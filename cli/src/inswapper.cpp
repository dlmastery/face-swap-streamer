// Inswapper — port of insightface.model_zoo.INSwapper.
// Reference: insightface/python-package/insightface/model_zoo/inswapper.py
//
// The op is:
//   1. similarity-align target face to 128×128 (arcface-style 5-pt template)
//   2. run inswapper_128_fp16.onnx with [aligned_face, source_emb] inputs
//   3. unwarp the result + soft mask, blend onto the source frame
//
// The aligned-crop math is the same template as arcface, just at a different
// size — we keep the destination kps below for clarity.

#include "faceswap/inswapper.hpp"

#include <fmt/core.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace faceswap {

namespace {

// SCRFD/arcface 5-pt destination template at 128×128. Insightface's
// face_align.estimate_norm() does:
//   ratio  = 128/128 = 1.0
//   diff_x = 8.0 * ratio = 8.0
//   dst    = arcface_dst_112 * ratio  -> arcface_dst_112
//   dst[:, 0] += diff_x               -> shifted +8 in x
// (NOT a 128/112 scale — that was the bug that put landmarks ~13 px too low.)
const std::array<cv::Point2f, 5> kInswapperDst = {{
    {46.2946f, 51.6963f},  // left eye   (38.2946 + 8, 51.6963)
    {81.5318f, 51.5014f},  // right eye  (73.5318 + 8, 51.5014)
    {64.0252f, 71.7366f},  // nose tip   (56.0252 + 8, 71.7366)
    {49.5493f, 92.3655f},  // mouth L    (41.5493 + 8, 92.3655)
    {78.7299f, 92.2041f},  // mouth R    (70.7299 + 8, 92.2041)
}};

constexpr int kCropSize = 128;

}  // namespace

void Inswapper::load(const fs::path& path, OnnxSession::Provider provider, int cuda_device) {
    session_.load(path.string(), provider, cuda_device);

    // emap is the last initializer of inswapper_128_fp16.onnx, a (512, 512)
    // matrix that transforms arcface embeddings into the inswapper latent
    // space. We pre-extract it via cli/scripts/extract_emap.py to a raw
    // float32 binary alongside the ONNX file.
    const fs::path emap_path = path.parent_path() / "inswapper_emap.bin";
    std::ifstream f(emap_path, std::ios::binary);
    if (!f)
        throw std::runtime_error(fmt::format(
            "missing inswapper_emap.bin (run: conda run -n dlc python "
            "cli/scripts/extract_emap.py)  expected: {}", emap_path.string()));
    f.seekg(0, std::ios::end);
    const std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    if (bytes != std::streamsize(emap_dim_) * emap_dim_ * 4)
        throw std::runtime_error(fmt::format(
            "inswapper_emap.bin wrong size: got {} bytes, want {}",
            bytes, emap_dim_ * emap_dim_ * 4));
    emap_.resize(static_cast<std::size_t>(emap_dim_) * emap_dim_);
    f.read(reinterpret_cast<char*>(emap_.data()), bytes);
    if (!f) throw std::runtime_error("inswapper_emap.bin read truncated");
}

std::vector<float> Inswapper::transform_embedding(const std::vector<float>& src_emb) const {
    if (emap_.empty())
        throw std::runtime_error("Inswapper::transform_embedding: emap not loaded");
    if (static_cast<int>(src_emb.size()) != emap_dim_)
        throw std::runtime_error(fmt::format(
            "embedding dim {} != emap dim {}", src_emb.size(), emap_dim_));

    // latent[j] = sum_i src_emb[i] * emap[i, j]   (row-major emap)
    std::vector<float> latent(emap_dim_, 0.0f);
    for (int i = 0; i < emap_dim_; ++i) {
        const float a = src_emb[i];
        const float* row = emap_.data() + std::size_t(i) * emap_dim_;
        for (int j = 0; j < emap_dim_; ++j) latent[j] += a * row[j];
    }

    // L2-normalise (matches Python's `latent /= np.linalg.norm(latent)`)
    double s = 0.0;
    for (float v : latent) s += static_cast<double>(v) * v;
    s = std::sqrt(std::max(s, 1e-12));
    for (float& v : latent) v = static_cast<float>(v / s);
    return latent;
}

void Inswapper::aligned_crop(const cv::Mat& bgr, const Face& face,
                             cv::Mat& out_crop, cv::Matx23f& out_inv) const {
    std::vector<cv::Point2f> src(face.kps.begin(), face.kps.end());
    std::vector<cv::Point2f> dst(kInswapperDst.begin(), kInswapperDst.end());
    cv::Mat M = cv::estimateAffinePartial2D(src, dst, cv::noArray(), cv::LMEDS);
    cv::warpAffine(bgr, out_crop, M, {kCropSize, kCropSize});

    cv::Mat invM;
    cv::invertAffineTransform(M, invM);
    out_inv = cv::Matx23f((float)invM.at<double>(0,0), (float)invM.at<double>(0,1), (float)invM.at<double>(0,2),
                          (float)invM.at<double>(1,0), (float)invM.at<double>(1,1), (float)invM.at<double>(1,2));
}

cv::Mat Inswapper::paste_back(const cv::Mat& src_frame,
                              const cv::Mat& swapped_128,
                              const cv::Matx23f& inv) const {
    // Direct port of the Python paste_back in insightface/inswapper.py:
    //   warp white mask + swapped face by IM
    //   img_white[img_white>20] = 255
    //   measure mask bbox, derive erode k = max(size/10, 10), blur k = max(size/20, 5)
    //   erode + Gaussian blur with odd kernel (2k+1)
    //   blend by mask/255
    cv::Mat invMat(2, 3, CV_32F, const_cast<float*>(inv.val));

    cv::Mat warped;
    cv::warpAffine(swapped_128, warped, invMat, src_frame.size(),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

    cv::Mat mask128(kCropSize, kCropSize, CV_32F, cv::Scalar(255.0f));
    cv::Mat img_white;
    cv::warpAffine(mask128, img_white, invMat, src_frame.size(),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    // Threshold: any pixel > 20 becomes 255 (matches Python).
    cv::threshold(img_white, img_white, 20.0, 255.0, cv::THRESH_BINARY);

    // Bounding box of the warped white region — sets adaptive kernel sizes.
    cv::Mat mask_u8;
    img_white.convertTo(mask_u8, CV_8UC1);
    std::vector<cv::Point> nonzero;
    cv::findNonZero(mask_u8, nonzero);
    if (nonzero.empty()) return src_frame.clone();
    cv::Rect bb = cv::boundingRect(nonzero);
    // Python uses bbox span (max-min, no +1): mimic with width-1, height-1.
    const int mh = std::max(0, bb.height - 1);
    const int mw = std::max(0, bb.width  - 1);
    const int mask_size = std::max(1, (int)std::round(std::sqrt(double(mh) * mw)));
    const int erode_k   = std::max(mask_size / 10, 10);
    // Python: k = max(mask_size//20, 5); blur_size = (2*k+1, 2*k+1)
    const int blur_half = std::max(mask_size / 20, 5);
    const int blur_k    = 2 * blur_half + 1;

    cv::erode(img_white, img_white,
              cv::getStructuringElement(cv::MORPH_RECT, {erode_k, erode_k}));
    cv::GaussianBlur(img_white, img_white, {blur_k, blur_k}, 0.0);
    img_white *= (1.0 / 255.0);  // CV_32F values now in [0, 1]

    cv::Mat mask3;
    cv::cvtColor(img_white, mask3, cv::COLOR_GRAY2BGR);

    cv::Mat src_f, warped_f;
    src_frame.convertTo(src_f, CV_32FC3);
    warped   .convertTo(warped_f, CV_32FC3);

    cv::Mat blended = src_f.mul(cv::Scalar::all(1.0) - mask3) + warped_f.mul(mask3);
    cv::Mat out;
    blended.convertTo(out, CV_8UC3);
    return out;
}

cv::Mat Inswapper::swap(cv::Mat bgr_frame,
                        const Face& target_face,
                        const Face& src_face) const {
    if (src_face.embedding.empty())
        throw std::runtime_error("Inswapper::swap: source embedding is empty");

    cv::Mat aligned;
    cv::Matx23f inv;
    aligned_crop(bgr_frame, target_face, aligned, inv);

    cv::Mat blob = cv::dnn::blobFromImage(
        aligned, 1.0 / 255.0, cv::Size(kCropSize, kCropSize),
        cv::Scalar(0, 0, 0), /*swapRB=*/true);

    // Apply emap transform on the source embedding — this is the missing step
    // that turned earlier C++ output into garbage.
    const std::vector<float> latent = transform_embedding(src_face.embedding);
    std::vector<int64_t> emb_shape{1, static_cast<int64_t>(latent.size())};

    auto outs = session_.run(
        session_.input_names(),
        {reinterpret_cast<const float*>(blob.data), latent.data()},
        {{1, 3, kCropSize, kCropSize}, emb_shape},
        session_.output_names());

    if (outs.empty()) throw std::runtime_error("Inswapper::swap: empty model output");

    // Inswapper output: 1×3×128×128 in 0..1 RGB. Convert back to BGR uint8.
    const auto& out = outs.front();
    cv::Mat swapped_chw(3, kCropSize * kCropSize, CV_32F);
    std::memcpy(swapped_chw.data, out.data.data(), out.data.size() * sizeof(float));
    std::vector<cv::Mat> chans(3);
    for (int c = 0; c < 3; ++c) {
        chans[c] = cv::Mat(kCropSize, kCropSize, CV_32F,
                           const_cast<float*>(out.data.data()) + c * kCropSize * kCropSize);
    }
    std::swap(chans[0], chans[2]);  // RGB -> BGR
    cv::Mat swapped_f, swapped_u8;
    cv::merge(chans, swapped_f);
    swapped_f.convertTo(swapped_u8, CV_8UC3, 255.0);

    return paste_back(bgr_frame, swapped_u8, inv);
}

}  // namespace faceswap
