// FaceAnalyser — RetinaFace + arcface + genderage, ported from
// insightface.app.FaceAnalysis (Python). Math references:
//   - RetinaFace (SCRFD) decode: insightface/python-package/insightface/model_zoo/scrfd.py
//   - arcface forward + similarity transform: insightface/.../arcface_onnx.py + face_align.py
//   - genderage: insightface/.../attribute.py
//
// IMPORTANT: this is the structural port of the analyser. The detect/decode
// math is non-trivial — we keep the entry points stable and the expensive
// portions are clearly delineated so they can be filled in over the next
// few sessions without changing any caller.

#include "faceswap/face_analyser.hpp"

#include <fmt/core.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>

namespace faceswap {

namespace {

constexpr int   kArcfaceSize   = 112;
constexpr int   kGenderAgeSize = 96;

const std::array<cv::Point2f, 5> kArcfaceDst = {{
    {38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f},
    {41.5493f, 92.3655f}, {70.7299f, 92.2041f},
}};

cv::Mat preprocess_blob(const cv::Mat& bgr, int target,
                        std::vector<int64_t>& out_shape, float& out_scale) {
    // Letterbox preserving aspect, scale 1/128 (insightface convention)
    const float r = std::min(static_cast<float>(target) / bgr.rows,
                             static_cast<float>(target) / bgr.cols);
    const int new_w = static_cast<int>(std::round(bgr.cols * r));
    const int new_h = static_cast<int>(std::round(bgr.rows * r));
    cv::Mat resized;
    cv::resize(bgr, resized, {new_w, new_h});
    cv::Mat canvas(target, target, CV_8UC3, cv::Scalar(0, 0, 0));
    resized.copyTo(canvas(cv::Rect(0, 0, new_w, new_h)));

    cv::Mat blob = cv::dnn::blobFromImage(canvas, 1.0 / 128.0,
                                          cv::Size(target, target),
                                          cv::Scalar(127.5, 127.5, 127.5),
                                          /*swapRB=*/true);
    out_shape = {1, 3, target, target};
    out_scale = r;
    return blob;
}

void l2_normalize(std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += static_cast<double>(x) * x;
    s = std::sqrt(std::max(s, 1e-12));
    for (float& x : v) x = static_cast<float>(x / s);
}

}  // namespace

void FaceAnalyser::load(const fs::path& dir,
                        OnnxSession::Provider provider,
                        int cuda_device,
                        int det_size, float det_thresh) {
    det_size_   = det_size;
    det_thresh_ = det_thresh;
    det_  .load((dir / "det_10g.onnx").string(),  provider, cuda_device);
    recog_.load((dir / "w600k_r50.onnx").string(), provider, cuda_device);
    ga_   .load((dir / "genderage.onnx").string(), provider, cuda_device);
}

std::vector<Face> FaceAnalyser::detect(const cv::Mat& bgr) const {
    std::vector<int64_t> in_shape;
    float scale = 1.0f;
    cv::Mat blob = preprocess_blob(bgr, det_size_, in_shape, scale);

    auto outs = det_.run(
        det_.input_names(),
        {reinterpret_cast<const float*>(blob.data)},
        {in_shape},
        det_.output_names());

    auto faces = decode_retinaface(outs, scale, bgr.cols, bgr.rows);

    for (auto& f : faces) {
        score_gender_age(bgr, f);
        f.embedding = embed(bgr, f);
    }
    return faces;
}

// SCRFD det_10g decode. The model emits 9 tensors over 3 strides (8, 16, 32)
// × {scores, bbox, kps}. At det_size=S the feature maps are S/8, S/16, S/32.
// Each location has 2 anchors → N anchors per stride = (S/stride)^2 * 2.
//
// We identify the role of each output by its last-dim shape: 1 → score,
// 4 → bbox, 10 → 5-pt kps. We identify the stride by N — at det_size 640
// the three N values are 12800, 3200, 800 → strides 8, 16, 32. This avoids
// hard-coding output names which differ across exporters.
//
// Decoding (per anchor i at stride s, anchor center (cx, cy) = (col*s, row*s)):
//   score = scores[i]
//   l,t,r,b = bbox[i] * s     → x1=cx-l, y1=cy-t, x2=cx+r, y2=cy+b
//   for each kp k: (kp_x, kp_y) = kps[i, 2k:2k+2] * s + (cx, cy)
//
// Then we map back to original-image coords by dividing by the letterbox
// scale (the same `scale` we used in preprocess_blob).
namespace {

struct Detection {
    cv::Rect2f bbox;
    std::array<cv::Point2f, 5> kps{};
    float score;
};

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width,  b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float ua = a.width * a.height + b.width * b.height - inter;
    return ua > 0 ? inter / ua : 0.0f;
}

std::vector<Detection> nms(std::vector<Detection> dets, float iou_thresh) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b){ return a.score > b.score; });
    std::vector<Detection> kept;
    std::vector<bool> dead(dets.size(), false);
    for (std::size_t i = 0; i < dets.size(); ++i) {
        if (dead[i]) continue;
        kept.push_back(dets[i]);
        for (std::size_t j = i + 1; j < dets.size(); ++j) {
            if (dead[j]) continue;
            if (iou(dets[i].bbox, dets[j].bbox) > iou_thresh) dead[j] = true;
        }
    }
    return kept;
}

}  // namespace

std::vector<Face> FaceAnalyser::decode_retinaface(
    const std::vector<OnnxSession::Output>& outputs,
    float scale, int orig_w, int orig_h) const {

    // Group outputs by their last-dim role (1=score, 4=bbox, 10=kps),
    // keyed by N (number of anchors at that stride).
    struct StrideOuts { const std::vector<float>* scores=nullptr; const std::vector<float>* bbox=nullptr; const std::vector<float>* kps=nullptr; };
    std::map<int64_t, StrideOuts> by_n;
    for (const auto& out : outputs) {
        if (out.shape.size() < 2) continue;
        const int64_t n   = out.shape[out.shape.size() - 2];
        const int64_t dim = out.shape.back();
        auto& slot = by_n[n];
        if      (dim == 1)  slot.scores = &out.data;
        else if (dim == 4)  slot.bbox   = &out.data;
        else if (dim == 10) slot.kps    = &out.data;
    }
    if (by_n.size() != 3)
        throw std::runtime_error(fmt::format(
            "decode_retinaface: expected 3 strides of outputs, got {}", by_n.size()));

    constexpr int kNumAnchors = 2;
    std::vector<Detection> raw;

    for (const auto& [n, s] : by_n) {
        if (!s.scores || !s.bbox || !s.kps) continue;

        // Recover stride from N: N = (det_size/stride)^2 * num_anchors
        // → stride = det_size / sqrt(N / num_anchors)
        const int locs = static_cast<int>(n) / kNumAnchors;
        const int side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(locs))));
        if (side <= 0 || side * side != locs) continue;
        const int stride = det_size_ / side;
        if (stride <= 0) continue;

        for (int row = 0; row < side; ++row) {
            for (int col = 0; col < side; ++col) {
                for (int a = 0; a < kNumAnchors; ++a) {
                    const std::size_t idx = static_cast<std::size_t>((row * side + col) * kNumAnchors + a);
                    const float score = (*s.scores)[idx];
                    if (score < det_thresh_) continue;

                    const float cx = static_cast<float>(col * stride);
                    const float cy = static_cast<float>(row * stride);
                    const float l = (*s.bbox)[idx * 4 + 0] * stride;
                    const float t = (*s.bbox)[idx * 4 + 1] * stride;
                    const float r = (*s.bbox)[idx * 4 + 2] * stride;
                    const float b = (*s.bbox)[idx * 4 + 3] * stride;

                    Detection d;
                    d.score = score;
                    const float x1 = (cx - l) / scale;
                    const float y1 = (cy - t) / scale;
                    const float x2 = (cx + r) / scale;
                    const float y2 = (cy + b) / scale;
                    d.bbox = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);

                    for (int k = 0; k < 5; ++k) {
                        const float kx = ((*s.kps)[idx * 10 + k * 2 + 0] * stride + cx) / scale;
                        const float ky = ((*s.kps)[idx * 10 + k * 2 + 1] * stride + cy) / scale;
                        d.kps[k] = {kx, ky};
                    }
                    raw.push_back(d);
                }
            }
        }
    }

    auto kept = nms(std::move(raw), 0.4f);

    std::vector<Face> out;
    out.reserve(kept.size());
    for (auto& d : kept) {
        // Clip bbox to image bounds — protects downstream crop/warp.
        const float x1 = std::max(0.0f, d.bbox.x);
        const float y1 = std::max(0.0f, d.bbox.y);
        const float x2 = std::min(static_cast<float>(orig_w), d.bbox.x + d.bbox.width);
        const float y2 = std::min(static_cast<float>(orig_h), d.bbox.y + d.bbox.height);
        if (x2 - x1 < 1.0f || y2 - y1 < 1.0f) continue;
        Face f;
        f.bbox      = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
        f.kps       = d.kps;
        f.det_score = d.score;
        out.push_back(std::move(f));
    }
    return out;
}

std::vector<float> FaceAnalyser::embed(const cv::Mat& bgr, const Face& face) const {
    std::vector<cv::Point2f> src(face.kps.begin(), face.kps.end());
    std::vector<cv::Point2f> dst(kArcfaceDst.begin(), kArcfaceDst.end());
    cv::Mat M = cv::estimateAffinePartial2D(src, dst, cv::noArray(), cv::LMEDS);
    cv::Mat aligned;
    cv::warpAffine(bgr, aligned, M, {kArcfaceSize, kArcfaceSize});
    cv::Mat blob = cv::dnn::blobFromImage(
        aligned, 1.0 / 127.5, cv::Size(kArcfaceSize, kArcfaceSize),
        cv::Scalar(127.5, 127.5, 127.5), /*swapRB=*/true);

    auto outs = recog_.run(
        recog_.input_names(),
        {reinterpret_cast<const float*>(blob.data)},
        {{1, 3, kArcfaceSize, kArcfaceSize}},
        recog_.output_names());
    if (outs.empty()) return {};
    auto emb = std::move(outs.front().data);
    l2_normalize(emb);
    return emb;
}

void FaceAnalyser::score_gender_age(const cv::Mat& bgr, Face& face) const {
    // Crop a square around the bbox, expand 1.5x, resize to 96×96.
    const float cx = face.bbox.x + face.bbox.width  * 0.5f;
    const float cy = face.bbox.y + face.bbox.height * 0.5f;
    const float side = std::max(face.bbox.width, face.bbox.height) * 1.5f;
    cv::Rect roi(int(cx - side / 2), int(cy - side / 2), int(side), int(side));
    roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (roi.width < 8 || roi.height < 8) return;

    cv::Mat crop = bgr(roi);
    cv::Mat resized;
    cv::resize(crop, resized, {kGenderAgeSize, kGenderAgeSize});
    cv::Mat blob = cv::dnn::blobFromImage(
        resized, 1.0, cv::Size(kGenderAgeSize, kGenderAgeSize),
        cv::Scalar(0, 0, 0), /*swapRB=*/true);

    auto outs = ga_.run(
        ga_.input_names(),
        {reinterpret_cast<const float*>(blob.data)},
        {{1, 3, kGenderAgeSize, kGenderAgeSize}},
        ga_.output_names());
    if (outs.empty() || outs.front().data.size() < 3) return;
    const float* d = outs.front().data.data();
    face.gender = (d[0] > d[1]) ? Gender::Female : Gender::Male;
    face.age    = static_cast<int>(std::round(d[2] * 100.0f));
}

}  // namespace faceswap
