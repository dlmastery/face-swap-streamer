#pragma once

#include "faceswap/onnx_session.hpp"
#include "faceswap/types.hpp"

#include <opencv2/core.hpp>

namespace faceswap {

/// Thin C++ port of insightface.model_zoo.INSwapper. Takes a 1×3×128×128
/// face crop + a source-face arcface embedding, runs inswapper_128_fp16.onnx,
/// and pastes the swapped face back into the source frame with a feathered
/// alpha mask.
class Inswapper {
public:
    Inswapper() = default;

    void load(const fs::path& model_path,
              OnnxSession::Provider provider,
              int cuda_device);

    /// Swap `target_face` in `bgr_frame` with `src_face`'s identity, in-place.
    /// Returns the modified frame (same dims as input).
    cv::Mat swap(cv::Mat bgr_frame,
                 const Face& target_face,
                 const Face& src_face) const;

private:
    /// 5-point similarity transform → 128×128 aligned face crop, plus the
    /// inverse matrix for pasting back.
    void aligned_crop(const cv::Mat& bgr, const Face& face,
                      cv::Mat& out_crop, cv::Matx23f& out_inv) const;

    /// Feathered paste-back: warp the swapped 128×128 + alpha mask back
    /// into the source frame using the inverse affine.
    cv::Mat paste_back(const cv::Mat& src_frame,
                       const cv::Mat& swapped_128,
                       const cv::Matx23f& inv) const;

    /// Apply the emap transform: latent = src_emb @ emap, then L2-normalise.
    /// Inswapper's ONNX takes embeddings in its own latent space, not raw
    /// arcface space — without this the swap output is garbage.
    std::vector<float> transform_embedding(const std::vector<float>& src_emb) const;

private:
    mutable OnnxSession session_;
    int input_size_ = 128;
    std::vector<float> emap_;       ///< 512×512 row-major (loaded from inswapper_emap.bin)
    int emap_dim_ = 512;
};

}  // namespace faceswap
