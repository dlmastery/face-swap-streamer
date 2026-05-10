#pragma once

#include "faceswap/onnx_session.hpp"
#include "faceswap/types.hpp"

#include <memory>
#include <opencv2/core.hpp>
#include <vector>

namespace faceswap {

/// Runs RetinaFace detection (det_10g.onnx) + gender/age (genderage.onnx)
/// + arcface embedding (w600k_r50.onnx) over an image and returns the
/// detected faces with their embeddings, gender, and 5-point landmarks.
///
/// Equivalent to insightface.app.FaceAnalysis(name="buffalo_l").get(img),
/// re-implemented in C++ so we can run it without Python.
class FaceAnalyser {
public:
    FaceAnalyser() = default;

    /// Loads all four models from `analyser_dir`. Expected files:
    ///   det_10g.onnx, w600k_r50.onnx, genderage.onnx
    /// (we skip 2d106det.onnx and 1k3d68.onnx — not used for face-swap)
    void load(const fs::path& analyser_dir,
              OnnxSession::Provider provider,
              int cuda_device,
              int det_size,
              float det_thresh);

    /// Detect + score + embed every face in `bgr`. The default execution
    /// is single-frame; batched API arrives in a later commit.
    std::vector<Face> detect(const cv::Mat& bgr) const;

    int   det_size()   const { return det_size_; }
    float det_thresh() const { return det_thresh_; }

private:
    /// RetinaFace decode: turn the raw det_10g outputs (anchor scores +
    /// bbox offsets + 5-kps offsets) into Face structs. Implemented in
    /// face_analyser.cpp; the math is straight from insightface's
    /// scrfd.py / retinaface.py.
    std::vector<Face> decode_retinaface(
        const std::vector<OnnxSession::Output>& outputs,
        float scale, int orig_w, int orig_h) const;

    /// 5-point similarity transform → 112×112 aligned crop, then
    /// arcface forward → 512-dim L2-normalised embedding.
    std::vector<float> embed(const cv::Mat& bgr, const Face& face) const;

    /// genderage.onnx forward — sets face.gender + face.age.
    void score_gender_age(const cv::Mat& bgr, Face& face) const;

private:
    mutable OnnxSession det_;
    mutable OnnxSession recog_;
    mutable OnnxSession ga_;
    int   det_size_   = 640;
    float det_thresh_ = 0.30f;
};

}  // namespace faceswap
