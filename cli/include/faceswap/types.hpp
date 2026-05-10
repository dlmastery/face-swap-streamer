#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace faceswap {

namespace fs = std::filesystem;

enum class Gender : std::uint8_t { Unknown, Male, Female };

inline char gender_letter(Gender g) {
    switch (g) {
        case Gender::Male:   return 'M';
        case Gender::Female: return 'F';
        default:             return '?';
    }
}

/// One detected face in a frame. Mirrors what insightface.app.Face exposes,
/// minus fields we don't use.
struct Face {
    cv::Rect2f bbox;                    ///< (x, y, w, h) in source image pixels
    std::array<cv::Point2f, 5> kps{};   ///< 5-point landmarks (eye L, eye R, nose, mouth L, mouth R)
    float det_score = 0.0f;             ///< detector confidence
    Gender gender   = Gender::Unknown;  ///< from genderage model
    int   age       = 0;
    std::vector<float> embedding;       ///< L2-normalised arcface embedding (512 dim)
};

/// One uploaded source image (e.g. --male / --female), its detected face,
/// and the per-video reference embedding lock.
struct SourceSpec {
    fs::path path;
    Gender   gender = Gender::Unknown;
    int      age    = 0;
    Face     src_face{};                ///< detected face from the source image
    std::vector<float> ref_emb;         ///< filled in per video by reference extractor
    int      ref_frame = -1;            ///< the frame_idx the reference came from
    int      ref_pool  = 0;             ///< how many candidates clustered onto it
    int      ref_votes = 0;
};

/// CLI / pipeline configuration. Built from argv at program start.
struct Config {
    fs::path male_image;
    fs::path female_image;
    fs::path video_path;        ///< single-video mode if non-empty
    fs::path video_dir;         ///< directory mode if non-empty
    fs::path output_dir;        ///< where swapped MP4s land

    // Performance knobs
    int concurrency  = 2;       ///< how many videos to process in parallel on the GPU
    int det_size     = 640;
    float det_thresh = 0.30f;
    float ref_thresh = 0.18f;
    int  q_depth     = 128;     ///< per-stage queue depth in the swap pipeline

    // Model files (defaults assume layout from setup.ps1 — see README)
    fs::path models_dir = "models";
    fs::path face_analyser_dir;       ///< has det_10g.onnx, w600k_r50.onnx, genderage.onnx, ...
    fs::path inswapper_path;          ///< inswapper_128_fp16.onnx
    fs::path ffmpeg_exe;              ///< auto-detected if empty

    // Execution provider preferences
    bool use_cuda = true;
    bool use_trt  = false;            ///< off by default; -DUSE_TRT=ON build still needed
    int  cuda_device = 0;

    // Logging
    int  verbosity = 1;               ///< 0 quiet, 1 normal, 2 debug
};

/// One target video being swapped, with its progress + final output path.
struct VideoJob {
    fs::path target_path;
    fs::path output_path;             ///< where the final swapped MP4 lands
    std::string filename;             ///< basename of target, for display

    int width = 0, height = 0;
    double fps = 0.0;
    int total_frames = 0;
    int current_frame = 0;
    int swap_count = 0;
    double proc_fps = 0.0;

    std::string error;                ///< populated if the job failed
};

}  // namespace faceswap
