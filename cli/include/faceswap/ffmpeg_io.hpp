#pragma once

#include "faceswap/types.hpp"

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace faceswap {

/// Spawn ffmpeg as a subprocess to encode raw BGR frames + the original
/// audio track into an .mp4 with +faststart. RAII — destructor closes
/// stdin and waits for ffmpeg to exit.
///
/// We use a subprocess instead of linking libavformat because:
///   * libavformat's API is hostile and version-fragile
///   * the OS already has ffmpeg
///   * subprocess is deterministic and easy to debug
class FfmpegEncoder {
public:
    FfmpegEncoder();
    ~FfmpegEncoder();

    FfmpegEncoder(const FfmpegEncoder&) = delete;
    FfmpegEncoder& operator=(const FfmpegEncoder&) = delete;

    /// Open the encoder. `audio_source` is the original target.mp4 path —
    /// ffmpeg muxes its audio track into the output.
    void open(const fs::path& ffmpeg_exe,
              const fs::path& audio_source,
              const fs::path& output_mp4,
              int width, int height, double fps);

    /// Push one BGR frame to ffmpeg's stdin. Returns false if the pipe
    /// has died (e.g. ffmpeg crashed).
    bool write_frame(const cv::Mat& bgr);

    /// Close stdin and wait for ffmpeg to finish. Returns ffmpeg's exit code.
    int close();

    bool is_open() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p;
};

/// List of files matching --dir <folder>. Filters to common video extensions.
std::vector<fs::path> list_video_files(const fs::path& dir);

}  // namespace faceswap
