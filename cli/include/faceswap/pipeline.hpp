#pragma once

#include "faceswap/face_analyser.hpp"
#include "faceswap/inswapper.hpp"
#include "faceswap/types.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <thread>

namespace faceswap {

/// Process one video end-to-end. The pipeline is a 4-stage thread chain:
///
///   reader (cv2.VideoCapture) ──► detect_q
///   detect (FaceAnalyser)     ──► swap_q
///   swap   (Inswapper)        ──► encode_q
///   encoder (libavformat OR ffmpeg subprocess) ──► output mp4
///
/// All stages run concurrently — the writer never blocks the GPU work and
/// vice-versa. C++ frees us from the GIL so detect+swap can genuinely
/// overlap with read+write.
///
/// Returns when the video is fully processed or `cancel` is signalled.
struct StreamingOpts {
    int  q_depth = 128;
    int  num_writer_threads = 1;
    float ref_thresh = 0.18f;
    bool verbose = false;

    /// Called from the worker thread on every progress tick (~2 Hz).
    /// Used to print progress bars from main without polling.
    std::function<void(const VideoJob&)> on_progress;
};

void run_streaming(
    VideoJob& job,
    const std::vector<SourceSpec>& sources,
    const FaceAnalyser& analyser,
    const Inswapper& swapper,
    const StreamingOpts& opts,
    std::atomic_bool& cancel);

/// Run N videos in parallel on the same GPU. Each pipeline runs in its own
/// thread; ORT serialises the actual GPU calls (detect + swap) but all the
/// CPU work (read, write, queue) overlaps freely. Bounded by VRAM.
struct BatchOpts : StreamingOpts {
    int concurrency = 2;
};

void run_batch(
    std::vector<VideoJob>& jobs,
    const std::vector<SourceSpec>& sources,
    const FaceAnalyser& analyser,
    const Inswapper& swapper,
    const BatchOpts& opts,
    std::atomic_bool& cancel);

}  // namespace faceswap
