// Pipeline: 4-stage threaded face-swap for a single video, plus a batch
// runner that processes N videos in parallel on the same GPU.
//
// Stage layout:
//   reader  ──► detect_q ──► detector ──► swap_q ──► swapper ──► encode_q ──► encoder
//
// Threading: each stage gets one thread. Bounded queues throttle each stage
// to its slowest neighbour. ORT internally serialises GPU calls so detector
// and swapper share the device safely; everything else (read, paste, write)
// runs lock-free on the CPU side.
//
// Batch mode runs `concurrency` independent pipelines on a worker pool. Each
// pipeline has its own queues — only the model sessions are shared (and they
// are thread-safe on the .Run() boundary).

#include "faceswap/pipeline.hpp"
#include "faceswap/ffmpeg_io.hpp"
#include "faceswap/reference.hpp"

#include <fmt/core.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

namespace faceswap {

namespace {

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t cap) : cap_(cap) {}

    void push(T v) {
        std::unique_lock lk(m_);
        cv_full_.wait(lk, [&]{ return q_.size() < cap_ || closed_; });
        if (closed_) return;
        q_.push_back(std::move(v));
        cv_empty_.notify_one();
    }

    /// Returns std::nullopt iff the queue is empty *and* closed (sentinel).
    std::optional<T> pop() {
        std::unique_lock lk(m_);
        cv_empty_.wait(lk, [&]{ return !q_.empty() || closed_; });
        if (q_.empty()) return std::nullopt;
        T v = std::move(q_.front());
        q_.pop_front();
        cv_full_.notify_one();
        return v;
    }

    void close() {
        std::scoped_lock lk(m_);
        closed_ = true;
        cv_empty_.notify_all();
        cv_full_.notify_all();
    }

private:
    std::size_t cap_;
    std::deque<T> q_;
    std::mutex m_;
    std::condition_variable cv_empty_, cv_full_;
    bool closed_ = false;
};

struct DetectMsg {
    int   frame_idx;
    cv::Mat frame;
};
struct SwapMsg {
    int   frame_idx;
    cv::Mat frame;
    std::vector<Face> faces;
};
struct EncodeMsg {
    int   frame_idx;
    cv::Mat frame;
    int   swaps_in_frame;
};

float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || a.size() != b.size()) return -1.0f;
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += static_cast<double>(a[i]) * b[i];
    return static_cast<float>(s);
}

const SourceSpec* match_source(const std::vector<SourceSpec>& sources,
                               const Face& f, float ref_thresh) {
    const SourceSpec* best = nullptr;
    float best_score = ref_thresh;
    for (const auto& s : sources) {
        if (s.gender != f.gender)        continue;
        if (s.ref_emb.empty())           continue;
        const float c = cosine(s.ref_emb, f.embedding);
        if (c >= best_score) { best_score = c; best = &s; }
    }
    return best;
}

}  // namespace

void run_streaming(VideoJob& job,
                   const std::vector<SourceSpec>& sources_in,
                   const FaceAnalyser& analyser,
                   const Inswapper& swapper,
                   const StreamingOpts& opts,
                   std::atomic_bool& cancel) {

    cv::VideoCapture cap(job.target_path.string());
    if (!cap.isOpened()) {
        job.error = "could not open input video";
        return;
    }
    job.width        = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    job.height       = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    job.fps          = cap.get(cv::CAP_PROP_FPS);
    job.total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    if (job.fps <= 0.1) job.fps = 25.0;  // sane default

    // Per-video reference extraction (closes its own VideoCapture)
    auto sources = sources_in;
    try {
        extract_reference_embeddings(job.target_path, analyser, sources);
    } catch (const std::exception& e) {
        job.error = fmt::format("reference extraction failed: {}", e.what());
        return;
    }

    fs::path ffmpeg_exe = std::getenv("FFMPEG_BIN") ? std::getenv("FFMPEG_BIN") : "ffmpeg";

    FfmpegEncoder encoder;
    try {
        encoder.open(ffmpeg_exe, job.target_path, job.output_path,
                     job.width, job.height, job.fps);
    } catch (const std::exception& e) {
        job.error = fmt::format("ffmpeg open failed: {}", e.what());
        return;
    }

    BoundedQueue<DetectMsg> detect_q(opts.q_depth);
    BoundedQueue<SwapMsg>   swap_q  (opts.q_depth);
    BoundedQueue<EncodeMsg> enc_q   (opts.q_depth);

    std::atomic_int  swap_count{0};
    std::atomic_int  proc_idx{0};
    const auto t0 = std::chrono::steady_clock::now();

    // Reader thread
    std::thread reader([&]{
        cv::Mat frame;
        int idx = 0;
        while (!cancel.load(std::memory_order_relaxed) && cap.read(frame) && !frame.empty()) {
            detect_q.push({idx++, frame.clone()});
        }
        cap.release();
        detect_q.close();
    });

    // Detect thread
    std::thread detect([&]{
        while (auto msg = detect_q.pop()) {
            if (cancel.load(std::memory_order_relaxed)) break;
            std::vector<Face> faces;
            try { faces = analyser.detect(msg->frame); }
            catch (const std::exception& e) {
                if (opts.verbose) fmt::print(stderr, "[detect] {}\n", e.what());
            }
            swap_q.push({msg->frame_idx, std::move(msg->frame), std::move(faces)});
        }
        swap_q.close();
    });

    // Swap thread
    std::thread swap([&]{
        while (auto msg = swap_q.pop()) {
            if (cancel.load(std::memory_order_relaxed)) break;
            int swaps_here = 0;
            for (const auto& f : msg->faces) {
                const auto* src = match_source(sources, f, opts.ref_thresh);
                if (!src) continue;
                try {
                    msg->frame = swapper.swap(msg->frame, f, src->src_face);
                    ++swaps_here;
                } catch (const std::exception& e) {
                    if (opts.verbose) fmt::print(stderr, "[swap] frame {}: {}\n",
                                                 msg->frame_idx, e.what());
                }
            }
            swap_count.fetch_add(swaps_here, std::memory_order_relaxed);
            enc_q.push({msg->frame_idx, std::move(msg->frame), swaps_here});
        }
        enc_q.close();
    });

    // Writer thread (also drives progress callback on a 0.5 s heartbeat)
    std::thread writer([&]{
        auto last_tick = std::chrono::steady_clock::now();
        while (auto msg = enc_q.pop()) {
            if (cancel.load(std::memory_order_relaxed)) break;
            if (!encoder.write_frame(msg->frame)) {
                job.error = "ffmpeg pipe died";
                cancel.store(true, std::memory_order_relaxed);
                break;
            }
            const int idx = proc_idx.fetch_add(1, std::memory_order_relaxed) + 1;
            const auto now = std::chrono::steady_clock::now();
            if (now - last_tick > std::chrono::milliseconds(500) && opts.on_progress) {
                last_tick = now;
                job.current_frame = idx;
                job.swap_count    = swap_count.load(std::memory_order_relaxed);
                const double dt   = std::chrono::duration<double>(now - t0).count();
                job.proc_fps      = dt > 0 ? idx / dt : 0.0;
                opts.on_progress(job);
            }
        }
    });

    reader.join();
    detect.join();
    swap.join();
    writer.join();

    int rc = encoder.close();
    if (rc != 0 && job.error.empty()) job.error = fmt::format("ffmpeg exited with code {}", rc);

    job.current_frame = proc_idx.load(std::memory_order_relaxed);
    job.swap_count    = swap_count.load(std::memory_order_relaxed);
    const double dt   = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    job.proc_fps      = dt > 0 ? job.current_frame / dt : 0.0;
    if (opts.on_progress) opts.on_progress(job);
}

void run_batch(std::vector<VideoJob>& jobs,
               const std::vector<SourceSpec>& sources,
               const FaceAnalyser& analyser,
               const Inswapper& swapper,
               const BatchOpts& opts,
               std::atomic_bool& cancel) {

    const int conc = std::max(1, std::min(opts.concurrency, (int)jobs.size()));
    std::atomic_int next_idx{0};

    auto worker = [&]{
        while (!cancel.load(std::memory_order_relaxed)) {
            const int i = next_idx.fetch_add(1, std::memory_order_relaxed);
            if (i >= (int)jobs.size()) break;
            run_streaming(jobs[i], sources, analyser, swapper, opts, cancel);
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(conc);
    for (int i = 0; i < conc; ++i) pool.emplace_back(worker);
    for (auto& t : pool) t.join();
}

}  // namespace faceswap
