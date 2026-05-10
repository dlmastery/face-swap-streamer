// Face-swap CLI: high-throughput offline conversion.
//
// Usage:
//   faceswap --male m.jpg --female f.jpg --video clip.mp4 --output out/
//   faceswap --male m.jpg --female f.jpg --dir clips/ --output out/ --concurrency 3
//
// See cli/README.md for the full flag reference and tuning guidance.

#include <CLI/CLI.hpp>
#include <fmt/color.h>
#include <fmt/core.h>

#include "faceswap/face_analyser.hpp"
#include "faceswap/ffmpeg_io.hpp"
#include "faceswap/inswapper.hpp"
#include "faceswap/onnx_session.hpp"
#include "faceswap/pipeline.hpp"
#include "faceswap/reference.hpp"
#include "faceswap/types.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using faceswap::Config;
using faceswap::SourceSpec;
using faceswap::VideoJob;
using faceswap::Gender;

static std::atomic_bool g_cancel{false};

extern "C" void on_sigint(int) {
    g_cancel.store(true, std::memory_order_relaxed);
}

namespace {

fs::path which_ffmpeg() {
    if (const char* env = std::getenv("FFMPEG_BIN"); env && *env) return env;
#ifdef _WIN32
    const char* candidates[] = {"ffmpeg.exe", "ffmpeg"};
#else
    const char* candidates[] = {"ffmpeg"};
#endif
    for (auto* name : candidates) return name;  // PATH lookup deferred to popen
    return "ffmpeg";
}

void resolve_model_paths(Config& cfg) {
    if (cfg.face_analyser_dir.empty())
        cfg.face_analyser_dir = cfg.models_dir / "buffalo_l";
    if (cfg.inswapper_path.empty())
        cfg.inswapper_path = cfg.models_dir / "inswapper_128_fp16.onnx";
    if (cfg.ffmpeg_exe.empty())
        cfg.ffmpeg_exe = which_ffmpeg();
}

void validate(const Config& cfg) {
    if (cfg.male_image.empty() && cfg.female_image.empty())
        throw CLI::ValidationError("--male / --female", "at least one source image is required");
    if (cfg.video_path.empty() == cfg.video_dir.empty())
        throw CLI::ValidationError("--video / --dir", "exactly one of --video or --dir must be set");
    if (cfg.output_dir.empty())
        throw CLI::ValidationError("--output", "output directory is required");
    if (!cfg.male_image.empty() && !fs::exists(cfg.male_image))
        throw CLI::ValidationError("--male", fmt::format("file not found: {}", cfg.male_image.string()));
    if (!cfg.female_image.empty() && !fs::exists(cfg.female_image))
        throw CLI::ValidationError("--female", fmt::format("file not found: {}", cfg.female_image.string()));
    if (!cfg.video_path.empty() && !fs::exists(cfg.video_path))
        throw CLI::ValidationError("--video", fmt::format("file not found: {}", cfg.video_path.string()));
    if (!cfg.video_dir.empty() && !fs::is_directory(cfg.video_dir))
        throw CLI::ValidationError("--dir", fmt::format("not a directory: {}", cfg.video_dir.string()));
    if (!fs::exists(cfg.face_analyser_dir))
        throw CLI::ValidationError("models", fmt::format("face analyser dir missing: {}", cfg.face_analyser_dir.string()));
    if (!fs::exists(cfg.inswapper_path))
        throw CLI::ValidationError("models", fmt::format("inswapper model missing: {}", cfg.inswapper_path.string()));
}

/// Detect the source face once per --male/--female image. We pin gender from
/// the flag name (rather than trusting genderage on a single still image,
/// which is noisier than the human telling us "this is the male").
SourceSpec load_source(const fs::path& path, Gender gender,
                       const faceswap::FaceAnalyser& analyser) {
    SourceSpec spec;
    spec.path = path;
    spec.gender = gender;

    cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (img.empty())
        throw std::runtime_error(fmt::format("could not read image: {}", path.string()));

    auto faces = analyser.detect(img);
    if (faces.empty())
        throw std::runtime_error(fmt::format("no face detected in {}", path.string()));

    // Take the largest face (largest bbox area). Same heuristic as worker.py.
    auto& chosen = *std::max_element(
        faces.begin(), faces.end(),
        [](const auto& a, const auto& b) {
            return a.bbox.width * a.bbox.height < b.bbox.width * b.bbox.height;
        });
    spec.src_face = chosen;
    spec.src_face.gender = gender;  // pin from CLI, not from the model
    spec.age = chosen.age;
    return spec;
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, on_sigint);

    CLI::App app{"faceswap — offline face-swap CLI", "faceswap"};
    Config cfg;

    app.add_option("--male",       cfg.male_image,    "Source image for the male face");
    app.add_option("--female",     cfg.female_image,  "Source image for the female face");
    app.add_option("--video",      cfg.video_path,    "Single target video (mp4/mov/mkv/webm)");
    app.add_option("--dir",        cfg.video_dir,     "Directory of target videos (batch mode)");
    app.add_option("--output",     cfg.output_dir,    "Output directory (created if missing)")
        ->required();

    auto* perf = app.add_option_group("Performance");
    perf->add_option("--concurrency", cfg.concurrency, "Number of videos to process in parallel (batch mode)")
        ->default_val(2)->check(CLI::Range(1, 16));
    perf->add_option("--threads",     cfg.q_depth,    "Per-stage queue depth")
        ->default_val(128)->check(CLI::Range(8, 1024));
    perf->add_option("--det-size",    cfg.det_size,   "Face detector input edge (multiple of 32)")
        ->default_val(640);
    perf->add_option("--det-thresh",  cfg.det_thresh, "Face detector confidence threshold")
        ->default_val(0.30f);
    perf->add_option("--ref-thresh",  cfg.ref_thresh, "Reference embedding match threshold (cosine)")
        ->default_val(0.18f);
    perf->add_flag("--cpu",     [&](std::int64_t){ cfg.use_cuda = false; }, "Disable CUDA, run on CPU only");
    perf->add_flag("--trt",     cfg.use_trt,    "Enable TensorRT (experimental)");
    perf->add_option("--device", cfg.cuda_device, "CUDA device index")->default_val(0);

    auto* paths = app.add_option_group("Paths");
    paths->add_option("--models",          cfg.models_dir,         "Models root directory");
    paths->add_option("--face-analyser",   cfg.face_analyser_dir,  "Override buffalo_l directory");
    paths->add_option("--inswapper",       cfg.inswapper_path,     "Override inswapper model path");
    paths->add_option("--ffmpeg",          cfg.ffmpeg_exe,         "Path to ffmpeg binary");

    app.add_flag_function("-v,--verbose", [&](std::int64_t n){ cfg.verbosity = static_cast<int>(n) + 1; },
                          "Increase verbosity (-vv for debug)");

    CLI11_PARSE(app, argc, argv);

    try {
        resolve_model_paths(cfg);
        validate(cfg);
        fs::create_directories(cfg.output_dir);

        const auto provider = cfg.use_trt
            ? faceswap::OnnxSession::Provider::TensorRT
            : (cfg.use_cuda ? faceswap::OnnxSession::Provider::CUDA
                            : faceswap::OnnxSession::Provider::CPU);

        // ---- Load models ------------------------------------------------------
        fmt::print(fg(fmt::color::cyan), "[load]");
        fmt::print(" face analyser from {}\n", cfg.face_analyser_dir.string());
        faceswap::FaceAnalyser analyser;
        analyser.load(cfg.face_analyser_dir, provider, cfg.cuda_device,
                      cfg.det_size, cfg.det_thresh);

        fmt::print(fg(fmt::color::cyan), "[load]");
        fmt::print(" inswapper from {}\n", cfg.inswapper_path.string());
        faceswap::Inswapper swapper;
        swapper.load(cfg.inswapper_path, provider, cfg.cuda_device);

        // ---- Detect source faces ---------------------------------------------
        std::vector<SourceSpec> sources;
        if (!cfg.male_image.empty())
            sources.push_back(load_source(cfg.male_image, Gender::Male, analyser));
        if (!cfg.female_image.empty())
            sources.push_back(load_source(cfg.female_image, Gender::Female, analyser));
        fmt::print(fg(fmt::color::cyan), "[load]");
        fmt::print(" {} source face(s):", sources.size());
        for (const auto& s : sources) fmt::print(" [{}] {}", faceswap::gender_letter(s.gender), s.path.filename().string());
        fmt::print("\n");

        // ---- Build job list --------------------------------------------------
        std::vector<VideoJob> jobs;
        if (!cfg.video_path.empty()) {
            VideoJob j;
            j.target_path = cfg.video_path;
            j.filename    = cfg.video_path.filename().string();
            j.output_path = cfg.output_dir / (cfg.video_path.stem().string() + "_swapped.mp4");
            jobs.push_back(std::move(j));
        } else {
            for (auto& p : faceswap::list_video_files(cfg.video_dir)) {
                VideoJob j;
                j.target_path = p;
                j.filename    = p.filename().string();
                j.output_path = cfg.output_dir / (p.stem().string() + "_swapped.mp4");
                jobs.push_back(std::move(j));
            }
        }
        if (jobs.empty()) {
            fmt::print(fg(fmt::color::red), "[err] no videos to process\n");
            return 2;
        }
        fmt::print(fg(fmt::color::cyan), "[plan]");
        fmt::print(" {} video(s), concurrency={}\n", jobs.size(), cfg.concurrency);

        // ---- Run pipeline (single or batch) ----------------------------------
        const auto t0 = std::chrono::steady_clock::now();

        faceswap::BatchOpts opts;
        opts.q_depth     = cfg.q_depth;
        opts.ref_thresh  = cfg.ref_thresh;
        opts.verbose     = cfg.verbosity >= 2;
        opts.concurrency = cfg.concurrency;
        opts.on_progress = [](const VideoJob& j) {
            if (j.total_frames > 0) {
                fmt::print(fg(fmt::color::green), "\r[{}] frame {}/{} swaps={} fps={:.1f}",
                           j.filename, j.current_frame, j.total_frames, j.swap_count, j.proc_fps);
                std::fflush(stdout);
            }
        };

        if (jobs.size() == 1) {
            faceswap::run_streaming(jobs.front(), sources, analyser, swapper, opts, g_cancel);
        } else {
            faceswap::run_batch(jobs, sources, analyser, swapper, opts, g_cancel);
        }

        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(t1 - t0).count();

        fmt::print("\n");
        int ok = 0, fail = 0;
        for (const auto& j : jobs) {
            if (j.error.empty()) {
                fmt::print(fg(fmt::color::green), "[ok]");
                fmt::print(" {} → {} ({} frames, {} swaps)\n",
                           j.filename, j.output_path.filename().string(),
                           j.current_frame, j.swap_count);
                ++ok;
            } else {
                fmt::print(fg(fmt::color::red), "[fail]");
                fmt::print(" {}: {}\n", j.filename, j.error);
                ++fail;
            }
        }
        fmt::print("\nDone in {:.1f}s — {} ok, {} failed.\n", elapsed, ok, fail);
        return fail ? 1 : 0;

    } catch (const std::exception& e) {
        fmt::print(fg(fmt::color::red), "[fatal] {}\n", e.what());
        return 1;
    }
}
