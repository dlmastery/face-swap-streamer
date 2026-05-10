#include "faceswap/ffmpeg_io.hpp"

#include <fmt/core.h>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <sys/types.h>
  #include <sys/wait.h>
  #include <unistd.h>
#endif

namespace faceswap {

namespace {

#ifdef _WIN32
struct WinPipe {
    HANDLE read_end  = nullptr;
    HANDLE write_end = nullptr;
};

WinPipe make_pipe(bool inherit_read, bool inherit_write) {
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE r = nullptr, w = nullptr;
    if (!CreatePipe(&r, &w, &sa, 0))
        throw std::runtime_error("CreatePipe failed");
    if (!inherit_read)  SetHandleInformation(r, HANDLE_FLAG_INHERIT, 0);
    if (!inherit_write) SetHandleInformation(w, HANDLE_FLAG_INHERIT, 0);
    return {r, w};
}
#endif

std::string ext_lower(const fs::path& p) {
    std::string s = p.extension().string();
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

}  // namespace

struct FfmpegEncoder::Impl {
#ifdef _WIN32
    PROCESS_INFORMATION pi{};
    HANDLE stdin_w = nullptr;
#else
    pid_t pid = -1;
    int   stdin_fd = -1;
#endif
    int   width = 0, height = 0;
    bool  open  = false;
};

FfmpegEncoder::FfmpegEncoder() = default;
FfmpegEncoder::~FfmpegEncoder() { close(); }

void FfmpegEncoder::open(const fs::path& ffmpeg_exe,
                         const fs::path& audio_source,
                         const fs::path& output_mp4,
                         int width, int height, double fps) {
    if (p && p->open) close();
    p = std::make_unique<Impl>();
    p->width = width; p->height = height;

    fs::create_directories(output_mp4.parent_path());

    // ffmpeg argv:
    //   -y -loglevel error
    //   -f rawvideo -pix_fmt bgr24 -s WxH -r FPS -i -        (video from stdin)
    //   -i audio_source -map 0:v:0 -map 1:a:0?               (audio from target.mp4)
    //   -c:v libx264 -preset veryfast -pix_fmt yuv420p
    //   -c:a aac -shortest -movflags +faststart
    //   output.mp4
    std::vector<std::string> argv = {
        ffmpeg_exe.string(),
        "-y", "-loglevel", "error", "-hide_banner",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", fmt::format("{}x{}", width, height),
        "-r", fmt::format("{:.6f}", fps),
        "-i", "-",
    };
    const bool have_audio = !audio_source.empty() && fs::exists(audio_source);
    if (have_audio) {
        argv.insert(argv.end(), {"-i", audio_source.string(),
                                 "-map", "0:v:0", "-map", "1:a:0?"});
    }
    argv.insert(argv.end(), {
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-crf", "20",
    });
    if (have_audio) argv.insert(argv.end(), {"-c:a", "aac", "-shortest"});
    argv.insert(argv.end(), {"-movflags", "+faststart", output_mp4.string()});

#ifdef _WIN32
    auto pipe = make_pipe(/*inherit_read=*/true, /*inherit_write=*/false);
    p->stdin_w = pipe.write_end;

    std::string cmd;
    for (auto& a : argv) {
        if (!cmd.empty()) cmd += ' ';
        if (a.find_first_of(" \t\"") != std::string::npos) {
            cmd += '"';
            for (char c : a) { if (c == '"') cmd += '\\'; cmd += c; }
            cmd += '"';
        } else {
            cmd += a;
        }
    }

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput  = pipe.read_end;
    si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);

    std::vector<char> cmdbuf(cmd.begin(), cmd.end());
    cmdbuf.push_back('\0');
    if (!CreateProcessA(nullptr, cmdbuf.data(),
                        nullptr, nullptr, TRUE, 0, nullptr, nullptr,
                        &si, &p->pi)) {
        CloseHandle(pipe.read_end);
        CloseHandle(pipe.write_end);
        throw std::runtime_error(fmt::format("CreateProcess failed for ffmpeg ({})", GetLastError()));
    }
    CloseHandle(pipe.read_end);  // child owns its end now
#else
    int pipefd[2];
    if (pipe(pipefd) != 0) throw std::runtime_error("pipe() failed");
    pid_t pid = fork();
    if (pid < 0) {
        ::close(pipefd[0]); ::close(pipefd[1]);
        throw std::runtime_error("fork() failed");
    }
    if (pid == 0) {  // child
        dup2(pipefd[0], STDIN_FILENO);
        ::close(pipefd[0]); ::close(pipefd[1]);
        std::vector<char*> cargv;
        for (auto& a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
        cargv.push_back(nullptr);
        execvp(cargv[0], cargv.data());
        _exit(127);
    }
    ::close(pipefd[0]);
    p->pid = pid;
    p->stdin_fd = pipefd[1];
#endif
    p->open = true;
}

bool FfmpegEncoder::write_frame(const cv::Mat& bgr) {
    if (!p || !p->open) return false;
    if (bgr.cols != p->width || bgr.rows != p->height) {
        throw std::runtime_error(fmt::format(
            "frame size mismatch: encoder={}x{} got={}x{}",
            p->width, p->height, bgr.cols, bgr.rows));
    }
    cv::Mat contig = bgr.isContinuous() ? bgr : bgr.clone();
    const std::size_t bytes = static_cast<std::size_t>(contig.cols) * contig.rows * 3;
#ifdef _WIN32
    DWORD written = 0;
    BOOL ok = WriteFile(p->stdin_w, contig.data, (DWORD)bytes, &written, nullptr);
    return ok && written == bytes;
#else
    const std::uint8_t* d = contig.data;
    std::size_t left = bytes;
    while (left) {
        ssize_t n = ::write(p->stdin_fd, d, left);
        if (n <= 0) return false;
        d    += n;
        left -= (std::size_t)n;
    }
    return true;
#endif
}

int FfmpegEncoder::close() {
    if (!p || !p->open) return 0;
#ifdef _WIN32
    if (p->stdin_w) { CloseHandle(p->stdin_w); p->stdin_w = nullptr; }
    DWORD code = 0;
    WaitForSingleObject(p->pi.hProcess, INFINITE);
    GetExitCodeProcess(p->pi.hProcess, &code);
    CloseHandle(p->pi.hProcess);
    CloseHandle(p->pi.hThread);
    p->open = false;
    return (int)code;
#else
    if (p->stdin_fd >= 0) { ::close(p->stdin_fd); p->stdin_fd = -1; }
    int status = 0;
    waitpid(p->pid, &status, 0);
    p->open = false;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

bool FfmpegEncoder::is_open() const { return p && p->open; }

std::vector<fs::path> list_video_files(const fs::path& dir) {
    std::vector<fs::path> out;
    if (!fs::is_directory(dir)) return out;
    for (auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string e = ext_lower(entry.path());
        if (e == ".mp4" || e == ".mov" || e == ".mkv" || e == ".webm" || e == ".avi") {
            out.push_back(entry.path());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace faceswap
