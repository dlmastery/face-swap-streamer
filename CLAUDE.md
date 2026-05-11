# CLAUDE.md

Operator's manual for AI agents (Claude Code, Cursor, etc.) and humans
working on this repo. Read [README.md](README.md) for the user-facing
pitch and [DESIGN.md](DESIGN.md) for the architecture deep-dive. This
file tells you **how to build, run, debug, and extend the project
without breaking it**.

If you only have time for one thing, read the [TL;DR](#tldr-getting-it-running),
then [Things that broke before](#things-that-broke-before--dont-re-break-them).

For deeper reading once you're past the basics:

- [README.md](README.md) — user-facing overview, quickstart
- [USERGUIDE.md](USERGUIDE.md) — end-user walkthrough, FAQ
- [DESIGN.md](DESIGN.md) — *why* the architecture is shaped this way
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the *how*: dataflow, threading, HTTP API, file layout
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md) — perf tuning, per-stage timings, what's been tried
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — symptom → cause → fix table
- [docs/CHANGELOG.md](docs/CHANGELOG.md) — every commit and what it taught us
- [docs/HACKING.md](docs/HACKING.md) — onboarding for new developers
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — PR process, code of conduct
- [OBS-setup.md](OBS-setup.md) — virtual webcam recipe

---

## Table of contents

1. [TL;DR — getting it running](#tldr-getting-it-running)
2. [Hardware + software prerequisites](#hardware--software-prerequisites)
3. [First-run from a fresh clone](#first-run-from-a-fresh-clone)
4. [Project goal](#project-goal)
5. [Architecture in one diagram](#architecture-in-one-diagram)
6. [Two conda envs (this is important)](#two-conda-envs-this-is-important)
7. [Code map](#code-map)
8. [Common tasks (commands you'll use)](#common-tasks-commands-youll-use)
9. [Editing workflow](#editing-workflow)
10. [HTTP API surface](#http-api-surface)
11. [Things that broke before](#things-that-broke-before--dont-re-break-them)
12. [Troubleshooting matrix](#troubleshooting-matrix)
13. [Performance + GPU memory](#performance--gpu-memory)
14. [Things NOT to commit](#things-not-to-commit)
15. [Workflow tips for AI agents](#workflow-tips-for-ai-agents)
16. [Smoke-test before committing](#smoke-test-before-committing)
17. [How to update upstream tools](#how-to-update-upstream-tools)
18. [Security notes](#security-notes)
19. [Known limitations](#known-limitations)

---

## TL;DR — getting it running

```powershell
git clone https://github.com/dlmastery/face-swap-streamer.git
cd face-swap-streamer
.\setup.ps1                                  # one-time; ~10 min, ~9 GB
conda run -n dlc python webapp.py            # start the web app
# open http://localhost:8080/
```

Drop your face photo + a video into the upload form. Wait through the
five phase pills (load → detect → reference → stream → finalise). The
live HLS player appears with a "Click to unmute" overlay; one click and
you have audio.

If you're modifying code, you don't need to re-run `setup.ps1` — just
edit and restart `webapp.py`.

---

## Hardware + software prerequisites

### Minimum

- Windows 10 (build 19044+) or Windows 11
- 16 GB RAM
- NVIDIA GPU with ≥ 6 GB VRAM and CUDA-12-compatible driver (R535 or newer)
- 15 GB free disk

### Recommended (what this was built on)

- Windows 11
- 32 GB RAM
- RTX 4090 Laptop / Desktop / equivalent (≥ 16 GB VRAM)
- NVIDIA driver R595+
- NVMe SSD (model loading is I/O-heavy)

### Required software (must be on PATH)

| Tool | Verify with | If missing |
|---|---|---|
| Anaconda or Miniconda | `conda --version` | https://www.anaconda.com/download |
| Git | `git --version` | https://git-scm.com/download/win |
| `gh` (GitHub CLI) — only for pushing | `gh auth status` | https://cli.github.com/ |
| ffmpeg / ffplay | `ffmpeg -version` | `winget install Gyan.FFmpeg` (preferred — Anaconda's ffmpeg ships without SDL2 so its `ffplay` is broken) |
| NVIDIA driver | `nvidia-smi` | https://www.nvidia.com/Download/index.aspx |

### CPU-only fallback?

Possible but slow. The code's `providers=["CUDAExecutionProvider",
"CPUExecutionProvider"]` means it falls back to CPU automatically if
CUDA fails. Expect ~1-2 fps wall-clock on a modern CPU instead of 7-25
fps on RTX 4090. Most of the docs/scripts assume GPU.

---

## First-run from a fresh clone

These are the exact commands a brand-new user types, in order. If any
step fails, see the [Troubleshooting matrix](#troubleshooting-matrix).

```powershell
# 1. clone
git clone https://github.com/dlmastery/face-swap-streamer.git
cd face-swap-streamer

# 2. one-shot install — creates conda envs, clones upstreams, downloads models
.\setup.ps1
# expect: ~10 min, ~9 GB on disk
# this provisions:
#   - conda env  faceswap (Py 3.12) for FaceFusion
#   - conda env  dlc      (Py 3.11) for Deep-Live-Cam + the web app
#   - .\facefusion\        (cloned from facefusion/facefusion)
#   - .\deep-live-cam\     (cloned from hacksider/Deep-Live-Cam)
#   - model files (inswapper_128_fp16.onnx ~265 MB, GFPGANv1.4.pth ~333 MB)
#   - patches to facefusion\facefusion\conda.py + deep-live-cam\run.py
#     for cuDNN DLL discovery

# 3. verify CUDA loaded in both envs
conda run -n faceswap python test-cuda.py
conda run -n dlc      python test-cuda-dlc.py
# expect: both print "VERDICT: CUDA works" and a 2x2 identity matrix

# 4. start the webapp
conda run -n dlc python webapp.py
# expect log line: "[webapp] starting on http://localhost:8080/"

# 5. open browser
start http://localhost:8080/
```

If you don't have `gh` or you're not the repo owner, you can stop after
step 4 — pushing isn't needed for local use.

---

## Project goal

A user uploads a photo and a video. The backend swaps the user's face
onto the matching person in the video and **streams the result back to
the browser with synchronised audio while it's still being processed**
(HLS via ffmpeg's tee muxer). When the swap finishes, the user
downloads the muxed MP4.

Several independent paths share the repo:

| Path | Tool | Purpose |
|---|---|---|
| **A** | FaceFusion 3.6 | Highest-quality offline render via CLI |
| **B** | Deep-Live-Cam 2.1.2 | Real-time GUI swap (webcam / virtual camera) |
| **C** | OBS Studio | Loop a swapped MP4 into a virtual webcam |
| **★** | **`webapp.py`** (Flask :8080) | **Original web app** — single-process 4-stage thread pipeline. 4-8 fps at 1080p. Most documented path. |
| **★★★** | **`webapp_mp.py`** (Flask any port, defaults :8080) | **Multiprocessing variant** — N worker processes via `multiprocessing.shared_memory`. **33 fps steady-state at 1080p** on i9 + 4090 with N=6. Same UI, same models, same outputs as `webapp.py`. Set `FACESWAP_PORT=8082 FACESWAP_WORKERS=6 FACESWAP_DET_SIZE=480` for the peak config. |
| **★★** | **FastAPI :8081 + Next.js :3000** (`server/`, `web/`) | **Production-grade rewrite** with batch upload + WebSocket status + ZIP download |
| **CLI** | **`cli/faceswap.exe`** | **C++ port** of the swap pipeline for batch / headless use, no Python required |

The web app reuses Deep-Live-Cam's `insightface` install but is its own
codepath; it doesn't shell out to FaceFusion or DLC. The C++ CLI is a
fresh re-implementation (RetinaFace decode, arcface alignment, inswapper,
paste-back, ffmpeg I/O) — same model files, different runtime.

---

## Architecture in one diagram

```
browser  ── upload ──►  Flask /start (webapp.py)
                              │ spawn worker thread
                              ▼
                       _run_job(job)
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
      InsightFace       inswapper-128     ffmpeg subprocess
      buffalo_l         128×128 -> face   raw BGR pipe + target.mp4 audio
      (detect+gender)   on detected box       │  cwd = job_dir
            │                 │               ▼
            └─────► reference ┘         tee muxer
                    embedding           ┌─────────────┐
                    cluster              │             │
                    (auto-extract)       ▼             ▼
                                     hls/playlist  swapped.mp4
                                       + .ts segs   (frag MP4)
                                          │             │
                                /job/<id>/hls/...   /job/<id>/download
                                          │             │
                                          ▼             ▼
                                    hls.js in     Final muxed
                                    <video>       download
                                    pre-buffer 15s
                                    + click-to-unmute
```

See [DESIGN.md](DESIGN.md) for why each piece is shaped this way.

---

## Two conda envs (this is important)

| Env | Python | Used by | Why |
|---|---|---|---|
| `faceswap` | 3.12 | `swap-song.ps1`, `swap-album.ps1` (FaceFusion path A) | FaceFusion 3.6 pins numpy 2.x, onnx 1.21, onnxruntime-gpu 1.24 |
| `dlc` | 3.11 | `webapp.py`, `stream-swap.py`, `play-song.ps1` (paths B + ★) | Deep-Live-Cam pins numpy <2, onnx 1.18, onnxruntime-gpu 1.23 |

These dependency sets are **mutually incompatible**. Do not try to merge
them. Always use `conda run -n <env> python ...` (or `conda activate
<env>` first); never call `python` directly without first knowing which
env.

`webapp.py` lives in the `dlc` env because it reuses DLC's
`insightface` install for face detection and the inswapper.

---

## Code map

| File | Purpose |
|---|---|
| `webapp.py` | Flask app — the **original** single-process artefact. ~700 lines: dataclass `Job`, model loader, worker `_run_job`, `_spawn_ffmpeg`, Flask routes, two big HTML templates (`INDEX_HTML`, `VIEWER_HTML`). 4-8 fps at 1080p. Stable, simple, no extra dependencies. |
| `webapp_mp.py` | Flask app — **multiprocessing variant**. Same routes / HTML / API as `webapp.py` but `_run_job` spawns N worker processes (`server/swap_worker.py`) that share frames via `multiprocessing.shared_memory`. Same models, same outputs. Adds `n_workers` + `worker_warmup_ms` + `paste` to `/status` JSON. **33 fps at 1080p steady-state** with `FACESWAP_WORKERS=6 FACESWAP_DET_SIZE=480` on RTX 4090 + i9. See `docs/perf-bench.md` for the autoresearch + winning config. |
| `server/swap_worker.py` | Per-worker entry point for `webapp_mp.py`. Loads `FaceAnalysis` + `INSwapper` once at process start, then loops on `SwapRequest`/`SwapResponse` over `multiprocessing.Queue`. Uses `FramePool` (shared-memory ring) to avoid IPC frame copies. ~310 lines. Reused inside each spawned worker via `mp.Process(target=worker_main, ...)`. |
| `stream-swap.py` | CLI version of the streaming pipeline. Outputs to `ffplay` window or a tiny built-in MJPEG http server. Useful for debugging the swap loop without Flask in the way |
| `extract-ref.py` | Standalone helper: scan a video, return the clearest face of a given gender |
| `probe.py` | Compatibility check on an `(image, video)` pair — reports if both are readable and a face is detectable in each |
| `test-cuda.py` / `test-cuda-dlc.py` | Verify `onnxruntime-gpu` actually loads CUDA in each env. Run these first if a job inexplicably falls back to CPU |
| `swap-song.ps1` | PS wrapper around `facefusion.py headless-run` (path A) |
| `swap-album.ps1` | Batch wrapper that processes every video in a folder via `swap-song.ps1` |
| `play-song.ps1` | PS wrapper around `deep-live-cam/run.py` (path B GUI) |
| `setup.ps1` | One-shot installer — creates envs, clones upstreams, installs deps, downloads models, applies cuDNN patches |
| `requirements-webapp.txt` | Pip deps for the `dlc` env beyond what DLC's own `requirements.txt` installs |
| `requirements-facefusion.txt` | CUDA runtime libs for the `faceswap` env beyond what FaceFusion's `install.py` installs |
| `OBS-setup.md` | Path C walk-through (no code) |
| `webapp_jobs/<id>/` | Per-job working dir: source.jpg, target.mp4, hls/, swapped.mp4, ffmpeg.log |
| `server/main.py`, `server/worker.py`, `server/schemas.py` | FastAPI rewrite (port 8081). Worker module shares ML pipeline with Flask via the same insightface session; HTTP/WebSocket layer is async/Pydantic. |
| `server/jobs/<id>/` | FastAPI per-job working dir (analogous to `webapp_jobs/`) |
| `web/` | Next.js 16 frontend (port 3000). App router, hls.js viewer, multi-file uploader, batch viewer, WebSocket client. |
| `cli/CMakeLists.txt`, `cli/include/`, `cli/src/` | **C++ CLI**. Headers + cpp for `OnnxSession`, `FaceAnalyser` (RetinaFace + arcface + genderage), `Inswapper`, `extract_reference_embeddings`, `run_streaming` / `run_batch`, `FfmpegEncoder`, plus `main.cpp`. |
| `cli/scripts/setup.ps1` | One-shot installer for the C++ build: winget cmake + MSVC, downloads ONNX Runtime 1.18.1 + OpenCV 4.10.0 win pack into `cli/third_party/`, copies `buffalo_l/` + inswapper into `cli/models/`. |
| `cli/scripts/build.ps1` | Loads `vcvars64.bat`, runs `cmake -G "Visual Studio 17 2022"`, builds Release. |
| `cli/scripts/extract_emap.py` | One-time helper: pulls the 512×512 emap matrix out of `inswapper_128_fp16.onnx` (its last graph initializer) into `cli/models/inswapper_emap.bin` so the C++ Inswapper can apply the arcface→latent transform. |
| `cli/README.md` | Build instructions, flag reference, batch tuning table. |

---

## Common tasks (commands you'll use)

### Run the webapp (foreground)

Single-process / 4-thread pipeline (the original — 4-8 fps at 1080p):
```powershell
conda run -n dlc python webapp.py
```

Multiprocessing variant (33 fps steady-state at 1080p — see `docs/perf-bench.md`):
```powershell
$env:FACESWAP_PORT       = "8082"   # avoid colliding with webapp.py if both running
$env:FACESWAP_WORKERS    = "6"      # N worker processes (default 4; 8 OOMs/race on 16 GB cards)
$env:FACESWAP_DET_SIZE   = "480"    # smaller detector → frees GPU for swap (+5%)
$env:FACESWAP_VIDEO_ENCODER = "h264_nvenc"   # NVENC keeps the encoder off the CPU
conda run -n dlc python webapp_mp.py
```

Both expose the same UI / HTML / `/status` JSON. Differences:
- `webapp_mp.py` spawns N workers at *job* start, which adds ~30 s of warmup to the first frame. Subsequent jobs in the same Flask process re-pay the warmup (no warm-pool yet).
- `webapp_mp.py`'s `/status` JSON adds `n_workers`, `worker_warmup_ms`, and a `paste` timer; otherwise identical schema.
- Pick `webapp.py` if you want the simpler "one job at a time, fewer moving parts" path. Pick `webapp_mp.py` if you want max throughput (4090 + i9 hits ~33 fps).

### Run in background, capture log
```powershell
Start-Process -WindowStyle Hidden -FilePath conda `
    -ArgumentList @('run','-n','dlc','python','webapp.py') `
    -RedirectStandardOutput out\webapp.log
```

### Tail the webapp log
```powershell
Get-Content out\webapp.log -Tail 50 -Wait
```

### Inspect a job's ffmpeg
Each job's ffmpeg stderr is drained to `webapp_jobs/<job_id>/ffmpeg.log`.
**Read this file first** when a job goes wrong — it shows what ffmpeg
complained about, which is otherwise invisible.

```powershell
$j = (Get-ChildItem webapp_jobs -Directory | Sort-Object LastWriteTime -Desc | Select-Object -First 1).Name
Get-Content "webapp_jobs\$j\ffmpeg.log" -Tail 30
```

### Path A: high-quality offline FaceFusion render
```powershell
.\swap-song.ps1 -Source .\source\me.jpg -Target .\songs\song.mp4
```

Quality presets: `fast`, `balanced` (default), `cinema`. Add `-Upscale`
for `frame_enhancer real_esrgan_x2_fp16`.

### Path B: real-time GUI swap (Deep-Live-Cam)
```powershell
.\play-song.ps1
```

### Path C: virtual webcam for calls
See [`OBS-setup.md`](OBS-setup.md). No code involved — it's an OBS
configuration walk-through.

### Run the CLI streamer (no web UI)
```powershell
conda run -n dlc python stream-swap.py `
    --source .\source\me.jpg `
    --target .\songs\song.mp4 `
    --gender M `
    --save .\out\swapped.mp4
```
Outputs to a `ffplay` window. Add `--web` for a small built-in MJPEG
http server instead.

### Run the C++ CLI (offline batch swap, no Python required at runtime)

One-time build (~10 min — fetches ORT 1.18.1 + OpenCV 4.10.0):
```powershell
pwsh -File cli/scripts/setup.ps1     # winget cmake + MSVC, downloads deps
conda run -n dlc python cli/scripts/extract_emap.py   # one-time emap dump
pwsh -File cli/scripts/build.ps1     # cmake configure + Release build
```

Single-video run:
```powershell
$env:FFMPEG_BIN = "C:\Users\<u>\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
.\cli\build\bin\Release\faceswap.exe `
    --male  .\source\sreeni.jpg `
    --video .\songs\song.mp4 `
    --output .\out\
```

Batch run (N videos in parallel on the same GPU):
```powershell
.\cli\build\bin\Release\faceswap.exe `
    --male  .\source\sreeni.jpg `
    --female .\source\her.jpg `
    --dir   .\songs\ `
    --output .\out\ `
    --concurrency 2
```

Observed perf on RTX 4090 Laptop, 1080p input:
- `--concurrency 1`: ~16-17 fps wall-clock (single-stream)
- `--concurrency 2`: ~33 fps aggregate (two pipelines × ~16-17 fps)
- bottleneck above 2 is GPU; CPU paste-back has plenty of headroom

The CLI shares its model files and inswapper math with the Python web
app, but its swap math went through a few fix-up passes (see
`Things that broke before` issues #16, #17, #18) — if a regression
shows up vs the Python output, the suspect order is: emap → alignment
template → blur kernel.

### Verify CUDA actually loaded
```powershell
conda run -n faceswap python test-cuda.py
conda run -n dlc      python test-cuda-dlc.py
```
Both should print `VERDICT: CUDA works` and run a tiny ONNX inference
on GPU. If they fall back to CPU, see "cuDNN" below.

### Stop a running webapp
Ctrl+C in its terminal, OR:
```powershell
Get-Process python | Where-Object { $_.MainModule.FileName -match 'envs\\dlc' } | Stop-Process
```

### Clear stale jobs
```powershell
Remove-Item -Recurse -Force webapp_jobs\* -ErrorAction SilentlyContinue
```

### Re-apply cuDNN patches (if you re-cloned upstreams)
```powershell
.\setup.ps1 -Force
```

---

## Editing workflow

The most common edit cycle for an AI agent:

1. Edit `webapp.py` (or another file).
2. Stop the running webapp (Ctrl+C the terminal, or `TaskStop` in
   Claude Code).
3. Restart: `conda run -n dlc python webapp.py`.
4. Wait ~5 s for Flask to bind + ~30 s for models to pre-warm.
5. Test in browser at <http://localhost:8080/>.
6. If a job fails, read `webapp_jobs/<id>/ffmpeg.log` first.

We don't use Flask's reloader (`use_reloader=False`) on purpose — it
double-loads the GPU models, which is slow and OOMs the GPU.

For HTML/JS-only changes, you can also just hard-refresh the browser
(`Ctrl+F5`) without restarting the server, since the templates are
inlined into `webapp.py` and re-rendered each request.

---

## HTTP API surface

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Upload form (HTML) |
| `POST` | `/start` | Multipart upload (`source` image, `target` video). Creates a job dir, spawns worker thread, redirects to `/job/<id>` |
| `GET` | `/job/<id>` | Viewer page (HTML with hls.js) |
| `GET` | `/job/<id>/status` | Worker state (JSON: `phase`, `message`, `current_frame`, `total_frames`, `swap_count`, `proc_fps`, `detected_gender`, `ref_frame`, …). Polled at ~2.5 Hz |
| `GET` | `/job/<id>/hls/playlist.m3u8` | HLS playlist — re-fetched periodically by hls.js |
| `GET` | `/job/<id>/hls/seg_NNNNN.ts` | HLS media segments — fetched on demand |
| `GET` | `/job/<id>/file` | Inline-served fragmented MP4 (`Content-Type: video/mp4`, Range-aware). Used as `<video>` `src` fallback |
| `GET` | `/job/<id>/download` | Same MP4 with `Content-Disposition: attachment` for the Download button |

The `/hls/...` route whitelists filenames ending in `.m3u8` or `.ts` and
rejects path-traversal attempts.

Job phases (in order):
`queued → loading_models → detecting_source → finding_reference →
streaming → finalising → done`. Plus `error` (terminal).

---

## Things that broke before — don't re-break them

### 1. cuDNN DLL discovery on Windows

`onnxruntime-gpu` doesn't bundle cuDNN/cuBLAS. They come from the
`nvidia-cudnn-cu12`, `nvidia-cublas-cu12`, etc. pip packages, which
install DLLs into `<env>/Lib/site-packages/nvidia/<lib>/bin/`.

Python 3.8+ on Windows uses a "secure DLL search" policy that **ignores
PATH for native imports**. You must call `os.add_dll_directory(...)`
**and keep the returned cookie alive in a long-lived list**.

`webapp.py` and `stream-swap.py` already do this at the top of the
file:

```python
_dll_cookies = []  # MUST be a kept reference — GC'd cookies = lost paths
if sys.platform == "win32":
    _sp = os.path.join(sys.prefix, "Lib", "site-packages")
    for _sub in ("cudnn", "cublas", "cuda_runtime", "curand", "cufft",
                 "cuda_nvrtc", "nvjitlink"):
        _bin = os.path.join(_sp, "nvidia", _sub, "bin")
        if os.path.isdir(_bin):
            _dll_cookies.append(os.add_dll_directory(_bin))
            os.environ["PATH"] = _bin + os.pathsep + os.environ["PATH"]
```

A common bug is `[os.add_dll_directory(p) for p in dirs]` — the
returned list is discarded, the cookies get GC'd, the directories are
removed from the search path almost immediately, and onnxruntime
silently falls back to CPU.

`setup.ps1` patches FaceFusion's `facefusion/conda.py` and
Deep-Live-Cam's `run.py` to do the same thing.

### 2. ffmpeg `tee` muxer + Windows paths

The `tee` muxer URL syntax uses `:` to separate options. Windows
drive-letter paths like `C:/Users/evija/...` collide with this. ffmpeg
will silently fail to write any output.

**Fix:** in `_spawn_ffmpeg(...)`, run ffmpeg with `cwd=<job_dir>` and
use relative paths (`hls/playlist.m3u8`, `swapped.mp4`) inside the tee
URL. Don't put absolute paths in there.

### 3. Browser autoplay policy

`<video>.play()` is silently rejected if the page has audio AND the
user hasn't interacted with the page. The viewer page now starts the
player **muted** so autoplay works, then shows a "Click to unmute"
overlay.

If you change the autoplay logic, make sure `player.muted = true` is
set **before** `player.play()`, and that the unmute overlay is wired
to set `player.muted = false`.

### 4. Pre-buffer for slower-than-realtime swap

The face-swap pipeline runs at ~7-25 fps wall-clock depending on input
resolution. The source video plays at 25 fps. If hls.js starts playing
as soon as it has 1 fragment, it'll outpace the producer and stutter
every second.

The viewer pre-buffers 15 seconds (`PREBUFFER_TARGET`) before pressing
play, and re-buffers to 8 seconds (`REBUFFER_TARGET`) on `waiting`
events. Don't lower these without testing on a slow GPU.

### 5. `python -m venv` from anaconda is broken on Windows

`python -m venv` from Anaconda's Python creates a venv whose
`sys.path` includes `<anaconda>\Lib`. Pip operations leak into the
Anaconda base env. We hit this and accidentally downgraded pillow +
pydantic-core in the user's base env once.

**Always use conda envs**, never `python -m venv` if your base Python
is from Anaconda.

### 6. PowerShell heredoc + `git commit -m`

`git commit -m @'...'@` mangles the message — PowerShell splits the
heredoc into multiple positional arguments. Use:

```powershell
git commit -F .commit-msg.tmp
```

with the message in a temp file. `setup.ps1` and the example commands
in this file follow that pattern.

### 7. Reference-embedding GC

The auto-extract reference logic stores `numpy` embeddings in a list of
tuples. Make sure the list outlives the loop that builds it — losing
the reference makes `np.dot` later return wrong shapes.

### 8. ONNX Runtime "fallback to CPU" silent disaster

When you give onnxruntime a list like `[("TensorrtExecutionProvider",
{...}), "CUDAExecutionProvider", "CPUExecutionProvider"]` and the FIRST
provider fails to initialise (TRT lib missing, cuDNN missing, etc.),
**onnxruntime does NOT fall through to the next provider in your list**
— it falls all the way to **CPU only**. This is silent (just a one-line
`EP Error... Falling back to ['CPUExecutionProvider'] and retrying.`
in stderr) and you'll spend forever wondering why your 4090 is at 0 %
util.

Defence in depth in `_ensure_models()`:

1. **Detect TRT availability** before listing it as a provider:
   ```python
   try:
       import tensorrt  # noqa
       trt_available = "TensorrtExecutionProvider" in ort.get_available_providers()
   except ImportError:
       trt_available = False
   ```
   Don't put TRT in the providers list unless `trt_available` is true.

2. **Verify the active provider after load**:
   ```python
   active = _swapper.session.get_providers()
   if active == ["CPUExecutionProvider"]:
       raise RuntimeError("inswapper loaded on CPU only — CUDA failed")
   ```
   Better to crash startup than silently grind frames at 1 fps.

3. The face analyser uses CUDA (not TRT) because some of its models have
   dynamic-shape inputs (`det_10g.onnx` has `'?'` dims) that TRT can't
   compile efficiently. CUDA is plenty fast at 640×640 anyway.

### 9. TensorRT DLL discovery

`tensorrt-cu12` puts its DLLs at `<env>/Lib/site-packages/tensorrt_libs/`
— a different layout from the other `nvidia-*-cu12` packages (which put
DLLs at `nvidia/<lib>/bin/`). The DLL discovery loop in `webapp.py` /
`stream-swap.py` includes BOTH paths:

```python
_bin_dirs = [
    *(os.path.join(_sp, "nvidia", sub, "bin")
      for sub in ("cudnn", "cublas", "cuda_runtime", "curand", "cufft",
                  "cuda_nvrtc", "nvjitlink")),
    os.path.join(_sp, "tensorrt_libs"),  # TRT — different layout
]
```

If you add another nvidia-* package with yet another layout, append it
here.

### 10. TRT engine build holds the GIL

While ONNX Runtime / TensorRT compiles a TRT engine on first run
(~60–90 s for inswapper), the C++ side does CPU-bound work that
sometimes holds the Python GIL. Flask's threaded request handlers
become slow during this window. The page-load timeout in test scripts
should account for this — wait at least 90 s after a fresh restart
before declaring the server dead.

This only happens once per cache directory; the engine is then
serialised to `webapp_jobs/.trt_cache/` and reused.

### 12. Pipeline thread coordination

The 4-stage pipeline (`reader → detect → swap → writer`) uses three
`queue.Queue(maxsize=128)` between stages. Things that broke before:

- **Reader put() blocked on full queue while we tried to join it** →
  always drain upstream queues in the `finally:` block before joining
  threads, otherwise the join times out.
- **Setting a `playStarted=true` flag *before* the play() promise
  resolved** (analogous to setting "broken=true" before pipe write
  succeeds). Always flip the success flag *after* the operation
  resolves, in the success branch.
- **Sending END to the wrong queue at exit** — each stage forwards
  END to the next. Reader puts END on read_q; detector pops END,
  forwards to detect_q; main loop pops END from detect_q, then puts
  END on write_q for the writer.

The `_run_job` finally block:
1. `job.stop_flag.set()` — reader notices on next iter and exits
2. drain `read_q` and `detect_q` (best-effort) so any blocked put()s
   unblock and propagate END
3. `write_q.put(END)` — writer drains and exits
4. `t_writer.join(timeout=30)` then `t_detect.join(timeout=10)`
   then `t_reader.join(timeout=10)` — must be in this order so that
   any frames in flight reach ffmpeg before we close cap
5. `cap.release()`

### 13.5 Fragmented MP4 isn't a real MP4

We originally output the downloadable MP4 via the same ffmpeg tee
that produced HLS, with `movflags=+frag_keyframe+empty_moov+
default_base_moof`. That works for browser MSE players (the inline
`<video>` we use for the in-page VOD fallback) but **fails on every
native player**: phones reported "format not supported", desktop
apps played audio only because video fragments were unreadable,
ffprobe couldn't even parse the file.

Fix: drop MP4 from the streaming ffmpeg's tee output. Stream HLS only
during processing. Then in the finalise phase, run a second ffmpeg
pass that remuxes (no re-encode) the .ts segments into a real MP4
with `+faststart`:

```python
def _remux_to_mp4(job: Job) -> None:
    cmd = [FFMPEG_EXE, "-y", "-allowed_extensions", "ALL",
           "-i", os.path.join(job.hls_dir, "playlist.m3u8"),
           "-c", "copy", "-bsf:a", "aac_adtstoasc",
           "-movflags", "+faststart",
           job.out_audio_path]
    rc = subprocess.call(cmd)
    if rc != 0: raise RuntimeError(f"remux failed (rc={rc})")
```

`-bsf:a aac_adtstoasc` is required because TS segments carry AAC in
ADTS framing while MP4 needs raw AAC frames. `+faststart` puts the
moov atom at the front so iOS Safari and progressive download work.

Lesson: fragmented MP4 is a streaming protocol format. Always remux
to non-fragmented MP4 if the file will be opened by anything except
an MSE-based player.

### 13. Multi-source matching

`Job.sources` is a list of `SourceSpec` (path, gender, age,
src_face, ref_emb, ref_frame, ref_votes, ref_pool). The form has two
file inputs both with `name="source"`; Flask's
`request.files.getlist("source")` collects them. Per-frame, the
detector stacks all detected faces' embeddings into `(T, D)`,
multiplies by the pre-stacked `(S, D)` source references → `(T, S)`
similarity matrix; argmax per target row picks the matching source
(if it clears `REFERENCE_THRESH=0.22`). Each source can swap
multiple faces per frame (e.g. crowd shots), each face matches its
single best source.

Same-gender second source: the auto-extract is greedy — it claims
the largest cluster of its gender, then masks those candidates so
the next same-gender source picks a different cluster. Works for two
distinct people of the same gender; degrades to "first source claims
the lead" when the video only has one matching-gender person.

### 14. Browser autoplay rescue (UX, not a crash)

`<video>.play()` can be silently rejected by the browser's autoplay
policy if the user hasn't interacted with the page yet — even when
`muted=true`. Symptom: video is decoded, ready, but `paused=true` and
the player looks "stuck on a frame". The previous Playwright debug
showed `readyState=4, currentTime=91.75, paused=false, muted=true` —
i.e. it WAS playing once user clicked, but the first attempt had
silently failed.

Three layers of mitigation in `VIEWER_HTML`:

```html
<video id="player" playsinline controls muted autoplay></video>
```
- `muted autoplay` HTML attributes — browsers handle these via the
  declarative path more permissively than calling `play()` from JS.

```js
function tryStartPlayback() {
  if (playStarted) return;
  if (bufferedAhead() < PREBUFFER_TARGET) return;
  player.muted = true;
  player.play()
    .then(() => { playStarted = true; /* show unmute pill */ })
    .catch(err => setTimeout(tryStartPlayback, 1000));   // RETRY!
}
```
- Set `playStarted = true` only **after** the promise resolves —
  otherwise a rejection leaves `playStarted` stuck at true and we
  never retry.

```js
document.addEventListener('click', () => {
  if (!playStarted && bufferedAhead() >= 1) {
    player.muted = true; player.play().then(() => playStarted = true);
  }
});
```
- Universal click-anywhere rescue: if Chrome refuses every autoplay
  attempt, one click anywhere on the page kicks playback alive.

### 15. C++ pimpl + `unique_ptr<Impl>` needs ctor in `.cpp`

`FfmpegEncoder` (and `OnnxSession`) use the pimpl idiom with
`std::unique_ptr<Impl> p`. If you `= default` the *constructor* in the
header, MSVC instantiates `unique_ptr<Impl>::~unique_ptr` *at the call
site* — where `Impl` is incomplete — and you get
`C2027 use of undefined type 'Impl'` and `C2338 can't delete an
incomplete type`.

Fix: declare `FfmpegEncoder();` in the header, define
`FfmpegEncoder::FfmpegEncoder() = default;` in the `.cpp` (where `Impl`
is complete). Same rule for `~FfmpegEncoder() = default;` and any
defaulted move ops.

### 16. C++ inswapper without the emap transform → garbage swaps

The Python `inswapper.get()` does an extra step that's *not* in the
ONNX model:

```python
latent = source_face.normed_embedding.reshape((1,-1))
latent = np.dot(latent, self.emap)        # 512×512 matrix
latent /= np.linalg.norm(latent)
pred = session.run(..., {input_names[1]: latent})
```

`emap` is the **last graph initializer** in `inswapper_128_fp16.onnx`
(a 512×512 float32 matrix). It transforms arcface embeddings into the
inswapper's own latent space. If you feed the raw arcface embedding
straight to the model, you get visually-broken swaps that *look* like
they're working (output has face-shaped pixels) but the identity is
wrong and edges are degenerate.

The C++ port pre-extracts emap once via `cli/scripts/extract_emap.py`
into `cli/models/inswapper_emap.bin`, then `Inswapper::transform_embedding`
loads it at startup and applies `latent = src_emb @ emap; latent /= ||latent||`
before each `session.run`. **Do not skip this even if the swap "looks
plausible" without it.**

### 17. C++ inswapper alignment template — +8 X shift, NOT a 128/112 scale

The 5-point destination kps for the 128×128 inswapper crop come from
`insightface/utils/face_align.py::estimate_norm`:

```python
if image_size % 128 == 0:                 # ← inswapper hits this
    ratio = float(image_size) / 128.0    # = 1.0 for 128
    diff_x = 8.0 * ratio                 # = 8.0
dst = arcface_dst * ratio                 # arcface_dst unchanged
dst[:, 0] += diff_x                       # shift X by +8
```

So at 128 the template is **the unscaled 112-arcface template, X-shifted
by +8** — *not* `arcface_dst * (128/112)`. The wrong scale puts the chin
landmarks ~13 px too low, which produces visibly misaligned chin/jaw
swaps that look like a regression vs Python.

Correct:
```cpp
const std::array<cv::Point2f, 5> kInswapperDst = {{
    {46.2946f, 51.6963f}, {81.5318f, 51.5014f}, {64.0252f, 71.7366f},
    {49.5493f, 92.3655f}, {78.7299f, 92.2041f},
}};
```

### 18. C++ paste-back: `2k+1`, not `k`, for the Gaussian blur kernel

Python's `inswapper.py` paste-back computes the blur kernel like this:

```python
k = max(mask_size // 20, 5)
blur_size = tuple(2*i + 1 for i in (k, k))
img_white = cv2.GaussianBlur(img_white, blur_size, 0)
```

`k` is the *half-kernel*; the actual kernel size is `2k+1`. If you copy
that to C++ and pass `k` directly to `cv::GaussianBlur`, your blur radius
is roughly half what Python uses, and you can see the swap-mask edge as
a soft seam at 1080p. Same `mask_size = sqrt(mask_h * mask_w)` formula
on a bbox span that's `(max-min)`, not `(width)` (off-by-one matters at
small face sizes).

### 19. ORT 1.18 forward declarations: `struct`, not `class`

`Ort::Env`, `Ort::Session`, `Ort::SessionOptions` are declared as
`struct` in ORT 1.18's `onnxruntime_cxx_api.h`. If you forward-declare
them as `class` in your header, MSVC issues C4099 warnings. Use
`namespace Ort { struct Session; struct Env; struct SessionOptions; }`
(or include the full header in your `.cpp`).

### 20. Anaconda's bundled `ffmpeg.exe` is broken outside the conda env

`C:\Users\<u>\anaconda3\Library\bin\ffmpeg.exe` runs but fails with
`error while loading shared libraries: ?: cannot open shared object
file: No such file or directory` if you call it without conda
activation (it can't find its dependent DLLs from the conda Library
tree). For the C++ CLI we use the gyan.dev winget build instead:

```
C:\Users\<u>\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_..._8wekyb3d8bbwe\ffmpeg-X.Y-full_build\bin\ffmpeg.exe
```

…and pass that as `FFMPEG_BIN` env var. Gyan's full build is
self-contained, has libx264 + h264_nvenc + aac, no DLL dependencies.

### 21. `curl` on Windows + corporate-issued root certs → revocation check fails

```
curl: (35) schannel: next InitializeSecurityContext failed:
CRYPT_E_NO_REVOCATION_CHECK (0x80092012)
```

Symptom: GitHub release downloads (ORT, OpenCV) intermittently 0-byte
or never start. Root cause: schannel's strict OCSP revocation policy
hits a TLS cert it can't get a CRL for (common with MDM-managed
machines). Fix: pass `--ssl-no-revoke` to curl. We use this in
`cli/scripts/setup.ps1`'s download step.

### 22. PowerShell auto-variable `$PID` is read-only

Don't write `foreach ($pid in $list)` — PowerShell's built-in `$PID`
is read-only and you'll get
`Cannot overwrite variable PID because it is read-only or constant`.
Use `$id` or `$processId` instead. Same applies to `$HOME`, `$PWD`,
and a few others.

---

## Troubleshooting matrix

| Symptom | First place to look | Likely fix |
|---|---|---|
| Live player blank, no spinner movement | Browser DevTools → Network → `playlist.m3u8` and `seg_*.ts` | If 404: ffmpeg failed; check `webapp_jobs/<id>/ffmpeg.log`. If 200 + `paused=true`: autoplay blocked; ensure `player.muted=true` before `play()` |
| Live player blank, spinner stuck on "Buffering 0/15s" | `webapp_jobs/<id>/ffmpeg.log` | ffmpeg exited or crashed — see ffmpeg's last lines |
| ffmpeg.log empty | webapp.log | ffmpeg never spawned. Check `FFMPEG_EXE` is found at top of `webapp.py` |
| Job stuck on "loading_models" >2 min | `out/webapp.log` | Model download is silent — check `~/.insightface/models/buffalo_l/` is filling. If network is slow, just wait |
| Swap is slow / GPU shows 0% util | `nvidia-smi` while a job runs | cuDNN didn't load. Run `test-cuda-dlc.py`. If FAILED, re-install nvidia pip packages: `conda run -n dlc pip install --force-reinstall nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-curand-cu12 nvidia-cufft-cu12` |
| Faces flicker on/off (passthrough then swap then passthrough) | n/a | Lower `--det-thresh` to 0.30, or `REFERENCE_THRESH` in `webapp.py` to 0.18. Or feed a clearer source image |
| Wrong person gets swapped (e.g. female lead instead of male) | `/status` JSON `detected_gender` | If wrong, the source image's face is ambiguous; provide a clearer front-facing photo. The auto-extract will re-derive the reference accordingly |
| Audio missing in stream | `ffprobe target.mp4` | If no audio track in source, ffmpeg's `-map 1:a:0?` silently drops audio (the `?` makes it optional). Check the target file actually has audio |
| Audio + video desync | n/a | Expected when swap is much slower than realtime. Reduce target resolution before upload, or use Path A for offline-quality output |
| `gh repo create` says "already exists" | `gh repo view` | Repo was created previously; just `git push` |
| `gh push` 403 | `gh auth status` | Re-auth: `gh auth login` |
| GPU at 0%, CPU at 70%+ during a job | Webapp log: look for `inswapper active providers` | If line is `['CPUExecutionProvider']`, your TRT (or cuDNN) lib failed to load and ORT silently fell to CPU. See issue #8. Fix: install `tensorrt-cu12`, or remove TRT from providers list and rely on CUDA |
| Inswapper takes forever to load on first run | (normal) | TRT is building the engine (~60–90 s). Subsequent runs reuse cached engine in `webapp_jobs/.trt_cache/` |
| Live player looks blank but progress bar advances | Playwright: `evaluate("document.getElementById('player').paused")` | If `paused=true` and `readyState=4`: autoplay policy blocked initial play(). Click anywhere on the page — code retries play() on every click. The "Click to unmute" pill in the bottom-left also unblocks it |
| Webapp won't bind 8080 | `Get-NetTCPConnection -LocalPort 8080` | Another process is using it. Change `app.run(port=8080)` to another port, or kill the other process |
| Out-of-memory on GPU | `nvidia-smi` | Models hold ~5 GB; if other CUDA processes (Stable Diffusion, etc.) are running, kill them. Or restart your machine |
| `ImportError: DLL load failed while importing onnxruntime` | (cmd) | Same root cause as cuDNN — see issue #1 |

---

## Performance + GPU memory

Observed on RTX 4090 Laptop (16 GB VRAM), 640×480 Bollywood mp4 with
TRT enabled, walking through the speedup history:

| commit | what's in the loop | proc fps | GPU avg util |
|---|---|---|---|
| serial baseline    | cv2.read → fa.get → sw.get → ffmpeg.write all on one thread |  7.5 | ~17 % |
| `8735818` writer thread       | + async `ffmpeg.stdin.write` thread        | 10.1 | ~17 % |
| `660e7d1` reader thread + Q=32| + async `cv2.read` thread                  | 10.8 | ~17 % |
| `0a966ce` det 480 + Q=64      | + 480 detector input + bigger queues       | 11.7 | ~18 % |
| `2a0e0dd` multi-face          | + 1-or-2-source matching, batched embed   | 12+  | ~18 % |
| `d4fc024` 4-stage pipeline    | + dedicated detect thread, Q=128          | 12+  | ~18 % |

Total: about **+60 % throughput** with the same model + same detector
input, vs the serial baseline. After this point GPU is no longer the
bottleneck on this workload — average util sits around 18 % and only
peaks briefly. Further gains would need a bigger model (so the GPU
actually has more work) or a different I/O path (NVDEC, zero-copy
to ffmpeg).

Theoretical ceiling at higher resolutions on the same hardware:

| Resolution | observed fps | est. GPU util |
|---|---|---|
| 480×360  | 30-45  | ~25 % |
| 640×480  | 18-25  | ~20 % |
| 1280×720 | 12-18  | ~30 % |
| 1920×1080| 8-13   | ~40 % |

Bottleneck per-frame breakdown at 1080p (rough):
- `cv2.VideoCapture.read` ~3-5 ms (CPU decode)
- `fa.get` ~10-15 ms (GPU detect — biggest)
- per-face `sw.get` ~5-10 ms (GPU swap)
- `frame.tobytes()` ~3-5 ms (CPU memcpy)
- `ffmpeg.stdin.write` ~1-3 ms (pipe + ffmpeg encode)
- Python glue ~1-2 ms (queue puts/gets, attribute access, GIL switching)

Total ~25-40 ms/frame → 25-40 fps theoretical, observed ~10-13 fps —
the 2× gap is GIL serialisation between threads + ORT serialising GPU
calls across stages.

### Defaults that *are* enabled

- **TensorRT** for the inswapper if `tensorrt-cu12` is installed in
  the env (it's listed in `requirements-webapp.txt`). First run
  builds an engine (~60–90 s); cached to `webapp_jobs/.trt_cache/`
  per-architecture (e.g. `..._sm89.engine` for RTX 4090). Worth
  ~30–50 % uplift on the swap step. If `tensorrt` isn't present we
  stay on CUDA — never CPU (see issue #8).
- **det_size=480** for face detection (env: `FACESWAP_DET_SIZE`).
  ~1.7× faster than the model's native 640.
- **4-stage pipeline**: reader → detect → swap → writer threads with
  `queue.Queue(maxsize=128)` between them. Reader runs ~5–10 s ahead
  of swap so GPU isn't starved by I/O hiccups.

Install TRT (if missing):
```powershell
conda run -n dlc pip install tensorrt-cu12
```

### Tuning knobs (env vars)

| var | default | what it does |
|---|---|---|
| `FACESWAP_FACE_MODEL` | `buffalo_l` | Face analyser bundle. `buffalo_s` is ~2× faster but produced visible quality regression on test footage; left as opt-in. |
| `FACESWAP_DET_SIZE`   | `480`       | Face detector input (square). 640 = native, slower; 320 = even faster, misses small faces. |

### Tried-and-rejected speedups

- **`buffalo_s` face model** (env-toggle still works): less accurate
  embeddings → more reference-match misses → visible swap flicker
  on tested footage. Worth re-trying on cleaner sources.
- **CUDA Graph capture in TRT** (`trt_cuda_graph_enable: True`): no
  measurable gain on this workload, and it triggered a regression in
  one earlier broken commit. Disabled.
- **Frame-skip + interpolate**: process every 2nd frame, copy swap to
  in-between. Doubles speed but visible artefacts on fast cuts —
  unacceptable for music videos.

### Speedups still on the table

- **NVDEC for video decode** — `cv2.VideoCapture` does software
  decode. Switching to PyAV with `hwaccel=cuda`, or piping ffmpeg
  with `-hwaccel cuda` to a numpy buffer, would save the 3–5 ms/frame
  CPU decode + an unnecessary CPU→GPU memcpy.
- **Avoid `frame.tobytes()`** — currently 3–5 ms memcpy per 1080p
  frame to convert numpy → bytes for ffmpeg. Could pass the numpy
  buffer directly via `ffmpeg.stdin.write(frame.data)` (zero-copy of
  the underlying memory).
- **TRT for the static-shape sub-models** of the face analyser
  (`genderage`, `w600k_r50`, `2d106det`). Detection sub-model has
  dynamic shapes and stays on CUDA.
- **Multiprocessing** if the GIL becomes the floor: split detect
  and swap into separate processes with shared-memory frame buffers.
  Bigger refactor.

---

## Things NOT to commit

The `.gitignore` excludes them, but be deliberate. The repo is **public**.

| Path | Why |
|---|---|
| `source/` | Personal face photos |
| `songs/` | Likely copyrighted music videos |
| `out/`   | Derivative works of copyrighted material |
| `webapp_jobs/` | User uploads (potentially identifying) |
| `*.onnx`, `*.pth` | Big model files; download via `setup.ps1` |
| `facefusion/`, `deep-live-cam/` | Cloned upstream repos with their own git |
| `.venv/`, `__pycache__/`, `*.log` | Build artefacts |
| `.claude/`, `.playwright-mcp/` | Agent-tool runtime state |
| `.commit-msg.tmp` | PowerShell-heredoc workaround for `git commit -F` |

If you're adding a new file type that should be excluded, update
`.gitignore` in the same commit.

---

## Workflow tips for AI agents

- **Restart the webapp after editing `webapp.py`.** It's a Flask dev
  server in non-reload mode (we run with `use_reloader=False` to avoid
  double-loading the GPU models).
- **Models stay loaded across requests.** The first request after a
  restart pays the ~30 s model warm-up; subsequent requests are fast.
  Pre-warming is started on a background thread at server start
  (`threading.Thread(target=_ensure_models, daemon=True).start()`).
- **Single job at a time.** Models are global, so concurrent jobs would
  serialise on the GIL anyway. If you need true concurrency, give each
  job its own `FaceAnalysis` instance and accept the VRAM cost.
- **Use `conda run -n dlc python <script>`**, not `&
  <env>\python.exe <script>` — the latter doesn't set `CONDA_PREFIX`,
  which some upstream bootstrap code (e.g. FaceFusion's `conda.py`)
  depends on.
- **For Playwright debugging** of the live page, the
  `mcp__plugin_playwright_*` tools work against
  `http://localhost:8080/` directly. Useful for catching browser-side
  bugs (autoplay, hls.js errors, network 404s) that the server logs
  won't reveal. Inspect via `browser_evaluate()` to read
  `player.readyState`, `player.paused`, `player.error`.
- **Don't add a JS framework.** The viewer page intentionally avoids
  React/Vue — it's plain HTML + ~150 lines of vanilla JS. Reaching
  for a build step is overkill here.
- **Don't add a database.** Job state lives in an in-memory dict and
  per-job dirs on disk. Adding SQLite/Postgres for a single-user app
  is overkill.
- **PowerShell quoting hell**: when shelling out from Python or
  invoking via `Bash`, prefer building arg arrays and using
  `subprocess.Popen([...])` (or PowerShell's array splatting `& exe
  @argList`) over single-string commands. Quoting backslashes in
  Windows paths inside double-quoted PowerShell strings is a tarpit.

---

## Smoke-test before committing

There's no formal test suite. The smoke test is:

1. `conda run -n dlc python webapp.py` — wait for "starting on"
2. `Invoke-WebRequest http://localhost:8080/` should return 200 + ~14
   KB
3. Open <http://localhost:8080/> in a browser. Drop a known-good source
   image + a short MP4 (10-30 s).
4. Watch all 5 phase pills go green: load → detect → reference →
   stream → finalise.
5. After "Buffering 15/15s" message clears, the muted HLS player should
   start. Click "Click to unmute" → audio plays.
6. Confirm the "Download MP4" link returns a file you can play in VLC.
7. Open <http://localhost:8080/job/<id>> in a fresh tab — it should
   replay correctly as a VOD with full scrubber.
8. Check `webapp_jobs/<id>/ffmpeg.log` shows no errors.

If steps 4–8 all pass, the change is OK to commit.

For the FaceFusion path (path A), run:
```powershell
.\swap-song.ps1 -Source .\source\test.jpg -Target .\songs\test.mp4 -Quality fast
```
and confirm `out\test_swapped.mp4` opens in VLC.

---

## How to update upstream tools

When `facefusion` or `deep-live-cam` ships a new version:

```powershell
# FaceFusion
cd facefusion
git pull
cd ..
conda run -n faceswap python facefusion\install.py --onnxruntime cuda --force-reinstall
# re-apply our cuDNN patch — setup.ps1 detects this and skips if already patched:
.\setup.ps1 -SkipDeepLiveCam
```

```powershell
# Deep-Live-Cam
cd deep-live-cam
git pull
cd ..
conda run -n dlc pip install -r deep-live-cam\requirements.txt
.\setup.ps1 -SkipFaceFusion
```

Test with the smoke test above. If the upstream API has changed and
breaks `webapp.py`, pin the upstream commit in `setup.ps1` (replace the
`git clone --depth 1` with `git clone && git checkout <sha>`).

---

## Security notes

This is built as a **single-user local tool**. It's not safe to expose
to the public internet as-is.

- **No authentication.** Anyone who can reach the port can upload
  files and consume your GPU.
- **Binds to `0.0.0.0`** for LAN convenience. If you don't want LAN
  access, change `app.run(host="0.0.0.0", port=8080)` to
  `host="127.0.0.1"` in `webapp.py`.
- **No upload type validation** beyond MIME type sniffing by the
  browser. The backend trusts that what you uploaded is image / video.
  ffmpeg + OpenCV are the actual parsers; both are robust to malformed
  input but not bulletproof. Don't accept uploads from untrusted
  sources.
- **No rate limiting.** A bad client could DOS by uploading huge
  files. The 4 GB limit on `MAX_CONTENT_LENGTH` is the only guard.
- **HLS files are served unauthenticated** at predictable
  `/job/<uuid>/hls/...` URLs. UUIDs are random hex (12 chars / 48
  bits), so they're unguessable, but anyone who knows a job ID can
  watch its stream.

For real production:

- Put it behind a reverse proxy (nginx, Caddy) with HTTPS + auth.
- Replace `app.run(...)` with `waitress-serve --port=8080 webapp:app`
  on Windows or `gunicorn` on Linux.
- Add per-user job isolation; the current single-job model is for
  one user.
- Consider [`flask-talisman`](https://pypi.org/project/flask-talisman/)
  for security headers, [`flask-limiter`](https://pypi.org/project/Flask-Limiter/)
  for rate limiting.

---

## Known limitations

- **Single GPU, single job at a time.** Concurrent jobs would share
  the same `FaceAnalysis` instance, so the second job's calls would
  block on the first.
- **Slower than real-time on 1080p+ video.** A 4-minute 1080p song
  takes ~6-8 minutes wall-clock to process on RTX 4090.
- **No frame interpolation.** Faces that aren't detected in a frame
  pass through unswapped (you'll see brief "original face" frames in
  fast cuts).
- **Audio in HLS stream is just the original.** It's not modified for
  anything (no voice changing, no lip-sync); the only thing edited is
  the visual.
- **Browser support: Chrome/Firefox/Safari/Edge with hls.js or native
  HLS.** Internet Explorer doesn't work. Mobile Safari plays HLS
  natively but the autoplay overlay click is required.
- **Windows-first.** Linux + macOS would work after porting `setup.ps1`
  to bash and the FaceFusion/DLC `os.add_dll_directory` patches to
  `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH`. Not done.
- **No `<EXT-X-ENDLIST>` until ffmpeg fully exits.** Until then, hls.js
  treats the stream as live; once the marker arrives, it switches to
  VOD mode automatically (full scrubber, replay).
- **Models download on first run.** The InsightFace `buffalo_l`
  bundle (~290 MB) downloads silently to `~/.insightface/models/`
  the first time `FaceAnalysis(name='buffalo_l')` runs. There's no
  progress bar.
