# CLAUDE.md

Operator's manual for AI agents (Claude Code, Cursor, etc.) and humans
working on this repo. Read [README.md](README.md) for the user-facing
pitch and [DESIGN.md](DESIGN.md) for the architecture deep-dive. This
file tells you **how to build, run, debug, and extend the project
without breaking it**.

If you only have time for one thing, read the [TL;DR](#tldr-getting-it-running),
then [Things that broke before](#things-that-broke-before--dont-re-break-them).

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

Three independent paths share the repo:

| Path | Tool | Purpose |
|---|---|---|
| **A** | FaceFusion 3.6 | Highest-quality offline render via CLI |
| **B** | Deep-Live-Cam 2.1.2 | Real-time GUI swap (webcam / virtual camera) |
| **C** | OBS Studio | Loop a swapped MP4 into a virtual webcam |
| **★** | **`webapp.py`** | **The web app — what most of this repo is about** |

The web app reuses Deep-Live-Cam's `insightface` install but is its own
codepath; it doesn't shell out to FaceFusion or DLC.

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
| `webapp.py` | Flask app — the main artefact. ~700 lines: dataclass `Job`, model loader, worker `_run_job`, `_spawn_ffmpeg`, Flask routes, two big HTML templates (`INDEX_HTML`, `VIEWER_HTML`) |
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

---

## Common tasks (commands you'll use)

### Run the webapp (foreground)
```powershell
conda run -n dlc python webapp.py
```

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

### 11. Browser autoplay rescue (UX, not a crash)

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

Observed on RTX 4090 Laptop (16 GB VRAM):

| Resolution | CUDA-only fps | TensorRT FP16 fps | VRAM |
|---|---|---|---|
| 480×360  | 25-30 | 35-45 | 4.5 GB |
| 640×480  | 18-22 | 28-35 | 4.8 GB |
| 1280×720 | 8-12  | 14-20 | 5.5 GB |
| 1920×1080| 5-8   | 9-13  | 6.2 GB |

Bottleneck is `inswapper_128` inference (the 128×128 swap, then
`paste_back=True`'s warp). InsightFace face detection is faster.

### TensorRT (recommended on RTX cards)

If `tensorrt-cu12` is installed in the env (it's listed in
`requirements-webapp.txt`), `webapp.py` automatically uses TensorRT for
the inswapper. First run builds the engine (~60–90 s); cached to
`webapp_jobs/.trt_cache/` thereafter. Worth ~30–50 % throughput uplift.

Install (if missing):
```powershell
conda run -n dlc pip install tensorrt-cu12
```
~2 GB download. Restart `webapp.py` — it'll auto-detect.

If TRT is **not** present, the code stays on CUDA-only (don't let it
fall back to CPU; see issue #8).

### Other speedups (not enabled by default)

- **NVDEC for video decode** — `cv2.VideoCapture` is software decode.
  Switch to PyAV with `hwaccel=cuda` for ~10–20 % I/O savings.
- **Lower `det_size` to 480** — face detector input. Faster, may miss
  small faces in dance/wide shots.
- **Smaller face model `buffalo_s`** — ~2× faster than `buffalo_l`,
  slightly lower accuracy.
- **Frame-skip + interpolate** — process every 2nd frame, copy swap to
  the in-between. Doubles speed; visible artefacts on fast cuts.

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
