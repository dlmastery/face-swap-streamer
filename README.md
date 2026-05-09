# 🎭 face-swap-streamer

> **Drop a photo of yourself (or two — one per duet lead), drop a video, watch
> the swap stream live to your browser with synchronised audio while the GPU
> processes it.** When the run finishes, download a standard MP4 that plays
> on every phone, tablet, and desktop player.

A self-contained Windows + NVIDIA stack: **Flask** web app, **HLS** live
streaming with audio, **TensorRT-accelerated** [InsightFace](https://github.com/deepinsight/insightface)
inswapper, a **4-stage thread pipeline** that keeps the RTX GPU fed, and
**1- or 2-source matching** so duets actually work.

```
┌──────────────────────┐  drag-and-drop          ┌─────────────────────────┐
│  http://localhost   │  Face #1 + (Face #2)    │  Flask + InsightFace    │
│                     │ ────────────────────►  │  + TensorRT inswapper   │
│  Live HLS player    │                         │  + 4-stage pipeline     │
│  (audio synced)     │ ◄────────────────────  │  + ffmpeg HLS+remux     │
│                     │  hls.js / `<video>`     │                         │
└──────────────────────┘                          └─────────────────────────┘
```

---

## Table of contents

1. [What it does](#what-it-does)
2. [Quickstart](#quickstart)
3. [How it looks](#how-it-looks)
4. [Architecture](#architecture)
5. [Performance](#performance)
6. [Where the docs live](#where-the-docs-live)
7. [Three CLI fallback paths](#three-cli-fallback-paths)
8. [Privacy + licensing + ethics](#privacy--licensing--ethics)

---

## What it does

| | |
|---|---|
| 📤 Upload | 1 face image (required) + an optional 2nd face for duets, plus a target video |
| 🤖 Auto-detect | Each source's gender + age from InsightFace |
| 🎯 Auto-find | Each source is locked to the largest matching-gender cluster in the video — no manual tagging |
| 🎞️ Live stream | Browser plays HLS (h264 + AAC) **while the swap is still happening**, with original song audio synced |
| 📥 Download | Standard MP4 (h264 + AAC, +faststart, non-fragmented) that plays on iOS Safari, Android, VLC, QuickTime, Windows Media Player, anywhere |
| 🚀 Fast | RTX 4090 hits 8–13 fps on 1080p, 18–25 fps on 480p with TensorRT |
| 🔌 Local | Single-user, runs on `localhost:8080`, no internet, no telemetry |

---

## Quickstart

### 0. Prerequisites

- Windows 10 or 11
- Anaconda or Miniconda
- Git
- NVIDIA GPU with CUDA-12-compatible driver (R535+), ≥ 6 GB VRAM
- ~10 GB free disk

### 1. Clone + install

```powershell
git clone https://github.com/dlmastery/face-swap-streamer.git
cd face-swap-streamer
.\setup.ps1
```

`setup.ps1` provisions everything: two conda envs (`faceswap` Py 3.12 +
`dlc` Py 3.11 — they have incompatible deps), clones FaceFusion +
Deep-Live-Cam, downloads the `inswapper_128_fp16.onnx` + `GFPGANv1.4.pth`
model weights, installs the CUDA pip libs (`nvidia-cudnn-cu12`, …),
applies two patches to upstream tools so they find the cuDNN DLLs on
Windows. ~10 minutes, ~9 GB on disk.

### 2. Start the web app

```powershell
conda run -n dlc python webapp.py
```

Open <http://localhost:8080/>. Drop in your face, optionally a second
face, and a video. Click "Start live swap".

### 3. Watch the live HLS stream (with audio)

The viewer page walks through five phases — `load models → detect your
face → find target person → stream → finalise` — and then transitions
to the live HLS player. After 15 s of pre-buffering it auto-plays muted
(browsers block autoplay-with-sound until you interact); click the
"🔊 Click to unmute" pill bottom-left of the player to hear the song.

### 4. Download the finished MP4

When the finalise phase completes, the page shows a prominent "Download
MP4 (with audio)" card. Standard MP4, plays on every device.

---

## How it looks

### Upload page

```
┌────────────────────────────────────────────────────────────────┐
│   ★ Live face-swap streaming                                   │
│                                                                │
│        Your face, in any video.                                │
│   Streamed live to your browser.                               │
│                                                                │
│   Drop in a photo of yourself and a video. We auto-detect your │
│   gender, lock onto the matching person in the footage, and    │
│   stream the swap with synchronised audio — frame by frame,    │
│   while it processes.                                          │
│                                                                │
│   ┌────────────────────┐        ┌────────────────────┐         │
│   │ 👤 Face #1         │   →    │ 🎬 Target video    │         │
│   │ (required)         │        │                    │         │
│   │ drag image here    │        │ drag mp4 here      │         │
│   └────────────────────┘        └────────────────────┘         │
│   ┌────────────────────┐                                       │
│   │ 👤 Face #2         │                                       │
│   │ (optional, duets)  │                                       │
│   └────────────────────┘                                       │
│                                                                │
│   [    Start live swap    ]                                    │
└────────────────────────────────────────────────────────────────┘
```

### Viewer page (during streaming)

```
┌────────────────────────────────────────────────────────────────┐
│  Faceswap · job · a8c2e91d3f7b                    ← new swap  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │                                                          │ │
│   │             [ live HLS player video frame ]              │ │
│   │                                              ● live · audio │
│   │                                                          │ │
│   │   🔊 Click to unmute                                     │ │
│   └──────────────────────────────────────────────────────────┘ │
│   ████████████████████░░░░░░░░░░░░░░░░░░░░░░  47 %            │
│   progress 3140 / 6664   fps 12.4   swaps 482                  │
│   src1 F/40 (f3700, 22/28)   src2 M/44 (f300, 24/33)           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Architecture

A **4-stage thread pipeline** keeps the RTX GPU fed by overlapping I/O,
detection, swap, and pipe-write:

```
                  Q_DEPTH=128 (each queue ~ 800 MB at 1080p)
   ┌──────┐ read_q ┌───────┐ detect_q ┌──────┐ write_q ┌────────┐
   │reader│───────►│detect │─────────►│ swap │────────►│ writer │
   │ cv2  │        │fa.get │          │sw.get│         │ ffmpeg │
   └──────┘        └───────┘          └──────┘         └────────┘
       │              │                  │                │
       ▼              ▼                  ▼                ▼
  decode mp4     embed match        inswapper_128    pipe to ffmpeg
  (CPU)          (GPU + numpy)      (TRT FP16, GPU)  HLS .ts segments
```

ffmpeg is in turn fed via the tee muxer — actually no, it writes a
single live HLS stream during processing; the downloadable MP4 is a
**second pass** that remuxes the .ts segments via `-c copy -movflags
+faststart` (~5 s) so the result plays in every native player.

Reference matching:
- Source images each get their face's embedding → gender → "find this
  gender's biggest unused cluster in the video"
- Per frame, every detected face's embedding is compared (cosine) to
  every source's reference; argmax picks the best source if it clears
  the threshold

**See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** for the full
picture (job lifecycle, HLS muxing, autoplay rescue, etc.).

---

## Performance

Observed on RTX 4090 Laptop, 16 GB VRAM, with TensorRT inswapper +
det_size 480 + 4-stage pipeline (current default):

| Resolution  | proc fps | GPU avg util |
|---|---|---|
| 480 × 360   | 30 – 45 | ~25 % |
| 640 × 480   | 18 – 25 | ~20 % |
| 1280 × 720  | 12 – 18 | ~30 % |
| 1920 × 1080 | 8 – 13  | ~40 % |

Cumulative speedup history on the same Bollywood test footage:

| commit  | what changed | proc fps |
|---|---|---|
| `e36b4db` | TRT detect + CUDA fallback (no silent CPU)        | 7.5  |
| `8735818` | + async writer thread                             | 10.1 |
| `660e7d1` | + async reader thread, queue depth 32             | 10.8 |
| `0a966ce` | + det_size 480, queue depth 64                    | 11.7 |
| `2a0e0dd` | + 1- or 2-source matching, batched embedding dot  | 12+  |
| `d4fc024` | + 4-stage pipeline (detect on its own thread), Q=128 | 12+  |
| `40ba7a1` | MP4 remux to faststart (no fps impact, fixes mobile playback) | 12+  |

**See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)** for per-stage
timings, env-var tuning (`FACESWAP_FACE_MODEL`, `FACESWAP_DET_SIZE`),
and the speedups that didn't pan out.

---

## Where the docs live

| File | For who | What's in it |
|---|---|---|
| [README.md](README.md) | everyone | this file — overview, quickstart |
| [USERGUIDE.md](USERGUIDE.md) | end users | step-by-step walkthrough, FAQ, what to do when something looks wrong |
| [CLAUDE.md](CLAUDE.md) | AI agents + maintainers | operator manual, things-that-broke list, troubleshooting matrix |
| [DESIGN.md](DESIGN.md) | curious reader | why we built it this way (pipeline, HLS, embedding lock, conda envs) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | contributors | deep architecture: dataflow, threading model, HTTP API, file layout |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | optimisers | per-stage timings, tuning knobs, what to try next |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | end users + ops | symptom → cause → fix table, log file pointers |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | everyone | version history with commit refs and lessons |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | new contributors | how to add features, test, commit style |
| [docs/HACKING.md](docs/HACKING.md) | new developers | onboarding — what to read in what order, code map, common pitfalls |
| [OBS-setup.md](OBS-setup.md) | streamers | use a swapped MP4 as a virtual webcam in Discord / Zoom / Teams |

---

## Three CLI fallback paths

The repo also wraps two upstream tools as PowerShell helpers, in case
you'd rather not use the web app:

| Path | Tool | What it does | Wrapper |
|---|---|---|---|
| **A** | [FaceFusion 3.6](https://github.com/facefusion/facefusion) | Highest-quality offline render with the full processor chain (face_swapper + face_enhancer + expression_restorer + frame_enhancer) | `swap-song.ps1`, `swap-album.ps1` |
| **B** | [Deep-Live-Cam 2.1.2](https://github.com/hacksider/Deep-Live-Cam) | Real-time GUI swap (webcam + virtual camera) | `play-song.ps1` |
| **C** | OBS Studio (separate install) | Loop a swapped MP4 as a virtual webcam for Discord / Zoom / Teams | `OBS-setup.md` |

These are independent of the web app — if you only want the web app,
ignore them.

---

## Privacy + licensing + ethics

- **Single-user local tool.** Flask dev server, no auth. Don't expose
  to the internet without a reverse proxy + auth.
- **Personal photos and copyrighted videos stay on your machine.** The
  `.gitignore` excludes `source/`, `songs/`, `out/`, `webapp_jobs/` so
  you can never accidentally commit them.
- **Models** (inswapper_128, GFPGAN, buffalo_l) come from upstream
  releases under their own licenses (MIT for InsightFace).
- **Don't use this to make non-consensual deepfakes.** This tool is
  for personal entertainment, education, and creative use with consent.
  Many jurisdictions have specific laws about synthetic media of real
  people; know yours.

License: this repo's code is **MIT**. Upstream tools have their own —
[FaceFusion: OpenRAIL-AS](https://github.com/facefusion/facefusion/blob/master/LICENSE.md),
[Deep-Live-Cam: AGPL-3.0](https://github.com/hacksider/Deep-Live-Cam/blob/main/LICENSE),
InsightFace inswapper: MIT, hls.js: Apache-2.0. Read them before
redistributing.
