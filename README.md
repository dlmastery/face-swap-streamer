# Faceswap streamer

> Drop a photo of yourself (or two — one per duet lead) + a video. The web app
> auto-detects each face's gender, locks onto the matching person in the footage,
> and **streams the swap live to your browser with synchronised audio** — frame
> by frame, while it's still being processed. When the run finishes, you can
> download the finished MP4.

Latest perf on RTX 4090 with TensorRT inswapper + 4-stage thread pipeline:
8–13 fps on 1080p, 18–25 fps on 480p, audio in the live stream.

A self-contained Windows + NVIDIA stack built around three open-source
face-swap tools, with a Flask web app on top for live HLS streaming.

![browser preview placeholder — the home page is a glass-morphism card with
two drag-and-drop zones (face / video) and an animated gradient
background; the viewer page shows the live HLS player with phase pills
(load models → detect face → find reference → stream → finalise) and a
prominent "Download MP4 (with audio)" card when the swap completes.]

## What's in the box

| Path | Tool | What it does | When to use |
|---|---|---|---|
| **A** | [FaceFusion 3.6](https://github.com/facefusion/facefusion) | Offline render of an MP4 with full processor chain (`face_swapper`, `face_enhancer`, `expression_restorer`, `frame_enhancer`) | Highest quality output you can get; great for archival renders |
| **B** | [Deep-Live-Cam 2.1.2](https://github.com/hacksider/Deep-Live-Cam) | Real-time webcam / GUI-based playback swap | Live calls, OBS virtual camera |
| **C** | This repo's `webapp.py` | **Web UI: upload → live HLS stream with audio in browser → download** | What this README is mainly about |

Wrappers / helpers for paths A and B are included as PowerShell scripts.

## Why HLS streaming with audio

The naive way to "stream a face swap" is multipart-JPEG (MJPEG) — but MJPEG
has no audio channel. This project pipes raw BGR frames into a single
`ffmpeg` subprocess that:

1. Encodes h264 video + AAC audio (audio sourced directly from the original
   target file via `-i target.mp4 -map 1:a:0?`)
2. Outputs **two destinations** in one encoding pass via the `tee` muxer:
   - **HLS playlist** (`playlist.m3u8` + `seg_*.ts`) for the live browser stream
   - **Fragmented MP4** for the final download (writeable progressively)
3. Browser plays the HLS stream via [hls.js](https://github.com/video-dev/hls.js/)
   in a regular `<video>` element

Result: audio is in the stream **while it's still being processed**, not just
in the post-processed download.

## Requirements

- Windows 10/11
- NVIDIA GPU with CUDA 12-compatible driver (R535+ recommended)
- Anaconda or Miniconda
- Git
- ~10 GB free disk for models + envs

Tested on RTX 4090 Laptop, driver 595.97, Windows 11.

## Install

```powershell
git clone https://github.com/dlmastery/faceswap.git
cd faceswap
.\setup.ps1
```

`setup.ps1` will:

1. Create two isolated conda envs (`faceswap` Python 3.12 + `dlc` Python 3.11
   — separate because their pinned numpy/protobuf/onnxruntime versions
   conflict)
2. Clone FaceFusion + Deep-Live-Cam into subdirectories
3. Install all Python dependencies including the NVIDIA CUDA runtime libs
   (`nvidia-cudnn-cu12`, `nvidia-cublas-cu12`, etc.) that `onnxruntime-gpu`
   needs but doesn't bundle on Windows
4. Download the inswapper-128 + GFPGAN model weights
5. Apply two small patches that fix Windows DLL search for both upstream
   tools (see [DESIGN.md](DESIGN.md#cudnn-dll-discovery))

Total install size is around ~9 GB (mostly tensorflow + torch + the ONNX
models).

## Usage

### Web app (the main thing)

```powershell
conda run -n dlc python webapp.py
```

Open <http://localhost:8080/> and drag in a face image (Face #1, required) and
optionally a second one (Face #2, for duet swaps). Pick a video. The page
shows a phase spinner while the workflow runs:

1. **load models** — face analyser + inswapper into VRAM (one-time, ~30 s
   model warmup; one-time ~60–90 s TensorRT engine build on first run)
2. **detect your face** — gender/age detection from each uploaded source
3. **find target person** — single video scan extracts a reference cluster
   per source: each one is matched to the largest unused cluster of its
   gender (so two leads of a duet won't compete for the same target)
4. **stream** — HLS player appears, audio plays from the start, swap is
   happening live frame-by-frame, pre-buffer 15 s before playback so it
   doesn't stall when processing dips below realtime
5. **finalise** — ffmpeg writes the HLS endlist marker + closes the MP4

When the run completes, the "Download MP4 (with audio)" card appears and
the player swaps to the muxed file (full scrub bar, replay).

The live stream starts muted because browsers block autoplay-with-sound
without a user gesture; click the small "🔊 Click to unmute" pill in the
bottom-left of the player to turn audio on.

### FaceFusion CLI (Path A — best quality)

```powershell
# one song
.\swap-song.ps1 -Source .\source\me.jpg -Target .\songs\kesariya.mp4

# cinema preset + 2× upscale
.\swap-song.ps1 -Source .\source\me.jpg -Target .\songs\kesariya.mp4 `
    -Quality cinema -Upscale -OpenWhenDone

# whole folder
.\swap-album.ps1 -Source .\source\me.jpg -SkipExisting
```

### Deep-Live-Cam (Path B — real-time GUI)

```powershell
.\play-song.ps1
```

In the GUI: select your face → switch target to **Video** → pick an MP4 →
click **Live** for real-time playback (or **Start** for offline render).

### OBS virtual webcam (Path C)

See [`OBS-setup.md`](OBS-setup.md) for using a swapped MP4 as a virtual
webcam in Discord / Zoom / Teams.

## Repo layout

```
faceswap/
├── webapp.py               # Flask web app (HLS streaming + audio + download)
├── stream-swap.py          # CLI version of the streaming pipeline (ffplay output)
├── swap-song.ps1           # PS wrapper around FaceFusion CLI
├── swap-album.ps1          # Batch over a folder of MP4s
├── play-song.ps1           # Launch Deep-Live-Cam GUI
├── extract-ref.py          # Helper: scan a video for the clearest face of a gender
├── probe.py                # Compatibility check on (image, video) pair
├── test-cuda.py            # Verify onnxruntime CUDA actually loads
├── test-cuda-dlc.py        # Same, in the DLC env
├── setup.ps1               # One-shot installer
├── requirements-webapp.txt # pip deps for the dlc env
├── requirements-facefusion.txt
├── README.md               # this file
├── DESIGN.md               # architecture + decisions
└── OBS-setup.md            # Path C guide
```

`facefusion/`, `deep-live-cam/`, model files, your photos, your songs, and
your output directories are all `.gitignore`'d — `setup.ps1` provisions
the upstream code and models locally.

## Privacy / licensing

- **Don't commit your face photo or copyrighted music videos** to the repo —
  the `.gitignore` excludes `source/`, `songs/`, `out/`, and `webapp_jobs/`
  by default.
- Model weights (`inswapper_128_fp16.onnx`, `GFPGANv1.4.pth`) come from
  upstream releases and are downloaded by `setup.ps1`.
- This is a **single-user local tool**. The Flask dev server has no auth and
  binds to `0.0.0.0` for LAN convenience; if you need real production
  hosting, put it behind a reverse proxy with auth.
- This project is intended for personal entertainment and education with
  consent. Do not use it to make non-consensual deepfakes.

## License

Code in this repo is MIT. Upstream tools have their own licenses
([FaceFusion: OpenRAIL-AS](https://github.com/facefusion/facefusion/blob/master/LICENSE.md),
[Deep-Live-Cam: AGPL-3.0](https://github.com/hacksider/Deep-Live-Cam/blob/main/LICENSE),
[InsightFace inswapper: MIT](https://github.com/deepinsight/insightface)) — read
them before redistributing.
