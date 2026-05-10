# faceswap (C++ CLI)

Offline, GPU-accelerated face-swap CLI. Same model stack as the Python
`webapp.py` / FastAPI server (RetinaFace + arcface + inswapper_128_fp16),
but reimplemented in C++17 so it can run without Python and process
multiple videos in parallel without GIL serialization.

## What it does

```
faceswap --male m.jpg --female f.jpg --video clip.mp4 --output out/
faceswap --male m.jpg --female f.jpg --dir clips/ --output out/ --concurrency 3
```

Given one or two source faces (male and/or female), it swaps them onto every
matching person in a target video — or every video in a directory — and
writes muxed `*_swapped.mp4` files (with the original audio) to the output
folder.

## Why this exists alongside the web app

The web app (Flask :8080 / FastAPI :8081) is for interactive single-job /
small-batch use over the network. The CLI is for:

- **Long batches** (50+ videos) where you want to kick it off and walk away.
- **Higher throughput** — the CLI runs *N* full pipelines concurrently on the
  same GPU (`--concurrency N`), saturating GPU + I/O without Python's GIL
  funneling everything through one thread.
- **Embeddable** in shell scripts, scheduled jobs, or air-gapped boxes that
  can't run a web server.

The web app is unchanged; both paths share the same on-disk models.

## Build

This is Windows-first (the rest of the project is). One-time setup:

```powershell
pwsh -File cli/scripts/setup.ps1
pwsh -File cli/scripts/build.ps1
```

`setup.ps1` is idempotent. It installs CMake + MSVC Build Tools via winget,
fetches ONNX Runtime GPU 1.18.1 into `cli/third_party/`, points `OpenCV_DIR`
at the conda `dlc` env's OpenCV, and copies the buffalo_l + inswapper models
out of `~/.insightface/models/` into `cli/models/`.

`build.ps1` loads `vcvars64.bat` into the current shell, runs
`cmake -G Ninja`, and produces `cli/build/faceswap.exe`. Re-run any time you
edit the source.

Linux build is straightforward but not yet scripted — the CMakeLists.txt
already supports it; you just need ORT, OpenCV, and ffmpeg from your distro.

## Flags

| Flag | Purpose | Default |
|---|---|---|
| `--male <jpg>` | Source image for the male face | — |
| `--female <jpg>` | Source image for the female face | — |
| `--video <mp4>` | Single target video (mutually exclusive with `--dir`) | — |
| `--dir <folder>` | Directory of target videos (`*.mp4`/`*.mov`/`*.mkv`/`*.webm`) | — |
| `--output <folder>` | Where swapped MP4s are written | required |
| `--concurrency N` | Videos to process in parallel (batch mode) | `2` |
| `--threads N` | Per-stage queue depth (memory tradeoff) | `128` |
| `--det-size N` | RetinaFace input edge (multiple of 32) | `640` |
| `--det-thresh F` | Detection confidence threshold | `0.30` |
| `--ref-thresh F` | Reference embedding match cosine threshold | `0.18` |
| `--cpu` | Force CPU only (debugging) | off |
| `--trt` | Try TensorRT EP first (experimental) | off |
| `--device N` | CUDA device index | `0` |
| `--models <dir>` | Models root (expects `buffalo_l/` + `inswapper_128_fp16.onnx`) | `models/` |
| `--ffmpeg <path>` | Override ffmpeg binary | PATH lookup |
| `-v`, `-vv` | Verbosity (debug) | `1` |

At least one of `--male` / `--female` is required; exactly one of
`--video` / `--dir` is required; `--output` is required.

## Tuning batch concurrency

`--concurrency` controls how many videos run their pipelines at once on the
same GPU. ORT serializes the actual GPU calls (detector + swapper) so you
won't double inference throughput, but you *will* keep the GPU busier by
overlapping CPU work (frame read, paste-back, encode) across videos.

| GPU VRAM | Suggested `--concurrency` | Notes |
|---|---|---|
| 8 GB  | 2 | safe default |
| 12 GB | 3 | good for RTX 3060/3080-Ti |
| 16 GB | 4 | RTX 4080 sweet spot |
| 24 GB | 6 | RTX 4090; diminishing returns past this |

If you start swapping to disk or seeing `cudaErrorOutOfMemory`, drop one.

## Architecture

```
       reader ─► detect_q ─► detector ─► swap_q ─► swapper ─► encode_q ─► encoder
                                  ▲                  ▲
                              shared GPU          shared GPU
                              (ORT thread-safe)   (ORT thread-safe)
```

- `OnnxSession` wraps `Ort::Session`; one per model file, owned per process.
- `FaceAnalyser` = detector + arcface + genderage (matches insightface's
  `FaceAnalysis(name="buffalo_l")`).
- `Inswapper` = the 128×128 face-swap forward pass + paste-back blend.
- `extract_reference_embeddings()` does the per-video, per-gender clustering
  used by all swap stages — same algorithm as `worker.py` in the Python build.
- `run_streaming()` drives the four-stage pipeline for one video.
- `run_batch()` runs N `run_streaming()` instances on a thread pool.
- `FfmpegEncoder` is a stdin pipe to `ffmpeg.exe` for h264 + aac muxing
  (more reliable than linking libavformat).

## Status

End-to-end working. Build, swap, batch, audio mux all verified on RTX
4090 Laptop, 1080p Bollywood music videos.

| Component | State |
|---|---|
| CMake build (MSVC 14.4, `Visual Studio 17 2022` generator) | ✅ |
| ONNX Runtime 1.18.1 GPU + CUDA EP | ✅ |
| OpenCV 4.10.0 (official Windows pack) | ✅ |
| RetinaFace (SCRFD) decode + NMS | ✅ |
| arcface embedding + genderage | ✅ |
| Reference clustering per-gender | ✅ |
| Inswapper (emap transform + 5-pt similarity-warp + paste-back) | ✅ |
| FfmpegEncoder (h264 libx264 + AAC mux + `+faststart`) | ✅ |
| `--video` single-file mode | ✅ |
| `--dir` batch mode | ✅ |
| `--concurrency N` parallel pipelines on the same GPU | ✅ |

## Bugs caught during the port (logged in CLAUDE.md issues #15-22)

These were all real regressions vs the Python output that took a smoke
test to spot. Read CLAUDE.md before changing anything in `inswapper.cpp`
or `face_analyser.cpp`:

1. **emap transform** — Python multiplies the source embedding by a
   512×512 matrix (last initializer of `inswapper_128_fp16.onnx`)
   before running the model. Skipping this gives near-garbage output
   that *looks* like a face but isn't the right identity.
   `cli/scripts/extract_emap.py` dumps it once into
   `cli/models/inswapper_emap.bin`; `Inswapper::transform_embedding`
   loads + applies it.

2. **Inswapper alignment template at 128×128** is *not* the 112-arcface
   template scaled by 128/112. It's the 112 template **shifted by +8 in
   X**. Wrong template puts chin landmarks ~13 px too low, producing
   visibly-misaligned chin/jaw swaps.

3. **Paste-back blur kernel is `2k+1`**, not `k`. Python uses
   `k = max(mask_size//20, 5); blur = (2k+1, 2k+1)`. Using `k` directly
   leaves a visible blend seam at 1080p.

4. **pimpl + `unique_ptr<Impl>`** requires the constructor + destructor
   to be defined in the `.cpp` (where `Impl` is complete), not
   `= default`'d in the header.

5. **TLS revocation check** can break GitHub release downloads on
   corporate-managed Windows boxes (`schannel: CRYPT_E_NO_REVOCATION_CHECK`).
   `setup.ps1` (and any future curl in this dir) passes `--ssl-no-revoke`.

6. **Anaconda's bundled `ffmpeg.exe`** can't run outside its conda env
   (DLL discovery). Use the gyan.dev winget build via `FFMPEG_BIN`
   instead.
