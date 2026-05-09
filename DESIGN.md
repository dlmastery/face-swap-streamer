# Design

This document explains the non-obvious decisions in the codebase. Read it
after the [README](README.md) — that one tells you what the project does;
this one tells you *why it's built that way*.

## High-level dataflow

```
┌────────────────────────┐
│  Browser (upload page) │  drag-and-drop two face slots + 1 video,
└──────────┬─────────────┘  hls.js for live playback
           │ POST /start  (multipart: source[]=face1.jpg, source[]=face2.jpg, target=video.mp4)
           ▼
┌────────────────────────┐
│  Flask (webapp.py)     │  spawns one orchestrator thread per job
└──────────┬─────────────┘
           │
   ┌───────▼─────────────────────────────────────────────────┐
   │  _run_job orchestrator                                  │
   │  • for each uploaded source: detect face, gender, age   │
   │  • single video scan: collect every face's              │
   │    (frame_idx, gender, embedding, area*det_score)       │
   │  • per-source greedy cluster assignment — each source   │
   │    claims the largest unused cluster of its gender,     │
   │    so two F sources won't both target the same actress  │
   │  • spawn ffmpeg (raw BGR pipe + target.mp4 audio        │
   │    -> tee muxer -> HLS + fragmented MP4)                │
   │  • spin up 4-stage pipeline threads                     │
   └───────┬─────────────────────────────────────────────────┘
           │
   ┌───────▼─────────────────────────────────────────────────┐
   │  4-stage pipeline (one thread each, queue.Queue(128))   │
   │                                                         │
   │   reader   ──(read_q)──▶  detect  ──(detect_q)──▶       │
   │     │                       │                           │
   │     │                       │                           │
   │  cv2.read              fa.get + np.argmax               │
   │  raw BGR               -> (frame, picks)                │
   │                                                         │
   │       main worker  ──(write_q)──▶  writer               │
   │            │                          │                 │
   │       sw.get per pick           ffmpeg.stdin.write      │
   │       paste_back                                        │
   └───────┬─────────────────────────────────────────────────┘
           │
   ┌───────▼─────────────────────────────────────────────────┐
   │  ffmpeg (tee muxer)                                     │
   │   • h264 + AAC, one encode pass, two outputs            │
   │   • output A: HLS playlist.m3u8 + seg_NNNNN.ts          │
   │   • output B: fragmented MP4 (writeable progressively)  │
   └─┬──────────────────────────────────┬────────────────────┘
     │                                  │
     ▼                                  ▼
  /job/<id>/hls/playlist.m3u8     /job/<id>/download
  (browser <video> via hls.js)    (muxed MP4 with audio)
```

The pipeline has three bounded queues. With Q_DEPTH=128 and 1080p
frames, that's ~1.6 GB total in-flight memory. With a fast SSD and
RAM the reader runs ~5–10 s ahead of the swap, so the GPU is never
starved by transient I/O hiccups (HLS segment flushes, page-cache
writes, fa.get latency spikes).

## Why two conda envs?

FaceFusion and Deep-Live-Cam pin **mutually incompatible** dependency
versions:

| Package          | FaceFusion 3.6 | Deep-Live-Cam 2.1 |
|------------------|----------------|-------------------|
| numpy            | 2.2.1          | < 2.0             |
| onnx             | 1.21.0         | 1.18.0            |
| onnxruntime-gpu  | 1.24.4         | 1.23.2            |
| protobuf         | 7.34.1         | 4.25.1            |

Installing both into one env triggers downgrade churn that breaks at least
one of them. We use:

- `faceswap` (Python 3.12) — for FaceFusion (Path A)
- `dlc` (Python 3.11) — for Deep-Live-Cam **and** the web app
  (`webapp.py` reuses DLC's `insightface` install)

This keeps each tool happy without conflicts.

## Why not `python -m venv`?

We initially tried `python -m venv` from Anaconda's Python 3.12. On
Windows, the resulting venv's `sys.path` includes `<anaconda>\Lib`, so
`pip install` operates against Anaconda's `site-packages` instead of the
venv's. This corrupted the user's Anaconda base env (pillow downgrade,
pydantic-core downgrade) before we noticed.

Conda envs **are** properly isolated on Windows. Always use `conda create
-n <env>`, never `python -m venv` if your base Python is from Anaconda.

## cuDNN DLL discovery

`onnxruntime-gpu` on Windows requires cuDNN 9.x and cuBLAS 12.x as
**native DLLs**. The wheel doesn't bundle them — you have to install
them separately as pip packages:

```
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 \
            nvidia-cuda-nvrtc-cu12 nvidia-curand-cu12 nvidia-cufft-cu12
```

This installs DLLs into `<env>/Lib/site-packages/nvidia/<lib>/bin/`. But
**Python 3.8+ on Windows uses a "secure DLL search" policy** that ignores
`PATH` for native imports — you have to call `os.add_dll_directory(...)`
**and keep the returned cookie alive**.

Both upstream tools were patched to do this:

```python
# Crucial: store the cookies so they don't get GC'd, otherwise the
# directory is removed from the search path immediately.
_dll_cookies = []
for sub in ("cudnn", "cublas", "cuda_runtime", "cuda_nvrtc",
            "curand", "cufft", "nvjitlink"):
    bin_dir = os.path.join(site_packages, "nvidia", sub, "bin")
    if os.path.isdir(bin_dir):
        _dll_cookies.append(os.add_dll_directory(bin_dir))
```

A common bug: writing `[os.add_dll_directory(p) for p in dirs]` and
discarding the list. The cookies are GC'd, the registration disappears,
and onnxruntime falls back silently to CPU. Always assign to a kept
reference.

## Reference-embedding face matching

A naive face-swap script picks "the largest face in each frame". For a
duet (two protagonists), this flips the swap target between frames as
camera angles change. The fix has two layers:

### Layer 1 — auto-detect source gender

```python
src_face = max(fa.get(src_bgr), key=largest_area)
gender   = src_face.sex   # 'M' or 'F'
```

We swap only onto faces of the same gender as the source. This catches
~90% of the duet problem.

### Layer 2 — embedding-based reference lock

Gender prediction has ~5–10% error rate on profile / low-res / partially
occluded faces. So we additionally lock onto a specific person's face
**embedding** auto-extracted from the target video:

```python
# Sample one frame every 2 sec, collect male-face embeddings
candidates = []
for t in range(0, total_frames, fps * 2):
    cap.set(POS_FRAMES, t)
    fr = cap.read()
    males = [f for f in fa.get(fr) if f.sex == src_gender]
    if males:
        best = max(males, key=lambda f: face_width(f) * f.det_score)
        if face_width(best) >= 50:
            candidates.append((score, best.normed_embedding, t))

# Cluster: each candidate "votes" for similar candidates (sim > 0.30).
# Pick the centroid of the largest cluster — that's the recurring lead.
embs    = np.stack([c[1] for c in candidates])
votes   = (embs @ embs.T > 0.30).sum(axis=1)
winner  = argmax(votes * scores)
ref_emb = candidates[winner][1]
```

Then per-frame, instead of "largest face", we pick the face whose
embedding has the highest cosine similarity to `ref_emb`. We also reject
the swap entirely if the best similarity is below a threshold (default
0.22) — that drops shots where the lead isn't on screen, instead of
swapping onto an extra.

Cosine threshold of 0.22 is forgiving (handles angle/lighting variation);
0.30 is stricter (might drop borderline shots). Pick based on whether
you want fewer false-positives or fewer dropouts.

## HLS streaming with synchronised audio

### The problem

MJPEG (multipart-JPEG over HTTP) is the simplest "live frames to
browser" format, but it carries no audio. We want the user to **hear**
the song while the swap is processing.

### Why not WebRTC / DASH

- WebRTC: lowest latency but needs signaling + peer connection
  setup (~250 lines of glue + an STUN server). Overkill for localhost.
- DASH: similar to HLS but with more complex segment management. HLS
  works in every browser via hls.js.

### How the pipeline works

A single ffmpeg subprocess does it all:

```
ffmpeg
  -f rawvideo -pix_fmt bgr24 -s WxH -r FPS -i pipe:0      # input 0: BGR from python
  -i target.mp4                                            # input 1: original (for audio)
  -map 0:v:0 -map 1:a:0?                                   # take video from 0, audio from 1
  -c:v libx264 -preset ultrafast -tune zerolatency
  -pix_fmt yuv420p -profile:v high -level 4.1
  -g 50 -keyint_min 50 -sc_threshold 0                     # 2-sec keyframe interval
  -c:a aac -b:a 192k -ac 2 -ar 44100
  -shortest
  -f tee
  "[f=hls:hls_time=2:hls_list_size=0:hls_flags=independent_segments+append_list:
    hls_segment_filename=hls/seg_%05d.ts]hls/playlist.m3u8
   |
   [f=mp4:movflags=+faststart+frag_keyframe+empty_moov]swapped.mp4"
```

Key flags:

- `-tune zerolatency` + `-preset ultrafast`: trade compression efficiency
  for encode speed (we're doing it live)
- `-g 50` (= 2 sec at 25 fps): forces a keyframe every 2 sec, so each
  HLS segment is independently decodable. This is what `independent_segments`
  in the HLS playlist requires.
- `-shortest`: end output when the shorter input ends. Audio is the long
  one (full song); the shorter one is whatever pipe:0 gives us when we
  `close()` it.
- `movflags=+frag_keyframe+empty_moov`: write the MP4 as fragmented MP4
  so it's playable / downloadable while still being written. Without
  these, the moov atom only gets written at file close, so a partial MP4
  is unplayable.

### Pacing

Source video is 25 fps. Our swap pipeline runs at ~17–25 fps wall-clock
on an RTX 4090 (depends on resolution). We feed frames to ffmpeg as fast
as we can swap them; each frame is **labelled** as 1/25 sec apart via
`-r 25`, so the output timestamps are correct regardless of feed
rate. ffmpeg buffers audio in memory while it waits for video frames.

For a 4-minute song at 17 fps wall-clock, total processing is ~5.9 min
real time. The browser plays at 1× speed; if it catches up to the live
edge it'll buffer momentarily until more segments arrive.

## Job lifecycle

```
queued → loading_models → detecting_source → finding_reference →
streaming → finalising → done
                                                    │
                                                    └─ error (terminal)
```

The Flask `/job/<id>/status` endpoint returns the current phase + a
human-readable message + counters (`current_frame`, `swap_count`,
`proc_fps`). The viewer page polls it at ~2.5 Hz and updates the UI.

When `phase == streaming`, the viewer attaches hls.js to
`/job/<id>/hls/playlist.m3u8`. The HLS playlist initially contains few
segments (live tail), and grows as ffmpeg writes more. hls.js
re-fetches the playlist periodically.

When `phase == done`, the viewer reveals the prominent download card
linking to `/job/<id>/download` (the fragmented MP4 with audio). The
HLS player keeps working too — it auto-detects the endlist marker and
becomes a regular VOD player (full scrub bar etc).

## Trade-offs we made

| Decision | Pros | Cons |
|---|---|---|
| Embedding lock instead of face tracking | Robust to cuts | No tracking metadata (each frame is independent) |
| Single global model instances (one face analyser, one swapper) | Lower VRAM, faster jobs | Single-job-at-a-time webapp |
| Flask dev server | Zero ops complexity | Not for production behind a domain |
| HLS via tee muxer | One encoding pass, two outputs | Adds 2-second segment latency |
| Bind to `0.0.0.0` | LAN access | No auth — keep on a trusted network |

## Future work

- WebRTC track instead of HLS for sub-second latency (uses aiortc +
  signalling)
- Multi-user job isolation (each upload gets its own face analyser
  instance, or queue jobs)
- TensorRT execution provider for the inswapper (FP16 → ~30 % faster)
- Frame-skipping mode that processes every Nth frame and interpolates
  the rest, for real-time playback rates on slower GPUs
