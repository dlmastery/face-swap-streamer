# Architecture

Deep-dive into how the system is wired. If you're new, read
[../README.md](../README.md) first, then [../DESIGN.md](../DESIGN.md)
for the *why*, then this for the *how*.

---

## 1. System map

```
                                 ┌──────────────────────────────┐
                                 │   conda env: faceswap        │
                                 │   Python 3.12                │
                                 │                              │
   FaceFusion CLI (path A)  ◄────┤   facefusion/facefusion.py   │
                                 │   onnxruntime-gpu 1.24.4     │
                                 │   numpy 2.x  protobuf 7.x    │
                                 └──────────────────────────────┘

                                 ┌──────────────────────────────┐
                                 │   conda env: dlc             │
                                 │   Python 3.11                │
   Deep-Live-Cam GUI (path B)◄──┤                              │
                                 │   deep-live-cam/run.py       │
                                 │   onnxruntime-gpu 1.23.2     │
   Web app (path ★) ────────────►│   numpy 1.26  protobuf 4.x   │
                                 │   tensorrt-cu12, flask       │
                                 │   insightface 0.7.3          │
                                 │   webapp.py + stream-swap.py │
                                 └──────────────────────────────┘

                                 ┌──────────────────────────────┐
                                 │   GPU (NVIDIA, CUDA 12)      │
                                 │                              │
                                 │   • TensorRT 10.x runtime    │
                                 │   • cuDNN 9 + cuBLAS 12      │
                                 │     (from nvidia-* pip pkgs) │
                                 │   • inswapper TRT engine     │
                                 │     cached @ webapp_jobs/    │
                                 │       .trt_cache/...sm89.engine│
                                 └──────────────────────────────┘

                                 ┌──────────────────────────────┐
                                 │   ffmpeg 8.1 (Gyan build)    │
                                 │   • h264 + AAC encode        │
                                 │   • HLS muxer                │
                                 │   • MP4 remux (faststart)    │
                                 └──────────────────────────────┘
```

Two conda envs because FaceFusion 3.6 and Deep-Live-Cam pin
incompatible versions of numpy / onnx / onnxruntime / protobuf.
[../CLAUDE.md § "Two conda envs"](../CLAUDE.md#two-conda-envs-this-is-important)
has the version-conflict table.

---

## 2. Web-app dataflow

### 2.1 Job lifecycle

```
queued
  │
  ▼
loading_models      ── _ensure_models() — first run ~30 s warmup;
  │                    inswapper TRT build first time only (~60–90 s)
  ▼
detecting_source    ── per uploaded image: cv2.imread → fa.get → max-area face,
  │                    spec.gender, spec.age, spec.src_face populated
  ▼
finding_reference   ── single video scan, every {2 s × fps} frames sampled,
  │                    candidates of every needed gender collected;
  │                    per-source greedy cluster assignment
  ▼
streaming           ── 4-stage pipeline running, ffmpeg writing HLS +
  │                    .ts segments live
  ▼
finalising          ── streaming ffmpeg sees end-of-stdin, writes
  │                    HLS endlist marker; we then run a remux pass:
  │                    HLS .ts → swapped.mp4 (+faststart)
  ▼
done                ── viewer page swaps live HLS for inline VOD player,
                       reveals download button
```

`error` is a terminal state from any of the above.

### 2.2 Multi-source matching

```
uploaded source images:
   ┌──────────┐    ┌──────────┐
   │ face1.jpg│    │ face2.jpg│
   └─┬────────┘    └─┬────────┘
     │ fa.get        │ fa.get
     ▼               ▼
   {face, F, 40}   {face, M, 44}
            │  │
            │  └────────────────────────────────────────┐
            │                                            │
   ┌────────▼────────────────────────────────────────────▼────────┐
   │  scan target video, sample every 2 s, fa.get each frame      │
   │  → list of candidates: (frame, gender, embedding, score)     │
   └────────┬───────────────────────────┬─────────────────────────┘
            │                           │
            │ {F candidates}            │ {M candidates}
            ▼                           ▼
   ┌────────────────────┐      ┌────────────────────┐
   │ cluster by cosine  │      │ cluster by cosine  │
   │ similarity > 0.30  │      │ similarity > 0.30  │
   │ pick largest unused│      │ pick largest unused│
   │ → ref_emb          │      │ → ref_emb          │
   └────────────┬───────┘      └────────┬───────────┘
                │                       │
                ▼                       ▼
        spec1.ref_emb            spec2.ref_emb

per frame in stream loop:
   tgt_faces = fa.get(frame)               # T faces detected
   tgt_embs = stack of T embeddings        # shape (T, 512)
   ref_embs = stack of source ref_embs     # shape (S, 512)
   sims     = tgt_embs @ ref_embs.T        # (T, S)
   for each tgt face:
       best_src = argmax(sims[t])
       if sims[t, best_src] >= 0.22:
           swap this face with sources[best_src].src_face
       else:
           leave unswapped
```

Same-gender second source: the cluster step masks already-claimed
candidates so a second F source won't pick the same actress.

---

## 3. The 4-stage thread pipeline

Per job, four threads run with bounded queues between them:

```
                    Q_DEPTH=128 (each queue ~ 800 MB at 1080p)

   ┌──────────┐  read_q  ┌──────────┐ detect_q ┌──────────┐ write_q ┌──────────┐
   │  reader  │─────────►│  detect  │─────────►│   main   │────────►│  writer  │
   │ (thread) │          │ (thread) │          │ (thread) │         │ (thread) │
   └──────────┘          └──────────┘          └──────────┘         └──────────┘
        │                     │                      │                    │
        ▼                     ▼                      ▼                    ▼
   cv2.read()            fa.get(frame)          sw.get(frame,         ffmpeg.stdin
                         tgt_embs @ ref_embs    picked, src,          .write(bytes)
                         pick best source       paste_back=True)      one .ts segment
                                                                       per ~2 s
```

End-of-stream: each stage forwards an `END` sentinel down to the next
on its own exit. The main loop sees END from `detect_q` and breaks;
its `finally:` block sets the stop_flag, drains upstream queues,
forwards END to write_q, and joins all three threads in writer →
detect → reader order before releasing the OpenCV capture.

[../CLAUDE.md issue #12](../CLAUDE.md#12-pipeline-thread-coordination)
documents the exact failure modes and ordering rules.

---

## 4. Live HLS + downloadable MP4

### 4.1 During processing

ffmpeg subprocess takes raw BGR frames on stdin + audio from
`target.mp4`, encodes h264 + AAC, and writes **HLS only**:

```
hls/playlist.m3u8
hls/seg_00000.ts   (~2 sec, ~1.2 MB at 480p)
hls/seg_00001.ts
hls/seg_00002.ts
...
```

The browser fetches `playlist.m3u8` via [hls.js](https://github.com/video-dev/hls.js/),
plays segments as they arrive in a `<video>` element. Audio is in
the .ts segments, so the live stream has audio from frame 1.

### 4.2 At end of stream

When the streaming ffmpeg sees end-of-stdin it writes the
`#EXT-X-ENDLIST` marker into the playlist. A second ffmpeg pass then
remuxes the .ts segments into a standard non-fragmented MP4:

```
ffmpeg -y -allowed_extensions ALL -i hls/playlist.m3u8 \
       -c copy -bsf:a aac_adtstoasc -movflags +faststart \
       swapped.mp4
```

`-c copy` = no re-encode (~5 s for a 4-minute song). `-bsf:a
aac_adtstoasc` = repack AAC from ADTS framing (HLS) to MP4 raw frames.
`+faststart` = move moov atom to the front so iOS Safari /
mobile / progressive download work.

The result is **the same content** as the live HLS, just
re-containered. Plays everywhere.

### 4.3 Why not a single ffmpeg with tee output

We tried `[f=hls:...]playlist|[f=mp4:movflags=+frag_keyframe+empty_moov]
swapped.mp4` originally. The fragmented MP4 played fine in the
browser's MSE-based `<video>` element but **failed on every native
player**: phones reported "format not supported", desktop apps played
audio only, ffprobe couldn't parse it.

Lesson: fragmented MP4 is a streaming-protocol format, not a file
format. Always remux to non-fragmented MP4 if the file will be
opened by anything except an MSE-based player.

---

## 5. HTTP API surface

Implemented in `webapp.py`:

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/`                               | Upload form (HTML) |
| `POST` | `/start`                          | Multipart: 1+ `source` images, 1 `target` video. Creates job, returns 302 to viewer |
| `GET`  | `/job/<id>`                       | Viewer HTML (hls.js + status poll loop) |
| `GET`  | `/job/<id>/status`                | JSON: `phase`, `message`, `current_frame`, `total_frames`, `swap_count`, `proc_fps`, `sources[]` (gender, age, ref_frame, …) |
| `GET`  | `/job/<id>/hls/playlist.m3u8`     | HLS playlist — re-fetched periodically by hls.js |
| `GET`  | `/job/<id>/hls/seg_NNNNN.ts`      | HLS media segment — fetched on demand |
| `GET`  | `/job/<id>/file`                  | Inline-served MP4 (`video/mp4`, Range-aware) — `<video>` fallback |
| `GET`  | `/job/<id>/download`              | Same MP4 with `Content-Disposition: attachment` |

The `/hls/...` route whitelists filenames ending in `.m3u8` or `.ts`
and rejects path traversal (`..`, `/`, `\`).

`Cache-Control: no-store` on all HLS responses so the browser doesn't
disk-cache them — but they're still kept in MSE memory buffer
(`backBufferLength: 90 s`).

---

## 6. File layout

```
face-swap-streamer/
├── webapp.py                  ★ Flask app (≈900 lines incl. inline HTML/JS)
├── stream-swap.py             CLI version (ffplay output)
├── extract-ref.py             standalone helper
├── probe.py                   image+video compatibility check
├── test-cuda.py               verify ORT loads CUDA
├── test-cuda-dlc.py           same, in dlc env
├── swap-song.ps1              PS wrapper around facefusion headless-run
├── swap-album.ps1             batch swap-song over a folder
├── play-song.ps1              launch DLC GUI
├── setup.ps1                  one-shot installer
├── requirements-webapp.txt    pip deps for dlc env
├── requirements-facefusion.txt CUDA libs for faceswap env
│
├── README.md                  hero / quickstart
├── USERGUIDE.md               end-user walkthrough
├── DESIGN.md                  architectural decisions
├── CLAUDE.md                  operator's manual + things-that-broke
├── OBS-setup.md               OBS virtual-cam recipe
│
├── docs/
│   ├── ARCHITECTURE.md       this file
│   ├── PERFORMANCE.md
│   ├── TROUBLESHOOTING.md
│   ├── CHANGELOG.md
│   ├── CONTRIBUTING.md
│   └── HACKING.md
│
├── facefusion/                cloned by setup.ps1 (gitignored)
├── deep-live-cam/             cloned by setup.ps1 (gitignored)
├── source/                    user uploads (gitignored)
├── songs/                     user inputs (gitignored)
├── out/                       CLI outputs (gitignored)
├── webapp_jobs/               per-job dirs + .trt_cache/ (gitignored)
└── .gitignore
```

---

## 7. Threading + GIL summary

| Thread | Holds GIL during | Releases GIL during |
|---|---|---|
| Reader     | queue.put, attribute access | `cv2.VideoCapture.read()` (native decode) |
| Detect     | numpy stack/argmax, queue | `fa.get()` (ONNX inference) |
| Main (swap)| queue, frame.tobytes() (memcpy is C but Python-level) | `sw.get()` (ONNX + cv2 paste_back) |
| Writer     | queue | `ffmpeg.stdin.write()` (syscall) |

ORT serialises GPU calls across all four threads (its session is
thread-safe but holds an internal lock per call). So the GPU sees:
fa.get for frame N+2, sw.get for frame N+1, fa.get for frame N+3,
sw.get for frame N+2, … — interleaved, not in parallel.

The win from the pipeline isn't *parallel GPU work*; it's that CPU
work (decode, embedding match, pipe write, paste_back) overlaps with
GPU work, removing the serial wait time the original loop had.

---

## 8. Configuration knobs

Env vars read at server start:

| var | default | effect |
|---|---|---|
| `FACESWAP_FACE_MODEL` | `buffalo_l` | InsightFace bundle. `buffalo_s` = ~2× faster, less accurate; default has higher fidelity. |
| `FACESWAP_DET_SIZE`   | `480`       | Face detector input (square). Native is 640. |

Hard-coded constants you'd edit:

| const | location | default | what |
|---|---|---|---|
| `Q_DEPTH`           | `_run_job` worker setup | 128 | bounded queue depth between pipeline stages |
| `REFERENCE_THRESH`  | `_run_job`              | 0.22 | min cosine similarity for an embedding match to count |
| `PREBUFFER_TARGET`  | viewer JS               | 15 | seconds buffered before pressing play |
| `REBUFFER_TARGET`   | viewer JS               | 8   | seconds buffered before resuming after a stall |
| `hls_time`          | `_spawn_ffmpeg`         | 2   | seconds per HLS segment |
| `MAX_CONTENT_LENGTH`| Flask app config        | 4 GB | upload limit |

---

## 9. Gotchas a future change is likely to hit

- **Don't put `tensorrt` in the providers list unless `import
  tensorrt` succeeds.** ORT silently falls all the way to CPU on first-provider
  failure. See [../CLAUDE.md issue #8](../CLAUDE.md#8-onnx-runtime-fallback-to-cpu-silent-disaster).
- **Don't put absolute Windows paths in ffmpeg's tee URL.** Drive-
  letter colons collide with the option separator. Use cwd + relative
  paths. See [../CLAUDE.md issue #2](../CLAUDE.md#2-ffmpeg-tee-muxer--windows-paths).
- **Don't write a fragmented MP4 as the downloadable file.** It plays
  in browsers via MSE but breaks on every native player. Use HLS for
  live, remux to standard MP4 at the end. (This whole document
  exists partly because we hit this.)
- **Don't `os.add_dll_directory(...)` without storing the cookie.** Cookie
  GC removes the directory from search path immediately. See
  [../CLAUDE.md issue #1](../CLAUDE.md#1-cudnn-dll-discovery-on-windows).
- **Don't set `playStarted = true` before `play()` resolves.** A
  rejected promise leaves you wedged. See
  [../CLAUDE.md issue #14](../CLAUDE.md#14-browser-autoplay-rescue-ux-not-a-crash).
