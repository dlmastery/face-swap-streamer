# Changelog

All notable changes to face-swap-streamer, with the commits that
introduced them and the lessons each one taught.

---

## v0.11 — Multiprocessing webapp variant (33 fps at 1080p)

**`webapp_mp.py`** and **`server/swap_worker.py`** — new files. Original
`webapp.py` kept unchanged so both runtimes ship side-by-side.

### What it is

A drop-in companion to `webapp.py` that replaces the in-process 4-thread
pipeline with N worker processes coordinated via `multiprocessing.shared_memory`.
The master demuxes frames into a fixed-size shared-memory ring, dispatches
to whichever worker is free, and reassembles results in frame order before
writing to ffmpeg. Each worker loads its own `FaceAnalysis` + `INSwapper`
ORT sessions once at job-start, then runs the fused swap+paste on its slice
of frames with no GIL contention against the master or other workers.

Same upload form, same HLS player, same `/job/<id>/status` JSON (plus a
few extra keys: `n_workers`, `worker_warmup_ms`, `paste` timer).

### Why it exists

The single-process Flask sat at ~30 % GPU utilization regardless of workload.
The GIL was serializing every per-frame Python step — even though numpy
and cv2 release the GIL during their C calls, Python-side glue (queue ops,
attribute access, the per-frame match loop) held it just long enough that
the GPU sat idle between calls. Threading inside one process couldn't
break that pattern. Multiprocessing does — each worker has its own GIL,
so paste-back / Python glue from worker A runs concurrently with paste-back
/ Python glue from worker B, and ORT-level GPU calls overlap across
different processes at the CUDA-driver level.

### Numbers (RTX 4090 Laptop, 16 GB VRAM + i9, 1080p Bollywood music video)

| Config | proc_fps | Scaling | GPU util | VRAM |
|---|---|---|---|---|
| `webapp.py` (single-process baseline) | 4.1 | 1.0× | ~30 % | 3 GB |
| `webapp_mp.py` N=2 | 6.5 | 1.6× | 16-48 % | 5 GB |
| `webapp_mp.py` N=4 | 9.8 | 2.4× | 55-81 % | 8 GB |
| **`webapp_mp.py` N=6 + det_size=480 (full song)** | **33.0** ⭐ | **8.0×** | **87-95 %** | 10.5 GB |

The 33 fps number is the steady-state proc_fps on a full 5-minute 1080p
song (7902 frames). Short test clips show lower averages because the
~30 s worker warmup dominates over a 20 s clip.

### How to run

```powershell
$env:FACESWAP_PORT          = "8082"      # different port from webapp.py
$env:FACESWAP_WORKERS       = "6"
$env:FACESWAP_DET_SIZE      = "480"
$env:FACESWAP_VIDEO_ENCODER = "h264_nvenc"
conda run -n dlc python webapp_mp.py
```

Both `webapp.py` and `webapp_mp.py` can run simultaneously on different
ports. Same job-dir layout, same models, same uploads.

### Phases that did NOT make the cut (logged for future reference)

The full perf exploration ran through 5 phases of in-process optimisation
before settling on multiprocessing. The journey is in `docs/perf-bench.md`
and `docs/plans/2026-05-10-flask-gpu-saturation.md`. Notable dead ends:

- **Phase 3 (split paste-back to its own thread)**: regressed 4.4 → 3.5 fps.
  GIL contention + extra queue overhead exceeded the parallelism win when
  paste runs 13× longer than the GPU swap. Threads can't escape the GIL;
  only processes can.
- **Phase 4 (face batching in inswapper)**: deferred. Small leverage (~5 %
  at typical face densities) that doesn't compound with multiprocessing.
- **NVENC output encoder**: ships in `webapp_mp.py` (and is configurable
  via `FACESWAP_VIDEO_ENCODER`), but the writer was never the bottleneck.
  The win is freed CPU for paste-back inside each worker.

### Bugs caught and fixed during the rollout (commits on the
`perf-flask-gpu-saturation` branch)

- **5.5**: `insightface.app.common.Face` couldn't survive `pickle.dumps`
  across `mp.set_start_method("spawn")` — its `__reduce__` ended up calling
  a `None` constructor. Workaround: convert to plain dict in the master
  before pickling; re-wrap as `Face(d)` in the worker (where insightface
  is imported).
- **5.6**: At N≥6, all workers calling `onnx.load()` on the same
  `inswapper_128_fp16.onnx` concurrently triggered intermittent
  `google.protobuf.message.DecodeError: Error parsing message` — Windows
  occasionally hands one of the racers a partial buffer. Mitigation: 1-second
  per-worker stagger via `time.sleep(worker_id)` before opening any models.
  Sufficient for N≤6; N=8 needs a proper `multiprocessing.Lock` around the
  load (deferred — GPU is already saturated at N=6 anyway).

### Production config

For a 16 GB RTX 4090 Laptop + Core i9: `FACESWAP_WORKERS=6
FACESWAP_DET_SIZE=480`. For a 24 GB desktop 4090: same defaults; raising
to N=8 may give +20 % after the model-load lock fix lands.

### Roll-back

`webapp.py` is unchanged. Just stop `webapp_mp.py` and start `webapp.py`
on the same port — same UI, same models, same outputs (slower).

---

## v0.10 — C++ CLI port (offline batch swap, no Python at runtime)

**`cli/`** — fresh top-level directory.

A re-implementation of the entire swap pipeline in modern C++ (MSVC
14.4 / C++20). Same model files as the Python build (buffalo_l +
inswapper_128_fp16). Lives next to the existing Flask + FastAPI
servers, doesn't replace them.

What the CLI does:
- `--male m.jpg --female f.jpg --video clip.mp4 --output out/`
- `--male m.jpg --dir clips/ --output out/ --concurrency 2`
- Detects faces, extracts a per-gender reference cluster from the video,
  swaps every match, muxes the original audio back, writes
  `<basename>_swapped.mp4` with `+faststart`.
- `--concurrency N` runs N independent pipelines on the same GPU; ORT
  serializes the actual inference calls but C++ has no GIL so all the
  CPU work (read, paste-back, encode, queueing) overlaps freely.

Components:
- `OnnxSession` — RAII wrapper around `Ort::Session` with CUDA / TRT /
  CPU fallback chain.
- `FaceAnalyser` — RetinaFace decode + arcface embedding + genderage.
  `decode_retinaface` ports SCRFD anchor + bbox + 5-kps decode + NMS;
  identifies score / bbox / kps outputs by last-dim shape (1 / 4 / 10)
  so naming differences across exporters don't matter.
- `Inswapper` — 5-pt similarity-warp to 128×128, ONNX inference,
  feathered paste-back. Loads the 512×512 emap matrix from
  `cli/models/inswapper_emap.bin` (extracted once via
  `cli/scripts/extract_emap.py`) and applies it before session.run.
- `extract_reference_embeddings` — port of the Python algorithm:
  sample frames, cluster by gender, greedy-pick the largest unclaimed
  cluster per source so two same-gender faces don't fight over the
  same recurring person.
- `BoundedQueue<T>` + `run_streaming` — 4-stage pipeline (reader →
  detect → swap → encode) with bounded queues between stages.
- `run_batch` — N `run_streaming` instances on a thread pool sharing
  the model sessions.
- `FfmpegEncoder` — Win32 `CreateProcess` (POSIX `fork`+`execvp` on
  Linux) + raw BGR pixel pipe + audio-track mux from the source mp4.

Provisioning (`cli/scripts/setup.ps1`): winget cmake + MSVC Build Tools
2022, downloads ORT 1.18.1 win-x64-gpu + OpenCV 4.10.0 windows pack
into `cli/third_party/`, copies buffalo_l + inswapper from
`~/.insightface/models/` into `cli/models/`. ORT and OpenCV downloads
both pass `--ssl-no-revoke` because schannel's strict OCSP fails on
corporate-managed Windows boxes.

Build (`cli/scripts/build.ps1`): loads `vcvars64.bat` automatically,
`cmake -G "Visual Studio 17 2022" -A x64`, `cmake --build --config
Release --parallel`. POST_BUILD steps copy ORT + OpenCV DLLs alongside
the exe.

Bugs that took multiple smoke tests to surface (full prose in
CLAUDE.md issues #15-22):

- **#15 pimpl + `unique_ptr<Impl>`**: `= default` ctor in the header
  triggers `C2027 use of undefined type 'Impl'` because MSVC tries to
  instantiate `unique_ptr::~unique_ptr` at the call site. Fix: declare
  the ctor in the header, define it in the `.cpp`.
- **#16 emap transform**: missing the `latent = src_emb @ emap` step
  before `inswapper.run` produces face-shaped pixels but with the
  wrong identity. emap is the *last* graph initializer of the ONNX
  model, not in any Python file directly.
- **#17 inswapper alignment template**: the 128 template is the 112
  template **shifted by +8 in X**, not scaled by 128/112. Wrong scale
  puts chin landmarks ~13 px too low → visibly bad jaw blending.
- **#18 paste-back blur kernel**: Python's `k` is the half-kernel; the
  actual `cv2.GaussianBlur` size is `(2k+1, 2k+1)`. Using `k` directly
  leaves a soft seam at 1080p.
- **#19 ORT 1.18 forward decls** are `struct`, not `class` (C4099).
- **#20 anaconda ffmpeg.exe** can't run outside its conda env (DLL
  discovery). Use gyan.dev's winget build via `FFMPEG_BIN`.
- **#21 schannel CRYPT_E_NO_REVOCATION_CHECK** intermittently kills
  GitHub release downloads on corporate-managed boxes.
  `--ssl-no-revoke`.
- **#22 PowerShell `$PID`** is read-only. `foreach ($pid in $list)`
  errors with `Cannot overwrite variable PID`. Use `$id`.

Performance on RTX 4090 Laptop, 1080p Bollywood input
(Pal-Pal-Dil-Ke-Paas):
- Single-stream: 16.7 fps wall-clock end-to-end
- `--concurrency 2`: ~33 fps aggregate (~2× scaling)
- vs Python pipeline at the same input: ~8-13 fps single-stream → C++
  CLI is ~1.6× faster single-stream and ~2.5-4× faster at concurrency 2

The CLI is *not* a replacement for the Python web app — it's for
headless batch processing where a browser viewer isn't needed. The web
app remains the reference for *correctness*; if a regression shows up
in C++ output, compare against the Flask `_swapped.mp4` for the same
input.

---

## v0.7 — Real downloadable MP4

**`40ba7a1` Fix: produce a real MP4 (mobile + VLC compatible), not a fragmented one**

Downloaded MP4s before this commit were fragmented (`movflags=
+frag_keyframe+empty_moov+default_base_moof`). They played in the
browser via MSE, but iOS Safari, Android, VLC, QuickTime, Windows
Media Player all rejected them — phones reported "format not
supported", desktop apps played audio only.

Fix: drop MP4 from the streaming ffmpeg's tee output, then in the
finalise phase remux the (already-validated) HLS .ts segments into a
standard non-fragmented MP4 with `+faststart`:

```
ffmpeg -y -allowed_extensions ALL -i hls/playlist.m3u8 \
       -c copy -bsf:a aac_adtstoasc -movflags +faststart \
       swapped.mp4
```

`-c copy` = no re-encode, ~5 s for a 4-minute song. `aac_adtstoasc`
bitstream filter required for AAC ADTS → MP4 raw frames.

**Lesson:** fragmented MP4 is a streaming-protocol format, not a file
format. If the file will be opened by anything except an MSE-based
player, always remux to non-fragmented.

---

## v0.6 — Multi-face + 4-stage pipeline

**`d4fc024` 4-stage pipeline (reader → detect → swap → writer) + Q=128**

Splits face detection from the swap step into its own thread. Now
detection of frame N+1 starts while the swap on frame N is still
finalising — meaningful pipeline overlap (subject to ORT's GPU lock).
On this workload the GPU was already not the bottleneck (avg ~18%
util) so fps gain was minimal, but the layout sets us up for any
future GPU-side optimisation.

**`2a0e0dd` Multi-face swap: upload 1 or 2 source images**

Bollywood duets work end-to-end. Form has two drop-zones (Face #1
required, Face #2 optional). Backend treats them as a list:

- New `SourceSpec` dataclass: `path`, `gender`, `age`, `src_face`,
  `ref_emb`, `ref_frame`, `ref_votes`, `ref_pool`
- `Job.sources` list of these; legacy fields mirror `sources[0]` for
  status-JSON back-compat
- `/start` accepts `request.files.getlist("source")`
- Per-source greedy cluster assignment (each source claims the
  largest unused cluster of its gender, so two same-gender sources
  don't collide)
- Per-frame: stack target embeddings, dot-product against stacked
  reference embeddings, argmax per target → swap with that source

---

## v0.5 — Pipeline + det_size

**`0a966ce` det_size 640 → 480 + Q_DEPTH 32 → 64**

`buffalo_l`'s detector at 480 runs ~1.7× faster than 640 with a small
accuracy drop. `FACESWAP_DET_SIZE` env var to override.

Q_DEPTH 32 → 64 gives the reader several seconds of slack so the GPU
isn't starved by ffmpeg's HLS segment flushes (every 2 s).

`baseline → +35 % (writer) → +7 % (reader) → +8 % (det 480)` —
cumulative +56 % throughput.

**`660e7d1` Async reader thread + larger queues (Q=8 → 32)**

Mirrors the writer thread from the previous commit. Main loop sees
only `queue.get()` / `queue.put()` — no blocking I/O. Cleanup logic in
`finally`: set stop_flag, drain read_q so reader's pending put doesn't
deadlock, send END to writer, join both threads before
`cap.release()`.

Cv2 decode of a 640×480 h264 stream is fast (~2 ms/frame) so the win
was small (+7 %). Larger on 1080p+ inputs.

**`8735818` Async writer thread**

The original main worker loop was blocking on every
`ffmpeg.stdin.write()` — a few ms each frame, but accumulating to a
30–40 % wait when the GPU could've been processing the next frame.

Push frame bytes onto `queue.Queue(maxsize=8)`, drain in a single
writer thread. +35 % throughput, GPU peaks rose from ~35 % to ~59 %.

**Lesson:** before adding more threads, profile to see whether the
*current* threads are blocked on I/O or on compute. The writer was
the cheap win because pipe writes were where the loop stalled.

---

## v0.4 — TRT enabled, no silent CPU fallback

**`e36b4db` Enable real GPU usage: detect TRT before listing it, fall back to CUDA (not CPU)**

The previous TRT-aware loader had a silent CPU-fallback bug: when
`tensorrt` wasn't installed, ORT fell back ALL THE WAY to CPU instead
of trying CUDA. Symptom: GPU at 0 %, CPU at 70 %+, inswapper grinding
through frames at 1–2 fps.

Fix:

1. Detect TRT availability *before* listing it as a provider:
   `import tensorrt` AND `"TensorrtExecutionProvider" in
   ort.get_available_providers()`. Otherwise hand ORT a CUDA-only
   list.
2. After load, assert `_swapper.session.get_providers() !=
   ['CPUExecutionProvider']` — better to crash startup than to grind
   on CPU for 4 minutes per song.
3. DLL search path includes `tensorrt_libs/` (TRT's pip package puts
   DLLs there, not under `nvidia/<lib>/bin/`).

`requirements-webapp.txt` adds `tensorrt-cu12>=10.0` (~2 GB,
optional).

**Lesson:** ORT's "fall back to next provider" is fall-back-to-CPU,
not fall-back-to-the-next-listed. Detect each provider's prerequisites
before naming it.

---

## v0.3 — Reliable autoplay + UI polish

**`6ce108b` Reliable autoplay + TensorRT inswapper + CLAUDE.md operator manual**

Autoplay reliability:

- `<video muted autoplay>` HTML attributes — declarative path is
  more permissive than calling `play()` from JS
- `tryStartPlayback()` only sets `playStarted=true` *after* `play()`
  resolves, retries every 1s on rejection
- Universal click-anywhere rescue: any click on the page also
  triggers play()
- "Click to unmute" overlay shrunk from full-page blur to a small
  pulsing pill (the full overlay made the playing video look blank)

TensorRT inswapper with FP16 + engine caching at
`webapp_jobs/.trt_cache/`. ~30–50 % uplift on RTX cards. First job
pays a ~60–90 s engine build; cached for all subsequent jobs.

CLAUDE.md (initial 750-line operator manual): prereqs, first-run,
code map, HTTP API, the five Windows-specific bugs we hit and fixed,
troubleshooting matrix, perf numbers, security, smoke-test
checklist.

**Lesson:** browser autoplay policy is the trickiest cross-browser
issue in the codebase. Always set `muted=true` *before* `play()`,
always retry on rejection, always offer a click-rescue.

---

## v0.2 — HLS streaming with audio

**`36a6310` Fix HLS streaming: ffmpeg tee path, pre-buffer, autoplay unmute overlay**

Three bugs:

1. ffmpeg tee muxer URL parser treats `:` as an option separator, so
   Windows absolute paths (`C:/Users/...`) silently broke output. Run
   ffmpeg with `cwd=<job_dir>` and use relative paths inside the tee
   URL. Also drain ffmpeg's stderr to `<job_dir>/ffmpeg.log` on a
   background thread so future failures aren't silent.

2. Browsers block autoplay-with-audio until user interacts. Video was
   decoded and ready (`readyState=4, 1920x1080`) but stuck on
   `play()` reject. Start the player muted so autoplay succeeds, then
   surface a "Click to unmute" overlay button.

3. Swap pipeline runs slower than realtime (~7 fps wall-clock vs 25
   fps source) → the live edge moved faster than the buffer could
   refill, causing constant 1-second stutters. Pre-buffer 15 seconds
   before starting playback, and on `waiting` events wait until the
   buffer recovers to 8 seconds before resuming.

**Lesson:** Windows path quirks bite once per project. Always
test ffmpeg invocations with absolute Windows paths, OR use cwd +
relative paths.

---

## v0.1 — Initial release

**`f115d94` Initial commit: live face-swap web app with HLS audio streaming**

The first working version:

- Flask app with drag-drop upload, auto gender detect,
  embedding-locked reference matching, HLS+MP4 tee-muxer streaming
  so the browser hears the song audio while the swap is still
  processing
- `stream-swap.py`: CLI version that streams to ffplay or saves an
  audio-muxed MP4
- `swap-song.ps1` / `swap-album.ps1`: PowerShell wrappers around
  FaceFusion 3.6 headless-run for highest-quality offline renders
- `play-song.ps1`: launcher for Deep-Live-Cam GUI (path B —
  real-time playback)
- `setup.ps1`: one-shot installer (conda envs, upstream clones,
  model downloads, cuDNN DLL-search patches for FaceFusion conda.py
  and DLC run.py)
- `README.md`, `DESIGN.md`, `OBS-setup.md`: docs covering
  architecture, the cuDNN `os.add_dll_directory` issue, why two
  conda envs, the HLS pipeline, and the embedding-clustering
  reference-extraction algorithm

---

## Lessons summary (highest-impact first)

1. **Don't trust ORT's "next provider" fallback.** It's fall-back-
   to-CPU, not fall-back-to-the-next-named-provider. Detect each
   provider's prereqs before listing it.
2. **Fragmented MP4 isn't a file format.** Always remux to standard
   non-fragmented MP4 for downloads.
3. **Windows + native deps + Python = DLL search hell.** Always
   `os.add_dll_directory` and store the cookies. Always test the
   built env on a clean Windows shell.
4. **ffmpeg tee URLs and Windows drive letters don't mix.** Use cwd
   + relative paths inside the tee URL.
5. **Browser autoplay policy is asymmetric — `muted` must be set
   before `play()`, and you must retry on rejection.** Always.
6. **Don't combine three speedup changes in one commit.** When one
   breaks, you don't know which. (We learned this with a `list index
   out of range` bug at frame 295.)
7. **Profile before adding threads.** The cheap thread (writer) won
   us 35 %; the second thread (reader) won 7 %; the third (detector)
   won effectively 0 % because GPU wasn't the bottleneck any more.
