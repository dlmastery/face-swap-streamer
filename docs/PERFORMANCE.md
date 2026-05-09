# Performance

How fast it goes, why it goes that fast, and what you can change to
make it go faster (or slower).

---

## Measured throughput (RTX 4090 Laptop, 16 GB VRAM)

Defaults: TensorRT inswapper + det_size 480 + 4-stage pipeline +
Q_DEPTH 128 + buffalo_l face model.

| Source resolution | proc fps | GPU avg util | wall-clock for 4-min song |
|---|---|---|---|
| 480 × 360   | 30 – 45 | ~25 % | 2:30 – 4:00 |
| 640 × 480   | 18 – 25 | ~20 % | 4:00 – 5:30 |
| 1280 × 720  | 12 – 18 | ~30 % | 5:30 – 8:30 |
| 1920 × 1080 | 8 – 13  | ~40 % | 8:00 – 12:00 |

Memory: ~5 GB VRAM resident for the analyser + inswapper + TRT engine,
~1.5 GB host RAM for the in-flight frame queues at 1080p.

---

## Per-frame breakdown (1080p)

What a typical frame's wall-clock budget looks like. The pipeline
overlaps these so the actual frame interval ≈ max(stage_time), not
sum, but knowing each stage's cost helps decide what to optimise.

| Stage | Time | GPU? | Notes |
|---|---|---|---|
| `cv2.VideoCapture.read()` | 3 – 5 ms | no — software decode | Could move to NVDEC for ~50 % saving |
| `fa.get(frame)` (face detector + analysers) | 10 – 15 ms | yes | Biggest GPU stage |
| Embedding match: `tgt_embs @ ref_embs.T` + argmax | < 1 ms | no | numpy, releases GIL for matmul |
| `sw.get(...)` per face (TRT inswapper + paste_back) | 5 – 10 ms | yes for inswap, no for paste_back | TRT FP16, ~1.5 × faster than CUDA |
| `frame.tobytes()` (BGR → bytes for ffmpeg pipe) | 3 – 5 ms | no | Python-level memcpy of 6 MB |
| `ffmpeg.stdin.write(bytes)` | 1 – 3 ms | no — syscall + ffmpeg internal queue | |
| Python glue (queue puts/gets, attribute access, GIL switch) | 1 – 2 ms | no | |
| **Total (serial sum)** | **23 – 40 ms** | | Theoretical 25 – 43 fps |
| **Observed pipelined** | **~80 ms / frame** | | Effective 12 fps — gap is GIL serialisation + ORT GPU serialisation |

The gap between theoretical and observed is the cost of:
- ORT serialising GPU calls — `fa.get` on frame N+2 can't run *while*
  `sw.get` is running for frame N+1, even though they're on different
  threads, because ORT holds an internal lock per session.
- GIL serialisation between Python operations across threads.

---

## Speedup history

| commit | what changed | proc fps | gain | notes |
|---|---|---|---|---|
| `e36b4db` | TRT detection (no silent CPU fallback) | 7.5  | baseline | All later numbers vs this |
| `8735818` | + async writer thread | 10.1 | +35 % | Decouples ffmpeg pipe writes from main loop |
| `660e7d1` | + async reader thread, Q=32 | 10.8 | +7 %  | cv2 decode runs ahead of swap |
| `0a966ce` | + det_size 640→480, Q=64 | 11.7 | +8 %  | Face detector is the slowest GPU stage |
| `2a0e0dd` | + multi-source matching, batched embed | 12+  | +3 %  | Mostly a feature, modest perf side-effect |
| `d4fc024` | + 4-stage pipeline (detect on its own thread), Q=128 | 12+  | ±0    | GPU is no longer the bottleneck, so this wins less |
| `40ba7a1` | + MP4 remux (mobile compat fix) | 12+  | ±0    | Container fix, no perf change |

**Cumulative: ~+60 % throughput vs the serial baseline.** After this,
GPU is only ~20 % utilised on average — the pipeline is mostly waiting
on Python / ORT serialisation rather than GPU compute.

---

## Tuning knobs

### Env vars (read at server start)

| var | default | effect |
|---|---|---|
| `FACESWAP_FACE_MODEL` | `buffalo_l` | `buffalo_s` is ~2 × faster on detection but produced visible swap-quality regression on tested footage. Try it on cleaner sources. |
| `FACESWAP_DET_SIZE`   | `480`       | 640 = native (slower, catches smaller/profile faces); 320 = even faster (misses more) |

Set on the command line before launching:

```powershell
$env:FACESWAP_FACE_MODEL = "buffalo_s"
$env:FACESWAP_DET_SIZE   = "640"
conda run -n dlc python webapp.py
```

### Hard-coded constants (edit `webapp.py`)

| constant | default | what changing it does |
|---|---|---|
| `Q_DEPTH`            | 128 | Smaller = less RAM, less slack against I/O hiccups. Bigger = more buffer at the cost of memory. |
| `REFERENCE_THRESH`   | 0.22 | Lower (e.g. 0.18) = more permissive matches, more false positives. Higher (e.g. 0.30) = stricter, more dropouts. |
| `PREBUFFER_TARGET`   | 15 (in JS) | Seconds buffered before live playback starts. Lower = faster start but more risk of stalls. |
| `REBUFFER_TARGET`    | 8 (in JS)  | Seconds to recover after a buffer underrun before resuming. |
| `hls_time`           | 2 | Seconds per HLS segment. Smaller = lower latency but more files. |

---

## What didn't work

### `buffalo_s` face model

The smaller bundle (`scrfd_500m` detector + arcface mbf recogniser)
runs ~2 × faster on detection. On test footage the embeddings produced
were less discriminative — the cosine-similarity match against the
reference cluster was weaker, so reference-match misses increased and
the swap visibly flickered between leads.

Verdict: **kept as opt-in via env var**, not the default. May be fine
on cleaner / higher-quality source images.

### CUDA Graph capture in the TRT inswapper

`trt_cuda_graph_enable: True` should reduce kernel-launch overhead by
recording and replaying a graph. On this workload (inswapper has
static 1×3×128×128 input, perfect for graphs) we measured **no
improvement**, and one earlier broken commit had a regression that
correlated with this flag being on. Best guess: ORT's TRT EP has its
own optimisations that already saturate.

Verdict: **disabled**.

### Frame-skip + interpolate (process every Nth frame)

Process every 2nd frame, copy the swap to the in-between frame.
Doubles throughput in principle but **visibly artefacts on fast cuts**
— the in-between frame keeps a swapped face from a previous shot. For
music videos with rapid cuts this is unacceptable.

Verdict: **rejected**. Could revisit with motion-aware interpolation
(optical-flow warp the swap) but that's a much bigger change.

---

## Speedups still on the table

In rough order of bang-for-buck:

### 1. NVDEC video decode (~10–20 % I/O save)

`cv2.VideoCapture.read()` does software decode on CPU. Switching to
PyAV with `hwaccel=cuda`, or piping ffmpeg's `-hwaccel cuda` decode
output through stdin into a numpy buffer, would:

- Save the 3–5 ms/frame CPU decode at 1080p
- Avoid an unnecessary CPU → GPU memcpy when the frame is then sent
  to the GPU detector

About 30 lines of code if we go via PyAV. ~10 % expected gain on
1080p.

### 2. Zero-copy frame to ffmpeg (~5 – 10 % save)

Today: `frame.tobytes()` → `ffmpeg.stdin.write(bytes)`. The `tobytes`
is a Python-level memcpy of 6 MB at 1080p.

If we pass `frame.data` (a memoryview) directly to `ffmpeg.stdin.write`,
no copy. ~3–5 ms saved per 1080p frame.

### 3. TRT for the static-shape sub-models of the face analyser

`buffalo_l` includes:

| sub-model | input shape | static? |
|---|---|---|
| `det_10g`     | 1 × 3 × ? × ?    | dynamic — TRT struggles |
| `1k3d68`      | None × 3 × 192 × 192 | static |
| `2d106det`    | None × 3 × 192 × 192 | static |
| `genderage`   | None × 3 × 96 × 96  | static |
| `w600k_r50`   | None × 3 × 112 × 112 | static |

The four static-shape ones could run on TRT with engine caching like
the inswapper does. The detector stays on CUDA. Modest win because
detection is the dominant cost and doesn't get TRT'd.

### 4. Multiprocessing instead of threading

If profiling reveals the GIL becomes the floor, split the pipeline
into separate Python processes communicating via shared-memory
buffers (`multiprocessing.shared_memory`). Avoids GIL contention
entirely.

Bigger refactor; only worth doing if (1)–(3) above don't get us
where we want to be.

### 5. Larger inswapper model (256 or 512 instead of 128)

Doesn't speed things up — it slows the swap step *and* increases
quality. Useful if quality on close-ups is the priority. Output
`paste_back` would be sharper.

### 6. Streaming ffmpeg with `-tune zerolatency`

We already pass this. Could investigate `-x264-params
"sliced-threads=1:rc-lookahead=0"` for even lower per-segment
latency. Trade-off: encode quality.

---

## How to profile

### Quick GPU-utilisation sample

```bash
for i in $(seq 1 12); do nvidia-smi --query-gpu=utilization.gpu \
  --format=csv,noheader,nounits; sleep 0.5; done | \
  awk '{ s+=$1; n++; if ($1>m) m=$1 } END {
    printf "avg=%.0f%%  max=%.0f%%  (n=%d)\n", s/n, m, n }'
```

### Per-stage timing

Add timers around each stage in `_run_job`:

```python
import time
t_pre = time.perf_counter()
tgt_faces = fa.get(frame)
t_after_detect = time.perf_counter()
# ...
t_after_swap = time.perf_counter()
print(f"detect={1000*(t_after_detect-t_pre):.1f}ms  "
      f"swap={1000*(t_after_swap-t_after_detect):.1f}ms",
      flush=True)
```

For more sustained profiling, wrap the worker in `cProfile` and dump
to `webapp_jobs/<id>/profile.pstats`, then load with `snakeviz`.

### Find which stage's queue is full (= upstream bottleneck)

Print `read_q.qsize()` and `detect_q.qsize()` periodically:

- `read_q` near 0 → reader can't keep up (decode is bottleneck)
- `detect_q` full → main worker can't keep up (swap is bottleneck)
- Both empty → writer can't keep up (ffmpeg is bottleneck)
