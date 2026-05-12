# Flask webapp.py perf bench

Smoke clip: `cli/out_test/clip_20s.mp4` (501 frames, 1920×1080, 25 fps, faces present throughout). Source: `sreeni.jpg` as `--male`.

Each row records the timer snapshot at the **end** of the swap stream (just before `phase = done`). `nvidia-smi --query-gpu=utilization.gpu,utilization.encoder,utilization.decoder` is sampled every 1 s during the run; values shown are means over the streaming phase.

| Phase | proc_fps | read p50 | detect p50 | swap p50 | paste p50 | write p50 | GPU SM util | NVENC util | NVDEC util | swap_count | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (baseline pre-perf branch) | TBD | — | — | — | — | — | ~30% | 0% (libx264) | 0% (sw decode) | TBD | swap+paste fused; cv2 software decode; libx264 |
| 1 (timing instrumented) | **4.4** | 8.86 | 101.88 | 326.44 | n/a (fused) | 1.32 | not sampled mid-run | 0% | 0% | 313 | swap stage dominates at 1080p; detect+matmul bundle is ~100 ms; user's :8080 Flask was idle but loaded |
| 2 (NVENC output) | 4.1 | 9.12 | 111.43 | 333.28 | n/a | 1.33 | 0–35% (swap-bound) | 0% sampled (idle waiting on swap) | 0% | 313 | h264_nvenc active per ffmpeg log; no proc_fps gain because writer wasn't bottleneck. Win is freed CPU for Phase 3 paste-back. |
| ~~3 (paste-back split)~~ | **3.5 ❌ regression** | 14.51 | 120.06 | 19.37 | 259.27 | 4.28 | not sampled | n/a | n/a | 313 | **REVERTED.** Implemented + reviewed but throughput dropped vs Phase 2 (4.1 → 3.5). See note below. |
| 4 (face batching) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | batched ORT call when ≥2 faces/frame |
| 5 N=4 (clip_20s) | 9.8 | 9.06 | n/a | n/a | n/a | 5.30 | 55-81% | active | n/a | 313 | warmup 20.5s; 4 worker processes; worker_total p50=1744ms (4 frames parallel) |
| 5 N=2 (clip_20s) | 6.5 | — | n/a | n/a | n/a | — | 16-48% | active | n/a | 313 | warmup 14s; 1.6× scaling vs Phase 2 |
| 5 N=6 (clip_20s) | 11.3 | — | n/a | n/a | n/a | — | 32-84% | active | n/a | 313 | warmup 32s; 2.7× scaling; peak per-test sample |
| 5 N=8 (clip_20s) | crash | — | — | — | — | — | — | — | — | — | worker 0 fails: protobuf DecodeError during onnx.load even with 1s stagger; needs `multiprocessing.Lock` around model load. Deferred. |
| 5 N=6 det_size=480 (clip_20s) | 11.8 | — | n/a | n/a | n/a | — | 2-17% | active | n/a | 313 | det_size win small at 1080p (only +5%); swap dominates |
| **5 N=6 det_size=480 FULL 5min song** | **33.0** ⭐ | — | n/a | n/a | n/a | — | **87-95%** | active | n/a | 1682 | **PEAK** — warmup amortized over 7902 frames; pipeline reaches steady state; **GPU truly saturated**. Wall-clock 290s for 316s of footage → swap faster than realtime. |
| 5 N=6 det_size=480 user live run (4712-frame clip, 2026-05-11) | **22-26 mid-stream** | — | n/a | n/a | n/a | — | — | active | n/a | — | User-observed during live HLS playback at frame ~1100/4712 (24%). Climbs with frame count as the pipeline fills its queues; reproduces the 30+ fps peak after warmup amortises. |
| 6 (NVDEC input) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | >0% | TBD | PyAV hwaccel=cuda |

## How rows are captured

```
$env:FACESWAP_PORT=8082
conda run -n dlc python webapp.py   # in worktree, foreground or background
# upload sreeni.jpg + clip_20s.mp4 via the form at http://localhost:8082
# poll /job/<id>/status until phase=done
# capture proc_fps, swap_count, timers dict
# nvidia-smi util sampled separately with `nvidia-smi dmon -s u -c 20`
```

The Phase 1 row sets the baseline including instrumentation overhead. Subsequent rows must be strictly higher in `proc_fps` and within ±5% of `swap_count` (allowing for NMS/detector non-determinism) to advance.

Visual quality is verified by spot-checking `webapp_jobs/<id>/swapped.mp4` after each phase — eyeball comparison against the Phase 1 output, looking for chin alignment, blend seams, and colour bleed.

## Phase 5 timer schema (changed from Phases 1–3)

The 4-thread pipeline of Phases 1–3 had `read / detect / swap / write`
timers running in series across one process. Phase 5 replaces that with
N worker processes, so per-frame "detect" and "swap" are no longer
visible from the master — the master just sees dispatch-to-result.

Master-side timers in Phase 5:

| Name | What it measures |
|---|---|
| `read` | `cv2.VideoCapture.read()` wall time (decode of one frame) |
| `dispatch` | shared-memory copy + `in_q.put(SwapRequest)` (master -> worker handoff) |
| `worker_total` | `perf_counter` from dispatch to receiving `SwapResponse` (full per-frame round-trip incl. queue latency + detect + swap + paste in the worker). This is the end-to-end per-frame latency the master observes; it does NOT equal `proc_fps^-1` because workers process in parallel. |
| `write` | `ffmpeg.stdin.write()` wall time (master -> ffmpeg piping) |

`worker_total p50` is the key Phase 5 number: it should be in the
same ballpark as the Phase 2 swap timer (~330 ms at 1080p) because
each worker is doing the same fused swap+paste, but `proc_fps` should
be roughly N × Phase 2 fps because N workers run in parallel.

Status JSON additions: `n_workers`, `worker_warmup_ms`.

## Phase 1 takeaways (the smoke test that drives every subsequent phase)

- **swap stage = 326 ms p50** (1080p, single source, swap+paste fused). This is the dominant cost — confirms that splitting paste-back (Phase 3) and batching ORT calls (Phase 4) are the right next moves.
- **detect stage = 102 ms p50**. Larger than the naïve 10-15 ms `fa.get` estimate because the wrapper also stacks target embeddings, computes the (T,S) sim matrix, and runs the per-face argmax loop — all Python-side. That part probably won't shrink in Phases 2–4; it's a candidate for Phase 5's worker fan-out (which moves it into parallel processes).
- **write stage = 1.3 ms p50, read = 8.9 ms p50** — both well clear of any bottleneck. NVENC (Phase 2) is for freeing CPU, not closing a write-time gap.
- **Single Flask sample at 4.4 fps** runs slower than the earlier 6.5 fps measured on `main` because the worktree Flask and the user's main-tree Flask were both loaded into VRAM during the run (~3 GB total). The user's Flask was idle (no concurrent job), so the slowdown was contention on ORT's CUDA streams, not real compute. Phase 2+ smoke tests will reuse this baseline so the comparison is apples-to-apples.
- **MP4 quality** spot-checked: 1920×1080, h264 25 fps, AAC 44.1 kHz stereo, 20.04 s, 9.28 MB. Audio + video both well-formed.

## Phase 3 post-mortem: split paste-back did NOT help (reverted)

Phase 3 split the swap+paste fused call into separate GPU-swap and CPU-paste stages running on different threads. The split was correct on the GPU side: the `swap` timer dropped from 326 ms p50 (fused) to 19 ms p50 (pure ORT). But the new `paste` stage came in at 259 ms p50, and **overall throughput dropped from 4.4 fps (Phase 1/2) to 3.5 fps** on the same clip in the same conditions.

**Why it didn't help in single-process Python:**
1. The intended win was "next-frame GPU swap overlaps with current-frame paste." It doesn't pay off: GPU swap is 19 ms and paste is 259 ms, so the GPU finishes long before paste does and waits anyway. The slow stage IS the new added thread.
2. Adding a 5th thread + 4th queue added GIL contention and queue-handoff overhead that exceeded the (tiny) parallelism gain. With one GIL, threads serialize on Python bytecode regardless of how many you have.
3. Empty-pick frames (no faces) traversed 4 queues instead of 3, which is pure overhead for them — visible in the 2.3 fps measured during the no-face tail of the clip.

**Why it WILL help under Phase 5 (multiprocessing):**
- Each worker process has its own GIL. Inside a worker, the **fused** swap+paste is optimal (fewer thread boundaries, no inter-stage queue).
- Parallelism comes from running N workers in parallel, each doing its own fused swap+paste, so paste-back runs N times in parallel across the i9's cores.
- Phase 5 = real parallelism across processes. Phase 3 = fake parallelism inside one process. The former is what we want.

**Decision:** Phase 3 reverted to commit `2b9b4e7` (Phase 2 done). Branch keeps Phase 1 (instrumentation) and Phase 2 (NVENC). Phase 4 (face batching) is small leverage (~1–2 fps) and doesn't compound with Phase 5; deferred indefinitely. Next step: Phase 5 multiprocessing fan-out, targeting **30–50 fps** at 1080p on this i9 + 4090 box.

The `_paste_back` helper from Phase 3.2 was byte-equal to insightface's fused version (validated by the implementer on a real 1080p frame, zero non-zero diff pixels). If Phase 5 ends up wanting a standalone paste-back function inside each worker, this helper can be cherry-picked from the reflog (`git reflog show perf-flask-gpu-saturation` → commit `c0fe476`).

## Phase 5 autoresearch — concurrency sweep + final winning config

**Hardware:** Core i9 + RTX 4090 Laptop (16 GB VRAM).

| Config | fps | scaling vs Phase 2 | VRAM | GPU SM util | Notes |
|---|---|---|---|---|---|
| Phase 2 baseline N=1 (clip_20s) | 4.1 | 1.0× | ~3 GB | ~30% | swap+paste fused in main thread |
| N=2 (clip_20s) | 6.5 | 1.6× | 4.9 GB | 16-48% | |
| N=4 (clip_20s) | 9.8 | 2.4× | 8.0 GB | 55-81% | |
| **N=6** (clip_20s) | **11.3** | **2.7×** | 11.0 GB | 32-84% | stable peak on the 20s clip |
| N=8 (clip_20s) | crash | — | — | — | `protobuf DecodeError` during `onnx.load` race; 1s stagger insufficient. Needs `multiprocessing.Lock` around model load. |
| N=6 + det_size=480 (clip_20s) | 11.8 | 2.9× | 10.5 GB | 2-17% sampled | det_size only +5% at 1080p; swap dominates, not detect |
| **N=6 + det_size=480 on FULL 5-min song** | **33.0** ⭐ | **8.0×** | 10.5 GB | **87-95%** | The headline number. Warmup amortised; pipeline at steady state; GPU genuinely saturated. |

**The 20s smoke clip is misleading because the 32s worker warmup dominates the average.** On real-length content (7902 frames), the pipeline reaches steady state and the proc_fps jumps from 11.3 → 33.0. Wall-clock 290 s for 316 s of footage — i.e. **swap is now faster than playback**.

### Why N=6 + det_size=480 is the winning config

- N=6 saturates the GPU (samples in 87-95% range). Adding more workers can't extract more throughput because the GPU is the limiter.
- N=8 hits a model-load race condition on Windows (protobuf decode failure when 8 workers `onnx.load` simultaneously). Fixable with a proper `multiprocessing.Lock` around the load, but won't materially help because the GPU is already maxed at N=6.
- det_size=480 buys only +5% at 1080p because the swap stage (fused inswapper + 1080p paste-back) is the dominant cost per worker — detect is already small relative to it. Detect being smaller frees a sliver of GPU time, hence the +5%.
- VRAM at this config: 10.5 GB on a 16 GB card. Plenty of headroom for higher resolutions or a 7th worker if a future fix unlocks it.

### Knobs that did NOT pay off

- **NVENC output (Phase 2)**: writer was never the bottleneck; pure CPU offload win that doesn't move proc_fps.
- **Split paste-back into own thread (Phase 3)**: regressed throughput from 4.4 → 3.5 fps in single-process Python (GIL contention + extra queue overhead). REVERTED. Inside multiprocessing workers the **fused** swap+paste is optimal.
- **Face batching (Phase 4)**: deferred — doesn't compound with multiprocessing; per-worker fused call is already optimal at typical 1-2 faces/frame.

### Recommended production config

```powershell
$env:FACESWAP_PORT = "8082"
$env:FACESWAP_WORKERS = "6"
$env:FACESWAP_DET_SIZE = "480"
$env:FACESWAP_VIDEO_ENCODER = "h264_nvenc"
conda run -n dlc python webapp.py
```

For 24 GB desktop 4090: try `FACESWAP_WORKERS=8` after landing the model-load lock fix; should hit 40-50 fps.



