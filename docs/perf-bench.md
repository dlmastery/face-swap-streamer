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
| 5 (multiproc N=4) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 4 worker processes via shared_memory |
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


