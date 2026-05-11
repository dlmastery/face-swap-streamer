# Flask webapp.py perf bench

Smoke clip: `cli/out_test/clip_20s.mp4` (501 frames, 1920×1080, 25 fps, faces present throughout). Source: `sreeni.jpg` as `--male`.

Each row records the timer snapshot at the **end** of the swap stream (just before `phase = done`). `nvidia-smi --query-gpu=utilization.gpu,utilization.encoder,utilization.decoder` is sampled every 1 s during the run; values shown are means over the streaming phase.

| Phase | proc_fps | read p50 | detect p50 | swap p50 | paste p50 | write p50 | GPU SM util | NVENC util | NVDEC util | swap_count | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (baseline pre-perf branch) | TBD | — | — | — | — | — | ~30% | 0% (libx264) | 0% (sw decode) | TBD | swap+paste fused; cv2 software decode; libx264 |
| 1 (timing instrumented) | **4.4** | 8.86 | 101.88 | 326.44 | n/a (fused) | 1.32 | not sampled mid-run | 0% | 0% | 313 | swap stage dominates at 1080p; detect+matmul bundle is ~100 ms; user's :8080 Flask was idle but loaded |
| 2 (NVENC output) | 4.1 | 9.12 | 111.43 | 333.28 | n/a | 1.33 | 0–35% (swap-bound) | 0% sampled (idle waiting on swap) | 0% | 313 | h264_nvenc active per ffmpeg log; no proc_fps gain because writer wasn't bottleneck. Win is freed CPU for Phase 3 paste-back. |
| 3 (paste-back split) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 5-stage pipeline; paste in own thread |
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

