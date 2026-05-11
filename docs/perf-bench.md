# Flask webapp.py perf bench

Smoke clip: `cli/out_test/clip_20s.mp4` (501 frames, 1920×1080, 25 fps, faces present throughout). Source: `sreeni.jpg` as `--male`.

Each row records the timer snapshot at the **end** of the swap stream (just before `phase = done`). `nvidia-smi --query-gpu=utilization.gpu,utilization.encoder,utilization.decoder` is sampled every 1 s during the run; values shown are means over the streaming phase.

| Phase | proc_fps | read p50 | detect p50 | swap p50 | paste p50 | write p50 | GPU SM util | NVENC util | NVDEC util | swap_count | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (baseline pre-perf branch) | TBD | — | — | — | — | — | ~30% | 0% (libx264) | 0% (sw decode) | TBD | swap+paste fused; cv2 software decode; libx264 |
| 1 (timing instrumented) | TBD | TBD | TBD | TBD | n/a (fused) | TBD | TBD | TBD | TBD | TBD | StageTimer in /status; no behaviour change vs baseline |
| 2 (NVENC output) | TBD | TBD | TBD | TBD | n/a | TBD | TBD | >0% | TBD | TBD | h264_nvenc |
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
