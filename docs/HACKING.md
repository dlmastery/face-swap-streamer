# Hacking guide

For a new developer — human or AI — landing in the repo for the
first time. What to read, in what order, and where to look when you
start cutting code.

---

## Read in this order

1. **[../README.md](../README.md)** — what the thing does, how to run
   it. 5 min read.
2. **[../USERGUIDE.md](../USERGUIDE.md)** — try the thing yourself.
   Upload a face + a video, watch the live stream, download the MP4.
   You can't usefully change the system without first using it.
3. **[ARCHITECTURE.md](ARCHITECTURE.md)** — the system map, the
   pipeline diagram, the HTTP API. 15 min read.
4. **[../DESIGN.md](../DESIGN.md)** — *why* it's built this way:
   conda envs, cuDNN DLL trickery, embedding lock, HLS tee +
   remux, etc. Context for the *what*.
5. **[../CLAUDE.md § Things that broke before](../CLAUDE.md#things-that-broke-before--dont-re-break-them)** —
   16 numbered hazards. **Skim this**, don't memorise. You'll come
   back to it when something doesn't work.
6. **[CHANGELOG.md](CHANGELOG.md)** — every commit and what it
   taught us. Useful when you're about to revert something.

Then dive into the code.

---

## Code map (most-frequently-edited first)

| File | What | When you'd touch it |
|---|---|---|
| `webapp.py` (~900 lines) | The whole web app: dataclass `Job`, `_ensure_models`, `_run_job`, `_spawn_ffmpeg`, `_remux_to_mp4`, Flask routes, two HTML templates inline | 95 % of changes |
| `stream-swap.py` | CLI version of the streaming pipeline. Mirrors the webapp's worker but outputs to ffplay. Useful for headless testing. | When debugging the swap loop without Flask in the way |
| `setup.ps1` | One-shot installer | When deps change, or upstream pinning changes |
| `requirements-webapp.txt` | pip deps for `dlc` env | New Python packages |
| `swap-song.ps1` etc. | PS wrappers for the CLI fallback paths | Rarely |
| Docs (`README.md`, `USERGUIDE.md`, `docs/*`) | What you're reading | Every meaningful behaviour change |

---

## Editing workflow

### The fast loop

```powershell
# 1. edit webapp.py
# 2. stop the running webapp (Ctrl+C in its terminal, or TaskStop)
# 3. relaunch
conda run -n dlc python webapp.py
# 4. wait ~5 s for Flask to bind, ~30 s for models to pre-warm
# 5. hard-refresh browser (Ctrl+F5)
# 6. test
```

We don't use Flask's reloader (`use_reloader=False`) — it would
double-load GPU models, OOM the GPU, and miss the threading-safety
guarantees we rely on.

### For HTML/JS-only changes

Hard-refresh the browser. The HTML templates are inlined in
`webapp.py` and re-rendered on every request, so server restart
isn't required.

### Where to read logs

| Log | What |
|---|---|
| `out/webapp.log` | Webapp stdout — Flask access, Python prints, tracebacks |
| `webapp_jobs/<id>/ffmpeg.log` | Per-job ffmpeg stderr |
| Browser DevTools console (F12) | hls.js errors, JS errors, Network failures |

Useful tail commands:

```powershell
Get-Content out/webapp.log -Tail 50 -Wait
Get-Content "webapp_jobs/$((Get-ChildItem webapp_jobs -Directory | Sort LastWriteTime -Desc)[0].Name)/ffmpeg.log" -Tail 30
```

---

## How to add a feature

### A new env var / hard-coded knob

1. Add `os.getenv("FACESWAP_FOO", default)` in `_ensure_models` or
   wherever the value is consumed
2. Document in [PERFORMANCE.md § Tuning knobs](PERFORMANCE.md#tuning-knobs)
   and [../CLAUDE.md § Tuning knobs](../CLAUDE.md)

### A new processor (e.g. face_enhancer post-step)

1. In `_run_job` worker thread, after `sw.get(...)` and before
   `write_q.put(...)`, run the new processor on the swapped frame
2. New globally-shared model? Add to `_ensure_models()`, lazy-load on
   first use, add to the active-providers verification
3. Add per-frame timing to the perf table in [PERFORMANCE.md](PERFORMANCE.md)

### A new pipeline stage

If you genuinely need *another* thread between existing stages:

1. Define a new `queue.Queue(maxsize=Q_DEPTH)` between the upstream
   and downstream stage
2. Define the thread function with a `try ... finally:` that always
   forwards the END sentinel
3. In the `_run_job` `finally:` block, drain the queue (best-effort)
   and join the new thread *between* its upstream and downstream
   joins — order matters (writer → detect → reader currently)
4. Re-read [../CLAUDE.md issue #12](../CLAUDE.md#12-pipeline-thread-coordination)
   before testing — there are subtle deadlocks if cleanup ordering
   is wrong

### A new HTTP endpoint

1. Add `@app.route(...)` near the existing routes
2. If it returns video/audio bytes, set `Cache-Control: no-store`
   and use `send_from_directory` for Range support
3. Document in [ARCHITECTURE.md § HTTP API surface](ARCHITECTURE.md#5-http-api-surface)

### A new "phase"

1. Add to the `PHASE_ORDER` list in the viewer JS
2. Add a phase pill in the viewer HTML
3. `_set(job, phase="new_phase", message="…")` from the worker
4. Update [USERGUIDE.md § What the screen shows you](../USERGUIDE.md#what-the-screen-shows-you-in-plain-english)
   if user-visible

---

## How to test a change

### The smoke test (always)

1. Restart the webapp
2. Open `http://localhost:8080/`
3. Drop a known-good source image + a known-good short MP4
4. Watch all phase pills go green: load → detect → reference → stream → finalise → done
5. Confirm live HLS plays with audio after click-to-unmute
6. Click "Download MP4 (with audio)", verify file plays in VLC

If steps 4–6 all pass, the change is OK to commit.

### A regression test for the multi-source path

1. Drop two source images (M + F)
2. Drop a duet video
3. Confirm `sources` array in `/job/<id>/status` has both entries
   with sensible gender/age and `ref_frame` values

### A regression test for the speed path

After any change that touches `_run_job`:

```bash
# during the run:
for i in $(seq 1 12); do nvidia-smi --query-gpu=utilization.gpu \
  --format=csv,noheader,nounits; sleep 0.5; done | \
  awk '{ s+=$1; n++; if ($1>m) m=$1 } END { printf "avg=%.0f%%  max=%.0f%%\n", s/n, m }'

# expected (RTX 4090, 480p input):
# avg=15-25%  max=50-80%
```

If GPU avg drops near 0, you broke CUDA loading — see
[../CLAUDE.md issue #8](../CLAUDE.md#8-onnx-runtime-fallback-to-cpu-silent-disaster).

---

## Commit style

Looking at `git log --oneline`:

```
40ba7a1  Fix: produce a real MP4 (mobile + VLC compatible), not a fragmented one
d4fc024  Optimization #6: 4-stage pipeline (reader -> detect -> swap -> writer)
2a0e0dd  Multi-face swap: upload 1 or 2 source images, swap each lead with the matching one
0a966ce  det_size 640 -> 480 + Q_DEPTH 32 -> 64
```

The pattern:

- **First line:** what changed, in imperative voice. Short.
- **Body:** why, what was tried, measured impact (fps numbers if
  perf-related), lessons.

Use `git commit -F <file>` instead of `-m "..."` because PowerShell
mangles multi-line `-m` heredocs. There's a `.commit-msg.tmp` pattern
for this in the existing commits — read [../CLAUDE.md issue #6](../CLAUDE.md).

---

## Things that bite first-timers

These are the ones I'd warn an experienced dev about, beyond the
explicit hazard list:

1. **Two conda envs, easy to use the wrong one.** Always
   `conda run -n dlc python webapp.py` (not `python webapp.py`
   from the wrong shell).
2. **The webapp pre-warms models on startup, so the first ~30 s
   shows "starting on ..." but `/start` requests will block.** That
   isn't a hang.
3. **TRT engine builds are CPU-bound and hold the GIL** — Flask
   gets sluggish during the ~60–90 s first-run engine build. Wait
   it out.
4. **Multi-line PowerShell `git commit -m` mangles the message.**
   Use `-F .commit-msg.tmp` (existing pattern).
5. **The HLS files are real files on disk under `webapp_jobs/<id>/hls/`.**
   Browser doesn't cache them (we set `no-store`), but `hls.js` keeps
   90 s in MSE memory buffer. If you're testing scrub-back behaviour,
   know what's where.
6. **Don't put `tensorrt` in the providers list unless the package
   imports cleanly.** ORT's silent CPU fallback will burn an hour
   before you notice.

---

## When you're stuck

1. Read the relevant section of [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
2. Search [../CLAUDE.md § Things that broke before](../CLAUDE.md#things-that-broke-before--dont-re-break-them)
   for symptoms similar to yours.
3. Check `out/webapp.log` and `webapp_jobs/<id>/ffmpeg.log`.
4. If the change is making the live player blank, drive Playwright:

   ```js
   // browser_evaluate function
   const v = document.getElementById('player');
   return { paused: v.paused, muted: v.muted, readyState: v.readyState,
            error: v.error?.message, currentTime: v.currentTime,
            buffered: v.buffered.length ? v.buffered.end(0) : null };
   ```

   Most blank-screen issues are autoplay-blocked or hls.js fetching
   a 404'd segment.
5. If still stuck, file a GitHub issue with logs + symptom.
