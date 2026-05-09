# Troubleshooting

Symptom → diagnosis → fix. Start with the symptom that matches what
you're seeing, follow the link to the deeper explanation if you want
the why.

For setup-time problems (install, conda envs, missing tools), see
[../README.md § Quickstart](../README.md#quickstart) and
[../CLAUDE.md § First-run](../CLAUDE.md#first-run-from-a-fresh-clone).

---

## Index

- [The webapp won't start](#the-webapp-wont-start)
- [Upload form doesn't load](#upload-form-doesnt-load)
- [Job stuck in a phase](#job-stuck-in-a-phase)
- [Live player problems](#live-player-problems)
- [Audio problems](#audio-problems)
- [Wrong person being swapped](#wrong-person-being-swapped)
- [Quality problems](#quality-problems)
- [Speed / GPU problems](#speed--gpu-problems)
- [Download problems](#download-problems)
- [Where to look when nothing else fits](#where-to-look-when-nothing-else-fits)

---

## The webapp won't start

### `ModuleNotFoundError: No module named 'flask'` (or anything pip-installed)

You're not in the `dlc` conda env.

```powershell
conda run -n dlc python webapp.py
```

If `dlc` doesn't exist: re-run `setup.ps1` from the repo root.

### `Address already in use` / `Only one usage of each socket address`

Port 8080 is taken. Either stop the other process:

```powershell
Get-NetTCPConnection -LocalPort 8080 | Select-Object OwningProcess
Stop-Process -Id <pid>
```

… or change the port in `webapp.py`'s `app.run(... port=8080 ...)`.

### `RuntimeError: inswapper loaded on CPU only`

The CUDA + cuDNN DLLs aren't on the search path. This is by design —
better to crash startup than silently grind on CPU.

Re-install the cuDNN pip package:

```powershell
conda run -n dlc pip install --force-reinstall \
    nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 \
    nvidia-cuda-nvrtc-cu12 nvidia-curand-cu12 nvidia-cufft-cu12
```

Then verify in a fresh shell:

```powershell
conda run -n dlc python test-cuda-dlc.py
```

It should print `VERDICT: CUDA works`. If not, check
[../CLAUDE.md issue #1](../CLAUDE.md#1-cudnn-dll-discovery-on-windows).

### Webapp logs show `EP Error … TensorRT libraries`

The `tensorrt-cu12` package isn't installed. Install it:

```powershell
conda run -n dlc pip install tensorrt-cu12
```

(About 2 GB. Or skip — the code falls back to CUDA without TRT.)

---

## Upload form doesn't load

### Browser shows `This site can't be reached`

Webapp isn't running, or you have the wrong URL. Confirm the webapp
is up:

```powershell
Invoke-WebRequest http://localhost:8080/ -UseBasicParsing
```

Should return `StatusCode: 200`. If not, see "the webapp won't start"
above.

### Browser shows the form but the drop-zones look broken

Hard-refresh (Ctrl+F5) to bypass cached HTML. The HTML is inlined in
`webapp.py` and is regenerated each restart.

If still broken: check the browser DevTools console (F12 → Console).
Most likely a typo in the inlined CSS/JS.

---

## Job stuck in a phase

### Stuck on `loading_models` for >2 minutes

First-time-ever launch needs to:
- Download `buffalo_l` (~290 MB) from InsightFace's model server
- Build the inswapper TRT engine (~60–90 s, CPU-bound)

Both are silent — no progress bar. Expected wait: 3–5 min on first
run, ~30 s on subsequent restarts.

To check: `ls webapp_jobs/.trt_cache/` should contain a
`...sm89.engine` file (the architecture suffix matches your GPU). If
that file is being written, things are progressing.

### Stuck on `finding_reference`

The video probably has no detectable face of the source's gender.
Either:

- The source face's gender was misclassified — try a clearer photo
- The video genuinely doesn't contain that gender (e.g. you uploaded
  a photo of yourself but the video has only the opposite gender)

The error message in the status JSON / viewer page will say
`no <gender> face found in the video`.

### Stuck on `streaming` with progress bar not advancing

Job has crashed. Check:

```powershell
$j = (Get-ChildItem webapp_jobs -Directory | Sort LastWriteTime -Desc | Select -First 1).Name
Get-Content "webapp_jobs/$j/ffmpeg.log" -Tail 30
```

Common causes in the ffmpeg log:
- `Conversion failed!` — the pipe died. Often happens if the source
  has no audio track and we mapped `1:a:0?` strictly. The `?` should
  make it optional but some ffmpeg versions are stricter.
- `Cannot allocate memory` — you ran out of RAM. Lower `Q_DEPTH`.

The webapp log itself (`out/webapp.log` or stdout) shows the Python
exception:

```powershell
Get-Content out/webapp.log -Tail 50 | Select-String -Pattern "error|Error|Traceback"
```

### Phase shows `error`

Read the `message` field — it's a one-line description. Common ones:

| Message | Cause |
|---|---|
| `no face detected in <file> — try a clearer, front-facing photo` | Source image has no detectable face |
| `no <gender> face found in the video` | Video has nobody of the source's gender |
| `could not open target video` | Bad file or unsupported codec |
| `cancelled` | Stop flag was set (you stopped or restarted the server) |
| `ffmpeg failed (rc=…)` | See `webapp_jobs/<id>/ffmpeg.log` |
| `inswapper loaded on CPU only — CUDA failed to initialise` | See [/the webapp won't start](#the-webapp-wont-start) |

---

## Live player problems

### Player shows a black/blank frame, progress bar still moves

Most common cause: **autoplay was blocked, the player is paused**.
Click anywhere on the page — there's a click-anywhere rescue handler
that triggers `play()`. The "🔊 Click to unmute" pill bottom-left of
the player also unblocks playback.

Less common: hls.js can't fetch a `.ts` segment because of a 404.
Check DevTools → Network → look for red `404`s on
`/job/<id>/hls/seg_NNNNN.ts`. If you see them, ffmpeg crashed mid-way
through producing the segment; check `webapp_jobs/<id>/ffmpeg.log`.

### Player keeps stalling every couple of seconds

Buffer is draining faster than the swap pipeline produces. Three
things:

- **It's a 1080p video on an underspecced GPU.** Expected. The
  download will be smooth.
- **Increase `PREBUFFER_TARGET`** in the viewer JS (default 15 s) to
  ride out longer dips.
- **Decrease the work per frame**: lower target resolution, smaller
  face model.

### "Buffering 0 / 15 s" stays at 0 forever

Streaming ffmpeg never wrote any segments. Check
`webapp_jobs/<id>/ffmpeg.log` for an error. Most common:

- ffmpeg can't find a working video encoder (very rare with the
  Gyan build we recommend)
- The HLS dir has wrong permissions (also rare on Windows)

---

## Audio problems

### Live stream is silent (no song audio)

After the muted-autoplay starts, **click the "🔊 Click to unmute"
pill in the bottom-left** of the player. Browsers refuse to play
sound until the user has interacted with the page; this is a one-
click fix.

### Audio plays but is out of sync with video

Live HLS sometimes drifts when the swap is significantly slower than
real-time — the player keeps audio in sync with the .ts segments,
but the video frames *inside* those segments were produced from
later source-video timestamps than the audio is at.

The downloaded MP4 doesn't have this problem (it's a clean remux at
the end).

### Downloaded MP4 has audio but no video

You're on an old build. The fragmented MP4 produced by commits
before `40ba7a1` was unplayable in many native players. Update to
the latest, re-run the swap.

To check your version: `git log --oneline -1` from the repo. Need
to be at `40ba7a1` or later.

---

## Wrong person being swapped

### Female lead is being swapped instead of male (or vice-versa)

The detected gender of your source photo was wrong. Status JSON
shows `detected_gender` — if it's wrong, the source photo is
ambiguous.

- Use a clearer, more front-facing photo
- Make sure the face fills more of the frame (>30 %)
- Avoid photos with strong shadow / heavy makeup that biases gender
  prediction

### Multiple sources, but the wrong one is being applied to the wrong person

The "greedy cluster assignment" picks the largest unused cluster
per source. If both sources are same gender and the algorithm
picks the wrong assignment, you can:

- Swap the upload order (the first source claims the largest
  cluster)
- For now, this is a known limitation for same-gender duets

### Faces flicker — sometimes swapped, sometimes not

The reference threshold (cosine 0.22) is sometimes not met on
profile shots / odd angles. Lower it in `webapp.py`
(`REFERENCE_THRESH = 0.18`), restart, retry.

---

## Quality problems

### The swapped face looks blurry / soft

Inswapper outputs a 128 × 128 face that's then warp-pasted back at
the original resolution. On a 1080p source, that's a ~5 × upscale on
close-ups. Limited by the model.

Mitigations:

- Use **FaceFusion CLI (path A)** for offline renders — it has a
  `face_enhancer` (GFPGAN) post-step that the web app doesn't apply.
- Or post-process the downloaded MP4 with GFPGAN/CodeFormer
  separately.

### Face boundary is visible (a "mask edge")

InsightFace's `paste_back` blends with a soft mask, but it's not
perfect. Same as above — FaceFusion's `face_enhancer` does a much
nicer job. The web app trades quality for live streamability.

### Wrong age / gender shown in source pills

The InsightFace age/gender model is rough — ±5 years is normal.
Doesn't affect the swap itself (that uses face embedding, not age).

---

## Speed / GPU problems

### `nvidia-smi` shows 0 % GPU during a job

If the job is processing (status JSON shows `phase: streaming`,
`current_frame` advancing), the GPU may simply be between bursts.
Sample over 5–10 seconds:

```bash
for i in $(seq 1 12); do nvidia-smi --query-gpu=utilization.gpu \
  --format=csv,noheader,nounits; sleep 0.5; done | \
  awk '{ s+=$1; n++; if ($1>m) m=$1 } END { printf "avg=%.0f%%  max=%.0f%%\n", s/n, m }'
```

Expected: `avg=15-30%  max=50-80%` during a job. If `avg=0%, max=0%`,
something's wrong:

- Check the webapp log: `[webapp] inswapper active providers: …` —
  if the line says `['CPUExecutionProvider']` only, see [issue #8](../CLAUDE.md#8-onnx-runtime-fallback-to-cpu-silent-disaster)
- Run `conda run -n dlc python test-cuda-dlc.py` — if it fails,
  cuDNN didn't load

### fps lower than expected

See [PERFORMANCE.md](PERFORMANCE.md) for the expected fps per
resolution and per-stage breakdown. Common reasons fps drops:

- Source video is higher resolution than you think (`ffprobe input.mp4`)
- Other GPU apps are running (Stable Diffusion, games)
- Background downloads are eating disk I/O bandwidth
- TRT engine wasn't built (look for `TensorrtExecutionProvider` in
  the active providers line)

---

## Download problems

### Download button is greyed out / 404

The job's `phase` isn't `done` yet. Wait for the green "Your swap is
ready" card to appear. Until then `swapped.mp4` doesn't exist on
disk yet (it's built in the `finalising` phase).

### Downloaded MP4 won't play

You're either on an old build or the remux step failed.

Check: `ffprobe webapp_jobs/<id>/swapped.mp4`. If it errors with
"Invalid data found", the file is the old fragmented format; you're
on a pre-`40ba7a1` build. `git pull`, restart, re-run.

If it shows `format_name=mov,mp4,m4a,3gp,3g2,mj2` and a duration,
the file is valid. Try a different player (VLC is most permissive).

### Mobile says "Format not supported"

Same root cause — pre-`40ba7a1` fragmented MP4. Update to current.

---

## Where to look when nothing else fits

### Per-job ffmpeg log
```
webapp_jobs/<job_id>/ffmpeg.log
```
The streaming ffmpeg's stderr. Has the actual error message when an
encode breaks.

### Webapp log
```
out/webapp.log
```
Flask access logs + Python prints + tracebacks. Search for
`Traceback` or `error`.

### Browser DevTools
F12 → Console (JS errors), Network (HTTP 404s on segments,
Cache-Control issues), Application → Frames (HLS playlist contents).

### Running processes
```powershell
Get-Process python | Sort-Object StartTime -Descending | Select-Object -First 5
```

If the webapp process isn't there, it died — check stdout (`out/webapp.log`)
for the exception that killed it.

### GPU memory
```powershell
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

If memory is stuck high after the job ends, the model didn't get
released. Restart the webapp.

### Final fallback

File a GitHub issue at
<https://github.com/dlmastery/face-swap-streamer/issues> with:

- The symptom
- The phase the job got stuck on (or the error message)
- Last 30 lines of `webapp.log`
- Last 30 lines of `webapp_jobs/<id>/ffmpeg.log` (if the issue was
  during streaming or finalising)
- Your GPU + driver version: `nvidia-smi`
