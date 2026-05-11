# Flask webapp.py — GPU saturation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Raise Flask webapp.py throughput from ~8–13 fps to **40–60 fps at 1080p** on Core i9 + RTX 4090 (currently ~30% GPU, ~50% CPU utilisation) — without regressing swap quality vs the current pipeline.

**Architecture:** Six independent phases applied in order. Phase 1 is diagnostic-only (no behaviour change). Phases 2–4 keep the single-process design but split paste-back into its own thread, switch the encoder to NVENC, and batch multi-face frames in inswapper. Phase 5 introduces multiprocessing worker fan-out for the swap loop (the big win). Phase 6 swaps cv2 software decode for PyAV/NVDEC.

**Tech Stack:** Flask, ONNX Runtime CUDA EP, insightface (buffalo_l + inswapper_128_fp16), OpenCV, ffmpeg (libx264 → h264_nvenc), Python `multiprocessing.shared_memory`, PyAV (Phase 6 only).

**Smoke-test asset:** `cli/out_test/clip_20s.mp4` (501 frames, 1920×1080, 25 fps — already extracted earlier in the session). Faces appear ~all the way through. Source: `C:\Users\evija\Downloads\sreeni.jpg` as `--male`. Expected ~150-260 swaps in the clip (depending on detection); proc_fps is the headline metric per phase.

**Baseline before Phase 1 (measured today):**
- Pal-Pal-Dil-Ke-Paas 1080p 7902-frame song
- Two-source M+F: 6.5 fps wall-clock, 6.7 fps single-source
- GPU util ~30%, CPU ~50%

---

## How to "test" each phase

There's no unit-test suite (intentional — `CLAUDE.md`). The smoke test for every phase:

1. Restart Flask: kill PID listening on 8080, `conda run -n dlc python webapp.py` in background, wait ~60–90 s for model warm-up.
2. Upload `sreeni.jpg` + `cli/out_test/clip_20s.mp4` via the form at <http://localhost:8080>.
3. Poll `GET /job/<id>/status` until `phase == done` (or `error`).
4. Read `proc_fps`, `current_frame`, `swap_count` from the final status JSON.
5. Open `webapp_jobs/<id>/swapped.mp4` and **visually confirm** the swap quality matches the pre-phase baseline (no chin-misalignment / colour-flicker / missing-face regressions).
6. Record the result in `docs/perf-bench.md` (one row per phase).

**Pass criteria per phase:** `proc_fps` strictly greater than the previous phase, swap_count within ±5% of baseline (allowing for non-determinism in tracker / NMS), no visible quality regression.

---

## Phase 1: Per-stage timing instrumentation (diagnostic)

Adds light-weight timing of every stage of the 4-thread pipeline so we know where the gap is. No behaviour change.

### Task 1.1: Add a `StageTimer` helper

**Files:**
- Modify: `webapp.py` (add ~25-line helper class near top of file, after the `Job` dataclass)

**Step 1: Add the helper class.**

```python
import threading, time
from collections import deque

class StageTimer:
    """Lightweight running stats for a pipeline stage.

    Each stage records elapsed ms per frame. We keep a 250-frame ring so
    p50/p95 reflect the last ~10-30 seconds without unbounded memory.
    """
    def __init__(self, name: str, ring_size: int = 250):
        self.name = name
        self._ring = deque(maxlen=ring_size)
        self._lock = threading.Lock()
        self.n_total = 0

    def record(self, ms: float) -> None:
        with self._lock:
            self._ring.append(ms)
            self.n_total += 1

    def snapshot(self) -> dict:
        with self._lock:
            if not self._ring:
                return {"name": self.name, "n": 0}
            buf = sorted(self._ring)
        p50 = buf[len(buf) // 2]
        p95 = buf[int(len(buf) * 0.95)]
        return {"name": self.name, "n_total": self.n_total,
                "p50_ms": round(p50, 2), "p95_ms": round(p95, 2),
                "max_ms": round(buf[-1], 2)}
```

**Step 2: Run a syntax check.**

Run: `conda run -n dlc python -c "import webapp"`
Expected: no traceback.

**Step 3: Commit.**

```bash
git add webapp.py
git commit -m "Phase 1.1: add StageTimer helper for pipeline instrumentation"
```

### Task 1.2: Wrap each pipeline stage in StageTimer

**Files:**
- Modify: `webapp.py` `_run_job` function (around the existing thread definitions)

**Step 1: Instantiate timers for each stage.**

In `_run_job`, near where `Q_DEPTH = 128` is declared, add:

```python
timers = {
    "read":   StageTimer("read"),
    "detect": StageTimer("detect"),
    "swap":   StageTimer("swap"),       # only sw.get GPU call
    "paste":  StageTimer("paste"),      # only paste_back inside sw.get
    "write":  StageTimer("write"),
}
job.timers = timers  # so /status can serialize them
```

**Step 2: Wrap reader loop's `cap.read()` call.**

```python
def _reader_loop():
    try:
        while not job.stop_flag.is_set():
            t = time.perf_counter()
            ok, fr = cap.read()
            timers["read"].record((time.perf_counter() - t) * 1000)
            if not ok:
                break
            read_q.put(fr)
    finally:
        read_q.put(END)
```

**Step 3: Wrap detect loop similarly.**

In `_detect_loop` around `fa.get(frame)` plus the per-frame matching block:

```python
def _detect_loop():
    try:
        while True:
            item = read_q.get()
            if item is END:
                return
            frame = item
            t = time.perf_counter()
            tgt_faces = fa.get(frame)
            picks = []
            if tgt_faces:
                tgt_embs = np.stack([f.normed_embedding for f in tgt_faces])
                sims = tgt_embs @ ref_embs.T
                for ti, tface in enumerate(tgt_faces):
                    si = int(np.argmax(sims[ti]))
                    if float(sims[ti, si]) >= REFERENCE_THRESH:
                        picks.append((tface, si))
            timers["detect"].record((time.perf_counter() - t) * 1000)
            detect_q.put((frame, picks))
    finally:
        detect_q.put(END)
```

**Step 4: Wrap the main swap loop and writer.**

In the main loop (~lines 472-489), instrument `sw.get` and `ffmpeg.stdin.write`:

```python
# inside the main `while True` after `frame, picks = item`:
t = time.perf_counter()
for tface, si in picks:
    frame = sw.get(frame, tface, ref_sources[si].src_face, paste_back=True)
    swap_count += 1
timers["swap"].record((time.perf_counter() - t) * 1000)
```

In `_writer_loop`:

```python
def _writer_loop():
    nonlocal broken
    while True:
        item = write_q.get()
        if item is END:
            return
        try:
            t = time.perf_counter()
            ffmpeg.stdin.write(item)
            timers["write"].record((time.perf_counter() - t) * 1000)
        except (BrokenPipeError, OSError):
            broken = True
            while True:
                x = write_q.get()
                if x is END:
                    return
```

**Step 5: Expose timers in `/status` JSON.**

Modify the `status()` route to include `timers` if present. After the existing `job.__dict__`-style status build:

```python
out["timers"] = {k: v.snapshot() for k, v in (getattr(job, "timers", {}) or {}).items()}
```

**Step 6: Smoke test.**

Restart Flask, upload `clip_20s.mp4`, poll status mid-stream, confirm the JSON contains a `timers` block with `read`, `detect`, `swap`, `write` p50/p95 values.

Expected qualitative shape (1080p, current baseline):
- `read` p50 ~ 3-5 ms
- `detect` p50 ~ 15-25 ms (includes Python glue)
- `swap` p50 ~ 30-60 ms (this is what we expect — the killer; swap+paste fused)
- `write` p50 ~ 1-3 ms

**Step 7: Commit.**

```bash
git add webapp.py
git commit -m "Phase 1.2: instrument pipeline stages with StageTimer + /status JSON"
```

### Task 1.3: Record Phase 1 baseline

**Files:**
- Create: `docs/perf-bench.md`

**Step 1: Save the timer snapshot for the smoke clip.**

```markdown
# Flask webapp perf bench

Smoke clip: `cli/out_test/clip_20s.mp4` (501 frames, 1920×1080).

| Phase | proc_fps | read p50 | detect p50 | swap p50 | write p50 | GPU util | Notes |
|---|---|---|---|---|---|---|---|
| Baseline (Phase 1) | TBD | TBD | TBD | TBD | TBD | TBD | swap+paste fused |
```

**Step 2: Run the smoke test, fill in numbers from /status, save nvidia-smi output during streaming.**

**Step 3: Commit.**

```bash
git add docs/perf-bench.md
git commit -m "Phase 1.3: baseline perf numbers on clip_20s.mp4"
```

---

## Phase 2: NVENC output encoder

Switch ffmpeg's video encoder from `libx264` (CPU) to `h264_nvenc` (dedicated GPU silicon). Frees CPU for paste-back; doesn't compete with compute SMs.

### Task 2.1: Make the encoder configurable via env var

**Files:**
- Modify: `webapp.py` `_spawn_ffmpeg(...)`

**Step 1: Replace the hardcoded `-c:v libx264` block with:**

```python
encoder = os.getenv("FACESWAP_VIDEO_ENCODER", "h264_nvenc")
if encoder == "h264_nvenc":
    venc_args = ["-c:v", "h264_nvenc", "-preset", "p4",
                 "-rc", "vbr", "-cq", "21", "-b:v", "0"]
else:
    venc_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
```

…and splice `venc_args` into the existing ffmpeg argv at the spot where `-c:v` currently sits.

**Step 2: Smoke test.**

Restart Flask. Upload `clip_20s.mp4`. Confirm:
- Job completes.
- `swapped.mp4` plays in VLC (NVENC bitstreams can be subtly different — confirm playable).
- `proc_fps` higher than Phase 1 baseline (writer queue should drain faster → less back-pressure on the swap loop).
- `nvidia-smi dmon -s u -c 10`: NVENC column non-zero during streaming.

**Step 3: Commit.**

```bash
git add webapp.py
git commit -m "Phase 2: NVENC output (FACESWAP_VIDEO_ENCODER=h264_nvenc default)"
```

**Step 4: Record in `docs/perf-bench.md`.**

---

## Phase 3: Split paste-back into its own pipeline stage

Current swap stage = GPU `sw.get` + CPU paste-back fused on the same thread. Splitting them lets paste-back overlap with the next frame's swap call.

`insightface`'s `INSwapper.get(...)` does BOTH steps in one call. We need to call it with `paste_back=False` (returns the swapped 128×128 + the inverse affine matrix M), then do the paste-back ourselves in a separate stage.

### Task 3.1: Verify the `paste_back=False` API path

**Files:**
- Read only: `deep-live-cam/models/...` insightface source (not in tree — check `~/.insightface` install or read the package directly).

**Step 1: Inspect the insightface inswapper.**

Run: `conda run -n dlc python -c "import inspect; from insightface.model_zoo.inswapper import INSwapper; print(inspect.getsource(INSwapper.get))"`

Confirm the function:
- When `paste_back=False` returns `(bgr_fake_128, M)`.
- When `paste_back=True` does warpAffine + erode + GaussianBlur + alpha blend internally.

Document the actual signature in this plan if it differs from expectation.

### Task 3.2: Extract the paste-back into a helper

**Files:**
- Modify: `webapp.py`

**Step 1: Add `_paste_back(bgr_frame, bgr_fake_128, M) -> np.ndarray` at module scope.**

Verbatim port of the insightface paste-back math:

```python
def _paste_back(img, bgr_fake, M):
    target_img = img
    fake_diff = bgr_fake.astype(np.float32) - 127.5
    bgr_fake = bgr_fake
    IM = cv2.invertAffineTransform(M)
    img_white = np.full((bgr_fake.shape[0], bgr_fake.shape[1]), 255, dtype=np.float32)
    bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]),
                               borderValue=0.0)
    img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]),
                                borderValue=0.0)
    img_white[img_white > 20] = 255
    fmask = (img_white == 255).astype(np.uint8)
    mask_h_inds, mask_w_inds = np.where(img_white == 255)
    if len(mask_h_inds) == 0:
        return target_img
    mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
    mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
    mask_size = int(np.sqrt(mask_h * mask_w))
    k = max(mask_size // 10, 10)
    kernel = np.ones((k, k), np.uint8)
    img_white = cv2.erode(img_white, kernel, iterations=1)
    k = max(mask_size // 20, 5)
    kernel_size = (k, k)
    blur_size = tuple(2 * i + 1 for i in kernel_size)
    img_white = cv2.GaussianBlur(img_white, blur_size, 0)
    img_white /= 255.0
    img_white = np.reshape(img_white, [img_white.shape[0], img_white.shape[1], 1])
    fake_merged = img_white * bgr_fake + (1 - img_white) * target_img.astype(np.float32)
    fake_merged = fake_merged.astype(np.uint8)
    return fake_merged
```

**Step 2: Run a syntax check.**

`conda run -n dlc python -c "import webapp"` — expect no traceback.

### Task 3.3: Refactor the swap loop into 5 stages

**Files:**
- Modify: `webapp.py` `_run_job`

**Step 1: Add a new queue `paste_q` between swap and writer.**

```python
paste_q: "queue.Queue[object]" = queue.Queue(maxsize=Q_DEPTH)
```

**Step 2: New main loop (now "swap only") emits to `paste_q`.**

Replace the current main `while True` body around lines 472-489 with:

```python
item = detect_q.get()
if item is END:
    paste_q.put(END)
    break
frame, picks = item
n += 1
t = time.perf_counter()
swapped_items = []
for tface, si in picks:
    bgr_fake, M = sw.get(frame, tface, ref_sources[si].src_face, paste_back=False)
    swapped_items.append((bgr_fake, M))
    swap_count += 1
timers["swap"].record((time.perf_counter() - t) * 1000)
paste_q.put((frame, swapped_items))
```

**Step 3: Add a paste worker thread.**

```python
def _paste_loop():
    while True:
        item = paste_q.get()
        if item is END:
            write_q.put(END)
            return
        frame, swapped_items = item
        t = time.perf_counter()
        for bgr_fake, M in swapped_items:
            frame = _paste_back(frame, bgr_fake, M)
        timers["paste"].record((time.perf_counter() - t) * 1000)
        # Pass numpy buffer directly to writer — skip .tobytes()
        write_q.put(frame)
```

**Step 4: Update the writer to handle ndarray instead of bytes.**

```python
def _writer_loop():
    nonlocal broken
    while True:
        item = write_q.get()
        if item is END:
            return
        try:
            t = time.perf_counter()
            ffmpeg.stdin.write(item.tobytes() if hasattr(item, "tobytes") else item)
            timers["write"].record((time.perf_counter() - t) * 1000)
        except (BrokenPipeError, OSError):
            broken = True
            ...
```

(Or push `item.data` if benchmark shows tobytes is the dominant write cost.)

**Step 5: Start the paste thread.**

```python
t_paste  = threading.Thread(target=_paste_loop,  daemon=True, name=f"job-{job.id}-paste")
t_paste.start()
```

And in the `finally:` block, join it between detector and writer:

```python
write_q.put(END)
t_writer.join(timeout=30)
paste_q.put(END)
t_paste.join(timeout=10)
t_detect.join(timeout=10)
t_reader.join(timeout=10)
```

**Step 6: Smoke test.**

Restart Flask. Upload `clip_20s.mp4`. Confirm:
- Job completes; phase pills all green.
- `proc_fps` higher than Phase 2.
- `swap` p50 drops dramatically (now only ORT.run time, ~5-10 ms/face).
- `paste` p50 reflects the bbox-adaptive cost (~20-40 ms at 1080p).
- Output MP4 is visually identical to Phase 2 (paste_back math is byte-exact).
- swap_count unchanged.

**Step 7: Commit.**

```bash
git add webapp.py
git commit -m "Phase 3: split paste-back into its own pipeline stage"
```

**Step 8: Record in `docs/perf-bench.md`.**

---

## Phase 4: Batch faces-per-frame in inswapper

When ≥2 faces are picked in the same frame, today's code runs `sw.get` sequentially per face. Batching them into one ORT call doubles inswapper throughput on multi-face frames.

### Task 4.1: Add `_swap_faces_batched(frame, picks, sw, sources)` helper

**Files:**
- Modify: `webapp.py`

**Step 1: Write the batched path.**

```python
def _swap_faces_batched(frame, picks, sw, sources):
    """Returns list of (bgr_fake_128, M) for each pick. Batches the
    inswapper forward across the K picks in this frame."""
    if not picks:
        return []
    aimgs, Ms, latents = [], [], []
    for tface, si in picks:
        # Reuse insightface's face_align.norm_crop2 — same alignment math
        from insightface.utils import face_align
        aimg, M = face_align.norm_crop2(frame, tface.kps, 128)
        aimgs.append(aimg)
        Ms.append(M)
        src_emb = sources[si].src_face.normed_embedding
        latent = src_emb.reshape((1, -1)) @ sw.emap
        latent /= np.linalg.norm(latent)
        latents.append(latent[0])
    blob = cv2.dnn.blobFromImages(aimgs, 1.0 / 255.0, (128, 128),
                                   (0.0, 0.0, 0.0), swapRB=True)
    latents_arr = np.stack(latents).astype(np.float32)
    preds = sw.session.run(sw.output_names,
                            {sw.input_names[0]: blob,
                             sw.input_names[1]: latents_arr})[0]
    out = []
    for i in range(len(picks)):
        bgr_fake = preds[i].transpose(1, 2, 0)[..., ::-1]
        bgr_fake = (bgr_fake * 255).astype(np.uint8)
        out.append((bgr_fake, Ms[i]))
    return out
```

**Step 2: Replace the per-face loop in the main swap stage with this call.**

Replace:
```python
for tface, si in picks:
    bgr_fake, M = sw.get(frame, tface, ref_sources[si].src_face, paste_back=False)
    swapped_items.append((bgr_fake, M))
    swap_count += 1
```
with:
```python
swapped_items = _swap_faces_batched(frame, picks, sw, ref_sources)
swap_count += len(picks)
```

**Step 3: Smoke test.**

Same clip. Confirm:
- swap_count unchanged.
- For two-source M+F clips, `swap` p50 should fall by ~30-40% in multi-face frames.
- Output looks identical to Phase 3.

**Step 4: Commit.**

```bash
git add webapp.py
git commit -m "Phase 4: batch inswapper across faces in the same frame"
```

---

## Phase 5: Multiprocessing fan-out (the big win)

Spawn N worker processes (default 4, env-configurable). Master demuxes frames from the video, dispatches by frame index round-robin via `multiprocessing.shared_memory`, workers do detect+swap+paste, master reorders and writes to ffmpeg.

Each worker owns its own `FaceAnalysis` + `INSwapper` (loaded once at worker startup). VRAM: ~2.5 GB/worker × 4 = ~10 GB. Fits comfortably on a 16 GB Laptop 4090, easily on a 24 GB desktop 4090.

### Task 5.1: Design the IPC protocol

**Files:**
- Create: `server/swap_worker.py` (NEW — even though Flask, keep worker module under server/ for shared use later)

**Step 1: Define the shared-memory frame pool.**

```python
# server/swap_worker.py
from multiprocessing import shared_memory
import numpy as np

class FramePool:
    """Pre-allocated shared-memory slots for in-flight frames. Workers and
    master share the same N×(H*W*3) bytes; each slot is owned by exactly
    one party at a time, tracked via a slot-index queue."""
    def __init__(self, n_slots: int, height: int, width: int):
        self.shape = (height, width, 3)
        self.itemsize = height * width * 3
        self.shms = [shared_memory.SharedMemory(create=True, size=self.itemsize)
                     for _ in range(n_slots)]
        self.arrs = [np.ndarray(self.shape, dtype=np.uint8, buf=shm.buf)
                     for shm in self.shms]

    def write(self, slot: int, frame: np.ndarray) -> None:
        np.copyto(self.arrs[slot], frame)

    def read(self, slot: int) -> np.ndarray:
        # Return a *copy* so caller can mutate freely
        return self.arrs[slot].copy()

    def close(self):
        for shm in self.shms:
            shm.close(); shm.unlink()
```

**Step 2: Define request/response messages.**

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SwapRequest:
    frame_idx: int                  # global frame number for reorder
    slot: int                       # which shm slot holds the input frame
    end: bool = False               # sentinel: no more frames

@dataclass
class SwapResponse:
    frame_idx: int
    slot: int                       # which shm slot holds the output (same as input — workers swap in place)
    swap_count: int
    error: Optional[str] = None
```

### Task 5.2: Write the worker entry point

**Files:**
- Modify: `server/swap_worker.py`

**Step 1: Worker main loop.**

```python
def worker_main(worker_id, in_q, out_q, shm_names, shape,
                ref_embs_bytes, ref_sources_pickled,
                det_size, det_thresh, ref_thresh,
                models_face_dir, inswapper_path):
    """Spawned via multiprocessing.Process. Loads models, processes
    SwapRequests until END."""
    # 1. Load models in this process (each process needs its own ORT
    #    sessions; can't pickle Sessions across fork).
    fa = _load_face_analyser(...)
    sw = _load_inswapper(...)

    # 2. Rehydrate shared state.
    ref_embs = np.frombuffer(ref_embs_bytes, dtype=np.float32).reshape(-1, 512)
    ref_sources = pickle.loads(ref_sources_pickled)

    # 3. Re-attach to the shm pool.
    shms = [shared_memory.SharedMemory(name=n) for n in shm_names]
    arrs = [np.ndarray(shape, dtype=np.uint8, buf=shm.buf) for shm in shms]

    while True:
        req = in_q.get()
        if req.end:
            out_q.put(SwapResponse(frame_idx=-1, slot=-1, swap_count=0))
            break
        frame = arrs[req.slot].copy()
        # detect → swap → paste (same logic as Phase 4 main loop)
        tgt_faces = fa.get(frame)
        picks = _match_picks(tgt_faces, ref_embs, ref_thresh)
        if picks:
            swapped_items = _swap_faces_batched(frame, picks, sw, ref_sources)
            for bgr_fake, M in swapped_items:
                frame = _paste_back(frame, bgr_fake, M)
        np.copyto(arrs[req.slot], frame)        # write result back into same slot
        out_q.put(SwapResponse(frame_idx=req.frame_idx,
                               slot=req.slot,
                               swap_count=len(picks)))
```

**Step 2: Syntax check.**

`conda run -n dlc python -c "import server.swap_worker"` — expect no traceback.

### Task 5.3: Refactor `_run_job` to drive workers

**Files:**
- Modify: `webapp.py` `_run_job`

**Step 1: At the start of streaming phase, spawn workers + create the FramePool.**

```python
import multiprocessing as mp
N_WORKERS = int(os.getenv("FACESWAP_WORKERS", "4"))
POOL_SIZE = N_WORKERS * 4  # 4 slots ahead per worker

frame_pool = FramePool(POOL_SIZE, in_h, in_w)
shm_names = [s.name for s in frame_pool.shms]

ref_embs_bytes = ref_embs.astype(np.float32).tobytes()
ref_sources_pickled = pickle.dumps(ref_sources)

in_qs  = [mp.Queue(maxsize=16) for _ in range(N_WORKERS)]
out_q  = mp.Queue()                       # workers all push to one master queue

procs = []
for w in range(N_WORKERS):
    p = mp.Process(target=worker_main,
                   args=(w, in_qs[w], out_q, shm_names, (in_h, in_w, 3),
                         ref_embs_bytes, ref_sources_pickled,
                         FACESWAP_DET_SIZE, FACESWAP_DET_THRESH, REFERENCE_THRESH,
                         FACE_ANALYSER_MODELS_DIR, SWAPPER_PATH),
                   daemon=True)
    p.start()
    procs.append(p)
```

**Step 2: Main loop becomes a demux+reorder loop.**

```python
free_slots = collections.deque(range(POOL_SIZE))
in_flight = {}       # frame_idx -> slot
done_ready = {}      # frame_idx -> slot (out-of-order completions)
next_to_write = 0
next_to_read  = 0

while True:
    # Dispatch up to POOL_SIZE-K frames ahead
    while free_slots and next_to_read < total_frames:
        ok, fr = cap.read()
        if not ok:
            break
        slot = free_slots.popleft()
        frame_pool.write(slot, fr)
        worker = next_to_read % N_WORKERS
        in_qs[worker].put(SwapRequest(frame_idx=next_to_read, slot=slot))
        in_flight[next_to_read] = slot
        next_to_read += 1

    # Collect any completed frames
    while True:
        try:
            resp = out_q.get(timeout=0.001)
        except queue.Empty:
            break
        done_ready[resp.frame_idx] = resp.slot
        swap_count += resp.swap_count

    # Write any frames whose turn it is, in order
    while next_to_write in done_ready:
        slot = done_ready.pop(next_to_write)
        write_q.put(frame_pool.read(slot))     # copy out; workers may reuse the slot
        free_slots.append(slot)
        next_to_write += 1

    if next_to_write >= total_frames:
        break
```

**Step 3: Tear down on completion.**

```python
for q_ in in_qs:
    q_.put(SwapRequest(frame_idx=-1, slot=-1, end=True))
for _ in range(N_WORKERS):
    out_q.get(timeout=30)  # drain end ACKs
for p in procs:
    p.join(timeout=30)
frame_pool.close()
```

**Step 4: Smoke test (CRITICAL — this is the big refactor).**

Restart Flask. Upload `clip_20s.mp4`.

Watch carefully for:
- All 4 workers spawning (look for `[worker N] models loaded` log lines)
- `nvidia-smi` shows 4 python processes (or 5 incl. master) on GPU.
- Memory accounting: `nvidia-smi --query-gpu=memory.used` ≤ 13 GB on a 16 GB card.
- `proc_fps` 3-4× Phase 4's number.
- swap_count within ±5% of Phase 4 baseline (some non-determinism in tracker/NMS allowed).
- Output MP4 plays + visually matches Phase 4.

**If anything is wrong:** revert this commit, capture the failure mode, do NOT proceed.

**Step 5: Commit.**

```bash
git add webapp.py server/swap_worker.py
git commit -m "Phase 5: multiprocessing fan-out for swap pipeline (N=4 workers default)"
```

**Step 6: Record in `docs/perf-bench.md`.**

---

## Phase 6: NVDEC input via PyAV

Saves CPU decode + the CPU→GPU memcpy. Only worth doing after Phase 5; before that, the master process bottleneck wasn't the decode.

### Task 6.1: Conditional PyAV import + decoder

**Files:**
- Modify: `webapp.py` (reader path inside the master, after Phase 5 lands).
- Modify: `requirements-webapp.txt` (add `av>=11.0`).

**Step 1: Replace cv2.VideoCapture in the master with PyAV.**

```python
import av
container = av.open(job.target_path, options={"hwaccel": "cuda"})
stream = container.streams.video[0]
stream.codec_context.options = {"hwaccel": "cuda"}
fps = float(stream.average_rate)
total_frames = stream.frames
def frame_iter():
    for frame in container.decode(video=0):
        yield frame.to_ndarray(format="bgr24")
```

(Catch: NVDEC outputs in YUV; PyAV does the CPU YUV→BGR conversion. To skip THAT, we'd need direct CUDA decode + zero-copy. Out of scope for this phase — first measure if PyAV-with-NVDEC is faster than cv2.VideoCapture.)

**Step 2: Wire `frame_iter()` into the master's read loop in place of `cap.read()`.**

**Step 3: Smoke test.**

Same clip. Confirm:
- Job completes; `proc_fps` ≥ Phase 5.
- `nvidia-smi dmon -s u`: NVDEC column non-zero during streaming.
- Frame count unchanged.

**Step 4: Commit.**

```bash
git add webapp.py requirements-webapp.txt
git commit -m "Phase 6: NVDEC input via PyAV (hwaccel=cuda)"
```

---

## Final smoke test

After all 6 phases, do a longer test against the Pal-Pal-Dil-Ke-Paas 7902-frame song with sreeni.jpg as `--male`:

```
Expected: ~50 fps proc_fps at 1080p
          ~60-70% GPU util during streaming (`nvidia-smi dmon -s u`)
          ~60-80% CPU util on the i9
          Total wall-clock: ~2.5 minutes (was ~12 min at Phase 0)
          swap_count within ±5% of Phase 0 baseline
          Output MP4 visually identical to Phase 0
```

If actual numbers are lower than this, the next investigation step is to inspect the post-Phase-5 timer outputs: which stage dominates `max(read, detect_p95, swap_p95, paste_p95, write_p95)` × worker_count?

---

## Risks + mitigation

| Risk | Likelihood | Mitigation |
|---|---|---|
| Phase 3 `paste_back=False` API differs across insightface versions | medium | Task 3.1 verifies signature first; commit-per-phase lets us revert |
| Phase 5 worker startup adds 30-60 s per job (re-loading models) | high | First swap job pays the cost; subsequent jobs use the existing workers if we keep them warm — *out of scope for this plan*, accept the cost |
| Phase 5 shared memory leak if a worker crashes | medium | `FramePool.close()` in the master's `finally:` block; worker crashes propagate to master via `out_q` poll |
| Phase 5 head-of-line blocking if one worker is slow | low at N=4 | reorder buffer is `done_ready` dict; small enough to never grow unbounded since we cap `next_to_read - next_to_write <= POOL_SIZE` |
| Phase 5 GPU OOM on Laptop 4090 16 GB | medium | Default N_WORKERS=4 → ~10 GB; env var lets user drop to 2-3 on smaller cards |
| Phase 6 PyAV-with-hwaccel-cuda not actually using NVDEC | medium | nvidia-smi dmon checks before/after; if no NVDEC activity, revert |
| Visual quality regression from any phase | medium | Manual MP4 inspection step in every phase + swap_count comparison |

---

## Roll-back / abort points

After Phase 1: easy revert (one helper class added; pure observation).
After Phase 2: revert to libx264 via `FACESWAP_VIDEO_ENCODER=libx264`.
After Phase 3: `git revert` the commit; pipeline returns to 4-stage.
After Phase 4: `git revert`; per-face loop returns.
After Phase 5: `git revert`; single-process pipeline returns. **This is the high-risk revert; isolate the change to one commit.**
After Phase 6: `git revert`; cv2.VideoCapture returns.

---

## Estimated wall-clock

- Phase 1: 30-45 min (instrumentation + record baseline)
- Phase 2: 15-30 min (env var + smoke test)
- Phase 3: 60-90 min (paste-back split + verify byte-for-byte parity)
- Phase 4: 45-60 min (batching helper + verify identical output)
- Phase 5: 3-4 hours (multiprocessing scaffold + worker module + smoke test + likely 1-2 round-trips of debugging)
- Phase 6: 30-60 min (PyAV decoder + verify NVDEC activity)

Total: ~6-9 hours of focused implementation + smoke testing.
