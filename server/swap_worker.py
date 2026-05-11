"""
Multiprocess swap-worker entry point (Phase 5 of the GPU-saturation plan).

Each worker process loads its own InsightFace `FaceAnalysis` (buffalo_l) +
`INSwapper` (inswapper_128_fp16.onnx) ONE TIME at startup, then runs a
single-threaded loop pulling frame slots out of shared memory, doing
detect + match + fused swap+paste, writing the result back into the
same slot, and acking via the shared result queue.

Why processes, not threads (see Phase 3 post-mortem `3edf180`):
- ORT releases the GIL during inference, but the Python glue around each
  ORT call (numpy slicing, paste-back, attribute lookups on Face objects,
  queue puts/gets) is GIL-serialised. Splitting into more in-process
  threads regressed throughput because the GIL contention got worse.
- Separate processes give each worker its own GIL and its own ORT/CUDA
  context. We pay the cost of model load × N workers at startup, then
  steady-state is fully parallel up to the GPU's compute ceiling.

Loaded in isolation (no webapp dependency) — webapp.py imports
`worker_main` and spawns processes; this module never imports webapp.

Sub-task 5.1: ships the entry point + DLL discovery + model loading.
Sub-task 5.2 will add SwapRequest / SwapResponse / FramePool.
"""
from __future__ import annotations
import os
import sys
import time
import pickle
import traceback
from dataclasses import dataclass
from typing import Optional


# ---- Windows DLL discovery for onnxruntime + CUDA ---------------------------
# Each spawned worker process gets a fresh Python interpreter; the
# os.add_dll_directory cookies from the parent (webapp.py) do NOT propagate.
# This block MUST run BEFORE `import onnxruntime` / `import insightface`.
# See CLAUDE.md issue #1.

_dll_cookies = []   # keep cookies alive — GC'd cookies = lost search paths


def _register_cuda_dll_dirs() -> None:
    """Add nvidia-*-cu12 and tensorrt_libs DLL dirs to Windows' secure DLL
    search path. Safe to call multiple times — duplicate dirs are silently
    ignored by Windows. No-op on non-Windows.
    """
    if sys.platform != "win32":
        return
    sp = os.path.join(sys.prefix, "Lib", "site-packages")
    bin_dirs = [
        # nvidia-cudnn-cu12, nvidia-cublas-cu12, etc.
        *(os.path.join(sp, "nvidia", sub, "bin")
          for sub in ("cudnn", "cublas", "cuda_runtime", "curand", "cufft",
                      "cuda_nvrtc", "nvjitlink")),
        # tensorrt-cu12 puts its DLLs under tensorrt_libs/ — different layout
        os.path.join(sp, "tensorrt_libs"),
    ]
    for b in bin_dirs:
        if os.path.isdir(b):
            try:
                _dll_cookies.append(os.add_dll_directory(b))
            except OSError:
                pass
            os.environ["PATH"] = b + os.pathsep + os.environ["PATH"]


_register_cuda_dll_dirs()


# ---- IPC dataclasses (Sub-task 5.2 will flesh these out) --------------------

@dataclass
class SwapRequest:
    """Master -> worker: process the frame in `slot_id` (frame index `frame_idx`).

    If `end` is True, all other fields are ignored — the worker should ack
    with a SwapResponse(frame_idx=-1) and exit cleanly.
    """
    frame_idx: int = -1
    slot_id: int = -1
    end: bool = False


@dataclass
class SwapResponse:
    """Worker -> master: frame at `frame_idx` is done in `slot_id`.

    `n_swapped` is how many faces this worker swapped into the frame
    (used by master to accumulate `swap_count`). `worker_id` is the
    sending worker so the master can debug per-worker stalls.

    If `frame_idx == -1` this is the end-of-stream ack.
    `error` is non-empty if the worker hit an exception while processing
    this frame — master should treat that as a fatal job error.
    """
    frame_idx: int = -1
    slot_id: int = -1
    n_swapped: int = 0
    worker_id: int = -1
    elapsed_ms: float = 0.0
    error: str = ""


class FramePool:
    """Pool of fixed-size shared-memory slots backing one BGR frame each.

    Owned by the master process; workers `attach` to slots by name via
    `multiprocessing.shared_memory.SharedMemory(name=...)`. Each slot is
    a contiguous uint8 buffer sized to fit `H * W * 3` bytes; the same
    buffer holds the input frame on dispatch and the swapped output on
    return (workers overwrite in place; master copies out to ffmpeg
    before recycling the slot).

    Sizing: at 1080p one slot is ~6.2 MB. With N_workers=4 and 4 slots
    per worker (16 slots) that's ~100 MB of pinned shared RAM — fine.

    Lifecycle from the master's POV:
        pool = FramePool(n_slots=N_workers * 4, shape=(H, W, 3))
        slot = pool.acquire()                 # blocks if pool empty
        arr  = pool.view(slot)                # numpy view, no copy
        arr[...] = decoded_frame              # fill in place
        in_q.put(SwapRequest(frame_idx=k, slot_id=slot))
        # ... worker processes, writes result back into the same slot ...
        resp = out_q.get()
        out_arr = pool.view(resp.slot_id)
        ffmpeg.stdin.write(out_arr.tobytes()) # or out_arr.copy() first
        pool.release(resp.slot_id)            # back into the free pool

    Workers never call acquire/release — they just `view(slot_id)` the
    slot the master picked, mutate it in place, and ack.

    Cleanup: `close()` releases the SharedMemory handles; on Windows
    Python's reference counter unlinks them when the last handle is
    dropped. We additionally call `unlink()` defensively on master shutdown
    because dangling shm names linger across process crashes on some
    Windows builds.
    """

    def __init__(self, n_slots: int, shape: tuple, dtype=None):
        import numpy as np
        from multiprocessing import shared_memory
        if dtype is None:
            dtype = np.uint8
        self.n_slots = int(n_slots)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self._np = np
        self._shm_mod = shared_memory

        nbytes = int(np.prod(self.shape) * self.dtype.itemsize)
        self.nbytes = nbytes
        self._shms = []                # list[SharedMemory]
        self._views = []               # list[np.ndarray] — one per slot
        for _ in range(self.n_slots):
            shm = shared_memory.SharedMemory(create=True, size=nbytes)
            self._shms.append(shm)
            self._views.append(
                np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
            )

        # Free-slot pool. Uses a thread-safe Queue so the master's
        # demux thread can `acquire()` (block on empty) while a
        # background result-drainer calls `release()` from another
        # thread. We keep it bounded to n_slots so accidental
        # double-release is loud (`Full`) instead of silent.
        import queue as _queue
        self._free: "_queue.Queue[int]" = _queue.Queue(maxsize=self.n_slots)
        for i in range(self.n_slots):
            self._free.put(i)

    @property
    def names(self) -> list:
        """Pass these to each worker so it can attach to the same slots."""
        return [s.name for s in self._shms]

    def acquire(self, timeout: Optional[float] = None) -> int:
        """Block until a slot is free; return its id. `timeout=None` waits
        forever; pass a small timeout to detect master stalls."""
        return self._free.get(timeout=timeout)

    def release(self, slot_id: int) -> None:
        """Return a slot to the free pool. Raises queue.Full on double-release."""
        self._free.put_nowait(int(slot_id))

    def view(self, slot_id: int):
        """Zero-copy numpy view of the slot. Master and worker both call this."""
        return self._views[int(slot_id)]

    def free_count(self) -> int:
        return self._free.qsize()

    def close(self) -> None:
        """Drop all SharedMemory handles + unlink on POSIX/Windows. Idempotent."""
        for shm in self._shms:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except (FileNotFoundError, OSError):
                # Already unlinked, or platform doesn't require it.
                pass
        self._shms.clear()
        self._views.clear()


# ---- Worker entry point ----------------------------------------------------

def worker_main(
    worker_id: int,
    in_q,                       # mp.Queue[SwapRequest]
    out_q,                      # mp.Queue[SwapResponse]
    shm_names: list,            # list[str] of SharedMemory names (one per slot)
    shape: tuple,               # (H, W, 3) — slot frame shape
    ref_embs_bytes: bytes,      # pickled numpy.ndarray, shape (S, D)
    ref_sources_pickled: bytes, # pickled list[SourceSpec-like dict]
    det_size: int,              # face detector input (square)
    det_thresh: float,          # detector confidence threshold
    ref_thresh: float,          # cosine-sim threshold for source-match
    models_face_dir: Optional[str],  # FACESWAP_FACE_MODEL or None
    inswapper_path: str,        # absolute path to inswapper_128_fp16.onnx
) -> None:
    """Process entry. Loads models, then loops on `in_q` until END.

    Sub-task 5.1: stub — load models, ack startup, drain in_q with end-of-stream
    response only. Real per-frame swap logic lands in sub-task 5.3 (after the
    master wiring is also in place — keeps the diff reviewable).
    """
    t0 = time.perf_counter()
    try:
        # Late imports — must follow _register_cuda_dll_dirs() above.
        # cv2 + numpy first; they're cheap and don't touch CUDA.
        import numpy as np  # noqa: F401  (used once SwapResponse handling lands)
        import cv2          # noqa: F401  (used by paste-back internals)
        import insightface
        from insightface.app import FaceAnalysis

        face_model = models_face_dir or "buffalo_l"
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        print(f"[worker-{worker_id}] loading FaceAnalysis({face_model}) "
              f"det_size={det_size} det_thresh={det_thresh}...", flush=True)
        fa = FaceAnalysis(name=face_model, providers=providers)
        fa.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=det_thresh)

        print(f"[worker-{worker_id}] loading inswapper from {inswapper_path}...",
              flush=True)
        sw = insightface.model_zoo.get_model(inswapper_path, providers=providers)

        # Verify CUDA actually loaded — see CLAUDE.md issue #8.
        try:
            active = sw.session.get_providers()
        except AttributeError:
            active = None
        if active and active == ["CPUExecutionProvider"]:
            raise RuntimeError(
                f"[worker-{worker_id}] inswapper loaded on CPU only — "
                f"CUDA failed to initialise. Check cuDNN DLL discovery."
            )

        load_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[worker-{worker_id}] models loaded in {load_ms:.0f} ms "
              f"(providers={active})", flush=True)

        # Unpickle reference embeddings + source faces (master computed these once).
        # ref_embs: (S, D) float32 normed embeddings, one row per source.
        # ref_sources: list[dict] with key 'src_face' holding the source-image Face
        #              that inswapper will read identity from. We pass dicts (not the
        #              SourceSpec dataclass) so this module has no webapp dependency.
        ref_embs = pickle.loads(ref_embs_bytes)
        ref_sources = pickle.loads(ref_sources_pickled)
        ref_embs_T = ref_embs.T  # (D, S) — pre-transpose for the per-frame matmul

        # Attach the shared-memory slots and build per-slot numpy views ONCE.
        # Per-frame we just index into `slot_views[slot_id]`; no slicing or
        # SharedMemory lookup in the hot loop.
        from multiprocessing import shared_memory
        slots = []
        slot_views = []
        for name in shm_names:
            try:
                shm = shared_memory.SharedMemory(name=name)
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"[worker-{worker_id}] shared memory '{name}' not found: {e}"
                ) from None
            slots.append(shm)
            slot_views.append(np.ndarray(shape, dtype=np.uint8, buffer=shm.buf))

        # Tell master we're ready (frame_idx=-2 is the startup-ack convention).
        out_q.put(SwapResponse(frame_idx=-2, worker_id=worker_id))

        # ---- Main per-frame loop -----------------------------------------------
        # In-process pipeline is intentionally single-threaded: every per-frame
        # step (decode-from-shm, detect, match, fused swap+paste, encode-to-shm)
        # is GIL-serialised against itself anyway, and parallelism comes from
        # running N of these processes in parallel. Adding threads here just
        # brings back the GIL contention we measured in Phase 3.
        while True:
            req: SwapRequest = in_q.get()
            if req.end:
                out_q.put(SwapResponse(frame_idx=-1, worker_id=worker_id))
                break

            tA = time.perf_counter()
            slot_id = int(req.slot_id)
            frame_idx = int(req.frame_idx)
            try:
                # In-place numpy view of the slot; master already wrote the input
                # BGR frame here. We mutate it in place with the swap result.
                frame = slot_views[slot_id]

                # Detect + match against the pre-stacked source references.
                tgt_faces = fa.get(frame)
                n_swapped = 0
                if tgt_faces:
                    tgt_embs = np.stack([f.normed_embedding for f in tgt_faces])
                    sims = tgt_embs @ ref_embs_T          # (T, S)
                    for ti, tface in enumerate(tgt_faces):
                        si = int(np.argmax(sims[ti]))
                        if float(sims[ti, si]) >= ref_thresh:
                            # Fused swap+paste — same call shape as the
                            # in-process Phase 2 path; matches its output byte-for-byte
                            # (same model, same source face, same target face).
                            swapped = sw.get(frame, tface,
                                             ref_sources[si]["src_face"],
                                             paste_back=True)
                            # insightface returns a NEW ndarray for the swapped frame
                            # (not in-place). Copy it back into the shared-memory slot
                            # so the master sees the result without an extra IPC payload.
                            frame[...] = swapped
                            n_swapped += 1

                elapsed = (time.perf_counter() - tA) * 1000.0
                out_q.put(SwapResponse(
                    frame_idx=frame_idx, slot_id=slot_id,
                    n_swapped=n_swapped, worker_id=worker_id,
                    elapsed_ms=elapsed,
                ))
            except Exception as e:
                # Per-frame failure shouldn't kill the worker — report and continue.
                # Master decides whether a single-frame error is fatal.
                tb = traceback.format_exc()
                print(f"[worker-{worker_id}] frame {frame_idx} error: {e}\n{tb}",
                      flush=True)
                out_q.put(SwapResponse(
                    frame_idx=frame_idx, slot_id=slot_id,
                    worker_id=worker_id,
                    error=f"{type(e).__name__}: {e}",
                ))

        # Release the shared-memory handles cleanly (master owns + unlinks).
        for s in slots:
            try:
                s.close()
            except Exception:
                pass
        print(f"[worker-{worker_id}] exited cleanly", flush=True)

    except Exception as e:
        # Surface to master so the job fails loudly instead of stalling.
        tb = traceback.format_exc()
        print(f"[worker-{worker_id}] FATAL: {e}\n{tb}", flush=True)
        try:
            out_q.put(SwapResponse(
                frame_idx=-1, worker_id=worker_id,
                error=f"{type(e).__name__}: {e}",
            ))
        except Exception:
            pass
        sys.exit(1)
