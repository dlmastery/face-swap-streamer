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
    """Placeholder — Sub-task 5.2 fleshes this into a real shared-memory pool.

    Will own a list[SharedMemory] of fixed-size BGR slots, hand out free
    slots to the master for frame decode, and recycle them once the
    master has written the swapped frame to ffmpeg.
    """
    def __init__(self):
        raise NotImplementedError("FramePool — wired in sub-task 5.2")


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
        ref_embs = pickle.loads(ref_embs_bytes)               # (S, D) ndarray
        ref_sources = pickle.loads(ref_sources_pickled)       # list of SourceSpec-likes
        _ = ref_embs, ref_sources, fa, sw, ref_thresh         # silence "unused" in 5.1

        # Attach shared-memory slots — sub-task 5.2 wires these into per-frame use.
        # For 5.1 we just resolve them once at startup and validate the names.
        from multiprocessing import shared_memory
        slots = []
        for name in shm_names:
            try:
                slots.append(shared_memory.SharedMemory(name=name))
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"[worker-{worker_id}] shared memory '{name}' not found: {e}"
                ) from None
        _ = shape, slots

        # ---- Main loop (sub-task 5.1: end-of-stream-only stub) -----------------
        # Sub-task 5.3 replaces this body with: decode slot -> fa.get -> match -> sw.get
        # -> write back into slot -> SwapResponse(frame_idx, slot_id, n_swapped).
        while True:
            req: SwapRequest = in_q.get()
            if req.end:
                out_q.put(SwapResponse(frame_idx=-1, worker_id=worker_id))
                break
            # Stub: not yet processing frames — should not happen in 5.1.
            out_q.put(SwapResponse(
                frame_idx=req.frame_idx,
                slot_id=req.slot_id,
                worker_id=worker_id,
                error="worker_main stub — sub-task 5.3 not wired yet",
            ))

        # Release the shared-memory handles cleanly.
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
