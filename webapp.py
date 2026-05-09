"""
faceswap webapp:
  upload image + mp4 -> auto-detect source gender -> auto-extract matching reference
  from video -> live MJPEG stream in browser -> audio-muxed mp4 download.

Run:
    conda run -n dlc python webapp.py
Then open http://localhost:8080/
"""
from __future__ import annotations
import os
import sys
import glob
import time
import uuid
import threading
import queue
import subprocess
from dataclasses import dataclass, field
from typing import Optional

# ---- Win Py 3.8+ secure DLL search: register CUDA + TensorRT dirs before
#      onnxruntime import. PATH alone is NOT enough on Py 3.8+ Windows;
#      os.add_dll_directory cookies must be kept alive (don't GC them).
_dll_cookies = []
if sys.platform == "win32":
    _sp = os.path.join(sys.prefix, "Lib", "site-packages")
    _bin_dirs = [
        # nvidia-cudnn-cu12, nvidia-cublas-cu12, etc.
        *(os.path.join(_sp, "nvidia", sub, "bin")
          for sub in ("cudnn", "cublas", "cuda_runtime", "curand", "cufft",
                      "cuda_nvrtc", "nvjitlink")),
        # tensorrt-cu12 puts its DLLs at site-packages/tensorrt_libs/ (different layout)
        os.path.join(_sp, "tensorrt_libs"),
    ]
    for _bin in _bin_dirs:
        if os.path.isdir(_bin):
            try:
                _dll_cookies.append(os.add_dll_directory(_bin))
            except OSError:
                pass
            os.environ["PATH"] = _bin + os.pathsep + os.environ["PATH"]

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from flask import Flask, request, jsonify, Response, redirect, url_for, send_from_directory


# ---- Configuration ---------------------------------------------------------

ROOT = r"C:\Users\evija\faceswap"
JOBS_DIR = os.path.join(ROOT, "webapp_jobs")
os.makedirs(JOBS_DIR, exist_ok=True)
SWAPPER_PATH = os.path.join(ROOT, "deep-live-cam", "models", "inswapper_128_fp16.onnx")

FFMPEG_EXE = next((p for p in [
    r"C:\Users\evija\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe",
    "ffmpeg",
] if p == "ffmpeg" or os.path.isfile(p)), None)


# ---- Job state -------------------------------------------------------------

@dataclass
class Job:
    id: str
    source_path: str
    target_path: str
    out_audio_path: str          # final audio-muxed MP4 (download)
    hls_dir: str                 # dir holding playlist.m3u8 + seg_*.ts
    phase: str = "queued"
    message: str = "Queued"
    detected_gender: str = ""
    detected_age: int = 0
    ref_frame: int = -1
    ref_votes: int = 0
    ref_pool: int = 0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    total_frames: int = 0
    current_frame: int = 0
    swap_count: int = 0
    proc_fps: float = 0.0
    error: str = ""
    started: float = field(default_factory=time.time)
    finished: float = 0.0
    stop_flag: threading.Event = field(default_factory=threading.Event)


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()

# Lazy-loaded global models (one set, shared across jobs — single-user assumption)
_models_lock = threading.Lock()
_face_analyser: Optional[FaceAnalysis] = None
_swapper = None


def _ensure_models():
    """Lazy-load the face analyser + inswapper. Called once on server start
    (background thread) and again at the top of every job (no-op if already
    loaded).

    Provider strategy:
      - face analyser: CUDA only (its models have dynamic shapes that TRT
        can't compile efficiently; CUDA is fast enough at 640x640).
      - inswapper: TensorRT if the `tensorrt` pip package is installed
        (30-50%% faster on RTX cards), otherwise plain CUDA. We never list
        TRT as a provider unless the lib is actually present — onnxruntime
        silently falls back ALL the way to CPU if its first listed provider
        fails to initialise, which is much worse than just using CUDA.
      - After load, we verify the active provider is CUDA (or TRT). If it
        somehow fell back to CPU, we raise so the failure is visible
        instead of grinding through frames at 1 fps.
    """
    global _face_analyser, _swapper
    with _models_lock:
        if _face_analyser is None:
            # det_size 480 vs 640: detection runs ~1.7x faster at 480 with only
            # a small drop in detection of small/profile faces. Detection is
            # the slowest single GPU stage on RTX cards, so this is a big lever.
            # If you find faces being missed in dance/wide shots, bump back to 640.
            det_size = int(os.getenv("FACESWAP_DET_SIZE", "480"))
            print(f"[webapp] loading face analyser (CUDA, det_size={det_size})...", flush=True)
            fa = FaceAnalysis(name="buffalo_l",
                              providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            fa.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=0.4)
            _face_analyser = fa

        if _swapper is None:
            # Detect TensorRT availability: the python package, the EP wheel,
            # AND the actual nvinfer DLLs all need to be present.
            trt_available = False
            try:
                import tensorrt  # noqa: F401
                # If we can also locate the EP shared library, TRT is usable.
                ort_providers = set(ort.get_available_providers()) if (ort := __import__("onnxruntime")) else set()
                trt_available = "TensorrtExecutionProvider" in ort_providers
            except ImportError:
                trt_available = False

            if trt_available:
                trt_cache = os.path.join(JOBS_DIR, ".trt_cache")
                os.makedirs(trt_cache, exist_ok=True)
                providers = [
                    ("TensorrtExecutionProvider", {
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": trt_cache,
                        "trt_fp16_enable": True,
                        "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,
                    }),
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]
                print("[webapp] loading inswapper (TensorRT + CUDA, ~60-90s engine build on first run)...", flush=True)
            else:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                print("[webapp] loading inswapper (CUDA — `pip install tensorrt-cu12` in dlc env for ~30-50%% extra speed)...", flush=True)

            _swapper = insightface.model_zoo.get_model(SWAPPER_PATH, providers=providers)

            # Verify CUDA actually loaded — onnxruntime silently falls back
            # to CPU on EP init failures (e.g. cuDNN missing). Catch this
            # loudly: a job running on CPU is much worse than a startup error.
            try:
                active = _swapper.session.get_providers()
            except AttributeError:
                active = None
            print(f"[webapp] inswapper active providers: {active}", flush=True)
            if active and active == ["CPUExecutionProvider"]:
                raise RuntimeError(
                    "inswapper loaded on CPU only — CUDA failed to initialise. "
                    "Run `conda run -n dlc python test-cuda-dlc.py` to diagnose; "
                    "usually means cuDNN/cuBLAS DLLs aren't on the DLL search path "
                    "(see CLAUDE.md issue #1)."
                )


# ---- Job worker ------------------------------------------------------------

def _set(job: Job, **kw):
    for k, v in kw.items():
        setattr(job, k, v)


def _spawn_ffmpeg(job: Job, w: int, h: int, fps: float) -> subprocess.Popen:
    """Spawn ffmpeg: BGR frames on stdin + audio from target.mp4, h264+aac out,
    tee muxer writes both live HLS and a fragmented MP4. Audio is muxed in DURING
    streaming so the browser hears it live.

    NB: we run ffmpeg with cwd=<job_dir> so paths inside the tee URL are relative.
    Windows drive-letter colons (C:/...) collide with tee's `:` option separator,
    so absolute paths break it silently — relative paths sidestep that entirely.
    """
    if not FFMPEG_EXE:
        raise RuntimeError("ffmpeg not found — install Gyan.FFmpeg or anaconda's ffmpeg")
    os.makedirs(job.hls_dir, exist_ok=True)
    job_dir = os.path.dirname(job.out_audio_path)
    target_abs = os.path.abspath(job.target_path)

    # All paths in the tee URL are relative to job_dir.
    playlist = "hls/playlist.m3u8"
    seg_pattern = "hls/seg_%05d.ts"
    final_mp4 = os.path.basename(job.out_audio_path)

    # Tee muxer: one encode pass, two outputs (live HLS + downloadable MP4).
    # MP4 uses fragmented format (frag_keyframe+empty_moov) so it can be written
    # progressively without seeking back to the moov atom.
    tee = (
        f"[f=hls:hls_time=2:hls_list_size=0:"
        f"hls_flags=independent_segments+append_list:"
        f"hls_segment_filename={seg_pattern}]{playlist}"
        f"|"
        f"[f=mp4:movflags=+frag_keyframe+empty_moov+default_base_moof]{final_mp4}"
    )

    cmd = [
        FFMPEG_EXE, "-y", "-hide_banner", "-loglevel", "info",
        # input 0: raw BGR video from python stdin
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{w}x{h}", "-framerate", str(fps),
        "-i", "pipe:0",
        # input 1: original target file (for its audio track) — absolute is fine here
        "-i", target_abs,
        # take video from input 0, audio (if any) from input 1
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
        "-g", str(int(round(fps * 2))),
        "-keyint_min", str(int(round(fps * 2))),
        "-sc_threshold", "0",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "44100",
        "-shortest",
        "-f", "tee", tee,
    ]
    print(f"[webapp] ffmpeg cwd={job_dir} cmd={' '.join(cmd[:8])}... tee(hls,mp4)", flush=True)
    # Capture stderr so we can surface real errors. Drain it on a background thread
    # to avoid the OS pipe filling up and stalling ffmpeg.
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            cwd=job_dir, bufsize=0)
    # Tail ffmpeg's stderr to a per-job log so we can debug failures.
    log_path = os.path.join(job_dir, "ffmpeg.log")
    def _drain():
        with open(log_path, "wb") as f:
            for line in iter(proc.stderr.readline, b""):
                f.write(line); f.flush()
    threading.Thread(target=_drain, daemon=True).start()
    return proc


def _run_job(job: Job):
    ffmpeg = None
    try:
        _set(job, phase="loading_models", message="Loading face-swap models (one-time, ~30s)…")
        _ensure_models()
        fa = _face_analyser
        sw = _swapper

        _set(job, phase="detecting_source", message="Detecting your face in the source image…")
        src_bgr = cv2.imread(job.source_path)
        if src_bgr is None:
            raise RuntimeError("could not read source image")
        src_faces = fa.get(src_bgr)
        if not src_faces:
            raise RuntimeError("no face detected in source image — try a clearer, front-facing photo")
        src_face = max(src_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        gender = src_face.sex
        _set(job, detected_gender=gender, detected_age=int(src_face.age))

        cap = cv2.VideoCapture(job.target_path)
        if not cap.isOpened():
            raise RuntimeError("could not open target video")
        in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _set(job, width=in_w, height=in_h, fps=float(fps), total_frames=total)

        # auto-extract reference embedding
        _set(job, phase="finding_reference",
             message=f"Scanning video for the {gender} face to swap onto…")
        step = max(1, int(fps * 2.0))
        candidates = []
        i = 0
        while i < total and len(candidates) < 60:
            if job.stop_flag.is_set():
                raise RuntimeError("cancelled")
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok:
                break
            faces = [f for f in fa.get(fr) if f.sex == gender]
            if faces:
                best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * f.det_score)
                if (best.bbox[2] - best.bbox[0]) >= 50:
                    candidates.append((float((best.bbox[2] - best.bbox[0]) * best.det_score),
                                       best.normed_embedding, i))
            i += step
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not candidates:
            raise RuntimeError(f"no {gender} face found in the video")

        embs = np.stack([c[1] for c in candidates])
        sim = embs @ embs.T
        votes = (sim > 0.30).sum(axis=1)
        winner = int(np.argmax(votes * np.array([c[0] for c in candidates])))
        ref_emb = candidates[winner][1]
        _set(job, ref_frame=int(candidates[winner][2]),
             ref_votes=int(votes[winner]), ref_pool=len(candidates))

        # ffmpeg HLS+MP4 pipeline
        ffmpeg = _spawn_ffmpeg(job, in_w, in_h, fps)

        _set(job, phase="streaming",
             message=f"Streaming swap (locked onto {gender} face from frame {job.ref_frame}) — audio is included")
        REFERENCE_THRESH = 0.22

        # ---- Async reader + writer threads --------------------------------------
        # Reader: cv2.read() into read_q so the main loop never blocks on decode.
        # Writer: drains write_q into ffmpeg.stdin so the main loop never blocks
        # on the pipe. Both queues are bounded → memory is capped, full queues
        # provide natural backpressure.
        # Q_DEPTH=64 → ~800 MB at 1080p (each queue holds raw BGR), plenty of
        # slack for momentary read/write stalls (e.g. ffmpeg flushing a HLS
        # segment every 2 s) while still bounded. With this much buffer the
        # reader can run several seconds ahead of the swap loop, so the GPU
        # is never starved by I/O hiccups.
        Q_DEPTH = 64
        END = object()
        read_q: "queue.Queue[object]" = queue.Queue(maxsize=Q_DEPTH)
        write_q: "queue.Queue[object]" = queue.Queue(maxsize=Q_DEPTH)
        broken = False

        def _reader_loop():
            try:
                while not job.stop_flag.is_set():
                    ok, fr = cap.read()
                    if not ok:
                        break
                    read_q.put(fr)
            finally:
                read_q.put(END)

        def _writer_loop():
            nonlocal broken
            while True:
                item = write_q.get()
                if item is END:
                    return
                try:
                    ffmpeg.stdin.write(item)
                except (BrokenPipeError, OSError):
                    broken = True
                    while True:
                        x = write_q.get()
                        if x is END:
                            return

        t_reader = threading.Thread(target=_reader_loop, daemon=True,
                                    name=f"job-{job.id}-reader")
        t_writer = threading.Thread(target=_writer_loop, daemon=True,
                                    name=f"job-{job.id}-writer")
        t_reader.start()
        t_writer.start()

        n = 0
        swap_count = 0
        t0 = time.time()
        last_log = t0
        try:
            while True:
                if job.stop_flag.is_set():
                    raise RuntimeError("cancelled")
                if broken:
                    break
                item = read_q.get()
                if item is END:
                    break
                frame = item
                n += 1
                tgt_faces = fa.get(frame)
                picked = None
                best_sim = -1.0
                for f in tgt_faces:
                    s = float(np.dot(f.normed_embedding, ref_emb))
                    if s > best_sim:
                        best_sim = s
                        picked = f
                if best_sim < REFERENCE_THRESH:
                    picked = None

                if picked is not None:
                    frame = sw.get(frame, picked, src_face, paste_back=True)
                    swap_count += 1

                # Hand off to the writer thread (will block if the queue is full,
                # which is fine — provides natural backpressure on the pipeline).
                if broken:
                    break
                write_q.put(frame.tobytes())

                now = time.time()
                if now - last_log > 0.5:
                    elapsed = now - t0
                    job.current_frame = n
                    job.swap_count = swap_count
                    job.proc_fps = n / elapsed if elapsed else 0.0
                    last_log = now
        finally:
            # Tell the writer to drain and exit, even on exceptions. The reader
            # exits on its own when cap.read() returns False; if we're bailing
            # early, set the stop_flag so it stops promptly on its next iter.
            job.stop_flag.set()
            # Drain any remaining items the reader might have queued so its
            # final put(END) doesn't block on a full queue and hang join().
            try:
                while True:
                    read_q.get_nowait()
            except queue.Empty:
                pass
            write_q.put(END)
            t_writer.join(timeout=30)
            t_reader.join(timeout=10)
            cap.release()
        # close ffmpeg cleanly so it writes the HLS endlist + finalises MP4
        try:
            ffmpeg.stdin.close()
        except Exception:
            pass
        _set(job, phase="finalising", message="Finalising MP4 + HLS playlist…")
        try:
            ffmpeg.wait(timeout=60)
        except subprocess.TimeoutExpired:
            ffmpeg.kill()
            raise RuntimeError("ffmpeg did not exit in time")
        if ffmpeg.returncode not in (0, None) and not broken:
            # ffmpeg stderr is drained to <job_dir>/ffmpeg.log by the spawn helper
            log_path = os.path.join(os.path.dirname(job.out_audio_path), "ffmpeg.log")
            err = ""
            try:
                with open(log_path, "rb") as f:
                    err = f.read().decode(errors="replace")
            except Exception:
                pass
            raise RuntimeError(f"ffmpeg failed (rc={ffmpeg.returncode}): {err[-800:]}")

        _set(job, phase="done", message="Done — audio + video saved", finished=time.time(),
             current_frame=n, swap_count=swap_count,
             proc_fps=n / max(time.time() - t0, 1e-6))
    except Exception as e:
        if ffmpeg is not None:
            try: ffmpeg.kill()
            except Exception: pass
        _set(job, phase="error", message=str(e), error=str(e), finished=time.time())
        print(f"[webapp] job {job.id} error: {e}", flush=True)


# ---- Flask app -------------------------------------------------------------

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4 GB upload limit
app.config["MAX_FORM_MEMORY_SIZE"] = 4 * 1024 * 1024 * 1024


@app.errorhandler(413)
def too_large(_):
    return "File too large (4 GB limit)", 413


def _cleanup_old_jobs(keep_hours: float = 6.0) -> None:
    """Remove job dirs older than keep_hours so disk doesn't fill up."""
    cutoff = time.time() - keep_hours * 3600
    try:
        for d in os.listdir(JOBS_DIR):
            full = os.path.join(JOBS_DIR, d)
            if not os.path.isdir(full):
                continue
            try:
                if os.path.getmtime(full) < cutoff and d not in JOBS:
                    import shutil
                    shutil.rmtree(full, ignore_errors=True)
            except Exception:
                pass
    except Exception:
        pass

INDEX_HTML = r"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><title>Faceswap · live stream</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root {
    color-scheme: dark;
    --bg-0:#05060c; --bg-1:#0c0f1c; --bg-2:#13182a;
    --ink-0:#f6f8fc; --ink-1:#c5cce0; --ink-2:#8c95b0;
    --accent-1:#7a5cff; --accent-2:#3aa1ff; --accent-3:#ff5cb1;
    --good:#52d6a3; --line:rgba(255,255,255,.07);
  }
  *, *::before, *::after { box-sizing: border-box; }
  html, body { margin:0; padding:0; }
  body {
    font-family: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--ink-0);
    background: var(--bg-0);
    min-height:100vh;
    overflow-x:hidden;
    -webkit-font-smoothing: antialiased;
  }

  /* animated aurora background */
  .aurora { position:fixed; inset:0; z-index:-2; overflow:hidden; background:var(--bg-0); }
  .aurora::before, .aurora::after, .aurora .blob {
    content:""; position:absolute; border-radius:50%; filter: blur(80px);
    opacity:.55; will-change: transform;
  }
  .aurora::before {
    width:600px; height:600px; left:-150px; top:-150px;
    background: radial-gradient(circle, var(--accent-1), transparent 60%);
    animation: float1 24s ease-in-out infinite;
  }
  .aurora::after {
    width:700px; height:700px; right:-200px; top:5%;
    background: radial-gradient(circle, var(--accent-2), transparent 60%);
    animation: float2 30s ease-in-out infinite;
  }
  .aurora .blob {
    width:550px; height:550px; left:30%; bottom:-200px;
    background: radial-gradient(circle, var(--accent-3), transparent 60%);
    animation: float3 36s ease-in-out infinite;
  }
  @keyframes float1 { 0%,100% { transform: translate(0,0) scale(1); }
                      50% { transform: translate(140px,80px) scale(1.1); } }
  @keyframes float2 { 0%,100% { transform: translate(0,0) scale(1); }
                      50% { transform: translate(-120px,140px) scale(1.05); } }
  @keyframes float3 { 0%,100% { transform: translate(0,0) scale(1); }
                      50% { transform: translate(80px,-100px) scale(1.15); } }

  /* film-grain overlay */
  .grain { position:fixed; inset:0; z-index:-1; pointer-events:none;
           opacity:.15; mix-blend-mode:overlay;
           background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9'/></filter><rect width='200' height='200' filter='url(%23n)' opacity='.5'/></svg>"); }

  header.top {
    position:sticky; top:0; z-index:10;
    padding:1rem 1.5rem; display:flex; align-items:center; justify-content:space-between;
    border-bottom:1px solid var(--line);
    backdrop-filter: blur(14px); background: rgba(5,6,12,0.45);
  }
  .brand { display:flex; align-items:center; gap:.6rem; font-weight:700; letter-spacing:-.01em; }
  .brand .dot { width:10px; height:10px; border-radius:50%;
                background: linear-gradient(135deg, var(--accent-1), var(--accent-3));
                box-shadow: 0 0 18px var(--accent-1); }
  .top a { color: var(--ink-1); text-decoration:none; font-size:.9rem; }

  main { max-width:1200px; margin:0 auto; padding: 4rem 1.5rem 6rem; }

  .hero { text-align:center; margin-bottom:3.5rem; }
  .hero .eyebrow {
    display:inline-block; padding:.4rem .9rem; border-radius:999px;
    background: rgba(122, 92, 255, 0.08); border:1px solid rgba(122, 92, 255, 0.3);
    color: #c4b3ff; font-size:.78rem; font-weight:500; letter-spacing:.06em;
    margin-bottom:1.2rem; text-transform: uppercase;
  }
  .hero h1 {
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    line-height:1.05; font-weight:800; letter-spacing:-.03em;
    margin: 0 0 1.1rem;
    background: linear-gradient(135deg, #ffffff 0%, #c5cce0 50%, #7a5cff 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  .hero p {
    color: var(--ink-1); font-size: clamp(1rem, 1.4vw, 1.18rem);
    line-height:1.55; max-width: 640px; margin: 0 auto;
  }

  /* Upload card */
  .card {
    background: linear-gradient(180deg, rgba(20,26,42,0.65) 0%, rgba(13,16,28,0.8) 100%);
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 2.2rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 30px 80px rgba(0,0,0,.45),
                inset 0 1px 0 rgba(255,255,255,.06);
    max-width: 920px; margin: 0 auto;
  }

  .drop-row { display:grid; grid-template-columns: 1fr 80px 1fr; gap:1.2rem; align-items:stretch; }
  @media (max-width: 700px) {
    .drop-row { grid-template-columns: 1fr; }
    .drop-row .arrow { transform: rotate(90deg); margin: -1rem auto; }
  }
  .arrow {
    display:flex; align-items:center; justify-content:center;
    color: var(--accent-1); font-size:1.8rem;
    animation: pulse 2.4s ease-in-out infinite;
  }
  @keyframes pulse {
    0%,100% { opacity:.55; transform: translateX(0); }
    50% { opacity:1; transform: translateX(6px); }
  }

  .drop {
    position:relative; border:2px dashed rgba(255,255,255,.12);
    border-radius:18px; padding:1.6rem;
    background: rgba(8,10,18,0.4);
    transition: all .2s ease;
    cursor:pointer; min-height: 220px;
    display:flex; flex-direction:column; align-items:center; justify-content:center; gap:.6rem;
    text-align:center;
  }
  .drop:hover { border-color: rgba(122,92,255,0.5); background: rgba(122,92,255,0.05); transform: translateY(-2px); }
  .drop.over { border-color: var(--accent-1); background: rgba(122,92,255,0.1); }
  .drop.has-file { border-style:solid; border-color: rgba(82,214,163,0.4); background: rgba(82,214,163,0.05); }
  .drop input[type=file] { position:absolute; inset:0; opacity:0; cursor:pointer; }
  .drop .icon {
    width:46px; height:46px; border-radius:12px;
    background: linear-gradient(135deg, rgba(122,92,255,0.2), rgba(58,161,255,0.2));
    display:flex; align-items:center; justify-content:center;
    color: var(--accent-2); font-size:1.6rem;
    border: 1px solid rgba(122,92,255,0.3);
  }
  .drop .label { font-weight:600; font-size:.95rem; color: var(--ink-0); }
  .drop .hint { font-size:.8rem; color: var(--ink-2); }
  .drop .preview { width:100%; max-width: 220px; aspect-ratio: 16/10;
                   background:#000; border-radius:10px; overflow:hidden; margin-top:.4rem;
                   display:flex; align-items:center; justify-content:center; }
  .drop .preview img, .drop .preview video {
    width:100%; height:100%; object-fit: cover;
  }
  .drop .filename {
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size:.8rem; color: var(--good); word-break: break-all;
    max-width: 100%;
  }

  .actions-row { margin-top:2rem; display:flex; gap:1rem; align-items:center; flex-wrap:wrap; }
  button.go {
    flex:1; min-width: 180px;
    padding: 1rem 1.4rem; border:none; border-radius: 12px;
    font-family: inherit; font-size:1rem; font-weight:600; cursor:pointer;
    background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 100%);
    color: white; letter-spacing:.01em;
    box-shadow: 0 14px 30px rgba(122,92,255,.35);
    transition: all .15s ease; position:relative; overflow:hidden;
  }
  button.go:hover { transform: translateY(-2px); box-shadow: 0 18px 40px rgba(122,92,255,.45); }
  button.go:disabled { opacity:.6; cursor:wait; transform:none; }
  .actions-row small { color: var(--ink-2); font-size:.82rem; }

  /* Feature pills */
  .features {
    margin-top: 3rem; display:grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1rem;
  }
  .feat {
    padding: 1.2rem 1.4rem; background: rgba(13,16,28,0.5);
    border:1px solid var(--line); border-radius: 14px;
  }
  .feat .ico { width:32px; height:32px; border-radius:8px;
               display:flex; align-items:center; justify-content:center;
               background: rgba(122,92,255,0.15); margin-bottom:.5rem;
               color: var(--accent-2); }
  .feat h3 { margin:0 0 .25rem; font-size:.95rem; font-weight:600; }
  .feat p { margin:0; color: var(--ink-2); font-size:.85rem; line-height:1.45; }

  footer { text-align:center; color: var(--ink-2); font-size:.82rem;
           padding: 3rem 1.5rem; }
</style></head><body>
<div class="aurora"><div class="blob"></div></div>
<div class="grain"></div>

<header class="top">
  <div class="brand"><span class="dot"></span> Faceswap</div>
  <a href="https://github.com/deepinsight/insightface" target="_blank" rel="noopener">powered by InsightFace</a>
</header>

<main>
  <section class="hero">
    <span class="eyebrow">Live face-swap streaming</span>
    <h1>Your face, in any video.<br>Streamed live to your browser.</h1>
    <p>Drop in a photo of yourself and a video. We auto-detect your gender, lock onto the
    matching person in the footage, and stream the swap with synchronised audio — frame by
    frame, while it processes.</p>
  </section>

  <form class="card" action="/start" method="POST" enctype="multipart/form-data" id="f">
    <div class="drop-row">
      <label class="drop" id="d_source">
        <div class="icon">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 21v-1a8 8 0 0 1 16 0v1"/></svg>
        </div>
        <div class="label">Your face</div>
        <div class="hint" id="h_source">PNG, JPG · 1024 px+ recommended</div>
        <div class="preview" id="p_source" style="display:none"><img alt=""></div>
        <div class="filename" id="n_source"></div>
        <input type="file" name="source" id="source" accept="image/*" required>
      </label>

      <div class="arrow" aria-hidden="true">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
      </div>

      <label class="drop" id="d_target">
        <div class="icon">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
        </div>
        <div class="label">Target video</div>
        <div class="hint" id="h_target">MP4, MOV, WebM · any length</div>
        <div class="preview" id="p_target" style="display:none"><video muted playsinline></video></div>
        <div class="filename" id="n_target"></div>
        <input type="file" name="target" id="target" accept="video/*" required>
      </label>
    </div>

    <div class="actions-row">
      <button type="submit" id="go" class="go">Start live swap</button>
      <small>First run loads models (~30 s). After that, every job is fast.</small>
    </div>
  </form>

  <section class="features">
    <div class="feat">
      <div class="ico">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
      </div>
      <h3>Auto gender + reference lock</h3>
      <p>Detects your face's gender from the source image, scans the video, and locks the swap
         onto the matching person — never the other co-star.</p>
    </div>
    <div class="feat">
      <div class="ico">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
      </div>
      <h3>HLS live streaming with audio</h3>
      <p>Browser plays the swap with the original song's audio while it's still being processed —
         no waiting for the full render to finish.</p>
    </div>
    <div class="feat">
      <div class="ico">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v6"/><path d="M12 22v-6"/><path d="m4.93 4.93 4.24 4.24"/><path d="m14.83 14.83 4.24 4.24"/><path d="M2 12h6"/><path d="M22 12h-6"/><path d="m4.93 19.07 4.24-4.24"/><path d="m14.83 9.17 4.24-4.24"/></svg>
      </div>
      <h3>Embedding-based matching</h3>
      <p>Cosine similarity to a clustered reference embedding — robust to profiles, low light,
         and multiple background extras.</p>
    </div>
  </section>
</main>

<footer>local · GPU-accelerated via CUDA · models cached after first run</footer>

<script>
function setupDrop(zoneId, inputId, previewId, nameId, hintId, isVideo) {
  const zone = document.getElementById(zoneId);
  const input = document.getElementById(inputId);
  const previewWrap = document.getElementById(previewId);
  const previewEl = previewWrap.querySelector(isVideo ? 'video' : 'img');
  const nameEl = document.getElementById(nameId);
  const hintEl = document.getElementById(hintId);
  const orig = hintEl.textContent;

  function show(file) {
    if (!file) return;
    nameEl.textContent = `${file.name} · ${(file.size/1024/1024).toFixed(1)} MB`;
    hintEl.textContent = `Looks good — ready to swap`;
    zone.classList.add('has-file');
    const url = URL.createObjectURL(file);
    previewEl.src = url;
    previewWrap.style.display = '';
    if (isVideo) previewEl.load();
  }
  input.addEventListener('change', e => show(e.target.files[0]));

  ['dragenter','dragover'].forEach(ev =>
    zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add('over'); }));
  ['dragleave','drop'].forEach(ev =>
    zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.remove('over'); }));
  zone.addEventListener('drop', e => {
    const f = e.dataTransfer.files[0]; if (!f) return;
    const dt = new DataTransfer(); dt.items.add(f); input.files = dt.files;
    show(f);
  });
}
setupDrop('d_source', 'source', 'p_source', 'n_source', 'h_source', false);
setupDrop('d_target', 'target', 'p_target', 'n_target', 'h_target', true);

document.getElementById('f').addEventListener('submit', () => {
  const b = document.getElementById('go');
  b.disabled = true; b.textContent = 'Uploading…';
});
</script>
</body></html>
"""

VIEWER_HTML = r"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><title>Faceswap · stream</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/hls.js@1.5.13/dist/hls.min.js"></script>
<style>
  :root {
    color-scheme: dark;
    --bg-0:#05060c; --bg-1:#0c0f1c; --bg-2:#13182a;
    --ink-0:#f6f8fc; --ink-1:#c5cce0; --ink-2:#8c95b0;
    --accent-1:#7a5cff; --accent-2:#3aa1ff; --accent-3:#ff5cb1;
    --good:#52d6a3; --line:rgba(255,255,255,.07);
  }
  *,*::before,*::after { box-sizing: border-box; }
  html, body { margin:0; padding:0; }
  body {
    font-family: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--ink-0); background: var(--bg-0); min-height:100vh;
    -webkit-font-smoothing: antialiased;
  }
  .aurora { position:fixed; inset:0; z-index:-1; overflow:hidden; }
  .aurora::before, .aurora::after {
    content:""; position:absolute; border-radius:50%; filter: blur(100px); opacity:.35;
  }
  .aurora::before { width:600px; height:600px; left:-200px; top:-200px;
    background: radial-gradient(circle, var(--accent-1), transparent 60%); }
  .aurora::after { width:700px; height:700px; right:-250px; bottom:-200px;
    background: radial-gradient(circle, var(--accent-2), transparent 60%); }

  header.top {
    position:sticky; top:0; z-index:10;
    padding:1rem 1.5rem; display:flex; align-items:center; justify-content:space-between;
    border-bottom:1px solid var(--line);
    backdrop-filter: blur(14px); background: rgba(5,6,12,0.65);
  }
  .brand { display:flex; align-items:center; gap:.6rem; font-weight:700; }
  .brand .dot { width:10px; height:10px; border-radius:50%;
                background: linear-gradient(135deg, var(--accent-1), var(--accent-3));
                box-shadow: 0 0 18px var(--accent-1); }
  .top h1 { margin:0; font-size:.95rem; font-weight:500; color: var(--ink-1);
            font-family: "JetBrains Mono", ui-monospace, monospace; }
  .top a { color: var(--ink-1); text-decoration:none; font-size:.9rem; opacity:.8; }
  .top a:hover { opacity:1; color: var(--accent-2); }

  main { max-width:1200px; margin:0 auto; padding: 2rem 1.5rem 4rem;
         display:flex; flex-direction:column; gap:1.2rem; align-items:center; }

  .stage { width:100%; aspect-ratio:16/9; background:#000;
           border-radius:18px; overflow:hidden; position:relative;
           box-shadow: 0 30px 80px rgba(0,0,0,.5);
           border: 1px solid var(--line); }
  .stage video { width:100%; height:100%; object-fit:contain; display:none; background:#000; }
  .stage video.live { display:block; }

  .prep { position:absolute; inset:0; display:flex; flex-direction:column;
          align-items:center; justify-content:center; padding:2rem; text-align:center;
          background: radial-gradient(800px 500px at 50% 30%, rgba(122,92,255,0.1) 0%, transparent 60%); }
  .ring { width:84px; height:84px; margin-bottom:1.2rem; position:relative; }
  .ring::before, .ring::after { content:""; position:absolute; inset:0; border-radius:50%;
    border:3px solid transparent; }
  .ring::before { border-top-color: var(--accent-1);
    animation: spin 1.1s cubic-bezier(.5,.05,.95,.5) infinite; }
  .ring::after { border-top-color: var(--accent-2); inset:10px;
    animation: spin 1.6s cubic-bezier(.5,.05,.95,.5) infinite reverse; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .phase { font-size:1.15rem; font-weight:600; letter-spacing:-.01em; }
  .msg { color: var(--ink-1); font-size:.92rem; margin-top:.4rem; max-width:520px;
         line-height:1.5; }
  .steps { margin-top:1.6rem; display:flex; gap:.4rem; justify-content:center; flex-wrap:wrap; }
  .step { padding:.4rem .8rem; border-radius:999px; background: rgba(255,255,255,.04);
    color: var(--ink-2); font-size:.74rem; border:1px solid transparent;
    font-family: "JetBrains Mono", ui-monospace, monospace; transition: all .2s; }
  .step.done { color: var(--good); border-color: rgba(82,214,163,.3);
               background: rgba(82,214,163,.08); }
  .step.active { color: var(--ink-0); border-color: var(--accent-1);
                 background: rgba(122,92,255,.15);
                 box-shadow: 0 0 24px rgba(122,92,255,.25); }

  .audio-pill { position:absolute; top:1rem; right:1rem;
    background: rgba(0,0,0,.7); border:1px solid rgba(82,214,163,.3);
    padding:.4rem .8rem; border-radius:999px; font-size:.78rem; color: var(--good);
    display:none; align-items:center; gap:.4rem; backdrop-filter: blur(8px); }
  .audio-pill.show { display:flex; }
  .audio-pill .dot { width:6px; height:6px; border-radius:50%; background: var(--good);
    animation: blink 1.4s ease-in-out infinite; }
  @keyframes blink { 50% { opacity: .3; } }

  /* "Click to unmute" — small corner button, NOT a full-page overlay.
     Browsers force us to start muted (autoplay+sound is blocked until the
     user interacts). The video plays normally; this button just toggles audio. */
  .unmute-overlay { position:absolute; bottom:1rem; left:1rem; display:none;
    cursor:pointer; z-index:5; pointer-events:none; }
  .unmute-overlay.show { display:block; }
  .unmute-overlay .btn {
    display:inline-flex; align-items:center; gap:.55rem; padding:.65rem 1.1rem;
    background: rgba(20,26,42,0.92); color: var(--ink-0);
    border-radius: 999px; font-weight:600; font-size:.86rem;
    border: 1px solid rgba(122,92,255,.5);
    box-shadow: 0 14px 40px rgba(0,0,0,.6), 0 0 0 4px rgba(122,92,255,.15);
    transition: transform .12s, box-shadow .12s;
    pointer-events:auto;
    animation: pulse-glow 2.4s ease-in-out infinite; }
  .unmute-overlay:hover .btn { transform: translateY(-2px);
    box-shadow: 0 18px 50px rgba(0,0,0,.7), 0 0 0 6px rgba(122,92,255,.25); }
  .unmute-overlay svg { width:18px; height:18px; }
  @keyframes pulse-glow {
    0%,100% { box-shadow: 0 14px 40px rgba(0,0,0,.6), 0 0 0 4px rgba(122,92,255,.15); }
    50%     { box-shadow: 0 14px 40px rgba(0,0,0,.6), 0 0 0 8px rgba(122,92,255,.35); }
  }

  /* Progress + meta */
  .meta-row { width:100%; display:flex; flex-direction:column; gap:.6rem; }
  .progress { width:100%; height:6px; background: rgba(255,255,255,.06);
              border-radius:3px; overflow:hidden; }
  .progress > div { height:100%; background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    width:0; transition: width .3s; }
  .meta { display:flex; gap:1.6rem; flex-wrap:wrap; color: var(--ink-2); font-size:.86rem;
          font-family: "JetBrains Mono", ui-monospace, monospace; }
  .meta .k { color: var(--ink-2); }
  .meta .v { color: var(--ink-0); font-weight:500; }

  /* Done card with prominent download */
  .done-card { width:100%;
    background: linear-gradient(135deg, rgba(82,214,163,.08), rgba(58,161,255,.08));
    border: 1px solid rgba(82,214,163,.25); border-radius:18px;
    padding: 1.5rem 2rem; display:none; flex-direction:column; gap:1rem;
    box-shadow: 0 20px 50px rgba(0,0,0,.4); }
  .done-card.show { display:flex; }
  .done-card h2 { margin:0; font-size:1.3rem; font-weight:700;
    background: linear-gradient(135deg, var(--good), var(--accent-2));
    -webkit-background-clip: text; background-clip: text; color: transparent; }
  .done-card p { margin:0; color: var(--ink-1); font-size:.92rem; }
  .download-btn { display:inline-flex; align-items:center; gap:.6rem;
    padding:.9rem 1.4rem; border-radius:10px; text-decoration:none; font-weight:600;
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
    color: white; font-size:.95rem; transition: all .15s;
    box-shadow: 0 12px 30px rgba(122,92,255,.35);
    font-family: inherit; align-self:flex-start; }
  .download-btn:hover { transform: translateY(-2px); box-shadow: 0 18px 40px rgba(122,92,255,.5); }
  .download-btn svg { width:18px; height:18px; }

  /* Error */
  .err { padding:1rem 1.4rem; background: rgba(255,90,90,.08); border:1px solid rgba(255,90,90,.3);
    border-radius:12px; color:#ffb4b4; font-size:.9rem; display:none; width:100%; }
  .err.show { display:block; }
</style></head><body>
<div class="aurora"></div>
<header class="top">
  <div class="brand"><span class="dot"></span> Faceswap</div>
  <h1>job · __JOB_ID__</h1>
  <a href="/">&larr; new swap</a>
</header>

<main>
  <div class="stage">
    <video id="player" playsinline controls muted autoplay></video>
    <div class="prep" id="prep">
      <div class="ring"></div>
      <div class="phase" id="phase">Loading…</div>
      <div class="msg" id="msg">Initialising…</div>
      <div class="steps">
        <div class="step" data-k="loading_models">load models</div>
        <div class="step" data-k="detecting_source">detect face</div>
        <div class="step" data-k="finding_reference">find reference</div>
        <div class="step" data-k="streaming">stream</div>
        <div class="step" data-k="finalising">finalise</div>
      </div>
    </div>
    <div class="audio-pill" id="audiopill"><span class="dot"></span>live · with audio</div>
    <div class="unmute-overlay" id="unmute" title="Click to unmute">
      <div class="btn">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>
        Click to unmute
      </div>
    </div>
  </div>

  <div class="meta-row">
    <div class="progress"><div id="bar"></div></div>
    <div class="meta">
      <span><span class="k">progress</span> <span class="v" id="m_progress">0 / 0</span></span>
      <span><span class="k">fps</span> <span class="v" id="m_fps">–</span></span>
      <span><span class="k">swaps</span> <span class="v" id="m_swap">0</span></span>
      <span id="m_extra"></span>
    </div>
  </div>

  <div class="done-card" id="done">
    <h2>Your swap is ready</h2>
    <p id="done_msg">Audio is included. The video above is the final result — controls let you scrub, replay, and full-screen.</p>
    <a class="download-btn" id="dl" href="#" download>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
      Download MP4 (with audio)
    </a>
  </div>

  <div class="err" id="err"></div>
</main>

<script>
const JOB = "__JOB_ID__";
const PHASE_ORDER = ["loading_models","detecting_source","finding_reference","streaming","finalising"];
const player = document.getElementById('player');
const prep = document.getElementById('prep');
const phaseEl = document.getElementById('phase');
const msgEl = document.getElementById('msg');
const bar = document.getElementById('bar');
const audioPill = document.getElementById('audiopill');
const doneCard = document.getElementById('done');
const errEl = document.getElementById('err');

function pill(state, k) {
  document.querySelectorAll(`.step[data-k="${k}"]`).forEach(el => {
    el.classList.remove("active","done");
    if (state) el.classList.add(state);
  });
}

const PHASE_LABELS = {
  queued: "Queued",
  loading_models: "Loading models",
  detecting_source: "Detecting your face",
  finding_reference: "Finding target person",
  streaming: "Streaming live",
  finalising: "Finalising",
  done: "Done",
  error: "Error",
};

let hls = null;
let streamShown = false;
let playStarted = false;
const PREBUFFER_TARGET = 15;   // seconds we want buffered ahead before pressing play
const REBUFFER_TARGET = 8;     // when we stall, wait for this many seconds before resuming
const unmuteOverlay = document.getElementById('unmute');

function bufferedAhead() {
  if (player.buffered.length === 0) return 0;
  return player.buffered.end(player.buffered.length - 1) - player.currentTime;
}

function setPhaseMsg(text) { msgEl.textContent = text; }

function tryStartPlayback() {
  if (playStarted) return;
  const ahead = bufferedAhead();
  setPhaseMsg(`Buffering ${ahead.toFixed(1)} / ${PREBUFFER_TARGET}s before starting…`);
  if (ahead < PREBUFFER_TARGET) return;
  // Browsers block autoplay-with-audio. Always start muted so play() succeeds.
  player.muted = true;
  player.play().then(() => {
    // Only flip the flag *after* play() actually succeeds — otherwise a rejected
    // promise (autoplay policy) would leave us stuck in "started" with paused video.
    playStarted = true;
    unmuteOverlay.classList.add('show');
    audioPill.classList.add('show');
    prep.style.display = 'none';
  }).catch(err => {
    console.warn('play rejected, retrying in 1s:', err && err.name);
    // Retry — once user has interacted with the page (any click anywhere counts)
    // the autoplay policy lifts and the next attempt will succeed.
    setTimeout(tryStartPlayback, 1000);
  });
}

// Any click on the stage counts as a user gesture for the autoplay policy.
// This is the universal "rescue" path: if Chrome refuses to autoplay,
// the user clicking anywhere on the player area will start it.
document.addEventListener('click', () => {
  if (!playStarted && bufferedAhead() >= 1) {
    player.muted = true;
    player.play().then(() => {
      playStarted = true;
      unmuteOverlay.classList.add('show');
      audioPill.classList.add('show');
      prep.style.display = 'none';
    }).catch(()=>{});
  }
}, { once: false });

function attachStream() {
  if (streamShown) return;
  const url = `/job/${JOB}/hls/playlist.m3u8`;
  if (window.Hls && Hls.isSupported()) {
    hls = new Hls({
      // Bigger buffer because the swap pipeline produces frames slower than
      // realtime — we want to soak up several seconds of slack.
      liveSyncDuration: PREBUFFER_TARGET,
      liveMaxLatencyDuration: 60,
      maxBufferLength: 60,
      maxMaxBufferLength: 120,
      backBufferLength: 90,
      lowLatencyMode: false,
      manifestLoadingMaxRetry: 60,
      manifestLoadingRetryDelay: 800,
      levelLoadingMaxRetry: 60,
      levelLoadingRetryDelay: 800,
      fragLoadingMaxRetry: 60,
      fragLoadingRetryDelay: 800,
    });
    hls.loadSource(url);
    hls.attachMedia(player);
    // Don't auto-play on MANIFEST_PARSED — wait for buffer to fill instead.
    hls.on(Hls.Events.BUFFER_APPENDED, tryStartPlayback);
    hls.on(Hls.Events.ERROR, (_, data) => {
      if (data.fatal) console.warn('hls fatal', data);
    });
  } else if (player.canPlayType('application/vnd.apple.mpegurl')) {
    // Safari native HLS — same buffer-then-play idea via timeupdate
    player.src = url;
    player.addEventListener('progress', tryStartPlayback);
  } else {
    errEl.classList.add('show');
    errEl.textContent = "Your browser doesn't support HLS. Try Chrome, Firefox, or Safari.";
    return;
  }

  // Stall handling: when the buffer drains (backend can't keep up), pause and
  // wait for a re-buffer instead of letting the player stutter every second.
  player.addEventListener('waiting', () => {
    if (playStarted) setPhaseMsg(`Buffering… (${bufferedAhead().toFixed(1)}s ahead)`);
    prep.style.display = 'flex';
  });
  player.addEventListener('playing', () => {
    prep.style.display = 'none';
  });
  // After a stall, only resume once we have REBUFFER_TARGET seconds again.
  let resumeTimer = null;
  player.addEventListener('waiting', () => {
    if (resumeTimer) clearInterval(resumeTimer);
    resumeTimer = setInterval(() => {
      if (bufferedAhead() >= REBUFFER_TARGET) {
        clearInterval(resumeTimer); resumeTimer = null;
        player.play().catch(()=>{});
      } else {
        setPhaseMsg(`Re-buffering ${bufferedAhead().toFixed(1)} / ${REBUFFER_TARGET}s…`);
      }
    }, 500);
  });

  // Unmute overlay click → enable audio.
  unmuteOverlay.addEventListener('click', () => {
    player.muted = false;
    player.volume = 1;
    unmuteOverlay.classList.remove('show');
  });

  player.classList.add('live');
  streamShown = true;
  setPhaseMsg(`Buffering 0 / ${PREBUFFER_TARGET}s before starting…`);
}

async function poll() {
  let r;
  try { r = await fetch(`/job/${JOB}/status`).then(r => r.json()); }
  catch(e) { setTimeout(poll, 1000); return; }

  const idx = PHASE_ORDER.indexOf(r.phase);
  PHASE_ORDER.forEach((k, i) => {
    if (i < idx) pill('done', k);
    else if (i === idx) pill('active', k);
    else pill('', k);
  });

  phaseEl.textContent = PHASE_LABELS[r.phase] || r.phase;
  msgEl.textContent = r.message;

  if (r.detected_gender) {
    document.getElementById('m_extra').innerHTML =
      `<span class="k">source</span> <span class="v">${r.detected_gender}/${r.detected_age}</span>` +
      (r.ref_frame >= 0 ? ` &nbsp; <span class="k">ref</span> <span class="v">f${r.ref_frame} (${r.ref_votes}/${r.ref_pool})</span>` : '');
  }

  if (r.total_frames > 0) {
    document.getElementById('m_progress').textContent = `${r.current_frame} / ${r.total_frames}`;
    bar.style.width = `${100 * r.current_frame / r.total_frames}%`;
  }
  if (r.proc_fps > 0) document.getElementById('m_fps').textContent = r.proc_fps.toFixed(1);
  document.getElementById('m_swap').textContent = r.swap_count;

  if ((r.phase === "streaming" || r.phase === "finalising" || r.phase === "done") && !streamShown) {
    attachStream();
  }

  if (r.phase === "done") {
    audioPill.classList.remove('show');
    unmuteOverlay.classList.remove('show');
    doneCard.classList.add('show');
    document.getElementById('dl').href = `/job/${JOB}/download`;
    // Stream is finished — let the existing HLS playback continue (it now has
    // the full playlist with #EXT-X-ENDLIST and acts as VOD with full scrub).
    return;
  }
  if (r.phase === "error") {
    prep.style.display = 'flex';
    errEl.classList.add('show');
    errEl.textContent = "Job failed: " + r.message;
    return;
  }
  setTimeout(poll, 400);
}
poll();
</script>
</body></html>
"""


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/start", methods=["POST"])
def start():
    src = request.files.get("source")
    tgt = request.files.get("target")
    if not src or not tgt:
        return "missing source or target", 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    src_ext = os.path.splitext(src.filename or "src.jpg")[1].lower() or ".jpg"
    tgt_ext = os.path.splitext(tgt.filename or "tgt.mp4")[1].lower() or ".mp4"
    src_path = os.path.join(job_dir, "source" + src_ext)
    tgt_path = os.path.join(job_dir, "target" + tgt_ext)
    src.save(src_path)
    tgt.save(tgt_path)

    job = Job(
        id=job_id,
        source_path=src_path,
        target_path=tgt_path,
        out_audio_path=os.path.join(job_dir, "swapped.mp4"),
        hls_dir=os.path.join(job_dir, "hls"),
    )
    with JOBS_LOCK:
        JOBS[job_id] = job
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return redirect(url_for("viewer", job_id=job_id))


@app.route("/job/<job_id>")
def viewer(job_id: str):
    if job_id not in JOBS:
        return "no such job", 404
    return Response(VIEWER_HTML.replace("__JOB_ID__", job_id), mimetype="text/html")


@app.route("/job/<job_id>/status")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"phase": "error", "message": "no such job"}), 404
    return jsonify({
        "phase": job.phase,
        "message": job.message,
        "detected_gender": job.detected_gender,
        "detected_age": job.detected_age,
        "ref_frame": job.ref_frame,
        "ref_votes": job.ref_votes,
        "ref_pool": job.ref_pool,
        "current_frame": job.current_frame,
        "total_frames": job.total_frames,
        "swap_count": job.swap_count,
        "proc_fps": job.proc_fps,
        "error": job.error,
    })


@app.route("/job/<job_id>/hls/<path:fname>")
def hls_file(job_id: str, fname: str):
    job = JOBS.get(job_id)
    if not job:
        return "no such job", 404
    if not os.path.isdir(job.hls_dir):
        return "stream not started", 404
    # Whitelist filenames so users can't escape the hls dir.
    if "/" in fname or "\\" in fname or ".." in fname:
        return "bad", 400
    if not (fname.endswith(".m3u8") or fname.endswith(".ts")):
        return "bad", 400
    path = os.path.join(job.hls_dir, fname)
    if not os.path.isfile(path):
        return "not yet", 404
    mt = "application/vnd.apple.mpegurl" if fname.endswith(".m3u8") else "video/mp2t"
    resp = send_from_directory(job.hls_dir, fname, mimetype=mt)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/job/<job_id>/download")
def download(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return "no such job", 404
    if not os.path.isfile(job.out_audio_path):
        return "not ready yet", 404
    nice = (os.path.splitext(os.path.basename(job.target_path))[0] or "swap") + "_swapped.mp4"
    return send_from_directory(os.path.dirname(job.out_audio_path),
                               os.path.basename(job.out_audio_path),
                               as_attachment=True, download_name=nice)


@app.route("/job/<job_id>/file")
def file_inline(job_id: str):
    """Inline streaming for the final-MP4 fallback player (Range-aware)."""
    job = JOBS.get(job_id)
    if not job:
        return "no such job", 404
    if not os.path.isfile(job.out_audio_path):
        return "not ready yet", 404
    return send_from_directory(os.path.dirname(job.out_audio_path),
                               os.path.basename(job.out_audio_path),
                               as_attachment=False, mimetype="video/mp4")


if __name__ == "__main__":
    _cleanup_old_jobs()
    # Pre-warm models in a background thread so the first job is faster.
    threading.Thread(target=_ensure_models, daemon=True).start()
    print("[webapp] starting on http://localhost:8080/  (jobs at " + JOBS_DIR + ")", flush=True)
    # Threaded server so the long-poll MJPEG stream doesn't block other requests.
    app.run(host="0.0.0.0", port=8080, threaded=True, debug=False, use_reloader=False)
