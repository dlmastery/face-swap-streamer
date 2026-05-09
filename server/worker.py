"""ML pipeline — face detection, embedding match, swap, HLS encode.

Lifted from the original webapp.py with these changes for production:
  - models loaded behind a lock, used as global singletons
  - per-job state isolated in `Job` dataclass (no shared mutable state)
  - status changes flow through an `on_change` callback so the HTTP layer
    can broadcast updates without the worker knowing about HTTP
  - 4-stage thread pipeline: reader → detector → main(swapper) → writer
  - HLS streaming via ffmpeg subprocess; finalised MP4 produced by a remux
    pass (`-c copy -movflags +faststart`) so the download plays everywhere
  - batch orchestration: one BatchJob spawns N child jobs that share
    detected source faces but each get their own reference extraction
    against their own video
"""
from __future__ import annotations

import os
import sys
import time
import uuid
import queue
import shutil
import threading
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---- Win Py 3.8+ secure DLL search: register CUDA dirs before importing
#      onnxruntime. PATH alone is NOT enough — must call os.add_dll_directory
#      and KEEP THE COOKIES ALIVE in a long-lived list.
_dll_cookies: list = []
if sys.platform == "win32":
    _sp = os.path.join(sys.prefix, "Lib", "site-packages")
    for _sub in ("cudnn", "cublas", "cuda_runtime", "curand", "cufft",
                 "cuda_nvrtc", "nvjitlink"):
        _bin = os.path.join(_sp, "nvidia", _sub, "bin")
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


# ---- Configuration -------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWAPPER_PATH = os.path.join(ROOT, "deep-live-cam", "models", "inswapper_128_fp16.onnx")
JOBS_DIR = os.path.join(ROOT, "server", "jobs")  # separate from old webapp_jobs/

FFMPEG_EXE = next((p for p in [
    r"C:\Users\evija\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe",
    "ffmpeg",
] if p == "ffmpeg" or os.path.isfile(p)), None)

os.makedirs(JOBS_DIR, exist_ok=True)


# ---- Domain models -------------------------------------------------------

@dataclass
class SourceSpec:
    """One uploaded source image, its detected face + per-video reference."""
    path: str
    gender: str = ""
    age: int = 0
    src_face: object = None         # insightface Face — kept alive for swap
    ref_emb: object = None          # numpy embedding of matching cluster
    ref_frame: int = -1
    ref_votes: int = 0
    ref_pool: int = 0


@dataclass
class Job:
    """One target video being swapped. Fully self-contained — no shared
    mutable state with other jobs."""
    id: str
    target_path: str
    target_filename: str
    job_dir: str
    out_audio_path: str             # final audio-muxed MP4
    hls_dir: str                    # playlist.m3u8 + seg_*.ts

    # Sources — copy-by-reference is OK because SourceSpec.src_face is set
    # once during detect_sources() and not mutated afterwards. ref_emb /
    # ref_frame / ref_votes / ref_pool ARE per-job and live on a per-job
    # copy of the SourceSpec.
    sources: list[SourceSpec] = field(default_factory=list)

    phase: str = "queued"
    message: str = "Queued"
    error: str = ""

    # Top-level mirror of sources[0] for legacy clients
    detected_gender: str = ""
    detected_age: int = 0
    ref_frame: int = -1
    ref_votes: int = 0
    ref_pool: int = 0

    # Video info + progress
    width: int = 0
    height: int = 0
    fps: float = 0.0
    total_frames: int = 0
    current_frame: int = 0
    swap_count: int = 0
    proc_fps: float = 0.0

    started: float = field(default_factory=time.time)
    finished: float = 0.0

    stop_flag: threading.Event = field(default_factory=threading.Event)
    on_change: Optional[Callable[["Job"], None]] = None


@dataclass
class BatchJob:
    """A multi-video upload: same source faces, N videos, sequential GPU
    processing. Children are full Jobs that the orchestrator runs in order."""
    id: str
    batch_dir: str
    sources: list[SourceSpec] = field(default_factory=list)
    jobs: list[Job] = field(default_factory=list)

    phase: str = "queued"
    message: str = ""
    started: float = field(default_factory=time.time)
    finished: float = 0.0
    on_change: Optional[Callable[["BatchJob"], None]] = None


# ---- Model loading -------------------------------------------------------

_models_lock = threading.Lock()
_face_analyser: Optional[FaceAnalysis] = None
_swapper = None


def ensure_models() -> None:
    """Lazy-load the face analyser + inswapper into VRAM. Called once on
    server start (background thread) and again from each job (no-op if
    already loaded). CUDA only — TensorRT is intentionally not used here.
    """
    global _face_analyser, _swapper
    with _models_lock:
        if _face_analyser is None:
            face_model = os.getenv("FACESWAP_FACE_MODEL", "buffalo_l")
            det_size = int(os.getenv("FACESWAP_DET_SIZE", "640"))
            det_thresh = float(os.getenv("FACESWAP_DET_THRESH", "0.3"))
            print(f"[worker] loading face analyser (CUDA, model={face_model}, "
                  f"det_size={det_size}, det_thresh={det_thresh})...", flush=True)
            fa = FaceAnalysis(name=face_model,
                              providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            fa.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=det_thresh)
            _face_analyser = fa

        if _swapper is None:
            print("[worker] loading inswapper (CUDA only)...", flush=True)
            _swapper = insightface.model_zoo.get_model(
                SWAPPER_PATH,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            try:
                active = _swapper.session.get_providers()
            except AttributeError:
                active = None
            print(f"[worker] inswapper active providers: {active}", flush=True)
            if active and active == ["CPUExecutionProvider"]:
                raise RuntimeError(
                    "inswapper loaded on CPU only — CUDA failed to initialise. "
                    "See CLAUDE.md issue #1 (cuDNN DLL discovery)."
                )


def get_face_analyser() -> FaceAnalysis:
    if _face_analyser is None:
        ensure_models()
    assert _face_analyser is not None
    return _face_analyser


def get_swapper():
    if _swapper is None:
        ensure_models()
    return _swapper


# ---- Helpers -------------------------------------------------------------

def _set(job: Job, **kw) -> None:
    """Mutate a job and fire the on_change callback. Centralised so the
    WebSocket layer reliably sees every state transition."""
    for k, v in kw.items():
        setattr(job, k, v)
    if job.on_change is not None:
        try:
            job.on_change(job)
        except Exception as e:
            # Don't let a flaky observer crash the worker.
            print(f"[worker] on_change observer raised: {e}", flush=True)


def _set_batch(batch: BatchJob, **kw) -> None:
    for k, v in kw.items():
        setattr(batch, k, v)
    if batch.on_change is not None:
        try:
            batch.on_change(batch)
        except Exception as e:
            print(f"[worker] batch on_change raised: {e}", flush=True)


# ---- Source detection (shared by single + batch) -------------------------

def detect_source_faces(source_paths: list[str]) -> list[SourceSpec]:
    """Read each source image, detect its primary face. Returns a list of
    SourceSpec ready to be referenced (without ref_emb, which is filled in
    per video by extract_reference_embeddings)."""
    fa = get_face_analyser()
    specs: list[SourceSpec] = []
    for path in source_paths:
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"could not read source image: {os.path.basename(path)}")
        faces = fa.get(img)
        if not faces:
            raise RuntimeError(
                f"no face detected in {os.path.basename(path)} — try a clearer, "
                "front-facing photo"
            )
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        specs.append(SourceSpec(
            path=path,
            gender=face.sex,
            age=int(face.age),
            src_face=face,
        ))
    return specs


# ---- ffmpeg --------------------------------------------------------------

def _spawn_ffmpeg(job: Job, w: int, h: int, fps: float) -> subprocess.Popen:
    """Spawn ffmpeg that takes raw BGR on stdin + audio from target.mp4 and
    writes only HLS (.m3u8 + .ts). The downloadable MP4 is built by a
    second pass (_remux_to_mp4) at end of stream — this gives us a real,
    non-fragmented +faststart MP4 that plays on every native player.

    Run with cwd=job_dir + relative paths in the HLS args because Windows
    drive-letter colons collide with hls option separators if absolute
    paths are used (silent failure)."""
    if not FFMPEG_EXE:
        raise RuntimeError("ffmpeg not found — install Gyan.FFmpeg")
    os.makedirs(job.hls_dir, exist_ok=True)
    job_dir = os.path.dirname(job.out_audio_path)
    target_abs = os.path.abspath(job.target_path)
    cmd = [
        FFMPEG_EXE, "-y", "-hide_banner", "-loglevel", "info",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{w}x{h}", "-framerate", str(fps),
        "-i", "pipe:0",
        "-i", target_abs,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
        "-g", str(int(round(fps * 2))),
        "-keyint_min", str(int(round(fps * 2))),
        "-sc_threshold", "0",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "44100",
        "-shortest",
        "-f", "hls",
        "-hls_time", "2",
        "-hls_list_size", "0",
        "-hls_flags", "independent_segments+append_list",
        "-hls_segment_filename", "hls/seg_%05d.ts",
        "hls/playlist.m3u8",
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            cwd=job_dir, bufsize=0)
    log_path = os.path.join(job_dir, "ffmpeg.log")

    def _drain():
        with open(log_path, "wb") as f:
            for line in iter(proc.stderr.readline, b""):
                f.write(line)
                f.flush()
    threading.Thread(target=_drain, daemon=True, name=f"ffmpeg-log-{job.id}").start()
    return proc


def _remux_to_mp4(job: Job) -> None:
    """Concat the finalised HLS .ts segments into a standard MP4 with the
    moov atom moved to the front (+faststart). Plays on iOS Safari, Android,
    QuickTime, VLC, Windows MP."""
    if not FFMPEG_EXE:
        raise RuntimeError("ffmpeg not found")
    playlist = os.path.join(job.hls_dir, "playlist.m3u8")
    if not os.path.isfile(playlist):
        raise RuntimeError(f"HLS playlist missing: {playlist}")
    cmd = [
        FFMPEG_EXE, "-y", "-hide_banner", "-loglevel", "warning",
        "-allowed_extensions", "ALL",
        "-i", playlist,
        "-c", "copy",
        "-bsf:a", "aac_adtstoasc",
        "-movflags", "+faststart",
        job.out_audio_path,
    ]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"HLS -> MP4 remux failed (rc={rc})")


# ---- Reference extraction (per video) ------------------------------------

def extract_reference_embeddings(job: Job) -> None:
    """For each of job.sources, find a recurring matching-gender face cluster
    in the target video and fill in spec.ref_emb / ref_frame / ref_votes /
    ref_pool. Single video scan, multi-source aware."""
    fa = get_face_analyser()

    cap = cv2.VideoCapture(job.target_path)
    try:
        if not cap.isOpened():
            raise RuntimeError("could not open target video")
        in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _set(job, width=in_w, height=in_h, fps=float(fps), total_frames=total)

        genders_needed = set(s.gender for s in job.sources)
        _set(job, phase="finding_reference",
             message=f"Scanning for {' + '.join(sorted(genders_needed))} face"
                     f"{'s' if len(genders_needed) > 1 else ''} to swap onto…")

        step = max(1, int(fps * 2.0))
        min_ref_face_w = int(os.getenv("FACESWAP_MIN_REF_FACE_W", "25"))
        max_samples = 120
        all_candidates: list = []   # (score, embedding, frame_idx, gender)
        i = 0
        while i < total and len(all_candidates) < max_samples:
            if job.stop_flag.is_set():
                raise RuntimeError("cancelled")
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok:
                break
            for face in fa.get(fr):
                if face.sex not in genders_needed:
                    continue
                w_face = face.bbox[2] - face.bbox[0]
                if w_face < min_ref_face_w:
                    continue
                all_candidates.append((float(w_face * face.det_score),
                                       face.normed_embedding, i, face.sex))
            i += step
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Per-gender greedy cluster assignment so two same-gender sources
        # can't land on the same person.
        used_idxs: set[int] = set()
        for gender in genders_needed:
            same_gender = [(idx, c) for idx, c in enumerate(all_candidates) if c[3] == gender]
            sources_this_gender = [s for s in job.sources if s.gender == gender]
            if not same_gender:
                raise RuntimeError(f"no {gender} face found in the video")
            embs = np.stack([c[1] for _, c in same_gender])
            sim = embs @ embs.T
            scores = np.array([c[0] for _, c in same_gender])
            for spec in sources_this_gender:
                mask = np.array([1.0 if i_ not in used_idxs else 0.0
                                 for i_, _ in same_gender])
                if mask.sum() == 0:
                    mask = np.ones(len(same_gender))
                votes = (sim > 0.30).sum(axis=1) * scores * mask
                local_winner = int(np.argmax(votes))
                global_idx, cand = same_gender[local_winner]
                spec.ref_emb = cand[1]
                spec.ref_frame = int(cand[2])
                spec.ref_votes = int((sim[local_winner] > 0.30).sum())
                spec.ref_pool = len(same_gender)
                similar = sim[local_winner] > 0.30
                for j, (gidx, _) in enumerate(same_gender):
                    if similar[j]:
                        used_idxs.add(gidx)

        if job.sources:
            primary = job.sources[0]
            _set(job, ref_frame=primary.ref_frame,
                 ref_votes=primary.ref_votes, ref_pool=primary.ref_pool,
                 detected_gender=primary.gender, detected_age=primary.age)
    finally:
        cap.release()


# ---- The streaming pipeline ---------------------------------------------

def run_streaming(job: Job) -> None:
    """The 4-stage thread pipeline: reader → detect → main(swap) → writer.
    Assumes job.sources[*].ref_emb is set (call extract_reference_embeddings
    first) and job.width/height/fps/total_frames populated."""
    fa = get_face_analyser()
    sw = get_swapper()

    cap = cv2.VideoCapture(job.target_path)
    if not cap.isOpened():
        raise RuntimeError("could not open target video")

    ffmpeg = _spawn_ffmpeg(job, job.width, job.height, job.fps)

    msg_genders = ", ".join(f"{s.gender}@frame{s.ref_frame}" for s in job.sources)
    _set(job, phase="streaming",
         message=f"Streaming swap ({len(job.sources)} source"
                 f"{'s' if len(job.sources) > 1 else ''}: {msg_genders}) — audio is included")

    REFERENCE_THRESH = float(os.getenv("FACESWAP_REF_THRESH", "0.18"))
    ref_embs = np.stack([s.ref_emb for s in job.sources])
    ref_sources = list(job.sources)

    Q_DEPTH = int(os.getenv("FACESWAP_Q_DEPTH", "128"))
    END = object()
    read_q: queue.Queue = queue.Queue(maxsize=Q_DEPTH)
    detect_q: queue.Queue = queue.Queue(maxsize=Q_DEPTH)
    write_q: queue.Queue = queue.Queue(maxsize=Q_DEPTH)
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

    def _detect_loop():
        try:
            while True:
                item = read_q.get()
                if item is END:
                    return
                frame = item
                tgt_faces = fa.get(frame)
                picks = []
                if tgt_faces:
                    tgt_embs = np.stack([f.normed_embedding for f in tgt_faces])
                    sims = tgt_embs @ ref_embs.T
                    for ti, tface in enumerate(tgt_faces):
                        si = int(np.argmax(sims[ti]))
                        if float(sims[ti, si]) >= REFERENCE_THRESH:
                            picks.append((tface, si))
                detect_q.put((frame, picks))
        finally:
            detect_q.put(END)

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

    t_reader = threading.Thread(target=_reader_loop, daemon=True, name=f"job-{job.id}-reader")
    t_detect = threading.Thread(target=_detect_loop, daemon=True, name=f"job-{job.id}-detect")
    t_writer = threading.Thread(target=_writer_loop, daemon=True, name=f"job-{job.id}-writer")
    t_reader.start()
    t_detect.start()
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
            item = detect_q.get()
            if item is END:
                break
            frame, picks = item
            n += 1
            for tface, si in picks:
                frame = sw.get(frame, tface, ref_sources[si].src_face, paste_back=True)
                swap_count += 1
            if broken:
                break
            write_q.put(frame.tobytes())

            now = time.time()
            if now - last_log > 0.5:
                elapsed = now - t0
                _set(job,
                     current_frame=n,
                     swap_count=swap_count,
                     proc_fps=n / elapsed if elapsed else 0.0)
                last_log = now
    finally:
        job.stop_flag.set()
        for q_ in (read_q, detect_q):
            try:
                while True:
                    q_.get_nowait()
            except queue.Empty:
                pass
        write_q.put(END)
        t_writer.join(timeout=30)
        t_detect.join(timeout=10)
        t_reader.join(timeout=10)
        cap.release()
        try:
            ffmpeg.stdin.close()
        except Exception:
            pass

    _set(job, phase="finalising", message="Finalising HLS playlist…")
    try:
        ffmpeg.wait(timeout=60)
    except subprocess.TimeoutExpired:
        ffmpeg.kill()
        raise RuntimeError("ffmpeg did not exit in time")
    if ffmpeg.returncode not in (0, None) and not broken:
        log_path = os.path.join(os.path.dirname(job.out_audio_path), "ffmpeg.log")
        err = ""
        try:
            with open(log_path, "rb") as f:
                err = f.read().decode(errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"ffmpeg failed (rc={ffmpeg.returncode}): {err[-800:]}")

    _set(job, phase="finalising",
         message="Building downloadable MP4 (+faststart)…")
    try:
        _remux_to_mp4(job)
    except Exception as e:
        print(f"[worker] remux warning: {e}", flush=True)


# ---- Public entry points -------------------------------------------------

def run_single_job(job: Job, sources: list[SourceSpec]) -> None:
    """Process one Job end-to-end. `sources` already has src_face/gender/age
    detected (we accept them pre-detected so a batch can detect once and
    reuse). Each call gets its own copy with per-job ref_emb / ref_frame."""
    try:
        _set(job, phase="loading_models", message="Loading face-swap models…")
        ensure_models()

        # Per-job copy of each SourceSpec (so we don't write per-video
        # ref_emb back onto the shared batch sources).
        job.sources = [
            SourceSpec(
                path=s.path,
                gender=s.gender,
                age=s.age,
                src_face=s.src_face,
            )
            for s in sources
        ]

        extract_reference_embeddings(job)
        run_streaming(job)

        _set(job, phase="done",
             message="Done — audio + video saved",
             finished=time.time())
    except Exception as e:
        _set(job,
             phase="error",
             message=str(e),
             error=str(e),
             finished=time.time())
        print(f"[worker] job {job.id} error: {e}", flush=True)


def run_batch(batch: BatchJob) -> None:
    """Detect source faces ONCE, then run each child job sequentially. The
    GPU is single-tenant for this app so sequential keeps things simple."""
    try:
        _set_batch(batch, phase="processing", message="Loading models + detecting source faces…")
        ensure_models()

        # Detect each source's face once, share across child jobs
        source_paths = [s.path for s in batch.sources]
        detected = detect_source_faces(source_paths)
        for shared, src_spec in zip(batch.sources, detected):
            shared.gender = src_spec.gender
            shared.age = src_spec.age
            shared.src_face = src_spec.src_face

        for i, job in enumerate(batch.jobs):
            _set_batch(batch, message=f"Processing video {i + 1} / {len(batch.jobs)}: "
                                      f"{job.target_filename}")
            run_single_job(job, batch.sources)

        any_error = any(j.phase == "error" for j in batch.jobs)
        _set_batch(batch,
                   phase="error" if any_error else "done",
                   message="Some videos failed" if any_error else f"All {len(batch.jobs)} videos done",
                   finished=time.time())
    except Exception as e:
        _set_batch(batch, phase="error", message=str(e), finished=time.time())
        print(f"[worker] batch {batch.id} error: {e}", flush=True)
