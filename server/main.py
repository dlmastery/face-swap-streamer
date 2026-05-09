"""FastAPI application — production-grade replacement for the Flask webapp.

Layered architecture:
  - this file owns HTTP/WebSocket concerns only
  - server/worker.py owns the ML pipeline (detection, swap, ffmpeg)
  - server/schemas.py owns the API contract (Pydantic types -> OpenAPI)

Run with:
    conda run -n dlc uvicorn server.main:app --host 0.0.0.0 --port 8081
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import (
    FastAPI, File, Form, HTTPException, Path as FPath, Request,
    UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import worker
from .schemas import (
    BatchCreatedResponse, BatchJobSummary, BatchStatus,
    HealthResponse, JobCreatedResponse, JobStatus, SourceInfo,
)


# ---- Registry: in-process state of currently-known jobs and batches ------

_jobs: dict[str, worker.Job] = {}
_batches: dict[str, worker.BatchJob] = {}
_registry_lock = threading.Lock()

# WebSocket fan-out state. Each entry is a list of (loop, queue) tuples —
# the worker thread pushes JSON onto the queue from a non-async context, the
# WS coroutine drains and sends. asyncio.Queue is thread-safe via call_soon_threadsafe.
_job_ws: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = {}
_batch_ws: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = {}
_ws_lock = threading.Lock()


# ---- Lifespan ------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm GPU models in a background thread so the first job is fast.
    threading.Thread(target=worker.ensure_models, daemon=True, name="model-prewarm").start()
    yield


app = FastAPI(
    title="face-swap-streamer",
    version="0.8.0",
    description="Live face-swap web service with batch + multi-source support.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # In dev, the Vite/Next dev server proxies; in prod we serve frontend from
    # the same origin. This list covers both.
    allow_origins=["http://localhost:3000", "http://localhost:5173",
                   "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload size cap. 8 GB total per request — comfortably accommodates 1 GB
# videos × N for batch uploads plus headroom. Set MAX_UPLOAD_GB env var to
# tune (e.g. 16 for 16 GB).
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_GB", "8")) * 1024 * 1024 * 1024


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.method == "POST":
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_UPLOAD:
            return JSONResponse(
                {"error": "upload too large",
                 "max_bytes": MAX_UPLOAD,
                 "max_human": f"{MAX_UPLOAD // (1024 ** 3)} GB"},
                status_code=413,
            )
    return await call_next(request)


# ---- Helpers -------------------------------------------------------------

def _job_to_status(job: worker.Job) -> JobStatus:
    return JobStatus(
        id=job.id,
        phase=job.phase,             # type: ignore[arg-type]
        message=job.message,
        error=job.error,
        sources=[
            SourceInfo(
                gender=s.gender,     # type: ignore[arg-type]
                age=s.age,
                ref_frame=s.ref_frame,
                ref_votes=s.ref_votes,
                ref_pool=s.ref_pool,
            )
            for s in job.sources
        ],
        detected_gender=job.detected_gender,  # type: ignore[arg-type]
        detected_age=job.detected_age,
        ref_frame=job.ref_frame,
        ref_votes=job.ref_votes,
        ref_pool=job.ref_pool,
        width=job.width,
        height=job.height,
        fps=job.fps,
        total_frames=job.total_frames,
        current_frame=job.current_frame,
        swap_count=job.swap_count,
        proc_fps=job.proc_fps,
        started=job.started,
        finished=job.finished,
    )


def _batch_to_status(batch: worker.BatchJob) -> BatchStatus:
    summaries: list[BatchJobSummary] = []
    done = 0
    err = 0
    for j in batch.jobs:
        progress = (100.0 * j.current_frame / j.total_frames) if j.total_frames else 0.0
        summaries.append(BatchJobSummary(
            id=j.id,
            target_filename=j.target_filename,
            phase=j.phase,           # type: ignore[arg-type]
            progress_pct=progress,
            proc_fps=j.proc_fps,
            error=j.error,
        ))
        if j.phase == "done":
            done += 1
        elif j.phase == "error":
            err += 1
    return BatchStatus(
        id=batch.id,
        phase=batch.phase,           # type: ignore[arg-type]
        message=batch.message,
        sources=[
            SourceInfo(
                gender=s.gender,     # type: ignore[arg-type]
                age=s.age,
                ref_frame=-1, ref_votes=0, ref_pool=0,
            )
            for s in batch.sources
        ],
        jobs=summaries,
        total_videos=len(batch.jobs),
        done_videos=done,
        error_videos=err,
        started=batch.started,
        finished=batch.finished,
    )


def _push_to_ws(slot_dict, registry_id: str, payload: dict) -> None:
    """Worker-thread → WebSocket bridge. Drops entries whose loop has died."""
    text = json.dumps(payload)
    with _ws_lock:
        targets = list(slot_dict.get(registry_id, ()))
    for loop, q in targets:
        try:
            loop.call_soon_threadsafe(q.put_nowait, text)
        except Exception:
            pass


def _make_job_on_change(job_id: str):
    def cb(job: worker.Job):
        _push_to_ws(_job_ws, job_id, _job_to_status(job).model_dump())
    return cb


def _make_batch_on_change(batch_id: str):
    def cb(batch: worker.BatchJob):
        _push_to_ws(_batch_ws, batch_id, _batch_to_status(batch).model_dump())
    return cb


def _save_uploads(upload_dir: str, files: list[UploadFile], prefix: str) -> list[tuple[str, str]]:
    """Save each UploadFile to upload_dir/{prefix}_{i}{ext}. Returns list of
    (saved_path, original_filename)."""
    os.makedirs(upload_dir, exist_ok=True)
    saved: list[tuple[str, str]] = []
    for i, f in enumerate(files):
        original = f.filename or f"{prefix}_{i}"
        ext = os.path.splitext(original)[1].lower() or (".jpg" if prefix == "src" else ".mp4")
        path = os.path.join(upload_dir, f"{prefix}_{i}{ext}")
        with open(path, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append((path, original))
    return saved


def _cleanup_finished_workspace() -> None:
    """Wipe everything under server/jobs/ except in-flight items."""
    if not os.path.isdir(worker.JOBS_DIR):
        return
    with _registry_lock:
        active_job_dirs = {os.path.basename(j.job_dir)
                           for j in _jobs.values() if j.phase not in ("done", "error")}
        active_batch_dirs = {os.path.basename(b.batch_dir)
                             for b in _batches.values() if b.phase not in ("done", "error")}
        # Forget terminal entries
        for jid in list(_jobs.keys()):
            if _jobs[jid].phase in ("done", "error"):
                del _jobs[jid]
        for bid in list(_batches.keys()):
            if _batches[bid].phase in ("done", "error"):
                del _batches[bid]
    keep = active_job_dirs | active_batch_dirs
    for entry in os.listdir(worker.JOBS_DIR):
        if entry in keep:
            continue
        full = os.path.join(worker.JOBS_DIR, entry)
        try:
            if os.path.isdir(full):
                shutil.rmtree(full, ignore_errors=True)
            else:
                os.remove(full)
        except Exception:
            pass


# ---- Routes: health ------------------------------------------------------

@app.get("/healthz", response_model=HealthResponse, tags=["health"])
async def healthz() -> HealthResponse:
    h = HealthResponse(
        status="ok",
        cuda=False,
        inswapper_loaded=worker._swapper is not None,
        face_analyser_loaded=worker._face_analyser is not None,
    )
    if worker._swapper is not None:
        try:
            providers = worker._swapper.session.get_providers()
            h.cuda = "CUDAExecutionProvider" in providers
        except Exception:
            pass
    return h


# ---- Routes: single job (one video) --------------------------------------

@app.post("/api/jobs", response_model=JobCreatedResponse, tags=["jobs"])
async def create_job(
    source: list[UploadFile] = File(..., description="1+ face images"),
    target: UploadFile = File(..., description="single target video"),
):
    """Single-video swap. For multi-video use POST /api/batches instead."""
    if not source:
        raise HTTPException(400, "at least one source image is required")
    _cleanup_finished_workspace()

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(worker.JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    saved_src = _save_uploads(job_dir, source, prefix="source")
    saved_tgt = _save_uploads(job_dir, [target], prefix="target")[0]

    # Detect sources up front so a 400 is returned synchronously if any
    # source has no detectable face — no point starting the job.
    try:
        worker.ensure_models()
        detected = worker.detect_source_faces([p for p, _ in saved_src])
    except RuntimeError as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, str(e))

    job = worker.Job(
        id=job_id,
        target_path=saved_tgt[0],
        target_filename=saved_tgt[1],
        job_dir=job_dir,
        out_audio_path=os.path.join(job_dir, "swapped.mp4"),
        hls_dir=os.path.join(job_dir, "hls"),
    )
    job.on_change = _make_job_on_change(job_id)
    with _registry_lock:
        _jobs[job_id] = job

    threading.Thread(
        target=worker.run_single_job,
        args=(job, detected),
        daemon=True,
        name=f"job-{job_id}",
    ).start()

    return JobCreatedResponse(
        id=job_id,
        status_url=f"/api/jobs/{job_id}/status",
        ws_url=f"/api/jobs/{job_id}/ws",
        hls_url=f"/api/jobs/{job_id}/hls/playlist.m3u8",
        download_url=f"/api/jobs/{job_id}/download",
    )


@app.get("/api/jobs/{job_id}/status", response_model=JobStatus, tags=["jobs"])
async def get_job_status(job_id: str = FPath(...)):
    with _registry_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return _job_to_status(job)


@app.get("/api/jobs/{job_id}/hls/{fname}", tags=["jobs"])
async def get_hls_file(job_id: str = FPath(...), fname: str = FPath(...)):
    with _registry_lock:
        job = _jobs.get(job_id)
    if not job or not os.path.isdir(job.hls_dir):
        raise HTTPException(404)
    if "/" in fname or "\\" in fname or ".." in fname:
        raise HTTPException(400, "bad filename")
    if not (fname.endswith(".m3u8") or fname.endswith(".ts")):
        raise HTTPException(400, "only .m3u8 / .ts allowed")
    path = os.path.join(job.hls_dir, fname)
    if not os.path.isfile(path):
        raise HTTPException(404)
    media_type = "application/vnd.apple.mpegurl" if fname.endswith(".m3u8") else "video/mp2t"
    resp = FileResponse(path, media_type=media_type)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/api/jobs/{job_id}/file", tags=["jobs"])
async def get_job_file_inline(job_id: str = FPath(...)):
    with _registry_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404)
    if not os.path.isfile(job.out_audio_path):
        raise HTTPException(404, "not ready yet")
    return FileResponse(job.out_audio_path, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/download", tags=["jobs"])
async def get_job_download(job_id: str = FPath(...)):
    with _registry_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404)
    if not os.path.isfile(job.out_audio_path):
        raise HTTPException(404, "not ready yet")
    nice = (os.path.splitext(job.target_filename)[0] or "swap") + "_swapped.mp4"
    return FileResponse(
        job.out_audio_path,
        media_type="video/mp4",
        filename=nice,
    )


@app.websocket("/api/jobs/{job_id}/ws")
async def job_ws(ws: WebSocket, job_id: str):
    await ws.accept()
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue(maxsize=64)
    with _ws_lock:
        _job_ws.setdefault(job_id, []).append((loop, q))
    # Send initial snapshot
    with _registry_lock:
        job = _jobs.get(job_id)
    if job:
        await ws.send_text(_job_to_status(job).model_dump_json())
    try:
        while True:
            text = await q.get()
            await ws.send_text(text)
    except WebSocketDisconnect:
        pass
    finally:
        with _ws_lock:
            slot = _job_ws.get(job_id, [])
            slot[:] = [(l, qq) for l, qq in slot if qq is not q]
            if not slot:
                _job_ws.pop(job_id, None)


# ---- Routes: batch (1+ videos) ------------------------------------------

@app.post("/api/batches", response_model=BatchCreatedResponse, tags=["batches"])
async def create_batch(
    source: list[UploadFile] = File(..., description="1+ face images"),
    target: list[UploadFile] = File(..., description="1+ target videos"),
):
    """Batch swap: same source faces applied to N videos sequentially."""
    if not source:
        raise HTTPException(400, "at least one source image is required")
    if not target:
        raise HTTPException(400, "at least one target video is required")
    _cleanup_finished_workspace()

    batch_id = uuid.uuid4().hex[:12]
    batch_dir = os.path.join(worker.JOBS_DIR, batch_id)
    os.makedirs(batch_dir, exist_ok=True)

    # Sources go in batch_dir/sources/ and are referenced by every child job
    sources_dir = os.path.join(batch_dir, "sources")
    saved_src = _save_uploads(sources_dir, source, prefix="source")

    try:
        worker.ensure_models()
        # Validate all sources have detectable faces before accepting the batch
        worker.detect_source_faces([p for p, _ in saved_src])
    except RuntimeError as e:
        shutil.rmtree(batch_dir, ignore_errors=True)
        raise HTTPException(400, str(e))

    shared_sources = [worker.SourceSpec(path=p) for p, _ in saved_src]

    # One child Job per uploaded video; each gets its own subdir under the batch
    child_jobs: list[worker.Job] = []
    for idx, tgt in enumerate(target):
        job_id = uuid.uuid4().hex[:12]
        job_dir = os.path.join(batch_dir, f"job_{idx:02d}_{job_id}")
        os.makedirs(job_dir, exist_ok=True)
        saved_tgt = _save_uploads(job_dir, [tgt], prefix="target")[0]
        job = worker.Job(
            id=job_id,
            target_path=saved_tgt[0],
            target_filename=saved_tgt[1],
            job_dir=job_dir,
            out_audio_path=os.path.join(job_dir, "swapped.mp4"),
            hls_dir=os.path.join(job_dir, "hls"),
        )
        job.on_change = _make_job_on_change(job_id)
        child_jobs.append(job)

    batch = worker.BatchJob(
        id=batch_id,
        batch_dir=batch_dir,
        sources=shared_sources,
        jobs=child_jobs,
    )
    batch.on_change = _make_batch_on_change(batch_id)
    with _registry_lock:
        _batches[batch_id] = batch
        for j in child_jobs:
            _jobs[j.id] = j

    threading.Thread(
        target=worker.run_batch,
        args=(batch,),
        daemon=True,
        name=f"batch-{batch_id}",
    ).start()

    return BatchCreatedResponse(
        id=batch_id,
        status_url=f"/api/batches/{batch_id}/status",
        ws_url=f"/api/batches/{batch_id}/ws",
        download_zip_url=f"/api/batches/{batch_id}/download",
        jobs=[j.id for j in child_jobs],
    )


@app.get("/api/batches/{batch_id}/status", response_model=BatchStatus, tags=["batches"])
async def get_batch_status(batch_id: str = FPath(...)):
    with _registry_lock:
        batch = _batches.get(batch_id)
    if not batch:
        raise HTTPException(404)
    return _batch_to_status(batch)


@app.get("/api/batches/{batch_id}/download", tags=["batches"])
async def download_batch_zip(batch_id: str = FPath(...)):
    """Stream a ZIP of every finished MP4 in the batch. Skips children that
    haven't finished or errored."""
    with _registry_lock:
        batch = _batches.get(batch_id)
    if not batch:
        raise HTTPException(404)

    def _stream_zip():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
            for j in batch.jobs:
                if j.phase != "done" or not os.path.isfile(j.out_audio_path):
                    continue
                base = os.path.splitext(j.target_filename)[0] or j.id
                zf.write(j.out_audio_path, arcname=f"{base}_swapped.mp4")
        buf.seek(0)
        chunk = buf.read(64 * 1024)
        while chunk:
            yield chunk
            chunk = buf.read(64 * 1024)

    headers = {"Content-Disposition": f'attachment; filename="batch_{batch_id}.zip"'}
    return StreamingResponse(_stream_zip(), media_type="application/zip", headers=headers)


@app.websocket("/api/batches/{batch_id}/ws")
async def batch_ws(ws: WebSocket, batch_id: str):
    await ws.accept()
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue(maxsize=64)
    with _ws_lock:
        _batch_ws.setdefault(batch_id, []).append((loop, q))
    with _registry_lock:
        batch = _batches.get(batch_id)
    if batch:
        await ws.send_text(_batch_to_status(batch).model_dump_json())
    try:
        while True:
            text = await q.get()
            await ws.send_text(text)
    except WebSocketDisconnect:
        pass
    finally:
        with _ws_lock:
            slot = _batch_ws.get(batch_id, [])
            slot[:] = [(l, qq) for l, qq in slot if qq is not q]
            if not slot:
                _batch_ws.pop(batch_id, None)


# ---- Static frontend (mounted at "/" in production) ----------------------

_STATIC_DIST = os.path.join(os.path.dirname(__file__), "static", "dist")
if os.path.isdir(_STATIC_DIST):
    app.mount("/", StaticFiles(directory=_STATIC_DIST, html=True), name="frontend")
