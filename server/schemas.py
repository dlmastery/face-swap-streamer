"""Pydantic schemas — the API contract between the FastAPI server and any
client (the Next.js frontend, but also any future client). These are the
*external* shapes; the worker uses its own dataclasses internally."""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


JobPhase = Literal[
    "queued",
    "loading_models",
    "detecting_source",
    "finding_reference",
    "streaming",
    "finalising",
    "done",
    "error",
]


class SourceInfo(BaseModel):
    """One uploaded source's detection state, exposed in the status payload."""
    gender: Literal["", "M", "F"] = ""
    age: int = 0
    ref_frame: int = -1
    ref_votes: int = 0
    ref_pool: int = 0


class JobStatus(BaseModel):
    """The complete job state, returned by GET /api/jobs/{id}/status and
    pushed over the WebSocket."""
    id: str
    phase: JobPhase
    message: str = ""
    error: str = ""

    # Detection / reference results — list keeps order matching the upload.
    sources: list[SourceInfo] = Field(default_factory=list)

    # Backwards-compat top-level mirror of sources[0] (legacy clients).
    detected_gender: Literal["", "M", "F"] = ""
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

    started: float = 0.0
    finished: float = 0.0


class JobCreatedResponse(BaseModel):
    """Returned by POST /api/jobs after a successful single-video upload."""
    id: str
    status_url: str = Field(description="GET this for the latest JobStatus")
    ws_url: str = Field(description="WebSocket for real-time JobStatus push")
    hls_url: str = Field(description="HLS playlist (only available once phase >= streaming)")
    download_url: str = Field(description="Final MP4 (only available once phase == done)")


# ---- Batch ---------------------------------------------------------------
# A "batch" is a multi-video upload: same source faces applied to N videos
# sequentially on a single GPU. Each video gets its own Job under the hood.

BatchPhase = Literal[
    "queued",
    "processing",   # at least one child job in flight
    "done",         # every child job is done
    "error",        # one or more child jobs failed
]


class BatchJobSummary(BaseModel):
    """One child job's summary inside a batch — minimal fields for the
    batch overview UI; full status comes from /api/jobs/{id}/status."""
    id: str
    target_filename: str = Field(description="Original uploaded video filename, for display")
    phase: JobPhase
    progress_pct: float = 0.0
    proc_fps: float = 0.0
    error: str = ""


class BatchStatus(BaseModel):
    id: str
    phase: BatchPhase
    message: str = ""
    sources: list[SourceInfo] = Field(default_factory=list)
    jobs: list[BatchJobSummary] = Field(default_factory=list)
    total_videos: int = 0
    done_videos: int = 0
    error_videos: int = 0
    started: float = 0.0
    finished: float = 0.0


class BatchCreatedResponse(BaseModel):
    """Returned by POST /api/batches after a successful multi-video upload."""
    id: str
    status_url: str
    ws_url: str
    download_zip_url: str = Field(description="ZIP of every finished MP4 (only valid once phase == done)")
    jobs: list[str] = Field(description="Child job IDs in upload order")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"] = "ok"
    cuda: bool = False
    inswapper_loaded: bool = False
    face_analyser_loaded: bool = False
    gpu_name: Optional[str] = None
    gpu_mem_used_mb: Optional[int] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
