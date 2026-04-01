"""FastAPI server wrapping the soccer-vision pipeline.

Provides endpoints for video upload, processing status, and result retrieval.
This is a synchronous, single-job server (Tier 1 — no background workers).
"""

import os
import shutil
import uuid
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from utils.schema import AppConfig
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Soccer Vision API",
    description="Video analytics pipeline for player counting per team",
    version="1.0.0",
)

# In-memory job store (Tier 1 — no database)
jobs: Dict[str, dict] = {}

# Directories
UPLOAD_DIR = "/tmp/soccer-vision/uploads"
# Use the config's output_dir so results land in the mounted volume (./output:/app/output).
# Fall back to /tmp only if config is unavailable at startup.
_startup_config_path = os.environ.get("CONFIG_PATH", "config.yaml")
if os.path.exists(_startup_config_path):
    from utils.schema import AppConfig as _AppConfig
    _startup_cfg = _AppConfig.from_yaml(_startup_config_path)
    OUTPUT_BASE_DIR = str(Path(_startup_cfg.output.output_dir).resolve())
else:
    OUTPUT_BASE_DIR = "/tmp/soccer-vision/output"

# Maximum upload size: 500 MB
MAX_UPLOAD_SIZE = 500 * 1024 * 1024


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    message: str = ""


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "soccer-vision"}


@app.post("/process", response_model=JobResponse)
async def process_video(
    file: UploadFile = File(...),
    frame_skip: int = Form(1),
    save_video: bool = Form(True),
    train_data_json: Optional[str] = Form(None),
):
    """Upload a video and start processing.

    Args:
        file: Video file upload (.mp4, .avi, .mov).
        frame_skip: Process every Nth frame.
        save_video: Whether to generate annotated output video.

    Returns:
        JobResponse with job_id and status.
    """
    # Validate file extension
    ext = Path(file.filename or "unknown.mp4").suffix.lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {ext}. Use .mp4, .avi, or .mov",
        )

    job_id = str(uuid.uuid4())[:8]

    # Save uploaded file with size limit enforcement
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")

    bytes_written = 0
    chunk_size = 1024 * 1024  # 1 MB chunks
    with open(input_path, "wb") as f:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_SIZE:
                f.close()
                os.remove(input_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum upload size is {MAX_UPLOAD_SIZE // (1024 * 1024)} MB.",
                )
            f.write(chunk)

    logger.info(f"Job {job_id}: Video uploaded — {file.filename} ({ext})")

    # Output directory for this job
    output_dir = os.path.join(OUTPUT_BASE_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize job state
    jobs[job_id] = {
        "status": JobStatus.PROCESSING,
        "progress": 0.0,
        "message": "Starting pipeline",
        "input_path": input_path,
        "output_dir": output_dir,
        "report": None,
    }

    # Run pipeline synchronously (Tier 1)
    try:
        # Load config
        config_path = os.environ.get("CONFIG_PATH", "config.yaml")
        if os.path.exists(config_path):
            config = AppConfig.from_yaml(config_path)
        else:
            config = AppConfig()

        # Override frame_skip from request
        config.output.frame_skip = frame_skip

        from pipeline import Pipeline

        pipeline = Pipeline(config)

        def update_progress(current: int, total: int, stage: str):
            if total > 0:
                jobs[job_id]["progress"] = round(current / total * 100, 1)
            jobs[job_id]["message"] = stage

        # Parse seeds/train_data if provided
        train_data = None
        if train_data_json:
            import json
            try:
                train_data = json.loads(train_data_json)
            except Exception as e:
                logger.warning(f"Failed to parse train_data_json, ignoring: {e}")

        report = pipeline.run(
            input_path=input_path,
            output_dir=output_dir,
            save_video=save_video,
            on_progress=update_progress,
            train_data=train_data,
        )

        jobs[job_id]["status"] = JobStatus.COMPLETE
        jobs[job_id]["progress"] = 100.0
        jobs[job_id]["message"] = "Processing complete"
        jobs[job_id]["report"] = report

        # Copy predictions.csv to the base output dir so label_tracks.py can
        # always find it at output/predictions.csv without knowing the job_id.
        src_pred = os.path.join(output_dir, "predictions.csv")
        if os.path.exists(src_pred):
            shutil.copy2(src_pred, os.path.join(OUTPUT_BASE_DIR, "predictions.csv"))
            logger.info(f"Job {job_id}: predictions.csv copied to {OUTPUT_BASE_DIR}/predictions.csv")

        logger.info(f"Job {job_id}: Complete — {len(report.frames)} frames processed")

    except Exception as e:
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["message"] = str(e)
        logger.exception(f"Job {job_id}: Failed — {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    return JobResponse(
        job_id=job_id,
        status=jobs[job_id]["status"],
        progress=jobs[job_id]["progress"],
        message=jobs[job_id]["message"],
    )


@app.get("/status/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Check processing progress for a job.

    Args:
        job_id: Job identifier returned from POST /process.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = jobs[job_id]
    return JobResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
    )


@app.get("/results/{job_id}/report")
async def get_report(job_id: str):
    """Download the JSON report for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = jobs[job_id]
    if job["status"] != JobStatus.COMPLETE:
        raise HTTPException(status_code=400, detail=f"Job not complete. Status: {job['status']}")

    report_path = os.path.join(job["output_dir"], "report.json")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report file not found")

    return FileResponse(
        report_path,
        media_type="application/json",
        filename=f"{job_id}_report.json",
    )


@app.get("/results/{job_id}/video")
async def get_video(job_id: str):
    """Download the annotated video for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = jobs[job_id]
    if job["status"] != JobStatus.COMPLETE:
        raise HTTPException(status_code=400, detail=f"Job not complete. Status: {job['status']}")

    video_path = os.path.join(job["output_dir"], "output.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{job_id}_output.mp4",
    )


@app.get("/results/{job_id}/summary")
async def get_summary(job_id: str):
    """Get the human-readable summary for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = jobs[job_id]
    if job["status"] != JobStatus.COMPLETE:
        raise HTTPException(status_code=400, detail=f"Job not complete. Status: {job['status']}")

    summary_path = os.path.join(job["output_dir"], "summary.txt")
    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail="Summary file not found")

    with open(summary_path, "r") as f:
        summary = f.read()

    return {"job_id": job_id, "summary": summary}
