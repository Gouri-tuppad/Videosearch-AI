from pathlib import Path
import logging
import threading
from datetime import datetime, date, time
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.crud import create_video, get_video_by_filename, get_video_by_hash
from app.db.session import get_db
from app.pipeline import compute_file_hash, process_new_video

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/videos", tags=["videos"])

UPLOAD_DIR = Path(settings.upload_folder)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_SIZE = settings.max_upload_size
ALLOWED_MIME_TYPES = set(settings.allowed_mime_types)


def _safe_filename(filename: str) -> str:
    path = Path(filename)
    stem = path.stem
    suffix = path.suffix or ".mp4"
    candidate = f"{stem}{suffix}"
    counter = 1

    while (UPLOAD_DIR / candidate).exists():
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1

    return candidate


def _run_pipeline_bg(video_path: str) -> None:
    """Run the processing pipeline in a background thread."""
    try:
        process_new_video(video_path)
    except Exception:
        logger.exception("Background pipeline failed for %s", video_path)
    finally:
        import gc
        gc.collect()


@router.post("/upload", status_code=status.HTTP_200_OK)
async def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type. Please upload a video file.")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds maximum upload size of 500MB")

    filename = _safe_filename(file.filename)
    destination = UPLOAD_DIR / filename

    logger.info(
        "Upload requested: filename=%s size=%d content_type=%s",
        filename,
        len(contents),
        file.content_type,
    )

    try:
        with destination.open("wb") as buffer:
            buffer.write(contents)
        logger.info("Saved uploaded video: %s", filename)
    except OSError as exc:
        logger.exception("Failed to save uploaded file: %s", filename)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file") from exc

    # ── Hash-based dedup: skip reprocessing if file already processed ──
    file_hash = compute_file_hash(destination)
    existing_by_hash = get_video_by_hash(db, file_hash)
    if existing_by_hash:
        logger.info("File hash match — video already processed: hash=%s video_id=%d", file_hash, existing_by_hash.id)
        # Remove the duplicate upload file if it has a different filename
        try:
            if destination.resolve() != (UPLOAD_DIR / existing_by_hash.filename).resolve():
                destination.unlink(missing_ok=True)
        except Exception:
            pass
        return {
            "filename": existing_by_hash.filename,
            "size": len(contents),
            "upload_status": "uploaded",
            "processing_status": existing_by_hash.status,
            "video_id": existing_by_hash.id,
            "message": "File already processed (hash match). Skipping reprocessing.",
        }

    # Check if video already exists in database by filename
    existing_video = get_video_by_filename(db, filename)
    if existing_video:
        logger.info("Video already exists in database: %s", filename)
        # Update hash if missing
        if not existing_video.file_hash:
            existing_video.file_hash = file_hash
            db.commit()
        # If already ready, skip reprocessing
        if existing_video.status == "ready":
            return {
                "filename": filename,
                "size": len(contents),
                "upload_status": "uploaded",
                "processing_status": "ready",
                "video_id": existing_video.id,
                "message": "Video already processed.",
            }
        # If still processing or errored, restart pipeline
        video_id = existing_video.id
    else:
        # Save to database with current date and time
        try:
            current_datetime = datetime.now()
            current_date = current_datetime.date()
            current_time = current_datetime.time()

            video = create_video(
                db=db,
                filename=filename,
                duration=0.0,
                fps=0.0,
                upload_date=current_date,
                upload_time=current_time,
                file_hash=file_hash,
                status="processing",
            )
            video_id = video.id
            logger.info("Created video record in database: id=%d filename=%s", video_id, filename)
        except IntegrityError as exc:
            db.rollback()
            logger.warning("Duplicate video filename (race condition): %s", filename)
            raise HTTPException(status_code=409, detail="Video with this filename already exists") from exc
        except Exception as exc:
            db.rollback()
            logger.exception("Failed to save video to database: %s", filename)
            raise HTTPException(status_code=500, detail="Failed to save video metadata to database") from exc

    # ── Start background pipeline ──────────────────────────────────────
    logger.info("Starting background pipeline for video_id=%d filename=%s", video_id, filename)
    t = threading.Thread(target=_run_pipeline_bg, args=(str(destination),), daemon=True)
    t.start()

    return {
        "filename": filename,
        "size": len(contents),
        "upload_status": "uploaded",
        "processing_status": "processing",
        "video_id": video_id,
    }


@router.get("/status/{video_id}", status_code=status.HTTP_200_OK)
async def get_video_status(video_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return the current processing status and progress for a video."""
    from app.db.crud import get_video
    video = get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return {
        "video_id": video.id,
        "filename": video.filename,
        "status": video.status,
        "progress": video.progress,
    }


@router.get("", status_code=status.HTTP_200_OK)
async def list_all_videos(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return all uploaded videos stored in the database."""
    from app.db.crud import list_videos
    videos = list_videos(db)
    return {
        "total": len(videos),
        "videos": [
            {
                "id": v.id,
                "video_name": v.filename,
                "duration": v.duration,
                "fps": v.fps,
                "upload_date": str(v.upload_date),
                "upload_time": str(v.upload_time),
                "status": v.status,
                "progress": v.progress,
            }
            for v in videos
        ],
    }
