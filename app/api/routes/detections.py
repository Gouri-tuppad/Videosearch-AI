import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.crud import list_detections_for_frame
from app.db.database import SessionLocal
from app.pipeline import process_new_video

router = APIRouter(prefix="/detections", tags=["detections"])

logger = logging.getLogger(__name__)


def get_db_session() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/process/{filename}", status_code=status.HTTP_200_OK)
def process_video(filename: str, db: Session = Depends(get_db_session)) -> dict[str, Any]:
    safe_filename = Path(filename).name
    video_path = Path(settings.upload_folder) / safe_filename
    logger.info("Start request to process video: %s", safe_filename)

    # Verify uploaded file
    logger.info("Verifying uploaded file exists: %s", video_path)
    if not video_path.exists():
        logger.info("Uploaded file not found: %s", video_path)
        raise HTTPException(status_code=404, detail="Video file not found")
    logger.info("Verified uploaded file exists: %s", video_path)

    try:
        result = process_new_video(video_path)
    except Exception as exc:
        logger.exception("Processing failed for video %s", safe_filename)
        raise HTTPException(status_code=500, detail=f"Video processing failed: {exc}") from exc

    logger.info("Finish processing request for: %s", safe_filename)
    return result


@router.get("/frame/{frame_id}", status_code=status.HTTP_200_OK)
def get_frame_detections(frame_id: int, db: Session = Depends(get_db_session)) -> list[dict[str, Any]]:
    detections = list_detections_for_frame(db, frame_id)
    return [
        {
            "id": detection.id,
            "object_name": detection.object_name,
            "confidence": detection.confidence,
            "bbox": [
                detection.bbox_xmin,
                detection.bbox_ymin,
                detection.bbox_xmax,
                detection.bbox_ymax,
            ],
        }
        for detection in detections
    ]


@router.get("/objects/{filename}", status_code=status.HTTP_200_OK)
def get_detected_objects(filename: str, db: Session = Depends(get_db_session)) -> dict[str, Any]:
    """Return all unique objects detected by YOLO in a specific video's frames."""
    from app.db.crud import get_video_by_filename, list_detected_objects_for_video

    safe_filename = Path(filename).name
    video = get_video_by_filename(db, safe_filename)
    if not video:
        raise HTTPException(status_code=404, detail=f"Video '{safe_filename}' not found in database")

    objects = list_detected_objects_for_video(db, video.id)
    return {
        "video_name": safe_filename,
        "video_id": video.id,
        "detected_objects": objects,
        "total_unique_objects": len(objects),
    }
