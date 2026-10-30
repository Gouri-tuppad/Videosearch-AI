from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.config import settings
from app.core.detector import detector
from app.db.database import engine
from app.faiss_index import INDEX_DIR, IndexNotFoundError, IndexCorruptedError, load_index
from app.search_service import ensure_clip_text_encoder_loaded

router = APIRouter(prefix="/health", tags=["health"])


def _check_database_connection() -> dict[str, str]:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"status": "ok", "details": "Database connection successful"}
    except Exception as exc:
        return {"status": "unavailable", "details": str(exc)}


def _check_yolo_model() -> dict[str, str]:
    try:
        detector.ensure_model_loaded()
        return {"status": "loaded", "details": "YOLO model loaded"}
    except Exception as exc:
        return {"status": "unavailable", "details": str(exc)}


def _check_clip_model() -> dict[str, str]:
    try:
        ensure_clip_text_encoder_loaded()
        return {"status": "loaded", "details": "CLIP model loaded"}
    except Exception as exc:
        return {"status": "unavailable", "details": str(exc)}


def _check_faiss_index() -> dict[str, str]:
    if not INDEX_DIR.exists():
        return {"status": "missing", "details": "FAISS index directory does not exist"}
    index_files = [path for path in INDEX_DIR.glob("*.index") if path.is_file()]
    if not index_files:
        return {"status": "missing", "details": "No FAISS index files found"}

    try:
        load_index(index_files[0].stem)
        return {"status": "available", "details": f"Found {len(index_files)} FAISS index file(s)"}
    except (IndexNotFoundError, IndexCorruptedError, Exception) as exc:
        return {"status": "unavailable", "details": str(exc)}


def _check_uploaded_video() -> dict[str, str]:
    upload_folder = Path(settings.upload_folder)
    if not upload_folder.exists():
        return {"status": "missing", "details": "Upload folder does not exist"}
    uploaded_files = [path for path in upload_folder.iterdir() if path.is_file()]
    if not uploaded_files:
        return {"status": "missing", "details": "No uploaded video files found"}
    return {"status": "available", "details": f"{len(uploaded_files)} uploaded video file(s) present"}


@router.get("", status_code=200)
async def health_check() -> JSONResponse:
    database = _check_database_connection()
    yolo_model = _check_yolo_model()
    clip_model = _check_clip_model()
    faiss_index = _check_faiss_index()
    uploaded_video = _check_uploaded_video()

    overall_ok = all(
        component["status"] in {"ok", "loaded", "available"}
        for component in (database, yolo_model, clip_model, faiss_index, uploaded_video)
    )
    status = "ok" if overall_ok else "degraded"

    payload = {
        "status": status,
        "database": database,
        "yolo_model": yolo_model,
        "clip_model": clip_model,
        "faiss_index": faiss_index,
        "uploaded_video": uploaded_video,
    }

    return JSONResponse(status_code=200 if overall_ok else 503, content=payload)
