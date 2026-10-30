
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes.health import router as health_router
from app.api.routes.video_info import router as video_info_router
from app.api.routes.videos import router as videos_router
from app.api.routes.detections import router as detections_router
from app.api.routes.search import router as search_router
from app.api.routes.clips import router as clips_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.db.database import Base, engine
from app.pipeline import ensure_clip_vision_encoder_loaded
from app.search_service import ensure_clip_text_encoder_loaded

setup_logging()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
)

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include all API routers before static file mounts so specific endpoints take precedence
app.include_router(health_router)
app.include_router(videos_router)
app.include_router(video_info_router)
app.include_router(detections_router)
app.include_router(search_router)
app.include_router(clips_router)

from pathlib import Path

Path(settings.clips_folder).mkdir(parents=True, exist_ok=True)
app.mount(
    "/clips",
    StaticFiles(directory=settings.clips_folder),
    name="clips",
)

# Mount data/ so that per-video clips (data/<video_id>/clips/) are servable
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/data",
    StaticFiles(directory=str(DATA_DIR)),
    name="data",
)

from fastapi import UploadFile, File, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.api.routes.videos import upload_video

@app.post("/upload", include_in_schema=False)
async def upload_video_root(file: UploadFile = File(...), db: Session = Depends(get_db)):
    return await upload_video(file=file, db=db)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"message": "Natural Language Video Search System API is running"}


@app.on_event("startup")
def startup_event() -> None:
    Base.metadata.create_all(bind=engine)
    # Load CLIP models in background threads to avoid blocking startup
    import threading
    threading.Thread(target=ensure_clip_text_encoder_loaded, daemon=True).start()
    threading.Thread(target=ensure_clip_vision_encoder_loaded, daemon=True).start()
