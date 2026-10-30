from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)


class Settings(BaseSettings):
    app_name: str = "Natural Language Video Search System"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: str = "development"
    database_url: str = "sqlite:///./app.db"

    upload_folder: Path = Path("videos")
    frames_folder: Path = Path("frames")
    clips_folder: Path = Path("clips")
    processing_folder: Path = Path("processing")
    faiss_index_folder: Path = Path("faiss_indexes")
    embeddings_folder: Path = Path("embeddings")

    clip_model_name: str = "openai/clip-vit-base-patch32"
    yolo_model_path: Path = Path("yolov8n.pt")
    similarity_threshold: float = 0.20
    clip_min_score: float = 0.25
    clip_score_margin: float = 0.03
    frame_extraction_interval: float = 0.5
    clip_padding_seconds: float = 1.5
    event_min_consecutive: int = 2
    event_gap_tolerance: float = 2.0
    event_max_length: float = 30.0
    yolo_confidence: float = 0.5
    max_upload_size: int = 500 * 1024 * 1024
    allowed_mime_types: tuple[str, ...] = ("video/mp4", "video/x-m4v", "video/quicktime")

    log_folder: Path = Path("logs")
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


settings = get_settings()
