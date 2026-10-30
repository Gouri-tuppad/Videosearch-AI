import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

import cv2

from app.core.config import settings

logger = logging.getLogger(__name__)


def _get_video_duration(video_path: Path) -> float:
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    try:
        fps = float(video.get(cv2.CAP_PROP_FPS))
        frame_count = float(video.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        video.release()

    if fps <= 0 or frame_count <= 0:
        raise ValueError("Unable to determine video duration")

    return frame_count / fps


def _safe_output_path(video_path: Path, timestamp: float, output_dir: Path) -> Path:
    safe_timestamp = f"{timestamp:.3f}".replace(".", "_")
    filename = f"{video_path.stem}_{safe_timestamp}.mp4"
    path = output_dir / filename
    counter = 1

    while path.exists():
        path = output_dir / f"{video_path.stem}_{safe_timestamp}_{counter}.mp4"
        counter += 1

    return path


def extract_clip(
    video_path: str | Path,
    timestamp: float,
    output_dir: str | Path | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> Dict[str, Any]:
    video_path = Path(video_path)
    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if timestamp < 0:
        logger.error("Invalid timestamp: %s", timestamp)
        raise ValueError("Timestamp must be non-negative")

    output_dir = Path(output_dir) if output_dir is not None else Path(settings.clips_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = _get_video_duration(video_path)
    if timestamp > duration:
        logger.error("Timestamp %s is outside video duration %s", timestamp, duration)
        raise ValueError("Timestamp is outside the video duration")

    padding = max(0.0, float(settings.clip_padding_seconds))
    clip_start = max(0.0, float(start_time) if start_time is not None else timestamp - padding)
    clip_end = min(duration, float(end_time) if end_time is not None else timestamp + padding)
    clip_duration = max(0.1, clip_end - clip_start)

    output_path = _safe_output_path(video_path, timestamp, output_dir)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{clip_start:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{clip_duration:.3f}",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-preset",
        "fast",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    logger.debug(
        "Extracting clip: video=%s timestamp=%s start=%s end=%s duration=%s output=%s",
        video_path,
        timestamp,
        clip_start,
        clip_end,
        clip_duration,
        output_path,
    )

    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info("Clip created: %s", output_path)
        logger.debug("FFmpeg output: %s", completed.stdout)
        if not output_path.exists():
            raise RuntimeError("Clip was not created.")

        if output_path.stat().st_size == 0:
            raise RuntimeError("Generated clip is empty.")

        logger.info("Clip size: %.2f MB", output_path.stat().st_size / (1024 * 1024))
    except subprocess.CalledProcessError as exc:
        logger.error(
            "FFmpeg failed for %s: %s",
            video_path,
            exc.stderr or exc.stdout or str(exc),
        )
        raise RuntimeError(
            f"FFmpeg failed to extract clip from {video_path}: {exc.stderr or exc.stdout or str(exc)}"
        ) from exc
    except FileNotFoundError as exc:
        logger.exception("FFmpeg executable not found")
        raise RuntimeError("FFmpeg is not installed or not found in PATH") from exc

    return {
        "clip_filename": output_path.name,
        "clip_path": str(output_path.resolve()),
        "start_time": round(clip_start, 3),
        "end_time": round(clip_end, 3),
    }
