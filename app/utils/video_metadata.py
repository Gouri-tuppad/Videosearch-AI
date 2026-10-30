from pathlib import Path
from typing import Any
import logging

import cv2

logger = logging.getLogger(__name__)


def get_video_metadata(video_path: str | Path) -> dict[str, Any]:
    video_path = Path(video_path)
    logger.info("Opening video for metadata: %s", video_path)
    video = None
    try:
        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        fps = float(video.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        duration = frame_count / fps if fps > 0 else 0.0

        logger.info("Video metadata loaded: fps=%s, width=%s, height=%s, frames=%s", fps, width, height, frame_count)

        return {
            "duration": round(duration, 3),
            "fps": round(fps, 3),
            "width": width,
            "height": height,
            "total_frames": frame_count,
        }
    except Exception:
        logger.exception("Failed to get video metadata for %s", video_path)
        raise
    finally:
        if video is not None:
            video.release()
            del video
        import gc
        gc.collect()
