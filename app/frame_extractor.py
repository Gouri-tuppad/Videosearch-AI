from pathlib import Path
from typing import Any
import subprocess
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path | None = None,
) -> list[dict[str, Any]]:

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    if output_dir is None:
        output_dir = Path(settings.frames_folder) / video_path.stem
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    interval = float(settings.frame_extraction_interval)

    if interval <= 0:
        interval = 1.0

    output_pattern = output_dir / "frame_%06d.jpg"

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval}",
        "-q:v",
        "2",
        str(output_pattern),
    ]

    logger.info(
        "Extracting frames with FFmpeg: video=%s interval=%s",
        video_path,
        interval,
    )

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )

        logger.debug(
            "FFmpeg output: %s",
            completed.stdout,
        )

        if completed.stderr:
            logger.debug(
                "FFmpeg stderr: %s",
                completed.stderr,
            )

    except FileNotFoundError as exc:
        logger.exception("FFmpeg was not found")
        raise RuntimeError(
            "FFmpeg is not installed or not available in PATH."
        ) from exc

    except subprocess.CalledProcessError as exc:
        logger.error(
            "FFmpeg frame extraction failed: %s",
            exc.stderr,
        )
        raise RuntimeError(
            f"FFmpeg frame extraction failed:\n{exc.stderr}"
        ) from exc

    frame_files = sorted(
        output_dir.glob("frame_*.jpg")
    )

    if not frame_files:
        raise RuntimeError(
            f"FFmpeg completed but no frames were generated in {output_dir}"
        )

    extracted: list[dict[str, Any]] = []

    for index, frame_path in enumerate(frame_files):

        timestamp = index * interval

        extracted.append(
            {
                "video_name": video_path.name,
                "frame_index": index,
                "frame_filename": frame_path.name,
                "timestamp": round(timestamp, 3),
            }
        )

    logger.info(
        "Extracted %d frames from %s",
        len(extracted),
        video_path,
    )

    return extracted