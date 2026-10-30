from pathlib import Path

import cv2
import numpy as np

from app.frame_extractor import extract_frames


def test_extract_frames_creates_timestamped_images(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, 2.0, (160, 120))

    try:
        for _ in range(6):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    output_dir = tmp_path / "frames" / video_path.stem
    frames = extract_frames(video_path, output_dir)

    assert len(frames) >= 1
    assert frames[0]["frame_filename"].startswith("frame_")
    assert frames[0]["frame_filename"].endswith(".jpg")
    assert (output_dir / frames[0]["frame_filename"]).exists()
    assert frames[0]["timestamp"] >= 0
