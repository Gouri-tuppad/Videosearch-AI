from pathlib import Path

import cv2
import numpy as np

from app.clip_generator import extract_clip


def test_extract_clip_bounds(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (160, 120))

    try:
        for _ in range(50):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    clip = extract_clip(video_path, timestamp=1.0, output_dir=tmp_path / "clips")
    clip_path = Path(clip["clip_path"])

    assert clip_path.exists()
    assert "sample" in clip["clip_filename"]
