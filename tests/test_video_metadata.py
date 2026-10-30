from pathlib import Path

import cv2
import numpy as np

from app.utils.video_metadata import get_video_metadata


def test_get_video_metadata_returns_expected_values(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 240))

    try:
        for _ in range(20):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    metadata = get_video_metadata(video_path)

    assert metadata["fps"] == 10.0
    assert metadata["width"] == 320
    assert metadata["height"] == 240
    assert metadata["total_frames"] == 20
    assert metadata["duration"] > 0
