import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.pipeline import process_new_video


class PipelineIsolationTests(unittest.TestCase):
    def test_process_new_video_returns_current_video_objects_and_unique_frame_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "videos" / "sample_video.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"fake video data")

            fake_session = MagicMock()
            fake_session.close.return_value = None

            def fake_extract(video, output_dir):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "frame_000001.jpg").write_bytes(b"fake frame")
                return [{"frame_filename": "frame_000001.jpg", "timestamp": 0.0}]

            def fake_generate(video_name, frame_dir, device=None):
                output_dir = root / "embeddings" / video_name
                output_dir.mkdir(parents=True, exist_ok=True)
                import numpy as np

                np.save(output_dir / "frame_embeddings.npy", np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
                (output_dir / "metadata.json").write_text(
                    '{"items": [{"frame_index": 0, "frame_name": "frame_000001.jpg", "timestamp": 0.0}]}',
                    encoding="utf-8",
                )

            with patch("app.pipeline.VIDEO_DIR", root / "videos"), \
                 patch("app.pipeline.FRAME_DIR", root / "frames"), \
                 patch("app.pipeline.CLIPS_DIR", root / "clips"), \
                 patch("app.pipeline.PROCESSING_DIR", root / "processing"), \
                 patch("app.pipeline.STATUS_FILE", root / "processing" / "status.json"), \
                 patch("app.pipeline.EMBEDDINGS_DIR", root / "embeddings"), \
                 patch("app.pipeline.INDEX_DIR", root / "faiss_indexes"), \
                 patch("app.pipeline.SessionLocal", return_value=fake_session), \
                 patch("app.pipeline._video_already_processed", return_value=False), \
                 patch("app.pipeline.get_video_metadata", return_value={"duration": 10.0, "fps": 1.0}), \
                 patch("app.pipeline.extract_frames", side_effect=fake_extract) as mock_extract, \
                 patch("app.pipeline._get_or_create_video", return_value=MagicMock(id=42)), \
                 patch("app.pipeline.create_frame", return_value=MagicMock(id=7)), \
                 patch("app.pipeline.create_detection"), \
                 patch("app.pipeline._get_clip_model", return_value=(MagicMock(), MagicMock())), \
                 patch("app.pipeline.generate_frame_embeddings", side_effect=fake_generate), \
                 patch("app.pipeline.build_index"), \
                 patch("app.pipeline.detector.predict", return_value=[{
                     "object_name": "cat",
                     "confidence": 0.9,
                     "bbox_xmin": 0.0,
                     "bbox_ymin": 0.0,
                     "bbox_xmax": 5.0,
                     "bbox_ymax": 5.0,
                 }]):
                result = process_new_video(video_path)

            self.assertIn("cat", result["detected_objects"])
            self.assertNotIn("person", [name.lower() for name in result["detected_objects"]])
            self.assertEqual(mock_extract.call_args[0][1].name, "sample_video.mp4")


if __name__ == "__main__":
    unittest.main()
