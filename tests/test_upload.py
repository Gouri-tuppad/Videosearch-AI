import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class UploadEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.upload_dir = Path(self.temp_dir.name) / "videos"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir_patcher = patch("app.api.routes.videos.UPLOAD_DIR", self.upload_dir)
        self.database_lookup_patcher = patch(
            "app.api.routes.videos.get_video_by_filename", return_value=None
        )
        self.create_video_patcher = patch(
            "app.api.routes.videos.create_video", return_value=type("Video", (), {"id": 1})()
        )
        self.pipeline_bg_patcher = patch("app.api.routes.videos._run_pipeline_bg")
        self.upload_dir_patcher.start()
        self.database_lookup_patcher.start()
        self.create_video_patcher.start()
        self.pipeline_bg_patcher.start()

    def tearDown(self) -> None:
        self.pipeline_bg_patcher.stop()
        self.create_video_patcher.stop()
        self.database_lookup_patcher.stop()
        self.upload_dir_patcher.stop()
        self.temp_dir.cleanup()

    def test_upload_accepts_mp4_file(self) -> None:
        client = TestClient(app)

        response = client.post(
            "/videos/upload",
            files={"file": ("sample.mp4", BytesIO(b"fake mp4 data"), "video/mp4")},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["filename"], "sample.mp4")
        self.assertEqual(payload["upload_status"], "uploaded")
        self.assertGreaterEqual(payload["size"], 0)
        self.assertTrue((self.upload_dir / "sample.mp4").exists())

    def test_upload_rejects_non_video_files(self) -> None:
        client = TestClient(app)

        response = client.post(
            "/videos/upload",
            files={"file": ("notes.txt", BytesIO(b"not a video"), "text/plain")},
        )

        self.assertEqual(response.status_code, 415)
        self.assertIn("video", response.json()["detail"].lower())

    def test_duplicate_filenames_are_suffixed(self) -> None:
        client = TestClient(app)

        first = client.post(
            "/videos/upload",
            files={"file": ("clip.mp4", BytesIO(b"first"), "video/mp4")},
        )
        second = client.post(
            "/videos/upload",
            files={"file": ("clip.mp4", BytesIO(b"second"), "video/mp4")},
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["filename"], "clip_1.mp4")
        self.assertTrue((self.upload_dir / "clip_1.mp4").exists())


if __name__ == "__main__":
    unittest.main()
