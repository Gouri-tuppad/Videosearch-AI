import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.api.routes.search import SearchRequest, search_video_route
from app.search_service import SearchResult


class SearchEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.videos_dir = Path(self.temp_dir.name) / "videos"
        self.clips_dir = Path(self.temp_dir.name) / "clips"
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        (self.videos_dir / "sample.mp4").write_bytes(b"fake video data")
        self.videos_dir_patcher = patch("app.api.routes.search.VIDEOS_DIR", self.videos_dir)
        self.clips_dir_patcher = patch("app.api.routes.search.CLIPS_DIR", self.clips_dir)
        self.videos_dir_patcher.start()
        self.clips_dir_patcher.start()

    def tearDown(self) -> None:
        self.clips_dir_patcher.stop()
        self.videos_dir_patcher.stop()
        self.temp_dir.cleanup()

    def test_returns_no_match_when_object_is_absent_but_similarity_is_low(self) -> None:
        with patch("app.api.routes.search.search_video", return_value=[
            SearchResult(frame_filename="frame_1.jpg", timestamp=10.0, score=0.1),
        ]), patch(
            "app.api.routes.search.get_video_by_filename", return_value=type("Video", (), {"id": 1, "filename": "sample.mp4"})()
        ), patch(
            "app.api.routes.search._match_events", return_value=None
        ):
            response = asyncio.run(
                search_video_route(SearchRequest(query="person", video_name="sample.mp4"), db=MagicMock())
            )

        self.assertFalse(response.found)
        self.assertEqual(response.message, "Not found in this video. Try another search.")
        self.assertEqual(response.query, "person")
        self.assertIsNone(response.timestamp)
        self.assertIsNone(response.similarity_score)
        self.assertIsNone(response.frame_name)
        self.assertIsNone(response.clip_path)

    def test_returns_clip_when_object_is_in_events_table(self) -> None:
        mock_event = MagicMock()
        mock_event.label = "person"
        mock_event.start = 2.0
        mock_event.end = 6.0
        mock_event.clip_path = "data/1/clips/person_2.0_6.0.mp4"
        mock_event.max_conf = 0.93

        with patch("app.api.routes.search.search_video") as mock_search, patch(
            "app.api.routes.search.get_video_by_filename", return_value=type("Video", (), {"id": 1, "filename": "sample.mp4"})()
        ), patch(
            "app.api.routes.search._match_events", return_value=[mock_event]
        ):
            response = asyncio.run(
                search_video_route(SearchRequest(query="person", video_name="sample.mp4"), db=MagicMock())
            )

        self.assertTrue(response.found)
        self.assertIn("person", response.message.lower())
        self.assertEqual(response.clip_path, "data/1/clips/person_2.0_6.0.mp4")
        self.assertEqual(response.similarity_score, 0.93)
        # Events-table match should NOT call CLIP search
        mock_search.assert_not_called()

    def test_returns_clip_when_synonym_matches_events_table(self) -> None:
        mock_event = MagicMock()
        mock_event.label = "person"
        mock_event.start = 1.0
        mock_event.end = 4.0
        mock_event.clip_path = "data/1/clips/person_1.0_4.0.mp4"
        mock_event.max_conf = 0.88

        with patch("app.api.routes.search.search_video") as mock_search, patch(
            "app.api.routes.search.get_video_by_filename", return_value=type("Video", (), {"id": 1, "filename": "sample.mp4"})()
        ), patch(
            "app.api.routes.search._match_events", return_value=[mock_event]
        ):
            # Query with synonym "human"
            response = asyncio.run(
                search_video_route(SearchRequest(query="human", video_name="sample.mp4"), db=MagicMock())
            )

        self.assertTrue(response.found)
        self.assertEqual(response.clip_path, "data/1/clips/person_1.0_4.0.mp4")
        mock_search.assert_not_called()

    def test_returns_no_match_when_object_is_absent_from_video(self) -> None:
        with patch("app.api.routes.search.search_video", return_value=[]) as mock_search, patch(
            "app.api.routes.search.get_video_by_filename", return_value=type("Video", (), {"id": 1, "filename": "sample.mp4"})()
        ), patch(
            "app.api.routes.search._match_events", return_value=None
        ):
            response = asyncio.run(
                search_video_route(SearchRequest(query="elephant", video_name="sample.mp4"), db=MagicMock())
            )

        self.assertFalse(response.found)
        self.assertEqual(response.message, "Not found in this video. Try another search.")
        self.assertIsNone(response.clip_path)

    def test_returns_clip_response_when_similarity_meets_threshold(self) -> None:
        with patch("app.api.routes.search.search_video_clip_fallback", return_value=[
            {"start": 8.0, "end": 12.0, "score": 0.8, "representative_frame": "frame_1.jpg"},
        ]), patch(
            "app.api.routes.search.get_video_by_filename", return_value=type("Video", (), {"id": 1, "filename": "sample.mp4"})()
        ), patch(
            "app.api.routes.search._match_events", return_value=None
        ), patch(
            "app.api.routes.search._cut_clip", return_value=None
        ):
            response = asyncio.run(
                search_video_route(SearchRequest(query="sunset in mountains", video_name="sample.mp4"), db=MagicMock())
            )

        self.assertTrue(response.found)
        self.assertEqual(response.query, "sunset in mountains")
        self.assertEqual(response.similarity_score, 0.8)
        self.assertEqual(response.frame_name, "frame_1.jpg")
        self.assertIn("sunset_in_mountains_8.0_12.0.mp4", response.clip_path or "")


if __name__ == "__main__":
    unittest.main()
