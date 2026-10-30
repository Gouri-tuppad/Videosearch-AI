import unittest
from unittest.mock import patch

from app.api.routes.search import _group_matches
from app.core.config import settings
from app.search_service import SearchResult


class TemporalGroupingTests(unittest.TestCase):
    def test_groups_nearby_frames_into_occurrences(self) -> None:
        matches = [
            SearchResult("frame_001.jpg", 1.0, 0.61, "video.mp4"),
            SearchResult("frame_002.jpg", 1.5, 0.72, "video.mp4"),
            SearchResult("frame_003.jpg", 2.0, 0.76, "video.mp4"),
            SearchResult("frame_004.jpg", 2.5, 0.70, "video.mp4"),
            SearchResult("frame_010.jpg", 5.0, 0.68, "video.mp4"),
            SearchResult("frame_011.jpg", 5.5, 0.73, "video.mp4"),
        ]

        with patch.object(settings, "frame_extraction_interval", 0.5):
            segments = _group_matches(matches)

        self.assertEqual(len(segments), 2)
        self.assertEqual({(segment["start_time"], segment["end_time"]) for segment in segments}, {(1.0, 2.5), (5.0, 5.5)})
        self.assertEqual(max(segment["best_similarity_score"] for segment in segments), 0.76)


if __name__ == "__main__":
    unittest.main()