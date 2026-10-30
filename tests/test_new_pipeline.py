from datetime import date, datetime
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from io import BytesIO
from pathlib import Path
import tempfile
import numpy as np

from app.main import app
from app.pipeline import _track_events, compute_file_hash
from app.search_service import expand_synonyms
from app.db.models import Event, Video
from app.db.database import SessionLocal, Base, engine


class NewPipelineUnitTests(unittest.TestCase):
    def setUp(self):
        session = SessionLocal()
        try:
            for fn in ["test_clips_video.mp4", "test_search_video.mp4"]:
                vid = session.query(Video).filter(Video.filename == fn).first()
                if vid:
                    session.query(Event).filter(Event.video_id == vid.id).delete()
                    session.delete(vid)
            session.commit()
        finally:
            session.close()

    def tearDown(self):
        session = SessionLocal()
        try:
            for fn in ["test_clips_video.mp4", "test_search_video.mp4"]:
                vid = session.query(Video).filter(Video.filename == fn).first()
                if vid:
                    session.query(Event).filter(Event.video_id == vid.id).delete()
                    session.delete(vid)
            session.commit()
        finally:
            session.close()

    def test_event_tracking_open_close_gap_rules(self):
        # Rule: open after >= 2 consecutive detections
        # Rule: allow 2s gap
        # Rule: close when absent > 2s or length > 30s
        frames = [
            (0.0, [{"label": "person", "confidence": 0.9}]),
            (1.0, [{"label": "person", "confidence": 0.95}]),  # opened at consecutive=2
            (2.0, [{"label": "person", "confidence": 0.85}]),
            (3.0, []),  # 1s gap (allowed)
            (4.0, [{"label": "person", "confidence": 0.92}]),  # still open
            (5.0, []),
            (6.0, []),
            (7.0, []),  # gap > 2s -> closed at t=4.0
        ]
        events = _track_events(frames, min_consecutive=2, gap_tolerance=2.0, max_length=30.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["label"], "person")
        self.assertEqual(events[0]["start"], 0.0)
        self.assertEqual(events[0]["end"], 4.0)
        self.assertEqual(events[0]["max_conf"], 0.95)

    def test_event_tracking_single_detection_not_opened(self):
        frames = [
            (0.0, [{"label": "car", "confidence": 0.8}]),
            (1.0, []),
            (2.0, []),
            (3.0, []),
        ]
        events = _track_events(frames, min_consecutive=2, gap_tolerance=2.0, max_length=30.0)
        self.assertEqual(len(events), 0)

    def test_event_tracking_max_length(self):
        frames = [(float(t), [{"label": "dog", "confidence": 0.9}]) for t in range(35)]
        events = _track_events(frames, min_consecutive=2, max_length=30.0)
        self.assertGreaterEqual(len(events), 2)
        for ev in events:
            self.assertLessEqual(ev["end"] - ev["start"], 31.0)

    def test_synonym_expansion(self):
        syns = expand_synonyms("puppy")
        self.assertIn("dog", syns)
        self.assertIn("puppy", syns)

        syns_car = expand_synonyms("automobile")
        self.assertIn("car", syns_car)

        syns_human = expand_synonyms("someone")
        self.assertIn("person", syns_human)

    def test_get_clips_endpoint(self):
        session = SessionLocal()
        try:
            # Create a test video and events
            video = Video(
                filename="test_clips_video.mp4",
                duration=10.0,
                fps=1.0,
                status="ready",
                progress=100.0,
                upload_date=date.today(),
                upload_time=datetime.now().time(),
            )
            session.add(video)
            session.commit()
            session.refresh(video)
            vid_id = video.id

            ev1 = Event(
                video_id=vid_id,
                label="person",
                start=0.0,
                end=3.0,
                clip_path=f"data/{vid_id}/clips/person_0.0_3.0.mp4",
                max_conf=0.95,
            )
            ev2 = Event(
                video_id=vid_id,
                label="person",
                start=5.0,
                end=8.0,
                clip_path=f"data/{vid_id}/clips/person_5.0_8.0.mp4",
                max_conf=0.88,
            )
            ev3 = Event(
                video_id=vid_id,
                label="car",
                start=1.0,
                end=4.0,
                clip_path=f"data/{vid_id}/clips/car_1.0_4.0.mp4",
                max_conf=0.91,
            )
            session.add_all([ev1, ev2, ev3])
            session.commit()
        finally:
            session.close()

        # Test GET /clips/{video_id}
        client = TestClient(app)
        resp = client.get(f"/clips/{vid_id}")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["video_id"], vid_id)
        self.assertEqual(data["total_events"], 3)
        self.assertIn("person", data["clips"])
        self.assertIn("car", data["clips"])
        self.assertEqual(len(data["clips"]["person"]), 2)
        self.assertEqual(len(data["clips"]["car"]), 1)
        self.assertEqual(data["clips"]["person"][0]["clip_path"], f"data/{vid_id}/clips/person_0.0_3.0.mp4")

    def test_search_events_first_with_synonyms(self):
        session = SessionLocal()
        try:
            video = Video(
                filename="test_search_video.mp4",
                duration=10.0,
                fps=1.0,
                status="ready",
                progress=100.0,
                upload_date=date.today(),
                upload_time=datetime.now().time(),
            )
            session.add(video)
            session.commit()
            session.refresh(video)
            vid_id = video.id

            ev = Event(
                video_id=vid_id,
                label="dog",
                start=2.0,
                end=6.0,
                clip_path=f"data/{vid_id}/clips/dog_2.0_6.0.mp4",
                max_conf=0.94,
            )
            session.add(ev)
            session.commit()
        finally:
            session.close()

        # Query with synonym "puppy" -> should hit events table without CLIP
        client = TestClient(app)
        resp = client.post("/search", json={"query": "puppy", "video_name": "test_search_video.mp4"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["found"])
        self.assertEqual(data["clip_path"], f"data/{vid_id}/clips/dog_2.0_6.0.mp4")
        self.assertIn("dog", data["message"].lower())

    def test_upload_dedup_by_hash(self):
        client = TestClient(app)
        session = SessionLocal()
        try:
            fake_content = b"fake video bytes for hash test"
            import hashlib
            file_hash = hashlib.sha256(fake_content).hexdigest()

            # Video already exists and is ready
            video = Video(
                filename="existing_hash_video.mp4",
                duration=10.0,
                fps=1.0,
                status="ready",
                progress=100.0,
                file_hash=file_hash,
                upload_date=date.today(),
                upload_time=datetime.now().time(),
            )
            session.add(video)
            session.commit()
            vid_id = video.id
        finally:
            session.close()

        # Upload file with identical content -> should skip reprocessing
        resp = client.post(
            "/videos/upload",
            files={"file": ("new_name.mp4", BytesIO(fake_content), "video/mp4")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["video_id"], vid_id)
        self.assertIn("already processed", data["message"].lower())
        self.assertEqual(data["processing_status"], "ready")

        # Cleanup
        s = SessionLocal()
        try:
            v = s.query(Video).filter(Video.id == vid_id).first()
            if v:
                s.delete(v)
            s.commit()
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
