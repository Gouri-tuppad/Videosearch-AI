"""Pytest tests for search behavior:
1. Query matching an event returns pre-cut clips with status "found" (logged path: events).
2. Nonsense query ("unicorn") returns status "not_found" with suggestions (logged path: not_found).
3. Synonym query ("puppy") expands and returns dog clips with status "found" (logged path: events).
"""
import logging
from datetime import date, datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.db.database import SessionLocal
from app.db.models import Event, Video
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def test_video_and_events():
    session = SessionLocal()
    video_filename = "test_behavior_video.mp4"
    # Clean up any leftover records
    existing = session.query(Video).filter(Video.filename == video_filename).first()
    if existing:
        session.query(Event).filter(Event.video_id == existing.id).delete()
        session.delete(existing)
        session.commit()

    video = Video(
        filename=video_filename,
        duration=25.0,
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

    dog_event = Event(
        video_id=vid_id,
        label="dog",
        start=2.0,
        end=7.0,
        clip_path=f"data/{vid_id}/clips/dog_2.0_7.0.mp4",
        max_conf=0.92,
    )
    car_event = Event(
        video_id=vid_id,
        label="car",
        start=10.0,
        end=15.0,
        clip_path=f"data/{vid_id}/clips/car_10.0_15.0.mp4",
        max_conf=0.88,
    )
    session.add_all([dog_event, car_event])
    session.commit()
    session.close()

    yield {"video_id": vid_id, "filename": video_filename}

    # Teardown
    cleanup_session = SessionLocal()
    vid = cleanup_session.query(Video).filter(Video.id == vid_id).first()
    if vid:
        cleanup_session.query(Event).filter(Event.video_id == vid.id).delete()
        cleanup_session.delete(vid)
        cleanup_session.commit()
    cleanup_session.close()


def test_query_matching_event_returns_clips(client, test_video_and_events, caplog):
    """A query matching an event returns pre-cut clips with status 'found'."""
    vid_id = test_video_and_events["video_id"]
    filename = test_video_and_events["filename"]

    with caplog.at_level(logging.INFO):
        response = client.post(
            "/search",
            json={"query": "dog", "video_name": filename},
        )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "found"
    assert data["query"] == "dog"
    assert len(data["results"]) >= 1

    clip_result = data["results"][0]
    assert clip_result["label"] == "dog"
    assert clip_result["start"] == 2.0
    assert clip_result["end"] == 7.0
    assert clip_result["clip_url"] == f"data/{vid_id}/clips/dog_2.0_7.0.mp4"
    assert clip_result["score"] == 0.92

    # Verify search path logged
    assert "Search path used: events" in caplog.text


def test_nonsense_query_returns_not_found(client, test_video_and_events, caplog):
    """A nonsense query ('unicorn') returns status 'not_found' with suggestions."""
    vid_id = test_video_and_events["video_id"]
    filename = test_video_and_events["filename"]

    # In case CLIP fallback executes, mock it to simulate rejection of nonsense query
    with caplog.at_level(logging.INFO), patch(
        "app.api.routes.search.search_video_clip_fallback", return_value=[]
    ):
        response = client.post(
            "/search",
            json={"query": "unicorn", "video_name": filename},
        )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "not_found"
    assert data["query"] == "unicorn"
    assert data["message"] == "Not found in this video. Try another search."
    assert data["results"] == []
    assert isinstance(data["suggestions"], list)
    assert "dog" in data["suggestions"]
    assert "car" in data["suggestions"]

    # Verify search path logged
    assert "Search path used: not_found" in caplog.text


def test_synonym_query_returns_dog_clips(client, test_video_and_events, caplog):
    """A synonym query ('puppy') expands and returns the dog clips with status 'found'."""
    vid_id = test_video_and_events["video_id"]
    filename = test_video_and_events["filename"]

    with caplog.at_level(logging.INFO):
        response = client.post(
            "/search",
            json={"query": "puppy", "video_name": filename},
        )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "found"
    assert data["query"] == "puppy"
    assert len(data["results"]) >= 1

    clip_result = data["results"][0]
    assert clip_result["label"] == "dog"
    assert clip_result["clip_url"] == f"data/{vid_id}/clips/dog_2.0_7.0.mp4"
    assert clip_result["score"] == 0.92

    # Verify search path logged
    assert "Search path used: events" in caplog.text

