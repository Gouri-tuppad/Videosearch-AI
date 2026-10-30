from typing import Any
from datetime import date, time

from sqlalchemy.orm import Session

from app.db.detection_models import Detection
from app.db.models import Event, Frame, Video


def create_video(db: Session, filename: str, duration: float, fps: float, upload_date: date, upload_time: time, file_hash: str | None = None, status: str = "uploading") -> Video:
    video = Video(filename=filename, duration=duration, fps=fps, upload_date=upload_date, upload_time=upload_time, file_hash=file_hash, status=status, progress=0.0)
    db.add(video)
    db.commit()
    db.refresh(video)
    return video


def get_video(db: Session, video_id: int) -> Video | None:
    return db.query(Video).filter(Video.id == video_id).first()


def get_video_by_filename(db: Session, filename: str) -> Video | None:
    return db.query(Video).filter(Video.filename == filename).first()


def get_video_by_hash(db: Session, file_hash: str) -> Video | None:
    """Find an already-processed video with this file hash."""
    return db.query(Video).filter(Video.file_hash == file_hash, Video.status == "ready").first()


def list_videos(db: Session, skip: int = 0, limit: int = 100) -> list[Video]:
    return db.query(Video).offset(skip).limit(limit).all()


def update_video(db: Session, video_id: int, **updates: Any) -> Video | None:
    video = get_video(db, video_id)
    if not video:
        return None
    for field, value in updates.items():
        setattr(video, field, value)
    db.commit()
    db.refresh(video)
    return video


def update_video_progress(db: Session, video_id: int, progress: float, status: str | None = None) -> None:
    """Update video processing progress (and optionally status)."""
    video = get_video(db, video_id)
    if video:
        video.progress = progress
        if status:
            video.status = status
        db.commit()


def delete_video(db: Session, video_id: int) -> bool:
    video = get_video(db, video_id)
    if not video:
        return False
    db.delete(video)
    db.commit()
    return True


def create_frame(db: Session, video_id: int, frame_name: str, timestamp: float) -> Frame:
    frame = Frame(video_id=video_id, frame_name=frame_name, timestamp=timestamp)
    db.add(frame)
    db.commit()
    db.refresh(frame)
    return frame


def list_frames_for_video(db: Session, video_id: int, skip: int = 0, limit: int = 100) -> list[Frame]:
    return db.query(Frame).filter(Frame.video_id == video_id).offset(skip).limit(limit).all()


def delete_frames_for_video(db: Session, video_id: int) -> int:
    deleted = db.query(Frame).filter(Frame.video_id == video_id).delete()
    db.commit()
    return deleted


def create_detection(
    db: Session,
    frame_id: int,
    object_name: str,
    confidence: float,
    bbox_xmin: float,
    bbox_ymin: float,
    bbox_xmax: float,
    bbox_ymax: float,
) -> Detection:
    detection = Detection(
        frame_id=frame_id,
        object_name=object_name,
        confidence=confidence,
        bbox_xmin=bbox_xmin,
        bbox_ymin=bbox_ymin,
        bbox_xmax=bbox_xmax,
        bbox_ymax=bbox_ymax,
    )
    db.add(detection)
    db.commit()
    db.refresh(detection)
    return detection


def list_detections_for_frame(db: Session, frame_id: int) -> list[Detection]:
    return db.query(Detection).filter(Detection.frame_id == frame_id).all()


def list_detected_objects_for_video(db: Session, video_id: int) -> list[dict]:
    """Return unique detected object names with their first occurrence timestamp for a video."""
    from sqlalchemy import func, distinct

    results = (
        db.query(
            Detection.object_name,
            func.min(Frame.timestamp).label("first_timestamp"),
            func.count(Detection.id).label("count"),
            func.max(Detection.confidence).label("max_confidence"),
        )
        .join(Frame, Detection.frame_id == Frame.id)
        .filter(Frame.video_id == video_id)
        .group_by(Detection.object_name)
        .order_by(func.count(Detection.id).desc())
        .all()
    )

    return [
        {
            "object_name": row.object_name,
            "first_timestamp": round(float(row.first_timestamp), 3),
            "detection_count": int(row.count),
            "max_confidence": round(float(row.max_confidence), 4),
        }
        for row in results
    ]


def list_detection_frame_names_for_video(db: Session, video_id: int, object_name: str) -> set[str]:
    """Return frame names where a specific object was detected in a video."""
    rows = (
        db.query(Frame.frame_name)
        .join(Detection, Detection.frame_id == Frame.id)
        .filter(Frame.video_id == video_id, Detection.object_name == object_name)
        .distinct()
        .all()
    )
    return {row[0] for row in rows}


# ── Event CRUD ──────────────────────────────────────────────────────────

def create_event(
    db: Session,
    video_id: int,
    label: str,
    start: float,
    end: float,
    clip_path: str | None,
    max_conf: float,
) -> Event:
    event = Event(
        video_id=video_id,
        label=label,
        start=round(start, 3),
        end=round(end, 3),
        clip_path=clip_path,
        max_conf=round(max_conf, 4),
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def bulk_create_events(db: Session, events: list[dict]) -> list[Event]:
    """Create multiple Event rows in a single commit."""
    objs = []
    for ev in events:
        objs.append(Event(
            video_id=ev["video_id"],
            label=ev["label"],
            start=round(ev["start"], 3),
            end=round(ev["end"], 3),
            clip_path=ev.get("clip_path"),
            max_conf=round(ev["max_conf"], 4),
        ))
    db.add_all(objs)
    db.commit()
    for o in objs:
        db.refresh(o)
    return objs


def list_events_for_video(db: Session, video_id: int) -> list[Event]:
    return db.query(Event).filter(Event.video_id == video_id).order_by(Event.label, Event.start).all()


def list_events_by_label(db: Session, video_id: int, label: str) -> list[Event]:
    return (
        db.query(Event)
        .filter(Event.video_id == video_id, Event.label == label)
        .order_by(Event.start)
        .all()
    )


def search_events_by_labels(db: Session, video_id: int, labels: list[str]) -> list[Event]:
    """Find events whose label matches any of the given labels (case-insensitive)."""
    from sqlalchemy import func
    lower_labels = [l.lower() for l in labels]
    return (
        db.query(Event)
        .filter(
            Event.video_id == video_id,
            func.lower(Event.label).in_(lower_labels),
        )
        .order_by(Event.start)
        .all()
    )


def delete_events_for_video(db: Session, video_id: int) -> int:
    deleted = db.query(Event).filter(Event.video_id == video_id).delete()
    db.commit()
    return deleted


def get_distinct_event_labels_for_video(db: Session, video_id: int, limit: int = 5) -> list[str]:
    """Return up to `limit` distinct event labels present in the video for suggestions."""
    if not hasattr(db, "query"):
        return []
    from sqlalchemy import distinct
    try:
        labels = [
            row[0]
            for row in db.query(distinct(Event.label))
            .filter(Event.video_id == video_id)
            .order_by(Event.label)
            .limit(limit)
            .all()
            if row[0]
        ]
        if len(labels) < limit:
            from app.db.detection_models import Detection
            from app.db.models import Frame
            det_rows = (
                db.query(distinct(Detection.object_name))
                .join(Frame, Detection.frame_id == Frame.id)
                .filter(Frame.video_id == video_id)
                .limit(limit - len(labels))
                .all()
            )
            for r in det_rows:
                if r[0] and r[0] not in labels:
                    labels.append(r[0])
        return labels[:limit]
    except Exception:
        return []
