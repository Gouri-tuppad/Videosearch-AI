"""GET /clips/{video_id} — return pre-generated clips grouped by label."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.crud import get_video, get_video_by_filename, list_events_for_video
from app.db.session import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/clips", tags=["clips"])


@router.get("/{video_id}", status_code=status.HTTP_200_OK)
async def get_clips(video_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return all pre-generated clips for a video, grouped by label."""
    video = None
    if video_id.isdigit():
        video = get_video(db, int(video_id))
    if not video:
        video = get_video_by_filename(db, video_id)
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

    events = list_events_for_video(db, video.id)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        grouped[ev.label].append({
            "id": ev.id,
            "label": ev.label,
            "start": ev.start,
            "end": ev.end,
            "clip_path": ev.clip_path,
            "max_conf": ev.max_conf,
        })

    res: dict[str, Any] = {
        "video_id": video.id,
        "filename": video.filename,
        "status": video.status,
        "total_events": len(events),
        "clips": dict(grouped),
        "labels": dict(grouped),
    }
    # Unpack grouped labels at root level for direct key access
    for k, v in grouped.items():
        if k not in res:
            res[k] = v

    return res
