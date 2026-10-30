from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status

from app.utils.video_metadata import get_video_metadata

router = APIRouter(prefix="/videos", tags=["videos"])


@router.get("/info", status_code=status.HTTP_200_OK)
async def get_video_info(video_path: str = Query(..., description="Path to the uploaded video file")) -> dict[str, Any]:
    path = Path(video_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    try:
        return get_video_metadata(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
