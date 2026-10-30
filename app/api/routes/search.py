"""Search API endpoints — Events-first with synonyms, CLIP+FAISS fallback, suggestions on not_found."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from app.clip_generator import extract_clip
from app.core.config import settings
from app.api.routes.detections import get_db_session
from app.db.crud import (
    get_distinct_event_labels_for_video,
    get_video,
    get_video_by_filename,
    list_detected_objects_for_video,
    list_detection_frame_names_for_video,
    list_events_for_video,
    search_events_by_labels,
)
from app.faiss_index import INDEX_DIR, IndexNotFoundError, IndexCorruptedError
from app.pipeline import _cut_clip, DATA_DIR
from app.search_service import (
    SearchResult,
    search_video,
    expand_synonyms,
    format_clip_query_prompt,
    search_video_clip_fallback,
)

VIDEOS_DIR = Path(settings.upload_folder)
CLIPS_DIR = Path(settings.clips_folder)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    video_name: str | None = Field(
        None,
        description="Optional video name or video ID",
    )

    @validator("query")
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("query must be a non-empty string")
        return value.strip()


class SearchResultItem(BaseModel):
    label: str
    start: float
    end: float
    clip_url: str
    score: float
    # Backward compatibility
    start_time: float | None = None
    end_time: float | None = None
    best_similarity_score: float | None = None
    clip_path: str | None = None
    representative_frame: str | None = None
    video_name: str | None = None


class SearchResponse(BaseModel):
    status: str = Field(..., description="Either 'found' or 'not_found'")
    query: str = Field(..., description="The original search query")
    results: List[SearchResultItem] = Field(default_factory=list, description="Matching video clip segments")
    message: str | None = Field(None, description="Human-readable result message")
    suggestions: List[str] = Field(default_factory=list, description="Suggested labels from this video")
    # Backward compatibility fields
    found: bool = False
    video_name: str | None = None
    threshold: float | None = None
    timestamp: str | None = None
    similarity_score: float | None = None
    frame_name: str | None = None
    clip_path: str | None = None
    best_similarity_score: float | None = None
    error: str | None = None


def _select_video_name(video_name: str | None) -> str:
    if video_name:
        return video_name

    if not INDEX_DIR.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No FAISS indexes found on the server.",
        )

    index_files = [path for path in INDEX_DIR.glob("*.index") if path.is_file()]
    if not index_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No FAISS indexes found on the server.",
        )
    if len(index_files) > 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Multiple FAISS indexes are available. Please provide video_name in the request body.",
        )

    return index_files[0].stem


def _format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _relative_clip_path(absolute_path: str) -> str:
    """Return clip path in 'clips/<filename>' format for the static file mount."""
    clip_path = Path(absolute_path)
    return f"clips/{clip_path.name}"


def _group_matches(results: List[SearchResult]) -> List[Dict[str, Any]]:
    """Group thresholded CLIP hits that belong to one temporal occurrence."""
    if not results:
        return []

    gap_limit = max(float(settings.frame_extraction_interval) * 1.5, 0.1)
    ordered = sorted(results, key=lambda item: item.timestamp)
    groups: List[List[SearchResult]] = [[ordered[0]]]

    for result in ordered[1:]:
        if result.timestamp - groups[-1][-1].timestamp <= gap_limit:
            groups[-1].append(result)
        else:
            groups.append([result])

    segments = []
    for group in groups:
        representative = max(group, key=lambda item: item.score)
        segments.append(
            {
                "start_time": round(group[0].timestamp, 3),
                "end_time": round(group[-1].timestamp, 3),
                "best_similarity_score": round(representative.score, 4),
                "representative_frame": representative.frame_filename,
                "video_name": representative.video_name,
            }
        )

    return sorted(segments, key=lambda item: item["best_similarity_score"], reverse=True)


def _not_found_response(
    video_name: str,
    query: str,
    suggestions: list[str] | None = None,
    best_similarity_score: float | None = None,
) -> SearchResponse:
    suggests = suggestions if suggestions is not None else []
    logger.info("Search path used: not_found for query='%s' (suggestions=%s)", query, suggests)
    return SearchResponse(
        status="not_found",
        found=False,
        video_name=video_name,
        query=query,
        threshold=float(settings.similarity_threshold),
        results=[],
        clip_path=None,
        message="Not found in this video. Try another search.",
        suggestions=suggests,
        best_similarity_score=best_similarity_score,
    )


# ── Step 1: Events-first search ────────────────────────────────────────

def _match_events(query: str, video_id: int, db: Session) -> list[Any] | None:
    """Try to find matching events by label (with synonyms).

    Returns a list of Event rows or None if no label matched.
    """
    if not hasattr(db, "query"):
        return None

    candidate_labels = expand_synonyms(query)
    if not candidate_labels:
        return None

    try:
        events = search_events_by_labels(db, video_id, candidate_labels)
        return events if events else None
    except Exception as exc:
        logger.warning("Error searching events by labels: %s", exc)
        return None


# ── Step 2: CLIP fallback ───────────────────────────────────────────────

def _clip_fallback_search(
    video_name: str,
    query: str,
    video_id: int,
    db: Session,
) -> SearchResponse:
    """Fall back to CLIP semantic search when no event label matched.

    - Only keeps hits with similarity >= CLIP_MIN_SCORE (default 0.25)
    - Uses text prompt 'a photo of {query}'
    - Requires top_score - mean_score >= 0.03
    - Merges adjacent hits (gap <= 2s) and returns top 3 segments
    - Cuts clips if they don't exist yet
    """
    try:
        segments = search_video_clip_fallback(
            video_name=video_name,
            query=query,
            clip_min_score=float(settings.clip_min_score),
            clip_score_margin=float(settings.clip_score_margin),
            gap_tolerance=2.0,
        )
    except Exception as exc:
        logger.exception("CLIP fallback search error for video=%s query=%s: %s", video_name, query, exc)
        segments = []

    if not segments:
        suggestions = get_distinct_event_labels_for_video(db, video_id, limit=5)
        return _not_found_response(video_name, query, suggestions=suggestions)

    # Cut clips for merged segments if they don't exist yet
    video_record = get_video(db, video_id) if hasattr(db, "query") else None
    duration = float(video_record.duration) if video_record and video_record.duration else 3600.0
    video_path = VIDEOS_DIR / video_name
    clips_dir = DATA_DIR / str(video_id) / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    results_items: list[SearchResultItem] = []
    safe_query = re.sub(r"[^a-zA-Z0-9_-]+", "_", query.strip().lower())

    for seg in segments:
        clip_filename = f"{safe_query}_{seg['start']:.1f}_{seg['end']:.1f}.mp4"
        clip_file = clips_dir / clip_filename
        if not clip_file.exists() or clip_file.stat().st_size == 0:
            if video_path.exists():
                try:
                    _cut_clip(
                        video_path=video_path,
                        label=safe_query,
                        start=seg["start"],
                        end=seg["end"],
                        padding=float(settings.clip_padding_seconds),
                        output_dir=clips_dir,
                        duration=duration,
                    )
                except Exception as exc:
                    logger.warning("Failed to cut clip for fallback segment: %s", exc)

        clip_url = f"data/{video_id}/clips/{clip_filename}"
        results_items.append(
            SearchResultItem(
                label=query,
                start=seg["start"],
                end=seg["end"],
                clip_url=clip_url,
                score=round(seg["score"], 4),
                start_time=seg["start"],
                end_time=seg["end"],
                best_similarity_score=round(seg["score"], 4),
                clip_path=clip_url,
                representative_frame=seg.get("representative_frame", ""),
                video_name=video_name,
            )
        )

    logger.info("Search path used: clip_fallback for query='%s' (matches=%d)", query, len(results_items))
    top = results_items[0]
    return SearchResponse(
        status="found",
        found=True,
        video_name=video_name,
        query=query,
        threshold=float(settings.clip_min_score),
        results=results_items,
        timestamp=f"{_format_timestamp(top.start)} - {_format_timestamp(top.end)}",
        similarity_score=top.score,
        frame_name=top.representative_frame,
        clip_path=top.clip_url,
        message="Matching segments found via CLIP semantic search.",
        best_similarity_score=top.score,
    )


# ── Main endpoint ──────────────────────────────────────────────────────

@router.post("", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search_video_route(
    request: SearchRequest,
    db: Session = Depends(get_db_session),
) -> SearchResponse:
    logger.info("Search requested: query=%s video_name=%s", request.query, request.video_name)

    video = None
    if request.video_name and request.video_name.isdigit():
        video = get_video(db, int(request.video_name)) if hasattr(db, "query") else None
    if not video:
        video_name = _select_video_name(request.video_name)
        video = get_video_by_filename(db, video_name) if hasattr(db, "query") else None
    else:
        video_name = video.filename

    if not video:
        # Check if index exists for this video_name
        video_id = 0
        suggestions = []
        if hasattr(db, "query"):
            suggestions = get_distinct_event_labels_for_video(db, video_id, limit=5)
        return _not_found_response(video_name, request.query, suggestions=suggestions)

    video_id = video.id

    # ── Step 1: events-table lookup (with synonyms) ────────────────
    matched_events = _match_events(request.query, video_id, db)

    if matched_events:
        logger.info("Search path used: events for query='%s' (matches=%d)", request.query, len(matched_events))
        response_results: list[SearchResultItem] = []
        for ev in matched_events:
            clip_path = ev.clip_path or f"data/{video_id}/clips/{ev.label}_{ev.start:.1f}_{ev.end:.1f}.mp4"
            response_results.append(
                SearchResultItem(
                    label=ev.label,
                    start=round(ev.start, 3),
                    end=round(ev.end, 3),
                    clip_url=clip_path,
                    score=round(ev.max_conf, 4),
                    start_time=round(ev.start, 3),
                    end_time=round(ev.end, 3),
                    best_similarity_score=round(ev.max_conf, 4),
                    clip_path=clip_path,
                    video_name=video_name,
                )
            )

        response_results.sort(key=lambda r: r.score, reverse=True)
        best_ev = response_results[0]
        return SearchResponse(
            status="found",
            found=True,
            video_name=video_name,
            query=request.query,
            threshold=float(settings.similarity_threshold),
            results=response_results,
            timestamp=f"{_format_timestamp(best_ev.start)} - {_format_timestamp(best_ev.end)}",
            similarity_score=best_ev.score,
            frame_name=None,
            clip_path=best_ev.clip_url,
            message=f"Found '{best_ev.label}' events in this video.",
            best_similarity_score=best_ev.score,
        )

    # ── Step 2: CLIP semantic fallback ─────────────────────────────
    return _clip_fallback_search(video_name, request.query, video_id, db)


@router.get("/{video_id}", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search_video_get_route(
    video_id: str,
    q: str,
    db: Session = Depends(get_db_session),
) -> SearchResponse:
    """GET /search/{video_id}?q=... endpoint."""
    video = None
    if video_id.isdigit():
        video = get_video(db, int(video_id)) if hasattr(db, "query") else None
    if not video:
        video = get_video_by_filename(db, video_id) if hasattr(db, "query") else None
    video_name = video.filename if video else video_id
    return await search_video_route(SearchRequest(query=q, video_name=video_name), db=db)
