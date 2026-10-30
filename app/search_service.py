from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np
import torch

from app.core.clip_loader import load_clip_model
from app.core.config import settings
from app.faiss_index import load_index, search_index

logger = logging.getLogger(__name__)

_DEFAULT_DEVICE = "cpu"

if TYPE_CHECKING:
    from transformers import CLIPModel, CLIPProcessor


@dataclass(frozen=True)
class SearchResult:
    frame_filename: str
    timestamp: float
    score: float
    video_name: str = ""


def _load_clip_text_encoder(device: str = _DEFAULT_DEVICE) -> tuple[Any, Any]:
    return load_clip_model(device)


def ensure_clip_text_encoder_loaded(device: str = _DEFAULT_DEVICE) -> None:
    _load_clip_text_encoder(device)


def _validate_query(query: str) -> str:
    if not isinstance(query, str):
        raise ValueError("Search query must be a string")
    normalized = query.strip()
    if not normalized:
        raise ValueError("Search query cannot be empty")
    return normalized


# ── Synonym expansion ──────────────────────────────────────────────────

# Mapping from common query terms → YOLO / COCO label synonyms.
# Keys are user-facing words; values include the canonical COCO label
# plus common alternatives so that "person" also matches queries for
# "human", "man", "woman", etc.
_SYNONYM_MAP: dict[str, list[str]] = {
    # People
    "person": ["person", "human", "man", "woman", "people", "pedestrian", "boy", "girl", "child", "kid"],
    "human": ["person", "human", "man", "woman", "people"],
    "man": ["person", "man"],
    "woman": ["person", "woman"],
    "people": ["person", "people"],
    "pedestrian": ["person", "pedestrian"],
    "boy": ["person", "boy"],
    "girl": ["person", "girl"],
    "child": ["person", "child", "kid"],
    "kid": ["person", "child", "kid"],
    "someone": ["person"],
    "somebody": ["person"],
    # Vehicles
    "car": ["car", "automobile", "vehicle"],
    "automobile": ["car", "automobile"],
    "vehicle": ["car", "truck", "bus", "vehicle"],
    "truck": ["truck", "lorry"],
    "lorry": ["truck", "lorry"],
    "bus": ["bus", "coach"],
    "bike": ["bicycle", "motorcycle", "bike"],
    "bicycle": ["bicycle", "bike"],
    "motorcycle": ["motorcycle", "motorbike", "bike"],
    "motorbike": ["motorcycle", "motorbike"],
    "airplane": ["airplane", "aeroplane", "plane", "aircraft", "jet"],
    "plane": ["airplane", "aeroplane", "plane", "aircraft"],
    "boat": ["boat", "ship", "vessel", "yacht"],
    "ship": ["boat", "ship"],
    "train": ["train", "locomotive"],
    # Animals
    "dog": ["dog", "puppy", "canine"],
    "puppy": ["dog", "puppy"],
    "cat": ["cat", "kitten", "feline"],
    "kitten": ["cat", "kitten"],
    "bird": ["bird"],
    "horse": ["horse", "pony"],
    "cow": ["cow", "cattle"],
    "sheep": ["sheep", "lamb"],
    "elephant": ["elephant"],
    "bear": ["bear"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    # Objects
    "phone": ["cell phone", "phone", "mobile", "smartphone"],
    "mobile": ["cell phone", "phone", "mobile"],
    "smartphone": ["cell phone", "smartphone"],
    "cell phone": ["cell phone", "phone", "mobile"],
    "laptop": ["laptop", "computer", "notebook"],
    "computer": ["laptop", "computer"],
    "tv": ["tv", "television", "monitor", "screen"],
    "television": ["tv", "television"],
    "monitor": ["tv", "monitor"],
    "bottle": ["bottle"],
    "cup": ["cup", "mug", "glass"],
    "mug": ["cup", "mug"],
    "glass": ["cup", "glass", "wine glass"],
    "chair": ["chair", "seat"],
    "couch": ["couch", "sofa"],
    "sofa": ["couch", "sofa"],
    "bed": ["bed"],
    "table": ["dining table", "table", "desk"],
    "desk": ["dining table", "table", "desk"],
    "book": ["book"],
    "clock": ["clock", "watch"],
    "watch": ["clock", "watch"],
    "knife": ["knife"],
    "fork": ["fork"],
    "spoon": ["spoon"],
    "bowl": ["bowl"],
    "banana": ["banana"],
    "apple": ["apple"],
    "orange": ["orange"],
    "sandwich": ["sandwich"],
    "pizza": ["pizza"],
    "cake": ["cake"],
    "donut": ["donut", "doughnut"],
    "doughnut": ["donut", "doughnut"],
    "umbrella": ["umbrella"],
    "handbag": ["handbag", "bag", "purse"],
    "bag": ["handbag", "backpack", "bag", "suitcase"],
    "backpack": ["backpack", "bag"],
    "suitcase": ["suitcase", "luggage", "bag"],
    "luggage": ["suitcase", "luggage"],
    "tie": ["tie", "necktie"],
    "necktie": ["tie", "necktie"],
    # Sports
    "ball": ["sports ball", "ball", "frisbee"],
    "sports ball": ["sports ball", "ball"],
    "frisbee": ["frisbee"],
    "skateboard": ["skateboard"],
    "surfboard": ["surfboard"],
    "tennis racket": ["tennis racket", "racket"],
    "racket": ["tennis racket", "racket"],
    "baseball bat": ["baseball bat", "bat"],
    "bat": ["baseball bat", "bat"],
    "kite": ["kite"],
    # Kitchen / food
    "oven": ["oven", "stove"],
    "stove": ["oven", "stove"],
    "microwave": ["microwave"],
    "toaster": ["toaster"],
    "sink": ["sink"],
    "refrigerator": ["refrigerator", "fridge"],
    "fridge": ["refrigerator", "fridge"],
    # Other
    "traffic light": ["traffic light", "signal"],
    "stop sign": ["stop sign"],
    "fire hydrant": ["fire hydrant", "hydrant"],
    "hydrant": ["fire hydrant", "hydrant"],
    "bench": ["bench"],
    "potted plant": ["potted plant", "plant", "houseplant"],
    "plant": ["potted plant", "plant"],
    "toilet": ["toilet"],
    "remote": ["remote", "remote control"],
    "keyboard": ["keyboard"],
    "mouse": ["mouse"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy bear", "stuffed animal"],
    "toothbrush": ["toothbrush"],
    "hair drier": ["hair drier", "hair dryer"],
    "vase": ["vase"],
}


def expand_synonyms(query: str) -> list[str]:
    """Expand a natural-language query into candidate YOLO/COCO labels.

    Returns a list of possible labels (always includes the original words).
    """
    normalized = re.sub(r"[^a-z0-9 ]+", " ", query.casefold()).strip()
    words = normalized.split()
    candidates: set[str] = set()

    # Try multi-word matches first (longest match wins)
    for n in range(min(4, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            if phrase in _SYNONYM_MAP:
                candidates.update(_SYNONYM_MAP[phrase])

    # Then single-word fallback
    for w in words:
        if w in _SYNONYM_MAP:
            candidates.update(_SYNONYM_MAP[w])
        else:
            # Add the raw word as-is so it can match labels directly
            candidates.add(w)

    # Also add the full normalized query
    candidates.add(normalized)

    return list(candidates)


def format_clip_query_prompt(query: str) -> str:
    """Format text prompt as 'a photo of {query}' unless already formatted."""
    cleaned = query.strip()
    lower = cleaned.lower()
    if lower.startswith("a photo of ") or lower.startswith("a picture of ") or lower.startswith("a video of "):
        return cleaned
    return f"a photo of {cleaned}"


# ── CLIP text encoding ─────────────────────────────────────────────────

def encode_text_query(query: str, device: str = _DEFAULT_DEVICE) -> np.ndarray:
    """Convert a natural language query into a CLIP text embedding."""
    valid_query = _validate_query(query)
    model, processor = _load_clip_text_encoder(device)

    # Log model identity for traceability
    model_name = getattr(getattr(model, "config", None), "name_or_path", None) or str(settings.clip_model_name)
    logger.info("Encoding text query using CLIP model: %s", model_name)

    inputs = processor(text=[valid_query], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.get_text_features(**inputs)

    # outputs may be a Tensor or a model output object with attributes
    embedding_tensor: torch.Tensor | None = None
    if isinstance(outputs, torch.Tensor):
        embedding_tensor = outputs
    else:
        if hasattr(outputs, "text_embeds"):
            embedding_tensor = getattr(outputs, "text_embeds")
        elif hasattr(outputs, "pooler_output"):
            embedding_tensor = getattr(outputs, "pooler_output")
        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
            embedding_tensor = outputs[0]
        elif isinstance(outputs, dict):
            if "text_embeds" in outputs and isinstance(outputs["text_embeds"], torch.Tensor):
                embedding_tensor = outputs["text_embeds"]
            elif "pooler_output" in outputs and isinstance(outputs["pooler_output"], torch.Tensor):
                embedding_tensor = outputs["pooler_output"]

    if embedding_tensor is None:
        logger.exception("Unable to extract text embedding tensor from CLIP model output for query: %s", valid_query)
        raise TypeError("Unable to extract text embedding tensor from CLIP model output")

    if embedding_tensor.dim() == 1:
        embedding_tensor = embedding_tensor.unsqueeze(0)

    embedding = embedding_tensor.cpu().numpy().astype(np.float32).reshape(-1)
    logger.info("Produced text embedding shape=%s dtype=%s for query=%s", embedding.shape, embedding.dtype, valid_query)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm
    return embedding

def search_video(
    video_name: str,
    query: str,
    top_k: int = 5,
    similarity_threshold: float | None = None,
    device: str = _DEFAULT_DEVICE,
) -> List[SearchResult]:
    """Search a video using a text query and return matching frame metadata."""

    _validate_query(query)

    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    if similarity_threshold is None:
        similarity_threshold = float(settings.similarity_threshold)

    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")

    logger.info(
        "Search requested: video=%s query=%s top_k=%d threshold=%.3f",
        video_name,
        query,
        top_k,
        similarity_threshold,
    )

    prompt = format_clip_query_prompt(query)
    query_embedding = encode_text_query(prompt, device=device)

    index, _ = load_index(video_name)
    results = search_index(
        video_name,
        query_embedding,
        top_k=max(top_k, int(index.ntotal)),
    )

    raw_scores = [float(result.get("score", 0.0)) for result in results]
    if raw_scores:
        logger.info(
            "Search FAISS raw scores for video=%s query=%s scores=%s",
            video_name,
            query,
            raw_scores,
        )
    else:
        logger.info(
            "Search FAISS returned no raw candidates for video=%s query=%s",
            video_name,
            query,
        )

    logger.info(
        "Search summary before filtering: video=%s query=%s top_k=%d threshold=%.4f raw_count=%d max_score=%.4f min_score=%.4f",
        video_name,
        query,
        top_k,
        similarity_threshold,
        len(raw_scores),
        max(raw_scores) if raw_scores else 0.0,
        min(raw_scores) if raw_scores else 0.0,
    )

    filtered: List[SearchResult] = []

    for result in results:
        score = float(result.get("score", 0.0))

        logger.info(
            "Candidate: frame=%s timestamp=%.3f score=%.4f threshold=%.4f",
            result.get("frame_name"),
            float(result.get("timestamp", 0.0)),
            score,
            similarity_threshold,
        )

        if score < similarity_threshold:
            logger.info(
                "Rejected: frame=%s score=%.4f",
                result.get("frame_name"),
                score,
            )
            continue

        logger.info(
            "Accepted: frame=%s score=%.4f",
            result.get("frame_name"),
            score,
        )

        filtered.append(
            SearchResult(
                frame_filename=result["frame_name"],
                timestamp=float(result["timestamp"]),
                score=score,
                video_name=video_name,
            )
        )

    if not filtered:
        logger.info(
            "OBJECT NOT FOUND: video=%s query=%s threshold=%.4f filtered_count=0",
            video_name,
            query,
            similarity_threshold,
        )
    else:
        logger.info(
            "OBJECT FOUND: video=%s query=%s threshold=%.4f filtered_count=%d",
            video_name,
            query,
            similarity_threshold,
            len(filtered),
        )

    return filtered


def merge_adjacent_hits(hits: list[dict], gap_tolerance: float = 2.0) -> list[dict]:
    """Merge frame hits where timestamp difference is <= gap_tolerance.

    Returns list of segments sorted by score descending.
    Each segment: {"start": float, "end": float, "score": float, "representative_frame": str}
    """
    if not hits:
        return []

    sorted_hits = sorted(hits, key=lambda x: float(x.get("timestamp", 0.0)))
    groups: list[list[dict]] = [[sorted_hits[0]]]

    for hit in sorted_hits[1:]:
        curr_ts = float(hit.get("timestamp", 0.0))
        prev_ts = float(groups[-1][-1].get("timestamp", 0.0))
        if curr_ts - prev_ts <= gap_tolerance:
            groups[-1].append(hit)
        else:
            groups.append([hit])

    segments = []
    for grp in groups:
        start_ts = float(grp[0].get("timestamp", 0.0))
        end_ts = float(grp[-1].get("timestamp", 0.0))
        if end_ts - start_ts < 1.0:
            end_ts = start_ts + 1.0
        best_hit = max(grp, key=lambda h: float(h.get("score", 0.0)))
        best_score = float(best_hit.get("score", 0.0))
        rep_frame = best_hit.get("frame_name") or best_hit.get("frame_filename") or ""
        segments.append({
            "start": round(start_ts, 3),
            "end": round(end_ts, 3),
            "score": round(best_score, 4),
            "representative_frame": rep_frame,
        })

    segments.sort(key=lambda s: s["score"], reverse=True)
    return segments


def search_video_clip_fallback(
    video_name: str,
    query: str,
    clip_min_score: float | None = None,
    clip_score_margin: float | None = None,
    gap_tolerance: float = 2.0,
    device: str = _DEFAULT_DEVICE,
) -> list[dict]:
    """Run CLIP + FAISS search fallback for phrases that don't match any label.

    1. Uses text prompt 'a photo of {query}'.
    2. Only keeps hits with similarity >= CLIP_MIN_SCORE (default 0.25).
    3. Requires top_score - mean_score >= CLIP_SCORE_MARGIN (default 0.03) to reject nonsense queries.
    4. Merges adjacent hits (gap <= 2s) and returns top 3 segments.
    """
    valid_query = _validate_query(query)
    min_score = float(settings.clip_min_score) if clip_min_score is None else float(clip_min_score)
    margin = float(settings.clip_score_margin) if clip_score_margin is None else float(clip_score_margin)

    prompt = format_clip_query_prompt(valid_query)
    query_embedding = encode_text_query(prompt, device=device)

    try:
        index, _ = load_index(video_name)
    except Exception as exc:
        logger.warning("Could not load FAISS index for %s: %s", video_name, exc)
        return []

    raw_candidates = search_index(video_name, query_embedding, top_k=max(10, int(index.ntotal)))
    if not raw_candidates:
        logger.info("FAISS returned 0 candidates for video=%s query='%s'", video_name, valid_query)
        return []

    scores = [float(c.get("score", 0.0)) for c in raw_candidates]
    top_score = max(scores)
    mean_score = float(np.mean(scores))

    logger.info(
        "CLIP fallback scoring: video=%s query='%s' top_score=%.4f mean_score=%.4f diff=%.4f (min_score=%.4f, margin=%.4f)",
        video_name,
        valid_query,
        top_score,
        mean_score,
        top_score - mean_score,
        min_score,
        margin,
    )

    # Rejection rule 1: top score must be >= CLIP_MIN_SCORE
    if top_score < min_score:
        logger.info("CLIP fallback rejected: top_score %.4f < min_score %.4f for query='%s'", top_score, min_score, valid_query)
        return []

    # Rejection rule 2: top score must beat video's average similarity by margin
    if (top_score - mean_score) < margin:
        logger.info(
            "CLIP fallback rejected (flat distribution / nonsense query): top_score - mean_score = %.4f < margin %.4f for query='%s'",
            top_score - mean_score,
            margin,
            valid_query,
        )
        return []

    # Only keep hits with score >= min_score
    valid_hits = [c for c in raw_candidates if float(c.get("score", 0.0)) >= min_score]
    if not valid_hits:
        logger.info("No hits met min_score=%.4f for query='%s'", min_score, valid_query)
        return []

    # Merge adjacent hits (gap <= 2s) and return top 3 segments
    merged_segments = merge_adjacent_hits(valid_hits, gap_tolerance=gap_tolerance)
    return merged_segments[:3]


def search_video_safe(
    video_name: str,
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.35,
    device: str = _DEFAULT_DEVICE,
) -> Dict[str, Any]:
    """Search a video and return structured response with error handling."""

    try:
        results = search_video(
            video_name=video_name,
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            device=device,
        )

        # No result passed the similarity threshold
        if not results:
            logger.info(
                "Object not found: video=%s query=%s threshold=%.3f",
                video_name,
                query,
                similarity_threshold,
            )

            return {
                "video_name": video_name,
                "query": query,
                "threshold": similarity_threshold,
                "found": False,
                "message": "Object not found in this video.",
                "results": [],
            }

        # At least one result passed the threshold
        logger.info(
            "Object found: video=%s query=%s matches=%d",
            video_name,
            query,
            len(results),
        )

        return {
            "video_name": video_name,
            "query": query,
            "threshold": similarity_threshold,
            "found": True,
            "message": f"Found '{query}' in the video.",
            "results": [
                {
                    "frame_filename": r.frame_filename,
                    "timestamp": r.timestamp,
                    "score": r.score,
                }
                for r in results
            ],
        }

    except Exception as exc:
        logger.exception(
            "Search failed for video %s",
            video_name,
        )

        return {
            "video_name": video_name,
            "query": query,
            "threshold": similarity_threshold,
            "found": False,
            "message": "Search failed.",
            "results": [],
            "error": str(exc),
        }
