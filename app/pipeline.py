"""Video processing pipeline.

Single-pass background worker triggered on upload:
  1. Sample frames (1/sec + scene changes)              → 30 %
  2. Run YOLOv8 (conf ≥ 0.5) + CLIP embeddings          → 70 %
  3. Store CLIP vectors in FAISS                          → 85 %
  4. Track events per label & cut clips in parallel       → 100 %

Stores events in SQLite and CLIP vectors in FAISS.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

import cv2
import numpy as np
import torch
from PIL import Image
from sqlalchemy.orm import Session

from app.core.clip_loader import load_clip_model
from app.core.config import settings
from app.core.detector import detector
from app.db.crud import (
    bulk_create_events,
    create_detection,
    create_frame,
    create_video,
    delete_events_for_video,
    delete_frames_for_video,
    delete_video,
    get_video,
    get_video_by_filename,
    list_videos,
    update_video,
    update_video_progress,
)
from app.db.database import SessionLocal
from app.faiss_index import (
    EMBEDDINGS_DIR,
    INDEX_DIR,
    IndexNotFoundError,
    build_index,
    load_index,
    _is_compatible_index_metadata,
)
from app.utils.video_metadata import get_video_metadata

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT_DIR / "videos"
FRAME_DIR = ROOT_DIR / "frames"
CLIPS_DIR = ROOT_DIR / "clips"
PROCESSING_DIR = ROOT_DIR / "processing"
DATA_DIR = ROOT_DIR / "data"
STATUS_FILE = PROCESSING_DIR / "status.json"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
_DEFAULT_DEVICE = "cpu"

if TYPE_CHECKING:
    from transformers import CLIPModel, CLIPProcessor


# ── Status helpers ──────────────────────────────────────────────────────

@dataclass
class PipelineStep:
    name: str
    status: str
    message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


@dataclass
class ProcessingStatus:
    video_name: str | None = None
    status: str = "idle"
    current_step: str | None = None
    error: str | None = None
    steps: List[PipelineStep] = field(default_factory=list)
    started_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_name": self.video_name,
            "status": self.status,
            "current_step": self.current_step,
            "error": self.error,
            "steps": [asdict(step) for step in self.steps],
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }


class PipelineError(Exception):
    pass


class ProcessingFailure(PipelineError):
    pass


def _ensure_directories() -> None:
    for path in (VIDEO_DIR, FRAME_DIR, EMBEDDINGS_DIR, INDEX_DIR, CLIPS_DIR, PROCESSING_DIR, DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for entry in path.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)


def _write_status(status: ProcessingStatus) -> None:
    PROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    status.updated_at = datetime.utcnow().isoformat()
    with STATUS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(status.to_dict(), handle, indent=2)


def _load_status() -> ProcessingStatus:
    if not STATUS_FILE.exists():
        return ProcessingStatus()
    with STATUS_FILE.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return ProcessingStatus(
        video_name=data.get("video_name"),
        status=data.get("status", "idle"),
        current_step=data.get("current_step"),
        error=data.get("error"),
        steps=[PipelineStep(**item) for item in data.get("steps", [])],
        started_at=data.get("started_at"),
        updated_at=data.get("updated_at"),
    )


def _start_step(status: ProcessingStatus, step_name: str, message: str | None = None) -> PipelineStep:
    step = PipelineStep(
        name=step_name,
        status="running",
        message=message,
        started_at=datetime.utcnow().isoformat(),
    )
    status.current_step = step_name
    status.status = "running"
    status.steps.append(step)
    if status.started_at is None:
        status.started_at = step.started_at
    _write_status(status)
    return step


def _complete_step(status: ProcessingStatus, step: PipelineStep, message: str | None = None) -> None:
    step.status = "completed"
    step.finished_at = datetime.utcnow().isoformat()
    if message:
        step.message = message
    _write_status(status)


def _fail_step(status: ProcessingStatus, step: PipelineStep, error: Exception) -> None:
    step.status = "failed"
    step.finished_at = datetime.utcnow().isoformat()
    step.message = str(error)
    status.status = "failed"
    status.error = str(error)
    _write_status(status)


def _reset_status(video_name: str | None = None) -> ProcessingStatus:
    status = ProcessingStatus(video_name=video_name, status="pending")
    _write_status(status)
    return status


# ── File hashing ────────────────────────────────────────────────────────

def compute_file_hash(filepath: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


# ── Data cleanup ────────────────────────────────────────────────────────

def _clear_video_data(video_name: str, session: Session | None = None) -> None:
    """Clear all generated data for a specific video only (frames, embeddings, index, clips).
    Does NOT remove the uploaded video file itself.
    """
    # Clear frame files for this video
    video_frame_dir = FRAME_DIR / video_name
    if video_frame_dir.exists():
        shutil.rmtree(video_frame_dir, ignore_errors=True)
        logger.info("Cleared frame directory for %s", video_name)

    # Clear embeddings for this video
    video_embeddings_dir = EMBEDDINGS_DIR / video_name
    if video_embeddings_dir.exists():
        shutil.rmtree(video_embeddings_dir, ignore_errors=True)
        logger.info("Cleared embeddings directory for %s", video_name)

    # Clear FAISS index for this video
    for suffix in (".index", "_metadata.json"):
        index_file = INDEX_DIR / f"{video_name}{suffix}"
        if index_file.exists():
            index_file.unlink(missing_ok=True)
    logger.info("Cleared FAISS index files for %s", video_name)

    # Clear clips generated for this video
    video_stem = Path(video_name).stem
    # Legacy flat clips dir
    if CLIPS_DIR.exists():
        for clip_file in CLIPS_DIR.iterdir():
            if clip_file.is_file() and clip_file.stem.startswith(video_stem):
                clip_file.unlink(missing_ok=True)
    # Per-video clips dir (data/<video_id>/clips/)
    # Will be handled by video_id if we have the session
    logger.info("Cleared clip files for %s", video_name)

    # Clear DB frame + detection + event records for this video
    if session is not None:
        video_record = get_video_by_filename(session, video_name)
        if video_record is not None:
            deleted_count = delete_frames_for_video(session, video_record.id)
            logger.info("Deleted %d old frame records (with cascading detections) for %s", deleted_count, video_name)
            deleted_events = delete_events_for_video(session, video_record.id)
            logger.info("Deleted %d old event records for %s", deleted_events, video_name)


# ── CLIP model helpers ──────────────────────────────────────────────────

def _get_clip_model(device: str = _DEFAULT_DEVICE) -> tuple[Any, Any]:
    logger.info("Loading CLIP vision encoder: %s", CLIP_MODEL_NAME)
    return load_clip_model(device)


def ensure_clip_vision_encoder_loaded(device: str = _DEFAULT_DEVICE) -> None:
    """Load the CLIP vision encoder once during application startup."""
    _get_clip_model(device)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings.astype(np.float32) / norms


# ── Frame sampling (1/sec + scene changes) ──────────────────────────────

def _sample_frames(video_path: Path, output_dir: Path) -> list[dict]:
    """Sample frames at 1 fps + scene-change frames.

    Returns a list of dicts with keys: frame, frame_idx, timestamp, path, frame_filename
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        del cap
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine scene-change frames via simple content detection
    scene_frames: set[int] = set()
    try:
        from scenedetect import detect, ContentDetector
        scene_list = detect(str(video_path), ContentDetector())
        for scene in scene_list:
            scene_frames.add(int(scene[0].get_frames()))
    except Exception:
        logger.debug("Scene detection unavailable or failed; using only 1 fps sampling")

    # 1 fps sample interval
    sample_interval = max(1, int(round(fps)))

    sampled: list[dict] = []
    prev_idx = -sample_interval  # ensure first frame is sampled
    frame_idx = 0
    prev_gray: np.ndarray | None = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Fast OpenCV scene cut detection when scenedetect list is empty
            small = cv2.resize(frame, (64, 36))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            is_scene = (frame_idx in scene_frames)
            if not scene_frames and prev_gray is not None:
                diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
                if diff > 30.0:
                    is_scene = True
            prev_gray = gray

            is_sample = (frame_idx % sample_interval == 0)

            # Avoid duplicate entries when a scene-change frame coincides with a 1fps sample
            if (is_sample or is_scene) and frame_idx != prev_idx:
                timestamp = frame_idx / fps
                frame_filename = f"frame_{frame_idx:06d}.jpg"
                frame_path = output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                sampled.append({
                    "frame": frame,
                    "frame_idx": frame_idx,
                    "timestamp": round(timestamp, 3),
                    "path": str(frame_path),
                    "frame_filename": frame_filename,
                })
                prev_idx = frame_idx

            frame_idx += 1
    finally:
        cap.release()
        del cap
        import gc
        gc.collect()
    logger.info("Sampled %d frames from %s (total raw frames: %d)", len(sampled), video_path.name, total_frames)
    return sampled


# Alias extract_frames to _sample_frames for pipeline and test compatibility
extract_frames = _sample_frames


# ── Event tracking ──────────────────────────────────────────────────────

@dataclass
class _TrackedEvent:
    label: str
    start: float
    end: float
    max_conf: float
    consecutive: int
    last_seen: float
    confirmed: bool  # True once consecutive >= min_consecutive


def _track_events(
    per_frame_detections: list[tuple[float, list[dict]]],
    min_consecutive: int = 2,
    gap_tolerance: float = 2.0,
    max_length: float = 30.0,
) -> list[dict]:
    """Build events from per-frame detections.

    Args:
        per_frame_detections: list of (timestamp, [{"label": str, "confidence": float}])

    Returns list of {"label", "start", "end", "max_conf"}
    """
    # Sort by timestamp
    per_frame_detections.sort(key=lambda x: x[0])

    # Active (open) trackers per label
    active: dict[str, _TrackedEvent] = {}
    closed: list[dict] = []

    for timestamp, detections in per_frame_detections:
        labels_in_frame = {d["label"] for d in detections}
        conf_map = {}
        for d in detections:
            lbl = d["label"]
            conf_map[lbl] = max(conf_map.get(lbl, 0.0), d["confidence"])

        # Update active trackers
        labels_to_close: list[str] = []
        for label, tracker in active.items():
            if label in labels_in_frame:
                # Label still present
                tracker.end = timestamp
                tracker.last_seen = timestamp
                tracker.consecutive += 1
                tracker.max_conf = max(tracker.max_conf, conf_map.get(label, 0.0))
                if tracker.consecutive >= min_consecutive:
                    tracker.confirmed = True
                # Check max length
                if tracker.confirmed and (tracker.end - tracker.start) > max_length:
                    closed.append({"label": tracker.label, "start": tracker.start, "end": tracker.end, "max_conf": tracker.max_conf})
                    labels_to_close.append(label)
            else:
                # Label absent
                gap = timestamp - tracker.last_seen
                if not tracker.confirmed:
                    # Drop unconfirmed candidate (failed consecutive requirement)
                    labels_to_close.append(label)
                elif gap > gap_tolerance:
                    # Confirmed event exceeded gap tolerance -> close it
                    closed.append({"label": tracker.label, "start": tracker.start, "end": tracker.end, "max_conf": tracker.max_conf})
                    labels_to_close.append(label)
                else:
                    # Within gap tolerance: keep open, reset consecutive count
                    tracker.consecutive = 0

        for lbl in labels_to_close:
            active.pop(lbl, None)

        # Open new trackers for newly appeared labels
        for label in labels_in_frame:
            if label not in active:
                active[label] = _TrackedEvent(
                    label=label,
                    start=timestamp,
                    end=timestamp,
                    max_conf=conf_map.get(label, 0.0),
                    consecutive=1,
                    last_seen=timestamp,
                    confirmed=False,
                )

    # Close remaining active events
    for tracker in active.values():
        if tracker.confirmed:
            closed.append({"label": tracker.label, "start": tracker.start, "end": tracker.end, "max_conf": tracker.max_conf})

    return closed


# ── Parallel clip cutting ───────────────────────────────────────────────

def _cut_clip_opencv(video_path: Path, start: float, end: float, output_path: Path, duration: float) -> bool:
    """Fallback clip cutting via OpenCV VideoCapture & VideoWriter."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        del cap
        return False
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            return False

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        start_frame = max(0, int(start * fps))
        end_frame = min(int(duration * fps), int(end * fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        curr = start_frame
        while curr <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            curr += 1

        out.release()
        del out
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as exc:
        logger.warning("OpenCV clip cutting fallback failed: %s", exc)
        return False
    finally:
        cap.release()
        del cap
        import gc
        gc.collect()


def _cut_clip(
    video_path: Path,
    label: str,
    start: float,
    end: float,
    padding: float,
    output_dir: Path,
    duration: float,
) -> str:
    """Cut a single clip with ffmpeg (or OpenCV fallback). Returns the clip filename."""
    clip_start = max(0.0, start - padding)
    clip_end = min(duration, end + padding)
    clip_duration = max(0.1, clip_end - clip_start)

    safe_label = label.replace(" ", "_").replace("/", "-")
    clip_filename = f"{safe_label}_{start:.1f}_{end:.1f}.mp4"
    output_path = output_dir / clip_filename

    command = [
        "ffmpeg", "-y",
        "-ss", f"{clip_start:.3f}",
        "-i", str(video_path),
        "-t", f"{clip_duration:.3f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-movflags", "+faststart",
        str(output_path),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info("Cut clip with ffmpeg: %s (%.1fs-%.1fs)", clip_filename, clip_start, clip_end)
            return clip_filename
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.info("FFmpeg cutting not available or failed (%s); trying OpenCV fallback for %s", exc, clip_filename)

    # Fallback to OpenCV if ffmpeg didn't produce the clip
    if _cut_clip_opencv(video_path, clip_start, clip_end, output_path, duration):
        logger.info("Cut clip with OpenCV fallback: %s (%.1fs-%.1fs)", clip_filename, clip_start, clip_end)
        return clip_filename

    logger.warning("Failed to cut clip: %s", clip_filename)
    return ""


def _cut_clips_parallel(
    video_path: Path,
    events: list[dict],
    padding: float,
    output_dir: Path,
    duration: float,
    max_workers: int = 4,
) -> dict[int, str]:
    """Cut all event clips in parallel. Returns mapping of event-index → clip filename."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for i, ev in enumerate(events):
            fut = pool.submit(
                _cut_clip,
                video_path,
                ev["label"],
                ev["start"],
                ev["end"],
                padding,
                output_dir,
                duration,
            )
            futures[fut] = i

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                clip_filename = fut.result()
                results[idx] = clip_filename
            except Exception as exc:
                logger.exception("Clip cutting failed for event %d", idx)
                results[idx] = ""

    return results


# ── CLIP embeddings generation ──────────────────────────────────────────

def generate_frame_embeddings(
    video_name: str,
    frames_or_dir: list[dict] | Path | str,
    device: str = _DEFAULT_DEVICE,
) -> None:
    """Generate CLIP image embeddings from sampled frames list or frame directory."""
    model, processor = _get_clip_model(device)
    model_name = getattr(getattr(model, "config", None), "name_or_path", None) or str(settings.clip_model_name)
    logger.info("Generating image embeddings using CLIP model: %s", model_name)

    items_to_embed = []
    if isinstance(frames_or_dir, (str, Path)):
        frame_dir = Path(frames_or_dir)
        frame_paths = sorted(frame_dir.glob("*.jpg"))
        interval = float(settings.frame_extraction_interval) if float(settings.frame_extraction_interval) > 0 else 1.0
        for idx, fp in enumerate(frame_paths):
            items_to_embed.append({
                "path": str(fp),
                "frame_idx": idx,
                "frame_filename": fp.name,
                "timestamp": round(idx * interval, 3),
            })
    else:
        for idx, s in enumerate(frames_or_dir):
            fn = s.get("frame_filename") or s.get("frame_name") or f"frame_{idx:06d}.jpg"
            fp = s.get("path") or fn
            items_to_embed.append({
                "path": str(fp),
                "frame_idx": s.get("frame_idx", idx),
                "frame_filename": fn,
                "timestamp": s.get("timestamp", 0.0),
            })

    if not items_to_embed:
        raise ValueError("No frames to embed")

    embeddings = []
    metadata_items = []

    for item in items_to_embed:
        frame_path = Path(item["path"])
        image = Image.open(frame_path).convert("RGB")

        metadata_items.append({
            "video_name": video_name,
            "frame_index": item["frame_idx"],
            "frame_filename": item["frame_filename"],
            "frame_name": item["frame_filename"],
            "timestamp": item["timestamp"],
        })

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            embedding = model.get_image_features(**inputs)

        embedding = embedding.cpu().numpy()
        embeddings.append(embedding)

    embeddings_array = np.vstack(embeddings)
    embeddings_array = _normalize_embeddings(embeddings_array)

    logger.info("Generated embeddings shape=%s dtype=%s for video=%s", embeddings_array.shape, embeddings_array.dtype, video_name)

    output_dir = EMBEDDINGS_DIR / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "frame_embeddings.npy", embeddings_array)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({"items": metadata_items}, handle, indent=2)

    logger.info("Generated CLIP embeddings for %d frames", len(items_to_embed))


# ── Duplicate detection ─────────────────────────────────────────────────

def _video_already_processed(db: Session, video_name: str) -> bool:
    video = get_video_by_filename(db, video_name)
    if video is None:
        return False
    # Ensure that both the FAISS index and the corresponding embeddings/metadata exist
    try:
        index, metadata = load_index(video_name)

        if not _is_compatible_index_metadata(metadata):
            logger.warning("FAISS index metadata incompatible or stale for %s; rebuilding index.", video_name)
            return False

        from app.faiss_index import EMBEDDINGS_DIR, METADATA_FILENAME, EMBEDDINGS_FILENAME

        embeddings_source = EMBEDDINGS_DIR / video_name
        metadata_file = embeddings_source / METADATA_FILENAME
        embeddings_file = embeddings_source / EMBEDDINGS_FILENAME

        if not metadata_file.exists() or not embeddings_file.exists():
            return False

        if index.ntotal != len(metadata["items"]):
            logger.warning("FAISS index count mismatch for %s; rebuilding index.", video_name)
            return False

        return True
    except IndexNotFoundError:
        return False
    except Exception:
        return False


# ── Video record management ─────────────────────────────────────────────

def _get_or_create_video(session: Session, video_name: str, metadata: dict[str, Any], file_hash: str | None = None) -> Any:
    existing_video = get_video_by_filename(session, video_name)
    if existing_video is not None:
        logger.info("Video record already exists for %s; updating with metadata", video_name)
        update_video(
            session,
            existing_video.id,
            duration=metadata["duration"],
            fps=metadata["fps"],
            file_hash=file_hash or existing_video.file_hash,
        )
        return existing_video

    logger.info("Creating new video record for %s", video_name)
    current_datetime = datetime.now()
    return create_video(
        session,
        filename=video_name,
        duration=metadata["duration"],
        fps=metadata["fps"],
        upload_date=current_datetime.date(),
        upload_time=current_datetime.time(),
        file_hash=file_hash,
        status="processing",
    )


# ── Main pipeline ───────────────────────────────────────────────────────

def process_new_video(video_path: str | Path, device: str = _DEFAULT_DEVICE) -> Dict[str, Any]:
    """Run the full processing pipeline on a video file.

    Stages:
        sampling   → 30 %
        detection  → 70 %
        embedding  → 85 %
        clipping   → 100 %
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info("Start processing video: %s", video_path.name)
    _ensure_directories()
    status = _reset_status(video_path.name)
    session = SessionLocal()
    step: PipelineStep | None = None
    detected_objects: List[str] = []

    try:
        # ── Clear old data ──────────────────────────────────────────
        try:
            logger.info("Clearing old data for video: %s", video_path.name)
            _clear_video_data(video_path.name, session)
        except Exception as exc:
            logger.warning("Non-fatal error clearing old data for %s: %s", video_path.name, exc)

        # ── Video metadata ──────────────────────────────────────────
        meta = get_video_metadata(video_path)
        file_hash = compute_file_hash(video_path)
        video = _get_or_create_video(session, video_path.name, meta, file_hash=file_hash)
        video_id = video.id
        duration = meta["duration"]

        # Update status to processing
        update_video_progress(session, video_id, 0.0, "processing")

        # ── Phase 1: Sample frames (→ 30 %) ────────────────────────
        try:
            step = _start_step(status, "sample_frames", "Sampling frames (1 fps + scene changes)")
            frame_output_dir = FRAME_DIR / video_path.name
            sampled_frames = extract_frames(video_path, frame_output_dir)
            _complete_step(status, step, f"Sampled {len(sampled_frames)} frames")
            update_video_progress(session, video_id, 30.0)
            logger.info("Phase 1 complete: sampled %d frames", len(sampled_frames))
        except Exception as exc:
            logger.exception("Frame sampling failed for %s", video_path.name)
            if step:
                _fail_step(status, step, exc)
            update_video_progress(session, video_id, 0.0, "failed")
            session.rollback()
            raise ProcessingFailure(str(exc)) from exc

        # ── Phase 2: YOLO detection (→ 70 %) ───────────────────────
        per_frame_detections: list[tuple[float, list[dict]]] = []
        try:
            step = _start_step(status, "detect_objects", "Running YOLOv8 object detection")
            detected_object_names: set[str] = set()
            conf_threshold = float(settings.yolo_confidence)

            for i, sample in enumerate(sampled_frames):
                frame_filename = sample.get("frame_filename") or sample.get("frame_name") or f"frame_{i:06d}.jpg"
                frame_path = sample.get("path") or str(frame_output_dir / frame_filename)
                timestamp = sample.get("timestamp", 0.0)

                # Create frame record in DB
                frame_obj = create_frame(
                    session,
                    video_id=video_id,
                    frame_name=frame_filename,
                    timestamp=timestamp,
                )

                # Run YOLO
                raw_detections = detector.predict(frame_path)
                frame_dets: list[dict] = []

                for det in raw_detections:
                    conf = det["confidence"]
                    if conf < conf_threshold:
                        continue
                    object_name = str(det["object_name"]).strip()
                    if object_name:
                        detected_object_names.add(object_name)
                    create_detection(
                        session,
                        frame_id=frame_obj.id,
                        object_name=object_name,
                        confidence=conf,
                        bbox_xmin=det["bbox_xmin"],
                        bbox_ymin=det["bbox_ymin"],
                        bbox_xmax=det["bbox_xmax"],
                        bbox_ymax=det["bbox_ymax"],
                    )
                    frame_dets.append({"label": object_name, "confidence": conf})

                per_frame_detections.append((timestamp, frame_dets))

                # Progress: 30 → 70
                progress = 30.0 + (i / max(len(sampled_frames), 1)) * 40.0
                update_video_progress(session, video_id, progress)

            detected_objects = sorted(detected_object_names)
            _complete_step(status, step, f"Detected {len(detected_objects)} unique objects")
            update_video_progress(session, video_id, 70.0)
            logger.info("Phase 2 complete: detected objects: %s", detected_objects)
        except Exception as exc:
            logger.exception("Object detection failed for %s", video_path.name)
            if step:
                _fail_step(status, step, exc)
            update_video_progress(session, video_id, 0.0, "failed")
            session.rollback()
            raise ProcessingFailure(str(exc)) from exc

        # ── Phase 3: CLIP embeddings + FAISS (→ 85 %) ──────────────
        try:
            step = _start_step(status, "generate_embeddings", "Generating CLIP embeddings")
            _get_clip_model(device)
            generate_frame_embeddings(video_path.name, sampled_frames, device=device)
            _complete_step(status, step, "CLIP embeddings generated")
            update_video_progress(session, video_id, 80.0)

            step = _start_step(status, "build_faiss_index", "Building FAISS search index")
            build_index(video_path.name)
            _complete_step(status, step, "FAISS index built")
            update_video_progress(session, video_id, 85.0)
            logger.info("Phase 3 complete: embeddings + FAISS index built")
        except Exception as exc:
            logger.exception("Embedding/index generation failed for %s", video_path.name)
            if step:
                _fail_step(status, step, exc)
            update_video_progress(session, video_id, 0.0, "failed")
            session.rollback()
            raise ProcessingFailure(str(exc)) from exc

        # ── Phase 4: Event tracking + clip cutting (→ 100 %) ───────
        try:
            step = _start_step(status, "track_and_clip", "Tracking events and cutting clips")

            events = _track_events(
                per_frame_detections,
                min_consecutive=int(settings.event_min_consecutive),
                gap_tolerance=float(settings.event_gap_tolerance),
                max_length=float(settings.event_max_length),
            )
            logger.info("Tracked %d events for %s", len(events), video_path.name)

            # Per-video clips directory: data/<video_id>/clips/
            video_clips_dir = DATA_DIR / str(video_id) / "clips"
            video_clips_dir.mkdir(parents=True, exist_ok=True)

            if events:
                clip_results = _cut_clips_parallel(
                    video_path,
                    events,
                    padding=float(settings.clip_padding_seconds),
                    output_dir=video_clips_dir,
                    duration=duration,
                )

                # Assign clip paths and store events in DB
                event_rows = []
                for i, ev in enumerate(events):
                    clip_filename = clip_results.get(i, "")
                    clip_path = f"data/{video_id}/clips/{clip_filename}" if clip_filename else None
                    event_rows.append({
                        "video_id": video_id,
                        "label": ev["label"],
                        "start": ev["start"],
                        "end": ev["end"],
                        "clip_path": clip_path,
                        "max_conf": ev["max_conf"],
                    })
                bulk_create_events(session, event_rows)
                logger.info("Stored %d events in DB for video_id=%d", len(event_rows), video_id)
            else:
                logger.info("No events detected for %s", video_path.name)

            _complete_step(status, step, f"Tracked {len(events)} events, clips generated")
            update_video_progress(session, video_id, 100.0, "ready")
            logger.info("Phase 4 complete: all clips cut, events stored")
        except Exception as exc:
            logger.exception("Event tracking/clipping failed for %s", video_path.name)
            if step:
                _fail_step(status, step, exc)
            update_video_progress(session, video_id, 0.0, "failed")
            session.rollback()
            raise ProcessingFailure(str(exc)) from exc

        status.status = "completed"
        status_dict = status.to_dict()
        status_dict["detected_objects"] = detected_objects
        _write_status(status)
        logger.info("Finish processing video: %s", video_path.name)
        return status_dict

    except Exception as exc:
        logger.exception("Video processing failed with ProcessingFailure for %s: %s", video_path.name, exc)
        if step:
            _fail_step(status, step, exc)
        try:
            if "video_id" in locals() and video_id is not None:
                update_video_progress(session, video_id, 0.0, "failed")
        except Exception as e:
            logger.warning("Failed to set video status to 'failed': %s", e)
        session.rollback()
        raise ProcessingFailure(str(exc)) from exc
    finally:
        try:
            session.commit()
        except Exception:
            session.rollback()
        session.close()


def prepare_for_new_upload(video_name: str | None = None) -> None:
    """Prepare the system for a new upload.
    If video_name is provided, only clear data for that specific video.
    Otherwise, just ensure directories exist.
    """
    _ensure_directories()
    if video_name:
        _clear_video_data(video_name)


def get_processing_status() -> Dict[str, Any]:
    return _load_status().to_dict()
