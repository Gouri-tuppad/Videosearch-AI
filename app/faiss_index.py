from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from app.core.config import settings

_faiss = None

def _get_faiss():
    global _faiss
    if _faiss is None:
        try:
            import faiss as faiss_module
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "FAISS dependency is not installed. Install faiss to enable search indexing."
            ) from exc
        _faiss = faiss_module
    return _faiss

logger = logging.getLogger(__name__)

INDEX_DIR = Path(settings.faiss_index_folder)
EMBEDDINGS_DIR = Path(settings.embeddings_folder)
METADATA_FILENAME = "metadata.json"
EMBEDDINGS_FILENAME = "frame_embeddings.npy"
INDEX_FILE_SUFFIX = ".index"
METADATA_FILE_SUFFIX = "_metadata.json"
INDEX_METADATA_SCHEMA_VERSION = 1


class FaissIndexError(Exception):
    """Base exception for FAISS index operations."""


class IndexNotFoundError(FaissIndexError):
    """Raised when a requested FAISS index does not exist."""


class IndexCorruptedError(FaissIndexError):
    """Raised when a FAISS index file is corrupted or cannot be loaded."""


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _index_path(video_name: str, index_dir: Path = INDEX_DIR) -> Path:
    return index_dir / f"{video_name}{INDEX_FILE_SUFFIX}"


def _metadata_path(video_name: str, index_dir: Path = INDEX_DIR) -> Path:
    return index_dir / f"{video_name}{METADATA_FILE_SUFFIX}"


def _embeddings_source_path(video_name: str) -> Path:
    return EMBEDDINGS_DIR / video_name


def _load_metadata_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_metadata(metadata: Dict[str, Any]) -> None:
    if "items" not in metadata or not isinstance(metadata["items"], list):
        raise ValueError("Metadata file must contain an 'items' list")

    for item in metadata["items"]:
        if not all(key in item for key in ("frame_index", "frame_name", "timestamp")):
            raise ValueError("Each metadata item must include frame_index, frame_name, and timestamp")


def _is_compatible_index_metadata(metadata: Dict[str, Any]) -> bool:
    return (
        metadata.get("index_schema_version") == INDEX_METADATA_SCHEMA_VERSION
        and metadata.get("index_type") == "IndexFlatIP"
        and metadata.get("embeddings_normalized") is True
    )


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")

    embeddings = embeddings.astype(np.float32)
    faiss = _get_faiss()
    faiss.normalize_L2(embeddings)
    return embeddings


def build_index(
    video_name: str,
    embeddings_dir: Path | str = EMBEDDINGS_DIR,
    index_dir: Path | str = INDEX_DIR,
) -> Tuple[faiss.Index, Dict[str, Any]]:
    """Builds and stores a FAISS index for a given video.

    The embedding source directory should contain a metadata file and a
    NumPy file with frame embeddings:

      embeddings/<video_name>/metadata.json
      embeddings/<video_name>/frame_embeddings.npy

    Metadata must contain a list of items with frame_index, frame_name, and timestamp.
    """
    embeddings_dir = Path(embeddings_dir)
    index_dir = Path(index_dir)
    _ensure_directory(index_dir)

    source_dir = embeddings_dir / video_name
    metadata_path = source_dir / METADATA_FILENAME
    embeddings_path = source_dir / EMBEDDINGS_FILENAME

    if not metadata_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embedding source missing for video '{video_name}'. "
            f"Expected files: {metadata_path}, {embeddings_path}"
        )

    metadata = _load_metadata_file(metadata_path)
    _validate_metadata(metadata)
    for item in metadata["items"]:
        item.setdefault("video_name", video_name)
        item.setdefault("frame_filename", item["frame_name"])

    embeddings = np.load(embeddings_path)
    embeddings = _normalize_embeddings(embeddings)

    if embeddings.shape[0] != len(metadata["items"]):
        raise ValueError(
            "Number of embeddings does not match number of metadata items"
        )

    dimension = embeddings.shape[1]
    faiss = _get_faiss()
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_path = index_dir / f"{video_name}{INDEX_FILE_SUFFIX}"
    metadata_out_path = index_dir / f"{video_name}{METADATA_FILE_SUFFIX}"
    metadata_out = dict(metadata)
    metadata_out["index_schema_version"] = INDEX_METADATA_SCHEMA_VERSION
    metadata_out["index_type"] = "IndexFlatIP"
    metadata_out["vector_dimension"] = dimension
    metadata_out["embeddings_normalized"] = True

    faiss.write_index(index, str(index_path))

    with metadata_out_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_out, handle, indent=2)

    logger.info(
        "Built FAISS index for video %s with %d vectors and dimension %d",
        video_name,
        embeddings.shape[0],
        dimension,
    )
    return index, metadata


def load_index(
    video_name: str,
    index_dir: Path | str = INDEX_DIR,
) -> Tuple[faiss.Index, Dict[str, Any]]:
    """Loads an existing FAISS index and its metadata for a video."""
    index_dir = Path(index_dir)
    index_path = _index_path(video_name, index_dir)
    metadata_path = _metadata_path(video_name, index_dir)

    if not index_path.exists() or not metadata_path.exists():
        raise IndexNotFoundError(f"Index for video '{video_name}' not found")

    faiss = _get_faiss()
    try:
        index = faiss.read_index(str(index_path))
        metadata = _load_metadata_file(metadata_path)
        _validate_metadata(metadata)
        if index.ntotal != len(metadata["items"]):
            raise IndexCorruptedError(
                f"Index count ({index.ntotal}) does not match metadata count ({len(metadata['items'])}) for video '{video_name}'"
            )
        if not _is_compatible_index_metadata(metadata):
            # Attempt to rebuild the index from the original embeddings if possible.
            embeddings_source = _embeddings_source_path(video_name)
            if (embeddings_source / METADATA_FILENAME).exists() and (embeddings_source / EMBEDDINGS_FILENAME).exists():
                logger.warning(
                    "Index metadata incompatible for %s; attempting to rebuild index from embeddings at %s",
                    video_name,
                    embeddings_source,
                )
                try:
                    # build_index will write a new index and compatible metadata
                    build_index(video_name)
                    index = faiss.read_index(str(index_path))
                    metadata = _load_metadata_file(metadata_path)
                    _validate_metadata(metadata)
                    if not _is_compatible_index_metadata(metadata):
                        raise IndexCorruptedError(
                            f"Rebuilt index for video '{video_name}' is still incompatible."
                        )
                except Exception as exc:
                    raise IndexCorruptedError(
                        f"Index metadata for video '{video_name}' is incompatible and rebuild failed: {exc}"
                    ) from exc
            else:
                raise IndexCorruptedError(
                    f"Index metadata for video '{video_name}' is incompatible with the current search pipeline. Rebuild the FAISS index."
                )
    except (OSError, ValueError, faiss.FaissError, IndexCorruptedError) as exc:
        if isinstance(exc, IndexCorruptedError):
            raise
        raise IndexCorruptedError(
            f"Failed to load index for video '{video_name}'"
        ) from exc

    return index, metadata


def _validate_query_embedding(query_embedding: np.ndarray) -> np.ndarray:
    if query_embedding.ndim != 1:
        raise ValueError("Query embedding must be a 1D array")
    faiss = _get_faiss()
    query = query_embedding.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query)
    return query


def search_index(
    video_name: str,
    query_embedding: np.ndarray,
    top_k: int = 5,
    index_dir: Path | str = INDEX_DIR,
) -> List[Dict[str, Any]]:
    """Searches a stored FAISS index and returns matching frame metadata."""
    index, metadata = load_index(video_name, index_dir=index_dir)
    query = _validate_query_embedding(query_embedding)

    if index.ntotal == 0:
        return []

    distances, indices = index.search(query, top_k)
    logger.info("Executed FAISS search for video %s top_k=%d", video_name, top_k)
    results: List[Dict[str, Any]] = []

    for score, frame_id in zip(distances[0], indices[0]):
        if frame_id < 0:
            continue
        item = metadata["items"][int(frame_id)]
        results.append(
            {
                "frame_index": item["frame_index"],
                "frame_name": item["frame_name"],
                "timestamp": item["timestamp"],
                "score": float(score),
            }
        )

    return results
