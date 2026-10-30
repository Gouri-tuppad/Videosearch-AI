from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING
import torch

from app.core.config import settings

logger = logging.getLogger(__name__)

_CLIP_MODEL: Any = None
_CLIP_PROCESSOR: Any = None

if TYPE_CHECKING:
    from transformers import CLIPModel, CLIPProcessor


def get_default_device() -> str:
    """Return 'cuda' if GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clip_model(device: str | None = None) -> tuple[Any, Any]:
    """Load and cache the CLIP model and processor globally.

    Moves the model to GPU if available, else CPU.
    """
    global _CLIP_MODEL, _CLIP_PROCESSOR
    target_device = device if device is not None else get_default_device()

    if _CLIP_MODEL is None or _CLIP_PROCESSOR is None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            try:
                from transformers.models.clip import CLIPModel, CLIPProcessor
            except ImportError as e:
                logger.error(f"Failed to import CLIP models: {e}")
                raise ImportError("CLIPModel and CLIPProcessor not found in transformers library") from e

        try:
            logger.info("Loading CLIP model from local cache...")
            _CLIP_MODEL = CLIPModel.from_pretrained(settings.clip_model_name, local_files_only=True)
            _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(settings.clip_model_name, local_files_only=True)
        except OSError:
            logger.info("Downloading CLIP model (first run only)...")
            _CLIP_MODEL = CLIPModel.from_pretrained(settings.clip_model_name, local_files_only=False)
            _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(settings.clip_model_name, local_files_only=False)

        _CLIP_MODEL.eval()
        _CLIP_MODEL.to(target_device)
        logger.info("CLIP model loaded successfully on device: %s", target_device)
    else:
        try:
            _CLIP_MODEL.to(target_device)
        except Exception:
            pass

    return _CLIP_MODEL, _CLIP_PROCESSOR


@torch.no_grad()
def infer_image_features(model: Any, inputs: dict[str, Any]) -> Any:
    """Run image inference using torch.no_grad()."""
    return model.get_image_features(**inputs)


@torch.no_grad()
def infer_text_features(model: Any, inputs: dict[str, Any]) -> Any:
    """Run text inference using torch.no_grad()."""
    return model.get_text_features(**inputs)
