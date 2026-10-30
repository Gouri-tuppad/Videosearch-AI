"""Script to verify that YOLO and CLIP models load and run successfully."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from app.core.detector import detector
from app.core.clip_loader import load_clip_model, get_default_device


def check_yolo() -> None:
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    temp_img_path = ROOT_DIR / "dummy_test_image.jpg"
    try:
        Image.fromarray(dummy_img).save(temp_img_path)
        detector.ensure_model_loaded()
        _ = detector.predict(temp_img_path)
        print("YOLO OK")
    finally:
        if temp_img_path.exists():
            temp_img_path.unlink(missing_ok=True)


def check_clip() -> None:
    device = get_default_device()
    model, processor = load_clip_model(device=device)
    dummy_pil = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    inputs = processor(images=dummy_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    assert image_features is not None
    assert image_features.shape[-1] == 512
    print("CLIP OK")


if __name__ == "__main__":
    check_yolo()
    check_clip()

