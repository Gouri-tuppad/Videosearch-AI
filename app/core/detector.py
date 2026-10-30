import logging
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


class Detector:
    def __init__(self, model_path: Path | str | None = None) -> None:
        resolved_path = Path(model_path) if model_path is not None else settings.yolo_model_path
        self.model_path = str(resolved_path)
        self.model = None

    def ensure_model_loaded(self) -> None:
        if self.model is None:
            try:
                from ultralytics import YOLO
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "YOLO dependency is not installed. Install ultralytics to enable object detection."
                ) from exc

            # Safe fallback for PyTorch 2.6+ weights_only loading of Ultralytics DetectionModel
            try:
                import torch
                from ultralytics.nn.tasks import DetectionModel
                torch.serialization.add_safe_globals([DetectionModel])
            except Exception as exc:
                logger.debug("Could not add DetectionModel to torch safe_globals: %s", exc)

            try:
                self.model = YOLO(self.model_path)
            except Exception as exc:
                error_msg = (
                    f"Failed to load YOLO model from '{self.model_path}': {exc}\n"
                    "Fix steps:\n"
                    "1. Upgrade ultralytics: pip install --upgrade 'ultralytics>=8.3.0'\n"
                    "2. Ensure PyTorch allows safe globals or re-download model weights\n"
                    f"3. Verify that the model weights file '{self.model_path}' exists and is not corrupted."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from exc

    def predict(self, image_path: str | Path) -> list[dict[str, float | str]]:
        self.ensure_model_loaded()
        results = self.model.predict(source=str(image_path), imgsz=640, verbose=False)
        detections = []

        for result in results:
            names = result.names if hasattr(result, "names") else self.model.names
            for box in result.boxes:
                cls_id = int(box.cls[0]) if hasattr(box.cls, "shape") else int(box.cls)
                object_name = names.get(cls_id, str(cls_id))
                bbox = box.xyxy[0].tolist()
                detections.append(
                    {
                        "object_name": object_name,
                        "confidence": float(box.conf[0]) if hasattr(box, "conf") else 0.0,
                        "bbox_xmin": float(bbox[0]),
                        "bbox_ymin": float(bbox[1]),
                        "bbox_xmax": float(bbox[2]),
                        "bbox_ymax": float(bbox[3]),
                    }
                )

        return detections


detector = Detector()
