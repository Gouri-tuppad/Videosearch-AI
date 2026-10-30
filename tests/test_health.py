import unittest
from unittest.mock import patch

from app.api.routes.health import _check_yolo_model


class HealthCheckTests(unittest.TestCase):
    def test_yolo_check_initializes_the_lazy_model(self) -> None:
        with patch("app.api.routes.health.detector.ensure_model_loaded") as ensure_model_loaded:
            result = _check_yolo_model()

        self.assertEqual(result["status"], "loaded")
        ensure_model_loaded.assert_called_once_with()

    def test_yolo_check_reports_model_initialization_errors(self) -> None:
        with patch(
            "app.api.routes.health.detector.ensure_model_loaded",
            side_effect=RuntimeError("Invalid YOLO weights"),
        ):
            result = _check_yolo_model()

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["details"], "Invalid YOLO weights")


if __name__ == "__main__":
    unittest.main()
