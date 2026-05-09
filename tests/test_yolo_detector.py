import unittest

import numpy as np

from core_perception.yolo_detector import YoloDetector


class _FakePredictModel:
    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(dict(kwargs))
        if kwargs.get("imgsz") == 448:
            raise AssertionError(
                "input size torch.Size([1, 3, 448, 448]) not equal to max model size (1, 3, 640, 640)"
            )
        return ["ok"]


class YoloDetectorCompatTests(unittest.TestCase):
    def test_extract_static_export_hw(self):
        exc = AssertionError(
            "input size torch.Size([1, 3, 448, 448]) not equal to max model size (1, 3, 640, 640)"
        )
        self.assertEqual(YoloDetector._extract_static_export_hw(exc), (640, 640))

    def test_predict_recovers_from_static_engine_size_mismatch(self):
        detector = YoloDetector.__new__(YoloDetector)
        detector.conf_threshold = 0.25
        detector.inference_imgsz = 448
        detector._predict_device = 0
        detector._use_half_precision = False
        detector._is_exported_model = True
        detector.model = _FakePredictModel()

        image = np.zeros((360, 640, 3), dtype=np.uint8)
        result = detector._predict(image)

        self.assertEqual(result, ["ok"])
        self.assertEqual(detector.inference_imgsz, 640)
        self.assertEqual(len(detector.model.calls), 2)
        self.assertEqual(detector.model.calls[0]["imgsz"], 448)
        self.assertEqual(detector.model.calls[1]["imgsz"], 640)


if __name__ == "__main__":
    unittest.main()
