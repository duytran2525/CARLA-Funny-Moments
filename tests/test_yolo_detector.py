import unittest

import numpy as np
import torch

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


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls, track_id=None):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)
        self.cls = torch.tensor(cls, dtype=torch.float32)
        self.id = None if track_id is None else torch.tensor(track_id, dtype=torch.float32)

    def __len__(self):
        return int(self.xyxy.shape[0])


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class YoloDetectorCompatTests(unittest.TestCase):
    def test_extract_static_export_hw(self):
        exc = AssertionError(
            "input size torch.Size([1, 3, 448, 448]) not equal to max model size (1, 3, 640, 640)"
        )
        self.assertEqual(YoloDetector._extract_static_export_hw(exc), (640, 640))

    def test_metrics_eval_classes_match_requested_order(self):
        expected = (
            "vehicle",
            "two_wheeler",
            "traffic_light_red",
            "traffic_sign",
            "pedestrian",
            "traffic_light_green",
            "stop_line",
        )
        self.assertEqual(tuple(YoloDetector.DEFAULT_DISPLAY_CLASSES), expected)
        self.assertEqual(tuple(YoloDetector.METRICS_EVAL_CLASSES), expected)

    def test_two_wheeler_aliases_match_metrics_class(self):
        detector = YoloDetector.__new__(YoloDetector)
        detector.class_names = {
            0: "bike",
            1: "motobike",
            2: "motorcycle",
            3: "bicycle",
            4: "two_wheeler",
        }
        detector.class_aliases = dict(YoloDetector.CLASS_ALIASES)

        for cls_id in range(5):
            self.assertEqual(detector._resolve_class_name(cls_id), "two_wheeler")

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

    def test_detect_and_track_falls_back_to_predict_when_tracker_returns_no_boxes(self):
        detector = YoloDetector.__new__(YoloDetector)
        detector.conf_threshold = 0.25
        detector.display_classes = {"vehicle"}
        detector.class_names = {0: "vehicle"}
        detector.class_aliases = dict(YoloDetector.CLASS_ALIASES)
        detector._enable_tracking_metrics_logging = False
        detector._metrics_eval_classes = set(YoloDetector.METRICS_EVAL_CLASSES)
        detector.current_frame_id = 1
        detector.tracking_logs = []

        detector._track = lambda _image: [_FakeResult(_FakeBoxes([], [], []))]
        detector._predict = lambda _image: [
            _FakeResult(_FakeBoxes([[10, 20, 110, 120]], [0.9], [0]))
        ]

        image = np.zeros((240, 320, 3), dtype=np.uint8)
        result = detector.detect_and_track(image)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["class"], "vehicle")
        self.assertEqual(result[0]["bbox"], [10, 20, 110, 120])
        self.assertIsNone(result[0]["track_id"])


if __name__ == "__main__":
    unittest.main()
