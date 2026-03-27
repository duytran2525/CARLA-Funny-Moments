import math
import os
from typing import Any, Dict, List, Optional

import cv2
import torch
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        self.conf_threshold = conf_threshold

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Khong tim thay file model YOLO: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[YOLO DETECTOR] Dang load model tu {model_path} len thiet bi {self.device}...")
        self.model = YOLO(model_path).to(self.device)
        self.class_names = self.model.names

        self.display_classes = [
            "pedestrian",
            "vehicle",
            "two_wheeler",
            "traffic_sign",
            "traffic_light_red",
            "traffic_light_green",
        ]
        self.target_classes = ["pedestrian", "vehicle", "two_wheeler"]
        self.brake_classes = set(self.target_classes)
        self.red_light_class = "traffic_light_red"
        self.green_light_class = "traffic_light_green"
        self.traffic_light_classes = {self.red_light_class, self.green_light_class}
        self.class_aliases = {
            "traffic_light_red": "traffic_light_red",
            "trafficlight_red": "traffic_light_red",
            "red_traffic_light": "traffic_light_red",
            "traffic_light_green": "traffic_light_green",
            "trafficlight_green": "traffic_light_green",
            "green_traffic_light": "traffic_light_green",
        }
        print(
            f"[YOLO DETECTOR] Hien thi: {self.display_classes} | "
            f"Phanh gap: {sorted(self.brake_classes)}"
        )

        self.reference_objects = {
            "pedestrian": {"real_h": 1.6, "real_w": 0.45},
            "vehicle": {"real_h": 1.5, "real_w": 1.8},
            "two_wheeler": {"real_h": 1.1, "real_w": 0.6},
            "traffic_sign": {"real_h": 0.7, "real_w": 0.7},
            "traffic_light_red": {"real_h": 0.8, "real_w": 0.35},
            "traffic_light_green": {"real_h": 0.8, "real_w": 0.35},
        }

        # ROI for traffic light filtering (reduce false detections from other lanes).
        self.traffic_light_max_y_ratio = 0.55
        self.traffic_light_regions = {
            "urban": {
                "x_min_ratio": 0.35,
                "x_max_ratio": 0.65,
                "max_distance": 15.0,
                "label": "Do thi",
            },
            "rural_right": {
                "x_min_ratio": 0.65,
                "x_max_ratio": 0.95,
                "max_distance": 7.0,
                "label": "Nong thon (le phai)",
            },
        }
        self.traffic_light_zone_priority = ("urban", "rural_right")
        self.green_override_ratio = 1.05

        # Small temporal smoothing for stable stop/go behavior.
        self.red_confirm_frames = 2
        self.green_release_frames = 2
        self.red_hold_frames = 6
        self._red_confirm_counter = 0
        self._green_confirm_counter = 0
        self._red_hold_counter = 0
        self._red_light_active = False
        self._last_red_light_active = False

        self._last_debug_info: Dict[str, Any] = {}

    def _normalize_class_name(self, class_name):
        return str(class_name).strip().lower().replace(" ", "_").replace("-", "_")

    def _resolve_class_name(self, cls_id):
        if isinstance(self.class_names, dict):
            raw_name = self.class_names.get(cls_id, str(cls_id))
        else:
            if 0 <= cls_id < len(self.class_names):
                raw_name = self.class_names[cls_id]
            else:
                raw_name = str(cls_id)
        normalized_name = self._normalize_class_name(raw_name)
        return self.class_aliases.get(normalized_name, normalized_name)

    def _classify_traffic_light_zone(self, center_x, center_y, width, height):
        if center_y >= height * self.traffic_light_max_y_ratio:
            return None

        urban = self.traffic_light_regions["urban"]
        rural_right = self.traffic_light_regions["rural_right"]

        # Region 1: urban lights in center.
        if width * urban["x_min_ratio"] < center_x <= width * urban["x_max_ratio"]:
            return "urban"

        # Region 2: rural lights on right side.
        if width * rural_right["x_min_ratio"] < center_x < width * rural_right["x_max_ratio"]:
            return "rural_right"

        return None

    def _signal_score(self, confidence, distance):
        return confidence / max(distance, 0.5)

    def _pick_better_signal(self, current_signal, candidate_signal):
        if current_signal is None:
            return candidate_signal
        if candidate_signal["score"] > current_signal["score"]:
            return candidate_signal
        if candidate_signal["score"] == current_signal["score"]:
            if candidate_signal["confidence"] >= current_signal["confidence"]:
                return candidate_signal
        return current_signal

    def _candidate_in_zone_range(self, signal, zone_name):
        if signal is None:
            return False
        zone_cfg = self.traffic_light_regions[zone_name]
        return signal["distance"] < zone_cfg["max_distance"]

    def _evaluate_traffic_light_decision(self, zone_signals):
        active_zone = None
        for zone_name in self.traffic_light_zone_priority:
            signals = zone_signals.get(zone_name, {})
            if self._candidate_in_zone_range(signals.get("red"), zone_name):
                active_zone = zone_name
                break
            if self._candidate_in_zone_range(signals.get("green"), zone_name):
                active_zone = zone_name
                break

        if active_zone is None:
            return {
                "active_zone": None,
                "red_trigger": False,
                "green_release": False,
                "reason": "",
            }

        zone_cfg = self.traffic_light_regions[active_zone]
        red_signal = zone_signals[active_zone].get("red")
        green_signal = zone_signals[active_zone].get("green")
        red_in_range = self._candidate_in_zone_range(red_signal, active_zone)
        green_in_range = self._candidate_in_zone_range(green_signal, active_zone)

        red_trigger = False
        green_release = False
        reason = ""

        if red_in_range and green_in_range:
            if green_signal["score"] >= red_signal["score"] * self.green_override_ratio:
                green_release = True
                reason = (
                    f"{zone_cfg['label']}: GREEN @ {green_signal['distance']:.1f}m "
                    f"(override RED)"
                )
            else:
                red_trigger = True
                reason = f"{zone_cfg['label']}: RED @ {red_signal['distance']:.1f}m"
        elif red_in_range:
            red_trigger = True
            reason = f"{zone_cfg['label']}: RED @ {red_signal['distance']:.1f}m"
        elif green_in_range:
            green_release = True
            reason = f"{zone_cfg['label']}: GREEN @ {green_signal['distance']:.1f}m"

        return {
            "active_zone": active_zone,
            "red_trigger": red_trigger,
            "green_release": green_release,
            "reason": reason,
        }

    def _update_red_light_state(self, red_trigger, green_release):
        if red_trigger:
            self._red_confirm_counter = min(
                self._red_confirm_counter + 1, self.red_confirm_frames + 10
            )
            self._green_confirm_counter = 0
        elif green_release:
            self._green_confirm_counter = min(
                self._green_confirm_counter + 1, self.green_release_frames + 10
            )
            self._red_confirm_counter = 0
        else:
            self._red_confirm_counter = max(self._red_confirm_counter - 1, 0)
            self._green_confirm_counter = max(self._green_confirm_counter - 1, 0)

        if not self._red_light_active and self._red_confirm_counter >= self.red_confirm_frames:
            self._red_light_active = True
            self._red_hold_counter = self.red_hold_frames

        if self._red_light_active:
            if red_trigger:
                self._red_hold_counter = self.red_hold_frames
            else:
                self._red_hold_counter -= 1

            if self._green_confirm_counter >= self.green_release_frames:
                self._red_light_active = False
                self._red_hold_counter = 0
            elif self._red_hold_counter <= 0:
                self._red_light_active = False

        return self._red_light_active

    def _build_roi_regions(self, width, height, active_zone):
        regions: List[Dict[str, Any]] = []
        top = 0
        bottom = int(height * self.traffic_light_max_y_ratio)
        for zone_name, cfg in self.traffic_light_regions.items():
            x1 = int(width * cfg["x_min_ratio"])
            x2 = int(width * cfg["x_max_ratio"])
            regions.append(
                {
                    "zone_name": zone_name,
                    "label": cfg["label"],
                    "box": (x1, top, x2, bottom),
                    "max_distance_m": cfg["max_distance"],
                    "active": zone_name == active_zone,
                }
            )
        return regions

    def get_last_debug_info(self):
        return self._last_debug_info

    def estimate_distance_optimized(self, class_name, box, height, width):
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1

        if box_height <= 0 or box_width <= 0:
            return float("inf")

        obj_params = self.reference_objects.get(class_name, {"real_h": 1.5, "real_w": 1.8})
        real_h = obj_params["real_h"]
        real_w = obj_params["real_w"]

        fov_deg = 90.0
        fov_rad = math.radians(fov_deg)
        focal_length = (width / 2) / math.tan(fov_rad / 2)

        d_height = (real_h * focal_length) / box_height
        d_width = (real_w * focal_length) / box_width
        return min(d_height, d_width)

    def detect_and_evaluate(self, raw_image, distance_threshold=5.0):
        if raw_image is None or raw_image.ndim != 3 or raw_image.shape[2] not in (3, 4):
            raise ValueError("raw_image phai la anh HxWx3 hoac HxWx4.")

        if raw_image.shape[2] == 4:
            img = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2BGR)
        else:
            img = raw_image.copy()

        height, width, _ = img.shape
        results = self.model(img, conf=self.conf_threshold, verbose=False)

        obstacle_emergency_flag = False
        processed_detections = []
        zone_signals: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {
            zone_name: {"red": None, "green": None}
            for zone_name in self.traffic_light_regions
        }

        if len(results) > 0:
            result = results[0]
            boxes = result.boxes

            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self._resolve_class_name(cls_id)

                if class_name not in self.display_classes:
                    continue

                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue

                xyxy = box.xyxy[0].cpu().numpy()
                distance = self.estimate_distance_optimized(class_name, xyxy, height, width)

                x1, y1, x2, y2 = map(int, xyxy)
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                roi_zone = None
                processed_detections.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "class_name": class_name,
                        "confidence": conf,
                        "distance": distance,
                        "center": (center_x, center_y),
                        "roi_zone": None,
                    }
                )

                if class_name in self.brake_classes and distance < distance_threshold:
                    obstacle_emergency_flag = True

                if class_name in self.traffic_light_classes:
                    roi_zone = self._classify_traffic_light_zone(center_x, center_y, width, height)
                    processed_detections[-1]["roi_zone"] = roi_zone
                    if roi_zone is None:
                        continue

                    color_key = "red" if class_name == self.red_light_class else "green"
                    candidate_signal = {
                        "class_name": class_name,
                        "distance": distance,
                        "confidence": conf,
                        "score": self._signal_score(conf, distance),
                        "box": (x1, y1, x2, y2),
                    }
                    zone_signals[roi_zone][color_key] = self._pick_better_signal(
                        zone_signals[roi_zone][color_key], candidate_signal
                    )

        traffic_decision = self._evaluate_traffic_light_decision(zone_signals)
        red_light_active = self._update_red_light_state(
            traffic_decision["red_trigger"],
            traffic_decision["green_release"],
        )
        emergency_flag = obstacle_emergency_flag or red_light_active

        if red_light_active and not self._last_red_light_active and traffic_decision["reason"]:
            print(f"[TRAFFIC_LIGHT] STOP | {traffic_decision['reason']}")
        if not red_light_active and self._last_red_light_active:
            print("[TRAFFIC_LIGHT] GO | release stop state.")
        self._last_red_light_active = red_light_active

        self._last_debug_info = {
            "roi_regions": self._build_roi_regions(
                width,
                height,
                traffic_decision["active_zone"],
            ),
            "active_roi_zone": traffic_decision["active_zone"],
            "obstacle_emergency": obstacle_emergency_flag,
            "red_light_frame_trigger": traffic_decision["red_trigger"],
            "green_light_frame_release": traffic_decision["green_release"],
            "red_light_active": red_light_active,
            "decision_reason": traffic_decision["reason"],
        }

        return processed_detections, emergency_flag
