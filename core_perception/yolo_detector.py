import math
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
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

        # Active ROI lock: only one zone is allowed to control stop/go at a time.
        self.zone_lock_acquire_frames = 2
        self.zone_lock_release_missing_frames = 4
        self.locked_zone: Optional[str] = None
        self._last_locked_zone: Optional[str] = None
        self._zone_lock_missing_counter = 0
        self._zone_lock_acquire_counter = {
            zone_name: 0 for zone_name in self.traffic_light_regions
        }

        # After a stable green signal in locked zone, ignore other-zone takeover briefly.
        self.green_immunity_frames = 10
        self._green_immunity_counter = 0
        self._green_immunity_zone: Optional[str] = None

        # Turn-phase suppression:
        # if vehicle is actively turning and recently observed GREEN in the same locked zone,
        # temporarily suppress RED triggers likely belonging to the cross lane after camera rotates.
        self.turn_steer_threshold = 0.22
        self.turn_speed_threshold_kmh = 5.0
        self.turn_confirm_frames = 3
        self.turn_hold_frames = 18
        self.turn_green_grace_frames = 30
        self._turn_confirm_counter = 0
        self._turn_hold_counter = 0
        self._turn_active = False
        self._turn_green_grace_counter = 0
        self._turn_green_grace_zone: Optional[str] = None

        # Trapezoid danger corridor in front of ego vehicle for obstacle emergency brake.
        self.obstacle_danger_region = {
            "top_y_ratio": 0.58,
            "bottom_y_ratio": 0.98,
            "top_half_width_ratio": 0.10,
            "bottom_half_width_ratio": 0.34,
            "label": "Obstacle corridor",
        }

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

    def _evaluate_zone_signal(self, zone_signals, zone_name):
        zone_cfg = self.traffic_light_regions[zone_name]
        zone_data = zone_signals.get(zone_name, {})
        red_signal = zone_data.get("red")
        green_signal = zone_data.get("green")
        red_in_range = self._candidate_in_zone_range(red_signal, zone_name)
        green_in_range = self._candidate_in_zone_range(green_signal, zone_name)

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
            "zone_name": zone_name,
            "has_signal": red_in_range or green_in_range,
            "red_trigger": red_trigger,
            "green_release": green_release,
            "reason": reason,
        }

    def _select_zone_candidate_for_lock(self, zone_evaluations):
        for zone_name in self.traffic_light_zone_priority:
            zone_eval = zone_evaluations.get(zone_name)
            if not zone_eval or not zone_eval["has_signal"]:
                continue

            # During green immunity, do not allow another zone to take control.
            if (
                self._green_immunity_counter > 0
                and self._green_immunity_zone is not None
                and zone_name != self._green_immunity_zone
            ):
                continue
            return zone_name
        return None

    def _update_locked_zone(self, zone_evaluations):
        if self._green_immunity_counter > 0:
            self._green_immunity_counter -= 1
            if self._green_immunity_counter <= 0:
                self._green_immunity_counter = 0
                self._green_immunity_zone = None

        # Keep current lock while its signal is still visible.
        if self.locked_zone is not None:
            locked_eval = zone_evaluations.get(self.locked_zone)
            if locked_eval and locked_eval["has_signal"]:
                self._zone_lock_missing_counter = 0
                return self.locked_zone

            self._zone_lock_missing_counter += 1
            if self._zone_lock_missing_counter < self.zone_lock_release_missing_frames:
                return self.locked_zone

            self._last_locked_zone = self.locked_zone
            self.locked_zone = None
            self._zone_lock_missing_counter = 0
            for zone_name in self._zone_lock_acquire_counter:
                self._zone_lock_acquire_counter[zone_name] = 0

        # Acquire lock when no active lock is available.
        candidate_zone = self._select_zone_candidate_for_lock(zone_evaluations)
        if candidate_zone is None:
            for zone_name in self._zone_lock_acquire_counter:
                self._zone_lock_acquire_counter[zone_name] = 0
            return None

        for zone_name in self._zone_lock_acquire_counter:
            if zone_name == candidate_zone:
                self._zone_lock_acquire_counter[zone_name] += 1
            else:
                self._zone_lock_acquire_counter[zone_name] = 0

        if self._zone_lock_acquire_counter[candidate_zone] >= self.zone_lock_acquire_frames:
            self.locked_zone = candidate_zone
            self._last_locked_zone = candidate_zone
            self._zone_lock_missing_counter = 0
            return self.locked_zone

        return None

    def _update_turn_phase(self, vehicle_steer, speed_kmh):
        if vehicle_steer is None:
            strong_turn = False
        else:
            speed_ok = True if speed_kmh is None else float(speed_kmh) >= self.turn_speed_threshold_kmh
            strong_turn = abs(float(vehicle_steer)) >= self.turn_steer_threshold and speed_ok

        if strong_turn:
            self._turn_confirm_counter = min(
                self._turn_confirm_counter + 1, self.turn_confirm_frames + 10
            )
        else:
            self._turn_confirm_counter = max(self._turn_confirm_counter - 1, 0)

        if not self._turn_active and self._turn_confirm_counter >= self.turn_confirm_frames:
            self._turn_active = True
            self._turn_hold_counter = self.turn_hold_frames

        if self._turn_active:
            if strong_turn:
                self._turn_hold_counter = self.turn_hold_frames
            else:
                self._turn_hold_counter -= 1

            if self._turn_hold_counter <= 0:
                self._turn_active = False
                self._turn_hold_counter = 0

        if self._turn_green_grace_counter > 0:
            self._turn_green_grace_counter -= 1
            if self._turn_green_grace_counter <= 0:
                self._turn_green_grace_counter = 0
                self._turn_green_grace_zone = None

        return self._turn_active

    def _evaluate_traffic_light_decision(self, zone_signals, turn_active=False):
        zone_evaluations = {
            zone_name: self._evaluate_zone_signal(zone_signals, zone_name)
            for zone_name in self.traffic_light_regions
        }

        active_zone = self._update_locked_zone(zone_evaluations)
        if active_zone is None:
            return {
                "active_zone": None,
                "red_trigger": False,
                "green_release": False,
                "reason": "",
                "locked_zone": None,
                "green_immunity_counter": self._green_immunity_counter,
                "green_immunity_zone": self._green_immunity_zone,
                "turn_phase_active": turn_active,
                "turn_red_suppressed": False,
                "turn_green_grace_counter": self._turn_green_grace_counter,
                "turn_green_grace_zone": self._turn_green_grace_zone,
                "zone_evaluations": zone_evaluations,
            }

        zone_eval = zone_evaluations.get(active_zone, {})
        red_trigger = bool(zone_eval.get("red_trigger", False))
        green_release = bool(zone_eval.get("green_release", False))
        reason = zone_eval.get("reason", "")
        turn_red_suppressed = False

        if green_release:
            self._green_immunity_counter = self.green_immunity_frames
            self._green_immunity_zone = active_zone
            self._turn_green_grace_counter = self.turn_green_grace_frames
            self._turn_green_grace_zone = active_zone

        if (
            red_trigger
            and turn_active
            and self._turn_green_grace_counter > 0
            and self._turn_green_grace_zone == active_zone
        ):
            red_trigger = False
            green_release = True
            turn_red_suppressed = True
            zone_label = self.traffic_light_regions[active_zone]["label"]
            reason = (
                f"{zone_label}: RED suppressed while turning "
                f"(recent GREEN context)"
            )

        return {
            "active_zone": active_zone,
            "red_trigger": red_trigger,
            "green_release": green_release,
            "reason": reason,
            "locked_zone": self.locked_zone,
            "green_immunity_counter": self._green_immunity_counter,
            "green_immunity_zone": self._green_immunity_zone,
            "turn_phase_active": turn_active,
            "turn_red_suppressed": turn_red_suppressed,
            "turn_green_grace_counter": self._turn_green_grace_counter,
            "turn_green_grace_zone": self._turn_green_grace_zone,
            "zone_evaluations": zone_evaluations,
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

    def _build_obstacle_danger_polygon(self, width, height):
        cfg = self.obstacle_danger_region
        center_x = width / 2.0
        top_y = max(0, min(height - 1, int(height * cfg["top_y_ratio"])))
        bottom_y = max(top_y + 1, min(height - 1, int(height * cfg["bottom_y_ratio"])))
        top_half_w = max(1, int(width * cfg["top_half_width_ratio"]))
        bottom_half_w = max(top_half_w + 1, int(width * cfg["bottom_half_width_ratio"]))

        polygon = [
            (int(center_x - top_half_w), top_y),
            (int(center_x + top_half_w), top_y),
            (int(center_x + bottom_half_w), bottom_y),
            (int(center_x - bottom_half_w), bottom_y),
        ]

        clamped = []
        for x, y in polygon:
            clamped.append(
                (
                    max(0, min(width - 1, int(x))),
                    max(0, min(height - 1, int(y))),
                )
            )
        return clamped

    def _point_in_polygon(self, point_x, point_y, polygon):
        if len(polygon) < 3:
            return False
        contour = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        # >= 0 means inside or on the boundary.
        return cv2.pointPolygonTest(contour, (float(point_x), float(point_y)), False) >= 0

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

    def detect_and_evaluate(
        self,
        raw_image,
        distance_threshold=5.0,
        vehicle_steer=None,
        speed_kmh=None,
    ):
        if raw_image is None or raw_image.ndim != 3 or raw_image.shape[2] not in (3, 4):
            raise ValueError("raw_image phai la anh HxWx3 hoac HxWx4.")

        if raw_image.shape[2] == 4:
            img = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2BGR)
        else:
            img = raw_image.copy()

        height, width, _ = img.shape
        results = self.model(img, conf=self.conf_threshold, verbose=False)

        obstacle_emergency_flag = False
        obstacle_trigger_candidate = None
        processed_detections = []
        obstacle_danger_polygon = self._build_obstacle_danger_polygon(width, height)
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
                        "ground_point": None,
                        "in_danger_roi": False,
                        "danger_match": False,
                    }
                )

                if class_name in self.brake_classes:
                    ground_point = (center_x, float(y2))
                    in_danger_roi = self._point_in_polygon(
                        ground_point[0],
                        ground_point[1],
                        obstacle_danger_polygon,
                    )
                    processed_detections[-1]["ground_point"] = ground_point
                    processed_detections[-1]["in_danger_roi"] = in_danger_roi

                    if in_danger_roi and distance < distance_threshold:
                        obstacle_emergency_flag = True
                        processed_detections[-1]["danger_match"] = True
                        if (
                            obstacle_trigger_candidate is None
                            or distance < obstacle_trigger_candidate["distance"]
                        ):
                            obstacle_trigger_candidate = {
                                "class_name": class_name,
                                "distance": distance,
                            }

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

        turn_phase_active = self._update_turn_phase(vehicle_steer, speed_kmh)
        traffic_decision = self._evaluate_traffic_light_decision(
            zone_signals,
            turn_active=turn_phase_active,
        )
        red_light_active = self._update_red_light_state(
            traffic_decision["red_trigger"],
            traffic_decision["green_release"],
        )
        emergency_flag = obstacle_emergency_flag or red_light_active
        obstacle_reason = ""
        if obstacle_trigger_candidate is not None:
            obstacle_reason = (
                f"In-path obstacle: {obstacle_trigger_candidate['class_name']} "
                f"@ {obstacle_trigger_candidate['distance']:.1f}m"
            )
        decision_reason = traffic_decision["reason"] or ""
        if obstacle_reason:
            decision_reason = f"{decision_reason} | {obstacle_reason}" if decision_reason else obstacle_reason

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
            "obstacle_danger_roi": {
                "label": self.obstacle_danger_region["label"],
                "distance_threshold_m": distance_threshold,
                "polygon": obstacle_danger_polygon,
            },
            "active_roi_zone": traffic_decision["active_zone"],
            "locked_zone": traffic_decision.get("locked_zone"),
            "last_locked_zone": self._last_locked_zone,
            "zone_lock_acquire_counter": dict(self._zone_lock_acquire_counter),
            "zone_lock_missing_counter": self._zone_lock_missing_counter,
            "zone_lock_acquire_frames": self.zone_lock_acquire_frames,
            "zone_lock_release_missing_frames": self.zone_lock_release_missing_frames,
            "obstacle_emergency": obstacle_emergency_flag,
            "obstacle_reason": obstacle_reason,
            "red_light_frame_trigger": traffic_decision["red_trigger"],
            "green_light_frame_release": traffic_decision["green_release"],
            "red_light_active": red_light_active,
            "green_immunity_counter": traffic_decision.get("green_immunity_counter", 0),
            "green_immunity_zone": traffic_decision.get("green_immunity_zone"),
            "turn_phase_active": traffic_decision.get("turn_phase_active", False),
            "turn_red_suppressed": traffic_decision.get("turn_red_suppressed", False),
            "turn_confirm_counter": self._turn_confirm_counter,
            "turn_hold_counter": self._turn_hold_counter,
            "turn_green_grace_counter": traffic_decision.get("turn_green_grace_counter", 0),
            "turn_green_grace_zone": traffic_decision.get("turn_green_grace_zone"),
            "zone_evaluations": traffic_decision.get("zone_evaluations", {}),
            "decision_reason": decision_reason,
        }

        return processed_detections, emergency_flag
