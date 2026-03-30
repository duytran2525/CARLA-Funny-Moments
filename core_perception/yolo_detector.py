import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class YoloDetector:
    def __init__(
        self,
        model_path,
        conf_threshold=0.5,
        camera_fov_deg=90.0,
        obstacle_base_distance_m=8.0,
        camera_mount_x_m=1.5,
        camera_mount_y_m=0.0,
        camera_mount_z_m=2.2,
        camera_pitch_deg=-8.0,
    ):
        self.conf_threshold = conf_threshold
        self.camera_fov_deg = float(camera_fov_deg)
        self.obstacle_base_distance_m = float(obstacle_base_distance_m)
        self.camera_mount_x_m = float(camera_mount_x_m)
        self.camera_mount_y_m = float(camera_mount_y_m)
        self.camera_mount_z_m = float(camera_mount_z_m)
        self.camera_pitch_deg = float(camera_pitch_deg)
        self.obstacle_stop_min_distance_m = 5.5
        self.obstacle_stop_max_distance_m = 18.0
        self.obstacle_reaction_time_s = 0.45
        self.obstacle_assumed_decel_mps2 = 5.5
        self.obstacle_stop_margin_m = 1.5
        self.depth_valid_min_m = 0.3
        self.depth_valid_max_m = 120.0
        self.path_min_forward_m = 0.8
        self.path_max_forward_m = 35.0
        self.path_base_half_width_m = 1.10
        self.path_width_growth_per_m = 0.035
        self.path_max_half_width_m = 2.80
        self.path_curve_width_gain = 0.55
        self.path_wheelbase_m = 2.85
        self.path_max_steer_angle_deg = 35.0
        self.road_plane_update_interval = 5
        self.road_plane_sample_stride = 4
        self.road_plane_ransac_iters = 28
        self.road_plane_inlier_threshold_m = 0.10
        self.road_plane_min_points = 500
        self.road_plane_smooth_alpha = 0.20
        self._road_plane_frame_counter = 0
        self._road_plane_valid = False
        self._road_plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._road_plane_d = 0.0
        self._intrinsics_cache: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
        self._camera_to_vehicle_rot: Optional[np.ndarray] = None

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

    def _clamp(self, value, low, high):
        return max(low, min(high, value))

    def _get_intrinsics(self, width, height):
        key = (int(width), int(height))
        cached = self._intrinsics_cache.get(key)
        if cached is not None:
            return cached

        fov_x_rad = math.radians(self.camera_fov_deg)
        fx = (width / 2.0) / max(math.tan(fov_x_rad / 2.0), 1e-6)
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        intrinsics = (float(fx), float(fy), float(cx), float(cy))
        self._intrinsics_cache[key] = intrinsics
        return intrinsics

    def _camera_to_vehicle_rotation(self):
        if self._camera_to_vehicle_rot is not None:
            return self._camera_to_vehicle_rot
        pitch_rad = math.radians(-self.camera_pitch_deg)
        rot_y = np.array(
            [
                [math.cos(pitch_rad), 0.0, math.sin(pitch_rad)],
                [0.0, 1.0, 0.0],
                [-math.sin(pitch_rad), 0.0, math.cos(pitch_rad)],
            ],
            dtype=np.float32,
        )
        base = np.array(
            [
                [0.0, 0.0, 1.0],   # z_cam -> x_vehicle
                [1.0, 0.0, 0.0],   # x_cam -> y_vehicle
                [0.0, -1.0, 0.0],  # -y_cam -> z_vehicle
            ],
            dtype=np.float32,
        )
        self._camera_to_vehicle_rot = rot_y @ base
        return self._camera_to_vehicle_rot

    def _camera_points_to_vehicle(self, points_cam):
        if points_cam.size == 0:
            return points_cam
        rot = self._camera_to_vehicle_rotation()
        t = np.array(
            [self.camera_mount_x_m, self.camera_mount_y_m, self.camera_mount_z_m],
            dtype=np.float32,
        )
        return points_cam @ rot.T + t

    def _vehicle_ray_from_pixel(self, u, v, width, height):
        fx, fy, cx, cy = self._get_intrinsics(width, height)
        x = (float(u) - cx) / fx
        y = (float(v) - cy) / fy
        ray_cam = np.array([x, y, 1.0], dtype=np.float32)
        ray_cam = ray_cam / max(np.linalg.norm(ray_cam), 1e-6)
        rot = self._camera_to_vehicle_rotation()
        ray_vehicle = rot @ ray_cam
        return ray_vehicle / max(np.linalg.norm(ray_vehicle), 1e-6)

    def _intersect_ray_with_road_plane(self, ray_vehicle):
        n = self._road_plane_normal
        d = float(self._road_plane_d)
        origin = np.array(
            [self.camera_mount_x_m, self.camera_mount_y_m, self.camera_mount_z_m],
            dtype=np.float32,
        )
        denom = float(np.dot(n, ray_vehicle))
        if abs(denom) < 1e-6:
            return None
        scale = -(float(np.dot(n, origin)) + d) / denom
        if scale <= 0.0:
            return None
        point = origin + float(scale) * ray_vehicle
        return point

    def _fit_road_plane_ransac(self, points_vehicle):
        num_points = points_vehicle.shape[0]
        if num_points < 3:
            return None

        best_inliers = None
        best_count = 0

        for _ in range(self.road_plane_ransac_iters):
            ids = np.random.choice(num_points, size=3, replace=False)
            p1, p2, p3 = points_vehicle[ids]
            normal = np.cross(p2 - p1, p3 - p1)
            norm = float(np.linalg.norm(normal))
            if norm < 1e-6:
                continue
            normal = normal / norm
            if normal[2] < 0.0:
                normal = -normal
            if normal[2] < 0.65:
                continue

            d = -float(np.dot(normal, p1))
            distances = np.abs(points_vehicle @ normal + d)
            inliers = distances < self.road_plane_inlier_threshold_m
            count = int(np.count_nonzero(inliers))
            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_inliers is None or best_count < max(120, int(num_points * 0.12)):
            return None

        inlier_points = points_vehicle[best_inliers]
        centroid = np.mean(inlier_points, axis=0)
        centered = inlier_points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        if normal[2] < 0.0:
            normal = -normal
        d = -float(np.dot(normal, centroid))
        return normal.astype(np.float32), d

    def _update_road_plane_from_depth(self, depth_map_m, width, height):
        if depth_map_m is None or depth_map_m.ndim != 2:
            return

        self._road_plane_frame_counter += 1
        should_update = (
            (not self._road_plane_valid)
            or (self._road_plane_frame_counter % self.road_plane_update_interval == 0)
        )
        if not should_update:
            return

        h, w = depth_map_m.shape
        if h != int(height) or w != int(width):
            return

        y_start = int(h * 0.55)
        x_start = int(w * 0.10)
        x_end = int(w * 0.90)
        stride = max(1, int(self.road_plane_sample_stride))
        roi = depth_map_m[y_start:h:stride, x_start:x_end:stride]
        if roi.size == 0:
            return

        valid = np.isfinite(roi)
        valid &= roi >= self.depth_valid_min_m
        valid &= roi <= min(70.0, self.depth_valid_max_m)
        if int(np.count_nonzero(valid)) < self.road_plane_min_points:
            return

        rows, cols = np.where(valid)
        depths = roi[rows, cols]
        u = x_start + cols * stride
        v = y_start + rows * stride

        fx, fy, cx, cy = self._get_intrinsics(width, height)
        xn = (u.astype(np.float32) - cx) / fx
        yn = (v.astype(np.float32) - cy) / fy
        inv_norm = 1.0 / np.sqrt(xn * xn + yn * yn + 1.0)
        x_cam = depths * xn * inv_norm
        y_cam = depths * yn * inv_norm
        z_cam = depths * inv_norm
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)
        points_vehicle = self._camera_points_to_vehicle(points_cam)

        front_mask = points_vehicle[:, 0] > 0.5
        front_mask &= points_vehicle[:, 0] < 60.0
        front_mask &= np.abs(points_vehicle[:, 1]) < 12.0
        front_mask &= points_vehicle[:, 2] > -4.0
        front_mask &= points_vehicle[:, 2] < 3.0
        points_vehicle = points_vehicle[front_mask]
        if points_vehicle.shape[0] < self.road_plane_min_points:
            return

        fitted = self._fit_road_plane_ransac(points_vehicle)
        if fitted is None:
            return
        normal_new, d_new = fitted

        if not self._road_plane_valid:
            self._road_plane_normal = normal_new
            self._road_plane_d = float(d_new)
            self._road_plane_valid = True
            return

        alpha = self._clamp(self.road_plane_smooth_alpha, 0.05, 1.0)
        blended = (1.0 - alpha) * self._road_plane_normal + alpha * normal_new
        self._road_plane_normal = blended / max(np.linalg.norm(blended), 1e-6)
        self._road_plane_d = float((1.0 - alpha) * self._road_plane_d + alpha * d_new)

    def _steer_to_curvature(self, vehicle_steer):
        if vehicle_steer is None:
            return 0.0
        steer = self._clamp(float(vehicle_steer), -1.0, 1.0)
        wheel_angle = steer * math.radians(self.path_max_steer_angle_deg)
        curvature = math.tan(wheel_angle) / max(self.path_wheelbase_m, 1e-3)
        return float(self._clamp(curvature, -0.45, 0.45))

    def _compute_path_horizon(self, speed_kmh, stop_threshold_m):
        horizon = max(12.0, float(stop_threshold_m) + 6.0)
        if speed_kmh is not None:
            speed_mps = max(0.0, float(speed_kmh) / 3.6)
            horizon = max(horizon, speed_mps * 2.2 + 8.0)
        return float(self._clamp(horizon, 10.0, self.path_max_forward_m))

    def _curved_path_center_lateral(self, forward_m, vehicle_steer):
        curvature = self._steer_to_curvature(vehicle_steer)
        return 0.5 * curvature * float(forward_m) * float(forward_m)

    def _curved_path_half_width(self, forward_m, vehicle_steer):
        curvature = abs(self._steer_to_curvature(vehicle_steer))
        base = self.path_base_half_width_m + self.path_width_growth_per_m * max(0.0, forward_m)
        curve_bonus = self.path_curve_width_gain * curvature * max(0.0, forward_m)
        return float(min(self.path_max_half_width_m, base + curve_bonus))

    def _point_in_curved_path(self, forward_m, lateral_m, vehicle_steer, horizon_m):
        if not math.isfinite(forward_m) or not math.isfinite(lateral_m):
            return False
        if forward_m < self.path_min_forward_m or forward_m > horizon_m:
            return False
        center = self._curved_path_center_lateral(forward_m, vehicle_steer)
        half_w = self._curved_path_half_width(forward_m, vehicle_steer)
        return abs(float(lateral_m) - center) <= half_w

    def _estimate_ground_point_vehicle(self, center_x, y_bottom, width, height):
        ray_vehicle = self._vehicle_ray_from_pixel(center_x, y_bottom, width, height)
        point_vehicle = self._intersect_ray_with_road_plane(ray_vehicle)
        if point_vehicle is None:
            return None
        return (float(point_vehicle[0]), float(point_vehicle[1]), float(point_vehicle[2]))

    def _compute_obstacle_distance_threshold(self, distance_threshold, speed_kmh):
        base_threshold = self.obstacle_base_distance_m
        if distance_threshold is not None:
            base_threshold = float(distance_threshold)

        if speed_kmh is None:
            dynamic_threshold = base_threshold
        else:
            speed_mps = max(0.0, float(speed_kmh) / 3.6)
            reaction_dist = speed_mps * self.obstacle_reaction_time_s
            braking_dist = (speed_mps * speed_mps) / max(
                2.0 * self.obstacle_assumed_decel_mps2, 0.1
            )
            dynamic_threshold = max(
                base_threshold,
                reaction_dist + braking_dist + self.obstacle_stop_margin_m,
            )

        return float(
            max(
                self.obstacle_stop_min_distance_m,
                min(self.obstacle_stop_max_distance_m, dynamic_threshold),
            )
        )

    def _depth_patch_distance(self, depth_map_m, x1, y1, x2, y2, prefer_nearest=False):
        if depth_map_m is None:
            return float("inf")
        if depth_map_m.ndim != 2:
            return float("inf")

        h, w = depth_map_m.shape
        x1i = max(0, min(w - 1, int(x1)))
        x2i = max(0, min(w, int(x2)))
        y1i = max(0, min(h - 1, int(y1)))
        y2i = max(0, min(h, int(y2)))
        if x2i <= x1i or y2i <= y1i:
            return float("inf")

        patch = depth_map_m[y1i:y2i, x1i:x2i]
        if patch.size == 0:
            return float("inf")

        valid = patch[np.isfinite(patch)]
        if valid.size == 0:
            return float("inf")
        valid = valid[
            (valid >= self.depth_valid_min_m) & (valid <= self.depth_valid_max_m)
        ]
        if valid.size == 0:
            return float("inf")

        percentile = 30.0 if prefer_nearest else 50.0
        return float(np.percentile(valid, percentile))

    def estimate_distance_from_depth(self, class_name, box, depth_map_m):
        if depth_map_m is None:
            return float("inf")
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 2 or box_h <= 2:
            return float("inf")

        if class_name in self.brake_classes:
            center_x = (x1 + x2) * 0.5
            patch_half_w = max(2.0, box_w * 0.12)
            patch_top = y1 + box_h * 0.72
            patch_bottom = y2
            return self._depth_patch_distance(
                depth_map_m,
                center_x - patch_half_w,
                patch_top,
                center_x + patch_half_w,
                patch_bottom,
                prefer_nearest=True,
            )

        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        patch_half_w = max(2.0, box_w * 0.20)
        patch_half_h = max(2.0, box_h * 0.20)
        return self._depth_patch_distance(
            depth_map_m,
            center_x - patch_half_w,
            center_y - patch_half_h,
            center_x + patch_half_w,
            center_y + patch_half_h,
            prefer_nearest=False,
        )

    def estimate_distance_optimized(self, class_name, box, height, width):
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1

        if box_height <= 0 or box_width <= 0:
            return float("inf")

        obj_params = self.reference_objects.get(class_name, {"real_h": 1.5, "real_w": 1.8})
        real_h = obj_params["real_h"]
        real_w = obj_params["real_w"]

        fov_deg = self.camera_fov_deg
        fov_rad = math.radians(fov_deg)
        focal_length = (width / 2) / math.tan(fov_rad / 2)

        d_height = (real_h * focal_length) / box_height
        d_width = (real_w * focal_length) / box_width
        return min(d_height, d_width)

    def detect_and_evaluate(
        self,
        raw_image,
        distance_threshold=None,
        depth_map_m=None,
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
        effective_obstacle_threshold = self._compute_obstacle_distance_threshold(
            distance_threshold,
            speed_kmh,
        )
        self._update_road_plane_from_depth(depth_map_m, width, height)
        path_horizon_m = self._compute_path_horizon(speed_kmh, effective_obstacle_threshold)
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
                distance_from_depth = self.estimate_distance_from_depth(
                    class_name,
                    xyxy,
                    depth_map_m,
                )
                distance = distance_from_depth
                distance_source = "depth"
                if not math.isfinite(distance):
                    distance = self.estimate_distance_optimized(
                        class_name,
                        xyxy,
                        height,
                        width,
                    )
                    distance_source = "bbox"

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
                        "distance_source": distance_source,
                        "center": (center_x, center_y),
                        "roi_zone": None,
                        "ground_point": None,
                        "ground_point_vehicle": None,
                        "in_danger_roi": False,
                        "path_check_mode": "none",
                        "danger_match": False,
                    }
                )

                if class_name in self.brake_classes:
                    ground_point = (center_x, float(y2))
                    ground_point_vehicle = self._estimate_ground_point_vehicle(
                        center_x,
                        float(y2),
                        width,
                        height,
                    )
                    in_danger_roi = False
                    path_check_mode = "legacy_2d"
                    if ground_point_vehicle is not None:
                        forward_m, lateral_m, _ = ground_point_vehicle
                        in_danger_roi = self._point_in_curved_path(
                            forward_m,
                            lateral_m,
                            vehicle_steer,
                            path_horizon_m,
                        )
                        path_check_mode = "curved_3d"
                    else:
                        in_danger_roi = self._point_in_polygon(
                            ground_point[0],
                            ground_point[1],
                            obstacle_danger_polygon,
                        )
                    processed_detections[-1]["ground_point"] = ground_point
                    processed_detections[-1]["ground_point_vehicle"] = ground_point_vehicle
                    processed_detections[-1]["in_danger_roi"] = in_danger_roi
                    processed_detections[-1]["path_check_mode"] = path_check_mode

                    if in_danger_roi and distance < effective_obstacle_threshold:
                        obstacle_emergency_flag = True
                        processed_detections[-1]["danger_match"] = True
                        if (
                            obstacle_trigger_candidate is None
                            or distance < obstacle_trigger_candidate["distance"]
                        ):
                            forward_hint = None
                            if ground_point_vehicle is not None:
                                forward_hint = float(ground_point_vehicle[0])
                            obstacle_trigger_candidate = {
                                "class_name": class_name,
                                "distance": distance,
                                "forward_m": forward_hint,
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
            if obstacle_trigger_candidate.get("forward_m") is not None:
                obstacle_reason = (
                    f"{obstacle_reason} "
                    f"(x={obstacle_trigger_candidate['forward_m']:.1f}m)"
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
                "distance_threshold_m": effective_obstacle_threshold,
                "polygon": obstacle_danger_polygon,
            },
            "obstacle_path_model": {
                "mode": "curved_3d",
                "steer": 0.0 if vehicle_steer is None else float(vehicle_steer),
                "curvature": float(self._steer_to_curvature(vehicle_steer)),
                "horizon_m": path_horizon_m,
                "min_forward_m": self.path_min_forward_m,
                "base_half_width_m": self.path_base_half_width_m,
                "width_growth_per_m": self.path_width_growth_per_m,
                "curve_width_gain": self.path_curve_width_gain,
                "max_half_width_m": self.path_max_half_width_m,
                "wheelbase_m": self.path_wheelbase_m,
                "max_steer_angle_deg": self.path_max_steer_angle_deg,
            },
            "road_plane": {
                "valid": bool(self._road_plane_valid),
                "normal": [
                    float(self._road_plane_normal[0]),
                    float(self._road_plane_normal[1]),
                    float(self._road_plane_normal[2]),
                ],
                "d": float(self._road_plane_d),
                "update_interval_frames": int(self.road_plane_update_interval),
            },
            "depth_available": depth_map_m is not None,
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
