from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

class SupervisorState(Enum):
    CRUISING = "cruising"
    STOPPING = "stopping"
    STOPPED = "stopped"
    RESUMING = "resuming"


class TrafficSupervisor:
    """
    Focused traffic supervisor:
    1. Stop for red lights in configured ROI zones.
    2. Emergency / strong brake for in-path obstacles inside yellow polygon.

    Public API is intentionally kept compatible with existing callers:
      - compute(...)
      - get_state()
      - get_debug_info()
    """

    def __init__(self, config: dict):
        self.config = dict(config or {})

        # Core thresholds.
        self.config.setdefault("confidence_threshold", 0.5)
        self.config.setdefault("red_light_distance_threshold", 30.0)
        self.config.setdefault("obstacle_distance_threshold", 5.0)
        self.config.setdefault("max_stopped_time", 30.0)

        # Red-light policy.
        self.config.setdefault("red_stopline_trigger_distance_m", 7.0)
        self.config.setdefault("red_stopline_approach_start_distance_m", 18.0)
        self.config.setdefault("red_stopline_approach_target_speed_kmh", 20.0)
        self.config.setdefault("red_stopline_approach_min_brake", 0.08)
        self.config.setdefault("red_stopline_approach_floor_brake_near", 0.35)
        self.config.setdefault("red_stopline_approach_max_brake", 0.95)
        self.config.setdefault("red_stopline_vehicle_max_decel_mps2", 8.0)
        self.config.setdefault("red_hard_stop_hold_seconds", 1.5)
        self.config.setdefault("red_hard_stop_min_brake", 1.0)
        self.config.setdefault("rural_red_trigger_distance_m", 8.0)
        self.config.setdefault("green_release_margin", 0.05)
        self.config.setdefault("green_immunity_frames", 10)
        self.config.setdefault("zone_release_missing_frames", 3)
        self.config.setdefault("stop_line_crawl_start_distance_m", 30.0)
        self.config.setdefault("stop_line_crawl_end_distance_m", 2.5)
        self.config.setdefault("stop_line_crawl_max_brake", 0.2)
        self.config.setdefault("stop_line_crawl_target_speed_kmh", 20.0)
        self.config.setdefault("stop_line_crawl_preview_brake", 0.03)

        # Obstacle brake profile.
        self.config.setdefault("obstacle_linear_full_brake_distance_m", 6.0)
        self.config.setdefault("obstacle_linear_max_brake", 0.5)
        self.config.setdefault("obstacle_linear_zero_brake_distance_m", 18.0)

        # Dynamic obstacle threshold model.
        self.obstacle_base_distance_m = 8.0
        self.obstacle_stop_min_distance_m = 5.5
        self.obstacle_stop_max_distance_m = 18.0
        self.obstacle_reaction_time_s = 0.45
        self.obstacle_assumed_decel_mps2 = 5.5
        self.obstacle_stop_margin_m = 1.5

        # Curved danger corridor model.
        self.path_wheelbase_m = 2.85
        self.path_max_steer_angle_deg = 35.0
        self.path_base_half_width_m = 1.1
        self.path_width_growth_per_m = 0.035
        self.path_curve_width_gain = 0.55
        self.path_max_half_width_m = 2.8
        self.path_min_forward_m = 0.8
        self.path_max_forward_m = 55.0
        self.path_horizon_base_m = 10.0
        self.path_horizon_speed_gain = 0.90
        self.path_horizon_steer_gain = 4.0

        self._obstacle_classes = {
            "vehicle",
            "pedestrian",
            "two_wheeler",
            "car",
            "truck",
            "person",
        }

        # Runtime state.
        self.state = SupervisorState.CRUISING
        self.stopped_time = 0.0
        self.frame_count = 0

        self.locked_zone: Optional[str] = None
        self.zone_missing_count = 0
        self.green_immunity_counter = 0

        self.last_danger_polygon: Optional[np.ndarray] = None

        # Debug channels expected by run_agents overlays.
        self._last_selected_target_type = "none"
        self._last_obstacle_reason = "none"
        self._last_obstacle_threshold_m = float(self.obstacle_base_distance_m)
        self._last_obstacle_in_path_count = 0
        self._last_obstacle_trigger: Optional[Dict[str, Any]] = None
        self._last_obstacle_linear_brake = 0.0
        self._last_final_brake = 0.0
        self._last_force_signal_brake_active = False
        self._last_force_signal_brake_target: Optional[str] = None
        self._green_release_active = False
        self._green_release_zone: Optional[str] = None
        self._last_red_stopline_distance_m = float("inf")
        self._last_red_approach_brake = 0.0
        self._red_hard_stop_latch_s = 0.0
        self._red_hard_stop_active = False
        self._last_stop_line_crawl_brake = 0.0
        self._last_stop_line_distance_m = float("inf")
        self._last_stop_line_mode = "none"

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _steer_to_curvature(self, vehicle_steer: Optional[float]) -> float:
        if vehicle_steer is None:
            return 0.0
        steer = self._clamp(float(vehicle_steer), -1.0, 1.0)
        wheel_angle = steer * math.radians(self.path_max_steer_angle_deg)
        curvature = math.tan(wheel_angle) / max(self.path_wheelbase_m, 1e-3)
        return float(self._clamp(curvature, -0.45, 0.45))

    def _curved_path_center_lateral(self, forward_m: float, vehicle_steer: Optional[float]) -> float:
        curvature = self._steer_to_curvature(vehicle_steer)
        return 0.5 * curvature * float(forward_m) * float(forward_m)

    def _curved_path_half_width(self, forward_m: float, vehicle_steer: Optional[float]) -> float:
        curvature = abs(self._steer_to_curvature(vehicle_steer))
        base = self.path_base_half_width_m + self.path_width_growth_per_m * max(0.0, forward_m)
        curve_bonus = self.path_curve_width_gain * curvature * max(0.0, forward_m)
        return float(min(self.path_max_half_width_m, base + curve_bonus))

    @staticmethod
    def _camera_to_vehicle_rotation(camera_pitch_deg: float) -> np.ndarray:
        pitch_rad = math.radians(-float(camera_pitch_deg))
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
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
        return rot_y @ base

    def _project_vehicle_to_image(
        self,
        point_vehicle: np.ndarray,
        frame_width: int,
        frame_height: int,
        camera_fov_deg: float,
        camera_mount_xyz: Tuple[float, float, float],
        camera_pitch_deg: float,
    ) -> Optional[Tuple[int, int]]:
        fx = (frame_width / 2.0) / max(math.tan(math.radians(camera_fov_deg) / 2.0), 1e-6)
        fy = fx
        cx = frame_width / 2.0
        cy = frame_height / 2.0

        r_c2v = self._camera_to_vehicle_rotation(camera_pitch_deg)
        r_v2c = r_c2v.T
        t = np.array(camera_mount_xyz, dtype=np.float32)
        p_rel = point_vehicle.astype(np.float32) - t
        p_cam = r_v2c @ p_rel

        if p_cam[2] <= 0.15:
            return None

        u = fx * (p_cam[0] / p_cam[2]) + cx
        v = fy * (p_cam[1] / p_cam[2]) + cy
        if not np.isfinite(u) or not np.isfinite(v):
            return None
        return int(round(u)), int(round(v))

    def _classify_traffic_light_zone(
        self,
        bbox: Tuple[int, int, int, int],
        image_shape: Optional[Tuple[int, int, int]] = None,
    ) -> Optional[str]:
        if bbox is None or len(bbox) < 4:
            return None
        if image_shape is None:
            image_shape = (480, 640, 3)

        img_h = int(image_shape[0])
        img_w = int(image_shape[1])
        if img_h <= 0 or img_w <= 0:
            return None

        x, y, w, h = [float(v) for v in bbox[:4]]
        x_center = x + 0.5 * w
        y_bottom = y + h

        x_ratio = x_center / float(img_w)
        y_bottom_ratio = y_bottom / float(img_h)

        # Keep aligned with run_agents overlay for top 60% band.
        if y_bottom_ratio > 0.60:
            return None

        if 0.35 <= x_ratio <= 0.65:
            return "urban"
        if 0.65 <= x_ratio <= 0.95:
            return "rural_right"
        return None

    def _build_obstacle_danger_polygon(
        self,
        image_shape: Optional[Tuple[int, int, int]] = None,
        vehicle_steer: float = 0.0,
        vehicle_speed_kmh: float = 0.0,
        camera_fov_deg: float = 90.0,
        camera_mount_xyz: Tuple[float, float, float] = (1.5, 0.0, 2.2),
        camera_pitch_deg: float = -8.0,
    ) -> Optional[np.ndarray]:
        if image_shape is None:
            image_shape = (480, 640, 3)

        frame_h = int(image_shape[0])
        frame_w = int(image_shape[1])
        if frame_h <= 0 or frame_w <= 0:
            return None

        speed_mps = max(0.0, float(vehicle_speed_kmh) / 3.6)
        steer_mag = min(1.0, abs(float(vehicle_steer)))
        horizon_m = (
            self.path_horizon_base_m
            + self.path_horizon_speed_gain * speed_mps
            + self.path_horizon_steer_gain * steer_mag
        )
        horizon_m = max(self.path_min_forward_m + 2.0, min(horizon_m, self.path_max_forward_m))

        sample_count = max(28, int(horizon_m * 2.5))
        forward_values = np.linspace(self.path_min_forward_m, horizon_m, sample_count, dtype=np.float32)

        left_pixels: List[Tuple[int, int]] = []
        right_pixels: List[Tuple[int, int]] = []

        for forward_m in forward_values:
            center_lateral = self._curved_path_center_lateral(float(forward_m), vehicle_steer)
            half_width = self._curved_path_half_width(float(forward_m), vehicle_steer)

            y_left = center_lateral - half_width
            y_right = center_lateral + half_width

            p_left = np.array([float(forward_m), float(y_left), 0.0], dtype=np.float32)
            p_right = np.array([float(forward_m), float(y_right), 0.0], dtype=np.float32)

            left_uv = self._project_vehicle_to_image(
                p_left,
                frame_w,
                frame_h,
                camera_fov_deg,
                camera_mount_xyz,
                camera_pitch_deg,
            )
            right_uv = self._project_vehicle_to_image(
                p_right,
                frame_w,
                frame_h,
                camera_fov_deg,
                camera_mount_xyz,
                camera_pitch_deg,
            )

            if left_uv is not None:
                left_pixels.append(left_uv)
            if right_uv is not None:
                right_pixels.append(right_uv)

        if len(left_pixels) < 4 or len(right_pixels) < 4:
            return None

        polygon_points = left_pixels + list(reversed(right_pixels))
        if len(polygon_points) < 3:
            return None

        polygon = np.array(polygon_points, dtype=np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, frame_w - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, frame_h - 1)
        return polygon

    @staticmethod
    def _point_in_polygon(point: Tuple[float, float], polygon: Optional[np.ndarray]) -> bool:
        if polygon is None:
            return False
        try:
            return cv2.pointPolygonTest(polygon, point, False) >= 0
        except Exception:
            return False

    @staticmethod
    def _estimate_distance_from_bbox(bbox: Tuple[int, int, int, int]) -> float:
        # Conservative fallback if metric distance is missing.
        _, _, w, h = [float(v) for v in bbox]
        pix = max(w, h, 1.0)
        return float(150.0 / pix)

    def _compute_obstacle_distance_threshold(
        self,
        distance_threshold: Optional[float],
        speed_kmh: Optional[float],
    ) -> float:
        base_threshold = float(self.obstacle_base_distance_m)
        if distance_threshold is not None:
            base_threshold = float(distance_threshold)

        if speed_kmh is None:
            dynamic_threshold = base_threshold
        else:
            speed_mps = max(0.0, float(speed_kmh) / 3.6)
            reaction_dist = speed_mps * self.obstacle_reaction_time_s
            braking_dist = (speed_mps * speed_mps) / max(2.0 * self.obstacle_assumed_decel_mps2, 0.1)
            dynamic_threshold = max(base_threshold, reaction_dist + braking_dist + self.obstacle_stop_margin_m)

        return float(
            max(
                self.obstacle_stop_min_distance_m,
                min(self.obstacle_stop_max_distance_m, dynamic_threshold),
            )
        )

    def _parse_detection(self, det: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        bbox = det.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None

        x, y, w, h = [int(v) for v in bbox[:4]]
        w = max(1, w)
        h = max(1, h)

        class_name = str(det.get("class_name", "")).strip().lower()
        confidence = float(det.get("confidence", 0.0))

        distance_m = float(det.get("distance_m", float("inf")))
        if not np.isfinite(distance_m):
            distance_m = self._estimate_distance_from_bbox((x, y, w, h))

        return {
            "class_name": class_name,
            "confidence": confidence,
            "bbox": (x, y, w, h),
            "distance_m": float(distance_m),
        }

    def _update_zone_lock(self, candidate_zone: Optional[str]) -> None:
        if candidate_zone is not None:
            self.locked_zone = str(candidate_zone)
            self.zone_missing_count = 0
            return

        if self.locked_zone is None:
            return

        self.zone_missing_count += 1
        release_after = max(1, int(self.config.get("zone_release_missing_frames", 3)))
        if self.zone_missing_count >= release_after:
            self.locked_zone = None
            self.zone_missing_count = 0

    def _select_red_signal(
        self,
        red_by_zone: Dict[str, Dict[str, Any]],
        green_by_zone: Dict[str, Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        urban_red = red_by_zone.get("urban")
        urban_green = green_by_zone.get("urban")
        rural_red = red_by_zone.get("rural_right")
        rural_green = green_by_zone.get("rural_right")

        release_margin = float(self.config.get("green_release_margin", 0.05))

        selected_red: Optional[Dict[str, Any]] = None
        active_zone: Optional[str] = None
        self._green_release_active = False
        self._green_release_zone = None

        if urban_red is not None or urban_green is not None:
            active_zone = "urban"
            if urban_green is not None and (
                urban_red is None
                or float(urban_green["confidence"]) >= float(urban_red["confidence"]) + release_margin
            ):
                self._green_release_active = True
                self._green_release_zone = "urban"
            else:
                selected_red = urban_red
        elif rural_red is not None or rural_green is not None:
            active_zone = "rural_right"
            if rural_green is not None and (
                rural_red is None
                or float(rural_green["confidence"]) >= float(rural_red["confidence"]) + release_margin
            ):
                self._green_release_active = True
                self._green_release_zone = "rural_right"
            else:
                selected_red = rural_red

        if self._green_release_active:
            immunity_frames = max(0, int(self.config.get("green_immunity_frames", 10)))
            self.green_immunity_counter = max(self.green_immunity_counter, immunity_frames)
            selected_red = None

        return selected_red, active_zone

    def _compute_red_brake(
        self,
        red_det: Optional[Dict[str, Any]],
        stop_line_distances: List[float],
        speed_kmh: float,
    ) -> Tuple[float, str]:
        self._last_red_stopline_distance_m = float("inf")
        self._last_red_approach_brake = 0.0
        if red_det is None:
            return 0.0, "none"

        zone = str(red_det.get("zone", ""))
        red_distance = float(red_det.get("distance_m", float("inf")))

        if zone == "urban":
            trigger_stopline_m = float(self.config.get("red_stopline_trigger_distance_m", 7.0))
            nearest_stopline = min(stop_line_distances) if stop_line_distances else float("inf")
            if np.isfinite(nearest_stopline):
                self._last_red_stopline_distance_m = float(nearest_stopline)
            if np.isfinite(nearest_stopline) and nearest_stopline <= trigger_stopline_m:
                return 1.0, "stop_line"

            # Pre-brake zone (15m -> 7m by default):
            # slow the vehicle down toward ~20km/h before entering hard-stop gate.
            approach_start_m = float(self.config.get("red_stopline_approach_start_distance_m", 15.0))
            approach_target_kmh = float(self.config.get("red_stopline_approach_target_speed_kmh", 20.0))
            approach_min_brake = float(self.config.get("red_stopline_approach_min_brake", 0.08))
            approach_floor_brake_near = float(self.config.get("red_stopline_approach_floor_brake_near", 0.35))
            approach_max_brake = float(self.config.get("red_stopline_approach_max_brake", 0.95))
            vehicle_max_decel = float(self.config.get("red_stopline_vehicle_max_decel_mps2", 8.0))
            vehicle_max_decel = max(0.1, vehicle_max_decel)

            if np.isfinite(nearest_stopline) and nearest_stopline <= approach_start_m:
                # Distance remaining until hard-stop gate at 7m.
                remaining_to_gate_m = max(0.1, nearest_stopline - trigger_stopline_m)

                speed_mps = max(0.0, float(speed_kmh) / 3.6)
                target_mps = max(0.0, float(approach_target_kmh) / 3.6)
                if speed_mps <= target_mps:
                    brake_from_kinematics = 0.0
                else:
                    req_decel = ((speed_mps * speed_mps) - (target_mps * target_mps)) / (2.0 * remaining_to_gate_m)
                    req_decel = max(0.0, req_decel)
                    brake_from_kinematics = req_decel / vehicle_max_decel

                # Distance floor prevents "too-late" weak brake near 7m.
                approach_range_m = max(0.5, approach_start_m - trigger_stopline_m)
                approach_progress = (approach_start_m - nearest_stopline) / approach_range_m
                approach_progress = self._clamp(float(approach_progress), 0.0, 1.0)
                brake_floor = approach_min_brake + (approach_floor_brake_near - approach_min_brake) * approach_progress

                approach_brake = max(float(brake_from_kinematics), float(brake_floor))
                approach_brake = float(self._clamp(approach_brake, 0.0, approach_max_brake))
                self._last_red_approach_brake = approach_brake
                return approach_brake, "red_light"

            # Fallback when stop-line detector misses but light is close in urban ROI.
            fallback_red_m = min(
                float(self.config.get("red_light_distance_threshold", 30.0)),
                12.0,
            )
            if np.isfinite(red_distance) and red_distance <= fallback_red_m:
                return 0.85, "red_light"
            return 0.0, "none"

        if zone == "rural_right":
            trigger_red_m = float(self.config.get("rural_red_trigger_distance_m", 8.0))
            if np.isfinite(red_distance) and red_distance <= trigger_red_m:
                return 1.0, "red_light"

        return 0.0, "none"

    def _compute_stop_line_crawl_brake(
        self,
        stop_line_distances: List[float],
        speed_kmh: float,
        has_red_signal: bool,
        has_green_signal: bool,
    ) -> float:
        self._last_stop_line_crawl_brake = 0.0
        self._last_stop_line_distance_m = float("inf")
        self._last_stop_line_mode = "none"

        # If there is a valid traffic-light decision, stop_line crawl is disabled.
        if has_red_signal:
            self._last_stop_line_mode = "disabled_by_red"
            return 0.0
        if has_green_signal:
            self._last_stop_line_mode = "disabled_by_green"
            return 0.0
        if not stop_line_distances:
            return 0.0

        nearest = min(float(d) for d in stop_line_distances if np.isfinite(float(d)))
        if not np.isfinite(nearest):
            return 0.0
        self._last_stop_line_distance_m = float(nearest)

        start_distance_m = float(self.config.get("stop_line_crawl_start_distance_m", 30.0))
        end_distance_m = float(self.config.get("stop_line_crawl_end_distance_m", 2.5))
        max_brake = float(self.config.get("stop_line_crawl_max_brake", 0.2))
        target_speed_kmh = float(self.config.get("stop_line_crawl_target_speed_kmh", 20.0))
        preview_brake = float(self.config.get("stop_line_crawl_preview_brake", 0.03))

        start_distance_m = max(1.0, start_distance_m)
        end_distance_m = max(0.5, min(end_distance_m, start_distance_m - 0.1))

        # No need to brake if vehicle is already at or below crawl speed target.
        if speed_kmh <= target_speed_kmh:
            self._last_stop_line_mode = "below_target_speed"
            return 0.0

        # Far preview: keep a tiny brake so driver stack clearly sees crawl intent.
        if nearest > start_distance_m:
            brake = float(self._clamp(preview_brake, 0.0, max_brake))
            self._last_stop_line_crawl_brake = brake
            self._last_stop_line_mode = "preview_far"
            return brake

        if nearest <= end_distance_m:
            dist_factor = 1.0
        else:
            dist_factor = (start_distance_m - nearest) / max(1e-6, start_distance_m - end_distance_m)
        dist_factor = self._clamp(float(dist_factor), 0.0, 1.0)

        overspeed_ratio = (float(speed_kmh) - target_speed_kmh) / max(1.0, target_speed_kmh)
        overspeed_factor = self._clamp(float(overspeed_ratio), 0.0, 1.0)
        if overspeed_factor > 0.0:
            overspeed_factor = max(0.25, overspeed_factor)

        brake = max_brake * dist_factor * overspeed_factor
        brake = float(self._clamp(brake, 0.0, max_brake))

        self._last_stop_line_crawl_brake = brake
        self._last_stop_line_mode = "active" if brake > 0.0 else "inactive"
        return brake

    def _compute_obstacle_brake(
        self,
        detections: List[Dict[str, Any]],
        danger_polygon: Optional[np.ndarray],
        speed_kmh: float,
        distance_threshold: Optional[float],
    ) -> float:
        self._last_obstacle_reason = "none"
        self._last_obstacle_trigger = None
        self._last_obstacle_in_path_count = 0

        threshold_m = self._compute_obstacle_distance_threshold(distance_threshold, speed_kmh)
        self._last_obstacle_threshold_m = float(threshold_m)

        if danger_polygon is None:
            self._last_obstacle_linear_brake = 0.0
            return 0.0

        nearest: Optional[Dict[str, Any]] = None
        min_distance_m = float("inf")

        for det in detections:
            class_name = str(det.get("class_name", "")).strip().lower()
            if class_name not in self._obstacle_classes:
                continue

            confidence = float(det.get("confidence", 0.0))
            if confidence < 0.25:
                continue

            bbox = det.get("bbox")
            if bbox is None or len(bbox) < 4:
                continue
            x, y, w, h = [int(v) for v in bbox[:4]]
            bottom_center = (int(x + w / 2.0), int(y + h))
            if not self._point_in_polygon(bottom_center, danger_polygon):
                continue

            self._last_obstacle_in_path_count += 1
            distance_m = float(det.get("distance_m", float("inf")))
            if not np.isfinite(distance_m):
                distance_m = self._estimate_distance_from_bbox((x, y, w, h))

            if distance_m < min_distance_m:
                min_distance_m = float(distance_m)
                nearest = {
                    "class_name": class_name,
                    "confidence": confidence,
                    "distance_m": float(distance_m),
                    "bbox": (x, y, w, h),
                }

        if nearest is None or not np.isfinite(min_distance_m):
            self._last_obstacle_linear_brake = 0.0
            return 0.0

        full_brake_dist = float(self.config.get("obstacle_linear_full_brake_distance_m", 6.0))
        linear_max_brake = float(self.config.get("obstacle_linear_max_brake", 0.5))
        zero_brake_dist = float(self.config.get("obstacle_linear_zero_brake_distance_m", 18.0))
        zero_brake_dist = max(full_brake_dist + 0.1, zero_brake_dist)

        if min_distance_m <= full_brake_dist:
            linear_brake = 1.0
        elif min_distance_m >= zero_brake_dist:
            linear_brake = 0.0
        else:
            ratio = (zero_brake_dist - min_distance_m) / (zero_brake_dist - full_brake_dist)
            linear_brake = float(linear_max_brake * self._clamp(float(ratio), 0.0, 1.0))

        obstacle_brake = float(self._clamp(linear_brake, 0.0, 1.0))

        # Emergency stop trigger for close in-path obstacle under dynamic threshold.
        if min_distance_m <= threshold_m:
            obstacle_brake = max(obstacle_brake, 0.6)
            self._last_obstacle_reason = (
                f"in-path {nearest['class_name']} {min_distance_m:.1f}m <= {threshold_m:.1f}m"
            )
        else:
            self._last_obstacle_reason = f"in-path {nearest['class_name']} {min_distance_m:.1f}m"

        if min_distance_m <= full_brake_dist:
            obstacle_brake = 1.0

        self._last_obstacle_trigger = nearest
        self._last_obstacle_linear_brake = float(self._clamp(obstacle_brake, 0.0, 1.0))
        return self._last_obstacle_linear_brake

    def _update_state(self, brake_force: float, current_speed: float, dt: float) -> None:
        brake_active = bool(brake_force > 1e-3)

        if brake_active:
            if float(current_speed) <= 0.1:
                self.state = SupervisorState.STOPPED
            else:
                self.state = SupervisorState.STOPPING
        else:
            if self.state == SupervisorState.STOPPED:
                self.state = SupervisorState.RESUMING
            else:
                self.state = SupervisorState.CRUISING

        if self.state == SupervisorState.STOPPED:
            self.stopped_time += max(0.0, float(dt))
        else:
            self.stopped_time = 0.0
            if self.state == SupervisorState.RESUMING:
                self.state = SupervisorState.CRUISING

        self.frame_count += 1

    def compute(
        self,
        detections: List[dict],
        current_speed: float,
        image_shape: Optional[Tuple[int, int, int]] = None,
        distance_threshold: Optional[float] = None,
        vehicle_steer: Optional[float] = None,
        dt: float = 0.033,
        danger_polygon: Optional[np.ndarray] = None,
    ) -> float:
        if image_shape is None:
            image_shape = (480, 640, 3)

        # Clear frame-local signal flags.
        self._last_force_signal_brake_active = False
        self._last_force_signal_brake_target = None

        speed_kmh = max(0.0, float(current_speed)) * 3.6
        steer_val = 0.0 if vehicle_steer is None else float(vehicle_steer)

        if self.green_immunity_counter > 0:
            self.green_immunity_counter -= 1

        if danger_polygon is None:
            danger_polygon = self._build_obstacle_danger_polygon(
                image_shape=image_shape,
                vehicle_steer=steer_val,
                vehicle_speed_kmh=speed_kmh,
            )
        self.last_danger_polygon = danger_polygon

        parsed: List[Dict[str, Any]] = []
        for det in detections or []:
            parsed_det = self._parse_detection(det)
            if parsed_det is not None:
                parsed.append(parsed_det)

        red_by_zone: Dict[str, Dict[str, Any]] = {}
        green_by_zone: Dict[str, Dict[str, Any]] = {}
        stop_line_distances: List[float] = []

        for det in parsed:
            class_name = str(det["class_name"])
            confidence = float(det["confidence"])
            bbox = det["bbox"]
            zone = self._classify_traffic_light_zone(bbox, image_shape)

            if class_name in {"traffic_light_red", "red_light"}:
                if confidence < 0.30:
                    continue
                if zone is None:
                    continue
                existing = red_by_zone.get(zone)
                if existing is None or float(det["confidence"]) > float(existing["confidence"]):
                    red_entry = dict(det)
                    red_entry["zone"] = zone
                    red_by_zone[zone] = red_entry
                continue

            if class_name in {"traffic_light_green", "green_light"}:
                if confidence < 0.30:
                    continue
                if zone is None:
                    continue
                existing = green_by_zone.get(zone)
                if existing is None or float(det["confidence"]) > float(existing["confidence"]):
                    green_entry = dict(det)
                    green_entry["zone"] = zone
                    green_by_zone[zone] = green_entry
                continue

            if class_name in {"stop_line", "stopline", "stop_line_marking"}:
                stop_line_distances.append(float(det["distance_m"]))

        selected_red, active_zone = self._select_red_signal(red_by_zone, green_by_zone)
        self._update_zone_lock(active_zone)

        red_brake, red_target = self._compute_red_brake(
            selected_red,
            stop_line_distances,
            speed_kmh=speed_kmh,
        )
        if red_brake > 0.0:
            self._last_force_signal_brake_active = True
            self._last_force_signal_brake_target = red_target

        stop_line_crawl_brake = self._compute_stop_line_crawl_brake(
            stop_line_distances=stop_line_distances,
            speed_kmh=speed_kmh,
            has_red_signal=(selected_red is not None),
            has_green_signal=bool(self._green_release_active),
        )

        obstacle_brake = self._compute_obstacle_brake(
            detections=parsed,
            danger_polygon=danger_polygon,
            speed_kmh=speed_kmh,
            distance_threshold=distance_threshold,
        )

        final_brake = 0.0
        selected_target = "none"

        if obstacle_brake >= red_brake and obstacle_brake > 0.0:
            final_brake = obstacle_brake
            selected_target = "obstacle"
        elif red_brake > 0.0:
            final_brake = red_brake
            selected_target = red_target
        elif stop_line_crawl_brake > 0.0:
            final_brake = stop_line_crawl_brake
            selected_target = "stop_line_crawl"

        # Red hard-stop latch:
        # keep brake command continuous even when detection flickers for a few frames.
        hold_seconds = max(0.0, float(self.config.get("red_hard_stop_hold_seconds", 1.5)))
        hold_min_brake = float(self._clamp(self.config.get("red_hard_stop_min_brake", 1.0), 0.0, 1.0))
        should_refresh_latch = bool(
            red_brake >= 0.99 and red_target == "stop_line"
        )
        if should_refresh_latch:
            self._red_hard_stop_latch_s = max(self._red_hard_stop_latch_s, hold_seconds)

        if self._red_hard_stop_latch_s > 0.0:
            final_brake = max(final_brake, hold_min_brake)
            if selected_target in ("none", "stop_line_crawl"):
                selected_target = "stop_line"
            self._red_hard_stop_latch_s = max(0.0, self._red_hard_stop_latch_s - max(1e-3, float(dt)))

        self._red_hard_stop_active = self._red_hard_stop_latch_s > 1e-6

        # Timeout escape to avoid permanent deadlock.
        max_stopped_time = float(self.config.get("max_stopped_time", 30.0))
        if self.state == SupervisorState.STOPPED and self.stopped_time > max_stopped_time:
            final_brake = 0.0
            selected_target = "none"
            self._last_obstacle_reason = "timeout_release"

        final_brake = float(self._clamp(final_brake, 0.0, 1.0))
        self._last_selected_target_type = selected_target
        self._last_final_brake = final_brake

        self._update_state(brake_force=final_brake, current_speed=float(current_speed), dt=float(dt))

        return final_brake

    def get_state(self) -> str:
        return self.state.value

    def get_debug_info(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "stopped_time": float(self.stopped_time),
            "frame_count": int(self.frame_count),
            "locked_zone": self.locked_zone,
            "zone_acquire_count": 0,
            "zone_missing_count": int(self.zone_missing_count),
            "in_turn_phase": False,
            "turn_hold_counter": 0,
            "turn_grace_counter": 0,
            "green_immunity_counter": int(self.green_immunity_counter),
            "selected_target_type": str(self._last_selected_target_type),
            "danger_polygon": self.last_danger_polygon,
            "danger_polygon_valid": self.last_danger_polygon is not None,
            "obstacle_emergency": bool(
                self._last_selected_target_type == "obstacle" and self._last_obstacle_linear_brake >= 0.6
            ),
            "obstacle_reason": str(self._last_obstacle_reason),
            "obstacle_threshold_m": float(self._last_obstacle_threshold_m),
            "obstacle_in_path_count": int(self._last_obstacle_in_path_count),
            "obstacle_trigger": self._last_obstacle_trigger,
            "obstacle_linear_brake": float(self._last_obstacle_linear_brake),
            "green_release_active": bool(self._green_release_active),
            "green_release_zone": self._green_release_zone,
            "red_stopline_distance_m": (
                None if not np.isfinite(self._last_red_stopline_distance_m) else float(self._last_red_stopline_distance_m)
            ),
            "red_stopline_approach_brake": float(self._last_red_approach_brake),
            "red_hard_stop_active": bool(self._red_hard_stop_active),
            "red_hard_stop_latch_s": float(self._red_hard_stop_latch_s),
            "stop_line_crawl_brake": float(self._last_stop_line_crawl_brake),
            "stop_line_distance_m": (
                None if not np.isfinite(self._last_stop_line_distance_m) else float(self._last_stop_line_distance_m)
            ),
            "stop_line_crawl_mode": str(self._last_stop_line_mode),
            "force_signal_brake_active": bool(self._last_force_signal_brake_active),
            "force_signal_brake_target": self._last_force_signal_brake_target,
            "prev_brake_force": float(self._last_final_brake),
        }
