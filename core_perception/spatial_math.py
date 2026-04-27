from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, fov_deg: float) -> "CameraIntrinsics":
        width_f = float(width)
        height_f = float(height)
        fov_rad = math.radians(float(fov_deg))
        fx = (width_f / 2.0) / max(math.tan(fov_rad / 2.0), 1e-6)
        fy = fx
        cx = width_f / 2.0
        cy = height_f / 2.0
        return cls(
            width=int(width),
            height=int(height),
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
        )


class DynamicIPM:
    """
    Dynamic Inverse Perspective Mapping using runtime pitch/roll.

    Workflow:
    tracked objects -> project_to_bev(...) -> object list with distance + velocity.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        camera_mount_x_m: float = 1.5,
        camera_mount_y_m: float = 0.0,
        camera_mount_z_m: float = 2.2,
        base_pitch_deg: float = -8.0,
        base_roll_deg: float = 0.0,
        max_distance_m: float = 120.0,
    ) -> None:
        self.intrinsics = intrinsics
        self.camera_mount_x_m = float(camera_mount_x_m)
        self.camera_mount_y_m = float(camera_mount_y_m)
        self.camera_mount_z_m = float(camera_mount_z_m)
        self.base_pitch_deg = float(base_pitch_deg)
        self.base_roll_deg = float(base_roll_deg)
        self.max_distance_m = float(max_distance_m)

        self._homography: Optional[np.ndarray] = None
        self._last_h_signature: Optional[Tuple[float, float]] = None
        self._track_history: Dict[int, Tuple[float, float, float]] = {}

    def project_to_bev(
        self,
        tracked_objects: Any,
        pitch: float,
        roll: float,
        ego_speed_kmh: float = 0.0,
        timestamp: Optional[float] = None,
    ):
        """
        Input tracked objects:
            {"class","bbox","conf","track_id", ...}

        Output objects (extended):
            {"class","bbox","conf","track_id","distance_m","velocity_kmh", ...}
        """
        ts = time.time() if timestamp is None else float(timestamp)
        self._update_homography(float(pitch), float(roll))

        single_input = False
        if isinstance(tracked_objects, dict):
            object_list: List[Dict[str, Any]] = [tracked_objects]
            single_input = True
        elif (
            isinstance(tracked_objects, (list, tuple))
            and len(tracked_objects) == 4
            and all(isinstance(v, (int, float, np.number)) for v in tracked_objects)
        ):
            object_list = [
                {
                    "class": "unknown",
                    "bbox": list(tracked_objects),
                    "conf": 1.0,
                    "track_id": None,
                }
            ]
            single_input = True
        else:
            object_list = list(tracked_objects)

        projected: List[Dict[str, Any]] = []
        
        # Chiều cao thực tế giả định (m) cho các vật thể lơ lửng trên không
        elevated_real_heights = {
            "traffic_light_red": 0.8,
            "traffic_light_green": 0.8,
            "traffic_light": 0.8,
            "traffic_sign": 0.7,
            "stop_sign": 0.7,
        }

        for obj in object_list:
            bbox = obj.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = [float(v) for v in bbox]
            class_name = str(obj.get("class", "unknown")).lower()

            if class_name in elevated_real_heights:
                # 1. Pinhole Fallback cho vật thể lơ lửng
                real_h = elevated_real_heights[class_name]
                bbox_h = max(1.0, y2 - y1)
                distance_m = float((self.intrinsics.fy * real_h) / bbox_h)
                
                # 2. Ước tính X, Y trong không gian xe
                center_u = 0.5 * (x1 + x2)
                center_v = 0.5 * (y1 + y2)
                ray_cam = np.array([
                    (center_u - self.intrinsics.cx) / max(self.intrinsics.fx, 1e-6),
                    (center_v - self.intrinsics.cy) / max(self.intrinsics.fy, 1e-6),
                    1.0
                ], dtype=np.float64)
                
                p_cam = ray_cam * distance_m
                rot = self._camera_to_vehicle_rotation(float(pitch), float(roll))
                p_veh = rot @ p_cam
                
                x_m, y_m = float(p_veh[0]), float(p_veh[1])
                bev_xy_m = [x_m, y_m]
                rel_velocity_kmh = self._estimate_relative_speed(obj.get("track_id"), x_m, y_m, ts)
            else:
                # Xử lý mặc định (IPM) cho vật thể chạm đất (xe, người...)
                bottom_center_u = 0.5 * (x1 + x2)
                bottom_center_v = y2

                ground_xy = self._project_pixel_to_ground(
                    bottom_center_u, bottom_center_v, float(pitch), float(roll)
                )
                if ground_xy is None:
                    distance_m = float("inf")
                    bev_xy_m = [float("nan"), float("nan")]
                    rel_velocity_kmh = 0.0
                else:
                    x_m, y_m = ground_xy
                    distance_m = float(math.hypot(x_m, y_m))
                    bev_xy_m = [float(x_m), float(y_m)]
                    rel_velocity_kmh = self._estimate_relative_speed(obj.get("track_id"), x_m, y_m, ts)

            enriched = dict(obj)
            enriched["distance_m"] = distance_m
            enriched["velocity_kmh"] = float(ego_speed_kmh)
            enriched["bev_xy_m"] = bev_xy_m
            enriched["relative_velocity_kmh"] = float(rel_velocity_kmh)
            projected.append(enriched)

        if single_input:
            return projected[0] if projected else {}
        return projected

    def _update_homography(self, pitch: float, roll: float) -> None:
        sig = (round(float(pitch), 5), round(float(roll), 5))
        if self._last_h_signature == sig:
            return
        self._last_h_signature = sig

        if cv2 is None:
            self._homography = None
            return

        w = self.intrinsics.width
        h = self.intrinsics.height
        image_pts = np.array(
            [
                [0.0, h - 1.0],
                [w - 1.0, h - 1.0],
                [0.0, h * 0.62],
                [w - 1.0, h * 0.62],
            ],
            dtype=np.float32,
        )

        world_pts: List[List[float]] = []
        for u, v in image_pts:
            projected = self._ray_intersection_to_ground(float(u), float(v), pitch, roll)
            if projected is None:
                self._homography = None
                return
            world_pts.append([projected[0], projected[1]])

        world_arr = np.array(world_pts, dtype=np.float32)
        h_mat = cv2.getPerspectiveTransform(image_pts, world_arr)
        if not np.all(np.isfinite(h_mat)):
            self._homography = None
            return
        self._homography = h_mat

    def _project_pixel_to_ground(
        self,
        u: float,
        v: float,
        pitch: float,
        roll: float,
    ) -> Optional[Tuple[float, float]]:
        if self._homography is not None:
            uv1 = np.array([float(u), float(v), 1.0], dtype=np.float64)
            mapped = self._homography @ uv1
            if abs(float(mapped[2])) > 1e-6:
                x_m = float(mapped[0] / mapped[2])
                y_m = float(mapped[1] / mapped[2])
                if (
                    math.isfinite(x_m)
                    and math.isfinite(y_m)
                    and 0.0 <= x_m <= self.max_distance_m
                ):
                    return x_m, y_m

        return self._ray_intersection_to_ground(u, v, pitch, roll)

    def _ray_intersection_to_ground(
        self,
        u: float,
        v: float,
        pitch: float,
        roll: float,
    ) -> Optional[Tuple[float, float]]:
        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        cx = self.intrinsics.cx
        cy = self.intrinsics.cy

        ray_cam = np.array(
            [
                (float(u) - cx) / max(fx, 1e-6),
                (float(v) - cy) / max(fy, 1e-6),
                1.0,
            ],
            dtype=np.float64,
        )
        ray_cam /= max(float(np.linalg.norm(ray_cam)), 1e-8)

        rot = self._camera_to_vehicle_rotation(pitch, roll)
        ray_vehicle = rot @ ray_cam

        origin = np.array(
            [
                self.camera_mount_x_m,
                self.camera_mount_y_m,
                self.camera_mount_z_m,
            ],
            dtype=np.float64,
        )
        denom = float(ray_vehicle[2])
        if denom >= -1e-6:
            return None

        t = -float(origin[2]) / denom
        if t <= 0.0:
            return None

        point_vehicle = origin + t * ray_vehicle
        x_m = float(point_vehicle[0])
        y_m = float(point_vehicle[1])
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            return None
        if x_m < 0.0 or x_m > self.max_distance_m:
            return None
        return x_m, y_m

    def _camera_to_vehicle_rotation(self, pitch: float, roll: float) -> np.ndarray:
        total_pitch_deg = self.base_pitch_deg + float(pitch)
        total_roll_deg = self.base_roll_deg + float(roll)

        pitch_rad = math.radians(-total_pitch_deg)
        roll_rad = math.radians(total_roll_deg)

        rot_y = np.array(
            [
                [math.cos(pitch_rad), 0.0, math.sin(pitch_rad)],
                [0.0, 1.0, 0.0],
                [-math.sin(pitch_rad), 0.0, math.cos(pitch_rad)],
            ],
            dtype=np.float64,
        )
        rot_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(roll_rad), -math.sin(roll_rad)],
                [0.0, math.sin(roll_rad), math.cos(roll_rad)],
            ],
            dtype=np.float64,
        )
        base = np.array(
            [
                [0.0, 0.0, 1.0],   # z_cam -> x_vehicle
                [1.0, 0.0, 0.0],   # x_cam -> y_vehicle
                [0.0, -1.0, 0.0],  # -y_cam -> z_vehicle
            ],
            dtype=np.float64,
        )
        return rot_x @ rot_y @ base

    def _estimate_relative_speed(
        self,
        track_id: Any,
        x_m: float,
        y_m: float,
        timestamp: float,
    ) -> float:
        if track_id is None:
            return 0.0
        try:
            tid = int(track_id)
        except Exception:
            return 0.0

        prev = self._track_history.get(tid)
        self._track_history[tid] = (float(timestamp), float(x_m), float(y_m))
        if prev is None:
            return 0.0

        prev_ts, prev_x, prev_y = prev
        dt = float(timestamp) - float(prev_ts)
        if dt <= 1e-3:
            return 0.0

        dist_m = math.hypot(float(x_m) - float(prev_x), float(y_m) - float(prev_y))
        return float((dist_m / dt) * 3.6)
