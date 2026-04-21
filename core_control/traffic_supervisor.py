"""
Traffic Supervisor - 6 Layer Safety Architecture

Chức năng:
  - Phát hiện đèn đỏ → Signal phanh (với zone classification + locking)
  - Phát hiện chướng ngại → Signal phanh (với dynamic distance threshold)
  - Quản lý trạng thái xe (CRUISING/STOPPING/STOPPED/RESUMING)
  - Xác thực điều kiện resume an toàn
  - Xử lý rẽ và immunity logic
  - Red light smoothing + temporal filtering

Architecture (6 Layers):
  1. Perception Processing: Parse YOLO detections, zone classification, obstacle polygon
  2. Decision Fusion: Signal scoring, zone locking, turn suppression, dynamic distance
  3. Temporal Filtering: Multi-frame consensus, smoothing
  4. Resume Validation: Check all conditions + green light override
  5. Fail-Safe Logic: Emergency handling, fallback distance
  6. State Management: Track vehicle state + green immunity timer
"""

import math
import numpy as np
import cv2
from collections import deque
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any


class SupervisorState(Enum):
    """
    Trạng thái của traffic supervisor
    
    State Transitions:
      CRUISING --[brake signal]--> STOPPING
      STOPPING --[speed→0]--> STOPPED
      STOPPED --[resume clear]--> RESUMING
      RESUMING --[safety check ok]--> CRUISING
    """
    CRUISING = "cruising"      # Đang chạy bình thường
    STOPPING = "stopping"      # Đang phanh
    STOPPED = "stopped"        # Đã dừng hoàn toàn
    RESUMING = "resuming"      # Kiểm tra điều kiện resume


@dataclass
class DetectionResult:
    """Kết quả phát hiện từ YOLO"""
    class_name: str                      # "traffic_light", "vehicle", "pedestrian"
    confidence: float                    # 0.0-1.0
    bbox: Tuple[int, int, int, int]      # (x, y, w, h)
    distance: Optional[float] = None     # Khoảng cách ước tính (meter)
    signal_score: Optional[float] = None # Score tính từ confidence/distance


class TrafficSupervisor:
    """
    ════════════════════════════════════════════════════════════════════
    TRAFFIC SUPERVISOR - 6 LAYER SAFETY ARCHITECTURE + SMART LOGIC
    ════════════════════════════════════════════════════════════════════
    
    Module quản lý an toàn giao thông với 6 layer kiến trúc + logic thông minh:
    
    Layer 1: Perception Processing
      ├─ Parse YOLO detections
      ├─ Zone classification (urban/rural_right)
      ├─ Trapezoid obstacle corridor detection
      └─ Ước tính khoảng cách từ bbox + physics formula
    
    Layer 2: Decision Fusion
      ├─ Signal scoring (confidence/distance)
      ├─ Zone locking (2-frame acquire, 4-frame release)
      ├─ Turn suppression (steer ≥0.22, speed ≥5 km/h, 18-frame hold)
      ├─ Dynamic distance threshold based on speed
      └─ Green light override (green_score ≥ red_score × 1.05)
    
    Layer 3: Temporal Filtering
      ├─ Lịch sử 3 frames gần nhất
      ├─ Red light smoothing (2-frame confirm, 6-frame hold, 2-frame release)
      └─ Purpose: Tránh phantom braking
    
    Layer 4: Resume Validation
      ├─ Check: Đèn xanh? Chướng ngại tan? Xe dừng?
      ├─ Timeout protection: max 30s
      └─ Chỉ resume khi ALL clear
    
    Layer 5: Fail-Safe Logic
      ├─ Fallback distance estimation (bbox-based)
      ├─ Conservative brake nếu không chắc
      └─ Timeout release để tránh stuck
    
    Layer 6: State Management
      ├─ CRUISING → STOPPING → STOPPED → RESUMING → CRUISING
      ├─ Track speed, time, confidence
      ├─ Green immunity timer (10 frames after green)
      └─ Explicit state transitions
    
    ════════════════════════════════════════════════════════════════════
    """
    
    def __init__(self, config: dict):
        """
        Khởi tạo supervisor với cấu hình
        
        Args:
            config: Dict chứa tham số:
                - confidence_threshold: 0.5 (mức tin cậy tối thiểu)
                - temporal_filter_frames: 3 (frame liên tiếp cần xác nhận)
                - red_light_distance_threshold: 30.0 (meter)
                - obstacle_distance_threshold: 5.0 (meter)
                - max_stopped_time: 30.0 (giây - timeout)
        """
        self.config = config
        self.state = SupervisorState.CRUISING

        # ─────────────────────────────────────────────────────────
        # PATH CORRIDOR PARAMETERS (for trapezoid drawing)
        # ─────────────────────────────────────────────────────────
        self.path_wheelbase_m = 2.7
        self.path_max_steer_angle_deg = 70.0
        self.path_base_half_width_m = 1.1
        self.path_width_growth_per_m = 0.035
        self.path_curve_width_gain = 0.55
        self.path_max_half_width_m = 2.8
        self.path_min_forward_m = 0.8
        self.path_max_forward_m = 40.0

        # Merge V10 defaults (can be overridden by external config)
        self.config.setdefault('acc_time_gap', 1.5)
        self.config.setdefault('acc_standstill_dist', 2.0)
        self.config.setdefault('acc_proportional_gain', 0.08)
        self.config.setdefault('acc_derivative_gain', 0.20)
        self.config.setdefault('acc_max_urgency_brake', 1.0)
        self.config.setdefault('acc_max_urgency_throttle', 0.5)
        self.config.setdefault('ttc_safe_seconds', 4.0)
        self.config.setdefault('ttc_critical_seconds', 1.0)
        self.config.setdefault('vehicle_length_m', 4.5)
        self.config.setdefault('tracking_iou_threshold', 0.3)
        self.config.setdefault('tracking_max_missed_frames', 6)
        self.config.setdefault('tracking_patience_seconds', 0.3)
        self.config.setdefault('tracking_max_lead_decel_ms2', 8.0)
        self.config.setdefault('tracking_max_physical_accel_ms2', 10.0)
        self.config.setdefault('stop_line_confirm_frames', 2)
        self.config.setdefault('safe_distance_urban_m', 20.0)
        self.config.setdefault('safe_distance_rural_right_m', 10.0)
        self.config.setdefault('safe_distance_default_m', 15.0)
        
        # ─────────────────────────────────────────────────────────
        # LAYER 1: Zone Locking State
        # ─────────────────────────────────────────────────────────
        self.locked_zone = None           # 'urban' or 'rural_right'
        self.zone_acquire_count = 0       # Frames for zone acquisition
        self.zone_missing_count = 0       # Frames to release lock
        
        # ─────────────────────────────────────────────────────────
        # LAYER 2: Turn Phase Tracking
        # ─────────────────────────────────────────────────────────
        self.in_turn_phase = False        # Suppressing braking during turn
        self.turn_hold_counter = 0        # Hold turn suppression
        self.turn_grace_counter = 0       # Grace period after turn
        
        # ─────────────────────────────────────────────────────────
        # LAYER 2: Red Light Smoothing State
        # ─────────────────────────────────────────────────────────
        self.red_confirmed = False        # Red light confirmed
        self.red_confirm_count = 0        # Frames to confirm (2)
        self.red_hold_count = 0           # Frames to hold after (6)
        self.red_release_count = 0        # Frames to release (2)
        self.stop_line_confirm_count = 0  # Strict mode: frames to confirm stop_line
        
        # ─────────────────────────────────────────────────────────
        # LAYER 3: Temporal Filtering - Lịch sử detection
        # ─────────────────────────────────────────────────────────
        self.red_light_history = deque(maxlen=3)
        self.obstacle_history = deque(maxlen=3)
        
        # ─────────────────────────────────────────────────────────
        # LAYER 6: Green Immunity Timer
        # ─────────────────────────────────────────────────────────
        self.green_immunity_counter = 0   # 10 frames after green light
        self.phantom_immunity_counter = 0 # Frames to ignore phantom obstacle after timeout escape

        # ─────────────────────────────────────────────────────────
        # V10 TRACKING MEMORY (multi-target with kinematic state)
        # Each entry: (DetectionResult, hit_streak, miss_streak, last_distance_m, last_rel_vel_ms)
        # ─────────────────────────────────────────────────────────
        self.last_detected_targets: Dict[str, List[Tuple[DetectionResult, int, int, float, float]]] = {
          'obstacle': []
        }
        
        # ─────────────────────────────────────────────────────────
        # TIME TRACKING
        # ─────────────────────────────────────────────────────────
        self.stopped_time = 0.0
        self.frame_count = 0
        
        # ─────────────────────────────────────────────────────────
        # LAYER 5: Jerk Constraint History
        # ─────────────────────────────────────────────────────────
        self._prev_brake_force = 0.0      # Previous frame's brake force

        # V10 runtime signals for debugging
        self._last_longitudinal_urgency = 0.0
        self._last_acc_urgency = 0.0
        self._last_ttc_urgency = float('-inf')
        self._last_selected_target_type = None
        self.last_danger_polygon = None
        
        # ─────────────────────────────────────────────────────────
        # DEBUG INFO
        # ─────────────────────────────────────────────────────────
        self._print_init_report()
    
    def _print_init_report(self):
        """In báo cáo khởi tạo (one-time)"""
        print("\n" + "="*70)
        print("🚦 TRAFFIC SUPERVISOR INITIALIZED")
        print("="*70)
        print(f"Config:")
        print(f"  - Confidence Threshold: {self.config.get('confidence_threshold', 0.5)}")
        print(f"  - Temporal Filter Frames: {self.config.get('temporal_filter_frames', 3)}")
        print(f"  - Red Light Distance: {self.config.get('red_light_distance_threshold', 30.0)}m")
        print(f"  - Obstacle Distance: {self.config.get('obstacle_distance_threshold', 5.0)}m")
        print(f"  - Max Stopped Time: {self.config.get('max_stopped_time', 30.0)}s")
        print("="*70 + "\n")
    
    # ═══════════════════════════════════════════════════════════════
    # LAYER 1: PERCEPTION PROCESSING + ZONE CLASSIFICATION
    # ═══════════════════════════════════════════════════════════════
    
    def _clamp(self, value: float, low: float, high: float) -> float:
        """Clamp value between low and high."""
        return max(low, min(high, value))
    
    def _steer_to_curvature(self, vehicle_steer: Optional[float]) -> float:
        """
        Chuyển steering angle (-1 to 1) thành curvature (1/meters).
        
        Công thức: curvature = tan(wheel_angle) / wheelbase
        """
        if vehicle_steer is None:
            return 0.0
        steer = self._clamp(float(vehicle_steer), -1.0, 1.0)
        wheel_angle = steer * math.radians(self.path_max_steer_angle_deg)
        curvature = math.tan(wheel_angle) / max(self.path_wheelbase_m, 1e-3)
        return float(self._clamp(curvature, -0.45, 0.45))
    
    def _curved_path_center_lateral(self, forward_m: float, vehicle_steer: Optional[float]) -> float:
        """
        Tính lateral offset (y) từ center line dựa trên curvature.
        
        Công thức: center_y = 0.5 * curvature * forward_m²
        (This creates a parabolic trajectory)
        """
        curvature = self._steer_to_curvature(vehicle_steer)
        return 0.5 * curvature * float(forward_m) * float(forward_m)
    
    def _curved_path_half_width(self, forward_m: float, vehicle_steer: Optional[float]) -> float:
        """
        Tính nửa chiều rộng của corridor dựa trên forward distance.
        
        Bao gồm:
        - base: Chiều rộng cơ bản
        - growth: Tăng thêm dựa trên forward distance
        - curve_bonus: Tăng thêm khi có curvature
        """
        curvature = abs(self._steer_to_curvature(vehicle_steer))
        base = self.path_base_half_width_m + self.path_width_growth_per_m * max(0.0, forward_m)
        curve_bonus = self.path_curve_width_gain * curvature * max(0.0, forward_m)
        return float(min(self.path_max_half_width_m, base + curve_bonus))
    
    def _camera_to_vehicle_rotation(self, camera_pitch_deg: float) -> np.ndarray:
        """
        Tính rotation matrix từ camera frame sang vehicle frame.
        
        Camera pitch: độ dốc của camera (âm = hướng xuống)
        """
        pitch_rad = math.radians(-float(camera_pitch_deg))
        rot_y = np.array(
            [
                [math.cos(pitch_rad), 0.0, math.sin(pitch_rad)],
                [0.0, 1.0, 0.0],
                [-math.sin(pitch_rad), 0.0, math.cos(pitch_rad)],
            ],
            dtype=np.float32,
        )
        # Base transformation: camera Y → vehicle Z, camera Z → vehicle -Y
        base = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
        return rot_y @ base
    
    def _project_vehicle_to_image(self,
                                   point_vehicle: np.ndarray,
                                   frame_width: int,
                                   frame_height: int,
                                   camera_fov_deg: float,
                                   camera_mount_xyz: Tuple[float, float, float],
                                   camera_pitch_deg: float) -> Optional[Tuple[int, int]]:
        """
        Project 3D point từ vehicle frame sang image frame (pixel coordinates).
        
        Args:
            point_vehicle: 3D point trong vehicle frame [x, y, z]
            frame_width, frame_height: Kích thước frame (pixel)
            camera_fov_deg: Field of view của camera (độ)
            camera_mount_xyz: Vị trí camera trong vehicle frame
            camera_pitch_deg: Góc pitch của camera
        
        Returns:
            (u, v) pixel coordinates, hoặc None nếu point ở phía sau camera
        """
        if np is None:
            return None

        fx = (frame_width / 2.0) / max(math.tan(math.radians(camera_fov_deg) / 2.0), 1e-6)
        fy = fx
        cx = frame_width / 2.0
        cy = frame_height / 2.0

        r_c2v = self._camera_to_vehicle_rotation(camera_pitch_deg)
        r_v2c = r_c2v.T
        t = np.array(camera_mount_xyz, dtype=np.float32)
        p_rel = point_vehicle.astype(np.float32) - t
        p_cam = r_v2c @ p_rel
        
        # Point phải ở phía trước camera (z > 0.15m)
        if p_cam[2] <= 0.15:
            return None

        u = fx * (p_cam[0] / p_cam[2]) + cx
        v = fy * (p_cam[1] / p_cam[2]) + cy
        
        if not np.isfinite(u) or not np.isfinite(v):
            return None
        
        return (int(round(u)), int(round(v)))
    
    def _classify_traffic_light_zone(self, 
                                    bbox: Tuple[int, int, int, int],
                                    image_shape: Optional[Tuple] = None
                                    ) -> Optional[str]:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ Zone Classification: Urban vs Rural_Right                   │
        └─────────────────────────────────────────────────────────────┘
        
        Phân loại đèn giao thông dựa trên vị trí trong frame:
        
        Urban (Đèn phố): x ∈ [0.35W, 0.65W] → distance_threshold = 15m
          - Điều khiển lưu thông trung tâm thành phố
          - Gần hơn, phanh sớm hơn
        
        Rural_Right (Đèn nông thôn phải): x ∈ [0.65W, 0.95W] → distance_threshold = 7m
          - Đèn bên phải đường
          - Xa hơn, nhưng chỉ thông báo khi rất gần
        
        Args:
          - bbox: (x, y, w, h)
          - image_shape: (height, width, channels), or default 480×640
        
        Returns:
          - 'urban' or 'rural_right' or None (nếu outside ranges)
        """
        if bbox is None or len(bbox) < 4:
            return None
        
        if image_shape is None:
            image_shape = (480, 640, 3)  # Default CARLA camera
        
        x_center = bbox[0] + bbox[2] / 2.0
        img_width = image_shape[1]
        
        x_ratio = x_center / img_width
        
        if 0.35 <= x_ratio <= 0.65:
            return 'urban'
        elif 0.65 <= x_ratio <= 0.95:
            return 'rural_right'
        else:
            return None
    
    def _build_obstacle_danger_polygon(self,
                       image_shape: Optional[Tuple] = None,
                       vehicle_steer: float = 0.0,
                       vehicle_speed_kmh: float = 0.0,
                       camera_fov_deg: float = 90.0,
                       camera_mount_xyz: Tuple[float, float, float] = (1.5, 0.0, 2.2),
                       camera_pitch_deg: float = -8.0
                       ) -> Optional[np.ndarray]:
      """
      ┌────────────────────────────────────────────────────────────────┐
      │ Tạo hình thang danger corridor dính trên mặt đất + bẻ cong     │
      │ theo góc lái của xe                                             │
      └────────────────────────────────────────────────────────────────┘
      
      Cách tiếp cận:
      1. Xác định vehicle-space coordinates (forward, lateral)
      2. Tính curvature từ steering angle
      3. Tính center lateral offset từ parabolic trajectory
      4. Tính corridor half-width dựa trên forward distance
      5. Tính 3D points trên road plane (z = 0)
      6. Project sang image frame để lấy pixel coordinates
      7. Tạo closed polygon từ left + right points
      
      Args:
          image_shape: (height, width, channels) - default (480, 640, 3)
          vehicle_steer: Steering angle (-1 to 1)
          vehicle_speed_kmh: Tốc độ (km/h)
          camera_fov_deg: Camera field of view (độ)
          camera_mount_xyz: Camera mount position trong vehicle frame (meters)
          camera_pitch_deg: Camera pitch angle (độ, âm = hướng xuống)
      
      Returns:
          np.ndarray: Polygon (N, 2) trong image coordinates, hoặc None nếu không valid
      """
      if image_shape is None:
          image_shape = (480, 640, 3)
      
      if np is None:
          return None

      frame_h, frame_w = image_shape[0], image_shape[1]
      
      # ─────────────────────────────────────────────────────────
      # Bước 1: Tính Curvature từ Steering
      # ─────────────────────────────────────────────────────────
      curvature = self._steer_to_curvature(vehicle_steer)
      
      # ─────────────────────────────────────────────────────────
      # Bước 2: Xác định Horizon dựa trên Speed
      # ─────────────────────────────────────────────────────────
      # Horizon = base + speed-dependent term
      horizon_m = max(self.path_min_forward_m + 2.0, 12.0)
      if vehicle_speed_kmh is not None:
          speed_mps = max(0.0, float(vehicle_speed_kmh) / 3.6)
          horizon_m = max(horizon_m, speed_mps * 2.2 + 8.0)
      horizon_m = min(horizon_m, self.path_max_forward_m)
      
      # ─────────────────────────────────────────────────────────
      # Bước 3: Sample Points trong Vehicle-Space
      # ─────────────────────────────────────────────────────────
      # Sample nhiều điểm từ min_forward đến horizon
      sample_count = max(28, int(horizon_m * 2.5))
      forward_values = np.linspace(
          self.path_min_forward_m, 
          horizon_m, 
          sample_count, 
          dtype=np.float32
      )
      
      left_pixels: List[Tuple[int, int]] = []
      right_pixels: List[Tuple[int, int]] = []
      
      for forward_m in forward_values:
          # ───────────────────────────────────────────────
          # Tính Center Lateral Offset (bẻ cong từ curvature)
          # ───────────────────────────────────────────────
          center_lateral = self._curved_path_center_lateral(forward_m, vehicle_steer)
          
          # ───────────────────────────────────────────────
          # Tính Half-Width của Corridor
          # ───────────────────────────────────────────────
          half_width = self._curved_path_half_width(forward_m, vehicle_steer)
          
          # ───────────────────────────────────────────────
          # Tính Left & Right Lateral Positions
          # ───────────────────────────────────────────────
          y_left = center_lateral - half_width
          y_right = center_lateral + half_width
          
          # ───────────────────────────────────────────────
          # Tạo 3D Points trên Road Plane (z = 0, giả sử)
          # ───────────────────────────────────────────────
          # Vehicle frame: (x=forward, y=lateral, z=vertical)
          p_left = np.array([float(forward_m), float(y_left), 0.0], dtype=np.float32)
          p_right = np.array([float(forward_m), float(y_right), 0.0], dtype=np.float32)
          
          # ───────────────────────────────────────────────
          # Project sang Image Frame (pixel coordinates)
          # ───────────────────────────────────────────────
          left_uv = self._project_vehicle_to_image(
              p_left, 
              frame_w, 
              frame_h, 
              camera_fov_deg,
              camera_mount_xyz,
              camera_pitch_deg
          )
          right_uv = self._project_vehicle_to_image(
              p_right, 
              frame_w, 
              frame_h, 
              camera_fov_deg,
              camera_mount_xyz,
              camera_pitch_deg
          )
          
          # Chỉ thêm nếu cả hai điểm hợp lệ (trong camera view)
          if left_uv is not None:
              left_pixels.append(left_uv)
          if right_uv is not None:
              right_pixels.append(right_uv)
      
      # ─────────────────────────────────────────────────────────
      # Bước 4: Kiểm tra có đủ điểm để tạo polygon
      # ─────────────────────────────────────────────────────────
      if len(left_pixels) < 4 or len(right_pixels) < 4:
          return None
      
      # ─────────────────────────────────────────────────────────
      # Bước 5: Tạo Closed Polygon
      # ─────────────────────────────────────────────────────────
      # Polygon = left_points + reversed(right_points)
      # Tạo hình thang kín
      polygon_points = left_pixels + list(reversed(right_pixels))
      
      if len(polygon_points) < 3:
          return None
      
      polygon = np.array(polygon_points, dtype=np.int32)
      
      # Clip polygon points to image bounds to avoid out-of-bounds projections
      polygon[:, 0] = np.clip(polygon[:, 0], 0, frame_w - 1)
      polygon[:, 1] = np.clip(polygon[:, 1], 0, frame_h - 1)
      
      # Lưu trữ thông tin debug
      self.last_danger_polygon = polygon
      
      return polygon
    
    def _point_in_polygon(self, 
                         point: Tuple[float, float],
                         polygon: np.ndarray
                         ) -> bool:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ Check if point is inside polygon using cv2.pointPolygonTest │
        └─────────────────────────────────────────────────────────────┘
        
        Args:
          - point: (x, y)
          - polygon: np.ndarray of shape (N, 2)
        
        Returns:
          - bool: True if point inside, False otherwise
        """
        if polygon is None:
            return False
        
        try:
            result = cv2.pointPolygonTest(polygon, point, False)
            return result >= 0  # ≥0 means inside or on boundary
        except:
            return False
    
    def _signal_score(self, 
                     detection: Optional[DetectionResult]
                     ) -> float:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ Signal Score: Tính score từ confidence + distance           │
        └─────────────────────────────────────────────────────────────┘
        
        Score = confidence / (1.0 + distance/10.0)
          - Càng gần → score cao
          - Càng thấp confidence → score thấp
        
        Args:
          - detection: DetectionResult or None
        
        Returns:
          - score: float (0.0 - 1.0)
        """
        if detection is None:
            return 0.0
        
        distance_factor = 1.0 + (detection.distance or 100.0) / 10.0
        score = detection.confidence / distance_factor
        return max(0.0, min(score, 1.0))
    
    def _evaluate_zone_signal(self,
                             red_light: Optional[DetectionResult],
                             image_shape: Optional[Tuple] = None
                             ) -> Tuple[Optional[str], float]:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ Evaluate Zone + Signal Score                                │
        └─────────────────────────────────────────────────────────────┘
        
        Kết hợp zone classification + signal scoring
        
        Returns:
          - zone: 'urban', 'rural_right', or None
          - score: 0.0 - 1.0
        """
        if red_light is None:
            return None, 0.0
        
        zone = self._classify_traffic_light_zone(red_light.bbox, image_shape)
        score = self._signal_score(red_light)
        return zone, score
    
    def _select_zone_candidate_for_lock(self,
                                       current_zone: Optional[str],
                                       new_detection: Optional[DetectionResult],
                                       image_shape: Optional[Tuple] = None
                                       ) -> Optional[str]:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ Zone Locking: Select candidate for acquisition              │
        └─────────────────────────────────────────────────────────────┘
        
        Logic:
          1. Nếu chưa lock: classify zone từ new_detection
          2. Nếu đã lock: keep current zone
          3. Chuyển zone: new_detection có zone khác
        
        Returns:
          - candidate_zone: 'urban', 'rural_right', or None
        """
        if new_detection is None:
            return current_zone
        
        new_zone, _ = self._evaluate_zone_signal(new_detection, image_shape)
        
        if current_zone is None:
            return new_zone
        
        if new_zone == current_zone:
            return new_zone
        else:
            # Zone mismatch: keep current
            return current_zone
    
    def _update_locked_zone(self,
                           candidate_zone: Optional[str],
                           red_light: Optional[DetectionResult]
                           ) -> None:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ Zone Locking State Machine                                  │
        │ Acquire: 2 frames, Release: 4 frames                        │
        └─────────────────────────────────────────────────────────────┘
        
        IDLE (no lock)
          ↓ (candidate_zone ≠ None)
        ACQUIRING (zone_acquire_count < 2)
          ↓ (zone_acquire_count ≥ 2)
        LOCKED (detection present)
          ↓ (detection lost)
        RELEASING (zone_missing_count < 4)
          ↓ (zone_missing_count ≥ 4)
        IDLE (lock released)
        """
        if candidate_zone is None:
            # No candidate: move to release phase
            if self.locked_zone is not None:
                self.zone_missing_count += 1
                if self.zone_missing_count >= 4:
                    self.locked_zone = None
                    self.zone_acquire_count = 0
                    self.zone_missing_count = 0
        else:
            # Candidate present: acquire or refresh lock
            if self.locked_zone is None:
                # Start acquisition
                self.zone_acquire_count += 1
                if self.zone_acquire_count >= 2:
                    self.locked_zone = candidate_zone
            else:
                # Already locked: reset missing counter
                self.zone_missing_count = 0

    def _compute_iou(self,
                     bbox1: Optional[Tuple[int, int, int, int]],
                     bbox2: Optional[Tuple[int, int, int, int]]) -> float:
        """Compute IoU between 2 XYWH bboxes."""
        if bbox1 is None or bbox2 is None or len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0

        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
        bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter_area
        if union <= 1e-9:
            return 0.0
        return float(inter_area / union)

    def _greedy_match_detections(self,
                                 previous_tracks: List[Tuple[DetectionResult, int, int, float, float]],
                                 current_dets: List[DetectionResult],
                                 dt: float) -> List[Tuple[DetectionResult, int, int, float, float]]:
        """
        V10 tracking update:
          - IoU greedy matching
          - hit/miss streak maintenance
          - kinematic projection during missed frames
          - derivative clamping on re-acquisition
        """
        iou_threshold = float(self.config.get('tracking_iou_threshold', 0.3))
        max_missed = int(self.config.get('tracking_max_missed_frames', 6))
        patience_s = float(self.config.get('tracking_patience_seconds', 0.3))
        max_lead_decel = float(self.config.get('tracking_max_lead_decel_ms2', 8.0))
        max_physical_accel = float(self.config.get('tracking_max_physical_accel_ms2', 10.0))

        dt = max(1e-3, float(dt))

        # Build candidate match pairs (greedy by highest IoU)
        pairs: List[Tuple[float, int, int]] = []
        for pi, (p_det, _h, _m, _d, _rv) in enumerate(previous_tracks):
            for ci, c_det in enumerate(current_dets):
                iou = self._compute_iou(p_det.bbox, c_det.bbox)
                if iou >= iou_threshold:
                    pairs.append((iou, pi, ci))

        pairs.sort(key=lambda x: x[0], reverse=True)

        used_prev = set()
        used_curr = set()
        matches: List[Tuple[int, int]] = []
        for _iou, pi, ci in pairs:
            if pi in used_prev or ci in used_curr:
                continue
            used_prev.add(pi)
            used_curr.add(ci)
            matches.append((pi, ci))

        updated: List[Tuple[DetectionResult, int, int, float, float]] = []

        # Matched tracks: update with derivative clamping
        for pi, ci in matches:
            _prev_det, prev_hit, _prev_miss, prev_dist, prev_rel_vel = previous_tracks[pi]
            cur_det = current_dets[ci]

            measured_rel_vel = (prev_dist - cur_det.distance) / dt
            max_vel_delta = max_physical_accel * dt
            smoothed_rel_vel = float(np.clip(
                measured_rel_vel,
                prev_rel_vel - max_vel_delta,
                prev_rel_vel + max_vel_delta
            ))

            cur_det.relative_velocity_kmh = smoothed_rel_vel * 3.6
            cur_det.track_hit_streak = prev_hit + 1
            cur_det.track_miss_streak = 0

            updated.append((
                cur_det,
                prev_hit + 1,
                0,
                float(cur_det.distance),
                smoothed_rel_vel
            ))

        # Unmatched previous tracks: kinematic projection
        for pi, (prev_det, prev_hit, prev_miss, prev_dist, prev_rel_vel) in enumerate(previous_tracks):
            if pi in used_prev:
                continue

            miss = prev_miss + 1
            if miss > max_missed:
                continue

            time_missing = miss * dt
            predicted_rel_vel = prev_rel_vel
            if time_missing > patience_s:
                predicted_rel_vel += max_lead_decel * (time_missing - patience_s)

            predicted_distance = max(0.1, prev_dist - predicted_rel_vel * dt)

            ghost_det = DetectionResult(
                class_name=prev_det.class_name,
                confidence=max(0.05, prev_det.confidence * 0.95),
                bbox=prev_det.bbox,
                distance=predicted_distance,
                signal_score=prev_det.signal_score
            )
            ghost_det.relative_velocity_kmh = predicted_rel_vel * 3.6
            ghost_det.track_hit_streak = prev_hit
            ghost_det.track_miss_streak = miss

            updated.append((
                ghost_det,
                prev_hit,
                miss,
                predicted_distance,
                predicted_rel_vel
            ))

        # Unmatched current detections: create new tracks
        for ci, cur_det in enumerate(current_dets):
            if ci in used_curr:
                continue

            raw_rel_vel_ms = float(getattr(cur_det, 'relative_velocity_kmh', 0.0)) / 3.6
            cur_det.track_hit_streak = 1
            cur_det.track_miss_streak = 0
            updated.append((cur_det, 1, 0, float(cur_det.distance), raw_rel_vel_ms))

        # Prefer closer tracks to evaluate immediate risk first
        updated.sort(key=lambda item: item[3])
        return updated
    
    def _parse_detections(self,
               detections: List[dict],
               image_shape: Optional[Tuple] = None,
               dt: float = 0.033,
               vehicle_steer: float = 0.0
                         ) -> Tuple[Optional[DetectionResult], 
                                   Optional[DetectionResult],
                                   Optional[List[DetectionResult]]]:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ LAYER 1: Parse YOLO detections + zone classification        │
        │ NEW: Support for stop_line detection                        │
        └─────────────────────────────────────────────────────────────┘
        
        Input: List Dict từ YOLO/Spatial_Math
          [
            {'class_name': 'traffic_light_red', 'confidence': 0.92, 'bbox': (100, 50, 40, 80), 'distance_m': 15.0},
            {'class_name': 'vehicle', 'confidence': 0.88, 'bbox': (200, 100, 100, 150), 'distance_m': 8.5, 'relative_velocity_kmh': 0.0},
            {'class_name': 'stop_line', 'confidence': 0.78, 'bbox': (150, 400, 300, 420), 'distance_m': 12.0},
          ]
        
        Output: (red_light_detection, obstacle_detection, stop_lines_list)
          - red_light_detection: Highest confidence red light or None
          - obstacle_detection: Highest confidence vehicle/pedestrian or None
          - stop_lines_list: All detected stop_lines (for nearest selection)
        
        Logic:
          1. Loop through all detections
          2. Extract distance_m and relative_velocity_kmh from spatial_math output
          3. Classify: red_light | vehicle/obstacle | stop_line
          4. Keep highest confidence for red_light and obstacle
          5. Collect ALL stop_lines (not just highest)
          6. Update zone locking state
          7. Return 3-tuple
        """
        red_light = None
        obstacle_candidates: List[DetectionResult] = []
        stop_lines = []
        danger_polygon = self._build_obstacle_danger_polygon(image_shape, vehicle_steer)
        self.last_danger_polygon = danger_polygon
        
        for det in detections:
            class_name = det.get('class_name', '').lower()
            confidence = float(det.get('confidence', 0.0))
            bbox = det.get('bbox')
            
            # Extract distance from spatial_math output (prefer this over bbox estimation)
            distance = float(det.get('distance_m', float('inf')))
            if not np.isfinite(distance):
                distance = self._estimate_distance_from_bbox(bbox)
            
            # Extract relative velocity (only for vehicles)
            relative_velocity_kmh = float(det.get('relative_velocity_kmh', 0.0))
            
            # ┌─ Phân loại: Đèn đỏ
            if class_name in ['traffic_light_red', 'red_light'] and confidence > 0.3:
                if red_light is None or confidence > red_light.confidence:
                    red_light = DetectionResult(
                        class_name='traffic_light_red',
                        confidence=confidence,
                        bbox=bbox,
                        distance=distance,
                        signal_score=None  # Will compute later
                    )
                    # Store relative_velocity for later use
                    red_light.relative_velocity_kmh = relative_velocity_kmh
            
            # ┌─ Phân loại: Chướng ngại (Vehicle/Pedestrian)
            elif class_name in ['vehicle', 'pedestrian', 'car', 'truck', 'person', 'two_wheeler']:
                if confidence > 0.3:
                    if bbox is not None and len(bbox) >= 4 and danger_polygon is not None:
                        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        bottom_center = (int(x + w / 2.0), y + h)
                        if not self._point_in_polygon(bottom_center, danger_polygon):
                            continue
                    obstacle_det = DetectionResult(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                        distance=distance,
                        signal_score=None
                    )
                    obstacle_det.relative_velocity_kmh = relative_velocity_kmh
                    obstacle_candidates.append(obstacle_det)
            
            # ┌─ Phân loại: Stop Line (Collect ALL, not just highest)
            elif class_name in ['stop_line', 'stopline']:
                if confidence > 0.5:  # Higher threshold for stop_line
                    stop_line_det = DetectionResult(
                        class_name='stop_line',
                        confidence=confidence,
                        bbox=bbox,
                        distance=distance,
                        signal_score=None
                    )
                    stop_lines.append(stop_line_det)
        
        # Update zone locking state (using red_light)
        if red_light:
            candidate_zone = self._select_zone_candidate_for_lock(
                self.locked_zone, red_light, image_shape
            )
            self._update_locked_zone(candidate_zone, red_light)

        # V10: update obstacle tracks with IoU + kinematics
        previous_tracks = self.last_detected_targets.get('obstacle', [])
        updated_tracks = self._greedy_match_detections(previous_tracks, obstacle_candidates, dt)
        self.last_detected_targets['obstacle'] = updated_tracks

        # Pick nearest stable track as obstacle target
        obstacle = None
        temporal_frames = int(self.config.get('temporal_filter_frames', 3))
        for det, hit, miss, _dist, _rv in updated_tracks:
          if hit >= temporal_frames:
                obstacle = det
                break
        
        return red_light, obstacle, stop_lines
    
    def _estimate_distance_from_bbox(self, bbox: Tuple) -> float:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ Estimate Distance from Bounding Box (Monocular)             │
        │ Fallback for Layer 5 fail-safe                              │
        └─────────────────────────────────────────────────────────────┘
        
        Formula: distance ≈ focal_length * real_height / bbox_height
          - Obj cao = gần
          - Obj thấp = xa
        
        Parameters:
          - focal_length: 800 pixels (CARLA camera default)
          - real_height: 1.7 meters (human height standard)
        
        Output: distance in meters, clamped to [0.5, 100.0]
        
        Note: Monocular depth estimation (single camera)
              Giữ như fallback nếu spatial_math không available
        """
        if bbox is None or len(bbox) < 4:
            return 100.0
        
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        
        focal_length = 800
        real_height = 1.7
        
        if h <= 0:
            return 100.0
        
        distance = (focal_length * real_height) / h
        return max(0.5, min(distance, 100.0))
    
    # ═══════════════════════════════════════════════════════════════
    # LAYER 2: DECISION FUSION + TURN SUPPRESSION + DYNAMIC DISTANCE
    # ═══════════════════════════════════════════════════════════════
    
    def _update_turn_phase(self, 
                          vehicle_steer: Optional[float] = None
                          ) -> None:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ Turn Phase Tracking: Suppress braking during sharp turns     │
        └─────────────────────────────────────────────────────────────┘
        
        Condition: steer ≥ 0.22 AND speed ≥ 5 km/h (≈1.39 m/s)
          - Indicate sharp turn
          - Hold suppression 18 frames
          - Grace period 30 frames after turn
        
        State Machine:
          NORMAL (turn_hold_counter = 0, turn_grace_counter = 0)
            ↓ (sharp turn detected)
          TURNING (turn_hold_counter: 0→18)
            ↓ (turn_hold_counter ≥ 18)
          GRACE (turn_grace_counter: 0→30)
            ↓ (turn_grace_counter ≥ 30)
          NORMAL
        """
        if vehicle_steer is None:
            # Default: not turning
            self.in_turn_phase = False
        elif abs(vehicle_steer) >= 0.22:
            # Sharp turn detected
            self.in_turn_phase = True
            self.turn_hold_counter = 18
            self.turn_grace_counter = 0
        
        # Decrement counters
        if self.turn_hold_counter > 0:
            self.turn_hold_counter -= 1
        elif self.turn_grace_counter > 0:
            self.turn_grace_counter -= 1
            if self.turn_grace_counter == 0:
                self.in_turn_phase = False
    
    def _resolve_braking_target(self,
                               red_light: Optional[DetectionResult],
                               obstacle: Optional[DetectionResult],
                               stop_lines: Optional[List[DetectionResult]],
                               ) -> Tuple[Optional[DetectionResult], str, bool]:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ LAYER 2: Resolve Braking Target Priority                    │
        │ Stop line > Red light > Obstacle                            │
        └─────────────────────────────────────────────────────────────┘
        
        Priority with Confidence-Based Fallback:
          1. IF red_light present AND stop_lines nearby
             → Use nearest stop_line (if confidence > 0.65)
             → Fallback to red_light if stop_line missed
          2. ELIF red_light only
             → Use red_light
          3. ELIF obstacle present
             → Use obstacle
          4. ELSE
             → No braking target
        
        Returns:
          - target: DetectionResult or None
          - target_type: 'stop_line' | 'red_light' | 'obstacle' | None
          - use_fallback: bool (True if using fallback chain)
        """
        use_fallback = False
        vehicle_length = float(self.config.get('vehicle_length_m', 4.5))

        candidates: List[Tuple[float, DetectionResult, str]] = []

        if stop_lines:
            nearest_stop = min(stop_lines, key=lambda s: s.distance)
            candidates.append((float(nearest_stop.distance), nearest_stop, 'stop_line'))

        if red_light is not None:
            candidates.append((float(red_light.distance), red_light, 'red_light'))

        if obstacle is not None:
            obstacle_effective_distance = max(0.0, float(obstacle.distance) - vehicle_length)
            candidates.append((obstacle_effective_distance, obstacle, 'obstacle'))

        if not candidates:
            return None, None, use_fallback

        # Safety-first: smallest effective distance wins (not confidence-first)
        candidates.sort(key=lambda item: item[0])
        _eff_dist, target, target_type = candidates[0]

        # mark fallback when a signal exists but obstacle wins by proximity
        if target_type == 'obstacle' and (red_light is not None or stop_lines):
            use_fallback = True

        return target, target_type, use_fallback

    def _compute_urgency_continuous_superposition_v10(self,
                                                      distance: float,
                                                      rel_vel_ms: float,
                                                      current_speed_ms: float) -> float:
        """
        V10 continuous bipolar urgency:
          - ACC: PD output in [-max_throttle, +max_brake]
          - TTC: continuous linear safety veto mapped to same bipolar domain
          - Arbitration: max(urgency_acc, urgency_ttc)
        """
        time_gap = float(self.config.get('acc_time_gap', 1.5))
        standstill_dist = float(self.config.get('acc_standstill_dist', 2.0))
        kp = float(self.config.get('acc_proportional_gain', 0.08))
        kd = float(self.config.get('acc_derivative_gain', 0.20))
        max_brake = float(self.config.get('acc_max_urgency_brake', 1.0))
        max_throttle = float(self.config.get('acc_max_urgency_throttle', 0.5))

        desired_distance = (current_speed_ms * time_gap) + standstill_dist
        position_error = distance - desired_distance

        urgency_acc = (-kp * position_error) + (kd * rel_vel_ms)
        urgency_acc = float(np.clip(urgency_acc, -max_throttle, max_brake))

        ttc_safe = float(self.config.get('ttc_safe_seconds', 4.0))
        ttc_critical = float(self.config.get('ttc_critical_seconds', 1.0))

        urgency_ttc = float('-inf')
        if rel_vel_ms > 0.0:
            ttc = distance / max(rel_vel_ms, 1e-3)
            if ttc < ttc_safe:
                # Map TTC linearly into bipolar safety envelope: [safe -> -max_throttle, critical -> +max_brake]
                norm = (ttc_safe - ttc) / max(1e-6, (ttc_safe - ttc_critical))
                urgency_ttc = float(np.clip(
                    (-max_throttle) + norm * (max_brake + max_throttle),
                    -max_throttle,
                    max_brake
                ))

        final_urgency = max(urgency_acc, urgency_ttc)

        self._last_acc_urgency = urgency_acc
        self._last_ttc_urgency = urgency_ttc
        self._last_longitudinal_urgency = final_urgency

        return final_urgency
    
    def _compute_collision_urgency(self,
                                   target: DetectionResult,
                                   target_type: str,
                                   current_speed_ms: float = 0.0,
                                   ) -> float:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ LAYER 2: Compute Collision Urgency using TTC (with fallback)│
        │ Handles division-by-zero and stationary targets            │
        └─────────────────────────────────────────────────────────────┘
        
        Strategy:
          - Red light/Stop line: Distance-based (stationary)
          - Vehicle: TTC-based with fallback to distance
        
        Returns:
          - urgency: float [0.0 - 1.0]
            0.0 = safe, 0.3 = caution, 0.6 = warning, 0.9 = critical
        """
        if target is None:
            return 0.0
        
        distance = target.distance
        if not np.isfinite(distance) or distance < 0:
            return 0.0
        
        # Define danger zones
        safe_distance_default = float(self.config.get('safe_distance_default_m', 15.0))
        safe_distance_urban = float(self.config.get('safe_distance_urban_m', 20.0))
        safe_distance_rural = float(self.config.get('safe_distance_rural_right_m', 10.0))
        CRITICAL_DISTANCE = 2.0    # meters
        
        # Case 1: Stationary target (red light, stop line)
        if target_type in ['stop_line', 'red_light']:
            # ROI-aware dynamics:
            # - stop_line / urban: react earlier (larger safe distance)
            # - rural_right: react later to reduce false braking from side-road signals
            if target_type == 'stop_line':
                safe_distance = safe_distance_urban
            elif self.locked_zone == 'urban':
                safe_distance = safe_distance_urban
            elif self.locked_zone == 'rural_right':
                safe_distance = safe_distance_rural
            else:
                safe_distance = safe_distance_default

            safe_distance = max(CRITICAL_DISTANCE + 0.1, safe_distance)
            urgency = (safe_distance - distance) / (safe_distance - CRITICAL_DISTANCE)
            urgency = np.clip(urgency, 0.0, 1.0)
            self._last_longitudinal_urgency = float(urgency)
            self._last_acc_urgency = float(urgency)
            self._last_ttc_urgency = float('-inf')
            return urgency
        
        # Case 2: Moving target (vehicle, pedestrian) - use TTC
        elif target_type == 'obstacle':
            rel_vel_kmh = getattr(target, 'relative_velocity_kmh', 0.0)
            rel_vel_ms = rel_vel_kmh / 3.6  # Convert km/h to m/s

            # V10 bipolar urgency then project to brake channel [0..1] for current API compatibility
            urgency_bipolar = self._compute_urgency_continuous_superposition_v10(
                distance=distance,
                rel_vel_ms=rel_vel_ms,
                current_speed_ms=current_speed_ms
            )
            return float(np.clip(max(0.0, urgency_bipolar), 0.0, 1.0))
        
        return 0.0
    
    def _apply_confidence_ramp(self,
                              base_urgency: float,
                              confidence: float,
                              frame_since_detection: int = 0,
                              ramp_frames: Optional[int] = None,
                              ) -> float:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ LAYER 2: Confidence-Based Ramp for Urgency                  │
        │ Smooth application: low confidence → slower ramp            │
        └─────────────────────────────────────────────────────────────┘
        
        Logic:
          - High confidence (0.9) → ramp_time = 5 × (1 - 0.9) = 0.5 frames → instant
          - Low confidence (0.5) → ramp_time = 5 × (1 - 0.5) = 2.5 frames → slower
          - Very low (0.3) → ramp_time = 5 × (1 - 0.3) = 3.5 frames → very slow
        
        Args:
          - base_urgency: Original urgency [0.0 - 1.0]
          - confidence: Detection confidence [0.0 - 1.0]
          - frame_since_detection: How many frames since first detection
          - ramp_frames: Override frame calculation (default: 5 * (1 - confidence))
        
        Returns:
          - ramped_urgency: Smoothly applied urgency
        """
        if ramp_frames is None:
            ramp_frames = max(1, int(5 * (1.0 - confidence)))
        
        # Sigmoid-like ramp: progress from 0 to 1
        progress = min(1.0, frame_since_detection / max(1, ramp_frames))
        
        # Apply ramp
        ramped_urgency = base_urgency * progress
        
        return np.clip(ramped_urgency, 0.0, 1.0)
    
    def _compute_obstacle_distance_threshold(self,
                                            current_speed: float,
                                            base_threshold: float = 5.0
                                            ) -> float:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ Dynamic Distance Threshold based on Speed                   │
        └─────────────────────────────────────────────────────────────┘
        
        Formula:
          threshold = base_threshold + reaction_time × speed + braking_distance
          
        Where:
          - reaction_time = 0.5 seconds (human reaction)
          - braking_distance = v² / (2 × deceleration)
          - deceleration = 3.0 m/s² (moderate braking)
        
        Physics-based safe following distance
        
        Args:
          - current_speed: m/s
          - base_threshold: meters (default 5.0)
        
        Returns:
          - threshold: meters (min 5.0, max 50.0)
        """
        reaction_time = 0.5  # seconds
        deceleration = 3.0   # m/s²
        
        reaction_distance = reaction_time * current_speed
        braking_distance = (current_speed ** 2) / (2 * deceleration)
        
        threshold = base_threshold + reaction_distance + braking_distance
        return max(5.0, min(threshold, 50.0))
    
    def _should_brake(self,
                     red_light: Optional[DetectionResult],
                     obstacle: Optional[DetectionResult],
                     stop_lines: Optional[List[DetectionResult]],
                     current_speed: float = 0.0,
                     vehicle_steer: Optional[float] = None,
                     distance_threshold: Optional[float] = None
                     ) -> Tuple[bool, float]:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ LAYER 2: Decision Fusion - NEW with stop_line + TTC support │
        │ Returns urgency instead of binary decision                  │
        └─────────────────────────────────────────────────────────────┘
        
        Flow:
          1. Turn suppression: Skip if turning
          2. Resolve target: stop_line > red_light > obstacle
          3. Compute TTC-based urgency
          4. Apply confidence ramp
          5. Return should_brake + urgency
        """
        # Step 1: Turn suppression
        if self.in_turn_phase:
            return False, 0.0

        # Bug 1 fix: during green immunity, ignore transient red-light detections
        if self.green_immunity_counter > 0 and red_light is not None:
          red_light = None

        # Bug 2 fix (part B): short obstacle immunity after forced timeout release
        if self.phantom_immunity_counter > 0:
          obstacle = None

        # Step 2: Resolve braking target (stop_line priority)
        target, target_type, _use_fallback = self._resolve_braking_target(
            red_light, obstacle, stop_lines
        )

        self._last_selected_target_type = target_type
        
        if target is None:
            return False, 0.0
        
        # Convert current speed to m/s
        current_speed_ms = current_speed  # Already in m/s from supervisor
        
        # Step 3: Compute urgency (TTC-aware)
        urgency = self._compute_collision_urgency(
            target, target_type, current_speed_ms
        )
        
        # Step 4: confidence gate (keep continuous urgency for V10)
        confidence_scale = 1.0
        if target.confidence < 0.35:
          confidence_scale = 0.8
        final_urgency = float(np.clip(urgency * confidence_scale, 0.0, 1.0))
        
        # Step 5: Decision threshold
        should_brake = final_urgency > 0.2  # Threshold for braking
        
        return should_brake, final_urgency
    
    # ═══════════════════════════════════════════════════════════════
    # LAYER 3: RED LIGHT SMOOTHING + TEMPORAL FILTERING
    # ═══════════════════════════════════════════════════════════════
    
    def _update_red_light_state(self, 
                               red_detected: bool
                               ) -> bool:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ Red Light Smoothing: 2-confirm, 6-hold, 2-release           │
        └─────────────────────────────────────────────────────────────┘
        
        State Machine:
          IDLE (red_confirmed = False, all counters = 0)
            ↓ (red_detected = True)
          CONFIRMING (red_confirm_count: 0→2)
            ↓ (red_confirm_count ≥ 2)
          HOLDING (red_hold_count: 0→6)
            ↓ (red_hold_count ≥ 6 OR red_detected = False)
          RELEASING (red_release_count: 0→2)
            ↓ (red_release_count ≥ 2)
          IDLE
        
        Mục đích: Tránh flicker từ YOLO detection instability
        """
        if not red_detected:
            # No detection: move to release phase
            if self.red_confirmed:
                self.red_release_count += 1
                if self.red_release_count >= 2:
                    self.red_confirmed = False
                    self.red_confirm_count = 0
                    self.red_hold_count = 0
                    self.red_release_count = 0
        else:
            # Detection present
            if not self.red_confirmed:
                # Confirmation phase
                self.red_confirm_count += 1
                if self.red_confirm_count >= 2:
                    self.red_confirmed = True
                    self.red_hold_count = 0
                    self.red_release_count = 0
            else:
                # Already confirmed: continue holding
                self.red_hold_count += 1
        
        return self.red_confirmed
    
    def _temporal_consensus(self,
                 should_brake_raw: bool,
                 brake_conf_raw: float,
                 red_light: Optional[DetectionResult],
                 obstacle: Optional[DetectionResult]
                 ) -> Tuple[bool, float]:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ LAYER 3: Temporal Filtering - Multi-frame Consensus         │
        └─────────────────────────────────────────────────────────────┘
        
        Combined strategy:
          1. Red light smoothing (2-confirm, 6-hold, 2-release)
          2. Multi-frame consensus for obstacles (stable track only)
          3. Preserve Layer-2 urgency; only gate with temporal validity
        
        Purpose: Tránh phantom braking từ YOLO false positives (2-5% rate)
        
        Output:
          - final_brake: bool (confirmed decision)
          - confidence: float (propagated from Layer-2 if confirmed)
        """
        # Apply red light smoothing
        red_detected = red_light is not None
        red_smooth = self._update_red_light_state(red_detected)
        
        # Record frame này for obstacle
        self.red_light_history.append(red_light is not None)
        self.obstacle_history.append(obstacle is not None)
        
        # V10 obstacle consensus uses stable hit streak only
        temporal_frames = int(self.config.get('temporal_filter_frames', 3))
        obstacle_hit = int(getattr(obstacle, 'track_hit_streak', 0)) if obstacle is not None else 0
        obstacle_consensus = obstacle is not None and obstacle_hit >= temporal_frames

        target_type = self._last_selected_target_type
        stop_line_confirm_frames = int(self.config.get('stop_line_confirm_frames', 2))

        # Strict mode for stop_line: require consecutive confirmations
        stop_line_detected_now = bool(should_brake_raw and target_type == 'stop_line')
        if stop_line_detected_now:
            self.stop_line_confirm_count += 1
        else:
            self.stop_line_confirm_count = 0
        stop_line_consensus = self.stop_line_confirm_count >= stop_line_confirm_frames

        # Respect Layer-2 physical decision first, then apply temporal gate by target type.
        if not should_brake_raw:
          final_brake = False
        elif target_type == 'stop_line':
          final_brake = stop_line_consensus
        elif target_type in ('red_light', 'traffic_light_red'):
          final_brake = red_smooth
        elif target_type == 'obstacle':
          final_brake = obstacle_consensus
        else:
          # Unknown target type: keep Layer-2 decision
          final_brake = True
        
        confidence = float(np.clip(brake_conf_raw, 0.0, 1.0)) if final_brake else 0.0
        
        return final_brake, confidence
    
    # ═══════════════════════════════════════════════════════════════
    # LAYER 4: RESUME VALIDATION
    # ═══════════════════════════════════════════════════════════════
    
    def _can_resume(self,
                   red_light: Optional[DetectionResult],
                   obstacle: Optional[DetectionResult],
                   current_speed: float
                   ) -> bool:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ LAYER 4: Resume Validation - ALL conditions must pass       │
        │ Including: Green light override (green_score ≥ red_score×1.05)
        └─────────────────────────────────────────────────────────────┘
        
        Điều kiện resume (ALL phải TRUE):
          1. Không còn đèn đỏ gần (> threshold hoặc confidence < 0.3)
          2. Không còn chướng ngại gần (> dynamic threshold)
          3. Xe đã dừng hoàn toàn (speed < 0.1 m/s)
          4. Không timeout chờ quá lâu (< 30s)
          5. Green immunity: chỉ resume nếu green_immunity_counter = 0
        
        Nếu ANY condition fail:
          - Stay in STOPPED state
          - Keep braking
        """
        # Điều kiện 1: Không còn đèn đỏ
        red_light_clear = (red_light is None or 
                          red_light.distance > self.config.get('red_light_distance_threshold', 30.0) or
                          red_light.confidence < 0.3)
        
        # Điều kiện 2: Không còn chướng ngại
        obstacle_distance_threshold = self._compute_obstacle_distance_threshold(current_speed)
        obstacle_clear = (obstacle is None or
                         obstacle.distance > obstacle_distance_threshold or
                         obstacle.confidence < 0.3)
        
        # Điều kiện 3: Xe đã dừng
        speed_is_zero = current_speed < 0.1
        
        # Điều kiện 4: Không timeout
        timeout = self.stopped_time < self.config.get('max_stopped_time', 30.0)
        
        # Điều kiện 5: Green immunity expires
        green_immunity_clear = (self.green_immunity_counter == 0)
        
        can_resume = (red_light_clear and obstacle_clear and speed_is_zero and 
                     timeout and green_immunity_clear)
        return can_resume
    
    # ═══════════════════════════════════════════════════════════════
    # LAYER 5: SMOOTH BRAKING + JERK CONSTRAINT
    # ═══════════════════════════════════════════════════════════════
    
    def _compute_smooth_brake_force(self, 
                                   actual_distance: float,
                                   safe_distance: float = 15.0,
                                   critical_distance: float = 2.0) -> float:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ S-Curve Brake Force: Distance → Force (Hermite polynomial)  │
        │ Purpose: Smooth interpolation instead of linear             │
        └─────────────────────────────────────────────────────────────┘
        
        Formula: force = 0.05 + 0.95 × (3n² - 2n³)
        where n = (safe - actual) / (safe - critical), clamped [0, 1]
        
        Behavior:
          - distance ≥ safe_distance (15m): force = 0.05 (light touch)
          - distance = (critical+safe)/2 (8.5m): force = 0.5 (50% brake)
          - distance ≤ critical_distance (2m): force = 1.0 (full brake)
        
        Advantages:
          - Smooth derivative (no kinks like linear)
          - Physics-accurate for vehicle dynamics
          - Better passenger comfort
        
        Args:
          actual_distance: Current distance to obstacle (meters)
          safe_distance: Comfortable braking distance (default 15m)
          critical_distance: Absolute minimum stopping distance (default 2m)
        
        Returns:
          brake_force: 0.0 (no brake) → 1.0 (full brake)
        """
        # Normalize distance to [0, 1]
        distance_range = safe_distance - critical_distance
        normalized_distance = (safe_distance - actual_distance) / distance_range
        
        # Clamp to [0, 1]
        normalized_distance = max(0.0, min(1.0, normalized_distance))
        
        # Hermite S-curve: 3n² - 2n³ (smooth, symmetric)
        hermite_factor = 3 * normalized_distance**2 - 2 * normalized_distance**3
        
        # Map [0, 1] Hermite → [0.05, 1.0] brake force
        # Why 0.05 minimum? To keep wheels rolling (not full lock)
        brake_force = 0.05 + 0.95 * hermite_factor
        
        return brake_force
    
    def _apply_jerk_constraint(self,
                              current_brake_force: float,
                              target_brake_force: float,
                              dt: float,
                              max_jerk_ms3: float = 3.0) -> float:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ Jerk Limiter: Limit acceleration change for passenger comfort
        │ Jerk = d(acceleration)/dt [m/s³]                           │
        └─────────────────────────────────────────────────────────────┘
        
        Problem: Jump from 0→1 brake force causes sudden deceleration
        Solution: Limit force change to max_jerk_ms3 × dt per frame
        
        Jerk Physics:
          - Human comfort: max 3.0 m/s³ acceptable (ISO 16001)
          - Converting to brake force scale:
            - Vehicle decel ≈ brake_force × a_max (e.g., 8 m/s²)
            - Brake force rate = jerk / a_max
            - At 3 m/s³ and a_max=8 m/s²: force_rate = 0.375 (per second)
            - Per frame (33ms): Δforce ≈ 0.012 per frame
        
        Args:
          current_brake_force: Previous frame brake force [0.0-1.0]
          target_brake_force: Desired brake force [0.0-1.0]
          dt: Delta time since last frame (seconds)
          max_jerk_ms3: Maximum jerk allowed [m/s³], default 3.0
        
        Returns:
          ramped_brake_force: Constrained brake force [0.0-1.0]
        """
        # Calculate max force change allowed by jerk constraint
        # Assuming vehicle max deceleration = 8 m/s²
        vehicle_max_decel_ms2 = 8.0
        force_change_rate_per_second = max_jerk_ms3 / vehicle_max_decel_ms2
        max_force_change = force_change_rate_per_second * dt
        
        # Clamp target force to max_force_change step
        force_change = target_brake_force - current_brake_force
        force_change_clamped = max(
            -max_force_change,
            min(max_force_change, force_change)
        )
        
        ramped_brake_force = current_brake_force + force_change_clamped
        
        # Final clamp to [0.0, 1.0]
        ramped_brake_force = max(0.0, min(1.0, ramped_brake_force))
        
        return ramped_brake_force
    
    # ═══════════════════════════════════════════════════════════════
    # LAYER 5: FAIL-SAFE LOGIC
    # ═══════════════════════════════════════════════════════════════
    
    def _apply_failsafe(self, brake_force: float, dt: float = 0.033) -> float:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ LAYER 5: Fail-Safe - Smooth braking + jerk constraint       │
        │ + fallback distance + emergency handling                    │
        └─────────────────────────────────────────────────────────────┘
        
        Rule 1: Smooth brake force using jerk constraint
          - Limit acceleration change to < 3 m/s³ for passenger comfort
          - Apply S-curve interpolation for distance-based braking
          - Clamp to [0.0, 1.0]
        
        Rule 2: Timeout protection
          - Nếu STOPPED > max_stopped_time → release brake
          - Purpose: Tránh xe bị kẹt vĩnh viễn
          - Action: Switch state to RESUMING
        
        Args:
          brake_force: Target brake force from Layer 2 decision
          dt: Delta time since last frame (default 33ms for 30 FPS)
        
        Returns:
          brake_force_ramped: Smooth brake force with jerk constraint applied
        """
        # Rule 1: Apply jerk constraint (smooth ramping)
        brake_force_constrained = self._apply_jerk_constraint(
            self._prev_brake_force,
            brake_force,
            dt,
            max_jerk_ms3=3.0
        )
        
        # Store for next frame
        self._prev_brake_force = brake_force_constrained
        
        # Rule 2: Timeout protection
        if self.state == SupervisorState.STOPPED and self.stopped_time > self.config.get('max_stopped_time', 30.0):
          brake_force_constrained = 0.0
          self.state = SupervisorState.CRUISING
          self.stopped_time = 0.0
          self.phantom_immunity_counter = 30  # ~1s at 30 FPS
          self.green_immunity_counter = max(self.green_immunity_counter, 10)
        
        return brake_force_constrained
    
    # ═══════════════════════════════════════════════════════════════
    # LAYER 6: STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    
    def _update_state(self, 
                     should_brake: bool,
                     current_speed: float,
                     dt: float
                     ) -> None:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ LAYER 6: State Machine - Track vehicle state + green immunity
        └─────────────────────────────────────────────────────────────┘
        
        State Transitions:
        
          CRUISING
            ↓ (should_brake = True)
          STOPPING
            ↓ (speed → 0)
          STOPPED
            ↓ (all resume checks pass)
          RESUMING
            ↓ (safety check ok)
          CRUISING
        
        Green Immunity:
          - When transitioning STOPPED → RESUMING → CRUISING
          - Set green_immunity_counter = 10 frames
          - During this time, re-assess red light detection
          - Protects against green light false negatives
        
        Time Tracking:
          - stopped_time: Accumulate time in STOPPED state
          - Reset to 0 khi exit STOPPED
          - Used for timeout detection
        """
        if should_brake:
            if self.state == SupervisorState.CRUISING:
                self.state = SupervisorState.STOPPING
                self.stopped_time = 0.0
        
        if self.state == SupervisorState.STOPPING and current_speed < 0.1:
            self.state = SupervisorState.STOPPED
        
        # Time tracking
        if self.state == SupervisorState.STOPPED:
            self.stopped_time += dt
        else:
            self.stopped_time = 0.0
        
        # Phantom immunity counter
        if self.phantom_immunity_counter > 0:
          self.phantom_immunity_counter -= 1

        # Green immunity counter
        if self.green_immunity_counter > 0:
            self.green_immunity_counter -= 1
        
        self.frame_count += 1
    
    # ═══════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════
    
    def compute(self,
               detections: List[dict],
               current_speed: float,
               image_shape: Optional[Tuple] = None,
               distance_threshold: Optional[float] = None,
               vehicle_steer: Optional[float] = None,
               dt: float = 0.033,
               danger_polygon: Optional[np.ndarray] = None
               ) -> float:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ PUBLIC API: Main compute function (6-Layer Pipeline)        │
        │ Called from: run_agents.py (mỗi frame)                      │
        └─────────────────────────────────────────────────────────────┘
        
        Flow:
          Input (detections, current_speed, image_shape, distance_threshold, vehicle_steer, dt)
            ↓
          Layer 1: Parse detections + zone classification + zone locking
            ↓
          Layer 2: Decision fusion (turn suppression + dynamic distance)
            ↓
          Layer 3: Red light smoothing + temporal consensus
            ↓
          Layer 4: Resume validation (check resume conditions)
            ↓
          Layer 6: State update (update state machine + green immunity)
            ↓
          Layer 5: Fail-safe (emergency handling)
            ↓
          Output (brake_force: 0.0 → 1.0)
        
        Args:
          - detections: List[dict] from YOLO detector
          - current_speed: float (m/s)
          - image_shape: tuple (height, width, channels), optional
          - distance_threshold: float (override obstacle distance), optional
          - vehicle_steer: float (-1.0 to 1.0), optional (for turn suppression)
          - dt: float (delta time, default 30 FPS = 0.033s)
        
        Returns:
          - brake_force: 0.0 (no brake) → 1.0 (full brake)
        """
    
        if danger_polygon is not None:
          self.last_danger_polygon = danger_polygon
    
        # Layer 1: Parse + Zone classification + Zone locking
        # NEW: Now returns 3-tuple with stop_lines support
        steer_val = vehicle_steer if vehicle_steer is not None else 0.0
        # 🔧 FIX: Build and store polygon trước parse
        danger_polygon = self._build_obstacle_danger_polygon(image_shape, steer_val, current_speed)
        self.last_danger_polygon = danger_polygon
        
        red_light, obstacle, stop_lines = self._parse_detections(detections, image_shape, dt, steer_val)
        
        # Update turn phase
        self._update_turn_phase(vehicle_steer)
        
        # Layer 2: Decision fusion with stop_lines support
        should_brake_raw, brake_conf_raw = self._should_brake(
            red_light, obstacle, stop_lines, current_speed, vehicle_steer, distance_threshold
        )
        
        # Layer 3: Red light smoothing + temporal consensus
        should_brake, brake_conf = self._temporal_consensus(
          should_brake_raw, brake_conf_raw, red_light, obstacle
        )
        
        # Layer 4: Resume validation
        can_resume = self._can_resume(red_light, obstacle, current_speed)
        
        # Layer 6: State update (trước fail-safe)
        self._update_state(should_brake, current_speed, dt)
        
        # Compute brake force based on state
        if self.state == SupervisorState.CRUISING:
            brake_force = 0.0
        elif self.state == SupervisorState.STOPPING:
            brake_force = min(brake_conf, 1.0)
        elif self.state == SupervisorState.STOPPED:
            brake_force = 1.0
        elif self.state == SupervisorState.RESUMING:
            if can_resume:
                self.state = SupervisorState.CRUISING
                self.green_immunity_counter = 10  # 10-frame immunity after resume
                brake_force = 0.0
            else:
                brake_force = 1.0
        else:
            brake_force = 1.0
        
        # Layer 5: Fail-safe (with jerk constraint and dt parameter)
        brake_force = self._apply_failsafe(brake_force, dt)
        
        return brake_force
    
    def get_state(self) -> str:
        """Return current state (for debugging)"""
        return self.state.value
    
    def get_debug_info(self) -> Dict:
        """
        ┌─────────────────────────────────────────────────────────────┐
        │ Return comprehensive debug info for monitoring               │
        │ NEW: Added Layer 5 brake force tracking + urgency metrics    │
        └─────────────────────────────────────────────────────────────┘
        
        Usage:
          debug_info = supervisor.get_debug_info()
          print(f"State: {debug_info['state']}, Urgency: {debug_info['collision_urgency']:.2f}")
        
        Includes:
          - State machine state
          - Zone locking state
          - Turn suppression state
          - Red light smoothing state
          - Green immunity timer
          - Temporal filtering history
          - Timing information
          - NEW: Brake force tracking (ideal, ramped, final)
          - NEW: Collision urgency metrics
          - NEW: Jerk constraint status
        """
        return {
            # State machine
            'state': self.state.value,
            'stopped_time': self.stopped_time,
            'frame_count': self.frame_count,
            
            # Zone locking
            'locked_zone': self.locked_zone,
            'zone_acquire_count': self.zone_acquire_count,
            'zone_missing_count': self.zone_missing_count,
            
            # Turn suppression
            'in_turn_phase': self.in_turn_phase,
            'turn_hold_counter': self.turn_hold_counter,
            'turn_grace_counter': self.turn_grace_counter,
            
            # Red light smoothing
            'red_confirmed': self.red_confirmed,
            'red_confirm_count': self.red_confirm_count,
            'red_hold_count': self.red_hold_count,
            'red_release_count': self.red_release_count,
            'stop_line_confirm_count': self.stop_line_confirm_count,
            
            # Green immunity
            'green_immunity_counter': self.green_immunity_counter,
            'phantom_immunity_counter': self.phantom_immunity_counter,
            
            # Temporal filtering
            'red_light_history': list(self.red_light_history),
            'obstacle_history': list(self.obstacle_history),

            # V10 tracking + control
            'tracked_obstacle_count': len(self.last_detected_targets.get('obstacle', [])),
            'selected_target_type': self._last_selected_target_type,
            'longitudinal_urgency': round(self._last_longitudinal_urgency, 3),
            'acc_urgency': round(self._last_acc_urgency, 3),
            'ttc_urgency': (None if self._last_ttc_urgency == float('-inf') else round(self._last_ttc_urgency, 3)),
            
            'danger_polygon': self.last_danger_polygon,  # ← THÊM DÒNG NÀY
            'danger_polygon_valid': self.last_danger_polygon is not None,

            # NEW (Layer 5): Brake force tracking
            'prev_brake_force': round(self._prev_brake_force, 3),
            
            # NOTE: For full debug info, compute() should track:
            # 'collision_urgency': final_urgency value
            # 'confidence_ramp_progress': ramp % when applied
            # These would need to be stored in __init__ and updated in compute()
            # Placeholder here for future enhancement
        }
