import numpy as np
import random


class PurePursuitController:
    """
    Bộ điều khiển Lateral Control (Vô lăng) Pure Pursuit - Bản Thực chiến V4.1.
    Tích hợp: Dynamic Look-ahead, Low-pass Filter (Chống rung), Rate Limiter (Chống lật).
    """

    def __init__(
        self,
        wheelbase: float = 2.87,
        max_steer_angle_deg: float = 35.0,
        k_lookahead: float = 0.5,  # Hệ số nhân tốc độ cho Look-ahead
        min_lookahead: float = 3.0,  # Điểm nhìn tối thiểu (mét)
        max_lookahead: float = 15.0,  # Điểm nhìn tối đa (mét)
        steer_alpha: float = 0.3,  # Hệ số Low-pass filter (Càng nhỏ càng mượt, nhưng trễ)
        max_steer_rate: float = 1.0,  # Tốc độ vặn vô lăng tối đa (Normalized/sec)
        dt: float = 0.05,  # Thời gian 1 frame (20 FPS)
        lookahead_jitter: float = 0.0,  # ±jitter fraction on look-ahead (e.g. 0.15 = ±15%)
    ):
        self.L = wheelbase
        self.max_steer_rad = np.radians(max_steer_angle_deg)

        # Look-ahead params
        self.k_lookahead = k_lookahead
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.lookahead_jitter = float(lookahead_jitter)

        # Smoothing & Limits
        self.steer_alpha = steer_alpha
        self.max_steer_change = max_steer_rate * dt

        # State (Ký ức)
        self.prev_steer = 0.0

    def get_target_waypoint(self, waypoints, current_speed_ms):
        """
        Tính toán khoảng cách Look-ahead liên tục và nội suy điểm mục tiêu.
        waypoints: Mảng numpy shape (5, 2) tọa độ Ego-centric.
        """
        # Apply look-ahead jitter (±jitter%) for humanized driving
        if self.lookahead_jitter > 0.0:
            jitter_factor = 1.0 + (random.random() - 0.5) * 2.0 * self.lookahead_jitter
        else:
            jitter_factor = 1.0

        ld_target = max(
            self.min_lookahead,
            min(self.max_lookahead, self.k_lookahead * current_speed_ms * jitter_factor),
        )

        distances = np.linalg.norm(waypoints, axis=1)

        for i in range(len(distances)):
            if distances[i] >= ld_target:
                return waypoints[i][0], waypoints[i][1]

        return waypoints[-1][0], waypoints[-1][1]

    def compute_steering(self, waypoints, current_speed_kmh):
        """
        Nhận vào 5 Waypoint và Tốc độ -> Trả ra vô lăng [-1.0, 1.0] đã làm mượt.
        """
        current_speed_ms = current_speed_kmh / 3.6

        target_x, target_y = self.get_target_waypoint(waypoints, current_speed_ms)

        Ld_sq = target_x**2 + target_y**2
        if Ld_sq < 1e-4:
            raw_steer_normalized = 0.0
        else:
            curvature = (2.0 * self.L * target_y) / Ld_sq
            steering_rad = np.arctan(curvature)
            raw_steer_normalized = float(
                np.clip(steering_rad / self.max_steer_rad, -1.0, 1.0)
            )

        smoothed_steer = (self.steer_alpha * raw_steer_normalized) + (
            (1.0 - self.steer_alpha) * self.prev_steer
        )

        final_steer = np.clip(
            smoothed_steer,
            self.prev_steer - self.max_steer_change,
            self.prev_steer + self.max_steer_change,
        )

        self.prev_steer = final_steer
        return float(final_steer)

    def reset(self):
        """Khởi động lại vô lăng khi respawn xe."""
        self.prev_steer = 0.0
