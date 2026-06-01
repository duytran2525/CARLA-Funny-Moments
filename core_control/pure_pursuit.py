import math
import numpy as np


class PurePursuitController:
    """
    Pure Pursuit lateral controller — V5 (Roundabout-safe).

    Improvements over V4.1:
    - Interpolated look-ahead along the waypoint polyline (not just nearest
      discrete point) — critical for tight curves where waypoints are sparse.
    - Curvature-adaptive look-ahead that shortens when the path curves sharply.
    - Higher default responsiveness (steer_alpha, max_steer_rate) tuned for
      urban driving with roundabouts and tight intersections at ≤30 km/h.
    - Speed-dependent steer_alpha: more reactive at low speed, smoother at
      high speed.
    """

    def __init__(
        self,
        wheelbase: float = 2.87,
        max_steer_angle_deg: float = 35.0,
        k_lookahead: float = 0.45,
        min_lookahead: float = 3.0,       # ↑ from 2.0 — look further ahead to avoid cutting corners
        max_lookahead: float = 12.0,
        steer_alpha: float = 0.40,        # ↓ from 0.55 — smoother turn-in
        max_steer_rate: float = 1.5,      # ↓ from 3.0 — limit jerk at turn entry
        dt: float = 0.05,
    ):
        self.L = wheelbase
        self.max_steer_rad = np.radians(max_steer_angle_deg)

        self.k_lookahead = k_lookahead
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead

        self.steer_alpha = steer_alpha
        self.max_steer_rate = max_steer_rate
        self.dt = dt
        self.max_steer_change = max_steer_rate * dt

        self.prev_steer = 0.0

    # ── Look-ahead with polyline interpolation ──────────────────────────
    def _interpolate_on_polyline(
        self,
        waypoints: np.ndarray,
        target_dist: float,
    ) -> tuple[float, float]:
        """
        Walk along the waypoint polyline and interpolate the exact point at
        `target_dist` metres from the ego origin.  Falls back to the farthest
        waypoint if the polyline is shorter than target_dist.
        """
        if len(waypoints) == 0:
            return 0.0, 0.0
        if len(waypoints) == 1:
            return float(waypoints[0, 0]), float(waypoints[0, 1])

        cum_dist = 0.0
        for i in range(1, len(waypoints)):
            seg = np.linalg.norm(waypoints[i] - waypoints[i - 1])
            if seg < 1e-6:
                continue
            if cum_dist + seg >= target_dist:
                # Interpolate within this segment
                remaining = target_dist - cum_dist
                t = remaining / seg
                pt = waypoints[i - 1] + t * (waypoints[i] - waypoints[i - 1])
                return float(pt[0]), float(pt[1])
            cum_dist += seg

        # Polyline too short — return last point
        return float(waypoints[-1, 0]), float(waypoints[-1, 1])

    def _estimate_path_curvature(self, waypoints: np.ndarray) -> float:
        """
        Estimate average curvature of the first ~8 m of the path.
        Returns curvature in rad/m (0 = straight).
        """
        if len(waypoints) < 3:
            return 0.0

        total_angle = 0.0
        total_dist = 0.0
        for i in range(1, min(len(waypoints) - 1, 8)):
            v1 = waypoints[i] - waypoints[i - 1]
            v2 = waypoints[i + 1] - waypoints[i]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            total_angle += np.arccos(cos_a)
            total_dist += n1
        if total_dist < 0.5:
            return 0.0
        return float(total_angle / total_dist)

    def get_target_waypoint(
        self,
        waypoints: np.ndarray,
        current_speed_ms: float,
    ) -> tuple[float, float]:
        """
        Compute dynamic look-ahead distance and return the interpolated
        target point on the waypoint polyline.
        """
        # Base look-ahead from speed
        ld_base = max(
            self.min_lookahead,
            min(self.max_lookahead, self.k_lookahead * current_speed_ms),
        )

        # Shorten look-ahead when path is curvy (roundabout protection)
        curv = self._estimate_path_curvature(waypoints)
        curvature_factor = max(0.4, 1.0 - 3.0 * curv)  # 0.4..1.0
        ld_target = max(self.min_lookahead, ld_base * curvature_factor)

        return self._interpolate_on_polyline(waypoints, ld_target)

    def compute_steering(
        self,
        waypoints: np.ndarray,
        current_speed_kmh: float,
    ) -> float:
        """
        Given N waypoints (ego-frame, shape [N,2]) and current speed,
        return steering in [-1.0, 1.0].
        """
        waypoints = np.asarray(waypoints, dtype=np.float64)
        if waypoints.ndim != 2 or waypoints.shape[0] < 1:
            return float(self.prev_steer)

        current_speed_ms = current_speed_kmh / 3.6

        target_x, target_y = self.get_target_waypoint(waypoints, current_speed_ms)

        Ld_sq = target_x ** 2 + target_y ** 2
        if Ld_sq < 1e-4:
            raw_steer_normalized = 0.0
        else:
            curvature = (2.0 * self.L * target_y) / Ld_sq
            steering_rad = math.atan(curvature)
            raw_steer_normalized = float(
                np.clip(steering_rad / self.max_steer_rad, -1.0, 1.0)
            )

        # Speed-adaptive alpha: more reactive at low speed
        if current_speed_ms < 4.0:  # < ~14 km/h
            alpha = min(0.80, self.steer_alpha + 0.25)
        elif current_speed_ms < 8.0:  # < ~29 km/h
            alpha = self.steer_alpha
        else:
            alpha = max(0.30, self.steer_alpha - 0.10)

        smoothed_steer = alpha * raw_steer_normalized + (1.0 - alpha) * self.prev_steer

        # Rate limiter
        max_change = self.max_steer_change
        final_steer = float(np.clip(
            smoothed_steer,
            self.prev_steer - max_change,
            self.prev_steer + max_change,
        ))

        self.prev_steer = final_steer
        return final_steer

    def reset(self) -> None:
        """Reset steering state on respawn."""
        self.prev_steer = 0.0
