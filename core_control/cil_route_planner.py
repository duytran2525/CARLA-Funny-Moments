from __future__ import annotations

import math
from typing import Any, Dict, Optional


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


class CILRoutePlanner:
    """Owns route endpoint selection, planner polyline extraction, and route context."""

    def __init__(self, route_lookahead_m: float = 9.0, arrival_distance_m: float = 3.0) -> None:
        self.route_lookahead_m = max(4.0, float(route_lookahead_m))
        self.arrival_distance_m = max(1.0, float(arrival_distance_m))
        self.start_location: Any = None
        self.destination_location: Any = None
        self._last_adaptive_target_kmh: Optional[float] = None
        self.route_history_xy: list[tuple[float, float]] = []
        self.last_route_context: Dict[str, float] = {
            "route_valid": 0.0,
            "target_x_m": self.route_lookahead_m,
            "target_y_m": 0.0,
            "heading_error_deg": 0.0,
            "curvature_1pm": 0.0,
            "distance_to_turn_m": 90.0,
            "distance_to_junction_m": 90.0,
            "turn_urgency": 0.0,
            "junction_proximity": 0.0,
        }

    def reset_runtime_state(self) -> None:
        self.route_history_xy = []
        self._last_adaptive_target_kmh = None
        self.last_route_context = {
            "route_valid": 0.0,
            "target_x_m": self.route_lookahead_m,
            "target_y_m": 0.0,
            "heading_error_deg": 0.0,
            "curvature_1pm": 0.0,
            "distance_to_turn_m": 90.0,
            "distance_to_junction_m": 90.0,
            "turn_urgency": 0.0,
            "junction_proximity": 0.0,
        }

    @staticmethod
    def _xy_distance(a: Any, b: Any) -> float:
        return math.hypot(float(a.x - b.x), float(a.y - b.y))

    @staticmethod
    def _planner_item_to_waypoint(item: Any):
        if hasattr(item, "transform"):
            return item
        if isinstance(item, (tuple, list)) and len(item) >= 1 and hasattr(item[0], "transform"):
            return item[0]
        return None

    @staticmethod
    def _normalize_angle_deg(angle_deg: float) -> float:
        return (float(angle_deg) + 180.0) % 360.0 - 180.0

    @staticmethod
    def _sample_polyline_point(
        points_xy: list[tuple[float, float]],
        cumulative_s: list[float],
        target_s: float,
    ) -> Optional[tuple[float, float]]:
        if not points_xy or not cumulative_s:
            return None
        if len(points_xy) == 1:
            return points_xy[0]

        target = max(0.0, float(target_s))
        if target <= cumulative_s[0]:
            return points_xy[0]
        if target >= cumulative_s[-1]:
            return points_xy[-1]

        for idx in range(1, len(cumulative_s)):
            if cumulative_s[idx] < target:
                continue
            s0 = cumulative_s[idx - 1]
            s1 = cumulative_s[idx]
            p0 = points_xy[idx - 1]
            p1 = points_xy[idx]
            ratio = (target - s0) / max(1e-6, s1 - s0)
            x = p0[0] + ratio * (p1[0] - p0[0])
            y = p0[1] + ratio * (p1[1] - p0[1])
            return (float(x), float(y))
        return points_xy[-1]

    @staticmethod
    def _signed_curvature_3pts(
        p0: tuple[float, float],
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> float:
        ax = float(p1[0] - p0[0])
        ay = float(p1[1] - p0[1])
        bx = float(p2[0] - p1[0])
        by = float(p2[1] - p1[1])
        cx = float(p2[0] - p0[0])
        cy = float(p2[1] - p0[1])

        a = math.hypot(ax, ay)
        b = math.hypot(bx, by)
        c = math.hypot(cx, cy)
        if a <= 1e-4 or b <= 1e-4 or c <= 1e-4:
            return 0.0

        cross = ax * cy - ay * cx
        return float((2.0 * cross) / max(1e-6, a * b * c))

    def configure_endpoints(
        self,
        spawn_points: list[Any],
        vehicle_location: Any,
        configured_spawn_index: int,
        configured_destination_index: int,
    ) -> list[str]:
        warnings: list[str] = []
        self.start_location = vehicle_location
        self.destination_location = None

        if not spawn_points:
            warnings.append("No spawn points found. Route endpoint configuration skipped.")
            return warnings

        spawn_count = len(spawn_points)
        start_idx = None
        if configured_spawn_index >= 0:
            start_idx = configured_spawn_index % spawn_count
            configured_start = spawn_points[start_idx].location
            self.start_location = configured_start
            if vehicle_location is not None and self._xy_distance(vehicle_location, configured_start) > 2.0:
                warnings.append(
                    "Ego actual pose differs from configured spawn_point by >2m; keeping configured spawn_point as S marker."
                )
        elif vehicle_location is None:
            self.start_location = spawn_points[0].location
            warnings.append(
                "vehicle.spawn_point is negative and ego location is unavailable. Falling back to spawn index 0 as route start."
            )

        if configured_destination_index >= 0:
            dest_idx = configured_destination_index % spawn_count
            if start_idx is not None and dest_idx == start_idx and spawn_count > 1:
                dest_idx = (dest_idx + 1) % spawn_count
                warnings.append(
                    f"vehicle.destination_point equals spawn_point; shifted destination to index {dest_idx}."
                )
        else:
            base_idx = start_idx if start_idx is not None else 0
            dest_idx = (base_idx + 1) % spawn_count
            warnings.append(
                "vehicle.destination_point is negative. Using deterministic fallback destination index "
                f"{dest_idx}."
            )

        self.destination_location = spawn_points[dest_idx].location
        return warnings

    def distance_to_destination(self, vehicle_location: Any) -> Optional[float]:
        if vehicle_location is None or self.destination_location is None:
            return None
        dx = float(vehicle_location.x - self.destination_location.x)
        dy = float(vehicle_location.y - self.destination_location.y)
        dz = float(vehicle_location.z - self.destination_location.z)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def collect_route_locations(
        self,
        nav_agent: Any,
        max_points: int = 260,
        anchor_location: Any = None,
    ) -> list[Any]:
        if nav_agent is None:
            return []

        planner = None
        if hasattr(nav_agent, "get_local_planner"):
            planner = nav_agent.get_local_planner()
        if planner is None:
            return []

        route_locations: list[Any] = []

        def append_location(location: Any) -> None:
            if location is None:
                return
            if route_locations:
                prev = route_locations[-1]
                if self._xy_distance(location, prev) < 0.10:
                    return
            route_locations.append(location)

        def append_from(items: Any, limit: int) -> None:
            if not items:
                return
            for item in list(items)[: max(1, int(limit))]:
                waypoint = self._planner_item_to_waypoint(item)
                if waypoint is None:
                    continue
                append_location(waypoint.transform.location)

        buffer_attr = getattr(planner, "_waypoint_buffer", None)
        append_from(buffer_attr, max(8, max_points // 4))

        queue_attr = getattr(planner, "_waypoints_queue", None)
        append_from(queue_attr, max_points)

        # Keep planner polyline pure to avoid distorting route context/command logic.
        # S/D are rendered separately via markers in the map overlay.
        if anchor_location is not None and not route_locations:
            route_locations.append(anchor_location)

        if max_points > 0 and len(route_locations) > max_points:
            if self.destination_location is not None:
                trimmed = route_locations[: max_points - 1]
                trimmed.append(self.destination_location)
                route_locations = trimmed
            else:
                route_locations = route_locations[:max_points]

        return route_locations

    def update_route_history(self, location: Any, max_size: int = 1200, min_step_m: float = 0.30) -> None:
        if location is None:
            return

        current = (float(location.x), float(location.y))
        if self.route_history_xy:
            prev_x, prev_y = self.route_history_xy[-1]
            if math.hypot(current[0] - prev_x, current[1] - prev_y) < float(min_step_m):
                return

        self.route_history_xy.append(current)
        if len(self.route_history_xy) > int(max_size):
            self.route_history_xy = self.route_history_xy[-int(max_size) :]

    def _fallback_route_context_from_destination(
        self,
        base_context: Dict[str, float],
        vehicle_loc: Any,
        forward: Any,
        right: Any,
        lookahead_m: float,
        near_junction: bool,
    ) -> Dict[str, float]:
        if self.destination_location is None:
            self.last_route_context = base_context
            return base_context

        dx = float(self.destination_location.x - vehicle_loc.x)
        dy = float(self.destination_location.y - vehicle_loc.y)
        dist_xy = math.hypot(dx, dy)

        target_x = float(forward.x) * dx + float(forward.y) * dy
        target_y = -(float(right.x) * dx + float(right.y) * dy)
        target_x = clamp(target_x, -80.0, 80.0)
        target_y = clamp(target_y, -40.0, 40.0)

        x_for_heading = target_x if abs(target_x) > 1e-3 else (1e-3 if target_y >= 0.0 else -1e-3)
        heading_error_deg = math.degrees(math.atan2(target_y, x_for_heading))
        curvature_1pm = math.tan(math.radians(heading_error_deg)) / max(6.0, float(lookahead_m))

        heading_urgency = clamp((abs(heading_error_deg) - 4.0) / 22.0, 0.0, 1.0)
        turn_urgency = max(heading_urgency, clamp(abs(curvature_1pm) / 0.10, 0.0, 1.0))
        junction_proximity = 0.45 if near_junction else 0.0

        fallback = dict(base_context)
        fallback.update(
            {
                "route_valid": 0.6,
                "target_x_m": float(target_x),
                "target_y_m": float(target_y),
                "heading_error_deg": float(heading_error_deg),
                "curvature_1pm": float(curvature_1pm),
                "distance_to_turn_m": float(max(2.0, min(90.0, 0.35 * dist_xy))),
                "distance_to_junction_m": float(12.0 if junction_proximity > 0.0 else 90.0),
                "turn_urgency": float(turn_urgency),
                "junction_proximity": float(junction_proximity),
            }
        )
        self.last_route_context = fallback
        return fallback

    def _distance_to_next_junction_m(
        self,
        vehicle_loc: Any,
        route_locations: list[Any],
        world_map: Any,
        max_probe_m: float = 60.0,
    ) -> float:
        if world_map is None or not route_locations or vehicle_loc is None:
            return float("inf")

        travelled = 0.0
        prev_loc = vehicle_loc
        for idx, loc in enumerate(route_locations[:80]):
            seg = self._xy_distance(loc, prev_loc)
            prev_loc = loc
            if seg < 0.15:
                continue

            travelled += seg
            if travelled > max_probe_m:
                break
            if idx % 3 != 0 and travelled > 4.0:
                continue

            try:
                waypoint = world_map.get_waypoint(loc, project_to_road=True)
            except Exception:
                waypoint = None
            if waypoint is not None and getattr(waypoint, "is_junction", False):
                return float(travelled)

        return float("inf")

    def compute_route_context(
        self,
        vehicle_location: Any,
        vehicle_transform: Any,
        route_locations: list[Any],
        world_map: Any = None,
        lookahead_m: Optional[float] = None,
        near_junction: bool = False,
    ) -> Dict[str, float]:
        if lookahead_m is None:
            lookahead_m = self.route_lookahead_m
        lookahead_m = max(4.0, float(lookahead_m))

        context: Dict[str, float] = {
            "route_valid": 0.0,
            "target_x_m": float(lookahead_m),
            "target_y_m": 0.0,
            "heading_error_deg": 0.0,
            "curvature_1pm": 0.0,
            "distance_to_turn_m": 90.0,
            "distance_to_junction_m": 90.0,
            "turn_urgency": 0.0,
            "junction_proximity": 0.0,
        }

        if vehicle_location is None or vehicle_transform is None:
            self.last_route_context = context
            return context

        forward = vehicle_transform.get_forward_vector()
        right = vehicle_transform.get_right_vector()

        if not route_locations:
            return self._fallback_route_context_from_destination(
                base_context=context,
                vehicle_loc=vehicle_location,
                forward=forward,
                right=right,
                lookahead_m=lookahead_m,
                near_junction=near_junction,
            )

        nearest_idx = -1
        nearest_dist = float("inf")
        for idx, loc in enumerate(route_locations):
            dist = self._xy_distance(loc, vehicle_location)
            if dist < nearest_dist:
                nearest_idx = idx
                nearest_dist = dist

        if nearest_idx < 0:
            return self._fallback_route_context_from_destination(
                base_context=context,
                vehicle_loc=vehicle_location,
                forward=forward,
                right=right,
                lookahead_m=lookahead_m,
                near_junction=near_junction,
            )

        start_idx = max(0, nearest_idx - 2)
        route_local_world = route_locations[start_idx : min(len(route_locations), start_idx + 90)]
        if len(route_local_world) < 2:
            return self._fallback_route_context_from_destination(
                base_context=context,
                vehicle_loc=vehicle_location,
                forward=forward,
                right=right,
                lookahead_m=lookahead_m,
                near_junction=near_junction,
            )

        points_xy: list[tuple[float, float]] = []
        for loc in route_local_world:
            dx = float(loc.x - vehicle_location.x)
            dy = float(loc.y - vehicle_location.y)
            x_forward = float(forward.x) * dx + float(forward.y) * dy
            y_right = float(right.x) * dx + float(right.y) * dy
            y_left = -y_right
            if x_forward < -5.0:
                continue
            points_xy.append((x_forward, y_left))

        if len(points_xy) < 2:
            return self._fallback_route_context_from_destination(
                base_context=context,
                vehicle_loc=vehicle_location,
                forward=forward,
                right=right,
                lookahead_m=lookahead_m,
                near_junction=near_junction,
            )

        filtered_points: list[tuple[float, float]] = [points_xy[0]]
        for point in points_xy[1:]:
            prev = filtered_points[-1]
            if math.hypot(point[0] - prev[0], point[1] - prev[1]) < 0.20:
                continue
            filtered_points.append(point)

        if len(filtered_points) < 2:
            return self._fallback_route_context_from_destination(
                base_context=context,
                vehicle_loc=vehicle_location,
                forward=forward,
                right=right,
                lookahead_m=lookahead_m,
                near_junction=near_junction,
            )

        cumulative_s: list[float] = [0.0]
        for idx in range(1, len(filtered_points)):
            p0 = filtered_points[idx - 1]
            p1 = filtered_points[idx]
            cumulative_s.append(cumulative_s[-1] + math.hypot(p1[0] - p0[0], p1[1] - p0[1]))

        path_len = cumulative_s[-1]
        if path_len <= 1e-3:
            return self._fallback_route_context_from_destination(
                base_context=context,
                vehicle_loc=vehicle_location,
                forward=forward,
                right=right,
                lookahead_m=lookahead_m,
                near_junction=near_junction,
            )

        target_s = min(path_len, max(4.0, lookahead_m))
        target_point = self._sample_polyline_point(filtered_points, cumulative_s, target_s)
        if target_point is None:
            target_point = filtered_points[-1]

        target_x = float(clamp(target_point[0], -80.0, 80.0))
        target_y = float(target_point[1])
        x_for_heading = target_x if abs(target_x) > 1e-3 else (1e-3 if target_y >= 0.0 else -1e-3)
        heading_error_deg = math.degrees(math.atan2(target_y, x_for_heading))

        s0 = min(path_len, max(2.5, 0.40 * target_s))
        s1 = min(path_len, max(5.0, 0.75 * target_s))
        s2 = min(path_len, max(8.0, 1.20 * target_s))
        p0 = self._sample_polyline_point(filtered_points, cumulative_s, s0) or filtered_points[0]
        p1 = self._sample_polyline_point(filtered_points, cumulative_s, s1) or filtered_points[-1]
        p2 = self._sample_polyline_point(filtered_points, cumulative_s, s2) or filtered_points[-1]
        curvature_1pm = self._signed_curvature_3pts(p0, p1, p2)

        distance_to_turn_m = 90.0
        heading_prev = None
        for idx in range(1, len(filtered_points)):
            s = cumulative_s[idx]
            p_prev = filtered_points[idx - 1]
            p_now = filtered_points[idx]
            seg_dx = float(p_now[0] - p_prev[0])
            seg_dy = float(p_now[1] - p_prev[1])
            ds = max(1e-3, math.hypot(seg_dx, seg_dy))
            heading_now = math.degrees(math.atan2(seg_dy, seg_dx))

            if heading_prev is None:
                heading_prev = heading_now
                continue

            heading_delta = self._normalize_angle_deg(heading_now - heading_prev)
            local_curvature = math.radians(heading_delta) / ds
            heading_prev = heading_now
            if s < 1.5:
                continue
            if abs(heading_delta) > 7.0 or abs(local_curvature) > 0.065 or abs(filtered_points[idx][1]) > 2.0:
                distance_to_turn_m = float(s)
                break

        distance_to_junction_m = self._distance_to_next_junction_m(
            vehicle_loc=vehicle_location,
            route_locations=route_local_world,
            world_map=world_map,
            max_probe_m=60.0,
        )

        if near_junction:
            distance_to_junction_m = min(distance_to_junction_m, 0.0)

        if math.isfinite(distance_to_junction_m):
            junction_proximity = math.exp(-distance_to_junction_m / 14.0)
        else:
            distance_to_junction_m = 90.0
            junction_proximity = 0.0

        turn_center_m = max(8.0, lookahead_m)
        turn_urgency = 1.0 / (1.0 + math.exp((distance_to_turn_m - turn_center_m) / 3.0))
        heading_urgency = clamp((abs(heading_error_deg) - 3.0) / 18.0, 0.0, 1.0)
        curvature_urgency = clamp(abs(curvature_1pm) / 0.10, 0.0, 1.0)
        turn_urgency = max(turn_urgency, heading_urgency, curvature_urgency)

        context = {
            "route_valid": 1.0,
            "target_x_m": target_x,
            "target_y_m": target_y,
            "heading_error_deg": float(heading_error_deg),
            "curvature_1pm": float(curvature_1pm),
            "distance_to_turn_m": float(distance_to_turn_m),
            "distance_to_junction_m": float(distance_to_junction_m),
            "turn_urgency": clamp(float(turn_urgency), 0.0, 1.0),
            "junction_proximity": clamp(float(junction_proximity), 0.0, 1.0),
        }
        self.last_route_context = context
        return context

    @staticmethod
    def _argmax_weight(weights: Any) -> Optional[int]:
        if weights is None:
            return None
        if hasattr(weights, "detach"):
            try:
                values = weights.detach().cpu().numpy().tolist()
            except Exception:
                values = None
            if isinstance(values, list) and len(values) >= 4:
                return int(max(range(4), key=lambda idx: float(values[idx])))
        if isinstance(weights, (list, tuple)) and len(weights) >= 4:
            return int(max(range(4), key=lambda idx: float(weights[idx])))
        return None

    def command_from_context(self, route_context: Dict[str, float], blend_weights: Any = None) -> int:
        route_valid = float(route_context.get("route_valid", 0.0))
        heading_error_deg = float(route_context.get("heading_error_deg", 0.0))
        curvature_1pm = float(route_context.get("curvature_1pm", 0.0))
        target_y_m = float(route_context.get("target_y_m", 0.0))
        turn_urgency = float(route_context.get("turn_urgency", 0.0))
        distance_to_turn_m = float(route_context.get("distance_to_turn_m", 90.0))
        distance_to_junction_m = float(route_context.get("distance_to_junction_m", 90.0))

        if route_valid < 0.5:
            best = self._argmax_weight(blend_weights)
            return 0 if best is None else int(best)

        signed_turn = heading_error_deg + math.degrees(math.atan(curvature_1pm * 8.0)) + 8.0 * clamp(target_y_m / 3.0, -1.0, 1.0)

        if turn_urgency < 0.25 and distance_to_turn_m > 25.0 and abs(signed_turn) < 6.0:
            return 0

        if signed_turn > 5.5:
            return 1
        if signed_turn < -5.5:
            return 2

        if turn_urgency >= 0.45 or distance_to_turn_m < 14.0 or distance_to_junction_m < 12.0:
            return 3

        best = self._argmax_weight(blend_weights)
        return 0 if best is None else int(best)

    def compute_adaptive_target_speed_kmh(
        self,
        base_target_speed_kmh: float,
        current_speed_kmh: float,
        route_context: Dict[str, float],
        destination_distance_m: Optional[float],
        dt_s: float = 0.05,
    ) -> tuple[float, Dict[str, float]]:
        base_target = max(5.0, float(base_target_speed_kmh))
        adaptive_target = base_target
        dt_s = clamp(float(dt_s), 0.01, 0.20)

        dest_cap = base_target
        distance_m: Optional[float] = None
        if destination_distance_m is not None and math.isfinite(float(destination_distance_m)):
            distance_m = max(0.0, float(destination_distance_m))
            stop_buffer = max(3.6, self.arrival_distance_m + 0.8)
            if distance_m <= 45.0:
                comfort_decel_ms2 = 2.8
                available = max(0.0, distance_m - stop_buffer)
                v_stop_kmh = math.sqrt(max(0.0, 2.0 * comfort_decel_ms2 * available)) * 3.6
                dest_cap = clamp(v_stop_kmh, 9.0, base_target)
                adaptive_target = min(adaptive_target, dest_cap)

                if distance_m < 18.0:
                    near_dest_cap = clamp(4.5 + 0.65 * distance_m, 4.0, base_target)
                    adaptive_target = min(adaptive_target, near_dest_cap)
                    dest_cap = min(dest_cap, near_dest_cap)

        far_destination = distance_m is None or distance_m > 28.0

        turn_urgency = clamp(float(route_context.get("turn_urgency", 0.0)), 0.0, 1.0)
        distance_to_turn_m = max(0.0, float(route_context.get("distance_to_turn_m", 90.0)))
        curvature_abs = abs(float(route_context.get("curvature_1pm", 0.0)))
        junction_proximity = clamp(float(route_context.get("junction_proximity", 0.0)), 0.0, 1.0)

        turn_cap = base_target
        if turn_urgency > 0.05 or distance_to_turn_m < 40.0:
            urgency_cap = base_target * (1.0 - 0.34 * turn_urgency)
            curvature_cap = 56.0 / (1.0 + 9.0 * curvature_abs)
            proximity_scale = clamp((distance_to_turn_m - 2.0) / 24.0, 0.68, 1.0)
            turn_floor = 15.0 if far_destination else 9.0
            turn_cap = max(turn_floor, min(base_target, min(urgency_cap, curvature_cap) * proximity_scale))
            adaptive_target = min(adaptive_target, turn_cap)

        junction_cap = base_target
        if junction_proximity > 0.20:
            junction_floor = 15.0 if far_destination else 9.0
            junction_cap = clamp(base_target * (1.0 - 0.24 * junction_proximity), junction_floor, base_target)
            adaptive_target = min(adaptive_target, junction_cap)

        if far_destination and turn_urgency < 0.35 and distance_to_turn_m > 10.0 and junction_proximity < 0.45:
            recovery_floor = max(20.0, 0.74 * base_target)
            adaptive_target = max(adaptive_target, recovery_floor)

        if self._last_adaptive_target_kmh is None:
            self._last_adaptive_target_kmh = float(adaptive_target)
        else:
            max_drop_kmh = 28.0 * dt_s
            max_rise_kmh = 55.0 * dt_s
            lo = self._last_adaptive_target_kmh - max_drop_kmh
            hi = self._last_adaptive_target_kmh + max_rise_kmh
            adaptive_target = clamp(adaptive_target, lo, hi)
            self._last_adaptive_target_kmh = float(adaptive_target)

        adaptive_target = clamp(adaptive_target, 4.0, base_target)
        overspeed_kmh = max(0.0, float(current_speed_kmh) - adaptive_target)

        speed_plan = {
            "base_target_kmh": float(base_target),
            "adaptive_target_kmh": float(adaptive_target),
            "dest_cap_kmh": float(dest_cap),
            "turn_cap_kmh": float(turn_cap),
            "junction_cap_kmh": float(junction_cap),
            "overspeed_kmh": float(overspeed_kmh),
            "far_destination": 1.0 if far_destination else 0.0,
        }
        return float(adaptive_target), speed_plan
