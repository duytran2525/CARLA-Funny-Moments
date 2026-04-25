from __future__ import annotations

import math
from typing import Any, Optional


class CILRoutePlanner:
    """Owns route endpoints, planner polyline extraction, and route history."""

    def __init__(self, arrival_distance_m: float = 3.0) -> None:
        self.arrival_distance_m = max(1.0, float(arrival_distance_m))
        self.start_location: Any = None
        self.destination_location: Any = None
        self.route_history_xy: list[tuple[float, float]] = []

    def reset_runtime_state(self) -> None:
        self.route_history_xy = []

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

    def _nearest_route_index(
        self,
        route_locations: list[Any],
        anchor_location: Any,
        max_probe: int = 120,
    ) -> int:
        if not route_locations or anchor_location is None:
            return 0

        probe_count = max(1, min(int(max_probe), len(route_locations)))
        best_idx = 0
        best_dist = float("inf")
        for idx, loc in enumerate(route_locations[:probe_count]):
            dist = self._xy_distance(loc, anchor_location)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def _sanitize_route_polyline(
        self,
        route_locations: list[Any],
        anchor_location: Any,
        max_gap_m: float = 35.0,
    ) -> list[Any]:
        if not route_locations:
            return []

        start_idx = self._nearest_route_index(route_locations, anchor_location)

        sanitized: list[Any] = [route_locations[start_idx]]
        for loc in route_locations[start_idx + 1 :]:
            seg = self._xy_distance(loc, sanitized[-1])
            if seg < 0.10:
                continue
            if seg > float(max_gap_m):
                break
            sanitized.append(loc)

        if len(sanitized) < 3 and start_idx > 0:
            sanitized = [route_locations[0]]
            for loc in route_locations[1:]:
                seg = self._xy_distance(loc, sanitized[-1])
                if seg < 0.10:
                    continue
                if seg > float(max_gap_m):
                    break
                sanitized.append(loc)

        return sanitized

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

        if route_locations:
            route_locations = self._sanitize_route_polyline(
                route_locations,
                anchor_location=anchor_location,
                max_gap_m=35.0,
            )

        if anchor_location is not None and not route_locations:
            route_locations.append(anchor_location)

        if max_points > 0 and len(route_locations) > max_points:
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
