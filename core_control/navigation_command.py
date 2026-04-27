from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

DEFAULT_COMMAND_PREP_TIME_S = 1.8
DEFAULT_COMMAND_TRIGGER_MIN_M = 8.0
DEFAULT_COMMAND_TRIGGER_MAX_M = 25.0
DEFAULT_COMMAND_MAX_ARMED_FRAMES = 160
COMMAND_TURN_ARM_MAX_M = 12.0
COMMAND_STRAIGHT_ARM_MAX_M = 16.0
COMMAND_NEAR_JUNCTION_HOLD_M = 10.0
COMMAND_PROGRESS_EPS_M = 0.35
REFERENCE_ROUTE_MAX_ITEMS = 4096


def _xy_distance(a: Any, b: Any) -> float:
    return math.hypot(float(a.x - b.x), float(a.y - b.y))


def _planner_item_to_waypoint(item: Any):
    if hasattr(item, "transform"):
        return item
    if isinstance(item, (tuple, list)) and len(item) >= 1 and hasattr(item[0], "transform"):
        return item[0]
    return None


def _finalize_reference_route_entries(raw_entries: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    if not raw_entries:
        return []

    dedup_entries: list[Dict[str, Any]] = []
    for entry in raw_entries:
        location = entry.get("location")
        if location is None:
            continue
        raw_command = int(entry.get("raw_command", 0))
        is_junction = bool(entry.get("is_junction", False))

        if dedup_entries and _xy_distance(dedup_entries[-1]["location"], location) < 0.10:
            if int(dedup_entries[-1].get("raw_command", 0)) == 0 and raw_command != 0:
                dedup_entries[-1]["raw_command"] = int(raw_command)
            if is_junction:
                dedup_entries[-1]["is_junction"] = True
            continue

        dedup_entries.append(
            {
                "location": location,
                "raw_command": int(raw_command),
                "is_junction": bool(is_junction),
            }
        )

    route_plan: list[Dict[str, Any]] = [
        {
            "location": entry["location"],
            "command": 0,
            "is_junction": bool(entry.get("is_junction", False)),
        }
        for entry in dedup_entries
    ]

    idx = 0
    route_len = len(dedup_entries)
    while idx < route_len:
        raw_command = int(dedup_entries[idx].get("raw_command", 0))
        if raw_command not in (1, 2, 3):
            idx += 1
            continue

        episode_end = idx
        while episode_end + 1 < route_len and int(dedup_entries[episode_end + 1].get("raw_command", 0)) == raw_command:
            episode_end += 1

        search_end = min(route_len, episode_end + 10)
        event_idx = None
        for probe_idx in range(idx, search_end):
            if bool(dedup_entries[probe_idx].get("is_junction", False)):
                event_idx = int(probe_idx)
                break
        if event_idx is None:
            idx = episode_end + 1
            continue

        current_command = int(route_plan[event_idx].get("command", 0))
        if current_command == 0 or (current_command == 3 and raw_command in (1, 2)):
            route_plan[event_idx]["command"] = int(raw_command)
        idx = episode_end + 1

    return route_plan


def _is_trivial_straight_junction(
    junction_waypoint: Any,
    last_non_junction_wp: Any,
    angle_threshold_deg: float = 25.0,
) -> bool:
    """Return True if STRAIGHT at this junction is trivial (no real turn option).

    A trivial junction is one where the entry waypoint has no meaningfully
    different forward branches (e.g. highway overpass, merge ramp).  In that
    case STRAIGHT should be downgraded to LANE_FOLLOW so the CIL model does
    not switch output branches unnecessarily.
    """
    if not getattr(junction_waypoint, "is_junction", False):
        return False

    # Use the last non-junction waypoint as the junction entry point.
    entry_wp = last_non_junction_wp
    if entry_wp is None:
        # Fallback: walk backward from the junction waypoint to find entry.
        probe = junction_waypoint
        for _ in range(30):
            try:
                prev_wps = probe.previous(1.5)
            except Exception:
                prev_wps = []
            if not prev_wps:
                break
            probe = prev_wps[0]
            if not getattr(probe, "is_junction", False):
                entry_wp = probe
                break

    if entry_wp is None:
        return False  # Cannot determine; assume real junction.

    # Check how many distinct forward branches exist from the entry waypoint.
    try:
        next_wps = entry_wp.next(2.5)
    except Exception:
        return False

    if not next_wps:
        return False
    if len(next_wps) <= 1:
        return True  # Only one way forward; trivial pass-through.

    # Multiple successors exist.  Check if any have a heading that differs
    # significantly from the entry heading (i.e. a real turning option).
    base_yaw = float(entry_wp.transform.rotation.yaw)
    threshold = max(15.0, float(angle_threshold_deg))
    for nwp in next_wps:
        yaw_diff = abs(
            ((float(nwp.transform.rotation.yaw) - base_yaw) + 180.0) % 360.0 - 180.0
        )
        if yaw_diff > threshold:
            return False  # A real turning alternative exists; not trivial.

    return True  # All branches go roughly the same direction; trivial.


def build_reference_route_plan_from_trace(trace_items: Any, max_items: int = REFERENCE_ROUTE_MAX_ITEMS) -> list[Dict[str, Any]]:
    raw_entries: list[Dict[str, Any]] = []
    if not trace_items:
        return []

    last_non_junction_wp: Any = None

    for item in list(trace_items)[: max(1, int(max_items))]:
        waypoint = _planner_item_to_waypoint(item)
        road_option = item[1] if isinstance(item, (tuple, list)) and len(item) >= 2 else None
        if waypoint is None or not hasattr(waypoint, "transform"):
            continue

        is_junction = bool(getattr(waypoint, "is_junction", False))
        raw_command = int(map_road_option_to_command(road_option))

        if not is_junction:
            last_non_junction_wp = waypoint

        # Downgrade STRAIGHT at trivial pass-through junctions to LANE_FOLLOW.
        if raw_command == 3 and is_junction:
            if _is_trivial_straight_junction(waypoint, last_non_junction_wp):
                raw_command = 0

        raw_entries.append(
            {
                "location": waypoint.transform.location,
                "raw_command": raw_command,
                "is_junction": is_junction,
            }
        )

    return _finalize_reference_route_entries(raw_entries)[:max_items]


def build_global_reference_route(
    world_map: Any,
    start_location: Any,
    destination_location: Any,
    sampling_resolution: float = 1.0,
    max_items: int = REFERENCE_ROUTE_MAX_ITEMS,
) -> list[Dict[str, Any]]:
    if world_map is None or start_location is None or destination_location is None:
        return []
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore[import-not-found]
    except Exception:
        return []

    try:
        grp = GlobalRoutePlanner(world_map, float(sampling_resolution))
        traced_route = grp.trace_route(start_location, destination_location)
    except Exception:
        return []

    return build_reference_route_plan_from_trace(traced_route, max_items=max_items)


def snapshot_planner_route(planner: Any, max_items: int = REFERENCE_ROUTE_MAX_ITEMS) -> list[Dict[str, Any]]:
    if planner is None:
        return []

    raw_entries: list[Dict[str, Any]] = []
    last_non_junction_wp: Any = None

    def append_from(items: Any, limit: int) -> None:
        nonlocal last_non_junction_wp
        if not items:
            return
        for item in list(items)[: max(1, int(limit))]:
            waypoint = _planner_item_to_waypoint(item)
            road_option = item[1] if isinstance(item, (tuple, list)) and len(item) >= 2 else None
            if waypoint is None or not hasattr(waypoint, "transform"):
                continue

            is_junction = bool(getattr(waypoint, "is_junction", False))
            raw_command = int(map_road_option_to_command(road_option))

            if not is_junction:
                last_non_junction_wp = waypoint

            # Downgrade STRAIGHT at trivial pass-through junctions.
            if raw_command == 3 and is_junction:
                if _is_trivial_straight_junction(waypoint, last_non_junction_wp):
                    raw_command = 0

            raw_entries.append(
                {
                    "location": waypoint.transform.location,
                    "raw_command": raw_command,
                    "is_junction": is_junction,
                }
            )
            if len(raw_entries) >= max_items:
                break

    buffer_attr = getattr(planner, "_waypoint_buffer", None)
    append_from(buffer_attr, max(8, max_items // 4))
    if len(raw_entries) < max_items:
        queue_attr = getattr(planner, "_waypoints_queue", None)
        append_from(queue_attr, max_items - len(raw_entries))

    return _finalize_reference_route_entries(raw_entries)[:max_items]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def map_road_option_to_command(road_option: Any) -> int:
    """Map CARLA planner road options to CIL commands, ignoring lane changes."""
    if road_option is None:
        return 0

    raw_value = getattr(road_option, "value", road_option)
    try:
        numeric_value = int(raw_value)
        numeric_map = {
            1: 1,  # LEFT
            2: 2,  # RIGHT
            3: 3,  # STRAIGHT
            5: 0,  # CHANGELANELEFT -> ignored for intersection-only CIL
            6: 0,  # CHANGELANERIGHT -> ignored for intersection-only CIL
        }
        if numeric_value in numeric_map:
            return numeric_map[numeric_value]
    except (TypeError, ValueError):
        pass

    option_name = getattr(road_option, "name", str(road_option)).lower().replace(" ", "")
    if "change_lane" in option_name or "changelane" in option_name:
        return 0
    if "left" in option_name:
        return 1
    if "right" in option_name:
        return 2
    if "straight" in option_name:
        return 3
    return 0


class NavigationCommandOracle:
    """Shared intersection-only navigation command state machine for CIL."""

    def __init__(
        self,
        get_planner: Callable[[], Any],
        get_current_waypoint: Callable[[], Any],
        get_vehicle_location: Optional[Callable[[], Any]] = None,
        get_reference_route: Optional[Callable[[], Any]] = None,
        prep_time_s: float = DEFAULT_COMMAND_PREP_TIME_S,
        trigger_min_m: float = DEFAULT_COMMAND_TRIGGER_MIN_M,
        trigger_max_m: float = DEFAULT_COMMAND_TRIGGER_MAX_M,
        max_armed_frames: int = DEFAULT_COMMAND_MAX_ARMED_FRAMES,
    ) -> None:
        self._get_planner = get_planner
        self._get_current_waypoint = get_current_waypoint
        self._get_vehicle_location = get_vehicle_location
        self._get_reference_route = get_reference_route
        self._prep_time_s = float(prep_time_s)
        self._trigger_min_m = float(trigger_min_m)
        self._trigger_max_m = float(trigger_max_m)
        self._max_armed_frames = max(1, int(max_armed_frames))
        self._last_debug: Dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        self._active_navigation_command = 0
        self._active_command_source = "none"
        self._command_phase = "cruise"
        self._command_latch_frames = 0
        self._command_entered_junction = False
        self._armed_best_distance_to_junction_m = float("inf")
        self._armed_no_progress_frames = 0
        self._reference_route_progress_idx = 0
        self._last_debug = {}

    @property
    def last_debug(self) -> Dict[str, Any]:
        return dict(self._last_debug)

    @staticmethod
    def planner_queue_size(planner: Any) -> Optional[int]:
        if planner is None:
            return None
        for attr_name in ("_waypoints_queue", "_waypoint_buffer"):
            items = getattr(planner, attr_name, None)
            if items is None:
                continue
            try:
                return int(len(items))
            except Exception:
                continue
        return None

    @staticmethod
    def _normalize_angle_deg(angle_deg: float) -> float:
        wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
        return float(wrapped)

    def _safe_current_waypoint(self) -> Any:
        try:
            return self._get_current_waypoint()
        except Exception:
            return None

    def _safe_planner(self) -> Any:
        try:
            return self._get_planner()
        except Exception:
            return None

    def _safe_vehicle_location(self) -> Any:
        if self._get_vehicle_location is None:
            return None
        try:
            return self._get_vehicle_location()
        except Exception:
            return None

    def _safe_reference_route(self) -> list[Dict[str, Any]]:
        if self._get_reference_route is None:
            return []
        try:
            route = self._get_reference_route()
        except Exception:
            return []
        if not route:
            return []
        try:
            return list(route)
        except Exception:
            return []

    def _distance_from_route_start_m(self, route_start_location: Any) -> float:
        if route_start_location is None:
            return float("inf")
        vehicle_location = self._safe_vehicle_location()
        if vehicle_location is None:
            return float("inf")
        return math.hypot(
            float(vehicle_location.x - route_start_location.x),
            float(vehicle_location.y - route_start_location.y),
        )

    def _is_in_junction(self) -> bool:
        waypoint = self._safe_current_waypoint()
        return bool(waypoint is not None and getattr(waypoint, "is_junction", False))

    def _distance_to_next_junction_m(self, max_probe_m: float = 70.0, step_m: float = 1.5) -> float:
        waypoint = self._safe_current_waypoint()
        if waypoint is None:
            return float("inf")
        if getattr(waypoint, "is_junction", False):
            return 0.0

        travelled = 0.0
        probe_wp = waypoint
        step = max(0.5, float(step_m))
        max_steps = max(1, int(max_probe_m / step))

        for _ in range(max_steps):
            try:
                next_wps = probe_wp.next(step)
            except Exception:
                next_wps = []
            if not next_wps:
                break

            if len(next_wps) == 1:
                probe_wp = next_wps[0]
            else:
                base_yaw = float(probe_wp.transform.rotation.yaw)
                probe_wp = min(
                    next_wps,
                    key=lambda wp: abs(
                        self._normalize_angle_deg(float(wp.transform.rotation.yaw) - base_yaw)
                    ),
                )

            travelled += step
            if getattr(probe_wp, "is_junction", False):
                return float(travelled)

        return float("inf")

    def _extract_upcoming_turn_signal_from_reference_route(
        self,
        reference_route: list[Dict[str, Any]],
    ) -> tuple[int, float]:
        vehicle_location = self._safe_vehicle_location()
        if vehicle_location is None or not reference_route:
            return 0, float("inf")

        route_len = len(reference_route)
        start_idx = max(0, min(route_len - 1, int(self._reference_route_progress_idx)))
        probe_start = max(0, start_idx - 4)
        probe_end = min(route_len, max(start_idx + 120, 120))
        if probe_end <= probe_start:
            probe_start = 0
            probe_end = route_len

        nearest_idx = min(
            range(probe_start, probe_end),
            key=lambda idx: _xy_distance(reference_route[idx]["location"], vehicle_location),
        )
        self._reference_route_progress_idx = max(int(self._reference_route_progress_idx), int(nearest_idx))

        distance_m = 0.0
        prev_loc = vehicle_location
        for idx in range(int(self._reference_route_progress_idx), route_len):
            item = reference_route[idx]
            item_loc = item.get("location")
            if item_loc is None:
                continue
            distance_m += _xy_distance(prev_loc, item_loc)
            prev_loc = item_loc
            command = int(item.get("command", 0))
            if command != 0:
                return int(command), float(distance_m)
        return 0, float("inf")

    def _extract_upcoming_turn_signal(self, planner: Any) -> tuple[int, float, str]:
        reference_route = self._safe_reference_route()
        if reference_route:
            command, distance_m = self._extract_upcoming_turn_signal_from_reference_route(reference_route)
            return int(command), float(distance_m), "fixed_route"

        if planner is None:
            return 0, float("inf"), "none"

        distance_to_junction_m = self._distance_to_next_junction_m()
        if not math.isfinite(distance_to_junction_m):
            distance_to_junction_m = float("inf")

        for attr_name in ("target_road_option", "_target_road_option"):
            command = map_road_option_to_command(getattr(planner, attr_name, None))
            if command != 0:
                return command, float(distance_to_junction_m), "planner"

        for queue_name in ("_waypoint_buffer", "_waypoints_queue"):
            queue_attr = getattr(planner, queue_name, None)
            if not queue_attr:
                continue
            try:
                planner_items = list(queue_attr)
            except Exception:
                planner_items = []
            for item in planner_items[:96]:
                if not isinstance(item, (tuple, list)) or len(item) < 2:
                    continue
                command = map_road_option_to_command(item[1])
                if command != 0:
                    return command, float(distance_to_junction_m), "planner"
        return 0, float("inf"), "none"

    def _command_trigger_distance_m(self, speed_kmh: float) -> float:
        speed_mps = max(0.0, float(speed_kmh)) / 3.6
        return clamp(
            speed_mps * self._prep_time_s,
            self._trigger_min_m,
            self._trigger_max_m,
        )

    def _command_arm_distance_m(self, speed_kmh: float, upcoming_command: int) -> float:
        base_distance_m = self._command_trigger_distance_m(speed_kmh)
        command = int(upcoming_command)

        if command in (1, 2):
            late_turn_cap_m = clamp(
                4.5 + 0.14 * float(speed_kmh),
                7.5,
                min(self._trigger_max_m, COMMAND_TURN_ARM_MAX_M),
            )
            return min(base_distance_m, float(late_turn_cap_m))

        if command == 3:
            straight_cap_m = clamp(
                6.0 + 0.15 * float(speed_kmh),
                9.0,
                min(self._trigger_max_m, COMMAND_STRAIGHT_ARM_MAX_M),
            )
            return min(base_distance_m, float(straight_cap_m))

        return float(base_distance_m)

    def update(
        self,
        speed_kmh: float,
        route_start_location: Any = None,
    ) -> tuple[int, Dict[str, Any]]:
        planner = self._safe_planner()
        in_junction = self._is_in_junction()
        upcoming_command, distance_to_upcoming_turn_m, upcoming_source = self._extract_upcoming_turn_signal(planner)
        command_source = str(upcoming_source) if upcoming_command in (1, 2, 3) else "none"
        distance_to_junction_m = self._distance_to_next_junction_m()
        trigger_distance_m = self._command_trigger_distance_m(speed_kmh)
        arm_distance_m = self._command_arm_distance_m(speed_kmh, upcoming_command)
        if not math.isfinite(distance_to_upcoming_turn_m):
            distance_to_upcoming_turn_m = float("inf")
        command_distance_m = distance_to_upcoming_turn_m
        if (not math.isfinite(command_distance_m)) and math.isfinite(distance_to_junction_m):
            command_distance_m = float(distance_to_junction_m)

        distance_from_start_m = self._distance_from_route_start_m(route_start_location)
        reset_reason = "none"

        if self._active_navigation_command == 0:
            self._command_latch_frames = 0
            self._command_entered_junction = False
            should_arm = (
                upcoming_command in (1, 2, 3)
                and command_source in {"planner", "fixed_route"}
                and command_distance_m <= arm_distance_m
            )
            if should_arm:
                self._active_navigation_command = int(upcoming_command)
                self._active_command_source = str(command_source)
                self._command_entered_junction = bool(in_junction)
                self._command_latch_frames = 1
                self._command_phase = "in_junction" if in_junction else "armed"
                self._armed_best_distance_to_junction_m = float(distance_to_junction_m)
                self._armed_no_progress_frames = 0
            else:
                self._command_phase = "cruise"
        else:
            self._command_latch_frames += 1
            if in_junction:
                self._command_phase = "in_junction"
                self._command_entered_junction = True
                self._armed_best_distance_to_junction_m = 0.0
                self._armed_no_progress_frames = 0
            elif self._command_entered_junction:
                reset_reason = "left_junction"
                self.reset()
            else:
                self._command_phase = "armed"
                near_junction_hold_m = max(
                    6.0,
                    min(COMMAND_NEAR_JUNCTION_HOLD_M, arm_distance_m + 1.5),
                )
                junction_still_near = (
                    math.isfinite(distance_to_junction_m)
                    and distance_to_junction_m <= near_junction_hold_m
                )
                if math.isfinite(distance_to_junction_m):
                    if distance_to_junction_m + COMMAND_PROGRESS_EPS_M < self._armed_best_distance_to_junction_m:
                        self._armed_best_distance_to_junction_m = float(distance_to_junction_m)
                        self._armed_no_progress_frames = 0
                    elif junction_still_near:
                        self._armed_no_progress_frames = 0
                    else:
                        self._armed_no_progress_frames += 1
                else:
                    self._armed_no_progress_frames += 1
                if upcoming_command not in (1, 2, 3) and not junction_still_near:
                    reset_reason = "planner_turn_lost"
                    self.reset()
                elif self._armed_no_progress_frames >= self._max_armed_frames:
                    reset_reason = "armed_stalled"
                    self.reset()

        command_debug: Dict[str, Any] = {
            "phase": self._command_phase,
            "upcoming_command": int(upcoming_command),
            "active_command": int(self._active_navigation_command),
            "active_source": str(self._active_command_source),
            "upcoming_source": command_source,
            "reset_reason": str(reset_reason),
            "in_junction": bool(in_junction),
            "distance_to_turn_m": float(command_distance_m),
            "distance_to_junction_m": float(distance_to_junction_m),
            "trigger_distance_m": float(trigger_distance_m),
            "arm_distance_m": float(arm_distance_m),
            "latch_frames": int(self._command_latch_frames),
            "armed_no_progress_frames": int(self._armed_no_progress_frames),
            "best_armed_distance_to_junction_m": float(self._armed_best_distance_to_junction_m),
            "route_progress_idx": int(self._reference_route_progress_idx),
            "distance_from_start_m": float(distance_from_start_m),
        }
        self._last_debug = command_debug
        return int(self._active_navigation_command), command_debug
