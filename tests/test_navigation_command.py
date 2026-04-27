from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from core_control.navigation_command import (
    NavigationCommandOracle,
    build_reference_route_plan_from_trace,
    map_road_option_to_command,
)


@dataclass
class FakeLocation:
    x: float
    y: float
    z: float = 0.0


class FakeWaypoint:
    def __init__(self, x: float, y: float, yaw: float = 0.0, is_junction: bool = False) -> None:
        self.transform = SimpleNamespace(
            location=FakeLocation(x=x, y=y, z=0.0),
            rotation=SimpleNamespace(yaw=yaw),
        )
        self.is_junction = bool(is_junction)
        self._next_waypoints = []

    def next(self, step: float):
        del step
        return list(self._next_waypoints)


class FakeRoadOption:
    def __init__(self, value=None, name: str = "") -> None:
        self.value = value
        self.name = name


class FakePlanner:
    def __init__(self, target_road_option=None, waypoint_queue=None) -> None:
        self.target_road_option = target_road_option
        self._target_road_option = target_road_option
        self._waypoints_queue = list(waypoint_queue or [])
        self._waypoint_buffer = []


def build_waypoint_chain(length: int, step_m: float = 1.5, junction_idx: int | None = None):
    waypoints = []
    for idx in range(length):
        is_junction = junction_idx is not None and idx >= int(junction_idx)
        waypoints.append(FakeWaypoint(x=float(idx) * float(step_m), y=0.0, is_junction=is_junction))
    for idx in range(length - 1):
        waypoints[idx]._next_waypoints = [waypoints[idx + 1]]
    return waypoints


class NavigationCommandOracleTests(unittest.TestCase):
    def test_map_road_option_ignores_lane_changes(self) -> None:
        self.assertEqual(map_road_option_to_command(FakeRoadOption(value=5, name="CHANGELANELEFT")), 0)
        self.assertEqual(map_road_option_to_command(FakeRoadOption(value=6, name="CHANGELANERIGHT")), 0)
        self.assertEqual(map_road_option_to_command(FakeRoadOption(value=1, name="LEFT")), 1)
        self.assertEqual(map_road_option_to_command(FakeRoadOption(value=3, name="STRAIGHT")), 3)

    def test_oracle_arms_in_junction_and_resets_after_exit(self) -> None:
        wp_before = FakeWaypoint(x=0.0, y=0.0, is_junction=False)
        wp_junction = FakeWaypoint(x=1.5, y=0.0, is_junction=True)
        wp_after = FakeWaypoint(x=3.0, y=0.0, is_junction=False)
        wp_before._next_waypoints = [wp_junction]
        wp_junction._next_waypoints = [wp_after]

        state = {"waypoint": wp_before}
        planner = FakePlanner(target_road_option=FakeRoadOption(value=1, name="LEFT"))

        oracle = NavigationCommandOracle(
            get_planner=lambda: planner,
            get_current_waypoint=lambda: state["waypoint"],
            get_vehicle_location=lambda: state["waypoint"].transform.location,
            prep_time_s=1.8,
            trigger_min_m=9.0,
            trigger_max_m=24.0,
            max_armed_frames=160,
        )

        command, debug = oracle.update(speed_kmh=30.0)
        self.assertEqual(command, 1)
        self.assertEqual(debug["phase"], "armed")

        state["waypoint"] = wp_junction
        command, debug = oracle.update(speed_kmh=30.0)
        self.assertEqual(command, 1)
        self.assertEqual(debug["phase"], "in_junction")

        state["waypoint"] = wp_after
        planner.target_road_option = None
        planner._target_road_option = None
        command, debug = oracle.update(speed_kmh=30.0)
        self.assertEqual(command, 0)
        self.assertEqual(debug["phase"], "cruise")
        self.assertEqual(debug["reset_reason"], "left_junction")

    def test_oracle_skips_lane_change_queue_entries(self) -> None:
        wp_before = FakeWaypoint(x=0.0, y=0.0, is_junction=False)
        wp_junction = FakeWaypoint(x=1.5, y=0.0, is_junction=True)
        wp_before._next_waypoints = [wp_junction]

        planner = FakePlanner(
            target_road_option=None,
            waypoint_queue=[
                (None, FakeRoadOption(value=5, name="CHANGELANELEFT")),
                (None, FakeRoadOption(value=2, name="RIGHT")),
            ],
        )

        oracle = NavigationCommandOracle(
            get_planner=lambda: planner,
            get_current_waypoint=lambda: wp_before,
            get_vehicle_location=lambda: wp_before.transform.location,
        )

        command, debug = oracle.update(speed_kmh=25.0)
        self.assertEqual(command, 2)
        self.assertEqual(debug["upcoming_command"], 2)

    def test_oracle_delays_right_turn_arm_until_near_junction(self) -> None:
        chain = build_waypoint_chain(length=18, step_m=1.5, junction_idx=15)
        state = {"waypoint": chain[0]}
        planner = FakePlanner(target_road_option=FakeRoadOption(value=2, name="RIGHT"))

        oracle = NavigationCommandOracle(
            get_planner=lambda: planner,
            get_current_waypoint=lambda: state["waypoint"],
            get_vehicle_location=lambda: state["waypoint"].transform.location,
            prep_time_s=1.8,
            trigger_min_m=9.0,
            trigger_max_m=24.0,
        )

        command, debug = oracle.update(speed_kmh=47.0)
        self.assertEqual(command, 0)
        self.assertEqual(debug["phase"], "cruise")
        self.assertGreater(debug["distance_to_junction_m"], debug["arm_distance_m"])

        state["waypoint"] = chain[10]
        command, debug = oracle.update(speed_kmh=28.0)
        self.assertEqual(command, 2)
        self.assertEqual(debug["phase"], "armed")
        self.assertLessEqual(debug["distance_to_junction_m"], debug["arm_distance_m"])

    def test_oracle_keeps_command_armed_when_holding_near_junction(self) -> None:
        chain = build_waypoint_chain(length=8, step_m=1.5, junction_idx=5)
        state = {"waypoint": chain[0]}
        planner = FakePlanner(target_road_option=FakeRoadOption(value=2, name="RIGHT"))

        oracle = NavigationCommandOracle(
            get_planner=lambda: planner,
            get_current_waypoint=lambda: state["waypoint"],
            get_vehicle_location=lambda: state["waypoint"].transform.location,
            prep_time_s=1.8,
            trigger_min_m=9.0,
            trigger_max_m=24.0,
            max_armed_frames=3,
        )

        command, debug = oracle.update(speed_kmh=20.0)
        self.assertEqual(command, 2)
        self.assertEqual(debug["phase"], "armed")

        for _ in range(6):
            command, debug = oracle.update(speed_kmh=0.2)

        self.assertEqual(command, 2)
        self.assertEqual(debug["phase"], "armed")
        self.assertEqual(debug["reset_reason"], "none")
        self.assertEqual(debug["armed_no_progress_frames"], 0)

    def test_oracle_prefers_fixed_reference_route_over_dynamic_planner_queue(self) -> None:
        chain = build_waypoint_chain(length=8, step_m=1.5, junction_idx=5)
        state = {"waypoint": chain[0]}
        planner = FakePlanner(target_road_option=FakeRoadOption(value=2, name="RIGHT"))
        fixed_route = [
            {
                "location": waypoint.transform.location,
                "command": 1 if idx >= 5 else 0,
                "is_junction": waypoint.is_junction,
            }
            for idx, waypoint in enumerate(chain)
        ]

        oracle = NavigationCommandOracle(
            get_planner=lambda: planner,
            get_current_waypoint=lambda: state["waypoint"],
            get_vehicle_location=lambda: state["waypoint"].transform.location,
            get_reference_route=lambda: fixed_route,
            prep_time_s=1.8,
            trigger_min_m=9.0,
            trigger_max_m=24.0,
        )

        command, debug = oracle.update(speed_kmh=20.0)
        self.assertEqual(command, 1)
        self.assertEqual(debug["upcoming_command"], 1)
        self.assertEqual(debug["upcoming_source"], "fixed_route")
        self.assertEqual(debug["active_source"], "fixed_route")

    def test_reference_route_builder_drops_non_junction_command_episode(self) -> None:
        wp0 = FakeWaypoint(x=0.0, y=0.0, is_junction=False)
        wp1 = FakeWaypoint(x=1.5, y=0.0, is_junction=True)
        wp2 = FakeWaypoint(x=3.0, y=0.0, is_junction=False)
        wp3 = FakeWaypoint(x=4.5, y=0.0, is_junction=False)
        wp4 = FakeWaypoint(x=6.0, y=0.0, is_junction=False)
        trace_items = [
            (wp0, FakeRoadOption(value=3, name="STRAIGHT")),
            (wp1, FakeRoadOption(value=3, name="STRAIGHT")),
            (wp2, FakeRoadOption(value=None, name="LANEFOLLOW")),
            (wp3, FakeRoadOption(value=2, name="RIGHT")),
            (wp4, FakeRoadOption(value=2, name="RIGHT")),
        ]

        route_plan = build_reference_route_plan_from_trace(trace_items)
        non_zero_commands = [
            (idx, int(item["command"]))
            for idx, item in enumerate(route_plan)
            if int(item.get("command", 0)) != 0
        ]

        self.assertEqual(non_zero_commands, [(1, 3)])


if __name__ == "__main__":
    unittest.main()
