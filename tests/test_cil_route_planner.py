from __future__ import annotations

import unittest
from dataclasses import dataclass

from core_control.cil_route_planner import CILRoutePlanner


@dataclass
class FakeLocation:
    x: float
    y: float
    z: float = 0.0


class CILRoutePlannerTests(unittest.TestCase):
    def test_reference_route_anchor_searches_beyond_first_probe_window(self) -> None:
        planner = CILRoutePlanner()
        route_plan = [
            {"location": FakeLocation(float(idx), 0.0), "command": 0, "is_junction": False}
            for idx in range(320)
        ]

        route_locations = planner.collect_reference_route_locations(
            route_plan,
            max_points=40,
            anchor_location=FakeLocation(225.2, 0.0),
        )

        self.assertGreaterEqual(len(route_locations), 2)
        self.assertAlmostEqual(route_locations[0].x, 225.0, delta=1.0)

    def test_reference_route_can_return_full_s_to_d_polyline_for_visualization(self) -> None:
        planner = CILRoutePlanner()
        route_plan = [
            {"location": FakeLocation(float(idx), 0.0), "command": 0, "is_junction": False}
            for idx in range(400)
        ]

        route_locations = planner.collect_reference_route_locations(
            route_plan,
            max_points=4096,
            anchor_location=None,
        )

        self.assertEqual(len(route_locations), 400)
        self.assertEqual(route_locations[0].x, 0.0)
        self.assertEqual(route_locations[-1].x, 399.0)


if __name__ == "__main__":
    unittest.main()
