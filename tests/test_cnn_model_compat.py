from __future__ import annotations

import unittest

import numpy as np
import torch

import run_agents
from core_perception.cnn_model import (
    CIL_NvidiaCNN,
    ConditionalSteeringCNN,
    NvidiaCNN,
    NvidiaCNNV2,
    WaypointPredictor,
    classify_checkpoint_state_dict,
)
from run_agents import AGENT_REGISTRY, CILAgent, CILYoloAgent


class CnnModelCompatibilityTests(unittest.TestCase):
    def test_checkpoint_classifier_detects_supported_architectures(self) -> None:
        self.assertEqual(classify_checkpoint_state_dict(NvidiaCNN().state_dict()), "steering")
        self.assertEqual(classify_checkpoint_state_dict(NvidiaCNNV2().state_dict()), "steering_v2")
        self.assertEqual(
            classify_checkpoint_state_dict(ConditionalSteeringCNN().state_dict()),
            "conditional_steering",
        )
        self.assertEqual(classify_checkpoint_state_dict(WaypointPredictor().state_dict()), "waypoint")

    def test_legacy_steering_models_forward(self) -> None:
        image = torch.zeros(2, 3, 66, 200)
        self.assertEqual(tuple(NvidiaCNN()(image).shape), (2, 1))
        self.assertEqual(tuple(NvidiaCNNV2()(image).shape), (2, 1))

    def test_conditional_steering_model_accepts_optional_inputs(self) -> None:
        image = torch.zeros(2, 3, 66, 200)
        speed = torch.tensor([0.2, 0.7], dtype=torch.float32)
        command = torch.tensor([0, 3], dtype=torch.long)
        model = ConditionalSteeringCNN()
        self.assertEqual(tuple(model(image).shape), (2, 1))
        self.assertEqual(tuple(model(image, speed, command).shape), (2, 1))

    def test_waypoint_model_forward_shape(self) -> None:
        image = torch.zeros(2, 9, 66, 200)
        speed = torch.tensor([0.5, 0.6], dtype=torch.float32)
        command = torch.tensor([0, 2], dtype=torch.long)
        output = CIL_NvidiaCNN()(image, command, speed)
        self.assertEqual(tuple(output.shape), (2, 16))

    def test_cil_agent_waypoint_inference_passes_speed_and_reads_sigma_slice(self) -> None:
        class StubWaypointModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.seen_command: torch.Tensor | None = None
                self.seen_speed: torch.Tensor | None = None
                self.seen_image_shape: tuple[int, ...] | None = None

            def forward(
                self,
                image: torch.Tensor,
                command: torch.Tensor,
                speed: torch.Tensor,
            ) -> torch.Tensor:
                self.seen_command = command.detach().cpu()
                self.seen_speed = speed.detach().cpu()
                self.seen_image_shape = tuple(image.shape)
                return torch.arange(16, dtype=torch.float32).view(1, 16)

        model = StubWaypointModel()
        agent = object.__new__(CILAgent)
        agent._device = torch.device("cpu")
        agent._model = model

        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        waypoints, mean_uncertainty = agent._predict_cil_waypoints(
            [frame, frame, frame],
            speed_kmh=60.0,
            command=2,
        )

        self.assertEqual(tuple(waypoints.shape), (5, 2))
        self.assertEqual(model.seen_image_shape, (1, 9, 66, 200))
        self.assertEqual(int(model.seen_command.item()), 2)
        self.assertAlmostEqual(float(model.seen_speed.item()), 0.5)
        self.assertAlmostEqual(mean_uncertainty, 12.0)

    def test_cil_lane_constraint_falls_back_from_oncoming_lane(self) -> None:
        class FakeLocation:
            def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
                self.x = x
                self.y = y
                self.z = z

        class FakeRotation:
            yaw = 0.0

        class FakeTransform:
            def __init__(self, location: FakeLocation | None = None) -> None:
                self.location = location or FakeLocation()
                self.rotation = FakeRotation()

        class FakeWaypoint:
            def __init__(self, lane_id: int, location: FakeLocation) -> None:
                self.lane_id = lane_id
                self.transform = FakeTransform(location)

            def next(self, distance: float):
                return [FakeWaypoint(self.lane_id, FakeLocation(x=distance, y=0.0, z=0.0))]

        class FakeMap:
            def __init__(self) -> None:
                self.ego_wp = FakeWaypoint(1, FakeLocation())

            def get_waypoint(self, location: FakeLocation, **_kwargs):
                if abs(location.x) < 1e-6 and abs(location.y) < 1e-6:
                    return self.ego_wp
                lane_id = -1 if location.y > 0.0 else 1
                return FakeWaypoint(lane_id, FakeLocation(x=location.x, y=location.y, z=location.z))

        class FakeWorld:
            def __init__(self) -> None:
                self.map = FakeMap()

            def get_map(self) -> FakeMap:
                return self.map

        class FakeVehicle:
            def get_transform(self) -> FakeTransform:
                return FakeTransform(FakeLocation())

        class FakeLaneType:
            Driving = object()

        class FakeCarla:
            Location = FakeLocation
            LaneType = FakeLaneType

        old_carla = run_agents.carla
        try:
            run_agents.carla = FakeCarla
            agent = object.__new__(CILAgent)
            agent.session = type("Session", (), {"world": FakeWorld()})()

            constrained = agent._constrain_waypoints_to_lane(
                np.array([[10.0, 5.0]], dtype=np.float32),
                FakeVehicle(),
                command=0,
            )
        finally:
            run_agents.carla = old_carla

        self.assertEqual(tuple(constrained.shape), (1, 2))
        self.assertGreaterEqual(float(constrained[0, 0]), 3.0)
        self.assertAlmostEqual(float(constrained[0, 1]), 0.0)

    def test_cil_yolo_agent_is_registered_separately(self) -> None:
        self.assertIs(AGENT_REGISTRY["cil_yolo"], CILYoloAgent)
        self.assertIs(AGENT_REGISTRY["cil"], CILAgent)


if __name__ == "__main__":
    unittest.main()
