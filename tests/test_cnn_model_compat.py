from __future__ import annotations

import unittest

import numpy as np
import torch

from core_perception.cnn_model import (
    CIL_NvidiaCNN,
    ConditionalSteeringCNN,
    NvidiaCNN,
    NvidiaCNNV2,
    WaypointPredictor,
    classify_checkpoint_state_dict,
)
from run_agents import CILAgent


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


if __name__ == "__main__":
    unittest.main()
