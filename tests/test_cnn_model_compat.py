from __future__ import annotations

import unittest

import torch

from core_perception.cnn_model import (
    CIL_NvidiaCNN,
    ConditionalSteeringCNN,
    NvidiaCNN,
    NvidiaCNNV2,
    WaypointPredictor,
    classify_checkpoint_state_dict,
)


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
        command = torch.tensor([0, 2], dtype=torch.long)
        output = CIL_NvidiaCNN()(image, command)
        self.assertEqual(tuple(output.shape), (2, 15))


if __name__ == "__main__":
    unittest.main()
