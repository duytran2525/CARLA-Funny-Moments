import torch
import torch.nn as nn


class NvidiaCNN(nn.Module):
    def __init__(self):
        super(NvidiaCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),

            nn.Flatten()
        )

        self.dense_layers = nn.Sequential(
            nn.Linear(1152, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),

            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x.squeeze()


class NvidiaCNNV2(nn.Module):
    """NvidiaCNN with BatchNorm after each Conv2d and Dropout in dense layers."""

    def __init__(self):
        super(NvidiaCNNV2, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Flatten(),
        )

        self.dense_layers = nn.Sequential(
            nn.Linear(1152, 100),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x.squeeze()


class CIL_NvidiaCNN(nn.Module):
    """
    Phase 2 – Conditional Imitation Learning CNN.

    Architecture:
    * Shared visual backbone identical to NvidiaCNNV2 (Conv + BatchNorm + ELU).
    * Speed branch: small FC network that embeds the normalised speed scalar.
    * Conditional command heads: one fully-connected "expert" head per high-level
      command.  At inference time only the head matching the active command
      contributes to the output, replicating the standard CIL paper design.

    Inputs:
        image   – (B, 3, 66, 200) YUV tensor, normalised to [-1, 1]
        speed   – (B,) float tensor, normalised speed in [0, 1]
        command – (B,) long tensor, command index in {0, 1, 2, 3}
                  0 = Follow Lane, 1 = Turn Left, 2 = Turn Right, 3 = Go Straight

    Output:
        steering – (B,) float tensor with predicted steering angle in [-1, 1]
    """

    NUM_COMMANDS = 4
    _CONV_FLAT_DIM = 1152  # 64 filters × 1 H × 18 W after 5 conv layers on a 66×200 input
    _SPEED_EMB_DIM = 32

    def __init__(self):
        super().__init__()

        # --- Shared visual backbone (same as NvidiaCNNV2) ---
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Flatten(),
        )

        # --- Speed embedding branch ---
        self.speed_branch = nn.Sequential(
            nn.Linear(1, 64),
            nn.ELU(),
            nn.Linear(64, self._SPEED_EMB_DIM),
            nn.ELU(),
        )

        # --- Conditional command heads (one per command) ---
        head_input_dim = self._CONV_FLAT_DIM + self._SPEED_EMB_DIM
        self.command_heads = nn.ModuleList(
            [self._make_head(head_input_dim) for _ in range(self.NUM_COMMANDS)]
        )

    @staticmethod
    def _make_head(input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor, speed: torch.Tensor,
                command: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image:   (B, 3, 66, 200)
            speed:   (B,) – normalised speed scalar per sample
            command: (B,) – integer command index per sample
        Returns:
            steering: (B,) – predicted steering angle
        """
        # Visual features
        vis_feat = self.conv_layers(image)                        # (B, 1152)

        # Speed features
        spd_feat = self.speed_branch(speed.unsqueeze(1).float())  # (B, 32)

        # Concatenate
        features = torch.cat([vis_feat, spd_feat], dim=1)         # (B, 1184)

        # Apply all heads, then select the correct one per sample
        # head_outputs: (B, NUM_COMMANDS, 1)
        head_outputs = torch.stack(
            [head(features) for head in self.command_heads], dim=1
        )

        # Gather the output of the active command head for each sample
        cmd_idx = command.long().unsqueeze(1).unsqueeze(2)        # (B, 1, 1)
        steering = head_outputs.gather(1, cmd_idx).squeeze(1)     # (B, 1)

        return steering.squeeze(-1)                               # (B,)
