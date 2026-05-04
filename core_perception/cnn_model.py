from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class WaypointScaling:
    """Hard-coded scaling factors for waypoint denormalization (meters)."""

    x_scale: float = 30.0
    y_scale: float = 10.0
    sigma_scale: float = 10.0
    sigma_eps: float = 1e-4


class AggressiveStem(nn.Module):
    """Aggressive stem to downsample early and save VRAM."""

    def __init__(self, in_channels: int = 9, out_channels: int = 32) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class DepthwiseBlock(nn.Module):
    """Depthwise separable block with optional stride for downsampling."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.dw_act = nn.ELU(inplace=True)

        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.pw_act = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.dw_bn(x)
        x = self.dw_act(x)
        x = self.pw(x)
        x = self.pw_bn(x)
        return self.pw_act(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        return torch.sigmoid(attn)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_map, max_map], dim=1)
        return torch.sigmoid(self.conv(attn))


class CBAM(nn.Module):
    """CBAM attention block (Channel + Spatial)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel_attn = ChannelAttention(channels)
        self.spatial_attn = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) with identity initialization."""

    def __init__(self, num_commands: int = 4, emb_dim: int = 64, channels: int = 256) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_commands, embedding_dim=emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, channels * 2),
        )
        self.channels = channels
        self._init_identity()

    def _init_identity(self) -> None:
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        with torch.no_grad():
            self.mlp[-1].bias[: self.channels].fill_(1.0)
            self.mlp[-1].bias[self.channels :].fill_(0.0)

    def forward(self, x: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(command)
        gamma_beta = self.mlp(emb)
        gamma, beta = torch.split(gamma_beta, self.channels, dim=1)
        gamma = gamma.view(-1, self.channels, 1, 1)
        beta = beta.view(-1, self.channels, 1, 1)
        return x * gamma + beta


class FPN(nn.Module):
    """Lightweight FPN with unified channel width."""

    def __init__(self, in_channels: Tuple[int, int, int], out_channels: int = 128) -> None:
        super().__init__()
        c3, c4, c5 = in_channels
        self.lateral3 = nn.Conv2d(c3, out_channels, kernel_size=1, bias=False)
        self.lateral4 = nn.Conv2d(c4, out_channels, kernel_size=1, bias=False)
        self.lateral5 = nn.Conv2d(c5, out_channels, kernel_size=1, bias=False)

        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.smooth5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p5 = self.lateral5(p5)
        p4 = self.lateral4(p4) + F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p3 = self.lateral3(p3) + F.interpolate(p4, size=p3.shape[-2:], mode="nearest")

        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        return p3, p4, p5


class WaypointPredictor(nn.Module):
    """Spatial-Temporal Waypoint Predictor with CBAM + FiLM + FPN."""

    def __init__(self, scaling: WaypointScaling | None = None) -> None:
        super().__init__()
        self.scaling = scaling or WaypointScaling()
        self.stem = AggressiveStem(in_channels=9, out_channels=32)

        self.block1 = DepthwiseBlock(32, 64, stride=2)   # P3: [64, 17, 50]
        self.block2 = DepthwiseBlock(64, 128, stride=2)  # P4: [128, 9, 25]
        self.block3 = DepthwiseBlock(128, 256, stride=2) # P5: [256, 5, 13]

        self.cbam = CBAM(256)
        self.film = FiLM(num_commands=4, emb_dim=64, channels=256)

        self.fpn = FPN(in_channels=(64, 128, 256), out_channels=128)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 15),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        x = x.to(memory_format=torch.channels_last)
        p2 = self.stem(x)
        p3 = self.block1(p2)
        p4 = self.block2(p3)
        p5 = self.block3(p4)

        p5 = self.cbam(p5)
        p5 = self.film(p5, command)

        f3, f4, f5 = self.fpn(p3, p4, p5)
        fused = f3 + F.interpolate(f4, size=f3.shape[-2:], mode="nearest") + F.interpolate(
            f5, size=f3.shape[-2:], mode="nearest"
        )

        out = self.head(fused)
        coords = torch.tanh(out[:, :10])
        sigma = F.softplus(out[:, 10:])

        coords = coords.view(-1, 5, 2)
        # Tạo một tensor chứa hệ số scale nằm trên cùng device (CPU/GPU) với coords
        scale_factors = torch.tensor([self.scaling.x_scale, self.scaling.y_scale], device=coords.device)
        # Nhân out-of-place (PyTorch sẽ tạo ra một tensor mới, không ghi đè đồ thị cũ)
        coords = coords * scale_factors
        coords = coords.view(-1, 10)

        sigma = sigma * self.scaling.sigma_scale + self.scaling.sigma_eps
        return torch.cat([coords, sigma], dim=1)