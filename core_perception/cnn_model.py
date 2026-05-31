from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as _tv_models
except ImportError:  # pragma: no cover
    _tv_models = None


def unwrap_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    state_dict = checkpoint
    if isinstance(checkpoint, Mapping):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint does not contain a state_dict mapping.")

    cleaned = {}
    for key, value in state_dict.items():
        clean_key = str(key)
        if clean_key.startswith("module."):
            clean_key = clean_key.replace("module.", "", 1)
        cleaned[clean_key] = value
    return cleaned


def classify_checkpoint_state_dict(state_dict: Mapping[str, Any]) -> str:
    keys = {str(key) for key in state_dict.keys()}
    # New RegNet-based waypoint predictor
    if any(key.startswith("backbone_stage") for key in keys) or "waypoint_head.weight" in keys:
        return "waypoint"
    # Legacy custom-backbone waypoint predictor
    if "stem.conv.weight" in keys or "film.embedding.weight" in keys or "head.4.weight" in keys:
        return "waypoint_legacy"
    if any(key.startswith("speed_branch.") for key in keys) or any(key.startswith("command_heads.") for key in keys):
        return "conditional_steering"
    if any(".running_mean" in key for key in keys):
        return "steering_v2"
    if "conv_layers.0.weight" in keys and "dense_layers.0.weight" in keys:
        return "steering"
    return "unknown"


@dataclass(frozen=True)
class WaypointScaling:
    """Hard-coded scaling factors for waypoint denormalization (meters)."""

    x_scale: float = 50.0
    y_scale: float = 25.0
    sigma_scale: float = 10.0
    sigma_eps: float = 1e-4


class PhysicsAwareStem(nn.Module):
    """
    Stem nhận thức Vật lý: Tách biệt Không gian (Shared Weights) và Động lực học (Frame Difference).
    Input: [Batch, 9, H, W] (3 frame RGB)
    """
    def __init__(self, in_channels: int = 9, out_channels: int = 32) -> None:
        super().__init__()
        # Xử lý đặc trưng Không gian (Nhìn cảnh vật) - Shared cho cả 3 frame
        self.spatial_extractor = nn.Conv2d(
            in_channels=3, out_channels=16, 
            kernel_size=5, stride=2, padding=2
        )
        
        # Xử lý đặc trưng Động học (Nhìn chuyển động qua phép trừ ảnh)
        self.motion_extractor = nn.Conv2d(
            in_channels=3, out_channels=8, 
            kernel_size=5, stride=2, padding=2
        )
        
        # Fuse: 3 frame (3*16) + 2 motion diff (2*8) = 48 + 16 = 64 channels
        self.temporal_fuse = nn.Conv2d(64, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tách 3 frame (Mỗi frame 3 channels)
        f1, f2, f3 = x[:, :3], x[:, 3:6], x[:, 6:]
        
        # Tính Optical Flow Proxy (Vận tốc và Gia tốc)
        diff_12 = f2 - f1
        diff_23 = f3 - f2
        
        # Trích xuất đặc trưng (Shared weights)
        feat_f1 = self.spatial_extractor(f1)
        feat_f2 = self.spatial_extractor(f2)
        feat_f3 = self.spatial_extractor(f3)
        
        feat_d12 = self.motion_extractor(diff_12)
        feat_d23 = self.motion_extractor(diff_23)
        
        # Ghép nối và Fuse
        fused = torch.cat([feat_f1, feat_f2, feat_f3, feat_d12, feat_d23], dim=1)
        return self.act(self.bn(self.temporal_fuse(fused)))


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
            nn.Linear(emb_dim + 1, emb_dim),
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

    def forward(self, x: torch.Tensor, command: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(command)
        speed = speed.view(-1, 1)
        condition = torch.cat([emb, speed], dim=1)
        gamma_beta = self.mlp(condition)
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
    """Spatial-Temporal Waypoint Predictor with RegNetY-400MF + CBAM + FiLM + FPN.

    Architecture:
        Input [B, 9, 66, 200]  (3 temporal YUV frames)
          -> PhysicsAwareStem       -> [B, 32, 33, 100]  (temporal + motion features)
          -> Adapter                -> [B, 32, 33, 100]  (bridge to RegNet trunk)
          -> RegNetY-400MF stages   -> 48 / 104 / 208 / 440 channels
          -> FiLM(cmd, speed, 208) on stage 3
          -> CBAM(440) + FiLM(cmd, speed, 440) on stage 4
          -> FPN(104, 208, 440 -> 128) + sum fusion
          -> Shared FC features (128-d)
          -> Waypoint head: 15 values (10 coords + 5 sigma)
          -> Speed head: 1 value (normalized speed prediction)

    Output: [B, 16]  where [:10]=waypoints, [10:15]=sigma, [15:16]=speed_pred
    """

    # RegNetY-400MF stage output channels (from torchvision)
    _REGNET_CHANNELS: Tuple[int, int, int, int] = (48, 104, 208, 440)

    def __init__(
        self,
        scaling: WaypointScaling | None = None,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.scaling = scaling or WaypointScaling()

        # ── Temporal pre-processor (kept from original design) ──
        self.stem = PhysicsAwareStem(in_channels=9, out_channels=32)

        # ── Adapter: bridge stem output to RegNet trunk input ──
        # RegNet stem normally outputs 32ch; we skip it and feed our
        # 32ch temporal features directly into the trunk stages.
        self.adapter = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ── RegNetY-400MF backbone (pretrained, stem skipped) ──
        self._init_backbone(pretrained_backbone)

        c1, c2, c3, c4 = self._REGNET_CHANNELS  # 48, 104, 208, 440

        # ── Attention & multi-level conditioning ──
        self.cbam = CBAM(c4)
        self.film_s3 = FiLM(num_commands=4, emb_dim=64, channels=c3)
        self.film_s4 = FiLM(num_commands=4, emb_dim=128, channels=c4)

        # ── Multi-scale Feature Pyramid ──
        self.fpn = FPN(in_channels=(c2, c3, c4), out_channels=128)

        # ── Shared feature extractor ──
        self.shared_features = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 4)),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.3),
        )

        # ── Waypoint + Uncertainty head ──
        self.waypoint_head = nn.Linear(128, 15)

        # ── Speed prediction head (auxiliary, CILRS-style regularizer) ──
        self.speed_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._init_non_backbone_weights()

    def _init_backbone(self, pretrained: bool) -> None:
        """Load RegNetY-400MF trunk stages from torchvision."""
        if _tv_models is None:
            raise ImportError(
                "torchvision is required for WaypointPredictor. "
                "Install with: pip install torchvision"
            )
        weights = (
            _tv_models.RegNet_Y_400MF_Weights.IMAGENET1K_V2
            if pretrained
            else None
        )
        _regnet = _tv_models.regnet_y_400mf(weights=weights)
        # Extract trunk stages, skip stem (we use PhysicsAwareStem instead)
        # and skip fc classifier (we use our own heads).
        self.backbone_stage1 = _regnet.trunk_output.block1  # 32->48, stride 2
        self.backbone_stage2 = _regnet.trunk_output.block2  # 48->104, stride 2
        self.backbone_stage3 = _regnet.trunk_output.block3  # 104->208, stride 2
        self.backbone_stage4 = _regnet.trunk_output.block4  # 208->440, stride 2
        del _regnet

    def _init_non_backbone_weights(self) -> None:
        """Initialize only non-backbone weights; backbone keeps pretrained."""
        for parent in (
            self.stem, self.adapter, self.cbam, self.film_s3, self.film_s4,
            self.fpn, self.shared_features, self.waypoint_head, self.speed_head,
        ):
            for module in parent.modules():
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
        self.film_s3._init_identity()
        self.film_s4._init_identity()

    def get_backbone_params(self) -> list[nn.Parameter]:
        """Return backbone parameters (for differential learning rate)."""
        params: list[nn.Parameter] = []
        for stage in (
            self.backbone_stage1, self.backbone_stage2,
            self.backbone_stage3, self.backbone_stage4,
        ):
            params.extend(stage.parameters())
        return params

    def get_non_backbone_params(self) -> list[nn.Parameter]:
        """Return non-backbone parameters."""
        backbone_ids = {id(p) for p in self.get_backbone_params()}
        return [p for p in self.parameters() if id(p) not in backbone_ids]

    def forward(
        self, x: torch.Tensor, command: torch.Tensor, speed: torch.Tensor,
    ) -> torch.Tensor:
        # ── Temporal pre-processing ──
        p2 = self.stem(x)             # [B, 32, H/2, W/2]
        p2 = self.adapter(p2)         # [B, 32, H/2, W/2]

        # ── RegNet backbone stages ──
        s1 = self.backbone_stage1(p2)  # [B, 48,  ...]
        s2 = self.backbone_stage2(s1)  # [B, 104, ...]
        s3 = self.backbone_stage3(s2)  # [B, 208, ...]
        s3 = self.film_s3(s3, command, speed)
        s4 = self.backbone_stage4(s3)  # [B, 440, ...]

        # ── Attention + Conditioning on deepest features ──
        s4 = self.cbam(s4)
        s4 = self.film_s4(s4, command, speed)

        # ── FPN multi-scale fusion ──
        f3, f4, f5 = self.fpn(s2, s3, s4)
        fused = (
            f3
            + F.interpolate(f4, size=f3.shape[-2:], mode="nearest")
            + F.interpolate(f5, size=f3.shape[-2:], mode="nearest")
        )

        # ── Shared features ──
        features = self.shared_features(fused)  # [B, 128]

        # ── Waypoint + Uncertainty output ──
        wp_out = self.waypoint_head(features)    # [B, 15]
        coords = torch.tanh(wp_out[:, :10])
        sigma = F.softplus(wp_out[:, 10:15])

        coords_view = coords.view(-1, 5, 2)
        x_scaled = coords_view[..., 0] * self.scaling.x_scale
        y_scaled = coords_view[..., 1] * self.scaling.y_scale
        coords = torch.stack([x_scaled, y_scaled], dim=-1).view(-1, 10)

        sigma = sigma * self.scaling.sigma_scale + self.scaling.sigma_eps

        # ── Speed prediction (auxiliary) ──
        speed_pred = self.speed_head(features)   # [B, 1]

        return torch.cat([coords, sigma, speed_pred], dim=1)  # [B, 16]


class CIL_NvidiaCNN(WaypointPredictor):
    """Compatibility alias for the temporal waypoint model used by CIL."""


class NvidiaCNN(nn.Module):
    """Legacy single-frame steering model without batch normalization."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(inplace=True),
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(1152, 100),
            nn.ELU(inplace=True),
            nn.Linear(100, 50),
            nn.ELU(inplace=True),
            nn.Linear(50, 10),
            nn.ELU(inplace=True),
            nn.Linear(10, 1),
        )

    def forward(self, x: torch.Tensor, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        features = self.conv_layers(x)
        flattened = torch.flatten(features, start_dim=1)
        return self.dense_layers(flattened)


class NvidiaCNNV2(nn.Module):
    """Legacy single-frame steering model with batch normalization."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ELU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.ELU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(1152, 100),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(100, 50),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(50, 10),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(10, 1),
        )

    def forward(self, x: torch.Tensor, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        features = self.conv_layers(x)
        flattened = torch.flatten(features, start_dim=1)
        return self.dense_layers(flattened)


class ConditionalSteeringCNN(nn.Module):
    """Legacy command-conditioned steering model used by older checkpoints."""

    num_commands = 4

    def __init__(self) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ELU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.ELU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
        )
        self.speed_branch = nn.Sequential(
            nn.Linear(1, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 32),
            nn.ELU(inplace=True),
        )
        self.command_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1184, 256),
                    nn.ELU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(256, 64),
                    nn.ELU(inplace=True),
                    nn.Dropout(p=0.25),
                    nn.Linear(64, 1),
                )
                for _ in range(self.num_commands)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        speed: torch.Tensor | None = None,
        command: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = int(x.shape[0])
        if speed is None:
            speed = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        if command is None:
            command = torch.zeros(batch_size, device=x.device, dtype=torch.long)

        speed = speed.view(batch_size, 1).to(dtype=x.dtype, device=x.device)
        command = command.view(batch_size).to(dtype=torch.long, device=x.device).clamp_(0, self.num_commands - 1)

        features = self.conv_layers(x)
        flattened = torch.flatten(features, start_dim=1)
        speed_features = self.speed_branch(speed)
        fused = torch.cat([flattened, speed_features], dim=1)

        outputs = []
        for row_idx in range(batch_size):
            head = self.command_heads[int(command[row_idx].item())]
            outputs.append(head(fused[row_idx : row_idx + 1]))
        return torch.cat(outputs, dim=0)


__all__ = [
    "CIL_NvidiaCNN",
    "ConditionalSteeringCNN",
    "NvidiaCNN",
    "NvidiaCNNV2",
    "WaypointPredictor",
    "WaypointScaling",
    "classify_checkpoint_state_dict",
    "unwrap_state_dict",
]
