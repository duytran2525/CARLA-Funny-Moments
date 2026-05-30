#!/usr/bin/env python3
"""
Kaggle Training Script – WaypointPredictor from CSV + image files.
Phiên bản tối ưu cho RTX 6000 Pro + dữ liệu recovery (15.5%).
Lane penalty đối xứng (cả trái & phải), bỏ qua recovery.
ĐÃ SỬA: rec_mask.squeeze(1) để an toàn với mọi batch size.
"""

from __future__ import annotations

import gc
import os
import random
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION – cập nhật đường dẫn cho môi trường của bạn
# ============================================================================
REPO_URL = "https://github.com/duytran2525/CARLA-Funny-Moments.git"
WORKING_DIR = "/kaggle/working/CARLA-Funny-Moments"

CSV_PATH = "/kaggle/input/datasets/trasuaolong/data-carla-recovery/data_carlav2/driving_log_with_recovery.csv"
IMAGE_ROOT = "/kaggle/input/datasets/trasuaolong/data-carla-recovery/data_carlav2"

# ============================================================================
# 1. WORKSPACE SETUP
# ============================================================================
def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

def _to_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(value)

def prepare_workspace() -> Path:
    script_root = Path(__file__).resolve().parents[1]
    target_root = Path(WORKING_DIR)
    kaggle_detected = Path("/kaggle/input").is_dir()
    bootstrap_default = kaggle_detected and not _is_relative_to(script_root, target_root)
    bootstrap_enabled = _env_flag("KAGGLE_BOOTSTRAP", bootstrap_default)

    print("🚀 ĐANG KHỞI TẠO HỆ THỐNG HUẤN LUYỆN (CSV MODE)...")
    project_dir = script_root
    if bootstrap_enabled:
        if _is_relative_to(script_root, target_root):
            print("  Đang chạy trong WORKING_DIR hiện tại; bỏ qua clone lại repo.")
        else:
            if target_root.exists():
                shutil.rmtree(target_root)
            subprocess.run(["git", "clone", REPO_URL, str(target_root)], check=True)
            project_dir = target_root.resolve()

    os.chdir(project_dir)
    (project_dir / "models").mkdir(exist_ok=True)
    project_text = str(project_dir)
    if project_text not in sys.path:
        sys.path.insert(0, project_text)

    print("\n⚙️ ĐANG GHI ĐÈ CẤU HÌNH YAML...")
    yaml_path = project_dir / "configs" / "train_params.yaml"
    with yaml_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["data_root"] = "/kaggle/working" if Path("/kaggle/working").is_dir() else str(project_dir)
    for key in ("csv_path", "data_csv"):
        config.pop(key, None)
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f)
    print("✅ Đã cập nhật train_params.yaml!")
    return project_dir

def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False

# ============================================================================
# 2. DATASET CLASS
# ============================================================================
class WaypointCarlaDatasetCSV(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_root: str,
        transform=None,
        is_training: bool = True,
        train_ratio: float = 0.75,
        seed: int = 42,
        geometric_offset: float = 0.35,
        include_side_cameras: bool = True,
        min_speed_kmh: float = 1.0,
        min_wp5_x_m: float = 3.0,
        check_images: bool = False,
        recovery_jitter_std: float = 0.0,
    ):
        self.image_root = Path(image_root)
        self.transform = transform
        self.is_training = is_training
        self.recovery_jitter_std = recovery_jitter_std

        print("  📋 Đang tải CSV tổng...")
        df_full = pd.read_csv(csv_path)

        # Lọc stationary
        if {"speed", "wp_5_x"}.issubset(df_full.columns) and min_speed_kmh > 0:
            speed = pd.to_numeric(df_full["speed"], errors="coerce")
            wp5x = pd.to_numeric(df_full["wp_5_x"], errors="coerce")
            mask = (speed < min_speed_kmh) & (wp5x < min_wp5_x_m)
            before = len(df_full)
            df_full = df_full[~mask].reset_index(drop=True)
            print(f"  🚗 Stationary filter: {before} → {len(df_full)} rows")

        # Stratified split
        strat_col = df_full["recovery_flag"].fillna(0).astype(int)
        indices = np.arange(len(df_full))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=1.0 - train_ratio,
            stratify=strat_col,
            random_state=seed,
        )
        selected_idx = train_idx if is_training else val_idx
        selected_df = df_full.iloc[selected_idx].reset_index(drop=True)

        # Camera configs
        camera_configs = [("images_center", 0.0)]
        if is_training and include_side_cameras:
            camera_configs.extend([
                ("images_left", geometric_offset),
                ("images_right", -geometric_offset),
            ])

        print(f"  🔨 Đang xây dựng sample list ({'Train' if is_training else 'Val'})...")
        self.samples = []
        self.recovery_flags = []
        skipped_wp = 0
        skipped_img = 0

        for _, row in selected_df.iterrows():
            wp = self._extract_waypoints(row)
            if wp is None:
                skipped_wp += 1
                continue

            command = int(row.get("command", 0))
            recovery = float(row.get("recovery_flag", 0.0))
            speed_kmh = float(row.get("speed", 0.0))

            fn_t0   = self._normalize_filename(row.get("image_filename", ""))
            fn_tm03 = self._normalize_filename(row.get("image_filename_tm03", ""))
            fn_tm06 = self._normalize_filename(row.get("image_filename_tm06", ""))
            if not fn_t0 or not fn_tm03 or not fn_tm06:
                skipped_img += 1
                continue

            for cam_dir, offset in camera_configs:
                col_name = f"{cam_dir.split('_')[-1]}_camera"
                full_cam_path = row.get(col_name)
                if pd.isna(full_cam_path) or not isinstance(full_cam_path, str):
                    skipped_img += 1
                    continue
                base_dir = os.path.dirname(full_cam_path)
                if not base_dir:
                    skipped_img += 1
                    continue

                img_paths = [f"{base_dir}/{fn}" for fn in (fn_t0, fn_tm03, fn_tm06)]

                if check_images:
                    missing = any(not (self.image_root / p).exists() for p in img_paths)
                    if missing:
                        skipped_img += 1
                        continue

                wp_cam = wp.copy()
                wp_cam[:, 1] += offset

                self.samples.append({
                    "img_paths": img_paths,
                    "waypoints": wp_cam,
                    "command": command,
                    "recovery_flag": recovery,
                    "speed_kmh": speed_kmh,
                })
                self.recovery_flags.append(int(recovery))

        tag = "Train" if is_training else "Val"
        print(f"  ✅ {tag}: {len(self.samples)} samples")
        if skipped_wp > 0:
            print(f"     Bỏ qua {skipped_wp} rows (thiếu waypoints)")
        if skipped_img > 0:
            print(f"     Bỏ qua {skipped_img} camera-triplets (thiếu ảnh/đường dẫn)")

        if len(self.samples) == 0:
            raise RuntimeError(f"{tag} dataset rỗng! Kiểm tra CSV_PATH, IMAGE_ROOT và cột *_camera.")

    @staticmethod
    def _normalize_filename(value) -> str:
        text = str(value or "").strip()
        if not text or text.lower() == "nan":
            return ""
        if not text.endswith(".jpg"):
            stem = text.split(".")[0] if "." in text else text
            text = stem.zfill(8) + ".jpg"
        return text

    @staticmethod
    def _extract_waypoints(row) -> np.ndarray | None:
        keys = [f"wp_{i}_{c}" for i in range(1, 6) for c in ("x", "y")]
        try:
            values = [float(row[k]) for k in keys]
        except (KeyError, TypeError, ValueError):
            return None
        return np.array(values, dtype=np.float32).reshape(5, 2)

    def __len__(self):
        return len(self.samples)

    def get_recovery_flags(self):
        return list(self.recovery_flags)

    # Augmentations
    @staticmethod
    def _random_brightness(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + (random.random() - 0.5)), 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    @staticmethod
    def _random_shadow(img):
        h, w = img.shape[:2]
        pts = np.array([[random.randint(0, w), 0], [random.randint(0, w), h],
                        [random.randint(0, w), h], [random.randint(0, w), 0]], np.int32)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = np.zeros_like(hsv[:, :, 2])
        cv2.fillPoly(mask, [pts], 255)
        hsv[:, :, 2][mask == 255] = (hsv[:, :, 2][mask == 255] * 0.5).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    @staticmethod
    def _random_blur(img):
        k = random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)

    @staticmethod
    def _random_noise(img):
        sigma = random.uniform(5, 15)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def _random_contrast(img):
        factor = random.uniform(0.7, 1.3)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

    @staticmethod
    def _random_cutout(img):
        h, w = img.shape[:2]
        ch = int(h * random.uniform(0.1, 0.3))
        cw = int(w * random.uniform(0.1, 0.3))
        cy, cx = random.randint(0, h), random.randint(0, w)
        y1, y2 = max(0, cy - ch // 2), min(h, cy + ch // 2)
        x1, x2 = max(0, cx - cw // 2), min(w, cx + cw // 2)
        img[y1:y2, x1:x2, :] = 0
        return img

    def __getitem__(self, idx):
        sample = self.samples[idx]
        waypoints = sample["waypoints"].copy()
        command = int(sample["command"])
        recovery = float(sample["recovery_flag"])
        speed_norm = min(float(sample.get("speed_kmh", 0.0)), 120.0) / 120.0

        do_flip = self.is_training and random.random() > 0.5
        do_brightness = self.is_training and random.random() > 0.5
        do_shadow = self.is_training and random.random() > 0.5
        do_blur = self.is_training and random.random() > 0.5
        do_noise = self.is_training and random.random() > 0.7
        do_contrast = self.is_training and random.random() > 0.5
        do_cutout = self.is_training and random.random() > 0.5

        # Lateral jitter cho recovery (nhẹ)
        if self.is_training and recovery > 0.5 and self.recovery_jitter_std > 0:
            waypoints[:, 1] += np.random.normal(0, self.recovery_jitter_std, size=waypoints.shape[0]).astype(np.float32)

        images = []
        for rel_path in sample["img_paths"]:
            full_path = self.image_root / rel_path
            img = cv2.imread(str(full_path))
            if img is None:
                raise FileNotFoundError(f"Không đọc được ảnh: {full_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h = img.shape[0]
            img = img[int(h * 0.45):, :, :]

            if self.is_training:
                if do_brightness: img = self._random_brightness(img)
                if do_shadow:     img = self._random_shadow(img)
                if do_blur:       img = self._random_blur(img)
                if do_noise:      img = self._random_noise(img)
                if do_contrast:   img = self._random_contrast(img)
                if do_cutout:     img = self._random_cutout(img)
                if do_flip:       img = cv2.flip(img, 1)

            img = cv2.resize(img, (200, 66))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            images.append(img)

        stacked = np.concatenate(images, axis=-1)
        tensor = torch.from_numpy(stacked).permute(2, 0, 1).float() / 255.0

        if self.transform:
            tensor = self.transform(tensor)

        if do_flip:
            waypoints[:, 1] = -waypoints[:, 1]
            if command == 1: command = 2
            elif command == 2: command = 1

        return (
            tensor,
            torch.tensor(waypoints, dtype=torch.float32),
            torch.tensor(command, dtype=torch.long),
            torch.tensor(recovery, dtype=torch.float32),
            torch.tensor(speed_norm, dtype=torch.float32),
        )

# ============================================================================
# 3. TRAINING
# ============================================================================
def main():
    project_dir = prepare_workspace()
    print("\n🔥 BẮT ĐẦU HUẤN LUYỆN (CSV + ẢNH TRỰC TIẾP)...")

    from core_perception.cnn_model import WaypointPredictor

    with (project_dir / "configs" / "train_params.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    primary_device = torch.device("cuda:0") if num_gpus > 0 else torch.device("cpu")
    print(f"Device: {num_gpus} GPU(s)")

    transform = transforms.Compose([transforms.Normalize(mean=[0.5] * 9, std=[0.5] * 9)])

    train_ratio = float(config.get("train_split", 0.75))
    geo_offset = float(config.get("geometric_offset", 0.35))
    include_side = _to_bool(config.get("include_side_cameras_train"), True)
    recovery_jitter = float(config.get("recovery_jitter_std", 0.0))

    print(f"\n📦 Đang tạo dataset... (side_cameras={include_side})")
    train_dataset = WaypointCarlaDatasetCSV(
        csv_path=CSV_PATH,
        image_root=IMAGE_ROOT,
        transform=transform,
        is_training=True,
        train_ratio=train_ratio,
        seed=seed,
        geometric_offset=geo_offset,
        include_side_cameras=include_side,
        check_images=False,
        recovery_jitter_std=recovery_jitter,
    )
    val_dataset = WaypointCarlaDatasetCSV(
        csv_path=CSV_PATH,
        image_root=IMAGE_ROOT,
        transform=transform,
        is_training=False,
        train_ratio=train_ratio,
        seed=seed,
        geometric_offset=geo_offset,
        include_side_cameras=False,
        check_images=False,
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val  : {len(val_dataset)} samples")

    # ── DataLoader ──
    def collate_fn(batch):
        imgs, wps, cmds, recs, speeds = zip(*batch)
        return (
            torch.stack(imgs),
            torch.stack(wps).float(),
            torch.stack(cmds).long(),
            torch.tensor(recs, dtype=torch.float32),
            torch.tensor(speeds, dtype=torch.float32),
        )

    batch_size = int(config.get("batch_size", 256))
    rec_weight = float(config.get("recovery_weight", 3.0))

    command_weights = {0: 1.0, 1: 5.0, 2: 5.0, 3: 2.0}
    weights = []
    for sample in train_dataset.samples:
        w = command_weights.get(int(sample.get("command", 0)), 1.0)
        if int(float(sample.get("recovery_flag", 0.0))) == 1:
            w *= rec_weight
        weights.append(w)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    print(f"DataLoader: batch={batch_size}, workers=train4/val2, recovery_weight={rec_weight}")

    # ── Model ──
    model = WaypointPredictor().to(primary_device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer & Loss ──
    huber = nn.SmoothL1Loss(reduction="mean")
    base_lr = float(config["learning_rate"])
    backbone_lr_ratio = float(config.get("backbone_lr_ratio", 0.1))
    optimizer = optim.Adam([
        {"params": model.get_non_backbone_params(), "lr": base_lr},
        {"params": model.get_backbone_params(), "lr": base_lr * backbone_lr_ratio},
    ])

    lr_scheduler_type = str(config.get("lr_scheduler", "plateau")).lower()
    if lr_scheduler_type == "cosine":
        effective_epochs = int(config["epochs"]) - int(config.get("warmup_epochs", 1))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, effective_epochs), eta_min=1e-6)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=float(config.get("lr_factor", 0.5)),
            patience=int(config.get("lr_patience", 3)),
        )

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda" if use_amp else "cpu", enabled=use_amp)

    lambda_wp = float(config.get("loss_lambda_wp", 1.0))
    lambda_gnll = float(config.get("loss_lambda_gnll", 0.02))
    lambda_smoothness = float(config.get("loss_lambda_smoothness", 0.1))
    lambda_speed = float(config.get("loss_lambda_speed", 0.5))
    lambda_lane = float(config.get("loss_lambda_lane", 0.3))

    grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
    sigma_min = float(config.get("sigma_min", 0.01))
    sigma_max = float(config.get("sigma_max", 10.0))
    warmup_epochs = int(config.get("warmup_epochs", 3))
    backbone_freeze_epochs = int(config.get("backbone_freeze_epochs", 5))

    model_save = project_dir / "models" / "waypoint_predictor_csv.pth"
    model_save.parent.mkdir(exist_ok=True)

    epochs = int(config["epochs"])
    patience = int(config.get("early_stopping_patience", 8))
    best_val = float("inf")
    no_improve = 0
    start_epoch = 0

    # Resume
    resume_path = config.get("resume_checkpoint")
    if resume_path and os.path.isfile(resume_path):
        print(f"\n🔄 ĐANG KHÔI PHỤC TỪ CHECKPOINT: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=primary_device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_val = checkpoint.get("val_loss", float("inf"))
        print(f"✅ Tiếp tục từ Epoch {start_epoch + 1}")

    print(f"\n{'='*60}")
    print(f"TRAINING | lr={base_lr} | grad_clip={grad_clip_norm} | rec_weight={rec_weight}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs):
        # Backbone freeze
        if epoch < backbone_freeze_epochs:
            for p in model.get_backbone_params():
                p.requires_grad = False
            if epoch == 0:
                print(f"  🧊 Backbone FROZEN for epochs 1-{backbone_freeze_epochs}")
        elif epoch == backbone_freeze_epochs:
            for p in model.get_backbone_params():
                p.requires_grad = True
            print(f"  🔥 Backbone UNFROZEN from epoch {epoch + 1}")

        if epoch < warmup_epochs:
            warmup_lr = base_lr * (epoch + 1) / max(1, warmup_epochs)
            for idx, pg in enumerate(optimizer.param_groups):
                scale = backbone_lr_ratio if idx == 1 else 1.0
                pg["lr"] = warmup_lr * scale

        # ── TRAIN ──
        model.train()
        running_loss = 0.0
        nan_batches = 0
        for i, (imgs, wps, cmds, recs, speeds) in enumerate(train_loader):
            imgs = imgs.to(primary_device, non_blocking=True)
            cmds = cmds.to(primary_device, non_blocking=True)
            wps = wps.to(primary_device, non_blocking=True)
            recs = recs.to(primary_device, non_blocking=True)
            speeds = speeds.to(primary_device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda" if use_amp else "cpu", enabled=use_amp):
                out = model(imgs, cmds, speeds)
                pred_wp = out[:, :10].view(-1, 5, 2)
                tgt_wp = wps.view(-1, 5, 2)
                loss_wp = huber(pred_wp, tgt_wp)

                # Lane penalty ĐỐI XỨNG – bỏ qua recovery
                abs_lateral_y = torch.abs(pred_wp[..., 1])
                lane_penalty = torch.where(
                    abs_lateral_y > 3.0,
                    (abs_lateral_y - 3.0).pow(2) * 2.0,
                    torch.zeros_like(abs_lateral_y),
                )
                rec_mask = recs.view(-1, 1) > 0.5
                lane_penalty = lane_penalty * (~rec_mask).float()
                loss_lane = lane_penalty.mean()

                pred_sig = out[:, 10:15].view(-1, 5, 1).expand(-1, 5, 2)
                pred_var = pred_sig.pow(2).clamp(sigma_min**2, sigma_max**2)
                loss_gnll = 0.5 * ((tgt_wp - pred_wp)**2 / pred_var + torch.log(pred_var))

                vel = pred_wp[:, 1:] - pred_wp[:, :-1]
                accel = vel[:, 1:] - vel[:, :-1]
                smoothness_loss = accel.pow(2).mean()

                pred_speed = out[:, 15]
                loss_speed = nn.functional.mse_loss(pred_speed, speeds)

                loss = (lambda_wp * loss_wp
                        + lambda_gnll * loss_gnll.mean()
                        + lambda_smoothness * smoothness_loss
                        + lambda_speed * loss_speed
                        + lambda_lane * loss_lane)

            if not torch.isfinite(loss):
                nan_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())

            if i % 50 == 0:
                print(f"  [Train] Ep {epoch+1} Batch {i}/{len(train_loader)} loss={running_loss/(i+1):.4f}")

        train_loss = running_loss / max(1, len(train_loader) - nan_batches)

        # ── VAL ──
        model.eval()
        val_loss = 0.0
        val_mae_norm = 0.0
        val_mae_rec = 0.0
        val_n_norm = 0
        val_n_rec = 0
        with torch.no_grad():
            for imgs, wps, cmds, recs, speeds in val_loader:
                imgs = imgs.to(primary_device, non_blocking=True)
                cmds = cmds.to(primary_device, non_blocking=True)
                wps = wps.to(primary_device, non_blocking=True)
                recs = recs.to(primary_device, non_blocking=True)
                speeds = speeds.to(primary_device, non_blocking=True)

                with torch.amp.autocast("cuda" if use_amp else "cpu", enabled=use_amp):
                    out = model(imgs, cmds, speeds)
                    pred_wp = out[:, :10].view(-1, 5, 2)
                    tgt_wp = wps.view(-1, 5, 2)
                    loss_wp = huber(pred_wp, tgt_wp)

                    abs_lateral_y = torch.abs(pred_wp[..., 1])
                    lane_penalty = torch.where(
                        abs_lateral_y > 3.0,
                        (abs_lateral_y - 3.0).pow(2) * 2.0,
                        torch.zeros_like(abs_lateral_y),
                    )
                    rec_mask = recs.view(-1, 1) > 0.5
                    lane_penalty = lane_penalty * (~rec_mask).float()
                    loss_lane = lane_penalty.mean()

                    pred_sig = out[:, 10:15].view(-1, 5, 1).expand(-1, 5, 2)
                    pred_var = pred_sig.pow(2).clamp(sigma_min**2, sigma_max**2)
                    loss_g = 0.5 * ((tgt_wp - pred_wp)**2 / pred_var + torch.log(pred_var))

                    vel = pred_wp[:, 1:] - pred_wp[:, :-1]
                    accel = vel[:, 1:] - vel[:, :-1]
                    smoothness_loss = accel.pow(2).mean()

                    pred_speed = out[:, 15]
                    loss_speed = nn.functional.mse_loss(pred_speed, speeds)

                    loss = (lambda_wp * loss_wp
                            + lambda_gnll * loss_g.mean()
                            + lambda_smoothness * smoothness_loss
                            + lambda_speed * loss_speed
                            + lambda_lane * loss_lane)

                bs = imgs.size(0)
                val_loss += float(loss.item()) * bs

                mae_per = (pred_wp - tgt_wp).abs().mean(dim=(1, 2))   # (B,)
                rec_mask_sq = rec_mask.squeeze(1)   # FIX: an toàn với mọi batch size
                norm_mask   = ~rec_mask_sq

                val_mae_norm += mae_per[norm_mask].sum().item()
                val_n_norm   += norm_mask.sum().item()
                if rec_mask_sq.any():
                    val_mae_rec += mae_per[rec_mask_sq].sum().item()
                    val_n_rec   += rec_mask_sq.sum().item()

        val_loss /= max(1, val_n_norm + val_n_rec)
        val_mae_norm = val_mae_norm / max(1, val_n_norm)
        val_mae_rec = val_mae_rec / max(1, val_n_rec)

        print(f"Epoch [{epoch+1:2d}/{epochs}] Train={train_loss:.4f} Val={val_loss:.4f} "
              f"MAE_norm={val_mae_norm:.4f} MAE_rec={val_mae_rec:.4f} "
              f"(n={val_n_norm}/{val_n_rec})")

        if epoch >= warmup_epochs:
            if lr_scheduler_type == "cosine":
                scheduler.step()
            else:
                scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_mae_norm": val_mae_norm,
                "val_mae_rec": val_mae_rec,
            }, model_save)
            print(f"  💾 Saved best -> {model_save}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE  |  Best val loss: {best_val:.4f}")
    print(f"   Model: {model_save}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()