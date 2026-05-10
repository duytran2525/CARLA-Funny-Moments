#!/usr/bin/env python3
"""
Kaggle Training Script – WaypointPredictor with H5 image blobs + CSV labels.

H5 structure (individual JPEG blobs, NOT arrays):
    Town01/images_center/00000001.jpg  [scalar void dataset, raw JPEG bytes]
    Town01/images_left/00000001.jpg
    Town01/images_right/00000001.jpg
    Town02/...

Labels loaded from driving_log.csv (one per town on the filesystem).

Usage (Kaggle notebook cell):
    1. Set H5_PATH, CSV_ROOT below
    2. Run cell
"""

from __future__ import annotations

import gc
import os
import random
import shutil
import sys
import warnings
from pathlib import Path

import cv2

# CRITICAL: Disable HDF5 file locking BEFORE importing h5py.
# Without this, multiple DataLoader workers deadlock on the same H5 file.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION — update these paths for your Kaggle environment
# ============================================================================
REPO_URL = "https://github.com/duytran2525/CARLA-Funny-Moments.git"
WORKING_DIR = "/kaggle/working/CARLA-Funny-Moments"

# Path to the H5 file containing JPEG image blobs
H5_PATH = "/kaggle/input/datasets/yudtrann/dataset-carlav3/carla_images_drive.h5"

# Directory that contains Town*/driving_log.csv
# Từ screenshot: datasetcarlav2/data_carlav2/Town01/driving_log.csv
# Set None để auto-detect, hoặc set đường dẫn chính xác:
CSV_ROOT: str | None = "/kaggle/input/datasets/yudtrann/datasetcarlav2/data_carlav2"

# ============================================================================
# 1. CLONE REPO & SETUP
# ============================================================================
print("🚀 ĐANG KHỞI TẠO HỆ THỐNG HUẤN LUYỆN (H5 MODE)...")

if os.path.exists(WORKING_DIR):
    shutil.rmtree(WORKING_DIR)
os.system(f"git clone {REPO_URL} {WORKING_DIR}")
os.chdir(WORKING_DIR)
os.makedirs("models", exist_ok=True)

sys.path.insert(0, WORKING_DIR)

print("\n⚙️ ĐANG GHI ĐÈ CẤU HÌNH YAML...")
yaml_path = os.path.join(WORKING_DIR, "configs", "train_params.yaml")
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
config["data_root"] = "/kaggle/working"
for key in ("csv_path", "data_csv"):
    config.pop(key, None)
with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(config, f)
print("✅ Đã cập nhật train_params.yaml!")


# ============================================================================
# 2. AUTO-DETECT CSV ROOT
# ============================================================================
def find_csv_root(h5_path: str, csv_root_hint: str | None) -> Path:
    """Search common locations for Town*/driving_log.csv."""
    candidates: list[Path] = []
    if csv_root_hint:
        candidates.append(Path(csv_root_hint))
    h5_dir = Path(h5_path).parent
    candidates.extend([
        h5_dir,
        h5_dir.parent,
        h5_dir / "data_carlav2",
    ])
    # Kaggle input: search up to 4 levels deep to handle
    # /kaggle/input/datasets/owner/name/subdir pattern
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for depth_pattern in ["*", "*/*", "*/*/*", "*/*/*/*"]:
            for d in sorted(kaggle_input.glob(depth_pattern)):
                if d.is_dir():
                    candidates.append(d)

    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen or not c.is_dir():
            continue
        seen.add(key)
        town_csvs = list(c.glob("Town*/driving_log.csv"))
        if town_csvs:
            print(f"  ✅ CSV_ROOT found: {c}  ({len(town_csvs)} town CSVs)")
            return c

    raise FileNotFoundError(
        "Không tìm thấy Town*/driving_log.csv. "
        "Hãy set CSV_ROOT = đường dẫn tới thư mục chứa Town01/, Town02/,... "
        "với driving_log.csv trong mỗi town."
    )


# ============================================================================
# 3. DATASET CLASS
# ============================================================================
class WaypointCarlaDatasetH5(Dataset):
    """
    Reads images from H5 (per-image JPEG blobs) + labels from CSV files.
    Uses temporal stacking (t0 + t-0.3s + t-0.6s) matching dataset.py behavior.
    """

    def __init__(
        self,
        h5_path: str,
        csv_root: Path,
        towns: list[str],
        transform=None,
        is_training: bool = True,
        train_ratio: float = 0.75,
        seed: int = 42,
        geometric_offset: float = 0.35,
        include_side_cameras: bool = True,
        min_speed_kmh: float = 1.0,
        min_wp5_x_m: float = 3.0,
    ):
        self.h5_path = str(h5_path)
        self.transform = transform
        self.is_training = is_training
        self._h5_handle: h5py.File | None = None  # lazy per-worker cache

        # ── Build H5 key lookup (set of filenames per town/camera) ──
        print("  📂 Đang quét cấu trúc H5...")
        self._h5_keys: dict[str, set[str]] = {}
        with h5py.File(self.h5_path, "r") as f:
            for town in towns:
                for cam in ("images_center", "images_left", "images_right"):
                    grp_path = f"{town}/{cam}"
                    if grp_path in f and isinstance(f[grp_path], h5py.Group):
                        self._h5_keys[grp_path] = set(f[grp_path].keys())
        total_blobs = sum(len(v) for v in self._h5_keys.values())
        print(f"    H5 chứa {total_blobs} JPEG blobs từ {len(towns)} towns")

        # ── Load & merge CSVs ──
        print("  📋 Đang tải driving_log.csv...")
        frames: list[pd.DataFrame] = []
        for town in towns:
            csv_path = csv_root / town / "driving_log.csv"
            if not csv_path.exists():
                print(f"    ⚠️ Bỏ qua {town}: không tìm thấy CSV")
                continue
            df = pd.read_csv(csv_path)
            df["_town"] = town
            frames.append(df)
            print(f"    {town}: {len(df)} rows")

        if not frames:
            raise FileNotFoundError("Không tải được CSV nào!")
        merged = pd.concat(frames, ignore_index=True)

        # ── Filter stationary rows ──
        if {"speed", "wp_5_x"}.issubset(merged.columns) and min_speed_kmh > 0:
            speed = pd.to_numeric(merged["speed"], errors="coerce")
            wp5x = pd.to_numeric(merged["wp_5_x"], errors="coerce")
            mask = (speed < min_speed_kmh) & (wp5x < min_wp5_x_m)
            before = len(merged)
            merged = merged[~mask].reset_index(drop=True)
            print(f"  🚗 Stationary filter: {before} → {len(merged)} rows")

        # ── Train/Val split (at row level, before camera expansion) ──
        rng = np.random.default_rng(seed)
        n = len(merged)
        perm = rng.permutation(n)
        split = int(round(train_ratio * n))
        if is_training:
            selected_indices = perm[:split]
        else:
            selected_indices = perm[split:]
        selected_df = merged.iloc[selected_indices].reset_index(drop=True)

        # ── Camera configs ──
        camera_configs = [("images_center", 0.0)]
        if is_training and include_side_cameras:
            camera_configs.extend([
                ("images_left", geometric_offset),
                ("images_right", -geometric_offset),
            ])

        # ── Build sample list ──
        print(f"  🔨 Đang xây dựng sample list ({'Train' if is_training else 'Val'})...")
        self.samples: list[dict] = []
        self.recovery_flags: list[int] = []
        skipped_wp = 0
        skipped_img = 0

        for _, row in selected_df.iterrows():
            # Extract waypoints
            wp = self._extract_waypoints(row)
            if wp is None:
                skipped_wp += 1
                continue

            town = str(row["_town"])

            # Get temporal filenames
            fn_t0 = self._normalize_filename(row.get("image_filename", ""))
            fn_tm03 = self._normalize_filename(row.get("image_filename_tm03", ""))
            fn_tm06 = self._normalize_filename(row.get("image_filename_tm06", ""))
            if not fn_t0 or not fn_tm03 or not fn_tm06:
                skipped_img += 1
                continue

            command = int(row.get("command", 0))
            recovery = float(row.get("recovery_flag", 0.0))

            for cam_dir, offset in camera_configs:
                # Verify all 3 temporal images exist in this camera
                key = f"{town}/{cam_dir}"
                available = self._h5_keys.get(key, set())
                if not (fn_t0 in available and fn_tm03 in available and fn_tm06 in available):
                    skipped_img += 1
                    continue

                wp_cam = wp.copy()
                wp_cam[:, 1] += offset

                self.samples.append({
                    "town": town,
                    "cam_dir": cam_dir,
                    "fn_t0": fn_t0,
                    "fn_tm03": fn_tm03,
                    "fn_tm06": fn_tm06,
                    "waypoints": wp_cam,
                    "command": command,
                    "recovery_flag": recovery,
                })
                self.recovery_flags.append(int(recovery))

        tag = "Train" if is_training else "Val"
        print(f"  ✅ {tag}: {len(self.samples)} samples")
        if skipped_wp > 0:
            print(f"     Bỏ qua {skipped_wp} rows (thiếu waypoints)")
        if skipped_img > 0:
            print(f"     Bỏ qua {skipped_img} camera-triplets (thiếu ảnh trong H5)")

        if len(self.samples) == 0:
            raise RuntimeError(
                f"{tag} dataset rỗng! Kiểm tra:\n"
                "  - CSV_ROOT chứa Town*/driving_log.csv\n"
                "  - H5 chứa Town*/images_center/*.jpg\n"
                "  - Tên ảnh trong CSV khớp với tên trong H5"
            )

    @staticmethod
    def _normalize_filename(value) -> str:
        text = str(value or "").strip()
        if not text or text.lower() == "nan":
            return ""
        # Ensure .jpg extension
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

    def __len__(self) -> int:
        return len(self.samples)

    def get_recovery_flags(self) -> list[int]:
        return list(self.recovery_flags)

    # ── H5 handle with periodic reopen to flush internal cache ──
    def _get_h5(self) -> h5py.File:
        """Return h5py.File handle. Reopens every 3000 calls to prevent
        h5py's internal object/metadata cache from growing unbounded."""
        self._h5_reads = getattr(self, '_h5_reads', 0) + 1
        if self._h5_handle is not None and self._h5_reads % 3000 == 0:
            self._h5_handle.close()
            self._h5_handle = None
            gc.collect()
        if self._h5_handle is None:
            self._h5_handle = h5py.File(self.h5_path, "r")
        return self._h5_handle

    # ── Image loading ──
    def _read_jpeg_from_h5(self, f: h5py.File, path: str) -> np.ndarray:
        """Read a JPEG blob from H5 and decode to BGR numpy array."""
        raw = f[path][()]
        if isinstance(raw, np.void):
            raw = raw.tobytes()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Không decode được JPEG: {path}")
        return img  # BGR

    # ── Augmentations (matching dataset.py) ──
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

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        town = sample["town"]
        cam_dir = sample["cam_dir"]
        waypoints = sample["waypoints"].copy()
        command = int(sample["command"])
        recovery = float(sample["recovery_flag"])

        # ── Augmentation decisions ──
        do_flip = self.is_training and random.random() > 0.5
        do_brightness = self.is_training and random.random() > 0.5
        do_shadow = self.is_training and random.random() > 0.5
        do_blur = self.is_training and random.random() > 0.5
        do_noise = self.is_training and random.random() > 0.7
        do_contrast = self.is_training and random.random() > 0.5
        do_cutout = self.is_training and random.random() > 0.5

        # ── Load 3 temporal images from H5 (cached handle) ──
        fnames = [sample["fn_t0"], sample["fn_tm03"], sample["fn_tm06"]]
        images = []
        f = self._get_h5()
        for fn in fnames:
            h5_key = f"{town}/{cam_dir}/{fn}"
            img = self._read_jpeg_from_h5(f, h5_key)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Crop top 45% (sky removal)
            h = img.shape[0]
            img = img[int(h * 0.45):, :, :]

            # Augmentations (applied consistently across 3 temporal frames)
            if self.is_training:
                if do_brightness:
                    img = self._random_brightness(img)
                if do_shadow:
                    img = self._random_shadow(img)
                if do_blur:
                    img = self._random_blur(img)
                if do_noise:
                    img = self._random_noise(img)
                if do_contrast:
                    img = self._random_contrast(img)
                if do_cutout:
                    img = self._random_cutout(img)
                if do_flip:
                    img = cv2.flip(img, 1)

            img = cv2.resize(img, (200, 66))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            images.append(img)

        # Stack 3 temporal frames → (66, 200, 9) → (9, 66, 200)
        stacked = np.concatenate(images, axis=-1)
        tensor = torch.from_numpy(stacked).permute(2, 0, 1).float() / 255.0

        if self.transform:
            tensor = self.transform(tensor)

        # Flip adjustments
        if do_flip:
            waypoints[:, 1] = -waypoints[:, 1]
            if command == 1:
                command = 2
            elif command == 2:
                command = 1

        return (
            tensor,
            torch.tensor(waypoints, dtype=torch.float32),
            torch.tensor(command, dtype=torch.long),
            torch.tensor(recovery, dtype=torch.float32),
        )


# ============================================================================
# 4. TRAINING
# ============================================================================
def main():
    print("\n🔥 BẮT ĐẦU HUẤN LUYỆN (H5)...")

    from core_perception.cnn_model import WaypointPredictor

    # ── Config ──
    with open(os.path.join(WORKING_DIR, "configs", "train_params.yaml"), "r") as f:
        config = yaml.safe_load(f)

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}")

    # ── GPU ──
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    primary_device = torch.device("cuda:0") if num_gpus > 0 else torch.device("cpu")
    use_multi_gpu = num_gpus > 1
    print(f"Device: {num_gpus} GPU(s)" if num_gpus else "Device: CPU")

    # ── Discover towns from H5 ──
    with h5py.File(H5_PATH, "r") as f:
        towns = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
    towns.sort()
    print(f"Towns trong H5: {towns}")

    # ── Find CSV root ──
    csv_root = find_csv_root(H5_PATH, CSV_ROOT)

    # ── Transform ──
    transform = transforms.Compose([transforms.Normalize(mean=[0.5] * 9, std=[0.5] * 9)])

    # ── Datasets ──
    train_ratio = float(config.get("train_split", 0.75))
    geo_offset = float(config.get("geometric_offset", 0.35))

    include_side = bool(config.get("include_side_cameras_train", True))
    print(f"\n📦 Đang tạo H5 datasets... (side_cameras={include_side})")
    train_dataset = WaypointCarlaDatasetH5(
        h5_path=H5_PATH,
        csv_root=csv_root,
        towns=towns,
        transform=transform,
        is_training=True,
        train_ratio=train_ratio,
        seed=seed,
        geometric_offset=geo_offset,
        include_side_cameras=include_side,
    )
    val_dataset = WaypointCarlaDatasetH5(
        h5_path=H5_PATH,
        csv_root=csv_root,
        towns=towns,
        transform=transform,
        is_training=False,
        train_ratio=train_ratio,
        seed=seed,
        geometric_offset=geo_offset,
        include_side_cameras=False,  # Val always center only
    )

    # Free H5 key lookup after building samples (saves ~70MB RAM)
    train_dataset._h5_keys = {}
    val_dataset._h5_keys = {}
    gc.collect()

    print(f"\nTrain: {len(train_dataset)} samples")
    print(f"Val  : {len(val_dataset)} samples")

    # ── DataLoader ──
    # CRITICAL: num_workers=0 to avoid h5py + multiprocessing memory leak.
    # h5py internally caches HDF5 metadata per forked process, growing unbounded
    # until Kaggle OOM-kills the kernel (~batch 900 with 2 workers).
    def collate_fn(batch):
        imgs, wps, cmds, recs = zip(*batch)
        return (
            torch.stack(imgs),
            torch.stack(wps).float(),
            torch.stack(cmds).long(),
            torch.tensor(recs, dtype=torch.float32),
        )

    batch_size = int(config.get("batch_size", 48))  # larger batch OK with 0 workers

    rec_flags = train_dataset.get_recovery_flags()
    rec_weight = float(config.get("recovery_weight", 2.0))
    weights = [rec_weight if int(fl) == 1 else 1.0 for fl in rec_flags]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )

    print(f"DataLoader: batch={batch_size}, workers=0 (h5py safe mode)")
    print(f"  train_steps={len(train_loader)}  val_steps={len(val_loader)}")

    # ── Model (single GPU to reduce memory overhead) ──
    model = WaypointPredictor().to(primary_device)
    use_multi_gpu = False  # DataParallel causes extra memory on GPU:0
    print(f"Single GPU mode: {primary_device}")

    # ── Optimizer / Scheduler ──
    huber = nn.SmoothL1Loss(reduction="mean")
    base_lr = float(config["learning_rate"])
    optimizer = optim.Adam(model.parameters(), lr=base_lr)

    lr_scheduler_type = str(config.get("lr_scheduler", "plateau")).lower()
    if lr_scheduler_type == "cosine":
        # CosineAnnealingLR: smooth decay over all epochs (after warmup)
        effective_epochs = int(config["epochs"]) - int(config.get("warmup_epochs", 1))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, effective_epochs), eta_min=1e-6,
        )
        print(f"  LR Scheduler: CosineAnnealing (T_max={effective_epochs}, eta_min=1e-6)")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=float(config.get("lr_factor", 0.5)),
            patience=int(config.get("lr_patience", 3)),
        )
        print(f"  LR Scheduler: ReduceLROnPlateau (factor={config.get('lr_factor', 0.5)})")
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda" if use_amp else "cpu", enabled=use_amp)

    lambda_wp = float(config.get("loss_lambda_wp", 1.0))
    lambda_gnll = float(config.get("loss_lambda_gnll", 0.05))

    # GNLL stability params
    grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
    sigma_min = float(config.get("sigma_min", 0.01))
    sigma_max = float(config.get("sigma_max", 10.0))
    warmup_epochs = int(config.get("warmup_epochs", 1))

    model_save = Path(WORKING_DIR) / "models" / "waypoint_predictor_h5.pth"
    model_save.parent.mkdir(exist_ok=True)

    epochs = int(config["epochs"])
    patience = int(config.get("early_stopping_patience", 8))
    best_val = float("inf")
    no_improve = 0

    print(f"\n{'=' * 60}")
    print(f"TRAINING | lr={base_lr} | grad_clip={grad_clip_norm} | sigma=[{sigma_min},{sigma_max}]")
    print(f"{'=' * 60}\n")

    for epoch in range(epochs):
        # Linear LR warmup
        if epoch < warmup_epochs:
            warmup_lr = base_lr * (epoch + 1) / max(1, warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # ── TRAIN ──
        model.train()
        running_loss = 0.0
        nan_batches = 0
        for i, (imgs, wps, cmds, _recs) in enumerate(train_loader):
            imgs = imgs.to(primary_device, non_blocking=True)
            cmds = cmds.to(primary_device, non_blocking=True)
            wps = wps.to(primary_device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda" if use_amp else "cpu", enabled=use_amp):
                out = model(imgs, cmds)
                pred_wp = out[:, :10].view(-1, 5, 2)
                # Model already applies softplus — only clamp here
                pred_sig = out[:, 10:].view(-1, 5, 1).expand(-1, 5, 2).clamp(sigma_min, sigma_max)
                tgt_wp = wps.view(-1, 5, 2)
                loss_wp = huber(pred_wp, tgt_wp)
                loss_gnll = 0.5 * ((tgt_wp - pred_wp) ** 2 / pred_sig + torch.log(pred_sig))
                loss = lambda_wp * loss_wp + lambda_gnll * loss_gnll.mean()

            # Skip NaN/Inf batches
            if not torch.isfinite(loss):
                nan_batches += 1
                optimizer.zero_grad(set_to_none=True)
                del out, pred_wp, pred_sig, tgt_wp, loss_wp, loss_gnll, loss, imgs, cmds, wps
                continue

            scaler.scale(loss).backward()

            # Unscale before clipping
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())

            # Explicit cleanup of intermediate tensors
            del out, pred_wp, pred_sig, tgt_wp, loss_wp, loss_gnll, loss, imgs, cmds, wps

            if i % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # CPU RAM check via /proc (Linux/Kaggle)
                cpu_mb = 0
                try:
                    with open('/proc/self/status') as sf:
                        for line in sf:
                            if line.startswith('VmRSS:'):
                                cpu_mb = int(line.split()[1]) // 1024
                                break
                except Exception:
                    pass
                gpu_mb = 0
                if torch.cuda.is_available():
                    gpu_mb = int(torch.cuda.memory_allocated(primary_device) / 1024**2)
                print(
                    f"  [Train] Ep {epoch+1} Batch {i}/{len(train_loader)} "
                    f"| loss={running_loss/max(1,i):.4f} | RAM={cpu_mb}MB GPU={gpu_mb}MB",
                    flush=True,
                )

        n_valid = max(1, len(train_loader) - nan_batches)
        train_loss = running_loss / n_valid
        if nan_batches > 0:
            print(f"  Skipped {nan_batches} NaN/Inf batches")

        # Force close H5 handles between train/val to flush h5py cache
        if train_dataset._h5_handle is not None:
            train_dataset._h5_handle.close()
            train_dataset._h5_handle = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── VAL ──
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_samples = 0
        with torch.no_grad():
            for i, (imgs, wps, cmds, _recs) in enumerate(val_loader):
                imgs = imgs.to(primary_device, non_blocking=True)
                cmds = cmds.to(primary_device, non_blocking=True)
                wps = wps.to(primary_device, non_blocking=True)
                with torch.amp.autocast("cuda" if use_amp else "cpu", enabled=use_amp):
                    out = model(imgs, cmds)
                    pred_wp = out[:, :10].view(-1, 5, 2)
                    pred_sig = out[:, 10:].view(-1, 5, 1).expand(-1, 5, 2).clamp(sigma_min, sigma_max)
                    tgt_wp = wps.view(-1, 5, 2)
                    loss_wp = huber(pred_wp, tgt_wp)
                    loss_g = 0.5 * ((tgt_wp - pred_wp) ** 2 / pred_sig + torch.log(pred_sig))
                    loss = lambda_wp * loss_wp + lambda_gnll * loss_g.mean()
                bs = imgs.size(0)
                val_loss += float(loss.item()) * bs
                val_mae += float((pred_wp - tgt_wp).abs().mean().item()) * bs
                val_samples += bs
                del out, pred_wp, pred_sig, tgt_wp, loss_wp, loss_g, loss, imgs, cmds, wps
                if i % 50 == 0:
                    gc.collect()

        # Flush val H5 handle too
        if val_dataset._h5_handle is not None:
            val_dataset._h5_handle.close()
            val_dataset._h5_handle = None
        gc.collect()

        val_loss = val_loss / max(1, val_samples)
        val_mae = val_mae / max(1, val_samples)

        mem_str = ""
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(primary_device) / 1024 ** 3
            reserv = torch.cuda.memory_reserved(primary_device) / 1024 ** 3
            mem_str = f"| Mem: {alloc:.2f}/{reserv:.2f} GB"

        print(
            f"Epoch [{epoch + 1:2d}/{epochs}] | Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} (MAE={val_mae:.4f}) {mem_str}"
        )

        # Only step scheduler after warmup
        if epoch >= warmup_epochs:
            if lr_scheduler_type == "cosine":
                scheduler.step()
            else:
                scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  LR: {current_lr:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            state = model.module.state_dict() if use_multi_gpu else model.state_dict()
            torch.save(
                {"model_state_dict": state, "epoch": epoch + 1,
                 "val_loss": val_loss, "val_mae": val_mae},
                model_save,
            )
            print(f"  Saved best (val={val_loss:.4f}, MAE={val_mae:.4f}) -> {model_save}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE  |  Best val loss: {best_val:.4f}")
    print(f"   Model: {model_save}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()


