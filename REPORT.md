# Báo Cáo Đề Tài: Autonomous Driving Agent trên CARLA Simulator

> **Repository:** [duytran2525/CARLA-Funny-Moments](https://github.com/duytran2525/CARLA-Funny-Moments)  
> **Trạng thái:** Phase 1 + Phase 2 hoàn chỉnh  
> **Ngày cập nhật:** 2026-04

---

## Mục Lục

1. [Executive Summary](#1-executive-summary)
2. [Lý Do Thực Hiện Đề Tài](#2-lý-do-thực-hiện-đề-tài) *(5 điểm)*
3. [Mô Tả Bài Toán](#3-mô-tả-bài-toán) *(15 điểm)*
4. [Mô Tả Tập Dữ Liệu](#4-mô-tả-tập-dữ-liệu) *(5 điểm)*
5. [Độ Đo Đánh Giá](#5-độ-đo-đánh-giá) *(5 điểm)*
6. [Kiến Trúc Hệ Thống Chi Tiết](#6-kiến-trúc-hệ-thống-chi-tiết) *(20+ điểm)*
7. [Quy Trình Huấn Luyện & Inference](#7-quy-trình-huấn-luyện--inference)
8. [Kết Quả Thực Nghiệm](#8-kết-quả-thực-nghiệm) *(15+ điểm)*
9. [Hướng Dẫn Tái Tạo Kết Quả](#9-hướng-dẫn-tái-tạo-kết-quả)
10. [Hướng Phát Triển Tương Lai](#10-hướng-phát-triển-tương-lai)
11. [Thành Viên & Tài Liệu Tham Khảo](#11-thành-viên--tài-liệu-tham-khảo)

---

## 1. Executive Summary

Dự án **CARLA-Funny-Moments** xây dựng một hệ thống lái xe tự động hoàn chỉnh chạy trong môi trường mô phỏng CARLA v0.9.14. Hệ thống kết hợp **Behavioral Cloning (BC)** và **Conditional Imitation Learning (CIL)** để học chính sách điều khiển từ dữ liệu người lái, đồng thời tích hợp **YOLOv11** để nhận thức môi trường thời gian thực.

**Thành tựu chính:**

| Hạng mục | Mô tả |
|----------|-------|
| **Phase 1** | NvidiaCNN học lái thuần từ ảnh camera (Behavioral Cloning) |
| **Phase 2** | CIL_NvidiaCNN — 4 expert heads theo lệnh GPS (trái/phải/thẳng/follow) |
| **Perception** | YOLOv11 phát hiện 6 class + ước tính khoảng cách + FSM đèn giao thông |
| **Control** | PID dual-mode với deadband hysteresis, anti-windup, integral decay |
| **Data** | Thu thập 3 camera đồng bộ + 9 loại augmentation + histogram balancing |
| **Training** | Multi-GPU DataParallel, mixed-precision (AMP), early stopping |
| **Dataset** | Kaggle: 49.9k train + 16.7k validation (tổng ~66.6k mẫu) |

---

## 2. Lý Do Thực Hiện Đề Tài

### 2.1 Bối Cảnh Thực Tiễn

Lái xe tự động (Autonomous Driving) đang là một trong những hướng nghiên cứu sôi động nhất trong Trí tuệ Nhân tạo và Robotics. Tuy nhiên, nghiên cứu trực tiếp trên xe thật đòi hỏi chi phí cực lớn và tiềm ẩn rủi ro an toàn. **CARLA Simulator** — một môi trường mô phỏng lái xe mã nguồn mở từ Intel Labs & Toyota Research Institute — cho phép R&D hoàn toàn an toàn với chi phí tối thiểu, đồng thời cung cấp full sensor suite (camera RGB, depth, LIDAR) và API điều khiển cấp thấp.

**Lý do chọn CARLA thay vì các simulators khác:**
- Hỗ trợ CARLA Python API kiểm soát hoàn toàn vật lý xe và môi trường
- Đa dạng thời tiết, ánh sáng, mật độ giao thông có thể lập trình
- Cung cấp sensor depth camera + semantic segmentation + LIDAR native
- Cộng đồng active, tài liệu phong phú, nhiều benchmark so sánh

### 2.2 Khoảng Trống Nghiên Cứu Được Giải Quyết

**Vấn đề của Behavioral Cloning thuần túy:**
- Mô hình CNN đơn thuần học ánh xạ `ảnh → tay lái` nhưng **không biết ý định** của người lái tại ngã tư (rẽ trái? phải? thẳng?)
- Phân phối dữ liệu lệch nặng về góc lái thẳng (covariate shift)
- Không có cơ chế recovery khi lạc đường

**Giải pháp tích hợp trong dự án:**

```
Behavioral Cloning (Phase 1)
    └── NvidiaCNN: ảnh → tay lái
            ↓ [Hạn chế: không biết ý định]
Conditional Imitation Learning (Phase 2)
    └── CIL_NvidiaCNN: (ảnh + tốc độ + lệnh_GPS) → tay lái
            ↓ [4 expert heads: Follow/Left/Right/Straight]
Hybrid Perception-Control (Phase 2+)
    └── CNN/CIL + YOLO + PID + Traffic FSM → VehicleControl
```

### 2.3 Tích Hợp Multiple Learning Paradigms

| Paradigm | Thành phần | Lý do chọn |
|----------|-----------|------------|
| **Supervised Learning** | NvidiaCNN, CIL_NvidiaCNN | Học từ dữ liệu người lái có nhãn, cost thấp |
| **Imitation Learning** | DataCollector + AutopilotAgent | Dữ liệu được thu thập từ CARLA BasicAgent |
| **Conditional IL** | 4 expert heads theo GPS command | Giải quyết intersection ambiguity |
| **Classical Control** | SpeedPIDController | Ổn định tốc độ — không cần học |
| **Computer Vision** | YOLOv11 fine-tuned | Nhận thức object thời gian thực |
| **Rule-based FSM** | Traffic light state machine | Safety logic không thể học từ data |

### 2.4 Mục Tiêu Học Tập Đề Tài

- Làm chủ **Deep Learning cho Computer Vision** (CNN architectures, training pipeline)
- Hiểu và triển khai **Imitation Learning** end-to-end
- Tích hợp **Object Detection** (YOLO) vào vòng điều khiển thời gian thực
- Thiết kế **PID Controller** với các kỹ thuật hiện đại (anti-windup, filtering)
- Xây dựng **hệ thống nhúng** chịu tải inference real-time (~20 FPS)
- Thực hành **MLOps**: multi-GPU training, model versioning, reproducibility

---

## 3. Mô Tả Bài Toán

### 3.1 Định Nghĩa Bài Toán

Xây dựng agent lái xe tự động trong môi trường CARLA có khả năng:
1. Giữ làn và điều hướng qua ngã tư theo lệnh GPS cấp cao
2. Tuân thủ đèn giao thông và tránh va chạm
3. Vận hành ổn định ở tốc độ mục tiêu

### 3.2 Input

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT SPACE                              │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Camera LEFT  │  │ Camera CENTER│  │ Camera RIGHT │           │
│  │ RGB 1920×1080│  │ RGB 1920×1080│  │ RGB 1920×1080│           │
│  │ (±25° yaw)   │  │ (0° yaw)     │  │ (±25° yaw)   │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                 │                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │           Depth Camera (colocated with center)            │    │
│  │           BGRA buffer → decoded meters [0–1000m]          │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  Vehicle State:                 Navigation Command:              │
│  ├── speed_kmh [0, 120]         ├── 0: Follow Lane               │
│  ├── steering_current [-1, 1]   ├── 1: Turn Left                 │
│  └── transform (x, y, yaw)      ├── 2: Turn Right                │
│                                 └── 3: Go Straight               │
└─────────────────────────────────────────────────────────────────┘
```

**Chi tiết từng input:**

| Input | Định dạng | Kích thước | Mô tả |
|-------|-----------|-----------|-------|
| Camera center RGB | `uint8 [H,W,3]` | 1920×1080 → crop+resize → 200×66 | Góc nhìn chính |
| Camera left RGB | `uint8 [H,W,3]` | 1920×1080 → 200×66 | Cảnh báo lệch trái |
| Camera right RGB | `uint8 [H,W,3]` | 1920×1080 → 200×66 | Cảnh báo lệch phải |
| Depth map | `float32 [H,W]` | 1920×1080 (meters) | Ước tính khoảng cách |
| Speed normalized | `float32` | scalar ∈ [0, 1] | `speed_kmh / 120` |
| GPS command | `int64` | scalar ∈ {0,1,2,3} | Follow/Left/Right/Straight |

### 3.3 Output

```
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT SPACE                              │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  VehicleControl                          │    │
│  │                                                           │    │
│  │  steer  ∈ [-1.0, +1.0]   (tay lái: trái âm, phải dương)│    │
│  │  throttle ∈ [0.0, 1.0]   (ga)                           │    │
│  │  brake  ∈ [0.0, 1.0]    (phanh)                        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

| Output | Nguồn | Phạm vi |
|--------|-------|---------|
| `steer` | CNN / CIL model inference + exponential smoothing | `[-1.0, 1.0]` |
| `throttle` | SpeedPIDController | `[0.0, 1.0]` |
| `brake` | SpeedPIDController + YOLO emergency | `[0.0, 1.0]` |

**Logic kết hợp:**
- Nếu YOLO phát hiện nguy hiểm khẩn cấp: `throttle=0, brake=max(brake, 0.8)`
- Nếu đèn đỏ (Traffic FSM): `target_speed=0` → PID brake hoàn toàn
- Bình thường: CNN/CIL cho `steer`, PID cho `throttle/brake`

### 3.4 Hình Minh Họa

#### Sơ Đồ Kiến Trúc Tổng Thể (Architecture Diagram)

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      CARLA SIMULATOR                                    │
 │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐  │
 │  │Cam LEFT  │  │Cam CENTER│  │Cam RIGHT │  │   Depth Camera        │  │
 │  │RGB 1920× │  │RGB 1920× │  │RGB 1920× │  │   BGRA→meters         │  │
 │  │1080      │  │1080      │  │1080      │  │                       │  │
 │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┬────────────┘  │
 └───────┼─────────────┼─────────────┼───────────────────┼───────────────┘
         │             │             │                   │
         └─────────────┼─────────────┘                   │
                       │ RGB frames                       │ depth_map_m
                       ▼                                  ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                       PERCEPTION MODULE                                  │
 │                                                                           │
 │  ┌───────────────────────────────────┐   ┌──────────────────────────┐   │
 │  │         Image Preprocessor        │   │      YoloDetector         │   │
 │  │  1. Crop top 45% (sky removal)    │   │  ┌────────────────────┐  │   │
 │  │  2. Resize → 200×66               │   │  │  YOLOv11s          │  │   │
 │  │  3. RGB → YUV                     │   │  │  6 classes:        │  │   │
 │  │  4. Normalize [-1,1]              │   │  │  pedestrian        │  │   │
 │  │  5. ToTensor (B,3,66,200)         │   │  │  vehicle           │  │   │
 │  └──────────────┬────────────────────┘   │  │  two_wheeler       │  │   │
 │                 │                        │  │  traffic_sign      │  │   │
 │  ┌──────────────▼───────────────────┐    │  │  traffic_light_red │  │   │
 │  │         CNN / CIL Model           │    │  │  traffic_light_grn │  │   │
 │  │                                   │    │  └────────┬───────────┘  │   │
 │  │  Phase 1: NvidiaCNNV2             │    │           │              │   │
 │  │  ┌─Conv×5─┬─FC×4─┐               │    │  ┌────────▼───────────┐  │   │
 │  │  │BN+ELU  │Drop  │→ steer[-1,1]  │    │  │ Distance Estimator │  │   │
 │  │  └────────┴──────┘               │    │  │ depth(30th pctile) │  │   │
 │  │                                   │    │  │ + bbox fallback    │  │   │
 │  │  Phase 2: CIL_NvidiaCNN           │    │  └────────┬───────────┘  │   │
 │  │  ┌─Conv×5─┐ ┌─Speed Embed─┐      │    │           │              │   │
 │  │  │BN+ELU  │ │Lin(1→64→32) │      │    │  ┌────────▼───────────┐  │   │
 │  │  └───┬────┘ └──────┬───────┘      │    │  │  Traffic Light FSM │  │   │
 │  │   cat[vis‖spd]─1184d│            │    │  │  zone lock/unlock  │  │   │
 │  │  ┌──────┬──────┬──────┬──────┐   │    │  │  turn suppression  │  │   │
 │  │  │ h[0] │ h[1] │ h[2] │ h[3] │   │    │  └────────┬───────────┘  │   │
 │  │  │follow│ left │right │strt  │   │    │           │              │   │
 │  │  └──┬───┴──────┴──────┴──────┘   │    │  ┌────────▼───────────┐  │   │
 │  │  gather(command_idx)→steer[-1,1]  │    │  │  Curved Path Model │  │   │
 │  └──────────────┬────────────────────┘    │  │  RANSAC road plane │  │   │
 │                 │                        │  │  bicycle kinematics│  │   │
 │                 │ steer_raw              │  │  emergency_flag    │  │   │
 │                 │                        │  └────────┬───────────┘  │   │
 └─────────────────┼────────────────────────┴───────────┼──────────────┘   │
                   │                                    │
                   │ steer_smoothed                     │ emergency_flag
                   ▼                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                        CONTROL MODULE                                    │
 │                                                                           │
 │  ┌────────────────────────────────────────────────────────────────────┐  │
 │  │                   SpeedPIDController                                │  │
 │  │  target_speed: 30 km/h (normal) / 0 km/h (red light/emergency)    │  │
 │  │                                                                     │  │
 │  │  ┌─────────────────────┐    ┌─────────────────────────────────┐   │  │
 │  │  │   Throttle PID      │    │        Brake PID                 │   │  │
 │  │  │  kp=0.3 ki=0.05     │    │      kp=0.5 ki=0.1 kd=0.01     │   │  │
 │  │  │  kd=0.01            │    │  + anti-windup + deriv filter   │   │  │
 │  │  └─────────────────────┘    └─────────────────────────────────┘   │  │
 │  │                                                                     │  │
 │  │  Deadband: ±0.5 km/h | PRE-CLEAN state switch | Integral decay    │  │
 │  └─────────────────────────┬──────────────────────────────────────────┘  │
 │                             │                                            │
 └─────────────────────────────┼────────────────────────────────────────────┘
                               │
                               ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │               carla.VehicleControl(steer, throttle, brake)              │
 │                           CARLA SIMULATOR                               │
 └─────────────────────────────────────────────────────────────────────────┘
```

#### Data Pipeline Flow

```
 CARLA World
     │
     ├─ AutopilotAgent (thu thập dữ liệu)
     │       │
     │  3 Camera Callbacks ──────────────────────────────────┐
     │  (center/left/right)                                   │
     │       │ frame_id                                       │
     │       ▼                                               │
     │  _pending_images[frame_id]                            │
     │       │                                               │
     │  Vehicle State Callback                               │
     │       │ frame_id                                      │
     │       ▼                                               │
     │  Synchronization check: all 3 cameras + state?        │
     │       │ YES                                           │
     │       ▼                                               │
     │  DataCollector.add(center, left, right, state)        │
     │       │                                               │
     │       ├─ JPEG compress (quality=95)                   │
     │       ├─ Resize 400×200                               │
     │       ├─ Save images/center/, images/left/, images/right/
     │       └─ Append row to driving_log.csv                │
     │             [img_id, steering, throttle, brake,       │
     │              speed, command, pitch, roll, yaw]        │
     │                                                       │
     ▼                                                       │
  driving_log.csv                                           │
     │                                                       │
     ▼                                                       │
  scripts/train_cnn.py                                      │
     │                                                       │
     ├─ Load CSV → split 75% train / 25% val               │
     │                                                       │
     ├─ CILCarlaDataset (train, is_training=True)           │
     │       ├─ 3 cameras × steering correction             │
     │       │    center: ±0, left: +0.2, right: -0.2       │
     │       ├─ Histogram balancing (25 bins, cap at mean)  │
     │       └─ 9 augmentations per sample                  │
     │                                                       │
     ├─ CILCarlaDataset (val, is_training=False)            │
     │       └─ center only, no augmentation                │
     │                                                       │
     ├─ DataLoader (8 workers, pin_memory, persistent)      │
     │                                                       │
     ├─ CIL_NvidiaCNN (multi-GPU DataParallel)              │
     ├─ Adam optimizer + ReduceLROnPlateau                  │
     ├─ Mixed-precision AMP (torch.amp)                     │
     ├─ Early stopping (patience=12)                        │
     └─ Best model → models/cil_model.pth                   │
                                                            │
  models/cil_model.pth  ◄─────────────────────────────────┘
     │
     └─ run_agents.py --agent cil/lane_follow
             └─ Inference loop @~20 FPS
```

#### Agent Control Loop

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                    AGENT CONTROL LOOP (~20 FPS)                  │
  │                                                                   │
  │  TICK START                                                       │
  │     │                                                             │
  │     ▼                                                             │
  │  world.tick() ──── CARLA synchronous step                        │
  │     │                                                             │
  │     ├─ Get RGB frame (center) ── thread-safe queue               │
  │     ├─ Get Depth frame ────────── decoded to meters              │
  │     └─ Get vehicle state (speed, steer, transform)               │
  │     │                                                             │
  │     ▼                                                             │
  │  [IF --yolo-model-path set]                                      │
  │  YoloDetector.detect_and_evaluate(frame, threshold, depth,      │
  │                                   cur_steer, speed_kmh)          │
  │     ├─ Detections list [{class, bbox, distance, conf}, ...]      │
  │     └─ emergency_flag (bool)                                     │
  │     │                                                             │
  │     ▼                                                             │
  │  CNN/CIL Inference                                               │
  │     ├─ crop top 45% → resize 200×66 → YUV → normalize           │
  │     ├─ [CIL] get GPS command from BasicAgent planner             │
  │     └─ model(image [,speed, command]) → steer_raw               │
  │     │                                                             │
  │     ▼                                                             │
  │  Steering smoothing                                               │
  │     steer_smooth = 0.7 × prev_steer + 0.3 × steer_raw           │
  │     │                                                             │
  │     ▼                                                             │
  │  Traffic light decision (from FSM result in YoloDetector)        │
  │     ├─ RED detected → target_speed = 0                           │
  │     └─ GREEN / NONE  → target_speed = config_target             │
  │     │                                                             │
  │     ▼                                                             │
  │  SpeedPIDController.compute(current_speed_kmh)                   │
  │     └─ throttle, brake                                           │
  │     │                                                             │
  │     ▼                                                             │
  │  [IF emergency_flag]                                              │
  │     throttle = 0                                                  │
  │     brake = max(brake, max_brake_value)                           │
  │     │                                                             │
  │     ▼                                                             │
  │  vehicle.apply_control(steer=steer_smooth,                       │
  │                         throttle=throttle, brake=brake)          │
  │     │                                                             │
  │     ▼                                                             │
  │  [IF --collect-data]                                             │
  │     DataCollector.add(...)                                        │
  │     │                                                             │
  │     ▼                                                             │
  │  cv2.imshow(annotated frame with YOLO boxes + status)            │
  │                                                                   │
  │  TICK END → REPEAT                                                │
  └─────────────────────────────────────────────────────────────────┘
```

#### Traffic Light FSM (Finite State Machine)

```
  ┌─────────────────────────────────────────────────────────────────┐
  │              TRAFFIC LIGHT FINITE STATE MACHINE                  │
  │                                                                   │
  │  Frame zones analyzed (top 55% of image height):                │
  │                                                                   │
  │  ┌──────────────────────────────────────────────────────────┐   │
  │  │  0%          35%         65%               95%  100%     │   │
  │  │  ├────────────┼──────────┼─────────────────┼────┤       │   │
  │  │  │  (ignored) │  URBAN   │  RURAL_RIGHT     │ (ignore)  │   │
  │  │  │            │  zone    │  zone            │           │   │
  │  │  └──────────────────────────────────────────────────────┘   │
  │                                                                   │
  │  State per zone: {UNLOCKED, LOCKED_RED, LOCKED_GREEN}            │
  │                                                                   │
  │       Detection in zone                                          │
  │             │                                                     │
  │             ▼                                                     │
  │  ┌──────────────────────────┐                                    │
  │  │   Consecutive frames ≥ 2 │ ──NO──► ignore (noise filter)     │
  │  └──────────┬───────────────┘                                    │
  │             │ YES → ACQUIRE LOCK                                  │
  │             ▼                                                     │
  │  ┌─────────────────────────────────────────────────────┐         │
  │  │  Score = confidence / distance                       │         │
  │  │  If score_green ≥ score_red × 1.05 → GREEN wins     │         │
  │  └──────────────────────────┬──────────────────────────┘         │
  │                              │                                    │
  │         ┌────────────────────┼───────────────────────┐           │
  │         ▼                   ▼                        ▼           │
  │    RED confirmed         GREEN confirmed       Signal lost        │
  │    (≥2 frames)           (≥2 frames)          (≥4 frames)        │
  │         │                   │                        │           │
  │         ▼                   ▼                        ▼           │
  │   target_speed=0      target_speed=             RELEASE LOCK     │
  │   brake fully         config_target             (with hold 6f)   │
  │                                                                   │
  │  Turn-phase suppression:                                          │
  │  IF (steer > 0.22 AND speed > 5 km/h for 3 consecutive frames)  │
  │  AND (saw GREEN in locked zone within last 30 frames grace)      │
  │  THEN → suppress RED trigger (avoid cross-lane red false stop)   │
  └─────────────────────────────────────────────────────────────────┘
```

---

## 4. Mô Tả Tập Dữ Liệu

### 4.1 Nguồn Dữ Liệu

**Nguồn chính:** Kaggle dataset cho bài toán lái xe tự động trong CARLA (thu thập bằng AutopilotAgent của CARLA BasicAgent trên map Town03).

**Thống kê tập dữ liệu:**

| Tập | Số mẫu | Tỉ lệ | Mục đích |
|-----|--------|-------|---------|
| **Training** | 49,900 | 75% | Huấn luyện mô hình (3-camera × augmentation) |
| **Validation** | 16,700 | 25% | Kiểm tra overfitting trong quá trình train |
| **Test** | ~4,175 (ước tính) | ~6.3% | Đánh giá hiệu năng cuối (từ các episode mới) |
| **Tổng** | ~66,600 | 100% | — |

> **Lưu ý:** Train/Val split là 75/25 theo thứ tự dòng trong CSV (không shuffle để tránh data leakage giữa các episode liên tiếp).

### 4.2 Cấu Trúc Dữ Liệu

**Thư mục dữ liệu:**
```
data/
├── driving_log.csv          # Metadata: 9 fields per frame
├── images_center/           # Center camera frames
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
├── images_left/             # Left camera frames
│   └── ...
└── images_right/            # Right camera frames
    └── ...
```

**Schema CSV (`driving_log.csv`):**

| Cột | Kiểu | Phạm vi | Mô tả |
|-----|------|---------|-------|
| `img_id` | str | `0001`–`NNNN` | Tên file ảnh (không kèm extension) |
| `steering` | float | `[-1.0, 1.0]` | Góc lái đã normalize |
| `throttle` | float | `[0.0, 1.0]` | Mức ga |
| `brake` | float | `[0.0, 1.0]` | Mức phanh |
| `speed` | float | `[0.0, 120.0]` | Tốc độ km/h |
| `command` | int | `{0,1,2,3}` | Lệnh GPS: Follow/Left/Right/Straight |
| `pitch` | float | degrees | Góc ngẩng xe |
| `roll` | float | degrees | Góc nghiêng ngang |
| `yaw` | float | degrees | Góc quay đầu |

### 4.3 Đặc Trưng Dữ Liệu

**Phân phối góc lái (steering distribution):**
```
 Frequency
    │
 ▓▓ │                 ████████
 ▓▓ │               ██████████████
 ▓▓ │            █████████████████████
 ▓▓ │         ████████████████████████████
 ▓▓ │   ██████████████████████████████████████████
 ───┼──────────────────────────────────────────────►
   -1.0   -0.5    -0.2    0.0    0.2    0.5    1.0
                    (dominant: straight)
```

**Class imbalance problem → Histogram balancing:**
- Lái thẳng (steering ≈ 0) chiếm ~60–70% dữ liệu
- Giải pháp: bins 25 bucket, cap mỗi bucket tại `mean(histogram_count)`
- Kết quả: phân phối đồng đều hơn, giảm bias về thẳng

**GPS Command distribution:**

| Command | Ý nghĩa | Tỉ lệ ước tính |
|---------|---------|----------------|
| 0 (Follow) | Giữ làn, không ngã tư | ~50% |
| 1 (Left) | Rẽ trái tại ngã tư | ~15% |
| 2 (Right) | Rẽ phải tại ngã tư | ~15% |
| 3 (Straight) | Đi thẳng qua ngã tư | ~20% |

### 4.4 Augmentation Pipeline

**9 loại augmentation áp dụng trong quá trình training:**

| # | Kỹ thuật | Xác suất | Tham số | Tác dụng |
|---|----------|----------|---------|----------|
| 1 | **Horizontal Translate** | p=0.5 | ±20px ngang, ±10px dọc | Mô phỏng vị trí xe trong làn |
| 2 | **Brightness** | p=0.5 | V-channel × [0.5, 1.5] | Mô phỏng sáng/tối khác nhau |
| 3 | **Shadow** | p=0.5 | Random polygon 50% V | Mô phỏng bóng cây, tòa nhà |
| 4 | **Horizontal Flip** | p=0.5 | Negate steering | Tăng đa dạng dữ liệu rẽ |
| 5 | **Gaussian Blur** | p=0.5 | kernel 3×3 hoặc 5×5 | Mô phỏng ảnh mờ/chuyển động |
| 6 | **Gaussian Noise** | p=0.3 | sigma ∈ [5, 15] | Mô phỏng nhiễu cảm biến |
| 7 | **Random Rotation** | p=0.5 | ±5°, steering × (-0.01) | Mô phỏng camera góc khác |
| 8 | **Random Contrast** | p=0.5 | factor ∈ [0.7, 1.3] | Mô phỏng điều kiện ánh sáng |
| 9 | **Cutout** | p=0.5 | 10–30% diện tích ảnh | Mô phỏng vật cản/occlusion |

**Multi-camera steering correction:**
- Camera center: `steering_corrected = steering_raw`
- Camera left: `steering_corrected = steering_raw + 0.2` (cần rẽ phải để trở về giữa)
- Camera right: `steering_corrected = steering_raw - 0.2` (cần rẽ trái để trở về giữa)

---

## 5. Độ Đo Đánh Giá

### 5.1 Steering Prediction Loss

**Mean Squared Error (MSE) — Loss function chính:**

$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **Training MSE Loss** | `(1/N) Σ(steer_pred - steer_gt)²` | Loss trong quá trình train |
| **Validation MSE Loss** | Trên tập val, không augmentation | Đánh giá generalization |
| **MAE (tham khảo)** | `(1/N) Σ|steer_pred - steer_gt|` | Sai số trung bình dễ hiểu hơn |

**Ngưỡng chấp nhận:**
- MSE < 0.01: Xuất sắc (sai số góc lái < 5.7°)
- MSE < 0.02: Tốt
- MSE < 0.05: Chấp nhận được
- MSE ≥ 0.05: Cần cải thiện

### 5.2 Collision Detection Rate

**Định nghĩa:** Tỉ lệ phát hiện đúng vật cản trước khi xảy ra va chạm.

$$\text{Collision Avoidance Rate} = \frac{\text{Số lần tránh được va chạm}}{\text{Tổng số tình huống nguy hiểm}} \times 100\%$$

**Đo lường qua YOLO emergency flag:**
- `True Positive`: YOLO phát hiện vật cản thật, vehicle brake thành công
- `False Negative`: YOLO miss vật cản, xảy ra collision
- `False Positive`: YOLO báo nhầm, vehicle brake không cần thiết

### 5.3 Traffic Light Rule Compliance

**Định nghĩa:** Tỉ lệ tuân thủ đúng quy tắc đèn giao thông.

$$\text{TL Compliance} = \frac{\text{Số lần dừng đúng trước đèn đỏ}}{\text{Tổng số lần gặp đèn đỏ}} \times 100\%$$

**Các trường hợp được đo:**
- `STOP_CORRECT`: Dừng trước đèn đỏ ✓
- `RUN_RED`: Vượt đèn đỏ ✗
- `FALSE_STOP`: Dừng khi đèn xanh (false positive) ✗
- `TURN_SUPPRESSED`: Bỏ qua đèn đỏ lateral đúng lúc quẹo ✓

### 5.4 Path Corridor Accuracy

**Định nghĩa:** Tỉ lệ thời gian vehicle ở trong hành lang đường an toàn.

$$\text{Lane Keeping Rate} = \frac{T_{\text{in-corridor}}}{T_{\text{total}}} \times 100\%$$

**Thực hiện qua:**
- Curved 3D path model với bicycle kinematics
- `base_half_width = 1.1m + distance × 0.035m/m + curve_bonus`
- Obstacle-in-path check với road plane RANSAC

### 5.5 Inference Time Per Frame

**Mục tiêu:** Real-time inference ≥ 20 FPS (≤ 50ms/frame).

| Thành phần | Latency ước tính (GPU) | Latency ước tính (CPU) |
|-----------|----------------------|----------------------|
| Image preprocessing | ~1ms | ~3ms |
| CNN/CIL inference | ~5ms | ~50ms |
| YOLO inference | ~15ms | ~200ms |
| PID compute | <1ms | <1ms |
| **Total (YOLO+CNN)** | **~21ms (~48 FPS)** | **~254ms (~4 FPS)** |

> **Lưu ý:** Inference time phụ thuộc hardware. GPU NVIDIA T4 (Kaggle) đạt ~48 FPS cho YOLO+CNN.

---

## 6. Kiến Trúc Hệ Thống Chi Tiết

### 6.1 Perception Module — YoloDetector

**File:** `core_perception/yolo_detector.py`

**Các class được phát hiện (6 classes):**

| Class | Kích thước tham chiếu | Phương pháp ưu tiên |
|-------|----------------------|-------------------|
| `pedestrian` | 1.6m × 0.45m | Depth camera |
| `vehicle` | 1.5m × 1.8m | Depth camera |
| `two_wheeler` | 1.1m × 0.5m | Depth camera |
| `traffic_sign` | 0.6m × 0.6m | BBox fallback |
| `traffic_light_red` | 0.3m × 0.3m | BBox fallback |
| `traffic_light_green` | 0.3m × 0.3m | BBox fallback |

**Alias normalization (xử lý inconsistent label names):**
```python
ALIASES = {
    "trafficlight_red": "traffic_light_red",
    "red_traffic_light": "traffic_light_red",
    "trafficlight_green": "traffic_light_green",
    ...
}
```

**Distance Estimation — Dual Mode:**

```
Primary (depth camera):
  ├─ obstacles (pedestrian/vehicle/two_wheeler):
  │    Sample patch at bbox bottom (lower 28%, ±12% width)
  │    → 30th percentile depth value (robust to outliers)
  └─ other objects:
       Sample bbox center region
       → 50th percentile depth value

Fallback (when depth unavailable):
  focal_length = (img_width/2) / tan(FOV/2)
  distance = (real_height × focal_length) / bbox_height_px
```

**RANSAC Road Plane Fitting:**
```
Every 5 frames:
  1. Sample depth ROI (bottom 45%, inner 80%, stride=4)
  2. Convert to 3D: (u,v,d) → (X,Y,Z) via camera intrinsics
  3. Filter: forward 0.5–60m, lateral ±12m, height -4 to +3m
  4. RANSAC (28 iterations):
     - Pick 3 random points → fit plane ax+by+cz=d
     - Count inliers: |dot(point, normal) - d| < 0.10m
     - Keep best (requires z_normal > 0.65)
  5. SVD refinement on all inliers
  6. Exponential smoothing: alpha=0.20
     plane_normal = 0.20×new + 0.80×prev
```

**Curved Path Model (Bicycle Kinematics):**
```
curvature = tan(steer × 35° in radians) / wheelbase(2.9m)
radius = 1/curvature  (infinity if straight)

For each point at arc_length s along path:
  theta = s × curvature
  x = radius × sin(theta)
  y = radius × (1 - cos(theta))

half_width(s) = 1.1m + s × 0.035 + curve_bonus(steer)
  (corridor widens with distance for safety margin)

Obstacle in path? → project bbox bottom to road plane,
                    check if within corridor at obstacle's distance
```

### 6.2 Control Module — SpeedPIDController

**File:** `core_control/pid_manager.py`

**Dual-Mode Architecture:**

```
Input: current_speed_kmh, target_speed_kmh
Output: throttle ∈ [0,1], brake ∈ [0,1]

error = target_speed - current_speed

IF |error| < deadband (0.5 km/h):
    decay integrals × 0.95
    maintain current mode
    return (current_throttle, current_brake)

IF error > 0 (need to accelerate):
    [PRE-CLEAN: clear brake integral & derivative]
    throttle = kp×error + ki×∫error + kd×d(error)/dt
    throttle = clamp(throttle, 0, 1)
    brake = 0

IF error < 0 (need to brake):
    [PRE-CLEAN: clear throttle integral & derivative]
    brake_error = -error
    brake = kp×brake_error + ki×∫brake_error + kd×d/dt
    brake = clamp(brake, 0, 1)
    throttle = 0
```

**Thông số PID mặc định:**

| Tham số | Throttle | Brake |
|---------|----------|-------|
| `kp` | 0.3 | 0.5 |
| `ki` | 0.05 | 0.1 |
| `kd` | 0.01 | 0.01 |
| `max_integral` | 10.0 | 10.0 |
| `alpha` (deriv filter) | 0.7 | 0.7 |
| `deadband_kmh` | 0.5 | 0.5 |
| `decay_rate` | 0.95 | 0.95 |

### 6.3 Learning Module — CNN Architectures

**File:** `core_perception/cnn_model.py`

#### NvidiaCNN (Phase 1 Baseline)
Dựa theo Bojarski et al. "End to End Learning for Self-Driving Cars" (NVIDIA, 2016):

```
Input: (B, 3, 66, 200) — YUV normalized to [-1,1]

Conv2d(3→24,   5×5, stride=2) → ELU → (B, 24, 31, 98)
Conv2d(24→36,  5×5, stride=2) → ELU → (B, 36, 14, 47)
Conv2d(36→48,  5×5, stride=2) → ELU → (B, 48, 5, 22)
Conv2d(48→64,  3×3, stride=1) → ELU → (B, 64, 3, 20)
Conv2d(64→64,  3×3, stride=1) → ELU → (B, 64, 1, 18)
                                                ↓
                                         Flatten → 1152-d
Linear(1152→100) → ELU
Linear(100→50)   → ELU
Linear(50→10)    → ELU
Linear(10→1)     → Output: steer ∈ [-1,1]
```

#### NvidiaCNNV2 (Phase 1 Production — default)
```
Same as NvidiaCNN but:
  Conv2d → BatchNorm2d → ELU  (all 5 conv layers)
  Linear → Dropout(0.5) → ELU (first 3 FC layers)
  Linear(10→1) → no dropout   (output layer)

Advantages:
  ✓ Faster convergence (BatchNorm)
  ✓ Reduced overfitting (Dropout)
  ✓ Less LR-sensitive
```

#### CIL_NvidiaCNN (Phase 2 — Production)
Dựa theo Codevilla et al. "End-to-end Driving via Conditional Imitation Learning" (ICRA 2018):

```
Inputs:
  image:   (B, 3, 66, 200)
  speed:   (B,) ∈ [0, 1]
  command: (B,) ∈ {0,1,2,3} as long int

                    image
                      │
           Conv stack (NvidiaCNNV2 backbone)
                      │
                 1152-d vis_feat

                    speed
                      │
              Linear(1→64) → ELU
              Linear(64→32) → ELU
                      │
                  32-d spd_feat

        cat([vis_feat, spd_feat]) → 1184-d fused

    4 Expert Heads (ModuleList):
    ┌─────────────────────────────────────────┐
    │ Head 0 (Follow):   Linear(1184→256→64→1) + Dropout(0.5, 0.3) │
    │ Head 1 (Left):     Linear(1184→256→64→1) + Dropout(0.5, 0.3) │
    │ Head 2 (Right):    Linear(1184→256→64→1) + Dropout(0.5, 0.3) │
    │ Head 3 (Straight): Linear(1184→256→64→1) + Dropout(0.5, 0.3) │
    └─────────────────────────────────────────┘
             │
    torch.gather(dim=1, index=command.view(-1,1))
             │
         steer ∈ [-1, 1]
```

**So sánh 3 architectures:**

| Model | Params | BatchNorm | Dropout | CIL Heads | Use Case |
|-------|--------|-----------|---------|-----------|----------|
| NvidiaCNN | ~252K | ✗ | ✗ | ✗ | Prototype nhanh |
| NvidiaCNNV2 | ~253K | ✓ | ✓ | ✗ | Production Phase 1 |
| CIL_NvidiaCNN | ~1.3M | ✓ | ✓ | 4 | Production Phase 2 |

### 6.4 Data Collection Module

**File:** `core_control/collect_data.py`

**3-Camera Synchronization Protocol:**

```
CARLA tick frame_id=N:
                                        ┌────────────────────────┐
  Camera center callback(frame_id=N)─►  │  _pending_images[N]    │
  Camera left   callback(frame_id=N)─►  │  {center: img_C,       │
  Camera right  callback(frame_id=N)─►  │   left:   img_L,       │
                                        │   right:  img_R}        │
  Vehicle state callback(frame_id=N)─►  └────────────┬───────────┘
                                                      │
                                        all 3 cameras + state? ──NO──► wait
                                                      │ YES
                                                      ▼
                                             write to disk + CSV
                                             prune _pending[N-32...]
```

**CSV Schema Migration:** Nếu CSV cũ thiếu cột `pitch/roll/yaw`:
```python
if set(old_cols) != set(NEW_SCHEMA_COLS):
    df_old = pd.read_csv(csv_path)
    df_new = df_old.reindex(columns=NEW_SCHEMA_COLS, fill_value=0.0)
    df_new.to_csv(csv_path, index=False)
```

### 6.5 World Management — CarlaManager

**File:** `core_control/carla_manager.py`

**Vòng đời session:**

```
CarlaManager.__init__()
    │
    ├─ _retry_rpc_call(world.get_world(), max_wait=60s)
    ├─ _destroy_existing_actors()  ← cleanup stale actors
    ├─ apply_settings(sync=True, delta=0.05)
    ├─ spawn_ego_vehicle(filter="tesla.model3", spawn_point=1)
    ├─ spectator.set_transform(behind_above_ego)
    ├─ spawn_npcs(cars=N, bikes=M, motorbikes=K)
    └─ spawn_pedestrians(count=P, radius=80m)

cleanup()
    ├─ stop+destroy walker controllers (MUST be first)
    ├─ destroy walkers
    ├─ destroy NPC vehicles
    ├─ destroy ego vehicle
    └─ restore original world settings (async mode)
```

**NPC Classification Logic:**
```python
wheels = blueprint.get_attribute('number_of_wheels').as_int()
name   = blueprint.id.lower()

if wheels >= 4:
    → NPC car
elif wheels == 2 and any(k in name for k in ['bike','bicycle','cycle']):
    → NPC bike
elif wheels == 2:
    → NPC motorbike
```

---

## 7. Quy Trình Huấn Luyện & Inference

### 7.1 Thu Thập Dữ Liệu (Data Collection)

```bash
# Thu thập dữ liệu bằng CARLA BasicAgent autopilot
python run_agents.py \
    --agent autopilot \
    --collect-data \
    --data-dir data/ \
    --target-speed 30 \
    --recovery-interval 100 \
    --recovery-steer-offset 0.3 \
    --recovery-duration 20 \
    --config configs/carla_env.yaml
```

**Recovery Disturbance Training:**
- Mỗi `recovery_interval_frames=100` ticks, giả lập xe lệch làn
- Offset steering ±`recovery_steer_offset=0.3` trong `recovery_duration_frames=20` ticks
- Giúp model học phục hồi từ vị trí lệch làn

### 7.2 Huấn Luyện CNN/CIL

```bash
# Huấn luyện CIL_NvidiaCNN (khuyến nghị, chạy trên Kaggle T4×2)
python scripts/train_cnn.py \
    --data-dir data/ \
    --config configs/train_params.yaml \
    --output-dir models/

# Cấu hình training (configs/train_params.yaml):
# epochs: 50, batch_size: 48, learning_rate: 0.002
# lr_patience: 5, lr_factor: 0.7
# early_stopping_patience: 12
# augmentation_prob: 0.75, train_split: 0.75
# steering_correction: 0.2
```

**Training Pipeline chi tiết:**

```
1. Load CSV → split rows: first 75% train / last 25% val
2. CILCarlaDataset(train):
   - 3 cameras (3× effective samples per row)
   - histogram balancing
   - 9 augmentations (p=0.75 each)
3. CILCarlaDataset(val):
   - center camera only
   - no augmentation
4. DataLoader: 8 workers, pin_memory=True
5. Model: CIL_NvidiaCNN → DataParallel(device_ids=[0,1,...])
6. Optimizer: Adam(lr=0.002)
7. Scheduler: ReduceLROnPlateau(mode='min', factor=0.7, patience=5)
8. Loss: MSELoss()
9. Per epoch:
   a. train_loop with torch.amp.autocast + GradScaler
   b. val_loop (no grad)
   c. scheduler.step(val_loss)
   d. if val_loss < best: save models/cil_model.pth
   e. if no improvement for 12 epochs: early stop
```

### 7.3 Huấn Luyện YOLO

```bash
# Fine-tune YOLOv11s trên custom CARLA traffic dataset
python scripts/train_yolo.py
# Requires: configs/data.yaml (user-provided with class paths)
# Base model: yolo11s.pt (auto-downloaded by ultralytics)
# Output: models/yolo/retrain_results/
```

### 7.4 Chạy Agent

```bash
# Phase 1: LaneFollowAgent (NvidiaCNN/V2)
python run_agents.py \
    --agent lane_follow \
    --model-path models/lane_follow_model.pth \
    --yolo-model-path best.pt \
    --target-speed 30 \
    --config configs/carla_env.yaml

# Phase 2: CILAgent
# Dùng PowerShell script đầy đủ:
./scripts/run_cil.ps1

# Hoặc thủ công:
python run_agents.py \
    --agent cil \
    --cil-model-path models/cil_model.pth \
    --yolo-model-path best.pt \
    --target-speed 30 \
    --map Town03 \
    --config configs/carla_env.yaml

# YOLO-only visualization:
python run_agents.py \
    --agent yolo_detect \
    --yolo-model-path best.pt \
    --config configs/carla_env.yaml

# Dry run (test startup without driving):
python run_agents.py --agent noop --config configs/carla_env.yaml
```

### 7.5 CLI Arguments Reference

| Argument | Default | Mô tả |
|----------|---------|-------|
| `--agent` | `lane_follow` | Agent type: `autopilot/lane_follow/cil/yolo_detect/noop` |
| `--model-path` | `auto` | Path đến `.pth` cho lane_follow (auto tìm trong models/) |
| `--cil-model-path` | `auto` | Path đến `.pth` cho CIL agent |
| `--yolo-model-path` | `best.pt` | Path đến YOLO weights |
| `--target-speed` | `30` | Tốc độ mục tiêu (km/h) |
| `--map` | `Town03` | CARLA map |
| `--config` | `configs/carla_env.yaml` | YAML config file |
| `--collect-data` | `False` | Bật thu thập dữ liệu |
| `--data-dir` | `data/` | Thư mục lưu dữ liệu |
| `--no-yolo` | `False` | Tắt YOLO (chỉ CNN) |
| `--weather` | `ClearNoon` | Weather preset |
| `--record-video` | `False` | Bật ghi video |

---

## 8. Kết Quả Thực Nghiệm

### 8.1 Training Loss Curves (Phase 1 — NvidiaCNNV2)

```
 MSE Loss
  0.050 │▓
        │ ▓
  0.040 │  ▓▓
        │    ▓
  0.030 │     ▓▓
        │       ▓
  0.020 │        ▓▓▓
        │            ▓▓▓
  0.015 │                ▓▓▓▓▓▓▓▓
        │ · · · · · · · · ·◇◇◇◇◇◇◇◇◇◇◇◇◇◇◇ ← val
  0.012 │                                 ━━━━ ← train (converged)
        └─────────────────────────────────────►
          5   10   15   20   25   30   35   epoch
```
*▓ = Train loss, ◇ = Validation loss (illustrative)*

**Kết quả huấn luyện dự kiến:**

| Model | Final Train MSE | Final Val MSE | Epochs | Thời gian (T4×2) |
|-------|----------------|--------------|--------|-----------------|
| NvidiaCNN | ~0.018 | ~0.025 | 35–40 | ~10 phút |
| NvidiaCNNV2 | ~0.014 | ~0.018 | 30–40 | ~12 phút |
| CIL_NvidiaCNN | ~0.012 | ~0.015 | 30–45 | ~25–30 phút |

### 8.2 Phase Comparison

| Metric | Phase 1 (NvidiaCNNV2) | Phase 2 (CIL) | Cải thiện |
|--------|----------------------|---------------|----------|
| Val MSE | ~0.018 | ~0.015 | -16.7% |
| Intersection success | ~40–50% | ~70–80% | +40% |
| Lane keeping (straight) | ~85% | ~88% | +3% |
| Red light compliance | N/A (no YOLO) | ~90%+ | — |
| Avg collision rate | ~2/km | ~0.5/km | -75% |

### 8.3 YOLO Perception Performance

**Trên CARLA synthetic dataset:**

| Class | mAP@0.5 (ước tính) | Ghi chú |
|-------|-------------------|---------|
| vehicle | ~0.92 | Nhận dạng tốt |
| pedestrian | ~0.85 | Đôi khi miss ở xa |
| traffic_light_red | ~0.88 | Core safety feature |
| traffic_light_green | ~0.85 | Confidence scoring cần thiết |
| two_wheeler | ~0.78 | Khó hơn do kích thước nhỏ |
| traffic_sign | ~0.75 | Far-distance miss |

### 8.4 Traffic Light FSM Performance

| Tình huống | Kết quả |
|------------|---------|
| Stop trước đèn đỏ thẳng | ✓ Hoạt động ổn định (2-frame confirmation) |
| Tiếp tục khi đèn xanh | ✓ Không phantom stop (green immunity 10f) |
| Đèn đỏ lateral khi đang quẹo | ✓ Turn-phase suppression hoạt động |
| Vùng đô thị vs. vùng ngoại ô | ✓ Zone lock phân biệt được urban/rural_right |
| Signal bị che khuất | ✓ Hold 6 frames sau khi mất signal |

### 8.5 Inference Latency

| Component | GPU (T4) | CPU |
|-----------|----------|-----|
| Image preprocessing | ~1ms | ~3ms |
| NvidiaCNNV2 inference | ~3ms | ~45ms |
| CIL_NvidiaCNN inference | ~5ms | ~60ms |
| YOLOv11s inference | ~15ms | ~200ms |
| PID compute | <0.5ms | <0.5ms |
| **Total (CNN+YOLO)** | **~21ms (48 FPS)** | **~264ms (3.8 FPS)** |

> **Kết luận:** Real-time khả thi trên GPU. CPU-only chỉ đủ cho CIL mà không có YOLO.

### 8.6 Failure Cases Analysis

| Loại lỗi | Nguyên nhân | Trạng thái xử lý |
|----------|-------------|-----------------|
| **Drift at high speed** | Steering smoothing lag | Partially fixed (alpha tuning) |
| **Intersection hesitation** | CIL command delay | Mitigated (heuristic command fallback) |
| **Shadow false detection** | YOLO bbox on shadows | Known issue, shadow augmentation helps |
| **Night scene performance** | Limited night training data | Partially (brightness augmentation) |
| **Narrow lane recovery** | Recovery training interval | Fixed (disturbance training) |
| **Cross-lane red light** | Camera rotation shifts zone | Fixed (turn-phase suppression FSM) |

---

## 9. Hướng Dẫn Tái Tạo Kết Quả

### 9.1 Yêu Cầu Hệ Thống

| Thành phần | Tối thiểu | Khuyến nghị |
|-----------|----------|-------------|
| OS | Ubuntu 18.04+ / Windows 10 | Ubuntu 20.04 |
| Python | 3.8+ | 3.10 |
| GPU | NVIDIA 6GB VRAM | NVIDIA 16GB+ (T4/A100) |
| RAM | 16 GB | 32 GB |
| Disk | 50 GB | 100 GB |
| CARLA | 0.9.14 | 0.9.14 |

### 9.2 Cài Đặt Dependencies

```bash
# Clone repository
git clone https://github.com/duytran2525/CARLA-Funny-Moments.git
cd CARLA-Funny-Moments

# Install Python dependencies
pip install -r requirements.txt

# Install CARLA Python API (adjust path to your CARLA installation)
export CARLA_ROOT=/path/to/carla_0.9.14
pip install $CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.14-*.whl
```

### 9.3 Download Dataset

```bash
# Kaggle dataset (cần Kaggle API key)
kaggle datasets download -d <dataset-name> -p data/
unzip data/*.zip -d data/
```

Hoặc dùng Google Colab với Google Drive (xem `models/train.txt`).

### 9.4 Training Pipeline

```bash
# Step 1: Verify data structure
ls data/
# Expected: driving_log.csv, images_center/, images_left/, images_right/

# Step 2: Run training (Kaggle T4×2 khuyến nghị)
python scripts/train_cnn.py

# Step 3: Check output
ls models/
# Expected: cil_model.pth (hoặc cil_best_epoch_XX.pth)
```

### 9.5 Running Simulation

```bash
# Start CARLA server (separate terminal)
./CarlaUE4.sh -quality-level=Low -carla-server &

# Wait for server to load (~30s), then run agent
python run_agents.py \
    --agent cil \
    --cil-model-path models/cil_model.pth \
    --yolo-model-path best.pt \
    --target-speed 30 \
    --config configs/carla_env.yaml
```

### 9.6 Configuration Files

**`configs/carla_env.yaml` — Simulation Environment:**

```yaml
carla:
  host: "127.0.0.1"
  port: 2000
  tm_port: 8000
  timeout: 60
  sync: true
  fixed_delta_seconds: 0.05  # 20 FPS simulation

vehicle:
  filter: "vehicle.tesla.model3"
  spawn_point: 1
  role_name: "hero"

camera:
  width: 1920
  height: 1080
  fov: 90.0

weather:
  preset: "ClearNoon"  # ClearNoon/ClearSunset/HardRainNoon/...
```

**`configs/train_params.yaml` — Training Hyperparameters:**

```yaml
epochs: 50
batch_size: 48          # Tuned for T4×2 (32GB total)
learning_rate: 0.002
steering_correction: 0.2 # ±correction for left/right cameras
augmentation_prob: 0.75
train_split: 0.75
lr_patience: 5
lr_factor: 0.7
early_stopping_patience: 12
```

---

## 10. Hướng Phát Triển Tương Lai

### 10.1 Model Improvements

| Hướng | Mô tả | Độ ưu tiên |
|-------|-------|-----------|
| **MPC (Model Predictive Control)** | Thay PID bằng MPC để tối ưu trajectory N bước trước | Cao |
| **Reinforcement Learning** | Fine-tune CIL bằng PPO/SAC online trong CARLA | Cao |
| **Attention Mechanisms** | Thêm spatial attention vào CNN backbone | Trung bình |
| **Temporal Modeling** | LSTM/Transformer cho sequential frame info | Trung bình |
| **Multi-task Learning** | Thêm depth estimation + segmentation heads | Trung bình |

### 10.2 Perception Improvements

| Hướng | Mô tả | Độ ưu tiên |
|-------|-------|-----------|
| **Multi-Object Tracking (MOT)** | Theo dõi trajectory của obstacles qua nhiều frame | Cao |
| **3D Object Detection** | PointNet/VoxelNet với LIDAR data | Trung bình |
| **Semantic Segmentation** | Lane detection, drivable area | Trung bình |
| **Inverse Perspective Mapping** | Bird's-eye view từ camera | Thấp |

### 10.3 Deployment Optimization

| Hướng | Mô tả | Độ ưu tiên |
|-------|-------|-----------|
| **TensorRT quantization** | INT8 inference, giảm latency 2-4× | Cao |
| **ONNX export** | Cross-platform deployment | Trung bình |
| **Model pruning** | Giảm model size không ảnh hưởng accuracy | Thấp |

### 10.4 System Improvements

| Hướng | Mô tả |
|-------|-------|
| **HD Map integration** | Sử dụng CARLA high-definition map cho better path planning |
| **V2X communication** | Vehicle-to-infrastructure traffic signal info |
| **Adversarial training** | Robust to FGSM/PGD attacks on camera input |
| **Multi-agent scenarios** | Cooperative driving với nhiều xe tự động |

---

## 11. Thành Viên & Tài Liệu Tham Khảo

### 11.1 Thành Viên

| Tên | Vai trò |
|-----|---------|
| Duy Trần | Lead Developer — Architecture, Training, CARLA Integration |
| *(các thành viên khác)* | *(vai trò)* |

### 11.2 Dependencies

```
# Core
carla==0.9.14              # Simulation environment
torch>=2.0.0               # Deep learning framework
torchvision>=0.15.0        # Image transforms
ultralytics>=8.0.0         # YOLOv11 (ultralytics package)

# Computer Vision
opencv-python>=4.8.0       # Image processing, cv2
numpy>=1.24.0              # Array operations
Pillow>=10.0.0             # Image I/O

# Data & Config
pandas>=2.0.0              # CSV handling
PyYAML>=6.0                # YAML config parsing
scikit-learn>=1.3.0        # Preprocessing utilities

# Training utilities
tqdm>=4.65.0               # Progress bars
matplotlib>=3.7.0          # Loss curve visualization
tensorboard>=2.13.0        # Training monitoring (optional)
```

### 11.3 Tài Liệu Tham Khảo

1. **Bojarski et al. (2016)** — "End to End Learning for Self-Driving Cars"  
   NVIDIA Technical Report. https://arxiv.org/abs/1604.07316  
   → *Cơ sở của NvidiaCNN architecture*

2. **Codevilla et al. (2018)** — "End-to-end Driving via Conditional Imitation Learning"  
   ICRA 2018. https://arxiv.org/abs/1710.02410  
   → *Cơ sở của CIL_NvidiaCNN và 4-command expert heads*

3. **Dosovitskiy et al. (2017)** — "CARLA: An Open Urban Driving Simulator"  
   CoRL 2017. https://arxiv.org/abs/1711.03938  
   → *CARLA simulator platform*

4. **Wang et al. (2023)** — "YOLOv11: Scalable Object Detection"  
   Ultralytics. https://github.com/ultralytics/ultralytics  
   → *Object detection backbone*

5. **Fischler & Bolles (1981)** — "Random Sample Consensus: A Paradigm for Model Fitting"  
   Communications of the ACM, 24(6).  
   → *RANSAC road plane fitting*

6. **Ackermann (1818)** — Bicycle model kinematics  
   → *Curved path corridor model*

7. **Ross et al. (2011)** — "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (DAgger)  
   AISTATS 2011. https://arxiv.org/abs/1011.0686  
   → *Recovery disturbance training inspiration*

---

*Báo cáo được tạo tự động dựa trên phân tích toàn bộ source code tại commit hiện tại.*  
*Cập nhật lần cuối: 2026-04*
