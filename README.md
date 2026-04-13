# CARLA-Funny-Moments 🚗

Autonomous Driving Agent trên CARLA Simulator — kết hợp Behavioral Cloning, Conditional Imitation Learning, YOLOv11 perception và PID control.

## Trạng Thái Dự Án

| Phase | Mô tả | Trạng thái |
|-------|-------|-----------|
| **Phase 1** | NvidiaCNNV2 — Behavioral Cloning (ảnh → tay lái) | ✅ Hoàn chỉnh |
| **Phase 2** | CIL_NvidiaCNN — Conditional Imitation Learning (4 expert heads) | ✅ Hoàn chỉnh |
| **Perception** | YOLOv11 + Depth + Traffic Light FSM + RANSAC road plane | ✅ Hoàn chỉnh |
| **Control** | PID dual-mode với deadband, anti-windup, integral decay | ✅ Hoàn chỉnh |
| **Phase 3** | RL fine-tuning / MPC | 🔜 Tương lai |

## Báo Cáo Đầy Đủ

📄 **[REPORT.md](./REPORT.md)** — Báo cáo kỹ thuật hoàn chỉnh bao gồm:
- Lý do thực hiện đề tài
- Mô tả bài toán (Input/Output/Diagrams)
- Mô tả tập dữ liệu (~66.6k mẫu)
- Độ đo đánh giá
- Kiến trúc hệ thống chi tiết (ASCII diagrams)
- Quy trình training & inference
- Kết quả thực nghiệm
- Hướng dẫn tái tạo kết quả

## Cấu Trúc Dự Án

```
CARLA-Funny-Moments/
├── core_perception/         # Nhận thức môi trường
│   ├── cnn_model.py         # NvidiaCNN / NvidiaCNNV2 / CIL_NvidiaCNN
│   ├── dataset.py           # PyTorch Dataset + 9 augmentation types
│   └── yolo_detector.py     # YOLOv11 + depth + FSM đèn giao thông
├── core_control/            # Điều khiển phương tiện
│   ├── carla_manager.py     # CARLA world & actor lifecycle
│   ├── pid_manager.py       # Dual-mode PID speed controller
│   ├── collect_data.py      # 3-camera synchronized recorder
│   └── sync_data.py         # Frame synchronization helper
├── scripts/
│   ├── train_cnn.py         # Multi-GPU CIL training script
│   └── train_yolo.py        # YOLO fine-tuning script
├── configs/
│   ├── carla_env.yaml       # CARLA simulation config
│   └── train_params.yaml    # Training hyperparameters
├── run_agents.py            # Main entry point (5 agent modes)
├── REPORT.md                # Báo cáo kỹ thuật đầy đủ
└── requirements.txt         # Python dependencies
```

## Quick Start

```bash
pip install -r requirements.txt

# Thu thập dữ liệu
python run_agents.py --agent autopilot --collect-data --data-dir data/

# Huấn luyện
python scripts/train_cnn.py

# Chạy agent CIL
python run_agents.py --agent cil --cil-model-path models/cil_model.pth --yolo-model-path best.pt
```

## Tài Liệu Tham Khảo Chính

- Bojarski et al. (2016) — NVIDIA End-to-End Learning
- Codevilla et al. (2018) — Conditional Imitation Learning (ICRA)
- Dosovitskiy et al. (2017) — CARLA Simulator
