# 🚗 Đồ án Xe tự lái CARLA & Dự báo Quỹ đạo Đa tác nhân (GTNet)

[![CARLA Version](https://img.shields.io/badge/CARLA-0.9.13%20%7C%200.9.15-blue.svg)](https://carla.org/)
[![Python Version](https://img.shields.io/badge/Python-3.8%20%7C%203.10-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](#)

Repository này chứa toàn bộ mã nguồn của đồ án nghiên cứu hệ thống tự hành CARLA, tích hợp mô hình dự báo quỹ đạo đa phương tiện chủ động (GTNet) cùng các giải pháp nhận dạng vật thể và điều khiển bám làn.

---

## 📌 Các Luồng Xử Lý Chính (Pipeline Agent Modes)

Hệ thống hỗ trợ 5 chế độ tác nhân điều khiển chính thông qua cờ `--agent` khi khởi chạy:

| Chế độ (`--agent`) | Cơ chế hoạt động | Module Perception | Cấu trúc Supervisor |
| :--- | :--- | :--- | :--- |
| **`lane_follow`** | Bám làn tự động sử dụng mạng CNN dự đoán góc lái. | Camera RGB + CNN | Không có (hoặc YOLO tùy chọn) |
| **`cil`** | Điều khiển theo lộ trình (Conditional Imitation Learning) bám waypoint. | Camera RGB + CIL CNN | GTNet Supervisor (Chủ động - Tùy chọn) |
| **`cil_yolo`** | Kết hợp điều khiển bám waypoint CIL và giám sát vật cản YOLO. | RGB + CIL + YOLO | YOLO + GTNet Supervisor (Brake Fusion) |
| **`yolo_detect`** | Chạy chế độ demo để hiển thị khả năng phát hiện vật thể thời gian thực. | YOLOv8 Detector | Traffic Supervisor (Phản động) |
| **`autopilot`** | Điều khiển xe sử dụng bộ điều hướng tích hợp mặc định của CARLA. | CARLA API | Không có |

---

## 📂 Sơ đồ Cấu trúc Repo

```text
├── configs/
│   ├── carla_env.yaml          # File cấu hình môi trường CARLA và tham số GTNet Supervisor
│   └── data.yaml               # Cấu hình dataset huấn luyện YOLO
├── core_control/
│   ├── gtnet_supervisor.py     # Bộ giám sát an toàn chủ động dựa trên quỹ đạo dự báo
│   ├── traffic_supervisor.py   # Bộ giám sát an toàn phản động (YOLO, đèn đỏ, vật cản)
│   ├── supervisor.py           # Quản lý supervisor cơ sở và arbitration
│   └── pid_controller.py       # Bộ điều khiển PID dọc và ngang
├── core_perception/
│   ├── multi_agent_model.py    # Định nghĩa kiến trúc mạng GTNet (GAT + GRU + WTA Loss)
│   ├── multi_agent_trajectory.py # Quản lý cửa sổ quỹ đạo và xây dựng đồ thị thích ứng
│   ├── multi_agent_dataset.py  # Bộ nạp dataset PyTorch dạng đồ thị
│   ├── dataset.py              # Bộ nạp dữ liệu cho mô hình CIL/Waypoint
│   └── yolo_detector.py        # Wrapper YOLOv8 phục vụ tracking và nhận dạng
├── scripts/
│   ├── build_multi_agent_dataset.py # Tiền xử lý dữ liệu raw CSV thành tensor mẫu đồ thị .pt
│   ├── fix_manifest_paths.py   # Chuẩn hóa đường dẫn manifest Windows/Linux
│   ├── test_dataset_loading.py # Kiểm tra tính toàn vẹn của dataset mẫu đồ thị
│   ├── kaggle_train_gtnet.py   # Script training tối ưu hóa cho Kaggle và Local
│   ├── validate_target_metrics.py # Script kiểm định metrics tiêu chuẩn của mô hình
│   ├── train_cnn.py            # Huấn luyện mô hình bám làn CIL
│   ├── train_yolo.py           # Huấn luyện mô hình YOLO
│   └── convert_yolo_to_engine.py # Biên dịch model YOLO sang TensorRT engine
├── tests/                      # Bộ unit tests kiểm thử độ ổn định
└── run_agents.py               # Script chạy chính của toàn bộ repo
```

---

## 🛠️ Cài đặt Môi trường

> [!IMPORTANT]
> **Yêu cầu hệ thống:**
> - Máy tính cài đặt card đồ họa NVIDIA (khuyên dùng RTX 3060 trở lên để chạy giả lập kèm YOLO/GTNet thời gian thực).
> - Phiên bản giả lập CARLA: **0.9.13** hoặc **0.9.15**.

### 1. Cấu hình Giả lập CARLA
CARLA Server phải được cài đặt riêng biệt. Bạn cần expose CARLA API cho Python bằng một trong hai cách:
1. Thêm biến môi trường hệ thống:
   - `CARLA_ROOT` trỏ tới thư mục gốc cài đặt CARLA.
   - `CARLA_PYTHONAPI` trỏ tới `<CARLA_ROOT>/PythonAPI/carla`.
2. Hoặc sao chép thư mục `PythonAPI/` đặt ngay cạnh thư mục gốc của repo này.

### 2. Cài đặt các thư viện Python
Sử dụng môi trường ảo (virtualenv hoặc conda) và cài đặt thông qua `pip`:
```powershell
pip install -r requirements.txt
```

---

## 🚀 Hướng dẫn Chạy Runtime

Cấu hình chi tiết nằm trong `configs/carla_env.yaml`. Bạn có thể khởi chạy tác nhân bằng CLI:

### 1. Khởi động CARLA Server
```powershell
# Chạy CARLA ở chất lượng đồ họa thấp và tắt hiển thị để tăng FPS (khuyên dùng khi thu thập dữ liệu/training)
.\CarlaUE4.exe -quality-level=Low -RenderOffScreen
```

### 2. Khởi chạy Tác nhân trên CARLA
```powershell
# Chạy bám làn CNN cơ bản
python run_agents.py --agent lane_follow

# Chạy CIL (Waypoint) cơ bản
python run_agents.py --agent cil --cil-model-path models/waypoint_predictor.pth

# Chạy CIL kết hợp YOLO + GTNet Supervisor (Bản đầy đủ an toàn nhất)
python run_agents.py --agent cil_yolo --cil-model-path models/waypoint_predictor.pth --yolo-model-path models/yolo26m_best.engine --enable-gtnet --gtnet-model-path models/gtnet_full.pt
```

<details>
<summary><b>⚙️ Click để xem toàn bộ các cờ CLI nâng cao của GTNet Supervisor</b></summary>

- `--enable-gtnet` / `--disable-gtnet`: Bật/tắt bộ giám sát quỹ đạo GTNet (Mặc định: tắt).
- `--gtnet-model-path <path>`: Đường dẫn tới file trọng số mô hình GTNet (`.pt`). Mặc định: `models/gtnet_full.pt`.
- `--gtnet-every-n-ticks <int>`: Tần suất chạy inference GTNet (mặc định: 2 ticks = chạy ở 10Hz khi giả lập ở 20Hz).
- `--gtnet-history-frames <int>`: Số frame lịch sử đưa vào mô hình (mặc định: 40 frames = 2.0 giây).
- `--gtnet-adjacency-mode <fixed|adaptive|checkpoint>`: Chế độ xác định đồ thị tương tác (mặc định: `checkpoint` - tự động nạp từ config lưu trong model).
- `--gtnet-fixed-adj-radius <float>`: Bán kính cố định của đồ thị kết nối khi chạy chế độ `fixed`.
- `--gtnet-max-actor-distance <float>`: Khoảng cách tối đa từ Ego tới NPC để đưa vào tính toán đồ thị (mặc định: 100.0m).
- `--gtnet-draw-debug` / `--no-gtnet-draw-debug`: Bật/tắt chế độ vẽ quỹ đạo dự đoán trực quan của tất cả NPC xung quanh lên màn hình CARLA.
</details>

---

## 🧠 Kiến trúc Mô hình GTNet Cải tiến

Kiến trúc **GTNet (Graph Trajectory Network)** cải tiến phối hợp mạng đồ thị chú ý và cấu trúc mạng học sâu tuần tự để giải quyết các tương tác giao thông phức tạp:

### 🌟 5 Cải tiến đột phá của mô hình:
1. **Edge-aware GAT (Mạng chú ý đồ thị học cạnh):** Thay thế cơ chế gộp thông tin social pooling đồng nhất bằng Graph Attention Network (GAT) 4 đầu. Mô hình tích hợp bộ mã hóa cạnh tương đối (`RelativeEdgeEncoder` mặc định `gat_edge_dim = 32`) để tự động học tầm quan trọng của các tác nhân xung quanh dựa trên khoảng cách hình học thực tế.
2. **Dự báo Đa mẫu (Multimodal Prediction):** Dự đoán đồng thời $K = 5$ quỹ đạo tương lai kèm phân phối xác suất tương ứng. Sử dụng hàm mất mát Winner-Takes-All (WTA) để chỉ cập nhật trọng số cho chế độ dự báo khớp với nhãn thực tế nhất, tránh lỗi trung bình hóa quỹ đạo (mean regression) tại các giao lộ.
3. **Bán kính tương tác thích ứng động (Adaptive Radius):** Tính toán bán kính liên kết đồ thị thích ứng tuyến tính theo vận tốc hiện tại $v_i$ của từng phương tiện:
   $$R_i = R_{\text{base}} + \alpha \cdot v_i$$
   *Tham số mặc định:* Bán kính cơ sở $R_{\text{base}} = 40.0\,\text{m}$, hệ số tỉ lệ $\alpha = 1.0$.
4. **Kỹ thuật ổn định hóa học máy (GAT Stability Fixes):** Tách biệt học danh mục giữa backbone và tầng chú ý bằng Dual Learning Rate (`gat_lr_scale = 0.5`). Đóng băng GAT trong 5 epoch đầu (`--gat-freeze-epochs 5`) và áp dụng cơ chế gradient clipping phân nhóm để triệt tiêu hiện tượng bùng nổ gradient.
5. **Tối ưu hóa huấn luyện:** Bộ lập lịch `WarmupCosineScheduler` (5 epoch tăng tiến LR, giảm dần theo Cosine). Tăng cường dữ liệu bằng Ego-rotation ngẫu nhiên ($\pm 10^{\circ}$) và History Frame Dropout ($10\%$). Tích hợp hàm hao hụt đa dạng hóa quỹ đạo (Diversity Loss với trọng số `0.08` tăng dần trong 15 epoch) để đẩy các mode đi xa nhau và chặn hiện tượng sụp đổ chế độ (mode collapse).

---

## 🛡️ Ứng dụng An toàn: Bộ giám sát GTNet Supervisor

GTNet Supervisor hoạt động song hành cùng YOLO Traffic Supervisor theo nguyên tắc kiểm soát phân cấp và phối hợp điều khiển tối cao (Control Arbitration):

- **Hành lang an toàn (Danger Zone):** Được định nghĩa động theo kích thước của xe Ego và khoảng cách phanh khẩn cấp.
- **Thuật toán Đồng thuận chế độ (Mode Consensus):** Do mô hình dự báo $K=5$ quỹ đạo khác nhau, bộ supervisor sẽ đếm số lượng mode của NPC cắt vào hành lang an toàn của Ego. Xe Ego chỉ kích hoạt phanh khẩn cấp khi có **tối thiểu 2 chế độ dự báo ($T \ge 2$)** cùng báo nguy hiểm. Thuật toán này giúp loại bỏ phanh ảo (false brakes) khi có 1 mode phụ rẽ nhánh cực đoan.
- **Brake Fusion (Control Arbitration):** Hợp nhất lệnh phanh của YOLO (phản động nhanh khi có vật cản sát sườn hoặc đèn đỏ) và GTNet (chủ động giảm tốc sớm khi xe phía trước chuyển làn cắt mặt) theo nguyên tắc lấy cực đại:
  $$\text{Brake}_{\text{Final}} = \max\left(\text{Brake}_{\text{Traffic\_Supervisor}},\; \text{Brake}_{\text{GTNet\_Supervisor}}\right)$$
- **Throttle Floor (Bảo vệ đâm đuôi):** Nếu phát hiện phương tiện bám đuôi phía sau có nguy cơ đâm vào Ego rất cao, bộ supervisor sẽ hạn chế cường độ phanh gấp và duy trì một mức ga tối thiểu để giữ khoảng cách an toàn (nếu khoảng trống phía trước xe cho phép).

---

## ⚙️ Quy trình Huấn luyện & Đánh giá GTNet

Quá trình huấn luyện đầy đủ từ dữ liệu thô (Raw CSV) đến mô hình kiểm định cuối cùng bao gồm các bước sau:

### Bước 1: Thu thập dữ liệu thô (Raw Data Collection)
Chạy giả lập CARLA Server trên máy local, sau đó thu thập dữ liệu tự động cho tất cả các Town bằng script PowerShell:
```powershell
# Chạy thu thập tự động 8 Town với bán kính thích ứng động
.\scripts\collect_all_towns_adaptive.ps1
```
Hoặc chạy thủ công trên một Town cụ thể:
```powershell
python collect_multi_agent_data.py --town Town01 --duration 500 --npc-vehicles 40
```

### Bước 2: Xây dựng Dataset mẫu đồ thị
Tiền xử lý các file CSV thô thành các tensor đồ thị đã căn chỉnh hệ trục tọa độ Ego-centric với độ dài 40 frames lịch sử và 60 frames tương lai:
```powershell
python scripts/build_multi_agent_dataset.py `
    --raw-csv data/multi_agent/raw/*.csv `
    --out-dir data/multi_agent/processed `
    --history-frames 40 `
    --future-frames 60 `
    --adaptive-radius `
    --radius-base 40.0 `
    --radius-alpha 1.0
```

> [!TIP]
> **Tương thích chéo hệ điều hành (Windows -> Linux/Kaggle):**
> Nếu bạn xây dựng dataset trên Windows nhưng muốn upload lên Kaggle Docker để train GPU miễn phí, hãy chạy script chuẩn hóa dấu gạch chéo ngược thành dấu gạch chéo xuôi trên manifest trước khi nén zip:
> ```powershell
> python scripts/fix_manifest_paths.py --manifest data/multi_agent/processed/manifest.csv --backup
> ```
> Kiểm tra tính toàn vẹn của dataset:
> ```powershell
> python scripts/test_dataset_loading.py --dataset-dir data/multi_agent/processed
> ```

### Bước 3: Huấn luyện mô hình (Training)
Chạy script `scripts/kaggle_train_gtnet.py` ở chế độ `full` để huấn luyện mô hình **GTNet Full** với tất cả cải tiến:
```bash
python scripts/kaggle_train_gtnet.py \
  --data-dir data/multi_agent/processed \
  --out-dir models/gtnet_full \
  --mode full \
  --epochs 80 \
  --batch-size 32 \
  --accum-steps 2 \
  --hidden-dim 384 \
  --num-modes 5 \
  --num-attention-heads 4 \
  --graph-layers 4 \
  --dropout 0.15 \
  --learning-rate 2e-4 \
  --weight-decay 1e-4 \
  --grad-clip 0.8 \
  --gat-lr-scale 0.5 \
  --gat-per-group-clip \
  --gat-freeze-epochs 5 \
  --radius-base 40.0 \
  --radius-alpha 1.0 \
  --encoder-dropout 0.15 \
  --early-stopping-patience 15 \
  --cosine-lr \
  --warmup-epochs 5 \
  --diversity-weight 0.08 \
  --diversity-ramp-epochs 15 \
  --early-stop-metric ade \
  --augment
```

#### Chạy Ablation Study (Nghiên cứu cấu trúc):
Huấn luyện lần lượt 8 biến thể (từ `000` đến `111`) để đo đạc sự đóng góp của từng thành phần độc lập:
```bash
python scripts/kaggle_train_gtnet.py \
  --data-dir data/multi_agent/processed \
  --out-dir models/ablation \
  --mode ablation \
  --epochs 50 \
  --batch-size 48 \
  --cosine-lr
```

### Bước 4: Kiểm định chỉ số (Validation)
Đánh giá mô hình đã lưu trên tập validation và so sánh với các điều kiện ràng buộc kỹ thuật của đồ án:
```powershell
python scripts/validate_target_metrics.py \
    --checkpoint models/gtnet_full/best.pt \
    --data-dir data/multi_agent/processed \
    --out-dir validation_results
```

---

## 📈 Đánh giá Hiệu năng (Performance Results)

Kết quả đánh giá mô hình được huấn luyện đầy đủ 80 epochs so sánh trực tiếp với Baseline (mô hình gốc) trên tập dữ liệu thử nghiệm độc lập:

| Chỉ số đánh giá (Metric) | Baseline (GCN gốc) | GTNet Full (Cải tiến) | Ngưỡng yêu cầu đồ án | Kết quả kiểm định |
| :--- | :---: | :---: | :---: | :---: |
| **minADE** (độ lệch trung bình) | 1.840 m | **0.441 m** | < 1.500 m | **ĐẠT (Vượt 70%)** |
| **minFDE** (độ lệch điểm cuối) | 4.120 m | **1.201 m** | < 2.700 m | **ĐẠT (Vượt 55%)** |
| **Miss Rate** (tỉ lệ trượt) | 38.00% | **16.07%** | < 20.00% | **ĐẠT** |
| **Inference Latency** (độ trễ) | 15.2 ms | **11.06 ms** | < 25.0 ms | **ĐẠT (Thời gian thực)** |

> [!NOTE]
> Nhờ áp dụng **Bán kính tương tác thích ứng** động, mô hình loại bỏ được các liên kết dư thừa ở dải tốc độ chậm, giúp giảm khối lượng tính toán đồ thị, đưa tốc độ xử lý trung bình về mức **11.06 ms/mẫu** (tương đương 90 FPS), hoàn toàn đáp ứng thời gian thực cho phần cứng nhúng trên xe tự lái.

---

## ⚙️ Huấn luyện Mô hình CIL & YOLO

### 1. Huấn luyện bám làn CIL
Script `scripts/train_cnn.py` hỗ trợ huấn luyện mạng dự đoán waypoint góc lái:
```powershell
python scripts/train_cnn.py
```
Checkpoint mặc định sẽ được ghi nhận tại `models/waypoint_predictor.pth`.

### 2. Huấn luyện YOLO
Cấu hình tập tin dataset tại `configs/data.yaml` hướng tới các thư mục ảnh đã gán nhãn thực tế. Chạy huấn luyện và convert TensorRT:
```powershell
# Chạy huấn luyện YOLO
python scripts/train_yolo.py --data configs/data.yaml

# Chuyển đổi trọng số sang TensorRT chạy tối ưu trên GPU
python scripts/convert_yolo_to_engine.py yolo26m_best.pt --fp16
```

---

## 🧪 Kiểm tra nhanh & Chạy Tests

Trước khi đẩy bất kỳ mã nguồn chỉnh sửa nào lên nhánh chính, hãy chắc chắn chạy kiểm tra tính ổn định hồi quy:

```powershell
# Chạy bộ unit tests kiểm thử model, dataset và GAT
python -m unittest discover -s tests -q

# Kiểm tra tính biên dịch hoàn thiện của code python
python -m compileall run_agents.py core_control core_perception scripts tests utils
```

---

## ❌ Xử lý Sự cố & Khắc phục lỗi

### 1. Lỗi tràn bộ nhớ VRAM (CUDA Out of Memory)
- **Giải pháp:** Giảm kích thước batch chẵn xuống nửa và bật tích lũy gradient tương ứng để duy trì hướng học ổn định:
  `--batch-size 16 --accum-steps 4`

### 2. Mô hình báo loss/metric NaN khi train
- **Giải pháp:** Đảm bảo `--gat-freeze-epochs` tối thiểu bằng `5` và đặt tốc độ học nhỏ `--learning-rate 1e-4`. Phiên bản v3 đã tích hợp bộ dọn dẹp bộ tích lũy gradient khi phát hiện NaN (`[FIX-1]`), giúp chương trình huấn luyện chạy xuyên suốt không bị crash giữa chừng.

### 3. File manifest.csv không tìm thấy hoặc sai đường dẫn trên Kaggle
- **Nguyên nhân:** Do tạo dataset trên hệ điều hành Windows sử dụng dấu gạch chéo ngược (`\`), khi đưa lên Linux/Kaggle Docker bị lỗi phân tách thư mục.
- **Giải pháp:** Chạy script chuẩn hóa dấu dẫn chéo trước khi nén zip dataset:
  `python scripts/fix_manifest_paths.py --manifest data/multi_agent/processed/manifest.csv`

---

## ⚠️ Lưu ý chung

- Các tệp tin mô hình lớn (`.pt`, `.engine`) nên được cấu hình thông qua Git LFS hoặc lưu trữ đám mây ngoài để tối ưu dung lượng của git repo.
- Đảm bảo các checkpoint đã được đặt đúng trong thư mục `models/` trước khi chạy các tác nhân tự hành trên CARLA.
