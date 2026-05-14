# CARLA-Funny-Moments

Đây là repo đồ án CARLA gồm 3 luồng chính:

- `lane_follow`: bám làn bằng CNN steering, có thể gắn thêm YOLO để giám sát an toàn.
- `cil`: dự đoán waypoint theo chuỗi frame và lệnh điều hướng.
- `yolo_detect`: demo perception YOLO kết hợp `TrafficSupervisor`.

## Cấu trúc repo

- `run_agents.py`: file chạy chính cho các agent trong CARLA.
- `core_control/`: điều khiển xe, PID, route planner, data collector, quản lý session CARLA.
- `core_perception/`: mô hình CNN, YOLO detector, object tracker, dataset loader.
- `scripts/train_cnn.py`: train mô hình waypoint cho CIL.
- `scripts/train_yolo.py`: train mô hình YOLO.
- `scripts/convert_yolo_to_engine.py`: convert YOLO từ `.pt` sang `.onnx` hoặc `.engine`.
- `configs/`: cấu hình môi trường CARLA, train và dataset YOLO.
- `tests/`: test hồi quy cho navigation, dataset compat và model compat.

## Yêu cầu môi trường

Các package Python nằm trong `requirements.txt`.

CARLA nên được cài riêng và expose bằng một trong các cách sau:

- biến môi trường `CARLA_ROOT`
- biến môi trường `CARLA_PYTHONAPI`
- thư mục `PythonAPI/` đặt gần repo

## Cài package

```powershell
pip install -r requirements.txt
```

## Chạy runtime

File cấu hình mặc định là `configs/carla_env.yaml`.

Ví dụ:

```powershell
python run_agents.py --agent lane_follow
python run_agents.py --agent cil --cil-model-path models/waypoint_predictor.pth
python run_agents.py --agent cil_yolo --cil-model-path models/waypoint_predictor.pth --yolo-model-path models/yolo26m_best.engine
python run_agents.py --agent yolo_detect --yolo-model-path models/yolo26m_best.engine
```

## Train mô hình CIL / Waypoint

Script `scripts/train_cnn.py` hỗ trợ 2 kiểu dữ liệu:

- một thư mục dataset duy nhất chứa `driving_log.csv` và `images_center/images_left/images_right`
- một thư mục gốc chứa nhiều dataset con như `Town03/driving_log.csv`, `Town05/driving_log.csv`

Checkpoint đầu ra mặc định:

```text
models/waypoint_predictor.pth
```

Chạy train:

```powershell
python scripts/train_cnn.py
```

Một số điểm đã được vá trong pipeline train:

- hỗ trợ cả schema CSV cũ và schema mới do `DataCollector` sinh ra
- chặn train khi dataset rỗng hoặc ảnh bị thiếu quá nhiều
- split train/val có seed để ổn định hơn

## Train YOLO

Template cấu hình dataset nằm ở `configs/data.yaml`.

Ví dụ:

```powershell
python scripts/train_yolo.py --data configs/data.yaml
python scripts/convert_yolo_to_engine.py yolo26m_best.pt --fp16
```

Bạn cần chỉnh lại `path`, `train`, `val` trong `configs/data.yaml` cho đúng dataset thực tế.

## Ghi chú kỹ thuật

- Runtime hiện phân loại checkpoint theo `state_dict`, không còn load nhầm kiến trúc chỉ vì tên file.
- `core_perception/dataset.py` hiện hỗ trợ cả layout CSV cũ lẫn layout mới của `DataCollector`.
- CIL hiện dùng history 3 frame thật, không còn lặp frame cuối.
- Mặc định trong `configs/carla_env.yaml`, HUD và telemetry CSV của CIL đã được tắt để chạy nhẹ hơn.

## Kiểm tra nhanh

```powershell
python -m unittest discover -s tests -q
python -m compileall run_agents.py core_control core_perception scripts tests utils
```

## Lưu ý

- Repo hiện có nhiều file model lớn; nên dùng Git LFS hoặc lưu trên release/storage riêng nếu làm việc nhóm.
- Nếu chạy `cil`, bạn cần có checkpoint waypoint tương thích, ví dụ `models/waypoint_predictor.pth`.
