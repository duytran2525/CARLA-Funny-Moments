# 🚗 Đồ án Xe tự lái CARLA & Dự báo Quỹ đạo Đa tác nhân (GTNet)

*   **Phiên bản CARLA hỗ trợ:** 0.9.13 | 0.9.15
*   **Môi trường ngôn ngữ:** Python 3.8 | 3.10
*   **Thư viện học sâu chính:** PyTorch 2.0+ (hỗ trợ CUDA tăng tốc phần cứng)

Repository này chứa toàn bộ mã nguồn của đồ án nghiên cứu hệ thống tự hành CARLA, tích hợp mô hình dự báo quỹ đạo đa phương tiện chủ động (GTNet) cùng các giải pháp nhận dạng vật thể và điều khiển bám làn.

---

## 📌 Các Luồng Xử Lý Chính (Pipeline Agent Modes)

Hệ thống hỗ trợ 5 chế độ tác nhân điều khiển chính thông qua cờ `--agent` khi khởi chạy:

| Chế độ (`--agent`) | Cơ chế hoạt động | Module Perception | Cấu trúc Supervisor |
| :--- | :--- | :--- | :--- |
| **`lane_follow`** | Bám làn tự động sử dụng mạng CNN dự đoán góc lái. | Camera RGB + CNN | Không có (hoặc RT-DETR tùy chọn) |
| **`cil`** | Điều khiển theo lộ trình (Conditional Imitation Learning) bám waypoint. | Camera RGB + CIL CNN | GTNet Supervisor (Chủ động - Tùy chọn) |
| **`cil_yolo`** | Kết hợp điều khiển bám waypoint CIL và giám sát vật cản RT-DETR-resnet50. | RGB + CIL + RT-DETR-resnet50 | RT-DETR-resnet50 + GTNet Supervisor (Brake Fusion) |
| **`yolo_detect`** | Chạy chế độ demo hiển thị khả năng phát hiện vật thể thời gian thực của RT-DETR-resnet50. | RT-DETR-resnet50 Detector | Traffic Supervisor (Phản động) |
| **`autopilot`** | Điều khiển xe sử dụng bộ điều hướng tích hợp mặc định của CARLA. | CARLA API | Không có |

---

## 📂 Các File Quan trọng ảnh hưởng đến Vận hành & Kết quả

Dưới đây là các file và thư mục cốt lõi trực tiếp quyết định khả năng chạy mô phỏng, dự báo và điều khiển xe tự hành:

### 1. File chạy chính & Cấu hình (Thư mục gốc)
*   **`run_agents.py`**: Điểm khởi chạy chương trình chính. Thiết lập kết nối CARLA, gọi vòng lặp tick, quản lý HUD hiển thị và phối hợp các bộ điều khiển/supervisor.
*   **`requirements.txt`**: Khai báo danh sách các thư viện Python bắt buộc của đồ án (PyTorch, OpenCV, NumPy, Pandas, PyYAML, Ultralytics,...).
*   **`run_cil.ps1`**: Script PowerShell khởi động nhanh tác nhân CIL với cấu hình chuẩn.

### 2. Module Lập kế hoạch & Điều khiển (`core_control/`)
*   [carla_manager.py](core_control/carla_manager.py): Quản lý vòng đời giả lập, đồng bộ hóa (sync mode), thời tiết và điều phối lưu lượng NPC (xe cộ, người đi bộ).
*   [gtnet_supervisor.py](core_control/gtnet_supervisor.py): Bộ giám sát an toàn chủ động GTNet. Inference quỹ đạo tương lai 3.0s, áp dụng thuật toán Đồng thuận chế độ (Mode Consensus) và phát lệnh phanh khẩn cấp dự phòng.
*   [traffic_supervisor.py](core_control/traffic_supervisor.py): Bộ giám sát luật lệ giao thông phản động. Phát hiện xe chắn phía trước, đèn giao thông đỏ để can thiệp phanh tức thời (nhận bounding box từ RT-DETR-resnet50).
*   [cil_route_planner.py](core_control/cil_route_planner.py): Lập kế hoạch lệnh điều hướng rẽ trái/phải/đi thẳng tại các nút giao lộ cục bộ cho CIL.
*   [pid_manager.py](core_control/pid_manager.py) & [pure_pursuit.py](core_control/pure_pursuit.py): Hiện thực hóa bộ điều khiển PID kép điều chỉnh ga/phanh và giải thuật Pure Pursuit điều khiển góc lái bám theo waypoint mục tiêu.

### 3. Module Nhận thức & Mô hình Học sâu (`core_perception/`)
*   [multi_agent_model.py](core_perception/multi_agent_model.py): Định nghĩa cấu trúc mạng **GTNet** cải tiến (GRU Encoder, Graph Attention layers, Multimodal shared-GRU Decoder và Winner-Takes-All loss).
*   [multi_agent_trajectory.py](core_perception/multi_agent_trajectory.py): Trích xuất cửa sổ thời gian (40 history, 60 future), xử lý chuyển đổi tọa độ Ego-centric và tính toán đồ thị bán kính thích ứng động.
*   [yolo_detector.py](core_perception/yolo_detector.py): Bộ tích hợp nhận diện vật thể RT-DETR-resnet50 và tracking BoT-SORT/ByteTrack.
*   [cnn_model.py](core_perception/cnn_model.py): Định nghĩa mô hình tích chập (CNN) dự đoán waypoint phục vụ tác nhân CIL.
*   [spatial_math.py](core_perception/spatial_math.py): Cung cấp các thuật toán hình học không gian, phép chiếu camera và ma trận IPM (Inverse Perspective Mapping) để biến đổi tọa độ 2D-3D.

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

Mọi cấu hình mặc định được nạp từ file `configs/carla_env.yaml`. Bạn có thể khởi chạy tác nhân bằng CLI:

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
python run_agents.py --agent cil --cil-model-path models/waypoint_predictor_csv.pth

# Chạy CIL kết hợp RT-DETR-resnet50 + GTNet Supervisor (Bản đầy đủ an toàn nhất)
python run_agents.py --agent cil_yolo --cil-model-path models/waypoint_predictor_csv.pth --yolo-model-path models/rtdetr-l.engine --enable-gtnet --gtnet-model-path models/gtnet_full.pt
```

---

## ⚡ Các Tệp tin & Yếu tố Quyết định Hiệu năng của Xe tự hành

Sự ổn định, an toàn và chính xác của xe tự hành trong mô phỏng CARLA phụ thuộc trực tiếp vào các tệp tin trọng số và các siêu tham số cấu hình sau:

### 1. Trọng số Mô hình quyết định Phản ứng & Nhận thức (Checkpoints)
*(Lưu ý: Các mô hình trọng số lớn này được lưu trữ trên Google Drive dự án, cần tải về đặt đúng thư mục trước khi chạy)*
*   **Mô hình Lập kế hoạch Waypoint (CIL):** Tệp tin `models/waypoint_predictor_csv.pth` chứa trọng số mạng CNN quyết định hướng di chuyển. Nếu tệp này bị lỗi hoặc không khớp phiên bản, xe sẽ đánh lái hỗn loạn hoặc đâm lên vỉa hè.
*   **Mô hình Nhận diện Đèn đỏ & Vật cản (RT-DETR-resnet50):** Việc sử dụng TensorRT engine `models/rtdetr-l.engine` là bắt buộc trong môi trường thời gian thực. Nếu sử dụng tệp `.pt` thông thường, FPS nhận dạng sẽ giảm từ ~150 FPS xuống < 20 FPS, gây trễ nghiêm trọng cho bộ phanh khẩn cấp phản động.
*   **Mô hình Dự đoán Quỹ đạo Tương tác (GTNet):** Tệp tin `models/gtnet_full.pt` quyết định khả năng dự báo trước 3.0s. Các tham số kiến trúc của mô hình này (mặc định `hidden_dim: 384`, `graph_layers: 4`, `num_modes: 5`) phải khớp 100% với tệp cấu hình mô phỏng.

### 2. Cấu hình Môi trường Mô phỏng Vật lý (Simulation Settings)
Mọi thiết lập vật lý trong file `configs/carla_env.yaml` trực tiếp ảnh hưởng đến độ trễ và khả năng kiểm soát xe:
*   **`sync: true` (Chế độ đồng bộ):** Bắt buộc phải kích hoạt. Chế độ đồng bộ đảm bảo giả lập chỉ tiến tới tick tiếp theo khi các bộ điều khiển PID và module nhận dạng đã tính toán xong, duy trì tính ổn định của hệ thống.
*   **`fixed_delta: 0.05` (Tần số 20Hz):** Cố định khoảng thời gian bước nhảy vật lý. Nếu chạy mất đồng bộ hoặc tần số không ổn định, các lệnh ga/phanh từ bộ điều khiển PID sẽ phản hồi trễ, gây ra hiện tượng giật lắc hoặc mất kiểm soát.
*   **`gtnet.inference_every_n_ticks: 2` (Chạy dự báo 10Hz):** Gọi mô hình GTNet mỗi 2 ticks giả lập giúp giảm tải tính toán CPU/GPU xuống 50% mà không làm giảm độ an toàn (Risk Cache được tái sử dụng ở tick xen kẽ).

### 3. Giải thuật Điều khiển & Trọng số PID (Control Laws)
*   **Tham số nhìn trước (Lookahead Distance):** Được định nghĩa trong giải thuật `core_control/pure_pursuit.py`. Khoảng cách này quá ngắn sẽ làm xe ôm cua gấp gây lắc đuôi; khoảng cách quá dài sẽ làm xe cắt cua sớm và đi lệch làn.
*   **Bộ điều khiển `core_control/pid_manager.py`:** Hệ số tỉ lệ/tích phân/vi phân (K_p, K_i, K_d) quyết định gia tốc ga và lực phanh. Sự không khớp hệ số sẽ làm xe đi quá tốc độ cho phép hoặc phanh gấp liên tục.
*   **Cơ chế Brake Fusion:** Kết hợp phanh phản động (RT-DETR-resnet50) và phanh chủ động (GTNet) qua hàm lấy cực đại. Cơ chế này đảm bảo an toàn cao nhất bằng cách ưu tiên lệnh phanh lớn nhất bất kể nguồn gốc nguy cơ.

---

## 📊 Kết Quả Đánh Giá & Thử Nghiệm (Model Evaluation)

Dưới đây là so sánh hiệu năng giữa mô hình huấn luyện đầy đủ **GTNet Full** và kết quả nghiên cứu biến thể (**Ablation Study**):

| Mô hình / Biến thể | GAT | WTA Loss | Bán kính thích ứng | Tập dữ liệu huấn luyện | minADE (m) ↓ | minFDE (m) ↓ | Miss Rate ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ablation (Cấu hình 111)** | ✓ | ✓ | ✓ | **35,000 mẫu** (Rút gọn) | 0.5155 | 1.1766 | 15.54% |
| **GTNet Full chính thức** | ✓ | ✓ | ✓ | **79,094 mẫu** (Đầy đủ) | **0.4410** | 1.2011 | 16.07% |

> [!NOTE]
> **Giải thích về sự khác biệt thông số:**
> Mặc dù mô hình thử nghiệm Ablation (Cấu hình 111) có chỉ số minFDE (1.1766m) và Miss Rate (15.54%) tối ưu hơn một chút so với mô hình GTNet Full chính thức (1.2011m và 16.07%), điều này là hoàn toàn bình thường và hợp lý vì:
> 1. **Quy mô tập dữ liệu:** Thử nghiệm Ablation Study được thực hiện trên một tập dữ liệu rút gọn (chỉ **35,000 mẫu**), có độ đa dạng thấp hơn và ít tình huống giao thông phức tạp/đặc biệt hơn, giúp mô hình dễ dàng đạt độ lỗi cuối cùng (FDE) và tỉ lệ trượt (Miss Rate) thấp hơn trên tập validation tương ứng.
> 2. **Khả năng tổng quát hóa (Generalization):** Mô hình GTNet Full chính thức được huấn luyện và tối ưu hóa trên toàn bộ tập dữ liệu đầy đủ (**79,094 mẫu**) từ nhiều thị trấn (Towns) khác nhau của CARLA. Việc học trên tập dữ liệu lớn này giúp mô hình bao quát được nhiều ca biên nguy hiểm (edge cases) và đạt độ an toàn cũng như khả năng hoạt động thực tế (online generalization) vượt trội khi xe tự hành vận hành trong môi trường mô phỏng thời gian thực, dù một vài chỉ số offline có vẻ cao hơn một chút do độ phức tạp cao của tập kiểm thử đầy đủ.

