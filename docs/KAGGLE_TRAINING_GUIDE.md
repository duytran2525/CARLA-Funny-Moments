# Hướng dẫn Training GTNet trên Kaggle

## Bước 1: Thu thập Dữ liệu (Local - CARLA)

### 1.1. Chạy CARLA Server

```powershell
# Mở terminal và chạy CARLA
cd C:\CARLA_0.9.15  # Thay đổi theo đường dẫn CARLA của bạn
.\CarlaUE4.exe -quality-level=Low -RenderOffScreen
```

### 1.2. Thu thập dữ liệu cho tất cả towns

```powershell
# Chạy script thu thập tự động
.\scripts\collect_all_towns.ps1
```

Hoặc thu thập từng town riêng lẻ:

```powershell
# Town01 - 5000 samples (500 giây)
python collect_multi_agent_data.py --town Town01 --duration 500 --npc-vehicles 40

# Town02
python collect_multi_agent_data.py --town Town02 --duration 500 --npc-vehicles 40

# Town03
python collect_multi_agent_data.py --town Town03 --duration 500 --npc-vehicles 40

# Town04
python collect_multi_agent_data.py --town Town04 --duration 500 --npc-vehicles 40

# Town05
python collect_multi_agent_data.py --town Town05 --duration 500 --npc-vehicles 40

# Town06
python collect_multi_agent_data.py --town Town06 --duration 500 --npc-vehicles 40

# Town07
python collect_multi_agent_data.py --town Town07 --duration 500 --npc-vehicles 40
```

**Lưu ý:**
- Mỗi town: 500 giây = 5000 frames (ở 10 FPS)
- Tổng thời gian: ~58 phút (7 towns × 8.3 phút)
- Dữ liệu raw sẽ được lưu trong `data/multi_agent/raw/`

### 1.3. Build Dataset

```powershell
# Build dataset với adaptive radius
python scripts/build_multi_agent_dataset.py `
    --csv data/multi_agent/raw/*.csv `
    --output data/multi_agent/processed `
    --adaptive-radius `
    --radius-base 20.0 `
    --radius-alpha 0.5
```

**Output:**
- Processed samples: `data/multi_agent/processed/samples/*.pt`
- Manifest: `data/multi_agent/processed/manifest.csv`
- Summary: `data/multi_agent/processed/build_summary.json`

## Bước 2: Upload Dataset lên Kaggle

### 2.1. Nén dataset

```powershell
# Nén thư mục processed
Compress-Archive -Path data\multi_agent\processed -DestinationPath gtnet_dataset.zip
```

### 2.2. Upload lên Kaggle

1. Truy cập https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload file `gtnet_dataset.zip`
4. Đặt tên: `gtnet-dataset`
5. Chọn "Public" hoặc "Private"
6. Click "Create"

## Bước 3: Setup Kaggle Notebook

### 3.1. Tạo Notebook mới

1. Truy cập https://www.kaggle.com/code
2. Click "New Notebook"
3. Chọn GPU: Settings → Accelerator → GPU T4 x2 (hoặc P100)

### 3.2. Add Dataset

1. Click "Add Data" ở sidebar phải
2. Tìm dataset `gtnet-dataset` của bạn
3. Click "Add"

### 3.3. Upload Code

**Option 1: Clone từ GitHub**

```python
!git clone https://github.com/your-username/CARLA-Funny-Moments.git
%cd CARLA-Funny-Moments
```

**Option 2: Upload files trực tiếp**

Upload các file sau vào Kaggle:
- `core_perception/multi_agent_model.py`
- `core_perception/multi_agent_trajectory.py`
- `core_perception/multi_agent_dataset.py`
- `scripts/kaggle_train_gtnet.py`

## Bước 4: Training trên Kaggle

### 4.1. Training Baseline (không có cải tiến)

```python
!python scripts/kaggle_train_gtnet.py \
    --data-dir /kaggle/input/gtnet-dataset/processed \
    --out-dir /kaggle/working/models/baseline \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --seed 42
```

**Thời gian dự kiến:** ~2-3 giờ trên GPU T4

### 4.2. Training GTNet Full (tất cả cải tiến)

```python
!python scripts/kaggle_train_gtnet.py \
    --data-dir /kaggle/input/gtnet-dataset/processed \
    --out-dir /kaggle/working/models/gtnet_full \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --enable-gat \
    --enable-multimodal \
    --enable-adaptive-radius \
    --num-modes 3 \
    --num-attention-heads 4 \
    --seed 42
```

**Thời gian dự kiến:** ~3-4 giờ trên GPU T4

### 4.3. Training Per-Town

```python
# Train riêng cho Town01
!python scripts/kaggle_train_gtnet.py \
    --data-dir /kaggle/input/gtnet-dataset/processed \
    --town-filter Town01 \
    --out-dir /kaggle/working/models/Town01 \
    --epochs 50 \
    --batch-size 32 \
    --enable-gat \
    --enable-multimodal \
    --enable-adaptive-radius \
    --seed 42
```

### 4.4. Ablation Study

```python
# Chạy ablation study (8 variants)
!python scripts/run_ablation_study.py \
    --data-dir /kaggle/input/gtnet-dataset/processed \
    --out-dir /kaggle/working/ablation \
    --epochs 30 \
    --batch-size 32 \
    --seed 42
```

**Thời gian dự kiến:** ~8-10 giờ (8 variants × 1-1.5 giờ)

## Bước 5: Download Checkpoints

```python
# Xem kết quả
!ls -lh /kaggle/working/models/gtnet_full/

# Download best checkpoint
from IPython.display import FileLink
FileLink('/kaggle/working/models/gtnet_full/best.pt')
```

## Bước 6: Validation

```python
# Validate model
!python scripts/validate_target_metrics.py \
    --checkpoint /kaggle/working/models/gtnet_full/best.pt \
    --data-dir /kaggle/input/gtnet-dataset/processed \
    --out-dir /kaggle/working/validation
```

## Kết quả Mong đợi

### Baseline
- **ADE**: ~1.84m
- **FDE**: ~4.12m
- **MissRate**: ~38%

### GTNet Full (với 5 cải tiến)
- **minADE**: <0.9m (target)
- **minFDE**: <1.8m (target)
- **MissRate**: <10% (target)

## Troubleshooting

### CUDA Out of Memory

**Giải pháp:**
```python
# Giảm batch size
--batch-size 16  # hoặc 8
```

### Training quá chậm

**Giải pháp:**
```python
# Giảm số epochs
--epochs 30

# Hoặc dùng quick ablation
--quick-ablation
```

### Dataset không tìm thấy

**Kiểm tra:**
```python
!ls /kaggle/input/
!ls /kaggle/input/gtnet-dataset/
```

## Tips & Tricks

### 1. Sử dụng Kaggle Sessions hiệu quả

- Kaggle cho phép 30 giờ GPU/tuần (free tier)
- Mỗi session tối đa 12 giờ
- Lưu checkpoints thường xuyên

### 2. Tối ưu Batch Size

| GPU | Recommended Batch Size |
|-----|------------------------|
| T4  | 32                     |
| P100| 64                     |
| V100| 128                    |

### 3. Early Stopping

```python
# Tăng patience nếu muốn train lâu hơn
--early-stopping-patience 10
```

### 4. Learning Rate Tuning

```python
# Nếu loss không giảm
--learning-rate 5e-4

# Nếu loss dao động
--learning-rate 2e-3
```

## Tham khảo

- **GTNet Paper**: [Graph Trajectory Networks](https://arxiv.org/abs/...)
- **GAT Paper**: [Graph Attention Networks (Veličković et al., ICLR 2018)](https://arxiv.org/abs/1710.10903)
- **WTA Loss**: [Multiple Futures Prediction (Lee et al., NeurIPS 2017)](https://arxiv.org/abs/1705.10292)
- **Documentation**: `scripts/README_train_per_town.md`
- **Ablation Study**: `scripts/README_ablation_study.md`
