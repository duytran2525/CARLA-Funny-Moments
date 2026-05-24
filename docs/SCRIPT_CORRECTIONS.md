# Các Điểm Cần Sửa trong Script Thuyết Trình

## ❌ Các Thông Số SAI cần sửa

### 1. Số lượng Modes (K)

**SAI:**
- Script nói: K = 6 modes

**ĐÚNG:**
- Implementation thực tế: **K = 3 modes** (default)
- Code: `num_modes: int = 3` trong `MultiAgentModelConfig`

**Sửa ở các slide:**
- Slide 4: "K = 6 modes" → "K = 3 modes"
- Slide 7: "K=6" → "K=3"
- Slide 13: "K=6 modes" → "K=3 modes"
- Slide 22, 23: "minADE₆", "minFDE₆" → "minADE₃", "minFDE₃"

---

### 2. Số Frames History và Future

**SAI:**
- Script nói: N_obs = 8 bước (3.2 giây), N_pred = 12 bước (4.8 giây)

**ĐÚNG:**
- Implementation thực tế: 
  - **history_frames = 20** (2.0 giây ở 10 FPS)
  - **future_frames = 30** (3.0 giây ở 10 FPS)
- Code: `WindowBuildConfig(history_frames=20, future_frames=30)`

**Sửa ở các slide:**
- Slide 4: "N_obs = 8 bước (3.2 giây)" → "N_obs = 20 bước (2.0 giây)"
- Slide 4: "N_pred = 12 bước (4.8 giây)" → "N_pred = 30 bước (3.0 giây)"
- Slide 5: "[N,8,2]" → "[N,20,2]"
- Slide 6: "8 bước lịch sử" → "20 bước lịch sử"
- Slide 7: "12 bước tương lai" → "30 bước tương lai"

---

### 3. Cải tiến 4: Transformer Encoder

**SAI:**
- Script nói: "Cải tiến 4: Transformer Encoder — thay GRU bằng Temporal Transformer"

**ĐÚNG:**
- **KHÔNG CÓ Transformer Encoder trong implementation hiện tại**
- Vẫn dùng GRU Encoder
- Code: `self.encoder = nn.GRU(...)` trong `MultiAgentTrajectoryPredictor`

**Hành động:**
- **XÓA hoàn toàn Slide 15** (Cải tiến 4: Transformer Encoder)
- **Đánh số lại:** Cải tiến 5 → Cải tiến 4
- **Sửa Slide 11:** Chỉ còn **4 cải tiến** (không phải 5)
- **Sửa Slide 17:** Bỏ "[Transformer Encoder]" khỏi kiến trúc

---

### 4. Cải tiến 5: Map Encoding

**SAI:**
- Script nói: "Cải tiến 5: Map Encoding & Data Augmentation"

**ĐÚNG:**
- **KHÔNG CÓ Map Encoding trong implementation hiện tại**
- Chỉ có Data Augmentation cơ bản (trong dataset builder)
- Không có CNN encoder cho BEV map

**Hành động:**
- **XÓA hoàn toàn Slide 16** (Cải tiến 5: Map Encoding)
- **Sửa Slide 11:** Chỉ còn **4 cải tiến**:
  1. GAT
  2. WTA Loss
  3. Adaptive Radius
  4. Enhanced Metrics (minADE, minFDE, MissRate)

---

### 5. Adjacency Radius

**SAI:**
- Script nói: r = 10m (baseline)

**ĐÚNG:**
- Implementation thực tế: **r = 40.0m** (default)
- Code: `adjacency_radius_m: float = 40.0` trong `WindowBuildConfig`

**Sửa ở các slide:**
- Slide 5: "r=10m" → "r=40m"
- Slide 6: "||p_i - p_j|| ≤ r=10m" → "||p_i - p_j|| ≤ r=40m"
- Slide 6: "vòng tròn r=10m" → "vòng tròn r=40m"

---

### 6. Adaptive Radius Formula

**SAI:**
- Script nói: r(v) = r_base + k_v · v̄ với r_base=5, k_v=0.2

**ĐÚNG:**
- Implementation thực tế: **r(v) = r_base + alpha · ||v||**
- Default: **r_base = 20.0m, alpha = 0.5**
- Code: `radius_base: float = 20.0, radius_alpha: float = 0.5`

**Sửa ở Slide 14:**
- "r(v) = r_base + k_v · v̄" → "r(v) = r_base + alpha · ||v||"
- "r_base=5, k_v=0.2" → "r_base=20.0, alpha=0.5"
- "Đường cao tốc 50km/h → r≈15m" → "Đường cao tốc 60km/h → r≈28m"
- "Khu dân cư 10km/h → r≈8m" → "Khu dân cư 10km/h → r≈21m"

---

## ✅ Các Điểm ĐÚNG (giữ nguyên)

1. ✅ GTNet sử dụng GRU Encoder + Graph Interaction + Decoder
2. ✅ GAT thay thế mean aggregation
3. ✅ WTA Loss cho multimodal prediction
4. ✅ Adaptive Radius dựa trên vận tốc
5. ✅ Multi-head attention với M=4 heads
6. ✅ Confidence Score kết hợp với WTA Loss
7. ✅ Soft Distance Weighting với Gaussian kernel
8. ✅ Two-stage training strategy
9. ✅ Social Consistency Regularization

---

## 📊 Số liệu Metrics cần cập nhật

### Baseline (cần verify bằng test thực tế)

**Trong script:**
- minADE₁ = 1.84m
- minFDE₁ = 4.12m
- MR = 38%

**Lưu ý:** Số liệu này cần được verify bằng cách chạy test thực tế trên dataset CARLA của bạn.

### Target (sau cải tiến)

**Trong script:**
- minADE₆ < 0.9m
- minFDE₆ < 1.8m
- MR < 10%

**Cần sửa:**
- minADE₆ → minADE₃ (vì K=3, không phải 6)
- minFDE₆ → minFDE₃

**Target thực tế từ requirements:**
- minADE < 1.5m
- minFDE < 2.7m
- MissRate < 0.20

---

## 🔧 Tóm tắt Hành động Sửa

### Xóa hoàn toàn:
1. ❌ Slide 15 (Transformer Encoder)
2. ❌ Slide 16 (Map Encoding)

### Sửa đổi:
1. ✏️ Slide 4, 5, 6, 7: Sửa N_obs=8→20, N_pred=12→30
2. ✏️ Slide 4, 7, 13, 22, 23: Sửa K=6→3
3. ✏️ Slide 5, 6: Sửa r=10m→40m
4. ✏️ Slide 11: Chỉ còn 4 cải tiến (không phải 5)
5. ✏️ Slide 14: Sửa công thức adaptive radius
6. ✏️ Slide 17: Bỏ Transformer và Map CNN khỏi kiến trúc
7. ✏️ Slide 18: Bỏ phần Two-stage training (không có trong code)
8. ✏️ Tất cả metrics: minADE₆→minADE₃, minFDE₆→minFDE₃

### Đánh số lại:
- Slide 17 (cũ) → Slide 15 (mới)
- Slide 18 (cũ) → Slide 16 (mới)
- Slide 19-25 → Slide 17-23

---

## 📝 Kiến trúc GTNet Thực tế

```
Input [N, 20, 6]
    ↓
GRU Encoder [N, 128]
    ↓
GAT (4 heads, Adaptive Radius) [N, 128]
    ↓
GRU Decoder (K=3 modes) [N, 3, 30, 2]
    ↓
WTA Loss + Confidence Loss
```

**Không có:**
- ❌ Transformer Encoder
- ❌ Map CNN
- ❌ BEV Map Encoding

---

## 🎯 Checklist Sửa Script

- [ ] Sửa K=6 → K=3 (tất cả slides)
- [ ] Sửa N_obs=8 → N_obs=20
- [ ] Sửa N_pred=12 → N_pred=30
- [ ] Sửa r=10m → r=40m
- [ ] Xóa Slide 15 (Transformer)
- [ ] Xóa Slide 16 (Map Encoding)
- [ ] Sửa Slide 11: 5 cải tiến → 4 cải tiến
- [ ] Sửa Slide 14: công thức adaptive radius
- [ ] Sửa Slide 17: kiến trúc (bỏ Transformer và Map)
- [ ] Sửa Slide 18: bỏ Two-stage training
- [ ] Đánh số lại slides sau khi xóa
- [ ] Cập nhật metrics: minADE₆→minADE₃

---

## 💡 Gợi ý Thêm

### Slide mới có thể thêm:

**Slide về Data Collection:**
- Thu thập 7 towns (Town01-07)
- 5000 samples mỗi town
- 10 FPS, 30-50 NPC vehicles
- Visibility radius: 100m

**Slide về Dataset Building:**
- Teleportation filter (>6m/0.1s)
- Ego-centric coordinates
- Adaptive adjacency matrix
- 6D features: (x, y, vx, vy, heading_x, heading_y)

**Slide về Training Strategy:**
- AdamW optimizer
- ReduceLROnPlateau scheduler
- Early stopping (patience=8)
- Gradient clipping (max_norm=1.0)
