# Trapezoid Obstacle Corridor - Refactoring Summary

## 📋 Tổng Quan
Đã viết lại phương thức vẽ obstacle corridor trong `traffic_supervisor.py` từ cách tính toán không chính xác (hình tam giác) sang cách tính toán chính xác sử dụng **vehicle-space 3D → image-space 2D projection**.

## 🔧 Cách Cũ vs Cách Mới

### ❌ Cách Cũ (Screen-Space)
```
- Dùng screen-space Y coordinates (pixel values)
- Tính ngang dựa trên Y thay vì forward distance
- Không phân biệt vehicle frame vs image frame
- Kết quả: Hình tam giác không chính xác
```

### ✅ Cách Mới (Vehicle-Space 3D Projection)
```
1. Sample points trong vehicle-space: forward_m ∈ [0.8m, horizon_m]
2. Tính curvature từ steering: k = tan(steer_angle) / wheelbase
3. Tính center offset: y_c = 0.5 × k × x²  (parabolic trajectory)
4. Tính half-width: w = base + growth × x + curve_gain × k × x
5. Tạo 3D points: P = (x, y±w, z=0) trong vehicle frame
6. Project sang image: (u,v) = camera_matrix × P_cam
7. Polygon = left_points + reversed(right_points)
```

## 📝 Helper Methods Mới

| Method | Tác dụng |
|--------|---------|
| `_clamp(value, low, high)` | Giới hạn giá trị |
| `_steer_to_curvature(steer)` | Steering → Curvature (1/m) |
| `_curved_path_center_lateral(fwd, steer)` | Forward distance → Lateral offset |
| `_curved_path_half_width(fwd, steer)` | Forward distance → Half-width (m) |
| `_camera_to_vehicle_rotation(pitch)` | Tính rotation matrix camera→vehicle |
| `_project_vehicle_to_image(P, ...)` | Project 3D point → image pixel |

## 🎯 Tính Năng

### ✅ Hình Thang Chính Xác
- Đáy: Rộng (gần camera)
- Đỉnh: Hẹp (xa camera)
- Kết quả: Trapezoid thực sự, không phải tam giác

### ✅ Bẻ Cong Theo Steering
```
Steering = 0°    → Hành lang thẳng
Steering = -30°  → Bẻ trái (parabolic)
Steering = +30°  → Bẻ phải (parabolic)
```

### ✅ Dính Trên Mặt Đất
- Tất cả points có z = 0 (road plane)
- Camera pitch được tính toán chính xác
- Không có distortion từ perspective

### ✅ Tùy Chỉnh Horizon Theo Speed
```
Speed  10 km/h → Horizon ngắn (~12m)
Speed  30 km/h → Horizon trung bình (~18m)
Speed  60 km/h → Horizon dài (~25m)
```

### ✅ Clipping Tự Động
- Các points nằm ngoài bounds được clip
- Polygon luôn nằm trong image bounds [0, W] × [0, H]

## 📊 Test Results

### TEST 1: Straight Line ✅
- Polygon generated: 126 points
- Structure: Symmetric trapezoid
- **PASSED**

### TEST 2: Left Turn ✅
- Polygon generated: 96 points
- Bias: Left side dominant (288.9 pixels)
- All points within bounds (clipped)
- **PASSED**

### TEST 3: Right Turn ✅
- Polygon generated: 112 points
- Bias: Right side dominant (261.6 pixels)
- All points within bounds (clipped)
- **PASSED**

### TEST 4: Speed Effect ✅
- Speed 10 km/h: Y range = [251, 479]
- Speed 30 km/h: Y range = [224, 479]
- Speed 60 km/h: Y range = [214, 479]
- Horizon expands with higher speed
- **PASSED**

### TEST 5: Helper Methods ✅
- Curvature calculation: Correct (clipped ±0.45 1/m)
- Lateral offset: Correct parabolic trajectory
- Half-width: Correct growth with forward distance
- **PASSED**

## 🔄 Backward Compatibility

✅ **Hoàn toàn backward compatible**
- Tất cả new parameters có default values
- Không break existing code
- Old calls vẫn hoạt động:
  ```python
  polygon = supervisor._build_obstacle_danger_polygon(image_shape, vehicle_steer)
  polygon = supervisor._build_obstacle_danger_polygon(image_shape, steer_val, current_speed)
  ```

## 📦 Default Parameters

```python
# Camera mounting
camera_fov_deg: float = 90.0
camera_mount_xyz: tuple = (1.5, 0.0, 2.2)  # (x, y, z) meters
camera_pitch_deg: float = -8.0

# Path parameters
path_wheelbase_m: float = 2.7
path_max_steer_angle_deg: float = 70.0
path_base_half_width_m: float = 1.1
path_width_growth_per_m: float = 0.035
path_curve_width_gain: float = 0.55
path_max_half_width_m: float = 2.8
path_min_forward_m: float = 0.8
path_max_forward_m: float = 40.0
```

## 🚀 Usage

```python
supervisor = TrafficSupervisor(config)

# Cách 1: Mặc định (straight, no speed)
polygon = supervisor._build_obstacle_danger_polygon(
    image_shape=(480, 640, 3)
)

# Cách 2: Với steering
polygon = supervisor._build_obstacle_danger_polygon(
    image_shape=(480, 640, 3),
    vehicle_steer=-0.3,
    vehicle_speed_kmh=30.0
)

# Cách 3: Full custom (rarely needed)
polygon = supervisor._build_obstacle_danger_polygon(
    image_shape=(480, 640, 3),
    vehicle_steer=0.2,
    vehicle_speed_kmh=25.0,
    camera_fov_deg=90.0,
    camera_mount_xyz=(1.5, 0.0, 2.2),
    camera_pitch_deg=-8.0
)
```

## 📌 Lưu Ý Kỹ Thuật

### Curvature Calculation
```
steer ∈ [-1, 1] → wheel_angle ∈ [-70°, +70°]
curvature = tan(wheel_angle) / wheelbase
curvature ∈ [-0.45, +0.45] (1/m) - CLIPPED
```

### Parabolic Trajectory
```
lateral_offset = 0.5 × curvature × forward_distance²
Đây là công thức kinematic steering model (Ackermann)
```

### Projection Pipeline
```
Vehicle Frame (x, y, z)
    ↓
Rotation: vehicle → camera (pitch transform)
    ↓
Translation: camera mount position
    ↓
Camera Frame (X, Y, Z)
    ↓
Intrinsic: f_x, f_y, c_x, c_y (from FOV)
    ↓
Image Frame (u, v) pixels
    ↓ Clip to bounds
    ↓
Final Polygon
```

## 🔗 File Changes

- **Modified**: `core_control/traffic_supervisor.py`
  - Thêm 6 helper methods
  - Viết lại `_build_obstacle_danger_polygon()`
  - Thêm clipping logic

- **Created**: `test_trapezoid_polygon.py`
  - 5 test cases (straight, left, right, speed, helpers)
  - Kiểm chứng logic đúng
  - All tests PASSED ✅

## 🎓 Learning Points

1. **Vehicle-Space vs Screen-Space**: Luôn phân biệt coordinate frames
2. **3D Projection**: Cần rotation + translation + intrinsic matrix
3. **Parabolic Path**: Steering model sinh ra đường cong parabol
4. **Clipping**: Out-of-bounds points cần xử lý an toàn
5. **Testing**: Helper methods cần unit tests để verify

## ✨ Tổng Kết

✅ Obstacle corridor giờ vẽ **hình thang chính xác** (thay vì hình tam giác)  
✅ Bẻ cong theo **steering angle** (parabolic trajectory)  
✅ Dính trên **mặt đất** (vehicle plane z=0)  
✅ **Tự động clipping** để an toàn  
✅ **Backward compatible** 100%  
✅ **Fully tested** - 5/5 tests PASSED  

Cách này hoàn toàn khác so với cách cũ - thay vì tính toán đơn giản trong screen-space, nó giờ dùng projection 3D chính xác với kiến thức về camera geometry!
