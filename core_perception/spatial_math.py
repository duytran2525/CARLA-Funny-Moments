"""
Spatial Math – Phase 2
======================
Chiếu điểm đáy bounding box từ không gian ảnh 2D thành tọa độ thực
(x_m, y_m) trên mặt đường (mét) bằng **Pinhole Camera Math** với giả định
mặt phẳng phẳng (flat-ground assumption).

Thay thế IPM warp matrix: không cần tính ma trận homography, chỉ cần biết
chiều cao camera, góc pitch và tiêu cự (tính từ FOV).

Ứng dụng
--------
* Cung cấp tọa độ (X, Y) mét cho ``object_tracker.py`` và ``adas_logic.py``.
* Tạo Pseudo-BEV Mini-map cho Phase 4 (visualization).

Nguyên lý toán học
------------------
Dựa trên mô hình camera lỗ kim (pinhole camera model):

    * Camera ở độ cao H (mét) so với mặt đường.
    * Camera nghiêng xuống góc α (rad) so với đường nằm ngang.
    * Một điểm P trên mặt đường nằm ở khoảng cách d phía trước và lệch
      lat sang một bên.

Công thức::

    θ_v = arctan((v - cy) / fy)          # góc dọc từ trục quang đến pixel
    total_angle = α + θ_v
    d  = H / tan(total_angle)             # khoảng cách phía trước (mét)
    lat = d * (u - cx) / fx              # lệch ngang (mét, + = phải)

Tham khảo
---------
* CARLA camera intrinsics: https://carla.readthedocs.io/en/stable/ref_sensors/
* Mallot et al., "Inverse Perspective Mapping Simplifies Optical Flow
  Computation and Obstacle Detection", BMVC 1991.
"""

from __future__ import annotations

import math
from typing import Tuple


class SpatialMath:
    """
    Chuyển đổi pixel → tọa độ thực trên mặt đường bằng Pinhole Camera Math.

    Thông số được đọc từ ``configs/camera_geometry.yaml`` (fov, height, pitch).
    Không cần ma trận warp hay homography.

    Parameters
    ----------
    image_width : int
        Chiều rộng ảnh camera (pixel). Mặc định 1920.
    image_height : int
        Chiều cao ảnh camera (pixel). Mặc định 1080.
    fov_deg : float
        Góc nhìn ngang (Horizontal FOV) của camera (độ). Mặc định 90°.
    cam_height : float
        Chiều cao camera so với mặt đường (mét). Mặc định 2.2m
        (từ CARLA camera transform z=2.2).
    cam_pitch_deg : float
        Góc nghiêng camera nhìn xuống so với đường nằm ngang (độ dương =
        nhìn xuống). CARLA đặt pitch=-8.0 (nhìn xuống 8°), vậy
        cam_pitch_deg=8.0. Mặc định 8.0°.

    Examples
    --------
    >>> sm = SpatialMath(image_width=1920, image_height=1080)
    >>> x_m, y_m, valid = sm.pixel_to_world(960, 900)
    >>> print(f"x={x_m:.1f}m, y={y_m:.1f}m, valid={valid}")
    """

    def __init__(
        self,
        image_width: int = 1920,
        image_height: int = 1080,
        fov_deg: float = 90.0,
        cam_height: float = 2.2,
        cam_pitch_deg: float = 8.0,
    ) -> None:
        self._setup_params(image_width, image_height, fov_deg, cam_height, cam_pitch_deg)

    def _setup_params(
        self,
        image_width: int,
        image_height: int,
        fov_deg: float,
        cam_height: float,
        cam_pitch_deg: float,
    ) -> None:
        """Thiết lập (hoặc cập nhật) tất cả các thông số camera nội tại."""
        self.W = image_width
        self.H = image_height
        self.cam_height = cam_height
        self.cam_pitch_rad = math.radians(cam_pitch_deg)

        # Tiêu cự (pixels) từ FOV ngang và kích thước ảnh (fx = fy, pixel vuông)
        hfov_rad = math.radians(fov_deg)
        self.fx = (image_width / 2.0) / math.tan(hfov_rad / 2.0)
        self.fy = self.fx

        # Tâm ảnh (optical center)
        self.cx = image_width / 2.0
        self.cy = image_height / 2.0

    def pixel_to_world(
        self, u: float, v: float
    ) -> Tuple[float, float, bool]:
        """
        Chuyển tọa độ pixel (u, v) thành tọa độ thực (x_m, y_m) trên mặt đường.

        Chỉ hoạt động với các điểm nằm trên mặt đất (không phải bầu trời).

        Parameters
        ----------
        u : float
            Tọa độ ngang (cột pixel), tính từ trái sang phải.
        v : float
            Tọa độ dọc (hàng pixel), tính từ trên xuống dưới.
            Nên là tọa độ điểm đáy (bottom) của bounding box để đảm bảo
            điểm nằm trên mặt đất.

        Returns
        -------
        x_m : float
            Khoảng cách từ camera về phía trước (mét). Dương = phía trước xe.
        y_m : float
            Lệch sang phải/trái so với trục xe (mét). Dương = bên phải.
        valid : bool
            ``False`` nếu điểm nằm phía sau camera, ở đường chân trời,
            hoặc ở xa vô hạn (total_angle ≤ 0).
        """
        # Góc dọc từ trục quang tới pixel (dương = phía dưới trung tâm ảnh)
        theta_v = math.atan2(v - self.cy, self.fy)

        # Tổng góc dưới đường nằm ngang (camera pitch + góc tới pixel)
        total_angle = self.cam_pitch_rad + theta_v

        # Điểm ở đường chân trời hoặc phía trên → không hợp lệ
        if total_angle <= 1e-6:
            return 0.0, 0.0, False

        # Khoảng cách về phía trước (mét)
        x_m = self.cam_height / math.tan(total_angle)
        if x_m <= 0:
            return 0.0, 0.0, False

        # Lệch ngang (mét, dương = bên phải)
        y_m = x_m * (u - self.cx) / self.fx

        return x_m, y_m, True

    def bbox_bottom_to_world(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> Tuple[float, float, bool]:
        """
        Tính tọa độ thực từ điểm giữa đáy (bottom-center) của bounding box.

        Điểm đáy giữa là ước lượng tốt nhất cho điểm tiếp xúc của đối
        tượng với mặt đường trong giả định flat-ground.

        Parameters
        ----------
        x1, y1, x2, y2 : float
            Tọa độ pixel của bounding box theo định dạng
            (left, top, right, bottom) – cùng format với YOLO output.

        Returns
        -------
        x_m, y_m, valid : float, float, bool
            Xem ``pixel_to_world`` để biết ý nghĩa.
        """
        u_center = (x1 + x2) / 2.0
        v_bottom = y2
        return self.pixel_to_world(u_center, v_bottom)

    def update_camera_params(
        self,
        image_width: int,
        image_height: int,
        fov_deg: float,
        cam_height: float = 2.2,
        cam_pitch_deg: float = 8.0,
    ) -> None:
        """
        Cập nhật thông số camera (ví dụ khi thay đổi độ phân giải lúc chạy).

        Parameters
        ----------
        image_width, image_height : int
            Kích thước ảnh mới (pixel).
        fov_deg : float
            FOV ngang mới (độ).
        cam_height : float
            Chiều cao camera mới (mét).
        cam_pitch_deg : float
            Góc nghiêng camera mới (độ, dương = nhìn xuống).
        """
        self._setup_params(image_width, image_height, fov_deg, cam_height, cam_pitch_deg)
