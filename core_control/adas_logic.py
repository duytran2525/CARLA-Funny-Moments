"""
ADAS Logic – Phase 2
====================
Module tính **Time-To-Collision (TTC)** và điều khiển **Adaptive Cruise Control (ACC)**
dựa trên các track đã có tọa độ (X, Y) mét từ ``object_tracker.SimpleTracker``.

Luồng Sinh tồn (Survival Pipeline)
------------------------------------
::

    YOLOv8 detections
        → SimpleTracker.update()                     # gán ID
        → SimpleTracker.update_world_positions(sm)   # pixel → mét
        → ADASModule.compute_control(tracks, ...)    # TTC + ACC → ga/phanh

Ngưỡng mặc định được đọc từ ``configs/adas_config.yaml``:
    * ``min_ttc``  = 2.0s → phanh khẩn cấp
    * ``safe_ttc`` = 5.0s → giảm tốc nhẹ
    * ``acc_distance`` = 10m → khoảng cách giữ ACC
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from core_perception.object_tracker import Track


class ADASModule:
    """
    Module ADAS Phase 2: Tính TTC và điều khiển Adaptive Cruise Control.

    Sơ đồ quyết định::

        TTC < min_ttc              → 🛑  Phanh khẩn cấp (max_brake)
        min_ttc ≤ TTC < safe_ttc  → ⚠️  Giảm tốc nhẹ (proportional brake)
        TTC ≥ safe_ttc             → ✅  Giữ tốc độ mục tiêu (ACC bình thường)

    Parameters
    ----------
    min_ttc : float
        Ngưỡng TTC (giây) kích hoạt phanh khẩn cấp. Mặc định 2.0s
        (khớp với ``adas_config.yaml``).
    safe_ttc : float
        Ngưỡng TTC (giây) xe hoạt động bình thường. Mặc định 5.0s.
    acc_distance : float
        Khoảng cách tối thiểu ACC giữ với xe trước (mét). Mặc định 10m.
    lane_width : float
        Bề rộng làn đường (mét) để lọc vật thể trong làn. Mặc định 3.5m.
    max_brake : float
        Mức phanh tối đa (0–1). Mặc định 0.8.
    max_throttle : float
        Mức ga tối đa khi ACC (0–1). Mặc định 0.6.
    """

    def __init__(
        self,
        min_ttc: float = 2.0,
        safe_ttc: float = 5.0,
        acc_distance: float = 10.0,
        lane_width: float = 3.5,
        max_brake: float = 0.8,
        max_throttle: float = 0.6,
    ) -> None:
        self.min_ttc = min_ttc
        self.safe_ttc = safe_ttc
        self.acc_distance = acc_distance
        self.lane_width = lane_width
        self.max_brake = max_brake
        self.max_throttle = max_throttle

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_ttc(self, track: Track, ego_speed_mps: float) -> Optional[float]:
        """
        Tính Time-To-Collision (giây) cho một đối tượng đang được theo dõi.

        Nếu có đủ lịch sử (≥ 2 điểm): sử dụng vận tốc tiếp cận thực tế.
        Nếu không: giả định vật thể đứng yên, closing_speed = ego_speed.

        Quy ước dấu vận tốc: ``world_x`` là khoảng cách phía trước (dương = xa hơn).
        ``obj_speed_mps > 0`` → đối tượng đang xa ra (world_x tăng).
        ``obj_speed_mps < 0`` → đối tượng đang lại gần (world_x giảm).

        Parameters
        ----------
        track : Track
            Track cần tính TTC.
        ego_speed_mps : float
            Tốc độ xe chủ (m/s).

        Returns
        -------
        float or None
            TTC tính bằng giây. ``None`` nếu xe đang ra xa hoặc tốc độ ≤ 0.
        """
        if track.world_x <= 0:
            return None

        # Ước lượng tốc độ đối tượng từ lịch sử world_x
        obj_speed_mps = 0.0
        if len(track.history) >= 2:
            t1, x1, _ = track.history[-2]
            t2, x2, _ = track.history[-1]
            dt = t2 - t1
            if dt > 1e-3:
                obj_speed_mps = (x2 - x1) / dt

        closing_speed = ego_speed_mps - obj_speed_mps
        if closing_speed <= 0:
            return None  # Xe không lại gần đối tượng

        return track.world_x / closing_speed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_control(
        self,
        tracks: List[Track],
        ego_speed_mps: float,
        target_speed_mps: float,
    ) -> Tuple[float, float, float]:
        """
        Tính lệnh ga/phanh theo TTC và tốc độ mục tiêu (ACC).

        Parameters
        ----------
        tracks : List[Track]
            Danh sách track đang hoạt động (đã có world_x, world_y từ SpatialMath).
        ego_speed_mps : float
            Tốc độ hiện tại của xe (m/s).
        target_speed_mps : float
            Tốc độ mục tiêu (m/s).

        Returns
        -------
        throttle : float
            Mức ga trong ``[0, max_throttle]``.
        brake : float
            Mức phanh trong ``[0, max_brake]``.
        min_ttc : float
            TTC nhỏ nhất (giây) để debug/log. ``float('inf')`` nếu không
            có vật cản trong làn.
        """
        # Chỉ xét vật thể trong làn xe, phía trước và trong ngưỡng acc_distance
        in_lane = [
            t for t in tracks
            if abs(t.world_y) < self.lane_width / 2.0
            and 0 < t.world_x <= max(self.acc_distance * 5, 100.0)
        ]

        min_ttc = float('inf')
        for track in in_lane:
            ttc = self._compute_ttc(track, ego_speed_mps)
            if ttc is not None and ttc < min_ttc:
                min_ttc = ttc

        # --- Quyết định dựa trên TTC ---
        if min_ttc < self.min_ttc:
            # Phanh khẩn cấp
            return 0.0, self.max_brake, min_ttc

        if min_ttc < self.safe_ttc:
            # Phanh tỉ lệ (gradient từ 0 đến max_brake * 0.5)
            ratio = 1.0 - (min_ttc - self.min_ttc) / (self.safe_ttc - self.min_ttc)
            brake = self.max_brake * ratio * 0.5
            return 0.0, brake, min_ttc

        # Không có vật cản nguy hiểm → điều khiển tốc độ mục tiêu (ACC)
        error = target_speed_mps - ego_speed_mps
        if error >= 0:
            throttle = min(0.2 + 0.05 * error, self.max_throttle)
            return throttle, 0.0, min_ttc
        else:
            brake = min((-error) / 5.0, self.max_brake)
            return 0.0, brake, min_ttc
