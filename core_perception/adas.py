"""
ADAS Module – Phase 2: Multi-Object Tracking + TTC + Adaptive Cruise Control.

Luồng Sinh tồn (Survival Pipeline)
------------------------------------
YOLOv8 detections
    → SimpleTracker (gán ID + lọc stale tracks)
    → SimpleTracker.update_world_positions(ipm) (pixel → mét qua IPM)
    → ADASModule.compute_control(tracks, ego_speed, target_speed)
    → Ra lệnh ga/phanh mượt mà

Lớp SimpleTracker
-----------------
Bộ theo dõi đa đối tượng dựa trên IoU, giao diện tương thích ByteTrack.
Không yêu cầu dependency ngoài (không cần supervision hay bytetrack package).

Lớp ADASModule
--------------
* Lọc vật thể trong làn (|lateral| < LANE_WIDTH/2).
* Tính TTC (Time-To-Collision) từ khoảng cách + vận tốc tương đối.
* Điều khiển ACC (Adaptive Cruise Control) theo 3 mức:
    1. TTC < MIN_TTC  → phanh khẩn cấp (emergency brake).
    2. MIN_TTC < TTC < SAFE_TTC → giảm tốc nhẹ (proportional brake).
    3. TTC > SAFE_TTC → giữ tốc độ mục tiêu (standard speed control).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Track dataclass
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """
    Đại diện cho một đối tượng đang được theo dõi.

    Attributes
    ----------
    track_id : int
        ID duy nhất của track (tăng dần từ 1).
    x1, y1, x2, y2 : float
        Tọa độ bounding box hiện tại (pixel).
    class_id : int
        Nhãn lớp từ YOLO (0 = person, 2 = car, ...).
    confidence : float
        Độ tin cậy của detection gần nhất.
    world_x : float
        Khoảng cách phía trước tính từ camera (mét). Được IPM cập nhật.
    world_y : float
        Lệch ngang so với trục xe (mét). Dương = bên phải.
    history : list
        Lịch sử ``(timestamp, world_x, world_y)`` để tính vận tốc tương đối.
    age : int
        Số frame track đã sống.
    missed_frames : int
        Số frame liên tiếp không tìm được detection khớp.
    """

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    confidence: float
    world_x: float = 0.0
    world_y: float = 0.0
    history: List[Tuple[float, float, float]] = field(default_factory=list)
    age: int = 0
    missed_frames: int = 0


# ---------------------------------------------------------------------------
# SimpleTracker
# ---------------------------------------------------------------------------

class SimpleTracker:
    """
    Bộ theo dõi đa đối tượng dựa trên IoU – giao diện tương thích ByteTrack.

    Thuật toán
    ----------
    Mỗi frame:
    1. Với mỗi track đang tồn tại, tìm detection có IoU cao nhất.
    2. IoU > ``iou_threshold`` → cập nhật tọa độ track.
    3. Không khớp với detection nào → tăng ``missed_frames``.
    4. Detection không khớp với track nào → tạo track mới.
    5. Track có ``missed_frames > max_missed`` → xóa khỏi danh sách.

    Parameters
    ----------
    iou_threshold : float
        Ngưỡng IoU tối thiểu để coi hai box là cùng đối tượng. Mặc định 0.3.
    max_missed : int
        Số frame tối đa một track được phép không có detection. Mặc định 5.
    """

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 5) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(a: List[float], b: List[float]) -> float:
        """Tính IoU giữa hai bounding box [x1, y1, x2, y2]."""
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: List[List[float]]) -> List[Track]:
        """
        Cập nhật tracker với danh sách detection từ YOLO trong một frame mới.

        Parameters
        ----------
        detections : List[List[float]]
            Mỗi phần tử có format: ``[x1, y1, x2, y2, confidence, class_id]``
            (tương tự output của ``YoloDetector.detect``).

        Returns
        -------
        List[Track]
            Danh sách tất cả các track đang hoạt động sau khi cập nhật.
        """
        matched_det_indices: set = set()

        # --- Khớp track hiện có với detection mới ---
        for tid in list(self._tracks.keys()):
            track = self._tracks[tid]
            track_box = [track.x1, track.y1, track.x2, track.y2]
            best_iou = self.iou_threshold
            best_di = -1

            for di, det in enumerate(detections):
                if di in matched_det_indices:
                    continue
                iou_val = self._iou(track_box, det[:4])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_di = di

            if best_di >= 0:
                det = detections[best_di]
                track.x1, track.y1 = det[0], det[1]
                track.x2, track.y2 = det[2], det[3]
                track.confidence = float(det[4])
                track.class_id = int(det[5]) if len(det) > 5 else 0
                track.missed_frames = 0
                track.age += 1
                matched_det_indices.add(best_di)
            else:
                track.missed_frames += 1

        # --- Xóa track quá cũ ---
        for tid in [t for t in list(self._tracks.keys())
                    if self._tracks[t].missed_frames > self.max_missed]:
            del self._tracks[tid]

        # --- Tạo track mới cho detection không khớp ---
        for di, det in enumerate(detections):
            if di not in matched_det_indices:
                new_track = Track(
                    track_id=self._next_id,
                    x1=float(det[0]), y1=float(det[1]),
                    x2=float(det[2]), y2=float(det[3]),
                    confidence=float(det[4]),
                    class_id=int(det[5]) if len(det) > 5 else 0,
                )
                self._tracks[self._next_id] = new_track
                self._next_id += 1

        return list(self._tracks.values())

    def update_world_positions(self, ipm, track_list: List[Track]) -> None:
        """
        Cập nhật tọa độ thế giới thực cho mỗi track qua IPM.

        Gọi sau ``update()`` để gắn tọa độ (mét) vào mỗi track, sau đó
        lưu vào ``track.history`` để tính vận tốc tương đối.

        Parameters
        ----------
        ipm : InversePerspectiveMapping
            Module IPM để chuyển đổi pixel → mét.
        track_list : List[Track]
            Danh sách track cần cập nhật (thường là kết quả của ``update()``).
        """
        now = time.monotonic()
        for track in track_list:
            x_m, y_m, valid = ipm.bbox_bottom_to_world(
                track.x1, track.y1, track.x2, track.y2
            )
            if valid:
                track.world_x = x_m
                track.world_y = y_m
                track.history.append((now, x_m, y_m))
                # Giữ lịch sử tối đa 10 frame gần nhất
                if len(track.history) > 10:
                    track.history.pop(0)

    def reset(self) -> None:
        """Xóa toàn bộ track (dùng khi restart episode)."""
        self._tracks.clear()
        self._next_id = 1


# ---------------------------------------------------------------------------
# ADASModule
# ---------------------------------------------------------------------------

class ADASModule:
    """
    Module ADAS Phase 2: Tính TTC và điều khiển Adaptive Cruise Control.

    Sơ đồ quyết định:
    ::

        TTC < MIN_TTC  → 🛑  Phanh khẩn cấp (max_brake)
        MIN_TTC ≤ TTC < SAFE_TTC → ⚠️  Giảm tốc nhẹ (proportional brake)
        TTC ≥ SAFE_TTC → ✅  Giữ tốc độ mục tiêu (ACC bình thường)

    Parameters
    ----------
    min_ttc : float
        Ngưỡng TTC (giây) kích hoạt phanh khẩn cấp. Mặc định 2.5s.
    safe_ttc : float
        Ngưỡng TTC (giây) xe hoạt động bình thường. Mặc định 5.0s.
    lane_width : float
        Bề rộng làn đường (mét) để lọc vật thể trong làn. Mặc định 3.5m.
    max_brake : float
        Mức phanh tối đa (0–1). Mặc định 0.8.
    max_throttle : float
        Mức ga tối đa khi ACC (0–1). Mặc định 0.6.
    """

    def __init__(
        self,
        min_ttc: float = 2.5,
        safe_ttc: float = 5.0,
        lane_width: float = 3.5,
        max_brake: float = 0.8,
        max_throttle: float = 0.6,
    ) -> None:
        self.min_ttc = min_ttc
        self.safe_ttc = safe_ttc
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
                # Quy ước dấu: world_x là khoảng cách phía trước (dương = xa hơn).
                # obj_speed_mps > 0 → đối tượng đang xa ra (world_x tăng)
                # obj_speed_mps < 0 → đối tượng đang lại gần (world_x giảm)
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
            Danh sách track đang hoạt động (đã có world_x, world_y từ IPM).
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
        # Chỉ xét vật thể trong làn xe và phía trước
        in_lane = [
            t for t in tracks
            if abs(t.world_y) < self.lane_width / 2.0 and t.world_x > 0
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

        # Không có vật cản nguy hiểm → điều khiển tốc độ mục tiêu
        error = target_speed_mps - ego_speed_mps
        if error >= 0:
            throttle = min(0.2 + 0.05 * error, self.max_throttle)
            return throttle, 0.0, min_ttc
        else:
            brake = min((-error) / 5.0, self.max_brake)
            return 0.0, brake, min_ttc
