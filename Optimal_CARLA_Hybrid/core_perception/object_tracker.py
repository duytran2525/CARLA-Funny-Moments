"""
Object Tracker – Phase 2
=========================
Bộ theo dõi đa đối tượng (Multi-Object Tracking) dựa trên IoU.
Giao diện tương thích với ByteTrack.

Nhận (X, Y) mét từ ``spatial_math.SpatialMath`` rồi gắn ID nhất quán
qua các frame liên tiếp. Lịch sử tọa độ được dùng bởi ``adas_logic.ADASModule``
để tính vận tốc tương đối và TTC.

Luồng sử dụng điển hình
------------------------
::

    from core_perception.spatial_math import SpatialMath
    from core_perception.object_tracker import SimpleTracker

    sm = SpatialMath()
    tracker = SimpleTracker()

    # Mỗi frame
    tracks = tracker.update(yolo_detections)          # gán ID
    tracker.update_world_positions(sm, tracks)        # pixel → mét
    # Truyền tracks sang adas_logic.ADASModule
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


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
        Khoảng cách phía trước tính từ camera (mét). Được SpatialMath cập nhật.
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

    def update_world_positions(self, spatial_math, track_list: List[Track]) -> None:
        """
        Cập nhật tọa độ thế giới thực cho mỗi track qua SpatialMath.

        Gọi sau ``update()`` để gắn tọa độ (mét) vào mỗi track, sau đó
        lưu vào ``track.history`` để tính vận tốc tương đối.

        Parameters
        ----------
        spatial_math : SpatialMath
            Module chiếu pixel → mét (từ ``spatial_math.py``).
        track_list : List[Track]
            Danh sách track cần cập nhật (thường là kết quả của ``update()``).
        """
        now = time.monotonic()
        for track in track_list:
            x_m, y_m, valid = spatial_math.bbox_bottom_to_world(
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
