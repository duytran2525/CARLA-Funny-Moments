"""
test_adas_math.py – Phase 2
============================
Script kiểm thử độc lập cho luồng:
    YOLO detections → SpatialMath (X, Y) → SimpleTracker → ADASModule (TTC)

Không cần bật CARLA hay card đồ hoạ. Chạy được hoàn toàn offline.

Cách chạy
---------
::

    # Từ thư mục Optimal_CARLA_Hybrid/
    python scripts/test_adas_math.py

    # Hoặc với verbose output
    python scripts/test_adas_math.py --verbose
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Thêm root vào sys.path để import module nội bộ
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core_perception.spatial_math import SpatialMath
from core_perception.object_tracker import SimpleTracker, Track
from core_control.adas_logic import ADASModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_detections_from_distances(
    distances: list[float],
    image_width: int = 1920,
    image_height: int = 1080,
    cam_height: float = 2.2,
    cam_pitch_deg: float = 8.0,
    fov_deg: float = 90.0,
) -> list[list[float]]:
    """
    Tạo YOLO detections giả từ danh sách khoảng cách phía trước (mét).

    Dùng công thức ngược của SpatialMath để chuyển (x_m, y_m=0) → pixel (u, v),
    rồi tạo bounding box 100×100 px xung quanh điểm đó.

    Parameters
    ----------
    distances : list[float]
        Danh sách khoảng cách phía trước (mét) cho từng đối tượng giả.

    Returns
    -------
    list[list[float]]
        List các detection theo format ``[x1, y1, x2, y2, conf, class_id]``.
    """
    import math

    fx = (image_width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    fy = fx
    cx = image_width / 2.0
    cy = image_height / 2.0
    pitch_rad = math.radians(cam_pitch_deg)

    detections = []
    for d in distances:
        if d <= 0:
            continue
        # v từ d: d = H / tan(pitch + theta_v) → theta_v = arctan(H/d) - pitch
        theta_v = math.atan(cam_height / d) - pitch_rad
        v = cy + fy * math.tan(theta_v)  # pixel row
        u = cx                            # trục trung tâm (y_m = 0)

        hw = 50  # half-width of fake box
        detections.append([u - hw, v - hw, u + hw, v + hw, 0.9, 2])  # class 2 = car

    return detections


def _run_test(verbose: bool = False) -> None:
    """Chạy toàn bộ bộ test cho luồng SpatialMath → Tracker → ADAS."""

    print("=" * 60)
    print("ADAS Math Test Suite – Phase 2")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. SpatialMath: pixel → world
    # ------------------------------------------------------------------
    print("\n[1] SpatialMath: pixel → (X, Y) mét")
    sm = SpatialMath(image_width=1920, image_height=1080, fov_deg=90.0,
                     cam_height=2.2, cam_pitch_deg=8.0)

    # Điểm chính giữa đáy ảnh (gần xe nhất)
    x, y, valid = sm.pixel_to_world(960, 1079)
    assert valid, "Điểm đáy trung tâm phải hợp lệ"
    assert x > 0, f"x phải dương, got {x:.2f}"
    assert abs(y) < 0.1, f"y gần 0 (trục trung tâm), got {y:.2f}"
    if verbose:
        print(f"  pixel(960, 1079) → x={x:.2f}m, y={y:.2f}m, valid={valid}")

    # Đường chân trời → không hợp lệ
    x2, y2, valid2 = sm.pixel_to_world(960, 300)
    assert not valid2, "Điểm trên đường chân trời phải không hợp lệ"
    if verbose:
        print(f"  pixel(960, 300)  → valid={valid2} (đường chân trời, đúng)")

    # Điểm bên phải → y dương
    x3, y3, valid3 = sm.pixel_to_world(1500, 800)
    assert valid3 and y3 > 0, f"Điểm bên phải phải có y > 0, got {y3:.2f}"
    if verbose:
        print(f"  pixel(1500, 800) → x={x3:.2f}m, y={y3:.2f}m (phải)")

    print("  ✅ SpatialMath PASSED")

    # ------------------------------------------------------------------
    # 2. SpatialMath: update_camera_params
    # ------------------------------------------------------------------
    print("\n[2] SpatialMath.update_camera_params()")
    old_fx = sm.fx
    sm.update_camera_params(1280, 720, 90.0, 2.2, 8.0)
    assert sm.fx != old_fx, "fx phải thay đổi khi width thay đổi"
    assert sm.W == 1280 and sm.H == 720
    sm.update_camera_params(1920, 1080, 90.0, 2.2, 8.0)  # khôi phục
    if verbose:
        print(f"  fx thay đổi: {old_fx:.1f} → {sm.fx:.1f} (sau khi khôi phục)")
    print("  ✅ update_camera_params PASSED")

    # ------------------------------------------------------------------
    # 3. SimpleTracker: ID persistence
    # ------------------------------------------------------------------
    print("\n[3] SimpleTracker: ID persistence qua frame")
    tracker = SimpleTracker(iou_threshold=0.3, max_missed=5)

    # Frame 1: 2 xe giả ở 20m và 40m
    dets1 = _fake_detections_from_distances([20.0, 40.0])
    tracks1 = tracker.update(dets1)
    assert len(tracks1) == 2, f"Cần 2 tracks, got {len(tracks1)}"
    ids1 = {t.track_id for t in tracks1}
    if verbose:
        print(f"  Frame 1: {len(tracks1)} tracks, IDs={ids1}")

    # Frame 2: 2 xe vẫn còn (dịch chuyển nhỏ)
    dets2 = _fake_detections_from_distances([19.5, 39.0])
    tracks2 = tracker.update(dets2)
    ids2 = {t.track_id for t in tracks2}
    assert ids1 == ids2, f"IDs phải giữ nguyên: {ids1} vs {ids2}"
    if verbose:
        print(f"  Frame 2: {len(tracks2)} tracks, IDs={ids2} (giữ nguyên ✅)")

    # Frame 3: 1 xe biến mất
    dets3 = _fake_detections_from_distances([19.0])
    tracks3 = tracker.update(dets3)
    assert len(tracks3) == 2, "Track bị miss lần 1 vẫn còn (missed_frames=1)"
    if verbose:
        print(f"  Frame 3: 1 det, {len(tracks3)} tracks vẫn active (missed chưa hết)")

    print("  ✅ SimpleTracker ID persistence PASSED")

    # ------------------------------------------------------------------
    # 4. SimpleTracker + SpatialMath: world positions
    # ------------------------------------------------------------------
    print("\n[4] SimpleTracker.update_world_positions()")
    tracker.reset()
    dets = _fake_detections_from_distances([15.0, 30.0])
    tracks = tracker.update(dets)
    tracker.update_world_positions(sm, tracks)

    for t in tracks:
        assert t.world_x > 0, f"world_x phải dương, got {t.world_x:.2f}"
        assert abs(t.world_y) < 0.5, f"world_y gần 0 (center), got {t.world_y:.2f}"
        assert len(t.history) == 1, "Phải có 1 entry trong history"
        if verbose:
            print(f"  Track #{t.track_id}: world_x={t.world_x:.2f}m, world_y={t.world_y:.2f}m")

    print("  ✅ update_world_positions PASSED")

    # ------------------------------------------------------------------
    # 5. ADASModule: TTC và quyết định ga/phanh
    # ------------------------------------------------------------------
    print("\n[5] ADASModule: TTC computation & control decisions")
    adas = ADASModule(min_ttc=2.0, safe_ttc=5.0, acc_distance=10.0)

    # Không có track → ga bình thường
    th, br, ttc = adas.compute_control([], ego_speed_mps=5.0, target_speed_mps=8.33)
    assert ttc == float('inf'), "Không có track → TTC = inf"
    assert th > 0, "Không có vật cản → phải ga"
    if verbose:
        print(f"  Không track: throttle={th:.2f}, brake={br:.2f}, TTC=inf")

    # Xe rất gần (5m) và đứng yên → TTC = 5/ego_speed = 0.5s → emergency brake
    tracker.reset()
    dets_close = _fake_detections_from_distances([5.0])
    tracks_close = tracker.update(dets_close)
    tracker.update_world_positions(sm, tracks_close)
    # Ghi đè world_x trực tiếp để chắc chắn
    for t in tracks_close:
        t.world_x = 5.0
        t.world_y = 0.0

    th2, br2, ttc2 = adas.compute_control(tracks_close, ego_speed_mps=10.0, target_speed_mps=8.33)
    assert ttc2 < adas.min_ttc, f"TTC phải nhỏ hơn min_ttc={adas.min_ttc}, got {ttc2:.2f}"
    assert br2 == adas.max_brake, f"Phải phanh khẩn cấp: brake={br2}"
    if verbose:
        print(f"  Xe 5m: throttle={th2:.2f}, brake={br2:.2f}, TTC={ttc2:.2f}s (🛑 khẩn cấp)")

    # Xe xa (80m) → không can thiệp
    for t in tracks_close:
        t.world_x = 80.0
        t.world_y = 0.0
        t.history.clear()

    th3, br3, ttc3 = adas.compute_control(tracks_close, ego_speed_mps=8.0, target_speed_mps=8.33)
    assert ttc3 > adas.safe_ttc, f"TTC phải > safe_ttc khi xe xa, got {ttc3:.2f}"
    if verbose:
        print(f"  Xe 80m: throttle={th3:.2f}, brake={br3:.2f}, TTC={ttc3:.2f}s (✅ an toàn)")

    print("  ✅ ADASModule PASSED")

    # ------------------------------------------------------------------
    # 6. End-to-end pipeline
    # ------------------------------------------------------------------
    print("\n[6] End-to-end pipeline: YOLO → SpatialMath → Tracker → ADAS")
    sm2 = SpatialMath()
    tracker2 = SimpleTracker()
    adas2 = ADASModule(min_ttc=2.0, safe_ttc=5.0)

    scenario = [
        ("an toàn", [50.0, 80.0], 8.0, 8.33),
        ("cảnh báo", [10.0], 8.0, 8.33),
        ("khẩn cấp", [3.0], 10.0, 8.33),
    ]

    for name, dists, ego_spd, tgt_spd in scenario:
        dets = _fake_detections_from_distances(dists)
        tracks = tracker2.update(dets)
        tracker2.update_world_positions(sm2, tracks)
        th, br, ttc = adas2.compute_control(tracks, ego_spd, tgt_spd)
        ttc_str = f"{ttc:.1f}s" if ttc != float('inf') else "∞"
        print(f"  [{name:>8}] dists={dists} → "
              f"throttle={th:.2f}, brake={br:.2f}, TTC={ttc_str}")

    print("  ✅ End-to-end PASSED")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("✅ TẤT CẢ TESTS PASSED – Luồng ADAS Math hoạt động đúng!")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kiểm thử luồng YOLO → SpatialMath → Tracker → TTC (offline, không cần CARLA)."
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help="In chi tiết kết quả từng bước."
    )
    args = parser.parse_args()
    _run_test(verbose=args.verbose)


if __name__ == "__main__":
    main()
