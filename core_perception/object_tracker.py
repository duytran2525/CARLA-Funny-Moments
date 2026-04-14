from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _bbox_to_xywh(bbox_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy.tolist()]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return np.array([cx, cy, w, h], dtype=np.float64)


def _xywh_to_bbox(xywh: np.ndarray) -> np.ndarray:
    cx, cy, w, h = [float(v) for v in xywh.tolist()]
    w = max(1.0, w)
    h = max(1.0, h)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.array([x1, y1, x2, y2], dtype=np.float64)


@dataclass
class _KalmanTrack:
    track_id: int
    class_name: str
    confidence: float
    state: np.ndarray  # [cx,cy,w,h,vx,vy,vw,vh]
    covariance: np.ndarray
    hits: int = 1
    age: int = 1
    miss: int = 0
    last_bbox: Optional[np.ndarray] = None


class KalmanObjectTracker:
    """
    Kalman-filter-based multi-object tracker.

    Input detections:
    [
        {"class": "...", "bbox": [x1,y1,x2,y2], "conf": 0.91},
        ...
    ]

    Output tracked detections:
    [
        {"class": "...", "bbox": [...], "conf": ..., "track_id": N, "raw_bbox": [...]},
        ...
    ]
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_age: int = 30,
        min_hits: int = 1,
        process_noise: float = 1.0,
        measurement_noise: float = 10.0,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_age = max(1, int(max_age))
        self.min_hits = max(1, int(min_hits))
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)

        self._tracks: Dict[int, _KalmanTrack] = {}
        self._next_track_id = 1
        self._last_timestamp: Optional[float] = None

        self._h = np.hstack([np.eye(4, dtype=np.float64), np.zeros((4, 4), dtype=np.float64)])

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
        self._last_timestamp = None

    def update(
        self,
        detections: Sequence[Dict[str, Any]],
        timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        parsed = self._sanitize_detections(detections)
        dt = self._resolve_dt(timestamp)
        f, q = self._build_motion(dt)

        predicted_bbox: Dict[int, np.ndarray] = {}
        for track_id, track in self._tracks.items():
            self._predict_track(track, f, q)
            predicted_bbox[track_id] = _xywh_to_bbox(track.state[:4])

        matches, unmatched_tracks, unmatched_det = self._associate(parsed, predicted_bbox)

        output_map: Dict[int, Dict[str, Any]] = {}
        for det_idx, track_id in matches:
            det = parsed[det_idx]
            track = self._tracks[track_id]
            self._update_track(track, det)
            if track.hits >= self.min_hits:
                output_map[det_idx] = self._build_output(track, det)

        for track_id in unmatched_tracks:
            track = self._tracks.get(track_id)
            if track is None:
                continue
            track.miss += 1
            track.age += 1

        for det_idx in unmatched_det:
            det = parsed[det_idx]
            track = self._create_track(det)
            if track.hits >= self.min_hits:
                output_map[det_idx] = self._build_output(track, det)

        self._drop_stale_tracks()
        return [output_map[idx] for idx in sorted(output_map)]

    def _resolve_dt(self, timestamp: Optional[float]) -> float:
        if timestamp is None:
            self._last_timestamp = None
            return 1.0 / 20.0
        ts = float(timestamp)
        if self._last_timestamp is None:
            self._last_timestamp = ts
            return 1.0 / 20.0
        dt = max(1e-3, ts - self._last_timestamp)
        self._last_timestamp = ts
        return dt

    def _build_motion(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        f = np.eye(8, dtype=np.float64)
        for i in range(4):
            f[i, i + 4] = dt

        q_scale = max(self.process_noise, 1e-6)
        q = np.diag(
            [
                1.0, 1.0, 1.0, 1.0,
                10.0, 10.0, 10.0, 10.0,
            ]
        ).astype(np.float64)
        q = q * q_scale
        return f, q

    @staticmethod
    def _sanitize_detections(detections: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for det in detections:
            bbox = det.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(
                {
                    "class": str(det.get("class", "unknown")),
                    "conf": float(det.get("conf", 0.0)),
                    "bbox": np.array([x1, y1, x2, y2], dtype=np.float64),
                }
            )
        return out

    def _predict_track(self, track: _KalmanTrack, f: np.ndarray, q: np.ndarray) -> None:
        track.state = f @ track.state
        track.covariance = f @ track.covariance @ f.T + q
        track.age += 1

    def _update_track(self, track: _KalmanTrack, det: Dict[str, Any]) -> None:
        z = _bbox_to_xywh(det["bbox"])
        r = np.eye(4, dtype=np.float64) * max(self.measurement_noise, 1e-6)

        y = z - (self._h @ track.state)
        s = self._h @ track.covariance @ self._h.T + r
        try:
            k = track.covariance @ self._h.T @ np.linalg.inv(s)
        except np.linalg.LinAlgError:
            k = track.covariance @ self._h.T @ np.linalg.pinv(s)

        track.state = track.state + k @ y
        ident = np.eye(8, dtype=np.float64)
        track.covariance = (ident - k @ self._h) @ track.covariance

        track.class_name = det["class"]
        track.confidence = 0.7 * float(track.confidence) + 0.3 * float(det["conf"])
        track.hits += 1
        track.miss = 0
        track.last_bbox = det["bbox"].copy()

    def _create_track(self, det: Dict[str, Any]) -> _KalmanTrack:
        xywh = _bbox_to_xywh(det["bbox"])
        state = np.zeros(8, dtype=np.float64)
        state[:4] = xywh
        covariance = np.diag(
            [25.0, 25.0, 25.0, 25.0, 100.0, 100.0, 100.0, 100.0]
        ).astype(np.float64)

        track = _KalmanTrack(
            track_id=self._next_track_id,
            class_name=det["class"],
            confidence=float(det["conf"]),
            state=state,
            covariance=covariance,
            last_bbox=det["bbox"].copy(),
        )
        self._tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    def _associate(
        self,
        detections: Sequence[Dict[str, Any]],
        predicted_bbox: Dict[int, np.ndarray],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        track_ids = list(predicted_bbox.keys())
        if not detections or not track_ids:
            return [], track_ids, list(range(len(detections)))

        candidates: List[Tuple[float, int, int]] = []
        for det_idx, det in enumerate(detections):
            det_bbox = det["bbox"]
            det_class = det["class"]
            for track_id in track_ids:
                track = self._tracks[track_id]
                if track.class_name != det_class:
                    continue
                iou = self._iou(det_bbox, predicted_bbox[track_id])
                if iou >= self.iou_threshold:
                    candidates.append((iou, det_idx, track_id))

        candidates.sort(key=lambda item: item[0], reverse=True)
        matched_det = set()
        matched_track = set()
        matches: List[Tuple[int, int]] = []
        for _, det_idx, track_id in candidates:
            if det_idx in matched_det or track_id in matched_track:
                continue
            matched_det.add(det_idx)
            matched_track.add(track_id)
            matches.append((det_idx, track_id))

        unmatched_tracks = [tid for tid in track_ids if tid not in matched_track]
        unmatched_det = [idx for idx in range(len(detections)) if idx not in matched_det]
        return matches, unmatched_tracks, unmatched_det

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in a.tolist()]
        bx1, by1, bx2, by2 = [float(v) for v in b.tolist()]
        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
        denom = area_a + area_b - inter
        if denom <= 1e-6:
            return 0.0
        return float(inter / denom)

    def _drop_stale_tracks(self) -> None:
        stale = [
            track_id
            for track_id, track in self._tracks.items()
            if track.miss > self.max_age
        ]
        for track_id in stale:
            self._tracks.pop(track_id, None)

    @staticmethod
    def _bbox_to_int_list(bbox: np.ndarray) -> List[int]:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox.tolist()]
        if x2 <= x1:
            x2 = x1 + 1
        if y2 <= y1:
            y2 = y1 + 1
        return [x1, y1, x2, y2]

    def _build_output(self, track: _KalmanTrack, det: Dict[str, Any]) -> Dict[str, Any]:
        bbox_pred = _xywh_to_bbox(track.state[:4])
        return {
            "class": track.class_name,
            "bbox": self._bbox_to_int_list(bbox_pred),
            "conf": float(det["conf"]),
            "track_id": int(track.track_id),
            "raw_bbox": self._bbox_to_int_list(det["bbox"]),
        }


# Compatibility alias for old references.
OCSortTracker = KalmanObjectTracker

