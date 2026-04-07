from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence


DEFAULT_CAMERA_SIDES: tuple[str, str, str] = ("center", "left", "right")


def normalize_camera_side(side: str) -> str:
    side_norm = side.strip().lower()
    if side_norm not in DEFAULT_CAMERA_SIDES:
        raise ValueError(f"Invalid camera side '{side}'. Expected center/left/right.")
    return side_norm


def build_synchronized_data(
    frame_id: int,
    pending_images: Mapping[int, Mapping[str, Any]],
    pending_states: Mapping[int, Mapping[str, Any]],
    required_sides: Sequence[str] = DEFAULT_CAMERA_SIDES,
) -> Optional[Dict[str, Any]]:
    images = pending_images.get(frame_id)
    state = pending_states.get(frame_id)
    if images is None or state is None:
        return None

    if not all(side in images for side in required_sides):
        return None

    return {
        "frame_id": frame_id,
        "images": {side: images[side] for side in required_sides},
        "state": dict(state),
    }
