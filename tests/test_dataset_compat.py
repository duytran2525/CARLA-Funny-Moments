from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from core_perception.dataset import WaypointCarlaDataset


def _write_rgb_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((120, 200, 3), 127, dtype=np.uint8)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to create test image: {path}")


def _base_row() -> dict[str, object]:
    row = {
        "command": 2,
        "recovery_flag": 1,
        "speed": 12.0,
    }
    for idx in range(1, 6):
        row[f"wp_{idx}_x"] = float(idx)
        row[f"wp_{idx}_y"] = float(idx) * 0.1
    return row


class WaypointDatasetCompatibilityTests(unittest.TestCase):
    def test_current_collector_schema_resolves_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for side in ("images_center", "images_left", "images_right"):
                _write_rgb_image(root / side / "00000001.jpg")

            csv_path = root / "driving_log.csv"
            row = _base_row()
            row.update(
                {
                    "image_filename": "00000001.jpg",
                    "image_filename_tm03": "00000001.jpg",
                    "image_filename_tm06": "00000001.jpg",
                    "img_id": "1",
                    "img_id_tm03": "1",
                    "img_id_tm06": "1",
                }
            )

            with csv_path.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)

            dataset = WaypointCarlaDataset(
                csv_file=csv_path,
                root_dir=root,
                is_training=False,
                filter_stationary=False,
            )
            self.assertEqual(len(dataset), 1)
            image_tensor, waypoint_tensor, command_tensor, recovery_tensor = dataset[0]
            self.assertEqual(tuple(image_tensor.shape), (9, 66, 200))
            self.assertEqual(tuple(waypoint_tensor.shape), (5, 2))
            self.assertEqual(int(command_tensor.item()), 2)
            self.assertEqual(float(recovery_tensor.item()), 1.0)
            self.assertEqual(list(dataset.get_recovery_flags()), [1])

    def test_legacy_schema_resolves_center_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            town_dir = root / "Town03"
            for side in ("center", "left", "right"):
                _write_rgb_image(town_dir / side / "00000001.jpg")

            csv_path = root / "legacy.csv"
            row = _base_row()
            row.update(
                {
                    "center_camera": "Town03/center/00000001.jpg",
                    "img_id": "1",
                    "img_id_tm03": "1",
                    "img_id_tm06": "1",
                }
            )

            with csv_path.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)

            dataset = WaypointCarlaDataset(
                csv_file=csv_path,
                root_dir=root,
                is_training=False,
                filter_stationary=False,
            )
            self.assertEqual(len(dataset), 1)
            image_tensor, waypoint_tensor, command_tensor, _recovery_tensor = dataset[0]
            self.assertEqual(tuple(image_tensor.shape), (9, 66, 200))
            self.assertEqual(tuple(waypoint_tensor.shape), (5, 2))
            self.assertEqual(int(command_tensor.item()), 2)


if __name__ == "__main__":
    unittest.main()
