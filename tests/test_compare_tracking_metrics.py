import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_tracking_metrics import (
    compare_fairness_rows,
    load_tracking_run,
    write_comparison_outputs,
)


def _write_metrics_dir(root: Path, tracker: str, pred_lines: list[str], metadata_overrides: dict | None = None) -> Path:
    metrics_dir = root / tracker
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "ground_truth.txt").write_text(
        "\n".join(
            [
                "1,101,10,10,20,20,1,-1,-1,-1",
                "2,101,12,10,20,20,1,-1,-1,-1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (metrics_dir / f"model_{tracker}_tracker_predictions.txt").write_text(
        "\n".join(pred_lines) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "agent": "yolo_detect",
        "map_name": "Town03",
        "sync": True,
        "fixed_delta": 0.05,
        "no_rendering": False,
        "seed": 42,
        "ticks": 100,
        "weather_preset": "ClearNoon",
        "vehicle_filter": "vehicle.tesla.model3",
        "configured_spawn_point": 1,
        "actual_initial_spawn_point": 1,
        "configured_destination_point": 10,
        "actual_route_destination_point": 10,
        "target_speed_kmh": 30.0,
        "tm_port": 8000,
        "npc_vehicle_count": 5,
        "npc_bike_count": 0,
        "npc_motorbike_count": 0,
        "npc_pedestrian_count": 0,
        "npc_enable_autopilot": True,
        "camera_width": 800,
        "camera_height": 600,
        "camera_fov": 90.0,
        "camera_mount": {"x_m": 1.5, "y_m": 0.0, "z_m": 2.2, "pitch_deg": -8.0},
        "yolo_backend": "planner",
        "yolo_nav_agent_type": "basic",
        "resolved_yolo_model_path": "D:/AI/CARLA-Funny-Moments/best.pt",
        "yolo_inference_imgsz": 448,
        "yolo_inference_every_n_ticks": 1,
        "yolo_tracker_config": f"{tracker}.yaml",
        "detector_uses_depth_input": False,
        "gt_occlusion_filter": {
            "enabled": True,
            "method": "projected_3d_bbox_visible_area_plus_depth_map",
            "min_visible_area_ratio": 0.20,
            "min_depth_visible_ratio": 0.20,
        },
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    (metrics_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics_dir


class CompareTrackingMetricsTests(unittest.TestCase):
    def test_compare_raw_metrics_dirs_and_fairness_pass(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            left_dir = _write_metrics_dir(
                root,
                "botsort",
                [
                    "1,7,10,10,20,20,0.9,-1,-1,-1",
                    "2,8,200,200,10,10,0.4,-1,-1,-1",
                ],
            )
            right_dir = _write_metrics_dir(
                root,
                "bytetrack",
                [
                    "1,7,10,10,20,20,0.9,-1,-1,-1",
                    "2,7,12,10,20,20,0.9,-1,-1,-1",
                ],
            )

            left = load_tracking_run(left_dir, name="BoTSORT")
            right = load_tracking_run(right_dir, name="ByteTrack")
            metrics_csv, fairness_csv, summary_txt = write_comparison_outputs(left, right, root / "compare")

            self.assertTrue(metrics_csv.exists())
            self.assertTrue(fairness_csv.exists())
            self.assertTrue(summary_txt.exists())
            self.assertIn("f1", metrics_csv.read_text(encoding="utf-8"))
            self.assertIn("fairness_check: PASS", summary_txt.read_text(encoding="utf-8"))

    def test_fairness_detects_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            left_dir = _write_metrics_dir(
                root,
                "botsort",
                ["1,7,10,10,20,20,0.9,-1,-1,-1"],
            )
            right_dir = _write_metrics_dir(
                root,
                "bytetrack",
                ["1,7,10,10,20,20,0.9,-1,-1,-1"],
                metadata_overrides={"map_name": "Town05"},
            )

            left = load_tracking_run(left_dir, name="BoTSORT")
            right = load_tracking_run(right_dir, name="ByteTrack")
            rows = compare_fairness_rows(left, right)

            map_row = next(row for row in rows if row["key"] == "map_name")
            self.assertEqual(map_row["status"], "mismatch")

    def test_tm_backend_missing_actual_destination_is_not_applicable(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            left_dir = _write_metrics_dir(
                root,
                "botsort",
                ["1,7,10,10,20,20,0.9,-1,-1,-1"],
                metadata_overrides={
                    "yolo_backend": "tm",
                    "actual_route_destination_point": None,
                },
            )
            right_dir = _write_metrics_dir(
                root,
                "bytetrack",
                ["1,7,10,10,20,20,0.9,-1,-1,-1"],
                metadata_overrides={
                    "yolo_backend": "tm",
                    "actual_route_destination_point": None,
                },
            )

            left = load_tracking_run(left_dir, name="BoTSORT")
            right = load_tracking_run(right_dir, name="ByteTrack")
            rows = compare_fairness_rows(left, right)

            destination_row = next(row for row in rows if row["key"] == "actual_route_destination_point")
            self.assertEqual(destination_row["BoTSORT"], "not_applicable_tm_backend")
            self.assertEqual(destination_row["ByteTrack"], "not_applicable_tm_backend")
            self.assertEqual(destination_row["status"], "match")

    def test_fairness_detects_different_ground_truth_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            left_dir = _write_metrics_dir(
                root,
                "botsort",
                ["1,7,10,10,20,20,0.9,-1,-1,-1"],
            )
            right_dir = _write_metrics_dir(
                root,
                "bytetrack",
                ["1,7,10,10,20,20,0.9,-1,-1,-1"],
            )
            (right_dir / "ground_truth.txt").write_text(
                "1,101,10,10,20,20,1,-1,-1,-1\n",
                encoding="utf-8",
            )

            left = load_tracking_run(left_dir, name="BoTSORT")
            right = load_tracking_run(right_dir, name="ByteTrack")
            rows = compare_fairness_rows(left, right)

            sha_row = next(row for row in rows if row["key"] == "metrics_raw.ground_truth_sha256")
            rows_row = next(row for row in rows if row["key"] == "metrics_raw.ground_truth_rows")
            self.assertEqual(sha_row["status"], "mismatch")
            self.assertEqual(rows_row["status"], "mismatch")


if __name__ == "__main__":
    unittest.main()
