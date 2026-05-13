import csv
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_tracking_metrics import (
    EVALUATION_CLASS_ORDER,
    copy_run_context_files,
    compute_simple_tracking_metrics,
    prepare_trackeval_bundle,
    write_simple_metrics_outputs,
    _classes_in_mot_files,
    _find_single_prediction_file,
    _read_seq_name_from_seqinfo,
)


class EvaluateTrackingMetricsTests(unittest.TestCase):
    def test_simple_metrics_and_output_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pred = root / "pred.txt"
            gt = root / "gt.txt"
            out_dir = root / "out"

            gt.write_text(
                "\n".join(
                    [
                        "1,101,10,10,20,20,1,-1,-1,-1",
                        "2,101,12,10,20,20,1,-1,-1,-1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            pred.write_text(
                "\n".join(
                    [
                        "1,7,10,10,20,20,0.9,-1,-1,-1",
                        "2,8,12,10,20,20,0.9,-1,-1,-1",
                        "2,9,200,200,10,10,0.4,-1,-1,-1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            metrics = compute_simple_tracking_metrics(pred, gt, iou_threshold=0.5)
            csv_path, txt_path = write_simple_metrics_outputs(metrics, out_dir)

            self.assertEqual(metrics["true_positives"], 2)
            self.assertEqual(metrics["false_positives"], 1)
            self.assertEqual(metrics["false_negatives"], 0)
            self.assertEqual(metrics["id_switches"], 1)
            self.assertTrue(csv_path.exists())
            self.assertTrue(txt_path.exists())
            self.assertTrue((out_dir / "tracking_metrics_per_class.csv").exists())
            self.assertIn("mota", csv_path.read_text(encoding="utf-8"))
            self.assertIn("id_switches: 1", txt_path.read_text(encoding="utf-8"))

    def test_simple_metrics_requires_matching_class_when_present(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pred = root / "pred.txt"
            gt = root / "gt.txt"

            gt.write_text("1,101,10,10,20,20,1,-1,-1,-1,pedestrian\n", encoding="utf-8")
            pred.write_text("1,7,10,10,20,20,0.9,-1,-1,-1,vehicle\n", encoding="utf-8")

            metrics = compute_simple_tracking_metrics(pred, gt, iou_threshold=0.5)

            self.assertEqual(metrics["class_match_mode"], "strict")
            self.assertEqual(metrics["true_positives"], 0)
            self.assertEqual(metrics["false_positives"], 1)
            self.assertEqual(metrics["false_negatives"], 1)
            self.assertEqual(metrics["class_pedestrian_false_negatives"], 1)
            self.assertEqual(metrics["class_vehicle_false_positives"], 1)
            csv_path, txt_path = write_simple_metrics_outputs(metrics, root / "out")
            self.assertIn("class_vehicle_false_positives", csv_path.read_text(encoding="utf-8"))
            self.assertIn("class=pedestrian", txt_path.read_text(encoding="utf-8"))

    def test_per_class_outputs_include_all_eval_classes_in_requested_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pred = root / "pred.txt"
            gt = root / "gt.txt"
            out_dir = root / "out"

            gt.write_text("1,101,10,10,20,20,1,-1,-1,-1,vehicle\n", encoding="utf-8")
            pred.write_text("1,7,10,10,20,20,0.9,-1,-1,-1,vehicle\n", encoding="utf-8")

            metrics = compute_simple_tracking_metrics(pred, gt, iou_threshold=0.5)
            _csv_path, txt_path = write_simple_metrics_outputs(metrics, out_dir)

            per_class_csv = out_dir / "tracking_metrics_per_class.csv"
            with per_class_csv.open("r", newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual([row["class"] for row in rows[: len(EVALUATION_CLASS_ORDER)]], EVALUATION_CLASS_ORDER)
            self.assertEqual(rows[1]["class"], "two_wheeler")
            self.assertEqual(rows[1]["gt_detections"], "0")
            summary_text = txt_path.read_text(encoding="utf-8")
            positions = [summary_text.index(f"class={class_name}") for class_name in EVALUATION_CLASS_ORDER]
            self.assertEqual(positions, sorted(positions))

    def test_classes_in_mot_files_uses_requested_order_and_aliases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pred = root / "pred.txt"
            gt = root / "gt.txt"
            pred.write_text(
                "\n".join(
                    [
                        "1,7,10,10,20,20,0.9,-1,-1,-1,stopline",
                        "1,8,10,10,20,20,0.9,-1,-1,-1,bike",
                        "1,9,10,10,20,20,0.9,-1,-1,-1,traffic-light-red",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            gt.write_text("1,101,10,10,20,20,1,-1,-1,-1,pedestrian\n", encoding="utf-8")

            # two_wheeler and pedestrian are in EVALUATION_CLASS_ORDER and sort first;
            # traffic_light_red and stop_line are now outside and sort alphabetically after.
            self.assertEqual(
                _classes_in_mot_files(pred, gt),
                ["two_wheeler", "pedestrian", "stop_line", "traffic_light_red"],
            )

    def test_prepare_trackeval_bundle(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pred = root / "pred.txt"
            gt = root / "gt.txt"
            pred.write_text("1,7,10,10,20,20,0.9,-1,-1,-1,vehicle\n", encoding="utf-8")
            gt.write_text("1,101,10,10,20,20,1,-1,-1,-1,vehicle\n", encoding="utf-8")

            data_root = prepare_trackeval_bundle(
                predictions_txt=pred,
                ground_truth_txt=gt,
                output_dir=root / "eval",
                seq_name="SEQ",
                tracker_name="TRK",
                benchmark_name="BENCH",
            )

            self.assertTrue((data_root / "gt" / "mot_challenge" / "BENCH" / "SEQ" / "gt" / "gt.txt").exists())
            self.assertTrue((data_root / "trackers" / "mot_challenge" / "BENCH" / "TRK" / "data" / "SEQ.txt").exists())
            self.assertTrue((data_root / "gt" / "mot_challenge" / "seqmaps" / "BENCH-all.txt").exists())
            trackeval_pred = data_root / "trackers" / "mot_challenge" / "BENCH" / "TRK" / "data" / "SEQ.txt"
            self.assertEqual(len(trackeval_pred.read_text(encoding="utf-8").strip().split(",")), 10)

    def test_prepare_trackeval_bundle_filters_class_and_rewrites_seqinfo(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pred = root / "pred.txt"
            gt = root / "gt.txt"
            seqinfo = root / "seqinfo.ini"
            pred.write_text(
                "\n".join(
                    [
                        "1,7,10,10,20,20,0.9,-1,-1,-1,vehicle",
                        "1,8,50,50,20,20,0.9,-1,-1,-1,pedestrian",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            gt.write_text(
                "\n".join(
                    [
                        "1,101,10,10,20,20,1,-1,-1,-1,vehicle",
                        "1,102,50,50,20,20,1,-1,-1,-1,pedestrian",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            seqinfo.write_text("[Sequence]\nname=old_seq\nseqLength=1\n", encoding="utf-8")

            data_root = prepare_trackeval_bundle(
                predictions_txt=pred,
                ground_truth_txt=gt,
                output_dir=root / "eval_vehicle",
                seq_name="vehicle_seq",
                tracker_name="TRK_vehicle",
                benchmark_name="BENCH",
                seqinfo_ini=seqinfo,
                class_filter="vehicle",
            )

            trackeval_pred = data_root / "trackers" / "mot_challenge" / "BENCH" / "TRK_vehicle" / "data" / "vehicle_seq.txt"
            trackeval_pred_text = trackeval_pred.read_text(encoding="utf-8")
            self.assertEqual(trackeval_pred_text.count("\n"), 1)
            self.assertNotIn("pedestrian", trackeval_pred_text)
            pred_fields = trackeval_pred_text.strip().split(",")
            self.assertEqual(len(pred_fields), 10)
            self.assertEqual(pred_fields[7], "1")
            rewritten_seqinfo = data_root / "gt" / "mot_challenge" / "BENCH" / "vehicle_seq" / "seqinfo.ini"
            self.assertIn("name=vehicle_seq", rewritten_seqinfo.read_text(encoding="utf-8"))

    def test_prepare_trackeval_per_class_separates_class_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pred = root / "pred.txt"
            gt = root / "gt.txt"
            pred.write_text("1,7,10,10,20,20,0.9,-1,-1,-1,vehicle\n", encoding="utf-8")
            gt.write_text("1,101,10,10,20,20,1,-1,-1,-1,pedestrian\n", encoding="utf-8")

            vehicle_root = prepare_trackeval_bundle(
                predictions_txt=pred,
                ground_truth_txt=gt,
                output_dir=root / "eval_vehicle",
                seq_name="vehicle_seq",
                tracker_name="TRK_vehicle",
                benchmark_name="BENCH",
                class_filter="vehicle",
            )
            vehicle_gt = vehicle_root / "gt" / "mot_challenge" / "BENCH" / "vehicle_seq" / "gt" / "gt.txt"
            vehicle_pred = vehicle_root / "trackers" / "mot_challenge" / "BENCH" / "TRK_vehicle" / "data" / "vehicle_seq.txt"
            self.assertEqual(vehicle_gt.read_text(encoding="utf-8"), "")
            self.assertEqual(vehicle_pred.read_text(encoding="utf-8").count("\n"), 1)

            pedestrian_root = prepare_trackeval_bundle(
                predictions_txt=pred,
                ground_truth_txt=gt,
                output_dir=root / "eval_pedestrian",
                seq_name="pedestrian_seq",
                tracker_name="TRK_pedestrian",
                benchmark_name="BENCH",
                class_filter="pedestrian",
            )
            pedestrian_gt = pedestrian_root / "gt" / "mot_challenge" / "BENCH" / "pedestrian_seq" / "gt" / "gt.txt"
            pedestrian_pred = pedestrian_root / "trackers" / "mot_challenge" / "BENCH" / "TRK_pedestrian" / "data" / "pedestrian_seq.txt"
            self.assertEqual(pedestrian_gt.read_text(encoding="utf-8").count("\n"), 1)
            self.assertEqual(pedestrian_pred.read_text(encoding="utf-8"), "")

    def test_metrics_dir_helpers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pred = root / "model_tracker_predictions.txt"
            seqinfo = root / "seqinfo.ini"
            pred.write_text("1,7,10,10,20,20,0.9,-1,-1,-1\n", encoding="utf-8")
            seqinfo.write_text("[Sequence]\nname=carla_run\n", encoding="utf-8")

            self.assertEqual(_find_single_prediction_file(root), pred)
            self.assertEqual(_read_seq_name_from_seqinfo(seqinfo), "carla_run")

    def test_copy_run_context_files_from_pred_gt_parent(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "run"
            out_dir = root / "out"
            run_dir.mkdir()
            pred = run_dir / "model_tracker_predictions.txt"
            gt = run_dir / "ground_truth.txt"
            metadata = run_dir / "run_metadata.json"
            pred.write_text("1,7,10,10,20,20,0.9,-1,-1,-1,vehicle\n", encoding="utf-8")
            gt.write_text("1,101,10,10,20,20,1,-1,-1,-1,vehicle\n", encoding="utf-8")
            metadata.write_text('{"agent":"yolo_detect"}\n', encoding="utf-8")

            copy_run_context_files(out_dir, pred_path=pred, gt_path=gt)

            self.assertEqual((out_dir / "run_metadata.json").read_text(encoding="utf-8"), metadata.read_text(encoding="utf-8"))
            self.assertEqual((out_dir / "ground_truth.txt").read_text(encoding="utf-8"), gt.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
