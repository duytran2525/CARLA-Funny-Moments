import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_tracking_metrics import (
    compute_simple_tracking_metrics,
    prepare_trackeval_bundle,
    write_simple_metrics_outputs,
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
            self.assertIn("mota", csv_path.read_text(encoding="utf-8"))
            self.assertIn("id_switches: 1", txt_path.read_text(encoding="utf-8"))

    def test_prepare_trackeval_bundle(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pred = root / "pred.txt"
            gt = root / "gt.txt"
            pred.write_text("1,7,10,10,20,20,0.9,-1,-1,-1\n", encoding="utf-8")
            gt.write_text("1,101,10,10,20,20,1,-1,-1,-1\n", encoding="utf-8")

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

    def test_metrics_dir_helpers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pred = root / "model_tracker_predictions.txt"
            seqinfo = root / "seqinfo.ini"
            pred.write_text("1,7,10,10,20,20,0.9,-1,-1,-1\n", encoding="utf-8")
            seqinfo.write_text("[Sequence]\nname=carla_run\n", encoding="utf-8")

            self.assertEqual(_find_single_prediction_file(root), pred)
            self.assertEqual(_read_seq_name_from_seqinfo(seqinfo), "carla_run")


if __name__ == "__main__":
    unittest.main()
