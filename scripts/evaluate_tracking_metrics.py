from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


MotRows = Dict[int, List[dict]]


def _max_frame_id_from_mot_txt(path: Path) -> int:
    max_frame = 0
    if not path.exists():
        return max_frame
    with path.open("r", encoding="utf-8") as file_obj:
        for raw in file_obj:
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 1:
                continue
            try:
                frame_id = int(float(parts[0]))
            except Exception:
                continue
            if frame_id > max_frame:
                max_frame = frame_id
    return max_frame


def _read_seq_name_from_seqinfo(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            for raw in file_obj:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("name="):
                    value = line.split("=", 1)[1].strip()
                    return value or None
    except Exception:
        return None
    return None


def _find_single_prediction_file(metrics_dir: Path) -> Optional[Path]:
    candidates = sorted(metrics_dir.glob("*_tracker_predictions.txt"))
    if not candidates:
        candidates = sorted(metrics_dir.glob("*pred*.txt"))
    if not candidates:
        return None
    return candidates[0]


def _write_default_seqinfo(path: Path, seq_name: str, seq_length: int) -> None:
    # Width/height here are placeholders used only when no seqinfo.ini was captured.
    path.write_text(
        "\n".join(
            [
                "[Sequence]",
                f"name={seq_name}",
                "imDir=img1",
                "frameRate=20",
                f"seqLength={max(1, int(seq_length))}",
                "imWidth=1",
                "imHeight=1",
                "imExt=.jpg",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _read_mot_txt(path: Path, ignore_negative_ids: bool = True) -> MotRows:
    rows_by_frame: MotRows = {}
    if not path.exists():
        return rows_by_frame

    with path.open("r", encoding="utf-8") as file_obj:
        for raw in file_obj:
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                frame_id = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6]) if len(parts) > 6 else 1.0
            except Exception:
                continue
            if frame_id <= 0 or w <= 0.0 or h <= 0.0:
                continue
            if ignore_negative_ids and track_id < 0:
                continue
            rows_by_frame.setdefault(frame_id, []).append(
                {
                    "id": track_id,
                    "bbox": (x, y, w, h),
                    "conf": conf,
                }
            )
    return rows_by_frame


def _copy_mot_txt_for_trackeval(source: Path, target: Path, ignore_negative_ids: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as source_obj, target.open("w", encoding="utf-8") as target_obj:
        for raw in source_obj:
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if ignore_negative_ids:
                if len(parts) < 2:
                    continue
                try:
                    track_id = int(float(parts[1]))
                except Exception:
                    continue
                if track_id < 0:
                    continue
            target_obj.write(line + "\n")


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def compute_simple_tracking_metrics(
    predictions_txt: Path,
    ground_truth_txt: Path,
    iou_threshold: float = 0.5,
) -> dict:
    predictions = _read_mot_txt(predictions_txt)
    ground_truth = _read_mot_txt(ground_truth_txt)

    frames = sorted(set(predictions.keys()) | set(ground_truth.keys()))
    total_gt = 0
    total_pred = 0
    tp = 0
    fp = 0
    fn = 0
    id_switches = 0
    iou_sum = 0.0
    last_match_by_gt_id: Dict[int, int] = {}

    threshold = float(iou_threshold)
    for frame_id in frames:
        frame_gt = ground_truth.get(frame_id, [])
        frame_pred = predictions.get(frame_id, [])
        total_gt += len(frame_gt)
        total_pred += len(frame_pred)

        candidate_pairs = []
        for gt_idx, gt in enumerate(frame_gt):
            for pred_idx, pred in enumerate(frame_pred):
                iou = _bbox_iou(gt["bbox"], pred["bbox"])
                if iou >= threshold:
                    candidate_pairs.append((iou, gt_idx, pred_idx))
        candidate_pairs.sort(reverse=True, key=lambda item: item[0])

        matched_gt = set()
        matched_pred = set()
        for iou, gt_idx, pred_idx in candidate_pairs:
            if gt_idx in matched_gt or pred_idx in matched_pred:
                continue
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            tp += 1
            iou_sum += float(iou)

            gt_id = int(frame_gt[gt_idx]["id"])
            pred_id = int(frame_pred[pred_idx]["id"])
            previous_pred_id = last_match_by_gt_id.get(gt_id)
            if previous_pred_id is not None and previous_pred_id != pred_id:
                id_switches += 1
            last_match_by_gt_id[gt_id] = pred_id

        fp += len(frame_pred) - len(matched_pred)
        fn += len(frame_gt) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    mota = 1.0 - ((fn + fp + id_switches) / total_gt) if total_gt > 0 else 0.0
    motp = iou_sum / tp if tp > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "method": "simple_iou_greedy",
        "iou_threshold": threshold,
        "frames": len(frames),
        "gt_detections": total_gt,
        "pred_detections": total_pred,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "id_switches": id_switches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mota": mota,
        "motp": motp,
    }


def write_simple_metrics_outputs(metrics: dict, output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tracking_metrics_summary.csv"
    txt_path = output_dir / "tracking_metrics_summary.txt"

    fieldnames = [
        "method",
        "iou_threshold",
        "frames",
        "gt_detections",
        "pred_detections",
        "true_positives",
        "false_positives",
        "false_negatives",
        "id_switches",
        "precision",
        "recall",
        "f1",
        "mota",
        "motp",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: metrics.get(key, "") for key in fieldnames})

    lines = [
        "Tracking Metrics Summary",
        f"method: {metrics['method']}",
        f"iou_threshold: {metrics['iou_threshold']:.3f}",
        f"frames: {metrics['frames']}",
        f"gt_detections: {metrics['gt_detections']}",
        f"pred_detections: {metrics['pred_detections']}",
        f"true_positives: {metrics['true_positives']}",
        f"false_positives: {metrics['false_positives']}",
        f"false_negatives: {metrics['false_negatives']}",
        f"id_switches: {metrics['id_switches']}",
        f"precision: {metrics['precision']:.6f}",
        f"recall: {metrics['recall']:.6f}",
        f"f1: {metrics['f1']:.6f}",
        f"mota: {metrics['mota']:.6f}",
        f"motp: {metrics['motp']:.6f}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, txt_path


def prepare_trackeval_bundle(
    predictions_txt: Path,
    ground_truth_txt: Path,
    output_dir: Path,
    seq_name: str,
    tracker_name: str,
    benchmark_name: str,
    seqinfo_ini: Optional[Path] = None,
) -> Path:
    data_root = output_dir / "trackeval_data"
    gt_root = data_root / "gt" / "mot_challenge"
    trackers_root = data_root / "trackers" / "mot_challenge"

    gt_seq_dir = gt_root / benchmark_name / seq_name
    gt_seq_gt_dir = gt_seq_dir / "gt"
    tracker_data_dir = trackers_root / benchmark_name / tracker_name / "data"
    seqmaps_dir = gt_root / "seqmaps"

    gt_seq_gt_dir.mkdir(parents=True, exist_ok=True)
    tracker_data_dir.mkdir(parents=True, exist_ok=True)
    seqmaps_dir.mkdir(parents=True, exist_ok=True)

    gt_target = gt_seq_gt_dir / "gt.txt"
    pred_target = tracker_data_dir / f"{seq_name}.txt"
    _copy_mot_txt_for_trackeval(ground_truth_txt, gt_target)
    _copy_mot_txt_for_trackeval(predictions_txt, pred_target, ignore_negative_ids=True)

    seqinfo_target = gt_seq_dir / "seqinfo.ini"
    if seqinfo_ini is not None and seqinfo_ini.exists():
        shutil.copy2(seqinfo_ini, seqinfo_target)
    else:
        seq_len = max(
            _max_frame_id_from_mot_txt(ground_truth_txt),
            _max_frame_id_from_mot_txt(predictions_txt),
        )
        _write_default_seqinfo(seqinfo_target, seq_name=seq_name, seq_length=seq_len)

    seqmap_file = seqmaps_dir / f"{benchmark_name}-all.txt"
    seqmap_file.write_text(f"name\n{seq_name}\n", encoding="utf-8")
    return data_root


def run_trackeval(
    data_root: Path,
    seq_name: str,
    tracker_name: str,
    benchmark_name: str,
) -> None:
    try:
        import trackeval  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "trackeval is not installed. Install with: pip install trackeval"
        ) from exc

    eval_config = trackeval.Evaluator.get_default_eval_config()
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}

    # Keep deterministic single-process evaluation for local experiments.
    eval_config["USE_PARALLEL"] = False
    eval_config["NUM_PARALLEL_CORES"] = 1

    dataset_config["GT_FOLDER"] = str(data_root / "gt" / "mot_challenge" / benchmark_name)
    dataset_config["TRACKERS_FOLDER"] = str(data_root / "trackers" / "mot_challenge" / benchmark_name)
    dataset_config["BENCHMARK"] = str(benchmark_name)
    dataset_config["SPLIT_TO_EVAL"] = "all"
    dataset_config["TRACKERS_TO_EVAL"] = [str(tracker_name)]
    dataset_config["SEQMAP_FOLDER"] = str(data_root / "gt" / "mot_challenge" / "seqmaps")
    dataset_config["SEQMAP_FILE"] = str(
        data_root / "gt" / "mot_challenge" / "seqmaps" / f"{benchmark_name}-all.txt"
    )
    dataset_config["SKIP_SPLIT_FOL"] = True
    dataset_config["DO_PREPROC"] = False

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [
        trackeval.metrics.HOTA(metrics_config),
        trackeval.metrics.CLEAR(metrics_config),
        trackeval.metrics.Identity(metrics_config),
    ]
    evaluator.evaluate(dataset_list, metrics_list)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare and run MOT tracking metrics (MOTA/HOTA/IDSW) with TrackEval."
    )
    parser.add_argument(
        "--metrics-dir",
        default="",
        help="Directory exported by run_agents.py under outputs/tracking_metrics/<run>; auto-finds pred/gt/seqinfo.",
    )
    parser.add_argument("--pred", default="", help="Path to tracker predictions txt (MOT format).")
    parser.add_argument("--gt", default="", help="Path to ground-truth txt (MOT format).")
    parser.add_argument(
        "--out-dir",
        default="outputs/tracking_metrics_eval",
        help="Output directory for TrackEval-compatible bundle.",
    )
    parser.add_argument("--seq-name", default="CARLA_SEQ_01", help="Sequence name used in TrackEval.")
    parser.add_argument("--tracker-name", default="BoTSORT", help="Tracker name used in TrackEval.")
    parser.add_argument("--benchmark", default="CARLA_MOT", help="Benchmark folder name for TrackEval.")
    parser.add_argument("--seqinfo", default="", help="Optional seqinfo.ini path.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for built-in simple metrics.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare folder structure and built-in summary; do not execute TrackEval.",
    )
    args = parser.parse_args()

    metrics_dir: Optional[Path] = None
    if args.metrics_dir:
        metrics_dir = Path(args.metrics_dir).resolve()
        if not metrics_dir.exists():
            raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    if args.pred:
        pred_path = Path(args.pred).resolve()
    elif metrics_dir is not None:
        pred_candidate = _find_single_prediction_file(metrics_dir)
        if pred_candidate is None:
            raise FileNotFoundError(f"No tracker prediction txt found in: {metrics_dir}")
        pred_path = pred_candidate.resolve()
    else:
        raise ValueError("Provide --metrics-dir or --pred.")

    if args.gt:
        gt_path = Path(args.gt).resolve()
    elif metrics_dir is not None:
        gt_path = (metrics_dir / "ground_truth.txt").resolve()
    else:
        raise ValueError("Provide --metrics-dir or --gt.")

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if metrics_dir is not None:
        metadata_candidate = metrics_dir / "run_metadata.json"
        if metadata_candidate.exists():
            metadata_target = out_dir / "run_metadata.json"
            shutil.copy2(metadata_candidate, metadata_target)
            print(f"[OK] Copied run metadata: {metadata_target}")

    seqinfo_path: Optional[Path] = None
    if args.seqinfo:
        seqinfo_candidate = Path(args.seqinfo).resolve()
        if seqinfo_candidate.exists():
            seqinfo_path = seqinfo_candidate
    elif metrics_dir is not None:
        seqinfo_candidate = metrics_dir / "seqinfo.ini"
        if seqinfo_candidate.exists():
            seqinfo_path = seqinfo_candidate.resolve()

    seq_name = str(args.seq_name).strip()
    if seq_name == "CARLA_SEQ_01":
        if seqinfo_path is not None:
            seq_name = _read_seq_name_from_seqinfo(seqinfo_path) or seq_name
        elif metrics_dir is not None and metrics_dir.name:
            seq_name = metrics_dir.name

    data_root = prepare_trackeval_bundle(
        predictions_txt=pred_path,
        ground_truth_txt=gt_path,
        output_dir=out_dir,
        seq_name=seq_name,
        tracker_name=str(args.tracker_name),
        benchmark_name=str(args.benchmark),
        seqinfo_ini=seqinfo_path,
    )
    print(f"[OK] Prepared TrackEval bundle at: {data_root}")

    metrics = compute_simple_tracking_metrics(
        predictions_txt=pred_path,
        ground_truth_txt=gt_path,
        iou_threshold=float(args.iou_threshold),
    )
    summary_csv, summary_txt = write_simple_metrics_outputs(metrics, out_dir)
    print(f"[OK] Wrote simple metrics CSV: {summary_csv}")
    print(f"[OK] Wrote simple metrics TXT: {summary_txt}")

    if args.prepare_only:
        return 0

    run_trackeval(
        data_root=data_root,
        seq_name=str(args.seq_name),
        tracker_name=str(args.tracker_name),
        benchmark_name=str(args.benchmark),
    )
    print("[OK] TrackEval finished. Check summary files under tracker output folders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
