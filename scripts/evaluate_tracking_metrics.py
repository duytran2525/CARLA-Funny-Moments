from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


MotRows = Dict[int, List[dict]]
UNKNOWN_CLASS = "__unknown__"
EVALUATION_CLASS_ORDER = [
    "vehicle",
    "two_wheeler",
    "traffic_light_red",
    "traffic_sign",
    "pedestrian",
    "traffic_light_green",
    "stop_line",
]
_EVALUATION_CLASS_INDEX = {
    class_name: index for index, class_name in enumerate(EVALUATION_CLASS_ORDER)
}
_CLASS_ALIASES = {
    "bike": "two_wheeler",
    "bicycle": "two_wheeler",
    "motobike": "two_wheeler",
    "motorbike": "two_wheeler",
    "motorcycle": "two_wheeler",
    "trafficlight_red": "traffic_light_red",
    "red_traffic_light": "traffic_light_red",
    "trafficlight_green": "traffic_light_green",
    "green_traffic_light": "traffic_light_green",
    "stopline": "stop_line",
    "stop_line_marking": "stop_line",
}
BASE_METRIC_FIELDS = [
    "method",
    "class_match_mode",
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
PER_CLASS_METRIC_SUFFIXES = [
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


def _normalize_metric_class_name(class_name: str) -> str:
    text = str(class_name).strip().lower()
    if not text or text == UNKNOWN_CLASS:
        return UNKNOWN_CLASS
    normalized = text.replace(" ", "_").replace("-", "_")
    return _CLASS_ALIASES.get(normalized, normalized)


def _safe_metric_class_name(class_name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in class_name).strip("_") or "unknown"


def _metric_class_sort_key(class_name: str) -> Tuple[int, str]:
    normalized = _normalize_metric_class_name(class_name)
    class_index = _EVALUATION_CLASS_INDEX.get(normalized)
    if class_index is not None:
        return class_index, ""
    return len(EVALUATION_CLASS_ORDER), _safe_metric_class_name(normalized)


def _ordered_metric_classes(class_names: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for class_name in class_names:
        normalized = _normalize_metric_class_name(class_name)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return sorted(ordered, key=_metric_class_sort_key)


def _empty_class_stats() -> dict:
    return {
        "gt_detections": 0,
        "pred_detections": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "id_switches": 0,
        "iou_sum": 0.0,
    }


def _split_class_metric_key(key: str) -> Optional[Tuple[str, str]]:
    if not key.startswith("class_"):
        return None
    for suffix in PER_CLASS_METRIC_SUFFIXES:
        marker = f"_{suffix}"
        if key.endswith(marker):
            return key[len("class_") : -len(marker)], suffix
    return None


def metric_key_sort_key(key: str) -> Tuple[int, int, str, int, str]:
    class_metric = _split_class_metric_key(key)
    if class_metric is None:
        return 2, len(EVALUATION_CLASS_ORDER), "", len(PER_CLASS_METRIC_SUFFIXES), str(key)

    class_name, suffix = class_metric
    class_index, fallback_class_name = _metric_class_sort_key(class_name)
    suffix_index = PER_CLASS_METRIC_SUFFIXES.index(suffix)
    return 1, class_index, fallback_class_name, suffix_index, str(key)


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


def copy_run_context_files(out_dir: Path, pred_path: Path, gt_path: Path, metrics_dir: Optional[Path] = None) -> None:
    """Copy raw GT and nearby metadata so later compare runs can verify fairness."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_candidates = []
    if metrics_dir is not None:
        metadata_candidates.append(metrics_dir / "run_metadata.json")
    metadata_candidates.extend([pred_path.parent / "run_metadata.json", gt_path.parent / "run_metadata.json"])

    for metadata_candidate in metadata_candidates:
        if metadata_candidate.exists():
            metadata_target = out_dir / "run_metadata.json"
            if metadata_candidate.resolve() != metadata_target.resolve():
                shutil.copy2(metadata_candidate, metadata_target)
            print(f"[OK] Copied run metadata: {metadata_target}")
            break

    if gt_path.exists():
        gt_target = out_dir / "ground_truth.txt"
        if gt_path.resolve() != gt_target.resolve():
            shutil.copy2(gt_path, gt_target)
        print(f"[OK] Copied raw ground truth: {gt_target}")


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


def _copy_seqinfo_with_name(source: Path, target: Path, seq_name: str) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    wrote_name = False
    output_lines = []
    for line in lines:
        if line.strip().lower().startswith("name="):
            output_lines.append(f"name={seq_name}")
            wrote_name = True
        else:
            output_lines.append(line)
    if not wrote_name:
        output_lines.insert(1 if output_lines and output_lines[0].strip().lower() == "[sequence]" else 0, f"name={seq_name}")
    target.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


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
                    "class": (
                        _normalize_metric_class_name(parts[10])
                        if len(parts) > 10 and parts[10].strip()
                        else UNKNOWN_CLASS
                    ),
                }
            )
    return rows_by_frame


def _copy_mot_txt_for_trackeval(
    source: Path,
    target: Path,
    ignore_negative_ids: bool = False,
    class_filter: Optional[str] = None,
    force_mot_class_id: Optional[int] = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    class_filter_norm = _normalize_metric_class_name(class_filter) if class_filter is not None else None
    with source.open("r", encoding="utf-8") as source_obj, target.open("w", encoding="utf-8") as target_obj:
        for raw in source_obj:
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            if ignore_negative_ids:
                if len(parts) < 2:
                    continue
                try:
                    track_id = int(float(parts[1]))
                except Exception:
                    continue
                if track_id < 0:
                    continue
            if class_filter_norm is not None:
                row_class = (
                    _normalize_metric_class_name(parts[10])
                    if len(parts) > 10 and parts[10].strip()
                    else UNKNOWN_CLASS
                )
                if row_class != class_filter_norm:
                    continue
            # TrackEval expects standard MOTChallenge columns. New logs may append
            # repo-local metadata such as class after column 10, so strip extras here.
            mot_parts = list(parts[:10])
            while len(mot_parts) < 10:
                mot_parts.append("-1")
            if force_mot_class_id is not None:
                mot_parts[7] = str(int(force_mot_class_id))
            target_obj.write(",".join(mot_parts[:10]) + "\n")


def _classes_in_mot_files(*paths: Path) -> List[str]:
    classes = set()
    for path in paths:
        rows_by_frame = _read_mot_txt(path, ignore_negative_ids=False)
        for rows in rows_by_frame.values():
            for row in rows:
                class_name = str(row.get("class", UNKNOWN_CLASS))
                if class_name != UNKNOWN_CLASS:
                    classes.add(class_name)
    return _ordered_metric_classes(classes)


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
    last_match_by_gt_key: Dict[Tuple[str, int], int] = {}
    class_stats: Dict[str, dict] = {
        class_name: _empty_class_stats() for class_name in EVALUATION_CLASS_ORDER
    }

    def stats_for(class_name: str) -> dict:
        return class_stats.setdefault(_normalize_metric_class_name(class_name), _empty_class_stats())

    threshold = float(iou_threshold)
    for frame_id in frames:
        frame_gt = ground_truth.get(frame_id, [])
        frame_pred = predictions.get(frame_id, [])
        total_gt += len(frame_gt)
        total_pred += len(frame_pred)
        for gt in frame_gt:
            stats_for(str(gt["class"]))["gt_detections"] += 1
        for pred in frame_pred:
            stats_for(str(pred["class"]))["pred_detections"] += 1

        candidate_pairs = []
        for gt_idx, gt in enumerate(frame_gt):
            for pred_idx, pred in enumerate(frame_pred):
                if str(gt["class"]) != str(pred["class"]):
                    continue
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

            class_name = str(frame_gt[gt_idx]["class"])
            class_stat = stats_for(class_name)
            class_stat["true_positives"] += 1
            class_stat["iou_sum"] += float(iou)
            gt_id = int(frame_gt[gt_idx]["id"])
            pred_id = int(frame_pred[pred_idx]["id"])
            previous_pred_id = last_match_by_gt_key.get((class_name, gt_id))
            if previous_pred_id is not None and previous_pred_id != pred_id:
                id_switches += 1
                class_stat["id_switches"] += 1
            last_match_by_gt_key[(class_name, gt_id)] = pred_id

        fp += len(frame_pred) - len(matched_pred)
        fn += len(frame_gt) - len(matched_gt)
        for pred_idx, pred in enumerate(frame_pred):
            if pred_idx not in matched_pred:
                stats_for(str(pred["class"]))["false_positives"] += 1
        for gt_idx, gt in enumerate(frame_gt):
            if gt_idx not in matched_gt:
                stats_for(str(gt["class"]))["false_negatives"] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    mota = 1.0 - ((fn + fp + id_switches) / total_gt) if total_gt > 0 else 0.0
    motp = iou_sum / tp if tp > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics = {
        "method": "simple_iou_greedy",
        "class_match_mode": "strict" if any(
            str(row.get("class", UNKNOWN_CLASS)) != UNKNOWN_CLASS
            for rows in list(predictions.values()) + list(ground_truth.values())
            for row in rows
        ) else "legacy_unknown_class",
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
    for class_name in _ordered_metric_classes(class_stats):
        safe_class = _safe_metric_class_name(class_name)
        stat = class_stats[class_name]
        class_tp = int(stat["true_positives"])
        class_fp = int(stat["false_positives"])
        class_fn = int(stat["false_negatives"])
        class_gt = int(stat["gt_detections"])
        class_precision = class_tp / (class_tp + class_fp) if (class_tp + class_fp) > 0 else 0.0
        class_recall = class_tp / class_gt if class_gt > 0 else 0.0
        class_f1 = (
            2.0 * class_precision * class_recall / (class_precision + class_recall)
            if (class_precision + class_recall) > 0
            else 0.0
        )
        class_mota = (
            1.0 - ((class_fn + class_fp + int(stat["id_switches"])) / class_gt)
            if class_gt > 0
            else 0.0
        )
        class_motp = float(stat["iou_sum"]) / class_tp if class_tp > 0 else 0.0
        prefix = f"class_{safe_class}_"
        metrics.update(
            {
                f"{prefix}gt_detections": class_gt,
                f"{prefix}pred_detections": int(stat["pred_detections"]),
                f"{prefix}true_positives": class_tp,
                f"{prefix}false_positives": class_fp,
                f"{prefix}false_negatives": class_fn,
                f"{prefix}id_switches": int(stat["id_switches"]),
                f"{prefix}precision": class_precision,
                f"{prefix}recall": class_recall,
                f"{prefix}f1": class_f1,
                f"{prefix}mota": class_mota,
                f"{prefix}motp": class_motp,
            }
        )
    return metrics


def _per_class_metric_rows(metrics: dict) -> List[dict]:
    class_names = set(EVALUATION_CLASS_ORDER)
    for key in metrics:
        class_metric = _split_class_metric_key(key)
        if class_metric is not None:
            class_names.add(class_metric[0])

    rows = []
    for safe_class in _ordered_metric_classes(class_names):
        row = {"class": safe_class}
        for suffix in PER_CLASS_METRIC_SUFFIXES:
            row[suffix] = metrics.get(f"class_{safe_class}_{suffix}", 0)
        rows.append(row)
    return rows


def _write_per_class_metrics_outputs(metrics: dict, output_dir: Path) -> Tuple[Path, Path]:
    csv_path = output_dir / "tracking_metrics_per_class.csv"
    txt_path = output_dir / "tracking_metrics_per_class.txt"
    rows = _per_class_metric_rows(metrics)
    fieldnames = ["class"] + PER_CLASS_METRIC_SUFFIXES

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = ["Tracking Metrics Per Class"]
    if not rows:
        lines.append("No class labels were found. Re-run the tracker after the class-aware logging change for per-class metrics.")
    for row in rows:
        lines.extend(
            [
                "",
                f"class: {row['class']}",
                f"gt_detections: {row['gt_detections']}",
                f"pred_detections: {row['pred_detections']}",
                f"true_positives: {row['true_positives']}",
                f"false_positives: {row['false_positives']}",
                f"false_negatives: {row['false_negatives']}",
                f"id_switches: {row['id_switches']}",
                f"precision: {float(row['precision']):.6f}",
                f"recall: {float(row['recall']):.6f}",
                f"f1: {float(row['f1']):.6f}",
                f"mota: {float(row['mota']):.6f}",
                f"motp: {float(row['motp']):.6f}",
            ]
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, txt_path


def write_simple_metrics_outputs(metrics: dict, output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tracking_metrics_summary.csv"
    txt_path = output_dir / "tracking_metrics_summary.txt"

    fieldnames = list(BASE_METRIC_FIELDS)
    for key in sorted(metrics, key=metric_key_sort_key):
        if key not in fieldnames:
            fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: metrics.get(key, "") for key in fieldnames})

    lines = [
        "Tracking Metrics Summary",
        f"method: {metrics['method']}",
        f"class_match_mode: {metrics.get('class_match_mode', '')}",
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
    per_class_rows = _per_class_metric_rows(metrics)
    if per_class_rows:
        lines.extend(["", "Per Class"])
        for row in per_class_rows:
            lines.append(
                "class={class_name} | gt={gt_detections} | pred={pred_detections} | "
                "tp={true_positives} | fp={false_positives} | fn={false_negatives} | "
                "idsw={id_switches} | precision={precision:.6f} | recall={recall:.6f} | "
                "f1={f1:.6f} | mota={mota:.6f} | motp={motp:.6f}".format(
                    class_name=row["class"],
                    gt_detections=row["gt_detections"],
                    pred_detections=row["pred_detections"],
                    true_positives=row["true_positives"],
                    false_positives=row["false_positives"],
                    false_negatives=row["false_negatives"],
                    id_switches=row["id_switches"],
                    precision=float(row["precision"]),
                    recall=float(row["recall"]),
                    f1=float(row["f1"]),
                    mota=float(row["mota"]),
                    motp=float(row["motp"]),
                )
            )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_per_class_metrics_outputs(metrics, output_dir)
    return csv_path, txt_path


def prepare_trackeval_bundle(
    predictions_txt: Path,
    ground_truth_txt: Path,
    output_dir: Path,
    seq_name: str,
    tracker_name: str,
    benchmark_name: str,
    seqinfo_ini: Optional[Path] = None,
    class_filter: Optional[str] = None,
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
    # TrackEval's MOTChallenge adapter only exposes the pedestrian class. For a
    # repo-local per-class bundle, rows are already filtered by our class column,
    # so mark the remaining rows as MOT pedestrian class id 1 to make TrackEval
    # evaluate that filtered class instead of silently treating it as classless.
    force_mot_class_id = 1 if class_filter is not None else None
    _copy_mot_txt_for_trackeval(
        ground_truth_txt,
        gt_target,
        class_filter=class_filter,
        force_mot_class_id=force_mot_class_id,
    )
    _copy_mot_txt_for_trackeval(
        predictions_txt,
        pred_target,
        ignore_negative_ids=True,
        class_filter=class_filter,
        force_mot_class_id=force_mot_class_id,
    )

    seqinfo_target = gt_seq_dir / "seqinfo.ini"
    if seqinfo_ini is not None and seqinfo_ini.exists():
        _copy_seqinfo_with_name(seqinfo_ini, seqinfo_target, seq_name=seq_name)
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
    do_preproc: bool = False,
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
    dataset_config["CLASSES_TO_EVAL"] = ["pedestrian"]
    dataset_config["SEQMAP_FOLDER"] = str(data_root / "gt" / "mot_challenge" / "seqmaps")
    dataset_config["SEQMAP_FILE"] = str(
        data_root / "gt" / "mot_challenge" / "seqmaps" / f"{benchmark_name}-all.txt"
    )
    dataset_config["SKIP_SPLIT_FOL"] = True
    dataset_config["DO_PREPROC"] = bool(do_preproc)

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
    parser.add_argument(
        "--run-overall-trackeval",
        action="store_true",
        help=(
            "Also run the unfiltered TrackEval bundle. This is class-agnostic for CARLA repo logs; "
            "prefer the per-class TrackEval bundles for BoT-SORT/ByteTrack comparison."
        ),
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
    copy_run_context_files(out_dir, pred_path=pred_path, gt_path=gt_path, metrics_dir=metrics_dir)

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

    observed_class_names = _classes_in_mot_files(pred_path, gt_path)
    class_names = (
        _ordered_metric_classes([*EVALUATION_CLASS_ORDER, *observed_class_names])
        if observed_class_names
        else []
    )
    per_class_trackeval_roots: List[Tuple[str, Path, str, str]] = []
    for class_name in class_names:
        safe_class = _safe_metric_class_name(class_name)
        class_seq_name = f"{seq_name}_{safe_class}"
        class_tracker_name = f"{args.tracker_name}_{safe_class}"
        class_root = prepare_trackeval_bundle(
            predictions_txt=pred_path,
            ground_truth_txt=gt_path,
            output_dir=out_dir / "trackeval_per_class" / safe_class,
            seq_name=class_seq_name,
            tracker_name=class_tracker_name,
            benchmark_name=str(args.benchmark),
            seqinfo_ini=seqinfo_path,
            class_filter=class_name,
        )
        per_class_trackeval_roots.append((safe_class, class_root, class_seq_name, class_name))
    if per_class_trackeval_roots:
        print(
            "[OK] Prepared per-class TrackEval bundles for: "
            + ", ".join(safe_class for safe_class, _, _, _ in per_class_trackeval_roots)
        )

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

    if observed_class_names and not args.run_overall_trackeval:
        print(
            "[INFO] Skipped unfiltered overall TrackEval because MOTChallenge TrackEval is class-agnostic "
            "for these repo logs. Use tracking_metrics_summary.csv for strict overall metrics and "
            "trackeval_per_class/* for HOTA/CLEAR/IDF1 per class."
        )
    else:
        run_trackeval(
            data_root=data_root,
            seq_name=seq_name,
            tracker_name=str(args.tracker_name),
            benchmark_name=str(args.benchmark),
            do_preproc=False,
        )
    for safe_class, class_root, class_seq_name, _class_name in per_class_trackeval_roots:
        class_gt = int(float(metrics.get(f"class_{safe_class}_gt_detections", 0) or 0))
        if class_gt <= 0:
            print(
                f"[INFO] Skipped TrackEval for class without GT detections: {safe_class}. "
                "Use tracking_metrics_per_class.csv for zero-GT/FP-only class rows."
            )
            continue
        run_trackeval(
            data_root=class_root,
            seq_name=class_seq_name,
            tracker_name=f"{args.tracker_name}_{safe_class}",
            benchmark_name=str(args.benchmark),
            do_preproc=True,
        )
    print("[OK] TrackEval finished. Check summary files under tracker output folders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
