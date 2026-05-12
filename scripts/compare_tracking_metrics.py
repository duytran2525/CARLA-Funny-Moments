from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    from scripts.evaluate_tracking_metrics import (
        _find_single_prediction_file,
        compute_simple_tracking_metrics,
        metric_key_sort_key,
    )
except ModuleNotFoundError:  # pragma: no cover - used when executed from scripts/
    from evaluate_tracking_metrics import (  # type: ignore
        _find_single_prediction_file,
        compute_simple_tracking_metrics,
        metric_key_sort_key,
    )


NUMERIC_METRICS = {
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
    "hota",
}

METRIC_ORDER = [
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
    "hota",
]

HIGHER_IS_BETTER = {
    "true_positives",
    "precision",
    "recall",
    "f1",
    "mota",
    "motp",
    "hota",
}

LOWER_IS_BETTER = {
    "false_positives",
    "false_negatives",
    "id_switches",
}

FAIRNESS_KEYS = [
    "agent",
    "map_name",
    "sync",
    "fixed_delta",
    "no_rendering",
    "seed",
    "ticks",
    "weather_preset",
    "vehicle_filter",
    "configured_spawn_point",
    "actual_initial_spawn_point",
    "configured_destination_point",
    "actual_route_destination_point",
    "target_speed_kmh",
    "tm_port",
    "npc_vehicle_count",
    "npc_bike_count",
    "npc_motorbike_count",
    "npc_pedestrian_count",
    "npc_enable_autopilot",
    "camera_width",
    "camera_height",
    "camera_fov",
    "camera_mount.x_m",
    "camera_mount.y_m",
    "camera_mount.z_m",
    "camera_mount.pitch_deg",
    "yolo_backend",
    "yolo_nav_agent_type",
    "resolved_yolo_model_path",
    "yolo_inference_imgsz",
    "yolo_inference_every_n_ticks",
    "detector_uses_depth_input",
    "gt_occlusion_filter.enabled",
    "gt_occlusion_filter.method",
    "gt_occlusion_filter.min_visible_area_ratio",
    "gt_occlusion_filter.min_depth_visible_ratio",
    "metrics_raw.ground_truth_sha256",
    "metrics_raw.ground_truth_rows",
    "metrics_raw.ground_truth_max_frame",
]

EXPECTED_DIFFERENCE_KEYS = [
    "yolo_tracker_config",
]


def _parse_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    if text == "":
        return ""
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except Exception:
        return text


def _read_summary_csv(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        try:
            row = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Metrics CSV is empty: {path}") from exc
    return {key: _parse_scalar(value) for key, value in row.items()}


def _write_summary_csv(metrics: Dict[str, Any], path: Path) -> None:
    fieldnames = [key for key in METRIC_ORDER if key in metrics]
    for key in sorted(metrics, key=metric_key_sort_key):
        if key not in fieldnames:
            fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: metrics.get(key, "") for key in fieldnames})


def _flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def _read_metadata_near(path: Path) -> Dict[str, Any]:
    candidates = []
    if path.is_file():
        candidates.extend([path.parent / "run_metadata.json", path.parent.parent / "run_metadata.json"])
    else:
        candidates.append(path / "run_metadata.json")
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as file_obj:
                loaded = json.load(file_obj)
            if isinstance(loaded, dict):
                return loaded
    return {}


def _ground_truth_path_for_stats(source: Path, metadata: Dict[str, Any]) -> Optional[Path]:
    candidates = []
    if source.is_file():
        candidates.append(source.parent / "ground_truth.txt")
    else:
        candidates.append(source / "ground_truth.txt")

    metadata_path = metadata.get("ground_truth_path")
    if metadata_path:
        try:
            candidates.append(Path(str(metadata_path)))
        except Exception:
            pass

    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate
        except Exception:
            continue
    return None


def _ground_truth_file_stats(path: Path) -> Dict[str, Any]:
    sha = hashlib.sha256()
    rows = 0
    max_frame = 0
    with path.open("rb") as file_obj:
        for raw in file_obj:
            sha.update(raw)
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            rows += 1
            first = line.split(",", 1)[0].strip()
            try:
                frame_id = int(float(first))
            except Exception:
                continue
            max_frame = max(max_frame, frame_id)
    return {
        "ground_truth_sha256": sha.hexdigest(),
        "ground_truth_rows": rows,
        "ground_truth_max_frame": max_frame,
    }


def _metadata_with_raw_stats(metadata: Dict[str, Any], source: Path) -> Dict[str, Any]:
    enriched = dict(metadata)
    gt_path = _ground_truth_path_for_stats(source, enriched)
    if gt_path is None:
        return enriched
    try:
        enriched["metrics_raw"] = _ground_truth_file_stats(gt_path)
    except Exception:
        pass
    return enriched


def _coerce_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _normalize_for_compare(value: Any) -> str:
    parsed = _parse_scalar(value)
    if isinstance(parsed, float):
        return f"{parsed:.9g}"
    if isinstance(parsed, bool):
        return "true" if parsed else "false"
    if parsed is None:
        return ""
    return str(parsed).strip()


def _format_value(value: Any) -> str:
    number = _coerce_float(value)
    if number is not None and not isinstance(value, bool):
        if abs(number - round(number)) < 1e-9:
            return str(int(round(number)))
        return f"{number:.6f}"
    if value is None:
        return ""
    return str(value)


def load_tracking_run(source: Path, name: str, iou_threshold: float = 0.5) -> Dict[str, Any]:
    source = source.resolve()
    if source.is_file():
        metrics = _read_summary_csv(source)
        metadata = _metadata_with_raw_stats(_read_metadata_near(source), source)
        return {
            "name": name,
            "source": str(source),
            "metrics": metrics,
            "metadata": metadata,
        }

    if not source.exists():
        raise FileNotFoundError(f"Tracking source not found: {source}")
    if not source.is_dir():
        raise ValueError(f"Tracking source must be a directory or CSV: {source}")

    summary_csv = source / "tracking_metrics_summary.csv"
    if summary_csv.exists():
        metrics = _read_summary_csv(summary_csv)
        metadata = _metadata_with_raw_stats(_read_metadata_near(source), source)
        return {
            "name": name,
            "source": str(source),
            "metrics": metrics,
            "metadata": metadata,
        }

    pred_path = _find_single_prediction_file(source)
    gt_path = source / "ground_truth.txt"
    if pred_path is None or not gt_path.exists():
        raise FileNotFoundError(
            "Directory must contain tracking_metrics_summary.csv or raw metrics files "
            f"(*_tracker_predictions.txt + ground_truth.txt): {source}"
        )

    metrics = compute_simple_tracking_metrics(
        predictions_txt=pred_path,
        ground_truth_txt=gt_path,
        iou_threshold=float(iou_threshold),
    )
    _write_summary_csv(metrics, source / "tracking_metrics_summary.csv")
    metadata = _metadata_with_raw_stats(_read_metadata_near(source), source)
    return {
        "name": name,
        "source": str(source),
        "metrics": metrics,
        "metadata": metadata,
    }


def compare_metric_rows(left_run: Dict[str, Any], right_run: Dict[str, Any]) -> list[Dict[str, Any]]:
    left_metrics = dict(left_run["metrics"])
    right_metrics = dict(right_run["metrics"])
    ordered_keys = [key for key in METRIC_ORDER if key in left_metrics or key in right_metrics]
    for key in sorted(set(left_metrics) | set(right_metrics), key=metric_key_sort_key):
        if key not in ordered_keys:
            ordered_keys.append(key)

    rows: list[Dict[str, Any]] = []
    left_name = str(left_run["name"])
    right_name = str(right_run["name"])
    for metric in ordered_keys:
        left_value = left_metrics.get(metric, "")
        right_value = right_metrics.get(metric, "")
        left_num = _coerce_float(left_value)
        right_num = _coerce_float(right_value)
        delta = ""
        winner = "n/a"
        preference = "neutral"

        if metric in HIGHER_IS_BETTER:
            preference = "higher"
        elif metric in LOWER_IS_BETTER:
            preference = "lower"
        elif metric.endswith(("_true_positives", "_precision", "_recall", "_f1", "_mota", "_motp")):
            preference = "higher"
        elif metric.endswith(("_false_positives", "_false_negatives", "_id_switches")):
            preference = "lower"

        if left_num is not None and right_num is not None:
            raw_delta = right_num - left_num
            delta = raw_delta
            if preference == "higher":
                if raw_delta > 1e-9:
                    winner = right_name
                elif raw_delta < -1e-9:
                    winner = left_name
                else:
                    winner = "tie"
            elif preference == "lower":
                if raw_delta < -1e-9:
                    winner = right_name
                elif raw_delta > 1e-9:
                    winner = left_name
                else:
                    winner = "tie"
            else:
                winner = "n/a"

        rows.append(
            {
                "metric": metric,
                left_name: left_value,
                right_name: right_value,
                "delta_right_minus_left": delta,
                "preferred_direction": preference,
                "winner": winner,
            }
        )
    return rows


def _metadata_value(flat_metadata: Dict[str, Any], key: str) -> Any:
    if key == "actual_route_destination_point":
        value = flat_metadata.get(key, "")
        backend = str(flat_metadata.get("yolo_backend", "")).strip().lower()
        if _normalize_for_compare(value) == "" and backend == "tm":
            return "not_applicable_tm_backend"
    return flat_metadata.get(key, "")


def compare_fairness_rows(left_run: Dict[str, Any], right_run: Dict[str, Any]) -> list[Dict[str, Any]]:
    left_flat = _flatten_dict(dict(left_run.get("metadata") or {}))
    right_flat = _flatten_dict(dict(right_run.get("metadata") or {}))

    rows: list[Dict[str, Any]] = []
    for key in FAIRNESS_KEYS:
        left_value = _metadata_value(left_flat, key)
        right_value = _metadata_value(right_flat, key)
        left_norm = _normalize_for_compare(left_value)
        right_norm = _normalize_for_compare(right_value)
        if left_norm == "" and right_norm == "":
            status = "missing"
        elif left_norm == "" or right_norm == "":
            status = "unknown"
        elif left_norm == right_norm:
            status = "match"
        else:
            status = "mismatch"
        rows.append(
            {
                "key": key,
                str(left_run["name"]): left_value,
                str(right_run["name"]): right_value,
                "status": status,
            }
        )

    for key in EXPECTED_DIFFERENCE_KEYS:
        left_value = _metadata_value(left_flat, key)
        right_value = _metadata_value(right_flat, key)
        rows.append(
            {
                "key": key,
                str(left_run["name"]): left_value,
                str(right_run["name"]): right_value,
                "status": "expected_difference",
            }
        )

    return rows


def _write_csv(path: Path, fieldnames: Iterable[str], rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fairness_status(rows: list[Dict[str, Any]]) -> str:
    relevant = [row for row in rows if row.get("status") != "expected_difference"]
    if any(row.get("status") == "mismatch" for row in relevant):
        return "FAIL"
    if any(row.get("status") in {"unknown", "missing"} for row in relevant):
        return "UNKNOWN"
    return "PASS"


def write_comparison_outputs(
    left_run: Dict[str, Any],
    right_run: Dict[str, Any],
    output_dir: Path,
) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    left_name = str(left_run["name"])
    right_name = str(right_run["name"])

    metric_rows = compare_metric_rows(left_run, right_run)
    fairness_rows = compare_fairness_rows(left_run, right_run)

    metrics_csv = output_dir / "tracking_metrics_comparison.csv"
    fairness_csv = output_dir / "tracking_fairness_check.csv"
    summary_txt = output_dir / "tracking_metrics_comparison.txt"

    _write_csv(
        metrics_csv,
        ["metric", left_name, right_name, "delta_right_minus_left", "preferred_direction", "winner"],
        metric_rows,
    )
    _write_csv(
        fairness_csv,
        ["key", left_name, right_name, "status"],
        fairness_rows,
    )

    fairness = _fairness_status(fairness_rows)
    lines = [
        "Tracking Metrics Comparison",
        f"left: {left_name} ({left_run['source']})",
        f"right: {right_name} ({right_run['source']})",
        f"fairness_check: {fairness}",
        "",
        "Metrics",
    ]
    for row in metric_rows:
        metric = str(row["metric"])
        if metric == "method":
            continue
        left_value = _format_value(row[left_name])
        right_value = _format_value(row[right_name])
        delta = _format_value(row["delta_right_minus_left"])
        winner = str(row["winner"])
        lines.append(
            f"{metric}: {left_name}={left_value} | {right_name}={right_value} | delta={delta} | winner={winner}"
        )

    mismatch_rows = [row for row in fairness_rows if row.get("status") == "mismatch"]
    unknown_rows = [row for row in fairness_rows if row.get("status") in {"unknown", "missing"}]
    lines.extend(["", "Fairness"])
    if mismatch_rows:
        lines.append("mismatches:")
        for row in mismatch_rows:
            lines.append(
                f"- {row['key']}: {left_name}={_format_value(row[left_name])} | "
                f"{right_name}={_format_value(row[right_name])}"
            )
    elif unknown_rows:
        lines.append("No mismatches found, but some metadata keys are missing/unknown.")
    else:
        lines.append("All fairness keys match; tracker config is allowed to differ.")

    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics_csv, fairness_csv, summary_txt


def _resolve_source(primary: str, fallback: str, label: str) -> Path:
    source = primary.strip() if primary else fallback.strip()
    if not source:
        raise ValueError(f"Provide --{label} or positional {label}.")
    return Path(source)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two CARLA tracking metric runs side-by-side and check whether "
            "the runs are fair except for tracker config."
        )
    )
    parser.add_argument("left", nargs="?", default="", help="Left metrics dir or tracking_metrics_summary.csv.")
    parser.add_argument("right", nargs="?", default="", help="Right metrics dir or tracking_metrics_summary.csv.")
    parser.add_argument("--left-source", default="", help="Left metrics dir or tracking_metrics_summary.csv.")
    parser.add_argument("--right-source", default="", help="Right metrics dir or tracking_metrics_summary.csv.")
    parser.add_argument("--left-name", default="BoTSORT", help="Display name for the left run.")
    parser.add_argument("--right-name", default="ByteTrack", help="Display name for the right run.")
    parser.add_argument(
        "--out-dir",
        default="outputs/tracking_metrics_compare",
        help="Directory for tracking_metrics_comparison.csv/txt and tracking_fairness_check.csv.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used only when a raw metrics dir has no summary CSV yet.",
    )
    args = parser.parse_args()

    left_source = _resolve_source(args.left_source, args.left, "left-source")
    right_source = _resolve_source(args.right_source, args.right, "right-source")

    left_run = load_tracking_run(left_source, name=str(args.left_name), iou_threshold=float(args.iou_threshold))
    right_run = load_tracking_run(right_source, name=str(args.right_name), iou_threshold=float(args.iou_threshold))

    metrics_csv, fairness_csv, summary_txt = write_comparison_outputs(
        left_run=left_run,
        right_run=right_run,
        output_dir=Path(args.out_dir).resolve(),
    )
    print(f"[OK] Wrote comparison CSV: {metrics_csv}")
    print(f"[OK] Wrote fairness CSV: {fairness_csv}")
    print(f"[OK] Wrote comparison TXT: {summary_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
