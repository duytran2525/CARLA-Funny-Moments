from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional


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
    shutil.copy2(ground_truth_txt, gt_target)
    shutil.copy2(predictions_txt, pred_target)

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

    dataset_config["GT_FOLDER"] = str(data_root / "gt" / "mot_challenge")
    dataset_config["TRACKERS_FOLDER"] = str(data_root / "trackers" / "mot_challenge")
    dataset_config["BENCHMARK"] = str(benchmark_name)
    dataset_config["SPLIT_TO_EVAL"] = "all"
    dataset_config["TRACKERS_TO_EVAL"] = [str(tracker_name)]
    dataset_config["SEQMAP_FOLDER"] = str(data_root / "gt" / "mot_challenge" / "seqmaps")
    dataset_config["SEQMAP_FILE"] = f"{benchmark_name}-all.txt"
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
    parser.add_argument("--pred", required=True, help="Path to tracker predictions txt (MOT format).")
    parser.add_argument("--gt", required=True, help="Path to ground-truth txt (MOT format).")
    parser.add_argument(
        "--out-dir",
        default="outputs/tracking_metrics_eval",
        help="Output directory for TrackEval-compatible bundle.",
    )
    parser.add_argument("--seq-name", default="CARLA_SEQ_01", help="Sequence name used in TrackEval.")
    parser.add_argument("--tracker-name", default="BoTSORT", help="Tracker name used in TrackEval.")
    parser.add_argument("--benchmark", default="CARLA_MOT", help="Benchmark folder name for TrackEval.")
    parser.add_argument("--seqinfo", default="", help="Optional seqinfo.ini path.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare folder structure; do not execute TrackEval.",
    )
    args = parser.parse_args()

    pred_path = Path(args.pred).resolve()
    gt_path = Path(args.gt).resolve()
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    seqinfo_path: Optional[Path] = None
    if args.seqinfo:
        seqinfo_candidate = Path(args.seqinfo).resolve()
        if seqinfo_candidate.exists():
            seqinfo_path = seqinfo_candidate

    data_root = prepare_trackeval_bundle(
        predictions_txt=pred_path,
        ground_truth_txt=gt_path,
        output_dir=out_dir,
        seq_name=str(args.seq_name),
        tracker_name=str(args.tracker_name),
        benchmark_name=str(args.benchmark),
        seqinfo_ini=seqinfo_path,
    )
    print(f"[OK] Prepared TrackEval bundle at: {data_root}")

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
