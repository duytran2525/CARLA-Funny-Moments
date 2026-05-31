from pathlib import Path

from scripts.evaluate_detection_models import (
    CarlaEvalConfig,
    DetectionEvalConfig,
    DetectionRecord,
    ModelConfig,
    RunConfig,
    write_all_outputs,
)


def test_write_all_outputs_creates_prediction_directories(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test_detection: {}\n", encoding="utf-8")

    model = ModelConfig(name="detector/a", path=tmp_path / "missing.pt")
    run_config = RunConfig(
        carla_cfg=CarlaEvalConfig(),
        eval_cfg=DetectionEvalConfig(
            models=[model],
            eval_classes=["vehicle"],
            output_dir=tmp_path,
        ),
        source_config_path=config_path,
    )
    ground_truth = {
        1: [DetectionRecord(1, "vehicle", (10.0, 10.0, 20.0, 20.0), 1.0, 101)]
    }
    predictions = {
        "detector/a": {
            1: [DetectionRecord(1, "vehicle", (10.0, 10.0, 20.0, 20.0), 0.9, 7)]
        }
    }

    metrics = write_all_outputs(
        out_dir=out_dir,
        run_config=run_config,
        ground_truth_by_frame=ground_truth,
        predictions_by_model=predictions,
        inference_times_by_model={"detector/a": [12.5]},
        frames_processed=1,
        dry_run=True,
    )

    model_dir = out_dir / "detector_a"
    assert (out_dir / "ground_truth.txt").exists()
    assert (model_dir / "predictions.txt").exists()
    assert (model_dir / "detection_metrics_summary.csv").exists()
    assert (out_dir / "detection_comparison.csv").exists()
    assert metrics["detector/a"]["true_positives"] == 1
