from pathlib import Path

from scripts.evaluate_detection_models import (
    CarlaEvalConfig,
    DetectionEvalConfig,
    DetectionRecord,
    GroundTruthFilterConfig,
    ModelConfig,
    RunConfig,
    camera_intrinsics_matrix,
    parse_run_config,
    project_vertices_to_box,
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
    assert (model_dir / "detection_metrics_summary.csv").read_text(encoding="utf-8-sig").splitlines()[0].startswith(
        "scope;class;frames"
    )
    assert (out_dir / "detection_comparison.csv").read_text(encoding="utf-8-sig").splitlines()[0].startswith(
        "metric;detector/a;best_model"
    )
    assert metrics["detector/a"]["true_positives"] == 1


def test_parse_run_config_excludes_stop_line_and_keeps_traffic_sign(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
test_detection:
  models:
    - name: detector
      path: models/missing.engine
  eval_classes:
    - vehicle
    - stop_line
    - traffic_sign
""",
        encoding="utf-8",
    )

    run_config = parse_run_config(config_path)

    assert "vehicle" in run_config.eval_cfg.eval_classes
    assert "traffic_sign" in run_config.eval_cfg.eval_classes
    assert "stop_line" not in run_config.eval_cfg.eval_classes


class _Point:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


def test_project_vertices_to_box_clips_bbox_crossing_near_plane() -> None:
    import numpy as np

    intrinsics = camera_intrinsics_matrix(640, 480, 90.0)
    world_to_camera = np.eye(4, dtype=np.float64)
    vertices = [
        _Point(x, y, z)
        for x in (-1.0, 3.0)
        for y in (-1.0, 1.0)
        for z in (-1.0, 1.0)
    ]

    projected = project_vertices_to_box(
        vertices=vertices,
        center_location=_Point(1.0, 0.0, 0.0),
        world_to_camera=world_to_camera,
        intrinsics=intrinsics,
        image_w=640,
        image_h=480,
        filters=GroundTruthFilterConfig(
            min_bbox_dim_px=1,
            min_bbox_area_px=1,
            min_visible_area_ratio=0.0,
            min_depth_visible_ratio=0.0,
            max_bbox_aspect_ratio=1000.0,
        ),
    )

    assert projected is not None
    x, y, w, h = projected.bbox
    assert 0.0 <= x < 640.0
    assert 0.0 <= y < 480.0
    assert w > 0.0
    assert h > 0.0
