from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from ultralytics import YOLO


def export_to_onnx(pt_path: Path, out_dir: Path, imgsz: int | tuple[int, int] = 640) -> Path:
    model = YOLO(str(pt_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {pt_path} -> ONNX (imgsz={imgsz})")
    # ultralytics YOLO export: returns list of export results or writes file to models/ by default
    result = model.export(format="onnx", imgsz=imgsz, device=0 if torch.cuda.is_available() else "cpu")
    # result can be str/Path or list; find the produced .onnx file
    candidates = []
    if isinstance(result, (list, tuple)):
        candidates = [Path(p) for p in result if str(p).lower().endswith(".onnx")]
    elif isinstance(result, (str, Path)):
        p = Path(result)
        if p.suffix.lower() == ".onnx":
            candidates = [p]

    # fallback: search cwd for recent .onnx
    if not candidates:
        for p in out_dir.rglob("*.onnx"):
            candidates.append(p)

    if not candidates:
        raise RuntimeError("ONNX export did not produce a .onnx file")

    # choose the first candidate
    onnx_path = candidates[0]
    target = out_dir / onnx_path.name
    if onnx_path.resolve() != target.resolve():
        shutil.move(str(onnx_path), str(target))
    print(f"ONNX model saved at: {target}")
    return target


def _ensure_tensorrt_module_alias() -> None:
    """Map tensorrt_bindings to tensorrt when only Python bindings are installed."""
    try:
        import tensorrt  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    try:
        import tensorrt_bindings as trt  # type: ignore

        if "tensorrt" not in sys.modules:
            sys.modules["tensorrt"] = trt
    except Exception:
        pass


def build_trt_engine(onnx_path: Path, engine_path: Path, fp16: bool = False, workspace: int = 4096) -> None:
    trtexec = shutil.which("trtexec")
    if trtexec is not None:
        cmd = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            f"--workspace={workspace}",
        ]
        if fp16:
            cmd.append("--fp16")

        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd)
        print(f"TensorRT engine written to: {engine_path}")
        return

    _ensure_tensorrt_module_alias()
    try:
        from ultralytics.utils.export.engine import onnx2engine
    except Exception as exc:
        raise FileNotFoundError(
            "trtexec not found in PATH and TensorRT Python export is unavailable."
        ) from exc

    workspace_gb = max(workspace / 1024.0, 0.25)
    print(
        "trtexec not found in PATH. Falling back to TensorRT Python API "
        f"(workspace={workspace_gb:.2f} GB, fp16={fp16})."
    )
    onnx2engine(
        str(onnx_path),
        str(engine_path),
        workspace=workspace_gb,
        half=fp16,
        verbose=False,
        prefix="TensorRT:",
    )
    print(f"TensorRT engine written to: {engine_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert Ultralytics YOLO .pt to TensorRT .engine via ONNX")
    parser.add_argument("input", help="Path to .pt file (e.g., yolo26l_best.pt)")
    parser.add_argument("--out-dir", help="Output folder for ONNX and engine", default="models")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (square) for export")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 when building TensorRT engine")
    parser.add_argument(
        "--workspace",
        type=int,
        default=4096,
        help="Workspace size in MB for TensorRT builder (auto-converted for Python fallback)",
    )
    parser.add_argument("--no-trt", action="store_true", help="Only export ONNX, do not run trtexec")

    args = parser.parse_args(argv)
    pt_path = Path(args.input)
    if not pt_path.exists():
        print(f"Input file not found: {pt_path}")
        return 2

    out_dir = Path(args.out_dir)

    try:
        onnx_path = export_to_onnx(pt_path, out_dir, imgsz=args.imgsz)
    except Exception as e:
        print("ONNX export failed:", e)
        return 3

    if args.no_trt:
        print("Skipping TensorRT engine build (--no-trt).")
        return 0

    engine_name = pt_path.stem + ".engine"
    engine_path = out_dir / engine_name

    try:
        build_trt_engine(onnx_path, engine_path, fp16=args.fp16, workspace=args.workspace)
    except subprocess.CalledProcessError as e:
        print("trtexec failed:", e)
        return 4
    except FileNotFoundError as e:
        print(e)
        print("You can manually run trtexec with the following command:")
        print(f"trtexec --onnx={onnx_path} --saveEngine={engine_path} --workspace={args.workspace} {'--fp16' if args.fp16 else ''}")
        return 5

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
