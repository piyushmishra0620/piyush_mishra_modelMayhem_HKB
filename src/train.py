from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO11 model on the PCB defect detection dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the dataset's YOLO data.yaml file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",
        help="Ultralytics model checkpoint or YAML (default: yolo11s.pt).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Training device, e.g. "0" for first CUDA GPU or "cpu". Auto if omitted.',
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--name", type=str, default="yolo11s_pcb")
    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT_ROOT / "runs" / "train",
        help="Directory in which Ultralytics stores the training run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision.",
    )
    return parser.parse_args()


def print_map50_90(metrics) -> None:
    """
    Ultralytics' standard map metric is mAP50-95.
    Its per-class AP matrix normally has ten columns for IoU 0.50..0.95.
    Averaging columns 0..8 gives an mAP50-90-style diagnostic.
    Check the competition evaluator's precise averaging/interpolation rules.
    """
    box_metrics = getattr(metrics, "box", None)
    if box_metrics is None:
        print("Could not read detection metrics from the validation result.")
        return

    print(f"Ultralytics mAP50-95: {box_metrics.map:.4f}")
    print(f"Ultralytics mAP50:    {box_metrics.map50:.4f}")

    ap = getattr(box_metrics, "ap", None)
    if ap is None:
        print("mAP50-90 diagnostic unavailable: metrics.box.ap was not exposed.")
        return

    # Convert a possible NumPy array / tensor into a NumPy array.
    if hasattr(ap, "detach"):
        ap = ap.detach().cpu().numpy()

    try:
        import numpy as np

        ap = np.asarray(ap)
        if ap.ndim == 2 and ap.shape[1] >= 9:
            map_50_90 = float(np.nanmean(ap[:, :9]))
            print(f"mAP50-90 diagnostic: {map_50_90:.4f}")
            print(
                "Note: this averages Ultralytics per-class AP values at "
                "IoU 0.50..0.90; verify it against the organizer's evaluator."
            )
        else:
            print(f"mAP50-90 diagnostic unavailable; AP matrix shape: {ap.shape}")
    except Exception as exc:
        print(f"Could not calculate mAP50-90 diagnostic: {exc}")


def main() -> None:
    args = parse_args()

    data_yaml = args.data.expanduser().resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}\n"
            "Pass the path to the extracted dataset's data.yaml file."
        )

    if args.epochs < 1 or args.imgsz < 1 or args.batch < 1:
        raise ValueError("--epochs, --imgsz and --batch must all be positive.")

    args.project = args.project.expanduser().resolve()
    args.project.mkdir(parents=True, exist_ok=True)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset YAML : {data_yaml}")
    print(f"Model        : {args.model}")
    print(f"Run output   : {args.project / args.name}")

    model = YOLO(args.model)

    train_config = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "weight_decay": 0.0005,
        "degrees": 5.0,
        "translate": 0.05,
        "scale": 0.20,
        "fliplr": 0.5,
        "flipud": 0.5,
        "pretrained": True,
        "seed": args.seed,
        "deterministic": True,
        "amp": not args.no_amp,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": True,
        "save": True,
        "plots": True,
        "verbose": True,
    }

    if args.device is not None:
        train_config["device"] = args.device

    # Keep this script under the __main__ guard for Windows multiprocessing.
    model.train(**train_config)

    run_dir = args.project / args.name
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"

    if not best_weights.is_file():
        raise FileNotFoundError(
            f"Training finished, but best.pt was not found at: {best_weights}"
        )

    # Save a predictable copy in the project's weights/ directory.
    project_weights_dir = PROJECT_ROOT / "weights"
    project_weights_dir.mkdir(parents=True, exist_ok=True)
    destination = project_weights_dir / "best.pt"
    shutil.copy2(best_weights, destination)

    print("\nTraining complete.")
    print(f"Best checkpoint : {best_weights}")
    print(f"Project copy    : {destination}")
    if last_weights.is_file():
        print(f"Last checkpoint : {last_weights}")

    # Validate the selected best checkpoint on the validation split.
    print("\nRunning validation using best.pt...")
    best_model = YOLO(str(best_weights))
    metrics = best_model.val(
        data=str(data_yaml),
        split="val",
        imgsz=args.imgsz,
        conf=0.001,
        plots=True,
        project=str(args.project),
        name=f"{args.name}_validation",
        exist_ok=True,
    )
    print("\nValidation metrics:")
    print_map50_90(metrics)


if __name__ == "__main__":
    main()
