
import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


# --------------------------------------------------
# 1. CLASS NAMES
# --------------------------------------------------

CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]


# --------------------------------------------------
# 2. INFERENCE FUNCTION
# --------------------------------------------------

def run_inference(
    weights_path: Path,
    source_path: Path,
    output_path: Path,
    image_size: int = 640,
    confidence: float = 0.001,
    iou_threshold: float = 0.7,
    max_detections: int = 300,
):
    """
    Perform inference on PCB images and generate predictions.csv.

    Args:
        weights_path: Path to trained best.pt weights.
        source_path: Image file or directory containing test images.
        output_path: Destination CSV file.
        image_size: Input image size for YOLO.
        confidence: Minimum confidence threshold.
        iou_threshold: NMS IoU threshold.
        max_detections: Maximum detections per image.
    """

    # Validate input paths
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}"
        )

    if not source_path.exists():
        raise FileNotFoundError(
            f"Input images not found: {source_path}"
        )

    # Load trained YOLO model
    print("\nLoading trained YOLO model...")

    model = YOLO(str(weights_path))

    print("Model loaded successfully!")

    # Perform inference
    print("\nRunning inference...")

    results = model.predict(
        source=str(source_path),
        imgsz=image_size,
        conf=confidence,
        iou=iou_threshold,
        max_det=max_detections,
        device="cpu",
        stream=False,
        verbose=False,
    )

    # Store predictions
    predictions = []

    for result in results:

        # Original image filename
        image_name = Path(result.path).name

        # Skip images without detections
        if result.boxes is None or len(result.boxes) == 0:
            continue

        # Bounding boxes in original image coordinates
        boxes = result.boxes.xyxy.cpu().numpy()

        # Confidence scores
        confidences = result.boxes.conf.cpu().numpy()

        # Predicted class indices
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, score, class_id in zip(
            boxes, confidences, class_ids
        ):

            x_min, y_min, x_max, y_max = box

            # Map class index to class name
            class_name = (
                model.names[class_id]
                if class_id in model.names
                else CLASS_NAMES[class_id]
            )

            predictions.append({
                "image_name": image_name,
                "class_name": class_name,
                "x_min": round(float(x_min), 2),
                "y_min": round(float(y_min), 2),
                "x_max": round(float(x_max), 2),
                "y_max": round(float(y_max), 2),
                "confidence": round(float(score), 6),
            })

    # --------------------------------------------------
    # 3. CREATE DATAFRAME
    # --------------------------------------------------

    columns = [
        "image_name",
        "class_name",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "confidence",
    ]

    df = pd.DataFrame(predictions, columns=columns)

    # --------------------------------------------------
    # 4. SAVE CSV
    # --------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(output_path, index=False)

    # --------------------------------------------------
    # 5. SUMMARY
    # --------------------------------------------------

    print("\nInference completed!")

    print(f"Images processed: {len(results)}")
    print(f"Total detections: {len(df)}")
    print(f"CSV saved at: {output_path}")

    if not df.empty:
        print("\nDetections per class:")
        print(df["class_name"].value_counts())

    else:
        print("\nNo defects detected.")

    return df


# --------------------------------------------------
# 6. COMMAND LINE INTERFACE
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="PCB Defect Detection Inference"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
        help="Path to trained YOLO weights"
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to test image or test image directory"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="submission/predictions.csv",
        help="Output CSV path"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold"
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold"
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum detections per image"
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent

    weights_path = Path(args.weights)
    source_path = Path(args.source)
    output_path = Path(args.output)

    if not weights_path.is_absolute():
        weights_path = project_root / weights_path

    if not source_path.is_absolute():
        source_path = project_root / source_path

    if not output_path.is_absolute():
        output_path = project_root / output_path

    run_inference(
        weights_path=weights_path,
        source_path=source_path,
        output_path=output_path,
        image_size=args.imgsz,
        confidence=args.conf,
        iou_threshold=args.iou,
        max_detections=args.max_det,
    )


if __name__ == "__main__":
    main()
