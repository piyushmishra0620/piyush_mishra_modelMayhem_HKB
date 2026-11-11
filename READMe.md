
# PCB Defect Detection using YOLO11

An object detection project that identifies and localizes six types of PCB defects using a YOLO11 deep learning model.

## 1. Defect Classes

The model detects the following six PCB defect categories:

1. `missing_hole`
2. `mouse_bite`
3. `open_circuit`
4. `short`
5. `spur`
6. `spurious_copper`

## 2. Dataset

**Dataset:** PCB Defect Detection v2

- [Roboflow Universe Dataset](https://universe.roboflow.com/uthm-project-visionbased-pcb-defect-detection/pcb-defect-detection-9ewqw-duvuw/dataset/2)
- [Google Drive Dataset ZIP](https://drive.google.com/file/d/1YMqqVntnZPkV0tBQYYSXzaZ8L7XCQA9U/view?usp=sharing)

The dataset contains PCB images annotated with bounding boxes for object detection.

The dataset is not included in this repository. Download and extract it separately when training the model.

## 3. Project Structure

```text
PCB-Defect-Detection/
│
├── notebooks/
│   └── pcb_defect_detection.ipynb
│
├── src/
│   ├── train.py
│   ├── inference.py
│   └── evaluate.py
│
├── weights/
│   └── best.pt
│
├── results/
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── PR_curve.png
│   └── F1_curve.png
│
├── submission/
│   └── predictions.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 4. Requirements

- Python 3.14.x
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- Pandas
- Matplotlib
- PyYAML

Install the dependencies:

```bash
python -m venv myenv
```

Activate the environment.

**Windows PowerShell:**

```powershell
.\myenv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
source myenv/bin/activate
```

Install packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Model Training

Training can be performed using Google Colab with a GPU or a compatible local environment.

Prepare the dataset and its YOLO-format `data.yaml` configuration file before training.

Run:

```bash
python src/train.py --data path/to/data.yaml
```

The trained model's best checkpoint is saved as:

```text
weights/best.pt
```

Training configuration and other available arguments can be checked using:

```bash
python src/train.py --help
```

## 6. Model Inference

Place the trained model inside the `weights/` directory.

Place the test images inside `test_images/`.

Run:

```bash
python src/inference.py --source test_images
```

For custom inference parameters:

```bash
python src/inference.py \
    --weights weights/best.pt \
    --source test_images \
    --output submission/predictions.csv \
    --imgsz 640 \
    --conf 0.001 \
    --iou 0.7 \
    --max-det 300
```

The predictions are exported to:

```text
submission/predictions.csv
```

## 7. Prediction CSV Format

The generated CSV follows this schema:

| Column | Description |
|---|---|
| image_name | Original image filename |
| class_name | Predicted defect category |
| x_min | Left coordinate of bounding box |
| y_min | Top coordinate of bounding box |
| x_max | Right coordinate of bounding box |
| y_max | Bottom coordinate of bounding box |
| confidence | Detection confidence score (0–1) |

Coordinates are expressed in absolute pixels relative to the original image dimensions.

Each detected defect produces one row. Images without detections have no corresponding rows.

## 8. Model Evaluation

The model can be evaluated on a labelled validation or test dataset using the evaluation script, if provided.

Evaluation should include:

- Mean Average Precision (mAP)
- Precision
- Recall
- Per-class detection performance
- Confusion matrix

For competition evaluation, use the official evaluator and its specified mAP@50:90 calculation.

## 9. Model Weights

The trained checkpoint is:

```text
best.pt
```

This file contains the trained YOLO model weights and is required for inference.

## 10. Notes

- Training requires the labelled dataset.
- Inference requires only the trained weights and input images.
- GPU acceleration is recommended for training.
- CPU inference is supported.
- Do not include virtual environments or large temporary dataset files in the repository.

## 11. Author

**Piyush Mishra**

PCB Defect Detection using Deep Learning and YOLO11.
