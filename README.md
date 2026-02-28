# AI-Integrated Traffic Violation Detection System (Indian Roads)

This project provides a **CPU-friendly, modular Python pipeline** for automated traffic enforcement from images:

- Helmet violation detection using **YOLOv8**
- Motorcycle + rider detection and association
- Indian license plate detection + crop extraction
- Image enhancement for blurry plates
- OCR via **EasyOCR** with Indian plate post-processing
- Violation report generation mapped to **MV Act Section 194D**

## Features

### 1) Helmet Violation Detection Module
Detects these classes from a custom YOLOv8 model:
- `driver_with_helmet`
- `driver_without_helmet`
- `passenger_with_helmet`
- `passenger_without_helmet`

Violation logic:
- If driver/passenger are detected without helmet, a violation is raised under **Section 194D**.

### 2) Vehicle Detection Module
Uses a pretrained YOLO model to detect:
- `motorcycle`
- `rider`/`person`

Then associates riders to motorcycles via IoU/containment heuristics.

### 3) License Plate Detection Module
Includes utilities for:
- Splitting YOLO-format plate dataset into train/val
- Training `yolov8n` on CPU
- Exporting the best model (`best.pt`)

### 4) Image Enhancement Module
For plate crops:
- Grayscale
- Bilateral filter
- Adaptive thresholding

### 5) OCR Module
Uses EasyOCR and post-processes text:
- Remove spaces and non-alphanumeric chars
- Uppercase
- Validate minimum Indian plate length (`>=8`)

### 6) Violation Report Generator
Produces structured records with:
- Timestamp
- Detected number plate
- Violation type
- MV Act section
- Penalty amount

### 7) CPU-First Design
Default configuration is optimized for Intel i5-like CPU systems:
- YOLO inference/training on `device=cpu`
- YOLOv8n for plate training

## Project Structure

```
traffic_volation/
├── requirements.txt
├── main.py
└── traffic_system/
    ├── __init__.py
    ├── constants.py
    ├── image_enhancement.py
    ├── ocr_module.py
    ├── plate_detection.py
    ├── report_generator.py
    ├── types.py
    ├── vehicle_detection.py
    └── violation_pipeline.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### A) Run End-to-End Inference on an Image

```bash
python main.py infer \
  --image /path/to/traffic_image.jpg \
  --helmet-model /path/to/helmet_best.pt \
  --vehicle-model yolov8n.pt \
  --plate-model /path/to/plate_best.pt \
  --save-visuals
```

Output:
- Console violation report
- Optional visualization under `outputs/`
- Optional cropped plate images under `outputs/crops/`

### B) Train Plate Detector (YOLOv8n, CPU)

Dataset should follow YOLO format:
- `images/` with `.jpg/.png`
- `labels/` with `.txt`
- Single class: `number_plate`

```bash
python main.py train-plates \
  --dataset-root /path/to/indian_plate_dataset \
  --output-root runs/plate_training \
  --epochs 50
```

This command:
1. Splits data into train/val
2. Generates `data.yaml`
3. Trains YOLOv8n on CPU
4. Exports best model path

## Optional Extensions Included in Code Design
- Bounding-box visualization (`--save-visuals`)
- Plate crop saving (`--save-crops`)
- Hooks for Flask dashboard and IoT integration (extensible in pipeline output format)
- Real-time logging can be added by calling pipeline repeatedly over camera frames

## Legal Note
This is a technical prototype for research/educational deployment. Ensure local legal/privacy compliance before real-world use.
