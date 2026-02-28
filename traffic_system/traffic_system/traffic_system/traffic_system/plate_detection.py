import random
import shutil
from pathlib import Path
from typing import List

import yaml
from ultralytics import YOLO

from traffic_system.constants import PLATE_CLASS_NAME
from traffic_system.types import Detection


class PlateDetector:
    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)

    def detect_plates(self, image_bgr) -> List[Detection]:
        results = self.model.predict(image_bgr, device="cpu", verbose=False)
        detections: List[Detection] = []

        for result in results:
            names = result.names
            for box in result.boxes:
                cls_idx = int(box.cls.item())
                cls_name = names[cls_idx]
                if cls_name == PLATE_CLASS_NAME:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append(
                        Detection(
                            cls_name=cls_name,
                            confidence=float(box.conf.item()),
                            bbox=(x1, y1, x2, y2),
                        )
                    )
        return detections


def split_yolo_dataset(dataset_root: Path, val_ratio: float = 0.2, seed: int = 42) -> Path:
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    all_images = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )

    random.Random(seed).shuffle(all_images)
    val_count = max(1, int(len(all_images) * val_ratio))
    val_images = set(all_images[:val_count])

    split_root = dataset_root / "split"
    for sub in ["train/images", "train/labels", "val/images", "val/labels"]:
        (split_root / sub).mkdir(parents=True, exist_ok=True)

    for img_path in all_images:
        label_path = labels_dir / f"{img_path.stem}.txt"
        split_name = "val" if img_path in val_images else "train"
        shutil.copy2(img_path, split_root / split_name / "images" / img_path.name)
        if label_path.exists():
            shutil.copy2(label_path, split_root / split_name / "labels" / label_path.name)

    data_yaml_path = split_root / "data.yaml"
    data_yaml = {
        "path": str(split_root),
        "train": "train/images",
        "val": "val/images",
        "names": [PLATE_CLASS_NAME],
        "nc": 1,
    }
    data_yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    return data_yaml_path


def train_plate_detector_cpu(
    data_yaml_path: Path,
    output_root: Path,
    epochs: int = 50,
    imgsz: int = 640,
) -> Path:
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=8,
        device="cpu",
        workers=2,
        project=str(output_root),
        name="plate_yolov8n_cpu",
    )
    return Path(results.save_dir) / "weights" / "best.pt"
