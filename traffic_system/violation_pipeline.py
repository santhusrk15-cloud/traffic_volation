from datetime import datetime
from pathlib import Path
from typing import List

import cv2

from traffic_system.constants import DEFAULT_PENALTY_194D, HELMET_CLASSES, VIOLATION_SECTION_194D
from traffic_system.image_enhancement import enhance_plate_image
from traffic_system.ocr_module import PlateOCR
from traffic_system.plate_detection import PlateDetector
from traffic_system.types import Detection, ViolationRecord
from traffic_system.vehicle_detection import VehicleDetector, draw_detections
from ultralytics import YOLO


class HelmetViolationDetector:
    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)

    def detect(self, image_bgr) -> List[Detection]:
        results = self.model.predict(image_bgr, device="cpu", verbose=False)
        detections: List[Detection] = []

        for result in results:
            names = result.names
            for box in result.boxes:
                cls_idx = int(box.cls.item())
                cls_name = names[cls_idx]
                if cls_name in HELMET_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append(
                        Detection(
                            cls_name=cls_name,
                            confidence=float(box.conf.item()),
                            bbox=(x1, y1, x2, y2),
                        )
                    )
        return detections


class TrafficViolationPipeline:
    def __init__(
        self,
        helmet_model_path: str,
        vehicle_model_path: str,
        plate_model_path: str,
        save_crops: bool = False,
    ) -> None:
        self.helmet_detector = HelmetViolationDetector(helmet_model_path)
        self.vehicle_detector = VehicleDetector(vehicle_model_path)
        self.plate_detector = PlateDetector(plate_model_path)
        self.ocr = PlateOCR()
        self.save_crops = save_crops

    def process_image(self, image_path: Path, output_dir: Path, save_visuals: bool = False) -> List[ViolationRecord]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        crop_dir = output_dir / "crops"
        if self.save_crops:
            crop_dir.mkdir(parents=True, exist_ok=True)

        helmet_dets = self.helmet_detector.detect(image)
        vehicle_dets = self.vehicle_detector.detect_motorcycles_and_riders(image)
        _ = self.vehicle_detector.associate_riders_to_vehicles(vehicle_dets)

        plate_dets = self.plate_detector.detect_plates(image)
        plate_texts = []
        for idx, plate in enumerate(plate_dets):
            x1, y1, x2, y2 = plate.bbox
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            enhanced = enhance_plate_image(crop)
            text = self.ocr.read_plate(enhanced)
            if text:
                plate_texts.append(text)
            if self.save_crops:
                cv2.imwrite(str(crop_dir / f"plate_{idx}.png"), enhanced)

        violations = [
            d for d in helmet_dets if d.cls_name in {"driver_without_helmet", "passenger_without_helmet"}
        ]
        primary_plate = plate_texts[0] if plate_texts else ""

        records = [
            ViolationRecord(
                timestamp=datetime.now(),
                detected_vehicle_number=primary_plate,
                violation_type=v.cls_name,
                mv_act_section=VIOLATION_SECTION_194D,
                penalty_amount=DEFAULT_PENALTY_194D,
            )
            for v in violations
        ]

        if save_visuals:
            vis_helmet = draw_detections(image, helmet_dets)
            vis_all = draw_detections(vis_helmet, vehicle_dets + plate_dets)
            cv2.imwrite(str(output_dir / f"{image_path.stem}_annotated.jpg"), vis_all)

        return records
