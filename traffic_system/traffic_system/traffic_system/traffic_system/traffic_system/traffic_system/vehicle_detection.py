from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

from traffic_system.types import Detection, VehicleRiderAssociation


class VehicleDetector:
    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)

    def detect_motorcycles_and_riders(self, image_bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(image_bgr, device="cpu", verbose=False)
        detections: List[Detection] = []

        for result in results:
            names = result.names
            for box in result.boxes:
                cls_idx = int(box.cls.item())
                cls_name = names[cls_idx]
                if cls_name in {"motorcycle", "person", "rider"}:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append(
                        Detection(
                            cls_name=cls_name,
                            confidence=float(box.conf.item()),
                            bbox=(x1, y1, x2, y2),
                        )
                    )
        return detections

    def associate_riders_to_vehicles(self, detections: List[Detection]) -> List[VehicleRiderAssociation]:
        vehicles = [d for d in detections if d.cls_name == "motorcycle"]
        riders = [d for d in detections if d.cls_name in {"person", "rider"}]
        associations: List[VehicleRiderAssociation] = []

        for vehicle in vehicles:
            best_rider = None
            best_overlap = 0.0
            for rider in riders:
                overlap = self._iou(vehicle.bbox, rider.bbox)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_rider = rider
            associations.append(
                VehicleRiderAssociation(
                    vehicle_bbox=vehicle.bbox,
                    rider_bbox=best_rider.bbox if best_rider else None,
                )
            )
        return associations

    @staticmethod
    def _iou(box_a, box_b) -> float:
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b

        inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
        inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
        area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)

        denom = area_a + area_b - inter_area
        return inter_area / denom if denom > 0 else 0.0


def draw_detections(image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
    vis = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{det.cls_name} {det.confidence:.2f}",
            (x1, max(10, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return vis
