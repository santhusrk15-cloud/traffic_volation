from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple


BBox = Tuple[int, int, int, int]


@dataclass
class Detection:
    cls_name: str
    confidence: float
    bbox: BBox


@dataclass
class VehicleRiderAssociation:
    vehicle_bbox: BBox
    rider_bbox: Optional[BBox]


@dataclass
class ViolationRecord:
    timestamp: datetime
    detected_vehicle_number: str
    violation_type: str
    mv_act_section: str
    penalty_amount: int
