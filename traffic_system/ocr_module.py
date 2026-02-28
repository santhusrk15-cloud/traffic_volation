import re
from typing import Optional

import easyocr
import numpy as np

from traffic_system.constants import MIN_INDIAN_PLATE_LENGTH


class PlateOCR:
    def __init__(self, languages: Optional[list[str]] = None) -> None:
        self.reader = easyocr.Reader(languages or ["en"], gpu=False)

    def read_plate(self, image: np.ndarray) -> str:
        results = self.reader.readtext(image, detail=0)
        merged = "".join(results)
        return self._post_process_text(merged)

    @staticmethod
    def _post_process_text(raw_text: str) -> str:
        text = re.sub(r"[^A-Za-z0-9]", "", raw_text.replace(" ", "")).upper()
        return text if len(text) >= MIN_INDIAN_PLATE_LENGTH else ""
