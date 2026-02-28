import cv2
import numpy as np


def enhance_plate_image(plate_bgr: np.ndarray) -> np.ndarray:
    """Enhance cropped plate image for OCR readability.

    Steps:
    1) Grayscale
    2) Bilateral filtering
    3) Adaptive thresholding
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    enhanced = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    return enhanced
