from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np
import cv2

Box = Tuple[int, int, int, int]


def inpaint_regions(image: np.ndarray, boxes: Iterable[Box]) -> np.ndarray:
    """Inpaint all given boxes on the image and return a cleaned copy.

    Uses OpenCV Telea algorithm for speed and decent quality.
    """
    cleaned = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        # Inflate slightly to cover outlines
        pad = 2
        x1i = max(0, x1 - pad)
        y1i = max(0, y1 - pad)
        x2i = min(cleaned.shape[1] - 1, x2 + pad)
        y2i = min(cleaned.shape[0] - 1, y2 + pad)
        mask[y1i:y2i, x1i:x2i] = 255
    cleaned = cv2.inpaint(cleaned, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cleaned
