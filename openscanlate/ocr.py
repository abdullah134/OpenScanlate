from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

try:
    import easyocr  # type: ignore
except Exception as e:  # pragma: no cover
    easyocr = None

# Optional: use Manga-OCR recognizer for Japanese text, while EasyOCR gives detection boxes
try:
    from manga_ocr import MangaOcr  # type: ignore
except Exception:
    MangaOcr = None  # type: ignore
from PIL import Image


@dataclass
class OcrBox:
    # Bounding box as (x1, y1, x2, y2) in pixels
    box: Tuple[int, int, int, int]
    text: str
    score: float


def _to_box(points: List[Tuple[float, float]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    return x1, y1, x2, y2


def detect_and_recognize(image: np.ndarray, use_manga_ocr: bool = False) -> List[OcrBox]:
    """Detect text regions and recognize text.

    Strategy:
    - Use EasyOCR for detection+recognition initially for all boxes.
    - If use_manga_ocr and MangaOcr is installed, reuse EasyOCR boxes, but re-recognize each crop with Manga-OCR.

    Returns list of OcrBox with bounding boxes and text.
    """
    if easyocr is None:
        raise RuntimeError(
            "easyocr is not installed. Please install dependencies from requirements.txt"
        )

    reader = easyocr.Reader(["ja", "en"], gpu=False)
    results = reader.readtext(image, detail=1)  # list of (bbox, text, score)

    ocr_boxes: List[OcrBox] = []
    if not results:
        return ocr_boxes

    # Optionally init Manga-OCR recognizer
    # Avoid typing errors if MangaOcr isn't available in this environment
    mo: Optional[object] = None
    if use_manga_ocr and MangaOcr is not None:
        mo = MangaOcr()

    for bbox, text, score in results:
        x1, y1, x2, y2 = _to_box(bbox)
        if mo is not None:
            # Re-recognize using Manga-OCR on the cropped region
            crop = image[y1:y2, x1:x2]
            try:
                pil_crop = Image.fromarray(crop)
                rec_text = mo(pil_crop)  # type: ignore[operator]
            except Exception:
                rec_text = text
        else:
            rec_text = text
        ocr_boxes.append(OcrBox(box=(x1, y1, x2, y2), text=rec_text, score=float(score)))

    return ocr_boxes
