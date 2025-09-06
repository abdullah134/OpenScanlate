from __future__ import annotations

from typing import Tuple, List
from PIL import Image, ImageDraw, ImageFont
import textwrap
import re

Box = Tuple[int, int, int, int]


def _wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Wrap text to fit within max_width using word-wise wrapping when possible, falling back to char-wise.

    Preserves explicit newlines by wrapping each paragraph independently.
    """
    if not text:
        return [""]

    result: List[str] = []
    paragraphs = text.splitlines() or [text]
    for para in paragraphs:
        if not para:
            result.append("")
            continue

        # Heuristic: if the line has whitespace, wrap by words, else wrap by characters (useful for Japanese)
        has_space = bool(re.search(r"\s", para))
        if has_space:
            words = para.split()
            line = ""
            for w in words:
                cand = w if not line else f"{line} {w}"
                if draw.textlength(cand, font=font) <= max_width:
                    line = cand
                else:
                    if line:
                        result.append(line)
                        line = w
                    else:
                        # Single very long word; fall back to char wrapping
                        for ch in w:
                            cand2 = line + ch
                            if draw.textlength(cand2, font=font) <= max_width:
                                line = cand2
                            else:
                                if line:
                                    result.append(line)
                                line = ch
            if line:
                result.append(line)
        else:
            line = ""
            for ch in para:
                cand = line + ch
                if draw.textlength(cand, font=font) <= max_width:
                    line = cand
                else:
                    if line:
                        result.append(line)
                    line = ch
            if line:
                result.append(line)
    return result or [""]


def draw_text_in_box(
    image: Image.Image,
    box: Box,
    text: str,
    font_path: str,
    font_size: int = 28,
    align: str = "center",
    *,
    min_font_size: int = 8,
    margin: int = 4,
) -> None:
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    avail_w = max(1, w - 2 * margin)
    avail_h = max(1, h - 2 * margin)

    draw = ImageDraw.Draw(image)

    # Try decreasing font sizes until text fits
    size = max(min_font_size, font_size)
    fitted_lines: List[str] = []
    fitted_font: ImageFont.FreeTypeFont | None = None
    fitted_h = 0
    while size >= min_font_size:
        font = ImageFont.truetype(font_path, size)
        lines = _wrap_text_to_width(draw, text.strip(), font, avail_w)
        # Measure block
        line_spacing = int(size * 0.25)
        total_h = 0
        max_line_w = 0
        for ln in lines:
            bbox = draw.textbbox((0, 0), ln, font=font, align=align)
            line_w = bbox[2] - bbox[0]
            line_h = bbox[3] - bbox[1]
            total_h += line_h
            max_line_w = max(max_line_w, line_w)
        if lines:
            total_h += (len(lines) - 1) * line_spacing

        if max_line_w <= avail_w and total_h <= avail_h:
            fitted_lines = lines
            fitted_font = font
            fitted_h = total_h
            break
        size -= 1

    # If never fit, draw with the smallest size and clip naturally within the image
    if fitted_font is None:
        fitted_font = ImageFont.truetype(font_path, min_font_size)
        fitted_lines = _wrap_text_to_width(draw, text.strip(), fitted_font, avail_w)
        line_spacing = int(min_font_size * 0.2)
        total_h = 0
        for ln in fitted_lines:
            bbox = draw.textbbox((0, 0), ln, font=fitted_font, align=align)
            total_h += (bbox[3] - bbox[1])
        fitted_h = total_h + max(0, (len(fitted_lines) - 1) * line_spacing)

    # Vertical placement (center within available area)
    curr_y = y1 + margin + max(0, (avail_h - fitted_h) // 2)
    for ln in fitted_lines:
        bbox = draw.textbbox((0, 0), ln, font=fitted_font, align=align)
        line_w = bbox[2] - bbox[0]
        line_h = bbox[3] - bbox[1]
        if align == "center":
            curr_x = x1 + margin + max(0, (avail_w - line_w) // 2)
        elif align == "right":
            curr_x = x2 - margin - line_w
        else:  # left
            curr_x = x1 + margin
        draw.text((curr_x, curr_y), ln, font=fitted_font, fill=(0, 0, 0))
        curr_y += line_h + int((fitted_font.size if hasattr(fitted_font, 'size') else size) * 0.25)


def paste_texts(
    cleaned_image: Image.Image,
    boxes: list[Box],
    texts: list[str],
    font_path: str,
    font_size: int = 28,
    align: str = "center",
) -> Image.Image:
    out = cleaned_image.copy()
    for box, text in zip(boxes, texts):
        draw_text_in_box(out, box, text, font_path, font_size, align)
    return out
