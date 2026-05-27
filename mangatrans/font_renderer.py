"""Font renderer — wrap PIL text rendering với multi-line, stroke, anti-alias.

Tách khỏi text.py để text.py không phụ thuộc PIL trực tiếp. typography_engine
gọi vào đây qua interface đơn giản.

Hỗ trợ:
- Multi-line draw centered theo line.
- Stroke (outline) + fill text — manga style.
- Adaptive line spacing.
- Vertical text (chuyển CJK → cột bằng cách break ký tự).
- Anti-alias bằng cách render lên RGBA layer + composite.

KHÔNG tự wrap text — caller phải đưa lines đã wrap (typography_engine làm việc đó).
KHÔNG tự decide font size — caller chỉ ra size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass
class TextStyle:
    """Style render. Single-source-of-truth cho từng câu render."""

    font_path: str
    font_size: int
    color_rgb: tuple[int, int, int] = (0, 0, 0)
    stroke_color_rgb: tuple[int, int, int] = (255, 255, 255)
    stroke_width: int = 2
    line_spacing: float = 1.05
    align: str = "center"   # 'center' | 'left' | 'right'
    vertical: bool = False
    kerning_em: float = 0.0  # extra letter spacing as fraction of em
    italic_skew: float = 0.0  # 0=no skew, 0.18=oblique simulation


def measure_lines(lines: Sequence[str], style: TextStyle) -> tuple[int, int]:
    """Trả (width, height) tổng của khối text."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(style.font_path, style.font_size)
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    if not lines:
        return 0, 0
    if style.vertical:
        # mỗi "line" thực ra là cột — width = sum cột, height = max chars * lh
        ascent, descent = font.getmetrics()
        lh = ascent + descent
        widths = []
        max_chars = 0
        for ln in lines:
            chars = list(ln)
            max_chars = max(max_chars, len(chars))
            w = max((draw.textlength(c, font=font) for c in chars), default=0)
            widths.append(int(w))
        total_w = int(sum(widths) + (len(widths) - 1) * widths[0] * 0.15)
        total_h = int(max_chars * lh * style.line_spacing)
        return total_w, total_h

    ascent, descent = font.getmetrics()
    lh = ascent + descent
    widths = [draw.textlength(ln, font=font) for ln in lines]
    total_h = lh + (len(lines) - 1) * lh * style.line_spacing
    return int(max(widths, default=0)), int(total_h)


def draw_lines(pil_img, anchor_xy: tuple[float, float],
               lines: Sequence[str], style: TextStyle,
               box_width: Optional[float] = None) -> None:
    """Draw text vào PIL image tại anchor_xy (top-left của khối).

    Nếu align != 'left' và box_width provided → căn line theo box_width.
    """
    from PIL import ImageDraw, ImageFont
    font = ImageFont.truetype(style.font_path, style.font_size)
    draw = ImageDraw.Draw(pil_img)
    ascent, descent = font.getmetrics()
    lh = ascent + descent
    x0, y0 = anchor_xy

    if style.vertical:
        _draw_vertical(draw, font, lines, x0, y0, style)
        return

    y = y0
    for line in lines:
        line_w = draw.textlength(line, font=font)
        if style.align == "center" and box_width is not None:
            x = x0 + (box_width - line_w) / 2
        elif style.align == "right" and box_width is not None:
            x = x0 + (box_width - line_w)
        else:
            x = x0
        eff_stroke = max(1, style.font_size // 30) if style.stroke_width else 0
        if style.stroke_width:
            eff_stroke = max(eff_stroke, style.stroke_width)
        draw.text(
            (x, y), line,
            fill=style.color_rgb,
            font=font,
            stroke_width=eff_stroke,
            stroke_fill=style.stroke_color_rgb,
        )
        y += lh * style.line_spacing


def _draw_vertical(draw, font, lines, x0, y0, style):
    """Render text dạng cột phải→trái. Mỗi `line` là 1 cột."""
    ascent, descent = font.getmetrics()
    lh = ascent + descent
    col_w_default = font.getlength("国") or font.getlength("M") or style.font_size

    cols = list(lines)
    # Cột phải nhất là cột đầu — manga đọc R→L
    x = x0
    for col in reversed(cols):
        y = y0
        for ch in col:
            cw = draw.textlength(ch, font=font) or col_w_default
            cx = x + (col_w_default - cw) / 2
            draw.text(
                (cx, y), ch,
                fill=style.color_rgb,
                font=font,
                stroke_width=max(1, style.font_size // 30)
                if style.stroke_width else 0,
                stroke_fill=style.stroke_color_rgb,
            )
            y += lh * style.line_spacing
        x += col_w_default * 1.15


def bgr_to_pil(image_bgr: np.ndarray):
    from PIL import Image
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_bgr(pil) -> np.ndarray:
    arr = np.array(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
