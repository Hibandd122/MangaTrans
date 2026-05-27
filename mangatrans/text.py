"""Text rendering: shape-aware text fit, bubble-aware sizing, multi-tier cascade.

Cascade (mỗi tier xuống = quality giảm, vẫn cố render thay vì bỏ trống):
    Tier 1: shape-aware mask + polygon_row_extents (polygon-based)
    Tier 2: shape-aware mask với min_size thấp hơn + padding nhỏ (tight pack)
    Tier 3: largest_inscribed_rect axis-aligned trong interior
    Tier 4: bbox gốc với min_size=8 (cuối cùng, không bao giờ vô hình)

Font sizing:
    Binary search [min_size, max_size]. Cap max_size theo text_bbox gốc
    (aspect-aware: bbox vuông = SFX/multi-line → 0.75×, dài = 1 line → 1.2×).
    Scale cap khi bubble >> text bbox: sqrt(area_ratio/3) clamp 2.5×.

Width per line:
    Polygon: scanline cắt polygon → đoạn chứa anchor_x (tail bubble bị bỏ).
    Mask fallback: longest run + filter rows có width >= 50% peak (loại tail).
    Mỗi line dùng MEDIAN width của band y → ổn định với rim noise.
"""
from __future__ import annotations

from typing import Optional

import cv2  # noqa: F401  (used by render_text_in_bbox / render_text_in_mask)
import numpy as np  # noqa: F401  (used downstream)

from .geometry import polygon_row_extents, row_extents


# --------------------------- Low-level helpers --------------------------- #

def _line_height(font) -> int:
    ascent, descent = font.getmetrics()
    return ascent + descent


def wrap_text(text: str, font, max_width: float, draw) -> list[str]:
    """Greedy word-wrap; fallback split-char chỉ khi 1 từ dài > max_width."""
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = ""
    for word in words:
        test = (current + " " + word) if current else word
        if draw.textlength(test, font=font) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
                current = word
            else:
                buf = ""
                for ch in word:
                    if draw.textlength(buf + ch, font=font) <= max_width:
                        buf += ch
                    else:
                        if buf:
                            lines.append(buf)
                        buf = ch
                current = buf
    if current:
        lines.append(current)
    return lines


def wrap_text_shape(text: str, font, width_for_line, draw,
                    max_lines: int = 30) -> Optional[list[str]]:
    """Greedy wrap với width budget khác nhau mỗi dòng (shape-aware).

    Returns None khi 1 từ đơn vượt budget → caller phải giảm font size.
    KHÔNG split char (tiếng Việt: "chưa" → "chư/a" không đọc được).
    """
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = ""
    i = 0
    wi = 0
    while wi < len(words):
        if i >= max_lines:
            return None
        max_w = width_for_line(i)
        if max_w <= 0:
            return None
        word = words[wi]
        if draw.textlength(word, font=font) > max_w:
            if current:
                lines.append(current)
                current = ""
                i += 1
                continue
            return None
        test = (current + " " + word) if current else word
        if draw.textlength(test, font=font) <= max_w:
            current = test
            wi += 1
        else:
            lines.append(current)
            current = ""
            i += 1
    if current:
        lines.append(current)
    return lines


def _cap_max_size(min_size: int, max_size: int,
                  text_bbox_hint, bubble_h: int, bubble_w: int,
                  is_sfx: bool = False) -> int:
    """Aspect-aware cap. Mục tiêu: chữ dịch SÁT kích thước chữ gốc, không phình.

    User feedback: trước đây render to gấp 2-4× chữ gốc → trông lạc. Tinh chỉnh:
      - Floor 24 (thay vì 55) → SFX/text nhỏ vẫn nhỏ.
      - Tight text nhỏ (base<60): dùng 0.95× thay 0.75× → bám sát stroke height gốc.
      - Scale-up khi bubble rộng: ngưỡng 5.0× (thay 3.0×), max 1.5× (thay 2.5×).
      - is_sfx=True: KHÔNG scale-up, cap = base × 1.0 (giữ tối đa stroke gốc).
    """
    if text_bbox_hint is None:
        return max_size
    tx1, ty1, tx2, ty2 = text_bbox_hint
    tbw = tx2 - tx1
    tbh = ty2 - ty1
    if tbw <= 0 or tbh <= 0:
        return max_size

    aspect = max(tbw, tbh) / min(tbw, tbh)
    base = min(tbw, tbh)

    if is_sfx:
        cap = max(int(base * 1.0), 24)
        return min(max_size, max(min_size + 4, cap))

    if aspect < 1.5:
        ratio = 0.95 if base < 60 else 0.85
        cap = max(int(base * ratio), 24)
    else:
        ratio = 1.0 if base < 60 else 1.1
        cap = max(int(base * ratio), 24)

    bubble_area = bubble_h * bubble_w
    text_area = tbw * tbh
    if text_area > 0 and bubble_area > 0:
        area_ratio = bubble_area / text_area
        if area_ratio > 5.0:
            scale = min((area_ratio / 5.0) ** 0.5, 1.5)
            cap = int(cap * scale)

    return min(max_size, max(min_size + 4, cap))


# --------------------------- Bbox render --------------------------- #

def render_text_in_bbox(pil_img, bbox, text, font_path,
                        padding=6, color=(0, 0, 0),
                        stroke_color=(255, 255, 255), stroke_width=2,
                        min_size=10, max_size=220, line_spacing=1.05,
                        text_bbox_hint=None, is_sfx=False) -> bool:
    """Render text vào axis-aligned bbox với binary-search font size."""
    from PIL import ImageDraw, ImageFont

    if not text.strip():
        return False
    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1 - 2 * padding)
    box_h = max(1, y2 - y1 - 2 * padding)

    max_size = _cap_max_size(min_size, max_size, text_bbox_hint,
                             y2 - y1, x2 - x1, is_sfx=is_sfx)

    draw = ImageDraw.Draw(pil_img)
    best_size = min_size
    best_lines = [text]
    lo, hi = min_size, max_size
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(font_path, mid)
        lines = wrap_text(text, font, box_w, draw)
        lh = _line_height(font)
        total_h = lh * len(lines) + (len(lines) - 1) * (lh * (line_spacing - 1))
        widest = max((draw.textlength(ln, font=font) for ln in lines), default=0)
        if widest <= box_w and total_h <= box_h:
            best_size, best_lines = mid, lines
            lo = mid + 1
        else:
            hi = mid - 1

    font = ImageFont.truetype(font_path, best_size)
    lh = _line_height(font)
    total_h = lh + (len(best_lines) - 1) * lh * line_spacing
    y = y1 + padding + (box_h - total_h) / 2
    eff_stroke = max(1, best_size // 30)
    for line in best_lines:
        line_w = draw.textlength(line, font=font)
        x = x1 + padding + (box_w - line_w) / 2
        draw.text(
            (x, y), line, fill=color, font=font,
            stroke_width=eff_stroke, stroke_fill=stroke_color,
        )
        y += lh * line_spacing
    return True


# --------------------------- Shape-aware render --------------------------- #

def render_text_in_mask(pil_img, mask, text, font_path,
                        text_bbox=None, polygon_contour=None,
                        padding=4, color=(0, 0, 0),
                        stroke_color=(255, 255, 255), stroke_width=2,
                        min_size=10, max_size=220, line_spacing=1.05,
                        is_sfx=False) -> bool:
    """Shape-aware render: mỗi dòng dùng width thực của bubble tại y dòng đó."""
    from PIL import ImageDraw, ImageFont

    if not text.strip() or mask is None:
        return False

    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return False

    bubble_top = int(ys.min()) + padding
    bubble_bot = int(ys.max()) - padding
    if bubble_top >= bubble_bot:
        return False

    bubble_h = bubble_bot - bubble_top
    bubble_w = int(xs.max() - xs.min())
    max_size = _cap_max_size(min_size, max_size, text_bbox, bubble_h, bubble_w,
                             is_sfx=is_sfx)

    if text_bbox is not None:
        _, ty1, _, ty2 = text_bbox
        anchor_y = (ty1 + ty2) / 2
        anchor_y = max(bubble_top + 1, min(bubble_bot - 1, anchor_y))
    else:
        anchor_y = (bubble_top + bubble_bot) / 2

    # Extents per scanline
    if polygon_contour is not None and len(polygon_contour) >= 3:
        anchor_x = (float((text_bbox[0] + text_bbox[2]) / 2)
                    if text_bbox is not None else None)
        extents = polygon_row_extents(polygon_contour, mask.shape[0],
                                      anchor_x=anchor_x)
        extents_filtered = extents  # polygon đã chọn segment chứa anchor
    else:
        extents = row_extents(mask)
        valid_widths = [(r[1] - r[0]) for r in extents if r is not None]
        if not valid_widths:
            return False
        width_floor = max(valid_widths) * 0.5
        extents_filtered = [
            (r if (r is not None and (r[1] - r[0]) >= width_floor) else None)
            for r in extents
        ]

    h_ext = len(extents)
    if not any(r is not None for r in extents_filtered):
        return False
    draw = ImageDraw.Draw(pil_img)

    def width_at_band(y_top, y_bot):
        y0 = max(int(y_top), bubble_top)
        y1 = min(int(y_bot), bubble_bot)
        if y0 > y1:
            return 0, 0
        rows = []
        for y in range(y0, y1 + 1):
            if 0 <= y < h_ext and extents_filtered[y] is not None:
                rows.append(extents_filtered[y])
        if not rows:
            for y in range(y0, y1 + 1):
                if 0 <= y < h_ext and extents[y] is not None:
                    rows.append(extents[y])
            if not rows:
                return 0, 0
        widths = sorted(r[1] - r[0] for r in rows)
        med_w = widths[len(widths) // 2]
        avg_cx = sum((r[0] + r[1]) / 2 for r in rows) / len(rows)
        return med_w, avg_cx

    def try_size(size):
        font = ImageFont.truetype(font_path, size)
        lh = _line_height(font)
        step = lh * line_spacing
        if lh > (bubble_bot - bubble_top):
            return None
        n_lines = 1
        for _ in range(8):
            total_h = lh + (n_lines - 1) * step
            if total_h > (bubble_bot - bubble_top):
                return None
            y_start = anchor_y - total_h / 2
            if y_start < bubble_top:
                y_start = bubble_top
            if y_start + total_h > bubble_bot:
                y_start = bubble_bot - total_h

            def width_for_line(i):
                y_top = y_start + i * step
                y_bot = y_top + lh
                if y_bot > bubble_bot + 0.5:
                    return 0
                bw, _ = width_at_band(y_top, y_bot)
                return max(0, bw - 2 * padding - 2 * stroke_width)

            lines = wrap_text_shape(text, font, width_for_line, draw, max_lines=30)
            if lines is None or len(lines) == 0:
                return None
            if len(lines) == n_lines:
                centers = []
                ok = True
                for i, ln in enumerate(lines):
                    y_top = y_start + i * step
                    y_bot = y_top + lh
                    bw, cx = width_at_band(y_top, y_bot)
                    eff_w = max(0, bw - 2 * padding - 2 * stroke_width)
                    if draw.textlength(ln, font=font) > eff_w:
                        ok = False
                        break
                    centers.append((cx, y_top))
                if not ok:
                    return None
                return (font, lines, centers)
            n_lines = max(1, len(lines))
        return None

    best = None
    lo, hi = min_size, max_size
    while lo <= hi:
        mid = (lo + hi) // 2
        r = try_size(mid)
        if r is not None:
            best = r
            lo = mid + 1
        else:
            hi = mid - 1

    if best is None:
        return False

    font, lines, centers = best
    # Honor caller's stroke_width:
    #   - stroke_width > 0  → exact px (per-bubble hint từ stroke_detector)
    #   - stroke_width == 0 → render plain (không vẽ stroke)
    #   - stroke_width < 0  → fallback heuristic dựa trên font size
    if stroke_width < 0:
        eff_stroke = max(1, font.size // 30)
    else:
        eff_stroke = stroke_width
    for ln, (cx, y_top) in zip(lines, centers):
        line_w = draw.textlength(ln, font=font)
        x = cx - line_w / 2
        draw.text(
            (x, y_top), ln, fill=color, font=font,
            stroke_width=eff_stroke, stroke_fill=stroke_color,
        )
    return True


# --------------------------- Renderer orchestrator --------------------------- #
#
# `TextRenderer` class đã bị xóa khi consolidate (2026-05-25). Render path duy
# nhất giờ là `TypographyEngine` (typography_engine.py) — engine đó import
# `wrap_text`, `wrap_text_shape`, `_cap_max_size`, `render_text_in_mask`,
# `render_text_in_bbox` từ file này.
