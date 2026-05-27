"""Text rendering — wrap/fit/render (consolidated).

Gộp `font_renderer.py` (PIL primitives) + `text.py` (wrap/fit helpers) +
`typography_engine.py` (tier-cascade orchestrator).

Cascade:
    Tier 1: shape-aware mask + polygon_row_extents
    Tier 2: shape-aware với min_size thấp hơn (tight pack)
    Tier 3: largest_inscribed_rect axis-aligned trong interior
    Tier 4: bbox gốc với min_size=8 (cuối cùng)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from .config import GeometryConfig, RenderConfig
from .geometry import (
    InteriorCache,
    find_bubble_interior,
    find_bubble_polygon,
    largest_inscribed_rect,
    polygon_row_extents,
    row_extents,
)
from .utils import get_logger


# =============================================================
# Section 1: Font renderer primitives (was font_renderer.py)
# =============================================================

@dataclass
class TextStyle:
    """Single-source-of-truth cho từng câu render."""

    font_path: str
    font_size: int
    color_rgb: tuple[int, int, int] = (0, 0, 0)
    stroke_color_rgb: tuple[int, int, int] = (255, 255, 255)
    stroke_width: int = 2
    line_spacing: float = 1.05
    align: str = "center"
    vertical: bool = False
    kerning_em: float = 0.0
    italic_skew: float = 0.0


def measure_lines(lines: Sequence[str], style: TextStyle) -> tuple[int, int]:
    """Trả (width, height) tổng của khối text."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(style.font_path, style.font_size)
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    if not lines:
        return 0, 0
    if style.vertical:
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
    """Draw text vào PIL image tại anchor_xy (top-left của khối)."""
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
    """Render text dạng cột phải→trái."""
    ascent, descent = font.getmetrics()
    lh = ascent + descent
    col_w_default = font.getlength("国") or font.getlength("M") or style.font_size

    cols = list(lines)
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


# =============================================================
# Section 2: Wrap + fit helpers (was text.py)
# =============================================================

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
    """Aspect-aware cap. Mục tiêu: bám sát stroke height gốc, không phình."""
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


def render_text_in_mask(pil_img, mask, text, font_path,
                        text_bbox=None, polygon_contour=None,
                        padding=4, color=(0, 0, 0),
                        stroke_color=(255, 255, 255), stroke_width=2,
                        min_size=10, max_size=220, line_spacing=1.05,
                        is_sfx=False) -> bool:
    """Shape-aware render: mỗi dòng dùng width thực của bubble tại y đó."""
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

    if polygon_contour is not None and len(polygon_contour) >= 3:
        anchor_x = (float((text_bbox[0] + text_bbox[2]) / 2)
                    if text_bbox is not None else None)
        extents = polygon_row_extents(polygon_contour, mask.shape[0],
                                      anchor_x=anchor_x)
        extents_filtered = extents
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


# =============================================================
# Section 3: Typography engine (was typography_engine.py)
# =============================================================

@dataclass
class FitResult:
    """Result của fit_to_bubble. None khi không fit."""

    success: bool = False
    tier: int = 0
    font_size: int = 0
    lines: list[str] = field(default_factory=list)
    line_centers: Optional[list[tuple[float, float]]] = None
    render_bbox: Optional[tuple[int, int, int, int]] = None
    debug: dict = field(default_factory=dict)


class TypographyEngine:
    """Fit text vào bubble theo tier cascade. Reusable across pages."""

    def __init__(self, render_cfg: RenderConfig, geom_cfg: GeometryConfig,
                 font_path_resolver=None):
        self.render_cfg = render_cfg
        self.geom_cfg = geom_cfg
        self._log = get_logger()
        self._resolver = font_path_resolver

    def fit_and_render(self, pil_img, item: dict, all_items: list[dict],
                       text: str, original_image: np.ndarray,
                       text_seg_mask: Optional[np.ndarray] = None,
                       interior_cache: Optional[InteriorCache] = None,
                       script_code: str = "vi") -> FitResult:
        """End-to-end fit + render 1 item."""
        font_path = self._resolve_font(script_code)

        result = self._try_shape_aware(
            pil_img, item, all_items, text, original_image,
            text_seg_mask, interior_cache, font_path,
        )
        if result.success:
            return result

        render_bbox = (result.render_bbox or tuple(item["bbox"]))
        ok = self._tier4_bbox(pil_img, render_bbox, text, font_path, item["bbox"])
        return FitResult(
            success=ok,
            tier=4 if ok else 0,
            font_size=0,
            lines=[text] if ok else [],
            render_bbox=render_bbox,
        )

    def _resolve_font(self, script_code: str) -> str:
        if self._resolver is not None:
            try:
                p = self._resolver(script_code)
                if p:
                    return p
            except Exception:  # noqa: BLE001
                pass
        return self.render_cfg.font_path

    def _try_shape_aware(self, pil_img, item, all_items, text,
                         original_image, text_seg_mask, interior_cache,
                         font_path) -> FitResult:
        cfg = self.render_cfg
        geom = self.geom_cfg
        if original_image is None:
            return FitResult()

        other_bxs = [r["bbox"] for r in all_items if r is not item]
        polygon_contour = None
        interior = None

        if text_seg_mask is not None:
            poly = find_bubble_polygon(original_image, item["bbox"],
                                       text_seg_mask, geom,
                                       other_bboxes=other_bxs)
            if poly is not None:
                interior = poly["mask"]
                polygon_contour = poly["contour"]
        if interior is None:
            interior = find_bubble_interior(original_image, item["bbox"], geom,
                                            other_bboxes=other_bxs,
                                            cache=interior_cache)
        if interior is None or not interior.any():
            return FitResult()

        tx1, ty1, tx2, ty2 = item["bbox"]
        text_area = max(1, (tx2 - tx1) * (ty2 - ty1))
        tbw, tbh = tx2 - tx1, ty2 - ty1
        tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        ys, xs = np.where(interior > 0)
        bx1, by1 = int(xs.min()), int(ys.min())
        bx2, by2 = int(xs.max()) + 1, int(ys.max()) + 1
        bubble_pix = int((interior > 0).sum())
        img_area = original_image.shape[0] * original_image.shape[1]
        covers_text = (bx1 <= tx1 and by1 <= ty1
                       and bx2 >= tx2 and by2 >= ty2)
        bubble_cx = (bx1 + bx2) / 2
        bubble_cy = (by1 + by2) / 2
        center_offset = max(abs(bubble_cx - tcx) / max(1, tbw),
                            abs(bubble_cy - tcy) / max(1, tbh))
        centered_ok = center_offset <= 1.5
        interior_trustworthy = (covers_text and centered_ok
                                and bubble_pix <= 0.12 * img_area)
        if not interior_trustworthy:
            return FitResult()

        r = self._render_shape(pil_img, interior, text, font_path,
                               text_bbox=item["bbox"],
                               polygon_contour=polygon_contour,
                               min_size=cfg.min_size,
                               padding=cfg.padding,
                               stroke_width=cfg.stroke_width,
                               tier=1)
        if r.success:
            return r

        r = self._render_shape(pil_img, interior, text, font_path,
                               text_bbox=item["bbox"],
                               polygon_contour=polygon_contour,
                               min_size=cfg.tight_min_size,
                               padding=cfg.tight_padding,
                               stroke_width=cfg.stroke_width,
                               tier=2)
        if r.success:
            return r

        cx_hint = (tx1 + tx2) // 2
        cy_hint = (ty1 + ty2) // 2
        inscribed = largest_inscribed_rect(interior, center_hint=(cx_hint, cy_hint))
        if inscribed is not None:
            ix1, iy1, ix2, iy2 = inscribed
            ox1, oy1 = max(ix1, tx1), max(iy1, ty1)
            ox2, oy2 = min(ix2, tx2), min(iy2, ty2)
            overlap = max(0, ox2 - ox1) * max(0, oy2 - oy1)
            if overlap >= 0.5 * text_area:
                iw, ih = ix2 - ix1, iy2 - iy1
                ex, ey = int(iw * 0.04), int(ih * 0.04)
                return FitResult(success=False,
                                 render_bbox=(ix1 + ex, iy1 + ey,
                                              ix2 - ex, iy2 - ey))
        return FitResult()

    def _render_shape(self, pil_img, interior, text, font_path,
                      text_bbox, polygon_contour, min_size, padding,
                      stroke_width, tier) -> FitResult:
        cfg = self.render_cfg
        ok = render_text_in_mask(
            pil_img, interior, text, font_path,
            text_bbox=text_bbox, polygon_contour=polygon_contour,
            padding=padding,
            color=cfg.color_rgb,
            stroke_color=cfg.stroke_color_rgb,
            stroke_width=stroke_width,
            min_size=min_size, max_size=cfg.max_size,
            line_spacing=cfg.line_spacing,
        )
        return FitResult(success=ok, tier=tier,
                         lines=[text] if ok else [])

    def _tier4_bbox(self, pil_img, bbox, text, font_path, text_bbox_hint) -> bool:
        cfg = self.render_cfg
        return render_text_in_bbox(
            pil_img, bbox, text, font_path,
            text_bbox_hint=text_bbox_hint,
            padding=cfg.padding,
            color=cfg.color_rgb,
            stroke_color=cfg.stroke_color_rgb,
            stroke_width=cfg.bbox_fallback_stroke,
            min_size=cfg.bbox_fallback_min_size,
            max_size=cfg.max_size,
            line_spacing=cfg.line_spacing,
        )
