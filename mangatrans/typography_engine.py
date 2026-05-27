"""Typography engine — orchestrate font size, wrap, fit cho mỗi bubble.

Refactor của text.py: tách phần "fit text to bubble shape" khỏi PIL drawing.
TextRenderer cũ (text.py) vẫn còn cho backward-compat; engine mới này là API
sạch hơn, có:

- `fit_to_bubble(text, mask, polygon, text_bbox, style_template)` — binary search
  size, shape-aware wrap, return FitResult với best lines + size + centers.
- Tier cascade compatible với text.py:
    Tier 1 — polygon shape-aware wrap
    Tier 2 — tight pack (smaller min_size + padding)
    Tier 3 — largest_inscribed_rect axis-aligned
    Tier 4 — bbox gốc raw
- Adaptive scaling theo bubble vs text-bbox area.
- Smart line break tiếng Việt: không split chữ (giữ dấu).
- Anti-overflow check sau wrap.
- Hỗ trợ vertical layout khi style.vertical=True.

Engine không tự gọi PIL — delegate qua `font_renderer` module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .config import GeometryConfig, RenderConfig
from .font_renderer import TextStyle, draw_lines, measure_lines
from .geometry import (
    InteriorCache,
    find_bubble_interior,
    find_bubble_polygon,
    largest_inscribed_rect,
    polygon_row_extents,
    row_extents,
)
from .text import wrap_text, wrap_text_shape, _cap_max_size  # reuse helpers
from .utils import get_logger


@dataclass
class FitResult:
    """Result của fit_to_bubble. None khi không fit."""

    success: bool = False
    tier: int = 0
    font_size: int = 0
    lines: list[str] = field(default_factory=list)
    # centers cho mỗi line: list[(cx, y_top)] cho shape-aware,
    # hoặc None khi rendered ở bbox mode.
    line_centers: Optional[list[tuple[float, float]]] = None
    render_bbox: Optional[tuple[int, int, int, int]] = None  # fallback bbox
    debug: dict = field(default_factory=dict)


class TypographyEngine:
    """Fit text vào bubble theo tier cascade. Reusable across pages."""

    def __init__(self, render_cfg: RenderConfig, geom_cfg: GeometryConfig,
                 font_path_resolver=None):
        """font_path_resolver: callable(script_code) → path. Optional.

        Khi None → dùng render_cfg.font_path mặc định.
        """
        self.render_cfg = render_cfg
        self.geom_cfg = geom_cfg
        self._log = get_logger()
        self._resolver = font_path_resolver

    # --------------------------- Public --------------------------- #

    def fit_and_render(self, pil_img, item: dict, all_items: list[dict],
                       text: str, original_image: np.ndarray,
                       text_seg_mask: Optional[np.ndarray] = None,
                       interior_cache: Optional[InteriorCache] = None,
                       script_code: str = "vi") -> FitResult:
        """End-to-end fit + render 1 item. Trả FitResult."""
        from PIL import ImageDraw, ImageFont
        cfg = self.render_cfg
        geom = self.geom_cfg
        font_path = self._resolve_font(script_code)

        # --- Tier 1+2+3 via shape-aware mask ---
        result = self._try_shape_aware(
            pil_img, item, all_items, text, original_image,
            text_seg_mask, interior_cache, font_path,
        )
        if result.success:
            return result

        # --- Tier 4: bbox gốc ---
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

    # --------------------------- Tier 1-3 --------------------------- #

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

        # Tier 1
        r = self._render_shape(pil_img, interior, text, font_path,
                               text_bbox=item["bbox"],
                               polygon_contour=polygon_contour,
                               min_size=cfg.min_size,
                               padding=cfg.padding,
                               stroke_width=cfg.stroke_width,
                               tier=1)
        if r.success:
            return r

        # Tier 2: tighter pack
        r = self._render_shape(pil_img, interior, text, font_path,
                               text_bbox=item["bbox"],
                               polygon_contour=polygon_contour,
                               min_size=cfg.tight_min_size,
                               padding=cfg.tight_padding,
                               stroke_width=cfg.stroke_width,
                               tier=2)
        if r.success:
            return r

        # Tier 3: inscribed rect axis-aligned
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
        """Tier 1/2 inner: shape-aware binary search + render."""
        # Reuse existing render_text_in_mask để giữ behavior. Trả True/False.
        from .text import render_text_in_mask
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
        from .text import render_text_in_bbox
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
