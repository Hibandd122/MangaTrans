"""Hybrid redraw orchestrator: classify → LaMa HD-tiled → cv2.inpaint fallback.

Bố cục:
- Component-level dispatch: SOLID/GRADIENT/TEXTURE (xem inpainter.classify_*).
- TEXTURE → LaMa per-tile (HD mode) hoặc whole-image (non-HD).
- Refiner pass thứ 2 chỉ trên core của tile để khử artifact center.
- Fallback: nếu LaMa exception → cv2.inpaint TELEA (chậm nhưng không crash).
- Feather Gaussian blend tránh seam ở edge tile / component.

Sai khác v15:
- Per-component fallback: bug ở LaMa 1 tile không phá toàn bộ ảnh.
- Cấu hình hoàn toàn qua InpaintConfig (không hardcode).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import InpaintConfig
from .inpainter import (
    BaseInpainter,
    classify_component_texture,
    fill_gradient,
    fill_solid,
)
from .utils import get_logger


def _feather(mask_u8: np.ndarray, ksize: int) -> np.ndarray:
    """Mask → float32 (H,W,1) feather alpha, range [0,1]."""
    if ksize % 2 == 0:
        ksize += 1
    f = cv2.GaussianBlur(mask_u8.astype(np.float32) / 255.0, (ksize, ksize), 0)
    return np.clip(f, 0, 1)[..., None]


class HybridRedrawer:
    """Per-component dispatch redraw: classify + LaMa + fallback.

    Lifecycle:
        rd = HybridRedrawer(inpainter, cfg)
        out = rd.redraw(image, mask)
        # ... possibly more pages
        rd.release()  # giải phóng LaMa GPU memory
    """

    def __init__(self, inpainter: Optional[BaseInpainter], config: InpaintConfig):
        self.inpainter = inpainter
        self.config = config
        self._log = get_logger()

    # --------------------------- Public API --------------------------- #

    def redraw(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Redraw mask region in image. Trả về ảnh đã inpaint (uint8, same shape)."""
        if mask is None or mask.size == 0 or int(mask.sum()) == 0:
            return image.copy()

        # Safety dilate +2px để bao trùm anti-alias rim của text/edge.
        # (Caller đã dilate kernel lớn, đây chỉ là buffer cuối.)
        mask_inp = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )

        if self.inpainter is None:
            # Không có LaMa → toàn bộ rơi vào cv2.inpaint fallback.
            return self._cv_inpaint(image, mask_inp)

        h, w = image.shape[:2]
        model_size = self.inpainter.target_size

        # Non-HD path: 1 pass cho cả ảnh.
        if not self.config.hd or max(h, w) <= model_size:
            return self._whole_image(image, mask_inp, model_size)

        return self._hd_tiled(image, mask_inp, model_size)

    def release(self) -> None:
        """Forward release tới inpainter (idempotent)."""
        if self.inpainter is not None:
            self.inpainter.release()

    # --------------------------- Whole-image path --------------------------- #

    def _whole_image(self, image: np.ndarray, mask: np.ndarray, model_size: int) -> np.ndarray:
        try:
            out = self.inpainter.run_tile(image, mask, target_size=model_size)
        except Exception as e:  # noqa: BLE001 — fallback bất kỳ exception nào
            self._log.warning(f"⚠️  LaMa whole-image fail ({e}), fallback cv2.inpaint")
            return self._cv_inpaint(image, mask)
        mask_f = _feather(mask, 13)
        blended = image.astype(np.float32) * (1 - mask_f) + out.astype(np.float32) * mask_f
        return np.clip(blended, 0, 255).astype(np.uint8)

    # --------------------------- HD-tiled path --------------------------- #

    def _hd_tiled(self, image: np.ndarray, mask: np.ndarray, model_size: int) -> np.ndarray:
        """Group CC → dispatch theo classify → tile-level LaMa với refiner."""
        cfg = self.config
        h, w = image.shape[:2]

        num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8))
        if num <= 1:
            return image.copy()

        result = image.copy().astype(np.float32)
        composite_mask = np.zeros((h, w), dtype=np.float32)
        n_solid = n_gradient = n_tiles = n_fallback = 0

        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            if area < cfg.min_component_area:
                continue

            comp_mask = (labels == i).astype(np.uint8) * 255
            comp_mask_dil = cv2.dilate(
                comp_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )

            # --- Classify dispatch ---
            if cfg.classify:
                kind, params = classify_component_texture(
                    image, comp_mask_dil,
                    rim_width=cfg.rim_width,
                    solid_std_thresh=cfg.solid_std_thresh,
                    gradient_residual_thresh=cfg.gradient_residual_thresh,
                )
            else:
                kind, params = "TEXTURE", None

            if kind == "SOLID" and params is not None:
                filled = fill_solid(image, comp_mask_dil, params["color"])
                mask_f = _feather(comp_mask_dil, 9)
                result = result * (1 - mask_f) + filled.astype(np.float32) * mask_f
                composite_mask = np.maximum(composite_mask, mask_f[..., 0])
                n_solid += 1
                continue
            if kind == "GRADIENT" and params is not None:
                filled = fill_gradient(image, comp_mask_dil,
                                       params["coef"], params["ref_color"])
                mask_f = _feather(comp_mask_dil, 9)
                result = result * (1 - mask_f) + filled.astype(np.float32) * mask_f
                composite_mask = np.maximum(composite_mask, mask_f[..., 0])
                n_gradient += 1
                continue

            # --- TEXTURE path: HD tile + refiner ---
            tile = self._compute_tile_bounds(x, y, bw, bh, w, h)
            x1, y1, x2, y2 = tile
            tile_img = image[y1:y2, x1:x2]
            tile_mask = mask[y1:y2, x1:x2]
            if tile_mask.sum() == 0:
                continue

            tile_out = self._run_tile_with_fallback(tile_img, tile_mask, model_size)
            if tile_out is None:
                n_fallback += 1
                # Fallback cuối: cv2.inpaint trên tile, paste lại.
                tile_out = cv2.inpaint(tile_img, tile_mask, 3, cv2.INPAINT_TELEA)
            else:
                # Refiner pass thứ 2 chỉ trên LaMa output (không refine cv2 fallback).
                if cfg.refine and tile_mask.sum() > 100:
                    refined = self._refine_tile(tile_out, tile_mask, bw, bh, model_size)
                    if refined is not None:
                        tile_out = refined

            tile_mask_f = _feather(tile_mask, 13)
            region = result[y1:y2, x1:x2]
            result[y1:y2, x1:x2] = (
                region * (1 - tile_mask_f)
                + tile_out.astype(np.float32) * tile_mask_f
            )
            composite_mask[y1:y2, x1:x2] = np.maximum(
                composite_mask[y1:y2, x1:x2], tile_mask_f[..., 0],
            )
            n_tiles += 1

        refine_lbl = " +refine" if cfg.refine else ""
        cls_lbl = (f" | classify: {n_solid} solid, {n_gradient} gradient"
                   if cfg.classify else "")
        fb_lbl = f" | fallback: {n_fallback}" if n_fallback else ""
        self._log.info(
            f"   - HD inpaint: {n_tiles} tiles @ {model_size}px, "
            f"pad={cfg.tile_pad}{refine_lbl}{cls_lbl}{fb_lbl}"
        )
        return np.clip(result, 0, 255).astype(np.uint8)

    # --------------------------- Helpers --------------------------- #

    def _compute_tile_bounds(self, x: int, y: int, bw: int, bh: int,
                             img_w: int, img_h: int) -> tuple[int, int, int, int]:
        """Mở rộng bbox CC ra tile vuông với padding context."""
        cfg = self.config
        min_pad = max(32, int(min(bw, bh) * 0.25))
        pad_x = max(int(bw * cfg.tile_pad), min_pad)
        pad_y = max(int(bh * cfg.tile_pad), min_pad)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + bw + pad_x)
        y2 = min(img_h, y + bh + pad_y)

        # Square + floor cfg.tile_min_side để LaMa luôn có đủ resolution.
        tw, th = x2 - x1, y2 - y1
        side = max(tw, th, cfg.tile_min_side)
        if tw < side:
            extra = side - tw
            x1 = max(0, x1 - extra // 2)
            x2 = min(img_w, x1 + side)
            x1 = max(0, x2 - side)
        if th < side:
            extra = side - th
            y1 = max(0, y1 - extra // 2)
            y2 = min(img_h, y1 + side)
            y1 = max(0, y2 - side)
        return x1, y1, x2, y2

    def _run_tile_with_fallback(self, tile_img: np.ndarray, tile_mask: np.ndarray,
                                model_size: int) -> Optional[np.ndarray]:
        """LaMa 1 tile, swallow exception. None → caller fallback cv2."""
        try:
            return self.inpainter.run_tile(tile_img, tile_mask, target_size=model_size)
        except Exception as e:  # noqa: BLE001
            self._log.warning(f"⚠️  LaMa tile fail ({e}), fallback cv2.inpaint")
            return None

    def _refine_tile(self, tile_out: np.ndarray, tile_mask: np.ndarray,
                     bw: int, bh: int, model_size: int) -> Optional[np.ndarray]:
        """Pass-2 refine: mask co lại 15% short side, chỉ refine lõi."""
        shrink_k = max(3, int(min(bw, bh) * 0.15))
        if shrink_k % 2 == 0:
            shrink_k += 1
        shrink_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink_k, shrink_k))
        tile_mask_core = cv2.erode(tile_mask, shrink_kernel, iterations=1)
        if tile_mask_core.sum() <= 50:
            return None
        try:
            tile_out2 = self.inpainter.run_tile(tile_out, tile_mask_core,
                                                target_size=model_size)
        except Exception as e:  # noqa: BLE001
            self._log.debug(f"refine fail ({e}) — giữ pass-1")
            return None
        core_f = _feather(tile_mask_core, 15)
        return (tile_out.astype(np.float32) * (1 - core_f)
                + tile_out2.astype(np.float32) * core_f)

    # --------------------------- Fallback --------------------------- #

    def _cv_inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Pure cv2 fallback. Chậm hơn LaMa nhưng deterministic, không crash."""
        self._log.info("   - cv2.inpaint TELEA fallback (no LaMa)")
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
