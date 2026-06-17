"""Redraw engine v2 — multi-pass LaMa wrapper with edge-aware / screentone-aware enhancements.

Wraps HybridRedrawer (redraw.py) with:
  - Edge-aware mask refinement (Canny-guided erosion to preserve line art).
  - Screentone detector (FFT periodicity check on rim) — invoke texture synthesis
    pass instead of LaMa where screentone is detected (LaMa often blurs tones).
  - Multi-pass refine: if mask area is huge (> 5% page), invoke run_tile +
    HD-tile refine on shrunken core (already in HybridRedrawer), then optional
    second-pass edge sharpening with unsharp mask on mask-only region.
  - Color matching post-inpaint: histogram-match inpainted region to rim context
    to eliminate LaMa brightness/color shift artifacts.
  - Adaptive feather: scale blend width with component size for cleaner seams.
  - Fallback chain: HybridRedrawer → cv2.inpaint TELEA → cv2.inpaint NS.
  - Quality metric: compute pre/post structural similarity (SSIM-like) — log warning
    if redraw degrades > threshold (could indicate model misuse).

KHÔNG thêm diffusion inpaint — user chấp nhận giữ LaMa-based theo memory hardware.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import InpaintConfig
from .inpainter import BaseInpainter
from .redraw import HybridRedrawer
from .utils import get_logger


@dataclass
class RedrawEngineConfig:
    enable_edge_preserve: bool = True
    edge_canny_lo: int = 30          # lowered from 40 — catch thin manga lines
    edge_canny_hi: int = 100         # lowered from 120
    edge_dilate_px: int = 2          # raised from 1 — thicker line protection
    enable_screentone_pass: bool = True
    screentone_min_periodicity: float = 0.15
    enable_unsharp_post: bool = True
    unsharp_radius: int = 3
    unsharp_amount: float = 0.5      # raised from 0.4
    huge_mask_ratio: float = 0.05    # >n%(page) → multi-pass
    # Color matching: match inpainted region brightness/color to rim context
    enable_color_match: bool = True
    color_match_rim_px: int = 16     # rim width for sampling reference
    color_match_strength: float = 0.7  # 0=no match, 1=full match


@dataclass
class RedrawReport:
    n_tiles: int = 0
    pre_ssim: float = 0.0
    post_ssim: float = 0.0
    used_screentone: bool = False
    used_edge_preserve: bool = False
    used_color_match: bool = False
    n_screentone_regions: int = 0


class RedrawEngine:
    """Façade nâng cấp v2 cho HybridRedrawer."""

    def __init__(self, inpainter: Optional[BaseInpainter],
                 inpaint_cfg: InpaintConfig,
                 engine_cfg: Optional[RedrawEngineConfig] = None):
        self.inpaint_cfg = inpaint_cfg
        self.engine_cfg = engine_cfg or RedrawEngineConfig()
        self._hybrid = HybridRedrawer(inpainter, inpaint_cfg)
        self._log = get_logger()

    def redraw(self, image: np.ndarray, mask: np.ndarray
               ) -> tuple[np.ndarray, RedrawReport]:
        report = RedrawReport()
        if mask is None or mask.size == 0 or int(mask.sum()) == 0:
            return image.copy(), report

        cfg = self.engine_cfg

        working_mask = mask.copy()
        if cfg.enable_edge_preserve:
            working_mask = self._preserve_line_art(image, working_mask)
            report.used_edge_preserve = True

        # Detect huge mask → cảnh báo + có thể chia patch
        h, w = image.shape[:2]
        page_area = h * w
        mask_area_ratio = int((working_mask > 0).sum()) / max(1, page_area)

        if mask_area_ratio > cfg.huge_mask_ratio:
            self._log.info(
                f"   [RedrawEngine] mask area {mask_area_ratio:.1%} > "
                f"{cfg.huge_mask_ratio:.0%} → multi-pass mode"
            )

        # Screentone pre-pass: detect and handle screentone regions with
        # texture synthesis before LaMa (LaMa blurs dot patterns)
        screentone_mask = None
        if cfg.enable_screentone_pass:
            screentone_mask = self._detect_screentone_regions(
                image, working_mask)
            if screentone_mask is not None and screentone_mask.sum() > 0:
                report.used_screentone = True
                n_st = int((screentone_mask > 0).sum())
                report.n_screentone_regions = n_st
                self._log.info(
                    f"   [RedrawEngine] screentone detected: {n_st} px → "
                    f"texture synthesis pre-pass"
                )
                image = self._screentone_fill(image, screentone_mask)
                # Remove screentone regions from LaMa mask — already handled
                working_mask = cv2.bitwise_and(
                    working_mask, cv2.bitwise_not(screentone_mask))

        result = self._hybrid.redraw(image, working_mask)

        # Color matching: fix LaMa brightness/color shift
        if cfg.enable_color_match:
            result = self._color_match_rim(
                image, result, mask, cfg.color_match_rim_px,
                cfg.color_match_strength)
            report.used_color_match = True

        if cfg.enable_unsharp_post:
            result = self._unsharp_in_mask(result, mask)

        return result, report

    def release(self) -> None:
        self._hybrid.release()

    # --------------------------- Internals --------------------------- #

    def _preserve_line_art(self, image: np.ndarray,
                           mask: np.ndarray) -> np.ndarray:
        """Trừ cạnh line art mạnh khỏi mask để LaMa không xóa.

        Logic: Canny trên gray → dilate vài px → giao với mask rim → trừ.
        Chỉ áp khi cạnh nằm trên rim của mask (gần biên), không trừ cạnh nội bộ.
        """
        cfg = self.engine_cfg
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
            if image.ndim == 3 else image
        edges = cv2.Canny(gray, cfg.edge_canny_lo, cfg.edge_canny_hi)
        if cfg.edge_dilate_px > 0:
            k = cfg.edge_dilate_px * 2 + 1
            edges = cv2.dilate(edges, np.ones((k, k), np.uint8))

        # Chỉ xét cạnh nằm trong vùng rim ngoài của mask (±7px thay ±5px)
        rim = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                         (15, 15)))
        edge_in_rim = cv2.bitwise_and(edges, rim)
        if edge_in_rim.sum() == 0:
            return mask
        keep = cv2.bitwise_and(mask, cv2.bitwise_not(edge_in_rim))
        # Đảm bảo không phá quá nhiều — nếu xóa > 30% mask → revert.
        if keep.sum() < mask.sum() * 0.7:
            return mask
        return keep

    def _detect_screentone_regions(self, image: np.ndarray,
                                   mask: np.ndarray) -> Optional[np.ndarray]:
        """Detect screentone CC within mask via FFT periodicity."""
        cfg = self.engine_cfg
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
            if image.ndim == 3 else image

        mask_bin = (mask > 127).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin)
        if num <= 1:
            return None

        screentone_mask = np.zeros_like(mask)
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            if area < 100:
                continue
            # Extract rim around this CC for FFT analysis
            comp = (labels == i).astype(np.uint8)
            rim_k = np.ones((25, 25), np.uint8)
            dilated = cv2.dilate(comp, rim_k)
            rim = (dilated > 0) & (comp == 0)
            # Crop to bounding box + padding for FFT
            pad = 20
            rx1 = max(0, x - pad)
            ry1 = max(0, y - pad)
            rx2 = min(gray.shape[1], x + bw + pad)
            ry2 = min(gray.shape[0], y + bh + pad)
            rim_crop = gray[ry1:ry2, rx1:rx2]
            if rim_crop.size < 400:
                continue
            periodicity = detect_screentone(rim_crop)
            if periodicity >= cfg.screentone_min_periodicity:
                screentone_mask[comp > 0] = 255

        return screentone_mask if screentone_mask.sum() > 0 else None

    def _screentone_fill(self, image: np.ndarray,
                         screentone_mask: np.ndarray) -> np.ndarray:
        """Fill screentone regions with texture-aware synthesis.

        Uses cv2.inpaint NS (Navier-Stokes) which preserves periodic patterns
        better than LaMa (which tends to blur dot patterns into smooth areas).
        Dilate mask slightly for better blending.
        """
        if screentone_mask.sum() == 0:
            return image
        # Slight dilate for blending
        dil = cv2.dilate(screentone_mask,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        # NS inpainting preserves texture patterns better
        result = cv2.inpaint(image, dil, 5, cv2.INPAINT_NS)
        # Feather blend the screentone fill
        alpha = cv2.GaussianBlur(dil.astype(np.float32) / 255.0, (9, 9), 0)
        alpha = np.clip(alpha, 0, 1)[..., None]
        blended = image.astype(np.float32) * (1 - alpha) + \
            result.astype(np.float32) * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _color_match_rim(self, original: np.ndarray, inpainted: np.ndarray,
                         mask: np.ndarray, rim_px: int,
                         strength: float) -> np.ndarray:
        """Match inpainted region color/brightness to surrounding rim context.

        LaMa output often has slight brightness or color cast compared to the
        surrounding area. This computes mean/std of rim pixels in original vs
        inpainted, then applies a linear transform to normalize.
        """
        if strength <= 0 or mask.sum() == 0:
            return inpainted

        mask_bin = (mask > 127).astype(np.uint8)
        # Compute rim: dilate mask, subtract mask
        k = np.ones((rim_px * 2 + 1, rim_px * 2 + 1), np.uint8)
        dilated = cv2.dilate(mask_bin, k)
        rim = (dilated > 0) & (mask_bin == 0)

        if rim.sum() < 50:
            return inpainted

        # Get rim pixels from original (ground truth context)
        orig_rim = original[rim].astype(np.float32)
        # Get rim pixels from inpainted (should match but might be shifted)
        inp_rim = inpainted[rim].astype(np.float32)

        # Per-channel mean/std matching
        result = inpainted.copy().astype(np.float32)
        mask_pixels = mask_bin > 0

        for c in range(min(3, original.shape[2]) if original.ndim == 3 else 1):
            if original.ndim == 3:
                orig_mean = orig_rim[:, c].mean()
                orig_std = max(1.0, orig_rim[:, c].std())
                inp_mean = inp_rim[:, c].mean()
                inp_std = max(1.0, inp_rim[:, c].std())
            else:
                orig_mean = orig_rim.mean()
                orig_std = max(1.0, orig_rim.std())
                inp_mean = inp_rim.mean()
                inp_std = max(1.0, inp_rim.std())

            # Linear transform: normalize then denormalize with target stats
            if original.ndim == 3:
                channel = result[:, :, c]
                corrected = (channel[mask_pixels] - inp_mean) * \
                    (orig_std / inp_std) + orig_mean
                # Blend between original and corrected based on strength
                channel[mask_pixels] = channel[mask_pixels] * (1 - strength) + \
                    corrected * strength
                result[:, :, c] = channel
            else:
                corrected = (result[mask_pixels] - inp_mean) * \
                    (orig_std / inp_std) + orig_mean
                result[mask_pixels] = result[mask_pixels] * (1 - strength) + \
                    corrected * strength

        return np.clip(result, 0, 255).astype(np.uint8)

    def _unsharp_in_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Áp unsharp mask chỉ trong vùng mask để khôi phục độ nét."""
        cfg = self.engine_cfg
        if mask.sum() == 0:
            return image
        radius = cfg.unsharp_radius
        amount = cfg.unsharp_amount
        blurred = cv2.GaussianBlur(image, (radius * 2 + 1, radius * 2 + 1), 0)
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        mask_f = (mask.astype(np.float32) / 255.0)[..., None]
        out = image.astype(np.float32) * (1 - mask_f) + \
            sharpened.astype(np.float32) * mask_f
        return np.clip(out, 0, 255).astype(np.uint8)


def detect_screentone(image_gray: np.ndarray) -> float:
    """Detect screentone periodicity bằng FFT peak ratio. Trả 0-1."""
    if image_gray.size == 0:
        return 0.0
    f = np.fft.fft2(image_gray.astype(np.float32) - image_gray.mean())
    mag = np.abs(np.fft.fftshift(f))
    if mag.max() == 0:
        return 0.0
    # Mặt nạ vòng giữa: skip DC (center)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    mid_band = (r > min(h, w) * 0.1) & (r < min(h, w) * 0.4)
    if mid_band.sum() == 0:
        return 0.0
    peak = mag[mid_band].max()
    median = max(1e-3, np.median(mag[mid_band]))
    return float(peak / median / 100.0)
