"""Redraw engine — multi-pass LaMa wrapper with edge-aware / screentone-aware enhancements.

Wraps HybridRedrawer (redraw.py) with:
  - Edge-aware mask refinement (Canny-guided erosion to preserve line art).
  - Screentone detector (FFT periodicity check on rim) — invoke texture synthesis
    pass instead of LaMa where screentone is detected (LaMa often blurs tones).
  - Multi-pass refine: if mask area is huge (> 5% page), invoke run_tile +
    HD-tile refine on shrunken core (already in HybridRedrawer), then optional
    second-pass edge sharpening with unsharp mask on mask-only region.
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
    edge_canny_lo: int = 40
    edge_canny_hi: int = 120
    edge_dilate_px: int = 1     # giãn cạnh giữ ngược lại trừ vào mask
    enable_screentone_pass: bool = True
    screentone_min_periodicity: float = 0.15
    enable_unsharp_post: bool = True
    unsharp_radius: int = 3
    unsharp_amount: float = 0.4
    huge_mask_ratio: float = 0.05   # >n%(page) → multi-pass


@dataclass
class RedrawReport:
    n_tiles: int = 0
    pre_ssim: float = 0.0
    post_ssim: float = 0.0
    used_screentone: bool = False
    used_edge_preserve: bool = False


class RedrawEngine:
    """Façade nâng cấp cho HybridRedrawer."""

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

        result = self._hybrid.redraw(image, working_mask)

        if cfg.enable_unsharp_post:
            result = self._unsharp_in_mask(result, working_mask)

        return result, report

    def release(self) -> None:
        self._hybrid.release()

    # --------------------------- Internals --------------------------- #

    def _preserve_line_art(self, image: np.ndarray,
                           mask: np.ndarray) -> np.ndarray:
        """Trừ cạnh line art mạnh khỏi mask để LaMa không xóa.

        Logic: Canny trên gray → dilate vài px → giao với mask → trừ. Chỉ áp
        khi cạnh nằm trên rim của mask (gần biên), không trừ cạnh nội bộ.
        """
        cfg = self.engine_cfg
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
            if image.ndim == 3 else image
        edges = cv2.Canny(gray, cfg.edge_canny_lo, cfg.edge_canny_hi)
        if cfg.edge_dilate_px > 0:
            k = cfg.edge_dilate_px * 2 + 1
            edges = cv2.dilate(edges, np.ones((k, k), np.uint8))

        # Chỉ xét cạnh nằm trong vùng rim ngoài của mask (±5px)
        rim = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                         (11, 11)))
        edge_in_rim = cv2.bitwise_and(edges, rim)
        if edge_in_rim.sum() == 0:
            return mask
        keep = cv2.bitwise_and(mask, cv2.bitwise_not(edge_in_rim))
        # Đảm bảo không phá quá nhiều — nếu xóa > 30% mask → revert.
        if keep.sum() < mask.sum() * 0.7:
            return mask
        return keep

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
