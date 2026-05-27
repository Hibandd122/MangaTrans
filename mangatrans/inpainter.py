"""Inpainting base module and factory.

Koharu integration (2026-05-27): Refactored to support multiple inpainting
backends via `create_inpainter` factory. Base class defines `run_tile`.
"""
from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

from .config import InpaintConfig
from .utils import get_logger


class BaseInpainter(ABC):
    """Abstract base class cho mọi inpaint backend."""

    def __init__(self, config: InpaintConfig):
        self.config = config

    @abstractmethod
    def run_tile(self, image_bgr: np.ndarray, mask_u8: np.ndarray,
                 target_size: Optional[int] = None,
                 force_cpu: bool = False) -> np.ndarray:
        """Inpaint 1 tile (image + mask)."""
        pass

    @property
    @abstractmethod
    def target_size(self) -> int:
        """Kích thước xử lý lý tưởng."""
        pass

    @contextlib.contextmanager
    def force_cpu_mode(self):
        """Context manager bật force-CPU."""
        yield

    def release(self) -> None:
        """Giải phóng resources."""
        pass


def create_inpainter(config: InpaintConfig) -> BaseInpainter:
    """Khởi tạo inpainter backend duy nhất (LamaInpainter)."""
    from .inpainters import LamaInpainter
    return LamaInpainter(config)


# --------------------------- Texture classifier --------------------------- #

def _component_rim_pixels(image: np.ndarray, comp_mask: np.ndarray, rim_width: int = 12):
    """Trả pixel rim annulus quanh comp_mask (loại bỏ chính mask)."""
    if comp_mask.dtype != np.uint8:
        comp_mask = comp_mask.astype(np.uint8)
    bin_mask = (comp_mask > 0).astype(np.uint8)
    k = np.ones((rim_width * 2 + 1, rim_width * 2 + 1), np.uint8)
    dil = cv2.dilate(bin_mask, k)
    rim = (dil > 0) & (bin_mask == 0)
    if not rim.any():
        return None, None
    ys, xs = np.where(rim)
    if image.ndim == 3:
        pix = image[ys, xs]
    else:
        pix = image[ys, xs][:, None]
    return pix, (ys, xs)


def classify_component_texture(image: np.ndarray, comp_mask: np.ndarray,
                               rim_width: int = 12,
                               solid_std_thresh: float = 8.0,
                               gradient_residual_thresh: float = 6.0) -> tuple[str, Optional[dict]]:
    """Classify rim quanh CC → SOLID / GRADIENT / TEXTURE.

    SOLID: rim std L channel < solid_std_thresh.
    GRADIENT: linear fit L = a + b·x + c·y residual_std < gradient_residual_thresh.
    TEXTURE: else.
    """
    pix, coords = _component_rim_pixels(image, comp_mask, rim_width)
    if pix is None or len(pix) < 20:
        return "TEXTURE", None

    if pix.shape[1] == 3:
        L = (0.114 * pix[:, 0] + 0.587 * pix[:, 1] + 0.299 * pix[:, 2]).astype(np.float32)
    else:
        L = pix[:, 0].astype(np.float32)

    std = float(L.std())
    if std < solid_std_thresh:
        med = np.median(pix, axis=0).astype(np.uint8)
        return "SOLID", {"color": med}

    ys, xs = coords
    A = np.stack([
        np.ones_like(xs, dtype=np.float32),
        xs.astype(np.float32),
        ys.astype(np.float32),
    ], axis=1)
    try:
        coef, *_ = np.linalg.lstsq(A, L, rcond=None)
        pred = A @ coef
        residual = L - pred
        if float(residual.std()) < gradient_residual_thresh:
            med = np.median(pix, axis=0).astype(np.float32)
            return "GRADIENT", {"coef": coef, "ref_color": med}
    except np.linalg.LinAlgError:
        pass

    return "TEXTURE", None


def fill_solid(image: np.ndarray, comp_mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Fill comp_mask in-place trên copy của image bằng color BGR scalar."""
    out = image.copy()
    ys, xs = np.where(comp_mask > 0)
    if len(ys) == 0:
        return out
    if image.ndim == 3:
        out[ys, xs] = color
    else:
        out[ys, xs] = int(color[0])
    return out


def fill_gradient(image: np.ndarray, comp_mask: np.ndarray,
                  coef: np.ndarray, ref_color: np.ndarray) -> np.ndarray:
    """Fill linear gradient L=a+bx+cy. Giữ chroma từ ref_color (median rim).

    Bug fix v15: ref_L có thể rất nhỏ (vùng tối) → scale overflow. Đã check
    ref_L < 1e-3 → scale = 1; thêm clip cuối để an toàn.
    """
    out = image.copy()
    ys, xs = np.where(comp_mask > 0)
    if len(ys) == 0:
        return out
    a, b, c = coef
    L_pred = a + b * xs.astype(np.float32) + c * ys.astype(np.float32)
    L_pred = np.clip(L_pred, 0, 255)
    if image.ndim == 3:
        ref_L = 0.114 * ref_color[0] + 0.587 * ref_color[1] + 0.299 * ref_color[2]
        if ref_L < 1.0:
            # Rim tối → giữ ref_color flat, không scale (tránh chia gần 0).
            new_pix = np.broadcast_to(ref_color[None, :], (len(ys), 3)).astype(np.uint8)
        else:
            scale = L_pred / ref_L
            # Cap scale 0.2..5.0 để extreme rim không gây sáng/tối phi lý
            scale = np.clip(scale, 0.2, 5.0)
            new_pix = np.clip(ref_color[None, :] * scale[:, None], 0, 255).astype(np.uint8)
        out[ys, xs] = new_pix
    else:
        out[ys, xs] = L_pred.astype(np.uint8)
    return out
