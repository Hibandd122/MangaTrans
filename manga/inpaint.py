"""Inpaint stack: pre-clean → LaMa TorchScript → HD-tiled dispatch → post-refine.

Gộp 4 file cũ thành 1:
- Pre-inpaint cleaning (`dilate_mask`, `smart_fill_uniform_regions`, `clean_bubbles_by_fill`).
- TorchScript LaMa inpainter (`LamaInpainter`) — chỉ hỗ trợ `anime-manga-big-lama.pt`.
- Texture classifier (SOLID/GRADIENT/TEXTURE) + deterministic fill.
- `HybridRedrawer` orchestrator (component-level dispatch + HD-tiled LaMa).
- `RedrawEngine` facade (edge-preserve + unsharp post).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import CleanerConfig, GeometryConfig, InpaintConfig
from .geometry import InteriorCache, find_bubble_interior
from .utils import get_logger


# ============================================================
# Section 1 — Pre-inpaint cleaning
# ============================================================

def dilate_mask(mask: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """Mở rộng mask để bao trùm rìa anti-alias của chữ."""
    if kernel_size <= 0:
        return mask.copy()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def smart_fill_uniform_regions(image: np.ndarray, mask: np.ndarray,
                               std_thresh: float = 18.0,
                               rim_size: int = 6) -> tuple[np.ndarray, np.ndarray, int]:
    """Fill CC trong mask bằng median rim color nếu rim đồng nhất."""
    h, w = image.shape[:2]
    mask_bin = (mask > 127).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin)
    if num <= 1:
        return image.copy(), mask.copy(), 0

    result = image.copy()
    remaining = np.zeros_like(mask)
    n_filled = 0

    kernel = np.ones((rim_size * 2 + 1, rim_size * 2 + 1), np.uint8)
    dilated_full = cv2.dilate(mask_bin, kernel, iterations=1)
    rim_full = dilated_full - mask_bin

    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < 4:
            continue
        comp_mask = (labels == i).astype(np.uint8)
        comp_dilated = cv2.dilate(comp_mask, kernel, iterations=1)
        comp_rim = (comp_dilated - comp_mask) & rim_full

        rim_pixels = image[comp_rim > 0]
        if len(rim_pixels) < 20:
            remaining[comp_mask > 0] = 255
            continue

        std = float(rim_pixels.std(axis=0).mean())
        if std > std_thresh:
            remaining[comp_mask > 0] = 255
            continue

        fill_color = np.median(rim_pixels, axis=0).astype(np.uint8)
        result[comp_mask > 0] = fill_color
        n_filled += 1

    return result, remaining, n_filled


def clean_bubbles_by_fill(image: np.ndarray, text_mask: np.ndarray, blocks: list[dict],
                          geom_cfg: GeometryConfig,
                          interior_cache: InteriorCache,
                          freetext_std_thresh: float = 25.0,
                          bubble_fill_retract: int = 4,
                          feather_ksize: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill bubble interior bằng median color sampled từ chính bubble."""
    result = image.copy()
    h, w = image.shape[:2]
    filled_union = np.zeros((h, w), dtype=np.uint8)

    text_dil_k = np.ones((5, 5), np.uint8)
    text_neighborhood = cv2.dilate(text_mask, text_dil_k)

    for blk in blocks:
        cls = blk.get("cls", 0)
        if cls != 0:
            continue

        other_bxs = [b["bbox"] for b in blocks if b is not blk]
        interior = find_bubble_interior(image, blk["bbox"], geom_cfg,
                                        other_bboxes=other_bxs, cache=interior_cache)
        if interior.sum() == 0:
            continue

        if bubble_fill_retract > 0:
            k = 2 * bubble_fill_retract + 1
            fill_mask = cv2.erode(interior, np.ones((k, k), np.uint8))
        else:
            fill_mask = interior

        sample_region = (fill_mask > 0) & (text_neighborhood == 0)
        if sample_region.sum() < 50:
            sample_region = (interior > 0) & (text_mask == 0)
            if sample_region.sum() < 30:
                continue

        avg_color = np.median(image[sample_region], axis=0).astype(np.uint8)

        if feather_ksize > 1:
            k = feather_ksize if feather_ksize % 2 == 1 else feather_ksize + 1
            alpha = cv2.GaussianBlur(
                fill_mask.astype(np.float32) / 255.0, (k, k), 0,
            )
            alpha = np.clip(alpha, 0, 1)[..., None]
            fill_layer = np.full_like(result, avg_color)
            result = (result.astype(np.float32) * (1 - alpha)
                      + fill_layer.astype(np.float32) * alpha)
            result = np.clip(result, 0, 255).astype(np.uint8)
        else:
            result[fill_mask > 0] = avg_color

        filled_union = cv2.bitwise_or(filled_union, fill_mask)

    remaining = cv2.bitwise_and(text_mask, cv2.bitwise_not(filled_union))
    return result, remaining, filled_union


# ============================================================
# Section 2 — LaMa TorchScript inpainter
# ============================================================

@dataclass
class _SessionEntry:
    obj: object
    device: str
    target_size: int


class LamaInpainter:
    """Wrap TorchScript LaMa model (.pt). Single-model cache."""

    def __init__(self, model_path: str, override_size: Optional[int] = None):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Không tìm thấy inpaint model: {model_path}")
        if not model_path.lower().endswith(".pt"):
            raise ValueError(
                "LamaInpainter chỉ hỗ trợ TorchScript .pt "
                "(ONNX đã bị bỏ — dùng anime-manga-big-lama.pt)."
            )
        self.model_path = model_path
        self._override_size = override_size
        self._entry: Optional[_SessionEntry] = None

    def _ensure_session(self) -> _SessionEntry:
        if self._entry is not None:
            return self._entry
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.jit.load(self.model_path, map_location=device).eval()
        target = self._override_size or 512
        self._entry = _SessionEntry(model, device, target)
        return self._entry

    @property
    def target_size(self) -> int:
        return self._ensure_session().target_size

    def run_tile(self, image_bgr: np.ndarray, mask_u8: np.ndarray,
                 target_size: Optional[int] = None) -> np.ndarray:
        """Inpaint 1 tile. Resize/pad theo target_size, output về size gốc."""
        import torch
        entry = self._ensure_session()
        size = target_size or entry.target_size
        h, w = image_bgr.shape[:2]

        if h <= size and w <= size:
            pad_h = size - h
            pad_w = size - w
            top = pad_h // 2
            bot = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            img_r = cv2.copyMakeBorder(image_bgr, top, bot, left, right, cv2.BORDER_REFLECT_101)
            msk_r = cv2.copyMakeBorder(mask_u8, top, bot, left, right, cv2.BORDER_CONSTANT, value=0)
            crop_after = (top, left, h, w)
        else:
            img_r = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
            msk_r = cv2.resize(mask_u8, (size, size), interpolation=cv2.INTER_NEAREST)
            crop_after = None

        img_rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
        img_t = (torch.from_numpy(img_rgb).float().div(255.0)
                 .permute(2, 0, 1).unsqueeze(0).to(entry.device))
        msk_t = (torch.from_numpy((msk_r > 127).astype(np.float32))
                 .unsqueeze(0).unsqueeze(0).to(entry.device))
        with torch.inference_mode():
            out_t = entry.obj(img_t, msk_t)
        out = out_t[0].permute(1, 2, 0).detach().cpu().numpy()
        if out.max() > 1.5:
            out = np.clip(out, 0, 255).astype(np.uint8)
        else:
            out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        if crop_after is not None:
            top, left, oh, ow = crop_after
            return out[top:top + oh, left:left + ow]
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)

    def release(self) -> None:
        """Giải phóng GPU memory. Idempotent."""
        if self._entry is None:
            return
        del self._entry.obj
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        self._entry = None


# ============================================================
# Section 3 — Texture classifier (SOLID / GRADIENT / TEXTURE)
# ============================================================

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
    """Classify rim quanh CC → SOLID / GRADIENT / TEXTURE."""
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
    """Fill linear gradient L=a+bx+cy. Giữ chroma từ ref_color (median rim)."""
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
            new_pix = np.broadcast_to(ref_color[None, :], (len(ys), 3)).astype(np.uint8)
        else:
            scale = L_pred / ref_L
            scale = np.clip(scale, 0.2, 5.0)
            new_pix = np.clip(ref_color[None, :] * scale[:, None], 0, 255).astype(np.uint8)
        out[ys, xs] = new_pix
    else:
        out[ys, xs] = L_pred.astype(np.uint8)
    return out


# ============================================================
# Section 4 — Hybrid redraw orchestrator
# ============================================================

def _feather(mask_u8: np.ndarray, ksize: int) -> np.ndarray:
    """Mask → float32 (H,W,1) feather alpha, range [0,1]."""
    if ksize % 2 == 0:
        ksize += 1
    f = cv2.GaussianBlur(mask_u8.astype(np.float32) / 255.0, (ksize, ksize), 0)
    return np.clip(f, 0, 1)[..., None]


class HybridRedrawer:
    """Per-component dispatch: classify + LaMa + fallback."""

    def __init__(self, inpainter: Optional[LamaInpainter], config: InpaintConfig):
        self.inpainter = inpainter
        self.config = config
        self._log = get_logger()

    def redraw(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Redraw mask region in image. Trả ảnh đã inpaint."""
        if mask is None or mask.size == 0 or int(mask.sum()) == 0:
            return image.copy()

        mask_inp = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )

        if self.inpainter is None:
            return self._cv_inpaint(image, mask_inp)

        h, w = image.shape[:2]
        model_size = self.inpainter.target_size

        if not self.config.hd or max(h, w) <= model_size:
            return self._whole_image(image, mask_inp, model_size)

        return self._hd_tiled(image, mask_inp, model_size)

    def release(self) -> None:
        if self.inpainter is not None:
            self.inpainter.release()

    def _whole_image(self, image: np.ndarray, mask: np.ndarray, model_size: int) -> np.ndarray:
        try:
            out = self.inpainter.run_tile(image, mask, target_size=model_size)
        except Exception as e:  # noqa: BLE001
            self._log.warning(f"⚠️  LaMa whole-image fail ({e}), fallback cv2.inpaint")
            return self._cv_inpaint(image, mask)
        mask_f = _feather(mask, 13)
        blended = image.astype(np.float32) * (1 - mask_f) + out.astype(np.float32) * mask_f
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _hd_tiled(self, image: np.ndarray, mask: np.ndarray, model_size: int) -> np.ndarray:
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

            tile = self._compute_tile_bounds(x, y, bw, bh, w, h)
            x1, y1, x2, y2 = tile
            tile_img = image[y1:y2, x1:x2]
            tile_mask = mask[y1:y2, x1:x2]
            if tile_mask.sum() == 0:
                continue

            tile_out = self._run_tile_with_fallback(tile_img, tile_mask, model_size)
            if tile_out is None:
                n_fallback += 1
                tile_out = cv2.inpaint(tile_img, tile_mask, 3, cv2.INPAINT_TELEA)
            else:
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

    def _compute_tile_bounds(self, x: int, y: int, bw: int, bh: int,
                             img_w: int, img_h: int) -> tuple[int, int, int, int]:
        cfg = self.config
        min_pad = max(32, int(min(bw, bh) * 0.25))
        pad_x = max(int(bw * cfg.tile_pad), min_pad)
        pad_y = max(int(bh * cfg.tile_pad), min_pad)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + bw + pad_x)
        y2 = min(img_h, y + bh + pad_y)

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
        try:
            return self.inpainter.run_tile(tile_img, tile_mask, target_size=model_size)
        except Exception as e:  # noqa: BLE001
            self._log.warning(f"⚠️  LaMa tile fail ({e}), fallback cv2.inpaint")
            return None

    def _refine_tile(self, tile_out: np.ndarray, tile_mask: np.ndarray,
                     bw: int, bh: int, model_size: int) -> Optional[np.ndarray]:
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

    def _cv_inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self._log.info("   - cv2.inpaint TELEA fallback (no LaMa)")
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


# ============================================================
# Section 5 — RedrawEngine facade (edge-preserve + unsharp post)
# ============================================================

@dataclass
class RedrawEngineConfig:
    enable_edge_preserve: bool = True
    edge_canny_lo: int = 40
    edge_canny_hi: int = 120
    edge_dilate_px: int = 1
    enable_screentone_pass: bool = True
    screentone_min_periodicity: float = 0.15
    enable_unsharp_post: bool = True
    unsharp_radius: int = 3
    unsharp_amount: float = 0.4
    huge_mask_ratio: float = 0.05


@dataclass
class RedrawReport:
    n_tiles: int = 0
    pre_ssim: float = 0.0
    post_ssim: float = 0.0
    used_screentone: bool = False
    used_edge_preserve: bool = False


class RedrawEngine:
    """Façade nâng cấp cho HybridRedrawer (edge-preserve + unsharp post)."""

    def __init__(self, inpainter: Optional[LamaInpainter],
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

    def _preserve_line_art(self, image: np.ndarray,
                           mask: np.ndarray) -> np.ndarray:
        cfg = self.engine_cfg
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
            if image.ndim == 3 else image
        edges = cv2.Canny(gray, cfg.edge_canny_lo, cfg.edge_canny_hi)
        if cfg.edge_dilate_px > 0:
            k = cfg.edge_dilate_px * 2 + 1
            edges = cv2.dilate(edges, np.ones((k, k), np.uint8))

        rim = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                         (11, 11)))
        edge_in_rim = cv2.bitwise_and(edges, rim)
        if edge_in_rim.sum() == 0:
            return mask
        keep = cv2.bitwise_and(mask, cv2.bitwise_not(edge_in_rim))
        if keep.sum() < mask.sum() * 0.7:
            return mask
        return keep

    def _unsharp_in_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
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
