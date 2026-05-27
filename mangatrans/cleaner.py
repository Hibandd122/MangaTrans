"""Pre-inpaint cleaning: bubble fill, smart-fill uniform regions.

Bubble fill: thay vì LaMa lấy toàn bộ bubble interior + text, ta fill bubble
trắng bằng median color sampled từ chính bubble → bubble sạch trắng, LaMa chỉ
xử lý phần SFX/text trên nền phức tạp. Tránh LaMa hallucinate trên nền trắng.

Smart-fill: với mỗi CC text còn lại sau bubble fill, check rim std. Nếu rim đồng
nhất (giấy trắng, panel trắng) → fill bằng median rim. Phức tạp → LaMa.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import CleanerConfig, GeometryConfig
from .geometry import find_bubble_interior, InteriorCache


def dilate_mask(mask: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """Mở rộng mask để bao trùm rìa anti-alias của chữ."""
    if kernel_size <= 0:
        return mask.copy()
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def smart_fill_uniform_regions(image: np.ndarray, mask: np.ndarray,
                               std_thresh: float = 18.0,
                               rim_size: int = 6) -> tuple[np.ndarray, np.ndarray, int]:
    """Fill CC trong mask bằng median rim color nếu rim đồng nhất.

    Tách CC, mỗi CC tính rim annulus, đo std L channel. Nếu < std_thresh:
    nền đồng nhất → fill (deterministic, không hallucinate). Còn lại để LaMa.

    Trả (image_after, mask_remaining, n_filled).
    """
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
    """Fill bubble interior bằng median color sampled từ chính bubble.

    Bubble (cls=0): fill nếu tìm được interior. Retract `bubble_fill_retract` px
    để KHÔNG paint chồng lên bubble outline → giữ stroke gốc, blend mượt.
    Sampled median được lấy từ pixel xa text (tránh anti-alias contamination).

    Free text (cls=1, SFX/cloud bubble): SKIP fill — cloud outline mềm thường
    nằm sát text, fill solid sẽ xóa luôn outline → "đểu". Để smart-fill (per-char
    rim) + LaMa redraw chỉ text strokes, outline gốc còn nguyên.

    Trả (image_cleaned, mask_remaining_for_lama, mask_bubble_union).
    """
    result = image.copy()
    h, w = image.shape[:2]
    filled_union = np.zeros((h, w), dtype=np.uint8)

    # Erode text mask 2px → pixels NEAR text (anti-alias rim) bị loại khỏi
    # median sample. Otherwise sample includes grey rim → fill có tint xám.
    text_dil_k = np.ones((5, 5), np.uint8)
    text_neighborhood = cv2.dilate(text_mask, text_dil_k)

    for blk in blocks:
        cls = blk.get("cls", 0)
        # Cls=1 (free text / cloud bubble / SFX) → SKIP. Smart-fill + LaMa
        # xử lý text strokes; outline gốc giữ nguyên.
        if cls != 0:
            continue

        other_bxs = [b["bbox"] for b in blocks if b is not blk]
        interior = find_bubble_interior(image, blk["bbox"], geom_cfg,
                                        other_bboxes=other_bxs, cache=interior_cache)
        if interior.sum() == 0:
            continue

        # Retract interior để fill không đụng outline.
        if bubble_fill_retract > 0:
            k = 2 * bubble_fill_retract + 1
            fill_mask = cv2.erode(interior, np.ones((k, k), np.uint8))
        else:
            fill_mask = interior

        # Sample median từ vùng SẠCH (xa text + xa outline).
        sample_region = (fill_mask > 0) & (text_neighborhood == 0)
        if sample_region.sum() < 50:
            # Vùng sample quá nhỏ → dùng interior - text trực tiếp.
            sample_region = (interior > 0) & (text_mask == 0)
            if sample_region.sum() < 30:
                continue

        avg_color = np.median(image[sample_region], axis=0).astype(np.uint8)

        # Feather fill: alpha=1 ở center, fade về 0 ở rim → blend mượt với
        # outline gốc, không tạo seam cứng.
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
