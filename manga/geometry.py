"""Bubble geometry: tìm interior mask, polygon contour, inscribed rect, row extents."""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from .config import GeometryConfig
from .utils import clamp_bbox


def exclude_other_bbox_regions(mask: np.ndarray, bbox, other_bboxes) -> np.ndarray:
    """Voronoi theo bbox extent: drop pixel gần bbox khác hơn bbox này."""
    if not other_bboxes or not mask.any():
        return mask
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask

    def _dist_to_bbox(bxs, bys, bx):
        bx1, by1, bx2, by2 = bx
        dx = np.maximum.reduce([np.zeros_like(bxs), bx1 - bxs, bxs - bx2])
        dy = np.maximum.reduce([np.zeros_like(bys), by1 - bys, bys - by2])
        return np.maximum(dx, dy)

    d_self = _dist_to_bbox(xs, ys, bbox)
    drop = np.zeros(len(xs), dtype=bool)
    for ob in other_bboxes:
        if tuple(ob) == tuple(bbox):
            continue
        d_other = _dist_to_bbox(xs, ys, ob)
        drop |= d_other < d_self
    if drop.any():
        mask = mask.copy()
        mask[ys[drop], xs[drop]] = 0
    return mask


class InteriorCache:
    """Cache global-CC fallback cho find_bubble_interior — per-image scope."""

    def __init__(self):
        self._entry: Optional[tuple] = None
        self._key: Optional[tuple] = None

    def get_or_compute(self, gray_image: np.ndarray, white_thresh: int, close_ksize: int):
        key = (gray_image.shape, white_thresh, close_ksize)
        if self._key == key and self._entry is not None:
            return self._entry
        _, white_bin = cv2.threshold(gray_image, white_thresh, 255, cv2.THRESH_BINARY)
        white_bin = cv2.morphologyEx(
            white_bin, cv2.MORPH_CLOSE,
            np.ones((close_ksize, close_ksize), np.uint8),
        )
        num, labels, stats, _ = cv2.connectedComponentsWithStats(white_bin)
        self._entry = (num, labels, stats)
        self._key = key
        return self._entry


def find_bubble_interior(image: np.ndarray, bbox, cfg: GeometryConfig,
                         other_bboxes: Optional[Sequence] = None,
                         cache: Optional[InteriorCache] = None) -> np.ndarray:
    """Tìm vùng bubble interior trắng bao quanh text bbox."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, w, h)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return np.zeros((h, w), dtype=np.uint8)

    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dark_dilate = cfg.dark_dilate
    dark_kernel = np.ones((dark_dilate, dark_dilate), np.uint8) if dark_dilate > 0 else None
    open_kernel = np.ones((3, 3), np.uint8)
    img_area = h * w

    for pad in (cfg.roi_pad_factor, cfg.roi_pad_factor * 2, cfg.roi_pad_factor * 3):
        pad_x = max(15, int(bw * pad))
        pad_y = max(15, int(bh * pad))
        rx1 = max(0, x1 - pad_x)
        ry1 = max(0, y1 - pad_y)
        rx2 = min(w, x2 + pad_x)
        ry2 = min(h, y2 + pad_y)
        roi_gray = gray_full[ry1:ry2, rx1:rx2]
        rh, rw = roi_gray.shape
        if rh == 0 or rw == 0:
            continue

        _, dark = cv2.threshold(roi_gray, cfg.dark_thresh, 255, cv2.THRESH_BINARY_INV)
        if dark_kernel is not None:
            dark = cv2.dilate(dark, dark_kernel)
        white_iso = cv2.bitwise_not(dark)
        white_iso = cv2.morphologyEx(white_iso, cv2.MORPH_OPEN, open_kernel)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(white_iso)
        if num <= 1:
            continue

        tx1_r, ty1_r = x1 - rx1, y1 - ry1
        tx2_r, ty2_r = x2 - rx1, y2 - ry1
        tcx_r = (tx1_r + tx2_r) // 2
        tcy_r = (ty1_r + ty2_r) // 2

        sample_pts = [(tcx_r, tcy_r)]
        for fy in (0.1, 0.3, 0.5, 0.7, 0.9):
            for fx in (0.1, 0.3, 0.5, 0.7, 0.9):
                px = int(tx1_r + (tx2_r - tx1_r) * fx)
                py = int(ty1_r + (ty2_r - ty1_r) * fy)
                if 0 <= px < rw and 0 <= py < rh:
                    sample_pts.append((px, py))
        for off in (5, 10, 15):
            sample_pts.extend([
                (tcx_r, max(0, ty1_r - off)),
                (tcx_r, min(rh - 1, ty2_r + off)),
                (max(0, tx1_r - off), tcy_r),
                (min(rw - 1, tx2_r + off), tcy_r),
            ])

        votes: dict[int, int] = {}
        for px, py in sample_pts:
            lbl = int(labels[py, px])
            if lbl == 0:
                continue
            votes[lbl] = votes.get(lbl, 0) + 1

        found_lbl = 0
        max_bubble_area = max(bw * bh * 12, 8000)
        for lbl, _v in sorted(votes.items(), key=lambda kv: -kv[1]):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            cx_ = int(stats[lbl, cv2.CC_STAT_LEFT])
            cy_ = int(stats[lbl, cv2.CC_STAT_TOP])
            cw_ = int(stats[lbl, cv2.CC_STAT_WIDTH])
            ch_ = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            slack = max(2, dark_dilate)
            if not (cx_ <= tx1_r + slack and cy_ <= ty1_r + slack
                    and cx_ + cw_ >= tx2_r - slack and cy_ + ch_ >= ty2_r - slack):
                continue
            if area > cfg.max_area_ratio * img_area:
                continue
            if area < bw * bh * 0.3:
                continue
            touches_border = (cx_ <= 0 or cy_ <= 0
                              or cx_ + cw_ >= rw or cy_ + ch_ >= rh)
            if area > max_bubble_area and not touches_border:
                continue
            found_lbl = lbl
            break

        if found_lbl != 0:
            bubble_roi = (labels == found_lbl).astype(np.uint8) * 255
            if dark_kernel is not None:
                restore_k = max(1, dark_dilate - 1)
                bubble_roi = cv2.dilate(
                    bubble_roi, np.ones((restore_k, restore_k), np.uint8),
                )
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[ry1:ry2, rx1:rx2] = bubble_roi
            full_mask = exclude_other_bbox_regions(full_mask, bbox, other_bboxes)
            if cfg.retract > 0:
                k = np.ones((cfg.retract * 2 + 1, cfg.retract * 2 + 1), np.uint8)
                full_mask = cv2.erode(full_mask, k)
            return full_mask

    if cache is None:
        cache = InteriorCache()
    num, labels, stats = cache.get_or_compute(gray_full, cfg.white_thresh, cfg.close_ksize)

    if num <= 1:
        return np.zeros((h, w), dtype=np.uint8)

    cx = max(0, min(w - 1, (x1 + x2) // 2))
    cy = max(0, min(h - 1, (y1 + y2) // 2))
    pad_out = 4
    sample_pts = [
        (cx, cy),
        (cx, max(0, y1 - pad_out)),
        (cx, min(h - 1, y2 + pad_out)),
        (max(0, x1 - pad_out), cy),
        (min(w - 1, x2 + pad_out), cy),
    ]

    found_lbl = 0
    for px, py in sample_pts:
        lbl = int(labels[py, px])
        if lbl == 0:
            continue
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area > cfg.max_area_ratio * img_area:
            continue
        cx_ = int(stats[lbl, cv2.CC_STAT_LEFT])
        cy_ = int(stats[lbl, cv2.CC_STAT_TOP])
        cw_ = int(stats[lbl, cv2.CC_STAT_WIDTH])
        ch_ = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        if cx_ <= x1 and cy_ <= y1 and cx_ + cw_ >= x2 and cy_ + ch_ >= y2:
            found_lbl = lbl
            break

    if found_lbl == 0:
        return np.zeros((h, w), dtype=np.uint8)

    bubble = (labels == found_lbl).astype(np.uint8) * 255
    bubble = exclude_other_bbox_regions(bubble, bbox, other_bboxes)
    if cfg.retract > 0:
        k = np.ones((cfg.retract * 2 + 1, cfg.retract * 2 + 1), np.uint8)
        bubble = cv2.erode(bubble, k)
    return bubble


def find_bubble_polygon(image: np.ndarray, bbox, text_seg_mask: np.ndarray,
                        cfg: GeometryConfig,
                        other_bboxes: Optional[Sequence] = None) -> Optional[dict]:
    """Polygon-based bubble interior dùng text seg mask làm seed."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, w, h)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    pad_x = max(15, int(bw * cfg.roi_pad_factor))
    pad_y = max(15, int(bh * cfg.roi_pad_factor))
    rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    roi_gray = gray[ry1:ry2, rx1:rx2]
    rh, rw = roi_gray.shape
    if rh == 0 or rw == 0:
        return None

    _, dark = cv2.threshold(roi_gray, cfg.dark_thresh, 255, cv2.THRESH_BINARY_INV)
    if cfg.dark_dilate > 0:
        dark = cv2.dilate(dark, np.ones((cfg.dark_dilate, cfg.dark_dilate), np.uint8))
    white_iso = cv2.bitwise_not(dark)
    white_iso = cv2.morphologyEx(white_iso, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(white_iso)
    if num <= 1:
        return None

    text_roi = text_seg_mask[ry1:ry2, rx1:rx2]
    text_in_bbox = np.zeros((rh, rw), dtype=np.uint8)
    bx1_r = max(0, x1 - rx1)
    by1_r = max(0, y1 - ry1)
    bx2_r = min(rw, x2 - rx1)
    by2_r = min(rh, y2 - ry1)
    text_in_bbox[by1_r:by2_r, bx1_r:bx2_r] = text_roi[by1_r:by2_r, bx1_r:bx2_r]
    text_dilated = cv2.dilate(text_in_bbox, np.ones((25, 25), np.uint8))

    img_area = h * w
    best_lbl, best_overlap = 0, 0
    for lbl in range(1, num):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < bw * bh * 0.3:
            continue
        if area > 0.15 * img_area:
            continue
        cc_mask = (labels == lbl)
        overlap = int((cc_mask & (text_dilated > 0)).sum())
        if overlap > best_overlap:
            best_overlap = overlap
            best_lbl = lbl

    if best_lbl == 0 or best_overlap == 0:
        return None

    bubble_roi = (labels == best_lbl).astype(np.uint8) * 255
    if cfg.dark_dilate > 0:
        restore_k = max(1, cfg.dark_dilate - 1)
        bubble_roi = cv2.dilate(bubble_roi, np.ones((restore_k, restore_k), np.uint8))

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[ry1:ry2, rx1:rx2] = bubble_roi
    full_mask = exclude_other_bbox_regions(full_mask, bbox, other_bboxes)
    if cfg.polygon_retract > 0:
        full_mask = cv2.erode(full_mask,
                              np.ones((cfg.polygon_retract * 2 + 1,
                                       cfg.polygon_retract * 2 + 1), np.uint8))

    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    tcx, tcy = float((x1 + x2) // 2), float((y1 + y2) // 2)
    best_contour, best_area = None, 0
    for c in contours:
        a = cv2.contourArea(c)
        if a < 200:
            continue
        d = cv2.pointPolygonTest(c, (tcx, tcy), True)
        if d >= -10 and a > best_area:
            best_contour, best_area = c, a
    if best_contour is None:
        return None

    perim = cv2.arcLength(best_contour, True)
    epsilon = max(2.0, perim * cfg.polygon_simplify_eps)
    simplified = cv2.approxPolyDP(best_contour, epsilon, True)

    if len(simplified) < 3:
        simplified = best_contour

    final_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(final_mask, [simplified], 255)

    M = cv2.moments(simplified)
    if M['m00'] > 0:
        centroid = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    else:
        centroid = (int(tcx), int(tcy))

    return {
        'mask': final_mask,
        'contour': simplified.reshape(-1, 2),
        'area': int(best_area),
        'centroid': centroid,
    }


def largest_inscribed_rect(mask: np.ndarray, center_hint=None) -> Optional[tuple[int, int, int, int]]:
    """Hình chữ nhật axis-aligned lớn nhất trong binary mask."""
    if not mask.any():
        return None
    m = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    h, w = m.shape

    if center_hint is not None:
        cx, cy = center_hint
        cx = max(0, min(w - 1, int(cx)))
        cy = max(0, min(h - 1, int(cy)))
        if m[cy, cx] == 0:
            _, _, _, max_loc = cv2.minMaxLoc(cv2.GaussianBlur(dist, (5, 5), 0))
            cx, cy = max_loc
    else:
        _, _, _, max_loc = cv2.minMaxLoc(cv2.GaussianBlur(dist, (5, 5), 0))
        cx, cy = max_loc

    left = right = top = bot = 0
    while cx - left - 1 >= 0 and m[cy, cx - left - 1] > 0:
        left += 1
    while cx + right + 1 < w and m[cy, cx + right + 1] > 0:
        right += 1
    while cy - top - 1 >= 0 and m[cy - top - 1, cx] > 0:
        top += 1
    while cy + bot + 1 < h and m[cy + bot + 1, cx] > 0:
        bot += 1

    hw = min(left, right)
    hh = min(top, bot)
    if hw <= 1 or hh <= 1:
        return None

    def rect_inside(rhw, rhh):
        if rhw <= 0 or rhh <= 0:
            return False
        x1 = max(0, cx - rhw)
        x2 = min(w, cx + rhw + 1)
        y1 = max(0, cy - rhh)
        y2 = min(h, cy + rhh + 1)
        sub = m[y1:y2, x1:x2]
        return sub.size > 0 and int(sub.min()) > 0

    if rect_inside(hw, hh):
        return (cx - hw, cy - hh, cx + hw, cy + hh)

    lo, hi, best = 0.0, 1.0, 0.0
    for _ in range(18):
        mid = (lo + hi) / 2
        if rect_inside(int(hw * mid), int(hh * mid)):
            best = mid
            lo = mid
        else:
            hi = mid
    if best <= 0:
        return None
    rhw, rhh = int(hw * best), int(hh * best)
    return (cx - rhw, cy - rhh, cx + rhw, cy + rhh)


def row_extents(mask: np.ndarray) -> list[Optional[tuple[int, int]]]:
    """Cho mỗi y trong mask, tìm RUN ngang dài nhất → (xmin, xmax)."""
    h = mask.shape[0]
    binmask = mask > 0
    out: list[Optional[tuple[int, int]]] = [None] * h

    row_has_pixel = binmask.any(axis=1)
    nonempty_rows = np.where(row_has_pixel)[0]
    if nonempty_rows.size == 0:
        return out

    for y in nonempty_rows:
        row = binmask[y]
        xs = np.where(row)[0]
        if xs.size == 1:
            v = int(xs[0])
            out[y] = (v, v)
            continue
        diffs = np.diff(xs)
        gap_idx = np.where(diffs > 1)[0]
        if gap_idx.size == 0:
            out[y] = (int(xs[0]), int(xs[-1]))
            continue
        starts = np.concatenate(([0], gap_idx + 1))
        ends = np.concatenate((gap_idx, [xs.size - 1]))
        run_lens = ends - starts
        best = int(np.argmax(run_lens))
        out[y] = (int(xs[starts[best]]), int(xs[ends[best]]))
    return out


def polygon_row_extents(contour: np.ndarray, h: int,
                        anchor_x: Optional[float] = None) -> list[Optional[tuple[int, int]]]:
    """Scanline intersect polygon contour. Trả (xmin, xmax) cho mỗi y."""
    if contour is None or len(contour) < 3:
        return [None] * h
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if anchor_x is None:
        anchor_x = float(pts[:, 0].mean())

    out: list[Optional[tuple[int, int]]] = [None] * h
    edges = []
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if y1 == y2:
            continue
        if y1 < y2:
            edges.append((y1, y2, x1, x2))
        else:
            edges.append((y2, y1, x2, x1))

    if not edges:
        return out

    arr = np.asarray(edges, dtype=np.float64)
    ymins = arr[:, 0]
    ymaxs = arr[:, 1]
    xa = arr[:, 2]
    xb = arr[:, 3]

    for y in range(h):
        yf = float(y) + 0.5
        active = (yf >= ymins) & (yf < ymaxs)
        if not active.any():
            continue
        t = (yf - ymins[active]) / (ymaxs[active] - ymins[active])
        xs = xa[active] + t * (xb[active] - xa[active])
        if xs.size < 2:
            continue
        xs.sort()
        best_seg = None
        best_len = 0.0
        contains = False
        for i in range(0, len(xs) - 1, 2):
            x_in, x_out = float(xs[i]), float(xs[i + 1])
            seg_len = x_out - x_in
            if x_in <= anchor_x <= x_out:
                best_seg = (x_in, x_out)
                contains = True
                break
            if not contains and seg_len > best_len:
                best_len = seg_len
                best_seg = (x_in, x_out)
        if best_seg is None:
            continue
        out[y] = (int(round(best_seg[0])), int(round(best_seg[1])))
    return out
