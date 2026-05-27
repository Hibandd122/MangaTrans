"""Text detection base module and post-processing.

Koharu integration (2026-05-27): Refactored to support multiple detection
backends via `create_detector` factory.

Post-processing (dedupe_blocks, split_merged_bubbles, recover_missed_text)
remains here and applies to all backends.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import DetectorConfig
from .utils import get_logger


@dataclass
class TextBlock:
    """1 bbox text. cls=0 = bubble, cls=1 = free text/SFX/narration.

    Dataclass thay dict để type-check + IDE hint. Tuy nhiên pipeline vẫn xài dict
    interface cho backward compatibility — convert qua to_dict / from_dict.
    """

    bbox: tuple[int, int, int, int]
    score: float
    cls: int

    def to_dict(self) -> dict:
        return {"bbox": list(self.bbox), "score": self.score, "cls": self.cls}

    @classmethod
    def from_dict(cls, d: dict) -> "TextBlock":
        return cls(bbox=tuple(d["bbox"]), score=float(d.get("score", 0.0)),
                   cls=int(d.get("cls", 0)))


class BaseTextDetector(ABC):
    """Abstract base class cho mọi text detector backend."""

    def __init__(self, config: DetectorConfig):
        self.config = config

    @abstractmethod
    def detect(self, image: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """Chạy detector. Trả (text_mask uint8 H×W, blocks list dict)."""
        pass

    def release(self) -> None:
        """Giải phóng resources."""
        pass


def create_detector(config: DetectorConfig) -> BaseTextDetector:
    """Khởi tạo detector backend duy nhất (ComicTextDetector)."""
    from .detectors import ComicTextDetector
    return ComicTextDetector(config)


# --------------------------- Hậu kì --------------------------- #

def dedupe_blocks(blocks: list[dict], iou_thresh: float = 0.35,
                  contain_thresh: float = 0.65) -> list[dict]:
    """Loại bbox trùng/lồng nhau — giữ bbox lớn hơn."""
    if not blocks:
        return blocks

    def _area(b: dict) -> int:
        x1, y1, x2, y2 = b["bbox"]
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _intersect(a: dict, b: dict) -> int:
        ax1, ay1, ax2, ay2 = a["bbox"]
        bx1, by1, bx2, by2 = b["bbox"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        return max(0, ix2 - ix1) * max(0, iy2 - iy1)

    order = sorted(range(len(blocks)), key=lambda i: -_area(blocks[i]))
    keep = [True] * len(blocks)
    for pi, i in enumerate(order):
        if not keep[i]:
            continue
        a = blocks[i]
        area_a = _area(a)
        if area_a == 0:
            keep[i] = False
            continue
        for j in order[pi + 1:]:
            if not keep[j]:
                continue
            b = blocks[j]
            inter = _intersect(a, b)
            if inter == 0:
                continue
            area_b = _area(b)
            iob = inter / area_b  # B luôn ≤ A do sort
            iou = inter / (area_a + area_b - inter)
            if iob > contain_thresh or iou > iou_thresh:
                keep[j] = False
    return [b for i, b in enumerate(blocks) if keep[i]]


def split_merged_bubbles(blocks: list[dict], text_mask: np.ndarray,
                         gap_thresh: int = 20, min_comp_area: int = 40) -> list[dict]:
    """Tách bbox gộp nhiều bubble dựa trên gap giữa text CC.

    Recursive cả 2 trục (Y trước, X sau) để bắt grid layout.
    """
    def _split_axis(comps_list, axis):
        if len(comps_list) <= 1:
            return [comps_list]
        sorted_c = sorted(comps_list, key=lambda c: c[axis])
        size_idx = 2 + axis  # cw cho axis 0, ch cho axis 1
        groups = [[sorted_c[0]]]
        for c in sorted_c[1:]:
            last_end = max(g[axis] + g[size_idx] for g in groups[-1])
            if c[axis] - last_end > gap_thresh:
                groups.append([c])
            else:
                groups[-1].append(c)
        return groups

    def _recursive(comps_list):
        if len(comps_list) <= 1:
            return [comps_list]
        y_groups = _split_axis(comps_list, axis=1)
        if len(y_groups) > 1:
            result = []
            for g in y_groups:
                result.extend(_recursive(g))
            return result
        x_groups = _split_axis(comps_list, axis=0)
        if len(x_groups) > 1:
            result = []
            for g in x_groups:
                result.extend(_recursive(g))
            return result
        return [comps_list]

    out: list[dict] = []
    for blk in blocks:
        x1, y1, x2, y2 = blk["bbox"]
        roi = text_mask[y1:y2, x1:x2]
        if roi.size == 0 or roi.sum() == 0:
            out.append(blk)
            continue
        num, _, stats, _ = cv2.connectedComponentsWithStats((roi > 127).astype(np.uint8))
        comps = []
        for i in range(1, num):
            cx, cy, cw, ch, area = stats[i]
            if area < min_comp_area:
                continue
            comps.append((cx, cy, cw, ch))
        if len(comps) <= 1:
            out.append(blk)
            continue
        groups = _recursive(comps)
        if len(groups) == 1:
            out.append(blk)
            continue
        pad = 4
        for g in groups:
            gx1 = min(c[0] for c in g) + x1 - pad
            gy1 = min(c[1] for c in g) + y1 - pad
            gx2 = max(c[0] + c[2] for c in g) + x1 + pad
            gy2 = max(c[1] + c[3] for c in g) + y1 + pad
            new_blk = dict(blk)
            new_blk["bbox"] = [int(max(0, gx1)), int(max(0, gy1)), int(gx2), int(gy2)]
            out.append(new_blk)
    return out


def recover_missed_text(image: np.ndarray, text_mask: np.ndarray, blocks: list[dict],
                        cfg: DetectorConfig) -> list[dict]:
    """Bắt bubble YOLO miss qua seg mask CC. Filter chặt để tránh false positive.

    Logic: CC trên mask close → nếu không overlap đáng kể với block hiện có,
    nằm trên nền trắng (annulus rim test), và density text đủ cao → thêm bbox
    với cls=1, score=0.30 (đánh dấu fallback).
    """
    if text_mask is None or text_mask.size == 0:
        return blocks
    h_img, w_img = text_mask.shape
    bin_mask = (text_mask > 0).astype(np.uint8)
    if bin_mask.sum() == 0:
        return blocks

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    k = cfg.recover_close_ksize
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(bin_mask * 255, cv2.MORPH_CLOSE, kernel)

    n, _labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    new_blocks = list(blocks)
    existing_boxes = [b["bbox"] for b in blocks]
    img_area = h_img * w_img

    for lbl in range(1, n):
        x, y, w, h, area = stats[lbl]
        if area < cfg.recover_min_area:
            continue
        if w < cfg.recover_min_dim or h < cfg.recover_min_dim:
            continue
        if w * h > cfg.recover_max_area_ratio * img_area:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > cfg.recover_max_aspect:
            continue
        bx2, by2 = x + w, y + h
        comp_area = w * h
        raw_pixels = int(bin_mask[y:by2, x:bx2].sum())
        if comp_area > 0 and raw_pixels / comp_area < cfg.recover_min_text_density:
            continue
        # Overlap với bbox hiện có
        overlap_max = 0.0
        for ex1, ey1, ex2, ey2 in existing_boxes:
            ix1, iy1 = max(x, ex1), max(y, ey1)
            ix2, iy2 = min(bx2, ex2), min(by2, ey2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if comp_area > 0 and inter / comp_area > overlap_max:
                overlap_max = inter / comp_area
        if overlap_max > cfg.recover_overlap_thresh:
            continue
        # Annulus whiteness test
        pad_s = 8
        ax1 = max(0, x - pad_s)
        ay1 = max(0, y - pad_s)
        ax2 = min(w_img, bx2 + pad_s)
        ay2 = min(h_img, by2 + pad_s)
        annulus = np.ones((ay2 - ay1, ax2 - ax1), dtype=bool)
        inner_x1 = x - ax1
        inner_y1 = y - ay1
        inner_x2 = bx2 - ax1
        inner_y2 = by2 - ay1
        annulus[inner_y1:inner_y2, inner_x1:inner_x2] = False
        ann_pix = gray[ay1:ay2, ax1:ax2][annulus]
        if ann_pix.size == 0 or ann_pix.mean() < cfg.recover_min_surround_white:
            continue
        pad = 6
        nx1 = max(0, x - pad)
        ny1 = max(0, y - pad)
        nx2 = min(w_img, bx2 + pad)
        ny2 = min(h_img, by2 + pad)
        new_blocks.append({
            "bbox": [int(nx1), int(ny1), int(nx2), int(ny2)],
            "score": 0.30,
            "cls": 1,
        })
    return new_blocks
