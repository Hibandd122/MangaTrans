"""Text/bubble detection — comic-text-detector ONNX + post-processing.

Gộp `detector.py` + `bubble_detector.py`:
- TextDetector: low-level ONNX inference → mask + raw bbox.
- BubbleDetector: facade chạy detect + dedupe + split + recover.
- TextBlock dataclass cho type-safety nếu cần.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from .config import DetectorConfig
from .utils import get_logger


@dataclass
class TextBlock:
    """1 bbox text. cls=0 = bubble, cls=1 = free text/SFX/narration."""

    bbox: tuple[int, int, int, int]
    score: float
    cls: int

    def to_dict(self) -> dict:
        return {"bbox": list(self.bbox), "score": self.score, "cls": self.cls}

    @classmethod
    def from_dict(cls, d: dict) -> "TextBlock":
        return cls(bbox=tuple(d["bbox"]), score=float(d.get("score", 0.0)),
                   cls=int(d.get("cls", 0)))


class TextDetector:
    """Wrap comic-text-detector ONNX. Lazy load session, cache đến khi reset."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self._session: Optional[ort.InferenceSession] = None
        self._log = get_logger()

    def _ensure_session(self) -> ort.InferenceSession:
        if self._session is None:
            if not os.path.isfile(self.config.model_path):
                raise FileNotFoundError(
                    f"Không tìm thấy model detection: {self.config.model_path}")
            available = set(ort.get_available_providers())
            preferred = ["DmlExecutionProvider", "CUDAExecutionProvider",
                         "CPUExecutionProvider"]
            providers = [p for p in preferred if p in available]
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = self.config.intra_op_threads
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._session = ort.InferenceSession(
                self.config.model_path, sess_options=so, providers=providers,
            )
            active = self._session.get_providers()
            self._log.info(f"   - Detector ONNX provider: {active[0]}")
        return self._session

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """Chạy detector. Trả (text_mask uint8 H×W, blocks list dict)."""
        cfg = self.config
        h, w = image.shape[:2]

        scale = cfg.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        delta_h = cfg.target_size - new_h
        delta_w = cfg.target_size - new_w
        top, bottom = delta_h // 2, delta_h - delta_h // 2
        left, right = delta_w // 2, delta_w - delta_w // 2
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=0)

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        inp = rgb.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None]

        sess = self._ensure_session()
        input_name = sess.get_inputs()[0].name
        output_names = [o.name for o in sess.get_outputs()]
        outputs = sess.run(None, {input_name: inp})
        named = dict(zip(output_names, outputs))

        if "seg" in named:
            raw = named["seg"]
        else:
            raw = next((o for o in outputs if o.ndim == 4 and o.shape[1] == 1), outputs[0])

        if raw.ndim == 4:
            prob = raw[0, 0]
        elif raw.ndim == 3:
            prob = raw[0]
        else:
            prob = raw

        if prob.shape[:2] != (cfg.target_size, cfg.target_size):
            prob = cv2.resize(prob, (cfg.target_size, cfg.target_size),
                              interpolation=cv2.INTER_LINEAR)
        prob = prob[top:top + new_h, left:left + new_w]
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        _, mask = cv2.threshold(prob, cfg.mask_thresh, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)

        blk_out = named.get("blk")
        if blk_out is not None and blk_out.ndim == 3 and blk_out.shape[2] >= 6:
            blocks = _parse_blk_output(
                blk_out, scale, top, left, h, w,
                cfg.blk_conf, cfg.blk_nms,
            )
        else:
            blocks = []
        return mask, blocks


def _parse_blk_output(blk_output: np.ndarray, scale: float, pad_top: int, pad_left: int,
                     orig_h: int, orig_w: int, conf_thresh: float,
                     nms_thresh: float) -> list[dict]:
    blks = blk_output[0]
    obj_conf = blks[:, 4]
    cls_conf = blks[:, 5:]
    cls_scores = cls_conf.max(axis=1)
    cls_ids = cls_conf.argmax(axis=1)
    scores = obj_conf * cls_scores

    keep = scores > conf_thresh
    if not keep.any():
        return []
    cx = blks[keep, 0]
    cy = blks[keep, 1]
    bw = blks[keep, 2]
    bh = blks[keep, 3]
    s = scores[keep].astype(np.float32)
    c = cls_ids[keep]

    boxes_xywh = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1).astype(np.float32)
    indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), s.tolist(), conf_thresh, nms_thresh)
    if len(indices) == 0:
        return []
    indices = np.array(indices).flatten()

    blocks: list[dict] = []
    for i in indices:
        x1 = (boxes_xywh[i, 0] - pad_left) / scale
        y1 = (boxes_xywh[i, 1] - pad_top) / scale
        x2 = (boxes_xywh[i, 0] + boxes_xywh[i, 2] - pad_left) / scale
        y2 = (boxes_xywh[i, 1] + boxes_xywh[i, 3] - pad_top) / scale
        x1i = int(max(0, min(orig_w - 1, x1)))
        y1i = int(max(0, min(orig_h - 1, y1)))
        x2i = int(max(0, min(orig_w, x2)))
        y2i = int(max(0, min(orig_h, y2)))
        if x2i - x1i < 5 or y2i - y1i < 5:
            continue
        blocks.append({
            "bbox": [x1i, y1i, x2i, y2i],
            "score": float(s[i]),
            "cls": int(c[i]),
        })
    blocks.sort(key=lambda b: (b["bbox"][1], -b["bbox"][0]))
    return blocks


# --------------------------- Post-processing --------------------------- #

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
            iob = inter / area_b
            iou = inter / (area_a + area_b - inter)
            if iob > contain_thresh or iou > iou_thresh:
                keep[j] = False
    return [b for i, b in enumerate(blocks) if keep[i]]


def split_merged_bubbles(blocks: list[dict], text_mask: np.ndarray,
                         gap_thresh: int = 20, min_comp_area: int = 40) -> list[dict]:
    """Tách bbox gộp nhiều bubble dựa trên gap giữa text CC."""
    def _split_axis(comps_list, axis):
        if len(comps_list) <= 1:
            return [comps_list]
        sorted_c = sorted(comps_list, key=lambda c: c[axis])
        size_idx = 2 + axis
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
    """Bắt bubble YOLO miss qua seg mask CC. Filter chặt để tránh false positive."""
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
        overlap_max = 0.0
        for ex1, ey1, ex2, ey2 in existing_boxes:
            ix1, iy1 = max(x, ex1), max(y, ey1)
            ix2, iy2 = min(bx2, ex2), min(by2, ey2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if comp_area > 0 and inter / comp_area > overlap_max:
                overlap_max = inter / comp_area
        if overlap_max > cfg.recover_overlap_thresh:
            continue
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


# --------------------------- Facade --------------------------- #

@dataclass
class BubbleDetectionResult:
    text_mask: np.ndarray
    blocks: list[dict]
    n_raw: int
    n_after_dedupe: int
    n_after_split: int


class BubbleDetector:
    """Facade chạy detect + dedupe + split + recover trong 1 call."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self._raw = TextDetector(self.config)
        self._log = get_logger()

    def detect(self, image: np.ndarray) -> BubbleDetectionResult:
        cfg = self.config
        text_mask, blocks_raw = self._raw.detect(image)
        n_raw = len(blocks_raw)
        blocks = dedupe_blocks(blocks_raw,
                               iou_thresh=cfg.dedupe_iou,
                               contain_thresh=cfg.dedupe_contain)
        n_dedup = len(blocks)
        blocks = split_merged_bubbles(blocks, text_mask,
                                      gap_thresh=cfg.split_gap_thresh,
                                      min_comp_area=cfg.split_min_comp_area)
        n_split = len(blocks)
        blocks = recover_missed_text(image, text_mask, blocks, cfg)
        if n_dedup < n_raw:
            self._log.info(f"   - Dedupe: {n_raw} → {n_dedup} bubble")
        if n_split > n_dedup:
            self._log.info(f"   - Split merged: {n_dedup} → {n_split} bubble")
        if len(blocks) > n_split:
            self._log.info(f"   - Recover missed: {n_split} → {len(blocks)} bubble")
        return BubbleDetectionResult(
            text_mask=text_mask,
            blocks=blocks,
            n_raw=n_raw,
            n_after_dedupe=n_dedup,
            n_after_split=n_split,
        )

    @property
    def raw(self) -> TextDetector:
        return self._raw
