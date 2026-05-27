"""Bubble detector — facade trên `detector.py` để API rõ hơn.

`detector.py` xử lý cả text-pixel mask + bbox. Module này giúp split semantic:
- BubbleDetector → tập trung vào bubble bbox (cls=0 round bubble / cls=1 free-text).
- post-processing pipeline (dedupe, split, recover) gom vào 1 chỗ.

Pipeline tiêu thụ qua facade này thay vì gọi nhiều helper rời rạc — dễ swap
backend model sau (Detectron2/YOLOv8 manga-specific) mà không phá pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import DetectorConfig
from .detector import (
    BaseTextDetector,
    create_detector,
    dedupe_blocks,
    recover_missed_text,
    split_merged_bubbles,
)
from .utils import get_logger


@dataclass
class BubbleDetectionResult:
    text_mask: np.ndarray
    blocks: list[dict]
    n_raw: int
    n_after_dedupe: int
    n_after_split: int


class BubbleDetector:
    """Facade gọn cho phần bubble detection của pipeline.

    Usage:
        det = BubbleDetector(DetectorConfig())
        result = det.detect(image)
        # result.blocks: list[{'bbox':[x1,y1,x2,y2], 'score':float, 'cls':int}]
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self._raw = create_detector(self.config)
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
    def raw(self) -> BaseTextDetector:
        """Truy cập low-level detector nếu cần (vd benchmark per-stage)."""
        return self._raw
