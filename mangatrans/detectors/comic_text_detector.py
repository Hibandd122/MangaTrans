"""Original comic-text-detector ONNX adapter."""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from ..config import DetectorConfig
from ..detector import BaseTextDetector
from ..utils import get_logger


class ComicTextDetector(BaseTextDetector):
    """Wrap comic-text-detector ONNX. Lazy load session."""

    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self._session = None
        self._log = get_logger()

    def _ensure_session(self):
        if self._session is None:
            import onnxruntime as ort
            if not os.path.isfile(self.config.model_path):
                raise FileNotFoundError(
                    f"Không tìm thấy model detection: {self.config.model_path}")
            available = set(ort.get_available_providers())
            preferred = ["DmlExecutionProvider", "CUDAExecutionProvider",
                         "CPUExecutionProvider"]
            providers = [p for p in preferred if p in available]
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._session = ort.InferenceSession(
                self.config.model_path, sess_options=so, providers=providers,
            )
            active = self._session.get_providers()
            self._log.info(f"   - Detector ONNX provider: {active[0]}")
        return self._session

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, list[dict]]:
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

        # --- Mask 'seg' ---
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

        # --- Bbox 'blk' ---
        blk_out = named.get("blk")
        if blk_out is not None and blk_out.ndim == 3 and blk_out.shape[2] >= 6:
            blocks = self._parse_blk_output(
                blk_out, scale, top, left, h, w,
                cfg.blk_conf, cfg.blk_nms,
            )
        else:
            blocks = []
        return mask, blocks

    def _parse_blk_output(self, blk_output: np.ndarray, scale: float, pad_top: int, pad_left: int,
                         orig_h: int, orig_w: int, conf_thresh: float,
                         nms_thresh: float) -> list[dict]:
        """Decode YOLO-style output `blk` (1,N,>=6) → bbox tọa độ ảnh gốc."""
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
