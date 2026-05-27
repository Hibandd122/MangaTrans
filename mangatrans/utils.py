"""Utility helpers: image IO, logging, model probing, cache.

Tránh side-effect ở module-level (vd `_cache = {}` global) — đổi sang explicit
context (LRU class hoặc bound trên pipeline instance).
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import cv2
import numpy as np


_LOGGER_INITIALIZED = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup root logger với format đẹp cho terminal. Idempotent."""
    global _LOGGER_INITIALIZED
    logger = logging.getLogger("mangatrans")
    if _LOGGER_INITIALIZED:
        logger.setLevel(level)
        return logger

    if hasattr(sys.stdout, "reconfigure"):
        # Đảm bảo output Unicode chạy ổn trên Windows console.
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(message)s",  # giữ tone "emoji + text" của codebase cũ
    ))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    _LOGGER_INITIALIZED = True
    return logger


def get_logger() -> logging.Logger:
    """Lấy logger module-scoped. Đảm bảo setup_logging đã chạy ít nhất 1 lần."""
    if not _LOGGER_INITIALIZED:
        setup_logging()
    return logging.getLogger("mangatrans")


def load_image(image_path: str) -> np.ndarray:
    """Đọc ảnh màu BGR. Raise FileNotFoundError nếu fail."""
    # cv2.imread không support Unicode path trên Windows — fallback qua numpy.
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        # Try unicode path workaround
        try:
            with open(image_path, "rb") as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception as e:
            raise FileNotFoundError(f"Không đọc được ảnh: {image_path}") from e
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
    return img


def save_image(image: np.ndarray, path: str) -> None:
    """Lưu ảnh BGR. Hỗ trợ Unicode path qua imencode."""
    # Tạo parent dir nếu chưa có
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"imencode fail cho {path}")
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def auto_pick_inpaint_model(preferred: Optional[str], candidates,
                            base_dir: str = ".") -> tuple[Optional[str], Optional[str]]:
    """Chọn inpaint model: ưu tiên `preferred` nếu tồn tại, sau đó duyệt candidates.

    Trả (absolute_path, friendly_name) hoặc (None, None).
    """
    if preferred and os.path.isfile(preferred):
        return os.path.abspath(preferred), os.path.basename(preferred)
    for fname, label in candidates:
        full = os.path.join(base_dir, fname)
        if os.path.isfile(full):
            return os.path.abspath(full), label
        if os.path.isfile(fname):
            return os.path.abspath(fname), label
    return None, None


def pick_first_existing_font(candidates) -> Optional[str]:
    """Trả font path đầu tiên tồn tại từ chuỗi candidates."""
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def ensure_dir(path: str) -> str:
    """Tạo directory nếu chưa có. Return path đã abs."""
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def clamp_bbox(bbox, w: int, h: int) -> tuple[int, int, int, int]:
    """Clamp bbox [x1,y1,x2,y2] vào ảnh kích thước (w,h). Đảm bảo x1<x2, y1<y2."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(x1 + 1, min(w, int(x2)))
    y2 = max(y1 + 1, min(h, int(y2)))
    return x1, y1, x2, y2


def to_uint8_image(arr: np.ndarray) -> np.ndarray:
    """Convert float / arbitrary-range array sang uint8 [0,255]."""
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype in (np.float32, np.float64):
        if float(arr.max()) > 1.5:
            return np.clip(arr, 0, 255).astype(np.uint8)
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return arr.astype(np.uint8)


def has_real_cjk(text: str) -> bool:
    """True nếu text chứa CJK char (Hiragana/Katakana/Han/Hangul)."""
    if not text:
        return False
    for ch in text:
        cp = ord(ch)
        if (0x3040 <= cp <= 0x30FF
                or 0x4E00 <= cp <= 0x9FFF
                or 0xAC00 <= cp <= 0xD7A3):
            return True
    return False


def has_latin_letters(text: str, min_count: int = 2) -> bool:
    """True nếu text có >=min_count Latin letter."""
    return sum(1 for ch in (text or "") if ch.isalpha() and ord(ch) < 128) >= min_count
