"""Optional OCRCache — per-crop sha256 → ocr_result.

Off by default (`RuntimeConfig.ocr_cache_enabled=False`) vì:
- Key = sha256(crop bytes) đắt cho crop 200x200.
- Manga page ít khi chứa bubble identical → hit rate thấp.
- Vẫn giữ implementation cho user power dùng (vd dịch lại cùng chap với
  glossary khác).

Thread-safe; reuse pattern atomic JSON từ TranslationCache.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from .checkpoint import atomic_write_json


def crop_hash(image_bgr: np.ndarray) -> str:
    """sha256 của crop bytes. Caller phải đảm bảo crop contiguous."""
    if not image_bgr.flags["C_CONTIGUOUS"]:
        image_bgr = np.ascontiguousarray(image_bgr)
    h = hashlib.sha256()
    h.update(str(image_bgr.shape).encode("utf-8"))
    h.update(b"\x00")
    h.update(image_bgr.dtype.str.encode("utf-8"))
    h.update(b"\x00")
    h.update(image_bgr.tobytes())
    return h.hexdigest()


class OCRCache:
    """Optional per-crop OCR result cache. Disabled by default."""

    def __init__(self, path: Optional[str], max_entries: int = 20_000,
                 debounce_s: float = 2.0):
        self.path = path
        self.max_entries = int(max_entries)
        self.debounce_s = float(max(0.0, debounce_s))
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._dirty = False
        self._last_flush_ts = 0.0
        self._stats = {"hits": 0, "misses": 0, "puts": 0}
        if self.path:
            self._load()

    def get(self, crop_key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._cache.get(crop_key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            self._stats["hits"] += 1
            return entry.get("result")

    def put(self, crop_key: str, result: Dict[str, Any]) -> None:
        with self._lock:
            self._cache[crop_key] = {"result": result, "ts": time.time()}
            self._stats["puts"] += 1
            self._dirty = True
            if len(self._cache) > self.max_entries:
                self._evict_locked()

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def flush(self) -> bool:
        if not self.path:
            return False
        with self._lock:
            if not self._dirty:
                return False
            snap = {"version": 1, "entries": dict(self._cache)}
        try:
            atomic_write_json(self.path, snap)
        except OSError as exc:
            sys.stderr.write(f"[OCRCache] flush failed: {exc}\n")
            return False
        with self._lock:
            self._dirty = False
            self._last_flush_ts = time.perf_counter()
        return True

    def flush_if_due(self) -> bool:
        now = time.perf_counter()
        with self._lock:
            if not self._dirty or not self.path:
                return False
            if now - self._last_flush_ts < self.debounce_s:
                return False
        return self.flush()

    def _load(self) -> None:
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            sys.stderr.write(f"[OCRCache] load failed ({exc}); starting empty\n")
            return
        entries = data.get("entries") or {}
        if isinstance(entries, dict):
            with self._lock:
                for k, v in entries.items():
                    if isinstance(v, dict) and "result" in v:
                        self._cache[k] = v

    def _evict_locked(self) -> None:
        target = int(self.max_entries * 0.95)
        if len(self._cache) <= target:
            return
        sorted_items = sorted(self._cache.items(), key=lambda kv: kv[1].get("ts", 0.0))
        drop = len(self._cache) - target
        for k, _ in sorted_items[:drop]:
            self._cache.pop(k, None)


class NullOCRCache(OCRCache):
    def __init__(self):
        super().__init__(path=None, max_entries=0, debounce_s=0.0)

    def get(self, crop_key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._stats["misses"] += 1
        return None

    def put(self, crop_key: str, result: Dict[str, Any]) -> None:
        return
