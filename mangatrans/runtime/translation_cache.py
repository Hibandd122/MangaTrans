"""TranslationCache — persistent JSON cache cho 1-1 text translations.

Cache key = sha256(f"{model}\\x00{target_lang}\\x00{text}"). Glossary KHÔNG nằm
trong key vì glossary chỉ nudges output — không phải hard constraint. Re-compute
khi glossary đổi sẽ kill giá trị cache trên 1 chap đang dịch (glossary grow per
page). Tài liệu rõ caveat này tại doc-string `get`.

Storage format: JSON dict { key: {translation, model, target_lang, text_preview, ts} }.
Atomic write qua tmp + os.replace, debounce. Thread-safe.

Eviction: chap nhỏ → không cần. Nếu `max_entries` reached, drop random 5% cũ
nhất theo `ts` (FIFO). Đủ cho usecase 50_000 entries.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .checkpoint import atomic_write_json


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def make_cache_key(model: str, target_lang: str, text: str) -> str:
    """sha256(model || NUL || target_lang || NUL || text) hex.

    NUL separator chống collision (vd text="A\\nB" vs lang="A", text="\\nB").
    """
    h = hashlib.sha256()
    h.update((model or "").encode("utf-8"))
    h.update(b"\x00")
    h.update((target_lang or "").encode("utf-8"))
    h.update(b"\x00")
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()


class TranslationCache:
    """Persistent translation cache. Thread-safe.

    Caveat: cache key KHÔNG có glossary. Hit có thể trả translation không khớp
    glossary mới nhất. Trên 1 chap dài, glossary chỉ thêm tên mới — nên hit
    cho text cũ vẫn hợp lệ. Để force refresh: xoá `.translation_cache.json`.
    """

    def __init__(
        self,
        path: Optional[str],
        max_entries: int = 50_000,
        debounce_s: float = 2.0,
    ):
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

    # ---- Core API ----

    def get(self, model: str, target_lang: str, text: str) -> Optional[str]:
        """Trả translation đã cache hoặc None."""
        if not text:
            return None
        key = make_cache_key(model, target_lang, text)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            self._stats["hits"] += 1
            return entry.get("translation")

    def put(self, model: str, target_lang: str, text: str, translation: str) -> None:
        if not text or translation is None:
            return
        key = make_cache_key(model, target_lang, text)
        entry = {
            "translation": translation,
            "model": model,
            "target_lang": target_lang,
            "text_preview": text[:80],
            "ts": time.time(),
        }
        with self._lock:
            self._cache[key] = entry
            self._stats["puts"] += 1
            self._dirty = True
            if len(self._cache) > self.max_entries:
                self._evict_locked()

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    # ---- Flush ----

    def flush_if_due(self) -> bool:
        now = time.perf_counter()
        with self._lock:
            if not self._dirty or not self.path:
                return False
            if now - self._last_flush_ts < self.debounce_s:
                return False
        return self.flush()

    def flush(self) -> bool:
        if not self.path:
            return False
        with self._lock:
            if not self._dirty:
                return False
            snapshot = {
                "version": 1,
                "ts": _utc_iso(),
                "entries": dict(self._cache),
            }
        try:
            atomic_write_json(self.path, snapshot)
        except OSError as exc:
            sys.stderr.write(f"[TranslationCache] flush failed: {exc}\n")
            return False
        with self._lock:
            self._dirty = False
            self._last_flush_ts = time.perf_counter()
        return True

    # ---- Internals ----

    def _load(self) -> None:
        if not self.path or not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            sys.stderr.write(f"[TranslationCache] load failed ({exc}); starting empty\n")
            return
        if not isinstance(data, dict):
            return
        entries = data.get("entries") or {}
        if not isinstance(entries, dict):
            return
        with self._lock:
            for k, v in entries.items():
                if isinstance(v, dict) and "translation" in v:
                    self._cache[k] = v

    def _evict_locked(self) -> None:
        # Drop 5% entries cũ nhất theo ts. Cheap O(n log n) — chỉ chạy khi overflow.
        target = int(self.max_entries * 0.95)
        if len(self._cache) <= target:
            return
        sorted_items = sorted(self._cache.items(), key=lambda kv: kv[1].get("ts", 0.0))
        drop = len(self._cache) - target
        for k, _ in sorted_items[:drop]:
            self._cache.pop(k, None)


class NullTranslationCache(TranslationCache):
    """No-op cache (cache disabled). Vẫn count stats để diagnostics."""

    def __init__(self) -> None:
        super().__init__(path=None, max_entries=0, debounce_s=0.0)

    def get(self, model: str, target_lang: str, text: str) -> Optional[str]:
        with self._lock:
            self._stats["misses"] += 1
        return None

    def put(self, model: str, target_lang: str, text: str, translation: str) -> None:
        return

    def flush_if_due(self) -> bool:
        return False

    def flush(self) -> bool:
        return False
