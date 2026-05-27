"""StructuredLogger — JSONL log + ring buffer cho crash report.

Tại sao tách khỏi `logging` chuẩn:
- Stage timing cần ghi event start/end với metadata (page_id, stage, latency_ms).
- Crash report cần last N events kèm trace → cần ring buffer in-memory.
- JSONL dễ grep/jq, không lẫn với emoji-prose log của legacy path.

Thread-safe: lock quanh write + rotate + ring buffer append. Drop-safe: nếu write
fail (disk full / file lock) log exception ra stderr và tiếp tục, không raise lên
caller (logging KHÔNG được làm crash pipeline).
"""
from __future__ import annotations

import collections
import contextlib
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Iterator, List, Optional


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class StructuredLogger:
    """JSONL writer + ring buffer. Append-only, thread-safe, idempotent close.

    File rotation policy: khi vượt `max_bytes`, rename `<path>` → `<path>.1`
    (overwrite cũ), mở file mới. Single backup là đủ — không cần multi-gen rotate
    cho debug log.
    """

    def __init__(
        self,
        path: Optional[str],
        ring_size: int = 30,
        max_bytes: int = 50 * 1024 * 1024,
    ):
        self.path = path
        self.max_bytes = int(max_bytes)
        self._ring: Deque[Dict[str, Any]] = collections.deque(maxlen=int(ring_size))
        self._lock = threading.Lock()
        self._fp = None
        self._closed = False
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
            # 'a' để resume sau crash không mất log cũ.
            try:
                self._fp = open(self.path, "a", encoding="utf-8")
            except OSError as exc:
                # Disk lỗi → fail-soft. Vẫn giữ ring buffer trong RAM.
                sys.stderr.write(f"[StructuredLogger] open failed: {exc}\n")
                self._fp = None

    # ---- Event API ----

    def event(self, event_type: str, **fields: Any) -> None:
        """Ghi 1 event vào JSONL + ring buffer. Không raise."""
        if self._closed:
            return
        record: Dict[str, Any] = {
            "ts": _utc_iso(),
            "event": event_type,
        }
        # Bỏ None để JSON gọn.
        for k, v in fields.items():
            if v is not None:
                record[k] = _jsonable(v)
        line = None
        with self._lock:
            self._ring.append(record)
            if self._fp is not None:
                try:
                    line = json.dumps(record, ensure_ascii=False)
                    self._fp.write(line + "\n")
                    self._fp.flush()
                    self._maybe_rotate_locked()
                except (OSError, ValueError) as exc:
                    sys.stderr.write(f"[StructuredLogger] write failed: {exc}\n")

    def snapshot_recent(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Snapshot last N events từ ring buffer. n=None → toàn bộ ring."""
        with self._lock:
            if n is None:
                return list(self._ring)
            if n <= 0:
                return []
            return list(self._ring)[-n:]

    # ---- Context manager cho stage timing ----

    @contextlib.contextmanager
    def stage_timer(self, stage: str, page_id: Optional[str] = None,
                    **extra: Any) -> Iterator[Dict[str, Any]]:
        """Emit `stage_start` + `stage_end` với latency_ms. Yield dict để stage gắn thêm."""
        info: Dict[str, Any] = {}
        t0 = time.perf_counter()
        self.event("stage_start", stage=stage, page_id=page_id, **extra)
        try:
            yield info
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.event("stage_end", stage=stage, page_id=page_id,
                       latency_ms=round(latency_ms, 2), ok=True, **info)
        except BaseException as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.event(
                "stage_error", stage=stage, page_id=page_id,
                latency_ms=round(latency_ms, 2), ok=False,
                error_type=type(exc).__name__, error_msg=str(exc)[:500],
                **info,
            )
            raise

    # ---- Lifecycle ----

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._fp is not None:
                try:
                    self._fp.flush()
                    self._fp.close()
                except OSError:
                    pass
                self._fp = None

    # ---- Internals ----

    def _maybe_rotate_locked(self) -> None:
        """Caller phải giữ lock. Rotate khi file > max_bytes."""
        if self._fp is None or not self.path:
            return
        try:
            size = self._fp.tell()
        except OSError:
            return
        if size < self.max_bytes:
            return
        try:
            self._fp.close()
            backup = self.path + ".1"
            if os.path.exists(backup):
                os.remove(backup)
            os.replace(self.path, backup)
            self._fp = open(self.path, "a", encoding="utf-8")
        except OSError as exc:
            sys.stderr.write(f"[StructuredLogger] rotate failed: {exc}\n")
            # Cố mở lại file cũ; nếu fail, log → stderr only.
            try:
                self._fp = open(self.path, "a", encoding="utf-8")
            except OSError:
                self._fp = None


def _jsonable(v: Any) -> Any:
    """Best-effort coerce → JSON-safe. Tránh raise trong logger."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, set):
        return sorted(_jsonable(x) for x in v)
    return repr(v)[:500]


class NullStructuredLogger(StructuredLogger):
    """No-op logger (path=None). Vẫn giữ ring buffer cho crash report tại runtime."""

    def __init__(self, ring_size: int = 30):
        super().__init__(path=None, ring_size=ring_size, max_bytes=0)
