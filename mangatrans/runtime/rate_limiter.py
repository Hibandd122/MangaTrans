"""TokenBucket — RPM rate limiter cho OpenRouter API.

Capacity = RPM, refill = RPM/60 token/giây. `acquire()` chờ đến khi có token.
`update_rpm(new)` thay đổi rate at runtime (adaptive trên 429).

Tách khỏi `translation_worker` để test riêng dễ.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional


class TokenBucket:
    """Async token bucket. Một thread / 1 event loop.

    Cấu trúc canonical: `tokens` floating; refill khi `acquire` được gọi (lazy).
    """

    def __init__(self, rpm: int, capacity: Optional[int] = None) -> None:
        if rpm <= 0:
            raise ValueError("rpm must be > 0")
        self._rpm = int(rpm)
        self._capacity = float(capacity if capacity is not None else rpm)
        self._tokens = float(self._capacity)
        self._last_refill = time.perf_counter()
        self._lock = asyncio.Lock()

    @property
    def rpm(self) -> int:
        return self._rpm

    @property
    def capacity(self) -> float:
        return self._capacity

    def tokens_available(self) -> float:
        """Snapshot tokens hiện tại sau khi refill lazy. Không acquire lock async
        nên có thể slightly stale dưới load — đủ chính xác cho diagnostics/test."""
        now = time.perf_counter()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return self._tokens
        rate_per_s = self._rpm / 60.0
        return min(self._capacity, self._tokens + elapsed * rate_per_s)

    async def acquire(self, tokens: float = 1.0) -> None:
        """Wait đến khi có đủ `tokens`. Caller có thể bị cancel."""
        if tokens <= 0:
            return
        while True:
            async with self._lock:
                self._refill_locked()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                # Cần thêm `needed` token; tốc độ refill = rpm/60 token/s.
                needed = tokens - self._tokens
                rate_per_s = max(self._rpm / 60.0, 1e-6)
                wait_s = needed / rate_per_s
            # Sleep ngoài lock để producer khác có thể acquire/refill.
            await asyncio.sleep(min(wait_s, 5.0))

    def update_rpm(self, new_rpm: int) -> None:
        """Thay đổi rate tại runtime. Capacity cũng update theo new_rpm."""
        new_rpm = max(1, int(new_rpm))
        # Không cần async lock — single-thread asyncio, ghi atomic.
        self._rpm = new_rpm
        self._capacity = float(new_rpm)
        # Clamp tokens hiện tại về capacity mới.
        if self._tokens > self._capacity:
            self._tokens = self._capacity

    def _refill_locked(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        rate_per_s = self._rpm / 60.0
        self._tokens = min(self._capacity, self._tokens + elapsed * rate_per_s)
        self._last_refill = now
