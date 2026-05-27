"""GPUMutex — asyncio.Semaphore(1) wrapper với current holder tracking.

Wrapper rất mỏng quanh `asyncio.Semaphore(1)` để watchdog có thể hỏi "page nào
đang giữ GPU?" + "đã giữ bao lâu?". Helper context manager async with-statement.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional


class GPUMutex:
    """Serialize GPU work giữa các page in-flight.

    1 lúc chỉ 1 stage GPU chạy (detect/lang/ocr/inpaint). Stage CPU/IO khác
    không acquire mutex này nên parallel tự nhiên.
    """

    def __init__(self) -> None:
        self._sem = asyncio.Semaphore(1)
        self._holder: Optional[str] = None
        self._stage: Optional[str] = None
        self._acquired_at: float = 0.0

    @asynccontextmanager
    async def acquire(self, page_id: str, stage: str) -> AsyncIterator[None]:
        await self._sem.acquire()
        self._holder = page_id
        self._stage = stage
        self._acquired_at = time.perf_counter()
        try:
            yield
        finally:
            self._holder = None
            self._stage = None
            self._acquired_at = 0.0
            self._sem.release()

    def current_holder(self) -> Optional[str]:
        return self._holder

    def current_stage(self) -> Optional[str]:
        return self._stage

    def held_for_s(self) -> float:
        if self._acquired_at == 0.0:
            return 0.0
        return time.perf_counter() - self._acquired_at

    def is_locked(self) -> bool:
        return self._holder is not None
