"""Watchdog — daemon thread phát hiện stage stuck → cancel + requeue.

Vấn đề: Python KHÔNG preempt được C extension đang stuck (Paddle OCR forward,
EasyOCR readtext, LaMa torch.jit.forward). Khi 1 stage chạy quá timeout:
- Watchdog set `task.cancel_requested = True` (cờ scheduler đọc tại boundary).
- Watchdog gọi `loop.call_soon_threadsafe(scheduler._cancel_task, page_id)`
  → coroutine `_run_stage` raise CancelledError ở await point gần nhất.
- Stage code Python (file IO, dataset prep) sẽ cancel ngay.
- Stage code C (model forward) phải chạy xong rồi mới raise — không thể đụng
  được. Sau khi forward trả về, asyncio sẽ raise CancelledError.

→ Watchdog đảm bảo:
  ✓ Stage bị stuck trong forward → SẼ retry sau khi forward (eventually) return.
  ✓ Stage bị stuck trong Python loop → cancel ngay.
  ✗ Stage bị stuck trong native deadlock (CUDA hang) → process restart cần thiết.

Per-stage timeout đọc từ `RuntimeConfig.watchdog_stage_timeout_s`. Heartbeat
do scheduler update sau mỗi stage_start (`task.last_heartbeat_ts = perf_counter()`).
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Callable, Dict, Optional, Protocol

from ..utils import get_logger
from .config import RuntimeConfig
from .page_task import PageTask, StageName


class _SchedulerProto(Protocol):
    """Subset của Scheduler API mà Watchdog dùng. Tách Protocol để test mock dễ."""

    def cancel_task(self, page_id: str) -> None: ...


class Watchdog:
    """Daemon thread giám sát timeout stage của các page đang chạy.

    Lifecycle:
      - `start(loop, scheduler)` — bắt đầu poll background.
      - `register(task)` / `unregister(page_id)` — track 1 page in-flight.
      - `heartbeat(page_id, stage)` — gọi mỗi khi stage start (scheduler).
      - `stop(timeout=5)` — graceful shutdown.

    Thread-safety: state behind `self._lock`. Watchdog đọc snapshot trước khi
    check, không hold lock khi gọi scheduler.cancel_task (tránh deadlock).
    """

    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self._log = get_logger()
        self._lock = threading.RLock()
        self._tracked: Dict[str, PageTask] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._scheduler: Optional[_SchedulerProto] = None
        self._on_freeze: Optional[Callable[[PageTask, str, float], None]] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stats = {"freezes_detected": 0, "cancels_issued": 0}

    # --------------------------- Public API --------------------------- #

    def start(
        self,
        loop: asyncio.AbstractEventLoop,
        scheduler: _SchedulerProto,
        on_freeze: Optional[Callable[[PageTask, str, float], None]] = None,
    ) -> None:
        """Bật daemon. Idempotent."""
        if not self.cfg.watchdog_enable:
            self._log.info("[Watchdog] disabled qua RuntimeConfig.")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._loop = loop
        self._scheduler = scheduler
        self._on_freeze = on_freeze
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="mangatrans-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Graceful shutdown. Idempotent."""
        self._stop_event.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._thread = None

    def register(self, task: PageTask) -> None:
        """Track 1 page in-flight. Reset heartbeat."""
        with self._lock:
            task.last_heartbeat_ts = time.perf_counter()
            self._tracked[task.page_id] = task

    def unregister(self, page_id: str) -> None:
        """Drop tracking khi page DONE/FAILED."""
        with self._lock:
            self._tracked.pop(page_id, None)

    def heartbeat(self, page_id: str, stage: StageName) -> None:
        """Gọi mỗi khi 1 stage bắt đầu — reset timestamp."""
        with self._lock:
            task = self._tracked.get(page_id)
            if task is None:
                return
            task.current_stage = stage
            task.last_heartbeat_ts = time.perf_counter()

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def tracked_page_ids(self) -> list[str]:
        with self._lock:
            return list(self._tracked.keys())

    # --------------------------- Internals --------------------------- #

    def _run(self) -> None:
        """Main poll loop. Sleep theo `watchdog_poll_interval_s`."""
        self._log.info(
            f"[Watchdog] started, poll mỗi {self.cfg.watchdog_poll_interval_s}s"
        )
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception as exc:  # noqa: BLE001
                # KHÔNG crash watchdog — log + continue.
                self._log.warning(f"[Watchdog] poll fail (ignored): {exc}")
            # Sleep interruptible.
            if self._stop_event.wait(self.cfg.watchdog_poll_interval_s):
                break
        self._log.info("[Watchdog] stopped.")

    def _poll_once(self) -> None:
        """Snapshot tracked tasks → check timeout → issue cancel."""
        now = time.perf_counter()
        to_cancel: list[tuple[PageTask, str, float]] = []
        with self._lock:
            for task in list(self._tracked.values()):
                if task.cancel_requested:
                    # Đã cancel từ trước, chờ scheduler dọn.
                    continue
                stage = task.current_stage
                if stage is None or task.last_heartbeat_ts == 0.0:
                    continue
                timeout = self.cfg.stage_timeout(stage.value)
                elapsed = now - task.last_heartbeat_ts
                if elapsed >= timeout:
                    to_cancel.append((task, stage.value, elapsed))

        # Issue cancel OUTSIDE lock — call_soon_threadsafe có thể block ngắn,
        # và on_freeze callback của caller có thể tốn time.
        for task, stage_name, elapsed in to_cancel:
            self._issue_cancel(task, stage_name, elapsed)

    def _issue_cancel(self, task: PageTask, stage_name: str, elapsed: float) -> None:
        """Mark cancel + push vào event loop."""
        with self._lock:
            task.cancel_requested = True
            self._stats["freezes_detected"] += 1
            self._stats["cancels_issued"] += 1

        self._log.warning(
            f"⚠️  [Watchdog] page={task.page_id} stage={stage_name} "
            f"freeze {elapsed:.1f}s — issuing cancel"
        )

        if self._on_freeze is not None:
            try:
                self._on_freeze(task, stage_name, elapsed)
            except Exception as exc:  # noqa: BLE001
                self._log.warning(f"[Watchdog] on_freeze callback fail: {exc}")

        loop = self._loop
        sched = self._scheduler
        if loop is None or sched is None:
            return
        # Marshal vào event loop. Có thể loop đã close (scheduler shut down) —
        # call_soon_threadsafe sẽ raise RuntimeError; ta swallow.
        try:
            loop.call_soon_threadsafe(sched.cancel_task, task.page_id)
        except RuntimeError as exc:
            # Loop closed — kệ, scheduler dọn rồi.
            self._log.debug(f"[Watchdog] cancel skip (loop closed): {exc}")
