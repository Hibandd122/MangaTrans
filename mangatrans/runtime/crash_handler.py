"""CrashHandler — install sys/threading/asyncio excepthook, dump crash JSON.

Mục tiêu: không bao giờ silent-die. Mọi exception thoát khỏi pipeline (uncaught
trong thread, asyncio task) phải:
1. Ghi structured event `uncaught_exception`.
2. Dump crash report JSON kèm last 30 events + memory snapshot + runtime cfg.
3. Để Python tự exit theo default (không suppress).

Caller install 1 lần khi tạo ChapterRunner; uninstall khi runner kết thúc để
test isolation hoạt động.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .checkpoint import atomic_write_json
from .memory_monitor import MemoryMonitor
from .structured_log import StructuredLogger


def _utc_iso_compact() -> str:
    """UTC timestamp safe cho filename (no `:`)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class CrashHandler:
    """Install excepthook 3-way: sys, threading, asyncio.

    Lưu ý:
    - asyncio handler được scheduler set trực tiếp vào loop (không global).
    - sys.excepthook trigger khi REPL/main thread chết uncaught.
    - threading.excepthook trigger trong worker thread của ThreadPoolExecutor.
    """

    def __init__(
        self,
        logger: StructuredLogger,
        crash_dir: str,
        memory_monitor: Optional[MemoryMonitor] = None,
        runtime_config_snapshot: Optional[Dict[str, Any]] = None,
        pipeline_config_digest: Optional[str] = None,
        context_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        self.logger = logger
        self.crash_dir = crash_dir
        self.memory_monitor = memory_monitor
        self.runtime_config_snapshot = runtime_config_snapshot or {}
        self.pipeline_config_digest = pipeline_config_digest
        self.context_provider = context_provider

        self._prev_sys: Optional[Callable] = None
        self._prev_threading: Optional[Callable] = None
        self._installed = False
        self._lock = threading.Lock()
        self._dump_count = 0

        os.makedirs(self.crash_dir, exist_ok=True)

    # ---- Install / uninstall ----

    def install(self) -> None:
        with self._lock:
            if self._installed:
                return
            self._prev_sys = sys.excepthook
            sys.excepthook = self._sys_hook  # type: ignore[assignment]
            self._prev_threading = getattr(threading, "excepthook", None)
            if hasattr(threading, "excepthook"):
                threading.excepthook = self._threading_hook  # type: ignore[assignment]
            self._installed = True

    def uninstall(self) -> None:
        with self._lock:
            if not self._installed:
                return
            try:
                sys.excepthook = self._prev_sys or sys.__excepthook__
            except Exception:  # noqa: BLE001
                pass
            try:
                if self._prev_threading is not None and hasattr(threading, "excepthook"):
                    threading.excepthook = self._prev_threading  # type: ignore[assignment]
            except Exception:  # noqa: BLE001
                pass
            self._installed = False

    # ---- Reported errors (caught) ----

    def report_caught(self, where: str, page_id: Optional[str],
                      exc: BaseException) -> None:
        """Caller tự catch nhưng muốn record."""
        self.logger.event(
            "caught_exception",
            where=where,
            page_id=page_id,
            error_type=type(exc).__name__,
            error_msg=str(exc)[:500],
        )

    # ---- Manual dump (asyncio loop hooks) ----

    def dump(self, kind: str, exc_type: type, exc_value: BaseException,
             tb: Any) -> Optional[str]:
        """Write crash JSON; trả về path nếu thành công."""
        try:
            tb_text = "".join(traceback.format_exception(exc_type, exc_value, tb))
        except Exception:  # noqa: BLE001
            tb_text = repr(exc_value)
        mem_snap: Dict[str, Any] = {}
        try:
            if self.memory_monitor is not None:
                mem_snap = self.memory_monitor.snapshot()
        except Exception:  # noqa: BLE001
            mem_snap = {"error": "monitor failed"}
        try:
            ctx = self.context_provider() if self.context_provider else {}
        except Exception:  # noqa: BLE001
            ctx = {}
        with self._lock:
            self._dump_count += 1
            n = self._dump_count
        payload = {
            "kind": kind,
            "pid": os.getpid(),
            "exc_type": getattr(exc_type, "__name__", repr(exc_type)),
            "exc_msg": str(exc_value)[:500],
            "traceback": tb_text[:8000],
            "memory_snapshot": mem_snap,
            "runtime_config": self.runtime_config_snapshot,
            "pipeline_config_digest": self.pipeline_config_digest,
            "recent_events": self.logger.snapshot_recent(),
            "context": ctx,
        }
        filename = f".mangatrans_crash_{_utc_iso_compact()}_{n:03d}.json"
        path = os.path.join(self.crash_dir, filename)
        try:
            atomic_write_json(path, payload)
        except OSError as exc:
            sys.stderr.write(f"[CrashHandler] dump failed: {exc}\n")
            return None
        # Log event tham chiếu file.
        try:
            self.logger.event("crash_dump_written", path=path, kind=kind)
        except Exception:  # noqa: BLE001
            pass
        return path

    # ---- Hooks ----

    def _sys_hook(self, exc_type: type, exc_value: BaseException, tb: Any) -> None:
        # Ignore KeyboardInterrupt — đó là user cancel, không phải bug.
        if issubclass(exc_type, KeyboardInterrupt):
            if self._prev_sys is not None:
                try:
                    self._prev_sys(exc_type, exc_value, tb)
                except Exception:  # noqa: BLE001
                    pass
            return
        self.dump("sys_excepthook", exc_type, exc_value, tb)
        if self._prev_sys is not None:
            try:
                self._prev_sys(exc_type, exc_value, tb)
            except Exception:  # noqa: BLE001
                pass

    def _threading_hook(self, args: Any) -> None:
        # `args` là threading.ExceptHookArgs (NamedTuple).
        exc_type = getattr(args, "exc_type", None)
        exc_value = getattr(args, "exc_value", None)
        tb = getattr(args, "exc_traceback", None)
        thread = getattr(args, "thread", None)
        if exc_type is None or issubclass(exc_type, SystemExit):
            return
        self.logger.event(
            "thread_uncaught",
            thread=getattr(thread, "name", None),
            error_type=getattr(exc_type, "__name__", repr(exc_type)),
            error_msg=str(exc_value)[:500] if exc_value else "",
        )
        if exc_value is not None:
            self.dump("thread_excepthook", exc_type, exc_value, tb)
        # Chain về previous hook (default sys.stderr print).
        if self._prev_threading is not None:
            try:
                self._prev_threading(args)
            except Exception:  # noqa: BLE001
                pass

    def asyncio_handler(self, loop: Any, context: Dict[str, Any]) -> None:
        """Set qua `loop.set_exception_handler(crash.asyncio_handler)`."""
        msg = context.get("message", "")
        exc = context.get("exception")
        self.logger.event(
            "asyncio_exception",
            message=str(msg)[:200],
            error_type=type(exc).__name__ if exc else None,
            error_msg=str(exc)[:500] if exc else None,
        )
        if exc is not None:
            self.dump(
                "asyncio_handler", type(exc), exc, exc.__traceback__,
            )
