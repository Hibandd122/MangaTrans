"""MetricsRegistry — counter/gauge nhẹ + health printer 30s.

Không dùng prometheus_client vì thừa cho local CLI. Chỉ cần in-memory dict +
print định kỳ ra stdout. Thread-safe.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional


class MetricsRegistry:
    """Counter (monotonic) + gauge (latest value). Thread-safe."""

    def __init__(self) -> None:
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()

    def incr(self, name: str, by: float = 1.0) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + float(by)

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "uptime_s": time.perf_counter() - self._t0,
            }

    # ---- Health printer ----

    def make_health_printer(
        self,
        interval_s: float = 30.0,
        printer: Optional[Callable[[str], None]] = None,
    ) -> "HealthPrinter":
        return HealthPrinter(self, interval_s, printer)


class HealthPrinter:
    """Daemon thread in metrics snapshot mỗi N giây ra stdout (hoặc custom sink).

    Idempotent stop(). Không raise nếu sink fail.
    """

    def __init__(
        self,
        registry: MetricsRegistry,
        interval_s: float,
        printer: Optional[Callable[[str], None]],
    ):
        self.registry = registry
        # Floor 0.01s để test nhanh; production user dùng default 30s qua RuntimeConfig.
        self.interval_s = float(max(0.01, interval_s))
        self._printer = printer or (lambda s: print(s, flush=True))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="MangaTrans-HealthPrinter", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                snap = self.registry.snapshot()
                msg = self._format(snap)
                self._printer(msg)
            except Exception:  # noqa: BLE001 - printer không được kill thread
                pass

    @staticmethod
    def _format(snap: Dict[str, Dict[str, float]]) -> str:
        up = snap.get("uptime_s", 0.0)
        cnt = snap.get("counters", {})
        gau = snap.get("gauges", {})
        parts = [f"[health uptime={up:.0f}s]"]
        if cnt:
            top = sorted(cnt.items())[:8]
            parts.append("counters=" + ",".join(f"{k}={int(v) if v == int(v) else round(v, 2)}" for k, v in top))
        if gau:
            top = sorted(gau.items())[:8]
            parts.append("gauges=" + ",".join(f"{k}={round(v, 2)}" for k, v in top))
        return " ".join(parts)
