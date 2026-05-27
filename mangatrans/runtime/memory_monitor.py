"""MemoryMonitor — psutil RAM + torch CUDA VRAM gauge.

Degrade gracefully nếu psutil/torch không sẵn:
- psutil missing → RAM snapshot trả về None, force_backpressure False.
- torch.cuda missing / CPU-only → VRAM snapshot trả về None, force_cpu False.

Public API:
- `snapshot()` → dict các gauge (process_rss_mb, system_used_pct, vram_free_mb,
  vram_total_mb, vram_used_mb).
- `should_force_cpu()` → True khi VRAM free < `low_water_mb`.
- `should_backpressure()` → True khi system memory used > high_water_pct.
- `is_oom_error(exc)` → matcher cho CUDA OOM (cross-platform).
- `cuda_empty_cache()` → no-op nếu không có torch/CUDA.
"""
from __future__ import annotations

import gc
import os
import sys
from typing import Any, Dict, Optional

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False


# Lazy torch import — chỉ load khi cần (cold start nặng).
def _try_import_torch():  # pragma: no cover - thin shim
    try:
        import torch  # type: ignore
        return torch
    except ImportError:
        return None


OOM_MARKERS = (
    "out of memory",
    "cuda out of memory",
    "cudaerrormemoryallocation",
    "alloc_failed",
    "outofmemory",
    "cuda_error_out_of_memory",
)


def is_oom_error(exc: BaseException) -> bool:
    """Cross-platform CUDA OOM matcher.

    Bắt cả torch.cuda.OutOfMemoryError, RuntimeError("CUDA out of memory"),
    onnxruntime OrtException OOM, MemoryError. Không bắt OSError generic.
    """
    if exc is None:
        return False
    # MemoryError thuần (CPU RAM cạn) cũng coi như OOM → fallback CPU không giúp,
    # nhưng caller có thể decide reduce depth.
    if isinstance(exc, MemoryError):
        return True
    name = type(exc).__name__.lower()
    if "outofmemory" in name or "oom" in name:
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in OOM_MARKERS)


class MemoryMonitor:
    """Read-only snapshot. Không spawn thread — caller poll khi cần."""

    def __init__(
        self,
        vram_low_water_mb: int = 500,
        vram_high_water_mb: int = 3500,
        memory_high_water_pct: float = 88.0,
        cuda_device: int = 0,
    ):
        self.vram_low_water_mb = int(vram_low_water_mb)
        self.vram_high_water_mb = int(vram_high_water_mb)
        self.memory_high_water_pct = float(memory_high_water_pct)
        self.cuda_device = int(cuda_device)
        self._torch = None
        self._cuda_available: Optional[bool] = None
        self._proc = psutil.Process(os.getpid()) if _HAS_PSUTIL else None

    # ---- Snapshot ----

    def snapshot(self) -> Dict[str, Any]:
        """Snapshot mọi gauge. Field None nếu không đo được."""
        snap: Dict[str, Any] = {}
        if _HAS_PSUTIL and self._proc is not None:
            try:
                rss_mb = self._proc.memory_info().rss / (1024 * 1024)
                snap["process_rss_mb"] = round(rss_mb, 1)
            except (psutil.Error, OSError):
                snap["process_rss_mb"] = None
            try:
                vm = psutil.virtual_memory()
                snap["system_used_pct"] = round(vm.percent, 1)
                snap["system_available_mb"] = round(vm.available / (1024 * 1024), 1)
            except (psutil.Error, OSError):
                snap["system_used_pct"] = None
                snap["system_available_mb"] = None
        else:
            snap["process_rss_mb"] = None
            snap["system_used_pct"] = None
            snap["system_available_mb"] = None

        vram = self._vram_snapshot()
        snap.update(vram)
        return snap

    def _vram_snapshot(self) -> Dict[str, Any]:
        if not self._cuda_check():
            return {
                "vram_free_mb": None,
                "vram_total_mb": None,
                "vram_used_mb": None,
            }
        try:
            free, total = self._torch.cuda.mem_get_info(self.cuda_device)
            free_mb = free / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            return {
                "vram_free_mb": round(free_mb, 1),
                "vram_total_mb": round(total_mb, 1),
                "vram_used_mb": round(total_mb - free_mb, 1),
            }
        except Exception:  # noqa: BLE001 - mọi error torch → fall-soft None
            return {
                "vram_free_mb": None,
                "vram_total_mb": None,
                "vram_used_mb": None,
            }

    def _cuda_check(self) -> bool:
        if self._cuda_available is not None:
            return self._cuda_available
        torch = _try_import_torch()
        self._torch = torch
        if torch is None:
            self._cuda_available = False
            return False
        try:
            self._cuda_available = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            self._cuda_available = False
        return self._cuda_available

    # ---- Decisions ----

    def should_force_cpu(self, snap: Optional[Dict[str, Any]] = None) -> bool:
        """True nếu VRAM free dưới low_water_mb → đẩy stage GPU sang CPU."""
        if snap is None:
            snap = self.snapshot()
        free = snap.get("vram_free_mb")
        if free is None:
            return False
        return free < self.vram_low_water_mb

    def should_backpressure(self, snap: Optional[Dict[str, Any]] = None) -> bool:
        """True nếu RAM hệ thống dùng > high_water_pct → giảm pipeline depth."""
        if snap is None:
            snap = self.snapshot()
        pct = snap.get("system_used_pct")
        if pct is None:
            return False
        return pct >= self.memory_high_water_pct

    # ---- Side-effect helpers ----

    def cuda_empty_cache(self) -> None:
        """Best-effort gc + torch.cuda.empty_cache. Không raise."""
        gc.collect()
        if not self._cuda_check():
            return
        try:
            self._torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def is_oom_error(exc: BaseException) -> bool:
        """Static alias (caller có thể không cần instance)."""
        return is_oom_error(exc)
