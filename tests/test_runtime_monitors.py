"""Unit tests for monitors + mutex + rate limiter + crash handler (Commit 3)."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import numpy as np
import pytest

from mangatrans.runtime.crash_handler import CrashHandler
from mangatrans.runtime.gpu_mutex import GPUMutex
from mangatrans.runtime.memory_monitor import (
    OOM_MARKERS,
    MemoryMonitor,
    is_oom_error,
)
from mangatrans.runtime.ocr_cache import NullOCRCache, OCRCache, crop_hash
from mangatrans.runtime.rate_limiter import TokenBucket
from mangatrans.runtime.structured_log import NullStructuredLogger


# --------------------------- GPUMutex --------------------------- #

class TestGPUMutex:
    def test_serialization_basic(self):
        async def run():
            mu = GPUMutex()
            order: List[str] = []

            async def task(pid: str):
                async with mu.acquire(pid, "detect"):
                    order.append(f"enter:{pid}")
                    await asyncio.sleep(0.05)
                    order.append(f"exit:{pid}")

            await asyncio.gather(task("p1"), task("p2"))
            # Phải có pattern enter/exit không xen kẽ.
            assert len(order) == 4
            # enter -> exit -> enter -> exit
            assert order[0].startswith("enter:") and order[1].startswith("exit:")
            assert order[2].startswith("enter:") and order[3].startswith("exit:")
            # 2 page khác nhau.
            assert order[0].split(":")[1] != order[2].split(":")[1]

        asyncio.run(run())

    def test_current_holder(self):
        async def run():
            mu = GPUMutex()
            assert mu.current_holder() is None
            assert mu.is_locked() is False

            async def task():
                async with mu.acquire("pX", "ocr"):
                    assert mu.current_holder() == "pX"
                    assert mu.current_stage() == "ocr"
                    assert mu.is_locked() is True
                    await asyncio.sleep(0.02)
                assert mu.current_holder() is None
                assert mu.is_locked() is False

            await task()

        asyncio.run(run())

    def test_held_for_s(self):
        async def run():
            mu = GPUMutex()

            async def task():
                async with mu.acquire("p1", "inpaint"):
                    await asyncio.sleep(0.1)
                    held = mu.held_for_s()
                    assert held >= 0.09

            await task()

        asyncio.run(run())

    def test_release_on_exception(self):
        async def run():
            mu = GPUMutex()
            with pytest.raises(RuntimeError):
                async with mu.acquire("p1", "detect"):
                    raise RuntimeError("boom")
            # Phải release dù raise.
            assert not mu.is_locked()
            # Lần tiếp theo acquire OK.
            async with mu.acquire("p2", "detect"):
                pass

        asyncio.run(run())


# --------------------------- TokenBucket --------------------------- #

class TestTokenBucket:
    def test_initial_capacity_full(self):
        async def run():
            b = TokenBucket(rpm=60)
            # Capacity 60 → có thể acquire 60 ngay.
            for _ in range(60):
                await b.acquire(1.0)
            # Token còn ~0.
            assert b.tokens_available() < 1.0

        asyncio.run(run())

    def test_refill_over_time(self):
        async def run():
            b = TokenBucket(rpm=60, capacity=1)  # 1/giây refill
            await b.acquire(1.0)
            # Sleep 1.2s → refill ~1 token.
            await asyncio.sleep(1.2)
            assert b.tokens_available() >= 0.9

        asyncio.run(run())

    def test_blocks_until_refill(self):
        async def run():
            b = TokenBucket(rpm=120, capacity=1)  # 2/giây refill
            t0 = time.perf_counter()
            await b.acquire(1.0)
            await b.acquire(1.0)
            elapsed = time.perf_counter() - t0
            # Phải mất ít nhất ~0.4s (1 refill ~0.5s).
            assert elapsed >= 0.3

        asyncio.run(run())

    def test_invalid_rpm_raises(self):
        with pytest.raises(ValueError):
            TokenBucket(rpm=0)
        with pytest.raises(ValueError):
            TokenBucket(rpm=-1)

    def test_update_rpm_clamps(self):
        b = TokenBucket(rpm=60)
        b.update_rpm(10)
        assert b.rpm == 10
        # Capacity giảm → tokens cũng giảm.
        assert b.tokens_available() <= 10

    def test_update_rpm_min(self):
        b = TokenBucket(rpm=60)
        b.update_rpm(0)
        assert b.rpm == 1
        b.update_rpm(-5)
        assert b.rpm == 1

    def test_acquire_zero(self):
        async def run():
            b = TokenBucket(rpm=60)
            await b.acquire(0)  # no-op, không deadlock

        asyncio.run(run())


# --------------------------- MemoryMonitor --------------------------- #

class TestMemoryMonitor:
    def test_snapshot_keys(self):
        mm = MemoryMonitor()
        snap = mm.snapshot()
        # 6 keys luôn có, value có thể None.
        for k in ("process_rss_mb", "system_used_pct", "system_available_mb",
                  "vram_free_mb", "vram_total_mb", "vram_used_mb"):
            assert k in snap

    def test_cuda_empty_cache_no_crash(self):
        mm = MemoryMonitor()
        mm.cuda_empty_cache()
        mm.cuda_empty_cache()  # idempotent

    def test_should_force_cpu_none(self):
        mm = MemoryMonitor(vram_low_water_mb=500)
        assert mm.should_force_cpu({"vram_free_mb": None}) is False

    def test_should_force_cpu_low(self):
        mm = MemoryMonitor(vram_low_water_mb=500)
        assert mm.should_force_cpu({"vram_free_mb": 100}) is True

    def test_should_force_cpu_high(self):
        mm = MemoryMonitor(vram_low_water_mb=500)
        assert mm.should_force_cpu({"vram_free_mb": 2000}) is False

    def test_should_backpressure(self):
        mm = MemoryMonitor(memory_high_water_pct=80.0)
        assert mm.should_backpressure({"system_used_pct": 50.0}) is False
        assert mm.should_backpressure({"system_used_pct": 90.0}) is True
        assert mm.should_backpressure({"system_used_pct": None}) is False


class TestIsOomError:
    def test_memory_error(self):
        assert is_oom_error(MemoryError("alloc"))

    def test_cuda_runtime_oom_message(self):
        assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate ..."))

    def test_lowercase_oom(self):
        for marker in OOM_MARKERS:
            assert is_oom_error(RuntimeError(marker))

    def test_named_oom_class(self):
        class CudaOutOfMemoryError(Exception):
            pass
        assert is_oom_error(CudaOutOfMemoryError("x"))

    def test_generic_runtime_error_not_oom(self):
        assert is_oom_error(RuntimeError("file not found")) is False
        assert is_oom_error(ValueError("bad arg")) is False

    def test_none_safe(self):
        assert is_oom_error(None) is False  # type: ignore[arg-type]


# --------------------------- CrashHandler --------------------------- #

class TestCrashHandler:
    def test_install_uninstall(self, tmp_path: Path):
        logger = NullStructuredLogger()
        original_sys = sys.excepthook
        original_thr = getattr(threading, "excepthook", None)
        crash = CrashHandler(logger, str(tmp_path))
        crash.install()
        assert sys.excepthook is not original_sys
        crash.uninstall()
        assert sys.excepthook is original_sys
        if original_thr is not None:
            assert threading.excepthook is original_thr

    def test_install_idempotent(self, tmp_path: Path):
        logger = NullStructuredLogger()
        crash = CrashHandler(logger, str(tmp_path))
        crash.install()
        first = sys.excepthook
        crash.install()
        assert sys.excepthook is first
        crash.uninstall()

    def test_report_caught_logs(self, tmp_path: Path):
        logger = NullStructuredLogger(ring_size=10)
        crash = CrashHandler(logger, str(tmp_path))
        crash.report_caught("test_stage", "p1", ValueError("nope"))
        recent = logger.snapshot_recent()
        assert any(r["event"] == "caught_exception" for r in recent)
        rec = [r for r in recent if r["event"] == "caught_exception"][-1]
        assert rec["where"] == "test_stage"
        assert rec["error_type"] == "ValueError"

    def test_dump_writes_json(self, tmp_path: Path):
        logger = NullStructuredLogger(ring_size=5)
        logger.event("stage_start", stage="detect", page_id="p1")
        crash = CrashHandler(
            logger, str(tmp_path),
            memory_monitor=MemoryMonitor(),
            runtime_config_snapshot={"enable_async": True},
            pipeline_config_digest="abc123",
        )
        try:
            raise RuntimeError("boom")
        except RuntimeError as e:
            path = crash.dump("manual", type(e), e, e.__traceback__)
        assert path is not None and os.path.isfile(path)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["kind"] == "manual"
        assert payload["exc_type"] == "RuntimeError"
        assert payload["exc_msg"] == "boom"
        assert "traceback" in payload
        assert payload["pipeline_config_digest"] == "abc123"
        assert payload["runtime_config"] == {"enable_async": True}
        # Ring buffer events kèm theo.
        events = payload["recent_events"]
        assert any(r["event"] == "stage_start" for r in events)
        # Memory snapshot keys.
        assert "memory_snapshot" in payload

    def test_dump_multiple_increments_counter(self, tmp_path: Path):
        logger = NullStructuredLogger()
        crash = CrashHandler(logger, str(tmp_path))
        for _ in range(3):
            try:
                raise RuntimeError("x")
            except RuntimeError as e:
                crash.dump("k", type(e), e, e.__traceback__)
            time.sleep(0.01)
        files = sorted(p.name for p in tmp_path.iterdir()
                       if p.name.startswith(".mangatrans_crash_"))
        assert len(files) == 3
        # Counter suffix _001, _002, _003.
        nums = [f.split("_")[-1].replace(".json", "") for f in files]
        assert sorted(nums) == ["001", "002", "003"]

    def test_threading_hook(self, tmp_path: Path):
        logger = NullStructuredLogger(ring_size=10)
        crash = CrashHandler(logger, str(tmp_path))
        # Simulate args.
        try:
            raise ValueError("worker fail")
        except ValueError as e:
            args = SimpleNamespace(
                exc_type=type(e),
                exc_value=e,
                exc_traceback=e.__traceback__,
                thread=threading.current_thread(),
            )
            crash._threading_hook(args)
        events = logger.snapshot_recent()
        assert any(r["event"] == "thread_uncaught" for r in events)
        # Crash file viết.
        files = list(tmp_path.glob(".mangatrans_crash_*.json"))
        assert len(files) == 1

    def test_threading_hook_systemexit_ignored(self, tmp_path: Path):
        logger = NullStructuredLogger(ring_size=10)
        crash = CrashHandler(logger, str(tmp_path))
        args = SimpleNamespace(
            exc_type=SystemExit,
            exc_value=SystemExit(0),
            exc_traceback=None,
            thread=threading.current_thread(),
        )
        crash._threading_hook(args)
        events = logger.snapshot_recent()
        assert not any(r["event"] == "thread_uncaught" for r in events)

    def test_asyncio_handler(self, tmp_path: Path):
        logger = NullStructuredLogger(ring_size=10)
        crash = CrashHandler(logger, str(tmp_path))
        try:
            raise RuntimeError("loop fail")
        except RuntimeError as e:
            loop = asyncio.new_event_loop()
            try:
                crash.asyncio_handler(loop, {"message": "task failed", "exception": e})
            finally:
                loop.close()
        events = logger.snapshot_recent()
        assert any(r["event"] == "asyncio_exception" for r in events)
        files = list(tmp_path.glob(".mangatrans_crash_*.json"))
        assert len(files) == 1


# --------------------------- OCRCache --------------------------- #

class TestCropHash:
    def test_stable(self):
        a = np.zeros((10, 10, 3), dtype=np.uint8)
        b = np.zeros((10, 10, 3), dtype=np.uint8)
        assert crop_hash(a) == crop_hash(b)

    def test_changes_on_data(self):
        a = np.zeros((10, 10, 3), dtype=np.uint8)
        b = np.ones((10, 10, 3), dtype=np.uint8)
        assert crop_hash(a) != crop_hash(b)

    def test_changes_on_shape(self):
        a = np.zeros((10, 10, 3), dtype=np.uint8)
        c = np.zeros((10, 11, 3), dtype=np.uint8)
        assert crop_hash(a) != crop_hash(c)


class TestOCRCache:
    def test_get_miss(self, tmp_path: Path):
        c = OCRCache(str(tmp_path / "oc.json"))
        assert c.get("x") is None
        assert c.stats()["misses"] == 1

    def test_put_get(self, tmp_path: Path):
        c = OCRCache(str(tmp_path / "oc.json"))
        c.put("k1", {"text": "hello"})
        assert c.get("k1") == {"text": "hello"}

    def test_persistence(self, tmp_path: Path):
        p = str(tmp_path / "oc.json")
        c1 = OCRCache(p, debounce_s=0.0)
        c1.put("k", {"text": "hi", "score": 0.9})
        c1.flush()
        c2 = OCRCache(p)
        assert c2.get("k") == {"text": "hi", "score": 0.9}

    def test_eviction(self, tmp_path: Path):
        c = OCRCache(str(tmp_path / "oc.json"), max_entries=10, debounce_s=0.0)
        for i in range(20):
            c.put(f"k{i}", {"i": i})
        # After eviction.
        assert len(c._cache) <= 10


class TestNullOCRCache:
    def test_always_miss(self):
        c = NullOCRCache()
        c.put("k", {"text": "x"})
        assert c.get("k") is None
