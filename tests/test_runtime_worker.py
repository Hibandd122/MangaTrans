"""Unit tests for translation_worker + watchdog (Commit 5)."""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from mangatrans.config import TranslateConfig
from mangatrans.runtime.config import RuntimeConfig
from mangatrans.runtime.page_task import PageState, PageTask, StageName
from mangatrans.runtime.translation_cache import (
    NullTranslationCache,
    TranslationCache,
)
from mangatrans.runtime.translation_worker import (
    RateLimitedTranslator,
    TranslateRequest,
    is_rate_limit_error,
)
from mangatrans.runtime.watchdog import Watchdog


# --------------------------- Fake Translator --------------------------- #

class FakeTranslator:
    """Mock Translator giả lập translate_batch — không network."""

    def __init__(
        self,
        target_lang: str = "vi",
        model: str = "fake/model",
        fail_n_times: int = 0,
        fail_with: Exception = None,
        delay_s: float = 0.0,
    ):
        self.config = TranslateConfig(target_lang=target_lang)
        self._model = model
        self.fail_n_times = fail_n_times
        self.fail_with = fail_with or RuntimeError("OpenRouter API HTTP 429: quota")
        self.delay_s = delay_s
        self.calls: List[tuple] = []
        self._lock = threading.Lock()
        self._fail_remaining = fail_n_times
        self.glossary = {}

    def resolve_model(self) -> str:
        return self._model

    def translate_batch(
        self, texts: List[str], position_tags: Optional[List[str]] = None
    ) -> List[str]:
        with self._lock:
            self.calls.append((tuple(texts), tuple(position_tags) if position_tags else None))
            if self._fail_remaining > 0:
                self._fail_remaining -= 1
                raise self.fail_with
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        return [f"VI:{t}" for t in texts]


# --------------------------- is_rate_limit_error --------------------------- #

class TestIsRateLimitError:
    def test_429_message(self):
        assert is_rate_limit_error(RuntimeError("OpenRouter API HTTP 429: quota"))

    def test_rate_limit_message(self):
        assert is_rate_limit_error(RuntimeError("Rate-limit exceeded"))

    def test_too_many_requests(self):
        assert is_rate_limit_error(RuntimeError("HTTP 429 Too Many Requests"))

    def test_quota(self):
        assert is_rate_limit_error(RuntimeError("monthly quota exceeded"))

    def test_retry_delay_json(self):
        assert is_rate_limit_error(RuntimeError('{"retryDelay":"12s"}'))

    def test_generic_runtime_not_match(self):
        assert is_rate_limit_error(RuntimeError("file not found")) is False

    def test_none_safe(self):
        assert is_rate_limit_error(None) is False  # type: ignore[arg-type]


# --------------------------- TranslateRequest --------------------------- #

class TestTranslateRequest:
    def test_basic(self):
        req = TranslateRequest(texts=["hi"], position_tags=["top-left"])
        assert req.texts == ["hi"]
        assert req.position_tags == ["top-left"]

    def test_no_tags(self):
        req = TranslateRequest(texts=["a", "b"])
        assert req.position_tags is None


# --------------------------- RateLimitedTranslator --------------------------- #

def _build_worker(
    translator: FakeTranslator,
    cache=None,
    rpm: int = 600,
    concurrency: int = 4,
    max_retries: int = 2,
    initial_backoff: float = 0.01,
    backoff_factor: float = 1.0,
    jitter: float = 0.0,
):
    cfg = RuntimeConfig(
        translation_rpm=rpm,
        translation_concurrency=concurrency,
        translation_max_retries=max_retries,
        translation_initial_backoff=initial_backoff,
        translation_backoff_factor=backoff_factor,
        translation_jitter_ratio=jitter,
    )
    executor = ThreadPoolExecutor(max_workers=concurrency)
    cache = cache if cache is not None else NullTranslationCache()
    return RateLimitedTranslator(translator, cfg, cache, executor), executor


class TestRateLimitedTranslatorHappyPath:
    def test_empty_texts(self):
        async def run():
            tr = FakeTranslator()
            worker, ex = _build_worker(tr)
            try:
                out = await worker.translate_batch(TranslateRequest(texts=[]))
                assert out == []
            finally:
                ex.shutdown(wait=True)

        asyncio.run(run())

    def test_basic_translation(self):
        async def run():
            tr = FakeTranslator()
            worker, ex = _build_worker(tr)
            try:
                req = TranslateRequest(texts=["こんにちは"])
                out = await worker.translate_batch(req)
                assert out == ["VI:こんにちは"]
                assert len(tr.calls) == 1
            finally:
                ex.shutdown(wait=True)

        asyncio.run(run())

    def test_multi_text(self):
        async def run():
            tr = FakeTranslator()
            worker, ex = _build_worker(tr)
            try:
                req = TranslateRequest(
                    texts=["a", "b", "c"],
                    position_tags=["top-left", "mid-center", "bot-right"],
                )
                out = await worker.translate_batch(req)
                assert out == ["VI:a", "VI:b", "VI:c"]
            finally:
                ex.shutdown(wait=True)

        asyncio.run(run())


class TestRateLimitedTranslatorCache:
    def test_cache_full_hit(self, tmp_path: Path):
        async def run():
            tr = FakeTranslator()
            cache = TranslationCache(str(tmp_path / "tc.json"), debounce_s=0.0)
            # Prime cache.
            cache.put(tr.resolve_model(), tr.config.target_lang, "hi", "PRE:hi")
            worker, ex = _build_worker(tr, cache=cache)
            try:
                out = await worker.translate_batch(TranslateRequest(texts=["hi"]))
                assert out == ["PRE:hi"]
                # Zero API call.
                assert len(tr.calls) == 0
                assert worker.stats["cache_hits"] == 1
                assert worker.stats["calls"] == 0
            finally:
                ex.shutdown(wait=True)

        asyncio.run(run())

    def test_cache_partial(self, tmp_path: Path):
        async def run():
            tr = FakeTranslator()
            cache = TranslationCache(str(tmp_path / "tc.json"), debounce_s=0.0)
            cache.put(tr.resolve_model(), tr.config.target_lang, "a", "PRE:a")
            worker, ex = _build_worker(tr, cache=cache)
            try:
                out = await worker.translate_batch(
                    TranslateRequest(texts=["a", "b", "c"])
                )
                # 'a' hit, 'b'+'c' miss → API call với chỉ 2 texts.
                assert out == ["PRE:a", "VI:b", "VI:c"]
                assert len(tr.calls) == 1
                assert tr.calls[0][0] == ("b", "c")
                assert worker.stats["cache_hits"] == 1
                assert worker.stats["cache_misses"] == 2
            finally:
                ex.shutdown(wait=True)

        asyncio.run(run())

    def test_cache_put_after_translate(self, tmp_path: Path):
        async def run():
            tr = FakeTranslator()
            cache = TranslationCache(str(tmp_path / "tc.json"), debounce_s=0.0)
            worker, ex = _build_worker(tr, cache=cache)
            try:
                await worker.translate_batch(TranslateRequest(texts=["x"]))
                # Second call → cache hit.
                out2 = await worker.translate_batch(TranslateRequest(texts=["x"]))
                assert out2 == ["VI:x"]
                assert len(tr.calls) == 1
            finally:
                ex.shutdown(wait=True)

        asyncio.run(run())


class TestRateLimitedTranslatorRetry:
    def test_retry_then_success(self):
        async def run():
            tr = FakeTranslator(fail_n_times=2)
            worker, ex = _build_worker(tr, max_retries=3, initial_backoff=0.01)
            try:
                out = await worker.translate_batch(TranslateRequest(texts=["a"]))
                assert out == ["VI:a"]
                # 2 fail + 1 success = 3 calls.
                assert len(tr.calls) == 3
                assert worker.stats["retries"] == 2
                assert worker.stats["rate_limit_hits"] == 2
            finally:
                ex.shutdown(wait=True)

        asyncio.run(run())

    def test_retry_exhaust_raises(self):
        async def run():
            tr = FakeTranslator(fail_n_times=99)
            worker, ex = _build_worker(tr, max_retries=2, initial_backoff=0.01)
            try:
                with pytest.raises(RuntimeError):
                    await worker.translate_batch(TranslateRequest(texts=["a"]))
                # 1 initial + 2 retries = 3 calls.
                assert len(tr.calls) == 3
                assert worker.stats["failures"] == 1
            finally:
                ex.shutdown(wait=True)

        asyncio.run(run())

    def test_429_halves_rpm(self):
        async def run():
            tr = FakeTranslator(fail_n_times=1)
            worker, ex = _build_worker(
                tr, rpm=20, max_retries=2, initial_backoff=0.01
            )
            try:
                start_rpm = worker._bucket.rpm
                await worker.translate_batch(TranslateRequest(texts=["a"]))
                # Sau 1 lần 429, RPM phải bị halve.
                assert worker._bucket.rpm < start_rpm
                assert worker._bucket.rpm == max(
                    worker.cfg.translation_rpm_min, start_rpm // 2
                )
            finally:
                ex.shutdown(wait=True)

        asyncio.run(run())

    def test_non_rate_limit_error_not_halve(self):
        async def run():
            tr = FakeTranslator(
                fail_n_times=1,
                fail_with=RuntimeError("parse error"),
            )
            worker, ex = _build_worker(tr, rpm=20, max_retries=2, initial_backoff=0.01)
            try:
                start_rpm = worker._bucket.rpm
                await worker.translate_batch(TranslateRequest(texts=["a"]))
                # Không 429 → RPM giữ nguyên.
                assert worker._bucket.rpm == start_rpm
            finally:
                ex.shutdown(wait=True)

        asyncio.run(run())


class TestRateLimitedTranslatorClose:
    def test_close_flushes_cache(self, tmp_path: Path):
        async def run():
            tr = FakeTranslator()
            cache = TranslationCache(str(tmp_path / "tc.json"), debounce_s=99.0)
            worker, ex = _build_worker(tr, cache=cache)
            try:
                await worker.translate_batch(TranslateRequest(texts=["a"]))
                worker.close()
                # File flushed.
                assert (tmp_path / "tc.json").exists()
            finally:
                ex.shutdown(wait=True)

        asyncio.run(run())


class TestRateLimitedTranslatorConcurrency:
    def test_concurrent_requests_serialized_by_glossary_lock(self):
        async def run():
            tr = FakeTranslator(delay_s=0.05)
            worker, ex = _build_worker(tr, concurrency=4)
            try:
                t0 = time.perf_counter()
                # 4 calls đồng thời. Lock serialize → 0.05s × 4 ~= 0.2s.
                results = await asyncio.gather(
                    *[
                        worker.translate_batch(TranslateRequest(texts=[f"t{i}"]))
                        for i in range(4)
                    ]
                )
                elapsed = time.perf_counter() - t0
                assert all(len(r) == 1 for r in results)
                # Phải >= ~0.15s (serialized) — nếu parallel sẽ ~0.06s.
                assert elapsed >= 0.15
            finally:
                ex.shutdown(wait=True)

        asyncio.run(run())


# --------------------------- Watchdog --------------------------- #

class FakeScheduler:
    """Mock Scheduler cho watchdog test."""

    def __init__(self):
        self.cancels: List[str] = []
        self._lock = threading.Lock()

    def cancel_task(self, page_id: str) -> None:
        with self._lock:
            self.cancels.append(page_id)


class TestWatchdog:
    def test_disabled_no_thread(self):
        cfg = RuntimeConfig(watchdog_enable=False)
        wd = Watchdog(cfg)
        loop = asyncio.new_event_loop()
        try:
            wd.start(loop, FakeScheduler())
            assert wd.is_running is False
        finally:
            loop.close()
            wd.stop()

    def test_start_stop_idempotent(self):
        cfg = RuntimeConfig(watchdog_poll_interval_s=0.05)
        wd = Watchdog(cfg)
        loop = asyncio.new_event_loop()
        try:
            wd.start(loop, FakeScheduler())
            assert wd.is_running is True
            wd.start(loop, FakeScheduler())  # idempotent
            assert wd.is_running is True
            wd.stop(timeout=2.0)
            assert wd.is_running is False
            wd.stop(timeout=2.0)  # idempotent
        finally:
            loop.close()

    def test_register_unregister(self):
        cfg = RuntimeConfig(watchdog_poll_interval_s=0.05)
        wd = Watchdog(cfg)
        task = PageTask.new("/tmp/a.png", "/tmp/a_out.png")
        wd.register(task)
        assert task.page_id in wd.tracked_page_ids()
        wd.unregister(task.page_id)
        assert task.page_id not in wd.tracked_page_ids()

    def test_heartbeat_resets_timer(self):
        cfg = RuntimeConfig(watchdog_poll_interval_s=0.05)
        wd = Watchdog(cfg)
        task = PageTask.new("/tmp/a.png", "/tmp/a_out.png")
        wd.register(task)
        first = task.last_heartbeat_ts
        time.sleep(0.02)
        wd.heartbeat(task.page_id, StageName.DETECT)
        assert task.last_heartbeat_ts > first
        assert task.current_stage == StageName.DETECT

    def test_freeze_triggers_cancel(self):
        cfg = RuntimeConfig(
            watchdog_poll_interval_s=0.05,
            watchdog_stage_timeout_s={"detect": 0.1},
        )
        wd = Watchdog(cfg)
        sched = FakeScheduler()

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        try:
            wd.start(loop, sched)
            task = PageTask.new("/tmp/a.png", "/tmp/a_out.png")
            wd.register(task)
            wd.heartbeat(task.page_id, StageName.DETECT)
            # Đợi vượt timeout + 1 poll cycle.
            time.sleep(0.4)
            assert task.cancel_requested is True
            assert wd.stats()["freezes_detected"] >= 1
            # Loop có thể đã nhận call_soon_threadsafe.
            time.sleep(0.1)
            assert task.page_id in sched.cancels
        finally:
            wd.stop(timeout=2.0)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2.0)
            loop.close()

    def test_no_freeze_within_timeout(self):
        cfg = RuntimeConfig(
            watchdog_poll_interval_s=0.05,
            watchdog_stage_timeout_s={"detect": 2.0},
        )
        wd = Watchdog(cfg)
        sched = FakeScheduler()
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        try:
            wd.start(loop, sched)
            task = PageTask.new("/tmp/a.png", "/tmp/a_out.png")
            wd.register(task)
            wd.heartbeat(task.page_id, StageName.DETECT)
            time.sleep(0.2)
            assert task.cancel_requested is False
            assert sched.cancels == []
        finally:
            wd.stop(timeout=2.0)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2.0)
            loop.close()

    def test_on_freeze_callback_invoked(self):
        cfg = RuntimeConfig(
            watchdog_poll_interval_s=0.05,
            watchdog_stage_timeout_s={"detect": 0.1},
        )
        wd = Watchdog(cfg)
        sched = FakeScheduler()
        called: List[tuple] = []

        def on_freeze(t, stage, elapsed):
            called.append((t.page_id, stage, elapsed))

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        try:
            wd.start(loop, sched, on_freeze=on_freeze)
            task = PageTask.new("/tmp/a.png", "/tmp/a_out.png")
            wd.register(task)
            wd.heartbeat(task.page_id, StageName.DETECT)
            time.sleep(0.4)
            assert len(called) >= 1
            assert called[0][0] == task.page_id
            assert called[0][1] == "detect"
        finally:
            wd.stop(timeout=2.0)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2.0)
            loop.close()

    def test_already_cancelled_not_recancelled(self):
        cfg = RuntimeConfig(
            watchdog_poll_interval_s=0.05,
            watchdog_stage_timeout_s={"detect": 0.05},
        )
        wd = Watchdog(cfg)
        sched = FakeScheduler()
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        try:
            wd.start(loop, sched)
            task = PageTask.new("/tmp/a.png", "/tmp/a_out.png")
            wd.register(task)
            wd.heartbeat(task.page_id, StageName.DETECT)
            # 2 cycle để chắc chắn pass timeout.
            time.sleep(0.3)
            first_count = wd.stats()["cancels_issued"]
            # Thêm 2 cycle nữa — không tăng vì đã cancel_requested.
            time.sleep(0.2)
            assert wd.stats()["cancels_issued"] == first_count
        finally:
            wd.stop(timeout=2.0)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2.0)
            loop.close()

    def test_no_stage_no_check(self):
        cfg = RuntimeConfig(
            watchdog_poll_interval_s=0.05,
            watchdog_stage_timeout_s={"detect": 0.05},
        )
        wd = Watchdog(cfg)
        sched = FakeScheduler()
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        try:
            wd.start(loop, sched)
            task = PageTask.new("/tmp/a.png", "/tmp/a_out.png")
            wd.register(task)
            # KHÔNG gọi heartbeat → current_stage=None → skip check.
            time.sleep(0.2)
            assert task.cancel_requested is False
        finally:
            wd.stop(timeout=2.0)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2.0)
            loop.close()
