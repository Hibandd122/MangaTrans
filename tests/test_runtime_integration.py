"""Integration tests cho Scheduler + ChapterRunner (Commit 6).

Mock toàn bộ MangaPipeline stage_* để KHÔNG load model. Verify async DAG,
GPU mutex serialization, watchdog cancel, OOM CPU fallback, resume, cache.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from mangatrans.config import PipelineConfig
from mangatrans.runtime.checkpoint import CheckpointStore
from mangatrans.runtime.config import RuntimeConfig
from mangatrans.runtime.gpu_mutex import GPUMutex
from mangatrans.runtime.memory_monitor import MemoryMonitor
from mangatrans.runtime.metrics import MetricsRegistry
from mangatrans.runtime.page_task import PageState, PageTask, StageName
from mangatrans.runtime.scheduler import (
    PageCancelled,
    Scheduler,
    StageFatal,
    StageRetry,
)
from mangatrans.runtime.structured_log import NullStructuredLogger
from mangatrans.runtime.translation_cache import NullTranslationCache, TranslationCache
from mangatrans.runtime.translation_worker import RateLimitedTranslator
from mangatrans.runtime.watchdog import Watchdog


# --------------------------- Fake pipeline ---------------------------- #

class FakeTranslator:
    """Mock Translator giả lập translate_batch."""

    class _Cfg:
        target_lang = "vi"

    def __init__(self):
        self.config = self._Cfg()
        self.glossary = {}

    def resolve_model(self) -> str:
        return "fake/model"

    def translate_batch(self, texts, position_tags=None):
        return [f"vi:{t}" for t in texts]

    def attach_glossary(self, path):
        pass

    def save_glossary(self):
        pass


@dataclass
class FakePipeline:
    """Mock MangaPipeline cho scheduler. Mỗi stage_* tăng counter + delay.

    Cho phép `fail_at` (stage_name → list exception cần raise đến hết list).
    """

    stage_delays: Dict[str, float] = field(default_factory=dict)
    stage_fail_queue: Dict[str, List[BaseException]] = field(default_factory=dict)
    raise_translation_errors: bool = True
    translator: Optional[Any] = None
    config: Optional[Any] = None
    enter_order: List[str] = field(default_factory=list)
    exit_order: List[str] = field(default_factory=list)
    gpu_concurrent_peak: int = 0
    _gpu_active: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    save_calls: List[str] = field(default_factory=list)

    GPU_STAGES = {"detect", "lang_detect", "ocr", "inpaint"}

    def __post_init__(self):
        if self.translator is None:
            self.translator = FakeTranslator()
        if self.config is None:
            self.config = PipelineConfig()

    def new_context(self) -> Dict[str, Any]:
        return {
            "image": None, "h": 100, "w": 100, "text_mask": None, "blocks": [],
            "detection": None, "ocr_results": [], "sfx_profiles": [],
            "blocks_to_clean": [], "mask_for_inpaint": None, "image_filled": None,
            "mask_for_lama": None, "result_image": None, "summary": {},
        }

    def _run_stage(self, name: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self.enter_order.append(name)
            if name in self.GPU_STAGES:
                self._gpu_active += 1
                self.gpu_concurrent_peak = max(self.gpu_concurrent_peak, self._gpu_active)

        # Mock fail.
        queue = self.stage_fail_queue.get(name)
        if queue:
            with self._lock:
                exc = queue.pop(0)
            try:
                raise exc
            finally:
                with self._lock:
                    if name in self.GPU_STAGES:
                        self._gpu_active -= 1
                    self.exit_order.append(name)

        delay = self.stage_delays.get(name, 0.0)
        if delay > 0:
            time.sleep(delay)

        with self._lock:
            if name in self.GPU_STAGES:
                self._gpu_active -= 1
            self.exit_order.append(name)
        return ctx

    def stage_load(self, ctx, input_path):
        return self._run_stage("load", ctx)

    def stage_detect(self, ctx):
        return self._run_stage("detect", ctx)

    def stage_lang_detect(self, ctx):
        return self._run_stage("lang_detect", ctx)

    def stage_ocr(self, ctx):
        return self._run_stage("ocr", ctx)

    def stage_sfx(self, ctx):
        return self._run_stage("sfx", ctx)

    def stage_translate(self, ctx, output_path):
        return self._run_stage("translate", ctx)

    def stage_save_json(self, ctx, output_path):
        self.save_calls.append(output_path + ".json")
        return self._run_stage("save_json", ctx)

    def stage_preserve_clean(self, ctx):
        return self._run_stage("preserve_clean", ctx)

    def stage_inpaint(self, ctx, force_cpu=False):
        ctx["_inpaint_force_cpu"] = force_cpu
        return self._run_stage("inpaint", ctx)

    def stage_render(self, ctx):
        return self._run_stage("render", ctx)

    def stage_save_png(self, ctx, output_path):
        # Tạo output file empty cho should_skip resume hoạt động.
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(b"\x00")
        ctx["summary"] = {"n_bubbles": 0, "n_translated": 0}
        return self._run_stage("save_png", ctx)


# --------------------------- Helpers ---------------------------- #

def _build_scheduler(
    pipeline,
    tmp_path: Path,
    runtime_overrides: Optional[Dict[str, Any]] = None,
):
    overrides = dict(runtime_overrides or {})
    cfg = RuntimeConfig(
        pipeline_depth=overrides.get("pipeline_depth", 3),
        cpu_pool_workers=overrides.get("cpu_pool_workers", 4),
        translation_rpm=overrides.get("translation_rpm", 600),
        translation_concurrency=overrides.get("translation_concurrency", 4),
        translation_max_retries=overrides.get("translation_max_retries", 1),
        translation_initial_backoff=overrides.get("translation_initial_backoff", 0.01),
        stage_retry_max=overrides.get("stage_retry_max", 1),
        watchdog_enable=overrides.get("watchdog_enable", False),
        enable_resume=overrides.get("enable_resume", True),
        enable_metrics_printer=False,
        crash_report_dir=str(tmp_path / "crashes"),
        structured_log_path=str(tmp_path / "log.jsonl"),
    )
    if "watchdog_stage_timeout_s" in overrides:
        cfg.watchdog_stage_timeout_s = overrides["watchdog_stage_timeout_s"]
    if "watchdog_poll_interval_s" in overrides:
        cfg.watchdog_poll_interval_s = overrides["watchdog_poll_interval_s"]

    executor = ThreadPoolExecutor(max_workers=cfg.cpu_pool_workers)
    cache = NullTranslationCache()
    checkpoint = CheckpointStore(str(tmp_path / "ckpt.json"), debounce_s=0.0)
    mem_monitor = MemoryMonitor()
    log = NullStructuredLogger(ring_size=20)
    metrics = MetricsRegistry()
    gpu_mutex = GPUMutex()
    worker = RateLimitedTranslator(pipeline.translator, cfg, cache, executor)
    watchdog = Watchdog(cfg) if cfg.watchdog_enable else None
    sched = Scheduler(
        pipeline, cfg, checkpoint, worker, gpu_mutex,
        mem_monitor, log, metrics, watchdog=watchdog, executor=executor,
    )
    return sched, executor, checkpoint, watchdog


# --------------------------- Tests ---------------------------- #

class TestSchedulerHappyPath:
    def test_single_page_complete(self, tmp_path: Path):
        async def run():
            pipe = FakePipeline()
            sched, ex, cp, _ = _build_scheduler(pipe, tmp_path)
            try:
                t = PageTask.new(
                    str(tmp_path / "p1.png"), str(tmp_path / "out" / "p1.png")
                )
                tasks = await sched.run([t])
                assert tasks[0].state == PageState.DONE
                # 11 stage chạy đủ.
                assert len(pipe.exit_order) == 11
                assert pipe.exit_order[0] == "load"
                assert pipe.exit_order[-1] == "save_png"
            finally:
                sched.close()
                ex.shutdown(wait=True)

        asyncio.run(run())

    def test_three_pages_complete(self, tmp_path: Path):
        async def run():
            pipe = FakePipeline()
            sched, ex, cp, _ = _build_scheduler(pipe, tmp_path)
            try:
                tasks = [
                    PageTask.new(
                        str(tmp_path / f"p{i}.png"),
                        str(tmp_path / "out" / f"p{i}.png"),
                    )
                    for i in range(3)
                ]
                results = await sched.run(tasks)
                assert all(t.state == PageState.DONE for t in results)
                # 3 page × 11 stage = 33 stage_run.
                assert len(pipe.exit_order) == 33
            finally:
                sched.close()
                ex.shutdown(wait=True)

        asyncio.run(run())


class TestSchedulerGpuMutex:
    def test_gpu_stage_never_overlaps(self, tmp_path: Path):
        async def run():
            # Delay nhỏ trên GPU stage để có cửa sổ race nếu mutex hỏng.
            pipe = FakePipeline(stage_delays={
                "detect": 0.05, "ocr": 0.05, "inpaint": 0.05,
            })
            sched, ex, cp, _ = _build_scheduler(
                pipe, tmp_path, runtime_overrides={"pipeline_depth": 3}
            )
            try:
                tasks = [
                    PageTask.new(
                        str(tmp_path / f"p{i}.png"),
                        str(tmp_path / "out" / f"p{i}.png"),
                    )
                    for i in range(3)
                ]
                await sched.run(tasks)
                # Quan trọng: gpu_concurrent_peak phải == 1 (mutex serialize).
                assert pipe.gpu_concurrent_peak == 1
            finally:
                sched.close()
                ex.shutdown(wait=True)

        asyncio.run(run())


class TestSchedulerRetry:
    def test_stage_transient_retry_succeeds(self, tmp_path: Path):
        async def run():
            pipe = FakePipeline(stage_fail_queue={
                "ocr": [RuntimeError("transient")],
            })
            sched, ex, cp, _ = _build_scheduler(
                pipe, tmp_path, runtime_overrides={"stage_retry_max": 2}
            )
            try:
                t = PageTask.new(
                    str(tmp_path / "p1.png"), str(tmp_path / "out" / "p1.png")
                )
                results = await sched.run([t])
                # Sau 1 fail + 1 success, page DONE.
                assert results[0].state == PageState.DONE
                # ocr ran 2 lần (fail + retry).
                assert pipe.exit_order.count("ocr") == 2
            finally:
                sched.close()
                ex.shutdown(wait=True)

        asyncio.run(run())

    def test_stage_fatal_marks_page_failed(self, tmp_path: Path):
        async def run():
            pipe = FakePipeline(stage_fail_queue={
                "ocr": [RuntimeError("e1"), RuntimeError("e2"),
                        RuntimeError("e3"), RuntimeError("e4")],
            })
            sched, ex, cp, _ = _build_scheduler(
                pipe, tmp_path, runtime_overrides={"stage_retry_max": 1}
            )
            try:
                t = PageTask.new(
                    str(tmp_path / "p1.png"), str(tmp_path / "out" / "p1.png")
                )
                results = await sched.run([t])
                assert results[0].state == PageState.FAILED
                assert "ocr" in (results[0].last_error or "").lower() or \
                       "RuntimeError" in (results[0].last_error or "")
            finally:
                sched.close()
                ex.shutdown(wait=True)

        asyncio.run(run())

    def test_one_page_fails_others_complete(self, tmp_path: Path):
        async def run():
            pipe = FakePipeline()
            # stage_retry_max=0 + 1 fail trong queue render → page nào hit
            # render trước nhất sẽ FAIL, các page khác chạy bình thường.
            # Deterministic vì chỉ 1 exception và không retry.
            pipe.stage_fail_queue["render"] = [
                RuntimeError("render fail page-X"),
            ]
            sched, ex, cp, _ = _build_scheduler(
                pipe, tmp_path, runtime_overrides={"stage_retry_max": 0}
            )
            try:
                tasks = [
                    PageTask.new(
                        str(tmp_path / f"p{i}.png"),
                        str(tmp_path / "out" / f"p{i}.png"),
                    )
                    for i in range(3)
                ]
                results = await sched.run(tasks)
                # Ít nhất 1 page FAIL và batch không kill.
                states = [t.state for t in results]
                assert PageState.FAILED in states
                assert PageState.DONE in states
            finally:
                sched.close()
                ex.shutdown(wait=True)

        asyncio.run(run())


class TestSchedulerOOM:
    def test_oom_inpaint_force_cpu_fallback(self, tmp_path: Path):
        async def run():
            # Fail inpaint với OOM 1 lần → retry với force_cpu=True.
            pipe = FakePipeline(stage_fail_queue={
                "inpaint": [RuntimeError("CUDA out of memory")],
            })
            sched, ex, cp, _ = _build_scheduler(
                pipe, tmp_path,
                runtime_overrides={"stage_retry_max": 2},
            )
            try:
                t = PageTask.new(
                    str(tmp_path / "p1.png"), str(tmp_path / "out" / "p1.png")
                )
                results = await sched.run([t])
                assert results[0].state == PageState.DONE
                assert "cpu_inpaint" in results[0].fallback_flags
                # inpaint ran 2x; 2nd time force_cpu=True.
                assert pipe.exit_order.count("inpaint") == 2
            finally:
                sched.close()
                ex.shutdown(wait=True)

        asyncio.run(run())


class TestSchedulerResume:
    def test_resume_skips_done_pages(self, tmp_path: Path):
        async def run():
            # Phase 1: run 2 page → cả 2 DONE, checkpoint persisted.
            pipe1 = FakePipeline()
            sched1, ex1, cp1, _ = _build_scheduler(pipe1, tmp_path)
            tasks1 = [
                PageTask.new(
                    str(tmp_path / f"p{i}.png"),
                    str(tmp_path / "out" / f"p{i}.png"),
                )
                for i in range(2)
            ]
            await sched1.run(tasks1)
            assert all(t.state == PageState.DONE for t in tasks1)
            sched1.close()
            ex1.shutdown(wait=True)

            assert os.path.isfile(str(tmp_path / "ckpt.json"))

            # Phase 2: new scheduler, load checkpoint, skip cả 2.
            pipe2 = FakePipeline()
            sched2, ex2, cp2, _ = _build_scheduler(pipe2, tmp_path)
            cp2.load()
            tasks2 = [
                PageTask.new(
                    str(tmp_path / f"p{i}.png"),
                    str(tmp_path / "out" / f"p{i}.png"),
                )
                for i in range(2)
            ]
            # restore_into đồng bộ state DONE.
            cp2.restore_into(tasks2)
            try:
                results = await sched2.run(tasks2)
                assert all(t.state == PageState.DONE for t in results)
                # Pipeline KHÔNG chạy stage nào.
                assert pipe2.exit_order == []
            finally:
                sched2.close()
                ex2.shutdown(wait=True)

        asyncio.run(run())


class TestSchedulerWatchdog:
    def test_watchdog_cancels_frozen_stage(self, tmp_path: Path):
        async def run():
            # detect "freeze" 1.5s, timeout 0.2s → watchdog cancel.
            pipe = FakePipeline(stage_delays={"detect": 1.5})
            sched, ex, cp, wd = _build_scheduler(
                pipe, tmp_path,
                runtime_overrides={
                    "watchdog_enable": True,
                    "watchdog_stage_timeout_s": {"detect": 0.2, "load": 30.0},
                    "watchdog_poll_interval_s": 0.05,
                    "stage_retry_max": 0,
                },
            )
            try:
                t = PageTask.new(
                    str(tmp_path / "p1.png"), str(tmp_path / "out" / "p1.png")
                )
                t0 = time.perf_counter()
                results = await sched.run([t])
                elapsed = time.perf_counter() - t0
                # Page FAILED (cancel) — không phải DONE.
                assert results[0].state == PageState.FAILED
                # Phải kết thúc nhanh hơn nhiều so với 1.5s nếu watchdog
                # cancel hiệu quả tại boundary.
                # Lưu ý: time.sleep KHÔNG bị preempt, nên elapsed ~= 1.5s
                # vẫn. Chỉ check task.cancel_requested.
                assert t.cancel_requested
            finally:
                sched.close()
                ex.shutdown(wait=True)

        asyncio.run(run())


class TestSchedulerCheckpoint:
    def test_checkpoint_records_progress(self, tmp_path: Path):
        async def run():
            pipe = FakePipeline()
            sched, ex, cp, _ = _build_scheduler(pipe, tmp_path)
            try:
                t = PageTask.new(
                    str(tmp_path / "p1.png"), str(tmp_path / "out" / "p1.png")
                )
                await sched.run([t])
                # Reload checkpoint từ disk, verify DONE.
                cp.flush()
                cp2 = CheckpointStore(str(tmp_path / "ckpt.json"))
                cp2.load()
                restored = cp2.get(t.page_id)
                assert restored is not None
                assert restored.state == PageState.DONE
                assert len(restored.completed_stages) == 11
            finally:
                sched.close()
                ex.shutdown(wait=True)

        asyncio.run(run())


class TestChapterRunnerLite:
    """Minimal smoke test cho ChapterRunner — verify wiring."""

    def test_runner_runs_with_fake_pipeline(self, tmp_path: Path):
        from mangatrans.runtime.chapter_runner import ChapterRunner

        pipe = FakePipeline()
        rt = RuntimeConfig(
            pipeline_depth=2,
            cpu_pool_workers=2,
            watchdog_enable=False,
            enable_metrics_printer=False,
            enable_resume=False,
            structured_log_path=str(tmp_path / "log.jsonl"),
            checkpoint_path=str(tmp_path / "ckpt.json"),
            translation_cache_path=str(tmp_path / "tc.json"),
            crash_report_dir=str(tmp_path / "crash"),
            translation_initial_backoff=0.01,
            stage_retry_max=0,
        )
        runner = ChapterRunner(PipelineConfig(), rt, pipeline=pipe)
        try:
            inputs = [
                (str(tmp_path / "in0.png"), str(tmp_path / "out" / "p0.png")),
                (str(tmp_path / "in1.png"), str(tmp_path / "out" / "p1.png")),
            ]
            results = runner.run(inputs, resume=False)
            assert len(results) == 2
            assert all(t.state == PageState.DONE for t in results)
        finally:
            runner.close()
