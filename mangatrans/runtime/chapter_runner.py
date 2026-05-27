"""ChapterRunner — sync facade quanh async Scheduler.

CLI / external caller dùng:

    runner = ChapterRunner(pipeline_cfg, runtime_cfg)
    summaries = runner.run([("in1.jpg", "out1.png"), ...], resume=True)
    runner.close()

Bên trong:
  1. Tạo MangaPipeline (raise_translation_errors=True).
  2. Tạo các runtime components (checkpoint, cache, mutex, worker, watchdog…).
  3. Build PageTask list, restore state từ checkpoint (resume).
  4. asyncio.run(Scheduler.run(tasks)).
  5. Cleanup tất cả (close worker, stop watchdog, release pipeline).

Backward-compat: ChapterRunner KHÔNG được gọi từ legacy `MangaPipeline.process_batch`.
Legacy path đi thẳng `pipeline.process_batch` (sync, swallow). User chuyển sang
async path qua CLI `--batch` (default tự bật).
"""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Optional, Sequence, Tuple

from ..config import PipelineConfig
from ..pipeline import MangaPipeline
from ..utils import get_logger
from .checkpoint import CheckpointStore, compute_config_digest
from .config import RuntimeConfig
from .crash_handler import CrashHandler
from .gpu_mutex import GPUMutex
from .memory_monitor import MemoryMonitor
from .metrics import HealthPrinter, MetricsRegistry
from .page_task import PageState, PageTask, StageName
from .scheduler import Scheduler
from .structured_log import NullStructuredLogger, StructuredLogger
from .translation_cache import NullTranslationCache, TranslationCache
from .translation_worker import RateLimitedTranslator
from .watchdog import Watchdog


def _default_path(out_dir: str, filename: str, override: Optional[str]) -> str:
    if override:
        return override
    return os.path.join(out_dir, filename)


class ChapterRunner:
    """Sync entry point cho async pipeline. Stateful — `close()` để release."""

    def __init__(
        self,
        pipeline_cfg: PipelineConfig,
        runtime_cfg: Optional[RuntimeConfig] = None,
        base_dir: str = ".",
        pipeline: Optional[MangaPipeline] = None,
    ):
        self.pipeline_cfg = pipeline_cfg
        self.runtime_cfg = runtime_cfg or RuntimeConfig()
        self._log = get_logger()

        # Pipeline: cho phép inject (test) hoặc tự tạo.
        if pipeline is None:
            self.pipeline = MangaPipeline(pipeline_cfg, base_dir=base_dir)
        else:
            self.pipeline = pipeline
        # Async path luôn raise translation errors để scheduler retry.
        self.pipeline.raise_translation_errors = self.runtime_cfg.raise_translation_errors

        self._scheduler: Optional[Scheduler] = None
        self._closed = False

    # --------------------------- Public API --------------------------- #

    def run(
        self,
        inputs: Sequence[Tuple[str, str]],
        resume: bool = True,
    ) -> List[PageTask]:
        """Process N page bất đồng bộ. Trả list PageTask đã update.

        Args:
            inputs: list (input_path, output_path).
            resume: bật resume nếu checkpoint cũ tồn tại.
        """
        if not inputs:
            return []
        out_dir = self._infer_out_dir(inputs)
        self._log.info(
            f"📦 [ChapterRunner] start chap ({len(inputs)} page) → {out_dir}"
        )

        # Resolve paths.
        rt = self.runtime_cfg
        cp_path = _default_path(out_dir, ".mangatrans_state.json", rt.checkpoint_path)
        cache_path = (
            _default_path(out_dir, ".translation_cache.json", rt.translation_cache_path)
            if rt.enable_translation_cache
            else None
        )
        log_path = _default_path(out_dir, ".mangatrans_log.jsonl", rt.structured_log_path)
        crash_dir = rt.crash_report_dir or out_dir

        # Components.
        struct_log = (
            StructuredLogger(
                log_path,
                ring_size=rt.structured_log_ring_size,
                max_bytes=rt.structured_log_max_bytes,
            )
            if log_path
            else NullStructuredLogger(ring_size=rt.structured_log_ring_size)
        )
        cache = (
            TranslationCache(cache_path, max_entries=rt.translation_cache_max_entries)
            if cache_path
            else NullTranslationCache()
        )
        mem_monitor = MemoryMonitor(
            vram_low_water_mb=rt.vram_low_water_mb,
            memory_high_water_pct=rt.memory_high_water_pct,
        )

        # Checkpoint resume.
        cfg_digest = compute_config_digest({
            "pipeline": self.pipeline_cfg,
            "runtime": self.runtime_cfg,
        })
        checkpoint = CheckpointStore(
            cp_path,
            config_digest=cfg_digest,
            debounce_s=rt.checkpoint_debounce_s,
        )
        if rt.enable_resume and rt.enable_checkpoint:
            checkpoint.load()
            if checkpoint.config_changed() and not rt.force_resume_on_config_change:
                self._log.warning(
                    "⚠️  [ChapterRunner] config digest đổi → rerun all. "
                    "Set --force-resume để bỏ qua."
                )
                checkpoint.clear()

        # Build tasks. Resume restore state từ checkpoint.
        tasks: List[PageTask] = [PageTask.new(inp, out) for inp, out in inputs]
        if rt.enable_resume and rt.enable_checkpoint:
            checkpoint.restore_into(tasks)
        else:
            for t in tasks:
                checkpoint.upsert(t)

        # Async components — phải tạo TRONG event loop (Semaphore bind loop).
        return asyncio.run(self._run_async(tasks, struct_log, cache, mem_monitor, checkpoint, crash_dir))

    async def _run_async(
        self,
        tasks: List[PageTask],
        struct_log: StructuredLogger,
        cache: TranslationCache,
        mem_monitor: MemoryMonitor,
        checkpoint: CheckpointStore,
        crash_dir: str,
    ) -> List[PageTask]:
        rt = self.runtime_cfg
        executor = ThreadPoolExecutor(
            max_workers=rt.cpu_pool_workers,
            thread_name_prefix="mt-chap",
        )
        try:
            worker = RateLimitedTranslator(
                self.pipeline.translator,
                rt,
                cache,
                executor,
            )
            gpu_mutex = GPUMutex()
            metrics = MetricsRegistry()
            crash = CrashHandler(
                struct_log,
                crash_dir,
                memory_monitor=mem_monitor,
                runtime_config_snapshot=self._runtime_snapshot(),
                pipeline_config_digest=compute_config_digest(self.pipeline_cfg),
            )
            crash.install()

            watchdog = Watchdog(rt) if rt.watchdog_enable else None
            health_printer: Optional[HealthPrinter] = None
            if rt.enable_metrics_printer and rt.health_print_interval_s > 0:
                health_printer = HealthPrinter(
                    metrics,
                    interval_s=rt.health_print_interval_s,
                    printer=lambda msg: self._log.info(msg),
                )
                health_printer.start()

            self._scheduler = Scheduler(
                self.pipeline, rt, checkpoint, worker, gpu_mutex,
                mem_monitor, struct_log, metrics,
                watchdog=watchdog, crash_handler=crash, executor=executor,
            )
            try:
                results = await self._scheduler.run(tasks)
            finally:
                self._scheduler.close()
                if health_printer is not None:
                    health_printer.stop()
                crash.uninstall()
                struct_log.close()
            return results
        finally:
            executor.shutdown(wait=False)

    def close(self) -> None:
        """Release pipeline + GPU memory. Idempotent."""
        if self._closed:
            return
        try:
            self.pipeline.release()
        except Exception as exc:  # noqa: BLE001
            self._log.warning(f"[ChapterRunner] pipeline.release fail: {exc}")
        self._closed = True

    # --------------------------- Helpers --------------------------- #

    def _infer_out_dir(self, inputs: Sequence[Tuple[str, str]]) -> str:
        """Chap out_dir = dirname của output đầu tiên."""
        first_out = inputs[0][1]
        out_dir = os.path.dirname(os.path.abspath(first_out)) or "."
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _runtime_snapshot(self) -> dict:
        from dataclasses import asdict
        try:
            return asdict(self.runtime_cfg)
        except TypeError:
            return {}
