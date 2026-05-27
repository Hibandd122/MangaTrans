"""Scheduler — async DAG runner cho MangaPipeline stages.

Mỗi page chạy qua 11 stage tuần tự (LOAD → SAVE_PNG). Pipeline depth N cho phép
N page in-flight song song:
  - Stage GPU (detect, lang, ocr, inpaint) bị `GPUMutex` serialize → chỉ 1 page
    chạm GPU tại 1 thời điểm, nhưng các page khác có thể chạy stage CPU/IO.
  - Translate stage gọi `RateLimitedTranslator` qua executor → vẫn ăn token bucket.

Per-stage error policy:
  - Bắt mọi Exception trên stage; nếu là OOM → empty cache + retry với
    force_cpu=True (chỉ stage inpaint hỗ trợ).
  - Nếu retryable (rate limit, transient IO) → exponential backoff retry trong
    cấp `cfg.stage_retry_max`.
  - Hết retry → page mark FAILED, checkpoint flush, scheduler tiếp page khác.
  - CancelledError từ watchdog → retry như fatal, max_page_retries lần.

Resume logic: scheduler đọc CheckpointStore khi start; page DONE + output file
tồn tại → skip; còn lại rerun từ stage_load (partial in-memory state không
persist; cache lo phần expensive).

Tại sao 1 file dài 400 LOC: scheduler là entry point chính, dễ debug khi gom
toàn bộ flow page (load → stage chain → cleanup) tại 1 chỗ. Tách thêm class
sẽ phải pass state qua nhiều method, khó trace hơn.
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ..pipeline import MangaPipeline
from ..utils import get_logger
from .checkpoint import CheckpointStore
from .config import RuntimeConfig
from .crash_handler import CrashHandler
from .gpu_mutex import GPUMutex
from .memory_monitor import MemoryMonitor, is_oom_error
from .metrics import MetricsRegistry
from .page_task import GPU_STAGES, PageState, PageTask, StageName
from .structured_log import StructuredLogger
from .translation_cache import TranslationCache
from .translation_worker import RateLimitedTranslator, TranslateRequest
from .watchdog import Watchdog


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class StageRetry(Exception):
    """Stage bị lỗi retryable — scheduler retry trong cấp `stage_retry_max`."""


class StageFatal(Exception):
    """Stage fail không retry — page mark FAILED."""


class PageCancelled(Exception):
    """Watchdog kích hoạt cancel — coroutine raise tại boundary."""


class Scheduler:
    """Async scheduler. Lifecycle:

      sched = Scheduler(pipeline, runtime_cfg, ...)
      await sched.run(tasks)   # async entrypoint
      sched.close()            # flush, release

    Lưu ý: scheduler KHÔNG sở hữu pipeline — caller (`ChapterRunner`) chịu
    trách nhiệm `pipeline.release()` cuối cùng.
    """

    def __init__(
        self,
        pipeline: MangaPipeline,
        runtime_cfg: RuntimeConfig,
        checkpoint: CheckpointStore,
        translator_worker: RateLimitedTranslator,
        gpu_mutex: GPUMutex,
        memory_monitor: MemoryMonitor,
        structured_logger: StructuredLogger,
        metrics: MetricsRegistry,
        watchdog: Optional[Watchdog] = None,
        crash_handler: Optional[CrashHandler] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        self.pipeline = pipeline
        self.cfg = runtime_cfg
        self.checkpoint = checkpoint
        self.translator_worker = translator_worker
        self.gpu_mutex = gpu_mutex
        self.memory_monitor = memory_monitor
        self.log_jsonl = structured_logger
        self.metrics = metrics
        self.watchdog = watchdog
        self.crash_handler = crash_handler
        self.executor = executor or ThreadPoolExecutor(
            max_workers=runtime_cfg.cpu_pool_workers,
            thread_name_prefix="mt-scheduler",
        )
        self._owns_executor = executor is None
        self._log = get_logger()

        # Async state.
        self._pipeline_sem = asyncio.Semaphore(runtime_cfg.pipeline_depth)
        # Track running coroutines by page_id để watchdog/cancel kết nối được.
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Patch state cho translator bridge — restore khi close.
        self._original_translate_batch: Optional[Callable] = None

    # --------------------------- Public API --------------------------- #

    async def run(self, tasks: List[PageTask]) -> List[PageTask]:
        """Process N page trong async, tôn trọng pipeline_depth.

        Trả list PageTask đã cập nhật state (sau scheduler chạy). Caller có
        thể inspect `.state`, `.summary`, `.last_error`.
        """
        if not tasks:
            return []
        self._loop = asyncio.get_event_loop()

        # Bridge translator: pipeline.translator.translate_batch → RateLimitedTranslator
        # Phải patch ON event loop để bridge có thể marshal coro back.
        self._install_translator_bridge()

        # Start watchdog nếu được.
        if self.watchdog is not None and not self.watchdog.is_running:
            self.watchdog.start(self._loop, self, on_freeze=self._on_freeze)

        # Tạo coroutines.
        coros = [self._run_page(t) for t in tasks]
        # gather với return_exceptions để 1 page exception không kill cả batch.
        results = await asyncio.gather(*coros, return_exceptions=True)
        for t, res in zip(tasks, results):
            if isinstance(res, BaseException):
                # Exception đã được _run_page catch + log; defensive guard.
                self._log.error(f"❌ [Scheduler] page {t.page_id} unhandled: {res}")
                if t.state not in (PageState.DONE, PageState.FAILED):
                    t.state = PageState.FAILED
                    t.last_error = f"{type(res).__name__}: {res}"
        # Final checkpoint flush.
        self.checkpoint.flush()
        self.translator_worker.cache.flush()
        return tasks

    def cancel_task(self, page_id: str) -> None:
        """Watchdog → cancel coroutine của page_id. Thread-safe entry."""
        task = self._running_tasks.get(page_id)
        if task is None or task.done():
            return
        task.cancel()

    def close(self) -> None:
        """Cleanup. Idempotent."""
        self._uninstall_translator_bridge()
        try:
            self.translator_worker.close()
        except Exception as exc:  # noqa: BLE001
            self._log.warning(f"[Scheduler] translator_worker close fail: {exc}")
        try:
            self.checkpoint.flush()
        except Exception as exc:  # noqa: BLE001
            self._log.warning(f"[Scheduler] checkpoint flush fail: {exc}")
        if self.watchdog is not None:
            self.watchdog.stop(timeout=5.0)
        if self._owns_executor:
            self.executor.shutdown(wait=False)

    # --------------------------- Page runner --------------------------- #

    async def _run_page(self, task: PageTask) -> None:
        """Chạy 1 page qua 11 stage. Catch tất cả lỗi, mark state."""
        page_id = task.page_id
        async with self._pipeline_sem:
            # Resume short-circuit: DONE + output file → skip.
            if self.cfg.enable_resume and self.checkpoint.should_skip(task):
                self._log.info(
                    f"⏭️  [Scheduler] skip {page_id} (DONE, output tồn tại)"
                )
                self.metrics.incr("pages_skipped")
                self.log_jsonl.event(
                    "page_skip_resume", page_id=page_id,
                    input=task.input_path, output=task.output_path,
                )
                task.state = PageState.DONE
                return

            task.state = PageState.RUNNING
            task.started_ts = _utc_iso()
            task.completed_stages = []  # rerun từ đầu
            self.checkpoint.upsert(task)
            if self.watchdog is not None:
                self.watchdog.register(task)
            self._running_tasks[page_id] = asyncio.current_task()  # type: ignore[assignment]

            self.log_jsonl.event(
                "page_start", page_id=page_id, input=task.input_path,
            )
            self._log.info(f"📄 [Scheduler] start page={page_id} → {task.output_path}")

            ctx = self.pipeline.new_context()
            try:
                await self._run_stages(task, ctx)
                task.state = PageState.DONE
                task.completed_ts = _utc_iso()
                task.summary = ctx.get("summary")
                self.checkpoint.upsert(task)
                self.checkpoint.flush_if_due()
                self.metrics.incr("pages_done")
                self.log_jsonl.event(
                    "page_done", page_id=page_id,
                    n_completed_stages=len(task.completed_stages),
                    fallback_flags=sorted(task.fallback_flags),
                )
            except StageFatal as exc:
                self._mark_failed(task, exc)
            except asyncio.CancelledError:
                self._log.warning(
                    f"⚠️  [Scheduler] page {page_id} cancelled (watchdog)"
                )
                self._mark_failed(task, PageCancelled("watchdog cancelled"))
                # Không re-raise — gather còn các coroutines khác.
            except BaseException as exc:  # noqa: BLE001
                self._mark_failed(task, exc)
                if self.crash_handler is not None:
                    self.crash_handler.report_caught(
                        "scheduler._run_page", page_id, exc,
                    )
            finally:
                if self.watchdog is not None:
                    self.watchdog.unregister(page_id)
                self._running_tasks.pop(page_id, None)
                # Cleanup ctx ASAP — giải phóng RAM (image, masks).
                ctx.clear()

    async def _run_stages(self, task: PageTask, ctx: Dict[str, Any]) -> None:
        """Chạy tuần tự 11 stage. Mỗi stage retry trong cấp stage_retry_max."""
        for stage in StageName.ordered():
            # Cancel barrier — watchdog đã set flag.
            if task.cancel_requested:
                raise PageCancelled(f"cancel_requested before stage={stage.value}")
            await self._execute_stage(task, ctx, stage)
            task.mark_stage_complete(stage)
            self.checkpoint.mark_stage_complete(task.page_id, stage)

    async def _execute_stage(
        self,
        task: PageTask,
        ctx: Dict[str, Any],
        stage: StageName,
    ) -> None:
        """Execute 1 stage với retry + GPU mutex + OOM CPU fallback."""
        cfg = self.cfg
        stage_name = stage.value
        last_exc: Optional[BaseException] = None

        for attempt in range(cfg.stage_retry_max + 1):
            if self.watchdog is not None:
                self.watchdog.heartbeat(task.page_id, stage)
            task.current_stage = stage

            self.log_jsonl.event(
                "stage_start", page_id=task.page_id, stage=stage_name,
                attempt=attempt,
            )
            stage_start = time.perf_counter()
            try:
                if stage in GPU_STAGES:
                    async with self.gpu_mutex.acquire(task.page_id, stage_name):
                        await self._dispatch_stage(task, ctx, stage)
                else:
                    await self._dispatch_stage(task, ctx, stage)
                latency_ms = int((time.perf_counter() - stage_start) * 1000)
                self.log_jsonl.event(
                    "stage_end", page_id=task.page_id, stage=stage_name,
                    attempt=attempt, latency_ms=latency_ms,
                )
                self.metrics.incr(f"stage_{stage_name}_ok")
                # Cleanup CUDA cache giữa stage GPU để giảm peak VRAM.
                if stage in GPU_STAGES:
                    self.memory_monitor.cuda_empty_cache()
                return
            except asyncio.CancelledError:
                # Watchdog cancel — propagate ngay.
                raise
            except BaseException as exc:  # noqa: BLE001
                last_exc = exc
                latency_ms = int((time.perf_counter() - stage_start) * 1000)
                self.metrics.incr(f"stage_{stage_name}_err")
                self.log_jsonl.event(
                    "stage_error", page_id=task.page_id, stage=stage_name,
                    attempt=attempt, latency_ms=latency_ms,
                    error_type=type(exc).__name__, error_msg=str(exc)[:300],
                )

                # OOM → CPU fallback cho inpaint, recheck VRAM, retry.
                if is_oom_error(exc):
                    self.metrics.incr("oom_events")
                    self.memory_monitor.cuda_empty_cache()
                    if stage == StageName.INPAINT:
                        if task.oom_retries >= cfg.oom_max_retries_per_page:
                            raise StageFatal(
                                f"INPAINT OOM > {cfg.oom_max_retries_per_page} lần"
                            ) from exc
                        task.oom_retries += 1
                        task.fallback_flags.add("cpu_inpaint")
                        self._log.warning(
                            f"💀 [Scheduler] inpaint OOM, retry với force_cpu "
                            f"({task.oom_retries}/{cfg.oom_max_retries_per_page})"
                        )
                        # Set retry attempt budget — bonus retry cho OOM.
                        continue
                    # OOM ở stage GPU khác → để retry loop quyết định.

                if attempt >= cfg.stage_retry_max:
                    # Hết retry → upgrade thành StageFatal.
                    raise StageFatal(
                        f"stage {stage_name} fail sau {attempt + 1} attempt: "
                        f"{type(exc).__name__}: {str(exc)[:200]}"
                    ) from exc

                stage_retries = task.stage_retries.get(stage_name, 0) + 1
                task.stage_retries[stage_name] = stage_retries

                # Backoff giữa retry — exponential.
                delay = min(30.0, cfg.translation_initial_backoff * (2 ** attempt))
                self._log.warning(
                    f"⚠️  [Scheduler] page={task.page_id} stage={stage_name} "
                    f"attempt {attempt + 1} fail ({type(exc).__name__}); "
                    f"retry sau {delay:.1f}s"
                )
                await asyncio.sleep(delay)
        # Defensive
        if last_exc:
            raise StageFatal(f"stage {stage_name} fail unreachable") from last_exc

    async def _dispatch_stage(
        self,
        task: PageTask,
        ctx: Dict[str, Any],
        stage: StageName,
    ) -> None:
        """Gọi stage method. Stage IO/CPU chạy executor; GPU stage cũng executor
        (vì PyTorch forward block GIL — caller đã hold mutex)."""
        loop = asyncio.get_event_loop()
        p = self.pipeline
        force_cpu = stage == StageName.INPAINT and "cpu_inpaint" in task.fallback_flags

        if stage == StageName.LOAD:
            ctx_out = await loop.run_in_executor(
                self.executor, p.stage_load, ctx, task.input_path
            )
        elif stage == StageName.DETECT:
            ctx_out = await loop.run_in_executor(self.executor, p.stage_detect, ctx)
        elif stage == StageName.LANG:
            ctx_out = await loop.run_in_executor(self.executor, p.stage_lang_detect, ctx)
        elif stage == StageName.OCR:
            ctx_out = await loop.run_in_executor(self.executor, p.stage_ocr, ctx)
        elif stage == StageName.SFX:
            ctx_out = await loop.run_in_executor(self.executor, p.stage_sfx, ctx)
        elif stage == StageName.TRANSLATE:
            ctx_out = await loop.run_in_executor(
                self.executor, p.stage_translate, ctx, task.output_path
            )
        elif stage == StageName.SAVE_JSON:
            ctx_out = await loop.run_in_executor(
                self.executor, p.stage_save_json, ctx, task.output_path
            )
        elif stage == StageName.PRESERVE_CLEAN:
            ctx_out = await loop.run_in_executor(
                self.executor, p.stage_preserve_clean, ctx
            )
        elif stage == StageName.INPAINT:
            ctx_out = await loop.run_in_executor(
                self.executor, p.stage_inpaint, ctx, force_cpu
            )
        elif stage == StageName.RENDER:
            ctx_out = await loop.run_in_executor(self.executor, p.stage_render, ctx)
        elif stage == StageName.SAVE_PNG:
            ctx_out = await loop.run_in_executor(
                self.executor, p.stage_save_png, ctx, task.output_path
            )
        else:
            raise StageFatal(f"unknown stage: {stage}")

        # Stages trả ctx (cùng instance) — update inplace bằng cách hợp nhất key.
        if ctx_out is not None and ctx_out is not ctx:
            ctx.update(ctx_out)

    # --------------------------- Translator bridge --------------------------- #

    def _install_translator_bridge(self) -> None:
        """Thay `pipeline.translator.translate_batch` bằng bridge → RateLimitedTranslator.

        TranslationPipeline.translate_page sẽ vô tình gọi worker qua sync proxy
        — nhận về cache hit / rate-limited HTTP call. Restore khi `close()`.
        """
        original = self.pipeline.translator.translate_batch
        self._original_translate_batch = original
        loop = self._loop

        def bridge(texts, position_tags=None):
            # Sync proxy chạy trong executor thread; marshal coroutine vào event loop.
            if loop is None or not loop.is_running():
                # Fallback — chạy trực tiếp Translator (legacy path).
                return original(texts, position_tags=position_tags)
            req = TranslateRequest(texts=list(texts), position_tags=position_tags)
            fut = asyncio.run_coroutine_threadsafe(
                self.translator_worker.translate_batch(req), loop
            )
            # Block executor thread đến khi worker xong; timeout safety.
            return fut.result(timeout=self.cfg.stage_timeout("translate") * 2)

        self.pipeline.translator.translate_batch = bridge

    def _uninstall_translator_bridge(self) -> None:
        if self._original_translate_batch is not None:
            try:
                self.pipeline.translator.translate_batch = self._original_translate_batch
            except Exception:  # noqa: BLE001
                pass
            self._original_translate_batch = None

    # --------------------------- Helpers --------------------------- #

    def _mark_failed(self, task: PageTask, exc: BaseException) -> None:
        task.state = PageState.FAILED
        task.completed_ts = _utc_iso()
        task.last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
        self.checkpoint.upsert(task)
        self.metrics.incr("pages_failed")
        self.log_jsonl.event(
            "page_failed", page_id=task.page_id,
            error_type=type(exc).__name__, error_msg=str(exc)[:300],
            completed_stages=[s.value for s in task.completed_stages],
        )
        self._log.error(
            f"❌ [Scheduler] page {task.page_id} FAIL: {type(exc).__name__}: {exc}"
        )

    def _on_freeze(self, task: PageTask, stage: str, elapsed: float) -> None:
        """Watchdog callback. Log + metrics."""
        self.metrics.incr("watchdog_freezes")
        self.log_jsonl.event(
            "stage_freeze", page_id=task.page_id, stage=stage,
            elapsed_s=round(elapsed, 2),
        )
