"""Runtime sub-package — async orchestration, checkpoint, watchdog, crash recovery.

Tách layer này khỏi `mangatrans/pipeline.py` để legacy sync path (`process_image`,
`process_batch`, `translate_image`) giữ nguyên semantics. Scheduler bọc các
`stage_*` method của `MangaPipeline` thành 1 DAG async với GPU mutex, token-bucket
rate limit, checkpoint JSON, watchdog daemon, crash report — tất cả opt-in qua
`RuntimeConfig.enable_async`.

Tại sao tách package riêng:
- Tránh kéo asyncio/psutil/checkpoint deps vào sync path khi user chỉ chạy 1 ảnh.
- Cho phép `from mangatrans.runtime import ChapterRunner` mà không phải import
  `pipeline` ngay (lazy chain qua `chapter_runner`).
- Test riêng từng layer (`tests/test_runtime_unit.py`) không cần load model.
"""
from __future__ import annotations

from .config import RuntimeConfig
from .page_task import PageState, PageTask, StageName

__all__ = [
    "RuntimeConfig",
    "PageTask",
    "PageState",
    "StageName",
    # Các tên dưới đây import lazy qua __getattr__ vì kéo theo asyncio/threading
    # nặng — chỉ load khi user thực sự gọi chapter runner.
    "ChapterRunner",
    "CheckpointStore",
    "RateLimitedTranslator",
    "StructuredLogger",
]


def __getattr__(name: str):  # pragma: no cover - thin shim
    if name == "ChapterRunner":
        from .chapter_runner import ChapterRunner
        return ChapterRunner
    if name == "CheckpointStore":
        from .checkpoint import CheckpointStore
        return CheckpointStore
    if name == "RateLimitedTranslator":
        from .translation_worker import RateLimitedTranslator
        return RateLimitedTranslator
    if name == "StructuredLogger":
        from .structured_log import StructuredLogger
        return StructuredLogger
    raise AttributeError(f"module 'mangatrans.runtime' has no attribute {name!r}")
