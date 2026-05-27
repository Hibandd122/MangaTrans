"""RuntimeConfig — tham số điều phối runtime async.

Tách khỏi `PipelineConfig` vì:
- Chỉ áp dụng khi `enable_async=True` (sync path không đọc tới).
- Field nhiều, dễ override qua CLI mà không làm rối `PipelineConfig`.
- Có thể serialize riêng vào checkpoint để resume sau crash với config y hệt.

Mọi default đều an toàn cho 1 chap manga (~100-200 page) trên GPU 4GB:
- pipeline_depth=3 → max 3 page in-flight, GPU mutex serialize stage GPU.
- translation_rpm=20 → an toàn cho free tier OpenRouter (limit thực ~30 RPM).
- watchdog timeout đặt theo p99 đo từ benchmark v17, có buffer 1.5x.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# Default per-stage watchdog timeout (giây). Dựa trên p99 của benchmark v17 + 50%
# safety margin. Detect/OCR nặng GPU; translate là HTTP nên timeout dài để network
# spike không trigger false-positive freeze.
DEFAULT_STAGE_TIMEOUT_S: Dict[str, float] = {
    "load": 120.0,
    "detect": 120.0,
    "lang_detect": 60.0,
    "ocr": 300.0,
    "sfx": 120.0,
    "translate": 180.0,
    "preserve_clean": 60.0,
    "inpaint": 180.0,
    "render": 60.0,
    "save_json": 60.0,
    "save_png": 60.0,
}


@dataclass
class RuntimeConfig:
    """Tham số runtime async. Tất cả opt-in qua CLI Runtime flag group.

    Tại sao tách 3 ngưỡng memory:
    - vram_high_water_mb: trigger CPU fallback inpaint khi free VRAM thấp.
    - memory_high_water_pct: trigger backpressure (giảm pipeline_depth) khi
      RSS gần ceiling (OS có thể giết process).
    - oom_max_retries_per_page: chặn loop OOM vô hạn → mark page FAILED.
    """

    # ---- Async orchestration ----
    enable_async: bool = True
    pipeline_depth: int = 3
    cpu_pool_workers: int = 4

    # ---- Translation rate limiting ----
    translation_rpm: int = 20
    translation_concurrency: int = 4
    translation_max_retries: int = 5
    translation_initial_backoff: float = 2.0
    translation_backoff_factor: float = 2.0
    translation_backoff_cap: float = 60.0
    translation_jitter_ratio: float = 0.25
    # Trên 429: halve RPM, sau N giây không 429 thì +1/min recover.
    translation_rpm_min: int = 5
    translation_rpm_recover_after_s: float = 60.0

    # ---- Retry / failure policy ----
    max_page_retries: int = 2
    stage_retry_max: int = 2  # retry mỗi stage trước khi page bị FAIL
    oom_max_retries_per_page: int = 3

    # ---- Watchdog ----
    watchdog_enable: bool = True
    watchdog_poll_interval_s: float = 5.0
    watchdog_stage_timeout_s: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_STAGE_TIMEOUT_S)
    )

    # ---- Memory / VRAM ----
    vram_oom_cpu_fallback: bool = True
    vram_high_water_mb: int = 3500       # GTX 1650 4GB → 3.5GB cảnh báo
    vram_low_water_mb: int = 500          # < 500MB free → force CPU
    memory_high_water_pct: float = 88.0
    memory_check_interval_s: float = 10.0

    # ---- Checkpoint ----
    enable_checkpoint: bool = True
    checkpoint_path: Optional[str] = None  # None → <output_dir>/.mangatrans_state.json
    checkpoint_debounce_s: float = 1.0
    enable_resume: bool = True
    force_resume_on_config_change: bool = False

    # ---- Caches ----
    enable_translation_cache: bool = True
    translation_cache_path: Optional[str] = None  # None → <output_dir>/.translation_cache.json
    translation_cache_max_entries: int = 50_000
    ocr_cache_enabled: bool = False               # off mặc định — đắt vì key = sha256(crop)
    ocr_cache_path: Optional[str] = None

    # ---- Logging / crash ----
    structured_log_path: Optional[str] = None     # None → <output_dir>/.mangatrans_log.jsonl
    structured_log_max_bytes: int = 50 * 1024 * 1024  # rotate trên 50MB
    structured_log_ring_size: int = 30            # last N events kèm crash report
    crash_report_dir: Optional[str] = None        # None → <output_dir>
    health_print_interval_s: float = 30.0

    # ---- Metrics ----
    enable_metrics_printer: bool = True

    # ---- Misc ----
    raise_translation_errors: bool = True  # async path BẬT để retry; legacy=False
    fail_fast_on_critical: bool = False    # True = dừng cả chap khi 1 page FAIL crit

    def for_legacy_sync(self) -> "RuntimeConfig":
        """Trả về RuntimeConfig đã tắt mọi feature async (cho legacy path)."""
        from dataclasses import replace
        return replace(
            self,
            enable_async=False,
            watchdog_enable=False,
            enable_checkpoint=False,
            enable_translation_cache=False,
            raise_translation_errors=False,
        )

    def stage_timeout(self, stage: str) -> float:
        """Lookup timeout cho stage; fallback 120s nếu unknown."""
        return float(self.watchdog_stage_timeout_s.get(stage, 120.0))
