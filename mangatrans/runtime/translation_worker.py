"""RateLimitedTranslator — async wrapper bọc Translator.translate_batch.

Tại sao tách worker này khỏi `Translator`:
1. Translator hiện tại blocking + lock-free → 2 page parallel cùng gọi sẽ race
   `self.glossary` (read-modify-write).
2. Cần rate limit (TokenBucket RPM) + concurrency cap (Semaphore N) ở layer trên,
   không inline trong Translator để giữ legacy sync path đơn giản.
3. Cần persistent cache HIT TRƯỚC khi tốn token bucket — text đã dịch không
   được tốn budget.

Pipeline cho 1 batch texts:
  1. Check cache cho từng text → tách `cached_idxs` vs `to_translate`.
  2. Nếu hết text (tất cả hit) → trả translations cache, không chạm API.
  3. Acquire token bucket (1 token/request) + semaphore concurrency.
  4. Run `Translator.translate_batch(to_translate)` trên executor (block IO).
  5. Trên 429: halve RPM (`update_rpm`), exponential backoff + jitter, retry
     tối đa `max_retries`.
  6. Put kết quả vào cache.
  7. Merge cached + new → trả về theo đúng index gốc.

Glossary mutation: Translator.translate_batch tự update `self.glossary` sau khi
parse. Để 2 page parallel không race, ta lock 1 thread chạm Translator tại một
thời điểm. Network call vẫn block trong lock — không tối ưu được vì Translator
API hiện tại không tách network/state. Trade-off chấp nhận: glossary write chỉ
thêm vài key, cost microseconds; network call mới là cost lớn (giây).
→ Solution thực dụng: dùng Semaphore(translation_concurrency) để giới hạn
parallelism, KHÔNG lock global Translator. Translator instance riêng cho
runtime worker, glossary load chung file nhưng write atomic mỗi page.
"""
from __future__ import annotations

import asyncio
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..translate import Translator
from ..utils import get_logger
from .config import RuntimeConfig
from .rate_limiter import TokenBucket
from .translation_cache import TranslationCache


# Regex match 429/quota/rate-limit trong message exception. Translator hiện tại
# raise `RuntimeError(f"OpenRouter API HTTP {e.code}: ...")` khi quota cạn.
_RATE_LIMIT_PATTERNS = (
    re.compile(r"HTTP 429", re.IGNORECASE),
    re.compile(r"rate.?limit", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
    re.compile(r"quota", re.IGNORECASE),
    re.compile(r"retryDelay", re.IGNORECASE),
)


def is_rate_limit_error(exc: BaseException) -> bool:
    """True nếu exception trông như 429/rate-limit. Heuristic match message."""
    if exc is None:
        return False
    msg = str(exc)
    return any(p.search(msg) for p in _RATE_LIMIT_PATTERNS)


@dataclass
class TranslateRequest:
    """1 request dịch — texts kèm position tags, immutable."""

    texts: List[str]
    position_tags: Optional[List[str]] = None


class RateLimitedTranslator:
    """Async wrapper quanh Translator. Cache + rate limit + retry.

    Lifecycle:
      - `__init__(translator, runtime_cfg, cache, executor)` — không I/O.
      - `await translate_batch(req)` — main entrypoint.
      - `close()` — flush cache, không shutdown executor (chia sẻ với scheduler).

    Tại sao executor share với scheduler:
      Translator.translate_batch là sync (urlopen blocking). Phải chạy trong
      executor để không block event loop. Executor là tài nguyên chung — đừng
      tạo riêng cho translator (dễ leak thread, lệch concurrency budget).
    """

    def __init__(
        self,
        translator: Translator,
        runtime_cfg: RuntimeConfig,
        cache: TranslationCache,
        executor: ThreadPoolExecutor,
    ):
        self.translator = translator
        self.cfg = runtime_cfg
        self.cache = cache
        self.executor = executor
        self._log = get_logger()

        self._bucket = TokenBucket(
            rpm=runtime_cfg.translation_rpm,
            capacity=runtime_cfg.translation_rpm,
        )
        # Cap concurrent in-flight Translator calls. Khác với RPM (=tốc độ),
        # đây là TRẦN parallel — kể cả khi RPM cho phép, không quá N call
        # đồng thời (tránh peak burst gây timeout).
        self._concurrency_sem = asyncio.Semaphore(runtime_cfg.translation_concurrency)
        # Lock bảo vệ glossary read-modify-write trong Translator. Translator
        # API hiện tại merge `self.glossary` post-parse → 2 page parallel race.
        # Đây là lock SYNC (thread) vì Translator chạy trên executor.
        self._glossary_lock = threading.Lock()
        # Adaptive RPM state.
        self._last_429_ts: float = 0.0
        self._original_rpm: int = runtime_cfg.translation_rpm

        self.stats = {
            "calls": 0,           # số request đã issue (sau cache miss)
            "cache_hits": 0,      # texts hit cache
            "cache_misses": 0,    # texts đi API
            "rate_limit_hits": 0, # số lần backoff vì 429
            "retries": 0,         # tổng số retry (mọi nguyên nhân)
            "failures": 0,        # request raise sau hết retry
        }

    # --------------------------- Public API --------------------------- #

    async def translate_batch(self, req: TranslateRequest) -> List[str]:
        """Dịch list text. Cache lookup → bucket → executor → retry → cache put.

        Trả translations theo đúng order với `req.texts`. Trên failure cuối cùng
        raise exception (scheduler sẽ catch + retry stage / mark FAILED).
        """
        if not req.texts:
            return []

        model = self.translator.resolve_model()
        target_lang = self.translator.config.target_lang

        # Tách cached vs cần dịch. Giữ index gốc để merge sau.
        cached: List[Optional[str]] = []
        miss_idxs: List[int] = []
        miss_texts: List[str] = []
        miss_pos: List[str] = []
        for i, text in enumerate(req.texts):
            hit = self.cache.get(model, target_lang, text)
            if hit is not None:
                cached.append(hit)
                self.stats["cache_hits"] += 1
            else:
                cached.append(None)
                miss_idxs.append(i)
                miss_texts.append(text)
                if req.position_tags and i < len(req.position_tags):
                    miss_pos.append(req.position_tags[i])
                self.stats["cache_misses"] += 1

        if not miss_texts:
            # Full cache hit → trả luôn, không tốn token bucket.
            return [c or "" for c in cached]

        miss_pos_arg = miss_pos if (req.position_tags and len(miss_pos) == len(miss_texts)) else None

        # Recover RPM nếu đã qua window sau lần 429 cuối.
        self._maybe_recover_rpm()

        translations = await self._call_with_retry(
            miss_texts, miss_pos_arg, model, target_lang
        )

        # Put vào cache + merge lại theo index gốc.
        for src_idx, src_text, tr in zip(miss_idxs, miss_texts, translations):
            if tr:
                self.cache.put(model, target_lang, src_text, tr)
            cached[src_idx] = tr
        # Cache flush (debounced).
        self.cache.flush_if_due()

        return [c or "" for c in cached]

    def close(self) -> None:
        """Flush cache. Không shutdown executor (caller sở hữu)."""
        try:
            self.cache.flush()
        except Exception as exc:  # noqa: BLE001
            self._log.warning(f"⚠️  Cache flush fail khi close: {exc}")

    # --------------------------- Internals --------------------------- #

    async def _call_with_retry(
        self,
        texts: List[str],
        position_tags: Optional[List[str]],
        model: str,
        target_lang: str,
    ) -> List[str]:
        """Acquire bucket → executor call → backoff retry.

        Trả translations cùng length với texts. Raise nếu hết retry.
        """
        cfg = self.cfg
        last_exc: Optional[BaseException] = None
        for attempt in range(cfg.translation_max_retries + 1):
            # Bucket + concurrency. Bucket acquire có thể chờ vài chục giây
            # khi RPM bị halve → asyncio không block thread khác.
            await self._bucket.acquire(1.0)
            async with self._concurrency_sem:
                self.stats["calls"] += 1
                loop = asyncio.get_event_loop()
                try:
                    result = await loop.run_in_executor(
                        self.executor,
                        self._call_translator_sync,
                        texts,
                        position_tags,
                    )
                    return result
                except BaseException as exc:  # noqa: BLE001
                    last_exc = exc
                    if is_rate_limit_error(exc):
                        self.stats["rate_limit_hits"] += 1
                        self._on_rate_limit()
                    if attempt >= cfg.translation_max_retries:
                        self.stats["failures"] += 1
                        # Hết retry — bóc exception cho scheduler xử lý.
                        raise
                    delay = self._backoff_delay(attempt)
                    self.stats["retries"] += 1
                    self._log.warning(
                        f"   [Translate] attempt {attempt + 1}/"
                        f"{cfg.translation_max_retries + 1} fail "
                        f"({type(exc).__name__}: {str(exc)[:120]}); "
                        f"chờ {delay:.1f}s rồi retry"
                    )
                    await asyncio.sleep(delay)
        # Defensive — không bao giờ tới đây vì raise trên hết retry.
        if last_exc:
            raise last_exc
        return [""] * len(texts)

    def _call_translator_sync(
        self,
        texts: List[str],
        position_tags: Optional[List[str]],
    ) -> List[str]:
        """Sync call trong executor. Lock quanh glossary R-M-W."""
        # Network call BLOCKING. Lock chỉ ôm phần state mutation. Translator
        # hiện tại không tách phase nên ta lock toàn bộ — chấp nhận để đảm bảo
        # correctness. Performance cost: serialize translation 1 thread tại 1
        # thời điểm. Phần lớn parallel value đến từ pipeline overlap stages
        # khác (detect/OCR/inpaint), không từ parallel HTTP.
        with self._glossary_lock:
            return self.translator.translate_batch(texts, position_tags=position_tags)

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff + jitter, cap configurable."""
        cfg = self.cfg
        base = cfg.translation_initial_backoff * (cfg.translation_backoff_factor ** attempt)
        base = min(cfg.translation_backoff_cap, base)
        jitter = random.uniform(0, base * cfg.translation_jitter_ratio)
        return base + jitter

    def _on_rate_limit(self) -> None:
        """Halve RPM (clamp min). Set timestamp để recover dần sau."""
        new_rpm = max(self.cfg.translation_rpm_min, self._bucket.rpm // 2)
        if new_rpm < self._bucket.rpm:
            self._log.info(
                f"   [Translate] 429 → giảm RPM {self._bucket.rpm} → {new_rpm}"
            )
            self._bucket.update_rpm(new_rpm)
        self._last_429_ts = time.perf_counter()

    def _maybe_recover_rpm(self) -> None:
        """Sau `recover_after_s` không 429, +1 RPM/min đến `_original_rpm`."""
        if self._bucket.rpm >= self._original_rpm:
            return
        if self._last_429_ts == 0.0:
            return
        elapsed = time.perf_counter() - self._last_429_ts
        if elapsed < self.cfg.translation_rpm_recover_after_s:
            return
        # +1 mỗi 60s qua threshold (clamp original).
        bonus = int((elapsed - self.cfg.translation_rpm_recover_after_s) / 60.0) + 1
        target = min(self._original_rpm, self._bucket.rpm + bonus)
        if target > self._bucket.rpm:
            self._log.info(
                f"   [Translate] recover RPM {self._bucket.rpm} → {target}"
            )
            self._bucket.update_rpm(target)
            # Reset timestamp để tránh recover liên tục lần sau (gradual).
            self._last_429_ts = time.perf_counter()
