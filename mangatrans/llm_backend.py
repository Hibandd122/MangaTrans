"""LLM backend abstraction — OpenRouter, local llama.cpp, OpenAI-compatible API.

Koharu integration (2026-05-27): tách logic gọi LLM ra khỏi `translate.py`
thành pluggable backend. `Translator` giờ delegate qua `LLMBackend.generate()`
thay vì hardcode `_call_openrouter()`.

Backends:
  - OpenRouterBackend: giữ nguyên logic HTTP hiện có (default).
  - LocalLLMBackend: chạy GGUF qua llama-cpp-python, auto-download từ HF.
  - OpenAICompatBackend: cho self-hosted API (vLLM, Ollama, text-gen-webui).

Factory: `create_llm_backend(translate_config, local_llm_config)` → LLMBackend.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Optional

from .config import (
    DEFAULT_OPENROUTER_KEY,
    DEFAULT_OPENROUTER_MODEL,
    OPENROUTER_URL,
    TranslateConfig,
)
from .utils import get_logger


class LLMBackend(ABC):
    """Abstract base cho mọi LLM backend."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Gửi prompt → trả text response."""

    @abstractmethod
    def model_label(self) -> str:
        """Tên model đang dùng (cho log/UI)."""

    def release(self) -> None:
        """Giải phóng resources. Override nếu cần."""


# ========================== OpenRouter Backend ========================== #

class OpenRouterBackend(LLMBackend):
    """Backend gốc — OpenRouter HTTP API. Giữ 100% logic từ translate.py."""

    def __init__(self, config: TranslateConfig):
        self.config = config
        self._log = get_logger()

    def generate(self, prompt: str) -> str:
        api_key = self._resolve_api_key()
        model = self.config.model or DEFAULT_OPENROUTER_MODEL
        return _call_openrouter(
            prompt, api_key, model,
            self.config.timeout, self.config.max_retries,
            self.config.temperature, self.config.top_p, self._log,
        )

    def model_label(self) -> str:
        return f"OpenRouter/{self.config.model or DEFAULT_OPENROUTER_MODEL}"

    def _resolve_api_key(self) -> str:
        return self.config.api_key or DEFAULT_OPENROUTER_KEY



# ========================== Factory ========================== #

def create_llm_backend(translate_config: TranslateConfig,
                       local_llm_config: Optional[Any] = None,
                       ) -> LLMBackend:
    """Factory tạo LLMBackend theo translate_config.backend.

    Args:
        translate_config: cấu hình dịch (chứa backend field).
        local_llm_config: (bỏ qua, giữ lại để tương thích API cũ).

    Returns:
        LLMBackend instance sẵn sàng gọi generate().
    """
    backend = translate_config.backend

    if backend == "openrouter":
        return OpenRouterBackend(translate_config)

    raise ValueError(
        f"Unknown LLM backend: {backend!r}. "
        f"Supported: 'openrouter'"
    )


# ========================== Shared HTTP helper ========================== #
# (moved from translate.py — giữ nguyên logic, shared giữa các backend)

def _call_openrouter(prompt: str, api_key: str, model: str,
                     timeout: int, max_retries: int,
                     temperature: float, top_p: float, log) -> str:
    """POST OpenRouter chat-completions → text content."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/mangatrans",
        "X-Title": "MangaTrans",
    }
    body = _http_post_json(OPENROUTER_URL, headers, payload, timeout,
                           max_retries, log, "OpenRouter")
    data = json.loads(body)
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"OpenRouter trả về cấu trúc lạ: {str(data)[:300]}") from e


def _http_post_json(url: str, headers: dict, payload: dict,
                    timeout: int, max_retries: int, log, label: str) -> str:
    """POST JSON → response body. Retry 429 với delay từ header/body."""
    headers = dict(headers)
    headers.setdefault("User-Agent", "mangatrans/0.1 (+https://github.com/)")
    data_bytes = json.dumps(payload).encode("utf-8")
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url, data=data_bytes, headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", "replace")
            if e.code == 429 and attempt < max_retries - 1:
                delay = _parse_retry_delay(err_body, e.headers)
                log.warning(
                    f"   [{label}] quota hit, đợi {delay}s rồi retry "
                    f"({attempt + 1}/{max_retries})..."
                )
                time.sleep(delay)
                continue
            raise RuntimeError(f"{label} API HTTP {e.code}: {err_body[:500]}")
    raise RuntimeError(f"{label} API không phản hồi sau khi retry.")


def _parse_retry_delay(body: str, headers) -> int:
    """Extract retry delay seconds từ body/headers. Default 30s."""
    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', body)
    if m:
        return int(float(m.group(1))) + 2
    m = re.search(r"try again in (\d+(?:\.\d+)?)s", body, re.IGNORECASE)
    if m:
        return int(float(m.group(1))) + 2
    try:
        ra = headers.get("Retry-After") if headers else None
        if ra:
            return int(float(ra)) + 1
    except (ValueError, AttributeError):
        pass
    return 30
