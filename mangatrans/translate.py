"""Translation: multi-backend LLM, batched call, JSON parse, glossary, reading order.

Class Translator manages glossary across pages (persistent JSON), retries 429,
parses `[position]` tags để LLM biết vị trí câu trong layout trang.

Backend (Koharu integration 2026-05-27):
  - OpenRouter (default): remote API qua `llm_backend.OpenRouterBackend`.
  - Local LLM: GGUF via llama-cpp-python qua `llm_backend.LocalLLMBackend`.
  - OpenAI-compat: vLLM/Ollama/etc. qua `llm_backend.OpenAICompatBackend`.

Reading order: phải→trái, trên→dưới (RTL manga). Group bbox theo y-band (default
25% page height) rồi sort x giảm dần.

Glossary extraction strategy: chỉ ghi nhận term capitalized xuất hiện LITERAL
trong cả source + target.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Optional

from .config import (
    DEFAULT_OPENROUTER_MODEL,
    OPENROUTER_URL,
    ROMANCE_SYSTEM_PROMPT,
    TranslateConfig,
)
from .llm_backend import LLMBackend, OpenRouterBackend, create_llm_backend
from .utils import get_logger


# Blacklist common English caps tránh false-positive khi extract glossary.
_GLOSSARY_BLACKLIST = {
    "Nothing", "Beats", "Drinking", "Alcohol", "With", "Cute", "Girls",
    "Yes", "No", "And", "But", "The", "You", "What", "Who", "Why", "How",
    "When", "Where", "This", "That", "These", "Those", "Hey", "Oh", "Ah",
    "Eh", "Huh", "Wow", "Well", "Nevermind", "Okay", "Really", "Truly",
    "Still", "Also", "Suppose", "Mean", "Just", "Very", "Much", "More",
    "Some", "Any", "All", "Each", "Both", "Either", "Neither", "One",
    "Two", "Three", "First", "Last", "Next", "Then", "Now", "Here",
    "There", "Today", "Tomorrow", "Yesterday", "Phew", "See", "Look",
    "Come", "Let", "Have", "Has", "Had", "Make", "Made", "Try", "Tried",
    "Take", "Took", "Give", "Gave", "Get", "Got", "Want", "Need",
    "Believe", "Know", "Knew", "Think", "Thought", "Feel", "Felt",
    "Say", "Said", "Tell", "Told", "Ask", "Asked", "Talk", "Talked",
    "Show", "Showed", "Shown", "Find", "Found", "Use", "Used", "Work",
    "Worked", "Live", "Lived", "Love", "Loved", "Like", "Liked",
    "Hate", "Hated", "Help", "Helped", "Listen", "Listened", "Hear",
    "Heard", "Watch", "Watched", "Read", "Write", "Wrote", "Eat",
    "Ate", "Drink", "Drank", "Sleep", "Slept", "Wake", "Woke",
    "Stand", "Stood", "Sit", "Sat", "Walk", "Walked", "Run", "Ran",
    "Stop", "Stopped", "Start", "Started", "End", "Ended", "Open",
    "Opened", "Close", "Closed", "Turn", "Turned", "Move", "Moved",
    "Bring", "Brought", "Send", "Sent", "Sell", "Sold", "Buy",
    "Bought", "Pay", "Paid", "Spend", "Spent", "Win", "Won", "Lose",
    "Lost", "Play", "Played", "Sing", "Sang", "Sung", "Dance",
    "Danced", "Cook", "Cooked", "Wait", "Waited", "Stay", "Stayed",
    "Leave", "Left", "Enter", "Entered", "Sound", "Sounds", "Good",
    "Bad", "Great", "Big", "Small", "Long", "Short", "Tall", "High",
    "Low", "Old", "New", "Young", "Beautiful", "Ugly", "Easy", "Hard",
    "Fast", "Slow", "Hot", "Cold", "Warm", "Cool", "Tender", "Super",
    "Impressive", "Talent", "Create", "Song", "Sticks", "Lot", "Right",
    "Wrong", "Same", "Different", "Important", "Together", "Alone",
    "Maybe", "Perhaps", "Probably", "Definitely", "Absolutely",
    "Forward", "Backward", "Around", "Through", "Across", "Above",
    "Below", "Under", "Over", "Between", "During", "Before", "After",
    "While", "Since", "Until", "Because", "Although", "Though",
    "Unless", "Whether", "Either", "Otherwise", "However", "Therefore",
    "Indeed", "Anyway", "Forever", "Never", "Always", "Sometimes",
    "Often", "Usually", "Rarely", "Seldom", "Almost", "Quite",
    "Rather", "Pretty", "Fairly", "Hardly", "Barely", "Nearly",
    "Approximately", "Exactly", "Completely", "Totally", "Entirely",
}
_NAME_RE = re.compile(r"\b[A-Z][A-Za-z]{2,}(?:-[A-Za-z]+)?\b")


def position_tag(bbox, page_w: int, page_h: int) -> str:
    """Bbox → 'top-left/middle-center/bottom-right' tag cho prompt."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    v = "top" if cy < page_h / 3 else ("bottom" if cy > 2 * page_h / 3 else "middle")
    h = "right" if cx > 2 * page_w / 3 else ("left" if cx < page_w / 3 else "center")
    return f"{v}-{h}"


def reading_order_indices(ocr_results: list[dict], page_h: int,
                          band_ratio: float = 0.25) -> list[int]:
    """Sort indices theo manga RTL reading order.

    Group y-band (mặc định 25% page height), trong mỗi band sort x giảm dần.
    """
    if not ocr_results:
        return []
    band_h = max(1, int(page_h * band_ratio))
    items: list[tuple[int, int, int]] = []
    for i, r in enumerate(ocr_results):
        x1, y1, x2, y2 = r["bbox"]
        cy = (y1 + y2) // 2
        cx = (x1 + x2) // 2
        band = cy // band_h
        items.append((band, -cx, i))  # -cx vì RTL
    items.sort()
    return [it[2] for it in items]


class Translator:
    """Multi-backend translation wrapper. Glossary persistent + reading-order aware.

    Backend pluggable qua `LLMBackend` (default: OpenRouter).
    """

    def __init__(self, config: TranslateConfig,
                 backend: Optional[LLMBackend] = None):
        self.config = config
        self._log = get_logger()
        self.glossary: dict[str, str] = {}
        self._glossary_path: Optional[str] = config.glossary_path
        # Backend: nếu caller không truyền → auto tạo từ config
        self._backend = backend

    @property
    def backend(self) -> LLMBackend:
        """Lazy init backend. Cho phép pipeline set sau __init__."""
        if self._backend is None:
            self._backend = OpenRouterBackend(self.config)
        return self._backend

    @backend.setter
    def backend(self, value: LLMBackend) -> None:
        self._backend = value

    # --------------------------- Public API --------------------------- #

    def resolve_api_key(self) -> str:
        """Trả OpenRouter API key. Backward-compat."""
        from .config import DEFAULT_OPENROUTER_KEY
        return DEFAULT_OPENROUTER_KEY

    def resolve_model(self) -> str:
        """Model label đang dùng."""
        return self.backend.model_label()

    def attach_glossary(self, path: Optional[str]) -> None:
        """Load glossary từ path (nếu file tồn tại). Idempotent."""
        self._glossary_path = path
        self.glossary = _load_glossary_file(path) if path else {}

    def save_glossary(self) -> None:
        if not self._glossary_path or not self.config.use_glossary:
            return
        _save_glossary_file(self._glossary_path, self.glossary)

    def translate_batch(self, texts: list[str],
                        position_tags: Optional[list[str]] = None) -> list[str]:
        """Dịch list text → list translated theo đúng thứ tự."""
        if not texts:
            return []
        cfg = self.config
        glossary = self.glossary if cfg.use_glossary else None

        prompt = _build_prompt(texts, cfg.target_lang, glossary, position_tags)
        raw_text = self.backend.generate(prompt)
        translations = _parse_translations(raw_text, len(texts), self._log)
        if cfg.use_glossary:
            new_pairs = extract_glossary_entries(texts, translations)
            if new_pairs:
                self.glossary.update(new_pairs)
        return translations


# --------------------------- Glossary --------------------------- #

def _load_glossary_file(path: str) -> dict[str, str]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:  # noqa: BLE001
        get_logger().warning(f"⚠️  Glossary load fail ({path}): {e}")
    return {}


def _save_glossary_file(path: str, data: dict[str, str]) -> None:
    try:
        d = os.path.dirname(path)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:  # noqa: BLE001
        get_logger().warning(f"⚠️  Glossary save fail ({path}): {e}")


def extract_glossary_entries(originals: list[str],
                             translations: list[str]) -> dict[str, str]:
    """Trích (term_gốc → term_dịch) chỉ với term tồn tại literal trong cả 2."""
    pairs: dict[str, str] = {}
    if not originals or not translations:
        return pairs
    blacklist_lc = {w.lower() for w in _GLOSSARY_BLACKLIST}
    for src, tgt in zip(originals, translations):
        if not src or not tgt:
            continue
        for m in _NAME_RE.findall(src):
            if m.lower() in blacklist_lc:
                continue
            cap_form = m[0].upper() + m[1:].lower()
            if m in tgt:
                pairs[m] = m
            elif cap_form in tgt:
                pairs[m] = cap_form
    return pairs


# --------------------------- OpenRouter call (backward-compat) ------------------- #
# Logic đã move sang llm_backend.py. Import lại cho backward-compat.

from .llm_backend import _call_openrouter, _http_post_json, _parse_retry_delay  # noqa: E402,F401


# --------------------------- Shared parser --------------------------- #

def _parse_translations(raw: str, expected_n: int, log) -> list[str]:
    """Parse text content → list[str], pad/truncate."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        translated = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.S)
        if not m:
            raise RuntimeError(f"Không parse được JSON: {raw[:300]}")
        translated = json.loads(m.group(0))

    if isinstance(translated, dict):
        for key in ("translations", "result", "output", "items"):
            if key in translated and isinstance(translated[key], list):
                translated = translated[key]
                break
        else:
            raise RuntimeError(f"JSON object không có array: {translated!r}")

    if not isinstance(translated, list):
        raise RuntimeError(f"Không trả về list: {translated!r}")
    if len(translated) != expected_n:
        log.warning(
            f"⚠️  LLM trả {len(translated)} mục, kỳ vọng {expected_n}. Sẽ pad/truncate."
        )
        translated = (translated + [""] * expected_n)[:expected_n]
    return [str(t) for t in translated]


# --------------------------- Prompt builder --------------------------- #

# Upgraded translator prompt — bám sát thể loại Romance/Shoujo/Josei nhưng
# rõ ràng hơn về: persona, xưng hô theo speaker-context, độ tự nhiên, độ ngắn
# vừa bubble, SFX 1-1, OCR garbled handling, đại từ đồng nhất giữa các câu.
_TRANSLATOR_PERSONA = """Bạn là dịch giả manga chuyên nghiệp, chuyên thể loại Romance / Shoujo / Josei,
văn phong tiếng Việt mềm mại, lãng mạn, tinh tế. Bạn dịch như fan-translator
có tâm: không máy móc, không lai căng từ Hán-Việt cứng nhắc, không bịa thêm
chi tiết. Bạn HIỂU rằng output sẽ vẽ đè vào bubble manga — câu dịch phải
ngắn vừa khít chỗ, mất nhịp dài lê thê là sai."""


def _build_prompt(texts: list[str], target_lang: str,
                  glossary: Optional[dict[str, str]],
                  position_tags: Optional[list[str]]) -> str:
    if position_tags and len(position_tags) == len(texts):
        numbered = "\n".join(
            f"{i + 1}. [{position_tags[i]}] {t}" for i, t in enumerate(texts)
        )
    else:
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))

    glossary_block = ""
    if glossary:
        items = [f"  - {k} → {v}" for k, v in sorted(glossary.items())]
        glossary_block = (
            "\n# GLOSSARY (dùng EXACTLY các bản dịch này — KHÔNG đổi giữa chừng)\n"
            + "\n".join(items) + "\n"
        )

    return (
        f"{_TRANSLATOR_PERSONA}\n\n"
        f"{ROMANCE_SYSTEM_PROMPT}\n"
        f"{glossary_block}\n"
        f"# NHIỆM VỤ\n"
        f"Dịch mỗi dòng dưới đây sang {target_lang} theo đúng các quy tắc trên.\n\n"
        f"# CONTEXT QUAN TRỌNG\n"
        "- Input đã sort theo thứ tự đọc manga (phải→trái, trên→dưới); câu liền kề thường cùng 1 cảnh hội thoại.\n"
        "- Tag [position] (top-left/middle-center/...) giúp định vị trên trang — KHÔNG dịch tag đó.\n"
        "- Giữ ĐỒNG NHẤT đại từ + tone giữa các câu trong cùng cảnh. Đại từ ở câu sau phải khớp với câu trước (không 'anh ấy' rồi 'cậu ấy' trong cùng cuộc nói).\n"
        "- Nếu source là SFX (BANG, COUGH, HMPH…) hoặc interjection (Oh!, Ah!, Eh?), dịch 1-1 sang SFX/interjection tiếng Việt tương ứng (Ầm!, Khụ, Hừ, Ồ!, A!, Hả?).\n"
        "- OCR có thể đọc sai SFX/handwriting → đoán SFX hợp lý từ ngữ cảnh romance, KHÔNG trả '...' cho Latin garbled.\n"
        "- Câu dịch KHÔNG dài hơn câu gốc đáng kể (bubble hạn chế chỗ). Ưu tiên ngắn gọn, tự nhiên.\n\n"
        f"# OUTPUT FORMAT\n"
        "- Trả về CHỈ một JSON array of strings, đúng số phần tử và đúng thứ tự với input.\n"
        "- Mỗi phần tử là câu dịch tiếng Việt tương ứng.\n"
        "- KHÔNG copy số thứ tự hay tag [position] vào output.\n"
        "- KHÔNG thêm giải thích, KHÔNG markdown, KHÔNG ký tự ``` quanh JSON.\n\n"
        f"# INPUT\n{numbered}\n\n"
        "JSON array:"
    )
