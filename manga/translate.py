"""Translation — OpenRouter backend + role-aware orchestrator (consolidated).

Gộp `translate.py` (Translator base) + `translation_pipeline.py` (role-aware).

- Translator: batched OpenRouter call, JSON parse, glossary persistent,
  reading-order aware (RTL manga).
- TranslationPipeline: wrap Translator + role-aware prompts (dialogue/
  narration/SFX), speaker continuity, honorifics preservation, name memory.

Provider: OpenRouter (OpenAI-compatible aggregator). Default model
`nvidia/nemotron-3-super-120b-a12b:free`.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from .config import (
    DEFAULT_OPENROUTER_MODEL,
    OPENROUTER_URL,
    ROMANCE_SYSTEM_PROMPT,
    TranslateConfig,
)
from .utils import get_logger


# =============================================================
# Section 1: Glossary blacklist + regex
# =============================================================

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

HONORIFIC_TOKENS = ("-san", "-chan", "-kun", "-sama", "-senpai", "-sensei",
                    "-dono", "-tan", "-nee", "-nii")
HONORIFIC_RE = re.compile(
    r"\b([A-Z][a-zA-Z]{1,15})(-(?:san|chan|kun|sama|senpai|sensei|dono|tan|nee|nii))\b"
)
NAME_PATTERN = re.compile(r"\b[A-Z][a-zA-Z]{1,15}\b")


# =============================================================
# Section 2: Reading order + position helpers
# =============================================================

def position_tag(bbox, page_w: int, page_h: int) -> str:
    """Bbox → 'top-left/middle-center/bottom-right' tag cho prompt."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    v = "top" if cy < page_h / 3 else ("bottom" if cy > 2 * page_h / 3 else "middle")
    h = "right" if cx > 2 * page_w / 3 else ("left" if cx < page_w / 3 else "center")
    return f"{v}-{h}"


def reading_order_indices(ocr_results: list[dict], page_h: int,
                          band_ratio: float = 0.25) -> list[int]:
    """Sort indices theo manga RTL reading order."""
    if not ocr_results:
        return []
    band_h = max(1, int(page_h * band_ratio))
    items: list[tuple[int, int, int]] = []
    for i, r in enumerate(ocr_results):
        x1, y1, x2, y2 = r["bbox"]
        cy = (y1 + y2) // 2
        cx = (x1 + x2) // 2
        band = cy // band_h
        items.append((band, -cx, i))
    items.sort()
    return [it[2] for it in items]


# =============================================================
# Section 3: Translator base (OpenRouter wrapper)
# =============================================================

class Translator:
    """OpenRouter translation wrapper. Glossary persistent + reading-order aware."""

    def __init__(self, config: TranslateConfig):
        self.config = config
        self._log = get_logger()
        self.glossary: dict[str, str] = {}
        self._glossary_path: Optional[str] = config.glossary_path

    def resolve_api_key(self) -> str:
        """Trả OpenRouter API key — hardcode DEFAULT_OPENROUTER_KEY (user explicit)."""
        from .config import DEFAULT_OPENROUTER_KEY
        return DEFAULT_OPENROUTER_KEY

    def resolve_model(self) -> str:
        return self.config.model or DEFAULT_OPENROUTER_MODEL

    def attach_glossary(self, path: Optional[str]) -> None:
        self._glossary_path = path
        self.glossary = _load_glossary_file(path) if path else {}

    def save_glossary(self) -> None:
        if not self._glossary_path or not self.config.use_glossary:
            return
        _save_glossary_file(self._glossary_path, self.glossary)

    def translate_batch(self, texts: list[str],
                        position_tags: Optional[list[str]] = None) -> list[str]:
        """Dịch list text → list translated theo đúng thứ tự. Retry JSON parse once."""
        if not texts:
            return []
        api_key = self.resolve_api_key()
        model = self.resolve_model()
        cfg = self.config
        glossary = self.glossary if cfg.use_glossary else None

        prompt = _build_prompt(
            texts, cfg.target_lang, glossary, position_tags,
            system_prompt=cfg.system_prompt,
        )
        raw_text = _call_openrouter(
            prompt, api_key, model,
            cfg.timeout, cfg.max_retries, cfg.temperature, cfg.top_p, self._log,
        )
        try:
            translations = _parse_translations(raw_text, len(texts), self._log)
        except RuntimeError:
            # Retry once with stricter prompt on JSON parse failure
            self._log.warning("⚠️  JSON parse failed, retrying with stricter prompt...")
            retry_prompt = prompt + "\n\nIMPORTANT: Return ONLY a valid JSON array. No markdown, no explanation."
            raw_text = _call_openrouter(
                retry_prompt, api_key, model,
                cfg.timeout, cfg.max_retries, cfg.temperature * 0.5, cfg.top_p, self._log,
            )
            translations = _parse_translations(raw_text, len(texts), self._log)
        if cfg.use_glossary:
            new_pairs = extract_glossary_entries(texts, translations)
            if new_pairs:
                self.glossary.update(new_pairs)
        return translations


# =============================================================
# Section 4: Glossary IO
# =============================================================

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


# =============================================================
# Section 5: HTTP call + parse
# =============================================================

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
    """POST JSON → response body. Retry 429/502/503/504 với delay."""
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
            if e.code in (502, 503, 504) and attempt < max_retries - 1:
                log.warning(
                    f"   [{label}] API báo lỗi {e.code} (Gateway/Timeout), đợi 5s rồi thử lại "
                    f"({attempt + 1}/{max_retries})..."
                )
                time.sleep(5)
                continue
            raise RuntimeError(f"{label} API HTTP {e.code}: {err_body[:500]}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < max_retries - 1:
                log.warning(
                    f"   [{label}] Lỗi mạng/Timeout ({e}), đợi 5s rồi thử lại "
                    f"({attempt + 1}/{max_retries})..."
                )
                time.sleep(5)
                continue
            raise RuntimeError(f"{label} API lỗi mạng: {e}")
    raise RuntimeError(f"{label} API không phản hồi sau khi retry.")


def _parse_retry_delay(body: str, headers) -> int:
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


# =============================================================
# Section 6: Prompt builder
# =============================================================

_TRANSLATOR_PERSONA = """Bạn là dịch giả manga chuyên nghiệp, chuyên thể loại Romance / Shoujo / Josei,
văn phong tiếng Việt mềm mại, lãng mạn, tinh tế. Bạn dịch như fan-translator
có tâm: không máy móc, không lai căng từ Hán-Việt cứng nhắc, không bịa thêm
chi tiết. Bạn HIỂU rằng output sẽ vẽ đè vào bubble manga — câu dịch phải
ngắn vừa khít chỗ, mất nhịp dài lê thê là sai."""


def _build_prompt(texts: list[str], target_lang: str,
                  glossary: Optional[dict[str, str]],
                  position_tags: Optional[list[str]],
                  system_prompt: Optional[str] = None) -> str:
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

    persona_prompt = system_prompt if system_prompt else ROMANCE_SYSTEM_PROMPT

    return (
        f"{_TRANSLATOR_PERSONA}\n\n"
        f"{persona_prompt}\n"
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


# =============================================================
# Section 7: Translation pipeline (role-aware orchestrator)
# =============================================================

@dataclass
class TranslationPipelineConfig:
    """Cấu hình orchestrator."""

    honorifics_keep: bool = True
    name_memory_path: Optional[str] = None
    speaker_band_ratio: float = 0.18
    sfx_skip_preserve_pixels: bool = True
    sanitize_punctuation: bool = True
    role_aware_prompt: bool = True


@dataclass
class CharacterMemory:
    """Một character entry trong name_memory."""

    name_src: str
    name_target: str
    first_seen_page: int = 0
    occurrences: int = 1
    side_hint: str = "unknown"


class TranslationPipeline:
    """Translation orchestrator. Wrap Translator + speaker continuity."""

    def __init__(self, translator: Translator,
                 config: Optional[TranslationPipelineConfig] = None):
        self.translator = translator
        self.config = config or TranslationPipelineConfig()
        self._log = get_logger()
        self.characters: dict[str, CharacterMemory] = {}
        self._name_memory_path: Optional[str] = self.config.name_memory_path
        self._page_idx = 0

    def attach_memory(self, path: Optional[str]) -> None:
        self._name_memory_path = path
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, dict):
                        self.characters[k] = CharacterMemory(
                            name_src=v.get("name_src", k),
                            name_target=v.get("name_target", k),
                            first_seen_page=int(v.get("first_seen_page", 0)),
                            occurrences=int(v.get("occurrences", 1)),
                            side_hint=v.get("side_hint", "unknown"),
                        )
        except Exception as e:  # noqa: BLE001
            self._log.warning(f"⚠️  Name memory load fail: {e}")

    def save_memory(self) -> None:
        if not self._name_memory_path:
            return
        try:
            d = os.path.dirname(self._name_memory_path)
            if d and not os.path.isdir(d):
                os.makedirs(d, exist_ok=True)
            payload = {
                k: {
                    "name_src": v.name_src, "name_target": v.name_target,
                    "first_seen_page": v.first_seen_page,
                    "occurrences": v.occurrences, "side_hint": v.side_hint,
                }
                for k, v in self.characters.items()
            }
            with open(self._name_memory_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:  # noqa: BLE001
            self._log.warning(f"⚠️  Name memory save fail: {e}")

    def translate_page(self, ocr_results: list[dict],
                       page_w: int, page_h: int,
                       page_idx: int = 0) -> list[str]:
        """Translate cả page. Return list translation align ocr_results."""
        self._page_idx = page_idx
        cfg = self.config

        if cfg.sfx_skip_preserve_pixels:
            kept = [i for i, r in enumerate(ocr_results)
                    if not r.get("should_preserve_pixels")
                    and r.get("should_translate", True) is not False
                    and (r.get("text") or "").strip()]
        else:
            kept = [i for i, r in enumerate(ocr_results)
                    if r.get("should_translate", True) is not False
                    and (r.get("text") or "").strip()]

        if not kept:
            return [""] * len(ocr_results)

        sub = [ocr_results[i] for i in kept]
        ro = reading_order_indices(sub, page_h)
        ordered_global_idx = [kept[k] for k in ro]
        texts = [ocr_results[i]["text"] for i in ordered_global_idx]
        if cfg.sanitize_punctuation:
            texts = [_sanitize_punctuation(t) for t in texts]
        position_tags = [
            position_tag(ocr_results[i]["bbox"], page_w, page_h)
            for i in ordered_global_idx
        ]
        roles = [ocr_results[i].get("role", "dialogue")
                 for i in ordered_global_idx]

        speaker_hints = self._derive_speaker_hints(
            [ocr_results[i] for i in ordered_global_idx], page_h,
        )

        if cfg.role_aware_prompt:
            tagged_texts = [
                _tag_text_for_prompt(t, role, hint)
                for t, role, hint in zip(texts, roles, speaker_hints)
            ]
        else:
            tagged_texts = texts

        translations = self.translator.translate_batch(tagged_texts, position_tags)

        if cfg.honorifics_keep:
            translations = [self._ensure_honorifics_kept(src, tgt)
                            for src, tgt in zip(texts, translations)]

        self._update_character_memory(texts, translations, position_tags,
                                      ordered_global_idx, ocr_results)

        out = [""] * len(ocr_results)
        for gidx, tr in zip(ordered_global_idx, translations):
            out[gidx] = tr
        return out

    def _derive_speaker_hints(self, ordered_results: list[dict],
                              page_h: int) -> list[str]:
        if not ordered_results:
            return []
        band_h = max(1, int(page_h * self.config.speaker_band_ratio))
        hints: list[str] = []
        prev_band = None
        cluster_id = 0
        for r in ordered_results:
            x1, y1, x2, y2 = r["bbox"]
            cy = (y1 + y2) // 2
            band = cy // band_h
            if prev_band is None or band != prev_band:
                cluster_id += 1
            prev_band = band
            cx = (x1 + x2) // 2
            side = "right" if cx > (r.get("_page_w", 9999) // 2) else "left"
            hints.append(f"turn{cluster_id}-{side}")
        return hints

    def _ensure_honorifics_kept(self, src: str, tgt: str) -> str:
        """Nếu src có 'Name-san' mà tgt drop suffix → append lại."""
        if not src or not tgt:
            return tgt
        for m in HONORIFIC_RE.finditer(src):
            full = m.group(0)
            name = m.group(1)
            suffix = m.group(2)
            if name in tgt and full not in tgt:
                tgt = tgt.replace(name, name + suffix, 1)
        return tgt

    def _update_character_memory(self, sources: list[str], targets: list[str],
                                 position_tags: list[str],
                                 ordered_global_idx: list[int],
                                 ocr_results: list[dict]) -> None:
        pairs = extract_glossary_entries(sources, targets)
        for k, v in pairs.items():
            entry = self.characters.get(k)
            ptag = position_tags[0] if position_tags else "middle-center"
            side = "left" if "left" in ptag else (
                "right" if "right" in ptag else "unknown")
            if entry is None:
                self.characters[k] = CharacterMemory(
                    name_src=k, name_target=v,
                    first_seen_page=self._page_idx, occurrences=1,
                    side_hint=side,
                )
            else:
                entry.occurrences += 1
                if entry.side_hint == "unknown" and side != "unknown":
                    entry.side_hint = side


def _sanitize_punctuation(text: str) -> str:
    if not text:
        return text
    out = re.sub(r"\.{4,}", "...", text)
    out = re.sub(r"\?{3,}", "??", out)
    out = re.sub(r"[?!]{4,}", "?!", out)
    out = re.sub(r"\s+([!?,.])", r"\1", out)
    return out.strip()


def _tag_text_for_prompt(text: str, role: str, speaker_hint: str) -> str:
    role_tag = {"dialogue": "DLG", "narration": "NAR", "sfx": "SFX"}.get(role, "DLG")
    return f"[{role_tag}|{speaker_hint}] {text}"
