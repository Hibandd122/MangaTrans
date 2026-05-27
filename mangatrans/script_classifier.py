"""Script classifier — phân loại text region thành CJK/Latin/mixed/symbol.

Dùng sau OCR cho từng text region để:
- chọn font phù hợp (CJK font cho kana/han, Latin font cho ABC).
- chọn rendering direction (vertical Japanese vs horizontal).
- gate translation behavior (preserve CJK SFX, translate Latin).

KHÔNG dùng cho whole-page detection — đó là việc của LanguageDetector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .language_detector import SCRIPT_RANGES


@dataclass
class ScriptProfile:
    """Bóc tách script content của 1 đoạn text."""

    text: str
    has_hiragana: bool = False
    has_katakana: bool = False
    has_kanji: bool = False
    has_hangul: bool = False
    has_hanzi_only: bool = False  # han nhưng không có kana → coi như Chinese/han
    has_latin: bool = False
    has_vietnamese_diacritic: bool = False
    has_digit: bool = False
    has_punct: bool = False
    primary: str = "unknown"  # 'ja' | 'ko' | 'zh' | 'en' | 'vi' | 'sfx' | 'mixed' | 'unknown'
    counts: dict[str, int] = None
    is_short_sfx: bool = False  # candidate SFX/exclamation token


def _in_range(cp: int, ranges) -> bool:
    for lo, hi in ranges:
        if lo <= cp <= hi:
            return True
    return False


def classify_script(text: Optional[str], min_letter_count: int = 1) -> ScriptProfile:
    """Phân loại text → primary script.

    Logic:
    - Có kana → Japanese (kể cả lẫn kanji).
    - Có hangul → Korean.
    - Chỉ han (không kana) → coi Chinese ('zh').
    - Có Vietnamese diacritic (Latin Extended Additional) → Vietnamese.
    - Latin only → English ('en').
    - Mixed CJK + Latin (≥ 2 script families) → 'mixed'.
    - Quá ngắn (<3 letter visible) + có ! ? . → SFX candidate.
    """
    if not text:
        return ScriptProfile(text="")

    counts: dict[str, int] = {k: 0 for k in SCRIPT_RANGES}
    counts["digit"] = 0
    counts["punct"] = 0
    counts["letter_total"] = 0

    for ch in text:
        cp = ord(ch)
        if ch.isspace():
            continue
        if ch.isdigit():
            counts["digit"] += 1
            continue
        if not ch.isalpha() and cp < 0x3000:
            counts["punct"] += 1
            continue
        matched = False
        for key, ranges in SCRIPT_RANGES.items():
            if _in_range(cp, ranges):
                counts[key] += 1
                counts["letter_total"] += 1
                matched = True
                break
        if not matched and ch.isalpha():
            counts["latin"] += 1
            counts["letter_total"] += 1

    has_hiragana = counts["hiragana"] > 0
    has_katakana = counts["katakana"] > 0
    has_kanji = counts["kanji"] > 0
    has_hangul = counts["hangul"] > 0
    has_latin = counts["latin"] > 0
    has_viet_dia = counts["vietnamese_diacritic"] > 0
    has_hanzi_only = has_kanji and not (has_hiragana or has_katakana)

    primary = "unknown"
    if has_hiragana or has_katakana:
        primary = "ja"
    elif has_hangul:
        primary = "ko"
    elif has_hanzi_only:
        primary = "zh"
    elif has_viet_dia:
        primary = "vi"
    elif has_latin:
        primary = "en"

    # Mixed flag: nhiều hơn 1 script family
    families_present = sum([
        has_hiragana or has_katakana or has_kanji,
        has_hangul,
        has_latin,
    ])
    if families_present >= 2 and primary != "ja":
        primary = "mixed"

    is_short_sfx = (counts["letter_total"] <= 3
                    and counts["punct"] >= 1
                    and counts["letter_total"] >= 1)

    return ScriptProfile(
        text=text,
        has_hiragana=has_hiragana,
        has_katakana=has_katakana,
        has_kanji=has_kanji,
        has_hangul=has_hangul,
        has_hanzi_only=has_hanzi_only,
        has_latin=has_latin,
        has_vietnamese_diacritic=has_viet_dia,
        has_digit=counts["digit"] > 0,
        has_punct=counts["punct"] > 0,
        primary=primary,
        counts=counts,
        is_short_sfx=is_short_sfx,
    )


def script_for_block(text: str) -> str:
    """Shortcut: classify text → primary code only ('ja'/'ko'/'zh'/'vi'/'en'/...)."""
    return classify_script(text).primary
