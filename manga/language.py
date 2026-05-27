"""Multi-tier language detection + script classifier (consolidated).

Trước đây tách thành `language_detector.py` + `script_classifier.py`. Giờ gộp:
- `LanguageDetector` chấm điểm whole-page (preview-OCR + glyph + layout + punct).
- `classify_script` phân loại 1 đoạn text → primary script ('ja'/'ko'/'zh'/'vi'/'en'/'mixed').
- `SCRIPT_RANGES` shared Unicode block bounds cho cả hai.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .utils import get_logger


SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "hiragana": ((0x3040, 0x309F),),
    "katakana": ((0x30A0, 0x30FF), (0x31F0, 0x31FF)),
    "kanji": ((0x4E00, 0x9FFF), (0x3400, 0x4DBF)),
    "hangul": ((0xAC00, 0xD7A3), (0x1100, 0x11FF), (0x3130, 0x318F)),
    "hanzi": ((0x4E00, 0x9FFF),),
    "latin": ((0x0020, 0x007E), (0x00C0, 0x024F)),
    "vietnamese_diacritic": ((0x1E00, 0x1EFF),),
}

CJK_PUNCT = set("。、！？「」『』〝〞・…—《》〈〉【】（）")

LANG_CANDIDATES: tuple[dict, ...] = (
    {"langs": ("ja", "en"), "name": "Japanese", "code": "ja",
     "markers": ("hiragana", "katakana", "kanji")},
    {"langs": ("ko", "en"), "name": "Korean", "code": "ko",
     "markers": ("hangul",)},
    {"langs": ("ch_sim", "en"), "name": "Chinese Simplified", "code": "zh_sim",
     "markers": ("hanzi",)},
    {"langs": ("ch_tra", "en"), "name": "Chinese Traditional", "code": "zh_tra",
     "markers": ("hanzi",)},
    {"langs": ("en",), "name": "English", "code": "en",
     "markers": ("latin",)},
    {"langs": ("vi", "en"), "name": "Vietnamese", "code": "vi",
     "markers": ("latin", "vietnamese_diacritic")},
)


@dataclass
class LanguageDetectorConfig:
    max_sample_bubbles: int = 4
    bubble_vertical_ratio: float = 1.6
    native_glyph_bonus: float = 1.25
    vertical_jp_bonus: float = 1.15
    layout_threshold: float = 0.40
    cjk_punct_bonus: float = 1.10
    min_confidence_floor: float = 0.20
    enabled_candidates: tuple[str, ...] = field(
        default_factory=lambda: ("ja", "ko", "zh_sim", "zh_tra", "en", "vi"),
    )


@dataclass
class LanguageDetection:
    code: str
    name: str
    langs: tuple[str, ...]
    score: float
    raw_scores: dict[str, float] = field(default_factory=dict)
    primary_script: str = "unknown"
    mixed: bool = False
    glyph_counts: dict[str, int] = field(default_factory=dict)
    layout_hint: str = "horizontal"


class LanguageDetector:
    """Multi-tier whole-page language detector. Cache cross-page."""

    def __init__(self, config: Optional[LanguageDetectorConfig] = None):
        self.config = config or LanguageDetectorConfig()
        self._log = get_logger()
        self._cached: Optional[LanguageDetection] = None

    def detect(self, image: np.ndarray, blocks: list[dict],
               force: bool = False) -> LanguageDetection:
        if self._cached is not None and not force:
            return self._cached

        if not blocks:
            self._cached = self._fallback("en", "Không có bubble")
            return self._cached

        crops = self._extract_crops(image, blocks)
        if not crops:
            self._cached = self._fallback("en", "Không crop được bubble")
            return self._cached

        layout_hint = self._infer_layout(blocks)

        try:
            scores, glyph_counts, mixed = self._score_candidates(crops, layout_hint)
        except Exception as e:  # noqa: BLE001
            self._log.warning(f"[LangDetect] EasyOCR preview fail ({e}) → fallback en")
            self._cached = self._fallback("en", f"OCR fail: {e}")
            return self._cached

        if not scores:
            self._cached = self._fallback("en", "Không candidate nào OCR được")
            return self._cached

        best_code = max(scores, key=lambda k: scores[k])
        best_score = scores[best_code]
        cand = next(c for c in LANG_CANDIDATES if c["code"] == best_code)

        if best_score < self.config.min_confidence_floor:
            self._cached = self._fallback("en",
                                          f"Best score {best_score:.2f} < floor")
            return self._cached

        primary = self._primary_script(glyph_counts, cand["code"])
        self._cached = LanguageDetection(
            code=cand["code"],
            name=cand["name"],
            langs=cand["langs"],
            score=best_score,
            raw_scores=scores,
            primary_script=primary,
            mixed=mixed,
            glyph_counts=glyph_counts,
            layout_hint=layout_hint,
        )
        self._log.info(
            f"   [LangDetect] {cand['name']} (score={best_score:.2f}, "
            f"layout={layout_hint}, mixed={mixed})"
        )
        return self._cached

    def invalidate(self) -> None:
        self._cached = None

    @property
    def cached(self) -> Optional[LanguageDetection]:
        return self._cached

    def _extract_crops(self, image: np.ndarray,
                       blocks: list[dict]) -> list[np.ndarray]:
        n = min(self.config.max_sample_bubbles, len(blocks))
        ranked = sorted(
            blocks,
            key=lambda b: (
                -(b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]),
                -b.get("score", 0.0),
            ),
        )[:n]
        h, w = image.shape[:2]
        crops = []
        for b in ranked:
            x1, y1, x2, y2 = b["bbox"]
            x1 = max(0, int(x1) - 4)
            y1 = max(0, int(y1) - 4)
            x2 = min(w, int(x2) + 4)
            y2 = min(h, int(y2) + 4)
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
        return crops

    def _infer_layout(self, blocks: list[dict]) -> str:
        if not blocks:
            return "horizontal"
        vertical = 0
        for b in blocks:
            x1, y1, x2, y2 = b["bbox"]
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            if bh / bw >= self.config.bubble_vertical_ratio:
                vertical += 1
        ratio = vertical / len(blocks)
        return "vertical" if ratio >= self.config.layout_threshold else "horizontal"

    def _score_candidates(self, crops: list[np.ndarray],
                          layout_hint: str) -> tuple[dict[str, float],
                                                     dict[str, int], bool]:
        try:
            import easyocr
        except ImportError as e:
            raise RuntimeError("Cần cài easyocr") from e

        scores: dict[str, float] = {}
        glyph_counts: dict[str, int] = {k: 0 for k in SCRIPT_RANGES}
        enabled = set(self.config.enabled_candidates)

        for cand in LANG_CANDIDATES:
            if cand["code"] not in enabled:
                continue
            try:
                reader = easyocr.Reader(list(cand["langs"]),
                                        gpu=True, verbose=False)
            except Exception as e:  # noqa: BLE001
                self._log.debug(f"   [LangDetect] init {cand['name']} fail: {e}")
                continue

            confs: list[float] = []
            local_glyphs: dict[str, int] = {k: 0 for k in SCRIPT_RANGES}
            cjk_punct_hits = 0
            for crop in crops:
                try:
                    res = reader.readtext(crop, detail=1, paragraph=False)
                except Exception:  # noqa: BLE001
                    continue
                if not res:
                    continue
                confs.append(sum(r[2] for r in res) / len(res))
                for r in res:
                    text = r[1]
                    for ch in text:
                        if ch in CJK_PUNCT:
                            cjk_punct_hits += 1
                        for key, ranges in SCRIPT_RANGES.items():
                            cp = ord(ch)
                            for lo, hi in ranges:
                                if lo <= cp <= hi:
                                    local_glyphs[key] += 1
                                    break

            if not confs:
                continue

            base_score = sum(confs) / len(confs)
            marker_hit = sum(local_glyphs[m] for m in cand["markers"])
            if marker_hit > 0:
                base_score *= self.config.native_glyph_bonus
            if cand["code"] == "ja" and layout_hint == "vertical":
                if local_glyphs["hiragana"] + local_glyphs["katakana"] > 0:
                    base_score *= self.config.vertical_jp_bonus
            if cand["code"] in {"ja", "zh_sim", "zh_tra"} and cjk_punct_hits > 0:
                base_score *= self.config.cjk_punct_bonus

            if cand["code"] in {"zh_sim", "zh_tra"}:
                if local_glyphs["hiragana"] + local_glyphs["katakana"] > 0:
                    base_score *= 0.6

            scores[cand["code"]] = base_score

            for k, v in local_glyphs.items():
                glyph_counts[k] += v

        floor = self.config.min_confidence_floor
        n_above = sum(1 for s in scores.values() if s >= floor)
        mixed = n_above >= 2
        return scores, glyph_counts, mixed

    def _primary_script(self, glyph_counts: dict[str, int],
                        code: str) -> str:
        if code == "ja":
            if glyph_counts.get("hiragana", 0) >= glyph_counts.get("katakana", 0):
                return "hiragana+kanji"
            return "katakana+kanji"
        if code == "ko":
            return "hangul"
        if code in ("zh_sim", "zh_tra"):
            return "hanzi"
        if code == "vi":
            return "latin+diacritic"
        return "latin"

    def _fallback(self, code: str, reason: str) -> LanguageDetection:
        cand = next(c for c in LANG_CANDIDATES if c["code"] == code)
        self._log.info(f"   [LangDetect] fallback {cand['name']} — {reason}")
        return LanguageDetection(
            code=cand["code"],
            name=cand["name"],
            langs=cand["langs"],
            score=0.0,
            primary_script=self._primary_script({}, code),
            mixed=False,
        )


# --------------------------- Per-text script classifier --------------------------- #

@dataclass
class ScriptProfile:
    text: str
    has_hiragana: bool = False
    has_katakana: bool = False
    has_kanji: bool = False
    has_hangul: bool = False
    has_hanzi_only: bool = False
    has_latin: bool = False
    has_vietnamese_diacritic: bool = False
    has_digit: bool = False
    has_punct: bool = False
    primary: str = "unknown"
    counts: dict[str, int] = None
    is_short_sfx: bool = False


def _in_range(cp: int, ranges) -> bool:
    for lo, hi in ranges:
        if lo <= cp <= hi:
            return True
    return False


def classify_script(text: Optional[str], min_letter_count: int = 1) -> ScriptProfile:
    """Phân loại text → primary script ('ja'/'ko'/'zh'/'vi'/'en'/'mixed')."""
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
    return classify_script(text).primary
