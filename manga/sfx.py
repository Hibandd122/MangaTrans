"""SFX detector — phân loại text region thành DIALOGUE / NARRATION / SFX."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .language import ScriptProfile, classify_script


_ACTION_SFX = {
    "bang", "boom", "crash", "smash", "thud", "thump", "whack", "pow",
    "crack", "snap", "kaboom", "blam", "wham", "thwack", "dom", "don",
    "swoosh", "whoosh", "zoom", "vroom", "dash", "dart",
    "slash", "stab", "slice", "chop", "rip", "tear",
}
_EMOTION_SFX = {
    "haha", "hehe", "hihi", "haa", "hah", "lol",
    "sigh", "phew", "huh", "hmm", "hmph",
    "wow", "yay", "argh", "ugh", "arghh", "yikes",
    "ah", "oh", "eh", "oi", "ow", "ouch", "wah", "uwah",
}
_AMBIENT_SFX = {
    "drip", "tick", "tock", "buzz", "hum", "chirp", "rustle",
    "creak", "click", "clack", "tap", "tap-tap", "whisper",
    "shhh", "hush", "rumble", "patter", "splash", "plop",
    "ring", "ding", "ringring", "ringg",
}


@dataclass
class SFXProfile:
    role: str  # 'dialogue' | 'narration' | 'sfx'
    subtype: str = "unknown"
    should_translate: bool = True
    should_preserve_pixels: bool = False
    confidence: float = 0.5
    reason: str = ""


@dataclass
class SFXDetectorConfig:
    short_token_max: int = 3
    bubble_class_sfx: int = 1
    narration_aspect_max: float = 0.6
    narration_min_chars: int = 12
    huge_size_ratio: float = 0.025


class SFXDetector:
    """Classifier per-block. Stateless ngoài config."""

    def __init__(self, config: Optional[SFXDetectorConfig] = None):
        self.config = config or SFXDetectorConfig()

    def classify(self, block: dict, ocr_text: str,
                 page_w: int, page_h: int,
                 ocr_conf: Optional[float] = None) -> SFXProfile:
        profile = classify_script(ocr_text or "")

        x1, y1, x2, y2 = block["bbox"]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        cls = block.get("cls", 0)
        bbox_area = bw * bh
        page_area = max(1, page_w * page_h)
        area_ratio = bbox_area / page_area
        aspect_hw = bh / bw

        cfg = self.config

        if cls == 0 and profile.counts and profile.counts["letter_total"] >= 2:
            return SFXProfile(
                role="dialogue",
                should_translate=True,
                confidence=0.85,
                reason="cls=bubble, có letter ≥ 2",
            )

        if cls == cfg.bubble_class_sfx:
            if aspect_hw <= cfg.narration_aspect_max and len(ocr_text) >= cfg.narration_min_chars:
                return SFXProfile(
                    role="narration",
                    should_translate=True,
                    confidence=0.7,
                    reason="cls=free-text, bbox dẹt ngang, text dài",
                )

        text_trim = (ocr_text or "").strip()
        text_letters = "".join(ch for ch in text_trim if ch.isalpha())
        is_short = len(text_letters) <= cfg.short_token_max
        is_huge_bbox = area_ratio >= cfg.huge_size_ratio
        is_pure_cjk = (profile.primary in {"ja", "zh"}
                       and not profile.has_latin)
        is_punct_heavy = profile.has_punct and profile.counts \
            and profile.counts["letter_total"] <= 3

        moderate_bbox = area_ratio >= cfg.huge_size_ratio * 0.6
        low_conf = ocr_conf is not None and ocr_conf < 0.4
        text_letter_count = len(text_letters)
        sparse_letters = moderate_bbox and text_letter_count < 4

        preserve_pixels = (is_huge_bbox
                          or is_pure_cjk
                          or (moderate_bbox and low_conf)
                          or sparse_letters)

        if cls == cfg.bubble_class_sfx and (is_pure_cjk or is_huge_bbox or sparse_letters):
            subtype = self._subtype(text_trim, profile)
            reason_parts = []
            if is_pure_cjk:
                reason_parts.append("CJK pure")
            if is_huge_bbox:
                reason_parts.append("bbox lớn")
            if sparse_letters:
                reason_parts.append("ít letter (OCR fragmented)")
            return SFXProfile(
                role="sfx",
                subtype=subtype,
                should_translate=True,
                should_preserve_pixels=preserve_pixels,
                confidence=0.75,
                reason="cls=free-text, " + " + ".join(reason_parts),
            )

        if cls == cfg.bubble_class_sfx and (is_short or is_punct_heavy):
            subtype = self._subtype(text_trim, profile)
            return SFXProfile(
                role="sfx",
                subtype=subtype,
                should_translate=True,
                should_preserve_pixels=True,
                confidence=0.65,
                reason="cls=free-text, text ngắn / punct-heavy (stylized SFX)",
            )

        if cls == 0 and is_short and profile.counts \
                and profile.counts["letter_total"] >= 1:
            return SFXProfile(
                role="dialogue",
                subtype=self._subtype(text_trim, profile),
                should_translate=True,
                confidence=0.6,
                reason="bubble + text ngắn (interjection)",
            )

        if cls == cfg.bubble_class_sfx:
            return SFXProfile(
                role="narration",
                should_translate=True,
                confidence=0.55,
                reason="cls=free-text default → narration",
            )

        return SFXProfile(
            role="dialogue",
            should_translate=True,
            confidence=0.5,
            reason="fallback dialogue",
        )

    def _subtype(self, text: str, profile: ScriptProfile) -> str:
        t = (text or "").strip().lower().strip("?!.,'\"")
        if not t:
            return "unknown"
        compact = "".join(ch for i, ch in enumerate(t)
                          if i == 0 or ch != t[i - 1])

        if (t in _EMOTION_SFX or compact in _EMOTION_SFX
                or any(t.startswith(e) for e in _EMOTION_SFX)):
            return "emotion"
        if t in _ACTION_SFX or compact in _ACTION_SFX:
            return "action"
        if t in _AMBIENT_SFX or compact in _AMBIENT_SFX:
            return "ambient"

        if profile.has_kanji and profile.has_punct:
            return "action"
        if profile.has_katakana and not profile.has_hiragana:
            return "action"
        if profile.has_hiragana:
            return "ambient"
        return "unknown"


def classify_blocks(ocr_results: list[dict],
                    blocks: list[dict],
                    page_w: int,
                    page_h: int,
                    config: Optional[SFXDetectorConfig] = None
                    ) -> list[SFXProfile]:
    """Convenience: classify mọi block. ocr_results align với blocks theo idx."""
    detector = SFXDetector(config)
    profiles: list[SFXProfile] = []
    by_idx = {r["block_idx"]: r for r in ocr_results}
    for i, blk in enumerate(blocks):
        ocr = by_idx.get(i, {"text": ""})
        profiles.append(detector.classify(blk, ocr.get("text", ""),
                                          page_w, page_h))
    return profiles
