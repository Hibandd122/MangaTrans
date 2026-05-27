"""Translation pipeline — role-aware orchestrator trên Translator (translate.py).

Bổ sung so với Translator base:
  - Role-aware prompt blocks: dialogue / narration / SFX dịch theo style khác nhau.
  - Speaker continuity: track speaker hint dựa trên position (left/right side of page),
    bubble proximity & y-band cluster → suggest 1 đại từ ổn định cho mỗi cluster.
  - Honorifics preservation: keep suffix -san/-chan/-kun/-sama/-senpai trong output VI.
  - Name memory: tách glossary ra "character names" (cap name pattern) vs "terms"
    (other glossary words). Names persist với metadata first-seen page index.
  - SFX preservation policy: SFX với should_preserve_pixels → bỏ khỏi batch.
  - Slang/emotion sanitization: normalize "....." → "...", "?!?!" → "?!".

Module thuần state — không gọi network. Network handled bởi Translator.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from .translate import (
    Translator,
    extract_glossary_entries,
    position_tag,
    reading_order_indices,
)
from .utils import get_logger


HONORIFIC_TOKENS = ("-san", "-chan", "-kun", "-sama", "-senpai", "-sensei",
                    "-dono", "-tan", "-nee", "-nii")
# Phát hiện honorific để đảm bảo dịch giữ nguyên token.
HONORIFIC_RE = re.compile(
    r"\b([A-Z][a-zA-Z]{1,15})(-(?:san|chan|kun|sama|senpai|sensei|dono|tan|nee|nii))\b"
)
NAME_PATTERN = re.compile(r"\b[A-Z][a-zA-Z]{1,15}\b")


@dataclass
class TranslationPipelineConfig:
    """Cấu hình orchestrator."""

    honorifics_keep: bool = True
    name_memory_path: Optional[str] = None  # None → output_dir/.names.json
    speaker_band_ratio: float = 0.18         # band y-cluster để gom speaker turn
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
    side_hint: str = "unknown"   # 'left' | 'right' | 'unknown'  từ position_tag


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

    # --------------------------- Memory IO --------------------------- #

    def attach_memory(self, path: Optional[str]) -> None:
        """Load character memory từ JSON. Idempotent."""
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

    # --------------------------- Public API --------------------------- #

    def translate_page(self, ocr_results: list[dict],
                       page_w: int, page_h: int,
                       page_idx: int = 0) -> list[str]:
        """Translate cả page. Return list translation align ocr_results.

        Bỏ qua bubble có should_preserve_pixels=True (SFX preserve nguyên bản).
        """
        self._page_idx = page_idx
        cfg = self.config

        # Filter translate candidates
        if cfg.sfx_skip_preserve_pixels:
            kept = [i for i, r in enumerate(ocr_results)
                    if not r.get("should_preserve_pixels")]
        else:
            kept = list(range(len(ocr_results)))

        if not kept:
            return [""] * len(ocr_results)

        sub = [ocr_results[i] for i in kept]
        ro = reading_order_indices(sub, page_h)
        ordered_idx_in_kept = ro
        ordered_global_idx = [kept[k] for k in ordered_idx_in_kept]
        texts = [ocr_results[i]["text"] for i in ordered_global_idx]
        if cfg.sanitize_punctuation:
            texts = [_sanitize_punctuation(t) for t in texts]
        position_tags = [
            position_tag(ocr_results[i]["bbox"], page_w, page_h)
            for i in ordered_global_idx
        ]
        roles = [ocr_results[i].get("role", "dialogue")
                 for i in ordered_global_idx]

        # Speaker continuity hint: y-band cluster + side
        speaker_hints = self._derive_speaker_hints(
            [ocr_results[i] for i in ordered_global_idx], page_w, page_h,
        )

        # Mở rộng prompt nếu role-aware
        if cfg.role_aware_prompt:
            tagged_texts = [
                _tag_text_for_prompt(t, role, hint)
                for t, role, hint in zip(texts, roles, speaker_hints)
            ]
        else:
            tagged_texts = texts

        translations = self.translator.translate_batch(tagged_texts, position_tags)

        # Honorifics + character memory update
        if cfg.honorifics_keep:
            translations = [self._ensure_honorifics_kept(src, tgt)
                            for src, tgt in zip(texts, translations)]

        self._update_character_memory(texts, translations, position_tags,
                                      ordered_global_idx, ocr_results)

        # Align về thứ tự gốc của ocr_results
        out = [""] * len(ocr_results)
        for gidx, tr in zip(ordered_global_idx, translations):
            out[gidx] = tr
        return out

    # --------------------------- Internals --------------------------- #

    def _derive_speaker_hints(self, ordered_results: list[dict],
                              page_w: int, page_h: int) -> list[str]:
        """Cluster bubble theo y-band → assume cùng cluster là cùng cuộc thoại."""
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
            # side hint từ position
            cx = (x1 + x2) // 2
            side = "right" if cx > (page_w // 2) else "left"
            hints.append(f"turn{cluster_id}-{side}")
        return hints

    def _ensure_honorifics_kept(self, src: str, tgt: str) -> str:
        """Nếu src có 'Name-san' mà tgt drop suffix → append lại."""
        if not src or not tgt:
            return tgt
        for m in HONORIFIC_RE.finditer(src):
            full = m.group(0)        # "Yuki-san"
            name = m.group(1)         # "Yuki"
            suffix = m.group(2)       # "-san"
            if name in tgt and full not in tgt:
                # Replace bare name → name+suffix (first occurrence only)
                tgt = tgt.replace(name, name + suffix, 1)
        return tgt

    def _update_character_memory(self, sources: list[str], targets: list[str],
                                 position_tags: list[str],
                                 ordered_global_idx: list[int],
                                 ocr_results: list[dict]) -> None:
        """Hấp thụ glossary pairs có character name pattern + side hint."""
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


# --------------------------- Helpers --------------------------- #

def _sanitize_punctuation(text: str) -> str:
    """Bình thường hóa dấu lặp quá đà từ OCR (....., ?!?!?!)."""
    if not text:
        return text
    # rút gọn ... lặp: 4+ dots → 3
    out = re.sub(r"\.{4,}", "...", text)
    # 3+ ? → ??
    out = re.sub(r"\?{3,}", "??", out)
    # ?!?!?! → ?!
    out = re.sub(r"[?!]{4,}", "?!", out)
    # space trước !? trim
    out = re.sub(r"\s+([!?,.])", r"\1", out)
    return out.strip()


def _tag_text_for_prompt(text: str, role: str, speaker_hint: str) -> str:
    """Prefix role + speaker để LLM có context. Ngắn để không ăn token."""
    role_tag = {"dialogue": "DLG", "narration": "NAR", "sfx": "SFX"}.get(role, "DLG")
    return f"[{role_tag}|{speaker_hint}] {text}"
