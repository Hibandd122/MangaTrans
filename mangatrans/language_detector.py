"""Multi-tier language detector cho manga page.

3 tầng signal kết hợp để chọn ngôn ngữ gốc của trang:
  1. Glyph signal — preview-OCR mỗi candidate lang combo trên sample bubble,
     đếm non-ASCII glyph + confidence trung bình (tầng mạnh nhất, có dấu hiệu thực).
  2. Layout signal — tỷ lệ bubble vertical (h/w >= 1.6) gợi ý Japanese tateshou
     reading; cluster bbox dạng cột gợi ý CJK; bbox ngang đều gợi ý Latin/Korean.
  3. Punctuation signal — preview OCR text scan dấu CJK fullwidth ('。', '、',
     '「', '」', '？', '！') vs Latin ('.', ',', '?', '!').

Mỗi tier vote một set langs; final pick = candidate có total weighted score cao nhất.
Khi không signal nào > confidence floor, fallback 'en'.

Cache instance-level: mỗi page reuse kết quả của trang đầu (manga 1 chap thường
cùng ngôn ngữ). Manual `.invalidate()` để force re-detect khi chuyển chap.

KHÔNG dùng external lang lib (fasttext/langdetect) — input là crops manga, text
quá ngắn và noisy → ML lang ID dễ false. Heuristic visual + glyph ổn định hơn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .utils import get_logger


# Unicode block bounds cho từng script. Range mở để cover edge cases.
SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "hiragana": ((0x3040, 0x309F),),
    "katakana": ((0x30A0, 0x30FF), (0x31F0, 0x31FF)),
    "kanji": ((0x4E00, 0x9FFF), (0x3400, 0x4DBF)),  # CJK Unified + Ext A
    "hangul": ((0xAC00, 0xD7A3), (0x1100, 0x11FF), (0x3130, 0x318F)),
    "hanzi": ((0x4E00, 0x9FFF),),  # overlap kanji — phân biệt qua presence of kana
    "latin": ((0x0020, 0x007E), (0x00C0, 0x024F)),
    "vietnamese_diacritic": ((0x1E00, 0x1EFF),),
}

# CJK fullwidth punctuation set
CJK_PUNCT = set("。、！？「」『』〝〞・…—《》〈〉【】（）")

# Candidate language combos cho EasyOCR preview. Mỗi combo có:
#   - langs: tuple lang codes EasyOCR
#   - name: hiển thị cho log
#   - markers: script keys liên quan → matched glyph cộng điểm
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
    """Cấu hình detection. Tweak qua PipelineConfig nếu cần."""

    max_sample_bubbles: int = 4         # crops đem ra preview OCR
    bubble_vertical_ratio: float = 1.6  # h/w >= 1.6 = vertical bubble (Nhật)
    native_glyph_bonus: float = 1.25    # nhân conf khi candidate match native glyph
    vertical_jp_bonus: float = 1.15     # bonus Japanese khi >40% bubble vertical
    layout_threshold: float = 0.40
    cjk_punct_bonus: float = 1.10
    min_confidence_floor: float = 0.20  # dưới mức này → fallback en
    enabled_candidates: tuple[str, ...] = field(
        default_factory=lambda: ("ja", "ko", "zh_sim", "zh_tra", "en", "vi"),
    )


@dataclass
class LanguageDetection:
    """Kết quả detect."""

    code: str                          # 'ja' | 'ko' | 'zh_sim' | 'zh_tra' | 'en' | 'vi'
    name: str
    langs: tuple[str, ...]             # EasyOCR lang list
    score: float                       # tổng score sau bonus
    raw_scores: dict[str, float] = field(default_factory=dict)
    primary_script: str = "unknown"    # tag thuận tiện cho downstream (ocr_router)
    mixed: bool = False                # >1 language detected vượt floor
    glyph_counts: dict[str, int] = field(default_factory=dict)
    layout_hint: str = "horizontal"    # 'vertical' | 'horizontal'


class LanguageDetector:
    """Multi-tier language detector. Cache cross-page.

    Thread-safe: lock quanh `_cached` cho async pipeline có thể gọi detect()
    từ nhiều page concurrent (GPU mutex serialize Paddle/EasyOCR nhưng cache
    write vẫn cần atomic).
    """

    def __init__(self, config: Optional[LanguageDetectorConfig] = None):
        self.config = config or LanguageDetectorConfig()
        self._log = get_logger()
        self._cached: Optional[LanguageDetection] = None
        import threading as _threading
        self._lock = _threading.Lock()

    # --------------------------- Public API --------------------------- #

    def detect(self, image: np.ndarray, blocks: list[dict],
               force: bool = False) -> LanguageDetection:
        """Detect ngôn ngữ chính của page.

        Args:
            image: BGR full page.
            blocks: list bubble bbox dict ({'bbox': [x1,y1,x2,y2], 'score': float}).
            force: bỏ qua cache.
        """
        with self._lock:
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

        # Try EasyOCR-based glyph detection. Failure → fallback heuristic.
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
        """Force re-detect ở lần `detect()` tiếp theo (vd qua chap mới)."""
        with self._lock:
            self._cached = None

    @property
    def cached(self) -> Optional[LanguageDetection]:
        return self._cached

    # --------------------------- Internals --------------------------- #

    def _extract_crops(self, image: np.ndarray,
                       blocks: list[dict]) -> list[np.ndarray]:
        n = min(self.config.max_sample_bubbles, len(blocks))
        # Pick bubble lớn + score cao — text rõ ràng nhất
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
        """Tỉ lệ bubble dạng cột (h/w >= threshold) — gợi ý vertical Nhật."""
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
        """Preview OCR mỗi candidate combo → glyph + conf score.

        Trả (scores_per_code, glyph_counts_across_all, mixed_flag).
        """
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise RuntimeError("Cần cài paddleocr") from e

        scores: dict[str, float] = {}
        glyph_counts: dict[str, int] = {k: 0 for k in SCRIPT_RANGES}
        cand_text_glyph: dict[str, dict[str, int]] = {}
        enabled = set(self.config.enabled_candidates)

        paddle_lang_map = {
            "ja": "japan",
            "ko": "korean",
            "zh_sim": "ch",
            "zh_tra": "chinese_cht",
            "en": "en",
            "vi": "latin",
        }

        for cand in LANG_CANDIDATES:
            if cand["code"] not in enabled:
                continue
            plang = paddle_lang_map.get(cand["code"])
            if not plang:
                continue

            try:
                reader = PaddleOCR(use_angle_cls=True, lang=plang, show_log=False)
            except Exception as e:  # noqa: BLE001
                self._log.debug(f"   [LangDetect] init {cand['name']} fail: {e}")
                continue

            confs: list[float] = []
            local_glyphs: dict[str, int] = {k: 0 for k in SCRIPT_RANGES}
            cjk_punct_hits = 0
            for crop in crops:
                try:
                    res = reader.ocr(crop, cls=True)
                except Exception:  # noqa: BLE001
                    continue
                if not res or not res[0]:
                    continue
                
                lines = res[0]
                confs.append(sum(r[1][1] for r in lines) / len(lines))
                for r in lines:
                    text = r[1][0]
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
            # Bonus khi candidate match script markers
            marker_hit = sum(local_glyphs[m] for m in cand["markers"])
            if marker_hit > 0:
                base_score *= self.config.native_glyph_bonus
            # Bonus Japanese khi layout vertical & có kana
            if cand["code"] == "ja" and layout_hint == "vertical":
                if local_glyphs["hiragana"] + local_glyphs["katakana"] > 0:
                    base_score *= self.config.vertical_jp_bonus
            # Bonus CJK candidate khi punctuation match
            if cand["code"] in {"ja", "zh_sim", "zh_tra"} and cjk_punct_hits > 0:
                base_score *= self.config.cjk_punct_bonus

            # Phân biệt zh (no kana) vs ja (có kana mà có kanji thì vẫn ja)
            if cand["code"] in {"zh_sim", "zh_tra"}:
                if local_glyphs["hiragana"] + local_glyphs["katakana"] > 0:
                    base_score *= 0.6  # có kana mà claim Chinese → penalize

            cand_text_glyph[cand["code"]] = local_glyphs
            scores[cand["code"]] = base_score

            for k, v in local_glyphs.items():
                glyph_counts[k] += v

        # Mixed flag: hơn 1 candidate vượt floor
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
