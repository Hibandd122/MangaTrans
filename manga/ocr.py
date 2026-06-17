"""OCR routing + gibberish filter (consolidated).

PaddleOCR-only engine. PP-OCRv5 server models for high-accuracy manga text
recognition. Multi-variant preprocessing with retry on low confidence.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from .config import OCRConfig
from .language import LanguageDetection
from .utils import get_logger


# --------------------------- Gibberish filter --------------------------- #

def is_likely_gibberish(text: str, min_chars: int = 2,
                       min_letter_ratio: float = 0.4) -> bool:
    """True nếu text RÁC HOÀN TOÀN — chỉ skip nếu không có nghĩa thật."""
    if not text:
        return True
    s = text.strip()
    if len(s) < min_chars:
        return True
    n_letter = 0
    n_visible = 0
    for ch in s:
        if ch.isspace():
            continue
        n_visible += 1
        if ch.isalpha():
            n_letter += 1
            continue
        cp = ord(ch)
        if (0x3040 <= cp <= 0x30FF
                or 0x4E00 <= cp <= 0x9FFF
                or 0xAC00 <= cp <= 0xD7A3):
            n_letter += 1
    if n_letter == 0:
        return True
    if n_visible <= 4 and n_letter / max(1, n_visible) < min_letter_ratio:
        return True
    return False


# --------------------------- OCR artifact cleanup --------------------------- #

_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def _dedup_repeated_words(text: str) -> str:
    """Collapse các từ lặp liên tiếp giống nhau — OCR multi-box hay nối duplicate."""
    if not text:
        return text
    words = text.split()
    if len(words) < 2:
        return text
    out = [words[0]]
    for w in words[1:]:
        prev = out[-1]
        if len(w) > 3 and w.lower() == prev.lower():
            continue
        out.append(w)
    return " ".join(out)


def _clean_ocr_artifacts(text: str) -> str:
    """Sửa misread phổ biến: `;`/`:` → `.`/`,`; dedup từ lặp liên tiếp."""
    if not text:
        return text
    s = text.strip()
    if _CJK_RE.search(s):
        return s
    s = re.sub(r"[_]+$", ".", s)
    s = re.sub(r"[;:]+$", ".", s)
    s = re.sub(r"[;:](\\s+[a-z])", r",\\1", s)
    s = re.sub(r"[;:](\\s+[A-Z])", r".\\1", s)
    s = _dedup_repeated_words(s)
    return s


# --------------------------- Config / dataclasses --------------------------- #

@dataclass
class OCRRouterConfig:
    engine: str = "auto"
    use_paddleocr_for_latin: bool = True
    use_manga_ocr_for_ja: bool = False       # disabled — PaddleOCR only
    use_tesseract_for_en: bool = False        # disabled — PaddleOCR only
    confidence_floor: float = 0.55            # raised from 0.50 for stricter filtering
    confidence_secondary: float = 0.30
    max_retries: int = 3                      # raised from 2 — more preprocessing attempts
    paddleocr_disable_mkldnn: bool = True
    # Use server models for higher accuracy (slower but better on manga text)
    use_server_model: bool = True
    preprocess_variants: tuple[str, ...] = field(
        default_factory=lambda: ("raw", "default", "clahe", "upscale2x",
                                 "denoise+sharpen", "threshold", "rotate90"),
    )


@dataclass
class OCRResult:
    text: str
    confidence: float
    engine: str
    variant: str
    retries: int = 0


# --------------------------- Engine adapter (PaddleOCR only) --------------------------- #

class _EnginePaddleOCR:
    """Wrap PaddleOCR (PP-OCRv5). Server models for accuracy, mobile for speed.

    Lang map (paddleocr 3.x):
      en, vi (→ en, latin script)  → 'en'
      ch_sim, ch_tra, zh           → 'ch'
      japan                        → 'japan'
      korean                       → 'korean'
    """

    _LANG_MAP = {
        "en": "en", "vi": "en",
        "ch_sim": "ch", "ch_tra": "ch", "zh": "ch",
        "ja": "japan", "ko": "korean",
    }

    # Server models — higher accuracy, slower. Use for manga OCR quality.
    _DET_MODEL_SERVER = {
        "en": "PP-OCRv5_server_det",
        "ch": "PP-OCRv5_server_det",
        "japan": "PP-OCRv5_server_det",
        "korean": "PP-OCRv5_server_det",
    }
    _REC_MODEL_SERVER = {
        "en": "en_PP-OCRv5_server_rec",
        "ch": "ch_PP-OCRv5_server_rec",
    }

    # Mobile models — faster fallback
    _DET_MODEL_MOBILE = {
        "en": "PP-OCRv5_mobile_det",
        "ch": "PP-OCRv5_mobile_det",
        "japan": "PP-OCRv5_mobile_det",
        "korean": "PP-OCRv5_mobile_det",
    }
    _REC_MODEL_MOBILE = {
        "en": "en_PP-OCRv5_mobile_rec",
    }

    def __init__(self, disable_mkldnn: bool = True, use_server: bool = True):
        self._predictors: dict[str, object] = {}
        self._disable_mkldnn = disable_mkldnn
        self._use_server = use_server
        self._init_lock = threading.Lock()
        if disable_mkldnn:
            os.environ.setdefault("FLAGS_use_mkldnn", "false")
            os.environ.setdefault("FLAGS_enable_pir_in_executor", "false")

    def _pick_lang(self, langs: tuple[str, ...]) -> str:
        for l in langs:
            if l in self._LANG_MAP:
                return self._LANG_MAP[l]
        return "en"

    def _ensure(self, paddle_lang: str):
        with self._init_lock:
            if paddle_lang not in self._predictors:
                from paddleocr import PaddleOCR
                kwargs = dict(
                    lang=paddle_lang,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
                if self._use_server:
                    det_model = self._DET_MODEL_SERVER.get(paddle_lang)
                    rec_model = self._REC_MODEL_SERVER.get(paddle_lang)
                else:
                    det_model = self._DET_MODEL_MOBILE.get(paddle_lang)
                    rec_model = self._REC_MODEL_MOBILE.get(paddle_lang)
                if det_model:
                    kwargs["text_detection_model_name"] = det_model
                if self._disable_mkldnn:
                    kwargs["enable_mkldnn"] = False
                self._predictors[paddle_lang] = PaddleOCR(**kwargs)
        return self._predictors[paddle_lang]

    def read(self, image: np.ndarray, langs: tuple[str, ...]) -> tuple[str, float]:
        try:
            import paddleocr  # noqa: F401
        except ImportError as e:
            raise RuntimeError("Cần cài paddleocr") from e
        paddle_lang = self._pick_lang(langs)
        predictor = self._ensure(paddle_lang)
        result = predictor.predict(image)
        if not result:
            return "", 0.0
        item = result[0] if isinstance(result, list) else result
        if not isinstance(item, dict):
            return "", 0.0
        texts = item.get("rec_texts") or []
        scores = item.get("rec_scores") or []
        if not texts:
            return "", 0.0
        polys = item.get("rec_polys") or []
        order = list(range(len(texts)))
        if polys and len(polys) == len(texts):
            order.sort(key=lambda i: (
                float(np.mean([p[1] for p in polys[i]])),
                float(np.mean([p[0] for p in polys[i]])),
            ))
        ordered_texts = [texts[i].strip() for i in order if texts[i] and texts[i].strip()]
        text = " ".join(ordered_texts)
        conf = float(np.mean(scores)) if scores else 0.0
        return text, conf


# --------------------------- Preprocessing variants --------------------------- #

def _variant_raw(crop: np.ndarray) -> np.ndarray:
    """No preprocessing — raw crop. Often best for clean white bubbles."""
    if crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    # Only upscale if very small
    if h < 40:
        scale = min(2.0, 60.0 / max(h, 1))
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)
    return crop


def _variant_default(crop: np.ndarray) -> np.ndarray:
    """Light unsharp-mask (1.3/-0.3) + upscale if small. Gentler than before
    to reduce phantom strokes on clean manga bubbles."""
    if crop.size == 0:
        return crop
    blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(crop, 1.3, blurred, -0.3, 0)
    h, w = sharp.shape[:2]
    if h < 60:
        scale = min(2.0, 80.0 / max(h, 1))
        sharp = cv2.resize(sharp, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)
    return sharp


def _variant_clahe(crop: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement — great for low-contrast manga scans."""
    if crop.size == 0:
        return crop
    if crop.ndim == 3:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        result = clahe.apply(crop)
    h, w = result.shape[:2]
    if h < 60:
        scale = min(2.0, 80.0 / max(h, 1))
        result = cv2.resize(result, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_CUBIC)
    return result


def _variant_upscale2x(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    return cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)


def _variant_denoise_sharpen(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return crop
    denoised = cv2.fastNlMeansDenoisingColored(crop, None, 7, 7, 5, 11) \
        if crop.ndim == 3 else cv2.fastNlMeansDenoising(crop, None, 7, 5, 11)
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(denoised, 1.3, blurred, -0.3, 0)
    return _variant_upscale2x(sharp) if sharp.shape[0] < 60 else sharp


def _variant_threshold(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 31, 9)
    if crop.ndim == 3:
        th = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    return th


def _variant_rotate90(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return crop
    return cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)


_VARIANT_FN: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "raw": _variant_raw,
    "default": _variant_default,
    "clahe": _variant_clahe,
    "upscale2x": _variant_upscale2x,
    "denoise+sharpen": _variant_denoise_sharpen,
    "threshold": _variant_threshold,
    "rotate90": _variant_rotate90,
}


# --------------------------- Router (PaddleOCR only) --------------------------- #

class OCRRouter:
    """PaddleOCR-only dispatcher with multi-variant retry."""

    def __init__(self, ocr_cfg: OCRConfig,
                 router_cfg: Optional[OCRRouterConfig] = None):
        self.ocr_cfg = ocr_cfg
        self.router_cfg = router_cfg or OCRRouterConfig()
        self._log = get_logger()
        self._paddleocr: Optional[_EnginePaddleOCR] = None

    def run_blocks(self, image: np.ndarray, blocks: list[dict],
                   detection: LanguageDetection) -> list[dict]:
        """OCR mỗi block. Trả list dict tương thích pipeline cũ."""
        results: list[dict] = []
        pad = self.ocr_cfg.crop_pad
        h, w = image.shape[:2]
        n = len(blocks)
        for i, blk in enumerate(blocks):
            x1, y1, x2, y2 = blk["bbox"]
            x1 = max(0, int(x1) - pad)
            y1 = max(0, int(y1) - pad)
            x2 = min(w, int(x2) + pad)
            y2 = min(h, int(y2) + pad)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            res = self.read_crop(crop, detection)
            cleaned_text = _clean_ocr_artifacts(res.text)
            item = {
                "bbox": [x1, y1, x2, y2],
                "block_idx": i,
                "score": blk.get("score", 0.0),
                "cls": blk.get("cls", 0),
                "ocr_conf": res.confidence,
                "text": cleaned_text,
                "ocr_engine": res.engine,
                "ocr_variant": res.variant,
            }
            if not cleaned_text:
                item.update({
                    "ocr_failed": True,
                    "should_translate": False,
                    "should_preserve_pixels": True,
                })
            results.append(item)
            self._log.info(
                f"  [{i + 1}/{n}] ({x1},{y1})-({x2},{y2}) "
                f"-> {cleaned_text!r} [{res.engine}/{res.variant}, "
                f"conf={res.confidence:.2f}]"
            )
        return results

    def read_crop(self, crop: np.ndarray,
                  detection: LanguageDetection) -> OCRResult:
        """OCR 1 crop với multi-variant retry. PaddleOCR only."""
        cfg = self.router_cfg

        best: Optional[OCRResult] = None
        for variant_name in cfg.preprocess_variants[: cfg.max_retries + 1]:
            processed = _VARIANT_FN.get(variant_name, _variant_raw)(crop)
            text, conf = self._try_engine(processed, detection.langs)
            r = OCRResult(text=text, confidence=conf,
                          engine="paddleocr", variant=variant_name)
            if best is None or conf > best.confidence:
                best = r
            if conf >= cfg.confidence_floor and text:
                return r
        return best or OCRResult(text="", confidence=0.0,
                                 engine="paddleocr", variant="raw")

    def release(self) -> None:
        self._paddleocr = None

    def _try_engine(self, crop: np.ndarray,
                    langs: tuple[str, ...]) -> tuple[str, float]:
        try:
            if self._paddleocr is None:
                self._paddleocr = _EnginePaddleOCR(
                    disable_mkldnn=self.router_cfg.paddleocr_disable_mkldnn,
                    use_server=self.router_cfg.use_server_model)
            return self._paddleocr.read(crop, langs)
        except Exception as e:  # noqa: BLE001
            self._log.debug(f"   [OCRRouter] paddleocr fail: {e}")
        return "", 0.0
