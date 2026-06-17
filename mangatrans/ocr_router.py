"""OCR Router — PaddleOCR-only, PP-OCRv5 server models, multi-variant retry.

Khi user chỉ định `--ocr-engine auto`:
  Router dùng PaddleOCR làm engine duy nhất (server model cho accuracy).
  Mỗi block: chạy raw → nếu confidence < threshold → retry với preprocessing
  variants (CLAHE, denoise, threshold, rotate90).

Lazy import: PaddleOCR chỉ load khi cần.
"""
from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from .config import OCRConfig
from .language_detector import LanguageDetection
from .utils import get_logger


# Common OCR misreads cho Latin text — manga dialogue hiếm khi dùng `;` `:`.
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def _dedup_repeated_words(text: str) -> str:
    """Collapse các từ lặp liên tiếp giống nhau — OCR multi-box hay nối duplicate.

    Ví dụ: "Sorry sorry about that" (cùng từ ghép 2 box) → "Sorry about that".
    Case-insensitive match; giữ form đầu tiên (preserves cap). KHÔNG dedup nếu
    user thật sự lặp ("ha ha", "no no") — chỉ skip dedup khi từ <= 3 char.
    """
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
    """Sửa misread phổ biến: trailing `;` `:` `_` → `.`; mid-sentence `;`/`:` → `,`;
    dedup từ lặp liên tiếp.

    Heuristic chỉ áp cho Latin (không CJK). Manga dialogue rất hiếm dùng `;`
    hoặc `:`, gần như luôn là OCR lỗi của `.` / `,`.
    """
    if not text:
        return text
    s = text.strip()
    if _CJK_RE.search(s):
        return s
    # Trailing _ hoặc - (cuối câu) → .
    s = re.sub(r"[_]+$", ".", s)
    # Trailing ;/: (kể cả lặp) → .
    s = re.sub(r"[;:]+$", ".", s)
    # Mid-sentence `;`/`:` + space + lowercase → `,`
    s = re.sub(r"[;:](\\s+[a-z])", r",\\1", s)
    # Mid-sentence `;`/`:` + space + Capital → `.` (hai câu nối nhau)
    s = re.sub(r"[;:](\\s+[A-Z])", r".\\1", s)
    s = _dedup_repeated_words(s)
    return s


@dataclass
class OCRRouterConfig:
    """Cấu hình router. Bổ sung cho OCRConfig hiện có."""

    # 'auto' | 'paddleocr'
    engine: str = "auto"
    use_paddleocr_for_latin: bool = True
    use_manga_ocr_for_ja: bool = False       # disabled — PaddleOCR only
    confidence_floor: float = 0.55           # raised for stricter filtering
    confidence_secondary: float = 0.30
    max_retries: int = 3                     # raised — more preprocessing attempts
    paddleocr_disable_mkldnn: bool = True    # tránh PIR/oneDNN crash trên Windows
    # Use server models for higher accuracy
    use_server_model: bool = True
    preprocess_variants: tuple[str, ...] = field(
        default_factory=lambda: ("raw", "default", "clahe", "upscale2x",
                                 "denoise+sharpen", "threshold", "rotate90"),
    )


@dataclass
class OCRResult:
    """1 OCR output trên 1 crop."""

    text: str
    confidence: float
    engine: str
    variant: str  # preprocessing variant đã dùng
    retries: int = 0


class OCRBackendUnavailable(RuntimeError):
    """Configured OCR backend is not installed or cannot start."""


def _missing_engine_message(engine: str) -> str:
    if engine in {"paddleocr", "paddleocr_vl"}:
        return (
            "OCR backend PaddleOCR chưa sẵn sàng. "
            "Cài bằng: pip install paddleocr paddlepaddle"
        )
    return f"OCR backend {engine!r} chưa sẵn sàng."


# --------------------------- Engine adapter (PaddleOCR only) ------------ #

class _EnginePaddleOCR:
    """Wrap PaddleOCR (PP-OCRv5). Server models for accuracy. Reuse predictor per lang.

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

    def __init__(self, disable_mkldnn: bool = True, use_server: bool = True):
        self._predictor = None
        self._active_lang: Optional[str] = None
        self._disable_mkldnn = disable_mkldnn
        self._use_server = use_server
        if disable_mkldnn:
            # Phải set TRƯỚC khi import paddle — nếu paddle đã load thì flag vô hiệu
            import os
            os.environ.setdefault("FLAGS_use_mkldnn", "false")
            os.environ.setdefault("FLAGS_enable_pir_in_executor", "false")

    def _pick_lang(self, langs: tuple[str, ...]) -> str:
        for l in langs:
            if l in self._LANG_MAP:
                return self._LANG_MAP[l]
        return "en"

    # Server models — higher accuracy, optimal for manga OCR quality
    _DET_MODEL_SERVER = {
        "en": "PP-OCRv5_server_det",
        "ch": "PP-OCRv5_server_det",
        "japan": "PP-OCRv5_server_det",
        "korean": "PP-OCRv5_server_det",
    }
    _REC_MODEL_SERVER = {
        "en": "en_PP-OCRv5_server_rec",
        "ch": "ch_PP-OCRv5_server_rec",
        "japan": "japan_PP-OCRv5_server_rec",
        "korean": "korean_PP-OCRv5_server_rec",
    }

    # Mobile fallback
    _DET_MODEL_MOBILE = {
        "en": "PP-OCRv5_mobile_det",
        "ch": "PP-OCRv5_mobile_det",
        "japan": "PP-OCRv5_mobile_det",
        "korean": "PP-OCRv5_mobile_det",
    }
    _REC_MODEL_MOBILE = {
        "en": "en_PP-OCRv5_mobile_rec",
    }

    def _ensure(self, paddle_lang: str):
        if self._predictor is None or getattr(self, "_active_lang", None) != paddle_lang:
            from .paddle_cache import get_paddleocr
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
            if rec_model:
                kwargs["text_recognition_model_name"] = rec_model
            if self._disable_mkldnn:
                kwargs["enable_mkldnn"] = False
            self._predictor = get_paddleocr(**kwargs)
            self._active_lang = paddle_lang
        return self._predictor

    def read(self, image: np.ndarray, langs: tuple[str, ...]) -> tuple[str, float]:
        paddle_lang = self._pick_lang(langs)
        predictor = self._ensure(paddle_lang)
        # PaddleOCR 3.x: predict() trả list of dict
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
        # Sort theo y-coordinate (reading order). rec_polys = list polygon 4 điểm.
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
    if h < 40:
        scale = min(2.0, 60.0 / max(h, 1))
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)
    return crop


def _variant_default(crop: np.ndarray) -> np.ndarray:
    """Light unsharp-mask (1.3/-0.3) + upscale nếu nhỏ. Gentler than before
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
    # Gentler unsharp-mask — 1.3/-0.3 instead of 1.5/-0.5
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(denoised, 1.3, blurred, -0.3, 0)
    return _variant_upscale2x(sharp) if sharp.shape[0] < 60 else sharp


def _variant_threshold(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # Adaptive threshold giúp text trên scan noisy
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 31, 9)
    if crop.ndim == 3:
        th = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    return th


def _variant_rotate90(crop: np.ndarray) -> np.ndarray:
    """Xoay 90° để bắt vertical text bị detect ngang."""
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


# --------------------------- Router (PaddleOCR only) ------------------- #

class OCRRouter:
    """PaddleOCR-only dispatcher with multi-variant retry. Lazy init engine."""

    def __init__(self, ocr_cfg: OCRConfig,
                 router_cfg: Optional[OCRRouterConfig] = None):
        self.ocr_cfg = ocr_cfg
        self.router_cfg = router_cfg or OCRRouterConfig()
        self._log = get_logger()
        self._paddleocr: Optional[_EnginePaddleOCR] = None

    # --------------------------- Public --------------------------- #

    def run_blocks(self, image: np.ndarray, blocks: list[dict],
                   detection: Optional[LanguageDetection] = None) -> list[dict]:
        """OCR mỗi block. Trả list dict tương thích pipeline cũ (key 'text','ocr_conf')."""
        if detection is None:
            detection = LanguageDetection(code="ja", name="Japanese", score=1.0, langs=("ja", "en"))
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

        if not self._is_available():
            raise OCRBackendUnavailable(_missing_engine_message("paddleocr"))

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

    # --------------------------- Internals --------------------------- #

    def _is_available(self) -> bool:
        try:
            if importlib.util.find_spec("paddleocr") is None:
                return False
            return True
        except Exception:  # noqa: BLE001
            return False

    def _try_engine(self, crop: np.ndarray,
                    langs: tuple[str, ...]) -> tuple[str, float]:
        try:
            if self._paddleocr is None:
                self._paddleocr = _EnginePaddleOCR(
                    disable_mkldnn=self.router_cfg.paddleocr_disable_mkldnn,
                    use_server=self.router_cfg.use_server_model)
            return self._paddleocr.read(crop, langs)
        except (ImportError, ModuleNotFoundError, OSError) as e:
            self._log.warning(f"   [OCRRouter] paddleocr fail: {e}")
            raise OCRBackendUnavailable(_missing_engine_message("paddleocr")) from e
        except Exception as e:  # noqa: BLE001
            self._log.warning(f"   [OCRRouter] paddleocr OCR crop fail: {e}")
        return "", 0.0
