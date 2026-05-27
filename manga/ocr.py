"""OCR routing + gibberish filter (consolidated).

Gộp `ocr.py` (gibberish) + `ocr_router.py` (multi-engine dispatcher).
Engines: PaddleOCR (Latin/zh), EasyOCR (multi-lang), manga-ocr (Japanese),
Tesseract (Latin fallback). Lazy import — không dep cứng vào engine ngoài.
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
    s = re.sub(r"[;:](\s+[a-z])", r",\1", s)
    s = re.sub(r"[;:](\s+[A-Z])", r".\1", s)
    s = _dedup_repeated_words(s)
    return s


# --------------------------- Config / dataclasses --------------------------- #

@dataclass
class OCRRouterConfig:
    engine: str = "auto"
    use_paddleocr_for_latin: bool = True
    use_manga_ocr_for_ja: bool = True
    use_tesseract_for_en: bool = False
    confidence_floor: float = 0.50
    confidence_secondary: float = 0.30
    max_retries: int = 2
    tesseract_cmd: Optional[str] = None
    paddleocr_disable_mkldnn: bool = True
    preprocess_variants: tuple[str, ...] = field(
        default_factory=lambda: ("default", "upscale2x", "denoise+sharpen",
                                 "threshold", "rotate90"),
    )


@dataclass
class OCRResult:
    text: str
    confidence: float
    engine: str
    variant: str
    retries: int = 0


# --------------------------- Engine adapters --------------------------- #

class _EngineEasyOCR:
    """Wrap EasyOCR. Reuse reader instance cho từng lang combo."""

    def __init__(self, gpu: bool = True):
        self._readers: dict[tuple[str, ...], object] = {}
        self.gpu = gpu

    def read(self, image: np.ndarray, langs: tuple[str, ...]) -> tuple[str, float]:
        try:
            import easyocr
        except ImportError as e:
            raise RuntimeError("Cần cài easyocr") from e
        if langs not in self._readers:
            self._readers[langs] = easyocr.Reader(list(langs),
                                                  gpu=self.gpu, verbose=False)
        reader = self._readers[langs]
        results = reader.readtext(image, detail=1, paragraph=False)
        if not results:
            return "", 0.0
        results.sort(key=lambda r: (
            float(np.mean([p[1] for p in r[0]])),
            float(np.mean([p[0] for p in r[0]])),
        ))
        text = " ".join(r[1].strip() for r in results if r[1].strip())
        conf = float(np.mean([r[2] for r in results]))
        return text, conf


class _EnginePaddleOCR:
    """Wrap PaddleOCR (PP-OCRv5). Reuse predictor per lang."""

    _LANG_MAP = {
        "en": "en", "vi": "en",
        "ch_sim": "ch", "ch_tra": "ch", "zh": "ch",
        "ja": "japan", "ko": "korean",
    }

    _DET_MODEL_BY_LANG = {
        "en": "PP-OCRv5_mobile_det",
        "ch": "PP-OCRv5_mobile_det",
        "japan": "PP-OCRv5_mobile_det",
        "korean": "PP-OCRv5_mobile_det",
    }
    _REC_MODEL_BY_LANG = {
        "en": "en_PP-OCRv5_mobile_rec",
    }

    def __init__(self, disable_mkldnn: bool = True):
        self._predictors: dict[str, object] = {}
        self._disable_mkldnn = disable_mkldnn
        if disable_mkldnn:
            os.environ.setdefault("FLAGS_use_mkldnn", "false")
            os.environ.setdefault("FLAGS_enable_pir_in_executor", "false")

    def _pick_lang(self, langs: tuple[str, ...]) -> str:
        for l in langs:
            if l in self._LANG_MAP:
                return self._LANG_MAP[l]
        return "en"

    def _ensure(self, paddle_lang: str):
        if paddle_lang not in self._predictors:
            from paddleocr import PaddleOCR
            kwargs = dict(
                lang=paddle_lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            det_model = self._DET_MODEL_BY_LANG.get(paddle_lang)
            if det_model:
                kwargs["text_detection_model_name"] = det_model
            rec_model = self._REC_MODEL_BY_LANG.get(paddle_lang)
            if rec_model:
                kwargs["text_recognition_model_name"] = rec_model
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


class _EngineMangaOCR:
    """Wrap manga-ocr. Chuyên Nhật, accuracy cao trên kana + kanji + vertical."""

    def __init__(self):
        self._reader = None

    def _ensure(self):
        if self._reader is None:
            from manga_ocr import MangaOcr
            self._reader = MangaOcr()
        return self._reader

    def read(self, image: np.ndarray, langs: tuple[str, ...]) -> tuple[str, float]:
        del langs
        from PIL import Image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        pil = Image.fromarray(rgb)
        text = self._ensure()(pil)
        text = (text or "").strip()
        conf = 0.85 if text else 0.0
        return text, conf


class _EngineTesseract:
    """Wrap pytesseract. Hữu ích cho Latin in ấn rõ."""

    def __init__(self, tesseract_cmd: Optional[str] = None):
        self._cmd = tesseract_cmd

    def read(self, image: np.ndarray, langs: tuple[str, ...]) -> tuple[str, float]:
        try:
            import pytesseract
        except ImportError as e:
            raise RuntimeError("Cần cài pytesseract") from e
        if self._cmd:
            pytesseract.pytesseract.tesseract_cmd = self._cmd

        tess_map = {"en": "eng", "ja": "jpn", "ko": "kor",
                    "ch_sim": "chi_sim", "ch_tra": "chi_tra", "vi": "vie"}
        tess_langs = "+".join(tess_map.get(l, "eng") for l in langs)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        try:
            data = pytesseract.image_to_data(gray, lang=tess_langs,
                                             output_type=pytesseract.Output.DICT)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Tesseract fail: {e}") from e

        words = [w for w in data["text"] if w.strip()]
        confs = [int(c) for c, w in zip(data["conf"], data["text"])
                 if w.strip() and str(c).lstrip("-").isdigit() and int(c) >= 0]
        text = " ".join(words)
        conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
        return text, conf


# --------------------------- Preprocessing variants --------------------------- #

def _variant_default(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return crop
    blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(crop, 1.5, blurred, -0.5, 0)
    h, w = sharp.shape[:2]
    if h < 60:
        scale = min(2.0, 80.0 / max(h, 1))
        sharp = cv2.resize(sharp, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)
    return sharp


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
    sharp = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)
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
    "default": _variant_default,
    "upscale2x": _variant_upscale2x,
    "denoise+sharpen": _variant_denoise_sharpen,
    "threshold": _variant_threshold,
    "rotate90": _variant_rotate90,
}


# --------------------------- Router --------------------------- #

class OCRRouter:
    """Dispatch theo language + confidence. Lazy init engines."""

    def __init__(self, ocr_cfg: OCRConfig,
                 router_cfg: Optional[OCRRouterConfig] = None):
        self.ocr_cfg = ocr_cfg
        self.router_cfg = router_cfg or OCRRouterConfig()
        self._log = get_logger()
        self._paddleocr: Optional[_EnginePaddleOCR] = None
        self._easyocr: Optional[_EngineEasyOCR] = None
        self._manga_ocr: Optional[_EngineMangaOCR] = None
        self._tesseract: Optional[_EngineTesseract] = None
        self._available = {"paddleocr": None, "easyocr": None,
                           "manga_ocr": None, "tesseract": None}

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
            results.append({
                "bbox": [x1, y1, x2, y2],
                "block_idx": i,
                "score": blk.get("score", 0.0),
                "cls": blk.get("cls", 0),
                "ocr_conf": res.confidence,
                "text": cleaned_text,
                "ocr_engine": res.engine,
                "ocr_variant": res.variant,
            })
            self._log.info(
                f"  [{i + 1}/{n}] ({x1},{y1})-({x2},{y2}) "
                f"-> {cleaned_text!r} [{res.engine}/{res.variant}, "
                f"conf={res.confidence:.2f}]"
            )
        return results

    def read_crop(self, crop: np.ndarray,
                  detection: LanguageDetection) -> OCRResult:
        """OCR 1 crop với routing + retry."""
        primary, secondary = self._choose_engines(detection)
        cfg = self.router_cfg

        best: Optional[OCRResult] = None
        for variant_name in cfg.preprocess_variants[: cfg.max_retries + 1]:
            processed = _VARIANT_FN.get(variant_name, _variant_default)(crop)
            text, conf = self._try_engine(primary, processed, detection.langs)
            r = OCRResult(text=text, confidence=conf,
                          engine=primary, variant=variant_name)
            if best is None or conf > best.confidence:
                best = r
            if conf >= cfg.confidence_floor and text:
                return r
            if conf < cfg.confidence_secondary and secondary and secondary != primary:
                text2, conf2 = self._try_engine(secondary, processed, detection.langs)
                if conf2 > best.confidence:
                    best = OCRResult(text=text2, confidence=conf2,
                                     engine=secondary, variant=variant_name)
                if conf2 >= cfg.confidence_floor and text2:
                    return best
        return best or OCRResult(text="", confidence=0.0,
                                 engine=primary, variant="default")

    def release(self) -> None:
        self._paddleocr = None
        self._easyocr = None
        self._manga_ocr = None
        self._tesseract = None

    def _choose_engines(self, detection: LanguageDetection
                        ) -> tuple[str, Optional[str]]:
        cfg = self.router_cfg
        if cfg.engine != "auto":
            return cfg.engine, "easyocr" if cfg.engine != "easyocr" else None

        code = detection.code
        if code == "ja" and cfg.use_manga_ocr_for_ja and self._is_available("manga_ocr"):
            return "manga_ocr", "easyocr"
        if code in ("en", "vi", "zh") and cfg.use_paddleocr_for_latin \
                and self._is_available("paddleocr"):
            return "paddleocr", "easyocr"
        if code == "en" and cfg.use_tesseract_for_en and self._is_available("tesseract"):
            return "tesseract", "easyocr"
        if self._is_available("paddleocr"):
            return "easyocr", "paddleocr"
        secondary = "tesseract" if self._is_available("tesseract") else None
        return "easyocr", secondary

    def _is_available(self, engine: str) -> bool:
        if self._available[engine] is not None:
            return self._available[engine]
        try:
            if engine == "paddleocr":
                __import__("paddleocr")
            elif engine == "manga_ocr":
                __import__("manga_ocr")
            elif engine == "tesseract":
                __import__("pytesseract")
            elif engine == "easyocr":
                __import__("easyocr")
            self._available[engine] = True
        except ImportError:
            self._available[engine] = False
            self._log.debug(f"   [OCRRouter] {engine} không cài → disable")
        except Exception as e:  # noqa: BLE001
            self._available[engine] = False
            self._log.debug(f"   [OCRRouter] {engine} import lỗi: {e} → disable")
        return self._available[engine]

    def _try_engine(self, name: str, crop: np.ndarray,
                    langs: tuple[str, ...]) -> tuple[str, float]:
        try:
            if name == "paddleocr":
                if self._paddleocr is None:
                    self._paddleocr = _EnginePaddleOCR(
                        disable_mkldnn=self.router_cfg.paddleocr_disable_mkldnn)
                return self._paddleocr.read(crop, langs)
            if name == "easyocr":
                if self._easyocr is None:
                    self._easyocr = _EngineEasyOCR(gpu=self.ocr_cfg.gpu)
                return self._easyocr.read(crop, langs)
            if name == "manga_ocr":
                if self._manga_ocr is None:
                    self._manga_ocr = _EngineMangaOCR()
                return self._manga_ocr.read(crop, langs)
            if name == "tesseract":
                if self._tesseract is None:
                    self._tesseract = _EngineTesseract(self.router_cfg.tesseract_cmd)
                return self._tesseract.read(crop, langs)
        except Exception as e:  # noqa: BLE001
            self._log.debug(f"   [OCRRouter] engine {name} fail: {e}")
        return "", 0.0
