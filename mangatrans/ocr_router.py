"""OCR Router — chọn OCR engine theo language + confidence, có retry + preprocessing.

Engines hỗ trợ:
  - paddleocr (PP-OCRv5, accuracy cao cho Latin + zh, primary cho en/vi/zh)
  - easyocr   (multi-lang, fallback hoặc primary cho ko)
  - manga-ocr (Nhật, accuracy cao cho kana+kanji vertical)
  - tesseract (binary external, fallback cho Latin in ấn rõ)

Khi user chỉ định `--ocr-engine auto`:
  Router map từ language code → preferred engine:
      ja  → manga-ocr (nếu có) > easyocr ja
      ko  → easyocr ko
      zh  → paddleocr ch > easyocr ch_sim/ch_tra
      en  → paddleocr en > easyocr en > tesseract eng
      vi  → paddleocr en (latin script) > easyocr vi
      mixed/unknown → easyocr với lang combo
  Mỗi block: chạy primary → nếu confidence < threshold → retry với preprocessing
  upscale/denoise/sharpen → nếu vẫn fail → secondary engine fallback.

Lazy import: từng engine chỉ load khi cần. Không có dep cứng vào paddleocr/manga-ocr/tesseract.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from .config import OCRConfig
from .language_detector import LanguageDetection
from .utils import get_logger


# Common EasyOCR misreads cho Latin text — manga dialogue hiếm khi dùng `;` `:`.
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
    s = re.sub(r"[;:](\s+[a-z])", r",\1", s)
    # Mid-sentence `;`/`:` + space + Capital → `.` (hai câu nối nhau)
    s = re.sub(r"[;:](\s+[A-Z])", r".\1", s)
    s = _dedup_repeated_words(s)
    return s


@dataclass
class OCRRouterConfig:
    """Cấu hình router. Bổ sung cho OCRConfig hiện có."""

    # 'auto' | 'paddleocr' | 'easyocr' | 'manga_ocr' | 'tesseract' | 'paddleocr_vl' | 'mit_48px'
    engine: str = "auto"
    use_paddleocr_for_latin: bool = True   # paddleocr primary cho en/vi/zh
    use_manga_ocr_for_ja: bool = True
    use_tesseract_for_en: bool = False
    confidence_floor: float = 0.50   # < floor → retry preprocessing
    confidence_secondary: float = 0.30  # < secondary → switch engine
    max_retries: int = 2
    tesseract_cmd: Optional[str] = None  # path tới tesseract.exe nếu cần
    paddleocr_disable_mkldnn: bool = True  # tránh PIR/oneDNN crash trên Windows
    preprocess_variants: tuple[str, ...] = field(
        default_factory=lambda: ("default", "upscale2x", "denoise+sharpen",
                                 "threshold", "rotate90"),
    )


@dataclass
class OCRResult:
    """1 OCR output trên 1 crop."""

    text: str
    confidence: float
    engine: str
    variant: str  # preprocessing variant đã dùng
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
    """Wrap PaddleOCR (PP-OCRv5). Reuse predictor per lang.

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

    def __init__(self, disable_mkldnn: bool = True):
        self._predictors: dict[str, object] = {}
        self._disable_mkldnn = disable_mkldnn
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

    # Mobile det model — đo trên chap 002: ~8s/page so với server_det ~13s
    # (~40% nhanh hơn), quality identical hoặc tốt hơn nhẹ ("flock" vs "Plock"
    # ở 1 case). Crop của manga bubble không nhỏ tới mức cần server_det.
    _DET_MODEL_BY_LANG = {
        "en": "PP-OCRv5_mobile_det",
        "ch": "PP-OCRv5_mobile_det",
        "japan": "PP-OCRv5_mobile_det",
        "korean": "PP-OCRv5_mobile_det",
    }
    _REC_MODEL_BY_LANG = {
        "en": "en_PP-OCRv5_mobile_rec",
        # Để PaddleOCR tự pick mặc định cho ch/japan/korean (đã optimize)
    }

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


class _EnginePaddleOCRVL:
    """Wrap PaddleOCR-VL-1.5 (Multimodal Vision-Language OCR)."""

    def __init__(self):
        self._predictor = None

    def _ensure(self):
        if self._predictor is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as e:
                raise RuntimeError("Cần cài paddleocr để dùng PaddleOCR-VL") from e
            self._predictor = PaddleOCR(use_vl=True, lang="en")
        return self._predictor

    def read(self, image: np.ndarray, langs: tuple[str, ...]) -> tuple[str, float]:
        predictor = self._ensure()
        result = predictor.ocr(image, cls=False)
        if not result or not result[0]:
            return "", 0.0
        
        texts = [res[1][0] for res in result[0] if res[1]]
        scores = [res[1][1] for res in result[0] if res[1]]
        if not texts:
            return "", 0.0
            
        return " ".join(texts), float(np.mean(scores))


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
        del langs  # always Japanese
        from PIL import Image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        pil = Image.fromarray(rgb)
        text = self._ensure()(pil)
        # manga-ocr không trả conf — ước lượng theo length / không-empty
        text = (text or "").strip()
        conf = 0.85 if text else 0.0
        return text, conf


class _EngineMIT48px:
    """Wrap MIT 48px OCR model cho low-res text."""

    def __init__(self):
        self._model = None

    def _ensure(self):
        if self._model is None:
            # Stub: require external MIT OCR package
            pass
        return self._model

    def read(self, image: np.ndarray, langs: tuple[str, ...]) -> tuple[str, float]:
        # Implement MIT 48px logic here once package is available
        # Currently falls back to empty
        return "", 0.0


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

        # Map EasyOCR lang codes → tesseract lang codes
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
    """Light unsharp-mask (cường độ 1.5/-0.5) + upscale nếu nhỏ. Đổi từ
    laplacian center=5 sang unsharp-mask để giảm phantom strokes — kernel cường
    độ cao tạo edge ảo → OCR thấy chữ ảo → 'thêm chữ'."""
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
    # Unsharp-mask thay laplacian — bớt phantom strokes
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)
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
        self._paddleocr_vl: Optional[_EnginePaddleOCRVL] = None
        self._easyocr: Optional[_EngineEasyOCR] = None
        self._manga_ocr: Optional[_EngineMangaOCR] = None
        self._tesseract: Optional[_EngineTesseract] = None
        self._mit_48px: Optional[_EngineMIT48px] = None
        # Track engine availability — đánh dấu False sau lần đầu fail load
        self._available = {"paddleocr": None, "easyocr": None,
                           "manga_ocr": None, "tesseract": None,
                           "paddleocr_vl": None, "mit_48px": None}

    # --------------------------- Public --------------------------- #

    def run_blocks(self, image: np.ndarray, blocks: list[dict],
                   detection: Optional[LanguageDetection] = None) -> list[dict]:
        """OCR mỗi block. Trả list dict tương thích pipeline cũ (key 'text','ocr_conf')."""
        if detection is None:
            detection = LanguageDetection(code="ja", confidence=1.0, langs=("ja", "en"))
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
            # Nếu primary fail sâu → thử secondary trên cùng variant
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
        self._paddleocr_vl = None
        self._easyocr = None
        self._manga_ocr = None
        self._tesseract = None
        self._mit_48px = None

    # --------------------------- Internals --------------------------- #

    def _choose_engines(self, detection: LanguageDetection
                        ) -> tuple[str, Optional[str]]:
        """Return (primary, secondary) engine names."""
        cfg = self.router_cfg
        if cfg.engine != "auto":
            return cfg.engine, "easyocr" if cfg.engine != "easyocr" else None

        code = detection.code
        # Japanese: manga-ocr > paddleocr japan > easyocr
        if code == "ja" and cfg.use_manga_ocr_for_ja and self._is_available("manga_ocr"):
            return "manga_ocr", "easyocr"
        # Latin (en/vi) or zh: paddleocr primary nếu available
        if code in ("en", "vi", "zh") and cfg.use_paddleocr_for_latin \
                and self._is_available("paddleocr"):
            return "paddleocr", "easyocr"
        if code == "en" and cfg.use_tesseract_for_en and self._is_available("tesseract"):
            return "tesseract", "easyocr"
        
        # Override bằng paddleocr_vl nếu backend được set explicit
        if cfg.engine == "paddleocr_vl" and self._is_available("paddleocr_vl"):
            return "paddleocr_vl", "easyocr"
        if cfg.engine == "mit_48px" and self._is_available("mit_48px"):
            return "mit_48px", "easyocr"

        # Mọi case khác: easyocr primary, paddleocr secondary nếu available
        if self._is_available("paddleocr"):
            return "easyocr", "paddleocr"
        secondary = "tesseract" if self._is_available("tesseract") else None
        return "easyocr", secondary

    def _is_available(self, engine: str) -> bool:
        if self._available[engine] is not None:
            return self._available[engine]
        try:
            if engine in ("paddleocr", "paddleocr_vl"):
                __import__("paddleocr")
            elif engine == "manga_ocr":
                __import__("manga_ocr")
            elif engine == "tesseract":
                __import__("pytesseract")
            elif engine == "easyocr":
                __import__("easyocr")
            elif engine == "mit_48px":
                pass # check logic cho mit_48px package
            self._available[engine] = True
        except ImportError:
            self._available[engine] = False
            self._log.debug(f"   [OCRRouter] {engine} không cài → disable")
        except Exception as e:  # noqa: BLE001 — paddleocr có thể crash khi import
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
            if name == "paddleocr_vl":
                if self._paddleocr_vl is None:
                    self._paddleocr_vl = _EnginePaddleOCRVL()
                return self._paddleocr_vl.read(crop, langs)
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
            if name == "mit_48px":
                if self._mit_48px is None:
                    self._mit_48px = _EngineMIT48px()
                return self._mit_48px.read(crop, langs)
        except Exception as e:  # noqa: BLE001
            self._log.debug(f"   [OCRRouter] engine {name} fail: {e}")
        return "", 0.0
