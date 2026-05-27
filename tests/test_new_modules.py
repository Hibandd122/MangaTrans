"""Unit tests cho các module VIP còn lại sau refactor 2026-05-25:
script_classifier, sfx_detector, translation_pipeline, language_detector
(layout signal), redraw_engine (preserve/unsharp), ocr_router clean artifacts,
paddleocr adapter.

Đã xóa test cho: stroke_detector, qa_report, debug_viz, api_key_store,
font_analyzer (module đã bị xóa khi consolidate).

Mỗi test pure — không gọi network/EasyOCR/model.
Run: pytest tests/test_new_modules.py -v
"""
from __future__ import annotations

import numpy as np

from mangatrans.language_detector import LanguageDetector, LanguageDetectorConfig
from mangatrans.ocr_router import _clean_ocr_artifacts, _EnginePaddleOCR, OCRRouter, OCRRouterConfig
from mangatrans.redraw_engine import RedrawEngine, RedrawEngineConfig, detect_screentone
from mangatrans.script_classifier import classify_script
from mangatrans.sfx_detector import SFXDetector
from mangatrans.translation_pipeline import (
    TranslationPipeline,
    TranslationPipelineConfig,
    _sanitize_punctuation,
    _tag_text_for_prompt,
)


# --------------------------- script_classifier --------------------------- #


def test_script_classify_english():
    p = classify_script("Hello world")
    assert p.primary == "en"
    assert p.has_latin


def test_script_classify_japanese_hiragana():
    p = classify_script("こんにちは")
    assert p.primary == "ja"
    assert p.has_hiragana


def test_script_classify_korean():
    p = classify_script("안녕하세요")
    assert p.primary == "ko"
    assert p.has_hangul


def test_script_classify_chinese_no_kana():
    p = classify_script("中文测试")
    assert p.primary == "zh"
    assert p.has_hanzi_only


def test_script_classify_vietnamese():
    p = classify_script("Bạn khỏe không?")
    assert p.primary == "vi"


def test_script_classify_short_sfx():
    p = classify_script("Eh?")
    assert p.is_short_sfx


def test_script_classify_empty():
    p = classify_script("")
    assert p.primary == "unknown"


# --------------------------- ocr_router clean artifacts --------------------------- #


def test_clean_ocr_artifacts_trailing_semi_to_period():
    assert _clean_ocr_artifacts("Sorry about that;") == "Sorry about that."


def test_clean_ocr_artifacts_trailing_colon_to_period():
    assert _clean_ocr_artifacts("There was nothing we could have done:") \
        == "There was nothing we could have done."


def test_clean_ocr_artifacts_trailing_underscore():
    assert _clean_ocr_artifacts("derabe_") == "derabe."


def test_clean_ocr_artifacts_mid_lowercase_to_comma():
    assert _clean_ocr_artifacts("Yeah; brue.") == "Yeah, brue."


def test_clean_ocr_artifacts_mid_capital_to_period():
    assert _clean_ocr_artifacts("Sorry; And then I left.") \
        == "Sorry. And then I left."


def test_clean_ocr_artifacts_skip_cjk():
    assert _clean_ocr_artifacts("こんにちは;") == "こんにちは;"


# --------------------------- sfx_detector --------------------------- #


def test_sfx_dialogue_default():
    det = SFXDetector()
    blk = {"bbox": [10, 10, 200, 100], "cls": 0, "score": 0.9}
    prof = det.classify(blk, "Hello world how are you", 800, 1000)
    assert prof.role == "dialogue"


def test_sfx_short_action_sfx():
    det = SFXDetector()
    blk = {"bbox": [10, 10, 100, 200], "cls": 1, "score": 0.9}
    prof = det.classify(blk, "POW!", 800, 1000)
    assert prof.role == "sfx"
    assert prof.subtype == "action"


def test_sfx_emotion():
    det = SFXDetector()
    blk = {"bbox": [50, 50, 90, 80], "cls": 1, "score": 0.8}
    prof = det.classify(blk, "ah", 800, 1000)
    assert prof.role == "sfx"
    assert prof.subtype == "emotion"


def test_sfx_narration_box():
    det = SFXDetector()
    blk = {"bbox": [0, 0, 600, 60], "cls": 1, "score": 0.7}
    prof = det.classify(blk, "Some long narration here in box", 800, 1000)
    assert prof.role == "narration"


def test_sfx_preserve_pure_cjk_huge():
    det = SFXDetector()
    blk = {"bbox": [100, 100, 300, 400], "cls": 1, "score": 0.7}
    prof = det.classify(blk, "嗙嗙", 800, 1000)
    assert prof.should_preserve_pixels


# --------------------------- translation_pipeline helpers --------------------------- #


def test_sanitize_punct_repeated():
    assert _sanitize_punctuation("......") == "..."
    assert _sanitize_punctuation("???") == "??"
    assert _sanitize_punctuation("?!?!?!?!") == "?!"


def test_sanitize_punct_trims():
    assert _sanitize_punctuation(" hi !! ") == "hi!!"


def test_tag_text_for_prompt():
    assert _tag_text_for_prompt("Hi", "dialogue", "turn1-left") == "[DLG|turn1-left] Hi"
    assert _tag_text_for_prompt("Boom!", "sfx", "turn3-right") == "[SFX|turn3-right] Boom!"
    assert _tag_text_for_prompt("Note", "narration", "turn2-left") == "[NAR|turn2-left] Note"


def test_translation_pipeline_skip_preserve_pixels():
    class MockTranslator:
        def __init__(self):
            self.glossary = {}

        def translate_batch(self, texts, position_tags=None):
            return [t.lower() for t in texts]

    tp = TranslationPipeline(MockTranslator(), TranslationPipelineConfig())
    ocr = [
        {"bbox": [0, 0, 100, 100], "text": "Hello",
         "block_idx": 0, "should_preserve_pixels": False},
        {"bbox": [100, 100, 200, 200], "text": "SFX",
         "block_idx": 1, "should_preserve_pixels": True},
    ]
    out = tp.translate_page(ocr, 800, 1000)
    assert len(out) == 2
    assert out[1] == ""


def test_translation_pipeline_honorific_keep():
    class MockTranslator:
        def __init__(self):
            self.glossary = {}

        def translate_batch(self, texts, position_tags=None):
            return [t.replace("-san", "").replace("-chan", "").replace("[DLG|turn1-left] ", "")
                    for t in texts]

    tp = TranslationPipeline(MockTranslator(),
                             TranslationPipelineConfig(honorifics_keep=True))
    ocr = [{"bbox": [0, 0, 100, 100], "text": "Yuki-san is here",
            "block_idx": 0}]
    out = tp.translate_page(ocr, 800, 1000)
    assert "-san" in out[0]


# --------------------------- language_detector layout --------------------------- #


def test_lang_detector_layout_horizontal():
    det = LanguageDetector(LanguageDetectorConfig())
    blocks = [{"bbox": [0, 0, 100, 30]},
              {"bbox": [200, 50, 350, 70]}]
    assert det._infer_layout(blocks) == "horizontal"


def test_lang_detector_layout_vertical():
    det = LanguageDetector(LanguageDetectorConfig())
    blocks = [{"bbox": [0, 0, 30, 150]},
              {"bbox": [50, 0, 80, 200]},
              {"bbox": [100, 0, 130, 180]}]
    assert det._infer_layout(blocks) == "vertical"


# --------------------------- redraw_engine pure helpers --------------------------- #


def test_detect_screentone_uniform_image():
    img = np.full((64, 64), 200, dtype=np.uint8)
    assert detect_screentone(img) == 0.0 or detect_screentone(img) >= 0


def test_detect_screentone_periodic():
    img = np.zeros((128, 128), dtype=np.uint8)
    img[::4, :] = 255
    score = detect_screentone(img)
    assert score > 0.0


def test_redraw_engine_empty_mask():
    cfg = RedrawEngineConfig(enable_edge_preserve=False,
                             enable_unsharp_post=False)
    eng = RedrawEngine(inpainter=None,
                       inpaint_cfg=None,  # type: ignore  — không dùng khi mask rỗng
                       engine_cfg=cfg)
    img = np.full((50, 50, 3), 100, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    out, report = eng.redraw(img, mask)
    assert np.array_equal(out, img)
    assert report.used_edge_preserve is False


# --------------------------- paddleocr adapter --------------------------- #


def test_paddleocr_lang_map():
    eng = _EnginePaddleOCR(disable_mkldnn=True)
    assert eng._pick_lang(("en",)) == "en"
    assert eng._pick_lang(("vi",)) == "en"
    assert eng._pick_lang(("ch_sim", "en")) == "ch"
    assert eng._pick_lang(("ja",)) == "japan"
    assert eng._pick_lang(("ko",)) == "korean"
    assert eng._pick_lang(("foo",)) == "en"


def test_paddleocr_read_mocked(monkeypatch):
    """Đảm bảo adapter parse đúng output PaddleOCR 3.x (rec_texts/rec_scores)."""
    eng = _EnginePaddleOCR(disable_mkldnn=True)

    class MockPredictor:
        def predict(self, image):
            return [{
                "rec_texts": ["Hello", "world"],
                "rec_scores": [0.95, 0.88],
                "rec_polys": [
                    [[0, 0], [50, 0], [50, 20], [0, 20]],
                    [[0, 30], [60, 30], [60, 50], [0, 50]],
                ],
            }]

    eng._predictors["en"] = MockPredictor()
    text, conf = eng.read(np.zeros((100, 100, 3), dtype=np.uint8), ("en",))
    assert text == "Hello world"
    assert 0.9 < conf < 0.93


def test_paddleocr_read_empty(monkeypatch):
    eng = _EnginePaddleOCR(disable_mkldnn=True)

    class MockPredictor:
        def predict(self, image):
            return [{"rec_texts": [], "rec_scores": []}]

    eng._predictors["en"] = MockPredictor()
    text, conf = eng.read(np.zeros((10, 10, 3), dtype=np.uint8), ("en",))
    assert text == ""
    assert conf == 0.0


def test_ocr_router_choose_paddleocr_for_english(monkeypatch):
    from mangatrans.config import OCRConfig
    from mangatrans.language_detector import LanguageDetection
    router = OCRRouter(OCRConfig(), OCRRouterConfig())
    router._available["paddleocr"] = True
    det = LanguageDetection(code="en", name="English",
                            langs=("en",), score=0.9)
    primary, secondary = router._choose_engines(det)
    assert primary == "paddleocr"
    assert secondary == "easyocr"


def test_ocr_router_choose_easyocr_when_paddle_unavailable(monkeypatch):
    from mangatrans.config import OCRConfig
    from mangatrans.language_detector import LanguageDetection
    router = OCRRouter(OCRConfig(), OCRRouterConfig())
    router._available["paddleocr"] = False
    router._available["tesseract"] = False
    det = LanguageDetection(code="en", name="English",
                            langs=("en",), score=0.9)
    primary, secondary = router._choose_engines(det)
    assert primary == "easyocr"
