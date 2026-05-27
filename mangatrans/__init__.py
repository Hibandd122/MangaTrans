"""MangaTrans — pipeline dịch manga: detect bubble → OCR → translate → inpaint → render.

Public API:
    from mangatrans import translate_image
    translate_image("page.jpg", "out.png")

    # hoặc dùng class trực tiếp
    from mangatrans import MangaPipeline, PipelineConfig
    pipe = MangaPipeline(PipelineConfig())
    pipe.process_image("chap/001.jpg", "out.png")

Modules:
    config            — PipelineConfig dataclass, defaults, prompts
    utils             — io, logging, model probing, caching helpers
    detector          — text/bubble detection (comic-text-detector ONNX)
    bubble_detector   — facade post-processing pipeline
    geometry          — bubble interior, polygon, inscribed rect, row extents
    cleaner           — bubble fill, smart fill uniform regions, mask prep
    inpainter         — LaMa session, texture classifier, deterministic fills
    redraw            — HD-tiled hybrid redraw (LaMa + classify + refine + fallback)
    redraw_engine     — Façade: edge-preserve + unsharp + HybridRedrawer
    language_detector — multi-tier page-level language detection
    script_classifier — block-level CJK/Latin/mixed classifier
    ocr               — gibberish filter helper (`is_likely_gibberish`)
    ocr_router        — multi-engine OCR dispatcher (PaddleOCR primary)
    sfx_detector      — dialogue/narration/SFX role classifier
    font_renderer     — PIL multi-line draw + stroke + vertical
    typography_engine — fit cascade trên FitResult + delegate render
    translate         — OpenRouter call, glossary, reading order, position tags
    translation_pipeline — role-aware orchestrator trên Translator
    text              — wrap_text / wrap_text_shape / inscribed-rect helpers
    pipeline          — end-to-end orchestrator
    cli               — argparse + entry point
"""
from .config import PipelineConfig

try:
    from .pipeline import MangaPipeline
except ImportError:  # pragma: no cover
    MangaPipeline = None  # type: ignore


def translate_image(input_path: str, output_path: str,
                    target_lang: str = "Vietnamese",
                    model=None) -> dict:
    """Fast one-shot dịch 1 ảnh manga.

    Tự khởi tạo `MangaPipeline` với `translate=ON` mặc định, dịch
    sang `target_lang`, dùng `model` (None → default OpenRouter model).
    Trả dict summary từ `process_image` (n_bubbles, n_ocr, n_translated,
    language).

    Note: với batch nhiều ảnh, tạo `MangaPipeline` 1 lần rồi gọi
    `process_batch()` để share session (model load tốn ~2-3s/ảnh).
    """
    from .config import TranslateConfig
    cfg = PipelineConfig(
        translate=TranslateConfig(
            enabled=True, target_lang=target_lang, model=model,
        ),
    )
    pipe = MangaPipeline(cfg)
    try:
        return pipe.process_image(input_path, output_path)
    finally:
        pipe.release()


__all__ = [
    "MangaPipeline",
    "PipelineConfig",
    "translate_image",
    "BubbleDetector",
    "LanguageDetector",
    "OCRRouter",
    "SFXDetector",
    "TypographyEngine",
    "RedrawEngine",
    "TranslationPipeline",
    "classify_script",
]
__version__ = "18.0.0"


def __getattr__(name: str):
    """Lazy import VIP modules để smoke-test không kéo dep chain."""
    if name == "BubbleDetector":
        from .bubble_detector import BubbleDetector
        return BubbleDetector
    if name == "LanguageDetector":
        from .language_detector import LanguageDetector
        return LanguageDetector
    if name == "OCRRouter":
        from .ocr_router import OCRRouter
        return OCRRouter
    if name == "SFXDetector":
        from .sfx_detector import SFXDetector
        return SFXDetector
    if name == "TypographyEngine":
        from .typography_engine import TypographyEngine
        return TypographyEngine
    if name == "RedrawEngine":
        from .redraw_engine import RedrawEngine
        return RedrawEngine
    if name == "TranslationPipeline":
        from .translation_pipeline import TranslationPipeline
        return TranslationPipeline
    if name == "classify_script":
        from .script_classifier import classify_script
        return classify_script
    raise AttributeError(f"module 'mangatrans' has no attribute {name!r}")
