"""manga — pipeline dịch manga: detect → OCR → translate → inpaint → render.

Public API:
    from manga import translate_image
    translate_image("page.jpg", "out.png")

    from manga import MangaPipeline, PipelineConfig
    pipe = MangaPipeline(PipelineConfig())
    pipe.process_image("chap/001.jpg", "out.png")

Modules (12 file consolidated):
    config      — PipelineConfig + dataclasses + prompts + constants
    utils       — io, logging, model probing, caching
    geometry    — bubble interior, polygon, inscribed rect, row extents
    language    — page-level lang detect + per-text script classifier
    detection   — TextDetector ONNX + dedupe + split + recover (facade)
    ocr         — gibberish filter + multi-engine OCR router
    sfx         — dialogue/narration/SFX role classifier
    inpaint     — cleaner + LaMa TorchScript + RedrawEngine façade
    render      — PIL primitives + wrap/fit + TypographyEngine cascade
    translate   — OpenRouter call + role-aware TranslationPipeline
    pipeline    — end-to-end orchestrator + CLI
"""
from .config import PipelineConfig, ROMANCE_SYSTEM_PROMPT

try:
    from .pipeline import MangaPipeline
except ImportError:  # pragma: no cover
    MangaPipeline = None  # type: ignore


def translate_image(input_path: str, output_path: str,
                    target_lang: str = "Vietnamese",
                    model=None) -> dict:
    """One-shot dịch 1 ảnh manga. Tự tạo MangaPipeline + release."""
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
    "ROMANCE_SYSTEM_PROMPT",
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
__version__ = "19.0.0"


def __getattr__(name: str):
    """Lazy import để smoke-test không kéo dep chain."""
    if name == "BubbleDetector":
        from .detection import BubbleDetector
        return BubbleDetector
    if name == "LanguageDetector":
        from .language import LanguageDetector
        return LanguageDetector
    if name == "OCRRouter":
        from .ocr import OCRRouter
        return OCRRouter
    if name == "SFXDetector":
        from .sfx import SFXDetector
        return SFXDetector
    if name == "TypographyEngine":
        from .render import TypographyEngine
        return TypographyEngine
    if name == "RedrawEngine":
        from .inpaint import RedrawEngine
        return RedrawEngine
    if name == "TranslationPipeline":
        from .translate import TranslationPipeline
        return TranslationPipeline
    if name == "classify_script":
        from .language import classify_script
        return classify_script
    raise AttributeError(f"module 'manga' has no attribute {name!r}")
