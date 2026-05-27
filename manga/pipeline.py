"""End-to-end MangaPipeline + CLI (consolidated).

Gộp `pipeline.py` + `cli.py`. Flow:
    detect → dedupe → split → recover_missed
    → LanguageDetector
    → OCRRouter (PaddleOCR / EasyOCR / manga-ocr)
    → SFXDetector
    → TranslationPipeline (role-aware, OpenRouter)
    → preserve stylized CJK SFX
    → bubble fill → smart fill → dilate → RedrawEngine (LaMa)
    → TypographyEngine (4-tier shape-aware fit + render)
    → save image + JSON
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Optional, Sequence

import cv2
import numpy as np

from .config import (
    DEFAULT_FONT_CANDIDATES,
    INPAINT_CANDIDATES,
    CleanerConfig,
    DetectorConfig,
    GeometryConfig,
    InpaintConfig,
    OCRConfig,
    PipelineConfig,
    RenderConfig,
    TranslateConfig,
)
from .detection import BubbleDetector
from .geometry import InteriorCache
from .inpaint import (
    LamaInpainter,
    RedrawEngine,
    clean_bubbles_by_fill,
    dilate_mask,
    smart_fill_uniform_regions,
)
from .language import LanguageDetection, LanguageDetector
from .ocr import OCRRouter
from .render import TypographyEngine, bgr_to_pil, pil_to_bgr
from .sfx import SFXDetector, SFXProfile
from .translate import Translator, TranslationPipeline
from .utils import (
    auto_pick_inpaint_model,
    ensure_dir,
    has_latin_letters,
    has_real_cjk,
    load_image,
    save_image,
    setup_logging,
)


_PRESERVE_SENTINEL = ("...", "…", "???")


def _is_real_translation(text: str) -> bool:
    if not text or not text.strip():
        return False
    return text.strip() not in _PRESERVE_SENTINEL


class MangaPipeline:
    """High-level orchestrator. Tái sử dụng session giữa nhiều page."""

    def __init__(self, config: Optional[PipelineConfig] = None,
                 base_dir: str = "."):
        self.config = config or PipelineConfig()
        self.base_dir = base_dir
        self._log = setup_logging(self.config.log_level)

        self.bubble_detector = BubbleDetector(self.config.detector)
        self.detector = self.bubble_detector.raw

        self.ocr_router = OCRRouter(self.config.ocr, self.config.ocr_router)

        if self.config.use_language_detector:
            self.language_detector: Optional[LanguageDetector] = LanguageDetector(
                self.config.language_detector,
            )
        else:
            self.language_detector = None

        if self.config.use_sfx_detector:
            self.sfx_detector: Optional[SFXDetector] = SFXDetector(
                self.config.sfx_detector,
            )
        else:
            self.sfx_detector = None

        self.translator = Translator(self.config.translate)
        self._translation_pipeline = TranslationPipeline(
            self.translator, self.config.translation_pipeline,
        )
        self._page_counter = 0

        self._typo_engine = TypographyEngine(
            self.config.render, self.config.geometry,
            font_path_resolver=None,
        )

        self._inpainter: Optional[LamaInpainter] = None
        self._redraw_engine: Optional[RedrawEngine] = None
        self._inpaint_label: Optional[str] = None
        self._resolve_inpainter()

    # --------------------------- Public API --------------------------- #

    def process_image(self, input_path: str, output_path: str) -> dict:
        """Process 1 image. Trả dict summary (counts, paths)."""
        cfg = self.config
        log = self._log
        log.info(f"📖 Đang đọc ảnh: {input_path}")
        image = load_image(input_path)
        h, w = image.shape[:2]

        log.info("🔍 Đang phát hiện vùng văn bản...")
        det_res = self.bubble_detector.detect(image)
        text_mask = det_res.text_mask
        blocks = det_res.blocks
        log.info(f"   - Tìm thấy {len(blocks)} bubble")

        detection: Optional[LanguageDetection] = None
        if self.language_detector is not None and blocks:
            detection = self.language_detector.detect(image, blocks)

        ocr_results: list[dict] = []
        if blocks:
            if detection is not None:
                log.info(
                    f"📝 OCR via Router (lang={detection.code}, "
                    f"engine={self.config.ocr_router.engine})..."
                )
            else:
                log.info("📝 OCR via Router (no lang detect)...")
            ocr_results = self.ocr_router.run_blocks(image, blocks, detection)

        sfx_profiles: list[Optional[SFXProfile]] = [None] * len(blocks)
        if self.sfx_detector is not None and ocr_results:
            by_idx = {r["block_idx"]: r for r in ocr_results}
            for i, blk in enumerate(blocks):
                ocr_rec = by_idx.get(i, {})
                ocr_text = ocr_rec.get("text", "")
                ocr_conf = ocr_rec.get("ocr_conf")
                prof = self.sfx_detector.classify(blk, ocr_text, w, h,
                                                  ocr_conf=ocr_conf)
                sfx_profiles[i] = prof
            for r in ocr_results:
                p = sfx_profiles[r["block_idx"]]
                if p is not None:
                    r["role"] = p.role
                    r["sfx_subtype"] = p.subtype
                    r["should_translate"] = p.should_translate
                    r["should_preserve_pixels"] = p.should_preserve_pixels

        if cfg.translate.enabled and ocr_results:
            self._translate_inplace(image, ocr_results, output_path)

        if ocr_results:
            json_path = os.path.splitext(output_path)[0] + ".json"
            ensure_dir(os.path.dirname(json_path) or ".")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(ocr_results, f, ensure_ascii=False, indent=2)
            log.info(f"📋 JSON đã lưu: {json_path}")

        blocks_to_clean, mask_for_inpaint = self._preserve_untranslated(
            blocks, ocr_results, text_mask, sfx_profiles,
            cfg.translate.enabled,
        )

        interior_cache = InteriorCache()
        log.info("⚪ Đang fill nền bubble bằng màu sampled...")
        image_filled, remaining_mask, _bubble_union = clean_bubbles_by_fill(
            image, mask_for_inpaint, blocks_to_clean,
            cfg.geometry, interior_cache,
            freetext_std_thresh=cfg.cleaner.bubble_freetext_std,
            bubble_fill_retract=cfg.cleaner.bubble_fill_retract,
            feather_ksize=cfg.cleaner.bubble_feather_ksize,
        )
        n_bubbles_filled = sum(
            1 for b in blocks_to_clean if b.get("cls", 0) == 0
        )
        log.info(f"   - Đã fill {n_bubbles_filled} bubble bằng white solid")

        log.info("🧽 Đang smart-fill vùng nền đồng nhất (per-character)...")
        pre_mask = dilate_mask(remaining_mask, cfg.cleaner.pre_dilate)
        image_filled, mask_after_smart, n_smart = smart_fill_uniform_regions(
            image_filled, pre_mask,
            std_thresh=cfg.cleaner.smart_std_thresh,
            rim_size=cfg.cleaner.smart_rim_size,
        )
        log.info(f"   - Smart-fill xử lý {n_smart} vùng nền đồng nhất")

        mask_for_lama = dilate_mask(mask_after_smart, cfg.cleaner.dilate_kernel)
        result_image = self._inpaint_step(image_filled, mask_for_lama)

        has_translation = any(r.get("translated") for r in ocr_results)
        if has_translation:
            log.info("🖋️  Đang render text dịch vào bubble...")
            result_image = self._render_step(
                result_image, ocr_results, image, text_mask,
            )

        ensure_dir(os.path.dirname(output_path) or ".")
        save_image(result_image, output_path)
        log.info(f"✅ Hoàn tất! Ảnh đã được lưu tại: {output_path}")

        return {
            "input": input_path,
            "output": output_path,
            "n_bubbles": len(blocks),
            "n_ocr": len(ocr_results),
            "n_translated": sum(1 for r in ocr_results
                                if r.get("translated")),
            "language": detection.code if detection else None,
        }

    def process_batch(self, input_paths: Sequence[str],
                      output_dir: str) -> list[dict]:
        """Process nhiều page với shared session. Trả list summary."""
        ensure_dir(output_dir)
        results = []
        for i, inp in enumerate(input_paths):
            base = os.path.splitext(os.path.basename(inp))[0]
            out_path = os.path.join(output_dir, f"{base}.png")
            self._log.info(f"\n📄 [{i + 1}/{len(input_paths)}] {inp}")
            try:
                summary = self.process_image(inp, out_path)
            except Exception as e:  # noqa: BLE001
                self._log.error(f"❌ {inp}: {e}")
                summary = {"input": inp, "error": str(e)}
            results.append(summary)
        return results

    def release(self) -> None:
        """Giải phóng GPU + model memory. Idempotent."""
        if self._inpainter is not None:
            self._inpainter.release()
        self.ocr_router.release()
        if self.translator.config.use_glossary:
            self.translator.save_glossary()

    # --------------------------- Translation step --------------------------- #

    def _translate_inplace(self, image: np.ndarray, ocr_results: list[dict],
                           output_path: str) -> None:
        cfg = self.config
        log = self._log

        if cfg.translate.use_glossary:
            glossary_path = (cfg.translate.glossary_path
                             or os.path.join(
                                 os.path.dirname(os.path.abspath(output_path)) or ".",
                                 ".glossary.json",
                             ))
            self.translator.attach_glossary(glossary_path)
            if self.translator.glossary:
                log.info(
                    f"   - Glossary loaded: {len(self.translator.glossary)} entries"
                    f" ({glossary_path})"
                )

        self._page_counter += 1
        mem_path = (cfg.translation_pipeline.name_memory_path
                    or os.path.join(
                        os.path.dirname(os.path.abspath(output_path)) or ".",
                        ".names.json",
                    ))
        self._translation_pipeline.attach_memory(mem_path)
        log.info(
            f"🌐 Đang dịch (role-aware) sang {cfg.translate.target_lang} "
            f"qua OpenRouter/{self.translator.resolve_model()}..."
        )
        page_h, page_w = image.shape[:2]
        try:
            translations = self._translation_pipeline.translate_page(
                ocr_results, page_w, page_h, page_idx=self._page_counter,
            )
        except Exception as e:  # noqa: BLE001
            log.warning(f"⚠️  Lỗi dịch: {e}")
            return
        for r, tr in zip(ocr_results, translations):
            if tr:
                r["translated"] = tr
                log.info(f"   {r['text']!r} -> {tr!r}")
        if cfg.translate.use_glossary:
            self.translator.save_glossary()
        self._translation_pipeline.save_memory()

    # --------------------------- Preserve step --------------------------- #

    def _preserve_untranslated(self, blocks: list[dict],
                               ocr_results: list[dict],
                               text_mask: np.ndarray,
                               sfx_profiles: list[Optional[SFXProfile]],
                               translate_enabled: bool
                               ) -> tuple[list[dict], np.ndarray]:
        """Mask out untranslated CJK SFX / stylized SFX khỏi text_mask."""
        log = self._log
        if not translate_enabled or not ocr_results \
                or not self.config.preserve_untranslated_cjk:
            return blocks, text_mask

        translated_block_indices = {
            r["block_idx"] for r in ocr_results
            if _is_real_translation(r.get("translated", ""))
        }
        block_idx_to_text = {r["block_idx"]: r.get("text", "")
                             for r in ocr_results}

        untranslated_bboxes: list[Sequence[int]] = []
        preserved_block_indices: set[int] = set()
        for i, blk in enumerate(blocks):
            sfx_pres = (sfx_profiles[i] is not None
                        and sfx_profiles[i].should_preserve_pixels)
            if i in translated_block_indices and not sfx_pres:
                continue
            ocr_text = block_idx_to_text.get(i, "")
            if sfx_pres or (has_real_cjk(ocr_text)
                            and not has_latin_letters(ocr_text)
                            and i not in translated_block_indices):
                untranslated_bboxes.append(blk["bbox"])
                preserved_block_indices.add(i)

        blocks_to_clean = [b for i, b in enumerate(blocks)
                           if i not in preserved_block_indices]
        if not untranslated_bboxes:
            log.info(f"   - Không có SFX/CJK preserve, inpaint toàn bộ "
                     f"{len(blocks_to_clean)} block")
            return blocks_to_clean, text_mask

        keep_mask = np.ones_like(text_mask)
        for bx1, by1, bx2, by2 in untranslated_bboxes:
            bx1 = max(0, int(bx1))
            by1 = max(0, int(by1))
            bx2 = min(text_mask.shape[1], int(bx2))
            by2 = min(text_mask.shape[0], int(by2))
            keep_mask[by1:by2, bx1:bx2] = 0
        masked = cv2.bitwise_and(text_mask, text_mask, mask=keep_mask)
        log.info(
            f"   - Giữ nguyên {len(untranslated_bboxes)} SFX/CJK (preserve), "
            f"inpaint {len(blocks_to_clean)} block còn lại"
        )
        return blocks_to_clean, masked

    # --------------------------- Inpaint step --------------------------- #

    def _resolve_inpainter(self) -> None:
        cfg = self.config
        path = cfg.inpaint.model_path
        if path and os.path.isfile(path):
            full_path, label = path, os.path.basename(path)
        else:
            full_path, label = auto_pick_inpaint_model(
                path, INPAINT_CANDIDATES, base_dir=self.base_dir,
            )
        if full_path is None:
            self._log.warning(
                "⚠️  Không tìm thấy inpaint model — sẽ dùng cv2.inpaint fallback."
            )
            self._inpainter = None
            self._inpaint_label = "cv2.inpaint (fallback)"
        else:
            self._inpainter = LamaInpainter(
                full_path, override_size=cfg.inpaint.target_size,
            )
            self._inpaint_label = label or os.path.basename(full_path)
            self._log.info(f"🧠 Inpaint model: {self._inpaint_label}  [{full_path}]")
        self._redraw_engine = RedrawEngine(
            self._inpainter, cfg.inpaint, cfg.redraw_engine,
        )

    def _inpaint_step(self, image_filled: np.ndarray,
                      mask_for_lama: np.ndarray) -> np.ndarray:
        log = self._log
        if mask_for_lama.sum() == 0:
            log.info("   - Không còn vùng nào cần inpaint, bỏ qua LaMa.")
            return image_filled
        mode_label = "HD-tiled" if self.config.inpaint.hd else "single-pass"
        log.info(
            f"🎨 Đang redraw nền (inpainting) với {self._inpaint_label} "
            f"[{mode_label} / RedrawEngine]..."
        )
        result, _report = self._redraw_engine.redraw(image_filled, mask_for_lama)
        return result

    # --------------------------- Render step --------------------------- #

    def _render_step(self, result_image: np.ndarray,
                     ocr_results: list[dict], original_image: np.ndarray,
                     text_mask: np.ndarray) -> np.ndarray:
        pil_img = bgr_to_pil(result_image)
        interior_cache = InteriorCache()
        script_code = "vi"
        translatable = [r for r in ocr_results
                        if r.get("translated")
                        and _is_real_translation(r["translated"])]
        for item in translatable:
            self._typo_engine.fit_and_render(
                pil_img, item, translatable, item["translated"],
                original_image=original_image,
                text_seg_mask=text_mask,
                interior_cache=interior_cache,
                script_code=script_code,
            )
        return pil_to_bgr(pil_img)


# =============================================================
# CLI
# =============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="manga",
        description=(
            "Manga translation pipeline: detect → OCR → translate → inpaint → render."
        ),
    )

    io = p.add_argument_group("I/O")
    io.add_argument("--input", "-i",
                    help="Input image path (hoặc dùng --batch)")
    io.add_argument("--output", "-o", default="cleaned.png",
                    help="Output path (mặc định cleaned.png)")
    io.add_argument("--batch", action="store_true",
                    help="Batch mode: --input là thư mục, --output cũng vậy")
    io.add_argument("--pattern", default="*.jpg,*.png,*.jpeg,*.webp",
                    help="Glob pattern (comma-sep) khi --batch")

    det = p.add_argument_group("Detection")
    det.add_argument("--detect-model", default="comic-text-detector.onnx",
                     help="Path tới comic-text-detector ONNX")

    inp = p.add_argument_group("Inpainting")
    inp.add_argument("--lama-model", default=None,
                     help="Inpaint model (None = auto-pick từ candidates)")
    inp.add_argument("--no-hd", action="store_true",
                     help="Tắt HD-tiled mode (nhanh hơn nhưng mất chi tiết)")
    inp.add_argument("--dilate-kernel", type=int, default=13,
                     help="Kích thước kernel giãn nở mask")

    tr = p.add_argument_group("Translation")
    tr.add_argument("--no-translate", action="store_true",
                    help="Tắt dịch — chỉ detect + inpaint + render bubble trống")
    tr.add_argument("--target-lang", default="Vietnamese")
    tr.add_argument("--llm-model", default=None,
                    help="Model OpenRouter")
    tr.add_argument("--glossary", default=None,
                    help="Path glossary JSON")
    tr.add_argument("--no-glossary", action="store_true")
    tr.add_argument("--names-memory", default=None,
                    help="Path JSON character name memory")

    rd = p.add_argument_group("Render")
    rd.add_argument("--font", default=DEFAULT_FONT_CANDIDATES[0],
                    help=f"Path font TTF (mặc định {DEFAULT_FONT_CANDIDATES[0]})")

    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p


def build_config(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig(
        detector=DetectorConfig(model_path=args.detect_model),
        geometry=GeometryConfig(),
        cleaner=CleanerConfig(dilate_kernel=args.dilate_kernel),
        inpaint=InpaintConfig(
            model_path=args.lama_model,
            hd=not args.no_hd,
        ),
        ocr=OCRConfig(),
        translate=TranslateConfig(
            enabled=not args.no_translate,
            target_lang=args.target_lang,
            model=args.llm_model,
            glossary_path=args.glossary,
            use_glossary=not args.no_glossary,
        ),
        render=RenderConfig(font_path=args.font),
        log_level=args.log_level,
    )
    if args.names_memory:
        from .translate import TranslationPipelineConfig
        if isinstance(cfg.translation_pipeline, TranslationPipelineConfig):
            cfg.translation_pipeline.name_memory_path = args.names_memory
    return cfg


def _expand_inputs(input_dir: str, pattern_csv: str) -> list[str]:
    paths: set[str] = set()
    for pat in pattern_csv.split(","):
        pat = pat.strip()
        if not pat:
            continue
        for p in glob.iglob(os.path.join(input_dir, pat)):
            if os.path.isfile(p):
                paths.add(os.path.abspath(p))
    return sorted(paths)


def main(argv: Optional[Sequence[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.input:
        parser.error("--input là bắt buộc")

    config = build_config(args)
    pipeline = MangaPipeline(config, base_dir=".")

    try:
        if args.batch:
            if not os.path.isdir(args.input):
                parser.error(f"--input phải là directory khi --batch ({args.input})")
            inputs = _expand_inputs(args.input, args.pattern)
            if not inputs:
                parser.error(
                    f"Không tìm thấy file nào trong {args.input} "
                    f"khớp {args.pattern}"
                )
            results = pipeline.process_batch(inputs, args.output)
            n_ok = sum(1 for r in results if "error" not in r)
            print(f"\n📊 Batch: {n_ok}/{len(results)} pages thành công")
            return 0 if n_ok == len(results) else 1

        pipeline.process_image(args.input, args.output)
        return 0
    finally:
        pipeline.release()


if __name__ == "__main__":
    sys.exit(main())
