"""End-to-end MangaPipeline orchestrator.

Flow (single path — sau refactor 2026-05-25):
    detect → dedupe → split → recover_missed
    → LanguageDetector
    → OCRRouter (PaddleOCR)
    → SFXDetector
    → TranslationPipeline (role-aware, OpenRouter)
    → preserve stylized CJK SFX
    → bubble fill → smart fill → dilate → RedrawEngine (LaMa + edge-preserve)
    → TypographyEngine (4-tier shape-aware fit + render)
    → save image + JSON

Class chia sẻ session/cache giữa pages → batch mode tiết kiệm overhead model
load. Mỗi page có InteriorCache riêng (drop sau khi xong) tránh cross-page poison.

Refactor v18.1 (2026-05-25): tách `process_image` thành 11 `stage_*` method để
`mangatrans.runtime.Scheduler` gọi riêng từng stage (GPU mutex, retry, watchdog
per stage). Legacy `process_image` giữ semantics y hệt — chỉ compose lại 11
stage qua một context dict `ctx`. `process_batch` swallow exception như cũ
(thuộc tính `raise_translation_errors=False` mặc định).
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np

from .bubble_detector import BubbleDetector
from .cleaner import clean_bubbles_by_fill, dilate_mask, smart_fill_uniform_regions
from .config import PipelineConfig
from .geometry import InteriorCache
from .inpainter import BaseInpainter
from .language_detector import LanguageDetector
from .ocr_router import OCRRouter
from .redraw_engine import RedrawEngine
from .sfx_detector import SFXDetector, SFXProfile
from .translate import Translator
from .translation_pipeline import TranslationPipeline
from .typography_engine import TypographyEngine
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
    """High-level orchestrator. Tái sử dụng session giữa nhiều page.

    `_STAGE_GPU` đánh dấu stage chạm GPU → Scheduler acquire `GPUMutex` trước
    khi gọi. Stage không trong set này chạy thread pool tự do (CPU/IO).
    """

    # Danh sách tên stage (string) chạm GPU. Phải match string value của StageName
    # trong `mangatrans.runtime.page_task` — tránh import vòng nên dùng literal.
    _STAGE_GPU: frozenset = frozenset({"detect", "lang_detect", "ocr", "inpaint"})

    def __init__(self, config: Optional[PipelineConfig] = None,
                 base_dir: str = "."):
        self.config = config or PipelineConfig()
        self.base_dir = base_dir
        self._log = setup_logging(self.config.log_level)

        # Detection facade
        self.bubble_detector = BubbleDetector(self.config.detector)
        # Backward-compat alias — một số test cũ truy cập .detector
        self.detector = self.bubble_detector.raw

        # OCR Router (PaddleOCR primary)
        self.ocr_router = OCRRouter(self.config.ocr, self.config.ocr_router)

        self.language_detector = (
            LanguageDetector(self.config.language_detector)
            if self.config.use_language_detector
            else None
        )
        self.sfx_detector = (
            SFXDetector(self.config.sfx_detector)
            if self.config.use_sfx_detector
            else None
        )

        # Translation: Translator (HTTP layer) + TranslationPipeline (role-aware)
        from .llm_backend import create_llm_backend
        backend = create_llm_backend(self.config.translate, self.config.local_llm)
        self.translator = Translator(self.config.translate, backend=backend)
        self._translation_pipeline = TranslationPipeline(
            self.translator, self.config.translation_pipeline,
        )
        self._page_counter = 0
        self.raise_translation_errors = False  # legacy mode: swallow errors
        self._gpu_lock = threading.Lock()  # GPU mutex cho multithreaded batch

        # Render: TypographyEngine (4-tier shape-aware fit)
        self._typo_engine = TypographyEngine(
            self.config.render, self.config.geometry,
            font_path_resolver=None,
        )

        # Inpaint: resolve path + create inpainter (lazy session)
        self._inpainter: Optional[BaseInpainter] = None
        self._redraw_engine: Optional[RedrawEngine] = None
        self._inpaint_label: Optional[str] = None
        self._resolve_inpainter()

    # --------------------------- Public API --------------------------- #

    def new_context(self) -> Dict[str, Any]:
        """Khởi tạo context dict trống cho 1 page. Scheduler dùng giữa các stage."""
        return {
            "image": None,
            "h": 0,
            "w": 0,
            "text_mask": None,
            "blocks": [],
            "detection": None,
            "ocr_results": [],
            "sfx_profiles": [],
            "blocks_to_clean": [],
            "mask_for_inpaint": None,
            "image_filled": None,
            "mask_for_lama": None,
            "result_image": None,
            "summary": {},
        }

    def process_image(self, input_path: str, output_path: str) -> dict:
        """Process 1 image. Backward-compat: trả dict summary giống v18.0."""
        ctx = self.new_context()
        ctx = self.stage_load(ctx, input_path)
        ctx = self.stage_detect(ctx)
        ctx = self.stage_lang_detect(ctx)
        ctx = self.stage_ocr(ctx)
        ctx = self.stage_sfx(ctx)
        ctx = self.stage_translate(ctx, output_path)
        ctx = self.stage_save_json(ctx, output_path)
        ctx = self.stage_preserve_clean(ctx)
        ctx = self.stage_inpaint(ctx)
        ctx = self.stage_render(ctx)
        ctx = self.stage_save_png(ctx, output_path)
        summary = ctx["summary"]
        summary["input"] = input_path
        summary["output"] = output_path
        return summary

    def process_image_threadsafe(self, input_path: str, output_path: str) -> dict:
        """Thread-safe version: acquire GPU lock cho GPU stages,
        release khi chạy CPU/IO (translate API, save) để luồng khác dùng GPU."""
        ctx = self.new_context()

        # --- Stage CPU/IO: load ảnh (không cần GPU) ---
        ctx = self.stage_load(ctx, input_path)

        # --- GPU block 1: detect + language preview + OCR ---
        with self._gpu_lock:
            ctx = self.stage_detect(ctx)
            ctx = self.stage_lang_detect(ctx)
            ctx = self.stage_ocr(ctx)

        ctx = self.stage_sfx(ctx)

        # --- CPU/IO: translate (HTTP API call — chờ network, không cần GPU) ---
        ctx = self.stage_translate(ctx, output_path)
        ctx = self.stage_save_json(ctx, output_path)

        # --- GPU block 2: preserve_clean + inpaint + render ---
        with self._gpu_lock:
            ctx = self.stage_preserve_clean(ctx)
            ctx = self.stage_inpaint(ctx)
            ctx = self.stage_render(ctx)

        # --- CPU/IO: save result PNG ---
        ctx = self.stage_save_png(ctx, output_path)

        summary = ctx["summary"]
        summary["input"] = input_path
        summary["output"] = output_path
        return summary

    def release(self) -> None:
        """Giải phóng GPU + model memory. Idempotent."""
        if self._inpainter is not None:
            self._inpainter.release()
        self.ocr_router.release()
        if self.translator.config.use_glossary:
            self.translator.save_glossary()

    # --------------------------- Stage methods (sync API) --------------------------- #

    def stage_load(self, ctx: Dict[str, Any], input_path: str) -> Dict[str, Any]:
        """Đọc ảnh BGR từ disk. Stage CPU/IO — không GPU."""
        self._log.info(f"📖 Đang đọc ảnh: {input_path}")
        image = load_image(input_path)
        ctx["image"] = image
        ctx["h"], ctx["w"] = image.shape[:2]
        return ctx

    def stage_detect(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """ONNX detect + dedupe + split + recover. Stage GPU."""
        self._log.info("🔍 Đang phát hiện vùng văn bản...")
        det_res = self.bubble_detector.detect(ctx["image"])
        ctx["text_mask"] = det_res.text_mask
        ctx["blocks"] = det_res.blocks
        self._log.info(f"   - Tìm thấy {len(det_res.blocks)} bubble")
        return ctx

    def stage_lang_detect(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Detect page language before OCR. Stage GPU/CPU depending on OCR backend."""
        ctx["detection"] = None
        if self.language_detector is not None and ctx["blocks"]:
            ctx["detection"] = self.language_detector.detect(
                ctx["image"], ctx["blocks"]
            )
        return ctx

    def stage_ocr(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """OCR via Router. Stage GPU/CPU depending on PaddleOCR runtime."""
        ocr_results: list[dict] = []
        blocks = ctx["blocks"]
        if blocks:
            detection = ctx.get("detection")
            if detection is not None:
                self._log.info(
                    f"📝 OCR via Router (lang={detection.code}, "
                    f"engine={self.config.ocr_router.engine})..."
                )
            else:
                self._log.info(
                    f"📝 OCR via Router (engine={self.config.ocr_router.engine})..."
                )
            ocr_results = self.ocr_router.run_blocks(
                ctx["image"], blocks, detection
            )
        ctx["ocr_results"] = ocr_results
        return ctx

    def stage_sfx(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Classify OCR blocks as dialogue, narration, or SFX."""
        blocks = ctx["blocks"]
        ocr_results = ctx["ocr_results"]
        sfx_profiles: list[Optional[SFXProfile]] = [None] * len(blocks)
        if self.sfx_detector is not None and ocr_results:
            by_idx = {r["block_idx"]: r for r in ocr_results}
            page_h, page_w = ctx["h"], ctx["w"]
            for i, blk in enumerate(blocks):
                ocr_rec = by_idx.get(i, {})
                prof = self.sfx_detector.classify(
                    blk,
                    ocr_rec.get("text", ""),
                    page_w,
                    page_h,
                    ocr_conf=ocr_rec.get("ocr_conf"),
                )
                sfx_profiles[i] = prof
            for r in ocr_results:
                idx = r.get("block_idx")
                if not isinstance(idx, int) or idx < 0 or idx >= len(sfx_profiles):
                    continue
                if r.get("ocr_failed"):
                    r["role"] = "unknown"
                    r["should_translate"] = False
                    r["should_preserve_pixels"] = True
                    continue
                p = sfx_profiles[idx]
                if p is not None:
                    r["role"] = p.role
                    r["sfx_subtype"] = p.subtype
                    r["should_translate"] = p.should_translate
                    r["should_preserve_pixels"] = p.should_preserve_pixels
        ctx["sfx_profiles"] = sfx_profiles
        return ctx

    def stage_translate(self, ctx: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """Role-aware translate. Stage CPU/IO (HTTP).

        `raise_translation_errors=True` → re-raise mọi lỗi (scheduler retry).
        Legacy sync giữ False (swallow + warn). Glossary write-back, name
        memory save vẫn chạy bên trong helper `_translate_inplace`.
        """
        cfg = self.config
        if cfg.translate.enabled and ctx["ocr_results"]:
            self._translate_inplace(ctx["image"], ctx["ocr_results"], output_path)
        return ctx

    def stage_save_json(self, ctx: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """Lưu JSON OCR results. Stage CPU/IO."""
        ocr_results = ctx["ocr_results"]
        if ocr_results:
            json_path = os.path.splitext(output_path)[0] + ".json"
            ensure_dir(os.path.dirname(json_path) or ".")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(ocr_results, f, ensure_ascii=False, indent=2)
            self._log.info(f"📋 JSON đã lưu: {json_path}")
        return ctx

    def stage_preserve_clean(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Preserve untranslated CJK + bubble fill + smart fill + dilate mask. Stage CPU."""
        cfg = self.config
        blocks_to_clean, mask_for_inpaint = self._preserve_untranslated(
            ctx["blocks"], ctx["ocr_results"], ctx["text_mask"],
            ctx["sfx_profiles"], cfg.translate.enabled,
        )
        ctx["blocks_to_clean"] = blocks_to_clean
        ctx["mask_for_inpaint"] = mask_for_inpaint

        interior_cache = InteriorCache()
        self._log.info("⚪ Đang fill nền bubble bằng màu sampled...")
        image_filled, remaining_mask, _bubble_union = clean_bubbles_by_fill(
            ctx["image"], mask_for_inpaint, blocks_to_clean,
            cfg.geometry, interior_cache,
            freetext_std_thresh=cfg.cleaner.bubble_freetext_std,
            bubble_fill_retract=cfg.cleaner.bubble_fill_retract,
            feather_ksize=cfg.cleaner.bubble_feather_ksize,
        )
        n_bubbles_filled = sum(
            1 for b in blocks_to_clean if b.get("cls", 0) == 0
        )
        self._log.info(f"   - Đã fill {n_bubbles_filled} bubble bằng white solid")

        self._log.info("🧽 Đang smart-fill vùng nền đồng nhất (per-character)...")
        pre_mask = dilate_mask(remaining_mask, cfg.cleaner.pre_dilate)
        image_filled, mask_after_smart, n_smart = smart_fill_uniform_regions(
            image_filled, pre_mask,
            std_thresh=cfg.cleaner.smart_std_thresh,
            rim_size=cfg.cleaner.smart_rim_size,
        )
        self._log.info(f"   - Smart-fill xử lý {n_smart} vùng nền đồng nhất")
        mask_for_lama = dilate_mask(mask_after_smart, cfg.cleaner.dilate_kernel)
        ctx["image_filled"] = image_filled
        ctx["mask_for_lama"] = mask_for_lama
        return ctx

    def stage_inpaint(self, ctx: Dict[str, Any], force_cpu: bool = False) -> Dict[str, Any]:
        """LaMa inpaint + edge-preserve. Stage GPU.

        `force_cpu=True` → scheduler bọc trong `inpainter.force_cpu_mode()`
        để mọi `run_tile` chạy CPU session (fallback khi VRAM OOM).
        """
        image_filled = ctx["image_filled"]
        mask_for_lama = ctx["mask_for_lama"]
        if force_cpu and self._inpainter is not None:
            with self._inpainter.force_cpu_mode():
                result = self._inpaint_step(image_filled, mask_for_lama)
        else:
            result = self._inpaint_step(image_filled, mask_for_lama)
        ctx["result_image"] = result
        return ctx

    def stage_render(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """TypographyEngine render translated text. Stage CPU."""
        ocr_results = ctx["ocr_results"]
        has_translation = any(r.get("translated") for r in ocr_results)
        if has_translation:
            self._log.info("🖋️  Đang render text dịch vào bubble...")
            ctx["result_image"] = self._render_step(
                ctx["result_image"], ocr_results, ctx["image"], ctx["text_mask"],
            )
        return ctx

    def stage_save_png(self, ctx: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """Lưu PNG kết quả. Stage CPU/IO. Build summary cuối."""
        ensure_dir(os.path.dirname(output_path) or ".")
        save_image(ctx["result_image"], output_path)
        self._log.info(f"✅ Hoàn tất! Ảnh đã được lưu tại: {output_path}")

        det = ctx.get("detection")
        ctx["summary"] = {
            "n_bubbles": len(ctx["blocks"]),
            "n_ocr": len(ctx["ocr_results"]),
            "n_translated": sum(1 for r in ctx["ocr_results"]
                                if r.get("translated")),
            "language": det.code if det else None,
        }
        return ctx

    # --------------------------- Translation step --------------------------- #

    def _translate_inplace(self, image: np.ndarray, ocr_results: list[dict],
                           output_path: str) -> None:
        """Role-aware translate via TranslationPipeline. In-place writes 'translated' field.

        Exception: nếu `raise_translation_errors=False` (legacy) → warn + return.
                   nếu True (async) → re-raise để scheduler retry/backoff.
        """
        cfg = self.config
        log = self._log
        translatable = [
            r for r in ocr_results
            if (r.get("text") or "").strip()
            and r.get("should_translate", True) is not False
            and not r.get("should_preserve_pixels")
        ]
        if not translatable:
            log.info("   - Không có OCR text hợp lệ để dịch, bỏ qua OpenRouter.")
            return

        # Glossary path
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

        # Name memory cho TranslationPipeline
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
            if self.raise_translation_errors:
                raise
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
            sfx_pres = (i < len(sfx_profiles)
                        and sfx_profiles[i] is not None
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
        """Sử dụng trực tiếp lama-manga."""
        cfg = self.config
        cfg.inpaint.model_path = os.path.join(self.base_dir, "anime-manga-big-lama.pt")
        self._inpaint_label = "anime-manga-big-lama.pt"
        from .inpainter import create_inpainter
        self._inpainter = create_inpainter(cfg.inpaint)
        self._log.info(f"🧠 Inpaint model: {self._inpaint_label}")
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
        """Render translated text via TypographyEngine (single path)."""
        from PIL import Image  # noqa: F401  (ensure PIL available)
        from .font_renderer import bgr_to_pil, pil_to_bgr
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
