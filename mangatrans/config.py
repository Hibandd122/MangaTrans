"""Cấu hình pipeline. Mọi default đều ở đây — không hardcode trong module khác.

Sử dụng dataclass thay vì hằng số module-level để hỗ trợ:
- Multi-pipeline trong cùng process (vd batch nhiều cấu hình A/B).
- Override per-page dễ qua dataclass.replace().
- Type checking và introspection.

Koharu model ecosystem (2026-05-27): thêm registry cho Detection, OCR,
Inpainting, LLM backends. Tất cả opt-in — default giữ nguyên model cũ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence


# OpenRouter: OpenAI-compatible aggregator (multi-model). Provider duy nhất —
# user đã chuyển hoàn toàn khỏi Gemini/Groq. Headers HTTP-Referer / X-Title để
# OpenRouter track app.
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
DEFAULT_OPENROUTER_KEY = ""


# Romance/Shoujo/Josei translation prompt — giữ nguyên từ v15 (đã được tinh chỉnh
# nhiều vòng cho thể loại này).
ROMANCE_SYSTEM_PROMPT = """BỐI CẢNH: Manga thể loại Romance / Shoujo / Josei. Câu chuyện tình cảm lãng mạn với nhiều cảm xúc tinh tế.

QUY TẮC DỊCH:
1. Giữ nguyên sắc thái tình cảm, lãng mạn của bản gốc — KHÔNG dịch khô khan hay máy móc.
2. Dùng tiếng Việt tự nhiên, mềm mại, phù hợp thể loại romance. Tránh từ Hán-Việt nặng nề.
3. Câu thoại ngắn gọn như manga, giữ nhịp cảm xúc. KHÔNG câu dài lê thê.
4. Giữ nguyên tên nhân vật gốc (Nhật/Hàn/Anh…). KHÔNG phiên âm hay đổi tên.
5. Xưng hô linh hoạt theo ngữ cảnh: anh-em, tớ-cậu, mình-bạn, tôi-anh/chị… tùy mối quan hệ.
6. Các câu tỏ tình, thổ lộ, độc thoại nội tâm phải giữ đúng cảm xúc gốc — rung động, e thẹn, đau lòng…
7. SFX/exclamation ngắn (1-2 từ) → dịch THẲNG sang Việt 1-1, KHÔNG paraphrase. VÍ DỤ:
   - "OH!!" / "Oh!" → "Ồ!!" / "Ồ!" (KHÔNG dịch thành "Ưm?" hay "Ô")
   - "AH!" / "Ah" → "A!" / "À"
   - "EH?" / "Eh?" → "Hả?" / "Ơ?"
   - "HEY!" → "Này!" (KHÔNG "Ê!")
   - "WOW!" → "Wow!" / "Ồ!"
   - "HUH?" → "Hử?"
   - "HMPH" / "HMM" → "Hừ" / "Hmm"
   - "YES!" → "Được!" / "Vâng!"
   - "NO!" → "Không!"
   - "OW!" / "OUCH!" → "Á!" / "Ối!"
   - "WAH!" → "Oa!"
   - "HAHAHA" → "Hahaha"
8. **OCR ĐỌC SAI SFX**: input có thể là chuỗi rác ngắn do OCR đọc sai SFX/handwriting Nhật. ĐOÁN SFX hợp lý từ ngữ cảnh manga romance và dịch sang Việt. VÍ DỤ:
   - "DIs-" / "OIs?" / "CEh?" → có thể là "Ơ?" / "Hả?" / "Ah-" / "Phù-" (đoán SFX thở dài/ngạc nhiên)
   - "NEVEPMINO" → "Nevermind" → "Thôi" / "Không sao"
   - "IILL" → "I'll" → giữ ý câu
   - "BT" / "BUt__" / "B+" → có thể là "But..." → "Nhưng..." (đoán từ Latin garbled ngắn)
   - "Sory aboけ that" → "Sorry about that" → "Xin lỗi về chuyện đó"
   - Pure CJK ("嚣", "噩") = SFX gốc Nhật → dịch sang SFX Việt tương ứng ("Ầm!", "Hự!")
   - **CHỈ trả về "..." khi input là CJK thuần (Nhật/Hàn/Trung) mà bạn không biết SFX gì**. Với Latin (Anh/Việt/etc) dù garbled, LUÔN cố đoán + dịch — KHÔNG bao giờ trả "..." cho Latin.
9. KHÔNG thêm/bớt ý so với bản gốc. KHÔNG sáng tác thêm chi tiết.
10. Ngôn ngữ cơ thể, hành động (đỏ mặt, tim đập, nắm tay…) dịch sát và tự nhiên.
11. Output phải NGẮN GỌN — câu dịch không dài hơn câu gốc nhiều (manga bubble hạn chế chỗ).
12. THỨ TỰ ĐỌC: input được sort theo thứ tự đọc manga (phải→trái, trên→dưới). Các câu liền nhau thường là 1 lượt hội thoại → giữ ĐỒNG NHẤT về đại từ, xưng hô, tone giữa các câu trong cùng cảnh. Đại từ ở câu sau phải khớp với câu trước (không "anh ấy" rồi "cậu ấy" trong cùng cuộc nói).
13. GLOSSARY: nếu có "GLOSSARY" section ở dưới, MỌI tên/thuật ngữ trong glossary PHẢI dịch ĐÚNG như liệt kê — KHÔNG đổi sang biến thể khác giữa chừng. Tên mới (chưa có trong glossary) → giữ nguyên dạng gốc."""


# Model registries đã bị xoá để đơn giản hoá mã nguồn (chỉ dùng các model mặc định).


# Font duy nhất — Mali-Bold.ttf (handwritten bold, full Vietnamese coverage)
# ở repo root. BadComic.ttf cũ thiếu dấu VN (chỉ ~50% coverage) → render ra
# tofu boxes. Mali-Bold có 100% Latin Extended Additional.
import os
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_FONT_CANDIDATES: tuple[str, ...] = (
    os.path.join(_BASE_DIR, "Mali-Bold.ttf"),
)


@dataclass
class DetectorConfig:
    """Cấu hình text detector. Hỗ trợ nhiều backend từ Koharu ecosystem."""

    # Backend: 'comic-text-detector' | 'anime-text-yolo' |
    #          'comic-text-bubble-detector' | 'pp-doclayout' | 'speech-bubble-seg'
    backend: str = "comic-text-detector"
    model_path: str = "comic-text-detector.onnx"
    target_size: int = 1024
    mask_thresh: float = 0.3
    blk_conf: float = 0.4
    blk_nms: float = 0.45

    # ONNX Runtime perf: 8 intra-op threads tối ưu trên 12-core CPU (13s vs 17s default).
    intra_op_threads: int = 8

    # Dedupe
    dedupe_iou: float = 0.35
    dedupe_contain: float = 0.65

    # Split merged bubbles
    split_gap_thresh: int = 20
    split_min_comp_area: int = 40

    # Recover missed text (seg-mask CC fallback cho YOLO miss)
    recover_min_area: int = 200
    recover_min_dim: int = 8
    recover_overlap_thresh: float = 0.4
    recover_max_area_ratio: float = 0.10
    recover_min_text_density: float = 0.05
    recover_min_surround_white: int = 185
    recover_max_aspect: float = 30.0
    recover_close_ksize: int = 13


@dataclass
class GeometryConfig:
    """Cấu hình tìm bubble interior + polygon."""

    # find_bubble_interior heuristic
    white_thresh: int = 210
    close_ksize: int = 5
    retract: int = 3
    max_area_ratio: float = 0.15
    roi_pad_factor: float = 1.5
    dark_thresh: int = 160
    dark_dilate: int = 5

    # find_bubble_polygon
    polygon_retract: int = 2
    polygon_simplify_eps: float = 0.005


@dataclass
class CleanerConfig:
    """Cấu hình pre-inpaint cleaning (smart-fill + bubble fill)."""

    dilate_kernel: int = 13          # final dilate cho LaMa input
    pre_dilate: int = 3              # nhẹ trước smart-fill
    smart_std_thresh: float = 18.0   # rim std < 18 = nền đồng nhất
    smart_rim_size: int = 6
    bubble_freetext_std: float = 25.0  # free-text fill nếu nền đồng nhất (giờ unused, giữ cho backward-compat)
    bubble_fill_retract: int = 4     # erode bubble fill mask để chừa outline
    bubble_feather_ksize: int = 9    # Gaussian blur edge fill → blend mượt outline


@dataclass
class InpaintConfig:
    """Cấu hình inpaint. Hỗ trợ LaMa (default), FLUX.2, AOT."""

    # Backend: 'lama' | 'flux2' | 'aot'
    backend: str = "lama"
    model_path: Optional[str] = None  # None = auto-pick
    target_size: Optional[int] = None  # None = auto từ ONNX shape
    hd: bool = True
    tile_pad: float = 0.75
    refine: bool = True
    classify: bool = True

    # Classifier thresholds
    solid_std_thresh: float = 8.0
    gradient_residual_thresh: float = 6.0
    rim_width: int = 12

    # HD tile constraints
    tile_min_side: int = 256
    min_component_area: int = 20


@dataclass
class OCRConfig:
    """Cấu hình OCR (EasyOCR)."""

    langs: tuple[str, ...] = ("auto",)
    gpu: bool = True  # EasyOCR auto-fallback CPU nếu không có CUDA → an toàn để default ON
    auto_max_samples: int = 3
    gibberish_min_chars: int = 2
    gibberish_min_letter_ratio: float = 0.4
    crop_pad: int = 2


@dataclass
class TranslateConfig:
    """Cấu hình dịch. Hỗ trợ OpenRouter, local LLM (llama.cpp), OpenAI-compat."""

    enabled: bool = True
    target_lang: str = "Vietnamese"
    model: Optional[str] = None  # None → DEFAULT_OPENROUTER_MODEL
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.3
    top_p: float = 0.9

    # LLM backend: 'openrouter' | 'local_llm' | 'openai_compat'
    backend: str = "openrouter"

    # OpenAI-compatible API (cho vLLM, Ollama, text-generation-webui, etc.)
    api_base: Optional[str] = None  # e.g. http://localhost:8080/v1
    api_key: Optional[str] = None   # key cho self-hosted (often empty)

    # Glossary
    glossary_path: Optional[str] = None  # None → <output_dir>/.glossary.json
    use_glossary: bool = True

    # Reading order
    reading_band_ratio: float = 0.25  # y-band height ratio cho group


@dataclass
class RenderConfig:
    """Cấu hình render text dịch."""

    font_path: str = DEFAULT_FONT_CANDIDATES[0]
    font_candidates: Sequence[str] = field(default_factory=lambda: DEFAULT_FONT_CANDIDATES)
    min_size: int = 10
    max_size: int = 220
    padding: int = 4
    line_spacing: float = 1.05
    color_rgb: tuple[int, int, int] = (0, 0, 0)
    stroke_color_rgb: tuple[int, int, int] = (255, 255, 255)
    stroke_width: int = 2

    # Tier 2/4 fallback overrides
    tight_min_size: int = 10
    tight_padding: int = 2
    bbox_fallback_min_size: int = 8
    bbox_fallback_stroke: int = 1


@dataclass
class LocalLLMConfig:
    """Cấu hình local LLM via llama-cpp-python.

    Model GGUF auto-download từ HuggingFace nếu `model_path` là None.
    Registry name → repo/file mapping xem `LLM_MODEL_REGISTRY`.
    """

    model_name: str = "qwen3.5-4b"       # key trong LLM_MODEL_REGISTRY
    model_path: Optional[str] = None     # path GGUF; None = auto download
    n_ctx: int = 4096                    # context length
    n_gpu_layers: int = -1               # -1 = tất cả layers lên GPU
    n_threads: int = 4
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 2048
    models_dir: str = "models"           # thư mục lưu GGUF đã tải


@dataclass
class PipelineConfig:
    """Cấu hình tổng cho MangaPipeline. Compose từ sub-configs.

    Sau refactor 2026-05-25: bỏ các opt-in `use_*` cho ocr_router /
    redraw_engine / typography_engine / translation_pipeline / detect_stroke
    — luôn dùng engine mới (single path).

    Koharu integration (2026-05-27): thêm `local_llm` sub-config.
    """

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    cleaner: CleanerConfig = field(default_factory=CleanerConfig)
    inpaint: InpaintConfig = field(default_factory=InpaintConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    local_llm: LocalLLMConfig = field(default_factory=LocalLLMConfig)

    # Sub-configs cho VIP modules — lazy default trong __post_init__.
    language_detector: object = field(default=None)
    ocr_router: object = field(default=None)
    sfx_detector: object = field(default=None)
    redraw_engine: object = field(default=None)
    translation_pipeline: object = field(default=None)

    # Top-level
    log_level: str = "INFO"
    preserve_untranslated_cjk: bool = True   # giữ nguyên CJK SFX khi không dịch được
    use_language_detector: bool = True
    use_sfx_detector: bool = True            # SFX role classification

    def __post_init__(self) -> None:
        # Defer import để tránh vòng lặp config ↔ ocr_router.
        if self.language_detector is None:
            from .language_detector import LanguageDetectorConfig
            self.language_detector = LanguageDetectorConfig()
        if self.ocr_router is None:
            from .ocr_router import OCRRouterConfig
            self.ocr_router = OCRRouterConfig()
        if self.sfx_detector is None:
            from .sfx_detector import SFXDetectorConfig
            self.sfx_detector = SFXDetectorConfig()
        if self.redraw_engine is None:
            from .redraw_engine import RedrawEngineConfig
            self.redraw_engine = RedrawEngineConfig()
        if self.translation_pipeline is None:
            from .translation_pipeline import TranslationPipelineConfig
            self.translation_pipeline = TranslationPipelineConfig()
