"""Unit tests cho pure functions của mangatrans.

Run: pytest tests/test_unit.py -v
"""
import numpy as np
import pytest

from mangatrans.detector import dedupe_blocks, split_merged_bubbles
from mangatrans.geometry import (
    InteriorCache,
    exclude_other_bbox_regions,
    largest_inscribed_rect,
    polygon_row_extents,
    row_extents,
)
from mangatrans.inpainter import (
    classify_component_texture,
    fill_gradient,
    fill_solid,
)
from mangatrans.ocr import is_likely_gibberish
from mangatrans.translate import (
    extract_glossary_entries,
    position_tag,
    reading_order_indices,
)
from mangatrans.text import wrap_text, wrap_text_shape
from mangatrans.utils import clamp_bbox, has_latin_letters, has_real_cjk


# --------------------------- detector --------------------------- #

class TestDedupe:
    def test_empty(self):
        assert dedupe_blocks([]) == []

    def test_keeps_larger_when_contained(self):
        big = {"bbox": [0, 0, 100, 100], "score": 0.9, "cls": 0}
        small = {"bbox": [10, 10, 50, 50], "score": 0.8, "cls": 0}
        out = dedupe_blocks([big, small], contain_thresh=0.5)
        assert len(out) == 1
        assert out[0]["bbox"] == [0, 0, 100, 100]

    def test_no_overlap_keeps_both(self):
        a = {"bbox": [0, 0, 50, 50], "score": 0.9, "cls": 0}
        b = {"bbox": [100, 100, 150, 150], "score": 0.8, "cls": 0}
        out = dedupe_blocks([a, b])
        assert len(out) == 2

    def test_drops_zero_area(self):
        zero = {"bbox": [0, 0, 0, 0], "score": 0.9, "cls": 0}
        valid = {"bbox": [10, 10, 50, 50], "score": 0.8, "cls": 0}
        out = dedupe_blocks([zero, valid])
        assert len(out) == 1
        assert out[0]["bbox"] == [10, 10, 50, 50]


class TestSplitMerged:
    def test_passthrough_no_gap(self):
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[10:30, 10:30] = 255  # 1 component
        blk = {"bbox": [0, 0, 200, 100], "score": 0.9, "cls": 0}
        out = split_merged_bubbles([blk], mask)
        assert len(out) == 1

    def test_splits_two_separated_clusters(self):
        mask = np.zeros((100, 400), dtype=np.uint8)
        mask[40:60, 10:80] = 255   # cluster 1
        mask[40:60, 300:380] = 255  # cluster 2 (gap > 20)
        blk = {"bbox": [0, 0, 400, 100], "score": 0.9, "cls": 0}
        out = split_merged_bubbles([blk], mask, gap_thresh=20, min_comp_area=10)
        assert len(out) == 2


# --------------------------- geometry --------------------------- #

class TestRowExtents:
    def test_empty_mask(self):
        m = np.zeros((10, 20), dtype=np.uint8)
        assert row_extents(m) == [None] * 10

    def test_single_run(self):
        m = np.zeros((3, 20), dtype=np.uint8)
        m[1, 5:15] = 255
        out = row_extents(m)
        assert out[0] is None
        assert out[1] == (5, 14)
        assert out[2] is None

    def test_picks_longest_run(self):
        m = np.zeros((1, 20), dtype=np.uint8)
        m[0, 1:3] = 255   # run 2px
        m[0, 5:15] = 255  # run 10px — should win
        out = row_extents(m)
        assert out[0] == (5, 14)


class TestPolygonExtents:
    def test_triangle(self):
        # Tam giác đỉnh trên, đáy ở y=10
        contour = np.array([[5, 0], [0, 10], [10, 10]])
        out = polygon_row_extents(contour, 12)
        # Tại y=5 (midline), x range ~ [2.5, 7.5]
        assert out[5] is not None
        x1, x2 = out[5]
        assert 0 <= x1 < 5 < x2 <= 10

    def test_degenerate_too_few_points(self):
        contour = np.array([[0, 0], [10, 10]])
        out = polygon_row_extents(contour, 12)
        assert out == [None] * 12


class TestInscribedRect:
    def test_filled_region_with_border(self):
        # distanceTransform yêu cầu có pixel = 0 → để border 5px
        m = np.zeros((100, 100), dtype=np.uint8)
        m[5:95, 5:95] = 255
        r = largest_inscribed_rect(m)
        assert r is not None
        x1, y1, x2, y2 = r
        assert x2 - x1 > 50 and y2 - y1 > 50

    def test_empty_mask(self):
        m = np.zeros((100, 100), dtype=np.uint8)
        assert largest_inscribed_rect(m) is None


class TestExcludeOther:
    def test_passthrough_no_others(self):
        m = np.ones((50, 50), dtype=np.uint8) * 255
        out = exclude_other_bbox_regions(m, [0, 0, 50, 50], None)
        assert np.array_equal(out, m)

    def test_drops_closer_to_other(self):
        m = np.zeros((100, 100), dtype=np.uint8)
        m[10:20, 60:70] = 255  # gần bbox other hơn bbox self
        out = exclude_other_bbox_regions(
            m, bbox=[0, 0, 10, 10], other_bboxes=[[60, 0, 70, 10]],
        )
        assert out.sum() < m.sum()


class TestInteriorCache:
    def test_get_or_compute_caches(self):
        cache = InteriorCache()
        img = np.full((50, 50), 220, dtype=np.uint8)
        r1 = cache.get_or_compute(img, 200, 3)
        r2 = cache.get_or_compute(img, 200, 3)
        assert r1 is r2  # identity check — same tuple returned


# --------------------------- inpainter texture --------------------------- #

class TestTextureClassify:
    def test_solid_rim_returns_solid(self):
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        kind, params = classify_component_texture(img, mask, rim_width=8)
        assert kind == "SOLID"
        assert params is not None and "color" in params

    def test_textured_rim_returns_texture(self):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        kind, _ = classify_component_texture(img, mask, rim_width=8)
        assert kind == "TEXTURE"

    def test_tiny_mask_returns_texture(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[10, 10] = 255
        kind, _ = classify_component_texture(img, mask, rim_width=1)
        # rim 1px quanh 1 pixel = ~8 pixels < 20 → TEXTURE fallback
        assert kind == "TEXTURE"


class TestFills:
    def test_fill_solid(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 2:5] = 255
        out = fill_solid(img, mask, np.array([10, 20, 30], dtype=np.uint8))
        assert (out[2:5, 2:5] == [10, 20, 30]).all()
        assert (out[0, 0] == 0).all()  # ngoài mask không đổi

    def test_fill_gradient_dark_ref(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 2:5] = 255
        coef = np.array([10.0, 0.0, 0.0])  # const L
        ref = np.array([0, 0, 0], dtype=np.uint8)  # ref_L ~ 0
        out = fill_gradient(img, mask, coef, ref)
        # Vùng dark: scale guard kích hoạt → giữ ref_color flat
        assert out.shape == img.shape


# --------------------------- ocr filter --------------------------- #

class TestGibberish:
    def test_empty(self):
        assert is_likely_gibberish("")
        assert is_likely_gibberish("   ")

    def test_pure_punct(self):
        assert is_likely_gibberish("!!!")
        assert is_likely_gibberish("...")

    def test_normal_word_not_gibberish(self):
        assert not is_likely_gibberish("hello")

    def test_cjk_not_gibberish(self):
        assert not is_likely_gibberish("こんにちは")

    def test_short_sfx_kept(self):
        # User wants Gemini to guess SFX — relax filter
        assert not is_likely_gibberish("Eh?")


# --------------------------- translate --------------------------- #

class TestPositionTag:
    def test_corners(self):
        # 1000x1000 page
        assert position_tag([0, 0, 100, 100], 1000, 1000) == "top-left"
        assert position_tag([900, 900, 1000, 1000], 1000, 1000) == "bottom-right"
        assert position_tag([400, 400, 600, 600], 1000, 1000) == "middle-center"


class TestReadingOrder:
    def test_rtl_order(self):
        # Manga RTL: x càng lớn → đọc trước trong cùng band
        results = [
            {"bbox": [100, 100, 200, 200]},  # left
            {"bbox": [400, 100, 500, 200]},  # right (đọc trước)
        ]
        idx = reading_order_indices(results, page_h=400)
        assert idx == [1, 0]

    def test_top_band_before_bottom(self):
        results = [
            {"bbox": [100, 350, 200, 400]},  # bottom
            {"bbox": [100, 50, 200, 100]},   # top (đọc trước)
        ]
        idx = reading_order_indices(results, page_h=400)
        assert idx == [1, 0]


class TestGlossaryExtract:
    def test_capitalized_name_preserved(self):
        srcs = ["Chishiro is here"]
        tgts = ["Chishiro đang ở đây"]
        pairs = extract_glossary_entries(srcs, tgts)
        assert pairs == {"Chishiro": "Chishiro"}

    def test_blacklisted_word_skipped(self):
        srcs = ["Nothing happened"]
        tgts = ["Nothing đã xảy ra"]
        pairs = extract_glossary_entries(srcs, tgts)
        assert "Nothing" not in pairs

    def test_all_caps_normalized(self):
        srcs = ["CHISHIRO speaks"]
        tgts = ["Chishiro nói"]
        pairs = extract_glossary_entries(srcs, tgts)
        assert pairs == {"CHISHIRO": "Chishiro"}


# --------------------------- text wrap --------------------------- #

class TestWrap:
    @pytest.fixture
    def font_draw(self):
        from PIL import Image, ImageDraw, ImageFont
        # Mock font với fixed char width via PIL default
        font = ImageFont.load_default()
        img = Image.new("RGB", (200, 50))
        draw = ImageDraw.Draw(img)
        return font, draw

    def test_wrap_empty(self, font_draw):
        font, draw = font_draw
        assert wrap_text("", font, 100, draw) == []

    def test_wrap_short_text_one_line(self, font_draw):
        font, draw = font_draw
        out = wrap_text("hi", font, 1000, draw)
        assert out == ["hi"]

    def test_wrap_shape_returns_none_for_oversize_word(self, font_draw):
        font, draw = font_draw
        # 0 budget cho mọi dòng → fail
        out = wrap_text_shape("hello world", font, lambda i: 0, draw)
        assert out is None


# --------------------------- utils --------------------------- #

class TestCJKDetection:
    def test_hiragana(self):
        assert has_real_cjk("ありがとう")

    def test_katakana(self):
        assert has_real_cjk("カタカナ")

    def test_han(self):
        assert has_real_cjk("漢字")

    def test_hangul(self):
        assert has_real_cjk("한글")

    def test_latin_only(self):
        assert not has_real_cjk("hello world")

    def test_empty(self):
        assert not has_real_cjk("")


class TestLatinDetection:
    def test_pure_latin(self):
        assert has_latin_letters("hello", min_count=2)

    def test_below_threshold(self):
        assert not has_latin_letters("a", min_count=2)

    def test_punct_only(self):
        assert not has_latin_letters("!!!")


class TestClampBbox:
    def test_in_bounds(self):
        assert clamp_bbox([10, 10, 50, 50], 100, 100) == (10, 10, 50, 50)

    def test_neg_clamped_to_zero(self):
        assert clamp_bbox([-5, -3, 20, 30], 100, 100) == (0, 0, 20, 30)

    def test_over_bounds_clamped(self):
        assert clamp_bbox([10, 10, 200, 200], 100, 100) == (10, 10, 100, 100)

    def test_ensures_min_size_one(self):
        # x2<=x1 → x2 = x1+1
        x1, y1, x2, y2 = clamp_bbox([50, 50, 50, 50], 100, 100)
        assert x2 > x1 and y2 > y1
