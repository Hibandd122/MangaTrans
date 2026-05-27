"""OCR helpers — gibberish filter (legacy `OCREngine` class đã bị xóa).

Sau refactor 2026-05-25 luôn dùng `OCRRouter` (ocr_router.py) — file này
chỉ còn `is_likely_gibberish()` được pipeline.py / tests dùng.
"""
from __future__ import annotations


# --------------------------- Gibberish filter --------------------------- #

def is_likely_gibberish(text: str, min_chars: int = 2,
                       min_letter_ratio: float = 0.4) -> bool:
    """True nếu text RÁC HOÀN TOÀN — chỉ skip nếu không có nghĩa thật.

    Note v15: trước filter mạnh hơn (loại "DIs-", "OIs?") nhưng user muốn giữ lại
    để LLM đoán SFX gốc → relax chỉ skip case empty/không-letter.
    """
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
