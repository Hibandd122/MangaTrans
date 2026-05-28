import threading
from typing import Any, Dict

_PADDLE_CACHE: Dict[str, Any] = {}
_LOCK = threading.Lock()

def get_paddleocr(lang: str, enable_mkldnn: bool = False, **kwargs):
    """Lấy instance PaddleOCR từ global cache để tránh lỗi 'PDX has already been initialized' 
    và tiết kiệm thời gian load model nhiều lần.
    """
    key = f"{lang}_{enable_mkldnn}_{kwargs}"
    with _LOCK:
        if key not in _PADDLE_CACHE:
            import os
            if not enable_mkldnn:
                os.environ.setdefault("FLAGS_use_mkldnn", "false")
                os.environ.setdefault("FLAGS_enable_pir_in_executor", "false")
            
            # Xử lý riêng use_vl cho tương thích ngược
            use_vl = kwargs.pop("use_vl", False)
            if use_vl:
                try:
                    from paddleocr import PaddleOCRVL as _PaddleOCR
                except ImportError:
                    from paddleocr import PaddleOCR as _PaddleOCR
                    kwargs["use_vl"] = True
            else:
                from paddleocr import PaddleOCR as _PaddleOCR

            _PADDLE_CACHE[key] = _PaddleOCR(
                lang=lang, 
                enable_mkldnn=enable_mkldnn,
                **kwargs
            )
        return _PADDLE_CACHE[key]
