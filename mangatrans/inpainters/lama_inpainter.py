"""LaMa inpainter — TorchScript .pt only (anime-manga-big-lama)."""
from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..config import InpaintConfig
from ..inpainter import BaseInpainter


@dataclass
class _SessionEntry:
    obj: object          # torch.jit.ScriptModule
    device: str          # 'cuda' | 'cpu'
    target_size: int


class LamaInpainter(BaseInpainter):
    """Wrap TorchScript LaMa model (.pt). 2 session slot: GPU mặc định + CPU fallback."""

    def __init__(self, config: InpaintConfig):
        super().__init__(config)
        self.model_path = config.model_path
        self._override_size = getattr(config, 'override_size', None)
        
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Không tìm thấy inpaint model: {self.model_path}")
        if not self.model_path.lower().endswith(".pt"):
            raise ValueError(
                "LamaInpainter chỉ hỗ trợ TorchScript .pt "
                "(ONNX đã bị bỏ — dùng anime-manga-big-lama.pt)."
            )

        self._entry: Optional[_SessionEntry] = None
        self._cpu_entry: Optional[_SessionEntry] = None
        self._force_cpu = False
        self._lock = threading.Lock()

    def _ensure_session(self, want_cpu: bool = False) -> _SessionEntry:
        import torch
        with self._lock:
            if want_cpu:
                if self._cpu_entry is not None:
                    return self._cpu_entry
                model = torch.jit.load(self.model_path, map_location="cpu").eval()
                target = self._override_size or 768
                self._cpu_entry = _SessionEntry(model, "cpu", target)
                return self._cpu_entry
            if self._entry is not None:
                return self._entry
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = torch.jit.load(self.model_path, map_location=device).eval()
            target = self._override_size or 768
            self._entry = _SessionEntry(model, device, target)
            return self._entry

    @contextlib.contextmanager
    def force_cpu_mode(self):
        prev = self._force_cpu
        self._force_cpu = True
        try:
            yield
        finally:
            self._force_cpu = prev

    @property
    def target_size(self) -> int:
        entry = self._ensure_session(want_cpu=self._force_cpu)
        return entry.target_size

    def run_tile(self, image_bgr: np.ndarray, mask_u8: np.ndarray,
                 target_size: Optional[int] = None,
                 force_cpu: bool = False) -> np.ndarray:
        import torch
        use_cpu = bool(force_cpu or self._force_cpu)
        entry = self._ensure_session(want_cpu=use_cpu)
        size = target_size or entry.target_size
        h, w = image_bgr.shape[:2]

        if h <= size and w <= size:
            pad_h = size - h
            pad_w = size - w
            top = pad_h // 2
            bot = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            img_r = cv2.copyMakeBorder(image_bgr, top, bot, left, right, cv2.BORDER_REFLECT_101)
            msk_r = cv2.copyMakeBorder(mask_u8, top, bot, left, right, cv2.BORDER_CONSTANT, value=0)
            crop_after = (top, left, h, w)
        else:
            img_r = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
            msk_r = cv2.resize(mask_u8, (size, size), interpolation=cv2.INTER_NEAREST)
            crop_after = None

        img_rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
        img_t = (torch.from_numpy(img_rgb).float().div(255.0)
                 .permute(2, 0, 1).unsqueeze(0).to(entry.device))
        msk_t = (torch.from_numpy((msk_r > 127).astype(np.float32))
                 .unsqueeze(0).unsqueeze(0).to(entry.device))
        with torch.inference_mode():
            out_t = entry.obj(img_t, msk_t)
        out = out_t[0].permute(1, 2, 0).detach().cpu().numpy()
        if out.max() > 1.5:
            out = np.clip(out, 0, 255).astype(np.uint8)
        else:
            out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        if crop_after is not None:
            top, left, oh, ow = crop_after
            return out[top:top + oh, left:left + ow]
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)

    def release(self) -> None:
        with self._lock:
            if self._entry is not None:
                del self._entry.obj
                self._entry = None
            if self._cpu_entry is not None:
                del self._cpu_entry.obj
                self._cpu_entry = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
