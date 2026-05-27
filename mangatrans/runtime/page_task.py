"""PageTask + StageName — đơn vị công việc trong scheduler.

Tại sao có dataclass riêng (không reuse `dict`):
- Type-safe state transitions (PENDING → RUNNING → DONE | FAILED).
- `completed_stages` rõ ràng để resume từ stage giữa chừng.
- `to_dict`/`from_dict` cho checkpoint round-trip (JSON sidecar).
- Track `retries`, `fallback_flags` (cpu_inpaint, …) cho diagnostics.
"""
from __future__ import annotations

import enum
import hashlib
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set


class StageName(str, enum.Enum):
    """Stage trong pipeline async. Order matches `Scheduler._STAGE_ORDER`.

    Stringify-as-value cho JSON checkpoint dễ đọc.
    """

    LOAD = "load"
    DETECT = "detect"
    LANG = "lang_detect"
    OCR = "ocr"
    SFX = "sfx"
    TRANSLATE = "translate"
    SAVE_JSON = "save_json"
    PRESERVE_CLEAN = "preserve_clean"
    INPAINT = "inpaint"
    RENDER = "render"
    SAVE_PNG = "save_png"

    @classmethod
    def ordered(cls) -> List["StageName"]:
        """Thứ tự thực thi cố định cho 1 page."""
        return [
            cls.LOAD, cls.DETECT, cls.LANG, cls.OCR, cls.SFX, cls.TRANSLATE,
            cls.SAVE_JSON, cls.PRESERVE_CLEAN, cls.INPAINT, cls.RENDER, cls.SAVE_PNG,
        ]


class PageState(str, enum.Enum):
    """Macro-state của 1 page trong checkpoint."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


# Các stage cần GPU mutex — scheduler sẽ acquire `GPUMutex` trước khi gọi.
GPU_STAGES: frozenset = frozenset({
    StageName.DETECT, StageName.LANG, StageName.OCR, StageName.INPAINT,
})


def compute_page_id(input_path: str) -> str:
    """Stable page ID từ abs path. 16 hex char đủ tránh collision trong 1 chap."""
    abs_path = os.path.abspath(input_path)
    digest = hashlib.sha256(abs_path.encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass
class PageTask:
    """Đơn vị công việc cho scheduler. Mutable — scheduler update tại runtime."""

    page_id: str
    input_path: str
    output_path: str

    state: PageState = PageState.PENDING
    current_stage: Optional[StageName] = None
    completed_stages: List[StageName] = field(default_factory=list)

    retries: int = 0
    stage_retries: Dict[str, int] = field(default_factory=dict)
    oom_retries: int = 0
    fallback_flags: Set[str] = field(default_factory=set)

    last_error: Optional[str] = None
    started_ts: Optional[str] = None
    completed_ts: Optional[str] = None

    # Trong-memory only, không persist.
    cancel_requested: bool = field(default=False, repr=False)
    last_heartbeat_ts: float = field(default=0.0, repr=False)
    summary: Optional[Dict[str, Any]] = field(default=None, repr=False)

    @classmethod
    def new(cls, input_path: str, output_path: str) -> "PageTask":
        return cls(
            page_id=compute_page_id(input_path),
            input_path=os.path.abspath(input_path),
            output_path=os.path.abspath(output_path),
        )

    # ---- State transition helpers ----

    def mark_stage_complete(self, stage: StageName) -> None:
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)
        if self.current_stage == stage:
            self.current_stage = None

    def is_stage_complete(self, stage: StageName) -> bool:
        return stage in self.completed_stages

    def is_done(self) -> bool:
        return self.state == PageState.DONE

    # ---- Serialization ----

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict cho checkpoint."""
        data = asdict(self)
        # Loại field volatile.
        for k in ("cancel_requested", "last_heartbeat_ts", "summary"):
            data.pop(k, None)
        data["state"] = self.state.value
        data["current_stage"] = self.current_stage.value if self.current_stage else None
        data["completed_stages"] = [s.value for s in self.completed_stages]
        data["fallback_flags"] = sorted(self.fallback_flags)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageTask":
        completed = [StageName(s) for s in data.get("completed_stages", [])]
        cur = data.get("current_stage")
        return cls(
            page_id=data["page_id"],
            input_path=data["input_path"],
            output_path=data["output_path"],
            state=PageState(data.get("state", "PENDING")),
            current_stage=StageName(cur) if cur else None,
            completed_stages=completed,
            retries=int(data.get("retries", 0)),
            stage_retries=dict(data.get("stage_retries", {})),
            oom_retries=int(data.get("oom_retries", 0)),
            fallback_flags=set(data.get("fallback_flags", [])),
            last_error=data.get("last_error"),
            started_ts=data.get("started_ts"),
            completed_ts=data.get("completed_ts"),
        )
