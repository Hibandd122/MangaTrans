"""CheckpointStore — JSON sidecar resume state per chap.

Schema v1: 1 file `.mangatrans_state.json` ở output dir, chứa toàn bộ PageTask
snapshot. Atomic write qua tmp → os.replace (Windows-safe). Debounce 1s để
tránh fsync mỗi stage trên chap 100+ page.

Tại sao JSON thay vì SQLite:
- 1 chap ~100 page × ~500B/page = ~50KB → JSON đọc/ghi nhanh, dễ debug.
- Không cần concurrent reader (1 process duy nhất).
- User có thể mở bằng editor xem state.

Resume policy (đã chốt trong plan):
- DONE + output file exists → skip.
- DONE + output file missing → rerun (file bị xoá ngoài).
- RUNNING/FAILED → rerun from scratch (partial in-memory state không persist).
- Config digest mismatch → log warning, rerun all (override `force_resume`).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .page_task import PageState, PageTask, StageName

SCHEMA_VERSION = 1


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def compute_config_digest(payload: Any) -> str:
    """sha256 digest của config tuỳ ý.

    Dùng để invalidate checkpoint khi user đổi config quan trọng (model,
    target_lang, …). Recursively encode dict/list/dataclass.
    """
    norm = _normalize(payload)
    blob = json.dumps(norm, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _normalize(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if is_dataclass(v):
        return _normalize(asdict(v))
    if isinstance(v, dict):
        return {str(k): _normalize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_normalize(x) for x in v]
    return repr(v)


def atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    """Atomic write JSON via tmp + os.replace. Idempotent dir creation.

    Trên Windows os.replace cross-volume sẽ fail. Caller phải đảm bảo path nằm
    trên cùng filesystem (tmp file tạo trong CÙNG dir của target).
    """
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    # delete=False để ta tự control move; suffix .tmp để dễ identify nếu crash giữa chừng.
    fd, tmp_path = tempfile.mkstemp(prefix=".ckpt_", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # Best-effort cleanup tmp; không che lỗi gốc.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


class CheckpointStore:
    """In-memory dict + debounced flush ra disk. Thread-safe.

    Lifecycle:
        store = CheckpointStore(path, config_digest=digest)
        store.load()                          # restore previous state
        for task in tasks:
            store.upsert(task)
            store.flush_if_due()
        store.flush()                         # final sync
    """

    def __init__(
        self,
        path: str,
        config_digest: Optional[str] = None,
        debounce_s: float = 1.0,
    ):
        self.path = path
        self.config_digest = config_digest or ""
        self.debounce_s = float(max(0.0, debounce_s))

        self._lock = threading.Lock()
        self._tasks: Dict[str, PageTask] = {}
        self._last_flush_ts: float = 0.0
        self._dirty: bool = False
        self._loaded_digest: Optional[str] = None
        self._loaded_at_iso: Optional[str] = None

    # ---- Load / save ----

    def load(self) -> Dict[str, PageTask]:
        """Đọc checkpoint nếu tồn tại. Không raise nếu file hỏng — log + reset."""
        if not os.path.isfile(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            sys.stderr.write(f"[CheckpointStore] load failed ({exc}); starting fresh\n")
            return {}
        if not isinstance(data, dict):
            return {}
        if int(data.get("version", 0)) != SCHEMA_VERSION:
            sys.stderr.write(
                f"[CheckpointStore] schema version mismatch "
                f"({data.get('version')} vs {SCHEMA_VERSION}); starting fresh\n"
            )
            return {}
        self._loaded_digest = data.get("config_digest")
        self._loaded_at_iso = data.get("ts")
        tasks_raw = data.get("pages", {}) or {}
        loaded: Dict[str, PageTask] = {}
        for pid, payload in tasks_raw.items():
            try:
                loaded[pid] = PageTask.from_dict(payload)
            except (KeyError, ValueError, TypeError) as exc:
                sys.stderr.write(
                    f"[CheckpointStore] skip corrupted entry {pid}: {exc}\n"
                )
        with self._lock:
            self._tasks = loaded
        return loaded

    def config_changed(self) -> bool:
        """True nếu config_digest hiện tại khác file đã load. False nếu chưa load
        hoặc file không có digest."""
        if self._loaded_digest is None:
            return False
        return self._loaded_digest != self.config_digest

    @property
    def loaded_digest(self) -> Optional[str]:
        return self._loaded_digest

    # ---- Mutators ----

    def upsert(self, task: PageTask) -> None:
        with self._lock:
            self._tasks[task.page_id] = task
            self._dirty = True

    def mark_state(self, page_id: str, state: PageState, error: Optional[str] = None) -> None:
        with self._lock:
            t = self._tasks.get(page_id)
            if t is None:
                return
            t.state = state
            if error is not None:
                t.last_error = error
            if state == PageState.DONE:
                t.completed_ts = _utc_iso()
            self._dirty = True

    def mark_stage_complete(self, page_id: str, stage: StageName) -> None:
        with self._lock:
            t = self._tasks.get(page_id)
            if t is None:
                return
            t.mark_stage_complete(stage)
            self._dirty = True

    def get(self, page_id: str) -> Optional[PageTask]:
        with self._lock:
            return self._tasks.get(page_id)

    def all_tasks(self) -> List[PageTask]:
        with self._lock:
            return list(self._tasks.values())

    def clear(self) -> None:
        with self._lock:
            self._tasks.clear()
            self._dirty = True

    # ---- Flush ----

    def flush_if_due(self) -> bool:
        """Flush ra disk nếu đã quá `debounce_s` từ flush trước."""
        now = time.perf_counter()
        with self._lock:
            if not self._dirty:
                return False
            if now - self._last_flush_ts < self.debounce_s:
                return False
        return self.flush()

    def flush(self) -> bool:
        """Force flush (ignore debounce). Trả True nếu đã viết, False nếu skip."""
        with self._lock:
            if not self._dirty:
                return False
            snapshot = {
                "version": SCHEMA_VERSION,
                "config_digest": self.config_digest,
                "ts": _utc_iso(),
                "pages": {pid: t.to_dict() for pid, t in self._tasks.items()},
            }
        try:
            atomic_write_json(self.path, snapshot)
        except OSError as exc:
            sys.stderr.write(f"[CheckpointStore] flush failed: {exc}\n")
            return False
        with self._lock:
            self._dirty = False
            self._last_flush_ts = time.perf_counter()
        return True

    # ---- Resume helpers ----

    def should_skip(self, task: PageTask) -> bool:
        """True nếu task đã DONE và output file tồn tại."""
        prev = self.get(task.page_id)
        if prev is None or prev.state != PageState.DONE:
            return False
        out = prev.output_path or task.output_path
        return bool(out) and os.path.isfile(out)

    def restore_into(self, tasks: Iterable[PageTask], reset_partial: bool = True) -> List[PageTask]:
        """Cập nhật `tasks` từ snapshot đã load. Trả list tasks đã merge.

        `reset_partial=True`: page state RUNNING/FAILED → reset về PENDING (rerun
        từ đầu); DONE giữ nguyên. Lý do: in-memory ctx không persist nên không
        thể resume từ giữa stage.
        """
        out: List[PageTask] = []
        for t in tasks:
            prev = self.get(t.page_id)
            if prev is None:
                self.upsert(t)
                out.append(t)
                continue
            if prev.state == PageState.DONE:
                # Inject summary, completed_stages từ prev.
                t.state = PageState.DONE
                t.completed_stages = list(prev.completed_stages)
                t.completed_ts = prev.completed_ts
                t.retries = prev.retries
            elif reset_partial:
                t.state = PageState.PENDING
                t.current_stage = None
                t.completed_stages = []
                t.retries = prev.retries
                t.last_error = prev.last_error
            self.upsert(t)
            out.append(t)
        return out
