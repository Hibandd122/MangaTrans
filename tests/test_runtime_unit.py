"""Unit tests cho `mangatrans/runtime/` foundation modules (Commit 1).

Phạm vi commit 1: config, structured_log, metrics, page_task.
Test phải chạy độc lập, không load model, không cần GPU/network.

Chạy: `pytest -p no:seleniumbase tests/test_runtime_unit.py -v`
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from mangatrans.runtime.config import DEFAULT_STAGE_TIMEOUT_S, RuntimeConfig
from mangatrans.runtime.metrics import HealthPrinter, MetricsRegistry
from mangatrans.runtime.page_task import (
    GPU_STAGES,
    PageState,
    PageTask,
    StageName,
    compute_page_id,
)
from mangatrans.runtime.structured_log import (
    NullStructuredLogger,
    StructuredLogger,
    _jsonable,
)


# --------------------------- RuntimeConfig --------------------------- #

class TestRuntimeConfig:
    def test_defaults_safe(self):
        cfg = RuntimeConfig()
        assert cfg.enable_async is True
        assert cfg.pipeline_depth == 3
        assert cfg.translation_rpm == 20
        assert cfg.max_page_retries >= 1
        # Watchdog có timeout cho mọi stage chính.
        for stage in ("detect", "ocr", "translate", "inpaint"):
            assert cfg.stage_timeout(stage) > 0

    def test_stage_timeout_unknown_fallback(self):
        cfg = RuntimeConfig()
        # Stage không có → fallback ~120s.
        assert cfg.stage_timeout("nonexistent_stage") == 120.0

    def test_for_legacy_sync_disables_everything(self):
        cfg = RuntimeConfig().for_legacy_sync()
        assert cfg.enable_async is False
        assert cfg.watchdog_enable is False
        assert cfg.enable_checkpoint is False
        assert cfg.enable_translation_cache is False
        assert cfg.raise_translation_errors is False

    def test_stage_timeout_dict_isolated(self):
        # Mỗi instance phải có copy riêng để mutate không leak.
        c1 = RuntimeConfig()
        c2 = RuntimeConfig()
        c1.watchdog_stage_timeout_s["detect"] = 999.0
        assert c2.watchdog_stage_timeout_s["detect"] == DEFAULT_STAGE_TIMEOUT_S["detect"]


# --------------------------- PageTask / StageName --------------------------- #

class TestPageTask:
    def test_stage_ordered(self):
        order = StageName.ordered()
        assert order[0] == StageName.LOAD
        assert StageName.INPAINT in order
        assert order[-1] == StageName.SAVE_PNG
        # Tất cả unique.
        assert len(order) == len(set(order))

    def test_gpu_stages_subset(self):
        # GPU_STAGES phải là subset của StageName.
        for s in GPU_STAGES:
            assert isinstance(s, StageName)
        assert StageName.DETECT in GPU_STAGES
        assert StageName.INPAINT in GPU_STAGES
        assert StageName.SAVE_PNG not in GPU_STAGES

    def test_compute_page_id_stable(self):
        a = compute_page_id("D:/x/y/page1.jpg")
        b = compute_page_id("D:/x/y/page1.jpg")
        assert a == b
        assert len(a) == 16
        c = compute_page_id("D:/x/y/page2.jpg")
        assert a != c

    def test_compute_page_id_uses_abs(self, tmp_path):
        # Relative vs absolute cùng file phải ra cùng id (qua abspath).
        rel = "./page.jpg"
        abs_p = os.path.abspath(rel)
        assert compute_page_id(rel) == compute_page_id(abs_p)

    def test_new_constructor(self):
        pt = PageTask.new("page.jpg", "out.png")
        assert pt.state == PageState.PENDING
        assert pt.retries == 0
        assert pt.completed_stages == []
        assert pt.fallback_flags == set()
        assert os.path.isabs(pt.input_path)

    def test_mark_stage_complete_idempotent(self):
        pt = PageTask.new("a.jpg", "a.png")
        pt.current_stage = StageName.DETECT
        pt.mark_stage_complete(StageName.DETECT)
        pt.mark_stage_complete(StageName.DETECT)  # idempotent
        assert pt.completed_stages == [StageName.DETECT]
        assert pt.current_stage is None
        assert pt.is_stage_complete(StageName.DETECT)
        assert not pt.is_stage_complete(StageName.OCR)

    def test_to_dict_from_dict_roundtrip(self):
        pt = PageTask.new("x.jpg", "x.png")
        pt.state = PageState.RUNNING
        pt.current_stage = StageName.OCR
        pt.completed_stages = [StageName.LOAD, StageName.DETECT, StageName.LANG]
        pt.retries = 1
        pt.stage_retries = {"ocr": 2}
        pt.oom_retries = 1
        pt.fallback_flags = {"cpu_inpaint"}
        pt.last_error = "boom"
        pt.started_ts = "2026-05-25T01:02:03Z"
        d = pt.to_dict()
        # JSON serializable
        s = json.dumps(d)
        d2 = json.loads(s)
        restored = PageTask.from_dict(d2)
        assert restored.page_id == pt.page_id
        assert restored.input_path == pt.input_path
        assert restored.state == PageState.RUNNING
        assert restored.current_stage == StageName.OCR
        assert restored.completed_stages == [StageName.LOAD, StageName.DETECT, StageName.LANG]
        assert restored.retries == 1
        assert restored.stage_retries == {"ocr": 2}
        assert restored.oom_retries == 1
        assert restored.fallback_flags == {"cpu_inpaint"}
        assert restored.last_error == "boom"

    def test_to_dict_omits_volatile(self):
        pt = PageTask.new("y.jpg", "y.png")
        pt.cancel_requested = True
        pt.last_heartbeat_ts = 12345.0
        pt.summary = {"x": 1}
        d = pt.to_dict()
        assert "cancel_requested" not in d
        assert "last_heartbeat_ts" not in d
        assert "summary" not in d

    def test_from_dict_minimal(self):
        # Backward-compat: trường thiếu phải có default sensible.
        pt = PageTask.from_dict({
            "page_id": "abc123def4567890",
            "input_path": "/p/a.jpg",
            "output_path": "/p/a.png",
        })
        assert pt.state == PageState.PENDING
        assert pt.retries == 0
        assert pt.completed_stages == []
        assert pt.current_stage is None


# --------------------------- StructuredLogger --------------------------- #

class TestStructuredLogger:
    def test_null_logger_no_file(self):
        lg = NullStructuredLogger(ring_size=5)
        lg.event("test", x=1)
        assert len(lg.snapshot_recent()) == 1
        assert lg.snapshot_recent()[0]["event"] == "test"
        lg.close()

    def test_ring_buffer_size(self):
        lg = NullStructuredLogger(ring_size=3)
        for i in range(10):
            lg.event("e", i=i)
        recent = lg.snapshot_recent()
        assert len(recent) == 3
        # Latest 3.
        assert [r["i"] for r in recent] == [7, 8, 9]

    def test_snapshot_recent_n(self):
        lg = NullStructuredLogger(ring_size=10)
        for i in range(5):
            lg.event("e", i=i)
        assert len(lg.snapshot_recent(2)) == 2
        assert lg.snapshot_recent(0) == []
        assert len(lg.snapshot_recent(None)) == 5
        assert len(lg.snapshot_recent(100)) == 5  # ring cap

    def test_jsonl_write(self, tmp_path: Path):
        p = tmp_path / "log.jsonl"
        lg = StructuredLogger(str(p), ring_size=5)
        lg.event("hello", page_id="abc", n=42)
        lg.event("bye", page_id="abc", reason="done")
        lg.close()
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        r0 = json.loads(lines[0])
        assert r0["event"] == "hello"
        assert r0["page_id"] == "abc"
        assert r0["n"] == 42
        assert "ts" in r0
        r1 = json.loads(lines[1])
        assert r1["event"] == "bye"

    def test_jsonl_skips_none_fields(self, tmp_path: Path):
        p = tmp_path / "log.jsonl"
        lg = StructuredLogger(str(p), ring_size=5)
        lg.event("x", a=1, b=None, c="ok")
        lg.close()
        rec = json.loads(p.read_text(encoding="utf-8").strip())
        assert "b" not in rec
        assert rec["a"] == 1
        assert rec["c"] == "ok"

    def test_stage_timer_success(self, tmp_path: Path):
        p = tmp_path / "log.jsonl"
        lg = StructuredLogger(str(p), ring_size=10)
        with lg.stage_timer("detect", page_id="p1") as info:
            info["bubbles"] = 3
        lg.close()
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        start = json.loads(lines[0])
        end = json.loads(lines[1])
        assert start["event"] == "stage_start"
        assert end["event"] == "stage_end"
        assert end["ok"] is True
        assert end["bubbles"] == 3
        assert "latency_ms" in end

    def test_stage_timer_exception_re_raised(self, tmp_path: Path):
        p = tmp_path / "log.jsonl"
        lg = StructuredLogger(str(p), ring_size=10)
        with pytest.raises(ValueError):
            with lg.stage_timer("ocr", page_id="p2"):
                raise ValueError("boom")
        lg.close()
        events = [json.loads(line) for line in p.read_text(encoding="utf-8").strip().splitlines()]
        assert events[0]["event"] == "stage_start"
        assert events[1]["event"] == "stage_error"
        assert events[1]["error_type"] == "ValueError"
        assert events[1]["error_msg"] == "boom"
        assert events[1]["ok"] is False

    def test_thread_safety(self, tmp_path: Path):
        p = tmp_path / "log.jsonl"
        lg = StructuredLogger(str(p), ring_size=1000)
        N = 50
        THREADS = 8

        def worker(tid: int) -> None:
            for i in range(N):
                lg.event("e", tid=tid, i=i)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        lg.close()
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        # Mỗi thread N events.
        assert len(lines) == N * THREADS
        # Mỗi line là JSON valid (no torn write).
        for ln in lines:
            json.loads(ln)

    def test_rotation_on_size(self, tmp_path: Path):
        p = tmp_path / "log.jsonl"
        lg = StructuredLogger(str(p), ring_size=5, max_bytes=512)
        # Ghi đủ để vượt 512 bytes.
        for i in range(50):
            lg.event("e", payload="x" * 40, i=i)
        lg.close()
        # Phải có backup .1
        assert (tmp_path / "log.jsonl.1").exists()
        assert p.exists()

    def test_close_idempotent(self, tmp_path: Path):
        p = tmp_path / "log.jsonl"
        lg = StructuredLogger(str(p), ring_size=5)
        lg.close()
        lg.close()  # not raise
        lg.event("after_close")  # no-op
        # File phải tồn tại (empty hoặc trống), không crash.
        assert p.exists()


class TestJsonable:
    def test_primitives(self):
        assert _jsonable(1) == 1
        assert _jsonable("x") == "x"
        assert _jsonable(None) is None
        assert _jsonable(True) is True

    def test_collections(self):
        assert _jsonable([1, 2, 3]) == [1, 2, 3]
        assert _jsonable((1, 2)) == [1, 2]
        assert _jsonable({"a": 1}) == {"a": 1}
        assert _jsonable({1, 2, 3}) == [1, 2, 3]

    def test_unknown_fallback(self):
        class Foo:
            def __repr__(self) -> str:
                return "Foo()"
        v = _jsonable(Foo())
        assert isinstance(v, str)
        assert "Foo" in v


# --------------------------- MetricsRegistry --------------------------- #

class TestMetricsRegistry:
    def test_incr(self):
        m = MetricsRegistry()
        m.incr("pages_done")
        m.incr("pages_done")
        m.incr("pages_done", by=3)
        snap = m.snapshot()
        assert snap["counters"]["pages_done"] == 5

    def test_gauge_overwrites(self):
        m = MetricsRegistry()
        m.gauge("vram_mb", 1000.0)
        m.gauge("vram_mb", 500.0)
        snap = m.snapshot()
        assert snap["gauges"]["vram_mb"] == 500.0

    def test_uptime(self):
        m = MetricsRegistry()
        snap = m.snapshot()
        assert snap["uptime_s"] >= 0

    def test_thread_safety(self):
        m = MetricsRegistry()
        N = 1000
        T = 8

        def worker():
            for _ in range(N):
                m.incr("c")

        threads = [threading.Thread(target=worker) for _ in range(T)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert m.snapshot()["counters"]["c"] == N * T


class TestHealthPrinter:
    def test_start_stop(self):
        m = MetricsRegistry()
        sink: list = []
        hp = HealthPrinter(m, interval_s=0.05, printer=sink.append)
        hp.start()
        m.incr("x", 7)
        time.sleep(0.2)
        hp.stop(timeout=1.0)
        # Đã có ít nhất 1 print và message chứa counter x.
        joined = "\n".join(sink)
        assert "uptime" in joined
        assert "x=7" in joined

    def test_stop_idempotent(self):
        m = MetricsRegistry()
        hp = HealthPrinter(m, interval_s=0.05, printer=lambda _: None)
        hp.stop()
        hp.stop()  # no error
        hp.start()
        hp.stop()

    def test_printer_exception_does_not_crash_thread(self):
        m = MetricsRegistry()
        calls = {"n": 0}

        def bad(_: str) -> None:
            calls["n"] += 1
            raise RuntimeError("sink down")

        hp = HealthPrinter(m, interval_s=0.05, printer=bad)
        hp.start()
        time.sleep(0.2)
        hp.stop(timeout=1.0)
        # Bị raise nhiều lần nhưng thread không die → ít nhất 2 lần gọi.
        assert calls["n"] >= 1
