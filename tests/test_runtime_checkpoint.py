"""Unit tests cho checkpoint + translation_cache (Commit 2).

Mở rộng `tests/test_runtime_unit.py` — single file để pytest collect 1 module.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from mangatrans.runtime.checkpoint import (
    SCHEMA_VERSION,
    CheckpointStore,
    atomic_write_json,
    compute_config_digest,
)
from mangatrans.runtime.page_task import PageState, PageTask, StageName
from mangatrans.runtime.translation_cache import (
    NullTranslationCache,
    TranslationCache,
    make_cache_key,
)


# --------------------------- atomic_write_json --------------------------- #

class TestAtomicWriteJson:
    def test_creates_file(self, tmp_path: Path):
        p = tmp_path / "x.json"
        atomic_write_json(str(p), {"a": 1})
        assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}

    def test_overwrites(self, tmp_path: Path):
        p = tmp_path / "x.json"
        atomic_write_json(str(p), {"a": 1})
        atomic_write_json(str(p), {"b": 2})
        assert json.loads(p.read_text(encoding="utf-8")) == {"b": 2}

    def test_no_tmp_leftovers(self, tmp_path: Path):
        p = tmp_path / "x.json"
        atomic_write_json(str(p), {"a": 1})
        leftovers = [f.name for f in tmp_path.iterdir() if f.name.startswith(".ckpt_")]
        assert leftovers == []

    def test_creates_dirs(self, tmp_path: Path):
        p = tmp_path / "deep" / "dir" / "x.json"
        atomic_write_json(str(p), {"a": 1})
        assert p.exists()


# --------------------------- compute_config_digest --------------------------- #

class TestConfigDigest:
    def test_stable_dict_order(self):
        d1 = compute_config_digest({"a": 1, "b": 2})
        d2 = compute_config_digest({"b": 2, "a": 1})
        assert d1 == d2

    def test_changes_on_value_change(self):
        a = compute_config_digest({"model": "x"})
        b = compute_config_digest({"model": "y"})
        assert a != b

    def test_nested(self):
        a = compute_config_digest({"x": {"y": [1, 2, 3]}})
        b = compute_config_digest({"x": {"y": [1, 2, 3]}})
        c = compute_config_digest({"x": {"y": [1, 2, 4]}})
        assert a == b
        assert a != c

    def test_dataclass(self):
        from dataclasses import dataclass

        @dataclass
        class Foo:
            a: int = 1
            b: str = "z"

        d1 = compute_config_digest(Foo())
        d2 = compute_config_digest(Foo())
        d3 = compute_config_digest(Foo(b="x"))
        assert d1 == d2
        assert d1 != d3


# --------------------------- CheckpointStore --------------------------- #

class TestCheckpointStore:
    def _path(self, tmp_path: Path) -> str:
        return str(tmp_path / ".mangatrans_state.json")

    def test_load_missing_returns_empty(self, tmp_path: Path):
        st = CheckpointStore(self._path(tmp_path))
        assert st.load() == {}
        assert st.all_tasks() == []

    def test_upsert_then_flush(self, tmp_path: Path):
        p = self._path(tmp_path)
        st = CheckpointStore(p, config_digest="abc", debounce_s=0.0)
        t = PageTask.new("page1.jpg", "out1.png")
        st.upsert(t)
        assert st.flush() is True
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        assert data["version"] == SCHEMA_VERSION
        assert data["config_digest"] == "abc"
        assert t.page_id in data["pages"]

    def test_round_trip(self, tmp_path: Path):
        p = self._path(tmp_path)
        st1 = CheckpointStore(p, config_digest="d1", debounce_s=0.0)
        t = PageTask.new("a.jpg", "a.png")
        t.state = PageState.DONE
        t.completed_stages = [StageName.LOAD, StageName.DETECT, StageName.SAVE_PNG]
        t.retries = 2
        t.fallback_flags = {"cpu_inpaint"}
        st1.upsert(t)
        st1.flush()

        st2 = CheckpointStore(p, config_digest="d1")
        loaded = st2.load()
        assert t.page_id in loaded
        rt = loaded[t.page_id]
        assert rt.state == PageState.DONE
        assert rt.completed_stages == [StageName.LOAD, StageName.DETECT, StageName.SAVE_PNG]
        assert rt.retries == 2
        assert rt.fallback_flags == {"cpu_inpaint"}

    def test_mark_state_done_sets_ts(self, tmp_path: Path):
        st = CheckpointStore(self._path(tmp_path), debounce_s=0.0)
        t = PageTask.new("a.jpg", "a.png")
        st.upsert(t)
        st.mark_state(t.page_id, PageState.DONE)
        got = st.get(t.page_id)
        assert got.state == PageState.DONE
        assert got.completed_ts is not None

    def test_mark_stage_complete(self, tmp_path: Path):
        st = CheckpointStore(self._path(tmp_path), debounce_s=0.0)
        t = PageTask.new("a.jpg", "a.png")
        st.upsert(t)
        st.mark_stage_complete(t.page_id, StageName.DETECT)
        got = st.get(t.page_id)
        assert StageName.DETECT in got.completed_stages

    def test_debounce_skips_flush(self, tmp_path: Path):
        p = self._path(tmp_path)
        st = CheckpointStore(p, debounce_s=10.0)
        t = PageTask.new("a.jpg", "a.png")
        st.upsert(t)
        assert st.flush() is True       # first flush goes through
        st.upsert(t)
        # Within debounce window → skipped.
        assert st.flush_if_due() is False

    def test_corrupted_file_ignored(self, tmp_path: Path):
        p = self._path(tmp_path)
        Path(p).write_text("{ broken json", encoding="utf-8")
        st = CheckpointStore(p)
        assert st.load() == {}

    def test_schema_version_mismatch_ignored(self, tmp_path: Path):
        p = self._path(tmp_path)
        Path(p).write_text(json.dumps({"version": 99, "pages": {}}), encoding="utf-8")
        st = CheckpointStore(p)
        assert st.load() == {}

    def test_config_changed_detection(self, tmp_path: Path):
        p = self._path(tmp_path)
        st1 = CheckpointStore(p, config_digest="A", debounce_s=0.0)
        st1.upsert(PageTask.new("a.jpg", "a.png"))
        st1.flush()
        # Reload với digest khác.
        st2 = CheckpointStore(p, config_digest="B")
        st2.load()
        assert st2.config_changed() is True
        # Cùng digest → False.
        st3 = CheckpointStore(p, config_digest="A")
        st3.load()
        assert st3.config_changed() is False

    def test_should_skip_done_with_file(self, tmp_path: Path):
        p = self._path(tmp_path)
        out_file = tmp_path / "out.png"
        out_file.write_bytes(b"x")
        st = CheckpointStore(p, debounce_s=0.0)
        t = PageTask.new("a.jpg", str(out_file))
        t.state = PageState.DONE
        st.upsert(t)
        # Mock prev = same task DONE.
        assert st.should_skip(t) is True

    def test_should_skip_done_without_file(self, tmp_path: Path):
        st = CheckpointStore(self._path(tmp_path), debounce_s=0.0)
        t = PageTask.new("a.jpg", str(tmp_path / "missing.png"))
        t.state = PageState.DONE
        st.upsert(t)
        assert st.should_skip(t) is False

    def test_should_skip_failed(self, tmp_path: Path):
        st = CheckpointStore(self._path(tmp_path), debounce_s=0.0)
        out = tmp_path / "out.png"
        out.write_bytes(b"x")
        t = PageTask.new("a.jpg", str(out))
        t.state = PageState.FAILED
        st.upsert(t)
        assert st.should_skip(t) is False

    def test_restore_into_resets_partial(self, tmp_path: Path):
        p = self._path(tmp_path)
        st1 = CheckpointStore(p, debounce_s=0.0)
        t = PageTask.new("a.jpg", "a.png")
        t.state = PageState.RUNNING
        t.completed_stages = [StageName.LOAD]
        t.current_stage = StageName.OCR
        t.last_error = "interrupted"
        st1.upsert(t)
        st1.flush()

        st2 = CheckpointStore(p, debounce_s=0.0)
        st2.load()
        new_t = PageTask.new("a.jpg", "a.png")
        merged = st2.restore_into([new_t])[0]
        assert merged.state == PageState.PENDING
        assert merged.current_stage is None
        assert merged.completed_stages == []
        assert merged.last_error == "interrupted"  # carried for diagnostics

    def test_restore_into_keeps_done(self, tmp_path: Path):
        p = self._path(tmp_path)
        st1 = CheckpointStore(p, debounce_s=0.0)
        t = PageTask.new("a.jpg", "a.png")
        t.state = PageState.DONE
        t.completed_stages = list(StageName.ordered())
        t.completed_ts = "2026-01-01T00:00:00Z"
        st1.upsert(t)
        st1.flush()

        st2 = CheckpointStore(p, debounce_s=0.0)
        st2.load()
        new_t = PageTask.new("a.jpg", "a.png")
        merged = st2.restore_into([new_t])[0]
        assert merged.state == PageState.DONE
        assert merged.completed_ts == "2026-01-01T00:00:00Z"

    def test_thread_safety(self, tmp_path: Path):
        st = CheckpointStore(self._path(tmp_path), debounce_s=0.0)
        N = 50
        T = 6

        def worker(tid: int):
            for i in range(N):
                t = PageTask.new(f"p{tid}_{i}.jpg", f"p{tid}_{i}.png")
                st.upsert(t)
                st.mark_state(t.page_id, PageState.DONE)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(T)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(st.all_tasks()) == N * T

    def test_atomic_no_partial_on_crash(self, tmp_path: Path, monkeypatch):
        # Simulate failure during atomic_write_json sau khi tmp viết xong.
        p = self._path(tmp_path)
        # Pre-existing valid file.
        Path(p).write_text(json.dumps({"version": SCHEMA_VERSION, "pages": {}}), encoding="utf-8")
        original = json.dumps({"version": SCHEMA_VERSION, "pages": {}})

        st = CheckpointStore(p, debounce_s=0.0)
        # Force os.replace to fail.
        from mangatrans.runtime import checkpoint as ckpt_mod
        real_replace = os.replace

        def fail_replace(*a, **kw):
            raise OSError("simulated")
        monkeypatch.setattr(ckpt_mod.os, "replace", fail_replace)

        t = PageTask.new("a.jpg", "a.png")
        st.upsert(t)
        ok = st.flush()
        assert ok is False
        # File cũ vẫn nguyên (không bị truncate).
        monkeypatch.setattr(ckpt_mod.os, "replace", real_replace)
        assert Path(p).read_text(encoding="utf-8") == original


# --------------------------- TranslationCache --------------------------- #

class TestMakeCacheKey:
    def test_stable(self):
        a = make_cache_key("m1", "vi", "hello")
        b = make_cache_key("m1", "vi", "hello")
        assert a == b
        assert len(a) == 64

    def test_distinguishes_components(self):
        base = make_cache_key("m1", "vi", "hello")
        assert base != make_cache_key("m2", "vi", "hello")
        assert base != make_cache_key("m1", "en", "hello")
        assert base != make_cache_key("m1", "vi", "world")

    def test_no_separator_collision(self):
        # "A\nB" lang + "" text  vs  "A" lang + "\nB" text → khác key nhờ NUL sep.
        a = make_cache_key("m", "A\nB", "")
        b = make_cache_key("m", "A", "\nB")
        assert a != b


class TestTranslationCache:
    def test_get_miss_returns_none(self, tmp_path: Path):
        c = TranslationCache(str(tmp_path / "cache.json"))
        assert c.get("m", "vi", "hello") is None
        assert c.stats()["misses"] == 1

    def test_put_then_get(self, tmp_path: Path):
        c = TranslationCache(str(tmp_path / "cache.json"))
        c.put("m", "vi", "hello", "xin chào")
        assert c.get("m", "vi", "hello") == "xin chào"
        assert c.stats()["hits"] == 1

    def test_empty_text_skipped(self, tmp_path: Path):
        c = TranslationCache(str(tmp_path / "cache.json"))
        c.put("m", "vi", "", "skip")
        assert c.get("m", "vi", "") is None
        assert len(c) == 0

    def test_persistence_round_trip(self, tmp_path: Path):
        p = str(tmp_path / "cache.json")
        c1 = TranslationCache(p, debounce_s=0.0)
        c1.put("m", "vi", "hello", "xin chào")
        c1.put("m", "vi", "bye", "tạm biệt")
        c1.flush()

        c2 = TranslationCache(p)
        assert c2.get("m", "vi", "hello") == "xin chào"
        assert c2.get("m", "vi", "bye") == "tạm biệt"

    def test_corrupted_load_ignored(self, tmp_path: Path):
        p = str(tmp_path / "cache.json")
        Path(p).write_text("{ broken", encoding="utf-8")
        c = TranslationCache(p)
        assert len(c) == 0
        assert c.get("m", "vi", "hello") is None

    def test_eviction(self, tmp_path: Path):
        c = TranslationCache(str(tmp_path / "cache.json"), max_entries=20, debounce_s=0.0)
        for i in range(30):
            c.put("m", "vi", f"text{i}", f"trans{i}")
        # Sau eviction còn ~95% × 20 = 19 entries.
        assert len(c) <= 20
        # Entries cũ nhất bị drop trước.

    def test_thread_safety(self, tmp_path: Path):
        c = TranslationCache(str(tmp_path / "cache.json"), debounce_s=0.0)
        N = 100
        T = 6

        def worker(tid: int):
            for i in range(N):
                c.put("m", "vi", f"{tid}_{i}", f"trans_{tid}_{i}")
                got = c.get("m", "vi", f"{tid}_{i}")
                assert got == f"trans_{tid}_{i}"

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(T)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(c) == N * T

    def test_flush_idempotent_when_clean(self, tmp_path: Path):
        c = TranslationCache(str(tmp_path / "cache.json"), debounce_s=0.0)
        c.put("m", "vi", "x", "y")
        assert c.flush() is True
        assert c.flush() is False  # nothing dirty


class TestNullTranslationCache:
    def test_always_miss(self):
        c = NullTranslationCache()
        c.put("m", "vi", "hello", "xin chào")
        assert c.get("m", "vi", "hello") is None
        # Stats: 1 miss.
        assert c.stats()["misses"] == 1
        # put không count.
        assert c.stats()["puts"] == 0
