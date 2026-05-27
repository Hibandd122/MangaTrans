"""CLI smoke tests cho Runtime wiring (Commit 7).

Kiểm tra:
- `build_runtime_config` map argparse flag → RuntimeConfig đúng.
- CLI `--batch` mặc định dispatch sang ChapterRunner (mock MangaPipeline + Runner).
- `--no-async` fallback về process_batch legacy.
- Single-image LUÔN sync.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from mangatrans.cli import build_parser, build_runtime_config, main
from mangatrans.runtime.config import RuntimeConfig
from mangatrans.runtime.page_task import PageState, PageTask


class TestBuildRuntimeConfig:
    def test_defaults(self):
        p = build_parser()
        args = p.parse_args(["-i", "x.png"])
        rt = build_runtime_config(args)
        assert isinstance(rt, RuntimeConfig)
        assert rt.enable_async is True
        assert rt.pipeline_depth == 3
        assert rt.cpu_pool_workers == 4
        assert rt.translation_rpm == 20
        assert rt.translation_concurrency == 4
        assert rt.enable_resume is True
        assert rt.force_resume_on_config_change is False
        assert rt.enable_translation_cache is True
        assert rt.watchdog_enable is True
        assert rt.vram_oom_cpu_fallback is True

    def test_no_async(self):
        p = build_parser()
        args = p.parse_args(["-i", "x.png", "--no-async"])
        rt = build_runtime_config(args)
        assert rt.enable_async is False

    def test_overrides_all_flags(self):
        p = build_parser()
        args = p.parse_args([
            "-i", "x.png",
            "--pipeline-depth", "5",
            "--cpu-pool-workers", "8",
            "--translation-rpm", "60",
            "--translation-concurrency", "8",
            "--no-resume",
            "--force-resume",
            "--no-translation-cache",
            "--checkpoint-path", "/tmp/ckpt.json",
            "--crash-dir", "/tmp/crash",
            "--watchdog-disable",
            "--no-cpu-fallback",
        ])
        rt = build_runtime_config(args)
        assert rt.pipeline_depth == 5
        assert rt.cpu_pool_workers == 8
        assert rt.translation_rpm == 60
        assert rt.translation_concurrency == 8
        assert rt.enable_resume is False
        assert rt.force_resume_on_config_change is True
        assert rt.enable_translation_cache is False
        assert rt.checkpoint_path == "/tmp/ckpt.json"
        assert rt.crash_report_dir == "/tmp/crash"
        assert rt.watchdog_enable is False
        assert rt.vram_oom_cpu_fallback is False


def _make_done_tasks(pairs):
    tasks: List[PageTask] = []
    for inp, out in pairs:
        t = PageTask.new(inp, out)
        t.state = PageState.DONE
        tasks.append(t)
    return tasks


class TestCliBatchDispatch:
    def test_batch_async_uses_chapter_runner(self, tmp_path: Path):
        """--batch (default async) → ChapterRunner.run được gọi, KHÔNG dùng
        MangaPipeline.process_batch."""
        # Tạo input dir với 2 file PNG giả (chỉ cần tồn tại).
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        (in_dir / "p1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (in_dir / "p2.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        out_dir = tmp_path / "out"

        called = {"chapter_run": 0, "process_batch": 0}

        class FakeRunner:
            def __init__(self, cfg, rt_cfg, base_dir="."):
                pass

            def run(self, pairs, resume=True):
                called["chapter_run"] += 1
                return _make_done_tasks(pairs)

            def close(self):
                pass

        class FakePipeline:
            def __init__(self, cfg, base_dir="."):
                pass

            def process_batch(self, inputs, out_dir):
                called["process_batch"] += 1
                return [{"output": p} for p in inputs]

            def process_image(self, *args, **kwargs):
                pass

            def release(self):
                pass

        with patch("mangatrans.cli.MangaPipeline", FakePipeline), \
             patch("mangatrans.runtime.chapter_runner.ChapterRunner", FakeRunner):
            rc = main([
                "--batch", "-i", str(in_dir), "-o", str(out_dir),
                "--no-translate",
            ])

        assert rc == 0
        assert called["chapter_run"] == 1
        assert called["process_batch"] == 0

    def test_batch_no_async_falls_back_legacy(self, tmp_path: Path):
        """--batch --no-async → process_batch legacy, KHÔNG dùng ChapterRunner."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        (in_dir / "p1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        out_dir = tmp_path / "out"

        called = {"chapter_run": 0, "process_batch": 0}

        class FakeRunner:
            def __init__(self, *a, **kw):
                pass

            def run(self, *a, **kw):
                called["chapter_run"] += 1
                return []

            def close(self):
                pass

        class FakePipeline:
            def __init__(self, cfg, base_dir="."):
                pass

            def process_batch(self, inputs, out_dir):
                called["process_batch"] += 1
                return [{"output": p} for p in inputs]

            def release(self):
                pass

        with patch("mangatrans.cli.MangaPipeline", FakePipeline), \
             patch("mangatrans.runtime.chapter_runner.ChapterRunner", FakeRunner):
            rc = main([
                "--batch", "--no-async",
                "-i", str(in_dir), "-o", str(out_dir),
                "--no-translate",
            ])

        assert rc == 0
        assert called["chapter_run"] == 0
        assert called["process_batch"] == 1

    def test_single_image_stays_sync(self, tmp_path: Path):
        """Single-image (no --batch) LUÔN sync, không gọi ChapterRunner."""
        inp = tmp_path / "p.png"
        inp.write_bytes(b"\x89PNG\r\n\x1a\n")
        out = tmp_path / "out.png"

        called = {"chapter_run": 0, "process_image": 0}

        class FakeRunner:
            def __init__(self, *a, **kw):
                pass

            def run(self, *a, **kw):
                called["chapter_run"] += 1
                return []

            def close(self):
                pass

        class FakePipeline:
            def __init__(self, cfg, base_dir="."):
                pass

            def process_image(self, inp, out):
                called["process_image"] += 1

            def release(self):
                pass

        with patch("mangatrans.cli.MangaPipeline", FakePipeline), \
             patch("mangatrans.runtime.chapter_runner.ChapterRunner", FakeRunner):
            rc = main(["-i", str(inp), "-o", str(out), "--no-translate"])

        assert rc == 0
        assert called["chapter_run"] == 0
        assert called["process_image"] == 1

    def test_batch_fail_exit_code(self, tmp_path: Path):
        """Nếu có page FAILED → exit code 1."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        (in_dir / "p1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (in_dir / "p2.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        out_dir = tmp_path / "out"

        class FakeRunner:
            def __init__(self, *a, **kw):
                pass

            def run(self, pairs, resume=True):
                # 1 DONE, 1 FAILED.
                tasks = []
                for i, (inp, out) in enumerate(pairs):
                    t = PageTask.new(inp, out)
                    t.state = PageState.DONE if i == 0 else PageState.FAILED
                    tasks.append(t)
                return tasks

            def close(self):
                pass

        class FakePipeline:
            def __init__(self, cfg, base_dir="."):
                pass

            def release(self):
                pass

        with patch("mangatrans.cli.MangaPipeline", FakePipeline), \
             patch("mangatrans.runtime.chapter_runner.ChapterRunner", FakeRunner):
            rc = main([
                "--batch", "-i", str(in_dir), "-o", str(out_dir),
                "--no-translate",
            ])

        assert rc == 1
