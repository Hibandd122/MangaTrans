"""Benchmark + integration smoke test.

Chạy pipeline trên N pages thật, đo wall-clock từng stage, không cần Gemini key.
Mặc định: --translate OFF (chỉ detect+OCR+inpaint).

Usage:
    python tests/benchmark.py                          # default chap/001.jpg
    python tests/benchmark.py --inputs chap/*.jpg
    python tests/benchmark.py --inputs chap/001.jpg chap/002.jpg --translate
    python tests/benchmark.py --quick                  # skip OCR + inpaint heavy
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import List

# Make package importable khi gọi từ root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mangatrans.cli import build_config, build_parser  # noqa: E402
from mangatrans.pipeline import MangaPipeline  # noqa: E402


def expand_inputs(patterns: List[str]) -> List[str]:
    out: list[str] = []
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            out.extend(sorted(matches))
        elif os.path.isfile(pat):
            out.append(pat)
    return out


def fmt_secs(s: float) -> str:
    return f"{s:6.2f}s"


def main() -> int:
    # UTF-8 stdout cho Windows console (emoji trong log)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    bench = argparse.ArgumentParser()
    bench.add_argument("--inputs", nargs="+",
                       default=["chap/001.jpg"],
                       help="Glob / paths của test images")
    bench.add_argument("--output-dir", default="benchmark_out",
                       help="Where to save outputs")
    bench.add_argument("--translate", action="store_true",
                       help="Bật Gemini translation (cần API key)")
    bench.add_argument("--quick", action="store_true",
                       help="Skip translation, render — chỉ detect+inpaint")
    bench.add_argument("--no-hd", action="store_true",
                       help="Tắt HD-tiled inpaint (nhanh hơn nhưng kém quality)")
    args = bench.parse_args()

    inputs = expand_inputs(args.inputs)
    if not inputs:
        print(f"❌ Không tìm thấy file nào khớp: {args.inputs}")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    # Mock argparse namespace cho build_config
    parser = build_parser()
    cli_args = ["-i", inputs[0], "-o", "dummy.png"]
    if args.translate:
        cli_args.append("--translate")
    if args.no_hd:
        cli_args.append("--no-hd")
    if args.quick:
        cli_args.append("--no-classify")
    ns = parser.parse_args(cli_args)
    config = build_config(ns)

    print(f"\n📦 Pipeline init...")
    t0 = time.perf_counter()
    pipeline = MangaPipeline(config, base_dir=".")
    print(f"   Init: {fmt_secs(time.perf_counter() - t0)}")

    print(f"\n🏁 Benchmark {len(inputs)} page(s):")
    timings: list[tuple[str, float, dict]] = []
    t_total_start = time.perf_counter()
    try:
        for idx, inp in enumerate(inputs, 1):
            base = os.path.splitext(os.path.basename(inp))[0]
            out_path = os.path.join(args.output_dir, f"{base}.png")
            print(f"\n   [{idx}/{len(inputs)}] {inp}")
            t0 = time.perf_counter()
            try:
                summary = pipeline.process_image(inp, out_path)
                dt = time.perf_counter() - t0
                timings.append((base, dt, summary))
                print(f"      ✅ {fmt_secs(dt)}  "
                      f"bubbles={summary['n_bubbles']:3d}  "
                      f"ocr={summary['n_ocr']:3d}  "
                      f"translated={summary['n_translated']:3d}")
            except Exception as e:  # noqa: BLE001
                dt = time.perf_counter() - t0
                timings.append((base, dt, {"error": str(e)}))
                print(f"      ❌ {fmt_secs(dt)}  ERROR: {e}")
    finally:
        pipeline.release()

    total = time.perf_counter() - t_total_start
    n_ok = sum(1 for _, _, s in timings if "error" not in s)

    print("\n" + "=" * 64)
    print(f"📊 BENCHMARK SUMMARY")
    print("=" * 64)
    print(f"   Total pages:       {len(inputs)}")
    print(f"   Successful:        {n_ok}")
    print(f"   Failed:            {len(inputs) - n_ok}")
    print(f"   Total wall-clock:  {fmt_secs(total)}")
    if n_ok > 0:
        avg = sum(t for _, t, s in timings if "error" not in s) / n_ok
        print(f"   Avg per page:      {fmt_secs(avg)}")
    print(f"   Output dir:        {args.output_dir}/")
    print("=" * 64)

    return 0 if n_ok == len(inputs) else 1


if __name__ == "__main__":
    sys.exit(main())
