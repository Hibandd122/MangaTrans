"""Pytest config: disable third-party plugin autoload + register markers.

Một số plugin (seleniumbase) trong env user crash khi load do version mismatch.
PYTEST_DISABLE_PLUGIN_AUTOLOAD bắt buộc set TRƯỚC khi pytest collect plugins,
nên cần đặt qua sitecustomize hoặc shell. Đây chỉ là safety net cho future imports.
"""
import os
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")


def pytest_configure(config):
    # Custom markers — không gây warning khi test dùng @pytest.mark.slow.
    config.addinivalue_line(
        "markers",
        "slow: stress / long-running tests (chạy với `pytest -m slow`)",
    )
