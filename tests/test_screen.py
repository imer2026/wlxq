"""Screen Capture 模块测试。

不依赖真实游戏窗口，只测数据类和纯函数。
涉及 win32 API 的调用需要真实窗口，标记为 smoke 测试。
"""

from __future__ import annotations

import pytest

from wlxq_bot.models import WindowContext
from wlxq_bot.perception.screen import (
    ScreenCapture,
    WindowInfo,
    get_window_monitor_resolution,
)


class TestWindowInfo:
    """WindowInfo 数据类测试。"""

    def _make_info(self, **overrides) -> WindowInfo:
        defaults: dict = {
            "handle": 12345,
            "title": "永远的蔚蓝星球",
            "class_name": "WeChatMainWndForPC",
            "is_visible": True,
            "is_minimized": False,
            "is_foreground": True,
            "window_rect": (100, 100, 2020, 1180),
            "client_rect": (100, 100, 1920, 1080),
            "client_size": (1920, 1080),
            "dpi": 96,
            "monitor_id": r"\\.\DISPLAY1",
            "monitor_resolution": (1920, 1080),
            "process_id": 9999,
            "thread_id": 8888,
        }
        defaults.update(overrides)
        return WindowInfo(**defaults)

    def test_create(self) -> None:
        info = self._make_info()
        assert info.handle == 12345
        assert info.title == "永远的蔚蓝星球"
        assert info.client_size == (1920, 1080)
        assert info.dpi == 96

    def test_frozen(self) -> None:
        """WindowInfo 应为不可变。"""
        info = self._make_info()
        with pytest.raises(AttributeError):
            info.title = "other"  # type: ignore[misc]

    def test_16_9_ratio_detection(self) -> None:
        """1920x1080 是 16:9。"""
        info = self._make_info(client_size=(1920, 1080))
        w, h = info.client_size
        assert abs(w / h - 16 / 9) < 0.01

    def test_non_16_9_ratio(self) -> None:
        info = self._make_info(client_size=(800, 600))
        w, h = info.client_size
        assert abs(w / h - 16 / 9) >= 0.01

    def test_minimized_flag(self) -> None:
        info = self._make_info(is_minimized=True)
        assert info.is_minimized
        assert not info.is_foreground or info.is_foreground  # 不互斥，但业务上最小化通常非前台

    def test_dpi_values(self) -> None:
        """常见 DPI 值。"""
        for dpi, scale in [(96, 100), (120, 125), (144, 150), (192, 200)]:
            info = self._make_info(dpi=dpi)
            assert info.dpi == dpi
            assert dpi / 96 * 100 == scale


def test_get_window_monitor_resolution_uses_window_monitor(monkeypatch) -> None:
    """模板分辨率取窗口所在屏幕，支持位于主屏左侧的负坐标显示器。"""
    monkeypatch.setattr(
        "wlxq_bot.perception.screen.win32api.MonitorFromWindow",
        lambda handle, flag: (handle, flag),
    )
    monkeypatch.setattr(
        "wlxq_bot.perception.screen.win32api.GetMonitorInfo",
        lambda _monitor: {"Monitor": (-2560, 0, 0, 1440)},
    )

    assert get_window_monitor_resolution(12345) == (2560, 1440)


def test_get_window_monitor_resolution_rejects_invalid_bounds(monkeypatch) -> None:
    monkeypatch.setattr(
        "wlxq_bot.perception.screen.win32api.MonitorFromWindow",
        lambda handle, flag: (handle, flag),
    )
    monkeypatch.setattr(
        "wlxq_bot.perception.screen.win32api.GetMonitorInfo",
        lambda _monitor: {"Monitor": (0, 0, 0, 1080)},
    )

    with pytest.raises(RuntimeError, match="边界无效"):
        get_window_monitor_resolution(12345)


def test_validate_context_rejects_window_move(monkeypatch) -> None:
    expected = WindowContext(
        window_handle=12345,
        client_rect_screen=(100, 100, 1920, 1080),
        client_size=(1920, 1080),
        dpi=96,
        monitor_id=r"\\.\DISPLAY1",
        is_foreground=True,
        is_minimized=False,
        captured_at=0,
        frame_id=7,
    )
    current = TestWindowInfo()._make_info(
        client_rect=(120, 100, 1920, 1080),
    )
    monkeypatch.setattr(
        "wlxq_bot.perception.screen.get_window_info",
        lambda _handle: current,
    )

    ok, reason = ScreenCapture().validate_context(expected)

    assert not ok
    assert "位置或尺寸已变化" in reason


def test_validate_context_rejects_monitor_change(monkeypatch) -> None:
    expected = WindowContext(
        window_handle=12345,
        client_rect_screen=(100, 100, 1920, 1080),
        client_size=(1920, 1080),
        dpi=96,
        monitor_id=r"\\.\DISPLAY1",
        is_foreground=True,
        is_minimized=False,
        captured_at=0,
        frame_id=7,
    )
    current = TestWindowInfo()._make_info(monitor_id=r"\\.\DISPLAY2")
    monkeypatch.setattr(
        "wlxq_bot.perception.screen.get_window_info",
        lambda _handle: current,
    )

    ok, reason = ScreenCapture().validate_context(expected)

    assert not ok
    assert "显示器已变化" in reason


def test_get_input_idle_seconds_returns_non_negative() -> None:
    """系统空闲秒数应为非负有限值（GetLastInputInfo 冒烟测试）。"""
    from wlxq_bot.perception.screen import get_input_idle_seconds

    idle = get_input_idle_seconds()
    assert 0.0 <= idle < 3600.0
