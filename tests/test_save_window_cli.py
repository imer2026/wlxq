"""save-window 模板包选择测试，不依赖真实游戏窗口。"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from wlxq_bot.cli import app
from wlxq_bot.config import load_local_config
from wlxq_bot.perception.screen import WindowInfo

runner = CliRunner()


def _window_info() -> WindowInfo:
    return WindowInfo(
        handle=123,
        title="永远的蔚蓝星球",
        class_name="Chrome_WidgetWin_0",
        is_visible=True,
        is_minimized=False,
        is_foreground=True,
        window_rect=(100, 100, 1047, 1847),
        client_rect=(110, 110, 927, 1727),
        client_size=(927, 1727),
        dpi=144,
        monitor_id=r"\\.\DISPLAY2",
        monitor_resolution=(2560, 1440),
        process_id=2,
        thread_id=3,
    )


def _patch_window(monkeypatch, config_path: Path) -> None:
    monkeypatch.setattr("wlxq_bot.cli.LOCAL_CONFIG_PATH", config_path)
    monkeypatch.setattr("wlxq_bot.cli.enable_dpi_awareness", lambda: None)
    monkeypatch.setattr("wlxq_bot.cli.find_window_by_title", lambda _title: 123)
    monkeypatch.setattr("wlxq_bot.cli.get_window_info", lambda _handle: _window_info())


def test_save_window_defaults_to_window_monitor_resolution(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "local.yaml"
    _patch_window(monkeypatch, config_path)
    monkeypatch.setattr(
        "wlxq_bot.cli.get_window_monitor_resolution",
        lambda _handle: (2560, 1440),
    )

    result = runner.invoke(app, ["save-window"])

    assert result.exit_code == 0, result.output
    config = load_local_config(config_path)
    assert config is not None
    assert config.window.template_pack == "2560x1440"


def test_save_window_explicit_pack_overrides_monitor_resolution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "local.yaml"
    _patch_window(monkeypatch, config_path)
    monkeypatch.setattr(
        "wlxq_bot.cli.get_window_monitor_resolution",
        lambda _handle: (_ for _ in ()).throw(AssertionError("不应读取显示器分辨率")),
    )

    result = runner.invoke(app, ["save-window", "--template-pack", "calibrated"])

    assert result.exit_code == 0, result.output
    config = load_local_config(config_path)
    assert config is not None
    assert config.window.template_pack == "calibrated"
