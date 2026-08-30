"""exec 能力命令测试，不启动真实窗口或输入。"""

from __future__ import annotations

from typer.testing import CliRunner

from wlxq_bot.cli import app
from wlxq_bot.config import DefaultConfig, TasksConfig
from wlxq_bot.models import State


def test_exec_select_difficulty_passes_override(monkeypatch) -> None:
    captured: dict[str, str | None] = {}

    monkeypatch.setattr(
        "wlxq_bot.config.load_default_config",
        lambda _path: DefaultConfig(),
    )
    monkeypatch.setattr(
        "wlxq_bot.config.load_tasks_config",
        lambda _path: TasksConfig(),
    )
    monkeypatch.setattr("wlxq_bot.cli.load_local_config", lambda _path: None)
    monkeypatch.setattr("wlxq_bot.cli.enable_dpi_awareness", lambda: None)

    def fake_select_difficulty(self, coop_difficulties: str | None = None) -> State:
        captured["coop_difficulties"] = coop_difficulties
        return State.COMPLETED

    monkeypatch.setattr(
        "wlxq_bot.runner.Runner.select_difficulty",
        fake_select_difficulty,
    )

    result = CliRunner().invoke(
        app,
        ["exec", "select-difficulty", "--coop-difficulties", "1-10"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {"coop_difficulties": "1-10"}
    assert "目标合作难度已全部点击" in result.output


def test_exec_select_difficulty_returns_failure_when_not_completed(monkeypatch) -> None:
    monkeypatch.setattr(
        "wlxq_bot.config.load_default_config",
        lambda _path: DefaultConfig(),
    )
    monkeypatch.setattr(
        "wlxq_bot.config.load_tasks_config",
        lambda _path: TasksConfig(),
    )
    monkeypatch.setattr("wlxq_bot.cli.load_local_config", lambda _path: None)
    monkeypatch.setattr("wlxq_bot.cli.enable_dpi_awareness", lambda: None)
    monkeypatch.setattr(
        "wlxq_bot.runner.Runner.select_difficulty",
        lambda self, coop_difficulties=None: State.FIND_COOP,
    )

    result = CliRunner().invoke(app, ["exec", "select-difficulty"])

    assert result.exit_code == 1
    assert "难度选择未完成" in result.output
