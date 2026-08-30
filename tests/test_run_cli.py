"""run 命令合作难度覆盖测试。"""

from __future__ import annotations

from typer.testing import CliRunner

from wlxq_bot.cli import app
from wlxq_bot.config import DefaultConfig, TasksConfig
from wlxq_bot.models import State


def _patch_run_env(monkeypatch, captured: dict) -> None:
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

    def fake_run(
        self,
        task_name: str,
        main_c: str | None = None,
        start_state: str = "find_coop",
        coop_difficulties: str | None = None,
        skip_difficulty_selection: bool | None = None,
        max_rounds: int | None = None,
    ) -> State:
        captured["task_name"] = task_name
        captured["main_c"] = main_c
        captured["start_state"] = start_state
        captured["coop_difficulties"] = coop_difficulties
        captured["skip_difficulty_selection"] = skip_difficulty_selection
        captured["max_rounds"] = max_rounds
        return State.COMPLETED

    monkeypatch.setattr("wlxq_bot.runner.Runner.run", fake_run)


def test_run_cli_passes_coop_difficulties_override(monkeypatch) -> None:
    captured: dict[str, object] = {}
    _patch_run_env(monkeypatch, captured)

    result = CliRunner().invoke(
        app,
        ["run", "coop", "--main-c", "assault", "--coop-difficulties", "1-10"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "task_name": "coop",
        "main_c": "assault",
        "start_state": "find_coop",
        "coop_difficulties": "1-10",
        "skip_difficulty_selection": None,
        "max_rounds": None,
    }


def test_run_cli_passes_skip_difficulty_selection(monkeypatch) -> None:
    captured: dict[str, object] = {}
    _patch_run_env(monkeypatch, captured)

    result = CliRunner().invoke(
        app,
        ["run", "coop", "--main-c", "assault", "--skip-difficulty-selection"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "task_name": "coop",
        "main_c": "assault",
        "start_state": "find_coop",
        "coop_difficulties": None,
        "skip_difficulty_selection": True,
        "max_rounds": None,
    }


def test_run_cli_passes_max_rounds_override(monkeypatch) -> None:
    captured: dict[str, object] = {}
    _patch_run_env(monkeypatch, captured)

    result = CliRunner().invoke(
        app,
        ["run", "coop", "--main-c", "assault", "--max-rounds", "3"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "task_name": "coop",
        "main_c": "assault",
        "start_state": "find_coop",
        "coop_difficulties": None,
        "skip_difficulty_selection": None,
        "max_rounds": 3,
    }


def test_run_cli_rejects_non_positive_max_rounds(monkeypatch) -> None:
    captured: dict[str, object] = {}
    _patch_run_env(monkeypatch, captured)

    result = CliRunner().invoke(
        app,
        ["run", "coop", "--main-c", "assault", "--max-rounds", "0"],
    )

    # typer 的 min=1 参数校验直接拒绝，不会触达 Runner.run
    assert result.exit_code != 0
    assert "task_name" not in captured
