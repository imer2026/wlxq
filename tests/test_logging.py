"""日志初始化与 ``--debug`` 开关测试。

验证：
- ``setup_logging`` 正确设置级别、幂等（不重复添加 handler）。
- ``get_logger`` 对短名和完整模块名都归一到 ``wlxq_bot.*``，不重复前缀。
- CLI ``--debug`` 全局选项把根日志器切到 DEBUG，不带时为 INFO。
- 文件日志：默认 DEBUG 级别、独立于控制台、幂等、过期清理与失败兜底。
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

import pytest
from rich.logging import RichHandler
from typer.testing import CliRunner

import wlxq_bot.cli as cli_module
from wlxq_bot.cli import app
from wlxq_bot.utils.log import ROOT_LOGGER_NAME, get_logger, setup_logging

#: 文件日志命名规则：wlxq-bot_YYYYMMDD_HHMMSS.log
_FILE_LOG_PATTERN = re.compile(r"^wlxq-bot_\d{8}_\d{6}\.log$")

# 根日志器的级别与 handler 隔离由 tests/conftest.py 的 autouse 夹具统一处理。


def test_setup_logging_sets_level() -> None:
    setup_logging("DEBUG")
    assert logging.getLogger(ROOT_LOGGER_NAME).level == logging.DEBUG
    setup_logging("INFO")
    assert logging.getLogger(ROOT_LOGGER_NAME).level == logging.INFO


def test_setup_logging_idempotent_no_duplicate_handler() -> None:
    setup_logging("INFO")
    setup_logging("DEBUG")
    setup_logging("INFO")
    handlers = logging.getLogger(ROOT_LOGGER_NAME).handlers
    assert sum(1 for h in handlers if isinstance(h, RichHandler)) == 1


def test_rich_handler_shows_local_time_on_every_log_line() -> None:
    setup_logging("INFO")
    handler = next(
        item
        for item in logging.getLogger(ROOT_LOGGER_NAME).handlers
        if isinstance(item, RichHandler)
    )

    assert handler._log_render.show_time is True
    assert handler._log_render.omit_repeated_times is False
    assert handler._log_render.time_format == "%H:%M:%S"


def test_setup_logging_disables_propagate() -> None:
    setup_logging("INFO")
    assert logging.getLogger(ROOT_LOGGER_NAME).propagate is False


def test_get_logger_short_name() -> None:
    assert get_logger("screen").name == "wlxq_bot.screen"


def test_get_logger_full_name_no_duplicate_prefix() -> None:
    # 传 __name__ 这种完整名不应产生 wlxq_bot.wlxq_bot.xxx
    assert get_logger("wlxq_bot.cli").name == "wlxq_bot.cli"
    assert get_logger("wlxq_bot.perception.screen").name == "wlxq_bot.perception.screen"


def test_debug_flag_sets_debug_level() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--debug", "version"])
    assert result.exit_code == 0
    assert logging.getLogger(ROOT_LOGGER_NAME).level == logging.DEBUG


def test_no_debug_flag_sets_info_level() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert logging.getLogger(ROOT_LOGGER_NAME).level == logging.INFO


def test_debug_short_flag_works() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["-v", "version"])
    assert result.exit_code == 0
    assert logging.getLogger(ROOT_LOGGER_NAME).level == logging.DEBUG


# ---------------------------------------------------------------------------
# 文件日志
# ---------------------------------------------------------------------------


def _file_handlers(root: logging.Logger) -> list[logging.FileHandler]:
    return [h for h in root.handlers if isinstance(h, logging.FileHandler)]


def _file_log_path() -> Path:
    handlers = _file_handlers(logging.getLogger(ROOT_LOGGER_NAME))
    assert len(handlers) == 1, "应恰好存在一个文件 handler"
    return Path(handlers[0].baseFilename)


def test_setup_logging_creates_timestamped_file(tmp_path: Path) -> None:
    setup_logging("INFO", log_dir=tmp_path, argv=["wlxq-bot", "version"])
    assert _FILE_LOG_PATTERN.match(_file_log_path().name)


def test_file_log_records_debug_when_console_is_info(tmp_path: Path) -> None:
    """文件日志固定 DEBUG：控制台 INFO 时 DEBUG 记录仍要进文件。"""
    setup_logging("INFO", log_dir=tmp_path, argv=["wlxq-bot", "version"])
    logger = get_logger("test.file_log")
    logger.debug("文件专属调试 confidence=0.42")
    logger.info("普通信息")

    content = _file_log_path().read_text(encoding="utf-8")
    assert "普通信息" in content
    assert "文件专属调试 confidence=0.42" in content
    # 文件格式应包含 logger 名或模块名，便于回溯日志来源
    assert "test_logging" in content or "file_log" in content


def test_setup_logging_file_handler_idempotent(tmp_path: Path) -> None:
    """重复调用不叠加文件 handler，且更新文件级别。"""
    setup_logging("INFO", log_dir=tmp_path, argv=["wlxq-bot", "version"])
    setup_logging("DEBUG", log_dir=tmp_path, file_level="INFO", argv=["wlxq-bot", "version"])

    handlers = _file_handlers(logging.getLogger(ROOT_LOGGER_NAME))
    assert len(handlers) == 1
    assert handlers[0].level == logging.INFO


def test_setup_logging_without_log_dir_is_console_only() -> None:
    """不传 log_dir 时保持原有行为：只有控制台 handler。"""
    setup_logging("INFO")
    assert _file_handlers(logging.getLogger(ROOT_LOGGER_NAME)) == []


def test_file_log_startup_record_contains_command_line(tmp_path: Path) -> None:
    """文件首行附近记录完整命令行，便于事后确认这次跑的是什么。"""
    setup_logging(
        "INFO",
        log_dir=tmp_path,
        argv=["wlxq-bot", "run", "coop", "--main-c", "assault"],
    )
    content = _file_log_path().read_text(encoding="utf-8")
    assert "wlxq-bot run coop --main-c assault" in content


def test_file_log_cleanup_removes_expired_files(tmp_path: Path) -> None:
    """按修改时间清理超过保留期的 .log；其他文件不动。"""
    old = tmp_path / "wlxq-bot_20200101_000000.log"
    old.write_text("old", encoding="utf-8")
    fresh = tmp_path / "wlxq-bot_20991231_235959.log"
    fresh.write_text("fresh", encoding="utf-8")
    other = tmp_path / "notes.txt"
    other.write_text("keep me", encoding="utf-8")
    old_time = time.time() - 30 * 86400
    os.utime(old, (old_time, old_time))

    setup_logging("INFO", log_dir=tmp_path, retention_days=7, argv=["wlxq-bot", "version"])

    assert not old.exists()
    assert fresh.exists()
    assert other.exists()


def test_setup_logging_survives_file_error(tmp_path: Path) -> None:
    """日志目录创建失败（同名普通文件挡路）不阻塞命令，控制台照常输出。"""
    blocked = tmp_path / "blocked"
    blocked.write_text("占位", encoding="utf-8")

    setup_logging("INFO", log_dir=blocked, argv=["wlxq-bot", "version"])

    assert _file_handlers(logging.getLogger(ROOT_LOGGER_NAME)) == []
    get_logger("test.survive").info("仍可输出")


def test_cli_passes_logging_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI 回调读取 default.yaml 的 logging 段并传给 setup_logging。"""
    config_path = tmp_path / "default.yaml"
    config_path.write_text(
        "logging:\n  enabled: true\n  dir: 'mylogs'\n  file_level: 'INFO'\n  retention_days: 3\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_module, "DEFAULT_CONFIG_PATH", config_path)
    captured: dict[str, object] = {}

    def fake_setup(level: str, **kwargs: object) -> logging.Logger:
        captured.update(kwargs)
        return setup_logging(level)

    monkeypatch.setattr(cli_module, "setup_logging", fake_setup)
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert captured == {"log_dir": "mylogs", "file_level": "INFO", "retention_days": 3}


def test_cli_logging_disabled_passes_no_log_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """logging.enabled: false 时不传 log_dir，完全退回控制台模式。"""
    config_path = tmp_path / "default.yaml"
    config_path.write_text("logging:\n  enabled: false\n", encoding="utf-8")
    monkeypatch.setattr(cli_module, "DEFAULT_CONFIG_PATH", config_path)
    captured: dict[str, object] = {}

    def fake_setup(level: str, **kwargs: object) -> logging.Logger:
        captured.update(kwargs)
        return setup_logging(level)

    monkeypatch.setattr(cli_module, "setup_logging", fake_setup)
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert "log_dir" not in captured
