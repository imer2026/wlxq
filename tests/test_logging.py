"""日志初始化与 ``--debug`` 开关测试。

验证：
- ``setup_logging`` 正确设置级别、幂等（不重复添加 handler）。
- ``get_logger`` 对短名和完整模块名都归一到 ``wlxq_bot.*``，不重复前缀。
- CLI ``--debug`` 全局选项把根日志器切到 DEBUG，不带时为 INFO。
"""

from __future__ import annotations

import logging

import pytest
from rich.logging import RichHandler
from typer.testing import CliRunner

from wlxq_bot.cli import app
from wlxq_bot.utils.log import ROOT_LOGGER_NAME, get_logger, setup_logging


@pytest.fixture(autouse=True)
def _restore_root_logger():
    """每个测试前后保存/恢复根日志器级别与 handler，隔离全局副作用。"""
    root = logging.getLogger(ROOT_LOGGER_NAME)
    saved_level = root.level
    saved_handlers = list(root.handlers)
    saved_propagate = root.propagate
    yield
    root.setLevel(saved_level)
    root.handlers = saved_handlers
    root.propagate = saved_propagate


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
