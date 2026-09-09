"""pytest 公共夹具。"""

from __future__ import annotations

import logging

import pytest

from wlxq_bot.action.input import FakeInput
from wlxq_bot.action.safety import SafetyGuard
from wlxq_bot.utils.log import ROOT_LOGGER_NAME


@pytest.fixture
def fake_input() -> FakeInput:
    """Fake Input，测试用，不实际操作桌面。"""
    return FakeInput()


@pytest.fixture
def safety_guard() -> SafetyGuard:
    """Safety Guard 测试实例。"""
    return SafetyGuard(max_failures=3, frame_ttl_ms=500)


@pytest.fixture(autouse=True)
def _isolated_root_logger(monkeypatch: pytest.MonkeyPatch):
    """隔离根日志器全局状态，避免测试间相互污染。

    - 通过 CLI（真实回调）的测试会让 ``setup_logging`` 挂上文件 handler；
      每个测试结束后关闭并移除测试期间新增的 handler，防止泄漏到后续测试。
    - CLI 测试统一丢弃文件日志参数，避免 pytest 运行时往工作目录写真实
      ``logs/`` 文件；文件日志行为本身由 test_logging.py 直接测
      ``setup_logging``，CLI 接线由其 monkeypatch ``wlxq_bot.cli.setup_logging`` 验证。
    """
    import wlxq_bot.cli as cli_module

    real_setup = cli_module.setup_logging

    def _setup_without_file_log(level, **kwargs):
        kwargs.pop("log_dir", None)
        return real_setup(level, **kwargs)

    monkeypatch.setattr(cli_module, "setup_logging", _setup_without_file_log)

    root = logging.getLogger(ROOT_LOGGER_NAME)
    saved_level = root.level
    saved_handlers = list(root.handlers)
    saved_propagate = root.propagate
    yield
    for handler in root.handlers:
        if handler not in saved_handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
            root.removeHandler(handler)
    root.setLevel(saved_level)
    root.handlers = saved_handlers
    root.propagate = saved_propagate
