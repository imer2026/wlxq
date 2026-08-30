"""pytest 公共夹具。"""

from __future__ import annotations

import pytest

from wlxq_bot.action.input import FakeInput
from wlxq_bot.action.safety import SafetyGuard


@pytest.fixture
def fake_input() -> FakeInput:
    """Fake Input，测试用，不实际操作桌面。"""
    return FakeInput()


@pytest.fixture
def safety_guard() -> SafetyGuard:
    """Safety Guard 测试实例。"""
    return SafetyGuard(max_failures=3, frame_ttl_ms=500)
