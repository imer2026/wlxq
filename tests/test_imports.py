"""验证包结构和核心模块可正常导入。

这是骨架阶段的冒烟测试，确保目录结构和 import 没有问题。
"""

from __future__ import annotations


def test_package_import() -> None:
    import wlxq_bot

    assert wlxq_bot.__version__ == "0.1.0"


def test_cli_import() -> None:
    from wlxq_bot.cli import app

    assert app is not None


def test_config_import() -> None:
    from wlxq_bot.config import DefaultConfig, TasksConfig

    assert DefaultConfig is not None
    assert TasksConfig is not None


def test_models_import() -> None:
    from wlxq_bot.models import (
        State,
        WindowContext,
    )

    assert WindowContext is not None
    assert State.UNKNOWN.value == "unknown"


def test_perception_import() -> None:
    from wlxq_bot.perception.locator import Locator
    from wlxq_bot.perception.screen import ScreenCapture
    from wlxq_bot.perception.vision import Vision

    assert ScreenCapture is not None
    assert Vision is not None
    assert Locator is not None


def test_action_import() -> None:
    from wlxq_bot.action.executor import ActionExecutor
    from wlxq_bot.action.input import FakeInput, InputController
    from wlxq_bot.action.safety import SafetyGuard

    assert ActionExecutor is not None
    assert InputController is not None
    assert FakeInput is not None
    assert SafetyGuard is not None


def test_debug_import() -> None:
    from wlxq_bot.debug.recorder import DebugRecorder

    assert DebugRecorder is not None


def test_hero_classifier_import() -> None:
    from wlxq_bot.hero_classifier import HeroCellCropper, HeroFrameCollector
    from wlxq_bot.perception.hero_classifier import HeroCellClassifier

    assert HeroFrameCollector is not None
    assert HeroCellCropper is not None
    assert HeroCellClassifier is not None


def test_tasks_import() -> None:
    from wlxq_bot.tasks.base import Task
    from wlxq_bot.tasks.coop import CoopTask

    assert Task is not None
    assert CoopTask is not None


def test_utils_import() -> None:
    from wlxq_bot.utils.log import get_logger
    from wlxq_bot.utils.time import Timeout

    assert get_logger is not None
    assert Timeout is not None


def test_assets_import() -> None:
    from wlxq_bot.assets import TemplatePack, find_template_pack

    assert TemplatePack is not None
    assert find_template_pack is not None
