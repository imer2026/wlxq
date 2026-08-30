"""CoopGrabCoordinator 单元测试。

用 Fake 组件验证双线程抢合作逻辑，不真实截图、不真实点击用户桌面。
抢合作线程连点 join_coop、检查线程周期识别准备按钮，二者重叠。
"""

from __future__ import annotations

import time

from wlxq_bot.action.safety import SafetyGuard
from wlxq_bot.config import Hotspot, RunConfig
from wlxq_bot.models import Action, ActionResult, MatchResult, WindowContext
from wlxq_bot.orchestration.coop_grab import (
    _TRANSIENT_FAILURE_REASONS,
    CoopGrabCoordinator,
)


def make_ctx(
    foreground: bool = True,
    minimized: bool = False,
    frame_id: int = 1,
    age: float = 0.0,
) -> WindowContext:
    return WindowContext(
        window_handle=1,
        client_rect_screen=(0, 0, 923, 1727),
        client_size=(923, 1727),
        dpi=96,
        monitor_id="primary",
        is_foreground=foreground,
        is_minimized=minimized,
        captured_at=time.time() - age,
        frame_id=frame_id,
    )


class FakeScreen:
    """假截图器：返回预设上下文，可模拟最小化、失焦或截图失败。

    fail_after=N 表示前 N 次截图成功，之后抛错（模拟运行中截图开始失败）。
    foreground_plan 为每次截图的前台状态序列（含 run() 的起始截图），
    耗尽后回落到 foreground，用于模拟「失焦几帧后切回」。
    """

    def __init__(
        self,
        *,
        foreground: bool = True,
        minimized: bool = False,
        raise_on_capture: bool = False,
        fail_after: int | None = None,
        foreground_plan: list[bool] | None = None,
    ) -> None:
        self._foreground = foreground
        self._minimized = minimized
        self._raise = raise_on_capture
        self._fail_after = fail_after
        self._foreground_plan = list(foreground_plan) if foreground_plan else []
        self.count = 0

    def capture(self, handle: int):
        if self._raise or (self._fail_after is not None and self.count >= self._fail_after):
            raise RuntimeError("fake capture fail")
        self.count += 1
        if self.count <= len(self._foreground_plan):
            foreground = self._foreground_plan[self.count - 1]
        else:
            foreground = self._foreground
        return (
            make_ctx(
                foreground=foreground,
                minimized=self._minimized,
                frame_id=self.count,
            ),
            None,
        )


class FakePerception:
    """假识别：按预设序列返回准备按钮匹配结果。"""

    def __init__(
        self,
        matches: list[MatchResult | None],
        *,
        raise_on_match: bool = False,
        unexpected_error: bool = False,
    ) -> None:
        self._matches = list(matches)
        self.calls = 0
        self._raise_on_match = raise_on_match
        self._unexpected_error = unexpected_error

    def match_ready_button(self, ctx, frame):
        self.calls += 1
        if self._raise_on_match:
            raise RuntimeError("fake perception failure")
        if self._unexpected_error:
            raise TypeError("fake unexpected worker failure")
        if self.calls <= len(self._matches):
            return self._matches[self.calls - 1]
        return None


class FakeExecutor:
    """假执行器：记录点击，恒定执行成功。"""

    def __init__(self) -> None:
        self.actions = []

    def execute(self, ctx, action):
        self.actions.append(action)
        return ActionResult(executed=True, verified=True)


class FlakyExecutor:
    """假执行器：前 N 次以指定原因失败，之后成功。

    用于验证瞬态失败（上下文超龄）不退出抢合作，而非瞬态失败（几何变化）
    连续达上限仍保守停止。
    """

    def __init__(self, failures: int, reason: str) -> None:
        self._remaining = failures
        self._reason = reason
        self.actions = []

    def execute(self, ctx, action):
        if self._remaining > 0:
            self._remaining -= 1
            return ActionResult(executed=False, verified=False, failure_reason=self._reason)
        self.actions.append(action)
        return ActionResult(executed=True, verified=True)


def make_run_config(**overrides) -> RunConfig:
    base = {
        "find_coop_check_interval_seconds": 0.01,
        "find_coop_max_duration_seconds": 2.0,
        "find_coop_click_delay_min": 0.0,
        "find_coop_click_delay_max": 0.0,
    }
    base.update(overrides)
    return RunConfig(**base)


def make_coordinator(
    screen: FakeScreen,
    perception: FakePerception,
    executor: FakeExecutor,
    safety: SafetyGuard,
    get_idle_seconds=lambda: 0.0,
    activate=lambda handle: False,
    probe_window=lambda handle: None,
    **cfg,
) -> CoopGrabCoordinator:
    return CoopGrabCoordinator(
        screen=screen,
        perception=perception,
        grab_executor=executor,
        safety=safety,
        hotspots={"join_coop": Hotspot(x_ratio=0.6, y_ratio=0.75)},
        run_config=make_run_config(**cfg),
        window_handle=1,
        get_idle_seconds=get_idle_seconds,
        activate=activate,
        probe_window=probe_window,
    )


def test_finds_ready_button_while_grab_clicks():
    # 前 2 次没识别到，第 3 次识别到准备按钮
    perception = FakePerception([None, None, MatchResult("zhun_bei", (500, 900), 0.9)])
    screen = FakeScreen()
    executor = FakeExecutor()
    coordinator = make_coordinator(screen, perception, executor, SafetyGuard())

    result = coordinator.run()

    assert result.found
    assert result.reason == "ready_found"
    # 抢合作线程在检查线程识别到之前应已连点若干次 join_coop
    assert len(executor.actions) >= 1
    assert all(action.tag == "find_coop_click" for action in executor.actions)
    assert perception.calls >= 3


def test_window_lost_when_minimized_persists_past_foreground_wait():
    # 最小化后一直不恢复：挂起等待，持续超过 window_foreground_wait_seconds
    # 才判 window_lost（与主循环同策略），不再一帧否决
    screen = FakeScreen(minimized=True)
    coordinator = make_coordinator(
        screen,
        FakePerception([]),
        FakeExecutor(),
        SafetyGuard(),
        window_foreground_wait_seconds=0.05,
    )

    result = coordinator.run()

    assert not result.found
    assert result.reason == "window_lost"


def test_focus_loss_suspends_and_resumes_grab():
    # 起始前台，第 2~3 次截图失焦（挂起等待），第 4 次起切回前台，
    # 继续识别并抢到准备按钮——短暂失焦不再终止抢合作
    screen = FakeScreen(foreground_plan=[True, False, False, True])
    perception = FakePerception([None, None, MatchResult("zhun_bei", (500, 900), 0.9)])
    coordinator = make_coordinator(screen, perception, FakeExecutor(), SafetyGuard())

    result = coordinator.run()

    assert result.found
    assert result.reason == "ready_found"


def test_transient_grab_failures_do_not_abort():
    # 连点被「窗口上下文已超时」连续拒绝 5 次（旧逻辑 3 次即误判 window_lost）：
    # 属瞬态失败不累计，等检查线程刷新后继续，最终正常抢到
    executor = FlakyExecutor(failures=5, reason="窗口上下文已超时")
    perception = FakePerception([None, None, MatchResult("zhun_bei", (500, 900), 0.9)])
    coordinator = make_coordinator(FakeScreen(), perception, executor, SafetyGuard())

    result = coordinator.run()

    assert result.found
    assert result.reason == "ready_found"


def test_non_transient_grab_failures_still_abort():
    # 几何变化不是瞬态失败：连续达到上限仍判 window_lost
    executor = FlakyExecutor(failures=5, reason="客户区尺寸已变化")
    coordinator = make_coordinator(
        FakeScreen(),
        FakePerception([None, None, None, None, None]),
        executor,
        SafetyGuard(),
        find_coop_max_duration_seconds=3.0,
    )

    result = coordinator.run()

    assert not result.found
    assert result.reason == "window_lost"


def test_refocus_fires_when_system_idle():
    # 失焦挂起且系统空闲超过 refocus_idle_seconds：自动尝试把窗口切回前台
    # （激活后窗口仍未恢复，最终按超时退出）
    activated = []
    screen = FakeScreen(foreground=False)
    coordinator = make_coordinator(
        screen,
        FakePerception([]),
        FakeExecutor(),
        SafetyGuard(),
        window_foreground_wait_seconds=0.3,
        get_idle_seconds=lambda: 30.0,
        activate=activated.append,
    )

    result = coordinator.run()

    assert activated == [1]
    assert not result.found
    assert result.reason == "window_lost"


def test_transient_failure_reasons_match_safety_guard():
    # 守卫测试：瞬态失败原因的文案必须与 safety.check_action 实际返回一致，
    # 防止一侧改文案后另一侧静默失配、瞬态失败被重新计成窗口丢失
    guard = SafetyGuard()
    action = Action(kind="click", target=(100, 100))

    ok_inactive, reason_inactive = guard.check_action(
        make_ctx(foreground=False), action, time.time()
    )
    assert not ok_inactive
    assert reason_inactive in _TRANSIENT_FAILURE_REASONS

    ok_stale, reason_stale = guard.check_action(
        make_ctx(age=10.0), action, time.time()
    )
    assert not ok_stale
    assert reason_stale in _TRANSIENT_FAILURE_REASONS


def test_capture_failed_when_starting_capture_raises():
    screen = FakeScreen(raise_on_capture=True)
    coordinator = make_coordinator(
        screen,
        FakePerception([]),
        FakeExecutor(),
        SafetyGuard(),
    )

    result = coordinator.run()

    assert not result.found
    assert result.reason == "capture_failed"


def test_no_join_hotspot_returns_without_threads():
    screen = FakeScreen()
    coordinator = make_coordinator(screen, FakePerception([]), FakeExecutor(), SafetyGuard())
    coordinator._hotspots = {}  # 模拟未标定 join_coop

    result = coordinator.run()

    assert not result.found
    assert result.reason == "no_join_hotspot"


def test_timeout_when_never_found():
    perception = FakePerception([])  # 永远返回 None
    screen = FakeScreen()
    executor = FakeExecutor()
    coordinator = make_coordinator(
        screen,
        perception,
        executor,
        SafetyGuard(),
        find_coop_max_duration_seconds=0.1,
    )

    result = coordinator.run()

    assert not result.found
    assert result.reason == "timeout"


def test_capture_failed_when_capture_starts_failing_mid_run():
    # 起始截图成功（fail_after=1），之后检查线程每次截图都失败 → capture_failed
    screen = FakeScreen(fail_after=1)
    coordinator = make_coordinator(
        screen,
        FakePerception([None, None, None, None, None]),
        FakeExecutor(),
        SafetyGuard(),
        find_coop_max_duration_seconds=3.0,
    )

    result = coordinator.run()

    assert not result.found
    assert result.reason == "capture_failed"


def test_perception_failed_when_ready_matching_keeps_failing():
    coordinator = make_coordinator(
        FakeScreen(),
        FakePerception([], raise_on_match=True),
        FakeExecutor(),
        SafetyGuard(),
        find_coop_max_duration_seconds=3.0,
    )

    result = coordinator.run()

    assert not result.found
    assert result.reason == "perception_failed"


def test_unexpected_worker_error_does_not_wait_until_timeout():
    coordinator = make_coordinator(
        FakeScreen(),
        FakePerception([], unexpected_error=True),
        FakeExecutor(),
        SafetyGuard(),
        find_coop_max_duration_seconds=3.0,
    )

    result = coordinator.run()

    assert not result.found
    assert result.reason == "worker_failed"
