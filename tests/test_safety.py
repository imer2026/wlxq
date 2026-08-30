"""Safety Guard 测试。

验证动作前安全检查逻辑，不实际移动鼠标。
"""

from __future__ import annotations

import time

from wlxq_bot.action.executor import ActionExecutor
from wlxq_bot.action.input import FakeInput
from wlxq_bot.action.safety import SafetyGuard
from wlxq_bot.models import Action, WindowContext


def _make_ctx(
    captured_at: float | None = None,
    is_foreground: bool = True,
    is_minimized: bool = False,
) -> WindowContext:
    return WindowContext(
        window_handle=1,
        client_rect_screen=(0, 0, 1920, 1080),
        client_size=(1920, 1080),
        dpi=100,
        monitor_id="primary",
        is_foreground=is_foreground,
        is_minimized=is_minimized,
        captured_at=captured_at if captured_at is not None else time.time(),
        frame_id=1,
    )


class TestSafetyGuard:
    def test_stop_request_blocks_action(self, safety_guard: SafetyGuard) -> None:
        safety_guard.request_stop()
        ctx = _make_ctx()
        action = Action(kind="click", target=(100, 100))
        ok, reason = safety_guard.check_action(ctx, action, time.time())
        assert not ok
        assert "停止信号" in reason

    def test_minimized_blocks_action(self, safety_guard: SafetyGuard) -> None:
        ctx = _make_ctx(is_minimized=True)
        action = Action(kind="click", target=(100, 100))
        ok, reason = safety_guard.check_action(ctx, action, time.time())
        assert not ok
        assert "最小化" in reason

    def test_expired_frame_blocks_action(self, safety_guard: SafetyGuard) -> None:
        ctx = _make_ctx(captured_at=time.time() - 10)
        action = Action(kind="click", target=(100, 100))
        ok, reason = safety_guard.check_action(ctx, action, time.time())
        assert not ok
        assert "超时" in reason

    def test_out_of_bounds_blocks_action(self, safety_guard: SafetyGuard) -> None:
        ctx = _make_ctx()
        action = Action(kind="click", target=(9999, 9999))
        ok, reason = safety_guard.check_action(ctx, action, time.time())
        assert not ok
        assert "超出客户区" in reason

    def test_valid_action_passes(self, safety_guard: SafetyGuard) -> None:
        ctx = _make_ctx()
        action = Action(kind="click", target=(100, 100))
        ok, reason = safety_guard.check_action(ctx, action, time.time())
        assert ok
        assert reason == ""

    def test_drag_end_out_of_bounds(self, safety_guard: SafetyGuard) -> None:
        ctx = _make_ctx()
        action = Action(
            kind="drag",
            target=(100, 100),
            end=(9999, 9999),
        )
        ok, reason = safety_guard.check_action(ctx, action, time.time())
        assert not ok
        assert "超出客户区" in reason

    def test_failure_count(self, safety_guard: SafetyGuard) -> None:
        assert not safety_guard.record_failure()
        assert not safety_guard.record_failure()
        assert safety_guard.record_failure()  # 第3次达到上限

        safety_guard.reset_failures()
        assert not safety_guard.record_failure()


class TestActionExecutor:
    def test_revalidates_window_context_before_input(self) -> None:
        input_ctrl = FakeInput()
        executor = ActionExecutor(
            SafetyGuard(frame_ttl_ms=5000),
            input_ctrl,
            min_delay=0,
            max_delay=0,
            context_validator=lambda _ctx: (False, "客户区尺寸已变化"),
        )

        result = executor.execute(
            _make_ctx(),
            Action(kind="click", target=(100, 100)),
        )

        assert not result.executed
        assert "尺寸已变化" in result.failure_reason
        assert input_ctrl.calls == []

    def test_input_is_not_marked_verified_without_task_observation(self) -> None:
        executor = ActionExecutor(
            SafetyGuard(frame_ttl_ms=5000),
            FakeInput(),
            min_delay=0,
            max_delay=0,
        )

        result = executor.execute(
            _make_ctx(),
            Action(kind="click", target=(100, 100)),
        )

        assert result.executed
        assert not result.verified

    def test_wait_is_verified_without_external_state_change(self) -> None:
        executor = ActionExecutor(
            SafetyGuard(frame_ttl_ms=5000),
            FakeInput(),
            min_delay=0,
            max_delay=0,
        )

        result = executor.execute(
            _make_ctx(),
            Action(kind="wait", duration=0.001),
        )

        assert result.executed
        assert result.verified

    def test_explicit_post_delay_replaces_generic_random_delay(self) -> None:
        sleeps: list[float] = []
        executor = ActionExecutor(
            SafetyGuard(frame_ttl_ms=5000),
            FakeInput(),
            min_delay=0.3,
            max_delay=0.8,
            sleep=sleeps.append,
        )

        result = executor.execute(
            _make_ctx(),
            Action(kind="click", target=(100, 100), post_delay=1.5),
        )

        assert result.executed
        assert sleeps == [1.5]

    def test_wait_sleeps_exactly_its_duration_without_extra_delay(self) -> None:
        """wait 的 duration 就是全部等待：不叠加随机拟人间隔（轮询节奏修正）。"""
        sleeps: list[float] = []
        executor = ActionExecutor(
            SafetyGuard(frame_ttl_ms=5000),
            FakeInput(),
            min_delay=0.5,
            max_delay=1.0,
            sleep=sleeps.append,
        )

        result = executor.execute(
            _make_ctx(),
            Action(kind="wait", duration=0.25),
        )

        assert result.executed
        assert sleeps == [0.25]

    def test_zero_duration_wait_advances_immediately(self) -> None:
        """duration=0 的确认类等待立即推进（不被 falsy 兜底拖成 1 秒）。"""
        sleeps: list[float] = []
        executor = ActionExecutor(
            SafetyGuard(frame_ttl_ms=5000),
            FakeInput(),
            sleep=sleeps.append,
        )

        executor.execute(_make_ctx(), Action(kind="wait", duration=0.0))

        assert sleeps == [0.0]

    def test_click_gets_humanized_random_delay(self) -> None:
        """输入动作后叠加拟人随机间隔（配置区间内，导航点击不再机器枪式连点）。"""
        sleeps: list[float] = []
        executor = ActionExecutor(
            SafetyGuard(frame_ttl_ms=5000),
            FakeInput(),
            min_delay=0.5,
            max_delay=1.0,
            sleep=sleeps.append,
        )

        executor.execute(_make_ctx(), Action(kind="click", target=(100, 100)))

        assert len(sleeps) == 1
        assert 0.5 <= sleeps[0] <= 1.0
