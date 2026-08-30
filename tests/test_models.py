"""核心数据模型测试。"""

from __future__ import annotations

import time

from wlxq_bot.models import (
    Action,
    CoopRole,
    DebugEvent,
    MatchResult,
    Observation,
    State,
    WindowContext,
    board_roi_name,
)


class TestWindowContext:
    """WindowContext 测试。"""

    def _make_ctx(
        self,
        captured_at: float | None = None,
        is_foreground: bool = True,
        is_minimized: bool = False,
    ) -> WindowContext:
        return WindowContext(
            window_handle=12345,
            client_rect_screen=(100, 200, 1920, 1080),
            client_size=(1920, 1080),
            dpi=100,
            monitor_id="primary",
            is_foreground=is_foreground,
            is_minimized=is_minimized,
            captured_at=captured_at if captured_at is not None else time.time(),
            frame_id=1,
        )

    def test_client_to_screen(self) -> None:
        ctx = self._make_ctx()
        assert ctx.client_to_screen(0, 0) == (100, 200)
        assert ctx.client_to_screen(100, 50) == (200, 250)

    def test_contains(self) -> None:
        ctx = self._make_ctx()
        assert ctx.contains(0, 0)
        assert ctx.contains(1919, 1079)
        assert not ctx.contains(1920, 1080)
        assert not ctx.contains(-1, 0)

    def test_valid_when_fresh_and_foreground(self) -> None:
        ctx = self._make_ctx(captured_at=time.time())
        assert ctx.is_valid(time.time(), ttl_ms=500)

    def test_invalid_when_minimized(self) -> None:
        ctx = self._make_ctx(is_minimized=True)
        assert not ctx.is_valid(time.time(), ttl_ms=500)

    def test_invalid_when_not_foreground(self) -> None:
        ctx = self._make_ctx(is_foreground=False)
        assert not ctx.is_valid(time.time(), ttl_ms=500)

    def test_invalid_when_expired(self) -> None:
        ctx = self._make_ctx(captured_at=time.time() - 10)
        assert not ctx.is_valid(time.time(), ttl_ms=500)


class TestObservation:
    """Observation 测试。"""

    def test_best_match(self) -> None:
        matches = [
            MatchResult("btn", (10, 10), 0.8),
            MatchResult("btn", (20, 20), 0.95),
            MatchResult("other", (30, 30), 0.9),
        ]
        obs = Observation(frame_id=1, matches=matches)
        best = obs.best_match("btn")
        assert best is not None
        assert best.confidence == 0.95
        assert best.position == (20, 20)

    def test_best_match_missing(self) -> None:
        obs = Observation(frame_id=1)
        assert obs.best_match("missing") is None


class TestState:
    """State 枚举测试。"""

    def test_unknown_is_default(self) -> None:
        assert State.UNKNOWN.value == "unknown"

    def test_all_states_present(self) -> None:
        expected = {
            "unknown",
            "find_coop",
            "enter_match",
            "select_opening_skills",
            "build_main_c",
            "select_main_c_skills",
            "handle_result",
            "claim_reward",
            "check_round_limit",
            "completed",
            "blocking_dialog",
            "window_invalid",
        }
        actual = {s.value for s in State}
        assert actual == expected


class TestAction:
    """Action 测试。"""

    def test_click_action(self) -> None:
        a = Action(kind="click", target=(100, 200), reason="测试点击")
        assert a.kind == "click"
        assert a.target == (100, 200)
        assert a.reason == "测试点击"

    def test_wait_action(self) -> None:
        a = Action(kind="wait", duration=1.5)
        assert a.kind == "wait"
        assert a.duration == 1.5


class TestDebugEvent:
    """DebugEvent 测试。"""

    def test_event_creation(self) -> None:
        e = DebugEvent(
            frame_id=1,
            kind="capture",
            message="截图完成",
            data={"size": "1920x1080"},
            timestamp=1234567890.0,
        )
        assert e.kind == "capture"
        assert e.data["size"] == "1920x1080"


class TestCoopRole:
    """CoopRole 与棋盘 ROI 映射测试。"""

    def test_role_values(self) -> None:
        assert CoopRole.INITIATOR.value == "initiator"
        assert CoopRole.HELPER.value == "helper"

    def test_initiator_maps_left_board(self) -> None:
        assert board_roi_name(CoopRole.INITIATOR) == "bottom_left_board"

    def test_helper_maps_right_board(self) -> None:
        assert board_roi_name(CoopRole.HELPER) == "bottom_right_board"

    def test_both_roles_covered(self) -> None:
        names = {board_roi_name(r) for r in CoopRole}
        assert names == {"bottom_left_board", "bottom_right_board"}
