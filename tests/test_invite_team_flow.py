"""邀请者模式组队状态机单元测试（spec 2026-09-21-initiator-monkey-design.md §6）。

fake observation 驱动：INVITE_CLICK 5 步点击链、INVITE_WAITING 进房判定
（含复检与超时重发）、INVITE_START 双态按钮、INVITE_COOLDOWN 冷却回环。
"""

from __future__ import annotations

from wlxq_bot.config import Hotspot, InviteConfig, MainCProfile, RoiConfig, RunConfig
from wlxq_bot.models import CoopRole, MatchResult, State, WindowContext
from wlxq_bot.tasks.base import TaskContext
from wlxq_bot.tasks.coop import CoopTask

_INVITE_HOTSPOTS = {
    "invite_start_challenge": Hotspot(x_ratio=0.55, y_ratio=0.95),
    "invite_button": Hotspot(x_ratio=0.69, y_ratio=0.69),
    "invite_friend_tab": Hotspot(x_ratio=0.81, y_ratio=0.95),
    "invite_guild_recruit": Hotspot(x_ratio=0.31, y_ratio=0.87),
    "invite_one_click": Hotspot(x_ratio=0.69, y_ratio=0.86),
    "invite_kick": Hotspot(x_ratio=0.70, y_ratio=0.78),
    "invite_close_fallback": Hotspot(x_ratio=0.91, y_ratio=0.14),
    "opening_ready_done": Hotspot(x_ratio=0.68, y_ratio=0.30),
}

_STEPS = ("invite_button", "invite_one_click", "invite_friend_tab", "invite_guild_recruit", "invite_one_click")


def _window_ctx() -> WindowContext:
    return WindowContext(
        window_handle=1,
        client_rect_screen=(0, 0, 923, 1723),
        client_size=(923, 1723),
        dpi=96,
        monitor_id="primary",
        is_foreground=True,
        is_minimized=False,
        captured_at=0.0,
        frame_id=1,
    )


def _observation(raw: dict | None = None):
    from wlxq_bot.models import Observation

    return Observation(
        frame_id=1,
        source_frame_ids=(1,),
        matches=(),
        raw_data=raw or {},
    )


class _FakeClock:
    """受控单调时钟：time 属性手动推进。"""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _make_invite_task(
    invite: InviteConfig | None = None,
    clock=None,
) -> CoopTask:
    ctx = TaskContext(main_c="monkey", max_rounds=10)
    profile = MainCProfile(
        display_name="猴子",
        hero_template_dir="heroes/monkey",
        target_star_override=1,
        opening_summons=5,
        summon_batch_size=3,
        forbid_star3_merge=True,
        hero_keep_counts={"angel": 1, "snow": 1, "death_knight": 0},
        required_lineup_heroes=["angel", "snow"],
        transform_sacrifice_order=["death_knight", "snow"],
    )
    task = CoopTask(
        ctx,
        RunConfig(initial_board_capacity=7),
        CoopRole.INITIATOR,
        dict(_INVITE_HOTSPOTS),
        skill_candidate_roi=RoiConfig(x_ratio=0.1, y_ratio=0.2, width_ratio=0.6, height_ratio=0.4),
        main_c_profile=profile,
        invite_mode=True,
        invite_config=invite or InviteConfig(),
        clock=clock if clock is not None else _FakeClock(),
    )
    return task


class TestInviteClickChain:
    def test_starts_in_invite_click(self) -> None:
        task = _make_invite_task()
        assert task.ctx.current_state == State.INVITE_CLICK

    def test_chain_walks_five_steps_then_waits(self) -> None:
        task = _make_invite_task()
        ctx = _window_ctx()
        visited: list[str] = []
        for _ in range(6):
            action, to_state = task.decide_action(_observation(), ctx)
            assert to_state == State.INVITE_CLICK
            visited.append(str(action.target))
        action, to_state = task.decide_action(_observation(), ctx)
        assert to_state == State.INVITE_WAITING
        assert action.tag == "invite_wait_entry"
        # 点击链游标重置，供下一轮重发
        assert task._invite_step == 0


class TestInviteWaiting:
    def test_visible_text_keeps_waiting(self) -> None:
        clock = _FakeClock()
        task = _make_invite_task(clock=clock)
        clock.now = 100.0
        task.ctx.current_state = State.INVITE_WAITING
        task._invite_wait_since = 100.0
        obs = _observation({"invite_text_visible": True})
        action, to_state = task.decide_action(obs, _window_ctx())
        assert to_state == State.INVITE_WAITING
        assert action.tag == "invite_waiting"

    def test_invisible_starts_recheck_not_navigation(self) -> None:
        """邀请文字首次不可见：先复检，不直接打开帮助者信息。"""
        task = _make_invite_task()
        task.ctx.current_state = State.INVITE_WAITING
        task._invite_wait_since = 0.0
        obs = _observation({"invite_text_visible": False})
        action, to_state = task.decide_action(obs, _window_ctx())
        assert to_state == State.INVITE_WAITING
        assert action.tag == "invite_join_recheck"

    def test_second_invisible_opens_helper_info(self) -> None:
        """连续两帧不可见：判定有人进房，点邀请位置弹信息层。"""
        task = _make_invite_task()
        task.ctx.current_state = State.INVITE_WAITING
        task._invite_wait_since = 0.0
        task._invite_text_missing_frames = 1
        obs = _observation({"invite_text_visible": False})
        action, to_state = task.decide_action(obs, _window_ctx())
        assert to_state == State.INVITE_WAITING
        assert action.tag == "invite_open_helper_info"
        assert task._invite_helper_opened is True


class TestHelperInfoFlow:
    """帮助者信息层三帧阶段机：打开 → 读进度+关层 → 执行分流。"""

    def _opened_task(self) -> CoopTask:
        task = _make_invite_task()
        task.ctx.current_state = State.INVITE_WAITING
        task._invite_helper_opened = True  # 弹层已打开（帧 2）
        return task

    def test_frame2_close_then_frame3_accept(self) -> None:
        """进度 480 ≥475：帧2 点 × 关层；帧3 迁 INVITE_START。"""
        task = self._opened_task()
        close = MatchResult(position=(800, 240), confidence=0.9,
                            template_name="x", roi_name="close_button_area")
        obs = _observation({
            "invite_text_visible": False,
            "helper_stage_progress": 480,
            "close_button_match": close,
        })
        action, to_state = task.decide_action(obs, _window_ctx())
        assert action.tag == "invite_close_helper_info"
        assert to_state == State.INVITE_WAITING
        assert task._invite_pending_decision == "accept"
        # 帧 3：弹层已关，执行 accept
        action2, to_state2 = task.decide_action(
            _observation({"invite_text_visible": False}), _window_ctx()
        )
        assert action2.tag == "invite_helper_accepted"
        assert to_state2 == State.INVITE_START

    def test_low_progress_close_then_kick(self) -> None:
        """进度 200 <475：帧2 关层（兜底 × 位置）；帧3 点请离进冷却。"""
        task = self._opened_task()
        obs = _observation({
            "invite_text_visible": False,
            "helper_stage_progress": 200,
        })
        action, to_state = task.decide_action(obs, _window_ctx())
        assert action.tag == "invite_close_helper_fallback"
        assert to_state == State.INVITE_WAITING
        assert task._invite_pending_decision == "kick"
        action2, to_state2 = task.decide_action(
            _observation({"invite_text_visible": False}), _window_ctx()
        )
        assert action2.tag == "invite_kick_helper"
        assert to_state2 == State.INVITE_COOLDOWN

    def test_progress_misses_retry_then_stop(self) -> None:
        """进度读不到：关层重试（帧循环），达上限保守停止。"""
        task = self._opened_task()
        for i in range(1, task._INVITE_PROGRESS_MAX_MISSES):
            obs = _observation({
                "invite_text_visible": False,
                "helper_stage_progress": None,
            })
            result = task.decide_action(obs, _window_ctx())
            assert result is not None, f"第 {i} 次失败不应停止"
            action, _ = result
            assert action.tag in ("invite_close_helper_info", "invite_close_helper_fallback")
            # 关层后重开（模拟循环）
            task._invite_helper_opened = False
            task.decide_action(
                _observation({"invite_text_visible": False}), _window_ctx()
            )
            task._invite_helper_opened = True
        # 达上限：保守停止
        task._invite_helper_opened = True
        obs = _observation({"invite_text_visible": False, "helper_stage_progress": None})
        assert task.decide_action(obs, _window_ctx()) is None
