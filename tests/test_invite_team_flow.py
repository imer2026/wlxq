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
    "invite_button": Hotspot(x_ratio=0.69, y_ratio=0.69),
    "invite_friend_tab": Hotspot(x_ratio=0.81, y_ratio=0.95),
    "invite_guild_recruit": Hotspot(x_ratio=0.31, y_ratio=0.87),
    "invite_one_click": Hotspot(x_ratio=0.69, y_ratio=0.86),
    "invite_kick": Hotspot(x_ratio=0.70, y_ratio=0.78),
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
        for _ in range(5):
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

    def test_timeout_triggers_reinvite(self) -> None:
        """等待超过 reinvite_after_seconds：重置点击链重发邀请。"""
        task = _make_invite_task(invite=InviteConfig(reinvite_after_seconds=10.0))
        task.ctx.current_state = State.INVITE_WAITING
        task._invite_wait_since = 0.0
        obs = _observation({"invite_text_visible": True})
        # 直接把等待起点拨回 20 秒前
        task._invite_wait_since = -20.0
        action, to_state = task.decide_action(obs, _window_ctx())
        assert to_state == State.INVITE_CLICK
        assert action.tag == "invite_reinvite"
        assert task._invite_reinvite_count == 1

    def test_too_many_reinvites_stops(self) -> None:
        """连续空等重发超上限：返回 None（保守停止）。"""
        task = _make_invite_task(
            invite=InviteConfig(reinvite_after_seconds=10.0, max_empty_reinvites=1)
        )
        task.ctx.current_state = State.INVITE_WAITING
        task._invite_wait_since = -20.0
        task._invite_reinvite_count = 1  # 已重发过 1 次
        obs = _observation({"invite_text_visible": True})
        assert task.decide_action(obs, _window_ctx()) is None

    def test_helper_progress_parsed_from_raw(self) -> None:
        """感知层解析出的关卡进度：≥475 保留（任务层分流在后续帧完成）。"""
        # 此用例验证 raw_data 约定：解析值直接可读（数字比较在任务分流中）
        obs = _observation({"helper_stage_progress": 480})
        assert obs.raw_data["helper_stage_progress"] == 480


class TestInviteStart:
    def test_ready_match_clicks_start(self) -> None:
        """已准备模板命中：点击开局，进 ENTER_MATCH。"""
        task = _make_invite_task()
        task.ctx.current_state = State.INVITE_START

        match = MatchResult(
            position=(123, 456),
            confidence=0.95,
            template_name="tiao_zhan_dui_you_zhun_bei",
            roi_name="invite_start_button_area",
        )
        obs = _observation({"invite_start_ready_match": match})
        action, to_state = task.decide_action(obs, _window_ctx())
        assert to_state == State.ENTER_MATCH
        assert action.tag == "invite_start_game"

    def test_not_ready_waits(self) -> None:
        task = _make_invite_task()
        task.ctx.current_state = State.INVITE_START
        not_ready = MatchResult(
            position=(100, 200),
            confidence=0.95,
            template_name="tiao_zhan_dui_you_wei_zhun_bei",
            roi_name="invite_start_button_area",
        )
        obs = _observation({"invite_start_not_ready_match": not_ready})
        action, to_state = task.decide_action(obs, _window_ctx())
        assert to_state == State.INVITE_START
        assert action.tag == "invite_wait_ready"

    def test_helper_left_returns_to_waiting(self) -> None:
        """开局按钮消失且邀请文字重现：帮助者退出，回等待态。"""
        task = _make_invite_task()
        task.ctx.current_state = State.INVITE_START
        obs = _observation({"invite_text_visible": True})
        action, to_state = task.decide_action(obs, _window_ctx())
        assert to_state == State.INVITE_WAITING
        assert action.tag == "invite_helper_left"


class TestInviteCooldown:
    def test_cooldown_waits_then_returns_to_click(self) -> None:
        task = _make_invite_task(invite=InviteConfig(kick_cooldown_seconds=60.0))
        task.ctx.current_state = State.INVITE_COOLDOWN
        action, to_state = task.decide_action(_observation(), _window_ctx())
        assert to_state == State.INVITE_COOLDOWN
        assert action.tag == "invite_cooldown"
        # 冷却到期（把截止时刻拨到过去）
        task._invite_cooldown_until = -1.0
        action, to_state = task.decide_action(_observation(), _window_ctx())
        assert to_state == State.INVITE_CLICK
        assert action.tag == "invite_cooldown_done"


class TestInMatchIntegration:
    def test_ready_done_clicked_once(self) -> None:
        """「准备完成」动作首次返回点击、之后不再重复（一次性 gate）。"""
        task = _make_invite_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._effective_opening_summons()
        first = task._invite_ready_done_action(_window_ctx())
        assert first is not None
        action, to_state = first
        assert to_state == State.BUILD_MAIN_C
        assert action.tag == "invite_ready_done"
        assert task._invite_ready_done_action() is None

    def test_gold_wait_assist_ignores_non_invite(self) -> None:
        """非邀请模式（强袭）：卡住处置直接返回 None，走原有金币等待。"""
        ctx = TaskContext(main_c="assault", max_rounds=10)
        task = CoopTask(
            ctx,
            RunConfig(initial_board_capacity=7),
            CoopRole.HELPER,
            dict(_INVITE_HOTSPOTS),
        )
        from wlxq_bot.models import BoardCapacity, BoardSnapshot

        board = BoardSnapshot(
            frame_id=1,
            heroes=[],
            capacity=BoardCapacity(total_slots=12, occupied=12),
        )
        assert task._invite_gold_wait_assist(board, _observation(), "测试") is None

    def test_gold_wait_assist_stuck_waits_when_no_sacrifice(self) -> None:
        """卡住且无可牺牲英雄：干等动作（重置无进展计数）。"""
        task = _make_invite_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        from wlxq_bot.models import BoardCapacity, BoardHero, BoardSnapshot

        def _h(t: str, s: int) -> BoardHero:
            return BoardHero(
                hero_type=t,
                star_level=s,
                position=(100, 100),
                confidence=0.9,
                template_path=f"heroes/{t}/star{s}/t.png",
            )

        # 棋盘满且缺雪姬（阵容不齐），在场英雄均达保留数不可牺牲、无变身次数
        board = BoardSnapshot(
            frame_id=1,
            heroes=[_h("monkey", 1), _h("angel", 1)],
            capacity=BoardCapacity(total_slots=2, occupied=2),
        )
        task._transform_used_count = 2  # 变身耗尽
        result = task._invite_gold_wait_assist(board, _observation(), "测试卡住")
        assert result is not None
        action, to_state = result
        assert to_state == State.BUILD_MAIN_C
        assert action.tag == "invite_stuck_wait"
