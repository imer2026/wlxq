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
    "invite_popup_close": Hotspot(x_ratio=0.91, y_ratio=0.14),
    "invite_stage_challenge": Hotspot(x_ratio=0.50, y_ratio=0.91),
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

    def test_chain_walks_then_waits(self) -> None:
        """前 6 步为点击；第 7 步（关弹窗）× 未识别到时自动跳过并收尾。"""
        task = _make_invite_task()
        ctx = _window_ctx()
        visited: list[str] = []
        for _ in range(6):
            action, to_state = task.decide_action(_observation(), ctx)
            assert to_state == State.INVITE_CLICK
            visited.append(str(action.target))
        assert len(visited) == 6
        # 第 7 步：× 未识别到（弹窗已被游戏自动关闭）→ 跳过，直接收尾
        action, to_state = task.decide_action(_observation(), ctx)
        assert to_state == State.INVITE_WAITING
        assert action.tag == "invite_wait_entry"
        # 点击链游标重置，供下一轮重发
        assert task._invite_step == 0

    def test_chain_step7_clicks_matched_x(self) -> None:
        """第 7 步：× 识别到 → 点匹配位置关弹窗。"""
        task = _make_invite_task()
        ctx = _window_ctx()
        for _ in range(6):
            task.decide_action(_observation(), ctx)
        close = MatchResult(position=(840, 248), confidence=0.95,
                            template_name="x", roi_name="close_button_area")
        action, to_state = task.decide_action(
            _observation({"close_button_match": close}), ctx
        )
        assert to_state == State.INVITE_CLICK
        assert str(action.target) == str(close.position)


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

    def test_invisible_dismisses_residue_first(self) -> None:
        """邀请文字不可见：先关一次残留弹窗再判定（两帧）。

        2026-09-24 实机教训：邀请弹窗关闭后文字有恢复延迟，直接判进房
        会把自己的信息层当帮助者（误读自己 475 进度）。
        """
        task = _make_invite_task()
        task.ctx.current_state = State.INVITE_WAITING
        task._invite_wait_since = 0.0
        obs = _observation({"invite_text_visible": False})
        # × 未识别到（无残留弹窗）：不盲点，直接打开帮助者信息层
        action, to_state = task.decide_action(obs, _window_ctx())
        assert action.tag == "invite_open_helper_info"
        assert to_state == State.INVITE_WAITING
        assert task._invite_helper_opened is True
        # × 识别到（确有残留弹窗）：先点 ×，下一帧再走帮助者流程
        task2 = _make_invite_task()
        task2.ctx.current_state = State.INVITE_WAITING
        task2._invite_wait_since = 0.0
        close = MatchResult(position=(840, 248), confidence=0.9,
                            template_name="x", roi_name="close_button_area")
        obs2 = _observation({"invite_text_visible": False, "close_button_match": close})
        action2, _ = task2.decide_action(obs2, _window_ctx())
        assert action2.tag == "invite_dismiss_residue"
        assert task2._invite_dismiss_checked is True
        action3, _ = task2.decide_action(obs2, _window_ctx())
        assert action3.tag == "invite_open_helper_info"


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
        # 合格：关层与迁移一步完成（无帧 3）
        assert to_state == State.INVITE_START
        assert task._invite_pending_decision is None

    def test_close_x_not_visible_waits(self) -> None:
        """弹层开但 × 未识别到：等待重读，不盲点固定坐标。"""
        task = self._opened_task()
        obs = _observation({
            "invite_text_visible": False,
            "helper_stage_progress": 200,
        })
        action, to_state = task.decide_action(obs, _window_ctx())
        assert action.tag == "invite_wait_close_button"
        assert to_state == State.INVITE_WAITING
        assert task._invite_helper_opened is True  # 保持打开状态

    def test_low_progress_close_then_kick(self) -> None:
        """进度 200 <475：帧2 关层（× 识别到才点）；帧3 点请离进冷却。"""
        task = self._opened_task()
        close = MatchResult(position=(800, 240), confidence=0.9,
                            template_name="x", roi_name="close_button_area")
        obs = _observation({
            "invite_text_visible": False,
            "helper_stage_progress": 200,
            "close_button_match": close,
        })
        action, to_state = task.decide_action(obs, _window_ctx())
        assert action.tag == "invite_close_helper_info"
        assert to_state == State.INVITE_WAITING
        assert task._invite_pending_decision == "kick"
        action2, to_state2 = task.decide_action(
            _observation({"invite_text_visible": False}), _window_ctx()
        )
        assert action2.tag == "invite_kick_helper"
        assert to_state2 == State.INVITE_COOLDOWN

    def test_progress_misses_retry_then_stop(self) -> None:
        """进度持续读不到：重试计数推进，达上限保守停止。"""
        task = self._opened_task()
        obs = _observation({
            "invite_text_visible": False,
            "helper_stage_progress": None,
        })
        # 每次 decide_action 都处于"帧 2"（弹层开），进度 None → 计数推进。
        # × 识别到与否不影响计数（等待帧/关层帧都推进）
        stops = 0
        for i in range(1, task._INVITE_PROGRESS_MAX_MISSES + 2):
            task._invite_helper_opened = True  # 保持帧 2 语境
            result = task.decide_action(obs, _window_ctx())
            if result is None:
                stops = i
                break
            assert result[0].tag in (
                "invite_wait_close_button",
                "invite_close_helper_info",
            )
        assert stops == task._INVITE_PROGRESS_MAX_MISSES, (
            f"应在第 {task._INVITE_PROGRESS_MAX_MISSES} 次失败后停止，实际第 {stops} 次"
        )


class TestInviteStartConfirm:
    def test_start_then_confirm_challenge(self) -> None:
        """两段式开局：点已准备（停留等待）→ 下一点确认页【挑战】→ 选技能。"""
        task = _make_invite_task()
        task.ctx.current_state = State.INVITE_START
        ready = MatchResult(position=(500, 1500), confidence=0.95,
                            template_name="zhun_bei", roi_name="invite_start_button_area")
        obs = _observation({"invite_start_ready_match": ready})
        action, to_state = task.decide_action(obs, _window_ctx())
        assert action.tag == "invite_start_game"
        assert to_state == State.INVITE_START
        assert task._invite_confirm_pending is True
        action2, to_state2 = task.decide_action(
            _observation({"invite_text_visible": False}), _window_ctx()
        )
        assert action2.tag == "invite_confirm_challenge"
        assert to_state2 == State.SELECT_OPENING_SKILLS
        assert task._invite_confirm_pending is False

    def test_start_failed_resets_confirm_flag(self) -> None:
        """开始点击验证失败：on_action_failed 撤销确认页标记，可重新点击。"""
        task = _make_invite_task()
        task.ctx.current_state = State.INVITE_START
        ready = MatchResult(position=(500, 1500), confidence=0.95,
                            template_name="zhun_bei", roi_name="invite_start_button_area")
        obs = _observation({"invite_start_ready_match": ready})
        task.decide_action(obs, _window_ctx())
        assert task._invite_confirm_pending is True
        from wlxq_bot.models import Action

        task.on_action_failed(Action(kind="click", target=(0, 0), tag="invite_start_game"))
        assert task._invite_confirm_pending is False
