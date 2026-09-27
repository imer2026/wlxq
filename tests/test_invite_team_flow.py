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
        # 房间屏锚点（灰按钮可见）必须在场，进房判定才合法
        obs = _observation({"invite_text_visible": False, "invite_start_not_ready_visible": True})
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
        obs2 = _observation({
            "invite_text_visible": False,
            "invite_start_not_ready_visible": True,
            "close_button_match": close,
        })
        action2, _ = task2.decide_action(obs2, _window_ctx())
        assert action2.tag == "invite_dismiss_residue"
        assert task2._invite_dismiss_checked is True
        action3, _ = task2.decide_action(obs2, _window_ctx())
        assert action3.tag == "invite_open_helper_info"

    def test_non_room_frame_freezes_judgment(self) -> None:
        """非房间画面（金/灰按钮都不在，如加载页/对局内）：冻结进房判定。

        2026-09-26 实机回归：误入对局后每帧判「有人进房」，对局画面盲点
        邀请位直到进度读取计数耗尽保守停止。
        """
        task = _make_invite_task()
        task.ctx.current_state = State.INVITE_WAITING
        task._invite_wait_since = 0.0
        obs = _observation({"invite_text_visible": False})
        action, to_state = task.decide_action(obs, _window_ctx())
        assert action.tag == "invite_waiting_not_room"
        assert to_state == State.INVITE_WAITING
        assert task._invite_helper_opened is False  # 未点邀请位开层


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
    """单段式开局（2026-09-26）：组队房间只有一张屏，金色挑战=可开局，
    点击直下 SELECT_OPENING_SKILLS，无"确认页第二击"。"""

    @staticmethod
    def _ready_obs() -> dict:
        return {
            "invite_start_ready_match": MatchResult(
                position=(500, 1500),
                confidence=0.95,
                template_name="zhun_bei",
                roi_name="invite_start_button_area",
            ),
        }

    def test_start_click_goes_straight_to_opening(self) -> None:
        """金色挑战：点击开局，目标态直接是 SELECT_OPENING_SKILLS。"""
        invite = InviteConfig(start_verify_frames=12)
        task = _make_invite_task(invite=invite)
        task.ctx.current_state = State.INVITE_START
        action, to_state = task.decide_action(
            _observation(self._ready_obs()), _window_ctx()
        )
        assert action.tag == "invite_start_game"
        assert to_state == State.SELECT_OPENING_SKILLS
        # 慢过渡验证预算：游戏点击到切屏实测 ~2.4s，全局 5 帧不够
        assert action.verify_frames == 12

    def test_empty_slot_means_helper_left(self) -> None:
        """邀请文字重现（槽位回空）= 帮助者退出，回 INVITE_WAITING 重过滤。"""
        task = _make_invite_task()
        task.ctx.current_state = State.INVITE_START
        obs = _observation({"invite_text_visible": True})
        action, to_state = task.decide_action(obs, _window_ctx())
        assert action.tag == "invite_helper_left"
        assert to_state == State.INVITE_WAITING

    def test_unknown_frame_waits_without_transition(self) -> None:
        """金灰按钮和邀请文字都没有（加载页/未知过渡）：原地等待不跳状态。

        2026-09-26 实机回归：黑屏加载页被按"邀请文字缺失默认可见"判成
        帮助者退出，整个邀请流程脱轨。
        """
        task = _make_invite_task()
        task.ctx.current_state = State.INVITE_START
        action, to_state = task.decide_action(_observation(), _window_ctx())
        assert action.tag == "invite_start_ambiguous"
        assert to_state == State.INVITE_START

    def test_start_failure_allows_retry(self) -> None:
        """开始点击验证失败：无内部标记需要撤销，金色仍在即可重新点击。"""
        from wlxq_bot.models import Action

        task = _make_invite_task()
        task.ctx.current_state = State.INVITE_START
        task.on_action_failed(Action(kind="click", target=(0, 0), tag="invite_start_game"))
        action, to_state = task.decide_action(
            _observation(self._ready_obs()), _window_ctx()
        )
        assert action.tag == "invite_start_game"
        assert to_state == State.SELECT_OPENING_SKILLS

class TestInviteOpeningEscape:
    """挑战未生效/被拒回到组队大厅的自愈闭环（实机 2026-09-25 死等开局修复）。

    实机证据：帮助者进房后橙色【挑战】模板立即命中（与是否准备无关），
    点开始→确认页→挑战 可能被游戏拒绝（帮助者未准备），画面回到组队大厅；
    原实现滞留在 SELECT_OPENING_SKILLS 死等开局，帮助者后来点了准备也无人
    再点开始。
    """

    def _opening_task(self, invite: InviteConfig | None = None, clock=None) -> CoopTask:
        task = _make_invite_task(invite=invite, clock=clock)
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._opening_loaded = True  # 跳过 3s 载入等待门，直达开局识别循环
        return task

    @staticmethod
    def _ready_obs() -> dict:
        return {
            "invite_start_ready_visible": True,
            "invite_start_ready_match": MatchResult(
                position=(500, 1500),
                confidence=0.95,
                template_name="zhun_bei",
                roi_name="invite_start_button_area",
            ),
        }

    def test_back_at_lobby_reconfirms_then_reclicks_start(self) -> None:
        """回大厅（按钮可见）连续 2 帧确认 → 重新点开始，直下选技能态。"""
        task = self._opening_task()
        obs = _observation(self._ready_obs())
        first, to1 = task.decide_action(obs, _window_ctx())
        assert first.tag == "invite_back_lobby_confirm"
        assert to1 == State.SELECT_OPENING_SKILLS
        second, to2 = task.decide_action(obs, _window_ctx())
        assert second.tag == "invite_start_game"
        assert to2 == State.SELECT_OPENING_SKILLS
        assert task._match_started_at is None

    def test_single_frame_hit_keeps_waiting(self) -> None:
        """单帧误匹配不放弃对局：继续开局等待，不迁移状态。"""
        task = self._opening_task()
        action, to_state = task.decide_action(_observation(self._ready_obs()), _window_ctx())
        assert to_state == State.SELECT_OPENING_SKILLS
        assert action.tag == "invite_back_lobby_confirm"
        action2, to_state2 = task.decide_action(_observation(), _window_ctx())
        assert to_state2 == State.SELECT_OPENING_SKILLS
        assert action2.tag == "opening_interface_wait"

    def test_budget_exhausted_kicks_helper(self) -> None:
        """合格后 ready_timeout_seconds 仍开不了局 → 请离重邀，不再无限重试。"""
        clock = _FakeClock()
        invite = InviteConfig(ready_timeout_seconds=60.0)
        task = self._opening_task(invite=invite, clock=clock)
        task._invite_wait_since = 0.0
        clock.now = 61.0
        obs = _observation(self._ready_obs())
        action, to_state = task.decide_action(obs, _window_ctx())
        assert action.tag == "invite_kick_helper"
        assert to_state == State.INVITE_COOLDOWN
        assert task._invite_kick_count == 1

    def test_zero_timeout_never_kicks(self) -> None:
        """ready_timeout_seconds=0：一直等，预算永不耗尽。"""
        clock = _FakeClock()
        invite = InviteConfig(ready_timeout_seconds=0)
        task = self._opening_task(invite=invite, clock=clock)
        task._invite_wait_since = 0.0
        clock.now = 99999.0
        obs = _observation(self._ready_obs())
        task.decide_action(obs, _window_ctx())
        action, to_state = task.decide_action(obs, _window_ctx())
        assert action.tag == "invite_start_game"
        assert to_state == State.SELECT_OPENING_SKILLS

    def test_kick_limit_reached_stops_conservatively(self) -> None:
        """连续请离达上限：决策返回 None 保守停止。"""
        clock = _FakeClock()
        invite = InviteConfig(ready_timeout_seconds=60.0)
        task = self._opening_task(invite=invite, clock=clock)
        task._invite_wait_since = 0.0
        task._invite_kick_count = invite.max_consecutive_kicks
        clock.now = 61.0
        assert task.decide_action(_observation(self._ready_obs()), _window_ctx()) is None

    def test_invite_start_budget_kicks_before_clicking_start(self) -> None:
        """INVITE_START 态预算耗尽：不再反复点开始，直接请离。"""
        clock = _FakeClock()
        invite = InviteConfig(ready_timeout_seconds=60.0)
        task = _make_invite_task(invite=invite, clock=clock)
        task.ctx.current_state = State.INVITE_START
        task._invite_wait_since = 0.0
        clock.now = 61.0
        action, to_state = task.decide_action(_observation(self._ready_obs()), _window_ctx())
        assert action.tag == "invite_kick_helper"
        assert to_state == State.INVITE_COOLDOWN

    def test_start_click_verifies_match_timeout_backstop(self) -> None:
        """开始点击验证通过 → 起 match_start_timeout 兜底（invite 模式此前无兜底死等）。"""
        from wlxq_bot.models import Action

        task = _make_invite_task()
        assert task._match_started_at is None
        task.on_action_verified(
            Action(kind="click", target=(0, 0), tag="invite_start_game"),
            State.SELECT_OPENING_SKILLS,
        )
        assert task._match_started_at is not None


class TestInviteReinviteCadence:
    """两层重邀节奏（2026-09-26 用户定案）：全量链每 60s，轻量短链每 15s。

    轻量链 = 邀请 → 一键邀请（弹窗层）→ 关弹窗，不走好友/公会招募；
    轻量链不重置全量计时（全量仍自上次全量邀请起算 60s）。
    """

    def _waiting_task(self, clock):
        task = _make_invite_task(clock=clock)
        task.ctx.current_state = State.INVITE_WAITING
        task._invite_wait_since = 0.0
        task._invite_full_since = 0.0
        return task

    def test_light_reinvite_due_at_15s(self):
        clock = _FakeClock()
        task = self._waiting_task(clock)
        clock.now = 15.0
        action, to_state = task.decide_action(
            _observation({"invite_text_visible": True}), _window_ctx()
        )
        assert action.tag == "invite_reinvite_light"
        assert to_state == State.INVITE_CLICK
        assert task._invite_light_walk is True
        assert task._invite_step == 1  # 跳过开始挑战，从邀请按钮起步

    def test_light_chain_walks_three_steps(self):
        clock = _FakeClock()
        task = self._waiting_task(clock)
        clock.now = 20.0
        task.decide_action(_observation({"invite_text_visible": True}), _window_ctx())
        # 模拟 runner：轻量重发动作提交后状态切到 INVITE_CLICK
        task.ctx.current_state = State.INVITE_CLICK
        a2, _ = task.decide_action(_observation(), _window_ctx())
        assert a2.tag == "invite_step_1"  # 邀请按钮
        a3, _ = task.decide_action(_observation(), _window_ctx())
        assert a3.tag == "invite_step_2"  # 一键邀请（弹窗层）
        a4, s4 = task.decide_action(_observation(), _window_ctx())
        assert a4.tag == "invite_wait_entry"  # × 未命中 → 跳过关弹窗 → 完成
        assert s4 == State.INVITE_WAITING
        assert task._invite_full_since == 0.0  # 轻量链不重置全量计时

    def test_full_reinvite_due_at_60s(self):
        clock = _FakeClock()
        task = self._waiting_task(clock)
        clock.now = 60.0
        action, to_state = task.decide_action(
            _observation({"invite_text_visible": True}), _window_ctx()
        )
        assert action.tag == "invite_reinvite"
        assert to_state == State.INVITE_CLICK
        assert task._invite_light_walk is False
        assert task._invite_step == 0  # 全量链从头走（含开始挑战）

    def test_full_wins_when_both_due(self):
        """20s 轻量被全量优先级覆盖：60s 到点走全量而非轻量。"""
        clock = _FakeClock()
        task = self._waiting_task(clock)
        clock.now = 60.0
        # 先走一次轻量（t=20），再推进到 t=60：全量与轻量同时到期
        clock.now = 20.0
        task.decide_action(_observation({"invite_text_visible": True}), _window_ctx())
        task._invite_wait_since = 20.0
        clock.now = 60.0
        action, _ = task.decide_action(
            _observation({"invite_text_visible": True}), _window_ctx()
        )
        assert action.tag == "invite_reinvite"

    def test_under_15s_keeps_waiting(self):
        clock = _FakeClock()
        task = self._waiting_task(clock)
        clock.now = 10.0
        action, to_state = task.decide_action(
            _observation({"invite_text_visible": True}), _window_ctx()
        )
        assert action.tag == "invite_waiting"
        assert to_state == State.INVITE_WAITING

class TestInviteCooldownReturn:
    """请离后回流等待（2026-09-28 用户定案）：房间回空即回等待态。

    60s 全量大轮回锚在上次全量邀请（_invite_full_since），不因请离重置；
    回空后继续检查进房 + 轻量邀请；kick_cooldown_seconds 内未回空才重走
    邀请链兜底（截止自冷却首帧起算）。非房间画面（锚点缺失）冻结回空判定。
    测试模拟 runner：decide_action 返回目标态后手动提交 ctx.current_state。
    """

    def _cooldown_task(self, invite=None, clock=None) -> CoopTask:
        task = _make_invite_task(invite=invite, clock=clock)
        task.ctx.current_state = State.INVITE_COOLDOWN
        task._invite_full_since = 0.0  # 全量邀请 T=0 发出（大轮回锚点）
        return task

    @staticmethod
    def _room_obs(text_visible: bool) -> dict:
        return {
            "invite_start_ready_visible": True,  # 房间屏锚点（灰色未准备按钮）
            "invite_text_visible": text_visible,
        }

    def test_cooldown_enables_invite_observation(self):
        """冷却态走 invite_waiting 观察模式，并带房间锚点/× 识别 flags。"""
        from wlxq_bot.perception.coop import _INTERFACE_FLAGS_BY_STATE

        task = _make_invite_task()
        assert task.observation_mode() is None  # INVITE_CLICK 无专项识别
        task.ctx.current_state = State.INVITE_COOLDOWN
        assert task.observation_mode() == "invite_waiting"
        flags = _INTERFACE_FLAGS_BY_STATE[State.INVITE_COOLDOWN]
        assert {"close_button", "invite_start_ready", "invite_start_not_ready"} <= flags

    def test_room_empty_returns_to_waiting_without_reinvite_chain(self):
        """踢生效（文字重现）→ 立即回 INVITE_WAITING，不重走点击链。"""
        clock = _FakeClock()
        task = self._cooldown_task(clock=clock)
        clock.now = 26.0  # 踢后下一帧房间已回空
        action, to_state = task.decide_action(
            _observation(self._room_obs(True)), _window_ctx()
        )
        assert action.tag == "invite_cooldown_room_empty"
        assert to_state == State.INVITE_WAITING
        assert task._invite_cooldown_until is None
        assert task._invite_full_since == 0.0  # 大轮回未重置
        # 回等待态首帧：轻量/全量都未到期 → 普通等待（不是 invite_reinvite）
        task.ctx.current_state = State.INVITE_WAITING
        clock.now = 27.0
        action2, to_state2 = task.decide_action(
            _observation(self._room_obs(True)), _window_ctx()
        )
        assert action2.tag == "invite_waiting"
        assert to_state2 == State.INVITE_WAITING

    def test_full_cycle_due_after_kick_not_from_kick(self):
        """T=24 进房、T=26 踢：全量重邀仍在 T=60 到期（大轮回），而非踢后 +60s。"""
        clock = _FakeClock()
        task = self._cooldown_task(clock=clock)
        clock.now = 26.0
        _, to_state = task.decide_action(
            _observation(self._room_obs(True)), _window_ctx()
        )
        assert to_state == State.INVITE_WAITING
        task.ctx.current_state = State.INVITE_WAITING
        clock.now = 60.0  # 自 T=0 全量起算到期；若自踢时刻起算要到 T=86
        action, to_state2 = task.decide_action(
            _observation(self._room_obs(True)), _window_ctx()
        )
        assert action.tag == "invite_reinvite"
        assert to_state2 == State.INVITE_CLICK

    def test_light_reinvite_resumes_after_return(self):
        """回空后轻量邀请按节奏恢复：回等待态 15s 后发轻量短链。"""
        clock = _FakeClock()
        task = self._cooldown_task(clock=clock)
        clock.now = 26.0
        task.decide_action(_observation(self._room_obs(True)), _window_ctx())
        task.ctx.current_state = State.INVITE_WAITING
        clock.now = 26.5  # 等待态首帧重置轻量计时（wait_since≈26.5）
        first, _ = task.decide_action(_observation(self._room_obs(True)), _window_ctx())
        assert first.tag == "invite_waiting"
        clock.now = 42.0  # 距计时重置 15s
        action, to_state = task.decide_action(
            _observation(self._room_obs(True)), _window_ctx()
        )
        assert action.tag == "invite_reinvite_light"
        assert to_state == State.INVITE_CLICK

    def test_join_during_return_is_evaluated_not_reinvited(self):
        """回等待态后有人进房（文字消失+锚点在）→ 打开帮助者信息层评估。"""
        clock = _FakeClock()
        task = self._cooldown_task(clock=clock)
        clock.now = 26.0
        task.decide_action(_observation(self._room_obs(True)), _window_ctx())
        task.ctx.current_state = State.INVITE_WAITING
        clock.now = 40.0
        action, to_state = task.decide_action(
            _observation(self._room_obs(False)), _window_ctx()
        )
        assert action.tag == "invite_open_helper_info"
        assert to_state == State.INVITE_WAITING

    def test_timeout_without_empty_rewalks_chain(self):
        """kick_cooldown_seconds 内文字未重现 → 兜底回 INVITE_CLICK 重走全量链。"""
        clock = _FakeClock()
        invite = InviteConfig(kick_cooldown_seconds=60.0)
        task = self._cooldown_task(invite=invite, clock=clock)
        clock.now = 26.0  # 冷却首帧起算截止 86s
        action, to_state = task.decide_action(
            _observation(self._room_obs(False)), _window_ctx()
        )
        assert action.tag == "invite_cooldown"
        assert to_state == State.INVITE_COOLDOWN
        clock.now = 86.1
        action2, to_state2 = task.decide_action(
            _observation(self._room_obs(False)), _window_ctx()
        )
        assert action2.tag == "invite_cooldown_done"
        assert to_state2 == State.INVITE_CLICK
        assert task._invite_step == 0  # 全量链从头走

    def test_non_room_screen_freezes_empty_check(self):
        """锚点缺失（加载页/过渡）不判回空：即使文字误报可见也不回流。"""
        clock = _FakeClock()
        task = self._cooldown_task(clock=clock)
        clock.now = 30.0
        action, to_state = task.decide_action(
            _observation({"invite_text_visible": True}), _window_ctx()
        )
        assert action.tag == "invite_cooldown"
        assert to_state == State.INVITE_COOLDOWN
        assert task._invite_cooldown_until is not None  # 未消费冷却
