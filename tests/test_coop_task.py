"""CoopTask 状态机决策单元测试。

只输入 Observation 断言 State/Action，不启动真实游戏窗口。
覆盖 determine_state 优先级和 decide_action 召唤/合成/停止决策。
"""

from __future__ import annotations

import pytest

from wlxq_bot.config import Hotspot, MainCProfile, ModeProfile, RoiConfig, RunConfig
from wlxq_bot.models import (
    Action,
    BoardCapacity,
    BoardHero,
    BoardSnapshot,
    CoopRole,
    DifficultyCandidate,
    MatchResult,
    Observation,
    SkillCandidate,
    State,
    WindowContext,
)
from wlxq_bot.perception.locator import roi_column_centers
from wlxq_bot.perception.mode_tab import ModeTabRow
from wlxq_bot.tasks.base import TaskContext
from wlxq_bot.tasks.coop import (
    _EMPTY_BOARD_RETRY,
    _ENTRY_RECLICK_MISSES,
    CoopTask,
    _RecruitStep,
)

# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------

# 变身按钮 locator 测试夹具（与 configs/tasks.yaml 的 transform_button 同构）
TRANSFORM_LOCATOR = {
    "strategy": "template",
    "template": "buttons/bian_shen.png",
    "threshold": 0.85,
    "action_point": {"source": "match_center"},
    "search_dx_ratio": 0.11,
    "search_up_ratio": 0.093,
    "search_down_ratio": 0.015,
}


def make_window_ctx(cw: int = 923, ch: int = 1723) -> WindowContext:
    return WindowContext(
        window_handle=1,
        client_rect_screen=(0, 0, cw, ch),
        client_size=(cw, ch),
        dpi=96,
        monitor_id="primary",
        is_foreground=True,
        is_minimized=False,
        captured_at=0.0,
        frame_id=1,
    )


def make_task(
    main_c: str = "assault",
    max_rounds: int = 10,
    target_star: int = 2,
    minimum_summons: int = 3,
    batch_enabled: bool = False,
    hotspots: dict[str, Hotspot] | None = None,
    coop_difficulties: list[int] | None = None,
    difficulty_selection_only: bool = False,
    skip_difficulty_selection: bool = False,
    include_skill_candidate_roi: bool = True,
    main_c_profile: MainCProfile | None = None,
    mode_profile: ModeProfile | None = None,
    pet_difficulty_targets: list[str] | None = None,
    transform_button_locator: dict | None = None,
) -> CoopTask:
    ctx = TaskContext(main_c=main_c, max_rounds=max_rounds)
    run_cfg = RunConfig(
        max_rounds=max_rounds,
        minimum_summon_count_before_skills=minimum_summons,
        initial_board_capacity=max(7, minimum_summons),
        target_star_level=target_star,
        batch_summon_enabled=batch_enabled,
        batch_merge_enabled=batch_enabled,
    )
    spots = (
        hotspots
        if hotspots is not None
        else {
            "territory_nav": Hotspot(x_ratio=0.11, y_ratio=0.21),
            "open_mode_select": Hotspot(x_ratio=0.21, y_ratio=0.31),
            "open_difficulty_dialog": Hotspot(x_ratio=0.3, y_ratio=0.4),
            "difficulty_scroll_start": Hotspot(x_ratio=0.5, y_ratio=0.3),
            "difficulty_scroll_end": Hotspot(x_ratio=0.5, y_ratio=0.7),
            "close_difficulty_dialog": Hotspot(x_ratio=0.4, y_ratio=0.5),
            "coop_chat": Hotspot(x_ratio=0.5, y_ratio=0.6),
            "add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96),
            "join_coop": Hotspot(x_ratio=0.6, y_ratio=0.75),
            "select_skill": Hotspot(x_ratio=0.2, y_ratio=0.9),
            "mouse_park": Hotspot(x_ratio=0.5, y_ratio=0.06),
            "like": Hotspot(x_ratio=0.3, y_ratio=0.8),
            "claim_chest": Hotspot(x_ratio=0.5, y_ratio=0.8),
        }
    )
    return CoopTask(
        ctx,
        run_cfg,
        CoopRole.HELPER,
        spots,
        coop_difficulties=coop_difficulties or [2, 1],
        skill_candidate_roi=(
            RoiConfig(
                x_ratio=0.1,
                y_ratio=0.2,
                width_ratio=0.6,
                height_ratio=0.4,
            )
            if include_skill_candidate_roi
            else None
        ),
        difficulty_selection_only=difficulty_selection_only,
        skip_difficulty_selection=skip_difficulty_selection,
        main_c_profile=main_c_profile,
        mode_profile=mode_profile or default_entry_mode_profile(),
        pet_difficulty_targets=pet_difficulty_targets,
        transform_button_locator=transform_button_locator,
    )


def make_hero(
    hero_type: str = "assault",
    star: int = 1,
    pos: tuple[int, int] = (100, 100),
    cell_name: str = "",
) -> BoardHero:
    return BoardHero(
        hero_type=hero_type,
        star_level=star,
        position=pos,
        confidence=0.9,
        cell_name=cell_name,
    )


def make_mode_profile(
    mode_tab_hotspot: str | None = "recruit_coop_tab",
    mode_tab_text: str | None = None,
    mode_chat_hotspot: str = "coop_chat",
    difficulty_kind: str = "template",
    difficulty_targets: list[str] | None = None,
    like_chest: bool = True,
    ready_button: str = "ready_button",
    leave_team: str = "coop_leave_team",
    return_button: str = "return_button",
    mode_card_text: str | None = "合作",
    challenge_text: str | None = "开始挑战",
    weekly_buff_text: str | None = None,
) -> ModeProfile:
    return ModeProfile(
        display_name="测试模式",
        mode_chat_hotspot=mode_chat_hotspot,
        entry={
            "mode_tab_hotspot": mode_tab_hotspot,
            "mode_tab_text": mode_tab_text,
            "mode_card_text": mode_card_text,
            "challenge_text": challenge_text,
            "weekly_buff_text": weekly_buff_text,
            "difficulty": {"kind": difficulty_kind, "targets": difficulty_targets or []},
        },
        match={"ready_button": ready_button, "leave_team": leave_team},
        settlement={"like_chest": like_chest, "return_button": return_button},
    )


def make_board(heroes: list[BoardHero]) -> BoardSnapshot:
    return BoardSnapshot(
        frame_id=1,
        heroes=heroes,
        capacity=BoardCapacity(total_slots=12, occupied=len(heroes)),
    )


def difficulty_observation(*levels: int, frame_id: int = 1) -> Observation:
    """难度弹窗打开状态：可见难度候选 + 【合作模式】标识（he_zuo_mo_shi）命中。"""
    return Observation(
        frame_id=frame_id,
        raw_data={"difficulty_dialog_visible": True},
        difficulty_candidates=[
            DifficultyCandidate(
                level=level,
                position=(300, 200 + index * 100),
                confidence=0.9,
            )
            for index, level in enumerate(levels)
        ],
    )


def home_observation(frame_id: int = 1) -> Observation:
    return Observation(frame_id=frame_id, raw_data={"home_page_visible": True})


def advance_confirm_difficulty_open(task: CoopTask, *levels: int, frame_id: int = 4) -> None:
    """驱动打开确认阶段：1 帧任意画面（settle 等待）→ 连续 2 帧标题可见（命中）
    → 1 帧确认打开。levels 为弹窗可见时同屏的难度候选（默认 2,1）。"""
    visible_levels = levels or (2, 1)
    for observation in [Observation(frame_id=frame_id)] + [
        difficulty_observation(*visible_levels, frame_id=frame_id + 1 + index) for index in range(3)
    ]:
        decision = task.decide_action(observation, make_window_ctx())
        assert decision is not None
        task.on_action_verified(*decision)


def default_entry_mode_profile() -> ModeProfile:
    """make_task 未显式传档案时的默认值：领地入口链可驱动，其余行为等价
    旧 mode=None（coop_chat 兜底、结算点赞宝箱、无页签步骤）。"""
    return ModeProfile(
        display_name="测试模式",
        mode_chat_hotspot="coop_chat",
        entry={
            "mode_card_text": "合作",
            "challenge_text": "开始挑战",
        },
        match={"ready_button": "ready_button", "leave_team": "coop_leave_team"},
        settlement={"like_chest": True, "return_button": "return_button"},
    )


def territory_entry_observations(frame_id: int = 1) -> list[Observation]:
    """领地入口链的逐帧观察（6 帧各推进一步）：领地点击→返回确认→进入
    按钮点击→弹窗标题确认→选卡点击→挑战页确认。除导航文字逐帧变化外，
    其余识别数据每帧都在（各步骤只读取自己需要的那一份）。"""
    base = {
        "mode_select_visible": True,
        "mode_card_rows": [ModeTabRow("合作模式", 0.9, (250, 1174))],
        "challenge_text": "开始挑战",
    }

    def obs(index: int, nav: str) -> Observation:
        raw = {**base, "territory_nav_text": nav} if nav else dict(base)
        return Observation(frame_id=frame_id + index, raw_data=raw)

    return [obs(0, "领地"), obs(1, "返回"), obs(2, ""), obs(3, ""), obs(4, ""), obs(5, "")]


def advance_territory_entry_to_recruit(task: CoopTask, frame_id: int = 1) -> None:
    """驱动领地入口链到 SELECT_MODE_TAB（挑战页聊天点击已验证）。"""
    for observation in territory_entry_observations(frame_id):
        decision = task.decide_action(observation, make_window_ctx())
        assert decision is not None
        task.on_action_verified(*decision)
    decision = task.decide_action(Observation(frame_id=frame_id + 6), make_window_ctx())
    assert decision is not None and decision[0].tag == "open_challenge_chat"
    task.on_action_verified(*decision)


def advance_entry_to_difficulty_dialog(task: CoopTask, frame_id: int = 1) -> None:
    """驱动领地入口链到 CONFIRM_DIFFICULTY_OPEN（打开难度弹窗点击已验证）。"""
    advance_territory_entry_to_recruit(task, frame_id=frame_id)
    decision = task.decide_action(Observation(frame_id=frame_id + 7), make_window_ctx())
    assert decision is not None and decision[0].tag == "open_difficulty_dialog"
    task.on_action_verified(*decision)


def advance_initial_entry_to_join(task: CoopTask) -> None:
    advance_territory_entry_to_recruit(task, frame_id=1)
    # 页签（默认档案未配置 → 跳过）后打开难度弹窗
    decision = task.decide_action(Observation(frame_id=8), make_window_ctx())
    assert decision is not None and decision[0].tag == "open_difficulty_dialog"
    task.on_action_verified(*decision)
    # 打开确认：settle → 连续 2 次命中标题 → 确认打开进入勾选
    advance_confirm_difficulty_open(task, 2, 1, frame_id=8)
    # 勾选 1、2 后进入关闭步骤
    for frame_id in (10, 11):
        decision = task.decide_action(
            difficulty_observation(2, 1, frame_id=frame_id), make_window_ctx()
        )
        assert decision is not None
        task.on_action_verified(*decision)
    # 关闭步骤：弹窗仍打开 → 点击关闭；之后连续 2 次复核不可见 → 确认关闭
    close_click = task.decide_action(difficulty_observation(2, 1, frame_id=12), make_window_ctx())
    assert close_click is not None
    task.on_action_verified(*close_click)
    for frame_id in range(13, 16):
        decision = task.decide_action(Observation(frame_id=frame_id), make_window_ctx())
        assert decision is not None
        task.on_action_verified(*decision)


# ---------------------------------------------------------------------------
# determine_state
# ---------------------------------------------------------------------------


class TestDetermineState:
    def test_completed_when_round_reached(self):
        task = make_task(max_rounds=2)
        task.ctx.round_count = 2
        obs = Observation(frame_id=1)
        assert task.determine_state(obs) == State.COMPLETED

    def test_window_invalid_flag(self):
        task = make_task()
        obs = Observation(frame_id=1, raw_data={"window_invalid": True})
        assert task.determine_state(obs) == State.WINDOW_INVALID

    def test_ready_button_moves_find_coop_to_enter_match(self):
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        obs = Observation(frame_id=1, raw_data={"ready_button_visible": True})
        assert task.determine_state(obs) == State.ENTER_MATCH

    def test_cultivation_inertia_maintains_build(self):
        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        obs = Observation(frame_id=1)  # 无任何标志
        assert task.determine_state(obs) == State.BUILD_MAIN_C

    def test_cultivation_inertia_maintains_select_skills(self):
        task = make_task()
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        obs = Observation(frame_id=1)
        assert task.determine_state(obs) == State.SELECT_MAIN_C_SKILLS

    def test_unknown_when_no_match(self):
        task = make_task()
        task.ctx.current_state = State.UNKNOWN
        obs = Observation(frame_id=1)
        assert task.determine_state(obs) == State.UNKNOWN

    def test_return_button_marks_round_ended(self):
        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        obs = Observation(frame_id=1, raw_data={"return_button_visible": True})
        assert task.determine_state(obs) == State.HANDLE_RESULT


# ---------------------------------------------------------------------------
# decide_action - FIND_COOP
# ---------------------------------------------------------------------------


class TestFindCoopAction:
    def test_initial_recruit_entry_uses_confirmed_business_order(self):
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        tags = []
        entry_observations = territory_entry_observations(frame_id=1) + [
            Observation(frame_id=7),  # 挑战页聊天 → SELECT_MODE_TAB → 打开难度弹窗
            Observation(frame_id=8),  # 打开确认：固定等待（settle）
            difficulty_observation(2, 1, frame_id=9),  # 命中 1
            difficulty_observation(2, 1, frame_id=10),  # 命中 2
            difficulty_observation(2, 1, frame_id=11),  # 确认打开 → 进入勾选
            difficulty_observation(2, 1, frame_id=12),
            difficulty_observation(2, 1, frame_id=13),
            difficulty_observation(2, 1, frame_id=14),
            # 弹窗仍打开 → 点击关闭；之后连续 2 次复核不可见 → 确认关闭
            difficulty_observation(2, 1, frame_id=15),
            Observation(frame_id=16),
            Observation(frame_id=17),
            Observation(frame_id=18),
        ]
        for observation in entry_observations:
            decision = task.decide_action(observation, make_window_ctx())
            assert decision is not None
            action, to_state = decision
            tags.append(action.tag)
            assert action.verification == "immediate"
            assert to_state == State.FIND_COOP
            task.on_action_verified(action, to_state)

        assert tags == [
            "open_territory",
            "territory_open_confirmed",
            "open_mode_select",
            "mode_select_open_confirmed",
            "mode_card_selected",
            "challenge_page_confirmed",
            "open_challenge_chat",
            "open_difficulty_dialog",
            "difficulty_open_settle",
            "difficulty_open_hit",
            "difficulty_open_hit",
            "difficulty_open_confirmed",
            "select_difficulty:1",
            "select_difficulty:2",
            "close_difficulty_dialog",
            "difficulty_close_miss_check",
            "difficulty_close_miss_check",
            "difficulty_dialog_closed",
        ]
        # 进入 JOIN_COOP 后发出抢合作信号动作（由 Runner 双线程协调器执行）
        grab_decision = task.decide_action(Observation(frame_id=19), make_window_ctx())
        assert grab_decision is not None
        assert grab_decision[0].kind == "grab_coop"
        assert grab_decision[0].tag == "find_coop_grab"
        assert grab_decision[1] == State.FIND_COOP

    def test_skip_difficulty_selection_jumps_from_recruit_to_join(self):
        task = make_task(skip_difficulty_selection=True)
        task.ctx.current_state = State.FIND_COOP
        tags = []
        # 跳过模式：打开确认通过后直接进关闭步骤，不进入 SELECT_DIFFICULTIES
        for observation in territory_entry_observations(frame_id=1) + [
            Observation(frame_id=7),  # 挑战页聊天 → SELECT_MODE_TAB → 打开难度弹窗
            Observation(frame_id=8),  # settle
            difficulty_observation(10, frame_id=9),
            difficulty_observation(10, frame_id=10),
            difficulty_observation(10, frame_id=11),  # 确认打开
            difficulty_observation(10, frame_id=12),
            # 弹窗仍打开 → 点击关闭；之后连续 2 次复核不可见 → 确认关闭
            difficulty_observation(10, frame_id=13),
            Observation(frame_id=14),
            Observation(frame_id=15),
            Observation(frame_id=16),
        ]:
            decision = task.decide_action(observation, make_window_ctx())
            assert decision is not None
            action, to_state = decision
            tags.append(action.tag)
            assert to_state == State.FIND_COOP
            task.on_action_verified(action, to_state)

        # 只跳过勾选难度等级；弹窗仍开/关一次刷新最新合作邀请
        assert tags == [
            "open_territory",
            "territory_open_confirmed",
            "open_mode_select",
            "mode_select_open_confirmed",
            "mode_card_selected",
            "challenge_page_confirmed",
            "open_challenge_chat",
            "open_difficulty_dialog",
            "difficulty_open_settle",
            "difficulty_open_hit",
            "difficulty_open_hit",
            "difficulty_open_confirmed",
            "close_difficulty_dialog",
            "difficulty_close_miss_check",
            "difficulty_close_miss_check",
            "difficulty_dialog_closed",
        ]
        grab_decision = task.decide_action(Observation(frame_id=17), make_window_ctx())
        assert grab_decision is not None
        assert grab_decision[0].kind == "grab_coop"
        assert grab_decision[0].tag == "find_coop_grab"
        assert grab_decision[1] == State.FIND_COOP

    def test_skip_difficulty_selection_keeps_next_round_refresh(self):
        task = make_task(max_rounds=2, skip_difficulty_selection=True)
        task.ctx.current_state = State.CHECK_ROUND_LIMIT
        match = MatchResult("buttons/fan_hui.png", (500, 1000), 0.9)
        before = Observation(
            frame_id=1,
            raw_data={"return_button_visible": True, "return_button_match": match},
        )
        decision = task.decide_action(before, make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert to_state == State.FIND_COOP
        task.on_action_verified(action, to_state)
        task.ctx.round_count = 1
        task.ctx.current_state = to_state

        # 后续局的邀请刷新仍走「合作页聊天 → 开难度弹窗 → 关难度弹窗」
        chat_decision = task.decide_action(Observation(frame_id=2), make_window_ctx())
        assert chat_decision is not None
        assert chat_decision[0].tag == "open_coop_chat"
        task.on_action_verified(*chat_decision)

        open_refresh_decision = task.decide_action(Observation(frame_id=3), make_window_ctx())
        assert open_refresh_decision is not None
        assert open_refresh_decision[0].tag == "open_refresh_difficulty_dialog"
        task.on_action_verified(*open_refresh_decision)

        # 打开确认通过后进入刷新关闭步骤（同样先确认弹窗真的打开）
        advance_confirm_difficulty_open(task, 10, frame_id=4)
        assert task._recruit_step is _RecruitStep.CLOSE_REFRESH_DIFFICULTY_DIALOG

        # 弹窗仍打开 → 点击关闭；连续 2 次复核不可见 → 确认关闭
        close_refresh_decision = task.decide_action(
            difficulty_observation(10, frame_id=10), make_window_ctx()
        )
        assert close_refresh_decision is not None
        assert close_refresh_decision[0].tag == "close_refresh_difficulty_dialog"
        task.on_action_verified(*close_refresh_decision)

        for frame_id in range(11, 13):
            closed_step = task.decide_action(Observation(frame_id=frame_id), make_window_ctx())
            assert closed_step is not None
            assert closed_step[0].tag == "difficulty_close_miss_check"
            task.on_action_verified(*closed_step)

        closed_step = task.decide_action(Observation(frame_id=13), make_window_ctx())
        assert closed_step is not None
        assert closed_step[0].tag == "refresh_dialog_closed"
        task.on_action_verified(*closed_step)

        join_decision = task.decide_action(Observation(frame_id=14), make_window_ctx())
        assert join_decision is not None
        assert join_decision[0].kind == "grab_coop"

    def test_initial_recruit_requests_territory_nav_observation(self):
        """领地入口第一步不做首页校验：直接 OCR 底部导航领地格。"""
        task = make_task()
        assert task.observation_mode() == "territory_nav"

    def test_join_step_emits_grab_coop_signal(self):
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        advance_initial_entry_to_join(task)

        decision = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "grab_coop"
        assert decision[0].tag == "find_coop_grab"

    def test_missing_entry_hotspot_stops_without_skipping_to_join(self):
        # 领地确认后缺 open_mode_select hotspot：无可用动作 → decide 返回 None
        # （调度层按保守停止计数，绝不跳过入口直接抢）
        task = make_task(
            hotspots={
                "territory_nav": Hotspot(x_ratio=0.11, y_ratio=0.21),
                "join_coop": Hotspot(x_ratio=0.6, y_ratio=0.75),
            }
        )
        task.ctx.current_state = State.FIND_COOP
        first, second, *_ = territory_entry_observations()
        d1 = task.decide_action(first, make_window_ctx())
        assert d1 is not None and d1[0].tag == "open_territory"
        task.on_action_verified(*d1)
        d2 = task.decide_action(second, make_window_ctx())
        assert d2 is not None and d2[0].tag == "territory_open_confirmed"
        task.on_action_verified(*d2)
        assert task._recruit_step == _RecruitStep.OPEN_MODE_SELECT
        assert task.decide_action(Observation(frame_id=3), make_window_ctx()) is None

    def test_ready_button_is_clicked_by_match_center(self):
        task = make_task()
        task.ctx.current_state = State.ENTER_MATCH
        match = MatchResult("buttons/zhun_bei.png", (500, 900), 0.9)
        obs = Observation(
            frame_id=1,
            raw_data={"ready_button_visible": True, "ready_button_match": match},
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].target == (500, 900)
        assert decision[0].verification == "next_frame"
        assert decision[1] == State.SELECT_OPENING_SKILLS


# ---------------------------------------------------------------------------
# decide_action - 击杀奖励弹窗（tan_chuang）
# ---------------------------------------------------------------------------


class TestClosePopup:
    def test_popup_during_match_is_closed(self):
        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        match = MatchResult("buttons/tan_chuang.png", (400, 1100), 0.9)
        obs = Observation(
            frame_id=1,
            raw_data={"tan_chuang_visible": True, "tan_chuang_match": match},
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert action.tag == "close_popup"
        assert action.target == (400, 1100)
        assert action.verification == "next_frame"
        # 关闭后回到原状态，不改变流程
        assert to_state == State.BUILD_MAIN_C

    def test_popup_pre_empts_skill_selection(self):
        task = make_task()
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        match = MatchResult("buttons/tan_chuang.png", (400, 1100), 0.9)
        obs = Observation(
            frame_id=1,
            raw_data={"tan_chuang_visible": True, "tan_chuang_match": match},
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "close_popup"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS

    def test_popup_not_closed_during_find_coop(self):
        # FIND_COOP 不在 _MATCH_STATES，弹窗点击只在游戏内生效
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        obs = Observation(
            frame_id=1,
            raw_data={"tan_chuang_visible": True, "home_page_visible": True},
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag != "close_popup"

    def test_close_popup_verified_when_disappeared(self):
        task = make_task()
        action = Action(kind="click", tag="close_popup", target=(1, 1))
        before = Observation(frame_id=1, raw_data={"tan_chuang_visible": True})
        after = Observation(frame_id=2)  # 弹窗已消失
        assert task.verify_action(action, before, after)

    def test_close_popup_not_verified_when_still_visible(self):
        task = make_task()
        action = Action(kind="click", tag="close_popup", target=(1, 1))
        before = Observation(frame_id=1, raw_data={"tan_chuang_visible": True})
        after = Observation(frame_id=2, raw_data={"tan_chuang_visible": True})
        assert not task.verify_action(action, before, after)


# ---------------------------------------------------------------------------
# decide_action - BUILD_MAIN_C
# ---------------------------------------------------------------------------


class TestBuildMainCAction:
    def test_forced_summons_use_fast_interval_until_last(self):
        """强制召唤阶段快速连点：非最后一次用固定间隔，最后一次保留稳定等待。"""
        task = make_task(minimum_summons=3)
        task.ctx.current_state = State.BUILD_MAIN_C
        obs = Observation(frame_id=1, board=make_board([]))

        # 前 2 次：固定快速间隔（不做棋盘识别，点击发送即计数）
        for expected_count in (1, 2):
            decision = task.decide_action(obs, make_window_ctx())
            assert decision is not None
            action, _ = decision
            assert action.tag == "required_summon"
            assert action.post_delay == task._run_config.forced_summon_interval_seconds
            task.on_action_verified(*decision)
            assert task._summon_count == expected_count

        # 第 3 次（最后一次）：恢复正常稳定等待（接下来要识别棋盘做决策）
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        action, _ = decision
        assert 1.0 <= action.post_delay <= 2.0
        task.on_action_verified(*decision)
        assert task._summon_count == 3

    def test_wants_board_watch_off_during_forced_summons(self):
        """强制召唤阶段不做多帧棋盘识别；完成后恢复。"""
        task = make_task(minimum_summons=3)
        task.ctx.current_state = State.BUILD_MAIN_C
        assert task.wants_board_watch() is False
        task._summon_count = 3
        assert task.wants_board_watch() is True

    def test_required_summon_reaches_minimum_count(self):
        task = make_task(minimum_summons=2)
        task.ctx.current_state = State.BUILD_MAIN_C
        obs = Observation(frame_id=1, board=make_board([]))
        first = task.decide_action(obs, make_window_ctx())
        assert first is not None
        task.on_action_verified(*first)
        second = task.decide_action(obs, make_window_ctx())
        assert second is not None
        task.on_action_verified(*second)
        assert task._summon_count == 2

    def test_required_summon_ignores_unstable_board(self):
        """前置召唤期间识别不稳定也继续召唤，不把模型结果作为点击门禁。"""
        task = make_task(minimum_summons=3)
        task.ctx.current_state = State.BUILD_MAIN_C
        obs = Observation(frame_id=1, board=None)
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "click"

    def test_stops_when_target_main_c_exists(self):
        task = make_task(target_star=2)
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        board = make_board([make_hero("assault", star=2, pos=(100, 100))])
        obs = Observation(frame_id=1, board=board)
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert action.kind == "wait"
        assert to_state == State.SELECT_MAIN_C_SKILLS

    def test_stops_on_higher_star_main_c(self):
        task = make_task(target_star=2)
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        board = make_board([make_hero("assault", star=3)])
        obs = Observation(frame_id=1, board=board)
        decision = task.decide_action(obs, make_window_ctx())
        assert to_state_of(decision) == State.SELECT_MAIN_C_SKILLS

    def test_merge_when_candidate_exists(self):
        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        # 两个同类型同星级英雄 = 合法合成对
        board = make_board(
            [
                make_hero("monkey", star=1, pos=(100, 100)),
                make_hero("monkey", star=1, pos=(200, 200)),
            ]
        )
        obs = Observation(frame_id=1, board=board)
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert action.kind == "drag"
        # 尽量从下往上拖：起点取更靠下（y 更大）的英雄
        assert action.target == (200, 200)
        assert action.end == (100, 100)
        # 拖动时长可配置（拖太快英雄跟随不及会被弹回原格）
        assert action.duration == task._run_config.merge_drag_duration
        assert to_state == State.BUILD_MAIN_C

    def test_non_main_c_merge_preferred(self):
        """非主C合成对优先于主C合成对。"""
        task = make_task(main_c="assault")
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        board = make_board(
            [
                make_hero("assault", star=1, pos=(100, 100)),
                make_hero("assault", star=1, pos=(200, 200)),
                make_hero("monkey", star=1, pos=(300, 300)),
                make_hero("monkey", star=1, pos=(400, 400)),
            ]
        )
        obs = Observation(frame_id=1, board=board)
        decision = task.decide_action(obs, make_window_ctx())
        action, _ = decision
        # 非主C（monkey）应被优先选中，且从下往上拖（y=400 的格子为起点）
        assert action.target == (400, 400)
        assert action.end == (300, 300)

    def test_summon_one_when_no_merge(self):
        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        # 不同类型，无合成对
        board = make_board(
            [
                make_hero("assault", star=1, pos=(100, 100)),
                make_hero("monkey", star=2, pos=(200, 200)),
            ]
        )
        obs = Observation(frame_id=1, board=board)
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert action.kind == "click"
        assert to_state == State.BUILD_MAIN_C

    def test_empty_board_keeps_waiting_instead_of_stopping(self):
        """对局内棋盘识别为空（board=None 或 0 英雄）：持续等待，不退出任务。

        2026-08-21 用户决策：已进入对局后棋盘空必然是页面/弹窗/结算过渡遮挡，
        等待不发任何输入、比退出安全；超过告警节奏后依然继续等待。
        """
        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        for board in (None, make_board([])):
            task._empty_board_count = 0
            obs = Observation(frame_id=1, board=board)
            for _ in range(_EMPTY_BOARD_RETRY * 3):
                decision = task.decide_action(obs, make_window_ctx())
                assert decision is not None
                assert decision[0].kind == "wait"
                assert decision[1] == State.BUILD_MAIN_C

    def test_none_when_add_hero_hotspot_missing(self):
        task = make_task(hotspots={})
        task.ctx.current_state = State.BUILD_MAIN_C
        obs = Observation(frame_id=1, board=make_board([]))
        assert task.decide_action(obs, make_window_ctx()) is None


# ---------------------------------------------------------------------------
# decide_action - 其他状态
# ---------------------------------------------------------------------------


class TestOtherStates:
    def test_select_main_c_skills_waits(self):
        task = make_task()
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        decision = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert action.kind == "wait"
        assert to_state == State.SELECT_MAIN_C_SKILLS

    def test_popup_close_resets_main_skill_empty_checks(self):
        """弹窗关闭后重置空识别计数：关完弹窗不得立即触发「随便选一个」盲点。"""
        task = make_task()
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        task._awaiting_main_candidates = True
        task._main_skill_empty_checks = task._run_config.skill_recognition_frames
        task.on_action_verified(
            Action(kind="click", tag="close_popup", target=(1, 1)),
            State.SELECT_MAIN_C_SKILLS,
        )
        assert task._main_skill_empty_checks == 0

    def test_no_fallback_click_when_skill_page_closed(self):
        """弹窗把技能页打断关闭后：页面不在时不盲点技能卡，确认关闭后回到定时节奏。"""
        task = make_task()
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        task._awaiting_main_candidates = True
        # 空识别计数已达阈值，但页面已不在 → 不能触发 main_skill_fallback
        task._main_skill_empty_checks = task._run_config.skill_recognition_frames

        page_closed_obs = Observation(frame_id=1)  # 无候选、无【选技能】标志
        frames = task._run_config.main_skill_page_closed_frames
        for frame_id in range(1, frames + 1):
            page_closed_obs = Observation(frame_id=frame_id)
            decision = task.decide_action(page_closed_obs, make_window_ctx())
            assert decision is not None
            assert decision[0].tag == "main_skill_page_missing_check"
            task.on_action_verified(*decision)

        # 连续确认页面关闭 → 停止等待图标，回到定时选技能节奏（等待下一次检查时间）
        decision = task.decide_action(page_closed_obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_skill_page_closed"
        task.on_action_verified(*decision)
        assert task._awaiting_main_candidates is False

        follow_up = task.decide_action(Observation(frame_id=frames + 2), make_window_ctx())
        assert follow_up is not None
        assert follow_up[0].kind == "wait"
        assert follow_up[0].reason == "等待下一次金币检查"

    def test_main_skill_click_failure_schedules_next_attempt(self):
        """局内技能点击重试耗尽后：不中止任务，改为稍后再开技能页重试。"""
        task = make_task()
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        task._awaiting_main_candidates = True

        task.on_action_failed(Action(kind="click", tag="main_skill_fallback", target=(1, 1)))

        assert task._awaiting_main_candidates is False
        assert task._next_skill_at is not None
        # 未到下次检查时间 → 等待而不是再次点击
        decision = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "wait"

    def test_opening_skill_click_failure_blocks_then_unblocks(self):
        """开局技能点击重试耗尽后：页面还在时等待不盲点；页面关闭后解除阻塞。"""
        task = make_task()
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._opening_loaded = True

        task.on_action_failed(Action(kind="click", tag="opening_skill_candidate", target=(1, 1)))
        assert task._opening_clicks_blocked is True

        # 技能页仍在 → 只等待，不再发点击
        page_open = Observation(
            frame_id=1,
            skill_candidates=[SkillCandidate("assault", (100, 100), 0.9)],
        )
        decision = task.decide_action(page_open, make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "wait"
        assert decision[0].tag == "opening_skill_blocked_wait"

        # 页面关闭、召唤出现 → 解除阻塞并正常推进
        closed = Observation(frame_id=2, raw_data={"summon_button_visible": True})
        decision = task.decide_action(closed, make_window_ctx())
        assert decision is not None
        assert task._opening_clicks_blocked is False

    def test_kick_back_to_home_restarts_recruit_from_home(self):
        """等待开局期间连续识别到首页：判定本局被取消，回首页重新进入招募。"""
        task = make_task()
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._opening_loaded = True
        frames = task._run_config.home_return_confirm_frames
        for frame_id in range(1, frames):
            decision = task.decide_action(home_observation(frame_id=frame_id), make_window_ctx())
            assert decision is not None
            assert decision[0].tag == "home_return_check"
            task.on_action_verified(*decision)

        decision = task.decide_action(home_observation(frame_id=frames), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "kick_reentry_home"
        assert decision[1] == State.FIND_COOP
        task.on_action_verified(*decision)
        task.ctx.current_state = State.FIND_COOP

        # 重新进入招募：从领地入口开始，而不是停在开局等待
        follow_up = task.decide_action(
            Observation(frame_id=frames + 1, raw_data={"territory_nav_text": "领地"}),
            make_window_ctx(),
        )
        assert follow_up is not None
        assert follow_up[0].tag == "open_territory"

    def test_single_home_frame_flash_does_not_reenter(self):
        """单帧首页命中（加载画面相似）不触发重进：下一帧回对局界面即计数归零。"""
        task = make_task()
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._opening_loaded = True
        decision = task.decide_action(home_observation(frame_id=1), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "home_return_check"
        task.on_action_verified(*decision)

        resumed = task.decide_action(
            Observation(frame_id=2, raw_data={"summon_button_visible": True}),
            make_window_ctx(),
        )
        assert resumed is not None
        assert task._home_return_count == 0
        assert resumed[0].tag == "opening_exit_confirm_check"

    def test_match_start_timeout_restarts_recruit(self):
        """点准备后长时间无对局界面：按本局未开始处理，重新进入招募。"""
        task = make_task()
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._opening_loaded = True
        task._match_started_at = 0.0
        task._clock = lambda: task._run_config.match_start_timeout_seconds + 1.0

        decision = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "match_start_timeout"
        assert decision[1] == State.FIND_COOP
        task.on_action_verified(*decision)
        task.ctx.current_state = State.FIND_COOP

        follow_up = task.decide_action(
            Observation(frame_id=2, raw_data={"territory_nav_text": "领地"}),
            make_window_ctx(),
        )
        assert follow_up is not None
        assert follow_up[0].tag == "open_territory"

    def test_kick_reentry_skips_difficulty_reselection(self):
        """已点过准备（本会话难度已勾选）后被踢回：重新进入不重复勾选难度。"""
        task = make_task(coop_difficulties=[2, 1])
        task.ctx.current_state = State.FIND_COOP
        task.on_action_verified(
            Action(kind="click", tag="ready_match", target=(1, 1)),
            State.SELECT_OPENING_SKILLS,
        )
        # 被踢回首页后的重进路径：领地入口 → 挑战页聊天 → 打开难度弹窗
        for observation, expected in list(
            zip(
                territory_entry_observations(),
                [
                    "open_territory",
                    "territory_open_confirmed",
                    "open_mode_select",
                    "mode_select_open_confirmed",
                    "mode_card_selected",
                    "challenge_page_confirmed",
                ],
                strict=False,
            )
        ) + [
            (Observation(frame_id=7), "open_challenge_chat"),
            (Observation(frame_id=8), "open_difficulty_dialog"),
        ]:
            decision = task.decide_action(observation, make_window_ctx())
            assert decision is not None
            assert decision[0].tag == expected
            task.on_action_verified(*decision)

        # 打开确认通过后 → 直接关闭（不再 select_difficulty：重复点击会取消勾选）
        advance_confirm_difficulty_open(task, 2, 1, frame_id=4)
        decision = task.decide_action(difficulty_observation(2, 1, frame_id=10), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "close_difficulty_dialog"

    def test_leave_team_after_host_idle_timeout(self):
        """点准备后房主长时间不开始：点击【退队】退出本队重新抢合作。"""
        task = make_task()
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._opening_loaded = True
        task._match_started_at = 0.0
        task._clock = lambda: task._run_config.leave_team_after_seconds + 1.0
        match = MatchResult("buttons/he_zuo_tui_dui.png", (300, 1500), 0.9)
        obs = Observation(
            frame_id=1,
            raw_data={"leave_team_visible": True, "leave_team_match": match},
        )

        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert action.tag == "leave_team"
        assert action.target == (300, 1500)
        assert action.verification == "next_frame"
        assert to_state == State.FIND_COOP

        # 动作后验证：退队按钮消失才算退出大厅
        assert task.verify_action(action, obs, Observation(frame_id=2))
        assert not task.verify_action(
            action, obs, Observation(frame_id=2, raw_data={"leave_team_visible": True})
        )
        task.on_action_verified(*decision)
        task.ctx.current_state = State.FIND_COOP

        # 退队后落点探测：识别到首页 → 从领地入口进入招募
        detect = task.decide_action(home_observation(frame_id=3), make_window_ctx())
        assert detect is not None
        assert detect[0].tag == "entry_page_detected"
        task.on_action_verified(*detect)
        follow = task.decide_action(
            Observation(frame_id=4, raw_data={"territory_nav_text": "领地"}),
            make_window_ctx(),
        )
        assert follow is not None
        assert follow[0].tag == "open_territory"

    def test_leave_team_entry_detect_falls_back_to_coop_page(self):
        """退队后未识别到首页标志：按合作页面处理，从合作页面聊天进入。"""
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        task.on_action_verified(
            Action(kind="click", tag="leave_team", target=(1, 1)),
            State.FIND_COOP,
        )

        detect = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert detect is not None
        assert detect[0].tag == "entry_page_detected"
        task.on_action_verified(*detect)
        follow = task.decide_action(Observation(frame_id=2), make_window_ctx())
        assert follow is not None
        assert follow[0].tag == "open_coop_chat"

    def test_lobby_wait_before_leave_timeout(self):
        """房主未开始的等待期内（未超时）：只等待，不点退队。"""
        task = make_task()
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._opening_loaded = True
        task._match_started_at = 0.0
        task._clock = lambda: task._run_config.leave_team_after_seconds - 5.0
        obs = Observation(
            frame_id=1,
            raw_data={
                "leave_team_visible": True,
                "leave_team_match": MatchResult("buttons/he_zuo_tui_dui.png", (300, 1500), 0.9),
            },
        )

        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "wait"
        assert decision[0].tag == "lobby_wait"

    def test_teammate_icon_triggers_fallback_after_settle_frames(self):
        """局内技能页识别到队友图标：本组无主C卡，页面稳定后随机选，不等满识别帧。

        首帧（页面刚弹出，图标可能先于卡片渲染）只等待，下一帧才点击。
        """
        task = make_task()
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        task._awaiting_main_candidates = True
        assert task._main_skill_empty_checks == 0  # 未等满识别帧

        obs = Observation(
            frame_id=1,
            raw_data={
                "select_skill_button_visible": True,
                "teammate_skill_visible": True,
            },
        )
        settle = task._run_config.skill_fallback_settle_frames
        for _frame_index in range(1, settle + 1):
            first = task.decide_action(obs, make_window_ctx())
            assert first is not None
            assert first[0].kind == "wait"  # 稳定期内只等待，不点
            task.on_action_verified(*first)

        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_skill_fallback"
        assert decision[0].kind == "click"

    def test_opening_teammate_icon_triggers_fallback_after_settle_frames(self):
        """开局技能页同理：队友图标可见且页面稳定后随机选一张。"""
        task = make_task()
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._opening_loaded = True
        assert task._opening_empty_checks == 0

        obs = Observation(
            frame_id=1,
            raw_data={
                "select_skill_button_visible": True,
                "teammate_skill_visible": True,
            },
        )
        settle = task._run_config.skill_fallback_settle_frames
        for _ in range(settle):
            first = task.decide_action(obs, make_window_ctx())
            assert first is not None
            assert first[0].kind == "wait"
            task.on_action_verified(*first)

        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "opening_skill_fallback"

    def test_failed_merge_pair_skipped_for_other_pair(self):
        """已拖动失败的合成对本轮跳过，改选其他合法对。"""
        task = make_task(main_c="assault")
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        # 两个非主C对：monkey 对已失败（签名为拖动方向：下格在前），angel 对未试
        task._failed_merge_pairs = {((200, 200), (100, 100))}
        board = make_board(
            [
                make_hero("monkey", star=1, pos=(100, 100)),
                make_hero("monkey", star=1, pos=(200, 200)),
                make_hero("angel", star=1, pos=(300, 300)),
                make_hero("angel", star=1, pos=(400, 400)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "drag"
        assert decision[0].target == (400, 400)
        assert decision[0].end == (300, 300)

    def test_all_merge_pairs_failed_falls_back_to_summon(self):
        """全部合成对均已拖动失败：召唤新英雄改变棋盘，不结束任务。"""
        task = make_task(main_c="assault")
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        task._failed_merge_pairs = {((200, 200), (100, 100))}
        board = make_board(
            [
                make_hero("monkey", star=1, pos=(100, 100)),
                make_hero("monkey", star=1, pos=(200, 200)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "click"
        assert decision[0].tag == "summon_hero"

    def test_successful_merge_clears_failed_pairs(self):
        """合成成功 = 拖动机制正常：清空失败记忆，失败过的对可重试。"""
        task = make_task()
        task._failed_merge_pairs = {((100, 100), (200, 200))}
        task.on_action_verified(
            Action(kind="drag", target=(300, 300), end=(400, 400), tag="merge_heroes"),
            State.BUILD_MAIN_C,
        )
        assert task._failed_merge_pairs == set()

    def test_merge_failure_records_failed_pair(self):
        """合成重试耗尽：记录失败对位置，供决策跳过。"""
        task = make_task()
        task.on_action_failed(
            Action(kind="drag", target=(100, 100), end=(200, 200), tag="merge_heroes")
        )
        assert task._failed_merge_pairs == {((100, 100), (200, 200))}

    def test_skill_selections_trigger_topup_return(self):
        """强袭策略：局内选满 4 次技能后回培养阶段补主C数量。"""
        profile = MainCProfile(
            display_name="强袭",
            hero_template_dir="heroes/assault",
            avoid_main_c_merge=True,
            topup_after_skill_selections=4,
            topup_hero_count=4,
        )
        task = make_task(main_c_profile=profile)
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        task._main_skill_selections = 4

        decision = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_topup"
        assert decision[1] == State.BUILD_MAIN_C

    def test_topup_not_triggered_below_selection_count(self):
        """未选满次数不回培养：维持定时选技能节奏。"""
        profile = MainCProfile(
            display_name="强袭",
            hero_template_dir="heroes/assault",
            topup_after_skill_selections=4,
            topup_hero_count=4,
        )
        task = make_task(main_c_profile=profile)
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        task._main_skill_selections = 3

        decision = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "wait"
        assert decision[0].reason == "等待下一次金币检查"

    def test_first_cultivation_only_requires_star_level(self):
        """第一阶段（首次培养）：只看 2 星主C，不检查数量，尽早进选技能。"""
        profile = MainCProfile(
            display_name="强袭",
            hero_template_dir="heroes/assault",
            topup_after_skill_selections=4,
            topup_hero_count=4,
        )
        task = make_task(main_c_profile=profile)
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        # 只有 1 个 2 星强袭（数量 1 < 4），但第一阶段不看数量
        board = make_board([make_hero("assault", star=2, pos=(100, 100))])
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS

    def test_topup_phase_requires_hero_count(self):
        """第三阶段（数量回补）：2星主C存在但数量不足时继续召唤，补够才回选技能。"""
        profile = MainCProfile(
            display_name="强袭",
            hero_template_dir="heroes/assault",
            topup_after_skill_selections=4,
            topup_hero_count=4,
        )
        task = make_task(main_c_profile=profile)
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        task._topup_active = True  # 已进入回补阶段
        board = make_board([make_hero("assault", star=2, pos=(100, 100))])
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "summon_hero"

        # 数量补到 4（含 2 星）→ 回选技能
        board = make_board(
            [
                make_hero("assault", star=2, pos=(100, 100)),
                make_hero("assault", star=1, pos=(200, 200)),
                make_hero("assault", star=1, pos=(300, 300)),
                make_hero("assault", star=1, pos=(400, 400)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=2, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS

    def test_topup_active_flag_transitions(self):
        """回补标记随 main_c_topup 置位、main_c_ready 复位。"""
        task = make_task()
        task.on_action_verified(Action(kind="wait", tag="main_c_topup"), State.BUILD_MAIN_C)
        assert task._topup_active is True
        task.on_action_verified(Action(kind="wait", tag="main_c_ready"), State.SELECT_MAIN_C_SKILLS)
        assert task._topup_active is False

    def test_topup_not_retriggered_after_count_satisfied(self):
        """数量补够后本局不再复查：再选满 4 次技能不回培养（2026-09-07 用户策略）。"""
        profile = MainCProfile(
            display_name="强袭",
            hero_template_dir="heroes/assault",
            topup_after_skill_selections=4,
            topup_hero_count=4,
        )
        task = make_task(main_c_profile=profile)
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        task._topup_active = True
        board = make_board(
            [
                make_hero("assault", star=2, pos=(100, 100)),
                make_hero("assault", star=1, pos=(200, 200)),
                make_hero("assault", star=1, pos=(300, 300)),
                make_hero("assault", star=1, pos=(400, 400)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        task.on_action_verified(*decision)
        assert task._topup_done is True

        # 再选满 4 次技能：旧逻辑会再次触发回补，新逻辑本局不再检查数量
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        for _ in range(4):
            task.on_action_verified(
                Action(kind="click", tag="main_skill_candidate", target=(1, 1)),
                State.SELECT_MAIN_C_SKILLS,
            )
        assert task._main_skill_selections == 4
        decision = task.decide_action(Observation(frame_id=2), make_window_ctx())
        assert decision is not None
        assert decision[0].tag != "main_c_topup"
        assert decision[0].kind == "wait"
        assert decision[0].reason == "等待下一次金币检查"

    def test_first_entry_with_enough_count_skips_topup(self):
        """首次培养时数量就已达标：main_c_ready 即置位，本局不再回补复查。"""
        profile = MainCProfile(
            display_name="强袭",
            hero_template_dir="heroes/assault",
            topup_after_skill_selections=4,
            topup_hero_count=4,
        )
        task = make_task(main_c_profile=profile)
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        board = make_board(
            [
                make_hero("assault", star=2, pos=(100, 100)),
                make_hero("assault", star=1, pos=(200, 200)),
                make_hero("assault", star=1, pos=(300, 300)),
                make_hero("assault", star=1, pos=(400, 400)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        task.on_action_verified(*decision)
        assert task._topup_done is True

        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        for _ in range(4):
            task.on_action_verified(
                Action(kind="click", tag="main_skill_candidate", target=(1, 1)),
                State.SELECT_MAIN_C_SKILLS,
            )
        decision = task.decide_action(Observation(frame_id=2), make_window_ctx())
        assert decision is not None
        assert decision[0].tag != "main_c_topup"

    def test_topup_done_resets_for_new_round(self):
        """数量达标标志只对本局有效：新一局（准备/被踢重进）后复位。"""
        task = make_task()
        task._topup_done = True
        task.on_action_verified(
            Action(kind="click", tag="ready_match", target=(1, 1)),
            State.SELECT_OPENING_SKILLS,
        )
        assert task._topup_done is False
        task._topup_done = True
        task._reset_for_home_reentry()
        assert task._topup_done is False

    def test_topup_summons_instead_of_merge_when_board_has_space(self):
        """回补阶段棋盘有空位：有非主C合成对也不拖动，直接召唤（2026-09-07 用户策略）。"""
        profile = MainCProfile(
            display_name="强袭",
            hero_template_dir="heroes/assault",
            avoid_main_c_merge=True,
            topup_after_skill_selections=4,
            topup_hero_count=4,
        )
        task = make_task(main_c_profile=profile)
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        task._topup_active = True
        # 数量 1 < 4，monkey 存在合法对，棋盘 3/12 有空位
        board = make_board(
            [
                make_hero("assault", star=2, pos=(100, 100)),
                make_hero("monkey", star=1, pos=(300, 300)),
                make_hero("monkey", star=1, pos=(400, 400)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "click"
        assert decision[0].tag == "summon_hero"

    def test_topup_merges_when_board_full(self):
        """回补阶段棋盘占满且数量未达标：恢复合成腾格子。"""
        profile = MainCProfile(
            display_name="强袭",
            hero_template_dir="heroes/assault",
            avoid_main_c_merge=True,
            topup_after_skill_selections=4,
            topup_hero_count=4,
        )
        task = make_task(main_c_profile=profile)
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        task._topup_active = True
        heroes = [
            make_hero("monkey", star=1, pos=(100, 100)),
            make_hero("monkey", star=1, pos=(200, 200)),
        ]
        board = BoardSnapshot(
            frame_id=1,
            heroes=heroes,
            capacity=BoardCapacity(total_slots=2, occupied=2),
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "drag"
        assert decision[0].tag == "merge_heroes"

    def test_non_topup_merge_still_preferred_when_board_has_space(self):
        """非回补阶段（首次培养）保持原行为：有合法对优先合成。"""
        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        board = make_board(
            [
                make_hero("snow", star=1, pos=(100, 100)),
                make_hero("snow", star=1, pos=(200, 200)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "drag"
        assert decision[0].tag == "merge_heroes"

    def test_main_c_ready_resets_skill_selection_counter(self):
        """数量达标进入选技能时清零选择计数，开始新一轮「选 N 次 → 检查」。"""
        profile = MainCProfile(
            display_name="强袭",
            hero_template_dir="heroes/assault",
            topup_after_skill_selections=4,
            topup_hero_count=4,
        )
        task = make_task(main_c_profile=profile)
        task._main_skill_selections = 4
        task.on_action_verified(Action(kind="wait", tag="main_c_ready"), State.SELECT_MAIN_C_SKILLS)
        assert task._main_skill_selections == 0

    def test_main_skill_click_increments_selection_counter(self):
        task = make_task()
        task.on_action_verified(
            Action(kind="click", tag="main_skill_candidate", target=(1, 1)),
            State.SELECT_MAIN_C_SKILLS,
        )
        assert task._main_skill_selections == 1

    def test_avoid_main_c_merge_summons_when_board_has_space(self):
        """强袭策略：无非主C对、只剩强袭对且棋盘有空位 → 召唤而不合并强袭。"""
        profile = MainCProfile(
            display_name="强袭",
            hero_template_dir="heroes/assault",
            avoid_main_c_merge=True,
        )
        task = make_task(main_c_profile=profile)
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        # 两个 1 星强袭（唯一合成对），占用 2/7 有空位
        board = make_board(
            [
                make_hero("assault", star=1, pos=(100, 100)),
                make_hero("assault", star=1, pos=(200, 200)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "click"
        assert decision[0].tag == "summon_hero"

    def test_main_c_merge_allowed_as_last_resort_when_board_full(self):
        """强袭策略：棋盘占满时合并强袭对腾格子（最后手段）。"""
        profile = MainCProfile(
            display_name="强袭",
            hero_template_dir="heroes/assault",
            avoid_main_c_merge=True,
        )
        task = make_task(main_c_profile=profile)
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        heroes = [make_hero("assault", star=1, pos=(100 * i, 100 * i)) for i in range(1, 3)]
        board = make_board(heroes)
        board = BoardSnapshot(
            frame_id=1,
            heroes=heroes,
            capacity=BoardCapacity(total_slots=2, occupied=2),
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "drag"
        assert decision[0].tag == "merge_heroes"

    def test_non_main_c_merge_still_preferred_under_avoid(self):
        """强袭策略：有非主C对时照常合并非主C（不回避）。"""
        profile = MainCProfile(
            display_name="强袭",
            hero_template_dir="heroes/assault",
            avoid_main_c_merge=True,
        )
        task = make_task(main_c_profile=profile)
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        board = make_board(
            [
                make_hero("assault", star=1, pos=(100, 100)),
                make_hero("assault", star=1, pos=(200, 200)),
                make_hero("monkey", star=1, pos=(300, 300)),
                make_hero("monkey", star=1, pos=(400, 400)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "drag"
        assert decision[0].target == (400, 400)  # monkey 对，从下往上拖

    def test_star3_pair_avoided_when_lower_pair_available(self):
        """有 1/2 星对时优先合并低星对，不碰 3 星对（2026-08-21 用户策略）。"""
        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        board = make_board(
            [
                make_hero("snow", star=3, pos=(100, 100)),
                make_hero("snow", star=3, pos=(200, 200)),
                make_hero("angel", star=1, pos=(300, 300)),
                make_hero("angel", star=1, pos=(400, 400)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "drag"
        assert decision[0].tag == "merge_heroes"
        assert decision[0].target == (400, 400)  # angel 1星对，从下往上拖
        assert decision[0].end == (300, 300)

    def test_star3_pair_only_with_board_space_summons_instead(self):
        """只剩 3 星对且棋盘有空位 → 召唤新英雄避开（不合成 4 星弹赠送页）。"""
        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        board = make_board(
            [
                make_hero("snow", star=3, pos=(100, 100)),
                make_hero("snow", star=3, pos=(200, 200)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "click"
        assert decision[0].tag == "summon_hero"

    def test_star3_pair_forced_when_board_full_sets_gift_await(self):
        """棋盘占满只剩 3 星对 → 被迫合并，并挂起 4 星赠送技能页确认流程。"""
        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        heroes = [
            make_hero("snow", star=3, pos=(100, 100)),
            make_hero("snow", star=3, pos=(200, 200)),
        ]
        board = BoardSnapshot(
            frame_id=1,
            heroes=heroes,
            capacity=BoardCapacity(total_slots=2, occupied=2),
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "drag"
        assert decision[0].tag == "merge_heroes"
        assert task._awaiting_merge_gift is True
        assert task._merge_gift_settled is False
        assert task.wants_board_watch() is False  # 等待期切单帧快识别

    def test_summon_failure_in_topup_returns_to_skills(self):
        """回补阶段召唤未生效（金币不足）：放弃回补，按星级门禁回选技能。"""
        profile = MainCProfile(
            display_name="强袭",
            hero_template_dir="heroes/assault",
            topup_after_skill_selections=4,
            topup_hero_count=4,
        )
        task = make_task(main_c_profile=profile)
        task.ctx.current_state = State.BUILD_MAIN_C
        task._topup_active = True
        task._main_skill_selections = 4

        task.on_action_failed(Action(kind="click", tag="summon_hero", target=(1, 1)))

        assert task._topup_active is False
        assert task._main_skill_selections == 0
        # 数量不足（1 个强袭）但星级满足 → 星级门禁直接放行回选技能
        board = make_board([make_hero("assault", star=2, pos=(100, 100))])
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS

    def test_summon_failure_in_cultivation_sets_retry_cooldown(self):
        """培养阶段召唤未生效：进入冷却等待，冷却结束后重新召唤。"""
        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        task._clock = lambda: 100.0

        task.on_action_failed(Action(kind="click", tag="summon_hero", target=(1, 1)))
        assert task._summon_retry_at == 100.0 + task._run_config.summon_retry_delay_seconds

        # 冷却期内：只等待不召唤
        decision = task.decide_action(
            Observation(frame_id=1, board=make_board([])), make_window_ctx()
        )
        assert decision is not None
        assert decision[0].tag == "summon_retry_wait"

        # 冷却结束：恢复正常召唤决策
        task._clock = lambda: 100.0 + task._run_config.summon_retry_delay_seconds + 1.0
        decision = task.decide_action(
            Observation(frame_id=1, board=make_board([])), make_window_ctx()
        )
        assert decision is not None
        assert decision[0].tag == "required_summon"

    def test_merge_gift_skill_page_prefers_main_c_icon(self):
        """培养中出现赠送技能页（合成4星）：优先点主C技能卡，留在培养状态。"""
        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        candidates = [SkillCandidate("assault", (300, 800), 0.9)]
        obs = Observation(
            frame_id=1,
            raw_data={"select_skill_button_visible": True},
            skill_candidates=candidates,
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "merge_gift_skill_candidate"
        assert decision[0].target == (300, 800)
        assert decision[0].verification == "next_frame"
        assert decision[1] == State.BUILD_MAIN_C

    def test_merge_gift_skill_page_random_fallback(self):
        """赠送技能页识别不到主C图标：三列兜底随机选一张。"""
        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        obs = Observation(frame_id=1, raw_data={"select_skill_button_visible": True})

        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "merge_gift_skill_fallback"
        assert decision[0].kind == "click"
        assert decision[1] == State.BUILD_MAIN_C

        # 页面关闭即验证通过（棋盘恢复）
        assert task.verify_action(decision[0], obs, Observation(frame_id=2))

    def test_merge_gift_title_alone_enters_skill_page_branch(self):
        """仅【请选择1个额外技能】提示条命中（【选技能】图未命中）也进入赠送技能分支。

        2026-08-21 实机：赠送页整页等待选择却始终未命中【选技能】图，掉进空棋盘
        重试直至保守停止；提示条为页面主标识，命中即应选卡。
        """
        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        candidates = [SkillCandidate("assault", (300, 800), 0.9)]
        obs = Observation(
            frame_id=1,
            raw_data={"merge_gift_skill_page_visible": True},
            skill_candidates=candidates,
        )

        decision = task.decide_action(obs, make_window_ctx())

        assert decision is not None
        assert decision[0].tag == "merge_gift_skill_candidate"
        assert decision[0].target == (300, 800)
        assert decision[1] == State.BUILD_MAIN_C

    def _awaiting_gift_task(self) -> CoopTask:
        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        task._awaiting_merge_gift = True
        task._merge_gift_settled = False
        return task

    def test_merge_gift_confirm_settles_then_requires_two_hits(self):
        """被迫 3+3 后：先 settle 2 秒，连续 2 次命中提示条才选技能。"""
        task = self._awaiting_gift_task()
        cfg = task._run_config

        # 1. 合并后固定等待页面渲染
        decision = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "merge_gift_settle"
        assert decision[0].duration == cfg.merge_gift_settle_seconds

        # 2. 第一次命中：继续轮询等待，不动作
        hit1 = Observation(frame_id=2, raw_data={"merge_gift_skill_page_visible": True})
        decision = task.decide_action(hit1, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "merge_gift_poll"
        assert decision[0].duration == cfg.merge_gift_poll_interval_seconds
        assert task._awaiting_merge_gift is True

        # 3. 第二次命中（带主C技能卡候选）：进入选卡
        candidates = [SkillCandidate("assault", (300, 800), 0.9)]
        hit2 = Observation(
            frame_id=3,
            raw_data={"merge_gift_skill_page_visible": True},
            skill_candidates=candidates,
        )
        decision = task.decide_action(hit2, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "merge_gift_skill_candidate"
        assert task._awaiting_merge_gift is False

    def test_merge_gift_confirm_gives_up_after_miss_budget(self):
        """连续未命中达预算：放弃等待，恢复常规决策（棋盘为空走空棋盘重试）。"""
        task = self._awaiting_gift_task()
        task.decide_action(Observation(frame_id=1), make_window_ctx())  # settle

        for i in range(task._run_config.merge_gift_fail_misses - 1):
            decision = task.decide_action(Observation(frame_id=2 + i), make_window_ctx())
            assert decision is not None
            assert decision[0].tag == "merge_gift_poll"
            assert task._awaiting_merge_gift is True

        # 第 miss_budget 次：放弃等待，落入空棋盘重试
        decision = task.decide_action(Observation(frame_id=99), make_window_ctx())
        assert decision is not None
        assert "棋盘识别空" in decision[0].reason
        assert task._awaiting_merge_gift is False
        assert task.wants_board_watch() is True

    def test_merge_verify_passes_when_gift_page_visible(self):
        """提示条已弹出即证明 3+3 合成生效（棋盘被遮、常规验证必失败的路径）。"""
        task = make_task()
        before = Observation(frame_id=1, board=make_board([make_hero("snow", star=3)]))
        after = Observation(
            frame_id=2,
            board=None,
            raw_data={"merge_gift_skill_page_visible": True},
        )
        action = Action(kind="drag", target=(100, 100), end=(200, 200), tag="merge_heroes")
        assert task.verify_action(action, before, after) is True

    def test_merge_gift_selection_verified_clears_awaiting(self):
        task = self._awaiting_gift_task()
        task.on_action_verified(
            Action(kind="click", tag="merge_gift_skill_fallback", target=(1, 1)),
            State.BUILD_MAIN_C,
        )
        assert task._awaiting_merge_gift is False
        assert task.wants_board_watch() is True

    def test_double_reward_dialog_cancelled_first(self):
        """结算阶段弹出双倍奖励确认窗：优先点取消，回到原结算状态。"""
        task = make_task()
        task.ctx.current_state = State.HANDLE_RESULT
        match = MatchResult("buttons/qu_xiao_shuang_bei.png", (400, 1200), 0.9)
        obs = Observation(
            frame_id=1,
            raw_data={
                "double_reward_dialog_visible": True,
                "double_reward_cancel_visible": True,
                "double_reward_cancel_match": match,
            },
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert action.tag == "close_double_reward"
        assert action.target == (400, 1200)
        assert action.verification == "next_frame"
        assert to_state == State.HANDLE_RESULT
        # 弹窗消失即验证通过
        assert task.verify_action(action, obs, Observation(frame_id=2))
        assert not task.verify_action(
            action, obs, Observation(frame_id=2, raw_data={"double_reward_dialog_visible": True})
        )

    def test_double_reward_dialog_waits_for_cancel_match(self):
        """弹窗可见但取消按钮未识别到：等待下一帧，不盲点。"""
        task = make_task()
        task.ctx.current_state = State.CLAIM_REWARD
        obs = Observation(frame_id=1, raw_data={"double_reward_dialog_visible": True})
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "wait"
        assert decision[0].tag == "double_reward_dialog_wait"
        assert decision[1] == State.CLAIM_REWARD

    def test_skill_cap_stops_selection(self):
        """局内技能选择总数达到上限：不再选技能，等待对局结束。"""
        profile = MainCProfile(
            display_name="强袭",
            hero_template_dir="heroes/assault",
            topup_after_skill_selections=4,
            topup_hero_count=4,
            skill_selection_cap=9,
        )
        task = make_task(main_c_profile=profile)
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        task._main_skill_selections_total = 9

        decision = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "skill_cap_reached"
        assert decision[0].kind == "wait"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS

    def test_skill_cap_counts_total_across_topup_reset(self):
        """回补重置分段计数，但总数累计不清零：总数到 9 即触发上限。"""
        profile = MainCProfile(
            display_name="强袭",
            hero_template_dir="heroes/assault",
            topup_after_skill_selections=4,
            topup_hero_count=4,
            skill_selection_cap=9,
        )
        task = make_task(main_c_profile=profile)
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        # 回补前选满 4 次 → 触发回补 → 回补完成清零分段计数
        for _ in range(4):
            task.on_action_verified(
                Action(kind="click", tag="main_skill_candidate", target=(1, 1)),
                State.SELECT_MAIN_C_SKILLS,
            )
        task.on_action_verified(Action(kind="wait", tag="main_c_ready"), State.SELECT_MAIN_C_SKILLS)
        assert task._main_skill_selections == 0
        assert task._main_skill_selections_total == 4

        # 回补后再选 5 次（分段计数 5，总数 9）→ 触发上限
        for _ in range(5):
            task.on_action_verified(
                Action(kind="click", tag="main_skill_fallback", target=(1, 1)),
                State.SELECT_MAIN_C_SKILLS,
            )
        assert task._main_skill_selections_total == 9
        decision = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "skill_cap_reached"

    def test_completed_returns_none(self):
        task = make_task()
        task.ctx.current_state = State.COMPLETED
        assert task.decide_action(Observation(frame_id=1), make_window_ctx()) is None

    def test_difficulties_are_selected_from_small_to_large(self):
        """面板打开停在普通区顶部、彩虹小号先出现：从小到大单程点击。"""
        task = make_task(coop_difficulties=[10, 9, 8])
        task.ctx.current_state = State.FIND_COOP
        advance_entry_to_difficulty_dialog(task)
        advance_confirm_difficulty_open(task, 8, 10, 9, frame_id=2)

        observation = difficulty_observation(8, 10, 9)
        selected = []
        for _ in range(3):
            decision = task.decide_action(observation, make_window_ctx())
            assert decision is not None
            selected.append(decision[0].tag)
            assert decision[0].verification == "immediate"
            task.on_action_verified(*decision)

        assert selected == [
            "select_difficulty:8",
            "select_difficulty:9",
            "select_difficulty:10",
        ]

    def test_scroll_distance_scales_with_client_size(self):
        task = make_task(coop_difficulties=[10])
        task.ctx.current_state = State.FIND_COOP
        advance_entry_to_difficulty_dialog(task)
        advance_confirm_difficulty_open(task, 16, 15, 14, frame_id=2)

        decision = task.decide_action(
            difficulty_observation(16, 15, 14),
            make_window_ctx(cw=1000, ch=2000),
        )
        assert decision is not None
        action = decision[0]
        assert action.kind == "drag"
        assert action.target == (500, 600)
        assert action.end == (500, 1400)
        assert action.verification == "next_frame"

    def test_difficulty_only_mode_starts_in_dialog_and_stops_after_selection(self):
        task = make_task(
            coop_difficulties=[2, 1],
            difficulty_selection_only=True,
        )
        task.ctx.current_state = State.FIND_COOP

        assert task.observation_mode() == "coop_difficulty"
        for expected_tag in ["select_difficulty:1", "select_difficulty:2"]:
            decision = task.decide_action(difficulty_observation(2, 1), make_window_ctx())
            assert decision is not None
            assert decision[0].tag == expected_tag
            task.on_action_verified(*decision)

        decision = task.decide_action(difficulty_observation(2, 1), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "difficulty_selection_complete"
        assert decision[1] == State.COMPLETED

    def test_close_dialog_steps_use_light_dialog_mode(self):
        """勾选难度保持候选识别；打开/关闭确认弹窗只识别开关标识（轻量模式）。"""
        task = make_task(coop_difficulties=[2, 1])
        task.ctx.current_state = State.FIND_COOP

        # 领地入口 → 挑战页聊天 → 打开难度弹窗，进入打开确认步骤
        advance_entry_to_difficulty_dialog(task)
        assert task._recruit_step is _RecruitStep.CONFIRM_DIFFICULTY_OPEN
        assert task.observation_mode() == "difficulty_dialog"

        # 确认打开后进入勾选步骤（候选识别模式）
        advance_confirm_difficulty_open(task, frame_id=2)
        assert task._recruit_step is _RecruitStep.SELECT_DIFFICULTIES
        assert task.observation_mode() == "coop_difficulty"

        # 勾选完全部目标难度后进入关闭步骤
        for _ in range(2):
            decision = task.decide_action(difficulty_observation(2, 1), make_window_ctx())
            assert decision is not None
            task.on_action_verified(*decision)
        assert task._recruit_step is _RecruitStep.CLOSE_DIFFICULTY_DIALOG
        assert task.observation_mode() == "difficulty_dialog"

        # 下一局的「开/关难度弹窗刷新邀请」同样只用轻量模式
        task._recruit_step = _RecruitStep.CLOSE_REFRESH_DIFFICULTY_DIALOG
        assert task.observation_mode() == "difficulty_dialog"

    def test_join_step_requests_grab_mode(self):
        """抢合作子步骤：连点加入期间额外识别准备按钮（出现即抢到）。"""
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.JOIN_COOP
        assert task.observation_mode() == "coop_grab"

    def test_scroll_accepts_next_frame_even_when_visible_levels_do_not_change(self):
        task = make_task(coop_difficulties=[10])
        before = difficulty_observation(16, 15, 14, frame_id=1)
        after_same = difficulty_observation(16, 15, 14, frame_id=2)
        after_changed = difficulty_observation(13, 12, 11, frame_id=3)
        stale = difficulty_observation(13, 12, 11, frame_id=1)
        action = Action(
            kind="drag",
            target=(100, 100),
            end=(100, 200),
            tag="scroll_difficulties:smaller",
        )
        assert task.verify_action(action, before, after_same)
        assert task.verify_action(action, before, after_changed)
        assert not task.verify_action(action, before, stale)

    def test_straddle_scrolls_to_retry_instead_of_stopping(self):
        # 难度严格 16→1 连续：9 之后出现 5，说明 8 被误识别。
        # 不应保守停止，应下移一点重新识别 8。
        task = make_task(coop_difficulties=[8], difficulty_selection_only=True)
        task.ctx.current_state = State.FIND_COOP
        decision = task.decide_action(
            difficulty_observation(12, 11, 10, 9, 5),
            make_window_ctx(),
        )
        assert decision is not None
        action = decision[0]
        assert action.kind == "drag"
        assert action.tag == "scroll_difficulties:smaller"

    def test_unrecognized_difficulty_skipped_after_scroll_budget(self):
        # 8 始终识别不到：先滚动重试，预算用尽后跳过而非停止，最后完成。
        task = make_task(coop_difficulties=[8], difficulty_selection_only=True)
        task.ctx.current_state = State.FIND_COOP
        visible = difficulty_observation(12, 11, 10, 9, 5)
        budget = task._run_config.difficulty_max_scrolls
        for _ in range(budget):
            decision = task.decide_action(visible, make_window_ctx())
            assert decision[0].tag == "scroll_difficulties:smaller"
            task.on_action_verified(*decision)
        # 预算用尽：跳过 8
        decision = task.decide_action(visible, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "difficulty_skip:8"
        task.on_action_verified(*decision)
        # 唯一目标已跳过 → 完成
        decision = task.decide_action(visible, make_window_ctx())
        assert decision[0].tag == "difficulty_selection_complete"
        assert decision[1] == State.COMPLETED
        assert task._skipped_difficulties == {8}
        assert task._selected_difficulties == set()
        assert task.skipped_difficulties == (8,)

    def test_unrecognized_difficulty_is_skipped_in_normal_run(self):
        task = make_task(coop_difficulties=[8], difficulty_selection_only=True)
        task._difficulty_selection_only = False
        task.ctx.current_state = State.FIND_COOP
        visible = difficulty_observation(12, 11, 10, 9, 5)
        for _ in range(task._run_config.difficulty_max_scrolls):
            task.on_action_verified(*task.decide_action(visible, make_window_ctx()))

        decision = task.decide_action(visible, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "difficulty_skip:8"
        task.on_action_verified(*decision)

        next_decision = task.decide_action(visible, make_window_ctx())
        assert next_decision is not None
        assert next_decision[0].tag == "close_difficulty_dialog"
        assert task.skipped_difficulties == (8,)

    def test_empty_candidates_scroll_toward_rainbow_zone(self):
        """面板打开在普通难度区（无彩虹候选）：从下往上拉向更大难度滚动，不等待不跳过。"""
        task = make_task(coop_difficulties=[10, 8], difficulty_selection_only=True)
        task.ctx.current_state = State.FIND_COOP

        decision = task.decide_action(Observation(frame_id=1), make_window_ctx(cw=1000, ch=2000))

        assert decision is not None
        action = decision[0]
        assert action.kind == "drag"
        assert action.tag == "scroll_difficulties:larger"
        # 从下往上拉：起点在列表偏下（scroll_end=0.7），终点在偏上（scroll_start=0.3）
        assert action.target == (500, 1400)
        assert action.end == (500, 600)

    def test_empty_candidates_skip_only_after_scroll_budget(self):
        """空识别滚动预算耗尽才跳过；没见过彩虹行的跳过不重置预算，后续目标级联跳过。"""
        task = make_task(coop_difficulties=[10, 8], difficulty_selection_only=True)
        task.ctx.current_state = State.FIND_COOP
        empty = Observation(frame_id=1)

        budget = task._run_config.difficulty_max_scrolls
        for _ in range(budget):
            decision = task.decide_action(empty, make_window_ctx())
            assert decision is not None
            assert decision[0].tag == "scroll_difficulties:larger"
            task.on_action_verified(*decision)
        decision = task.decide_action(empty, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "difficulty_skip:8"
        task.on_action_verified(*decision)

        # 列表根本没进彩虹区：下一个目标不再空滚一遍，直接跳过
        follow = task.decide_action(empty, make_window_ctx())
        assert follow is not None
        assert follow[0].tag == "difficulty_skip:10"

    def test_skip_after_seen_candidates_resets_scroll_budget(self):
        """见过彩虹候选后跳过的目标会重置预算：下个目标仍有滚动机会。"""
        task = make_task(coop_difficulties=[8], difficulty_selection_only=True)
        task.ctx.current_state = State.FIND_COOP
        task._difficulty_scroll_count = task._run_config.difficulty_max_scrolls

        visible = difficulty_observation(12, 11, 10, 9)
        decision = task.decide_action(visible, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "difficulty_skip:8"
        task.on_action_verified(*decision)

        assert task._difficulty_scroll_count == 0

    def test_difficulty_summary_tracks_selected_and_skipped(self):
        task = make_task(coop_difficulties=[9, 8, 7], difficulty_selection_only=True)
        task.ctx.current_state = State.FIND_COOP
        # 7 可见（最小目标先选）→ 选中
        decision = task.decide_action(difficulty_observation(7, 6, 5), make_window_ctx())
        assert decision[0].tag == "select_difficulty:7"
        task.on_action_verified(*decision)
        # 8 始终误识别为 5 → 滚到预算用尽 → 跳过
        misread = difficulty_observation(12, 11, 10, 9, 5)
        for _ in range(task._run_config.difficulty_max_scrolls):
            task.on_action_verified(*task.decide_action(misread, make_window_ctx()))
        decision = task.decide_action(misread, make_window_ctx())
        assert decision[0].tag == "difficulty_skip:8"
        task.on_action_verified(*decision)
        # 9 可见 → 选中，随后完成
        decision = task.decide_action(difficulty_observation(9, 10, 11), make_window_ctx())
        assert decision[0].tag == "select_difficulty:9"
        task.on_action_verified(*decision)
        decision = task.decide_action(difficulty_observation(9, 10, 11), make_window_ctx())
        assert decision[1] == State.COMPLETED
        assert task._selected_difficulties == {9, 7}
        assert task._skipped_difficulties == {8}

    def test_unknown_returns_none(self):
        task = make_task()
        task.ctx.current_state = State.UNKNOWN
        assert task.decide_action(Observation(frame_id=1), make_window_ctx()) is None

    def test_enter_match_without_ready_match_stops(self):
        task = make_task()
        task.ctx.current_state = State.ENTER_MATCH
        assert task.decide_action(Observation(frame_id=1), make_window_ctx()) is None

    def test_handle_result_clicks_like(self):
        task = make_task()
        task.ctx.current_state = State.HANDLE_RESULT
        decision = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "like_result"
        assert decision[0].verification == "immediate"
        assert decision[1] == State.CLAIM_REWARD

    def test_claim_reward_clicks_once_without_result_verification(self):
        task = make_task()
        task.ctx.current_state = State.CLAIM_REWARD
        decision = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "claim_chest"
        assert decision[0].verification == "immediate"
        assert decision[1] == State.CHECK_ROUND_LIMIT

    def test_skill_candidates_random_pick(self):
        # 简化版：候选都是主C技能图标位置，随机点其中一个
        task = make_task()
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._opening_loaded = True
        candidates = [
            SkillCandidate("assault", (100, 100), 0.9),
            SkillCandidate("assault", (200, 100), 0.8),
        ]
        decision = task.decide_action(
            Observation(frame_id=1, skill_candidates=candidates),
            make_window_ctx(),
        )
        assert decision is not None
        assert decision[0].target in {(100, 100), (200, 100)}
        assert decision[0].tag == "opening_skill_candidate"

    def test_opening_skill_fallback_after_recognition_frames(self):
        # 【选技能】页面打开但连续 skill_recognition_frames 帧没识别到主C图标 → 随便选一个
        task = make_task()
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._opening_loaded = True
        task._opening_empty_checks = task._run_config.skill_recognition_frames
        obs = Observation(
            frame_id=1,
            raw_data={"select_skill_button_visible": True},  # 开局技能页面已打开
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "opening_skill_fallback"
        assert decision[0].kind == "click"
        assert decision[0].verification == "next_frame"
        expected_points = roi_column_centers(
            task._skill_candidate_roi,
            make_window_ctx().client_size,
            columns=3,
        )
        assert decision[0].target in expected_points
        # 选后留在本状态，等【选技能】消失再进召唤
        assert decision[1] == State.SELECT_OPENING_SKILLS

    def test_angel_opening_skill_picks_randomly_without_waiting(self):
        # 天使开局技能页不出现【选技能】图和主C技能图标，只出现天使标识 →
        # 无需等待识别帧，立即在三张技能卡中随机选一张
        task = make_task()
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._opening_loaded = True
        obs = Observation(frame_id=1, raw_data={"tian_shi_kai_ju_visible": True})
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "opening_angel_skill_fallback"
        assert decision[0].kind == "click"
        assert decision[0].verification == "next_frame"
        expected_points = roi_column_centers(
            task._skill_candidate_roi,
            make_window_ctx().client_size,
            columns=3,
        )
        assert decision[0].target in expected_points
        # 选后留在本状态，等天使标识消失（verify 通过）再继续开局流程
        assert decision[1] == State.SELECT_OPENING_SKILLS

    def test_angel_opening_skill_verifies_marker_gone(self):
        # 天使技能页没有【选技能】图和技能卡图标，以天使标识消失判定选完
        task = make_task()
        action = Action(kind="click", tag="opening_angel_skill_fallback", target=(1, 1))
        before = Observation(frame_id=1, raw_data={"tian_shi_kai_ju_visible": True})
        still_open = Observation(frame_id=2, raw_data={"tian_shi_kai_ju_visible": True})
        closed = Observation(frame_id=3)

        assert not task.verify_action(action, before, still_open)
        assert task.verify_action(action, before, closed)

    def test_skill_candidate_roi_missing_stops_conservatively(self):
        task = make_task(include_skill_candidate_roi=False)
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._opening_loaded = True
        task._opening_empty_checks = task._run_config.skill_recognition_frames
        obs = Observation(
            frame_id=1,
            raw_data={"select_skill_button_visible": True},
        )

        assert task.decide_action(obs, make_window_ctx()) is None

    def test_skill_fallback_verifies_page_closed(self):
        task = make_task()
        action = Action(kind="click", tag="opening_skill_fallback", target=(1, 1))
        before = Observation(
            frame_id=1,
            raw_data={"select_skill_button_visible": True},
        )
        after = Observation(frame_id=2)

        assert task.verify_action(action, before, after)

    def test_main_skill_priority_verifies_page_closed(self):
        # 局内选技能一组一选：选完页面收起即生效；未收起=点击未生效（如卡未
        # 渲染完时点击落空），交给重试。回归：此前比对 skill_card_options 混入
        # frame_id，前后帧必然不同，点击被吞（同组卡原样不动）也误判生效
        task = make_task()
        action = Action(kind="click", tag="main_skill_priority", target=(1, 1))

        def card_options(frame_id: int) -> list[dict]:
            return [
                {
                    "column": 0,
                    "name": "天穹支援",
                    "tier": 1,
                    "known": True,
                    "page": "SELECT_MAIN_C_SKILLS",
                    "frame_id": frame_id,
                    "position": (100, 200),
                }
            ]

        before = Observation(
            frame_id=1,
            raw_data={
                "select_skill_button_visible": True,
                "skill_card_options": card_options(1),
            },
        )
        swallowed = Observation(
            frame_id=2,
            raw_data={
                "select_skill_button_visible": True,
                "skill_card_options": card_options(2),
            },
        )
        light_still_open = Observation(
            frame_id=3, raw_data={"select_skill_button_visible": True}
        )
        closed = Observation(frame_id=4)

        assert not task.verify_action(action, before, swallowed)
        assert not task.verify_action(action, before, light_still_open)
        assert task.verify_action(action, before, closed)

    def test_merge_gift_skill_priority_verifies_banner_gone(self):
        # 赠送技能页一组一选：提示条消失即生效；未消失=未生效
        task = make_task()
        action = Action(kind="click", tag="merge_gift_skill_priority", target=(1, 1))
        before = Observation(frame_id=1, raw_data={"merge_gift_skill_page_visible": True})
        still_open = Observation(frame_id=2, raw_data={"merge_gift_skill_page_visible": True})
        closed = Observation(frame_id=3)

        assert not task.verify_action(action, before, still_open)
        assert task.verify_action(action, before, closed)

    def test_opening_skill_priority_compares_cards_without_frame_fields(self):
        # 开局选完一张会自动弹下一组：页面未收起时必须比对卡面区分新组与未
        # 点中。回归：比对剔除 frame_id 等帧相关字段——同一组卡重读（仅
        # frame_id 不同）不得判成"已变化"
        task = make_task()
        action = Action(kind="click", tag="opening_skill_priority", target=(1, 1))

        def card_option(frame_id: int, name: str, tier: int) -> dict:
            return {
                "column": 0,
                "name": name,
                "tier": tier,
                "known": True,
                "page": "SELECT_OPENING_SKILLS",
                "frame_id": frame_id,
                "position": (100, 200),
            }

        before = Observation(
            frame_id=1,
            raw_data={
                "select_skill_button_visible": True,
                "skill_card_options": [card_option(1, "天穹支援", 1)],
            },
        )
        same_group = Observation(
            frame_id=2,
            raw_data={
                "select_skill_button_visible": True,
                "skill_card_options": [card_option(2, "天穹支援", 1)],
            },
        )
        next_group = Observation(
            frame_id=3,
            raw_data={
                "select_skill_button_visible": True,
                "skill_card_options": [card_option(3, "星辰护盾", 2)],
            },
        )

        assert not task.verify_action(action, before, same_group)
        assert task.verify_action(action, before, next_group)

    def test_skill_gold_wait_on_red_cost(self):
        # 只认色金币门控：费用红色（游戏自身判买不起）→ 等
        # gold_red_wait_seconds（默认 1 秒）后重新识别再判
        task = make_task()
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        task._next_skill_at = 0.0
        obs = Observation(frame_id=1, raw_data={"gold": {"skill_red": True}})
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert action.tag == "skill_gold_wait"
        assert action.duration == 1.0
        assert to_state == State.SELECT_MAIN_C_SKILLS

    def test_white_cost_proceeds_to_select_skill_click(self):
        # 只认色金币门控：费用白色（红标志 False、无数值）→ 视为付得起，
        # 直接进入定时点选技能
        task = make_task()
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        task._next_skill_at = 0.0
        obs = Observation(frame_id=1, raw_data={"gold": {"skill_red": False}})
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        # 点击后指针停靠到 mouse_park，避免悬停高亮污染下一帧费用色判
        assert decision[0].tag == "open_main_skills"
        assert decision[0].hover_park is not None

    def test_summon_gold_wait_on_red_cost(self):
        # 召唤金币门控（只认色）：召唤费用红色 → 等 gold_red_wait_seconds
        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        board = make_board([make_hero("snow", star=1, pos=(100, 100))])
        obs = Observation(
            frame_id=1, board=board, raw_data={"gold": {"summon_red": True}}
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert action.tag == "summon_gold_wait"
        assert action.duration == 1.0
        assert to_state == State.BUILD_MAIN_C

    def test_summon_button_enters_summon_phase(self):
        # 【选技能】连续多帧消失、【召唤】可见 → 开局阶段结束，进入召唤
        task = make_task()
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._opening_loaded = True
        obs = Observation(frame_id=1, raw_data={"summon_button_visible": True})
        # 前 3 帧只是确认等待（页面可能只是两组技能卡之间的间隙）
        for frame_id in (1, 2, 3):
            decision = task.decide_action(
                Observation(frame_id=frame_id, raw_data={"summon_button_visible": True}),
                make_window_ctx(),
            )
            assert decision is not None
            assert decision[0].tag == "opening_exit_confirm_check"
            task.on_action_verified(*decision)
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert action.tag == "opening_complete"
        assert to_state == State.BUILD_MAIN_C

    def test_opening_page_gap_does_not_exit_prematurely(self):
        """选完一张卡后页面短暂关闭（下一组即将弹出）不能误判选完。"""
        task = make_task()
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._opening_loaded = True
        # 间隙：页面消失、召唤按钮露出 1 帧 → 只确认等待，不退出
        gap = task.decide_action(
            Observation(frame_id=1, raw_data={"summon_button_visible": True}),
            make_window_ctx(),
        )
        assert gap is not None
        assert gap[0].tag == "opening_exit_confirm_check"
        task.on_action_verified(*gap)

        # 下一组技能卡弹出 → 继续选技能，退出计数归零
        candidates = [SkillCandidate("assault", (100, 100), 0.9)]
        resumed = task.decide_action(
            Observation(frame_id=2, skill_candidates=candidates),
            make_window_ctx(),
        )
        assert resumed is not None
        assert resumed[0].tag == "opening_skill_candidate"
        task.on_action_verified(*resumed)
        assert task._opening_exit_empty_count == 0

    def test_opening_exits_immediately_after_max_selections(self):
        """开局技能最多选 3 次：选满后召唤按钮一出现即结束，不再等退出确认帧。"""
        task = make_task()
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._opening_loaded = True
        # 模拟已成功选满 3 次开局技能
        for _ in range(task._run_config.opening_skill_max_selections):
            task.on_action_verified(
                Action(kind="click", tag="opening_skill_candidate", target=(1, 1)),
                State.SELECT_OPENING_SKILLS,
            )
        assert task._opening_skill_selections == 3

        # 召唤按钮出现 → 第一帧就直接进入召唤阶段
        decision = task.decide_action(
            Observation(frame_id=1, raw_data={"summon_button_visible": True}),
            make_window_ctx(),
        )
        assert decision is not None
        action, to_state = decision
        assert action.tag == "opening_complete"
        assert to_state == State.BUILD_MAIN_C

    def test_opening_max_selections_resets_on_new_round(self):
        """新一轮对局（点准备）后开局选择计数归零。"""
        task = make_task()
        task.on_action_verified(
            Action(kind="click", tag="opening_skill_candidate", target=(1, 1)),
            State.SELECT_OPENING_SKILLS,
        )
        assert task._opening_skill_selections == 1
        task.on_action_verified(
            Action(kind="click", tag="ready_match", target=(1, 1)),
            State.SELECT_OPENING_SKILLS,
        )
        assert task._opening_skill_selections == 0

    def test_select_skill_button_takes_priority_over_summon(self):
        # 同时识别到选择技能和召唤按钮时，按「先选择技能」处理，进入选技能流程而非召唤
        task = make_task()
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._opening_loaded = True
        obs = Observation(
            frame_id=1,
            raw_data={
                "summon_button_visible": True,
                "select_skill_button_visible": True,
            },
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        # 页面已打开但未识别到技能卡 → 等待候选（不进召唤）
        assert decision[0].kind == "wait"
        assert decision[1] == State.SELECT_OPENING_SKILLS

    def test_return_increments_round_after_next_frame_verification(self):
        task = make_task(max_rounds=2)
        task.ctx.current_state = State.CHECK_ROUND_LIMIT
        match = MatchResult("buttons/fan_hui.png", (500, 1000), 0.9)
        before = Observation(
            frame_id=1,
            raw_data={"return_button_visible": True, "return_button_match": match},
        )
        decision = task.decide_action(before, make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert to_state == State.FIND_COOP
        assert task.verify_action(action, before, Observation(frame_id=2))
        task.on_action_verified(action, to_state)
        assert task.ctx.round_count == 1
        task.ctx.current_state = to_state
        next_decision = task.decide_action(Observation(frame_id=2), make_window_ctx())
        assert next_decision is not None
        assert next_decision[0].tag == "open_coop_chat"
        task.on_action_verified(*next_decision)

        open_refresh_decision = task.decide_action(Observation(frame_id=3), make_window_ctx())
        assert open_refresh_decision is not None
        assert open_refresh_decision[0].tag == "open_refresh_difficulty_dialog"
        task.on_action_verified(*open_refresh_decision)

        # 打开确认通过后进入刷新关闭步骤（同样先确认弹窗真的打开）
        advance_confirm_difficulty_open(task, 10, frame_id=4)
        assert task._recruit_step is _RecruitStep.CLOSE_REFRESH_DIFFICULTY_DIALOG

        # 弹窗仍打开 → 点击关闭；连续 2 次复核不可见 → 确认关闭
        close_refresh_decision = task.decide_action(
            difficulty_observation(10, frame_id=10), make_window_ctx()
        )
        assert close_refresh_decision is not None
        assert close_refresh_decision[0].tag == "close_refresh_difficulty_dialog"
        task.on_action_verified(*close_refresh_decision)

        for frame_id in range(11, 13):
            closed_step = task.decide_action(Observation(frame_id=frame_id), make_window_ctx())
            assert closed_step is not None
            assert closed_step[0].tag == "difficulty_close_miss_check"
            task.on_action_verified(*closed_step)

        closed_step = task.decide_action(Observation(frame_id=13), make_window_ctx())
        assert closed_step is not None
        assert closed_step[0].tag == "refresh_dialog_closed"
        task.on_action_verified(*closed_step)

        join_decision = task.decide_action(Observation(frame_id=14), make_window_ctx())
        assert join_decision is not None
        assert join_decision[0].kind == "grab_coop"
        assert join_decision[0].tag == "find_coop_grab"


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def to_state_of(decision):
    """从 decide_action 返回值取 to_state。"""
    assert decision is not None
    return decision[1]


# ---------------------------------------------------------------------------
# 培养阶段棋盘与合成日志
# ---------------------------------------------------------------------------


class TestBoardLogging:
    def test_board_contents_logged_when_changed(self, caplog):
        """棋盘内容变化时输出各格英雄，便于人工核对识别结果。"""
        import logging

        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        board = make_board(
            [
                make_hero("assault", star=1, pos=(100, 100)),
                make_hero("angel", star=2, pos=(200, 300)),
            ]
        )
        with caplog.at_level(logging.INFO, logger="wlxq_bot.tasks.coop"):
            decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "棋盘识别" in messages
        assert "assault1星@(100, 100)" in messages
        assert "angel2星@(200, 300)" in messages

    def test_board_log_not_repeated_for_unchanged_signature(self, caplog):
        """同一棋盘签名重复识别时不重复输出，避免高频刷屏。"""
        import logging

        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        board = make_board([make_hero("assault", star=1, pos=(100, 100))])
        with caplog.at_level(logging.INFO, logger="wlxq_bot.tasks.coop"):
            task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
            task.decide_action(Observation(frame_id=2, board=board), make_window_ctx())
        board_logs = [r for r in caplog.records if "棋盘识别" in r.getMessage()]
        assert len(board_logs) == 1

    def test_merge_action_logs_drag_from_to(self, caplog):
        """合成决策输出拖动起点和终点及英雄信息。"""
        import logging

        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        board = make_board(
            [
                make_hero("monkey", star=1, pos=(100, 100)),
                make_hero("monkey", star=1, pos=(200, 200)),
            ]
        )
        with caplog.at_level(logging.INFO, logger="wlxq_bot.tasks.coop"):
            decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "drag"
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "拖动合成非主C对: monkey1星 (200, 200) -> (100, 100)" in messages

    def test_close_difficulty_dialog_retries_while_candidates_visible(self):
        """弹窗没关住（【合作模式】标识仍可见）时留在关闭步骤继续点击，不进入抢合作。"""
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        # 走到关闭步骤：入口 → 打开确认 → 勾选 1、2
        advance_entry_to_difficulty_dialog(task)
        advance_confirm_difficulty_open(task, 2, 1, frame_id=4)
        for frame_id in (10, 11):
            decision = task.decide_action(
                difficulty_observation(2, 1, frame_id=frame_id), make_window_ctx()
            )
            assert decision is not None
            task.on_action_verified(*decision)

        # 连续多帧标识仍可见：每次都再次点击关闭，绝不发出 grab_coop
        for frame_id in range(12, 17):
            decision = task.decide_action(
                difficulty_observation(2, 1, frame_id=frame_id), make_window_ctx()
            )
            assert decision is not None
            assert decision[0].tag == "close_difficulty_dialog"
            task.on_action_verified(*decision)

        # 达到上限仍可见 → 保守停止（decide_action 返回 None）
        exhausted = task.decide_action(difficulty_observation(2, 1, frame_id=20), make_window_ctx())
        assert exhausted is None

    def test_close_difficulty_dialog_stops_after_budget_without_grab(self):
        """关闭预算耗尽时不进入抢合作（弹窗挡住 join_coop 会导致无效连点）。"""
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        advance_entry_to_difficulty_dialog(task)
        advance_confirm_difficulty_open(task, 2, 1, frame_id=4)
        for frame_id in (10, 11):
            decision = task.decide_action(
                difficulty_observation(2, 1, frame_id=frame_id), make_window_ctx()
            )
            assert decision is not None
            task.on_action_verified(*decision)

        # 默认 difficulty_close_max_attempts=5，全部失败
        for frame_id in range(12, 17):
            decision = task.decide_action(
                difficulty_observation(2, 1, frame_id=frame_id), make_window_ctx()
            )
            assert decision is not None
            assert decision[0].kind == "click"
            task.on_action_verified(*decision)

        assert (
            task.decide_action(difficulty_observation(2, 1, frame_id=99), make_window_ctx()) is None
        )

    def test_open_confirm_tolerates_slow_opening_dialog(self):
        """弹窗打开慢：settle 后先识别不到，随后出现并连续命中 → 确认打开，不误判。"""
        task = make_task(skip_difficulty_selection=True)
        task.ctx.current_state = State.FIND_COOP
        advance_entry_to_difficulty_dialog(task)

        # settle 固定等待
        settle = task.decide_action(Observation(frame_id=4), make_window_ctx())
        assert settle is not None
        assert settle[0].tag == "difficulty_open_settle"
        task.on_action_verified(*settle)

        # 弹窗还没弹开：连续未命中只累计，绝不误判打开、更不进入抢合作
        for frame_id in (5, 6, 7):
            decision = task.decide_action(Observation(frame_id=frame_id), make_window_ctx())
            assert decision is not None
            assert decision[0].tag == "difficulty_open_miss"
            task.on_action_verified(*decision)

        # 弹窗出现 → 连续 2 次命中 → 确认打开
        for frame_id in (8, 9):
            decision = task.decide_action(
                difficulty_observation(10, frame_id=frame_id), make_window_ctx()
            )
            assert decision is not None
            assert decision[0].tag == "difficulty_open_hit"
            task.on_action_verified(*decision)
        confirmed = task.decide_action(difficulty_observation(10, frame_id=10), make_window_ctx())
        assert confirmed is not None
        assert confirmed[0].tag == "difficulty_open_confirmed"
        task.on_action_verified(*confirmed)

        # 跳过模式：确认打开后直接进入关闭步骤
        follow = task.decide_action(difficulty_observation(10, frame_id=11), make_window_ctx())
        assert follow is not None
        assert follow[0].tag == "close_difficulty_dialog"

    def test_open_confirm_reclicks_then_stops_when_dialog_never_opens(self):
        """弹窗始终没打开：连续未命中达阈值 → 重点打开按钮；重试用尽 → 保守停止。"""
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        advance_entry_to_difficulty_dialog(task)

        cfg = task._run_config
        for _ in range(cfg.difficulty_open_max_reclicks):
            settle = task.decide_action(Observation(frame_id=4), make_window_ctx())
            assert settle is not None
            assert settle[0].tag == "difficulty_open_settle"
            task.on_action_verified(*settle)
            for _ in range(cfg.difficulty_open_fail_misses):
                decision = task.decide_action(Observation(frame_id=5), make_window_ctx())
                assert decision is not None
                assert decision[0].tag == "difficulty_open_miss"
                task.on_action_verified(*decision)
            retry = task.decide_action(Observation(frame_id=6), make_window_ctx())
            assert retry is not None
            assert retry[0].tag == "difficulty_open_retry"
            assert retry[0].kind == "click"
            task.on_action_verified(*retry)

        # 重试用尽仍打不开 → 保守停止（不带着「没打开」的假设进入后续步骤）
        settle = task.decide_action(Observation(frame_id=7), make_window_ctx())
        task.on_action_verified(*settle)
        for _ in range(cfg.difficulty_open_fail_misses):
            decision = task.decide_action(Observation(frame_id=8), make_window_ctx())
            assert decision is not None
            task.on_action_verified(*decision)
        assert task.decide_action(Observation(frame_id=9), make_window_ctx()) is None

    def test_close_difficulty_dialog_ignores_stale_candidates_when_marker_gone(self):
        """关闭验证以【合作模式】标识为准：标识消失后，残留的难度候选命中不再阻塞关闭。"""
        task = make_task(skip_difficulty_selection=True)
        task.ctx.current_state = State.FIND_COOP
        advance_entry_to_difficulty_dialog(task)
        advance_confirm_difficulty_open(task, 10, frame_id=4)

        # 弹窗打开（标识可见）→ 点击关闭
        close_click = task.decide_action(difficulty_observation(10, frame_id=10), make_window_ctx())
        assert close_click is not None
        assert close_click[0].tag == "close_difficulty_dialog"
        task.on_action_verified(*close_click)

        # 标识已消失、但难度候选模板仍残留命中 → 不阻塞：连续 2 次复核后确认关闭
        stale_candidates = [DifficultyCandidate(level=10, position=(300, 200), confidence=0.9)]
        for frame_id in range(11, 13):
            confirm = task.decide_action(
                Observation(frame_id=frame_id, difficulty_candidates=stale_candidates),
                make_window_ctx(),
            )
            assert confirm is not None
            assert confirm[0].tag == "difficulty_close_miss_check"
            task.on_action_verified(*confirm)

        closed = task.decide_action(
            Observation(frame_id=13, difficulty_candidates=stale_candidates),
            make_window_ctx(),
        )
        assert closed is not None
        assert closed[0].tag == "difficulty_dialog_closed"
        task.on_action_verified(*closed)

        grab = task.decide_action(Observation(frame_id=14), make_window_ctx())
        assert grab is not None
        assert grab[0].kind == "grab_coop"

    def test_logs_prefer_cell_name_over_pixel_position(self, caplog):
        """识别结果带格名时日志使用格名（如 1A），不再输出像素坐标。"""
        import logging

        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        board = make_board(
            [
                make_hero("assault", star=1, pos=(610, 908), cell_name="1A"),
                make_hero("assault", star=1, pos=(684, 982), cell_name="2B"),
            ]
        )
        with caplog.at_level(logging.INFO, logger="wlxq_bot.tasks.coop"):
            decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "drag"
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "assault1星@1A" in messages
        # 从下往上拖：2B（更靠下）是起点
        assert "拖动合成主C对: assault1星 2B -> 1A" in messages
        assert "(610, 908)" not in messages

    def test_claim_chest_waits_before_return(self):
        """领取宝箱后带稳定等待：返回按钮在结算动画播完前点击无响应。"""
        task = make_task()
        task.ctx.current_state = State.CLAIM_REWARD
        decision = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert action.tag == "claim_chest"
        assert to_state == State.CHECK_ROUND_LIMIT
        assert action.post_delay == pytest.approx(task._run_config.reward_claim_return_delay)


class TestModeProfileDispatch:
    """模式档案注入：页签步骤、模式聊天入口、结算分派。"""

    def _spots_with_tabs(self) -> dict[str, Hotspot]:
        spots = {
            "territory_nav": Hotspot(x_ratio=0.11, y_ratio=0.21),
            "open_mode_select": Hotspot(x_ratio=0.21, y_ratio=0.31),
            "recruit_coop_tab": Hotspot(x_ratio=0.25, y_ratio=0.35),
            "recruit_pet_tab": Hotspot(x_ratio=0.4, y_ratio=0.35),
            "open_difficulty_dialog": Hotspot(x_ratio=0.3, y_ratio=0.4),
            "difficulty_scroll_start": Hotspot(x_ratio=0.5, y_ratio=0.3),
            "difficulty_scroll_end": Hotspot(x_ratio=0.5, y_ratio=0.7),
            "close_difficulty_dialog": Hotspot(x_ratio=0.4, y_ratio=0.5),
            "coop_chat": Hotspot(x_ratio=0.5, y_ratio=0.6),
            "pet_chat": Hotspot(x_ratio=0.55, y_ratio=0.62),
            "add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96),
            "join_coop": Hotspot(x_ratio=0.6, y_ratio=0.75),
            "select_skill": Hotspot(x_ratio=0.2, y_ratio=0.9),
            "mouse_park": Hotspot(x_ratio=0.5, y_ratio=0.06),
            "like": Hotspot(x_ratio=0.3, y_ratio=0.8),
            "claim_chest": Hotspot(x_ratio=0.5, y_ratio=0.8),
        }
        return spots

    def test_build_strategy_default_and_unknown(self):
        from wlxq_bot.tasks.strategy import CultivationStrategy, build_strategy

        profile = make_mode_profile()
        strategy = build_strategy(profile, "assault", RunConfig(), None)
        assert isinstance(strategy, CultivationStrategy)
        bad = profile.model_copy(update={"strategy": "phased_assault"})
        with pytest.raises(ValueError):
            build_strategy(bad, "assault", RunConfig(), None)

    def test_coop_mode_clicks_mode_tab_before_difficulty_dialog(self):
        task = make_task(
            hotspots=self._spots_with_tabs(),
            mode_profile=make_mode_profile(),  # 合作档案：点 recruit_coop_tab
        )
        task.ctx.current_state = State.FIND_COOP
        tags = []
        for observation in territory_entry_observations(frame_id=1) + [
            Observation(frame_id=7),   # open_challenge_chat（挑战页聊天）
            Observation(frame_id=8),   # open_mode_tab（页签 hotspot 路径）
            Observation(frame_id=9),   # open_difficulty_dialog
            Observation(frame_id=10),  # settle
            difficulty_observation(2, 1, frame_id=11),
            difficulty_observation(2, 1, frame_id=12),
            difficulty_observation(2, 1, frame_id=13),
        ]:
            decision = task.decide_action(observation, make_window_ctx())
            assert decision is not None
            action, to_state = decision
            tags.append(action.tag)
            task.on_action_verified(action, to_state)
        assert tags == [
            "open_territory",
            "territory_open_confirmed",
            "open_mode_select",
            "mode_select_open_confirmed",
            "mode_card_selected",
            "challenge_page_confirmed",
            "open_challenge_chat",
            "open_mode_tab",
            "open_difficulty_dialog",
            "difficulty_open_settle",
            "difficulty_open_hit",
            "difficulty_open_hit",
            "difficulty_open_confirmed",
        ]

    def test_mode_tab_hotspot_missing_stops_conservatively(self):
        # 档案要求点页签但 tasks.yaml 缺该 hotspot：无可用动作 → decide 返回 None
        # （调度层按保守停止计数，绝不跳过页签直接抢）
        task = make_task(mode_profile=make_mode_profile())  # make_task 默认 spots 无 recruit_coop_tab
        task.ctx.current_state = State.FIND_COOP
        advance_territory_entry_to_recruit(task)
        assert task._recruit_step == _RecruitStep.SELECT_MODE_TAB
        assert task.decide_action(Observation(frame_id=8), make_window_ctx()) is None

    # ------------------------------------------------------------------
    # 页签 OCR 定位（mode_tab_text 配置后替代固定 hotspot 坐标）
    # ------------------------------------------------------------------

    def _task_on_recruit_page(self, mode_profile: ModeProfile) -> CoopTask:
        """构建已驱动到 SELECT_MODE_TAB 步骤的任务（领地入口 → 聊天已完成）。"""
        task = make_task(hotspots=self._spots_with_tabs(), mode_profile=mode_profile)
        task.ctx.current_state = State.FIND_COOP
        advance_territory_entry_to_recruit(task)
        assert task._recruit_step == _RecruitStep.SELECT_MODE_TAB
        return task

    def test_mode_tab_ocr_clicks_text_center(self):
        task = self._task_on_recruit_page(
            make_mode_profile(mode_tab_text="合作模式"),
        )
        assert task.observation_mode() == "mode_tab"
        observation = Observation(
            frame_id=3,
            raw_data={
                "mode_tab_rows": [
                    ModeTabRow(text="合作模式", confidence=0.71, position=(314, 1535)),
                    ModeTabRow(text="保卫宠物", confidence=0.75, position=(523, 1535)),
                    ModeTabRow(text="共战灭尊", confidence=0.77, position=(733, 1535)),
                ]
            },
        )
        decision = task.decide_action(observation, make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert action.tag == "open_mode_tab"
        assert action.kind == "click"
        assert action.target == (314, 1535)
        assert to_state == State.FIND_COOP

    def test_mode_tab_ocr_substring_match_prefers_confidence(self):
        task = self._task_on_recruit_page(
            make_mode_profile(mode_tab_text="保卫宠物"),
        )
        observation = Observation(
            frame_id=3,
            raw_data={
                "mode_tab_rows": [
                    # 子串目标命中多行时取置信度最高者（如皮肤换色产生重影识别）
                    ModeTabRow(text="保卫宠物", confidence=0.75, position=(523, 1535)),
                    ModeTabRow(text="保卫宠", confidence=0.60, position=(520, 1540)),
                    ModeTabRow(text="共战灭尊", confidence=0.90, position=(733, 1535)),
                ]
            },
        )
        decision = task.decide_action(observation, make_window_ctx())
        assert decision is not None
        assert decision[0].target == (523, 1535)

    def test_mode_tab_ocr_miss_waits_then_stops_conservatively(self):
        task = self._task_on_recruit_page(
            make_mode_profile(mode_tab_text="保卫宠物"),
        )
        task._mode_tab_miss_limit = 3
        # 未识别到页签文字：不盲点固定坐标，返回 None 等待下一帧
        for frame_id in (3, 4):
            assert task.decide_action(Observation(frame_id=frame_id), make_window_ctx()) is None
        # 连续 miss 达上限：报错停止（大概率不在招募页，任何点击都不安全）
        with pytest.raises(RuntimeError, match="保卫宠物"):
            task.decide_action(Observation(frame_id=5), make_window_ctx())

    def test_mode_tab_ocr_hit_resets_miss_counter(self):
        task = self._task_on_recruit_page(
            make_mode_profile(mode_tab_text="合作模式"),
        )
        task._mode_tab_miss_limit = 2
        miss = Observation(frame_id=3, raw_data={"mode_tab_rows": []})
        assert task.decide_action(miss, make_window_ctx()) is None
        hit = Observation(
            frame_id=4,
            raw_data={
                "mode_tab_rows": [ModeTabRow(text="合作模式", confidence=0.7, position=(314, 1535))]
            },
        )
        decision = task.decide_action(hit, make_window_ctx())
        assert decision is not None and decision[0].target == (314, 1535)
        assert task._mode_tab_misses == 0

    def test_mode_tab_hotspot_path_used_when_text_unset(self):
        # 未配置 mode_tab_text：保持旧行为（固定 hotspot 坐标），观察提示为空
        task = self._task_on_recruit_page(make_mode_profile(mode_tab_text=None))
        assert task.observation_mode() is None
        decision = task.decide_action(Observation(frame_id=3), make_window_ctx())
        assert decision is not None
        action, _ = decision
        assert action.tag == "open_mode_tab"
        # recruit_coop_tab (0.25, 0.35) → 客户区像素
        assert action.target == (
            int(0.25 * make_window_ctx().client_size[0]),
            int(0.35 * make_window_ctx().client_size[1]),
        )

    def test_next_round_refresh_clicks_mode_tab_after_chat(self):
        task = make_task(
            max_rounds=2,
            hotspots=self._spots_with_tabs(),
            mode_profile=make_mode_profile(),
        )
        task.ctx.current_state = State.CHECK_ROUND_LIMIT
        match = MatchResult("buttons/fan_hui.png", (500, 1000), 0.9)
        before = Observation(
            frame_id=1,
            raw_data={"return_button_visible": True, "return_button_match": match},
        )
        decision = task.decide_action(before, make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert to_state == State.FIND_COOP
        task.on_action_verified(action, to_state)
        task.ctx.round_count = 1
        task.ctx.current_state = to_state

        chat = task.decide_action(Observation(frame_id=2), make_window_ctx())
        assert chat is not None and chat[0].tag == "open_coop_chat"
        task.on_action_verified(*chat)
        tab = task.decide_action(Observation(frame_id=3), make_window_ctx())
        assert tab is not None and tab[0].tag == "open_mode_tab"
        task.on_action_verified(*tab)
        dialog = task.decide_action(Observation(frame_id=4), make_window_ctx())
        assert dialog is not None and dialog[0].tag == "open_refresh_difficulty_dialog"

    def test_return_action_requests_settlement_screenshot(self):
        """点击返回的动作带 save_frame_before：调度层在该动作执行前保存结算画面。"""
        task = make_task()
        task.ctx.current_state = State.CHECK_ROUND_LIMIT
        match = MatchResult("buttons/fan_hui.png", (500, 1000), 0.9)
        observation = Observation(
            frame_id=1,
            raw_data={"return_button_visible": True, "return_button_match": match},
        )

        decision = task.decide_action(observation, make_window_ctx())

        assert decision is not None
        action, _ = decision
        assert action.tag == "return_result"
        assert action.save_frame_before is True

    def test_pet_mode_uses_pet_chat_hotspot(self):
        task = make_task(
            hotspots=self._spots_with_tabs(),
            mode_profile=make_mode_profile(
                mode_tab_hotspot="recruit_pet_tab",
                mode_chat_hotspot="pet_chat",
            ),
        )
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.OPEN_COOP_CHAT
        decision = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert decision is not None
        action, _ = decision
        assert action.tag == "open_coop_chat"
        assert action.target == (int(0.55 * 923), int(0.62 * 1723))  # pet_chat 中心

    def test_pet_settlement_skips_like_and_chest(self):
        task = make_task(
            hotspots=self._spots_with_tabs(),
            mode_profile=make_mode_profile(like_chest=False),
        )
        task.ctx.current_state = State.HANDLE_RESULT
        like = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert like is not None
        assert like[0].tag == "like_skipped"
        assert like[1] == State.CLAIM_REWARD
        task.ctx.current_state = State.CLAIM_REWARD
        claim = task.decide_action(Observation(frame_id=2), make_window_ctx())
        assert claim is not None
        assert claim[0].tag == "claim_skipped"
        assert claim[1] == State.CHECK_ROUND_LIMIT

    def test_coop_mode_keeps_like_and_chest(self):
        task = make_task(
            hotspots=self._spots_with_tabs(),
            mode_profile=make_mode_profile(like_chest=True),
        )
        task.ctx.current_state = State.HANDLE_RESULT
        like = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert like is not None and like[0].tag == "like_result"

    def _pet_task(
        self,
        targets: list[str] | None = None,
        difficulty_selection_only: bool = False,
    ) -> CoopTask:
        return make_task(
            hotspots=self._spots_with_tabs(),
            difficulty_selection_only=difficulty_selection_only,
            mode_profile=make_mode_profile(
                mode_tab_hotspot="recruit_pet_tab",
                mode_chat_hotspot="pet_chat",
                difficulty_kind="ocr",
                difficulty_targets=targets if targets is not None else ["普通", "中等"],
                like_chest=False,
                ready_button="pet_ready_button",
                leave_team="pet_leave_team",
                return_button="pet_return_button",
            ),
            pet_difficulty_targets=targets if targets is not None else ["普通", "中等"],
        )

    @staticmethod
    def _pet_observation(
        rows: list[tuple[str, float]], frame_id: int
    ) -> Observation:
        from wlxq_bot.perception.pet_difficulty import DifficultyRow

        return Observation(
            frame_id=frame_id,
            raw_data={
                "pet_difficulty_rows": [
                    DifficultyRow(text=text, confidence=conf, position=(100 * i, 200 * i))
                    for i, (text, conf) in enumerate(rows, start=1)
                ],
                "difficulty_dialog_visible": True,
            },
        )

    def test_pet_difficulty_clicks_targets_in_configured_order(self):
        task = self._pet_task(["普通", "中等"])
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.SELECT_DIFFICULTIES
        clicks = []
        for i, rows in enumerate(
            [
                [("保卫宠物-普通", 0.85), ("保卫宠物-中等", 0.8)],
                # 勾选后两行仍在列表中（行位置不变）：中等行保持 (200, 400)
                [("保卫宠物-普通", 0.85), ("保卫宠物-中等", 0.8)],
            ],
            start=1,
        ):
            decision = task.decide_action(
                self._pet_observation(rows, frame_id=i), make_window_ctx()
            )
            assert decision is not None
            action, to_state = decision
            assert to_state == State.FIND_COOP
            clicks.append((action.tag, action.target))
            task.on_action_verified(action, to_state)
        assert clicks == [
            ("select_pet_difficulty:普通", (100, 200)),
            ("select_pet_difficulty:中等", (200, 400)),
        ]
        # 选完 → 进入关闭弹窗步骤
        assert task._recruit_step == _RecruitStep.CLOSE_DIFFICULTY_DIALOG

    def test_pet_difficulty_prefers_higher_confidence_row(self):
        task = self._pet_task(["普通"])
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.SELECT_DIFFICULTIES
        decision = task.decide_action(
            self._pet_observation(
                [("保卫宠物-普通", 0.7), ("保卫宠物-普通", 0.95)], frame_id=1
            ),
            make_window_ctx(),
        )
        assert decision is not None
        assert decision[0].target == (200, 400)  # 置信度 0.95 的行

    def test_pet_difficulty_miss_budget_stops_conservatively(self):
        task = self._pet_task(["普通"])
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.SELECT_DIFFICULTIES
        budget = task._run_config.difficulty_open_fail_misses
        with pytest.raises(RuntimeError, match="难度行持续不可见"):
            for i in range(budget + 1):
                decision = task.decide_action(
                    self._pet_observation([], frame_id=i + 1), make_window_ctx()
                )
                assert decision is not None
                assert decision[0].tag == "pet_difficulty_miss"
                task.on_action_verified(*decision)

    def test_pet_difficulty_selection_only_completes(self):
        task = self._pet_task(["普通"], difficulty_selection_only=True)
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.SELECT_DIFFICULTIES
        decision = task.decide_action(
            self._pet_observation([("保卫宠物-普通", 0.85)], frame_id=1),
            make_window_ctx(),
        )
        assert decision is not None
        task.on_action_verified(*decision)
        final = task.decide_action(
            self._pet_observation([("保卫宠物-普通", 0.85)], frame_id=2),
            make_window_ctx(),
        )
        assert final is not None
        assert final[0].tag == "pet_difficulty_selection_complete"
        assert final[1] == State.COMPLETED

    def test_skip_difficulty_selection_applies_to_pet(self):
        # 已选过难度（_difficulty_done_session）→ 确认打开后直接关闭，不进勾选
        task = self._pet_task(["普通"])
        task._difficulty_done_session = True
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.SELECT_DIFFICULTIES
        task._difficulty_open_is_refresh = False
        # 模拟 difficulty_open_confirmed 的既有分流逻辑（在 on_action_verified 中）
        task._recruit_step = _RecruitStep.CONFIRM_DIFFICULTY_OPEN
        task.on_action_verified(
            Action(kind="wait", duration=0.0, verification="immediate", tag="difficulty_open_confirmed"),
            State.FIND_COOP,
        )
        assert task._recruit_step == _RecruitStep.CLOSE_DIFFICULTY_DIALOG


class TestTerritoryEntry:
    """领地入口链（2026-09-09 新流程）：导航判定、选卡、挑战页确认、超时。"""

    def test_open_territory_clicks_when_nav_reads_lingdi(self):
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        decision = task.decide_action(
            Observation(frame_id=1, raw_data={"territory_nav_text": "领地"}),
            make_window_ctx(),
        )
        assert decision is not None
        action, to_state = decision
        assert action.tag == "open_territory"
        assert action.kind == "click"
        # territory_nav (0.11, 0.21) → 客户区像素
        assert action.target == (
            int(0.11 * make_window_ctx().client_size[0]),
            int(0.21 * make_window_ctx().client_size[1]),
        )
        assert to_state == State.FIND_COOP
        task.on_action_verified(*decision)
        assert task._recruit_step == _RecruitStep.CONFIRM_TERRITORY_OPEN

    def test_open_territory_skips_when_already_open(self):
        """导航已读「返回」=领地页已打开：跳过点击与等待直进模式选择。"""
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        decision = task.decide_action(
            Observation(frame_id=1, raw_data={"territory_nav_text": "返回"}),
            make_window_ctx(),
        )
        assert decision is not None
        assert decision[0].tag == "territory_open_confirmed"
        assert decision[0].kind == "wait"
        task.on_action_verified(*decision)
        assert task._recruit_step == _RecruitStep.OPEN_MODE_SELECT

    def test_open_territory_unreadable_waits_without_click(self):
        """导航文字读不到：不盲点，等待下一帧复识。"""
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        decision = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "territory_nav_pending"
        assert decision[0].kind == "wait"

    def test_territory_confirm_waits_for_fanhui_then_advances(self):
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.CONFIRM_TERRITORY_OPEN
        pending = task.decide_action(
            Observation(frame_id=1, raw_data={"territory_nav_text": "领地"}),
            make_window_ctx(),
        )
        assert pending is not None and pending[0].tag == "territory_open_pending"
        confirmed = task.decide_action(
            Observation(frame_id=2, raw_data={"territory_nav_text": "返回"}),
            make_window_ctx(),
        )
        assert confirmed is not None and confirmed[0].tag == "territory_open_confirmed"
        task.on_action_verified(*confirmed)
        assert task._recruit_step == _RecruitStep.OPEN_MODE_SELECT

    def test_select_mode_card_clicks_text_center(self):
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.SELECT_MODE_CARD
        observation = Observation(
            frame_id=1,
            raw_data={"mode_card_rows": [ModeTabRow("合作模式", 0.9, (250, 1174))]},
        )
        decision = task.decide_action(observation, make_window_ctx())
        assert decision is not None
        action, _ = decision
        assert action.tag == "mode_card_selected"
        assert action.kind == "click"
        assert action.target == (250, 1174)

    def test_confirm_challenge_advances_to_chat_then_tab(self):
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.CONFIRM_CHALLENGE_PAGE
        decision = task.decide_action(
            Observation(frame_id=1, raw_data={"challenge_text": "开始挑战"}),
            make_window_ctx(),
        )
        assert decision is not None and decision[0].tag == "challenge_page_confirmed"
        task.on_action_verified(*decision)
        assert task._recruit_step == _RecruitStep.OPEN_CHALLENGE_CHAT
        # 聊天点击验证后进页签步骤，且页签去向是首局难度弹窗
        chat = task.decide_action(Observation(frame_id=2), make_window_ctx())
        assert chat is not None and chat[0].tag == "open_challenge_chat"
        task.on_action_verified(*chat)
        assert task._recruit_step == _RecruitStep.SELECT_MODE_TAB
        assert task._after_tab_step == _RecruitStep.OPEN_DIFFICULTY_DIALOG

    def test_weekly_buff_full_path_click_then_advance(self):
        """配置了每周弹窗识别：挑战页确认先进关闭步骤，命中点文字框中心，
        弹窗消失后经宽限帧放行进招募（旧行为：直接进聊天步骤）。"""
        task = make_task(
            mode_profile=make_mode_profile(weekly_buff_text="知道")
        )
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.CONFIRM_CHALLENGE_PAGE
        decision = task.decide_action(
            Observation(frame_id=1, raw_data={"challenge_text": "开始挑战"}),
            make_window_ctx(),
        )
        assert decision is not None and decision[0].tag == "challenge_page_confirmed"
        task.on_action_verified(*decision)
        assert task._recruit_step == _RecruitStep.CLOSE_WEEKLY_BUFF

        # 弹窗在：点「知道」文字框中心（开环），步骤停留等弹窗关闭
        popup = Observation(
            frame_id=2, raw_data={"weekly_buff_rows": [ModeTabRow("知道", 0.665, (452, 1306))]}
        )
        decision = task.decide_action(popup, make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "click"
        assert decision[0].tag == "close_weekly_buff"
        assert decision[0].target == (452, 1306)
        task.on_action_verified(*decision)
        assert task._recruit_step == _RecruitStep.CLOSE_WEEKLY_BUFF

        # 点击后弹窗消失：宽限帧内继续等（弹窗可能延迟弹出/关闭有过渡帧），
        # 宽限耗尽后放行进招募聊天步骤
        empty = Observation(frame_id=3, raw_data={"weekly_buff_rows": []})
        pending = task.decide_action(empty, make_window_ctx())
        assert pending is not None and pending[0].tag == "weekly_buff_pending"
        task.on_action_verified(*pending)
        absent = task.decide_action(empty, make_window_ctx())
        assert absent is not None and absent[0].tag == "weekly_buff_absent"
        task.on_action_verified(*absent)
        assert task._recruit_step == _RecruitStep.OPEN_CHALLENGE_CHAT

    def test_weekly_buff_absent_advances_after_grace(self):
        """无弹窗（非首次进入）：连续宽限帧读不到按钮文字，直接放行进招募。"""
        task = make_task(
            mode_profile=make_mode_profile(weekly_buff_text="知道")
        )
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.CLOSE_WEEKLY_BUFF
        empty = Observation(frame_id=1, raw_data={"weekly_buff_rows": []})
        pending = task.decide_action(empty, make_window_ctx())
        assert pending is not None and pending[0].tag == "weekly_buff_pending"
        task.on_action_verified(*pending)
        absent = task.decide_action(empty, make_window_ctx())
        assert absent is not None and absent[0].tag == "weekly_buff_absent"
        task.on_action_verified(*absent)
        assert task._recruit_step == _RecruitStep.OPEN_CHALLENGE_CHAT

    def test_weekly_buff_ignores_unrelated_rows(self):
        """识别区读到无关文字（如弹窗缺席时的背景「刷新」）：不算命中。"""
        task = make_task(
            mode_profile=make_mode_profile(weekly_buff_text="知道")
        )
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.CLOSE_WEEKLY_BUFF
        noisy = Observation(
            frame_id=1, raw_data={"weekly_buff_rows": [ModeTabRow("刷新", 0.56, (400, 1300))]}
        )
        decision = task.decide_action(noisy, make_window_ctx())
        assert decision is not None and decision[0].tag == "weekly_buff_pending"

    def test_weekly_buff_timeout_stops_conservatively(self):
        """弹窗持续可见（点击未生效）超过入口等待上限：保守停止不无限重点。"""
        task = make_task(
            mode_profile=make_mode_profile(weekly_buff_text="知道")
        )
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.CLOSE_WEEKLY_BUFF
        task._entry_wait_started_at = 0.0
        task._clock = lambda: task._entry_wait_timeout_seconds + 1.0
        popup = Observation(
            frame_id=1, raw_data={"weekly_buff_rows": [ModeTabRow("知道", 0.665, (452, 1306))]}
        )
        assert task.decide_action(popup, make_window_ctx()) is None

    def test_weekly_buff_unconfigured_mode_skips_step(self):
        """未配置 weekly_buff_text（合作模式）：挑战页确认后直接进聊天步骤。"""
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.CONFIRM_CHALLENGE_PAGE
        decision = task.decide_action(
            Observation(frame_id=1, raw_data={"challenge_text": "开始挑战"}),
            make_window_ctx(),
        )
        assert decision is not None and decision[0].tag == "challenge_page_confirmed"
        task.on_action_verified(*decision)
        assert task._recruit_step == _RecruitStep.OPEN_CHALLENGE_CHAT

    def test_entry_wait_timeout_stops_conservatively(self):
        """等待超上限：保守停止（decide 返回 None），不盲点不无限重试。"""
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.CONFIRM_TERRITORY_OPEN
        task._entry_wait_started_at = 0.0
        task._clock = lambda: task._entry_wait_timeout_seconds + 1.0
        assert task.decide_action(
            Observation(frame_id=1, raw_data={"territory_nav_text": "领地"}),
            make_window_ctx(),
        ) is None

    def test_entry_wait_within_timeout_keeps_polling(self):
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.CONFIRM_TERRITORY_OPEN
        task._entry_wait_started_at = 0.0
        task._clock = lambda: 1.0
        decision = task.decide_action(
            Observation(frame_id=1, raw_data={"territory_nav_text": "领地"}),
            make_window_ctx(),
        )
        assert decision is not None and decision[0].tag == "territory_open_pending"

    def test_territory_confirm_reclicks_when_still_lingdi(self):
        """点击被吞（实机 2026-09-12）：导航连续多帧仍读「领地」时重点领地。

        领地没打开时重点按钮是安全的（不会误点「返回」退出）；重点后
        misses 与等待 deadline 重新起算。
        """
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.CONFIRM_TERRITORY_OPEN
        lingdi = Observation(frame_id=1, raw_data={"territory_nav_text": "领地"})
        for _ in range(_ENTRY_RECLICK_MISSES - 1):
            decision = task.decide_action(lingdi, make_window_ctx())
            assert decision is not None and decision[0].tag == "territory_open_pending"
        decision = task.decide_action(lingdi, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "open_territory"
        assert decision[0].kind == "click"
        task.on_action_verified(*decision)
        assert task._recruit_step == _RecruitStep.CONFIRM_TERRITORY_OPEN
        assert task._entry_reclicks == 1
        assert task._entry_misses == 0

    def test_territory_confirm_stops_after_reclicks_exhausted(self):
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.CONFIRM_TERRITORY_OPEN
        task._entry_reclicks = task._entry_max_reclicks
        task._entry_misses = _ENTRY_RECLICK_MISSES - 1
        lingdi = Observation(frame_id=1, raw_data={"territory_nav_text": "领地"})
        assert task.decide_action(lingdi, make_window_ctx()) is None

    def test_territory_confirm_no_reclick_when_nav_unreadable(self):
        """导航读不到文字时不重点（可能已在领地页，重点会误点「返回」）。"""
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.CONFIRM_TERRITORY_OPEN
        for frame_id in range(1, _ENTRY_RECLICK_MISSES + 5):
            decision = task.decide_action(
                Observation(frame_id=frame_id), make_window_ctx()
            )
            assert decision is not None
            assert decision[0].tag == "territory_open_pending"
        assert task._entry_reclicks == 0

    def test_mode_select_confirm_advances_on_card_rows_even_if_title_missed(self):
        """标题漏读但本模式卡名可读：弹窗必然已打开，直接放行选卡。"""
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.CONFIRM_MODE_SELECT_OPEN
        decision = task.decide_action(
            Observation(
                frame_id=1,
                raw_data={"mode_card_rows": [ModeTabRow("合作模式", 0.7, (250, 600))]},
            ),
            make_window_ctx(),
        )
        assert decision is not None and decision[0].tag == "mode_select_open_confirmed"
        task.on_action_verified(*decision)
        assert task._recruit_step == _RecruitStep.SELECT_MODE_CARD

    def test_challenge_confirm_returns_to_card_while_popup_visible(self):
        """弹窗仍打开=选卡点击未生效：连续多帧后回选卡步骤重读重点。"""
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.CONFIRM_CHALLENGE_PAGE
        popup = Observation(
            frame_id=1,
            raw_data={"mode_select_visible": True, "challenge_text": ""},
        )
        for _ in range(_ENTRY_RECLICK_MISSES - 1):
            decision = task.decide_action(popup, make_window_ctx())
            assert decision is not None and decision[0].tag == "challenge_page_pending"
        decision = task.decide_action(popup, make_window_ctx())
        assert decision is not None and decision[0].tag == "challenge_page_pending"
        task.on_action_verified(*decision)
        assert task._recruit_step == _RecruitStep.SELECT_MODE_CARD
        assert task._entry_reclicks == 1

    def test_challenge_confirm_keeps_polling_when_popup_gone(self):
        """弹窗已关闭时不回选卡（可能已在挑战页，重点会点到空白处）。"""
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.CONFIRM_CHALLENGE_PAGE
        for frame_id in range(1, _ENTRY_RECLICK_MISSES + 5):
            decision = task.decide_action(
                Observation(frame_id=frame_id, raw_data={"challenge_text": ""}),
                make_window_ctx(),
            )
            assert decision is not None
            assert decision[0].tag == "challenge_page_pending"
        assert task._recruit_step == _RecruitStep.CONFIRM_CHALLENGE_PAGE
        assert task._entry_reclicks == 0

    def test_missing_card_text_stops_conservatively(self):
        """模式档案缺卡片文字：无法选卡，保守停止而不是盲点。"""
        task = make_task(mode_profile=make_mode_profile(mode_card_text=None))
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.SELECT_MODE_CARD
        assert task.decide_action(Observation(frame_id=1), make_window_ctx()) is None


# ---------------------------------------------------------------------------
# 批次召唤 / 批次合成（2026-09-10 用户策略）
# ---------------------------------------------------------------------------


def make_full_board(heroes: list[BoardHero]) -> BoardSnapshot:
    """棋盘占满的快照：total_slots = 占用数，available = 0。"""
    return BoardSnapshot(
        frame_id=1,
        heroes=heroes,
        capacity=BoardCapacity(total_slots=len(heroes), occupied=len(heroes)),
    )


class TestBatchSummon:
    def _task_after_forced_phase(self, minimum_summons: int = 2) -> CoopTask:
        task = make_task(minimum_summons=minimum_summons, batch_enabled=True)
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = minimum_summons
        return task

    def test_batch_summon_fills_empties_before_merging(self):
        """新默认优先级（2026-09-10）：棋盘有空位时先批次召唤填满，再合并。"""
        task = self._task_after_forced_phase()
        board = make_board(
            [
                make_hero("snow", star=1, pos=(100, 100)),
                make_hero("snow", star=1, pos=(200, 200)),
            ]
        )
        obs = Observation(frame_id=1, board=board)
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        # 有空位也有合法对：先批次召唤填满（批后再识别、再合并）
        assert decision[0].tag == "batch_summon"

    def test_batch_summon_planned_by_empty_cells(self):
        """培养阶段识别后按空格数规划批次召唤，点击发送即消耗批次。"""
        task = self._task_after_forced_phase()
        board = make_board([make_hero("snow", star=1, pos=(100, 100))])
        obs = Observation(frame_id=1, board=board)

        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        action, _ = decision
        assert action.tag == "batch_summon"
        assert action.verification == "immediate"
        assert task._batch_summon_remaining == 11  # 12 格 - 1 占用

        task.on_action_verified(*decision)
        assert task._batch_summon_remaining == 10
        assert task._summon_count == 3
        # 批次进行中不做多帧棋盘识别
        assert task.wants_board_watch() is False

    def test_batch_summon_non_last_clicks_use_fast_interval(self):
        task = self._task_after_forced_phase()
        board = make_board([make_hero("snow", star=1, pos=(100, 100))])
        obs = Observation(frame_id=1, board=board)
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].post_delay == task._run_config.forced_summon_interval_seconds

    def test_batch_summon_last_click_keeps_stable_delay(self):
        task = self._task_after_forced_phase()
        task._batch_summon_remaining = 1
        obs = Observation(frame_id=1, board=make_board([]))
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert 1.0 <= decision[0].post_delay <= 2.0

    def test_batch_summon_waits_when_gold_insufficient(self):
        """批内每次点击前金币门控：不足时等待且不消耗批次。"""
        task = self._task_after_forced_phase()
        task._batch_summon_remaining = 3
        obs = Observation(
            frame_id=1,
            board=make_board([]),
            raw_data={"gold": {"current": 0, "summon_cost": 80, "summon_red": True}},
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        action, _ = decision
        assert action.tag == "summon_gold_wait"
        assert task._batch_summon_remaining == 3

    def test_batch_summon_gold_wait_merges_first(self):
        """金币不足不空等（2026-09-13）：等待恢复期间优先合成（合成不花金币）。"""
        task = self._task_after_forced_phase()
        task._batch_summon_remaining = 3
        board = make_board(
            [
                make_hero("snow", star=1, pos=(100, 100)),
                make_hero("snow", star=1, pos=(200, 200)),
            ]
        )
        obs = Observation(
            frame_id=1,
            board=board,
            raw_data={"gold": {"current": 0, "summon_cost": 80, "summon_red": True}},
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        action, _ = decision
        assert action.tag == "merge_heroes"
        assert action.kind == "drag"
        assert action.verification == "next_frame"
        assert "等待恢复期间合成" in action.reason
        # 批次不消耗：合成腾出的空格由批后下一轮规划补召唤
        assert task._batch_summon_remaining == 3

    def test_batch_summon_gold_wait_waits_without_pairs(self):
        """无合法合成对时退回原地等待（原行为）。"""
        task = self._task_after_forced_phase()
        task._batch_summon_remaining = 3
        obs = Observation(
            frame_id=1,
            board=make_board([make_hero("snow", star=1, pos=(100, 100))]),
            raw_data={"gold": {"current": 0, "summon_cost": 80, "summon_red": True}},
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "summon_gold_wait"
        assert task._batch_summon_remaining == 3

    def test_gold_wait_after_batch_merges_first(self):
        """批次发完金币转红走培养门控（2026-09-13 实机）：等待期先合成。

        实机日志：批次最后一次召唤买空金币，下一轮批次已归零、落入培养
        金币门控，棋盘占满且有可合对时原实现只会干等金币恢复。
        """
        task = self._task_after_forced_phase()
        task._batch_summon_remaining = 0
        board = make_full_board(
            [
                make_hero("snow", star=1, pos=(100, 100)),
                make_hero("snow", star=1, pos=(200, 200)),
            ]
        )
        obs = Observation(
            frame_id=1,
            board=board,
            raw_data={"gold": {"current": 0, "summon_cost": 80, "summon_red": True}},
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        action, _ = decision
        assert action.tag == "merge_heroes"
        assert action.kind == "drag"
        assert action.verification == "next_frame"
        assert "召唤金币不足" in action.reason
        assert "等待恢复期间合成" in action.reason

    def test_gold_wait_with_empty_slots_merges_first(self):
        """棋盘未满但金币不足（非回补）：不再等金币召唤，先合成。"""
        task = self._task_after_forced_phase()
        task._batch_summon_remaining = 0
        board = make_board(
            [
                make_hero("snow", star=1, pos=(100, 100)),
                make_hero("snow", star=1, pos=(200, 200)),
            ]
        )
        obs = Observation(
            frame_id=1,
            board=board,
            raw_data={"gold": {"current": 0, "summon_cost": 80, "summon_red": True}},
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "merge_heroes"

    def test_gold_wait_skips_merge_when_board_ready(self):
        """棋盘已达标时等待期不合成：金币恢复后直接转选技能，别再动棋盘。"""
        task = self._task_after_forced_phase()
        task._batch_summon_remaining = 0
        board = make_full_board(
            [
                make_hero("assault", star=2, pos=(100, 100)),
                make_hero("snow", star=1, pos=(200, 200)),
                make_hero("snow", star=1, pos=(300, 300)),
            ]
        )
        obs = Observation(
            frame_id=1,
            board=board,
            raw_data={"gold": {"current": 0, "summon_cost": 80, "summon_red": True}},
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "summon_gold_wait"

    def test_gold_wait_topup_with_empty_slots_keeps_waiting(self):
        """回补阶段有空位时培养门控等待期不合成：合成减少数量，与回补策略冲突。"""
        task = self._task_after_forced_phase()
        task._batch_summon_remaining = 0
        task._topup_active = True
        board = make_board(
            [
                make_hero("snow", star=1, pos=(100, 100)),
                make_hero("snow", star=1, pos=(200, 200)),
            ]
        )
        obs = Observation(
            frame_id=1,
            board=board,
            raw_data={"gold": {"current": 0, "summon_cost": 80, "summon_red": True}},
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "summon_gold_wait"

    def test_batch_gold_wait_topup_with_empty_slots_keeps_waiting(self):
        """回补阶段批内金币不足同样不合成（原实现会合成，与回补策略冲突）。"""
        task = self._task_after_forced_phase()
        task._batch_summon_remaining = 3
        task._topup_active = True
        board = make_board(
            [
                make_hero("snow", star=1, pos=(100, 100)),
                make_hero("snow", star=1, pos=(200, 200)),
            ]
        )
        obs = Observation(
            frame_id=1,
            board=board,
            raw_data={"gold": {"current": 0, "summon_cost": 80, "summon_red": True}},
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "summon_gold_wait"
        assert task._batch_summon_remaining == 3

    def test_board_watch_resumes_after_batch_finishes(self):
        task = self._task_after_forced_phase()
        task._batch_summon_remaining = 1
        obs = Observation(frame_id=1, board=make_board([]))
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        task.on_action_verified(*decision)
        assert task._batch_summon_remaining == 0
        assert task.wants_board_watch() is True

    def test_batch_summon_disabled_falls_back_to_single_summon(self):
        task = self._task_after_forced_phase()
        task._run_config.batch_summon_enabled = False
        board = make_board([make_hero("snow", star=1, pos=(100, 100))])
        obs = Observation(frame_id=1, board=board)
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        action, _ = decision
        assert action.tag == "summon_hero"
        assert action.verification == "next_frame"
        assert task._batch_summon_remaining == 0


class TestBatchMerge:
    def _task_ready_for_merge(self, main_c: str = "assault") -> CoopTask:
        task = make_task(main_c=main_c, batch_enabled=True)
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        return task

    def test_batch_merge_plans_disjoint_pairs(self):
        """4 个 1 星雪姬：两对互不交叉，一次排队，逐拖消费。"""
        task = self._task_ready_for_merge()
        heroes = [
            make_hero("snow", star=1, pos=(10, 10)),
            make_hero("snow", star=1, pos=(20, 20)),
            make_hero("snow", star=1, pos=(30, 10)),
            make_hero("snow", star=1, pos=(40, 40)),
            make_hero("death_knight", star=1, pos=(50, 50)),
        ]
        obs = Observation(frame_id=1, board=make_full_board(heroes))

        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        action, _ = decision
        assert action.tag == "batch_merge"
        assert action.verification == "immediate"
        # 4 个雪姬两两配对，死亡骑士无对
        assert len(task._batch_merge_queue) == 2
        # 队列内所有格子互不相交
        used = [
            pos
            for source, dest in task._batch_merge_queue
            for pos in (source.position, dest.position)
        ]
        assert len(set(used)) == len(used)
        # 批次进行中不做多帧棋盘识别
        assert task.wants_board_watch() is False

        # 逐拖消费队列
        task.on_action_verified(*decision)
        assert len(task._batch_merge_queue) == 1
        second = task.decide_action(obs, make_window_ctx())
        assert second is not None
        assert second[0].tag == "batch_merge"
        task.on_action_verified(*second)
        assert task._batch_merge_queue == []
        assert task.wants_board_watch() is True

    def test_batch_merge_excludes_main_c_and_three_star(self):
        """批次绝不动主C强袭，也不排 3 星对（合成 4 星会弹赠送技能页）。"""
        task = self._task_ready_for_merge(main_c="assault")
        heroes = [
            make_hero("assault", star=1, pos=(10, 10)),
            make_hero("assault", star=1, pos=(20, 20)),
            make_hero("angel", star=3, pos=(30, 30)),
            make_hero("angel", star=3, pos=(40, 40)),
            make_hero("snow", star=1, pos=(50, 50)),
            make_hero("snow", star=1, pos=(60, 60)),
        ]
        obs = Observation(frame_id=1, board=make_full_board(heroes))

        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "batch_merge"
        # 批次里只有雪姬对
        assert len(task._batch_merge_queue) == 1
        source, dest = task._batch_merge_queue[0]
        assert {source.hero_type, dest.hero_type} == {"snow"}
        # 主C对和3星对保留在候选中，由单动作策略兜底
        candidates = obs.board.find_merge_candidates("assault")  # type: ignore[union-attr]
        assert len(candidates) == 3

    def test_batch_merge_excludes_failed_pairs(self):
        """失败记忆中的对不进批次；批次空时落回单动作策略。"""
        task = self._task_ready_for_merge()
        heroes = [
            make_hero("snow", star=1, pos=(10, 10)),
            make_hero("snow", star=1, pos=(20, 20)),
        ]
        # 预先把该对标记为失败（下->上的拖动方向签名）
        task._failed_merge_pairs.add(((20, 20), (10, 10)))
        obs = Observation(frame_id=1, board=make_full_board(heroes))

        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag != "batch_merge"
        assert task._batch_merge_queue == []

    def test_batch_merge_disabled_keeps_single_action(self):
        task = self._task_ready_for_merge()
        task._run_config.batch_merge_enabled = False
        heroes = [
            make_hero("snow", star=1, pos=(10, 10)),
            make_hero("snow", star=1, pos=(20, 20)),
        ]
        obs = Observation(frame_id=1, board=make_full_board(heroes))
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        # 走原有单动作合成（next_frame 逐拖验证）
        assert decision[0].tag == "merge_heroes"
        assert decision[0].verification == "next_frame"
        assert task._batch_merge_queue == []

    def test_batch_merge_attribution_marks_stale_pairs_failed(self):
        """批末归因：棋盘上仍原样存在的对 = 拖动未生效，记入失败记忆。"""
        task = self._task_ready_for_merge()
        sig_done = ((20, 20), (10, 10))
        sig_failed = ((40, 40), (30, 30))
        task._pending_merge_batch_signatures = [sig_done, sig_failed]
        # 批后棋盘：sig_done 的对已合并消失，sig_failed 的对原样还在
        heroes = [
            make_hero("snow", star=2, pos=(10, 10)),
            make_hero("snow", star=1, pos=(30, 30)),
            make_hero("snow", star=1, pos=(40, 40)),
        ]
        board = make_full_board(heroes)
        task._attribute_merge_batch(board.find_merge_candidates("assault"))
        assert sig_failed in task._failed_merge_pairs
        assert sig_done not in task._failed_merge_pairs
        assert task._pending_merge_batch_signatures == []


class TestBatchStateReset:
    def test_reset_for_home_reentry_clears_batch_state(self):
        task = make_task()
        task._batch_summon_remaining = 2
        task._batch_merge_queue = [
            (
                make_hero("snow", star=1, pos=(10, 10)),
                make_hero("snow", star=1, pos=(20, 20)),
            )
        ]
        task._pending_merge_batch_signatures = [((20, 20), (10, 10))]
        task._reset_for_home_reentry()
        assert task._batch_summon_remaining == 0
        assert task._batch_merge_queue == []
        assert task._pending_merge_batch_signatures == []

    def test_main_c_ready_clears_batch_state(self):
        task = make_task()
        task._batch_summon_remaining = 2
        task._batch_merge_queue = [
            (
                make_hero("snow", star=1, pos=(10, 10)),
                make_hero("snow", star=1, pos=(20, 20)),
            )
        ]
        task._pending_merge_batch_signatures = [((20, 20), (10, 10))]
        task.on_action_verified(
            Action(kind="wait", tag="main_c_ready", verification="immediate"),
            State.SELECT_MAIN_C_SKILLS,
        )
        assert task._batch_summon_remaining == 0
        assert task._batch_merge_queue == []
        assert task._pending_merge_batch_signatures == []


# ---------------------------------------------------------------------------
# 局内技能页「刷新」分支（2026-09-12 用户确认：每局一次，刷后等 1-2 秒必选一张）
# ---------------------------------------------------------------------------


def _refresh_obs(frame_id: int = 1, **raw_extra) -> Observation:
    """三张卡均非主C系技能（档位 9/10/11）+ 刷新按钮命中的局内技能页帧。"""
    raw = {
        "select_skill_button_visible": True,
        "skill_card_options": [
            {"column": 0, "name": "圣羽加持", "tier": 9, "known": True, "position": (100, 600)},
            {"column": 1, "name": "光矢协同", "tier": 10, "known": True, "position": (460, 600)},
            {"column": 2, "name": "星辰护盾", "tier": 11, "known": False, "position": (820, 600)},
        ],
        "skill_refresh_match": MatchResult("shua_xin.png", (472, 1321), 0.99),
    }
    raw.update(raw_extra)
    return Observation(frame_id=frame_id, raw_data=raw)


def _skill_options_obs(names_with_tiers: list[tuple[str, int]]) -> Observation:
    return Observation(
        frame_id=1,
        raw_data={
            "select_skill_button_visible": True,
            "skill_card_options": [
                {"column": i, "name": n, "tier": t, "known": True, "position": (i, 0)}
                for i, (n, t) in enumerate(names_with_tiers)
            ],
        },
    )


class TestSkillRefresh:
    """局内技能页「刷新」决策：触发条件、每局预算、刷新后必选一张。"""

    def _awaiting_task(self) -> CoopTask:
        task = make_task()
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        task._awaiting_main_candidates = True
        return task

    def test_refresh_fires_when_no_main_c_skill(self):
        task = self._awaiting_task()
        decision = task.decide_action(_refresh_obs(), make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert action.kind == "click"
        assert action.tag == "skill_refresh"
        assert action.target == (472, 1321)
        assert to_state == State.SELECT_MAIN_C_SKILLS

    def test_no_refresh_when_main_c_skill_present(self):
        """三张卡里有任意一张主C系技能（档位 ≤8）就不刷，直接按优先级选。"""
        task = self._awaiting_task()
        decision = task.decide_action(
            _skill_options_obs([("圣羽加持", 9), ("天穹支援", 1), ("星辰护盾", 11)]),
            make_window_ctx(),
        )
        assert decision is not None
        assert decision[0].tag == "main_skill_priority"

    def test_no_refresh_after_budget_used(self):
        """本局刷新机会用完：同样没有主C系技能也必须直接选一张。"""
        task = self._awaiting_task()
        task._skill_refresh_used_count = task._run_config.skill_refresh_max_per_round
        decision = task.decide_action(_refresh_obs(), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_skill_priority"

    def test_no_refresh_without_card_options(self):
        """OCR 没读出卡名：不在识别不可靠时浪费每局唯一机会，走等帧流程。"""
        task = self._awaiting_task()
        decision = task.decide_action(
            _refresh_obs(skill_card_options=None), make_window_ctx()
        )
        assert decision is not None
        assert decision[0].tag != "skill_refresh"

    def test_button_miss_waits_then_falls_to_selection(self):
        """按钮未识别到：先限次等待重试（页面过渡帧），预算耗尽才直接选卡。"""
        task = self._awaiting_task()
        # 前几次 miss → 等待重试，不浪费刷新机会也不盲选
        for _i in range(task._run_config.skill_refresh_button_wait_max):
            decision = task.decide_action(
                _refresh_obs(skill_refresh_match=None), make_window_ctx()
            )
            assert decision is not None
            action, to_state = decision
            assert action.kind == "wait"
            assert action.tag == "skill_refresh_wait_button"
            assert to_state == State.SELECT_MAIN_C_SKILLS
            task.on_action_verified(action, to_state)

        # 等待预算耗尽 → 放弃刷新（本局机会保留），按优先级直接选卡
        decision = task.decide_action(
            _refresh_obs(skill_refresh_match=None), make_window_ctx()
        )
        assert decision is not None
        assert decision[0].tag == "main_skill_priority"

        # 刷新机会未被消耗（按钮没点到不算用掉）
        assert task._skill_refresh_used_count == 0

    def test_button_wait_budget_resets_on_page_reopen(self):
        """重新打开技能页（open_main_skills）后按钮等待预算从零开始。"""
        task = self._awaiting_task()
        task._skill_refresh_button_waits = task._run_config.skill_refresh_button_wait_max
        task.on_action_verified(
            Action(kind="click", tag="open_main_skills", target=(1, 1)),
            State.SELECT_MAIN_C_SKILLS,
        )
        assert task._skill_refresh_button_waits == 0

    def test_no_refresh_when_disabled(self):
        task = self._awaiting_task()
        task._run_config.skill_refresh_enabled = False
        decision = task.decide_action(_refresh_obs(), make_window_ctx())
        assert decision is not None
        assert decision[0].tag != "skill_refresh"

    def test_no_refresh_when_budget_zero(self):
        task = self._awaiting_task()
        task._run_config.skill_refresh_max_per_round = 0
        decision = task.decide_action(_refresh_obs(), make_window_ctx())
        assert decision is not None
        assert decision[0].tag != "skill_refresh"

    def test_refresh_verified_then_same_frame_selects(self):
        """刷新生效后不再固定等待：验证帧已 OCR 出新卡面，直接按优先级选卡。"""
        task = self._awaiting_task()
        task.decide_action(_refresh_obs(), make_window_ctx())  # 决策：点刷新
        task.on_action_verified(
            Action(kind="click", tag="skill_refresh", target=(1, 1)),
            State.SELECT_MAIN_C_SKILLS,
        )
        assert task._skill_refresh_used_count == 1
        assert task._awaiting_main_candidates is True

        # 验证帧（runner 验证通过后在同一帧上继续决策）：新卡面里有主C系技能
        # → 立即按优先级选，无任何额外等待
        decision = task.decide_action(
            _skill_options_obs([("天穹支援", 1), ("星辰护盾", 11), ("圣羽加持", 9)]),
            make_window_ctx(),
        )
        assert decision is not None
        action, _ = decision
        assert action.tag == "main_skill_priority"
        assert action.kind == "click"

    def test_refresh_verifies_cards_changed(self):
        """验证口径：页面仍开着且卡面变化；卡面没变或页面已关都不算生效。"""
        task = self._awaiting_task()
        action = Action(kind="click", tag="skill_refresh", target=(472, 1321))
        before = Observation(
            frame_id=1,
            raw_data={
                "select_skill_button_visible": True,
                "skill_card_options": [
                    {"column": i, "name": n, "tier": 9 + i, "known": True, "position": (i, 0)}
                    for i, n in enumerate(["甲", "乙", "丙"])
                ],
            },
        )

        def after_with(names: list[str], *, page_open: bool) -> Observation:
            raw = {
                "skill_card_options": [
                    {"column": i, "name": n, "tier": 9 + i, "known": True, "position": (i, 0)}
                    for i, n in enumerate(names)
                ]
            }
            if page_open:
                raw["select_skill_button_visible"] = True
            return Observation(frame_id=2, raw_data=raw)

        assert task.verify_action(action, before, after_with(["丁", "戊", "己"], page_open=True))
        assert not task.verify_action(action, before, after_with(["甲", "乙", "丙"], page_open=True))
        assert not task.verify_action(action, before, after_with(["丁", "戊", "己"], page_open=False))

    def test_verified_refresh_counts_budget(self):
        task = self._awaiting_task()
        task.on_action_verified(
            Action(kind="click", tag="skill_refresh", target=(1, 1)),
            State.SELECT_MAIN_C_SKILLS,
        )
        assert task._skill_refresh_used_count == 1
        assert task._skill_refresh_button_waits == 0
        # 刷新后仍处于等待候选状态：验证帧直接进入选卡决策
        assert task._awaiting_main_candidates is True

    def test_failed_refresh_marks_used_and_falls_to_selection(self):
        """刷新点击重试耗尽：机会按已用处理，恢复后直接按优先级选卡。"""
        task = self._awaiting_task()
        task.on_action_failed(Action(kind="click", tag="skill_refresh", target=(1, 1)))
        assert task._skill_refresh_used_count == 1
        decision = task.decide_action(_refresh_obs(), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_skill_priority"

    def test_ready_match_resets_refresh_budget(self):
        """新一局开始（点准备）：刷新机会恢复一次。"""
        task = self._awaiting_task()
        task._skill_refresh_used_count = 1
        task.on_action_verified(
            Action(kind="click", tag="ready_match", target=(1, 1)),
            State.SELECT_OPENING_SKILLS,
        )
        assert task._skill_refresh_used_count == 0

    def test_home_reentry_resets_refresh_budget(self):
        """被踢回首页/结算重开：刷新机会恢复一次。"""
        task = self._awaiting_task()
        task._skill_refresh_used_count = 1
        task._reset_for_home_reentry()
        assert task._skill_refresh_used_count == 0


class TestSkillClickPostDelay:
    """局内付费选技能流程点击用固定 post_delay（skill_click_post_delay_seconds）。

    替代全局 input.min/max_delay 的随机拟人间隔，加快「开页→OCR→选卡」
    循环；只作用于局内主流程（开页 / 按档位 / 图标命中 / 兜底选卡），
    开局技能页与 4 星赠送页保持默认间隔（0）——它们的验证帧需完整 OCR
    比对卡面，过早截图反而多等一轮数秒的完整识别。
    """

    def test_open_main_skills_uses_configured_post_delay(self):
        task = make_task()
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        task._next_skill_at = 0.0
        obs = Observation(frame_id=1, raw_data={"gold": {"skill_red": False}})
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "open_main_skills"
        assert decision[0].post_delay == task._run_config.skill_click_post_delay_seconds

    def test_priority_pick_uses_configured_post_delay(self):
        task = make_task()
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        task._awaiting_main_candidates = True
        decision = task.decide_action(
            _skill_options_obs([("天穹支援", 1), ("星辰护盾", 11), ("圣羽加持", 9)]),
            make_window_ctx(),
        )
        assert decision is not None
        assert decision[0].tag == "main_skill_priority"
        assert decision[0].post_delay == task._run_config.skill_click_post_delay_seconds

    def test_icon_candidate_pick_uses_configured_post_delay(self):
        # OCR 无结果但主C图标命中 → 图标选卡路径同样走固定间隔
        task = make_task()
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        task._awaiting_main_candidates = True
        candidates = [SkillCandidate("assault", (100, 100), 0.9)]
        decision = task.decide_action(
            Observation(frame_id=1, skill_candidates=candidates),
            make_window_ctx(),
        )
        assert decision is not None
        assert decision[0].tag == "main_skill_candidate"
        assert decision[0].post_delay == task._run_config.skill_click_post_delay_seconds

    def test_fallback_pick_uses_configured_post_delay(self):
        task = make_task()
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        task._awaiting_main_candidates = True
        task._main_skill_empty_checks = task._run_config.skill_recognition_frames
        obs = Observation(frame_id=1, raw_data={"select_skill_button_visible": True})
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_skill_fallback"
        assert decision[0].post_delay == task._run_config.skill_click_post_delay_seconds

    def test_opening_flow_keeps_default_pacing(self):
        # 开局技能页不适用固定间隔：验证帧要完整 OCR 比对卡面
        task = make_task()
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._opening_loaded = True
        candidates = [SkillCandidate("assault", (100, 100), 0.9)]
        decision = task.decide_action(
            Observation(frame_id=1, skill_candidates=candidates),
            make_window_ctx(),
        )
        assert decision is not None
        assert decision[0].tag == "opening_skill_candidate"
        assert decision[0].post_delay == 0.0


class TestSelectTransform:
    """变身目标纯决策（设计文档 §4.1）。"""

    def test_returns_first_three_star_non_main_c(self):
        task = make_task()
        board = make_board(
            [
                make_hero("snow", star=2, pos=(100, 100)),
                make_hero("angel", star=3, pos=(300, 300)),
                make_hero("death_knight", star=3, pos=(500, 500)),
            ]
        )
        target = task._strategy.select_transform(board, 0, 2)
        assert target is not None
        assert target.hero_type == "angel"
        assert target.star_level == 3

    def test_none_when_main_c_target_reached(self):
        task = make_task()
        board = make_board(
            [
                make_hero("assault", star=2, pos=(100, 100)),
                make_hero("snow", star=3, pos=(300, 300)),
            ]
        )
        assert task._strategy.select_transform(board, 0, 2) is None

    def test_none_when_charges_exhausted(self):
        task = make_task()
        board = make_board([make_hero("snow", star=3, pos=(300, 300))])
        assert task._strategy.select_transform(board, 2, 2) is None

    def test_none_when_below_min_star(self):
        task = make_task()
        board = make_board([make_hero("snow", star=2, pos=(300, 300))])
        assert task._strategy.select_transform(board, 0, 2) is None

    def test_four_star_non_main_c_triggers(self):
        """min_star_level 是下限：4 星非主 C 同样触发（>= 3）。"""
        task = make_task()
        board = make_board([make_hero("snow", star=4, pos=(300, 300))])
        target = task._strategy.select_transform(board, 0, 2)
        assert target is not None
        assert target.star_level == 4

    def test_none_when_board_empty(self):
        task = make_task()
        assert task._strategy.select_transform(make_board([]), 0, 2) is None


class TestTransformSequence:
    """变身阶段机：poll → confirm → verify（设计文档 §4.2）。

    驱动方式与仓库惯例一致：直接设置任务私有字段 + 构造 Observation 断言动作。
    """

    def _ready_task(self) -> CoopTask:
        task = make_task(transform_button_locator=TRANSFORM_LOCATOR)
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        return task

    def _button_obs(self, match: MatchResult | None) -> Observation:
        raw: dict = {"transform_button_visible": match is not None}
        if match is not None:
            raw["transform_button_match"] = match
        return Observation(frame_id=1, raw_data=raw)

    # ---- 触发 ----

    def test_triggers_click_hero_on_three_star_non_main_c(self):
        task = self._ready_task()
        board = make_board([make_hero("snow", star=3, pos=(100, 100))])
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "click"
        assert decision[0].tag == "transform_click_hero"
        assert decision[0].target == (100, 100)
        assert task._transform_stage == "poll"
        assert task._transform_target is not None
        assert task._transform_target.hero_type == "snow"

    def test_no_trigger_without_locator(self):
        """未接线 locator（如模板缺失降级）时不变身，回落常规决策。"""
        task = make_task()  # 默认不传 locator
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        board = make_board([make_hero("snow", star=3, pos=(100, 100))])
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag != "transform_click_hero"
        assert task._transform_stage is None

    def test_no_trigger_when_main_c_ready(self):
        task = self._ready_task()
        board = make_board(
            [
                make_hero("assault", star=2, pos=(100, 100)),
                make_hero("snow", star=3, pos=(300, 300)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert task._transform_stage is None

    def test_no_trigger_when_charges_exhausted(self):
        task = self._ready_task()
        task._transform_used_count = 2
        board = make_board([make_hero("snow", star=3, pos=(100, 100))])
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag != "transform_click_hero"

    # ---- poll ----

    def _enter_poll(self, task: CoopTask) -> None:
        board = make_board([make_hero("snow", star=3, pos=(100, 100))])
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None and decision[0].tag == "transform_click_hero"

    def test_poll_clicks_button_when_in_window(self):
        task = self._ready_task()
        self._enter_poll(task)
        match = MatchResult("bian_shen.png", (100, 45), 0.95)
        decision = task.decide_action(self._button_obs(match), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "click"
        assert decision[0].tag == "transform_click_button"
        assert decision[0].target == (100, 45)
        assert task._transform_stage == "confirm"
        assert task._transform_button_pos == (100, 45)

    def test_poll_waits_when_button_missing(self):
        task = self._ready_task()
        self._enter_poll(task)
        decision = task.decide_action(self._button_obs(None), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "wait"
        assert decision[0].tag == "transform_button_poll"

    def test_poll_rejects_match_outside_window(self):
        """窗口外命中（如点击落到了别的英雄上弹出的按钮）不算我们的按钮。"""
        task = self._ready_task()
        self._enter_poll(task)
        match = MatchResult("bian_shen.png", (500, 45), 0.95)
        decision = task.decide_action(self._button_obs(match), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "transform_button_poll"

    def test_poll_timeout_reclicks_hero_then_gives_up(self):
        task = self._ready_task()
        self._enter_poll(task)
        task._transform_poll_deadline = 0.0  # 轮询超时
        board = make_board([make_hero("snow", star=3, pos=(100, 100))])
        obs = Observation(frame_id=1, board=board)
        # 第一次超时：重点击英雄
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "transform_reclick_hero"
        assert task._transform_reclicks == 1
        # 耗尽重点击预算后再超时：放弃，回落常规决策；放弃记忆抑制立刻再触发
        task._transform_reclicks = task._run_config.transform_button_reclick_max
        task._transform_poll_deadline = 0.0
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag not in {
            "transform_click_hero",
            "transform_reclick_hero",
            "transform_button_poll",
        }
        assert task._transform_stage is None
        assert task._transform_used_count == 0
        assert task._transform_skip_signature is not None

    def test_giveup_suppression_lifts_when_board_evolves(self):
        """棋盘演化（如召唤新英雄）后放弃记忆解除，变身可再次触发。"""
        task = self._ready_task()
        self._enter_poll(task)
        task._transform_poll_deadline = 0.0
        task._transform_reclicks = task._run_config.transform_button_reclick_max
        board = make_board([make_hero("snow", star=3, pos=(100, 100))])
        decision = task.decide_action(
            Observation(frame_id=1, board=board), make_window_ctx()
        )
        assert decision is not None
        assert decision[0].tag != "transform_click_hero"
        # 召唤后棋盘多了一个英雄：签名变化，解除抑制
        board2 = make_board(
            [
                make_hero("snow", star=3, pos=(100, 100)),
                make_hero("angel", star=1, pos=(300, 300)),
            ]
        )
        decision = task.decide_action(
            Observation(frame_id=1, board=board2), make_window_ctx()
        )
        assert decision is not None
        assert decision[0].tag == "transform_click_hero"
        assert task._transform_skip_signature is None

    # ---- confirm ----

    def _enter_confirm(self, task: CoopTask) -> None:
        self._enter_poll(task)
        match = MatchResult("bian_shen.png", (100, 45), 0.95)
        decision = task.decide_action(self._button_obs(match), make_window_ctx())
        assert decision is not None and decision[0].tag == "transform_click_button"

    def test_confirm_button_gone_settles_and_deducts(self):
        task = self._ready_task()
        self._enter_confirm(task)
        decision = task.decide_action(self._button_obs(None), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "wait"
        assert decision[0].tag == "transform_settle"
        assert decision[0].duration == task._run_config.transform_settle_seconds
        assert task._transform_stage == "verify"
        assert task._transform_used_count == 1

    def test_confirm_button_still_there_reclicks(self):
        task = self._ready_task()
        self._enter_confirm(task)
        match = MatchResult("bian_shen.png", (105, 48), 0.94)  # 原位附近仍在
        decision = task.decide_action(self._button_obs(match), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "click"
        assert decision[0].tag == "transform_reclick_button"
        assert task._transform_used_count == 0

    def test_confirm_button_moved_counts_as_gone(self):
        """原位按钮消失、别处出现新按钮（如另一次点击残留）按已消失处理。"""
        task = self._ready_task()
        self._enter_confirm(task)
        match = MatchResult("bian_shen.png", (400, 700), 0.94)
        decision = task.decide_action(self._button_obs(match), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "transform_settle"
        assert task._transform_used_count == 1

    def test_confirm_exhausted_gives_up_without_deduct(self):
        task = self._ready_task()
        self._enter_confirm(task)
        task._transform_reclicks = task._run_config.transform_button_reclick_max
        match = MatchResult("bian_shen.png", (100, 45), 0.94)
        board = make_board([make_hero("snow", star=3, pos=(100, 100))])
        obs = Observation(
            frame_id=1,
            board=board,
            raw_data={"transform_button_visible": True, "transform_button_match": match},
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag not in {"transform_click_button", "transform_reclick_button"}
        assert task._transform_stage is None
        assert task._transform_used_count == 0
        assert task._transform_skip_signature is not None

    # ---- verify ----

    def _enter_verify(self, task: CoopTask) -> None:
        self._enter_confirm(task)
        decision = task.decide_action(self._button_obs(None), make_window_ctx())
        assert decision is not None and decision[0].tag == "transform_settle"

    def test_verify_success_retriggers_on_same_position(self):
        """变身出非主C：验证通过后立即对同一位置再次触发（连用，用户定案）。"""
        task = self._ready_task()
        self._enter_verify(task)
        board = make_board([make_hero("angel", star=3, pos=(100, 100))])
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "transform_click_hero"  # 第二次连用
        assert task._transform_stage == "poll"
        assert task._transform_target is not None
        assert task._transform_target.hero_type == "angel"
        assert task._transform_used_count == 1

    def test_verify_main_c_result_goes_to_skills(self):
        """变出主C：验证通过后正常达标，进入选技能。"""
        task = self._ready_task()
        self._enter_verify(task)
        board = make_board([make_hero("assault", star=3, pos=(100, 100))])
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS

    def test_verify_unchanged_type_retries_then_gives_up(self):
        task = self._ready_task()
        self._enter_verify(task)
        board = make_board([make_hero("snow", star=3, pos=(100, 100))])  # 类型未变
        obs = Observation(frame_id=1, board=board)
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "transform_verify_retry"
        task._transform_verify_attempts = task._run_config.transform_verify_attempts
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag not in {
            "transform_verify_retry",
            "transform_click_hero",  # 放弃记忆抑制立刻再触发
        }
        assert task._transform_stage is None
        assert task._transform_skip_signature is not None

    def test_verify_empty_board_waits(self):
        task = self._ready_task()
        self._enter_verify(task)
        decision = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "wait"
        assert decision[0].tag == "transform_verify_wait"

    # ---- 钩子与重置 ----

    def test_observation_mode_and_board_watch_by_stage(self):
        task = self._ready_task()
        self._enter_poll(task)
        assert task.observation_mode() == "transform_button"
        assert task.wants_board_watch() is False
        task._transform_stage = "verify"
        assert task.observation_mode() is None
        assert task.wants_board_watch() is True

    def test_round_reset_clears_counter(self):
        task = self._ready_task()
        self._enter_verify(task)
        task._transform_used_count = 2
        task._transform_skip_signature = (("snow", 3, (100, 100)),)
        task.on_action_verified(Action(kind="wait", tag="ready_match"), State.ENTER_MATCH)
        assert task._transform_used_count == 0
        assert task._transform_stage is None
        assert task._transform_skip_signature is None
