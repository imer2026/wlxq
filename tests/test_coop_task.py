"""CoopTask 状态机决策单元测试。

只输入 Observation 断言 State/Action，不启动真实游戏窗口。
覆盖 determine_state 优先级和 decide_action 召唤/合成/停止决策。
"""

from __future__ import annotations

import pytest

from wlxq_bot.config import Hotspot, MainCProfile, RoiConfig, RunConfig
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
from wlxq_bot.tasks.base import TaskContext
from wlxq_bot.tasks.coop import _EMPTY_BOARD_RETRY, CoopTask, _RecruitStep

# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------


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
    hotspots: dict[str, Hotspot] | None = None,
    coop_difficulties: list[int] | None = None,
    difficulty_selection_only: bool = False,
    skip_difficulty_selection: bool = False,
    include_skill_candidate_roi: bool = True,
    main_c_profile: MainCProfile | None = None,
) -> CoopTask:
    ctx = TaskContext(main_c=main_c, max_rounds=max_rounds)
    run_cfg = RunConfig(
        max_rounds=max_rounds,
        minimum_summon_count_before_skills=minimum_summons,
        initial_board_capacity=max(7, minimum_summons),
        target_star_level=target_star,
    )
    spots = (
        hotspots
        if hotspots is not None
        else {
            "home_chat": Hotspot(x_ratio=0.1, y_ratio=0.2),
            "open_recruit": Hotspot(x_ratio=0.2, y_ratio=0.3),
            "open_difficulty_dialog": Hotspot(x_ratio=0.3, y_ratio=0.4),
            "difficulty_scroll_start": Hotspot(x_ratio=0.5, y_ratio=0.3),
            "difficulty_scroll_end": Hotspot(x_ratio=0.5, y_ratio=0.7),
            "close_difficulty_dialog": Hotspot(x_ratio=0.4, y_ratio=0.5),
            "coop_chat": Hotspot(x_ratio=0.5, y_ratio=0.6),
            "add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96),
            "join_coop": Hotspot(x_ratio=0.6, y_ratio=0.75),
            "select_skill": Hotspot(x_ratio=0.2, y_ratio=0.9),
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


def advance_initial_entry_to_join(task: CoopTask) -> None:
    observations = [
        home_observation(frame_id=1),
        Observation(frame_id=2),
        Observation(frame_id=3),
    ]
    for observation in observations:
        decision = task.decide_action(observation, make_window_ctx())
        assert decision is not None
        task.on_action_verified(*decision)
    # 打开确认：settle → 连续 2 次命中标题 → 确认打开进入勾选
    advance_confirm_difficulty_open(task, 2, 1, frame_id=4)
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
        entry_observations = [
            home_observation(frame_id=1),
            Observation(frame_id=2),
            Observation(frame_id=3),
            Observation(frame_id=4),  # 打开确认：固定等待（settle）
            difficulty_observation(2, 1, frame_id=5),  # 命中 1
            difficulty_observation(2, 1, frame_id=6),  # 命中 2
            difficulty_observation(2, 1, frame_id=7),  # 确认打开 → 进入勾选
            difficulty_observation(2, 1, frame_id=8),
            difficulty_observation(2, 1, frame_id=9),
            # 弹窗仍打开 → 点击关闭；之后连续 2 次复核不可见 → 确认关闭
            difficulty_observation(2, 1, frame_id=10),
            Observation(frame_id=11),
            Observation(frame_id=12),
            Observation(frame_id=13),
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
            "open_home_chat",
            "open_recruit",
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
        grab_decision = task.decide_action(Observation(frame_id=14), make_window_ctx())
        assert grab_decision is not None
        assert grab_decision[0].kind == "grab_coop"
        assert grab_decision[0].tag == "find_coop_grab"
        assert grab_decision[1] == State.FIND_COOP

    def test_skip_difficulty_selection_jumps_from_recruit_to_join(self):
        task = make_task(skip_difficulty_selection=True)
        task.ctx.current_state = State.FIND_COOP
        tags = []
        # 跳过模式：打开确认通过后直接进关闭步骤，不进入 SELECT_DIFFICULTIES
        for observation in [
            home_observation(frame_id=1),
            Observation(frame_id=2),
            Observation(frame_id=3),
            Observation(frame_id=4),  # settle
            difficulty_observation(10, frame_id=5),
            difficulty_observation(10, frame_id=6),
            difficulty_observation(10, frame_id=7),  # 确认打开 → 直接关闭
            difficulty_observation(10, frame_id=8),  # 点击关闭
            Observation(frame_id=9),
            Observation(frame_id=10),
            Observation(frame_id=11),
        ]:
            decision = task.decide_action(observation, make_window_ctx())
            assert decision is not None
            action, to_state = decision
            tags.append(action.tag)
            assert to_state == State.FIND_COOP
            task.on_action_verified(action, to_state)

        # 只跳过勾选难度等级；弹窗仍开/关一次刷新最新合作邀请
        assert tags == [
            "open_home_chat",
            "open_recruit",
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
        grab_decision = task.decide_action(Observation(frame_id=12), make_window_ctx())
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

    def test_initial_recruit_requires_positive_home_page_match(self):
        task = make_task()
        task.ctx.current_state = State.FIND_COOP

        with pytest.raises(RuntimeError, match="当前界面不是游戏首页"):
            task.decide_action(Observation(frame_id=1), make_window_ctx())

    def test_initial_recruit_requests_home_page_observation(self):
        task = make_task()
        assert task.observation_mode() == "home_page"

    def test_join_step_emits_grab_coop_signal(self):
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        advance_initial_entry_to_join(task)

        decision = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "grab_coop"
        assert decision[0].tag == "find_coop_grab"

    def test_missing_entry_hotspot_stops_without_skipping_to_join(self):
        task = make_task(hotspots={"join_coop": Hotspot(x_ratio=0.6, y_ratio=0.75)})
        task.ctx.current_state = State.FIND_COOP
        assert task.decide_action(home_observation(), make_window_ctx()) is None

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

        # 重新进入招募：从首页确认开始，而不是停在开局等待
        follow_up = task.decide_action(home_observation(frame_id=frames + 1), make_window_ctx())
        assert follow_up is not None
        assert follow_up[0].tag == "open_home_chat"

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

        follow_up = task.decide_action(home_observation(frame_id=2), make_window_ctx())
        assert follow_up is not None
        assert follow_up[0].tag == "open_home_chat"

    def test_kick_reentry_skips_difficulty_reselection(self):
        """已点过准备（本会话难度已勾选）后被踢回：重新进入不重复勾选难度。"""
        task = make_task(coop_difficulties=[2, 1])
        task.ctx.current_state = State.FIND_COOP
        task.on_action_verified(
            Action(kind="click", tag="ready_match", target=(1, 1)),
            State.SELECT_OPENING_SKILLS,
        )
        # 被踢回首页后的重进路径：首页 → 聊天 → 招募 → 打开难度弹窗
        for observation, expected in [
            (home_observation(frame_id=1), "open_home_chat"),
            (Observation(frame_id=2), "open_recruit"),
            (Observation(frame_id=3), "open_difficulty_dialog"),
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

        # 退队后落点探测：识别到首页 → 从首页入口进入招募
        detect = task.decide_action(home_observation(frame_id=3), make_window_ctx())
        assert detect is not None
        assert detect[0].tag == "entry_page_detected"
        task.on_action_verified(*detect)
        follow = task.decide_action(home_observation(frame_id=4), make_window_ctx())
        assert follow is not None
        assert follow[0].tag == "open_home_chat"

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
        for observation in (home_observation(), Observation(frame_id=1), Observation(frame_id=1)):
            decision = task.decide_action(observation, make_window_ctx())
            assert decision is not None
            task.on_action_verified(*decision)
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
        for observation in (home_observation(), Observation(frame_id=1), Observation(frame_id=1)):
            decision = task.decide_action(observation, make_window_ctx())
            assert decision is not None
            task.on_action_verified(*decision)
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

        # 首页确认 → 聊天 → 招募 → 打开难度弹窗，进入打开确认步骤
        for observation in (
            home_observation(),
            Observation(frame_id=1),
            Observation(frame_id=1),
        ):
            decision = task.decide_action(observation, make_window_ctx())
            assert decision is not None
            task.on_action_verified(*decision)
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
        for observation in (
            home_observation(frame_id=1),
            Observation(frame_id=2),
            Observation(frame_id=3),
        ):
            decision = task.decide_action(observation, make_window_ctx())
            assert decision is not None
            task.on_action_verified(*decision)
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
        for observation in (
            home_observation(frame_id=1),
            Observation(frame_id=2),
            Observation(frame_id=3),
        ):
            decision = task.decide_action(observation, make_window_ctx())
            assert decision is not None
            task.on_action_verified(*decision)
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
        for observation in (
            home_observation(frame_id=1),
            Observation(frame_id=2),
            Observation(frame_id=3),
        ):
            decision = task.decide_action(observation, make_window_ctx())
            assert decision is not None
            task.on_action_verified(*decision)

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
        for observation in (
            home_observation(frame_id=1),
            Observation(frame_id=2),
            Observation(frame_id=3),
        ):
            decision = task.decide_action(observation, make_window_ctx())
            assert decision is not None
            task.on_action_verified(*decision)

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
        for observation in (
            home_observation(frame_id=1),
            Observation(frame_id=2),
            Observation(frame_id=3),
        ):
            decision = task.decide_action(observation, make_window_ctx())
            assert decision is not None
            task.on_action_verified(*decision)
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
