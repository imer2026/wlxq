"""CoopTask 状态机决策单元测试。

只输入 Observation 断言 State/Action，不启动真实游戏窗口。
覆盖 determine_state 优先级和 decide_action 召唤/合成/停止决策。
"""

from __future__ import annotations

import pytest

from wlxq_bot.config import (
    Hotspot,
    MainCProfile,
    ModeProfile,
    MountRideConfig,
    RoiConfig,
    RunConfig,
)
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
    _NO_PROGRESS_LIMIT,
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
    其余识别数据每帧都在（各步骤只读取自己需要的那一份）。挑战页锚点
    （challenge_text）只在挑战页确认帧存在——与实机一致：感知层仅在
    challenge_page 观察模式读挑战按钮区，入口前几帧读不到（领地入口
    探测消费该文字，前几帧携带会被误判为已在挑战页）。"""
    base = {
        "mode_select_visible": True,
        "mode_card_rows": [ModeTabRow("合作模式", 0.9, (250, 1174))],
    }

    def obs(index: int, nav: str, challenge_text: str | None = None) -> Observation:
        raw = {**base, "territory_nav_text": nav} if nav else dict(base)
        if challenge_text is not None:
            raw["challenge_text"] = challenge_text
        return Observation(frame_id=frame_id + index, raw_data=raw)

    return [
        obs(0, "领地"),
        obs(1, "返回"),
        obs(2, ""),
        obs(3, ""),
        obs(4, ""),
        obs(5, "", challenge_text="开始挑战"),
    ]


def advance_territory_entry_to_recruit(task: CoopTask, frame_id: int = 1) -> None:
    """驱动领地入口链到 SELECT_MODE_TAB（聊天点击已确认生效或按档案跳过）。

    配置了页签文字的档案：连续 2 帧读到本模式页签（chat_open_pending →
    chat_open_confirmed）才算进入招募页；未配置的档案无法验证，1 帧跳过
    （chat_open_unverified）。
    """
    for observation in territory_entry_observations(frame_id):
        decision = task.decide_action(observation, make_window_ctx())
        assert decision is not None
        task.on_action_verified(*decision)
    decision = task.decide_action(Observation(frame_id=frame_id + 6), make_window_ctx())
    assert decision is not None and decision[0].tag == "open_challenge_chat"
    task.on_action_verified(*decision)
    for offset in (7, 8):
        confirm = task.decide_action(
            Observation(
                frame_id=frame_id + offset,
                raw_data={
                    # 与实机 OCR 一致的三个页签行，档案按各自 mode_tab_text 子串命中
                    "mode_tab_rows": [
                        ModeTabRow("合作模式", 0.8, (314, 1535)),
                        ModeTabRow("保卫宠物", 0.8, (523, 1535)),
                        ModeTabRow("共战灭尊", 0.8, (733, 1535)),
                    ]
                },
            ),
            make_window_ctx(),
        )
        assert confirm is not None and confirm[0].kind == "wait"
        task.on_action_verified(*confirm)
        if confirm[0].tag == "chat_open_unverified":
            break


def advance_entry_to_difficulty_dialog(task: CoopTask, frame_id: int = 1) -> None:
    """驱动领地入口链到 CONFIRM_DIFFICULTY_OPEN（打开难度弹窗点击已验证）。"""
    advance_territory_entry_to_recruit(task, frame_id=frame_id)
    decision = task.decide_action(Observation(frame_id=frame_id + 9), make_window_ctx())
    assert decision is not None and decision[0].tag == "open_difficulty_dialog"
    task.on_action_verified(*decision)


def advance_initial_entry_to_join(task: CoopTask) -> None:
    advance_territory_entry_to_recruit(task, frame_id=1)
    # 页签（默认档案未配置 → 跳过）后打开难度弹窗
    decision = task.decide_action(Observation(frame_id=10), make_window_ctx())
    assert decision is not None and decision[0].tag == "open_difficulty_dialog"
    task.on_action_verified(*decision)
    # 打开确认：settle → 连续 2 次命中标题 → 确认打开进入勾选
    advance_confirm_difficulty_open(task, 2, 1, frame_id=10)
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
            Observation(frame_id=7),  # 挑战页聊天
            Observation(frame_id=8),  # 聊天生效确认：档案未配置页签文字 → 跳过
            Observation(frame_id=9),  # 页签未配置 → 打开难度弹窗
            Observation(frame_id=10),  # 打开确认：固定等待（settle）
            difficulty_observation(2, 1, frame_id=11),  # 命中 1
            difficulty_observation(2, 1, frame_id=12),  # 命中 2
            difficulty_observation(2, 1, frame_id=13),  # 确认打开 → 进入勾选
            difficulty_observation(2, 1, frame_id=14),
            difficulty_observation(2, 1, frame_id=15),
            # 弹窗仍打开 → 点击关闭；之后连续 2 次复核不可见 → 确认关闭
            difficulty_observation(2, 1, frame_id=16),
            Observation(frame_id=17),
            Observation(frame_id=18),
            Observation(frame_id=19),
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
            "chat_open_unverified",
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
        grab_decision = task.decide_action(Observation(frame_id=20), make_window_ctx())
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
            Observation(frame_id=7),  # 挑战页聊天
            Observation(frame_id=8),  # 聊天生效确认：档案未配置页签文字 → 跳过
            Observation(frame_id=9),  # 页签未配置 → 打开难度弹窗
            Observation(frame_id=10),  # settle
            difficulty_observation(10, frame_id=11),
            difficulty_observation(10, frame_id=12),
            difficulty_observation(10, frame_id=13),  # 确认打开
            # 弹窗仍打开 → 点击关闭；之后连续 2 次复核不可见 → 确认关闭
            difficulty_observation(10, frame_id=14),
            Observation(frame_id=15),
            Observation(frame_id=16),
            Observation(frame_id=17),
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
            "chat_open_unverified",
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
        grab_decision = task.decide_action(Observation(frame_id=18), make_window_ctx())
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

        # 聊天点击生效确认：档案未配置页签文字，1 帧跳过
        unverified = task.decide_action(Observation(frame_id=3), make_window_ctx())
        assert unverified is not None and unverified[0].kind == "wait"
        task.on_action_verified(*unverified)

        open_refresh_decision = task.decide_action(Observation(frame_id=4), make_window_ctx())
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
        """领地入口第一步不做首页校验：OCR 底部导航领地格 + 挑战页锚点
        （探测是否已在挑战页）；打开确认步骤只读导航文字。"""
        task = make_task()
        assert task.observation_mode() == "territory_nav_probe"
        task._recruit_step = _RecruitStep.CONFIRM_TERRITORY_OPEN
        assert task.observation_mode() == "territory_nav"

    def test_join_step_emits_grab_coop_signal(self):
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        advance_initial_entry_to_join(task)

        decision = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "grab_coop"
        assert decision[0].tag == "find_coop_grab"

    def test_join_step_closes_visible_popup_before_grabbing(self):
        # 兜底：带着全屏弹窗回到 FIND_COOP 时先关闭再抢，否则弹窗挡住
        # join_coop 会形成「弹窗不关、抢不停」的循环（实机 2026-09-14）
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        advance_initial_entry_to_join(task)

        match = MatchResult("buttons/x.png", (858, 190), 0.99)
        obs = Observation(
            frame_id=1,
            raw_data={"close_button_visible": True, "close_button_match": match},
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert action.kind == "click"
        assert action.tag == "close_grab_popup"
        assert action.target == (858, 190)
        assert action.verification == "next_frame"
        # 关闭后留在 FIND_COOP，下一拍重新决策（弹窗已关则发抢合作信号）
        assert to_state == State.FIND_COOP
        task.on_action_verified(action, to_state)
        next_decision = task.decide_action(Observation(frame_id=2), make_window_ctx())
        assert next_decision is not None
        assert next_decision[0].kind == "grab_coop"

    def test_close_grab_popup_verified_by_button_disappearance(self):
        task = make_task()
        action = Action(kind="click", tag="close_grab_popup", target=(858, 190))
        before = Observation(frame_id=1, raw_data={"close_button_visible": True})
        after = Observation(frame_id=2)  # 关闭按钮已消失
        assert task.verify_action(action, before, after)
        still_visible = Observation(frame_id=2, raw_data={"close_button_visible": True})
        assert not task.verify_action(action, before, still_visible)

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
        # 快速补点（2026-11-19）：小间隔替代全局 1~2s 拟人间隔，吞击后立即补点
        assert 0.1 <= action.post_delay <= 0.3
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
# decide_action - 4星赠送技能页监控（90秒狐狸满星等非合并触发）
# ---------------------------------------------------------------------------


class TestMergeGiftPageMonitor:
    """非 BUILD 状态弹出4星赠送技能页：decide_action 顶部优先选卡。

    2026-10-02 实机：90秒狐狸随机升级把3星狐狸升成4星弹此页，此前唯一入口
    是本机合成 4 星（BUILD 状态内处理）。提示条原本只在 BUILD 识别，选技能
    状态下页面遮住【选技能】图被误判"技能页已关闭"，盲点按钮空转到局结束。
    """

    def _gift_obs(self, frame_id: int = 1, **raw_extra) -> Observation:
        raw: dict = {"merge_gift_skill_page_visible": True}
        raw.update(raw_extra)
        return Observation(frame_id=frame_id, raw_data=raw)

    def test_gift_page_during_skill_selection_picks_by_priority(self):
        task = make_task()
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        obs = self._gift_obs(
            skill_card_options=[
                {"column": 0, "name": "圣羽加持", "tier": 9, "known": True, "position": (100, 600)},
                {"column": 1, "name": "天穹支援", "tier": 1, "known": True, "position": (460, 600)},
                {"column": 2, "name": "星辰护盾", "tier": 11, "known": True, "position": (820, 600)},
            ]
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert action.tag == "merge_gift_skill_priority"
        assert action.target == (460, 600)
        # 状态原样返回：选完卡页面关闭，技能选购流程无感续上
        assert to_state == State.SELECT_MAIN_C_SKILLS

    def test_gift_page_without_options_falls_back_to_columns(self):
        task = make_task()
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        decision = task.decide_action(self._gift_obs(), make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert action.tag == "merge_gift_skill_fallback"
        # 赠送页保持默认间隔（0）：验证帧需完整识别页面收起，过早截图反而多等
        assert action.post_delay == 0.0
        assert to_state == State.SELECT_MAIN_C_SKILLS

    def test_kill_popup_takes_priority_over_gift_page(self):
        # 击杀奖励弹窗可能盖在赠送页上：先关弹窗，下一帧再选卡
        task = make_task()
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        match = MatchResult("buttons/tan_chuang.png", (400, 1100), 0.9)
        obs = Observation(
            frame_id=1,
            raw_data={
                "merge_gift_skill_page_visible": True,
                "tan_chuang_visible": True,
                "tan_chuang_match": match,
            },
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "close_popup"

    def test_gift_page_in_build_keeps_existing_path(self):
        # BUILD 不走顶部监控（已有 _awaiting_merge_gift 确认流 + 兜底块），
        # 重构后行为不变：直接落兜底块选卡
        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        decision = task.decide_action(self._gift_obs(), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "merge_gift_skill_fallback"
        assert decision[1] == State.BUILD_MAIN_C

    def test_gift_page_not_monitored_outside_match_states(self):
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        decision = task.decide_action(
            Observation(
                frame_id=1,
                raw_data={
                    "merge_gift_skill_page_visible": True,
                    "home_page_visible": True,
                },
            ),
            make_window_ctx(),
        )
        assert decision is None or not decision[0].tag.startswith("merge_gift")

    def test_gift_pick_verified_by_page_close(self):
        # 验证闭环依赖提示条标志在本状态可见：选卡前页面在、选卡后收起即生效
        task = make_task()
        action = Action(kind="click", tag="merge_gift_skill_priority", target=(460, 600))
        before = Observation(frame_id=1, raw_data={"merge_gift_skill_page_visible": True})
        after = Observation(frame_id=2)  # 选完卡页面关闭
        assert task.verify_action(action, before, after)
        still_visible = Observation(frame_id=2, raw_data={"merge_gift_skill_page_visible": True})
        assert not task.verify_action(action, before, still_visible)


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

    def test_frozen_board_wait_does_not_exhaust_no_progress(self):
        """冻结干等是有意等待：棋盘静止不触发无进展 None（2026-09-26 实机事故）。

        锁定格冻结召唤、棋盘合到无可合成对后每帧走 frozen_board_wait，
        此前棋盘签名连续 _NO_PROGRESS_LIMIT 次不变会使 decide_action 返回
        None，runner 连续 3 次无可用动作后保守退出，而画面完全正常。
        """
        task = make_task(main_c="monkey")
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        task._board_capacity_frozen = True
        # 类型互异、无主C达标、无金币数据（跳过门控）：只能落入冻结干等
        board = make_board(
            [
                make_hero("death_knight", star=3, pos=(100, 100)),
                make_hero("assault", star=1, pos=(200, 200)),
                make_hero("angel", star=2, pos=(300, 300)),
                make_hero("snow", star=2, pos=(400, 400)),
            ]
        )
        obs = Observation(frame_id=1, board=board)
        for _ in range(_NO_PROGRESS_LIMIT + 4):
            decision = task.decide_action(obs, make_window_ctx())
            assert decision is not None
            action, to_state = decision
            assert action.tag == "frozen_board_wait"
            assert to_state == State.BUILD_MAIN_C

    def test_merge_success_unfreezes_summon(self):
        """合成成功解除召唤冻结（2026-09-27 实机事故）：冻结不再一冻一局。

        第 2 局实机：两批召唤未落位误触发冻结（识别把占格读丢/锁定格误读
        为空），此后合成腾出的真实空位永不复用，干等 4 分钟到局终。合成
        成功 = 棋盘已变化，"点不进去"的依据过时，必须解冻恢复试探召唤。
        """
        task = make_task(main_c="monkey", batch_enabled=True)
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        task._board_capacity_frozen = True
        task._batch_no_land_streak = 2
        task.on_action_verified(
            Action(kind="drag", tag="merge_heroes", target=(1, 1), end=(2, 2)),
            State.BUILD_MAIN_C,
        )
        assert task._board_capacity_frozen is False
        assert task._batch_no_land_streak == 0
        assert task._batch_planned_occupancy is None
        # 下一帧：识别容量报告有空位 → 恢复召唤规划（而非 frozen_board_wait）
        board = make_board(
            [
                make_hero("death_knight", star=3, pos=(100, 100)),
                make_hero("assault", star=1, pos=(200, 200)),
            ]
        )
        decision = task.decide_action(
            Observation(frame_id=2, board=board), make_window_ctx()
        )
        assert decision is not None
        action, to_state = decision
        assert to_state == State.BUILD_MAIN_C
        assert action.kind == "click"
        assert action.tag == "batch_summon"


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

    def test_kick_back_to_territory_restarts_recruit(self):
        """组队大厅被踢回领地页（2026-09-25 实机）：准备按钮消失且导航连续
        读到「返回」→ 按本局未开始处理，从领地入口重新抢合作。"""
        task = make_task()
        task.ctx.current_state = State.ENTER_MATCH
        frames = task._run_config.territory_return_confirm_frames
        for frame_id in range(1, frames):
            decision = task.decide_action(
                Observation(
                    frame_id=frame_id, raw_data={"territory_nav_text": "返回"}
                ),
                make_window_ctx(),
            )
            assert decision is not None
            assert decision[0].tag == "territory_return_check"
            assert decision[1] == State.ENTER_MATCH
            task.on_action_verified(*decision)

        decision = task.decide_action(
            Observation(frame_id=frames, raw_data={"territory_nav_text": "返回"}),
            make_window_ctx(),
        )
        assert decision is not None
        assert decision[0].tag == "kick_reentry_territory"
        assert decision[1] == State.FIND_COOP
        task.on_action_verified(*decision)
        task.ctx.current_state = State.FIND_COOP

        # 已在领地页（导航读「返回」）：入口跳过点领地直接进模式选择
        follow_up = task.decide_action(
            Observation(
                frame_id=frames + 1, raw_data={"territory_nav_text": "返回"}
            ),
            make_window_ctx(),
        )
        assert follow_up is not None
        assert follow_up[0].tag == "territory_open_confirmed"

    def test_kick_reentry_recovers_when_landed_on_home(self):
        """被踢落点若是首页（导航读「领地」）：同一条恢复链自校正为点领地。"""
        task = make_task()
        task.ctx.current_state = State.ENTER_MATCH
        task._territory_return_count = (
            task._run_config.territory_return_confirm_frames - 1
        )
        decision = task.decide_action(
            Observation(frame_id=1, raw_data={"territory_nav_text": "返回"}),
            make_window_ctx(),
        )
        assert decision is not None
        assert decision[0].tag == "kick_reentry_territory"
        task.on_action_verified(*decision)
        task.ctx.current_state = State.FIND_COOP

        follow_up = task.decide_action(
            Observation(frame_id=2, raw_data={"territory_nav_text": "领地"}),
            make_window_ctx(),
        )
        assert follow_up is not None
        assert follow_up[0].tag == "open_territory"

    def test_single_territory_frame_flash_does_not_reenter(self):
        """单帧读到「返回」（OCR 误读）不触发重进：准备按钮恢复即计数归零。"""
        task = make_task()
        task.ctx.current_state = State.ENTER_MATCH
        decision = task.decide_action(
            Observation(frame_id=1, raw_data={"territory_nav_text": "返回"}),
            make_window_ctx(),
        )
        assert decision is not None
        assert decision[0].tag == "territory_return_check"
        task.on_action_verified(*decision)

        resumed = task.decide_action(
            Observation(
                frame_id=2,
                raw_data={
                    "ready_button_visible": True,
                    "ready_button_match": MatchResult(
                        "buttons/zhun_bei.png", (499, 1633), 0.9
                    ),
                },
            ),
            make_window_ctx(),
        )
        assert resumed is not None
        assert resumed[0].tag == "ready_match"
        assert task._territory_return_count == 0

    def test_ready_screen_without_nav_text_stops_conservatively(self):
        """准备按钮消失且导航读不到文字（未知画面）：不盲判，保守停止。"""
        task = make_task()
        task.ctx.current_state = State.ENTER_MATCH
        assert task.decide_action(Observation(frame_id=1), make_window_ctx()) is None

    def test_reset_after_grab_kick_restarts_recruit_from_territory(self):
        """抢合作循环内被踢回首页（协调器 kicked_home，2026-10-05 实机）：
        Runner 调 reset_after_grab_kick 清进度，下一帧从领地入口重走招募，
        而不是停在 JOIN_COOP 对着首页空点 join_coop。"""
        task = make_task()
        task._recruit_step = _RecruitStep.JOIN_COOP
        task._summon_count = 5
        task._home_return_count = 1

        task.reset_after_grab_kick()

        assert task._recruit_step == _RecruitStep.OPEN_TERRITORY
        assert task._summon_count == 0
        assert task._home_return_count == 0

        # 下一帧 FIND_COOP：首页落点导航读「领地」→ 正常点领地
        task.ctx.current_state = State.FIND_COOP
        follow_up = task.decide_action(
            Observation(frame_id=1, raw_data={"territory_nav_text": "领地"}),
            make_window_ctx(),
        )
        assert follow_up is not None
        assert follow_up[0].tag == "open_territory"

    def test_ready_click_defers_to_stray_popup_close(self):
        """组队大厅弹窗开着时点准备被模态层吞掉（2026-09-25 实机）：
        先关弹窗（× 消失即验证），下一帧再点准备。"""
        task = make_task()
        task.ctx.current_state = State.ENTER_MATCH
        popup = Observation(
            frame_id=1,
            raw_data={
                "close_button_visible": True,
                "close_button_match": MatchResult("buttons/x.png", (838, 251), 0.95),
                "ready_button_visible": True,
                "ready_button_match": MatchResult(
                    "buttons/zhun_bei.png", (499, 1633), 0.87
                ),
            },
        )
        decision = task.decide_action(popup, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "close_grab_popup"
        assert decision[1] == State.ENTER_MATCH
        task.on_action_verified(*decision)

        clean = Observation(
            frame_id=2,
            raw_data={
                "ready_button_visible": True,
                "ready_button_match": MatchResult(
                    "buttons/zhun_bei.png", (499, 1633), 0.87
                ),
            },
        )
        decision = task.decide_action(clean, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "ready_match"
        assert decision[1] == State.SELECT_OPENING_SKILLS

    def test_enter_match_declares_grab_observation_mode(self):
        """组队大厅显式声明 coop_grab 观察模式：不依赖入口子步骤残留值。"""
        task = make_task()
        task.ctx.current_state = State.ENTER_MATCH
        assert task.observation_mode() == "coop_grab"

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
            (Observation(frame_id=8), "chat_open_unverified"),
            (Observation(frame_id=9), "open_difficulty_dialog"),
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
        """退队后无首页/领地信号：先复识，等满探测上限才按合作页面处理。"""
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        task.on_action_verified(
            Action(kind="click", tag="leave_team", target=(1, 1)),
            State.FIND_COOP,
        )
        task._entry_wait_started_at = 0.0
        task._clock = lambda: 1.0

        # 未超时：不急于判合作页，等待下一帧复识（OCR 抖动容错）
        detect = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert detect is not None
        assert detect[0].tag == "entry_page_detected"
        task.on_action_verified(*detect)
        pending = task.decide_action(Observation(frame_id=2), make_window_ctx())
        assert pending is not None
        assert pending[0].tag == "entry_page_detected"

        # 等满探测上限仍无信号：降级按合作页面处理，从合作页面聊天进入
        task._clock = lambda: 6.0
        degrade = task.decide_action(Observation(frame_id=3), make_window_ctx())
        assert degrade is not None
        assert degrade[0].tag == "entry_page_detected"
        task.on_action_verified(*degrade)
        follow = task.decide_action(Observation(frame_id=4), make_window_ctx())
        assert follow is not None
        assert follow[0].tag == "open_coop_chat"

    def test_leave_team_entry_detect_territory_page(self):
        """退队落在领地页（2026-10-30/11-01 两次实机）：读到导航「返回」走领地入口。

        回归用例：此前「非首页即合作页」的探测把领地页误判成合作页，
        点空聊天入口后卡重点循环。
        """
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        task.on_action_verified(
            Action(kind="click", tag="leave_team", target=(1, 1)),
            State.FIND_COOP,
        )

        # 退队后第一帧：无首页信箱，但导航读到「返回」=已在领地页
        detect = task.decide_action(
            Observation(frame_id=1, raw_data={"territory_nav_text": "返回"}),
            make_window_ctx(),
        )
        assert detect is not None
        assert detect[0].tag == "entry_page_detected"
        task.on_action_verified(*detect)
        # 领地入口第一步：仍在领地页 → 跳过点领地，直进模式选择弹窗步骤
        follow = task.decide_action(
            Observation(frame_id=2, raw_data={"territory_nav_text": "返回"}),
            make_window_ctx(),
        )
        assert follow is not None
        assert follow[0].tag == "territory_open_confirmed"
        task.on_action_verified(*follow)
        # 模式选择弹窗步骤：点「合作模式」入口
        advance = task.decide_action(Observation(frame_id=3), make_window_ctx())
        assert advance is not None
        assert advance[0].tag == "open_mode_select"

    def test_leave_team_entry_detect_home_nav_text(self):
        """退队落在首页（导航读「领地」而非信箱命中）：同样走领地入口。"""
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        task.on_action_verified(
            Action(kind="click", tag="leave_team", target=(1, 1)),
            State.FIND_COOP,
        )

        detect = task.decide_action(
            Observation(frame_id=1, raw_data={"territory_nav_text": "领地"}),
            make_window_ctx(),
        )
        assert detect is not None
        assert detect[0].tag == "entry_page_detected"
        task.on_action_verified(*detect)
        # 领地入口第一步：读「领地」=未进领地页，点底部导航领地格
        follow = task.decide_action(
            Observation(frame_id=2, raw_data={"territory_nav_text": "领地"}),
            make_window_ctx(),
        )
        assert follow is not None
        assert follow[0].tag == "open_territory"

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

    def test_gold_wait_merges_wildcard_pair_with_space(self):
        """金币不足例外（2026-10-01 定案）：等金币恢复期间幻天使对让位可合
        ——有空位但召唤被金币门控拦住，「保留幻天使等召唤」前提不成立。
        回归实机 22:27 frame=151：占用 8/9、金币 52<130、场上幻天使2星+
        死骑2星一对，干等 2 分钟。"""
        profile = MainCProfile(
            display_name="强袭",
            hero_template_dir="heroes/assault",
            avoid_main_c_merge=True,
        )
        task = make_task(main_c_profile=profile)
        task._wildcard_heroes = frozenset({"angel_phantom"})
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        board = make_board(
            [
                make_hero("angel_phantom", star=2, pos=(100, 100)),
                make_hero("death_knight", star=2, pos=(200, 100)),
                make_hero("assault", star=1, pos=(300, 300)),
                make_hero("assault", star=1, pos=(400, 400)),
            ]
        )
        obs = Observation(
            frame_id=1,
            board=board,
            raw_data={"gold": {"current": 52, "summon_cost": 130, "summon_red": False}},
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "drag"
        assert decision[0].tag == "merge_heroes"

    def test_first_entry_with_enough_count_still_enters_topup_once(self):
        """首次培养数量已达标：第一阶段不置位标志（2026-10-01 晚定案——
        第一阶段达标不短路第二阶段），选满 4 次仍回培养；第二阶段（无环节
        档案）达标访问时置位，本局不再回补——防「回补↔选技能」空转。"""
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
        # 第一阶段不置位
        assert task._topup_done is False

        # 选满 4 次 → 仍回培养（第二阶段发生）
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        for _ in range(4):
            task.on_action_verified(
                Action(kind="click", tag="main_skill_candidate", target=(1, 1)),
                State.SELECT_MAIN_C_SKILLS,
            )
        decision = task.decide_action(Observation(frame_id=2), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_topup"

        # 第二阶段达标访问：置位标志，本局不再回补
        task.on_action_verified(*decision)
        assert task._topup_active is True
        task.ctx.current_state = State.BUILD_MAIN_C
        decision = task.decide_action(Observation(frame_id=3, board=board), make_window_ctx())
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
        decision = task.decide_action(Observation(frame_id=4), make_window_ctx())
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

    def test_ready_button_two_frames_fast_path_enters_summon(self):
        """快速通道（2026-11-20）：连续 2 帧「页面关+准备完成可见」直接进召唤，
        不等满 3 帧页面关闭确认。全天观测 37 个组间间隙按钮可见帧数全部为 0，
        间隙不可能误触发。"""
        task = make_task()
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._opening_loaded = True
        raw = {"summon_button_visible": True, "zhun_bei_wan_cheng_visible": True}
        first = task.decide_action(
            Observation(frame_id=1, raw_data=dict(raw)), make_window_ctx()
        )
        assert first is not None
        # 第 1 帧：连击=1 不足以触发，走确认等待
        assert first[0].tag == "opening_exit_confirm_check"
        task.on_action_verified(*first)
        assert task._opening_ready_streak == 1
        second = task.decide_action(
            Observation(frame_id=2, raw_data=dict(raw)), make_window_ctx()
        )
        assert second is not None
        assert second[0].tag == "opening_complete"
        assert second[1] == State.BUILD_MAIN_C
        assert task._opening_ready_streak == 0

    def test_ready_button_streak_resets_on_page_reopen(self):
        """间隙里页面重开（下一组弹出）作废连击：重关后重新数。"""
        task = make_task()
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._opening_loaded = True
        raw = {"summon_button_visible": True, "zhun_bei_wan_cheng_visible": True}
        first = task.decide_action(
            Observation(frame_id=1, raw_data=dict(raw)), make_window_ctx()
        )
        task.on_action_verified(*first)
        assert task._opening_ready_streak == 1
        # 下一组弹出：页面开 → 连击作废
        candidates = [SkillCandidate("assault", (100, 100), 0.9)]
        resumed = task.decide_action(
            Observation(frame_id=2, skill_candidates=candidates), make_window_ctx()
        )
        assert resumed is not None
        assert resumed[0].tag == "opening_skill_candidate"
        assert task._opening_ready_streak == 0

    def test_ready_button_blink_resets_streak(self):
        """按钮亮 1 帧后熄灭：连击清零，回到 3 帧页面关闭确认节奏。"""
        task = make_task()
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._opening_loaded = True
        first = task.decide_action(
            Observation(
                frame_id=1,
                raw_data={"summon_button_visible": True, "zhun_bei_wan_cheng_visible": True},
            ),
            make_window_ctx(),
        )
        assert first is not None
        task.on_action_verified(*first)
        assert task._opening_ready_streak == 1
        second = task.decide_action(
            Observation(frame_id=2, raw_data={"summon_button_visible": True}),
            make_window_ctx(),
        )
        assert second is not None
        assert second[0].tag == "opening_exit_confirm_check"
        assert task._opening_ready_streak == 0

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

        # 聊天点击生效确认：档案未配置页签文字，1 帧跳过
        unverified = task.decide_action(Observation(frame_id=3), make_window_ctx())
        assert unverified is not None and unverified[0].kind == "wait"
        task.on_action_verified(*unverified)

        open_refresh_decision = task.decide_action(Observation(frame_id=4), make_window_ctx())
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


class TestStar3TopupStage:
    """3星主C培养环节（数量回补达标后的非强制追加，2026-09-30 用户定案）。

    强袭补到 4 个后留在培养阶段：合成 2 星对碰 3 星（类型随机）+ 变身
    3 星非主C 转化；达成 / 变身用尽（整局预算归零即退，2026-10-05 单门
    定案）/ 死局即回选技能。环节内合并优先（召唤只用于凑材料）、合成
    材料只用非主C对。
    """

    def _profile(self, **overrides) -> MainCProfile:
        fields = {
            "display_name": "强袭",
            "hero_template_dir": "heroes/assault",
            "avoid_main_c_merge": True,
            "topup_after_skill_selections": 4,
            "topup_hero_count": 4,
            "star3_topup_enabled": True,
        }
        fields.update(overrides)
        return MainCProfile(**fields)

    def _stage_task(self, **profile_overrides) -> CoopTask:
        task = make_task(main_c_profile=self._profile(**profile_overrides))
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        task._topup_active = True
        return task

    def _count4_board(self) -> BoardSnapshot:
        """强袭数量已达 4（2+1+1+1），另有一对非主C 1 星，棋盘有空位。"""
        return make_board(
            [
                make_hero("assault", star=2, pos=(100, 100)),
                make_hero("assault", star=1, pos=(200, 200)),
                make_hero("assault", star=1, pos=(300, 300)),
                make_hero("assault", star=1, pos=(400, 400)),
                make_hero("snow", star=1, pos=(500, 500)),
                make_hero("snow", star=1, pos=(600, 600)),
            ]
        )

    def test_holds_board_after_count_reached_and_merges_first(self):
        """数量到 4 后留下培养：有空位也优先合并非主C对（环节内合并优先）。"""
        task = self._stage_task()
        decision = task.decide_action(
            Observation(frame_id=1, board=self._count4_board()), make_window_ctx()
        )
        assert decision is not None
        assert decision[0].kind == "drag"
        assert decision[0].tag == "merge_heroes"
        # 留场标记已置位
        assert task._star3_active is True

    def test_exits_immediately_on_3star_assault(self):
        """达成：场上已有 3 星强袭，直接回选技能并记录环节完成。"""
        task = self._stage_task()
        board = make_board(
            [
                make_hero("assault", star=3, pos=(100, 100)),
                make_hero("assault", star=1, pos=(200, 200)),
                make_hero("assault", star=1, pos=(300, 300)),
                make_hero("assault", star=1, pos=(400, 400)),
                make_hero("snow", star=1, pos=(500, 500)),
                make_hero("snow", star=1, pos=(600, 600)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS
        assert task._star3_done is True
        assert task._topup_done is True

    def test_exits_when_transform_exhausted(self):
        """放弃·变身用尽：整局预算归零即退（2026-10-05 单门定案）。"""
        task = self._stage_task()
        task._transform_used_count = 2
        decision = task.decide_action(
            Observation(frame_id=1, board=self._count4_board()), make_window_ctx()
        )
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS
        assert task._star3_done is True

    def test_exits_dead_board_instead_of_merging_main_c(self):
        """死局退出：占满只剩主C对也不合（保数量），回选技能。"""
        task = self._stage_task()
        heroes = [
            make_hero("assault", star=2, pos=(100, 100)),
            make_hero("assault", star=1, pos=(200, 200)),
            make_hero("assault", star=1, pos=(300, 300)),
            make_hero("assault", star=1, pos=(400, 400)),
        ]
        board = BoardSnapshot(
            frame_id=1, heroes=heroes, capacity=BoardCapacity(total_slots=4, occupied=4)
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS
        assert task._star3_done is True

    def test_summons_to_build_material_when_no_pairs(self):
        """环节内无对可合但有空位：召唤凑材料（继续培养，不退出）。"""
        task = self._stage_task()
        board = make_board(
            [
                make_hero("assault", star=2, pos=(100, 100)),
                make_hero("assault", star=1, pos=(200, 200)),
                make_hero("assault", star=1, pos=(300, 300)),
                make_hero("assault", star=1, pos=(400, 400)),
                make_hero("snow", star=1, pos=(500, 500)),
                make_hero("death_knight", star=1, pos=(600, 600)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "click"
        assert decision[0].tag == "summon_hero"
        assert task._star3_done is False

    def test_gold_wait_still_merges_in_stage(self):
        """环节内等金币恢复期间照常合成（合成不花金币，环节合并优先）。

        _star3_active 由培养决策置位；金币门控在决策之前，环节首帧金币
        转红会多等一个周期（见实现注释），这里测稳态。
        """
        task = self._stage_task()
        task._star3_active = True
        obs = Observation(
            frame_id=1,
            board=self._count4_board(),
            raw_data={"gold": {"current": 0, "summon_cost": 80, "summon_red": True}},
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "drag"
        assert decision[0].tag == "merge_heroes"

    def test_batch_summon_deferred_when_pairs_exist(self):
        """批次召唤让位：环节内有非主C对时先合成（批次或单动作），不点召唤。"""
        task = make_task(main_c_profile=self._profile(), batch_enabled=True)
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        task._topup_active = True
        decision = task.decide_action(
            Observation(frame_id=1, board=self._count4_board()), make_window_ctx()
        )
        assert decision is not None
        assert decision[0].kind == "drag"
        assert decision[0].tag in {"merge_heroes", "batch_merge"}

    def test_pre_star3_topup_keeps_summon_first(self):
        """数量未达标（<4）的回补阶段保持原策略：有空位先召唤不合成。"""
        task = self._stage_task()
        board = make_board(
            [
                make_hero("assault", star=2, pos=(100, 100)),
                make_hero("snow", star=1, pos=(500, 500)),
                make_hero("snow", star=1, pos=(600, 600)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "click"
        assert decision[0].tag == "summon_hero"

    def test_star3_done_resets_for_new_round(self):
        """环节完成标志只对本局有效：准备/被踢重进后复位。"""
        task = make_task()
        task._star3_done = True
        task._star3_active = True
        task.on_action_verified(
            Action(kind="click", tag="ready_match", target=(1, 1)),
            State.SELECT_OPENING_SKILLS,
        )
        assert task._star3_done is False
        assert task._star3_active is False
        task._star3_done = True
        task._star3_active = True
        task._reset_for_home_reentry()
        assert task._star3_done is False
        assert task._star3_active is False

    def test_disabled_profile_keeps_old_exit(self):
        """档案未开环节：数量达标仍按旧行为直接回选技能。"""
        task = self._stage_task(star3_topup_enabled=False)
        decision = task.decide_action(
            Observation(frame_id=1, board=self._count4_board()), make_window_ctx()
        )
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS

    def test_stage1_count_satisfied_skips_star3(self):
        """第一阶段数量已达标不进环节（实机 2026-10-01 frame=61 事故）：
        星级门禁直接回选技能——环节只属于第二阶段（选满 4 次技能回补后）。
        且不置位达标标志：第一阶段达标不短路第二阶段（2026-10-01 晚实机
        局 20:45 frame=98——变身出 3星强袭+数量达标把标志置位，整局跳过
        3星环节和凑狐狸；用户定案"第一阶段要在第二阶段凑"）。"""
        task = self._stage_task()
        task._topup_active = False  # 第一阶段（未走过「选满4次技能→回培养」）
        decision = task.decide_action(
            Observation(frame_id=1, board=self._count4_board()), make_window_ctx()
        )
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS
        assert task._star3_active is False
        assert task._star3_done is False
        task.on_action_verified(*decision)
        # 第一阶段不置位：选满 4 次技能后第二阶段照常发生
        assert task._topup_done is False

    def test_stage1_satisfied_still_topup_after_four_skills(self):
        """第一阶段数量达标 → 选满 4 次技能后仍回培养进第二阶段（回归：
        2026-10-01 晚前实现把达标标志在第一阶段置位，第二阶段被短路）。"""
        task = make_task(main_c_profile=self._profile())
        task.ctx.current_state = State.SELECT_MAIN_C_SKILLS
        task._main_skill_selections = self._profile().topup_after_skill_selections
        task._topup_done = False
        decision = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_topup"
        assert decision[1] == State.BUILD_MAIN_C

    def test_stage2_star3_assault_count_topup_continues(self):
        """第二阶段回补：3 星强袭在场但数量 3 < 4——数量回补不跳过，
        继续召唤凑第 4 个（2026-10-01 用户定案：强袭数量就要 4 个；
        实机 frame=455 场景，我曾误加"3星在场跳过凑数"短路已撤销）。"""
        task = self._stage_task()
        board = make_board(
            [
                make_hero("assault", star=3, pos=(100, 100)),
                make_hero("assault", star=1, pos=(200, 200)),
                make_hero("assault", star=1, pos=(300, 300)),  # 数量 3 < 4
                make_hero("snow", star=1, pos=(500, 500)),
                make_hero("death_knight", star=1, pos=(600, 600)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        # 棋盘有空位：按回补规则召唤（不退出、也不做 3 星培养动作）
        assert decision[0].tag == "summon_hero"
        assert task._star3_active is False
        assert task._star3_done is False

    def test_stage2_surplus_assault_pairs_merge_before_dead_exit(self):
        """第二阶段富余强袭对可合（实机 frame=150 案例）：7 个强袭含 2 星对、
        棋盘占满无空位——合并强袭对碰 3 星（保 ≥4），不死局退出。"""
        task = self._stage_task()
        heroes = [
            # 7 强袭：3×2星（互成对）+ 4×1星
            make_hero("assault", star=2, pos=(100, 100)),
            make_hero("assault", star=2, pos=(200, 200)),
            make_hero("assault", star=2, pos=(300, 300)),
            make_hero("assault", star=1, pos=(400, 400)),
            make_hero("assault", star=1, pos=(500, 500)),
            make_hero("assault", star=1, pos=(600, 600)),
            make_hero("assault", star=1, pos=(700, 700)),
            # 2 个非主C，互不成对、无 3 星（无可变身对象）
            make_hero("snow", star=2, pos=(800, 800)),
            make_hero("angel", star=1, pos=(900, 900)),
        ]
        board = BoardSnapshot(
            frame_id=1, heroes=heroes, capacity=BoardCapacity(total_slots=9, occupied=9)
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "drag"
        assert decision[0].tag == "merge_heroes"
        assert task._star3_done is False


class TestTopupAfterStage:
    """凑 lineup 英雄子环节（3星环节出口追加，2026-10-01 用户定案）。

    达成/变身用尽退出且场上无目标英雄：有空位先召唤、满盘合准入对（非主C
    且不含目标英雄）腾格、满盘无对且变身可用再变身；满盘+无对+无变身放弃。
    死局退出不进子环节。
    """

    def _profile(self, **overrides) -> MainCProfile:
        fields = {
            "display_name": "强袭",
            "hero_template_dir": "heroes/assault",
            "avoid_main_c_merge": True,
            "topup_after_skill_selections": 4,
            "topup_hero_count": 4,
            "star3_topup_enabled": True,
            "star3_topup_after_hero": "fox",
        }
        fields.update(overrides)
        return MainCProfile(**fields)

    def _after_task(self, *, transform: bool = False, **profile_overrides) -> CoopTask:
        task = make_task(
            main_c_profile=self._profile(**profile_overrides),
            transform_button_locator=TRANSFORM_LOCATOR if transform else None,
        )
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        task._topup_active = True
        return task

    def test_entered_on_achieve_and_summons_first(self):
        """达成 3星强袭且场上无狐狸：进子环节，有空位召唤优先。"""
        task = self._after_task(transform=True)
        board = make_board(
            [
                make_hero("assault", star=3, pos=(100, 100)),
                make_hero("assault", star=1, pos=(200, 200)),
                make_hero("assault", star=1, pos=(300, 300)),
                make_hero("assault", star=1, pos=(400, 400)),
                # 3 星死骑在场：子环节召唤优先，不当变身目标
                make_hero("death_knight", star=3, pos=(500, 500)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "click"
        assert decision[0].tag == "summon_hero"
        assert task._star3_after_active is True
        assert task._star3_done is True
        assert task._topup_done is True

    def test_not_entered_when_hero_already_present(self):
        """达成退出时狐狸已在场：不进子环节，直接回选技能。"""
        task = self._after_task()
        board = make_board(
            [
                make_hero("assault", star=3, pos=(100, 100)),
                make_hero("assault", star=1, pos=(200, 200)),
                make_hero("assault", star=1, pos=(300, 300)),
                make_hero("assault", star=1, pos=(400, 400)),
                make_hero("fox", star=1, pos=(500, 500)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS
        assert task._star3_after_active is False

    def test_not_entered_on_deadlock_exit(self):
        """死局退出（满盘只剩主C对）：不进子环节，直接回选技能。"""
        task = self._after_task()
        heroes = [
            make_hero("assault", star=3, pos=(100, 100)),
            make_hero("assault", star=1, pos=(200, 200)),
            make_hero("assault", star=1, pos=(300, 300)),
            make_hero("assault", star=1, pos=(400, 400)),
        ]
        board = BoardSnapshot(
            frame_id=1, heroes=heroes, capacity=BoardCapacity(total_slots=4, occupied=4)
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS
        assert task._star3_after_active is False

    def test_entered_on_transform_exhausted_exit(self):
        """放弃·变身用尽退出（棋盘还活着）：同样进子环节召唤。"""
        task = self._after_task()
        task._transform_used_count = 2
        board = make_board(
            [
                make_hero("assault", star=2, pos=(100, 100)),
                make_hero("assault", star=1, pos=(200, 200)),
                make_hero("assault", star=1, pos=(300, 300)),
                make_hero("assault", star=1, pos=(400, 400)),
                make_hero("angel", star=1, pos=(500, 500)),
                make_hero("death_knight", star=1, pos=(600, 600)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "summon_hero"
        assert task._star3_after_active is True

    def test_exits_to_skills_when_fox_appears(self):
        """子环节中场上出现狐狸：达标退出回选技能并清标志。"""
        task = self._after_task()
        task._star3_after_active = True
        task._star3_done = True
        board = make_board(
            [
                make_hero("assault", star=3, pos=(100, 100)),
                make_hero("assault", star=1, pos=(200, 200)),
                make_hero("assault", star=1, pos=(300, 300)),
                make_hero("assault", star=1, pos=(400, 400)),
                make_hero("fox", star=1, pos=(500, 500)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS
        assert task._star3_after_active is False

    def test_full_board_merges_allowed_pair(self):
        """满盘：合准入对（天使对）腾格，不合主C对。"""
        task = self._after_task()
        task._star3_after_active = True
        task._star3_done = True
        heroes = [
            make_hero("assault", star=3, pos=(100, 100)),
            make_hero("assault", star=1, pos=(200, 200)),
            make_hero("assault", star=1, pos=(300, 300)),
            make_hero("assault", star=1, pos=(400, 400)),
            make_hero("assault", star=1, pos=(450, 450)),
            make_hero("assault", star=1, pos=(500, 500)),
            make_hero("angel", star=1, pos=(600, 600)),
            make_hero("angel", star=1, pos=(700, 700)),
        ]
        board = BoardSnapshot(
            frame_id=1, heroes=heroes, capacity=BoardCapacity(total_slots=8, occupied=8)
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "drag"
        assert decision[0].tag == "merge_heroes"
        # 拖的是天使对，不是主C对
        assert decision[0].target != decision[0].end

    def test_full_board_uses_transform_when_no_pair(self):
        """满盘无准入对 + 变身可用且有 3 星目标：变身碰狐狸。"""
        task = self._after_task(transform=True)
        task._star3_after_active = True
        task._star3_done = True
        heroes = [
            make_hero("assault", star=3, pos=(100, 100)),
            make_hero("assault", star=1, pos=(200, 200)),
            make_hero("assault", star=1, pos=(300, 300)),
            make_hero("assault", star=1, pos=(400, 400)),
            make_hero("death_knight", star=3, pos=(500, 500)),
        ]
        board = BoardSnapshot(
            frame_id=1, heroes=heroes, capacity=BoardCapacity(total_slots=5, occupied=5)
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "transform_click_hero"

    def test_full_board_gives_up_without_pair_or_transform(self):
        """满盘 + 无准入对 + 无变身：入口帧即放弃回选技能（不点满盘召唤）。"""
        task = self._after_task(transform=True)
        # 变身预算用尽 → transform_usable=False
        task._transform_used_count = task._run_config.transform_max_per_round
        heroes = [
            make_hero("assault", star=3, pos=(100, 100)),
            make_hero("assault", star=1, pos=(200, 200)),
            make_hero("assault", star=1, pos=(300, 300)),
            make_hero("assault", star=1, pos=(400, 400)),
            make_hero("death_knight", star=1, pos=(500, 500)),
        ]
        board = BoardSnapshot(
            frame_id=1, heroes=heroes, capacity=BoardCapacity(total_slots=5, occupied=5)
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS
        assert task._star3_after_active is False

    def test_summon_first_with_transform_target_and_space(self):
        """有空位时召唤优先：即使有 3 星可变身对象也不先变身。"""
        task = self._after_task(transform=True)
        task._star3_after_active = True
        task._star3_done = True
        board = make_board(
            [
                make_hero("assault", star=3, pos=(100, 100)),
                make_hero("assault", star=1, pos=(200, 200)),
                make_hero("assault", star=1, pos=(300, 300)),
                make_hero("assault", star=1, pos=(400, 400)),
                make_hero("death_knight", star=3, pos=(500, 500)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "click"
        assert decision[0].tag == "summon_hero"


class TestDeadlockUnlock:
    """卡死解锁低星变身（2026-10-01 用户定案）。

    3星环节死局出口与凑 lineup 英雄子环节放弃出口，退出前先试低星变身
    （候选/星级规则见 strategy.select_deadlock_transform）；变身不可用
    （未接线/预算尽/放弃记忆抑制）或无候选才真退出。
    """

    def _profile(self, **overrides) -> MainCProfile:
        fields = {
            "display_name": "强袭",
            "hero_template_dir": "heroes/assault",
            "avoid_main_c_merge": True,
            "topup_after_skill_selections": 4,
            "topup_hero_count": 4,
            "star3_topup_enabled": True,
            "star3_topup_after_hero": "fox",
        }
        fields.update(overrides)
        return MainCProfile(**fields)

    def _task(self, *, transform: bool = True) -> CoopTask:
        task = make_task(
            main_c_profile=self._profile(),
            transform_button_locator=TRANSFORM_LOCATOR if transform else None,
        )
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        task._topup_active = True
        # 裸 RunConfig 的通配集为空，注入实机 default.yaml 配置
        task._wildcard_heroes = frozenset({"angel_phantom"})
        return task

    @staticmethod
    def _frame1199_board() -> BoardSnapshot:
        """实机 frame 1199 复刻：满盘 9/9，强袭 4 只两对（被数量底线拦）、
        非主C无同星对、无 ≥3星变身材料 → 真死局形状。"""
        heroes = [
            make_hero("death_knight", star=2, pos=(100, 100)),
            make_hero("angel", star=2, pos=(200, 100)),
            make_hero("assault", star=2, pos=(300, 100)),
            make_hero("death_knight", star=1, pos=(400, 100)),
            make_hero("assault", star=2, pos=(500, 100)),
            make_hero("assault", star=1, pos=(600, 100)),
            make_hero("assault", star=1, pos=(700, 100)),
            make_hero("fox", star=2, pos=(800, 100)),
            make_hero("fox", star=1, pos=(900, 100)),
        ]
        return BoardSnapshot(
            frame_id=1, heroes=heroes, capacity=BoardCapacity(total_slots=9, occupied=9)
        )

    def test_stage_deadlock_transforms_low_star_instead_of_exit(self):
        """frame 1199 复刻回归：死局 → 变 2星多数派格位序第一只（死骑），
        环节不退出、进入变身动作状态机。"""
        task = self._task()
        decision = task.decide_action(
            Observation(frame_id=1, board=self._frame1199_board()), make_window_ctx()
        )
        assert decision is not None
        assert decision[0].tag == "transform_click_hero"
        assert decision[0].target == (100, 100)
        assert task._star3_done is False
        assert task._topup_done is False
        assert task._transform_stage == "poll"

    def test_stage_deadlock_exits_when_transform_not_wired(self):
        """变身未接线（模板缺失降级）：解锁不可用，按原死局退出回选技能。"""
        task = self._task(transform=False)
        decision = task.decide_action(
            Observation(frame_id=1, board=self._frame1199_board()), make_window_ctx()
        )
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS
        assert task._star3_done is True
        assert task._transform_stage is None

    def test_stage_deadlock_exits_when_budget_exhausted(self):
        """变身预算用尽（局内早已花完）：解锁不可用，死局退出。"""
        task = self._task()
        task._transform_used_count = task._run_config.transform_max_per_round
        decision = task.decide_action(
            Observation(frame_id=1, board=self._frame1199_board()), make_window_ctx()
        )
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS

    def test_stage_deadlock_exits_when_all_unlock_targets_given_up(self):
        """按英雄放弃记忆记满全部解锁候选（按钮逐一超时，如皮肤升级产物）：
        解锁无候选，死局退出回选技能——等价旧整盘签名抑制，且不挡
        未失败的目标（2026-10-03 记法变更）。"""
        task = self._task()
        task.decide_action(
            Observation(frame_id=1, board=self._frame1199_board()), make_window_ctx()
        )
        assert task._transform_failed_targets == set()  # 前置：本盘未放弃过
        # 模拟低星解锁候选逐一按钮超时被记
        task._transform_failed_targets = {
            ("death_knight", 2, (100, 100)),
            ("angel", 2, (200, 100)),
            ("death_knight", 1, (400, 100)),
            ("fox", 2, (800, 100)),
            ("fox", 1, (900, 100)),
        }
        task._transform_stage = None
        task._transform_target = None
        task._transform_used_count = 0
        decision = task.decide_action(
            Observation(frame_id=2, board=self._frame1199_board()), make_window_ctx()
        )
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS

    def test_substage_giveup_transforms_low_star_instead_of_exit(self):
        """子环节放弃出口：变 1星多数派格位序第一只（天使），子环节继续
        （1/3 直接变出狐狸当场达标）。"""
        task = self._task()
        task._star3_after_active = True
        task._star3_done = True
        task._topup_done = True
        heroes = [
            make_hero("assault", star=3, pos=(100, 100)),
            make_hero("assault", star=1, pos=(200, 100)),
            make_hero("assault", star=1, pos=(300, 100)),
            make_hero("assault", star=1, pos=(400, 100)),
            make_hero("angel", star=2, pos=(500, 100)),
            make_hero("death_knight", star=2, pos=(600, 100)),
            make_hero("angel", star=1, pos=(700, 100)),
            make_hero("death_knight", star=1, pos=(800, 100)),
            make_hero("monkey", star=1, pos=(900, 100)),
        ]
        board = BoardSnapshot(
            frame_id=1, heroes=heroes, capacity=BoardCapacity(total_slots=9, occupied=9)
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "transform_click_hero"
        assert decision[0].target == (700, 100)
        assert task._star3_after_active is True
        assert task._transform_stage == "poll"

    def test_substage_giveup_exits_when_pool_empty(self):
        """子环节放弃且无低星候选（只剩通配与数量 4 的主C）：真放弃退出。"""
        task = self._task()
        task._star3_after_active = True
        task._star3_done = True
        task._topup_done = True
        heroes = [
            make_hero("assault", star=3, pos=(100, 100)),
            make_hero("assault", star=1, pos=(200, 100)),
            make_hero("assault", star=1, pos=(300, 100)),
            make_hero("assault", star=1, pos=(400, 100)),
            make_hero("angel_phantom", star=1, pos=(500, 100)),
            make_hero("angel_phantom", star=2, pos=(600, 100)),
        ]
        board = BoardSnapshot(
            frame_id=1, heroes=heroes, capacity=BoardCapacity(total_slots=6, occupied=6)
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS
        assert task._star3_after_active is False

    def test_substage_full_board_merges_surplus_main_pair(self):
        """frame 629 复刻回归（2026-10-02 富余主C对放行）：强袭 6 只满盘 →
        合 assault2 对（消耗 2 剩 4，产物 3星随机有 1/4 是狐狸），不再放弃。"""
        task = self._task()
        task._star3_after_active = True
        task._star3_done = True
        task._topup_done = True
        assault2_positions = {(200, 100), (300, 100), (500, 100)}
        heroes = [
            make_hero("angel", star=1, pos=(100, 100)),
            make_hero("assault", star=2, pos=(200, 100)),
            make_hero("assault", star=2, pos=(300, 100)),
            make_hero("assault", star=3, pos=(400, 100)),
            make_hero("assault", star=2, pos=(500, 100)),
            make_hero("assault", star=1, pos=(600, 100)),
            make_hero("death_knight", star=1, pos=(700, 100)),
            make_hero("assault", star=1, pos=(800, 100)),
        ]
        board = BoardSnapshot(
            frame_id=1, heroes=heroes, capacity=BoardCapacity(total_slots=8, occupied=8)
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "drag"
        assert decision[0].tag == "merge_heroes"
        assert {decision[0].target, decision[0].end} <= assault2_positions
        assert task._star3_after_active is True

    def test_substage_five_assaults_still_gives_up_without_phantom(self):
        """强袭 5 只无幻天使：主C对消耗 2 破底线仍不合 → 放弃（原行为锁）。"""
        task = self._task(transform=False)
        task._star3_after_active = True
        task._star3_done = True
        task._topup_done = True
        heroes = [
            make_hero("assault", star=3, pos=(100, 100)),
            make_hero("assault", star=2, pos=(200, 100)),
            make_hero("assault", star=2, pos=(300, 100)),
            make_hero("assault", star=1, pos=(400, 100)),
            make_hero("assault", star=1, pos=(500, 100)),
            make_hero("angel", star=1, pos=(600, 100)),
            make_hero("death_knight", star=1, pos=(700, 100)),
        ]
        board = BoardSnapshot(
            frame_id=1, heroes=heroes, capacity=BoardCapacity(total_slots=7, occupied=7)
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS
        assert task._star3_after_active is False

    def test_stage_full_board_merges_phantom_main_pair_at_five(self):
        """环节内满盘 + 强袭 5 只 + 幻天使：幻天使+强袭对只消耗 1，
        可合（原先按消耗 2 拦死），环节留场不死局。"""
        task = self._task()
        heroes = [
            make_hero("assault", star=2, pos=(100, 100)),
            make_hero("assault", star=1, pos=(200, 100)),
            make_hero("assault", star=1, pos=(300, 100)),
            make_hero("assault", star=1, pos=(400, 100)),
            make_hero("assault", star=1, pos=(500, 100)),
            make_hero("angel_phantom", star=1, pos=(600, 100)),
            make_hero("angel", star=2, pos=(700, 100)),
            make_hero("death_knight", star=2, pos=(800, 100)),
        ]
        board = BoardSnapshot(
            frame_id=1, heroes=heroes, capacity=BoardCapacity(total_slots=8, occupied=8)
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "drag"
        assert decision[0].tag == "merge_heroes"
        assert (600, 100) in {decision[0].target, decision[0].end}
        assert task._star3_active is True
        assert task._star3_done is False


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
            Observation(frame_id=8),   # chat_open_unverified（档案未配置页签文字，跳过确认）
            Observation(frame_id=9),   # open_mode_tab（页签 hotspot 路径）
            Observation(frame_id=10),  # open_difficulty_dialog
            Observation(frame_id=11),  # settle
            difficulty_observation(2, 1, frame_id=12),
            difficulty_observation(2, 1, frame_id=13),
            difficulty_observation(2, 1, frame_id=14),
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
            "chat_open_unverified",
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
        # 未识别到页签文字：不盲点固定坐标，返回等待动作重试下一帧
        # （显式 wait 而不是 None——None 会被 Runner 计入无进展预算
        #   （error_restart_max_consecutive=3 < miss_limit=8），2026-09-16 实机）
        for frame_id in (3, 4):
            decision = task.decide_action(Observation(frame_id=frame_id), make_window_ctx())
            assert decision is not None
            assert decision[0].tag == "mode_tab_pending"
            assert decision[0].kind == "wait"
        # 连续 miss 达上限：报错停止（大概率不在招募页，任何点击都不安全）
        with pytest.raises(RuntimeError, match="保卫宠物"):
            task.decide_action(Observation(frame_id=5), make_window_ctx())

    def test_mode_tab_ocr_hit_resets_miss_counter(self):
        task = self._task_on_recruit_page(
            make_mode_profile(mode_tab_text="合作模式"),
        )
        task._mode_tab_miss_limit = 2
        miss = Observation(frame_id=3, raw_data={"mode_tab_rows": []})
        miss_decision = task.decide_action(miss, make_window_ctx())
        assert miss_decision is not None and miss_decision[0].tag == "mode_tab_pending"
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
        # 聊天点击生效确认：档案未配置页签文字，1 帧跳过
        unverified = task.decide_action(Observation(frame_id=3), make_window_ctx())
        assert unverified is not None and unverified[0].kind == "wait"
        task.on_action_verified(*unverified)
        tab = task.decide_action(Observation(frame_id=4), make_window_ctx())
        assert tab is not None and tab[0].tag == "open_mode_tab"
        task.on_action_verified(*tab)
        dialog = task.decide_action(Observation(frame_id=5), make_window_ctx())
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

    def test_return_auto_skipped_when_settlement_page_closed(self):
        """结算页被游戏自动关闭（2026-11-05/06 实机）：读到挑战页锚点即视同
        返回完成，局数+1 并从模式聊天入口开下一局，不再等返回按钮。"""
        task = make_task(
            max_rounds=2,
            hotspots=self._spots_with_tabs(),
            mode_profile=make_mode_profile(),
        )
        task.ctx.current_state = State.CHECK_ROUND_LIMIT

        decision = task.decide_action(
            Observation(frame_id=1, raw_data={"challenge_text": "开始挑战"}),
            make_window_ctx(),
        )
        assert decision is not None
        action, to_state = decision
        assert action.tag == "return_auto_skipped"
        assert action.kind == "wait"
        assert to_state == State.FIND_COOP

        # 视同返回完成：局数+1，下一局从模式聊天进入
        task.on_action_verified(action, to_state)
        assert task.ctx.round_count == 1
        task.ctx.current_state = to_state
        chat = task.decide_action(Observation(frame_id=2), make_window_ctx())
        assert chat is not None
        assert chat[0].tag == "open_coop_chat"

    def test_return_button_wins_over_challenge_anchor(self):
        """返回按钮与挑战锚点同帧可见时优先点返回（结算页在场）。"""
        task = make_task(
            hotspots=self._spots_with_tabs(),
            mode_profile=make_mode_profile(),
        )
        task.ctx.current_state = State.CHECK_ROUND_LIMIT
        match = MatchResult("buttons/fan_hui.png", (500, 1000), 0.9)

        decision = task.decide_action(
            Observation(
                frame_id=1,
                raw_data={
                    "return_button_visible": True,
                    "return_button_match": match,
                    "challenge_text": "开始挑战",
                },
            ),
            make_window_ctx(),
        )
        assert decision is not None
        assert decision[0].tag == "return_result"
        assert decision[0].target == (500, 1000)

    def test_return_auto_skipped_completes_on_last_round(self):
        """自动返回兜底命中最后一局：视同完成后进入 COMPLETED。"""
        task = make_task(max_rounds=1, mode_profile=make_mode_profile())
        task.ctx.current_state = State.CHECK_ROUND_LIMIT

        decision = task.decide_action(
            Observation(frame_id=1, raw_data={"challenge_text": "开始挑战"}),
            make_window_ctx(),
        )
        assert decision is not None
        assert decision[0].tag == "return_auto_skipped"
        assert decision[1] == State.COMPLETED

    def test_check_round_limit_requests_challenge_page_mode(self):
        """结算返回步骤的观察模式：须在 recruit_step 遗留值之前命中。"""
        task = make_task(
            hotspots=self._spots_with_tabs(),
            mode_profile=make_mode_profile(),
        )
        task.ctx.current_state = State.CHECK_ROUND_LIMIT
        # 结算态携带局内遗留的入口步骤值（如模式选择弹窗步骤），不得误命中
        task._recruit_step = _RecruitStep.SELECT_MODE_CARD
        assert task.observation_mode() == "challenge_page"

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

    def test_entry_probe_jumps_to_chat_when_already_on_challenge_page(self):
        """领地入口探测（2026-09-25）：启动即停在挑战页（读到「开始挑战」）
        时，整段领地→模式选择→选卡导航全部跳过，直接点聊天进招募。"""
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        decision = task.decide_action(
            Observation(frame_id=1, raw_data={"challenge_text": "开始挑战"}),
            make_window_ctx(),
        )
        assert decision is not None
        action, to_state = decision
        assert action.tag == "challenge_page_confirmed"
        assert to_state == State.FIND_COOP
        task.on_action_verified(*decision)
        assert task._recruit_step == _RecruitStep.OPEN_CHALLENGE_CHAT
        # 跳过后与正常到达挑战页同链：聊天 → 生效确认
        chat = task.decide_action(Observation(frame_id=2), make_window_ctx())
        assert chat is not None and chat[0].tag == "open_challenge_chat"

    def test_entry_probe_without_anchor_keeps_territory_flow(self):
        """锚点读空（不在挑战页）：按导航文字正常走领地入口。"""
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        decision = task.decide_action(
            Observation(
                frame_id=1,
                raw_data={"challenge_text": "", "territory_nav_text": "领地"},
            ),
            make_window_ctx(),
        )
        assert decision is not None
        assert decision[0].tag == "open_territory"

    def test_entry_probe_weekly_buff_profile_closes_buff_first(self):
        """配置每周弹窗的模式（保卫宠物）：探测命中先进关弹窗步骤。"""
        task = make_task(
            mode_profile=make_mode_profile(
                challenge_text="挑战", weekly_buff_text="知道"
            )
        )
        task.ctx.current_state = State.FIND_COOP
        decision = task.decide_action(
            Observation(frame_id=1, raw_data={"challenge_text": "挑战"}),
            make_window_ctx(),
        )
        assert decision is not None and decision[0].tag == "challenge_page_confirmed"
        task.on_action_verified(*decision)
        assert task._recruit_step == _RecruitStep.CLOSE_WEEKLY_BUFF

    def test_entry_probe_invite_mode_enters_click_chain(self):
        """邀请者启动即停在挑战页：跳过领地导航直接进邀请点击链。"""
        task = make_task()
        task._invite_mode = True
        task.ctx.current_state = State.FIND_COOP
        decision = task.decide_action(
            Observation(frame_id=1, raw_data={"challenge_text": "开始挑战"}),
            make_window_ctx(),
        )
        assert decision is not None
        assert decision[0].tag == "invite_nav_done"
        assert decision[1] == State.INVITE_CLICK
        task.on_action_verified(*decision)
        assert task._recruit_step == _RecruitStep.INVITE_ENTRY_DONE
        assert task._invite_step == 0

    def test_kick_reentry_on_challenge_page_skips_navigation(self):
        """被踢/等待超时重入时游戏停在挑战页：领地入口探测直接进招募。"""
        task = make_task()
        task.ctx.current_state = State.ENTER_MATCH
        task._territory_return_count = (
            task._run_config.territory_return_confirm_frames - 1
        )
        decision = task.decide_action(
            Observation(frame_id=1, raw_data={"territory_nav_text": "返回"}),
            make_window_ctx(),
        )
        assert decision is not None and decision[0].tag == "kick_reentry_territory"
        task.on_action_verified(*decision)
        task.ctx.current_state = State.FIND_COOP
        probe = task.decide_action(
            Observation(
                frame_id=2,
                raw_data={"challenge_text": "开始挑战", "territory_nav_text": ""},
            ),
            make_window_ctx(),
        )
        assert probe is not None and probe[0].tag == "challenge_page_confirmed"
        task.on_action_verified(*probe)
        assert task._recruit_step == _RecruitStep.OPEN_CHALLENGE_CHAT

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
        # 聊天点击先进生效确认（档案未配置页签文字 → 1 帧跳过），再进页签
        # 步骤；页签去向是首局难度弹窗
        chat = task.decide_action(Observation(frame_id=2), make_window_ctx())
        assert chat is not None and chat[0].tag == "open_challenge_chat"
        task.on_action_verified(*chat)
        assert task._recruit_step == _RecruitStep.CONFIRM_CHAT_OPEN
        unverified = task.decide_action(Observation(frame_id=3), make_window_ctx())
        assert unverified is not None and unverified[0].tag == "chat_open_unverified"
        task.on_action_verified(*unverified)
        assert task._recruit_step == _RecruitStep.SELECT_MODE_TAB
        assert task._after_tab_step == _RecruitStep.OPEN_DIFFICULTY_DIALOG


class TestChatOpenConfirm:
    """聊天点击生效确认：页签行读到本模式页签才算进入招募，读不到则重点。

    2026-09-16 实机回归：结算返回后挑战页盖着金币道具说明卡，聊天点击
    只关掉弹窗、页面未切换，页签 OCR 读不到被无进展预算保守停止。生效
    判据用页签行（2026-09-17 实机修正）：挑战页锚点「开始挑战」在招募
    频道页同样可见，锚点无法区分两页——首版锚点判据导致确认永不推进、
    9 次重点全部点在招募页空白处直到超时。
    """

    def _task_on_confirm(self, **kwargs) -> CoopTask:
        task = make_task(
            mode_profile=make_mode_profile(mode_tab_text="合作模式"),
            **kwargs,
        )
        task.ctx.current_state = State.FIND_COOP
        task._after_tab_step = _RecruitStep.OPEN_DIFFICULTY_DIALOG
        task._enter_chat_open_confirm()
        assert task._recruit_step == _RecruitStep.CONFIRM_CHAT_OPEN
        return task

    @staticmethod
    def _obs(frame_id: int, tab_visible: bool) -> Observation:
        # 挑战页的页签行区域读到的是「拥有：」等无关文字（2026-09-16 实机）
        row = (
            ModeTabRow("合作模式", 0.8, (314, 1535))
            if tab_visible
            else ModeTabRow("拥有：", 0.7, (314, 1535))
        )
        return Observation(frame_id=frame_id, raw_data={"mode_tab_rows": [row]})

    def test_observation_mode_reads_tab_row(self):
        task = self._task_on_confirm()
        assert task.observation_mode() == "mode_tab"

    def test_reclick_when_tab_missing_then_confirm_after_reclick(self):
        """事故回放：首次点击被弹窗吃掉（页签读不到）→ 重点 → 页面切换成功。"""
        task = self._task_on_confirm()
        ctx = make_window_ctx()

        # 首次点击未生效：页签行连续 4 帧读不到 → 前 3 帧等待，第 4 帧重点聊天
        for frame_id in (1, 2, 3):
            decision = task.decide_action(self._obs(frame_id, False), ctx)
            assert decision is not None and decision[0].tag == "chat_open_pending"
        reclick = task.decide_action(self._obs(4, False), ctx)
        assert reclick is not None
        assert reclick[0].tag == "chat_reclick"
        assert reclick[0].kind == "click"
        # 重点坐标 = 模式聊天按钮（coop_chat hotspot 换算的客户区像素）
        assert reclick[0].target == (int(0.5 * 923), int(0.6 * 1723))
        task.on_action_verified(*reclick)
        assert task._recruit_step == _RecruitStep.CONFIRM_CHAT_OPEN

        # 重点生效（关掉了挡路的弹窗并真的点到聊天）：页签连续 2 帧可读
        pending = task.decide_action(self._obs(5, True), ctx)
        assert pending is not None and pending[0].tag == "chat_open_pending"
        confirmed = task.decide_action(self._obs(6, True), ctx)
        assert confirmed is not None and confirmed[0].tag == "chat_open_confirmed"
        task.on_action_verified(*confirmed)
        assert task._recruit_step == _RecruitStep.SELECT_MODE_TAB

    def test_budget_exhausted_stops_conservatively(self):
        """重点预算用尽仍读不到页签：decide 返回 None（调度层保守停止）。"""
        task = self._task_on_confirm()
        task._chat_open_max_reclicks = 2
        ctx = make_window_ctx()

        frame_id = 0
        for _ in range(2):
            for _ in range(3):
                frame_id += 1
                decision = task.decide_action(self._obs(frame_id, False), ctx)
                assert decision is not None and decision[0].tag == "chat_open_pending"
            frame_id += 1
            reclick = task.decide_action(self._obs(frame_id, False), ctx)
            assert reclick is not None and reclick[0].tag == "chat_reclick"
            task.on_action_verified(*reclick)

        # 预算用尽后再读满连续帧：页签仍读不到 → 无可用动作（保守停止）
        for _ in range(3):
            frame_id += 1
            decision = task.decide_action(self._obs(frame_id, False), ctx)
            assert decision is not None and decision[0].tag == "chat_open_pending"
        frame_id += 1
        assert task.decide_action(self._obs(frame_id, False), ctx) is None

    def test_single_tab_hit_does_not_advance(self):
        """页签单帧读到不推进（连续 2 帧才算生效），读不到后计数清零。"""
        task = self._task_on_confirm()
        ctx = make_window_ctx()

        decision = task.decide_action(self._obs(1, True), ctx)
        assert decision is not None and decision[0].tag == "chat_open_pending"
        assert task._recruit_step == _RecruitStep.CONFIRM_CHAT_OPEN

        decision = task.decide_action(self._obs(2, False), ctx)
        assert decision is not None and decision[0].tag == "chat_open_pending"
        assert task._chat_open_tab_hits == 0

    def test_recruit_page_anchor_text_does_not_block_confirm(self):
        """2026-09-17 实机回归：招募页底部同样有「开始挑战」，确认不得依赖
        锚点消失——只要页签行可读就推进（锚点文字同时存在也不影响）。"""
        task = self._task_on_confirm()
        ctx = make_window_ctx()
        obs = Observation(
            frame_id=1,
            raw_data={
                # 同帧两者都可读：页签行 + 挑战页锚点文字（招募页实况）
                "mode_tab_rows": [ModeTabRow("合作模式", 0.8, (314, 1535))],
                "challenge_text": "开始挑战",
            },
        )
        decision = task.decide_action(obs, ctx)
        assert decision is not None and decision[0].tag == "chat_open_pending"
        confirmed = task.decide_action(
            Observation(frame_id=2, raw_data=dict(obs.raw_data)), ctx
        )
        assert confirmed is not None and confirmed[0].tag == "chat_open_confirmed"
        assert task._recruit_step == _RecruitStep.SELECT_MODE_TAB

    def test_profile_without_tab_text_skips_confirm(self):
        """档案未配置页签文字：页签行验证无从谈起，保持旧行为直接选页签。"""
        task = make_task(mode_profile=make_mode_profile(mode_tab_text=None))
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.OPEN_CHALLENGE_CHAT
        chat = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert chat is not None and chat[0].tag == "open_challenge_chat"
        task.on_action_verified(*chat)
        assert task._recruit_step == _RecruitStep.CONFIRM_CHAT_OPEN
        decision = task.decide_action(Observation(frame_id=2), make_window_ctx())
        assert decision is not None and decision[0].tag == "chat_open_unverified"
        assert task._recruit_step == _RecruitStep.SELECT_MODE_TAB

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

    def _mode_select_territory_observation(self, frame_id: int) -> Observation:
        """弹窗未开时领地页的读数：标题带是传送门标签，卡名行是领地建筑。"""
        return Observation(
            frame_id=frame_id,
            raw_data={
                "mode_select_visible": False,
                "mode_select_title_rows": [
                    ModeTabRow("拥有：", 0.8, (300, 320)),
                    ModeTabRow("合作模式", 0.8, (558, 363)),
                ],
                "mode_card_rows": [ModeTabRow("玩法大", 0.8, (200, 1180))],
            },
        )

    def test_mode_select_confirm_reclicks_when_territory_label_still_visible(self):
        """领地「拥有：」标签仍在=弹窗确定未开：重点进入按钮是安全的。

        2026-09-26 18:05 实机：点击前画面被临时覆盖物挡住，进入按钮点击
        被吃，本步骤此前不做重点，干等 60 秒后保守退出。
        """
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.CONFIRM_MODE_SELECT_OPEN
        decision = None
        for frame_id in range(1, _ENTRY_RECLICK_MISSES + 1):
            decision = task.decide_action(
                self._mode_select_territory_observation(frame_id), make_window_ctx()
            )
            assert decision is not None
            if frame_id < _ENTRY_RECLICK_MISSES:
                assert decision[0].tag == "mode_select_open_pending"
        assert decision[0].tag == "open_mode_select"
        assert "重点进入按钮" in decision[0].reason
        task.on_action_verified(*decision)
        assert task._recruit_step == _RecruitStep.CONFIRM_MODE_SELECT_OPEN
        assert task._entry_reclicks == 1

    def test_mode_select_confirm_keeps_waiting_when_reads_ambiguous(self):
        """领地标签消失但标题卡名都没命中（读数模糊）：不重点，只干等。

        弹窗可能开着只是 OCR 全漏读，重点会误点卡面（2026-09-17 教训）。
        """
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.CONFIRM_MODE_SELECT_OPEN
        for frame_id in range(1, _ENTRY_RECLICK_MISSES + 5):
            decision = task.decide_action(
                Observation(frame_id=frame_id), make_window_ctx()
            )
            assert decision is not None
            assert decision[0].tag == "mode_select_open_pending"
        assert task._entry_reclicks == 0

    def test_mode_select_confirm_stops_when_reclick_budget_exhausted(self):
        """重点预算用尽弹窗仍未开：保守停止，不无限重试。"""
        task = make_task()
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.CONFIRM_MODE_SELECT_OPEN
        task._entry_reclicks = task._entry_max_reclicks
        task._entry_misses = _ENTRY_RECLICK_MISSES
        assert (
            task.decide_action(
                self._mode_select_territory_observation(1), make_window_ctx()
            )
            is None
        )

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


class TestOpeningMergeFirst:
    """开局合并先行（2026-11-19 用户定案）：强制召唤点满后首批规划有可合对
    时召唤让位一轮——先合成腾格，下一轮把剩余空位一次批次召唤填满；
    无对可合直接召唤，标记一次性消费，局间重置、邀请者不受影响。"""

    def _task_through_forced_phase(
        self, minimum_summons: int = 3, **kwargs
    ) -> CoopTask:
        task = make_task(minimum_summons=minimum_summons, batch_enabled=True, **kwargs)
        task.ctx.current_state = State.BUILD_MAIN_C
        obs = Observation(frame_id=1, board=make_board([]))
        for expected in range(1, minimum_summons + 1):
            decision = task.decide_action(obs, make_window_ctx())
            assert decision is not None
            assert decision[0].tag == "required_summon"
            task.on_action_verified(*decision)
            assert task._summon_count == expected
        assert task._opening_merge_pending is True
        return task

    def _pair_board(self) -> Observation:
        return Observation(
            frame_id=2,
            board=make_board(
                [
                    make_hero("snow", star=1, pos=(10, 10)),
                    make_hero("snow", star=1, pos=(20, 20)),
                ]
            ),
        )

    def test_defers_batch_summon_when_mergeable_pair(self):
        """点满后首批规划有可合对：召唤让位，先排批次合成。"""
        task = self._task_through_forced_phase()
        decision = task.decide_action(self._pair_board(), make_window_ctx())
        assert decision is not None
        action, _ = decision
        # 有空位也有对：正常优先级是先召唤（见 TestBatchSummon），让位后改排合成
        assert action.tag == "batch_merge"
        assert len(task._batch_merge_queue) == 1
        assert task._batch_summon_remaining == 0
        # 标记一次性消费
        assert task._opening_merge_pending is False

    def test_plans_batch_summon_directly_without_pair(self):
        """点满后首批规划无可合对：标记作废，照旧批次召唤填空位。"""
        task = self._task_through_forced_phase()
        obs = Observation(
            frame_id=2,
            board=make_board(
                [
                    make_hero("snow", star=1, pos=(10, 10)),
                    make_hero("death_knight", star=1, pos=(20, 20)),
                ]
            ),
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "batch_summon"
        assert task._opening_merge_pending is False

    def test_deferral_consumed_after_one_merge_round(self):
        """让位只一轮：合成批消费完后，下一轮即使场上仍有对也恢复召唤先行。"""
        task = self._task_through_forced_phase()
        obs = self._pair_board()
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        task.on_action_verified(*decision)
        # 逐拖消费完合成队列（拖动后棋盘实际会变，这里只验证规划走向）
        while task._batch_merge_queue:
            step = task.decide_action(obs, make_window_ctx())
            assert step is not None
            assert step[0].tag == "batch_merge"
            task.on_action_verified(*step)
        # 队列清空后重新规划：标记已消费，恢复「有空位先召唤」
        next_decision = task.decide_action(obs, make_window_ctx())
        assert next_decision is not None
        assert next_decision[0].tag == "batch_summon"

    def test_disabled_config_keeps_summon_first(self):
        """开关关闭（opening_merge_first=False）：行为与旧优先级完全一致。"""
        task = make_task(minimum_summons=3, batch_enabled=True)
        task._run_config.opening_merge_first = False
        task.ctx.current_state = State.BUILD_MAIN_C
        obs = Observation(frame_id=1, board=make_board([]))
        for _ in range(3):
            decision = task.decide_action(obs, make_window_ctx())
            assert decision is not None
            task.on_action_verified(*decision)
        assert task._opening_merge_pending is False
        decision = task.decide_action(self._pair_board(), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "batch_summon"

    def test_invite_mode_never_sets_flag(self):
        """邀请者打法独立（准备完成+凑阵容批次）：点满不置标记、不受让位影响。"""
        task = self._task_through_forced_phase()
        task._invite_mode = True
        task._invite_ready_clicked = True  # 跳过邀请者「准备完成」分支直达规划
        task._opening_merge_pending = True  # 模拟残留，验证规划处不消费也不让位
        decision = task.decide_action(self._pair_board(), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "batch_summon"
        # 邀请者点满路径不置标记：单独验证置位条件
        fresh = make_task(minimum_summons=3, batch_enabled=True)
        fresh._invite_mode = True
        fresh.ctx.current_state = State.BUILD_MAIN_C
        obs = Observation(frame_id=1, board=make_board([]))
        for _ in range(3):
            step = fresh.decide_action(obs, make_window_ctx())
            assert step is not None
            fresh.on_action_verified(*step)
        assert fresh._opening_merge_pending is False

    def test_reset_for_next_round_clears_flag(self):
        task = self._task_through_forced_phase()
        task._reset_for_next_round()
        assert task._opening_merge_pending is False

    def test_reset_for_home_reentry_clears_flag(self):
        task = self._task_through_forced_phase()
        task._reset_for_home_reentry()
        assert task._opening_merge_pending is False

    def test_observation_mode_forced_summon_phase(self):
        """连点期声明 forced_summon 模式（perception 据此跳过棋盘分类）；
        点满后与非 BUILD_MAIN_C 状态不再声明。"""
        task = make_task(minimum_summons=3)
        task.ctx.current_state = State.BUILD_MAIN_C
        assert task.observation_mode() == "forced_summon"
        task._summon_count = 3
        assert task.observation_mode() != "forced_summon"
        task.ctx.current_state = State.SELECT_OPENING_SKILLS
        task._summon_count = 0
        assert task.observation_mode() != "forced_summon"


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

    def test_triggers_even_when_main_c_target_reached(self):
        """2026-09-17 变更：主C达标不再抑制变身——3星非主C同样返回目标。"""
        task = make_task()
        board = make_board(
            [
                make_hero("assault", star=2, pos=(100, 100)),
                make_hero("snow", star=3, pos=(300, 300)),
            ]
        )
        target = task._strategy.select_transform(board, 0, 2)
        assert target is not None
        assert target.hero_type == "snow"
        assert target.star_level == 3

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

    def test_first_phase_skips_low_star_targets(self):
        """一阶段只变 3/4 星（2026-10-05 用户定案）：低星非主C不是目标。"""
        task = self._ready_task()
        assert task._topup_active is False
        board = make_board([make_hero("snow", star=1, pos=(100, 100))])
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag != "transform_click_hero"
        assert task._transform_stage is None

    def test_topup_segment_skips_low_star_targets(self):
        """回补段（数量未满 4）与一阶段同门槛：只变 3/4 星，任意星级变身
        留给 3星环节/凑狐狸子环节（2026-10-05 用户定案）。"""
        task = self._ready_task()
        task._topup_active = True
        board = make_board([make_hero("snow", star=1, pos=(100, 100))])
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag != "transform_click_hero"
        assert task._transform_stage is None

    def test_star3_stage_transforms_any_star(self):
        """3星环节内放开任意星级（2026-10-05 用户定案）：1星非主C也是目标。"""
        task = make_task(
            main_c_profile=MainCProfile(
                display_name="强袭",
                hero_template_dir="heroes/assault",
                avoid_main_c_merge=True,
                topup_after_skill_selections=4,
                topup_hero_count=4,
                star3_topup_enabled=True,
            ),
            transform_button_locator=TRANSFORM_LOCATOR,
        )
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        task._topup_active = True
        task._star3_active = True
        board = make_board(
            [
                make_hero("assault", star=1, pos=(100, 100)),
                make_hero("assault", star=1, pos=(200, 200)),
                make_hero("assault", star=1, pos=(300, 300)),
                make_hero("assault", star=1, pos=(400, 400)),
                make_hero("snow", star=1, pos=(500, 500)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "click"
        assert decision[0].tag == "transform_click_hero"
        assert decision[0].target == (500, 500)
        assert task._transform_stage == "poll"

    def test_triggers_click_hero_on_three_star_non_main_c(self):
        """一阶段 3 星非主C 照常触发（高价值滚动不受阶段门槛限制）。"""
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
        """非回补阶段主C达标：优先转选技能（触发点位于 main_c_ready 之后）。

        2026-09-17 变更只放宽 select_transform 的条件；回补阶段数量未达标
        时的触发见下方 test_triggers_in_topup_with_target_main_c。
        """
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

    def test_triggers_in_topup_with_target_main_c(self):
        """回补阶段主C星级达标但数量未达标：合并产出3星非主C后下一轮触发变身。

        实机日志（2026-09-17）：主C强袭2星在场（target_star_level=2 已
        达标）、处于补数量阶段，棋盘出现 angel3星——旧策略因「无达标主C」
        抑制不变身，只继续召唤/合成。
        """
        task = make_task(
            main_c_profile=MainCProfile(
                display_name="强袭",
                hero_template_dir="heroes/assault",
                topup_after_skill_selections=4,
                topup_hero_count=4,
            ),
            transform_button_locator=TRANSFORM_LOCATOR,
        )
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        task._topup_active = True
        board = make_board(
            [
                make_hero("assault", star=2, pos=(100, 100)),
                make_hero("angel", star=3, pos=(300, 300)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "click"
        assert decision[0].tag == "transform_click_hero"
        assert decision[0].target == (300, 300)
        assert task._transform_stage == "poll"
        assert task._transform_target is not None
        assert task._transform_target.hero_type == "angel"

    def test_no_trigger_when_charges_exhausted(self):
        task = self._ready_task()
        task._topup_active = True
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
        assert task._transform_failed_targets == {("snow", 3, (100, 100))}

    def test_giveup_memory_survives_unrelated_board_evolution(self):
        """按英雄放弃记忆（2026-10-03 用户定案）：棋盘无关演化（召唤/合成/
        识别漂移）不复活已放弃的目标——旧整盘签名记法每演化一次就复活
        重烧 ~6s（实机 01:46 局皮肤升级 fox3 共 7 次）。"""
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
        # 召唤后棋盘多了英雄：snow3 仍在场且仍被记，不再对它触发
        board2 = make_board(
            [
                make_hero("snow", star=3, pos=(100, 100)),
                make_hero("angel", star=1, pos=(300, 300)),
            ]
        )
        decision = task.decide_action(
            Observation(frame_id=2, board=board2), make_window_ctx()
        )
        assert decision is not None
        assert decision[0].tag != "transform_click_hero"
        assert task._transform_failed_targets == {("snow", 3, (100, 100))}

    def test_giveup_memory_lifts_when_hero_changes(self):
        """被记的英雄自身变化（合掉/升星/换英雄）：记录随之清除，
        新出现的可变身目标重新给机会。"""
        task = self._ready_task()
        self._enter_poll(task)
        task._transform_poll_deadline = 0.0
        task._transform_reclicks = task._run_config.transform_button_reclick_max
        board = make_board([make_hero("snow", star=3, pos=(100, 100))])
        decision = task.decide_action(
            Observation(frame_id=1, board=board), make_window_ctx()
        )
        assert decision is not None
        assert task._transform_failed_targets == {("snow", 3, (100, 100))}
        # snow3 离场（如被合掉），angel3 入场：记录被棋盘识别清掉，可再触发
        board2 = make_board([make_hero("angel", star=3, pos=(300, 300))])
        decision = task.decide_action(
            Observation(frame_id=2, board=board2), make_window_ctx()
        )
        assert decision is not None
        assert decision[0].tag == "transform_click_hero"
        assert decision[0].target == (300, 300)
        assert task._transform_failed_targets == set()

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
        assert task._transform_failed_targets == {("snow", 3, (100, 100))}

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
            "transform_click_hero",  # 放弃记忆按英雄记：snow3 不再触发
        }
        assert task._transform_stage is None
        assert task._transform_failed_targets == {("snow", 3, (100, 100))}

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
        task._transform_failed_targets = {("snow", 3, (100, 100))}
        task.on_action_verified(Action(kind="wait", tag="ready_match"), State.ENTER_MATCH)
        assert task._transform_used_count == 0
        assert task._transform_stage is None
        assert task._transform_failed_targets == set()


# ---------------------------------------------------------------------------
# 邀请者「准备完成」：开局召唤完成后直接点击（2026-09-27 用户定案；
# 2026-09-26 定案的「凑齐猴子+天使+雪姬再点」已废弃）
# ---------------------------------------------------------------------------


def _monkey_invite_build_task(minimum_summons: int = 1):
    """邀请者猴子 BUILD 任务：强制召唤已完成，带准备完成 hotspot。"""
    from wlxq_bot.config import Hotspot

    profile = MainCProfile(
        display_name="猴子",
        hero_template_dir="heroes/monkey",
        required_lineup_heroes=["angel", "snow"],
        lineup_variant_groups={"angel": ["angel_phantom"]},
        hero_keep_counts={"angel": 1, "snow": 1, "death_knight": 0},
        transform_sacrifice_order=["death_knight", "snow"],
        target_star_override=1,
    )
    spots = {
        "add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96),
        "mouse_park": Hotspot(x_ratio=0.5, y_ratio=0.06),
        "opening_ready_done": Hotspot(x_ratio=0.68, y_ratio=0.30),
    }
    task = make_task(
        main_c="monkey",
        minimum_summons=minimum_summons,
        main_c_profile=profile,
        hotspots=spots,
        batch_enabled=True,
        transform_button_locator={"search_dx_ratio": 0.11},
    )
    task._invite_mode = True
    task.ctx.current_state = State.BUILD_MAIN_C
    task._summon_count = minimum_summons
    return task


class TestInviteReadyDone:
    def test_lineup_incomplete_clicks_ready_done(self):
        """阵容未齐：召唤完成即点准备完成，不等凑阵容（金币不足也不影响）。"""
        task = _monkey_invite_build_task()
        board = make_board([make_hero("monkey", 1), make_hero("angel", 1)])
        obs = Observation(
            frame_id=1,
            board=board,
            raw_data={"gold": {"current": 0, "summon_cost": 80, "summon_red": True}},
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert action.tag == "invite_ready_done"
        assert to_state == State.BUILD_MAIN_C
        # 回归防护：该按钮无验证规则，next_frame 会必然失败空转 5 帧
        # 全量观察（2026-09-27 定案固定坐标点击无动作后验证）
        assert action.verification == "immediate"

    def test_lineup_ready_clicks_ready_done(self):
        """阵容凑齐（猴子+天使+雪姬）同样直接点准备完成。"""
        task = _monkey_invite_build_task()
        board = make_board(
            [make_hero("monkey", 1), make_hero("angel", 1), make_hero("snow", 1)]
        )
        decision = task.decide_action(
            Observation(frame_id=1, board=board), make_window_ctx()
        )
        assert decision is not None
        action, to_state = decision
        assert action.tag == "invite_ready_done"
        assert to_state == State.BUILD_MAIN_C

    def test_board_full_clicks_ready_done(self):
        """满盘未凑齐：同样直接点准备完成，不再先合成/变身腾格子。"""
        task = _monkey_invite_build_task()
        heroes = [
            make_hero("monkey", 1, (100, 200)),
            make_hero("angel", 1, (200, 200)),
            make_hero("monkey", 2, (300, 200)),
            make_hero("monkey", 2, (400, 200)),
            make_hero("angel", 2, (500, 200)),
            make_hero("angel", 2, (600, 200)),
            make_hero("death_knight", 2, (700, 200)),
            make_hero("death_knight", 2, (800, 200)),
            make_hero("death_knight", 1, (900, 200)),
            make_hero("death_knight", 1, (1000, 200)),
            make_hero("death_knight", 3, (1100, 200)),
            make_hero("death_knight", 3, (1200, 200)),
        ]
        board = make_board(heroes)  # 满盘，无雪姬
        decision = task.decide_action(
            Observation(frame_id=1, board=board), make_window_ctx()
        )
        assert decision is not None
        action, _ = decision
        assert action.tag == "invite_ready_done"

    def test_round_reset_clears_invite_ready_clicked(self):
        """每局重置必须清理准备完成标记：否则第 2 局起永不点准备完成。"""
        task = _monkey_invite_build_task()
        task._invite_ready_clicked = True
        task._reset_for_home_reentry()
        assert task._invite_ready_clicked is False
        task._invite_ready_clicked = True
        task._reset_for_next_round()
        assert task._invite_ready_clicked is False

    def test_return_to_invite_click_resets_round_state(self):
        """回归（2026-09-27 实机）：邀请者结算直通 INVITE_CLICK 时局内状态
        也必须清零。此前只在回 FIND_COOP 时重置，_summon_count 带入第 2 局
        后开局强制召唤被跳过，空棋盘一路干等到队友带过 19 层。"""
        task = _monkey_invite_build_task()
        task.ctx.round_count = 0
        task._summon_count = 14
        task._invite_ready_clicked = True
        task._empty_board_count = 16
        task._batch_summon_remaining = 2
        task._failed_merge_pairs = {((1, 1), (2, 2))}

        match = MatchResult("buttons/fan_hui.png", (500, 1000), 0.9)
        before = Observation(
            frame_id=1,
            raw_data={"return_button_visible": True, "return_button_match": match},
        )
        task.ctx.current_state = State.CHECK_ROUND_LIMIT
        decision = task.decide_action(before, make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert to_state == State.INVITE_CLICK
        assert task.verify_action(action, before, Observation(frame_id=2))
        task.on_action_verified(action, to_state)

        assert task.ctx.round_count == 1
        assert task._summon_count == 0
        assert task._invite_ready_clicked is False
        assert task._empty_board_count == 0
        assert task._batch_summon_remaining == 0
        assert task._failed_merge_pairs == set()



class TestSummonRefusalSuppression:
    """召唤拒判（2026-10-06 用户定案）：点击后召唤费用不变 = 被游戏拒绝。

    2026-10-06 实机事故：棋盘满时识别抖动造出假空格，bot 102 连点召唤全部
    被游戏拒绝（费用全程钉死 215、金币一分未扣）无限空转。费用是只有召唤
    成功才会变化的「安静信号」（金币后台缓存每 ~2s 读按钮区），连续 N 次
    不变即判定棋盘已满，停用召唤出口；依据游戏拒绝行为而非识别占用，两个
    阵容（邀请者/强袭）共用同一条通路。
    """

    GOLD = {"current": 10000, "summon_cost": 215, "summon_red": False, "frame_id": 2}

    @staticmethod
    def _task(batch_enabled: bool = False) -> CoopTask:
        task = make_task(minimum_summons=2, batch_enabled=batch_enabled)
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = 2
        return task

    @staticmethod
    def _obs(board: BoardSnapshot, gold: dict) -> Observation:
        return Observation(frame_id=1, board=board, raw_data={"gold": dict(gold)})

    @staticmethod
    def _full_board_no_pairs() -> BoardSnapshot:
        """12 格全占满且无任何同型同星对（全异类 1 星）。"""
        types = [
            "assault", "snow", "monkey", "angel", "death_knight", "fox",
            "boar", "cat", "dog", "wolf", "bear", "frog",
        ]
        return make_board(
            [make_hero(name, star=1, pos=(100 * index, 100)) for index, name in enumerate(types)]
        )

    def test_three_unchanged_costs_suppress_summoning(self):
        """回归（2026-10-06 实机）：满盘召唤连被拒 3 次（费用不变）后停用召唤出口。"""
        task = self._task()
        board = self._full_board_no_pairs()
        for expected_count in (0, 1, 2):
            decision = task.decide_action(self._obs(board, self.GOLD), make_window_ctx())
            assert decision is not None
            assert decision[0].tag == "summon_hero"
            assert task._summon_refusal_count == expected_count
            assert task._summon_refusal_armed
            assert task._summon_cost_baseline == 215
        # 第 4 次入口：计数到 3 → 判满。识别仍报满盘、无合法对 → 判满干等
        decision = task.decide_action(self._obs(board, self.GOLD), make_window_ctx())
        assert decision is not None
        action, _ = decision
        assert task._summon_refusal_suppressed
        assert action.tag == "frozen_board_wait"
        assert "召唤被拒" in action.reason

    def test_cost_change_resets_count(self):
        """费用变化 = 有召唤落位：计数清零。"""
        task = self._task()
        task._arm_summon_cost_check(dict(self.GOLD), 1)
        task._evaluate_summon_refusal(dict(self.GOLD))
        task._arm_summon_cost_check(dict(self.GOLD), 1)
        task._evaluate_summon_refusal(dict(self.GOLD))
        assert task._summon_refusal_count == 2
        task._arm_summon_cost_check(dict(self.GOLD), 1)
        task._evaluate_summon_refusal(
            {"current": 10000, "summon_cost": 230, "summon_red": False, "frame_id": 2}
        )
        assert task._summon_refusal_count == 0
        assert not task._summon_refusal_suppressed

    def test_stale_frame_read_skips_without_counting(self):
        """回归（2026-10-14 实机三连误判）：读数来自点击派发帧或更早——召唤
        动画未刷新按钮费用的旧值「未变化」不构成拒绝证据，不计数。

        事故链：三批召唤全部落位（棋盘多出 2 只、开新格、费用 190→215），
        但评估消费了点击前的缓存旧值 → 连计 3 次假判满 → 召唤出口锁死 →
        干等保守退出。帧号新鲜度门让这类读数按缺失处理。
        """
        task = self._task()
        task._arm_summon_cost_check(dict(self.GOLD), 5)
        task._evaluate_summon_refusal(dict(self.GOLD, frame_id=5))
        assert task._summon_refusal_count == 0
        assert not task._summon_refusal_armed
        assert not task._summon_refusal_suppressed

    def test_cost_change_lifts_suppression(self):
        """判满后费用变化（腾出空位后召唤落位）：解除抑制。"""
        task = self._task()
        task._summon_refusal_suppressed = True
        task._summon_refusal_count = 3
        task._arm_summon_cost_check(dict(self.GOLD), 1)
        task._evaluate_summon_refusal(
            {"current": 10000, "summon_cost": 230, "summon_red": False, "frame_id": 2}
        )
        assert not task._summon_refusal_suppressed
        assert task._summon_refusal_count == 0

    def test_merge_success_lifts_suppression(self):
        """合成成功腾出空位（即使费用还没变）：解除抑制，恢复试探召唤。"""
        task = self._task()
        task._summon_refusal_suppressed = True
        task._summon_refusal_count = 3
        task._summon_full_wait_streak = 2
        task.on_action_verified(
            Action(kind="drag", tag="merge_heroes", target=(1, 1), end=(2, 2)),
            State.BUILD_MAIN_C,
        )
        assert not task._summon_refusal_suppressed
        assert task._summon_refusal_count == 0
        assert task._summon_full_wait_streak == 0

    def test_missing_cost_read_skips_without_counting(self):
        """读数缺失（OCR 空读/快照超龄）：跳过不计数也不复位。"""
        task = self._task()
        task._arm_summon_cost_check(dict(self.GOLD), 1)
        task._evaluate_summon_refusal({"current": 10000, "summon_red": False})
        assert task._summon_refusal_count == 0
        assert not task._summon_refusal_armed
        # 未武装时评估是空操作
        task._evaluate_summon_refusal(dict(self.GOLD))
        assert task._summon_refusal_count == 0

    def test_cap_cost_freezes_counter(self):
        """费用读数达封顶值：计数冻结（封顶后成功召唤费用也不变，信号无定义）。"""
        task = self._task()
        gold_cap = {
            "current": 99999, "summon_cost": 660, "summon_red": False, "frame_id": 2,
        }
        task._arm_summon_cost_check(dict(gold_cap), 1)
        task._evaluate_summon_refusal(dict(gold_cap))
        assert task._summon_refusal_count == 0
        assert not task._summon_refusal_suppressed
        # 已累计的计数在封顶段保持，不增不清
        task._summon_refusal_count = 2
        task._arm_summon_cost_check(dict(gold_cap), 1)
        task._evaluate_summon_refusal(dict(gold_cap))
        assert task._summon_refusal_count == 2
        assert not task._summon_refusal_suppressed

    def test_suppressed_board_skips_batch_summon(self):
        """判满后批次召唤停用：识别报有空格（假空格）也不再规划批次召唤。"""
        board = make_board(
            [
                make_hero("snow", star=1, pos=(100, 100)),
                make_hero("snow", star=1, pos=(200, 200)),
            ]
        )
        baseline = self._task(batch_enabled=True)
        decision = baseline.decide_action(self._obs(board, self.GOLD), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "batch_summon"
        suppressed = self._task(batch_enabled=True)
        suppressed._summon_refusal_suppressed = True
        decision = suppressed.decide_action(self._obs(board, self.GOLD), make_window_ctx())
        assert decision is not None
        assert decision[0].tag != "batch_summon"
        # 有合法对：落入合成（费用判据不改变合成优先级）
        assert decision[0].tag in {"merge_heroes", "batch_merge"}

    def test_full_board_wait_streak_ends_in_conservative_stop(self):
        """判满干等连击计满：保守停止（识别抖动会骗过棋盘签名看门狗，按动作计数兜底）。"""
        task = self._task()
        task._run_config.summon_full_wait_limit = 3
        task._summon_refusal_suppressed = True
        board = self._full_board_no_pairs()
        for _ in range(2):
            decision = task.decide_action(self._obs(board, self.GOLD), make_window_ctx())
            assert decision is not None
            assert decision[0].tag == "frozen_board_wait"
        assert task._summon_full_wait_streak == 2
        assert task.decide_action(self._obs(board, self.GOLD), make_window_ctx()) is None

    def test_failed_summon_action_disarms(self):
        """召唤点击未发出（重试耗尽）：解除武装，不参与费用不变计数。"""
        task = self._task()
        task._arm_summon_cost_check(dict(self.GOLD), 1)
        task.on_action_failed(Action(kind="click", tag="summon_hero", target=(1, 1)))
        assert not task._summon_refusal_armed

    def test_suppressed_topup_with_space_merges_instead_of_wait(self):
        """回归（2026-10-14 实机锁死）：判满期 + 回补阶段 + 有空位时放行合成。

        「有空位先召唤」的前提随召唤出口停用失效；仍拦合成会让唯一的解除
        路径（合成成功自动解除）也走不通——10-14 事故里 fox1★ 对被
        topup_fill_first 拦了 18 帧，干等计满保守退出。放行后合成即自愈。
        """
        board = make_board(
            [
                make_hero("snow", star=1, pos=(100, 100)),
                make_hero("snow", star=1, pos=(200, 200)),
            ]
        )
        task = self._task(batch_enabled=True)
        task._topup_active = True
        task._summon_refusal_suppressed = True
        decision = task.decide_action(self._obs(board, self.GOLD), make_window_ctx())
        assert decision is not None
        assert decision[0].tag in {"merge_heroes", "batch_merge"}
        # 对照：未判满时同局同盘仍走「有空位先召唤」
        baseline = self._task(batch_enabled=True)
        baseline._topup_active = True
        decision = baseline.decide_action(self._obs(board, self.GOLD), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "batch_summon"

    def test_suppressed_with_space_releases_main_c_pair_gate(self):
        """回归（2026-10-25 实机×2）：判满期 + 有空位 + 仅主C对 → 放行合成。

        保卫宠物两局都在 6/8、7/8 的假满盘上锁死：召唤出口被费用判据停用后，
        策略层「有空位保主C数量」仍拦下 assault1★ 对，合成成功这条唯一
        解锁路径够不着，干等计满保守退出。放行后判满帧直接拖主C对自愈。
        """
        board = make_board(
            [
                make_hero("assault", star=1, pos=(100, 100)),
                make_hero("assault", star=1, pos=(200, 100)),
                make_hero("fox", star=1, pos=(100, 200)),
            ]
        )
        task = self._task()
        task._topup_active = True
        task._summon_refusal_suppressed = True
        decision = task.decide_action(self._obs(board, self.GOLD), make_window_ctx())
        assert decision is not None
        assert decision[0].tag in {"merge_heroes", "batch_merge"}
        # 对照：未判满时同局同盘仍走「有空位先召唤」
        baseline = self._task(batch_enabled=True)
        baseline._topup_active = True
        decision = baseline.decide_action(self._obs(board, self.GOLD), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "batch_summon"

    def test_suppressed_with_space_releases_wildcard_pair_gate(self):
        """回归（2026-10-25 实机）：判满期 + 有空位 + 幻天使通配对让位可合。

        假满盘上仅剩的合法对常是幻天使对（strategy:665 日志逐帧刷屏），
        与金币不足例外共用同一窄例外语义：召唤不可用时第 5 层空位保留失效。
        """
        board = make_board(
            [
                make_hero("angel_phantom", star=1, pos=(100, 100)),
                make_hero("angel_phantom", star=1, pos=(200, 100)),
                make_hero("fox", star=1, pos=(100, 200)),
            ]
        )
        task = self._task()
        task._topup_active = True
        task._summon_refusal_suppressed = True
        decision = task.decide_action(self._obs(board, self.GOLD), make_window_ctx())
        assert decision is not None
        assert decision[0].tag in {"merge_heroes", "batch_merge"}
        baseline = self._task(batch_enabled=True)
        baseline._topup_active = True
        decision = baseline.decide_action(self._obs(board, self.GOLD), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "batch_summon"

    def test_round_reset_clears_refusal_state(self):
        """局间重置（结算/被踢共用）：判满状态与计数清零，不带入下一局。"""
        task = self._task()
        task._summon_refusal_suppressed = True
        task._summon_refusal_count = 2
        task._summon_refusal_armed = True
        task._summon_cost_baseline = 215
        task._summon_full_wait_streak = 4
        task._reset_for_home_reentry()
        assert not task._summon_refusal_suppressed
        assert task._summon_refusal_count == 0
        assert not task._summon_refusal_armed
        assert task._summon_cost_baseline is None
        assert task._summon_full_wait_streak == 0


# ---------------------------------------------------------------------------
# 坐骑定时点击（2026-11-15 用户定案，default.yaml run.mount_ride）
# ---------------------------------------------------------------------------


class _MountFakeClock:
    """坐骑调度假时钟：手动推进，测试锚定与爆发节奏。"""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_mount_task(**mount_kwargs) -> tuple[CoopTask, _MountFakeClock]:
    """坐骑定时点击任务：默认参数开启，假时钟可控，mount_ride hotspot 已配置。"""
    ctx = TaskContext(main_c="assault", max_rounds=10)
    run_cfg = RunConfig(
        max_rounds=10,
        minimum_summon_count_before_skills=3,
        initial_board_capacity=7,
        target_star_level=2,
        mount_ride=MountRideConfig(enabled=True, **mount_kwargs),
    )
    task = CoopTask(
        ctx,
        run_cfg,
        CoopRole.HELPER,
        {"mount_ride": Hotspot(x_ratio=0.9137, y_ratio=0.5483)},
    )
    clock = _MountFakeClock()
    task._clock = clock
    task.ctx.current_state = State.BUILD_MAIN_C
    return task, clock


def mount_absent(frame_id: int) -> Observation:
    """无 zhun_bei_wan_cheng 旗标的观察 = 「准备完成」不存在。"""
    return Observation(frame_id=frame_id)


def mount_present(frame_id: int) -> Observation:
    return Observation(
        frame_id=frame_id, raw_data={"zhun_bei_wan_cheng_visible": True}
    )


def arm_mount_by_frames(task: CoopTask, clock: _MountFakeClock) -> None:
    """走真实计数路径锚定：连续 4 帧缺席（默认 absent_frames=4）。"""
    for i in range(4):
        assert (
            task._mount_ride_consult(mount_absent(i + 1), None, State.BUILD_MAIN_C)
            is None
        )
    assert task._mount_ready_at is not None


def preset_mount_due(task: CoopTask, clock: _MountFakeClock, ago: float = 1.0) -> None:
    """直接置为「已锚定且已到期」，跳过计数路径（爆发窗口测试前置）。"""
    task._mount_armed = True
    task._mount_ready_at = clock.now - 30.0
    task._mount_next_click_at = clock.now - ago


class TestMountRide:
    def test_disabled_by_default_no_state_change(self):
        """默认配置（enabled=False）下调度器完全不参与，门不开。"""
        task = make_task()
        task.ctx.current_state = State.BUILD_MAIN_C
        for i in range(10):
            assert (
                task._mount_ride_consult(mount_absent(i + 1), None, State.BUILD_MAIN_C)
                is None
            )
        assert task._mount_armed is False

    def test_arms_only_after_build_state(self):
        """开局技能页阶段不计数；进入培养（选完开局技能）才开门。"""
        task, _clock = make_mount_task()
        for i in range(10):
            assert (
                task._mount_ride_consult(
                    mount_absent(i + 1), None, State.SELECT_OPENING_SKILLS
                )
                is None
            )
        assert task._mount_armed is False
        assert task._mount_ride_consult(mount_absent(11), None, State.BUILD_MAIN_C) is None
        assert task._mount_armed is True

    def test_absent_streak_resets_on_hit(self):
        """连续缺席被命中打断后重数（「连续」的本义）。"""
        task, clock = make_mount_task()
        for i in range(3):
            task._mount_ride_consult(mount_absent(i + 1), None, State.BUILD_MAIN_C)
        assert task._mount_ready_at is None
        assert task._mount_absent_streak == 3
        task._mount_ride_consult(mount_present(4), None, State.BUILD_MAIN_C)
        assert task._mount_absent_streak == 0
        assert task._mount_ever_seen is True
        for i in range(3):
            task._mount_ride_consult(mount_absent(5 + i), None, State.BUILD_MAIN_C)
        assert task._mount_ready_at is None
        task._mount_ride_consult(mount_absent(8), None, State.BUILD_MAIN_C)
        assert task._mount_ready_at == clock.now

    def test_degraded_anchor_without_template_hit(self):
        """模板全程不命中：4 帧后照样锚定（退化=选完技能即锚定），标记漂移。"""
        task, clock = make_mount_task()
        arm_mount_by_frames(task, clock)
        assert task._mount_ever_seen is False

    def test_first_click_after_start_delay(self):
        """锚定后 30 秒内不打扰正常流程（返回 None），到期即发首击。"""
        task, clock = make_mount_task()
        arm_mount_by_frames(task, clock)
        clock.advance(29.9)
        assert task._mount_ride_consult(mount_absent(5), None, State.BUILD_MAIN_C) is None
        assert task._mount_clicks_done == 0
        clock.advance(0.2)
        result = task._mount_ride_consult(mount_absent(6), make_window_ctx(), State.BUILD_MAIN_C)
        assert result is not None
        action, to_state = result
        assert action.kind == "click"
        assert action.tag == "mount_ride_click"
        assert to_state == State.BUILD_MAIN_C
        assert task._mount_clicks_done == 1

    def test_burst_interval_lower_bound_and_budget(self):
        """爆发窗口：3 秒下限、间隔未到时零打扰（不返回任何动作）、4 击后沉默。"""
        task, clock = make_mount_task()
        arm_mount_by_frames(task, clock)
        window_ctx = make_window_ctx()
        expected_point = task._hotspot_point("mount_ride", window_ctx)
        clock.advance(30.0)
        for n in range(1, 5):
            result = task._mount_ride_consult(
                mount_absent(10 + n), window_ctx, State.BUILD_MAIN_C
            )
            assert result is not None
            action, _ = result
            assert action.kind == "click"
            assert action.tag == "mount_ride_click"
            assert action.target == expected_point
            assert task._mount_clicks_done == n
            if n < 4:
                # 间隔未到：正常流程照跑（调度器返回 None，绝不返回等待动作）
                clock.advance(1.0)
                assert (
                    task._mount_ride_consult(
                        mount_absent(20 + n), window_ctx, State.BUILD_MAIN_C
                    )
                    is None
                )
                clock.advance(2.0)
        # 点满后永久沉默
        clock.advance(5.0)
        assert task._mount_ride_consult(mount_absent(30), window_ctx, State.BUILD_MAIN_C) is None

    def test_overlay_frames_skipped_without_reset(self):
        """弹窗/技能页/赠送页帧不计入也不打断（2026-11-15 用户定案）：盲帧跳过，命中才重数。"""
        task, clock = make_mount_task()
        for i in range(2):
            task._mount_ride_consult(mount_absent(i + 1), None, State.BUILD_MAIN_C)
        assert task._mount_absent_streak == 2
        # 四种遮挡帧：不 +1、不清零，连续性保留
        for n, extra in enumerate(
            (
                {"tan_chuang_visible": True},
                {"select_skill_button_visible": True},
                {"merge_gift_skill_page_visible": True},
                {"zeng_song_ji_neng_visible": True},
            )
        ):
            obs = Observation(frame_id=99 + n, raw_data=dict(extra))
            task._mount_ride_consult(obs, None, State.BUILD_MAIN_C)
            assert task._mount_absent_streak == 2, extra
        assert task._mount_ready_at is None
        # 对照：模板命中仍然清零重数（重新开始）
        task._mount_ride_consult(mount_present(10), None, State.BUILD_MAIN_C)
        assert task._mount_absent_streak == 0
        # 遮挡关完/命中过后接着数：第 4 个「无遮挡且未命中」帧锚定
        for i in range(3):
            task._mount_ride_consult(mount_absent(20 + i), None, State.BUILD_MAIN_C)
        assert task._mount_ready_at is None
        task._mount_ride_consult(mount_absent(23), None, State.BUILD_MAIN_C)
        assert task._mount_ready_at == clock.now

    def test_overlay_defers_without_consuming_budget(self):
        """技能页/赠送页在场时点击顺延、预算不消耗（防误点卡片）。"""
        task, clock = make_mount_task()
        preset_mount_due(task, clock)
        window_ctx = make_window_ctx()
        overlays = [
            {"select_skill_button_visible": True},
            {"merge_gift_skill_page_visible": True},
            {"zeng_song_ji_neng_visible": True},
        ]
        for i, extra in enumerate(overlays):
            obs = Observation(frame_id=100 + i, raw_data=dict(extra))
            assert task._mount_ride_consult(obs, window_ctx, State.BUILD_MAIN_C) is None
        assert task._mount_clicks_done == 0
        result = task._mount_ride_consult(mount_absent(200), window_ctx, State.BUILD_MAIN_C)
        assert result is not None
        assert result[0].tag == "mount_ride_click"

    def test_popup_preempts_due_mount_click(self):
        """被弹窗打断：击杀奖励弹窗优先关闭，坐骑不发点击。"""
        task, clock = make_mount_task()
        preset_mount_due(task, clock)
        match = MatchResult("buttons/tan_chuang.png", (400, 1100), 0.9)
        obs = Observation(
            frame_id=1,
            raw_data={"tan_chuang_visible": True, "tan_chuang_match": match},
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "close_popup"
        assert task._mount_clicks_done == 0

    def test_decide_returns_mount_click_when_due(self):
        """decide_action 分支插在原状态逻辑之前：到期帧干净即发坐骑点击。"""
        task, clock = make_mount_task()
        preset_mount_due(task, clock)
        decision = task.decide_action(mount_absent(1), make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert action.tag == "mount_ride_click"
        assert to_state == State.BUILD_MAIN_C

    def test_exception_disables_for_round_only(self):
        """调度异常：ERROR 后本局禁用坐骑，该帧起原流程完全接管。"""
        task, clock = make_mount_task()
        preset_mount_due(task, clock)

        def _boom(*_args, **_kwargs):
            raise RuntimeError("hotspot 换算爆炸")

        task._hotspot_point = _boom
        window_ctx = make_window_ctx()
        assert task._mount_ride_consult(mount_absent(1), window_ctx, State.BUILD_MAIN_C) is None
        assert task._mount_disabled_for_round is True
        assert task._mount_ride_consult(mount_absent(2), window_ctx, State.BUILD_MAIN_C) is None

    def test_round_reset_clears_mount_state(self):
        """局间重置（结算/被踢共用路径）：锚点与预算不带入下一局。"""
        task, _clock = make_mount_task()
        task._mount_armed = True
        task._mount_absent_streak = 2
        task._mount_ever_seen = True
        task._mount_ready_at = 123.0
        task._mount_next_click_at = 456.0
        task._mount_clicks_done = 2
        task._mount_disabled_for_round = True
        task._reset_for_home_reentry()
        assert task._mount_armed is False
        assert task._mount_absent_streak == 0
        assert task._mount_ever_seen is False
        assert task._mount_ready_at is None
        assert task._mount_next_click_at is None
        assert task._mount_clicks_done == 0
        assert task._mount_disabled_for_round is False

    def test_missing_hotspot_preserves_budget(self):
        """hotspot 未配置：不发点击不消耗预算，原流程照常（只警告一次）。"""
        task, clock = make_mount_task()
        del task._hotspots["mount_ride"]
        preset_mount_due(task, clock)
        window_ctx = make_window_ctx()
        assert task._mount_ride_consult(mount_absent(1), window_ctx, State.BUILD_MAIN_C) is None
        assert task._mount_clicks_done == 0
        assert task._mount_point_warned is True


class TestHelperReadyClick:
    """帮助者「准备完成」一次性点击（2026-11-19 用户定案，11-20 短暂改为
    「召唤点满即点」后当日回滚到本版）。

    第一阶段培养凑出 2 星主C时，画面无弹窗遮挡且识别到按钮就点匹配中心、
    随即回选技能；弹窗在场延后到干净帧再查（不计入检查）；干净帧无论命中
    与否都算查过，一局只查一次，未命中不重试（游戏 ~90s 自动开始兜底）。
    仅模式档案 helper_ready_click=true 的帮助者生效；邀请者沿用开局召唤
    后的既有点击。
    """

    def _mode_profile(self, enabled: bool) -> ModeProfile:
        return ModeProfile(
            display_name="测试模式",
            mode_chat_hotspot="coop_chat",
            entry={"mode_card_text": "合作", "challenge_text": "开始挑战"},
            match={"ready_button": "ready_button", "leave_team": "coop_leave_team"},
            settlement={"like_chest": True, "return_button": "return_button"},
            helper_ready_click=enabled,
        )

    def _build_task(self, *, enabled: bool = True) -> CoopTask:
        task = make_task(mode_profile=self._mode_profile(enabled))
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        return task

    @staticmethod
    def _ready_board() -> BoardSnapshot:
        """第一阶段达标棋盘：单个 2 星强袭（不看数量）。"""
        return make_board([make_hero("assault", star=2, pos=(100, 100))])

    @staticmethod
    def _button_observation(frame_id: int = 1, board: BoardSnapshot | None = None) -> Observation:
        return Observation(
            frame_id=frame_id,
            board=board if board is not None else TestHelperReadyClick._ready_board(),
            raw_data={
                "zhun_bei_wan_cheng_visible": True,
                "zhun_bei_wan_cheng_match": MatchResult(
                    "buttons/zhun_bei_wan_cheng.png", (630, 500), 0.9
                ),
            },
        )

    def test_clicks_button_then_returns_to_skill_selection(self):
        """干净帧命中按钮：点匹配中心、状态迁选技能、标记置位。"""
        task = self._build_task()
        decision = task.decide_action(self._button_observation(), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "click"
        assert decision[0].tag == "helper_ready_done"
        assert decision[0].target == (630, 500)
        assert decision[1] == State.SELECT_MAIN_C_SKILLS
        assert task._helper_ready_check_done is True

    def test_button_absent_falls_through_to_plain_exit(self):
        """干净帧未命中：落回原出口（wait + 选技能），也算查过。"""
        task = self._build_task()
        board = self._ready_board()
        decision = task.decide_action(
            Observation(frame_id=1, board=board), make_window_ctx()
        )
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS
        assert task._helper_ready_check_done is True

    def test_popup_defers_check_to_clean_frame(self):
        """弹窗遮挡帧：等待留在培养态、不计入检查；弹窗关后再查命中。"""
        task = self._build_task()
        decision = task.decide_action(
            Observation(
                frame_id=1,
                board=self._ready_board(),
                raw_data={"tan_chuang_visible": True},
            ),
            make_window_ctx(),
        )
        assert decision is not None
        assert decision[0].kind == "wait"
        assert decision[0].tag == "helper_ready_popup_wait"
        assert decision[1] == State.BUILD_MAIN_C
        assert task._helper_ready_check_done is False
        # 弹窗关闭后的干净帧：正常命中并点击
        decision = task.decide_action(self._button_observation(frame_id=2), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "helper_ready_done"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS

    def test_checked_once_per_round_and_reset_between_rounds(self):
        """已查过的局不再点（含第二阶段出口复用同分支）；按局重置后恢复。"""
        task = self._build_task()
        first = task.decide_action(self._button_observation(), make_window_ctx())
        assert first is not None and first[0].tag == "helper_ready_done"
        # 同局后续帧（如回补阶段再达标）不再点
        task.ctx.current_state = State.BUILD_MAIN_C
        second = task.decide_action(self._button_observation(frame_id=2), make_window_ctx())
        assert second is not None
        assert second[0].tag == "main_c_ready"
        # 按局重置（结算/被踢共用路径）后恢复可查
        task._reset_for_home_reentry()
        assert task._helper_ready_check_done is False

    def test_disabled_by_mode_profile(self):
        """档案未开启（如 pet 模式）：完全不参与，标记不置位。"""
        task = self._build_task(enabled=False)
        decision = task.decide_action(self._button_observation(), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert task._helper_ready_check_done is False

    def test_invite_mode_excluded(self):
        """邀请者模式不参与：沿用开局召唤后的既有点击链路。"""
        task = make_task(
            mode_profile=self._mode_profile(True),
            hotspots={"opening_ready_done": Hotspot(x_ratio=0.68, y_ratio=0.30)},
        )
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        task._invite_mode = True
        decision = task.decide_action(self._button_observation(), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "invite_ready_done"
        assert task._helper_ready_check_done is False
