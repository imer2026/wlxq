"""合作任务状态机与局内决策。"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import replace
from enum import Enum
from typing import Any, Literal

from wlxq_bot.config import (
    Hotspot,
    InviteConfig,
    MainCProfile,
    ModeProfile,
    RoiConfig,
    RunConfig,
)
from wlxq_bot.models import (
    Action,
    BoardHero,
    BoardSnapshot,
    CoopRole,
    MatchResult,
    MergeCandidate,
    Observation,
    SkillCandidate,
    State,
    Transition,
    WindowContext,
)
from wlxq_bot.perception.locator import hotspot_to_client_point, roi_column_centers
from wlxq_bot.perception.pet_difficulty import match_targets
from wlxq_bot.tasks.base import Task, TaskContext
from wlxq_bot.tasks.strategy import CultivationStrategy, build_strategy
from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)


def _hero_location(hero: BoardHero) -> str:
    """日志用英雄位置描述：有格名用格名（如 1A），否则退回像素坐标。"""
    return hero.cell_name or str(tuple(hero.position))


def _comparable_skill_options(
    options: list[dict[str, object]] | None,
) -> list[tuple[int, str, int, bool]] | None:
    """提取用于开局卡面比对的稳定字段，剔除随帧变化的噪声字段。

    ``skill_card_options`` 每帧由 OCR 重新产出：``frame_id`` 必然不同，
    ``position``/``page`` 由固定几何与页面推出、帧间恒定。比对只保留
    （列号, 技能名, 档位, 是否已知）；缺失或为空视为 None（交给重试）。
    """
    if not options:
        return None
    return [
        (
            option.get("column", index),
            option["name"],
            option["tier"],
            option.get("known", False),
        )
        for index, option in enumerate(options)
    ]


_MATCH_STATES: frozenset[State] = frozenset(
    {
        State.ENTER_MATCH,
        State.SELECT_OPENING_SKILLS,
        State.BUILD_MAIN_C,
        State.SELECT_MAIN_C_SKILLS,
    }
)

# 邀请者模式组队状态（对局未开始，无弹窗/棋盘/金币识别开销）
_INVITE_TEAM_STATES: frozenset[State] = frozenset(
    {
        State.INVITE_CLICK,
        State.INVITE_WAITING,
        State.INVITE_START,
        State.INVITE_COOLDOWN,
    }
)

_SETTLEMENT_STATES: frozenset[State] = frozenset(
    {
        State.HANDLE_RESULT,
        State.CLAIM_REWARD,
        State.CHECK_ROUND_LIMIT,
    }
)
# 空棋盘识别的告警节奏：每累计这么多次输出一次告警。不是停止条件——对局内
# 棋盘为空只可能是页面/弹窗/结算过渡遮挡，持续等待比退出安全（2026-08-21）
_EMPTY_BOARD_RETRY = 10
_NO_PROGRESS_LIMIT = 8
# 领地入口确认步骤：连续这么多帧未见目标标识就重点入口按钮（0.3s 轮询
# 间隔下约 3 秒）；重点次数上限由 mode_select_recognition.entry_max_reclicks 配置
_ENTRY_RECLICK_MISSES = 10
# 挑战页每周弹窗的消失宽限：弹窗可能晚于页面到达弹出，连续这么多帧读不到
# 按钮文字才放行进招募（0.3s 轮询间隔下约 1 秒）；弹窗只在本日首次进入出现，
# 正常情况（已关/无弹窗）本步骤仅多等这几帧
_WEEKLY_BUFF_GRACE_MISSES = 2
# 聊天点击生效确认：页签行（mode_tab OCR）连续这么多帧读到本模式页签才算
# 已进入招募（吸收 OCR 单帧漏读与页面过渡残影）；连续这么多帧读不到说明
# 点击未生效（被道具卡等弹窗吃掉），重点聊天按钮。注意不能用「挑战页
# 锚点消失」判定——招募频道页底部同样有「开始挑战」按钮，锚点两页皆可读
# （2026-09-17 实机：9 次重点全部点空直到超时）
_CHAT_OPEN_ADVANCE_MISSES = 2
_CHAT_OPEN_RECLICK_MISSES = 4
# 主C系技能的最高优先级档位（compute_skill_tiers：第 8 档=主C未被显式档覆盖的
# 技能，主C技能永不落第 11 档）。三张卡最低档 > 该值 = 本组没有任何主C系技能，
# 是「刷新」分支的触发条件之一
_MAIN_C_TIER_CEILING = 8


class _RecruitStep(Enum):
    """`FIND_COOP` 内部的固定招募业务步骤。"""

    # 领地入口（2026-09-09 新流程，取代首页→聊天→招募）：读底部导航领地格
    # 文字，「领地」=点击并等「返回」；「返回」=已在领地页，跳过等待直进
    # 模式选择弹窗步骤
    OPEN_TERRITORY = "open_territory"
    CONFIRM_TERRITORY_OPEN = "confirm_territory_open"
    OPEN_MODE_SELECT = "open_mode_select"
    CONFIRM_MODE_SELECT_OPEN = "confirm_mode_select_open"
    SELECT_MODE_CARD = "select_mode_card"
    CONFIRM_CHALLENGE_PAGE = "confirm_challenge_page"
    # 挑战页每周弹窗（保卫宠物=本周Buff，每天首次进入弹出）：读到按钮文字
    # 即点文字框中心关闭；模式档案未配置 weekly_buff_text 时跳过本步骤
    CLOSE_WEEKLY_BUFF = "close_weekly_buff"
    OPEN_CHALLENGE_CHAT = "open_challenge_chat"
    # 聊天点击生效确认：挑战页锚点（challenge_text）消失才算进入招募；
    # 仍读到锚点说明点击被道具卡等弹窗吃掉，重点聊天（预算
    # mode_select_recognition.chat_open_max_reclicks）
    CONFIRM_CHAT_OPEN = "confirm_chat_open"
    SELECT_MODE_TAB = "select_mode_tab"
    OPEN_DIFFICULTY_DIALOG = "open_difficulty_dialog"
    CONFIRM_DIFFICULTY_OPEN = "confirm_difficulty_open"
    SELECT_DIFFICULTIES = "select_difficulties"
    CLOSE_DIFFICULTY_DIALOG = "close_difficulty_dialog"
    OPEN_COOP_CHAT = "open_coop_chat"
    OPEN_REFRESH_DIFFICULTY_DIALOG = "open_refresh_difficulty_dialog"
    CLOSE_REFRESH_DIFFICULTY_DIALOG = "close_refresh_difficulty_dialog"
    JOIN_COOP = "join_coop"
    # 退队后画面可能在首页也可能在合作页面：先探测再选对应入口
    DETECT_ENTRY_PAGE = "detect_entry_page"


class CoopTask(Task):
    """寻找合作、准备、培养主 C、选技能和结算的一局循环。"""

    def __init__(
        self,
        ctx: TaskContext,
        run_config: RunConfig,
        role: CoopRole,
        hotspots: dict[str, Hotspot],
        coop_difficulties: list[int] | None = None,
        *,
        skill_candidate_roi: RoiConfig | None = None,
        difficulty_selection_only: bool = False,
        skip_difficulty_selection: bool = False,
        mode_profile: ModeProfile | None = None,
        pet_difficulty_targets: list[str] | None = None,
        pet_difficulty_min_confidence: float = 0.5,
        mode_tab_miss_limit: int = 8,
        entry_wait_timeout_seconds: float = 60.0,
        entry_max_reclicks: int = 3,
        chat_open_max_reclicks: int = 10,
        main_c_profile: MainCProfile | None = None,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
        transform_button_locator: dict[str, Any] | None = None,
        invite_mode: bool = False,
        invite_config: "InviteConfig | None" = None,
    ) -> None:
        super().__init__(ctx)
        self._run_config = run_config
        self._role = role
        self._hotspots = hotspots
        # 邀请者模式（spec 2026-09-21 §6）：发起合作等待帮助者，走 INVITE_* 状态
        self._invite_mode = invite_mode
        self._invite = invite_config or InviteConfig()
        self._invite_step = 0  # 5 步点击链游标
        self._invite_wait_since: float | None = None  # 当前等待段起点（clock 秒）
        self._invite_kick_count = 0  # 连续请离计数
        self._invite_reinvite_count = 0  # 连续空等重发计数
        self._invite_cooldown_until: float | None = None
        # 开局召唤次数档案覆盖（opening_summons>0 时替换 run 级配置）
        self._main_c_profile = main_c_profile
        self._profile_opening_summons = (
            main_c_profile.opening_summons if main_c_profile is not None else 0
        )
        # 邀请者「准备完成」已点标记（开局召唤完成后点一次，动作后验证）
        self._invite_ready_clicked = False
        self._skill_candidate_roi = skill_candidate_roi
        # 主C培养策略（合成/门禁/数量回补）：纯决策对象，状态机只负责调度；
        # 模式档案经 Runner 的 resolve_main_c_profile 应用覆盖后传入 profile
        self._strategy = build_strategy(mode_profile, ctx.main_c, run_config, main_c_profile)
        # 通配合并英雄（幻天使）：来自 run.wildcard_merge_heroes，传给候选生成
        self._wildcard_heroes = frozenset(run_config.wildcard_merge_heroes)
        self._remaining_difficulties = set(coop_difficulties or self._run_config.coop_difficulties)
        self._difficulty_selection_only = difficulty_selection_only
        # 游戏本次会话已手动选过难度：首局招募跳过难度弹窗三步直接抢合作
        self._skip_difficulty_selection = skip_difficulty_selection
        # 难度选择结果追踪，用于结尾汇总「选中 / 未识别跳过」
        self._selected_difficulties: set[int] = set()
        self._skipped_difficulties: set[int] = set()
        self._difficulty_summary_logged = False
        # 页签 OCR 连续识别不到的帧计数（达 mode_tab_miss_limit 报错停止）
        self._mode_tab_misses = 0
        self._mode_tab_miss_limit = max(1, mode_tab_miss_limit)
        # 领地入口各等待步骤（领地变返回/弹窗出现/挑战页出现）共用的秒级
        # 上限；超时打 ERROR 保守停止，不盲点不无限重试
        self._entry_wait_timeout_seconds = max(1.0, entry_wait_timeout_seconds)
        # 领地入口确认步骤的重点预算：连续多帧未见标识时重点入口按钮，
        # 用尽仍不成功才保守停止（同难度弹窗打开确认的重点模式）
        self._entry_max_reclicks = max(0, entry_max_reclicks)
        self._entry_reclicks = 0
        self._entry_misses = 0
        # 聊天点击生效确认的重点预算（2026-09-16 实机：结算返回后挑战页盖着
        # 金币道具说明卡，聊天点击只关掉弹窗未进招募）；页签行连续
        # _CHAT_OPEN_RECLICK_MISSES 帧读不到本模式页签即重点，用尽仍不成功
        # 才保守停止
        self._chat_open_max_reclicks = max(0, chat_open_max_reclicks)
        self._chat_open_reclicks = 0
        self._chat_open_tab_hits = 0
        self._chat_open_tab_misses = 0
        # 当前入口等待步骤的 deadline 起点；None = 尚未开始计时
        self._entry_wait_started_at: float | None = None
        self._clock = clock
        self._rng = rng or random.Random()

        self._recruit_step = (
            _RecruitStep.SELECT_DIFFICULTIES
            if difficulty_selection_only
            else _RecruitStep.OPEN_TERRITORY
        )
        if self._recruit_step is _RecruitStep.OPEN_TERRITORY:
            self._start_entry_wait()
        self._difficulty_scroll_count = 0
        # 本目标难度的尝试期间是否见过彩虹候选：决定跳过后是否重置滚动预算
        self._difficulty_saw_candidates = False
        # 弹窗打开确认：settle → 轮询连续命中难度弹窗标识（合作=【合作模式】标题图；
        # 保卫宠物=「选择难度」收起栏被面板盖住，见 perception 配置）
        # （见 _action_confirm_difficulty_open）；重点打开按钮时全部重置
        self._difficulty_open_settled = False
        self._difficulty_open_hits = 0
        self._difficulty_open_misses = 0
        self._difficulty_open_reclicks = 0
        # 本次打开确认来自下一局刷新路径（确认后进 CLOSE_REFRESH 而非勾选）
        self._difficulty_open_is_refresh = False
        # 弹窗关闭确认：连续不可见弹窗标识（见 _action_close_difficulty_dialog）
        self._difficulty_close_attempts = 0
        self._difficulty_close_streak = 0

        self._summon_count = 0
        self._empty_board_count = 0
        self._no_progress_count = 0
        self._last_board_signature: tuple | None = None
        # 召唤未生效（多为金币不足）后的重试时刻；None 表示无需等待
        self._summon_retry_at: float | None = None
        # 被迫合并 3 星对后的赠送技能页确认（2026-08-21 用户策略，见
        # _action_build_main_c）：settle → 轮询连续命中【请选择1个额外技能】
        # 提示条 → 连续 N 次才选技能；连续 M 次未命中放弃等待恢复常规决策
        self._awaiting_merge_gift = False
        self._merge_gift_settled = False
        self._merge_gift_hits = 0
        self._merge_gift_misses = 0
        # 已拖动失败（重试耗尽）的合成对位置签名：本轮内跳过，不再反复拖
        # 同一落空的对；召唤新英雄或合成成功后棋盘演化出新的对再尝试
        self._failed_merge_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        # 批次化快速阶段（2026-09-10 用户策略）：识别一次后把动作攒成批，
        # 批内单帧快识别（界面标志 + 金币红/白），批次发完恢复多帧识别统一验证
        # 剩余批次召唤次数：批内每次点击前做金币门控，不足则等待、不消耗批次
        self._batch_summon_remaining = 0
        # 待发送批次合成队列：[(起点英雄, 终点英雄)]，互不相交的非主C低星对
        self._batch_merge_queue: list[tuple[BoardHero, BoardHero]] = []
        # 上一批次合成的位置签名：批末识别时仍原样存在的对 = 拖动未生效，
        # 记入失败记忆（批次拖动发送即提交，不做逐动作棋盘验证）
        self._pending_merge_batch_signatures: list[
            tuple[tuple[int, int], tuple[int, int]]
        ] = []

        self._opening_loaded = False
        self._opening_empty_checks = 0
        self._opening_exit_empty_count = 0
        self._opening_skill_selections = 0
        self._opening_clicks_blocked = False
        self._opening_wait_count = 0
        self._home_return_count = 0
        # 本会话是否已完成过难度勾选（首局选中或手动选过）；被踢回首页重新
        # 进入招募时不再重复勾选难度（再次点击已选中的难度会取消勾选）
        self._difficulty_done_session = False

        # 模式档案（方案 A）：None = 旧行为（无页签步骤、coop_chat 硬编码、
        # 结算含点赞宝箱）；正式流程由 Runner 按模式注入
        self._mode = mode_profile
        # 页签点击后的去向：首局路径进打开难度弹窗，下一局刷新路径进打开刷新弹窗
        self._after_tab_step = _RecruitStep.OPEN_DIFFICULTY_DIALOG
        # 保卫宠物难度选择（OCR 开环）：待选目标按配置顺序点击；行文字子串匹配。
        # 保序去重：重复目标会点击同一难度两次，实机确认再次点击已勾选难度
        # 会取消勾选（配置层已校验拒绝，这里兜底防御）
        self._remaining_pet_targets = list(dict.fromkeys(pet_difficulty_targets or []))
        self._selected_pet_targets: list[str] = []
        self._pet_difficulty_misses = 0
        self._pet_min_confidence = pet_difficulty_min_confidence
        # 挑战页每周弹窗（保卫宠物=本周Buff）连续读不到按钮文字的帧计数：
        # 达 _WEEKLY_BUFF_GRACE_MISSES 放行进招募（弹窗可能晚于页面弹出）
        self._weekly_buff_misses = 0

        self._match_started_at: float | None = None
        self._next_skill_at: float | None = None
        self._awaiting_main_candidates = False
        self._main_skill_empty_checks = 0
        self._main_page_missing_checks = 0
        # 局内技能页「刷新」：本局已用次数、按钮未识别到的等待重试计数
        # （2026-09-12 用户确认：每局一次，刷新后必选一张；「等卡面换新」由
        # 动作后验证承担——验证帧 OCR 出新卡面即已可识别，不再额外固定等待）
        self._skill_refresh_used_count = 0
        self._skill_refresh_button_waits = 0
        # 变身（2026-09-13 用户策略，设计文档 2026-09-13-hero-transform）：
        # locator 未接线（配置缺失/模板缺失降级）时整分支禁用
        self._transform_locator = transform_button_locator
        # 阶段机：None=未在进行；poll=已点英雄轮询按钮；confirm=已点按钮
        # 确认消失；verify=settle 等待后棋盘验证
        self._transform_stage: str | None = None
        self._transform_target: BoardHero | None = None
        self._transform_button_pos: tuple[int, int] | None = None
        self._transform_reclicks = 0
        self._transform_verify_attempts = 0
        # 本局已用变身次数（按钮确认消失后扣减；重置点与技能刷新一致）
        self._transform_used_count = 0
        self._transform_poll_deadline = 0.0
        # 放弃记忆：放弃变身时的棋盘签名；棋盘未演化前不再触发（防死循环）
        self._transform_skip_signature: tuple | None = None
        self._main_skill_selections = 0
        # 本局局内技能选择总次数（回补重置不清零；达到档案上限后停止选技能）
        self._main_skill_selections_total = 0
        # 处于数量回补阶段（选满 N 次技能回到培养）：此时培养门禁才检查主C数量；
        # 首次培养只看星级，让技能选择尽早开始
        self._topup_active = False
        # 本局主C数量已确认达标（回补补够或首次培养时数量就已够）：数量在选技能
        # 期间不会减少，2026-09-07 用户策略——本局内不再回培养复查数量，一直选技能
        self._topup_done = False
        # 邀请者模式从 INVITE_CLICK 起步（发起邀请），不走 FIND_COOP 抢合作
        if self._invite_mode:
            self.ctx.current_state = State.INVITE_CLICK

    @property
    def in_match_states(self) -> bool:
        """当前是否处于对局进行态（等待开局/培养/选技能）。

        供 runner 在动作后验证时判断「对局结束信号优先于动作验证」：
        返回按钮在这些状态下出现即意味着本局结束，棋盘类动作的验证
        必然不再满足，继续跑完验证帧只是白等（2026-09-06 实机）。
        """
        return self.ctx.current_state in _MATCH_STATES

    @property
    def skipped_difficulties(self) -> tuple[int, ...]:
        """诊断难度选择中未识别到的目标，按从大到小返回。"""
        return tuple(sorted(self._skipped_difficulties, reverse=True))

    def build_transitions(self) -> list[Transition]:
        """返回顶层业务阶段图；招募入口等小步骤保留在状态内部。"""
        wait = Action(kind="wait", duration=0.3, verification="immediate", reason="占位动作")
        return [
            Transition("find_coop", State.FIND_COOP, State.ENTER_MATCH, wait),
            # 邀请者组队链（spec §6）：点击链 → 等待过滤 → 等准备开局；冷却回点击链
            Transition("invite_click", State.INVITE_CLICK, State.INVITE_WAITING, wait),
            Transition("invite_filter", State.INVITE_WAITING, State.INVITE_START, wait),
            Transition("invite_kick", State.INVITE_WAITING, State.INVITE_COOLDOWN, wait),
            Transition("invite_reentry", State.INVITE_START, State.ENTER_MATCH, wait),
            Transition("invite_cooldown", State.INVITE_COOLDOWN, State.INVITE_CLICK, wait),
            Transition("ready", State.ENTER_MATCH, State.SELECT_OPENING_SKILLS, wait),
            Transition("select_opening", State.SELECT_OPENING_SKILLS, State.BUILD_MAIN_C, wait),
            Transition("cultivate_loop", State.BUILD_MAIN_C, State.BUILD_MAIN_C, wait),
            Transition("main_c_ready", State.BUILD_MAIN_C, State.SELECT_MAIN_C_SKILLS, wait),
            Transition("round_result", State.SELECT_MAIN_C_SKILLS, State.HANDLE_RESULT, wait),
            Transition("like", State.HANDLE_RESULT, State.CLAIM_REWARD, wait),
            Transition("claim_chest", State.CLAIM_REWARD, State.CHECK_ROUND_LIMIT, wait),
            Transition("next_round", State.CHECK_ROUND_LIMIT, State.FIND_COOP, wait),
            Transition("finish", State.CHECK_ROUND_LIMIT, State.COMPLETED, wait),
        ]

    def determine_state(self, observation: Observation) -> State:
        """以停止、窗口、结算和准备标志为优先，其余保持状态惯性。"""
        if self.ctx.round_count >= self.ctx.max_rounds:
            return State.COMPLETED
        if observation.flag("window_invalid"):
            return State.WINDOW_INVALID
        if self.ctx.current_state in _MATCH_STATES and observation.flag("return_button_visible"):
            return State.HANDLE_RESULT
        if self.ctx.current_state == State.FIND_COOP and observation.flag("ready_button_visible"):
            return State.ENTER_MATCH
        # 邀请者组队态保持惯性（决策与状态迁移全部在 decide_action 内完成）
        if self.ctx.current_state in _INVITE_TEAM_STATES:
            return self.ctx.current_state
        if self.ctx.current_state not in {State.UNKNOWN, State.WINDOW_INVALID}:
            return self.ctx.current_state
        return State.UNKNOWN

    def decide_action(
        self,
        observation: Observation,
        window_ctx: WindowContext | None = None,
    ) -> tuple[Action, State] | None:
        state = self.ctx.current_state
        # 游戏内任意阶段出现击杀奖励弹窗都优先关闭，再继续原流程
        if state in _MATCH_STATES:
            popup = self._flag_match(observation, "tan_chuang_match")
            if popup is not None:
                return (
                    Action(
                        kind="click",
                        target=popup.position,
                        duration=0.08,
                        verification="next_frame",
                        tag="close_popup",
                        reason="识别到击杀奖励弹窗，点击关闭",
                    ),
                    state,
                )
        # 结算阶段出现【双倍奖励】确认弹窗（多为结算瞬间误点「双倍奖励」触发，
        # 实机确认 2026-08-16）：优先点击取消回到正常结算，再继续点赞/领宝箱
        if state in _SETTLEMENT_STATES and observation.flag("double_reward_dialog_visible"):
            cancel = self._flag_match(observation, "double_reward_cancel_match")
            if cancel is not None:
                return (
                    Action(
                        kind="click",
                        target=cancel.position,
                        duration=0.08,
                        verification="next_frame",
                        tag="close_double_reward",
                        reason="识别到双倍奖励确认弹窗，点击取消回到正常结算",
                    ),
                    state,
                )
            return (
                Action(
                    kind="wait",
                    duration=0.3,
                    verification="immediate",
                    tag="double_reward_dialog_wait",
                    reason="双倍奖励弹窗已弹出，等待取消按钮可识别",
                ),
                state,
            )
        if state == State.FIND_COOP:
            return self._action_find_coop(observation, window_ctx)
        if state == State.INVITE_CLICK:
            return self._action_invite_click(observation, window_ctx)
        if state == State.INVITE_WAITING:
            return self._action_invite_waiting(observation, window_ctx)
        if state == State.INVITE_START:
            return self._action_invite_start(observation, window_ctx)
        if state == State.INVITE_COOLDOWN:
            return self._action_invite_cooldown(observation, window_ctx)
        if state == State.ENTER_MATCH:
            return self._action_ready(observation)
        if state == State.SELECT_OPENING_SKILLS:
            return self._action_opening_skills(observation, window_ctx)
        if state == State.BUILD_MAIN_C:
            return self._action_build_main_c(observation, window_ctx)
        if state == State.SELECT_MAIN_C_SKILLS:
            return self._action_main_c_skills(observation, window_ctx)
        if state == State.HANDLE_RESULT:
            if not self._like_chest_enabled():
                return (
                    Action(
                        kind="wait",
                        duration=0.3,
                        verification="immediate",
                        tag="like_skipped",
                        reason="保卫宠物结算：跳过点赞",
                    ),
                    State.CLAIM_REWARD,
                )
            return self._hotspot_click(
                "like",
                "like_result",
                "结算页点赞",
                State.CLAIM_REWARD,
                window_ctx,
            )
        if state == State.CLAIM_REWARD:
            if not self._like_chest_enabled():
                return (
                    Action(
                        kind="wait",
                        duration=0.3,
                        verification="immediate",
                        tag="claim_skipped",
                        reason="保卫宠物结算：跳过领宝箱",
                    ),
                    State.CHECK_ROUND_LIMIT,
                )
            # 实机确认：宝箱动画未播完时点返回无响应，领取后稳定等待再进入返回判定
            return self._hotspot_click(
                "claim_chest",
                "claim_chest",
                "领取结算宝箱",
                State.CHECK_ROUND_LIMIT,
                window_ctx,
                post_delay=self._run_config.reward_claim_return_delay,
            )
        if state == State.CHECK_ROUND_LIMIT:
            return self._action_return(observation)
        return None

    # ------------------------------------------------------------------
    # 招募与准备
    # ------------------------------------------------------------------

    def wants_board_watch(self) -> bool:
        # 强制召唤阶段（技能解锁前）不以棋盘识别为门禁：点击发送即计数，
        # 跳过多帧棋盘识别，按固定间隔快速连点；召唤满配置次数后恢复正常识别。
        # 批次召唤/批次合成进行中同理：批内单帧快识别（界面标志 + 金币红/白），
        # 批次发完后恢复多帧识别统一验证。
        # 等待4星赠送技能页期间页面遮住棋盘、识别必然为空，同样退回单帧
        # 快识别（BUILD 状态标志集含标题图与击杀弹窗，弹窗优先不受影响）
        if self._awaiting_merge_gift:
            return False
        if self._transform_stage is not None:
            # 变身阶段机：poll/confirm 单帧快识别（含按钮专项模式）；
            # verify 恢复多帧棋盘识别做动作后验证
            return self._transform_stage == "verify"
        if self._batch_summon_remaining > 0 or self._batch_merge_queue:
            return False
        return self._summon_count >= self._run_config.minimum_summon_count_before_skills

    def observation_mode(self) -> str | None:
        if self.ctx.current_state in _INVITE_TEAM_STATES:
            # 邀请者组队态（spec §6）：等待态用限域 OCR 读「邀请」文字与
            # 帮助者关卡进度；其余组队态无专项识别（开局按钮走全局 locator
            # 标志，见 perception._INTERFACE_FLAGS_BY_STATE 的 INVITE_START）
            if self.ctx.current_state == State.INVITE_WAITING:
                return "invite_waiting"
            return None
        if self._transform_stage is not None:
            # 变身阶段机接管观察模式（设计 §4.2）：poll/confirm 用按钮专项
            # 模式；verify 恢复默认（无专项模式）。整阶段机拦截而不是只拦
            # poll/confirm——否则 verify 阶段落入下方招募步骤分支（如测试
            # 夹具停在 OPEN_TERRITORY 时返回 territory_nav，2026-09-13 执行
            # 反馈修正）
            if self._transform_stage in ("poll", "confirm"):
                return "transform_button"
            return None
        if self._recruit_step is _RecruitStep.DETECT_ENTRY_PAGE:
            # 退队落点探测：只区分首页/非首页，首页也改走领地入口
            return "home_page"
        if self._recruit_step in (
            _RecruitStep.OPEN_TERRITORY,
            _RecruitStep.CONFIRM_TERRITORY_OPEN,
        ):
            # 领地入口：OCR 底部导航领地格文字（领地=可点，返回=已打开）
            return "territory_nav"
        if self._recruit_step in (
            _RecruitStep.OPEN_MODE_SELECT,
            _RecruitStep.CONFIRM_MODE_SELECT_OPEN,
            _RecruitStep.SELECT_MODE_CARD,
        ):
            # 领地入口：OCR 模式选择弹窗标题（开关锚点）+ 本模式卡片文字
            return "mode_select"
        if self._recruit_step in (
            _RecruitStep.CONFIRM_CHALLENGE_PAGE,
            _RecruitStep.CLOSE_WEEKLY_BUFF,
        ):
            # 领地入口收尾：OCR 挑战页按钮文字（到达锚点）；每周弹窗关闭
            # 步骤复用本模式同帧读弹窗按钮行（weekly_buff_rows）
            return "challenge_page"
        if self._recruit_step in (
            _RecruitStep.CONFIRM_CHAT_OPEN,
            _RecruitStep.SELECT_MODE_TAB,
        ) and (self._mode is not None and self._mode.entry.mode_tab_text):
            # 聊天点击生效确认与页签选择共用页签行 OCR：确认以「读到本
            # 模式页签」为生效判据——挑战页该区域是「拥有：」等无关文字，
            # 且招募页底部同样有「开始挑战」按钮、锚点无法区分两页
            # （2026-09-17 实机）。仅进招募后的一两帧识别，不在高频循环内
            return "mode_tab"
        if self._recruit_step is _RecruitStep.SELECT_DIFFICULTIES:
            # 勾选难度需要识别候选列表（16 个难度模板）
            return "coop_difficulty"
        if self._recruit_step in (
            _RecruitStep.CONFIRM_DIFFICULTY_OPEN,
            _RecruitStep.CLOSE_DIFFICULTY_DIALOG,
            _RecruitStep.CLOSE_REFRESH_DIFFICULTY_DIALOG,
        ):
            # 打开/关闭确认只需难度弹窗标识判定弹窗开关，不做 16 个
            # 难度候选匹配：候选匹配会把单帧识别拖慢到超过截图时效
            # （实机 2026-08-17：慢速机器上每帧 6 秒导致动作全部被时效拒绝）
            return "difficulty_dialog"
        if self._recruit_step is _RecruitStep.JOIN_COOP:
            # 抢合作：连点加入期间游戏随时可能放行，准备按钮出现即抢到。
            # 合作只能自己点加入进入（游戏不会主动拉人，实机确认 2026-08-17），
            # 其余入口子步骤不查任何全局标志
            return "coop_grab"
        return None

    # ------------------------------------------------------------------
    # 邀请者模式组队（spec 2026-09-21-initiator-monkey-design.md §6）
    # ------------------------------------------------------------------

    def _invite_hotspot_point(
        self, name: str, window_ctx: WindowContext | None
    ) -> tuple[int, int] | None:
        point = self._hotspot_point(name, window_ctx)
        if point is None:
            logger.error("邀请者流程缺少 hotspot: %s（configs/tasks.yaml）", name)
        return point

    def _action_invite_click(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        """邀请点击链（spec §6，2026-09-24 流程确认）。

        第 0 步点挑战页【开始挑战】进入邀请页面（复用 coop_challenge_start_area
        文字框中心，与保卫宠物挑战按钮同款「锚点+点击点二合一」用法）；
        随后 5 步：邀请 → 一键邀请(弹窗) → 好友 → 公会招募 → 一键邀请(好友页)。
        每步点击后保留固定间隔供界面切换；链完成进入 INVITE_WAITING 并重置
        重发计数。
        """
        if self._invite_step == 0:
            # 第 0 步：挑战页【开始挑战】（进邀请页面）
            point = self._invite_hotspot_point("invite_start_challenge", window_ctx)
            if point is None:
                return None
            self._invite_step += 1
            return (
                Action(
                    kind="click",
                    target=point,
                    duration=0.08,
                    post_delay=0.8,
                    verification="immediate",
                    tag="invite_step_0",
                    reason="邀请点击链 1/7：开始挑战（进邀请页面）",
                ),
                State.INVITE_CLICK,
            )
        steps: tuple[tuple[str, str], ...] = (
            ("invite_button", "邀请按钮"),
            ("invite_one_click", "一键邀请（弹窗层）"),
            ("invite_friend_tab", "好友页签"),
            ("invite_guild_recruit", "公会招募"),
            ("invite_one_click", "一键邀请（好友页）"),
            ("invite_popup_close", "关闭邀请弹窗"),
        )
        index = self._invite_step - 1
        if index >= len(steps):
            self._invite_step = 0
            self._invite_reinvite_count = 0
            self._invite_wait_since = self._clock()
            return (
                Action(
                    kind="wait",
                    duration=0.5,
                    verification="immediate",
                    tag="invite_wait_entry",
                    reason="邀请已发出，进入等待帮助者进房",
                ),
                State.INVITE_WAITING,
            )
        hotspot_name, label = steps[index]
        point = self._invite_hotspot_point(hotspot_name, window_ctx)
        if point is None:
            return None
        action = (
            Action(
                kind="click",
                target=point,
                duration=0.08,
                post_delay=0.6,
                verification="immediate",
                tag=f"invite_step_{self._invite_step}",
                reason=f"邀请点击链 {self._invite_step + 1}/7：{label}",
            ),
            State.INVITE_CLICK,
        )
        self._invite_step += 1
        return action

    def _action_invite_waiting(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        """等待进房 + 帮助者过滤（spec §6 INVITE_WAITING）。

        「邀请」文字可识别 = 无人进房：超 reinvite_after_seconds 重走点击链。
        识别不到 = 有人进房（复检一次防遮挡误判）：点邀请位置弹帮助者信息，
        OCR 关卡进度，× 关闭弹层；≥阈值进 INVITE_START，<阈值请离进冷却。
        """
        invite_visible = self._invite_text_visible(observation)
        now = self._clock()
        if self._invite_wait_since is None:
            self._invite_wait_since = now
        if invite_visible:
            waited = now - self._invite_wait_since
            if waited >= self._invite.reinvite_after_seconds:
                self._invite_reinvite_count += 1
                if self._invite_reinvite_count > self._invite.max_empty_reinvites:
                    logger.error(
                        "连续 %d 次空等重发仍无人进房，保守停止", self._invite_reinvite_count
                    )
                    return None
                logger.info(
                    "等待 %.0f 秒无人进房，重走邀请点击链（第 %d 次重发）",
                    waited,
                    self._invite_reinvite_count,
                )
                self._invite_step = 0
                self._invite_wait_since = None
                return (
                    Action(
                        kind="wait",
                        duration=0.2,
                        verification="immediate",
                        tag="invite_reinvite",
                        reason="无人进房超时，重新发起邀请",
                    ),
                    State.INVITE_CLICK,
                )
            return (
                Action(
                    kind="wait",
                    duration=1.0,
                    verification="immediate",
                    tag="invite_waiting",
                    reason=f"等待帮助者进房（已等 {waited:.0f}s，识别到邀请文字）",
                ),
                State.INVITE_WAITING,
            )
        # 帮助者信息层处理（spec §6 INVITE_WAITING 第 1-3 步）。
        # 待执行分流决定（帧 3）优先于一切——弹层已关，直接执行 kick/accept；
        # 弹层已打开 → 跳过复检直接读进度+关层（弹层必然遮着邀请文字，
        # 再复检只会死循环）；未打开 → 先关一次残留弹窗再读文字——邀请
        # 弹窗关闭后文字有恢复延迟/弹窗可能没关掉，直接判进房会把自己的
        # 信息层当帮助者（2026-09-24 实机：误读自己 475 进度）
        if self._invite_pending_decision is not None:
            return self._invite_process_helper(observation, window_ctx)
        if self._invite_helper_opened:
            return self._invite_process_helper(observation, window_ctx)
        if not self._invite_dismiss_checked:
            # 首次判定进房前：点一次邀请弹窗 × 清残留，下一帧重读文字
            self._invite_dismiss_checked = True
            point = self._invite_hotspot_point("invite_popup_close", window_ctx)
            if point is None:
                return None
            return (
                Action(
                    kind="click",
                    target=point,
                    duration=0.08,
                    post_delay=1.0,
                    verification="immediate",
                    tag="invite_dismiss_residue",
                    reason="邀请文字不可见：先关一次残留弹窗再判定（防把弹窗当进房）",
                ),
                State.INVITE_WAITING,
            )
        self._invite_dismiss_checked = False
        self._invite_wait_since = None
        return self._invite_process_helper(observation, window_ctx)

    def _invite_process_helper(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        """帮助者信息层两帧阶段机（spec §6 INVITE_WAITING 第 1-3 步）。

        帧 1：点 invite_button 打开弹层（_invite_helper_opened=True）。
        帧 2：弹层已开，读 helper_stage_progress → 点 × 关层，并把分流
        决定存入 ``_invite_pending_decision``（accept / kick）。
        帧 3：弹层已关，按决定执行——合格转 INVITE_START；不合格点
        【请离】转 INVITE_COOLDOWN；进度连读失败达上限保守停止。
        """
        progress = observation.raw_data.get("helper_stage_progress")
        if not self._invite_helper_opened:
            if self._invite_pending_decision is not None:
                # 帧 3：弹层已关，执行分流决定
                decision = self._invite_pending_decision
                self._invite_pending_decision = None
                return self._invite_execute_decision(decision, window_ctx)
            # 帧 1：打开弹层
            self._invite_helper_opened = True
            point = self._invite_hotspot_point("invite_button", window_ctx)
            if point is None:
                return None
            return (
                Action(
                    kind="click",
                    target=point,
                    duration=0.08,
                    post_delay=1.0,
                    verification="immediate",
                    tag="invite_open_helper_info",
                    reason="有人进房，打开帮助者信息层读取关卡进度",
                ),
                State.INVITE_WAITING,
            )
        # 帧 2：弹层已开。× 必然在层上（close_button flag / 兜底点），
        # 关层与分流决定一起记下，下一帧执行
        close = self._flag_match(observation, "close_button_match")
        self._invite_helper_opened = False
        if close is not None:
            close_action = Action(
                kind="click",
                target=close.position,
                duration=0.08,
                post_delay=0.6,
                verification="immediate",
                tag="invite_close_helper_info",
                reason=f"帮助者进度读取完成（{progress}），点 × 关闭弹层",
            )
        else:
            point = self._invite_hotspot_point("invite_close_fallback", window_ctx)
            if point is None:
                return None
            close_action = Action(
                kind="click",
                target=point,
                duration=0.08,
                post_delay=0.6,
                verification="immediate",
                tag="invite_close_helper_fallback",
                reason=f"帮助者进度读取完成（{progress}），×未识别，点兜底位置关闭",
            )
        if progress is None:
            self._invite_progress_misses += 1
            if self._invite_progress_misses >= self._INVITE_PROGRESS_MAX_MISSES:
                logger.error(
                    "帮助者关卡进度连续 %d 次读取失败，保守停止（不盲踢不盲开局）",
                    self._invite_progress_misses,
                )
                return None
            logger.warning(
                "帮助者关卡进度第 %d 次读取失败，关层后重试",
                self._invite_progress_misses,
            )
            return close_action, State.INVITE_WAITING
        self._invite_progress_misses = 0
        threshold = self._invite.min_stage_progress
        if progress < threshold:
            self._invite_kick_count += 1
            if self._invite_kick_count > self._invite.max_consecutive_kicks:
                logger.error(
                    "连续请离 %d 个低进度帮助者（匹配池可能整体不合格），保守停止",
                    self._invite_kick_count,
                )
                return None
            self._invite_pending_decision = "kick"
            logger.info(
                "帮助者关卡进度 %d < %d，关层后请离（第 %d/%d 个）",
                progress,
                threshold,
                self._invite_kick_count,
                self._invite.max_consecutive_kicks,
            )
        else:
            self._invite_kick_count = 0
            self._invite_pending_decision = "accept"
            logger.info("帮助者关卡进度 %d ≥ %d，合格", progress, threshold)
        return close_action, State.INVITE_WAITING

    def _invite_execute_decision(
        self,
        decision: str,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        """帧 3：执行帧 2 记下的分流决定（弹层已关，界面无遮挡）。"""
        if decision == "kick":
            kick = self._invite_hotspot_point("invite_kick", window_ctx)
            if kick is None:
                return None
            logger.info(
                "请离低进度帮助者（第 %d/%d 个），%ds 冷却后重邀",
                self._invite_kick_count,
                self._invite.max_consecutive_kicks,
                int(self._invite.kick_cooldown_seconds),
            )
            return (
                Action(
                    kind="click",
                    target=kick,
                    duration=0.08,
                    post_delay=0.5,
                    verification="next_frame",
                    tag="invite_kick_helper",
                    reason="请离低于阈值的帮助者",
                ),
                State.INVITE_COOLDOWN,
            )
        self._invite_wait_since = self._clock()
        return (
            Action(
                kind="wait",
                duration=0.3,
                verification="immediate",
                tag="invite_helper_accepted",
                reason="帮助者合格，等待其点准备后开局",
            ),
            State.INVITE_START,
        )

    _invite_text_missing_frames: int = 0
    _invite_helper_opened: bool = False
    _invite_pending_decision: str | None = None
    _invite_dismiss_checked: bool = False
    _invite_progress_misses: int = 0
    _INVITE_PROGRESS_MAX_MISSES: int = 5

    def _invite_text_visible(self, observation: Observation) -> bool:
        """读感知层 OCR 结果：邀请文字可见 = 无人进房。缺失按可见（继续等）。"""
        value = observation.raw_data.get("invite_text_visible")
        return True if value is None else bool(value)

    def _action_invite_start(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        """等帮助者准备：未准备模板 → 继续等；已准备模板 → 点击开局。"""
        ready_match = self._flag_match(observation, "invite_start_ready_match")
        if ready_match is not None:
            return (
                Action(
                    kind="click",
                    target=ready_match.position,
                    duration=0.08,
                    verification="next_frame",
                    tag="invite_start_game",
                    reason="帮助者已准备，点击开始对局",
                ),
                State.ENTER_MATCH,
            )
        not_ready = self._flag_match(observation, "invite_start_not_ready_match")
        if not_ready is not None:
            return (
                Action(
                    kind="wait",
                    duration=2.0,
                    verification="immediate",
                    tag="invite_wait_ready",
                    reason="帮助者未准备，等待其点准备",
                ),
                State.INVITE_START,
            )
        # 两个模板都没识别到：帮助者可能已退出（邀请文字重现），回等待态。
        # 若连续多帧双模板全无且文字也不可见，说明邀请文字判定可能是误报
        # （残留弹窗误判进房，实际没人）——回等待态重新走判定（自愈）
        if self._invite_text_visible(observation):
            logger.info("开局按钮消失且邀请文字重现，帮助者已退出，回到等待")
            self._invite_wait_since = self._clock()
            self._invite_dismiss_checked = False
            return (
                Action(
                    kind="wait",
                    duration=0.5,
                    verification="immediate",
                    tag="invite_helper_left",
                    reason="帮助者退出房间，继续等待新帮助者",
                ),
                State.INVITE_WAITING,
            )
        return (
            Action(
                kind="wait",
                duration=1.0,
                verification="immediate",
                tag="invite_start_ambiguous",
                reason="开局按钮未识别（可能界面过渡），稍后重试",
            ),
            State.INVITE_START,
        )

    def _action_invite_cooldown(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        """请离后冷却：到时重走 5 步点击链。"""
        now = self._clock()
        if self._invite_cooldown_until is None:
            self._invite_cooldown_until = now + self._invite.kick_cooldown_seconds
        remaining = self._invite_cooldown_until - now
        if remaining > 0:
            return (
                Action(
                    kind="wait",
                    duration=min(remaining, 2.0),
                    verification="immediate",
                    tag="invite_cooldown",
                    reason=f"请离后冷却，{remaining:.0f}s 后重新邀请",
                ),
                State.INVITE_COOLDOWN,
            )
        self._invite_cooldown_until = None
        self._invite_step = 0
        self._invite_wait_since = None
        return (
            Action(
                kind="wait",
                duration=0.2,
                verification="immediate",
                tag="invite_cooldown_done",
                reason="冷却结束，重新发起邀请",
            ),
            State.INVITE_CLICK,
        )

    # ------------------------------------------------------------------
    # 邀请者模式局内扩展（spec §5.4/§5.5）
    # ------------------------------------------------------------------

    def _effective_opening_summons(self) -> int:
        """开局召唤次数：档案覆盖优先（opening_summons>0），否则 run 配置。"""
        if self._invite_mode and self._profile_opening_summons > 0:
            return self._profile_opening_summons
        return self._run_config.minimum_summon_count_before_skills

    def _effective_batch_size(self) -> int:
        """凑阵容阶段每批召唤次数（monkey 档案 summon_batch_size，默认 3）。"""
        if self._invite_mode and self._main_c_profile is not None:
            return self._main_c_profile.summon_batch_size
        return 12  # 非 invite：无批限制（原行为=全部空格）

    def _invite_gold_wait_assist(
        self,
        board: BoardSnapshot,
        observation: Observation,
        why: str,
    ) -> tuple[Action, State] | None:
        """邀请者卡住出口（spec §5.3/§5.5）：金币不足且无法行动时的处置。

        阵容不齐 + 棋盘满 + 无安全合成 → 变身碰辅助（次数未耗尽时）；
        变身也耗尽 → 干等（重置无进展计数，等对局结束）。返回 None 表示
        没有邀请者特定处置，调用方走原有金币等待。
        """
        if not self._invite_mode:
            return None
        if not board.capacity.total_slots or board.capacity.available > 0:
            return None  # 有空位：不存在卡住，走原有召唤等待
        missing = self._strategy.lineup_missing(board)
        if not missing:
            return None  # 阵容已齐：门禁未达只因主C星级，无变身意义
        if (
            self._run_config.transform_enabled
            and self._transform_locator is not None
            and self._transform_skip_signature is None
        ):
            target = self._strategy.select_transform_assist(
                board,
                self._transform_used_count,
                self._run_config.transform_max_per_round,
            )
            if target is not None:
                self._transform_stage = "poll"
                self._transform_target = target
                self._transform_button_pos = None
                self._transform_reclicks = 0
                self._transform_verify_attempts = 0
                self._transform_poll_deadline = (
                    self._clock() + self._run_config.transform_button_appear_timeout_seconds
                )
                logger.info(
                    "卡住变身碰辅助：%s%d星（缺 %s，已用 %d/%d 次）",
                    target.hero_type,
                    target.star_level,
                    "、".join(missing),
                    self._transform_used_count,
                    self._run_config.transform_max_per_round,
                )
                return (
                    Action(
                        kind="click",
                        target=target.position,
                        duration=0.08,
                        post_delay=self._run_config.transform_button_poll_interval_seconds,
                        verification="immediate",
                        tag="transform_click_hero",
                        reason="卡住：点击牺牲英雄变身碰缺的辅助",
                    ),
                    State.BUILD_MAIN_C,
                )
        # 变身耗尽/未接线：干等（保持现状，用户 2026-09-21 定案）
        self._no_progress_count = 0
        self._last_board_signature = None
        return (
            Action(
                kind="wait",
                duration=self._run_config.gold_red_wait_seconds,
                verification="immediate",
                tag="invite_stuck_wait",
                reason=(
                    f"{why}：棋盘满、阵容缺 {'、'.join(missing)}、"
                    "无安全合成且变身不可用，保持现状干等"
                ),
            ),
            State.BUILD_MAIN_C,
        )

    def _invite_ready_done_action(
        self, window_ctx: WindowContext | None = None
    ) -> tuple[Action, State] | None:
        """邀请者「准备完成」一次性点击。

        Returns:
            首次调用返回点击动作；已点过（或非邀请模式）返回 None。
        """
        if not self._invite_mode or self._invite_ready_clicked:
            return None
        point = self._hotspot_point("opening_ready_done", window_ctx)
        if point is None:
            return None
        self._invite_ready_clicked = True
        return (
            Action(
                kind="click",
                target=point,
                duration=0.08,
                post_delay=1.5,
                verification="next_frame",
                tag="invite_ready_done",
                reason="开局召唤完成，点击「准备完成」确认进入培养",
            ),
            State.BUILD_MAIN_C,
        )

    def _action_find_coop(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        if self._recruit_step != _RecruitStep.JOIN_COOP:
            return self._action_recruit_entry(observation, window_ctx)
        # 兜底：带着全屏弹窗回到 FIND_COOP（如抢合作保守停止后重进）时先关闭，
        # 否则弹窗挡住 join_coop 会形成「弹窗不关、抢不停」的循环（实机
        # 2026-09-14：连点误点自己头像弹出「个人信息」）。× 为跨弹窗皮肤
        # 一致的标准关闭按钮；模板未采集的分辨率包识别不到，行为同旧版
        popup = self._flag_match(observation, "close_button_match")
        if popup is not None:
            return (
                Action(
                    kind="click",
                    target=popup.position,
                    duration=0.08,
                    verification="next_frame",
                    tag="close_grab_popup",
                    reason="识别到全屏弹窗关闭按钮，先关闭再抢合作",
                ),
                State.FIND_COOP,
            )
        # JOIN_COOP：抢合作由 Runner 的双线程协调器执行（连点 join + 并行识别
        # 准备按钮）。任务只发出信号动作；发现准备按钮后回到主循环，由
        # _action_ready 单线程点击准备。
        return (
            Action(
                kind="grab_coop",
                tag="find_coop_grab",
                reason="抢合作：连点 join_coop 并行识别准备按钮",
            ),
            State.FIND_COOP,
        )

    def _action_select_mode_tab_ocr(
        self,
        observation: Observation,
        tab_text: str,
    ) -> tuple[Action, State] | None:
        """按 OCR 文字定位点击本模式页签（三个页签共用一条识别区）。

        命中即点文字框中心（开环，同难度勾选策略；页签点击幂等，后续步骤
        自有门禁）。识别不到时不盲点固定坐标：先等待下一帧重试（进入招募
        可能有动画），连续 miss 达上限报错停止——页签行读不到文字大概率
        不在招募页，此时任何点击都不安全。等待帧返回显式 wait 动作而不是
        None：None 会被 Runner 计入无进展预算（error_restart_max_consecutive=3
        小于 miss_limit=8），合法的页面切换等待会被提前保守停止（2026-09-16
        实机）。
        """
        rows = [
            row
            for row in (observation.raw_data.get("mode_tab_rows") or [])
            if tab_text in row.text
        ]
        if not rows:
            self._mode_tab_misses += 1
            if self._mode_tab_misses >= self._mode_tab_miss_limit:
                raise RuntimeError(
                    f"连续 {self._mode_tab_miss_limit} 帧未识别到「{tab_text}」页签文字，"
                    "请确认当前在招募页后重新运行（mode_tab OCR）"
                )
            logger.warning(
                "frame=%d 未识别到「%s」页签（第 %d/%d 帧），等待下一帧",
                observation.frame_id,
                tab_text,
                self._mode_tab_misses,
                self._mode_tab_miss_limit,
            )
            return self._entry_wait_action(
                "mode_tab_pending",
                f"未识别到「{tab_text}」页签文字，等待下一帧复识",
            )
        best = max(rows, key=lambda row: row.confidence)
        self._mode_tab_misses = 0
        logger.info(
            "OCR 定位「%s」页签 conf=%.2f client=%s",
            tab_text,
            best.confidence,
            best.position,
        )
        return (
            Action(
                kind="click",
                target=best.position,
                duration=0.08,
                verification="immediate",
                tag="open_mode_tab",
                reason=f"进入招募：点击「{tab_text}」页签",
            ),
            State.FIND_COOP,
        )

    def _start_entry_wait(self) -> None:
        """重置领地入口等待的 deadline 起点（每进入一个等待步骤时调用）。"""
        self._entry_wait_started_at = self._clock()

    def _enter_chat_open_confirm(self) -> None:
        """聊天点击已发送：进入生效确认步骤，重点预算与连续帧计数从零起算。

        入口等待 deadline 同步重新起算（整个确认步骤共用一个秒级上限，
        重点不重置——防止 OCR 抖动下重点预算与超时叠加拖太久）。
        """
        self._recruit_step = _RecruitStep.CONFIRM_CHAT_OPEN
        self._chat_open_reclicks = 0
        self._chat_open_tab_hits = 0
        self._chat_open_tab_misses = 0
        self._start_entry_wait()

    def _entry_deadline_exceeded(self, observation: Observation, what: str) -> bool:
        """领地入口等待是否已达秒级上限；达到时打 ERROR（调用方保守停止）。

        deadline 惰性起算：首次调用时从当前时刻开始计时。
        """
        if self._entry_wait_started_at is None:
            self._entry_wait_started_at = self._clock()
            return False
        elapsed = self._clock() - self._entry_wait_started_at
        if elapsed < self._entry_wait_timeout_seconds:
            return False
        logger.error(
            "frame=%d 领地入口等待超时（%.0fs/%.0fs）：%s，保守停止",
            observation.frame_id,
            elapsed,
            self._entry_wait_timeout_seconds,
            what,
        )
        return True

    def _entry_wait_action(self, tag: str, reason: str) -> tuple[Action, State]:
        """领地入口等待步骤的轮询动作：短暂等待后下一帧复识。"""
        return (
            Action(
                kind="wait",
                duration=0.3,
                verification="immediate",
                tag=tag,
                reason=reason,
            ),
            State.FIND_COOP,
        )

    def _action_open_territory(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        """领地入口第一步：按导航文字决定点领地、直进弹窗或继续等。

        读到「领地」=未进领地页，点击 territory_nav；读到「返回」=已在
        领地页（如手动打开过），跳过点击与等待直进模式选择；读不到文字
        不盲点（当前页面可能没有底部导航），等下一帧复识，超过入口等待
        上限保守停止。
        """
        text = str(observation.raw_data.get("territory_nav_text") or "")
        if "返回" in text:
            logger.info(
                "frame=%d 领地页已打开（导航读到「返回」），跳过点领地",
                observation.frame_id,
            )
            self._recruit_step = _RecruitStep.OPEN_MODE_SELECT
            self._start_entry_wait()
            return (
                Action(
                    kind="wait",
                    duration=0.0,
                    verification="immediate",
                    tag="territory_open_confirmed",
                    reason="领地页已打开（导航读到「返回」）",
                ),
                State.FIND_COOP,
            )
        if "领地" in text:
            return self._hotspot_click(
                "territory_nav",
                "open_territory",
                "领地入口：点击底部导航「领地」",
                State.FIND_COOP,
                window_ctx,
            )
        if self._entry_deadline_exceeded(observation, "底部导航未读到「领地/返回」文字"):
            return None
        return self._entry_wait_action(
            "territory_nav_pending",
            "底部导航未读到「领地/返回」文字，等待下一帧复识",
        )

    def _action_confirm_territory_open(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        """正向确认领地页真的打开：导航文字从「领地」变成「返回」。

        页面打开快慢不定、首次点击可能被吞（实机 2026-09-12：点击后 60 秒
        导航一直读「领地」）：导航仍读「领地」=页面确认未打开，连续多帧
        未变化时**重点领地按钮**（最多 entry_max_reclicks 次，每次重点重新
        起算等待上限）——再点一次是安全的，领地没开就不会误点「返回」退出；
        导航读不到文字时不重点（可能已在领地页，重点会误点「返回」），只
        等待。重点用尽仍打不开则保守停止——绝不带着「没打开」的假设点
        进入按钮。
        """
        text = str(observation.raw_data.get("territory_nav_text") or "")
        if "返回" in text:
            logger.info(
                "frame=%d 领地页确认打开（导航读到「返回」）", observation.frame_id
            )
            self._recruit_step = _RecruitStep.OPEN_MODE_SELECT
            self._start_entry_wait()
            return (
                Action(
                    kind="wait",
                    duration=0.0,
                    verification="immediate",
                    tag="territory_open_confirmed",
                    reason="领地页已确认打开（导航读到「返回」）",
                ),
                State.FIND_COOP,
            )
        self._entry_misses += 1
        if "领地" in text and self._entry_misses >= _ENTRY_RECLICK_MISSES:
            if self._entry_reclicks >= self._entry_max_reclicks:
                logger.error(
                    "frame=%d 重点领地 %d 次后导航仍未变成「返回」，保守停止",
                    observation.frame_id,
                    self._entry_reclicks,
                )
                return None
            self._entry_reclicks += 1
            self._entry_misses = 0
            self._start_entry_wait()
            return self._hotspot_click(
                "territory_nav",
                "open_territory",
                f"领地页未打开，重点领地按钮"
                f"（{self._entry_reclicks}/{self._entry_max_reclicks}）",
                State.FIND_COOP,
                window_ctx,
            )
        if self._entry_deadline_exceeded(observation, "点领地后导航未变成「返回」"):
            return None
        return self._entry_wait_action(
            "territory_open_pending",
            "等待领地页打开（导航仍为「领地」）",
        )

    def _action_confirm_mode_select_open(
        self,
        observation: Observation,
    ) -> tuple[Action, State] | None:
        """正向确认模式选择弹窗打开：标题区域 OCR 读到「模式选择」。

        点击进入按钮后弹窗有延迟；超过入口等待上限仍未出现说明点击未生效
        或页面异常，保守停止。标题偶发漏读时，本模式卡名文字可读同样证明
        弹窗已打开（卡名与标题同帧识别），直接放行选卡。本步骤不做盲点
        重点：进入按钮坐标落在弹窗卡面附近，弹窗已开时重点可能误选另一
        模式卡片。
        """
        card_text = self._mode.entry.mode_card_text if self._mode else None
        card_rows = observation.raw_data.get("mode_card_rows") or []
        card_hit = bool(card_text) and any(
            card_text in row.text for row in card_rows
        )
        if observation.flag("mode_select_visible") or card_hit:
            logger.info(
                "frame=%d 模式选择弹窗确认打开（标题命中=%s 卡名兜底=%s）",
                observation.frame_id,
                observation.flag("mode_select_visible"),
                card_hit,
            )
            self._recruit_step = _RecruitStep.SELECT_MODE_CARD
            self._start_entry_wait()
            return (
                Action(
                    kind="wait",
                    duration=0.0,
                    verification="immediate",
                    tag="mode_select_open_confirmed",
                    reason="模式选择弹窗已确认打开（标题命中或卡名可读）",
                ),
                State.FIND_COOP,
            )
        if self._entry_deadline_exceeded(observation, "点进入按钮后模式选择弹窗未出现"):
            return None
        return self._entry_wait_action(
            "mode_select_open_pending",
            "等待模式选择弹窗打开（标题未命中）",
        )

    def _action_select_mode_card(
        self,
        observation: Observation,
    ) -> tuple[Action, State] | None:
        """按 OCR 文字定位点击本模式的卡片（合作=「合作」/宠物=「保卫」）。

        命中即点文字框中心（开环，同页签/难度勾选策略；挑战页到达有独立
        门禁）。识别不到时不盲点：等待下一帧复识，超过入口等待上限保守
        停止——弹窗内读不到卡片文字大概率弹窗状态异常。
        """
        card_text = self._mode.entry.mode_card_text if self._mode else None
        if not card_text:
            logger.error(
                "frame=%d 模式档案未配置 mode_card_text，领地入口无法选卡",
                observation.frame_id,
            )
            return None
        rows = [
            row
            for row in (observation.raw_data.get("mode_card_rows") or [])
            if card_text in row.text
        ]
        if not rows:
            if self._entry_deadline_exceeded(
                observation, f"模式选择弹窗内未识别到「{card_text}」卡片"
            ):
                return None
            return self._entry_wait_action(
                "mode_card_pending",
                f"未识别到「{card_text}」卡片文字，等待下一帧复识",
            )
        best = max(rows, key=lambda row: row.confidence)
        logger.info(
            "OCR 定位「%s」模式卡 conf=%.2f client=%s",
            card_text,
            best.confidence,
            best.position,
        )
        return (
            Action(
                kind="click",
                target=best.position,
                duration=0.08,
                verification="immediate",
                tag="mode_card_selected",
                reason=f"领地入口：点击「{card_text}」模式卡",
            ),
            State.FIND_COOP,
        )

    def _action_confirm_challenge_page(
        self,
        observation: Observation,
    ) -> tuple[Action, State] | None:
        """正向确认已到达本模式挑战页：按钮区域 OCR 读到挑战文字。

        合作挑战页读「开始挑战」、保卫宠物挑战页读「挑战」（模式档案
        challenge_text）。超过入口等待上限仍未读到说明选卡未生效或页面
        异常，保守停止。
        """
        challenge_text = self._mode.entry.challenge_text if self._mode else None
        text = str(observation.raw_data.get("challenge_text") or "")
        if challenge_text and challenge_text in text:
            logger.info(
                "frame=%d 已到达挑战页（读到「%s」）", observation.frame_id, challenge_text
            )
            if self._invite_mode:
                # 邀请者（spec 2026-09-21 §6）：导航到合作挑战页即止——【邀请】
                # 按钮就在挑战页上，无难度选择、不进招募频道；INVITE_CLICK
                # 从【开始挑战】开始走 5 步邀请链
                self._invite_step = 0
                self._invite_wait_since = None
                return (
                    Action(
                        kind="wait",
                        duration=0.3,
                        verification="immediate",
                        tag="invite_nav_done",
                        reason="邀请者已到挑战页，开始邀请点击链",
                    ),
                    State.INVITE_CLICK,
                )
            # 配置了每周弹窗识别（保卫宠物=本周Buff）先关弹窗再进招募；
            # 等待 deadline 从进入本步骤起重新起算
            if self._mode is not None and self._mode.entry.weekly_buff_text:
                self._recruit_step = _RecruitStep.CLOSE_WEEKLY_BUFF
                self._start_entry_wait()
            else:
                self._recruit_step = _RecruitStep.OPEN_CHALLENGE_CHAT
            return (
                Action(
                    kind="wait",
                    duration=0.0,
                    verification="immediate",
                    tag="challenge_page_confirmed",
                    reason=f"挑战页已确认打开（读到「{challenge_text}」）",
                ),
                State.FIND_COOP,
            )
        self._entry_misses += 1
        # 弹窗标题仍可见=选卡点击未生效（弹窗没关、页面没切）：连续多帧
        # 如此就回选卡步骤重读卡名并重点。弹窗已关闭时不回退——此时可能
        # 已在挑战页（按钮文字漏读），重点选卡会点到挑战页空白处
        if (
            observation.flag("mode_select_visible")
            and self._entry_misses >= _ENTRY_RECLICK_MISSES
        ):
            if self._entry_reclicks >= self._entry_max_reclicks:
                logger.error(
                    "frame=%d 重点选卡 %d 次后仍未读到「%s」，保守停止",
                    observation.frame_id,
                    self._entry_reclicks,
                    challenge_text or "挑战文字",
                )
                return None
            self._entry_reclicks += 1
            self._entry_misses = 0
            self._recruit_step = _RecruitStep.SELECT_MODE_CARD
            self._start_entry_wait()
            return self._entry_wait_action(
                "challenge_page_pending",
                "选卡未生效（弹窗仍打开），回到选卡步骤重点",
            )
        if self._entry_deadline_exceeded(
            observation, f"点击模式卡后未读到「{challenge_text or '挑战文字'}」"
        ):
            return None
        return self._entry_wait_action(
            "challenge_page_pending",
            "等待挑战页打开（按钮文字未命中）",
        )

    def _action_close_weekly_buff(
        self,
        observation: Observation,
    ) -> tuple[Action, State] | None:
        """关闭挑战页每周弹窗（保卫宠物=本周Buff，每天首次进入弹出）。

        OCR 读按钮文字（模式档案 weekly_buff_text 子串匹配，「知道了」实测
        被读成「知道」故配两字），命中即点文字框中心（开环，同选卡/难度
        勾选策略）。弹窗可能晚于页面到达弹出：连续 _WEEKLY_BUFF_GRACE_MISSES
        帧读不到才放行进招募；点击后弹窗关闭同样经宽限帧确认。持续命中
        （点击未生效）或持续读不到都受入口等待上限约束，超时保守停止。
        """
        buff_text = self._mode.entry.weekly_buff_text if self._mode else None
        if not buff_text:
            # 档案未配置（不应到达）：直接放行，保持旧行为
            self._recruit_step = _RecruitStep.OPEN_CHALLENGE_CHAT
            return (
                Action(
                    kind="wait",
                    duration=0.0,
                    verification="immediate",
                    tag="weekly_buff_absent",
                    reason="模式档案未配置每周弹窗识别，跳过关闭步骤",
                ),
                State.FIND_COOP,
            )
        if self._entry_deadline_exceeded(observation, "挑战页每周弹窗持续可见（点击未生效）"):
            return None
        rows = [
            row
            for row in (observation.raw_data.get("weekly_buff_rows") or [])
            if buff_text in row.text
        ]
        if rows:
            best = max(rows, key=lambda row: row.confidence)
            self._weekly_buff_misses = 0
            logger.info(
                "OCR 定位「%s」弹窗按钮 conf=%.2f client=%s",
                buff_text,
                best.confidence,
                best.position,
            )
            return (
                Action(
                    kind="click",
                    target=best.position,
                    duration=0.08,
                    verification="immediate",
                    tag="close_weekly_buff",
                    reason=f"关闭挑战页每周弹窗：点击「{buff_text}」",
                ),
                State.FIND_COOP,
            )
        self._weekly_buff_misses += 1
        if self._weekly_buff_misses >= _WEEKLY_BUFF_GRACE_MISSES:
            logger.info(
                "frame=%d 挑战页每周弹窗不可见（连续 %d 帧），进招募",
                observation.frame_id,
                self._weekly_buff_misses,
            )
            self._recruit_step = _RecruitStep.OPEN_CHALLENGE_CHAT
            return (
                Action(
                    kind="wait",
                    duration=0.0,
                    verification="immediate",
                    tag="weekly_buff_absent",
                    reason="挑战页每周弹窗不可见（宽限帧已过）",
                ),
                State.FIND_COOP,
            )
        return self._entry_wait_action(
            "weekly_buff_pending",
            "等待挑战页每周弹窗识别（弹窗可能延迟弹出）",
        )

    def _action_confirm_chat_open(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        """聊天点击生效确认：页签行读到本模式页签才算进入招募。

        2026-09-16 实机：结算返回后挑战页盖着金币道具说明卡，固定坐标的
        聊天点击被弹窗吃掉（只关掉弹窗、页面未切换），页签 OCR 永远读不
        到而被无进展预算保守停止。2026-09-17 实机修正生效判据：不能用
        「挑战页锚点（challenge_text）消失」——招募频道页底部同样有
        「开始挑战」按钮，锚点两页皆可读，导致确认永不推进、重点全部
        点空直到超时。改为 OCR 页签行（与 SELECT_MODE_TAB 同一识别区）：
        连续 ``_CHAT_OPEN_ADVANCE_MISSES`` 帧读到本模式页签 = 已进入招募
        （挑战页该区域读到的是「拥有：」等无关文字）；连续
        ``_CHAT_OPEN_RECLICK_MISSES`` 帧读不到 = 点击未生效（大概率被
        弹窗遮挡），重点聊天按钮——重点会顺带关掉挡路的道具卡，预算
        ``chat_open_max_reclicks`` 用尽或入口等待超时才保守停止。页签单帧
        漏读由连续帧计数吸收；档案未配置 ``mode_tab_text`` 时无法验证，
        保持旧行为直接进页签步骤。
        """
        tab_text = self._mode.entry.mode_tab_text if self._mode else None
        if not tab_text:
            # 档案未配置页签文字：页签行验证无从谈起（锚点又不能区分两页），
            # 保持旧行为直接进页签步骤
            self._recruit_step = _RecruitStep.SELECT_MODE_TAB
            return (
                Action(
                    kind="wait",
                    duration=0.0,
                    verification="immediate",
                    tag="chat_open_unverified",
                    reason="模式档案未配置页签文字，跳过聊天点击确认",
                ),
                State.FIND_COOP,
            )
        rows = [
            row
            for row in (observation.raw_data.get("mode_tab_rows") or [])
            if tab_text in row.text
        ]
        if rows:
            self._chat_open_tab_misses = 0
            self._chat_open_tab_hits += 1
            if self._chat_open_tab_hits < _CHAT_OPEN_ADVANCE_MISSES:
                return self._entry_wait_action(
                    "chat_open_pending",
                    "页签行已读到本模式页签，等待下一帧确认",
                )
            logger.info(
                "frame=%d 聊天点击已生效（连续 %d 帧读到「%s」页签），进页签选择",
                observation.frame_id,
                self._chat_open_tab_hits,
                tab_text,
            )
            self._recruit_step = _RecruitStep.SELECT_MODE_TAB
            return (
                Action(
                    kind="wait",
                    duration=0.0,
                    verification="immediate",
                    tag="chat_open_confirmed",
                    reason="聊天点击生效（页签行已可读），进页签选择",
                ),
                State.FIND_COOP,
            )
        self._chat_open_tab_hits = 0
        self._chat_open_tab_misses += 1
        if self._entry_deadline_exceeded(observation, "聊天点击后页签行持续不可读"):
            return None
        if self._chat_open_tab_misses < _CHAT_OPEN_RECLICK_MISSES:
            return self._entry_wait_action(
                "chat_open_pending",
                "聊天点击后未读到页签，等待页面切换",
            )
        if self._chat_open_reclicks >= self._chat_open_max_reclicks:
            logger.error(
                "frame=%d 重点聊天 %d 次后页签行仍读不到「%s」，保守停止",
                observation.frame_id,
                self._chat_open_reclicks,
                tab_text,
            )
            return None
        self._chat_open_reclicks += 1
        self._chat_open_tab_misses = 0
        chat_hotspot = self._mode.mode_chat_hotspot if self._mode else "coop_chat"
        logger.warning(
            "frame=%d 聊天点击未生效（页签行读不到，可能被弹窗遮挡），重点聊天（第 %d/%d 次）",
            observation.frame_id,
            self._chat_open_reclicks,
            self._chat_open_max_reclicks,
        )
        return self._hotspot_click(
            chat_hotspot,
            "chat_reclick",
            f"聊天点击未生效，重点聊天（第 {self._chat_open_reclicks} 次）",
            State.FIND_COOP,
            window_ctx,
        )

    def _action_recruit_entry(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        """按代码中确认的首次/下一局入口步骤生成一个安全动作。"""
        if self._recruit_step == _RecruitStep.DETECT_ENTRY_PAGE:
            # 退队后的落点未实机确认（可能首页、可能合作页面）：先识别首页
            # 标志再选入口，两条路径都不重复勾选难度（本会话已选中）。
            # 首页也改走领地入口（2026-09-09：不再从首页点聊天进招募）
            if observation.flag("home_page_visible"):
                logger.info("退队后识别到游戏首页，从领地入口进入招募")
                self._recruit_step = _RecruitStep.OPEN_TERRITORY
                self._entry_reclicks = 0
                self._entry_misses = 0
                self._start_entry_wait()
            else:
                logger.info("退队后未识别到首页标志，按合作页面处理")
                self._recruit_step = _RecruitStep.OPEN_COOP_CHAT
            return (
                Action(
                    kind="wait",
                    duration=0.3,
                    verification="immediate",
                    tag="entry_page_detected",
                    reason="退队后确认所在页面，选择招募入口",
                ),
                State.FIND_COOP,
            )
        if self._recruit_step == _RecruitStep.OPEN_TERRITORY:
            return self._action_open_territory(observation, window_ctx)
        if self._recruit_step == _RecruitStep.CONFIRM_TERRITORY_OPEN:
            return self._action_confirm_territory_open(observation, window_ctx)
        if self._recruit_step == _RecruitStep.OPEN_MODE_SELECT:
            return self._hotspot_click(
                "open_mode_select",
                "open_mode_select",
                "领地入口：点击「合作模式」进入按钮",
                State.FIND_COOP,
                window_ctx,
            )
        if self._recruit_step == _RecruitStep.CONFIRM_MODE_SELECT_OPEN:
            return self._action_confirm_mode_select_open(observation)
        if self._recruit_step == _RecruitStep.SELECT_MODE_CARD:
            return self._action_select_mode_card(observation)
        if self._recruit_step == _RecruitStep.CONFIRM_CHALLENGE_PAGE:
            return self._action_confirm_challenge_page(observation)
        if self._recruit_step == _RecruitStep.CLOSE_WEEKLY_BUFF:
            return self._action_close_weekly_buff(observation)
        if self._recruit_step == _RecruitStep.CONFIRM_CHAT_OPEN:
            return self._action_confirm_chat_open(observation, window_ctx)
        if self._recruit_step == _RecruitStep.OPEN_CHALLENGE_CHAT:
            chat_hotspot = self._mode.mode_chat_hotspot if self._mode else "coop_chat"
            return self._hotspot_click(
                chat_hotspot,
                "open_challenge_chat",
                "挑战页进入招募：点击模式页面聊天",
                State.FIND_COOP,
                window_ctx,
            )
        if self._recruit_step == _RecruitStep.SELECT_MODE_TAB:
            tab_text = self._mode.entry.mode_tab_text if self._mode else None
            if not tab_text:
                tab_name = self._mode.entry.mode_tab_hotspot if self._mode else None
                if not tab_name:
                    # 未配置页签：直接进入页签后的目标步骤（旧行为）
                    self._recruit_step = self._after_tab_step
                    return self._action_recruit_entry(observation, window_ctx)
                return self._hotspot_click(
                    tab_name,
                    "open_mode_tab",
                    "进入招募：点击模式页签",
                    State.FIND_COOP,
                    window_ctx,
                )
            return self._action_select_mode_tab_ocr(observation, tab_text)
        if self._recruit_step == _RecruitStep.OPEN_DIFFICULTY_DIALOG:
            return self._hotspot_click(
                "open_difficulty_dialog",
                "open_difficulty_dialog",
                "首次进入招募：打开难度弹窗",
                State.FIND_COOP,
                window_ctx,
            )
        if self._recruit_step == _RecruitStep.CONFIRM_DIFFICULTY_OPEN:
            return self._action_confirm_difficulty_open(observation, window_ctx)
        if self._recruit_step == _RecruitStep.SELECT_DIFFICULTIES:
            # OCR 模式（保卫宠物）：按行文字勾选，不需要窗口换算的滚动参数
            if self._mode is not None and self._mode.entry.difficulty.kind == "ocr":
                return self._action_select_pet_difficulties(observation)
            return self._action_select_difficulties(observation, window_ctx)
        if self._recruit_step == _RecruitStep.CLOSE_DIFFICULTY_DIALOG:
            return self._action_close_difficulty_dialog(
                observation,
                window_ctx,
                click_tag="close_difficulty_dialog",
                closed_tag="difficulty_dialog_closed",
                reason_prefix="首次进入招募",
            )
        if self._recruit_step == _RecruitStep.OPEN_COOP_CHAT:
            chat_hotspot = self._mode.mode_chat_hotspot if self._mode else "coop_chat"
            return self._hotspot_click(
                chat_hotspot,
                "open_coop_chat",
                "下一局进入招募：点击模式页面聊天",
                State.FIND_COOP,
                window_ctx,
            )
        if self._recruit_step == _RecruitStep.OPEN_REFRESH_DIFFICULTY_DIALOG:
            return self._hotspot_click(
                "open_difficulty_dialog",
                "open_refresh_difficulty_dialog",
                "下一局刷新邀请：打开难度弹窗",
                State.FIND_COOP,
                window_ctx,
            )
        if self._recruit_step == _RecruitStep.CLOSE_REFRESH_DIFFICULTY_DIALOG:
            return self._action_close_difficulty_dialog(
                observation,
                window_ctx,
                click_tag="close_refresh_difficulty_dialog",
                closed_tag="refresh_dialog_closed",
                reason_prefix="下一局刷新邀请",
            )
        return None

    def _action_confirm_difficulty_open(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        """正向确认难度弹窗真的打开，再进入勾选/关闭步骤（2026-08-21 实机定案）。

        弹窗打开快慢不定：点击打开按钮后先固定等待 ``difficulty_open_settle_seconds``
        吸收时序差异，再每 ``difficulty_poll_interval_seconds`` 识别一次难度弹窗
        标识（合作=【合作模式】标题图；保卫宠物=「选择难度」收起栏被面板盖住），
        连续 ``difficulty_open_confirm_hits`` 次可见才算打开（弹出动画帧
        凑不够连续命中）。连续 ``difficulty_open_fail_misses`` 次不可见说明本次
        点击未生效：重点打开按钮（最多 ``difficulty_open_max_reclicks`` 次），
        重试用尽仍打不开则保守停止——绝不带着「没打开」的假设进入后续步骤
        （打开按钮位置落在面板列表区内，弹窗开着时重点会误点难度行、改动勾选）。
        """
        cfg = self._run_config
        if not self._difficulty_open_settled:
            return (
                Action(
                    kind="wait",
                    duration=cfg.difficulty_open_settle_seconds,
                    verification="immediate",
                    tag="difficulty_open_settle",
                    reason="等待难度弹窗打开（快慢不定，先固定等待）",
                ),
                State.FIND_COOP,
            )
        if self._difficulty_open_hits >= cfg.difficulty_open_confirm_hits:
            return (
                Action(
                    kind="wait",
                    duration=0.0,
                    verification="immediate",
                    tag="difficulty_open_confirmed",
                    reason="难度弹窗已确认打开（弹窗标识连续命中）",
                ),
                State.FIND_COOP,
            )
        if self._difficulty_open_misses >= cfg.difficulty_open_fail_misses:
            if self._difficulty_open_reclicks >= cfg.difficulty_open_max_reclicks:
                logger.error(
                    "难度弹窗重点 %d 次后仍识别不到弹窗标识，保守停止 frame=%d",
                    self._difficulty_open_reclicks,
                    observation.frame_id,
                )
                return None
            return self._hotspot_click(
                "open_difficulty_dialog",
                "difficulty_open_retry",
                f"难度弹窗未打开，重点打开按钮"
                f"（{self._difficulty_open_reclicks + 1}/{cfg.difficulty_open_max_reclicks}）",
                State.FIND_COOP,
                window_ctx,
            )
        if observation.flag("difficulty_dialog_visible"):
            return (
                Action(
                    kind="wait",
                    duration=cfg.difficulty_poll_interval_seconds,
                    verification="immediate",
                    tag="difficulty_open_hit",
                    reason=f"难度弹窗标识可见（连续 "
                    f"{self._difficulty_open_hits + 1}/{cfg.difficulty_open_confirm_hits}）",
                ),
                State.FIND_COOP,
            )
        return (
            Action(
                kind="wait",
                duration=cfg.difficulty_poll_interval_seconds,
                verification="immediate",
                tag="difficulty_open_miss",
                    reason=f"难度弹窗标识不可见（连续 "
                f"{self._difficulty_open_misses + 1}/{cfg.difficulty_open_fail_misses}）",
            ),
            State.FIND_COOP,
        )

    def _action_close_difficulty_dialog(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
        *,
        click_tag: str,
        closed_tag: str,
        reason_prefix: str,
    ) -> tuple[Action, State] | None:
        """关闭难度弹窗，并以难度弹窗标识判定弹窗开关（合作=he_zuo_mo_shi 标题图；
        保卫宠物=「选择难度」收起栏，不可见=打开）。

        打开确认步骤已保证弹窗真的打开过，这里统一按「连续
        ``difficulty_close_confirm_frames`` 次识别不到弹窗标识」判定关闭（间隔
        ``difficulty_poll_interval_seconds``）：单帧漏检不再致命；收起动画期间
        标识可见只触发再次点击（实机确认：面板已收起时再点招募按钮无反应，
        重复点击无副作用）。标识可见 = 没关住，重点关闭按钮，最多
        ``difficulty_close_max_attempts`` 次，仍可见则保守停止——绝不带着未关闭
        的弹窗进入抢合作（弹窗挡住 join_coop，连点还会落在难度行上误改勾选）。
        """
        cfg = self._run_config
        if observation.flag("difficulty_dialog_visible"):
            if self._difficulty_close_attempts >= cfg.difficulty_close_max_attempts:
                logger.error(
                    "难度弹窗关闭点击 %d 次后弹窗标识仍可见，保守停止 frame=%d"
                    "（合作=标题图仍在；保卫宠物=「选择难度」收起栏未出现）",
                    self._difficulty_close_attempts,
                    observation.frame_id,
                )
                return None
            return self._hotspot_click(
                "close_difficulty_dialog",
                click_tag,
                f"{reason_prefix}：难度弹窗仍打开，再次点击关闭"
                f"（{self._difficulty_close_attempts + 1}/{cfg.difficulty_close_max_attempts}）",
                State.FIND_COOP,
                window_ctx,
            )
        if self._difficulty_close_streak < cfg.difficulty_close_confirm_frames:
            return (
                Action(
                    kind="wait",
                    duration=cfg.difficulty_poll_interval_seconds,
                    verification="immediate",
                    tag="difficulty_close_miss_check",
                    reason=f"{reason_prefix}：等待难度弹窗关闭稳定"
                    f"（{self._difficulty_close_streak + 1}/{cfg.difficulty_close_confirm_frames}）",
                ),
                State.FIND_COOP,
            )
        return (
            Action(
                kind="wait",
                duration=0.0,
                verification="immediate",
                tag=closed_tag,
                reason=f"{reason_prefix}：难度弹窗已确认关闭",
            ),
            State.FIND_COOP,
        )

    def _action_select_difficulties(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        """按从小到大顺序点击目标难度；目标不可见时按比例滚动列表。

        面板每次打开都定位在普通难度区顶部（2026-08-19 游戏规则），滚入
        彩虹区后最先出现的是编号最小的目标，因此从小到大点击是单程顺序：
        顺着向下滚动一路点完，不需要回滚。识别为空表示尚未滚入彩虹区，
        此时向更大难度滚动（从下往上拉），而不是等待或跳过。
        """
        if not self._remaining_difficulties:
            self._log_difficulty_summary()
            if self._difficulty_selection_only:
                return (
                    Action(
                        kind="wait",
                        duration=0.0,
                        verification="immediate",
                        tag="difficulty_selection_complete",
                        reason="目标合作难度已全部点击或跳过",
                    ),
                    State.COMPLETED,
                )
            return None

        target_level = min(self._remaining_difficulties)
        visible = {item.level: item for item in observation.difficulty_candidates}
        target = visible.get(target_level)
        scroll_start = self._hotspot_point("difficulty_scroll_start", window_ctx)
        scroll_end = self._hotspot_point("difficulty_scroll_end", window_ctx)
        if target is not None:
            # 开环勾选（2026-08-19 实机定案）：识别难度文字 → 点击文字中心 →
            # 推进，不识别勾选框状态。合作/宠物等邀请弹窗只遮挡勾选框、
            # 从不遮挡行文字，点击几何上永远落在行上；代价是已选过的会话
            # 必须用 --skip-difficulty-selection 跳过（再次点击会取消勾选）。
            return (
                Action(
                    kind="click",
                    target=target.position,
                    duration=0.08,
                    verification="immediate",
                    tag=f"select_difficulty:{target_level}",
                    reason=f"选择合作难度 {target_level}",
                ),
                State.FIND_COOP,
            )

        visible_levels = sorted(visible, reverse=True)
        if not visible_levels:
            # 2026-08-19 游戏规则更新：面板每次打开都定位在普通难度区，
            # 普通行（合作模式-第N层）不匹配彩虹模板，识别为空表示尚未
            # 滚入彩虹区——向更大难度滚动（从下往上拉），而不是等待或跳过。
            if scroll_start is None or scroll_end is None:
                logger.warning("难度滚动 hotspot 缺失，跳过 target=%d", target_level)
                return self._difficulty_skip_action(target_level, "难度滚动 hotspot 缺失")
            if self._difficulty_scroll_count >= self._run_config.difficulty_max_scrolls:
                logger.warning(
                    "目标难度 %d 滚动 %d 次仍未识别到彩虹候选，跳过",
                    target_level,
                    self._difficulty_scroll_count,
                )
                return self._difficulty_skip_action(
                    target_level,
                    f"滚动 {self._difficulty_scroll_count} 次仍未识别到候选",
                )
            start, end = scroll_end, scroll_start
            direction = "larger"
            reason = "列表仍在普通难度区（无彩虹候选），向更大难度滚动"
        else:
            self._difficulty_saw_candidates = True
            if self._difficulty_scroll_count >= self._run_config.difficulty_max_scrolls:
                # 滚动预算用尽仍未识别到该目标：记录后跳过，不阻断正式流程。
                # 难度选择是可选集合，漏掉单个目标不影响后续抢合作。
                logger.warning(
                    "目标难度 %d 滚动 %d 次仍未识别到，跳过 visible=%s",
                    target_level,
                    self._difficulty_scroll_count,
                    visible_levels,
                )
                return self._difficulty_skip_action(
                    target_level,
                    f"滚动 {self._difficulty_scroll_count} 次仍未识别到",
                )

            if scroll_start is None or scroll_end is None:
                logger.warning("难度滚动 hotspot 缺失，跳过 target=%d", target_level)
                return self._difficulty_skip_action(target_level, "难度滚动 hotspot 缺失")

            if max(visible_levels) < target_level:
                # 可见难度全部小于目标：目标在列表更深处（下方），向更大难度滚动。
                start, end = scroll_end, scroll_start
                direction = "larger"
                reason = f"当前难度均低于 {target_level}，向较大难度滚动"
            else:
                # 其余情形（全部高于目标 / 跨过目标但未命中）一律向较小滚动。
                # 彩虹编号在列表内连续，跨过目标却未命中说明目标在屏上被
                # 误识别，回移一点重新识别；滚动预算用尽则由上方分支跳过。
                start, end = scroll_start, scroll_end
                direction = "smaller"
                reason = f"向较小难度滚动以定位 {target_level} visible={visible_levels}"

        return (
            Action(
                kind="drag",
                target=start,
                end=end,
                duration=self._run_config.difficulty_scroll_duration,
                verification="next_frame",
                tag=f"scroll_difficulties:{direction}",
                reason=reason,
            ),
            State.FIND_COOP,
        )

    def _action_select_pet_difficulties(
        self, observation: Observation
    ) -> tuple[Action, State] | None:
        """保卫宠物难度勾选（OCR 开环，2026-09-06 定案）。

        行文字来自 CoopPerception 的 pet_difficulty_rows（已按
        pet_difficulty_recognition.min_confidence 过滤）；用共享的
        match_targets 做子串匹配（targets 顺序即优先顺序，每个目标取
        置信度最高行），命中即点击行文字中心，点击发送即计为已选
        （同合作难度的开环哲学，不识别勾选框）；保留置信度二次防线。
        所有目标不可见时只等待计数，连续 difficulty_open_fail_misses 次
        仍不可见则保守停止——OCR 模式无滚动，绝不盲点。
        """
        if not self._remaining_pet_targets:
            self._log_pet_difficulty_summary()
            if self._difficulty_selection_only:
                return (
                    Action(
                        kind="wait",
                        duration=0.0,
                        verification="immediate",
                        tag="pet_difficulty_selection_complete",
                        reason="保卫宠物目标难度已全部点击",
                    ),
                    State.COMPLETED,
                )
            return None

        rows = observation.raw_data.get("pet_difficulty_rows") or []
        for target, best in match_targets(rows, self._remaining_pet_targets):
            if best.confidence < self._pet_min_confidence:
                continue
            logger.info(
                "frame=%d 保卫宠物难度 %s 命中行 %r 置信度 %.2f，点击文字中心",
                observation.frame_id,
                target,
                best.text,
                best.confidence,
            )
            return (
                Action(
                    kind="click",
                    target=best.position,
                    duration=0.08,
                    verification="immediate",
                    tag=f"select_pet_difficulty:{target}",
                    reason=f"选择保卫宠物难度 {target}",
                ),
                State.FIND_COOP,
            )

        if self._pet_difficulty_misses >= self._run_config.difficulty_open_fail_misses:
            raise RuntimeError(
                f"保卫宠物难度行持续不可见（{self._pet_difficulty_misses} 次），保守停止；"
                "请确认难度弹窗已打开且 pet_difficulty_list ROI 标定正确"
            )
        return (
            Action(
                kind="wait",
                duration=0.3,
                verification="immediate",
                tag="pet_difficulty_miss",
                reason="保卫宠物难度行不可见，等待重试",
            ),
            State.FIND_COOP,
        )

    def _log_pet_difficulty_summary(self) -> None:
        """保卫宠物难度选择汇总（仅输出一次）。"""
        if self._difficulty_summary_logged:
            return
        self._difficulty_summary_logged = True
        logger.info(
            "保卫宠物难度选择汇总：选中 %s",
            self._selected_pet_targets,
        )

    def _complete_difficulty_target(
        self, level: int, *, selected: bool, reset_scrolls: bool = True
    ) -> None:
        """一个目标难度处理完成（点击已选/跳过）。

        移出待选集合并记账；待选清空时输出汇总并推进到关闭弹窗步骤。
        """
        self._remaining_difficulties.discard(level)
        if selected:
            self._selected_difficulties.add(level)
        else:
            self._skipped_difficulties.add(level)
        if reset_scrolls:
            self._difficulty_scroll_count = 0
        self._difficulty_saw_candidates = False
        if not self._remaining_difficulties:
            self._log_difficulty_summary()
            if not self._difficulty_selection_only:
                self._recruit_step = _RecruitStep.CLOSE_DIFFICULTY_DIALOG
                self._difficulty_close_attempts = 0
                self._difficulty_close_streak = 0

    @staticmethod
    def _difficulty_skip_action(level: int, reason: str) -> tuple[Action, State]:
        """记录无法识别/定位的难度并继续，不阻断后续合作流程。"""
        return (
            Action(
                kind="wait",
                duration=0.0,
                verification="immediate",
                tag=f"difficulty_skip:{level}",
                reason=f"跳过难度 {level}: {reason}",
            ),
            State.FIND_COOP,
        )

    def _log_difficulty_summary(self) -> None:
        """难度选择结束时汇总输出选中与未识别跳过的难度（仅输出一次）。"""
        if self._difficulty_summary_logged:
            return
        self._difficulty_summary_logged = True
        selected = sorted(self._selected_difficulties, reverse=True)
        skipped = sorted(self._skipped_difficulties, reverse=True)
        logger.info(
            "难度选择汇总：选中 %s，未识别跳过 %s",
            selected,
            skipped,
        )

    def _action_ready(self, observation: Observation) -> tuple[Action, State] | None:
        match = self._flag_match(observation, "ready_button_match")
        if match is None:
            return None
        return (
            Action(
                kind="click",
                target=match.position,
                duration=0.08,
                verification="next_frame",
                tag="ready_match",
                reason="识别并点击准备按钮",
            ),
            State.SELECT_OPENING_SKILLS,
        )

    # ------------------------------------------------------------------
    # 技能选择
    # ------------------------------------------------------------------

    def _action_opening_skills(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        # 等待开局期间识别到首页标志：可能本局被取消/被踢回首页（实机确认
        # 房主可踢出已加入玩家，被踢后回到首页、对局不开始）。连续多帧确认后
        # 放弃本局、回到招募入口重新抢合作，而不是无限等待对局界面。
        if observation.flag("home_page_visible"):
            if self._home_return_count + 1 >= self._run_config.home_return_confirm_frames:
                logger.warning(
                    "frame=%d 连续 %d 帧识别到首页标志，本局被取消/被踢回首页，重新进入招募",
                    observation.frame_id,
                    self._home_return_count + 1,
                )
                return (
                    Action(
                        kind="wait",
                        duration=0.0,
                        verification="immediate",
                        tag="kick_reentry_home",
                        reason="被踢回首页，本局未开始，重新进入招募",
                    ),
                    State.FIND_COOP,
                )
            return (
                Action(
                    kind="wait",
                    duration=0.5,
                    verification="immediate",
                    tag="home_return_check",
                    reason="识别到首页标志，确认是否被踢回首页"
                    f"（{self._home_return_count + 1}/{self._run_config.home_return_confirm_frames}）",
                ),
                State.SELECT_OPENING_SKILLS,
            )
        self._home_return_count = 0

        if not self._opening_loaded:
            return (
                Action(
                    kind="wait",
                    duration=self._run_config.ready_wait_seconds,
                    verification="immediate",
                    tag="opening_load_wait",
                    reason="准备后等待对局界面稳定",
                ),
                State.SELECT_OPENING_SKILLS,
            )

        # 进入游戏后按「先选技能、再召唤」识别两种开局。
        # 【选技能】可见或已识别到技能卡 = 处于技能选择页面：开局技能是免费的，进入对局后
        # 自动弹出该页面（不点「选择技能」按钮、不花金币），且页面会遮住召唤按钮，必须选完
        # 才能召唤。开局页面与局内点「选择技能」按钮（花金币）弹出的页面是同一个。
        # 天使英雄的开局技能页是特例：不出现【选技能】图、也没有主C技能图标，只出现
        # 天使开局标识（tian_shi_kai_ju）——命中即在技能卡区域随机选一张。
        if (
            observation.flag("select_skill_button_visible")
            or observation.skill_candidates
            or observation.flag("tian_shi_kai_ju_visible")
        ):
            if self._opening_clicks_blocked:
                # 技能卡点击多次未生效（Runner 已重试并放弃）：不再盲点，等待对局
                # 自然推进——页面关闭、召唤出现或结算窗口由上层状态优先级接管
                return (
                    Action(
                        kind="wait",
                        duration=1.0,
                        verification="immediate",
                        tag="opening_skill_blocked_wait",
                        reason="开局技能点击多次未生效，等待对局推进",
                    ),
                    State.SELECT_OPENING_SKILLS,
                )
            if observation.skill_candidates:
                # 选一张主C技能卡；选后留在本状态，等【选技能】消失、【召唤】出现再进召唤
                return self._choose_skill_action(
                    observation.skill_candidates,
                    "opening_skill_candidate",
                    State.SELECT_OPENING_SKILLS,
                )
            priority_action = self._skill_priority_action(
                observation, "opening_skill_priority", State.SELECT_OPENING_SKILLS
            )
            if priority_action is not None:
                return priority_action
            if observation.flag("tian_shi_kai_ju_visible"):
                return self._skill_fallback_action(
                    "opening_angel_skill_fallback",
                    State.SELECT_OPENING_SKILLS,
                    window_ctx,
                    reason="天使开局技能页，随机选一个技能",
                )
            # 页面打开但未识别到主C技能图标（图标在动可能单帧漏检）→ 多帧重试；
            # 识别到队友图标则证明页面已稳定且本组无主C技能卡，无需等满识别帧，
            # 但仍要求页面已出现 skill_fallback_settle_frames 帧——页面刚弹出时
            # 图标可能先于卡片渲染出来，首帧即点会点空
            if self._opening_empty_checks >= self._run_config.skill_recognition_frames or (
                observation.flag("teammate_skill_visible")
                and self._opening_empty_checks >= self._run_config.skill_fallback_settle_frames
            ):
                return self._skill_fallback_action(
                    "opening_skill_fallback",
                    State.SELECT_OPENING_SKILLS,
                    window_ctx,
                )
            return (
                Action(
                    kind="wait",
                    duration=0.5,
                    verification="immediate",
                    tag="opening_skill_empty_check",
                    reason="开局技能页面已打开，等待主C技能图标出现",
                ),
                State.SELECT_OPENING_SKILLS,
            )

        # 【选技能】消失且【召唤】可见 → 开局阶段结束（本局无开局技能或已选完），进入召唤。
        # 每次选完一张卡后页面会先关闭、下一组技能卡再弹出，这个间隙里召唤按钮会
        # 短暂露出；单帧「页面消失」不能判定选完，必须连续 opening_exit_confirm_frames
        # 帧都确认页面不在才算结束。已实机确认开局技能最多 opening_skill_max_selections
        # 次（不会再多），选满后召唤按钮一出现即可直接结束，无需再等退出确认帧。
        if observation.flag("summon_button_visible"):
            # 开局页面已关闭：解除点击阻塞（若之前因连续失败放弃过选择）
            self._opening_clicks_blocked = False
            selections_done = (
                self._opening_skill_selections >= self._run_config.opening_skill_max_selections
            )
            if not selections_done and (
                self._opening_exit_empty_count < self._run_config.opening_exit_confirm_frames
            ):
                return (
                    Action(
                        kind="wait",
                        duration=0.5,
                        verification="immediate",
                        tag="opening_exit_confirm_check",
                        reason="技能页面暂时关闭，确认是否还有下一组开局技能"
                        f"（{self._opening_exit_empty_count + 1}/"
                        f"{self._run_config.opening_exit_confirm_frames}）",
                    ),
                    State.SELECT_OPENING_SKILLS,
                )
            return (
                Action(
                    kind="wait",
                    duration=0.0,
                    verification="immediate",
                    tag="opening_complete",
                    reason="技能页面已连续关闭，进入召唤阶段",
                ),
                State.BUILD_MAIN_C,
            )

        # 两者都没识别到：对局界面尚未稳定，继续等待；同时解除点击阻塞，
        # 待下一组技能页弹出时再正常尝试选择。
        self._opening_clicks_blocked = False
        # 组队大厅（点完准备、对局未开始）：识别到【退队】且房主超过
        # leave_team_after_seconds 仍未开始，主动退队重新抢合作，不干等
        leave_match = self._flag_match(observation, "leave_team_match")
        if leave_match is not None:
            waited = (
                0.0 if self._match_started_at is None else self._clock() - self._match_started_at
            )
            if waited > self._run_config.leave_team_after_seconds:
                logger.warning(
                    "frame=%d 已点准备等待 %.0fs 房主未开始对局，点击退队重新抢合作",
                    observation.frame_id,
                    waited,
                )
                return (
                    Action(
                        kind="click",
                        target=leave_match.position,
                        duration=0.08,
                        verification="next_frame",
                        tag="leave_team",
                        reason=f"房主 {waited:.0f}s 未开始对局，退队重新抢合作",
                    ),
                    State.FIND_COOP,
                )
            return (
                Action(
                    kind="wait",
                    duration=0.5,
                    verification="immediate",
                    tag="lobby_wait",
                    reason="已点准备，等待房主开始对局",
                ),
                State.SELECT_OPENING_SKILLS,
            )
        # 等待超过 match_start_timeout_seconds 仍无对局界面则按本局未开始处理
        # （如被踢后落在非首页界面），重新进入招募
        if self._match_started_at is not None:
            waited = self._clock() - self._match_started_at
            if waited > self._run_config.match_start_timeout_seconds:
                logger.warning(
                    "frame=%d 点准备后 %.0fs 仍未见对局界面，按本局未开始处理，重新进入招募",
                    observation.frame_id,
                    waited,
                )
                return (
                    Action(
                        kind="wait",
                        duration=0.0,
                        verification="immediate",
                        tag="match_start_timeout",
                        reason="等待对局界面超时，本局未开始，重新进入招募",
                    ),
                    State.FIND_COOP,
                )
        # 长时间等待仍无任何开局界面标志：多为房主迟迟不开始（技能页/召唤/
        # 退队按钮都不可见）。周期性输出诊断，提示当前将等到超时为止
        if self._opening_wait_count % 10 == 1:
            logger.warning(
                "frame=%d 等待开局已 %d 帧未识别到技能页/召唤/退队标识"
                "（多半是房主未开始对局；%.0fs 后将按未开始处理）",
                observation.frame_id,
                self._opening_wait_count,
                self._run_config.match_start_timeout_seconds,
            )
        return (
            Action(
                kind="wait",
                duration=0.5,
                verification="immediate",
                tag="opening_interface_wait",
                reason="等待对局界面稳定，识别开局类型",
            ),
            State.SELECT_OPENING_SKILLS,
        )

    def _action_main_c_skills(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        # 局内技能选择总次数达到档案上限：不再花金币选技能，等对局结束进结算
        # （返回按钮由状态优先级接管）
        # 赠送技能卡片优先关闭：同系列技能选满 3 个触发（开局 3 技能不触发），
        # 点击任意位置关闭，处理方式同 Boss 奖励弹窗
        bonus_match = self._flag_match(observation, "zeng_song_ji_neng_match")
        if observation.flag("zeng_song_ji_neng_visible") and bonus_match is not None:
            return (
                Action(
                    kind="click",
                    target=bonus_match.position,
                    duration=0.08,
                    verification="next_frame",
                    tag="close_bonus_popup",
                    reason="同系列技能选满触发的赠送技能卡片，点击关闭",
                ),
                State.SELECT_MAIN_C_SKILLS,
            )
        cap = self._strategy.skill_selection_cap
        if cap > 0 and self._main_skill_selections_total >= cap:
            return (
                Action(
                    kind="wait",
                    duration=1.0,
                    verification="immediate",
                    tag="skill_cap_reached",
                    reason=f"已达局内技能选择上限 {cap} 次，等待对局结束",
                ),
                State.SELECT_MAIN_C_SKILLS,
            )

        # 主C档案配置了数量回补：局内选满 N 次技能后回培养阶段把主C数量
        # 补到目标以上（棋盘数量判断在 BUILD_MAIN_C，该状态才识别棋盘）。
        # 数量已确认达标的局不再复查（2026-09-07 用户策略）：选技能期间
        # 数量不会减少，补够一次后一直选技能到对局结束
        if not self._topup_done and self._strategy.should_topup(self._main_skill_selections):
            logger.info(
                "已选 %d 次主C技能，回培养阶段检查/补充主C数量",
                self._main_skill_selections,
            )
            return (
                Action(
                    kind="wait",
                    duration=0.2,
                    verification="immediate",
                    tag="main_c_topup",
                    reason="回培养阶段补充主C数量",
                ),
                State.BUILD_MAIN_C,
            )

        now = self._clock()
        if self._next_skill_at is None:
            self._schedule_next_skill(now)

        if self._awaiting_main_candidates:
            # 刷新分支（优先于按档位选卡）：本组没有任何主C系技能时点刷新换卡；
            # 机会用完、OCR 未读出卡名或按钮在等待预算内仍未出现时返回 None，
            # 直接落入下方选卡流程——刷新机会已用尽后必选一张（2026-09-12 用户确认）。
            # 刷新点击生效后的「等卡面换新」由验证帧承担：验证通过即新卡面已
            # OCR 可读，本帧（验证帧）的 skill_card_options 直接供下方按档位选卡
            refresh_action = self._skill_refresh_action(observation)
            if refresh_action is not None:
                return refresh_action, State.SELECT_MAIN_C_SKILLS
            priority_action = self._skill_priority_action(
                observation,
                "main_skill_priority",
                State.SELECT_MAIN_C_SKILLS,
                post_delay=self._run_config.skill_click_post_delay_seconds,
            )
            if priority_action is not None:
                return priority_action
            if observation.skill_candidates:
                return self._choose_skill_action(
                    observation.skill_candidates,
                    "main_skill_candidate",
                    State.SELECT_MAIN_C_SKILLS,
                    post_delay=self._run_config.skill_click_post_delay_seconds,
                )
            if not observation.flag("select_skill_button_visible"):
                # 技能页不在（可能被击杀奖励弹窗打断后关闭）：连续多帧确认后停止
                # 等待，回到定时点选技能的节奏，绝不在页面不在时盲点技能卡位置
                if self._main_page_missing_checks >= self._run_config.main_skill_page_closed_frames:
                    self._awaiting_main_candidates = False
                    self._main_skill_empty_checks = 0
                    self._main_page_missing_checks = 0
                    self._schedule_next_skill(now)
                    return (
                        Action(
                            kind="wait",
                            duration=0.3,
                            verification="immediate",
                            tag="main_skill_page_closed",
                            reason="技能页已关闭，回到定时选技能节奏",
                        ),
                        State.SELECT_MAIN_C_SKILLS,
                    )
                return (
                    Action(
                        kind="wait",
                        duration=0.4,
                        verification="immediate",
                        tag="main_skill_page_missing_check",
                        reason="技能页暂不可见，等待出现或确认关闭",
                    ),
                    State.SELECT_MAIN_C_SKILLS,
                )
            if self._main_skill_empty_checks >= self._run_config.skill_recognition_frames or (
                observation.flag("teammate_skill_visible")
                and self._main_skill_empty_checks >= self._run_config.skill_fallback_settle_frames
            ):
                # 多帧没识别到主C技能图标（或识别到队友图标确认本组无主C卡，
                # 且页面已稳定出现足够帧数）→ 随便选一个，再等下一轮
                return self._skill_fallback_action(
                    "main_skill_fallback",
                    State.SELECT_MAIN_C_SKILLS,
                    window_ctx,
                    post_delay=self._run_config.skill_click_post_delay_seconds,
                )
            return (
                Action(
                    kind="wait",
                    duration=0.4,
                    verification="immediate",
                    tag="main_skill_empty_check",
                    reason="等待主C技能图标出现",
                ),
                State.SELECT_MAIN_C_SKILLS,
            )

        assert self._next_skill_at is not None
        if now < self._next_skill_at:
            return (
                Action(
                    kind="wait",
                    duration=min(1.0, self._next_skill_at - now),
                    verification="immediate",
                    reason="等待下一次金币检查",
                ),
                State.SELECT_MAIN_C_SKILLS,
            )

        # 付费选技能金币门控：费用红色（游戏自身判买不起）或数值不足时不点
        # （游戏会忽略该点击），短等后重查。金币信息缺失（未读取/读取失败）
        # 时放行，保持原有行为。
        gold = observation.raw_data.get("gold")
        if gold is not None:
            affordable = self._can_afford(gold, "skill")
            if gold.get("skill_red"):
                logger.info("选技能金币检查：费用红色（游戏判买不起）→ 等待恢复")
            elif gold.get("current") is None or gold.get("skill_cost") is None:
                logger.info("选技能金币检查：只认色模式，费用白色（买得起）→ 可以点")
            else:
                logger.info(
                    "选技能金币检查：现有 %s，需要 %s → %s",
                    gold.get("current"),
                    gold.get("skill_cost"),
                    "可以点" if affordable else "买不起，等待恢复",
                )
        if gold is not None and not self._can_afford(gold, "skill"):
            # 付不起：等待金币恢复（击杀回金），红色等 gold_red_wait_seconds
            # 后重新识别再判
            wait_for = self._run_config.gold_red_wait_seconds
            self._next_skill_at = self._clock() + wait_for
            if gold.get("skill_red"):
                detail = "选技能费用红色（游戏判买不起）"
            else:
                detail = f"现有 {gold.get('current')} < 选技能 {gold.get('skill_cost')}"
            return (
                Action(
                    kind="wait",
                    duration=wait_for,
                    verification="immediate",
                    tag="skill_gold_wait",
                    reason=f"金币不足等待恢复：{detail}",
                ),
                State.SELECT_MAIN_C_SKILLS,
            )

        decision = self._hotspot_click(
            "select_skill",
            "open_main_skills",
            "定时点击选技能按钮",
            State.SELECT_MAIN_C_SKILLS,
            window_ctx,
            post_delay=self._run_config.skill_click_post_delay_seconds,
        )
        if decision is None:
            return None
        action, to_state = decision
        # 点击后把指针停靠到无关位置：悬停在【选技能】上会触发游戏高亮，
        # 稀释下一帧费用红/白判定的红色占比（2026-09-10 实机：连续多次
        # 红色被误判为白色导致盲点）
        park = self._hotspot_point("mouse_park", window_ctx)
        if park is not None:
            action = replace(action, hover_park=park)
        return action, to_state

    def _choose_skill_action(
        self,
        candidates: list[SkillCandidate],
        tag: str,
        to_state: State,
        *,
        post_delay: float = 0.0,
    ) -> tuple[Action, State]:
        # 简化版：候选都是主C技能图标命中位置，随机选一个（多个时随机点其一）
        candidate = self._rng.choice(candidates)
        return (
            Action(
                kind="click",
                target=candidate.position,
                duration=0.08,
                verification="next_frame",
                tag=tag,
                reason=f"识别到主C技能图标，点击其一（共 {len(candidates)} 个）",
                post_delay=post_delay,
            ),
            to_state,
        )

    def wants_gold_read(self) -> bool:
        """需要金币感知的状态：选技能阶段、批次召唤期间、以及开局强制召唤之后的培养/回补召唤。

        开局强制召唤阶段（前 minimum_summon_count_before_skills 次）不读金币（无门控、直接点）。
        批次召唤的每次点击前都读（召唤费用红/白为毫秒级色判，2026-09-10 用户策略）。
        """
        if self.ctx.current_state == State.SELECT_MAIN_C_SKILLS:
            return True
        if self.ctx.current_state == State.BUILD_MAIN_C:
            # 批次召唤进行中：每次点击前都要金币门控
            if self._batch_summon_remaining > 0:
                return True
            # 开局强制召唤阶段无门控,不读;解锁技能系统后的召唤才有金币门控
            return self._summon_count >= self._run_config.minimum_summon_count_before_skills
        return False

    @staticmethod
    def _can_afford(gold: dict, cost_key: str) -> bool:
        """判断金币是否够付某项费用。

        红色(游戏自身的买不起信号)直接判负；数值缺失时视为付得起
        （调用方在金币信息缺失时应保持原有行为）。
        """
        if gold.get(f"{cost_key}_red"):
            return False
        current = gold.get("current")
        cost = gold.get(f"{cost_key}_cost")
        if current is None or cost is None:
            return True
        return current >= cost

    def _skill_priority_action(
        self,
        observation: Observation,
        tag: str,
        to_state: State,
        *,
        post_delay: float = 0.0,
    ) -> tuple[Action, State] | None:
        """按主C优先级档位选卡（感知层标题 OCR 主路径）。

        感知层在标题 OCR 成功时在 raw_data 给出每张卡的技能名/档位/点击
        位置；取最小档，同档内随机点一张。未配置优先级、本帧无 OCR 结果
        时返回 None，走原有图标识别/随机兜底流程。
        """
        options = observation.raw_data.get("skill_card_options")
        if not options:
            return None
        best_tier = min(option["tier"] for option in options)
        pool = [option for option in options if option["tier"] == best_tier]
        chosen = self._rng.choice(pool)
        names = "、".join(str(option["name"]) for option in pool)
        return (
            Action(
                kind="click",
                target=tuple(chosen["position"]),
                duration=0.08,
                verification="next_frame",
                tag=tag,
                reason=f"按优先级第 {best_tier} 档选卡：{chosen['name']}（同档候选：{names}）",
                post_delay=post_delay,
            ),
            to_state,
        )

    def _skill_refresh_action(self, observation: Observation) -> Action | None:
        """局内技能页「刷新」分支：三张卡均无主C系技能时点一次刷新换卡。

        触发条件（全部满足）：功能开启、本局刷新机会未用完、标题 OCR 读出
        全部卡名且最低档位 > 主C档上限（无任何主C系技能）。按钮命中即返回
        点击动作；按钮未命中返回限次等待动作（页面过渡帧上按钮可能尚未渲染
        完），等待预算耗尽返回 None 交调用方落入现有选卡流程。OCR 没读出
        卡名时不等待不刷新（识别不可靠时不浪费每局唯一的机会）。验证与
        后续等待见 verify_action / on_action_verified。
        """
        cfg = self._run_config
        if not cfg.skill_refresh_enabled or cfg.skill_refresh_max_per_round <= 0:
            return None
        if self._skill_refresh_used_count >= cfg.skill_refresh_max_per_round:
            return None
        options = observation.raw_data.get("skill_card_options")
        if not options:
            return None
        best_tier = min(int(option["tier"]) for option in options)
        if best_tier <= _MAIN_C_TIER_CEILING:
            return None
        match = self._flag_match(observation, "skill_refresh_match")
        if match is None:
            # 页面刚弹开后的首个决策帧按钮可能尚未渲染完（2026-09-12 实机：
            # 两次 miss 都在打开页面后的第一帧，静态截图同位置 0.99+ 命中）：
            # 限次等待重试，预算内不选卡；耗尽才按「按钮不可用」放弃刷新
            if self._skill_refresh_button_waits < cfg.skill_refresh_button_wait_max:
                return Action(
                    kind="wait",
                    duration=cfg.skill_refresh_button_wait_seconds,
                    verification="immediate",
                    tag="skill_refresh_wait_button",
                    reason=(
                        "刷新按钮未识别到（可能为页面过渡帧），"
                        f"等待重试（{self._skill_refresh_button_waits + 1}"
                        f"/{cfg.skill_refresh_button_wait_max}）"
                    ),
                )
            logger.warning(
                "刷新按钮连续 %d 次等待后仍未识别到，本局机会保留、直接选卡",
                cfg.skill_refresh_button_wait_max,
            )
            return None
        names = "、".join(str(option["name"]) for option in options)
        return Action(
            kind="click",
            target=match.position,
            duration=0.08,
            verification="next_frame",
            tag="skill_refresh",
            reason=(
                f"三张卡均无主C系技能（最低档 {best_tier}：{names}），"
                f"点刷新换卡（本局第 {self._skill_refresh_used_count + 1} 次）"
            ),
        )

    def _skill_fallback_action(
        self,
        tag: str,
        to_state: State,
        window_ctx: WindowContext | None,
        *,
        reason: str = "未识别到主C技能，随便选一个",
        post_delay: float = 0.0,
    ) -> tuple[Action, State] | None:
        """从技能 ROI 的三列卡片中心随机选一个（未识别到主C图标或天使技能页时使用）。"""
        if self._skill_candidate_roi is None or window_ctx is None:
            logger.error("技能候选 ROI 缺失，无法计算三张技能卡的兜底点击点")
            return None
        try:
            points = roi_column_centers(
                self._skill_candidate_roi,
                window_ctx.client_size,
                columns=3,
            )
        except ValueError as exc:
            logger.error("技能候选 ROI 无效，无法计算兜底点击点: %s", exc)
            return None
        point = self._rng.choice(points)
        return (
            Action(
                kind="click",
                target=point,
                duration=0.08,
                verification="next_frame",
                tag=tag,
                reason=reason,
                post_delay=post_delay,
            ),
            to_state,
        )

    # ------------------------------------------------------------------
    # 主 C 培养
    # ------------------------------------------------------------------

    def _action_build_main_c(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        # 召唤未生效后的冷却（金币恢复期）：等待后再重新尝试
        if self._summon_retry_at is not None:
            if self._clock() < self._summon_retry_at:
                return (
                    Action(
                        kind="wait",
                        duration=1.0,
                        verification="immediate",
                        tag="summon_retry_wait",
                        reason="召唤未生效（可能金币不足），等待后重试",
                    ),
                    State.BUILD_MAIN_C,
                )
            self._summon_retry_at = None

        # ---- 被迫合并 3 星对后的赠送技能页确认（2026-08-21 用户策略）----
        # 合并动作发出即挂起等待：先固定等待页面渲染，再按轮询间隔识别
        # 【请选择1个额外技能】提示条，连续 N 次命中才进入选卡（半渲染/
        # 动画过渡帧不动作）；连续 M 次未命中 = 拖动未生效或对局已结束
        # （页面随结算消失），放弃等待恢复常规决策。期间击杀奖励弹窗由
        # decide_action 顶部弹窗优先逻辑关闭，返回按钮出现即转入结算
        if self._awaiting_merge_gift:
            cfg = self._run_config
            if not self._merge_gift_settled:
                self._merge_gift_settled = True
                return (
                    Action(
                        kind="wait",
                        duration=cfg.merge_gift_settle_seconds,
                        verification="immediate",
                        tag="merge_gift_settle",
                        reason="3星对已合并，等待4星赠送技能页渲染",
                    ),
                    State.BUILD_MAIN_C,
                )
            if observation.flag("merge_gift_skill_page_visible"):
                self._merge_gift_hits += 1
                self._merge_gift_misses = 0
            else:
                self._merge_gift_misses += 1
                self._merge_gift_hits = 0
            if self._merge_gift_hits >= cfg.merge_gift_confirm_hits:
                self._awaiting_merge_gift = False
                logger.info(
                    "4星赠送技能页确认可见（连续 %d 次），选择技能",
                    self._merge_gift_hits,
                )
                # 当前帧提示条可见，落入下方选卡分支
            elif self._merge_gift_misses >= cfg.merge_gift_fail_misses:
                self._awaiting_merge_gift = False
                logger.info(
                    "合并后连续 %d 次未见赠送技能页（拖动未生效或对局已结束），恢复常规决策",
                    self._merge_gift_misses,
                )
                # 落入常规棋盘决策（棋盘为空则由空棋盘重试/结算接管）
            else:
                return (
                    Action(
                        kind="wait",
                        duration=cfg.merge_gift_poll_interval_seconds,
                        verification="immediate",
                        tag="merge_gift_poll",
                        reason=(
                            "等待4星赠送技能页确认 "
                            f"连续{self._merge_gift_hits}/{cfg.merge_gift_confirm_hits}"
                        ),
                    ),
                    State.BUILD_MAIN_C,
                )

        # ---- 变身阶段机（2026-09-13 用户策略，设计文档 §4.2）----
        # 进行中的变身优先于下方所有常规决策；返回 None 表示本次变身结束
        # （成功/放弃/预算耗尽），清空阶段后落入常规培养决策
        if self._transform_stage is not None:
            stage_action = self._transform_stage_action(observation, window_ctx)
            if stage_action is not None:
                return stage_action

        # 合成 4 星后系统赠送的技能选择页（实机确认 2026-08-16）：两个 3 星
        # 合成 4 星时弹出，3 选 1，页面遮住棋盘导致识别为空。优先点主C技能卡，
        # 识别不到就用三列兜底随机选一张；选完页面关闭、棋盘恢复。
        # 页面主标识为【请选择1个额外技能】提示条（2026-08-21 实机确认该文案
        # 仅此页出现且不选则常驻；【选技能】图在该页不稳定，曾整页等待却未命中
        # 导致掉进空棋盘重试直至保守停止）；页面被Boss奖励弹窗遮挡时由上方
        # 弹窗优先关闭逻辑兜底，关完即可识别。确认流程见上方 _awaiting 分支，
        # 此处兜底覆盖确认放弃后页面才出现等罕见路径
        if (
            observation.flag("merge_gift_skill_page_visible")
            or observation.flag("select_skill_button_visible")
            or observation.skill_candidates
        ):
            priority_action = self._skill_priority_action(
                observation, "merge_gift_skill_priority", State.BUILD_MAIN_C
            )
            if priority_action is not None:
                return priority_action
            if observation.skill_candidates:
                return self._choose_skill_action(
                    observation.skill_candidates,
                    "merge_gift_skill_candidate",
                    State.BUILD_MAIN_C,
                )
            return self._skill_fallback_action(
                "merge_gift_skill_fallback",
                State.BUILD_MAIN_C,
                window_ctx,
                reason="合成4星赠送技能页，随机选一个技能",
            )

        # ---- 批次进行中：继续发送批次动作（2026-09-10 用户策略）----
        # 批内单帧观察（无棋盘快照）：召唤每次点击前做金币门控，不足则等待、
        # 不消耗批次；合成按队列继续拖。全部发完后恢复多帧识别统一验证，
        # 由批末规划逻辑归因未生效的合成对
        if self._batch_summon_remaining > 0:
            return self._batch_summon_action(observation, window_ctx)
        if self._batch_merge_queue:
            return self._batch_merge_action()

        board = observation.board
        minimum_summons = self._effective_opening_summons()
        if self._summon_count < minimum_summons:
            point = self._hotspot_point("add_hero", window_ctx)
            if point is None:
                return None
            next_summon = self._summon_count + 1
            if next_summon >= minimum_summons:
                # 最后一次强制召唤后即将进入棋盘决策：保留正常稳定等待
                post_delay = self._rng.uniform(
                    self._run_config.summon_recognition_delay_min,
                    self._run_config.summon_recognition_delay_max,
                )
            else:
                # 强制阶段不以识别为门禁，按固定间隔快速连点即可
                post_delay = self._run_config.forced_summon_interval_seconds
            return (
                Action(
                    kind="click",
                    target=point,
                    duration=0.08,
                    post_delay=post_delay,
                    verification="immediate",
                    tag="required_summon",
                    hover_park=self._hotspot_point("mouse_park", window_ctx),
                    reason=f"开局强制召唤 {next_summon}/{minimum_summons}",
                ),
                State.BUILD_MAIN_C,
            )

        if board is None or not board.heroes:
            # 已进入对局后棋盘识别为空，必然是页面遮挡（赠送技能页、击杀
            # 弹窗、结算过渡）：等待不发任何输入，持续等待永远比退出安全
            # （2026-08-21 用户决策），页面恢复后继续培养，返回按钮出现即
            # 正常转入结算；每累计 _EMPTY_BOARD_RETRY 次输出一次告警便于
            # 从日志定位卡点
            self._empty_board_count += 1
            if self._empty_board_count % _EMPTY_BOARD_RETRY == 0:
                logger.warning(
                    "棋盘连续 %d 次识别为空，不停止，继续等待页面恢复或对局结束",
                    self._empty_board_count,
                )
            return (
                Action(
                    kind="wait",
                    duration=0.5,
                    verification="immediate",
                    reason=f"棋盘识别空，继续等待 ({self._empty_board_count})",
                ),
                State.BUILD_MAIN_C,
            )
        self._empty_board_count = 0

        # 邀请者「准备完成」（spec §5.4，2026-09-22 用户定案）：开局召唤
        # 计满后点一次固定坐标确认（与阵容无关），点击后进入常规凑阵容循环
        if self._invite_mode and not self._invite_ready_clicked:
            ready_action = self._invite_ready_done_action(window_ctx)
            if ready_action is None:
                return None
            return ready_action

        # 召唤金币门控（开局强制召唤之外的所有召唤）：费用不足时优先在等待
        # 空档合成（见下方分支），无对可合再原地等待金币恢复（击杀回金），
        # 一直等到够或对局自然结束——不设上限、不放弃回补（2026-09-05 定案）。
        # 放在无进展检查之前：等待本身不算停滞，重置无进展计数，避免等待
        # 期间误触保守停止。
        # 金币信息缺失（未读取/OCR 失败）时不拦截，保持原有行为。
        gold = observation.raw_data.get("gold")
        if gold is not None and not self._can_afford(gold, "summon"):
            self._no_progress_count = 0
            self._last_board_signature = None
            if gold.get("summon_red"):
                detail = "召唤费用红色（游戏判买不起）"
            else:
                detail = f"现有 {gold.get('current')} < 召唤 {gold.get('summon_cost')}"
            # 等金币恢复的空档同样优先合成（2026-09-13 用户策略，与批次
            # 召唤等待期同一策略）：批次最后一次召唤发完后金币转红会落到
            # 这里，棋盘占满时正应合成腾格子而不是干等。两个例外不合成：
            # 棋盘已达标（金币恢复后直接转选技能，别再动棋盘）；回补阶段
            # 有空位（合成减少英雄数量，与「有空位先召唤占位」冲突）。
            # 无合法对再原地等待。
            if not self._strategy.main_c_ready(
                board, require_hero_count=self._topup_active
            ) and not (
                self._topup_active and board.capacity.available > 0
            ):
                merge = self._gold_wait_merge_action(observation, "召唤金币不足")
                if merge is not None:
                    return merge
                # 邀请者出口（spec §5.3）：无安全合成且棋盘满时，变身碰辅助
                # 优先于干等——否则金币门控循环会吞掉每局 2 次的变身机会
                assist = self._invite_gold_wait_assist(board, observation, "召唤金币不足")
                if assist is not None:
                    return assist
            return (
                Action(
                    kind="wait",
                    duration=self._run_config.gold_red_wait_seconds,
                    verification="immediate",
                    tag="summon_gold_wait",
                    reason=(
                        f"召唤金币不足，等待恢复：{detail}"
                        + ("（回补阶段，等待后继续回补）" if self._topup_active else "")
                    ),
                ),
                State.BUILD_MAIN_C,
            )

        signature = self._board_signature(board)
        if signature == self._last_board_signature:
            self._no_progress_count += 1
            if self._no_progress_count >= _NO_PROGRESS_LIMIT:
                return None
        else:
            self._no_progress_count = 0
            # 棋盘内容变化时输出各格英雄，便于人工核对识别结果和后续动作
            logger.info(
                "棋盘识别 frame=%d 占用=%d/%d: %s",
                observation.frame_id,
                len(board.heroes),
                board.capacity.total_slots,
                ", ".join(
                    f"{hero.hero_type}{hero.star_level}星@{_hero_location(hero)}"
                    for hero in board.heroes
                ),
            )
        self._last_board_signature = signature

        if self._strategy.main_c_ready(board, require_hero_count=self._topup_active):
            # 数量已达标：本局内不再回培养复查数量（选技能期间数量不会减少）
            if self._strategy.hero_count_satisfied(board):
                self._topup_done = True
            return (
                Action(
                    kind="wait",
                    duration=0.2,
                    verification="immediate",
                    tag="main_c_ready",
                    reason=self._strategy.ready_reason(
                        board, require_hero_count=self._topup_active
                    ),
                ),
                State.SELECT_MAIN_C_SKILLS,
            )

        # ---- 变身触发（设计文档 §4.1：次数未满 + 3星非主C）----
        # 免费且可能直接提星/凑数量，优先于批次召唤/合成；2026-09-17 用户
        # 变更：不再要求「无达标主C」——主C已达标（含回补阶段）时，2星对
        # 合并产出3星非主C的下一轮棋盘识别即检查触发；locator 未接线（模板
        # 缺失降级）或开关关闭时整分支跳过；放弃记忆未解除（棋盘签名未演化）
        # 时抑制触发，防止「放弃 → 立刻再触发」死循环
        if (
            self._transform_skip_signature is not None
            and self._board_signature(board) != self._transform_skip_signature
        ):
            # 棋盘已演化（召唤/合成改变了英雄集合）：解除放弃记忆的抑制
            self._transform_skip_signature = None
        if (
            self._run_config.transform_enabled
            and self._transform_locator is not None
            and self._transform_skip_signature is None
        ):
            transform_target = self._strategy.select_transform(
                board,
                self._transform_used_count,
                self._run_config.transform_max_per_round,
            )
            if transform_target is not None:
                self._transform_stage = "poll"
                self._transform_target = transform_target
                self._transform_button_pos = None
                self._transform_reclicks = 0
                self._transform_verify_attempts = 0
                self._transform_poll_deadline = (
                    self._clock() + self._run_config.transform_button_appear_timeout_seconds
                )
                logger.info(
                    "触发变身：%s%d星 %s 变身碰主C（本局已用 %d/%d 次）",
                    transform_target.hero_type,
                    transform_target.star_level,
                    _hero_location(transform_target),
                    self._transform_used_count,
                    self._run_config.transform_max_per_round,
                )
                return (
                    Action(
                        kind="click",
                        target=transform_target.position,
                        duration=0.08,
                        post_delay=self._run_config.transform_button_poll_interval_seconds,
                        verification="immediate",
                        tag="transform_click_hero",
                        reason="点击3星非主C英雄，弹出变身按钮",
                    ),
                    State.BUILD_MAIN_C,
                )

        # ---- 批次规划（2026-09-10 用户策略）：识别一次、动作攒批、批后统一识别 ----
        # 先把空格一批召唤填满（每次点击前金币门控），填满后再把互不相交的
        # 非主C 1/2星对一批拖完；两档都可以按配置关闭，走原有逐动作识别验证。
        # 邀请者（spec §5.4）：按 summon_batch_size 分批，批间重新识别
        batch_limit = board.capacity.available
        if self._invite_mode and self._effective_batch_size() < batch_limit:
            batch_limit = self._effective_batch_size()
            logger.info(
                "邀请者批次召唤限量 %d（本批 %d/%d，空格 %d）",
                batch_limit,
                batch_limit,
                self._effective_batch_size(),
                board.capacity.available,
            )
        if self._run_config.batch_summon_enabled and batch_limit > 0:
            self._batch_summon_remaining = batch_limit
            logger.info(
                "规划批次召唤 %d 次（空格 %d/%d，frame=%d）",
                batch_limit,
                board.capacity.available,
                board.capacity.total_slots,
                observation.frame_id,
            )
            return self._batch_summon_action(observation, window_ctx)
        candidates = board.find_merge_candidates(
            self.ctx.main_c, wildcard_heroes=self._wildcard_heroes
        )
        # 回补阶段（2026-09-07 用户策略）：棋盘有空位时直接召唤不合成——
        # 合成要拖动加验证耗时，先召唤占满空位更快；棋盘占满且数量未达标
        # 才合成腾格子。批次合成遵循同一优先级
        topup_fill_first = self._topup_active and board.capacity.available > 0
        if self._run_config.batch_merge_enabled and not topup_fill_first:
            # 批末归因：上一批中仍原样存在的对记入失败记忆（拖动未生效）
            self._attribute_merge_batch(candidates)
            batch = self._strategy.select_merge_batch(
                candidates, self._failed_merge_pairs, board
            )
            if batch:
                return self._plan_merge_batch(batch, observation.frame_id)
        # 批次关闭或无可批对：走原有单动作策略
        pair = None
        if not topup_fill_first:
            # 合成/避让/失败对跳过等策略决策由 CultivationStrategy 负责；
            # 返回 None 表示本次不合并，落入下方召唤分支
            pair = self._strategy.select_merge(board, candidates, self._failed_merge_pairs)
        if pair is not None:
            return self._merge_drag_action(pair)

        point = self._hotspot_point("add_hero", window_ctx)
        if point is None:
            return None
        if self._topup_active and board.capacity.available > 0:
            reason = "回补阶段棋盘有空位，直接召唤不合成"
        else:
            reason = "无合法合成对，召唤1个英雄"
        if gold is not None:
            reason += (
                f"（金币：现有 {gold.get('current')} / "
                f"召唤需 {gold.get('summon_cost')}）"
            )
        return (
            Action(
                kind="click",
                target=point,
                duration=0.08,
                post_delay=self._rng.uniform(
                    self._run_config.summon_recognition_delay_min,
                    self._run_config.summon_recognition_delay_max,
                ),
                verification="next_frame",
                tag="summon_hero",
                hover_park=self._hotspot_point("mouse_park", window_ctx),
                reason=reason,
            ),
            State.BUILD_MAIN_C,
        )

    def _gold_wait_merge_action(
        self,
        observation: Observation,
        context: str,
    ) -> tuple[Action, State] | None:
        """金币等待期的合成动作（2026-09-13 用户策略）。

        合成不花金币，正好利用等金币恢复的空档：用与培养阶段相同的策略
        选合法对（非主C优先、主C数量保护、3星规避等规则全部生效），以
        单动作合成发出（next_frame 验证，拖动失败对记忆照常生效）。棋盘
        快照缺失或无可合对返回 None，调用方退回原地等待。context 为来源
        描述（如「批次召唤金币不足（本批剩余 2 次）」），用于日志与
        动作原因。
        """
        board = observation.board
        if board is None:
            return None
        candidates = board.find_merge_candidates(
            self.ctx.main_c, wildcard_heroes=self._wildcard_heroes
        )
        pair = self._strategy.select_merge(board, candidates, self._failed_merge_pairs)
        if pair is None:
            return None
        logger.info("%s，等待恢复期间优先合成", context)
        return self._merge_drag_action(
            pair,
            reason=f"{context}，等待恢复期间合成",
        )

    def _merge_drag_action(
        self,
        pair: MergeCandidate,
        reason: str | None = None,
    ) -> tuple[Action, State]:
        """单动作合成：日志、3 星对赠送技能页标记与拖动动作（next_frame 验证）。"""
        source, dest = CultivationStrategy.drag_direction(pair)
        pair_kind = "主C对" if pair.is_main_c else "非主C对"
        logger.info(
            "拖动合成%s: %s%d星 %s -> %s",
            pair_kind,
            source.hero_type,
            source.star_level,
            _hero_location(source),
            _hero_location(dest),
        )
        if max(source.star_level, dest.star_level) >= 3:
            # 被迫合并 3 星对：合成 4 星将弹赠送技能页，合并后进入确认流程
            self._awaiting_merge_gift = True
            self._merge_gift_settled = False
            self._merge_gift_hits = 0
            self._merge_gift_misses = 0
        return (
            Action(
                kind="drag",
                target=source.position,
                end=dest.position,
                duration=self._run_config.merge_drag_duration,
                verification="next_frame",
                tag="merge_heroes",
                reason=reason
                or (
                    "优先拖动合成非主C合法对"
                    if not pair.is_main_c
                    else "无其他合法对，拖动合成主C对"
                ),
            ),
            State.BUILD_MAIN_C,
        )

    def _transform_stage_action(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        """变身阶段机主体（设计文档 §4.2）：poll → confirm → verify。

        返回 None = 本次变身结束（成功、放弃或预算耗尽），调用方先经
        ``_clear_transform_state`` 清空后落入常规培养决策；任何失败路径都
        不结束任务、不盲点。settle 不是独立阶段：confirm 确认按钮消失后
        直接置 verify 并返回 settle 时长的等待动作。
        """
        cfg = self._run_config
        target = self._transform_target
        assert target is not None  # 阶段存在即有目标（触发处成对设置）

        def _wait(tag: str, duration: float, reason: str) -> tuple[Action, State]:
            return (
                Action(
                    kind="wait",
                    duration=duration,
                    verification="immediate",
                    tag=tag,
                    reason=reason,
                ),
                State.BUILD_MAIN_C,
            )

        if self._transform_stage == "poll":
            match = self._flag_match(observation, "transform_button_match")
            if (
                match is not None
                and self._match_in_transform_window(
                    match.position, target.position, window_ctx
                )
            ):
                self._transform_button_pos = match.position
                self._transform_stage = "confirm"
                return (
                    Action(
                        kind="click",
                        target=match.position,
                        duration=0.08,
                        post_delay=0.3,
                        verification="immediate",
                        tag="transform_click_button",
                        reason=f"点击变身按钮（置信度 {match.confidence:.2f}）",
                    ),
                    State.BUILD_MAIN_C,
                )
            if self._clock() < self._transform_poll_deadline:
                return _wait(
                    "transform_button_poll",
                    cfg.transform_button_poll_interval_seconds,
                    "轮询等待变身按钮弹出",
                )
            if self._transform_reclicks < cfg.transform_button_reclick_max:
                self._transform_reclicks += 1
                self._transform_poll_deadline = (
                    self._clock() + cfg.transform_button_appear_timeout_seconds
                )
                logger.info(
                    "变身按钮未弹出，重点击英雄 %d/%d（按钮约1秒内弹出，仅未弹出时重点击）",
                    self._transform_reclicks,
                    cfg.transform_button_reclick_max,
                )
                return (
                    Action(
                        kind="click",
                        target=target.position,
                        duration=0.08,
                        post_delay=cfg.transform_button_poll_interval_seconds,
                        verification="immediate",
                        tag="transform_reclick_hero",
                        reason="变身按钮未弹出，重点击英雄",
                    ),
                    State.BUILD_MAIN_C,
                )
            logger.warning(
                "变身按钮轮询超时（游戏侧次数已尽或按钮未渲染），放弃本次变身，"
                "不扣次数（计数失步自愈兜底）"
            )
            self._remember_transform_giveup(observation)
            self._clear_transform_state()
            return None

        if self._transform_stage == "confirm":
            match = self._flag_match(observation, "transform_button_match")
            button_tolerance = (
                max(20, int(0.04 * window_ctx.client_size[0]))
                if window_ctx is not None
                else 40
            )
            still_at_pos = (
                match is not None
                and self._transform_button_pos is not None
                and abs(match.position[0] - self._transform_button_pos[0])
                <= button_tolerance
                and abs(match.position[1] - self._transform_button_pos[1])
                <= button_tolerance
            )
            if not still_at_pos:
                # 按钮已消失（原位不在）：变身已开始——扣减本局次数（开环计次，
                # 与技能刷新同哲学），等待动画后棋盘验证
                self._transform_used_count += 1
                self._transform_stage = "verify"
                logger.info(
                    "变身按钮已消失（变身开始），等待动画稳定（本局已用 %d/%d 次）",
                    self._transform_used_count,
                    cfg.transform_max_per_round,
                )
                return _wait(
                    "transform_settle",
                    cfg.transform_settle_seconds,
                    f"变身动画稳定等待（{cfg.transform_settle_seconds:.0f}秒，实测约4秒）",
                )
            if self._transform_reclicks < cfg.transform_button_reclick_max:
                self._transform_reclicks += 1
                return (
                    Action(
                        kind="click",
                        target=self._transform_button_pos,
                        duration=0.08,
                        post_delay=0.3,
                        verification="immediate",
                        tag="transform_reclick_button",
                        reason="变身按钮未消失，重点按钮（点击可能未落上）",
                    ),
                    State.BUILD_MAIN_C,
                )
            logger.warning("变身按钮点击未生效（原位按钮仍在），放弃本次变身，不扣次数")
            self._remember_transform_giveup(observation)
            self._clear_transform_state()
            return None

        # stage == "verify"：多帧棋盘识别已恢复（wants_board_watch 对本阶段
        # 返回 True）；「类型改变且星级不变」才算成功（变身类型必变）
        board = observation.board
        if board is None or not board.heroes:
            return _wait(
                "transform_verify_wait",
                0.5,
                "棋盘识别为空（页面遮挡），等待恢复后验证变身",
            )
        near_tolerance = (
            int(0.03 * window_ctx.client_size[1]) if window_ctx is not None else 52
        )
        nearest = min(
            board.heroes,
            key=lambda hero: (
                (hero.position[0] - target.position[0]) ** 2
                + (hero.position[1] - target.position[1]) ** 2
            ),
        )
        near = (
            abs(nearest.position[0] - target.position[0]) <= near_tolerance
            and abs(nearest.position[1] - target.position[1]) <= near_tolerance
        )
        if (
            near
            and nearest.star_level == target.star_level
            and nearest.hero_type != target.hero_type
        ):
            logger.info(
                "变身验证通过：%s%d星 → %s%d星（同位置星级不变、类型必变）",
                target.hero_type,
                target.star_level,
                nearest.hero_type,
                nearest.star_level,
            )
            self._clear_transform_state()
            return None
        self._transform_verify_attempts += 1
        if self._transform_verify_attempts < cfg.transform_verify_attempts:
            return _wait(
                "transform_verify_retry",
                cfg.transform_settle_seconds,
                (
                    f"变身未确认（第 {self._transform_verify_attempts} 次：类型未变"
                    "或动画未完），再等动画后重识别"
                ),
            )
        logger.warning(
            "变身验证 %d 次未确认（类型未变或英雄缺失），放弃验证，继续常规决策",
            cfg.transform_verify_attempts,
        )
        self._remember_transform_giveup(observation)
        self._clear_transform_state()
        return None

    def _remember_transform_giveup(self, observation: Observation) -> None:
        """放弃变身时记住棋盘签名：棋盘演化前不再触发（防死循环）。

        单帧观察（poll/confirm 阶段）无棋盘快照，退回最近一次多帧识别
        记录的签名（触发时的棋盘）。
        """
        signature = (
            self._board_signature(observation.board)
            if observation.board is not None
            else self._last_board_signature
        )
        if signature is not None:
            self._transform_skip_signature = signature

    def _match_in_transform_window(
        self,
        match_pos: tuple[int, int],
        hero_pos: tuple[int, int],
        window_ctx: WindowContext | None,
    ) -> bool:
        """按钮命中位置是否落在目标英雄的搜索窗口内（设计文档 §4.3）。

        窗口相对英雄格中心：±search_dx_ratio 宽、向上 search_up_ratio 高、
        向下 search_down_ratio 高（比例参数在 tasks.yaml 的
        locators.transform_button，由棋盘格尺寸推导）。窗口外命中（如误点了
        别的英雄弹出的按钮）不算本次目标。window_ctx 缺失时不拦截。
        """
        if window_ctx is None:
            return True
        cfg = self._transform_locator or {}
        cw, ch = window_ctx.client_size
        dx_tol = float(cfg.get("search_dx_ratio", 0.11)) * cw
        up_tol = float(cfg.get("search_up_ratio", 0.093)) * ch
        down_tol = float(cfg.get("search_down_ratio", 0.015)) * ch
        dx = match_pos[0] - hero_pos[0]
        dy = match_pos[1] - hero_pos[1]
        return abs(dx) <= dx_tol and -up_tol <= dy <= down_tol

    def _clear_transform_state(self) -> None:
        """清空变身阶段状态（不重置本局已用次数计数器）。"""
        self._transform_stage = None
        self._transform_target = None
        self._transform_button_pos = None
        self._transform_reclicks = 0
        self._transform_verify_attempts = 0
        self._transform_poll_deadline = 0.0

    def _batch_summon_action(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        """批次召唤的单次点击：每次点击前做召唤费用金币门控（2026-09-10 用户策略）。

        金币不足时原地等待（不消耗批次，金币随击杀恢复，与培养阶段召唤的
        等待策略一致）；点击发送即计数，间隔 forced_summon_interval_seconds，
        最后一次保留稳定等待供批后识别。
        """
        gold = observation.raw_data.get("gold")
        if gold is not None and not self._can_afford(gold, "summon"):
            if gold.get("summon_red"):
                detail = "召唤费用红色（游戏判买不起）"
            else:
                detail = f"现有 {gold.get('current')} < 召唤 {gold.get('summon_cost')}"
            # 等金币恢复的空档优先合成（2026-09-13 用户策略）：合成不花
            # 金币，正好利用等待时间；但回补阶段棋盘有空位时不合成——
            # 合成减少英雄数量，与「有空位先召唤占位」的回补策略冲突；
            # 无合法对再原地等待
            board = observation.board
            topup_fill_first = self._topup_active and (
                board is not None and board.capacity.available > 0
            )
            if not topup_fill_first:
                merge = self._gold_wait_merge_action(
                    observation,
                    f"批次召唤金币不足（本批剩余 {self._batch_summon_remaining} 次）",
                )
                if merge is not None:
                    return merge
            return (
                Action(
                    kind="wait",
                    duration=self._run_config.gold_red_wait_seconds,
                    verification="immediate",
                    tag="summon_gold_wait",
                    reason=(
                        f"批次召唤金币不足，等待恢复：{detail}"
                        f"（本批剩余 {self._batch_summon_remaining} 次）"
                    ),
                ),
                State.BUILD_MAIN_C,
            )
        point = self._hotspot_point("add_hero", window_ctx)
        if point is None:
            return None
        remaining = self._batch_summon_remaining
        if remaining <= 1:
            # 批次最后一次：保留正常稳定等待，英雄落位后批后识别
            post_delay = self._rng.uniform(
                self._run_config.summon_recognition_delay_min,
                self._run_config.summon_recognition_delay_max,
            )
        else:
            post_delay = self._run_config.forced_summon_interval_seconds
        return (
            Action(
                kind="click",
                target=point,
                duration=0.08,
                post_delay=post_delay,
                verification="immediate",
                tag="batch_summon",
                hover_park=self._hotspot_point("mouse_park", window_ctx),
                reason=f"批次召唤填满空格（本批剩余 {remaining} 次）",
            ),
            State.BUILD_MAIN_C,
        )

    def _batch_merge_action(self) -> tuple[Action, State]:
        """批次合成的单次拖动：取队列首对，拖完由批后识别统一验证。"""
        source, dest = self._batch_merge_queue[0]
        return (
            Action(
                kind="drag",
                target=source.position,
                end=dest.position,
                duration=self._run_config.merge_drag_duration,
                post_delay=self._rng.uniform(
                    self._run_config.summon_recognition_delay_min,
                    self._run_config.summon_recognition_delay_max,
                ),
                verification="immediate",
                tag="batch_merge",
                reason=(
                    f"批次合成 {source.hero_type}{source.star_level}星 -> "
                    f"{dest.hero_type}{dest.star_level}星"
                    f"（队列剩余 {len(self._batch_merge_queue)} 对）"
                ),
            ),
            State.BUILD_MAIN_C,
        )

    def _plan_merge_batch(
        self,
        batch: list[MergeCandidate],
        frame_id: int,
    ) -> tuple[Action, State]:
        """把策略选出的批次对排队并发出第一拖；批末由 _attribute_merge_batch 归因。"""
        self._batch_merge_queue = [CultivationStrategy.drag_direction(p) for p in batch]
        self._pending_merge_batch_signatures = [
            (source.position, dest.position) for source, dest in self._batch_merge_queue
        ]
        logger.info(
            "规划批次合成 %d 对（frame=%d）: %s",
            len(batch),
            frame_id,
            ", ".join(
                f"{source.hero_type}{source.star_level}星@{_hero_location(source)}"
                f"->{dest.hero_type}{dest.star_level}星@{_hero_location(dest)}"
                for source, dest in self._batch_merge_queue
            ),
        )
        return self._batch_merge_action()

    def _attribute_merge_batch(self, candidates: list[MergeCandidate]) -> None:
        """批末归因：上一批中棋盘上仍原样存在的对 = 拖动未生效，记入失败记忆。

        批内拖动不做逐动作棋盘验证（发送即提交），统一在批后识别时按位置
        签名比对：对子原样还在 = 拖动落空；不再存在的对视为已生效（批次
        只排1/2星对，合成不会弹出4星赠送页，批内棋盘只会因本批拖动而变化）。
        """
        if not self._pending_merge_batch_signatures:
            return
        current = {CultivationStrategy.pair_signature(c) for c in candidates}
        failed = [sig for sig in self._pending_merge_batch_signatures if sig in current]
        if failed:
            self._failed_merge_pairs.update(failed)
            logger.warning(
                "批次合成 %d 对拖动未生效（棋盘原样），本轮内跳过: %s",
                len(failed),
                failed,
            )
        self._pending_merge_batch_signatures = []

    # ------------------------------------------------------------------
    # 结算和动作后验证
    # ------------------------------------------------------------------

    def _action_return(self, observation: Observation) -> tuple[Action, State] | None:
        match = self._flag_match(observation, "return_button_match")
        if match is None:
            return None
        # 下一轮入口：邀请者回 INVITE_CLICK（结算落在合作挑战页，先点开始
        # 挑战再走邀请链，spec §6）；抢合作回 FIND_COOP
        if self._invite_mode and self.ctx.round_count + 1 < self.ctx.max_rounds:
            next_state = State.INVITE_CLICK
        else:
            next_state = State.FIND_COOP
        to_state = (
            State.COMPLETED if self.ctx.round_count + 1 >= self.ctx.max_rounds else next_state
        )
        return (
            Action(
                kind="click",
                target=match.position,
                duration=0.08,
                verification="next_frame",
                # 一局结束、点击返回前保存结算画面（合作/保卫宠物共用此路径）
                save_frame_before=True,
                tag="return_result",
                reason="点击返回并完成本局",
            ),
            to_state,
        )

    def verify_action(self, action: Action, before: Observation, after: Observation) -> bool:
        if action.tag == "ready_match":
            return not after.flag("ready_button_visible")
        if action.tag == "close_popup":
            return not after.flag("tan_chuang_visible")
        if action.tag == "close_bonus_popup":
            # 以赠送技能卡片标识消失确认已关闭
            return not after.flag("zeng_song_ji_neng_visible")
        if action.tag == "close_double_reward":
            # 以双倍奖励弹窗标识消失确认已取消
            return not after.flag("double_reward_dialog_visible")
        if action.tag == "close_grab_popup":
            # 以全屏弹窗关闭按钮消失确认已关闭
            return not after.flag("close_button_visible")
        if action.tag == "summon_hero":
            return self._boards_changed(before.board, after.board)
        if action.tag == "merge_heroes":
            # 4星赠送技能页已弹出即证明 3+3 合成生效（页面只由本机合成 4 星
            # 触发），且页面遮住棋盘会使常规棋盘验证必然失败——2026-08-21
            # 实机日志确认此路径
            if after.flag("merge_gift_skill_page_visible"):
                return True
            if not self._boards_changed(before.board, after.board) or after.board is None:
                return False
            if action.end is None or before.board is None:
                return False
            source = next(
                (hero for hero in before.board.heroes if hero.position == action.target), None
            )
            if source is None:
                return False
            return any(
                hero.position == action.end and hero.star_level == source.star_level + 1
                for hero in after.board.heroes
            )
        if action.tag == "opening_angel_skill_fallback":
            # 天使技能页没有【选技能】图和技能卡图标，以天使标识消失确认已选完
            return not after.flag("tian_shi_kai_ju_visible")
        if action.tag == "leave_team":
            # 以【退队】按钮消失确认已退出组队大厅
            return not after.flag("leave_team_visible")
        if action.tag == "skill_refresh":
            # 刷新生效=页面仍开着且卡面变化（换出同三张的概率极低；届时重试
            # 耗尽走 on_action_failed 按已用处理，直接选卡，不会死循环）。
            # 验证帧走完整识别（本 tag 不在轻量集合里），skill_card_options
            # 可用于比对，同开局页比对口径
            if not after.flag("select_skill_button_visible"):
                return False
            before_cmp = _comparable_skill_options(before.raw_data.get("skill_card_options"))
            after_cmp = _comparable_skill_options(after.raw_data.get("skill_card_options"))
            return (
                before_cmp is not None and after_cmp is not None and before_cmp != after_cmp
            )
        if action.tag in {
            "opening_skill_priority",
            "main_skill_priority",
            "merge_gift_skill_priority",
        }:
            # 选满档位的最后一次点击直接信任:选满后不存在下一组,页面关闭
            # 与否无需逐帧等待;若点击未生效,页面保持打开,任务层会再走
            # 随机兜底补选,不会卡死
            if (
                action.tag == "opening_skill_priority"
                and self._opening_skill_selections + 1
                >= self._run_config.opening_skill_max_selections
            ):
                return True
            # 页面收起即生效:局内选技能与合成赠送页一组一选,选完页面关闭,
            # 下一组需要再点【选技能】(页面打开期间按钮不可点)。页面未收起
            # =点击未生效(如卡未渲染完时点击落空),交给重试。这类 tag 的验证
            # 帧走轻量识别(无 skill_card_options),只看页面标志,不比对卡面
            before_visible = before.flag(
                "select_skill_button_visible"
            ) or before.flag("merge_gift_skill_page_visible")
            after_visible = after.flag(
                "select_skill_button_visible"
            ) or after.flag("merge_gift_skill_page_visible")
            if before_visible and not after_visible:
                return True
            if action.tag != "opening_skill_priority":
                return False
            # 开局技能页例外:选完一张先收起、下一组自动弹出(实机 2026-08-15),
            # 页面未收起时可能已是新组,"页面还在"分不清新组还是未点中,必须
            # 比对卡面。比对剔除 frame_id 等帧相关字段——每帧重读的 frame_id
            # 必然不同,混进来会让比对恒为"已变化",点没点中都判生效
            before_cmp = _comparable_skill_options(before.raw_data.get("skill_card_options"))
            after_cmp = _comparable_skill_options(after.raw_data.get("skill_card_options"))
            return before_cmp is not None and after_cmp is not None and before_cmp != after_cmp
        if action.tag in {
            "opening_skill_candidate",
            "opening_skill_fallback",
            "main_skill_candidate",
            "main_skill_fallback",
            "merge_gift_skill_candidate",
            "merge_gift_skill_fallback",
        }:
            page_closed = (
                before.flag("select_skill_button_visible")
                or before.flag("merge_gift_skill_page_visible")
            ) and not (
                after.flag("select_skill_button_visible")
                or after.flag("merge_gift_skill_page_visible")
            )
            return page_closed or self._skill_signature(before) != self._skill_signature(after)
        if action.tag.startswith("scroll_difficulties:"):
            # 难度是可选条件：拖动后只要求拿到后续帧再重新识别。候选集合
            # 未变化可能是到达边界或目标模板漏识别，应计入有限滚动预算后跳过，
            # 不能让正式合作流程因某个难度识别不到而停止。
            return after.frame_id > before.frame_id
        if action.tag == "return_result":
            return not after.flag("return_button_visible")
        return False

    def on_action_verified(self, action: Action, to_state: State) -> None:
        tag = action.tag
        if tag == "skill_refresh_wait_button":
            # 刷新按钮等待重试计数；页面过渡帧过去后下一帧通常会命中
            self._skill_refresh_button_waits += 1
        elif tag == "skill_refresh":
            # 刷新生效：本局机会减一。验证帧（卡面已变化且 OCR 可读）会直接
            # 进入本帧决策，按优先级选卡——不再额外固定等待
            self._skill_refresh_used_count += 1
            self._skill_refresh_button_waits = 0
            logger.info(
                "技能页刷新已生效，直接按验证帧识别结果选卡（本局已用 %d/%d 次）",
                self._skill_refresh_used_count,
                self._run_config.skill_refresh_max_per_round,
            )
        elif tag == "open_territory":
            # 领地点击已发送：进入打开确认（轮询导航文字变「返回」），
            # deadline 从点击时刻重新起算；重点计数保留（同一入口链共用预算）
            self._recruit_step = _RecruitStep.CONFIRM_TERRITORY_OPEN
            self._entry_misses = 0
            self._start_entry_wait()
        elif tag == "open_mode_select":
            # 进入按钮点击已发送：确认弹窗打开后再选卡
            self._recruit_step = _RecruitStep.CONFIRM_MODE_SELECT_OPEN
            self._start_entry_wait()
        elif tag == "mode_card_selected":
            # 选卡点击已发送（开环）：确认到达挑战页后再点聊天
            self._recruit_step = _RecruitStep.CONFIRM_CHALLENGE_PAGE
            self._start_entry_wait()
        elif tag == "open_challenge_chat":
            # 首局经挑战页聊天进入招募：页签后进难度弹窗（区别于次局刷新）；
            # 先确认真的离开挑战页（点击可能被道具卡等弹窗吃掉）
            self._after_tab_step = _RecruitStep.OPEN_DIFFICULTY_DIALOG
            self._enter_chat_open_confirm()
        elif tag == "open_mode_tab":
            self._recruit_step = self._after_tab_step
        elif tag in {"open_difficulty_dialog", "open_refresh_difficulty_dialog"}:
            # 打开点击已发送：进入打开确认步骤（settle → 轮询连续命中弹窗标识）。
            # 首局与下一局刷新共用确认，确认成功后按来源分流到各自后续步骤
            self._difficulty_open_settled = False
            self._difficulty_open_hits = 0
            self._difficulty_open_misses = 0
            self._difficulty_open_is_refresh = tag == "open_refresh_difficulty_dialog"
            self._recruit_step = _RecruitStep.CONFIRM_DIFFICULTY_OPEN
        elif tag == "difficulty_open_settle":
            self._difficulty_open_settled = True
        elif tag == "difficulty_open_hit":
            self._difficulty_open_hits += 1
            self._difficulty_open_misses = 0
        elif tag == "difficulty_open_miss":
            self._difficulty_open_misses += 1
            self._difficulty_open_hits = 0
        elif tag == "difficulty_open_retry":
            # 重点点击已发送：重置确认计数，重走 settle → 轮询
            self._difficulty_open_reclicks += 1
            self._difficulty_open_settled = False
            self._difficulty_open_hits = 0
            self._difficulty_open_misses = 0
        elif tag == "difficulty_open_confirmed":
            # 弹窗确认打开：重置关闭确认计数，按来源进入勾选或关闭。
            # 跳过模式只跳过勾选难度等级；被踢回首页后的重新进入同样不再勾选
            # （本会话已选中的难度再次点击会取消勾选）
            self._difficulty_close_attempts = 0
            self._difficulty_close_streak = 0
            if self._difficulty_open_is_refresh:
                self._recruit_step = _RecruitStep.CLOSE_REFRESH_DIFFICULTY_DIALOG
            elif self._skip_difficulty_selection or self._difficulty_done_session:
                self._recruit_step = _RecruitStep.CLOSE_DIFFICULTY_DIALOG
            else:
                self._recruit_step = _RecruitStep.SELECT_DIFFICULTIES
        elif tag.startswith("select_difficulty:"):
            # 开环：点击已发送即视为已选（勾选框状态不识别，见 decide 注释）
            level = int(tag.partition(":")[2])
            self._complete_difficulty_target(level, selected=True)
        elif tag.startswith("difficulty_skip:"):
            level = int(tag.partition(":")[2])
            # 见过彩虹行才重置预算（下个目标可能还要滚动）；从没见过说明
            # 列表根本没进彩虹区，保留耗尽的预算让后续目标直接级联跳过。
            self._complete_difficulty_target(
                level, selected=False, reset_scrolls=self._difficulty_saw_candidates
            )
        elif tag.startswith("scroll_difficulties:"):
            self._difficulty_scroll_count += 1
        elif tag.startswith("select_pet_difficulty:"):
            # 保卫宠物开环勾选：点击已发送即视为已选（不识别勾选框状态）
            target = tag.partition(":")[2]
            if target in self._remaining_pet_targets:
                self._remaining_pet_targets.remove(target)
            self._selected_pet_targets.append(target)
            self._pet_difficulty_misses = 0
            if not self._remaining_pet_targets:
                self._log_pet_difficulty_summary()
                if not self._difficulty_selection_only:
                    self._recruit_step = _RecruitStep.CLOSE_DIFFICULTY_DIALOG
                    self._difficulty_close_attempts = 0
                    self._difficulty_close_streak = 0
        elif tag == "pet_difficulty_miss":
            self._pet_difficulty_misses += 1
        elif tag == "open_coop_chat":
            # 下一局经合作页聊天刷新邀请：同样先确认离开挑战页再选页签
            self._after_tab_step = _RecruitStep.OPEN_REFRESH_DIFFICULTY_DIALOG
            self._enter_chat_open_confirm()
        elif tag in {"close_difficulty_dialog", "close_refresh_difficulty_dialog"}:
            # 点击已发送但弹窗未必关住：留在本步骤，等弹窗标识连续不可见才推进
            self._difficulty_close_attempts += 1
            self._difficulty_close_streak = 0
        elif tag == "difficulty_close_miss_check":
            self._difficulty_close_streak += 1
        elif tag in {"difficulty_dialog_closed", "refresh_dialog_closed"}:
            self._difficulty_close_streak = 0
            self._recruit_step = _RecruitStep.JOIN_COOP
        elif tag == "ready_match":
            self._match_started_at = self._clock()
            self._opening_loaded = False
            self._opening_skill_selections = 0
            self._opening_clicks_blocked = False
            self._opening_wait_count = 0
            self._home_return_count = 0
            self._main_skill_selections = 0
            self._main_skill_selections_total = 0
            self._skill_refresh_used_count = 0
            self._skill_refresh_button_waits = 0
            self._topup_active = False
            self._topup_done = False
            self._summon_retry_at = None
            self._transform_used_count = 0
            self._transform_skip_signature = None
            self._clear_transform_state()
            # 点过准备 = 本会话已完成难度勾选（无论程序选的还是手动选的）
            self._difficulty_done_session = True
        elif tag == "opening_load_wait":
            self._opening_loaded = True
        elif tag == "opening_interface_wait":
            self._opening_wait_count += 1
        elif tag == "opening_skill_empty_check":
            self._opening_empty_checks += 1
            # 技能页面还在（等图标出现）→ 退出确认计数归零
            self._opening_exit_empty_count = 0
        elif tag in {
            "opening_skill_candidate",
            "opening_skill_fallback",
            "opening_angel_skill_fallback",
            "opening_skill_priority",
        }:
            # 开局每选一次（命中或兜底）都重置空识别计数，给页面刷新/下一组技能留时间
            self._opening_empty_checks = 0
            self._opening_exit_empty_count = 0
            self._opening_skill_selections += 1
        elif tag == "opening_exit_confirm_check":
            self._opening_exit_empty_count += 1
        elif tag == "home_return_check":
            self._home_return_count += 1
        elif tag in {"kick_reentry_home", "match_start_timeout"}:
            # 本局未开始即被取消：清空对局内进度，从首页重新走招募入口
            self._reset_for_home_reentry()
        elif tag == "leave_team":
            # 主动退队：清空对局内进度，先探测落点页面再选招募入口
            self._reset_for_home_reentry()
            self._recruit_step = _RecruitStep.DETECT_ENTRY_PAGE
        elif tag in {"required_summon", "summon_hero", "batch_summon"}:
            self._summon_count += 1
            if tag == "batch_summon":
                # 批次召唤点击已发送：消耗批次；归零后下一轮恢复多帧识别
                self._batch_summon_remaining -= 1
        elif tag == "merge_heroes":
            # 合成成功 = 拖动机制正常；清空失败记忆，让之前的失败对可重试
            self._failed_merge_pairs.clear()
        elif tag == "batch_merge":
            # 批次合成单拖已发送：弹出队首；拖动生效同样证明机制正常，
            # 清空失败记忆（与单动作合成语义一致）
            self._batch_merge_queue.pop(0)
            self._failed_merge_pairs.clear()
        elif tag == "main_c_ready":
            # 数量已达标：清空技能选择计数，开始新一轮「选 N 次 → 检查数量」；
            # 同时丢弃可能残留的批次状态（培养已结束，批次不再有意义）
            self._main_skill_selections = 0
            self._topup_active = False
            self._batch_summon_remaining = 0
            self._batch_merge_queue = []
            self._pending_merge_batch_signatures = []
            self._schedule_next_skill(self._clock())
        elif tag == "main_c_topup":
            # 进入数量回补阶段：培养门禁从「只看星级」收紧为「星级 + 数量」
            self._topup_active = True
        elif tag == "close_popup":
            # 弹窗遮挡期间技能图标不可见；关闭后重置空识别计数，给技能页重新稳定、
            # 图标重新识别留时间，避免一关弹窗就立刻触发「随便选一个」
            self._main_skill_empty_checks = 0
            self._main_page_missing_checks = 0
            self._opening_empty_checks = 0
        elif tag == "open_main_skills":
            self._awaiting_main_candidates = True
            self._main_skill_empty_checks = 0
            self._main_page_missing_checks = 0
            # 新一次打开技能页：按钮等待重试预算从零开始
            self._skill_refresh_button_waits = 0
        elif tag == "main_skill_empty_check":
            self._main_skill_empty_checks += 1
        elif tag == "main_skill_page_missing_check":
            self._main_page_missing_checks += 1
        elif tag in {"main_skill_candidate", "main_skill_fallback", "main_skill_priority"}:
            self._awaiting_main_candidates = False
            self._main_page_missing_checks = 0
            self._skill_refresh_button_waits = 0
            self._main_skill_selections += 1
            self._main_skill_selections_total += 1
            self._schedule_next_skill(self._clock())
        elif tag in {
            "merge_gift_skill_candidate",
            "merge_gift_skill_fallback",
            "merge_gift_skill_priority",
        }:
            # 赠送技能页选卡完成、页面关闭：结束 3 星合并后的等待状态
            self._awaiting_merge_gift = False
            self._merge_gift_settled = False
            self._merge_gift_hits = 0
            self._merge_gift_misses = 0
        elif tag == "return_result":
            self.ctx.round_count += 1
            if to_state == State.FIND_COOP:
                self._reset_for_next_round()

    def on_action_failed(self, action: Action) -> None:
        """动作重试耗尽、Runner 放弃该动作时调整内部进度，均不结束任务。"""
        tag = action.tag
        if tag == "skill_refresh":
            # 刷新点击未生效（或换出同三张）：本局机会按已用处理，页面仍开时
            # 下一帧直接按优先级选卡——刷新不可用后必须选一张
            self._skill_refresh_used_count += 1
            logger.warning(
                "技能页刷新未生效，本局不再刷新（已用 %d/%d），直接选卡",
                self._skill_refresh_used_count,
                self._run_config.skill_refresh_max_per_round,
            )
        elif tag in {"main_skill_candidate", "main_skill_fallback", "main_skill_priority"}:
            # 本次局内选技能未生效：停止等待图标，按随机间隔稍后重新打开技能页
            self._awaiting_main_candidates = False
            self._main_skill_empty_checks = 0
            self._main_page_missing_checks = 0
            self._schedule_next_skill(self._clock())
        elif tag in {
            "opening_skill_candidate",
            "opening_skill_fallback",
            "opening_angel_skill_fallback",
            "opening_skill_priority",
        }:
            # 开局技能点击多次未生效：不再盲点，等页面变化后再重新尝试选择
            self._opening_clicks_blocked = True
        elif tag == "merge_heroes":
            # 合成拖动重试耗尽（英雄被弹回原位）：记住该失败对，本轮内跳过；
            # 棋盘后续演化（召唤/其他合成成功）出新的对再尝试
            if action.target is not None and action.end is not None:
                self._failed_merge_pairs.add((action.target, action.end))
        elif tag == "summon_hero":
            if self._topup_active:
                # 回补阶段召唤不动（金币不足）：放弃本次回补回选技能；
                # 对局继续、金币随击杀恢复，选满 4 次技能后会再尝试回补
                logger.warning("回补阶段召唤未生效（可能金币不足），暂停回补，回选技能继续对局")
                self._topup_active = False
                self._main_skill_selections = 0
            else:
                # 培养阶段：等待一段时间（金币恢复）再重新尝试召唤
                self._summon_retry_at = self._clock() + self._run_config.summon_retry_delay_seconds

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _schedule_next_skill(self, now: float) -> None:
        """下次金币检查时间：按金币检查间隔固定节奏，不再随机抖动。"""
        self._next_skill_at = now + self._run_config.gold_check_interval_seconds

    def _reset_for_next_round(self) -> None:
        self._reset_for_home_reentry()
        # 结算返回后停在合作页面：从合作页面聊天进入下一轮邀请刷新
        self._recruit_step = _RecruitStep.OPEN_COOP_CHAT

    def _reset_for_home_reentry(self) -> None:
        """本局未开始即被取消（被踢回首页/等待超时）：清空对局内进度。

        与结算后重置不同，此时画面在游戏首页，招募入口从领地入口重新开始；
        难度勾选不重复（_difficulty_done_session 保持已置位）。
        """
        self._recruit_step = _RecruitStep.OPEN_TERRITORY
        self._entry_reclicks = 0
        self._entry_misses = 0
        self._start_entry_wait()
        self._summon_count = 0
        self._batch_summon_remaining = 0
        self._batch_merge_queue = []
        self._pending_merge_batch_signatures = []
        self._empty_board_count = 0
        self._no_progress_count = 0
        self._last_board_signature = None
        self._failed_merge_pairs = set()
        self._awaiting_merge_gift = False
        self._merge_gift_settled = False
        self._merge_gift_hits = 0
        self._merge_gift_misses = 0
        self._summon_retry_at = None
        self._opening_loaded = False
        self._opening_empty_checks = 0
        self._opening_exit_empty_count = 0
        self._opening_skill_selections = 0
        self._opening_clicks_blocked = False
        self._opening_wait_count = 0
        self._home_return_count = 0
        self._match_started_at = None
        self._next_skill_at = None
        self._awaiting_main_candidates = False
        self._main_skill_empty_checks = 0
        self._main_page_missing_checks = 0
        self._main_skill_selections = 0
        self._main_skill_selections_total = 0
        self._skill_refresh_used_count = 0
        self._skill_refresh_button_waits = 0
        self._topup_active = False
        self._topup_done = False
        self._transform_used_count = 0
        self._transform_skip_signature = None
        self._clear_transform_state()

    @staticmethod
    def _board_signature(board: BoardSnapshot) -> tuple:
        return tuple(
            sorted((hero.hero_type, hero.star_level, hero.position) for hero in board.heroes)
        )

    def _boards_changed(
        self,
        before: BoardSnapshot | None,
        after: BoardSnapshot | None,
    ) -> bool:
        if before is None or after is None or not after.heroes:
            return False
        return self._board_signature(before) != self._board_signature(after)

    @staticmethod
    def _skill_signature(observation: Observation) -> tuple:
        return tuple(
            sorted(
                (candidate.skill_id, candidate.position)
                for candidate in observation.skill_candidates
            )
        )

    @staticmethod
    def _difficulty_signature(observation: Observation) -> tuple[int, ...]:
        return tuple(sorted(item.level for item in observation.difficulty_candidates))

    @staticmethod
    def _flag_match(observation: Observation, name: str) -> MatchResult | None:
        value = observation.flag(name, None)
        return value if isinstance(value, MatchResult) else None

    def _like_chest_enabled(self) -> bool:
        """结算是否执行点赞与领取宝箱（保卫宠物为 False，只点返回）。"""
        return self._mode is None or self._mode.settlement.like_chest

    def _hotspot_click(
        self,
        name: str,
        tag: str,
        reason: str,
        to_state: State,
        window_ctx: WindowContext | None,
        *,
        verification: Literal["immediate", "next_frame"] = "immediate",
        post_delay: float = 0.0,
    ) -> tuple[Action, State] | None:
        point = self._hotspot_point(name, window_ctx)
        if point is None:
            return None
        return (
            Action(
                kind="click",
                target=point,
                duration=0.08,
                post_delay=post_delay,
                verification=verification,
                tag=tag,
                reason=reason,
            ),
            to_state,
        )

    def _hotspot_point(
        self,
        name: str,
        window_ctx: WindowContext | None,
    ) -> tuple[int, int] | None:
        spot = self._hotspots.get(name)
        if spot is None or window_ctx is None:
            return None
        return hotspot_to_client_point(spot, window_ctx.client_size)
