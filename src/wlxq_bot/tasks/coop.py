"""合作任务状态机与局内决策。"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from enum import Enum
from typing import Literal

from wlxq_bot.config import Hotspot, MainCProfile, RoiConfig, RunConfig
from wlxq_bot.models import (
    Action,
    BoardHero,
    BoardSnapshot,
    CoopRole,
    MatchResult,
    Observation,
    SkillCandidate,
    State,
    Transition,
    WindowContext,
)
from wlxq_bot.perception.locator import hotspot_to_client_point, roi_column_centers
from wlxq_bot.tasks.base import Task, TaskContext
from wlxq_bot.tasks.strategy import CultivationStrategy
from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)


def _hero_location(hero: BoardHero) -> str:
    """日志用英雄位置描述：有格名用格名（如 1A），否则退回像素坐标。"""
    return hero.cell_name or str(tuple(hero.position))


_MATCH_STATES: frozenset[State] = frozenset(
    {
        State.ENTER_MATCH,
        State.SELECT_OPENING_SKILLS,
        State.BUILD_MAIN_C,
        State.SELECT_MAIN_C_SKILLS,
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


class _RecruitStep(Enum):
    """`FIND_COOP` 内部的固定招募业务步骤。"""

    VERIFY_HOME_PAGE = "verify_home_page"
    OPEN_RECRUIT = "open_recruit"
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
        main_c_profile: MainCProfile | None = None,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(ctx)
        self._run_config = run_config
        self._role = role
        self._hotspots = hotspots
        self._skill_candidate_roi = skill_candidate_roi
        # 主C培养策略（合成/门禁/数量回补）：纯决策对象，状态机只负责调度
        self._strategy = CultivationStrategy(ctx.main_c, run_config, main_c_profile)
        self._remaining_difficulties = set(coop_difficulties or self._run_config.coop_difficulties)
        self._difficulty_selection_only = difficulty_selection_only
        # 游戏本次会话已手动选过难度：首局招募跳过难度弹窗三步直接抢合作
        self._skip_difficulty_selection = skip_difficulty_selection
        # 难度选择结果追踪，用于结尾汇总「选中 / 未识别跳过」
        self._selected_difficulties: set[int] = set()
        self._skipped_difficulties: set[int] = set()
        self._difficulty_summary_logged = False
        self._clock = clock
        self._rng = rng or random.Random()

        self._recruit_step = (
            _RecruitStep.SELECT_DIFFICULTIES
            if difficulty_selection_only
            else _RecruitStep.VERIFY_HOME_PAGE
        )
        self._difficulty_scroll_count = 0
        # 本目标难度的尝试期间是否见过彩虹候选：决定跳过后是否重置滚动预算
        self._difficulty_saw_candidates = False
        # 弹窗打开确认：settle → 轮询连续命中【合作模式】标题图（见
        # _action_confirm_difficulty_open）；重点打开按钮时全部重置
        self._difficulty_open_settled = False
        self._difficulty_open_hits = 0
        self._difficulty_open_misses = 0
        self._difficulty_open_reclicks = 0
        # 本次打开确认来自下一局刷新路径（确认后进 CLOSE_REFRESH 而非勾选）
        self._difficulty_open_is_refresh = False
        # 弹窗关闭确认：连续不可见标题图（见 _action_close_difficulty_dialog）
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

        self._match_started_at: float | None = None
        self._next_skill_at: float | None = None
        self._awaiting_main_candidates = False
        self._main_skill_empty_checks = 0
        self._main_page_missing_checks = 0
        self._main_skill_selections = 0
        # 本局局内技能选择总次数（回补重置不清零；达到档案上限后停止选技能）
        self._main_skill_selections_total = 0
        # 处于数量回补阶段（选满 N 次技能回到培养）：此时培养门禁才检查主C数量；
        # 首次培养只看星级，让技能选择尽早开始
        self._topup_active = False

    @property
    def skipped_difficulties(self) -> tuple[int, ...]:
        """诊断难度选择中未识别到的目标，按从大到小返回。"""
        return tuple(sorted(self._skipped_difficulties, reverse=True))

    def build_transitions(self) -> list[Transition]:
        """返回顶层业务阶段图；招募入口等小步骤保留在状态内部。"""
        wait = Action(kind="wait", duration=0.3, verification="immediate", reason="占位动作")
        return [
            Transition("find_coop", State.FIND_COOP, State.ENTER_MATCH, wait),
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
        if state == State.ENTER_MATCH:
            return self._action_ready(observation)
        if state == State.SELECT_OPENING_SKILLS:
            return self._action_opening_skills(observation, window_ctx)
        if state == State.BUILD_MAIN_C:
            return self._action_build_main_c(observation, window_ctx)
        if state == State.SELECT_MAIN_C_SKILLS:
            return self._action_main_c_skills(observation, window_ctx)
        if state == State.HANDLE_RESULT:
            return self._hotspot_click(
                "like",
                "like_result",
                "结算页点赞",
                State.CLAIM_REWARD,
                window_ctx,
            )
        if state == State.CLAIM_REWARD:
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
        # 跳过多帧棋盘识别，按固定间隔快速连点；第 5 次后恢复正常识别。
        # 等待4星赠送技能页期间页面遮住棋盘、识别必然为空，同样退回单帧
        # 快识别（BUILD 状态标志集含标题图与击杀弹窗，弹窗优先不受影响）
        if self._awaiting_merge_gift:
            return False
        return self._summon_count >= self._run_config.minimum_summon_count_before_skills

    def observation_mode(self) -> str | None:
        if self._recruit_step in (_RecruitStep.VERIFY_HOME_PAGE, _RecruitStep.DETECT_ENTRY_PAGE):
            return "home_page"
        if self._recruit_step is _RecruitStep.SELECT_DIFFICULTIES:
            # 勾选难度需要识别候选列表（16 个难度模板）
            return "coop_difficulty"
        if self._recruit_step in (
            _RecruitStep.CONFIRM_DIFFICULTY_OPEN,
            _RecruitStep.CLOSE_DIFFICULTY_DIALOG,
            _RecruitStep.CLOSE_REFRESH_DIFFICULTY_DIALOG,
        ):
            # 打开/关闭确认只需【合作模式】标题图判定弹窗开关，不做 16 个
            # 难度候选匹配：候选匹配会把单帧识别拖慢到超过截图时效
            # （实机 2026-08-17：慢速机器上每帧 6 秒导致动作全部被时效拒绝）
            return "difficulty_dialog"
        if self._recruit_step is _RecruitStep.JOIN_COOP:
            # 抢合作：连点加入期间游戏随时可能放行，准备按钮出现即抢到。
            # 合作只能自己点加入进入（游戏不会主动拉人，实机确认 2026-08-17），
            # 其余入口子步骤不查任何全局标志
            return "coop_grab"
        return None

    def _action_find_coop(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        if self._recruit_step != _RecruitStep.JOIN_COOP:
            return self._action_recruit_entry(observation, window_ctx)
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

    def _action_recruit_entry(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        """按代码中确认的首次/下一局入口步骤生成一个安全动作。"""
        if self._recruit_step == _RecruitStep.DETECT_ENTRY_PAGE:
            # 退队后的落点未实机确认（可能首页、可能合作页面）：先识别首页标志
            # 再选入口，两条路径都不重复勾选难度（本会话已选中）
            if observation.flag("home_page_visible"):
                logger.info("退队后识别到游戏首页，从首页进入招募")
                self._recruit_step = _RecruitStep.VERIFY_HOME_PAGE
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
        if self._recruit_step == _RecruitStep.VERIFY_HOME_PAGE:
            if not observation.flag("home_page_visible"):
                logger.error(
                    "frame=%d 合作任务启动失败：当前界面未识别为游戏首页",
                    observation.frame_id,
                )
                raise RuntimeError("当前界面不是游戏首页，请返回游戏首页后重新运行合作任务")
            return self._hotspot_click(
                "home_chat",
                "open_home_chat",
                "已确认游戏首页，点击首页聊天",
                State.FIND_COOP,
                window_ctx,
            )
        if self._recruit_step == _RecruitStep.OPEN_RECRUIT:
            return self._hotspot_click(
                "open_recruit",
                "open_recruit",
                "首次进入招募：点击招募",
                State.FIND_COOP,
                window_ctx,
            )
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
            return self._hotspot_click(
                "coop_chat",
                "open_coop_chat",
                "下一局进入招募：点击合作页面聊天",
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
        吸收时序差异，再每 ``difficulty_poll_interval_seconds`` 识别一次【合作模式】
        标题图，连续 ``difficulty_open_confirm_hits`` 次可见才算打开（弹出动画帧
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
                    reason="难度弹窗已确认打开（标题图连续命中）",
                ),
                State.FIND_COOP,
            )
        if self._difficulty_open_misses >= cfg.difficulty_open_fail_misses:
            if self._difficulty_open_reclicks >= cfg.difficulty_open_max_reclicks:
                logger.error(
                    "难度弹窗重点 %d 次后仍识别不到【合作模式】标识，保守停止 frame=%d",
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
                    reason=f"难度弹窗标题图可见（连续 "
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
                reason=f"难度弹窗标题图不可见（连续 "
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
        """关闭难度弹窗，并以【合作模式】标题图（he_zuo_mo_shi）判定弹窗开关。

        打开确认步骤已保证弹窗真的打开过，这里统一按「连续
        ``difficulty_close_confirm_frames`` 次识别不到标题图」判定关闭（间隔
        ``difficulty_poll_interval_seconds``）：单帧漏检不再致命；收起动画期间
        标题图可见只触发再次点击（实机确认：面板已收起时再点招募按钮无反应，
        重复点击无副作用）。标题图可见 = 没关住，重点关闭按钮，最多
        ``difficulty_close_max_attempts`` 次，仍可见则保守停止——绝不带着未关闭
        的弹窗进入抢合作（弹窗挡住 join_coop，连点还会落在难度行上误改勾选）。
        """
        cfg = self._run_config
        if observation.flag("difficulty_dialog_visible"):
            if self._difficulty_close_attempts >= cfg.difficulty_close_max_attempts:
                logger.error(
                    "难度弹窗关闭点击 %d 次后仍识别到【合作模式】标识，保守停止 frame=%d",
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
        # 补到目标以上（棋盘数量判断在 BUILD_MAIN_C，该状态才识别棋盘）
        if self._strategy.should_topup(self._main_skill_selections):
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
            if observation.skill_candidates:
                return self._choose_skill_action(
                    observation.skill_candidates,
                    "main_skill_candidate",
                    State.SELECT_MAIN_C_SKILLS,
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
                    reason="等待下一次随机技能检查时间",
                ),
                State.SELECT_MAIN_C_SKILLS,
            )

        return self._hotspot_click(
            "select_skill",
            "open_main_skills",
            "定时点击选技能按钮",
            State.SELECT_MAIN_C_SKILLS,
            window_ctx,
        )

    def _choose_skill_action(
        self,
        candidates: list[SkillCandidate],
        tag: str,
        to_state: State,
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
            ),
            to_state,
        )

    def _skill_fallback_action(
        self,
        tag: str,
        to_state: State,
        window_ctx: WindowContext | None,
        *,
        reason: str = "未识别到主C技能，随便选一个",
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

        board = observation.board
        minimum_summons = self._run_config.minimum_summon_count_before_skills
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
                    reason=f"技能解锁前逐次召唤 {next_summon}/{minimum_summons}",
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

        candidates = board.find_merge_candidates(self.ctx.main_c)
        # 合成/避让/失败对跳过等策略决策由 CultivationStrategy 负责；
        # 返回 None 表示本次不合并，落入下方召唤分支
        pair = self._strategy.select_merge(board, candidates, self._failed_merge_pairs)
        if pair is not None:
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
                    reason=(
                        "优先拖动合成非主C合法对"
                        if not pair.is_main_c
                        else "无其他合法对，拖动合成主C对"
                    ),
                ),
                State.BUILD_MAIN_C,
            )

        point = self._hotspot_point("add_hero", window_ctx)
        if point is None:
            return None
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
                reason="无合法合成对，召唤1个英雄",
            ),
            State.BUILD_MAIN_C,
        )

    # ------------------------------------------------------------------
    # 结算和动作后验证
    # ------------------------------------------------------------------

    def _action_return(self, observation: Observation) -> tuple[Action, State] | None:
        match = self._flag_match(observation, "return_button_match")
        if match is None:
            return None
        to_state = (
            State.COMPLETED if self.ctx.round_count + 1 >= self.ctx.max_rounds else State.FIND_COOP
        )
        return (
            Action(
                kind="click",
                target=match.position,
                duration=0.08,
                verification="next_frame",
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
        if action.tag == "close_double_reward":
            # 以双倍奖励弹窗标识消失确认已取消
            return not after.flag("double_reward_dialog_visible")
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
        if tag == "open_home_chat":
            self._recruit_step = _RecruitStep.OPEN_RECRUIT
        elif tag == "open_recruit":
            self._recruit_step = _RecruitStep.OPEN_DIFFICULTY_DIALOG
        elif tag in {"open_difficulty_dialog", "open_refresh_difficulty_dialog"}:
            # 打开点击已发送：进入打开确认步骤（settle → 轮询连续命中标题图）。
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
        elif tag == "open_coop_chat":
            self._recruit_step = _RecruitStep.OPEN_REFRESH_DIFFICULTY_DIALOG
        elif tag in {"close_difficulty_dialog", "close_refresh_difficulty_dialog"}:
            # 点击已发送但弹窗未必关住：留在本步骤，等标题图连续不可见才推进
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
            self._topup_active = False
            self._summon_retry_at = None
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
        elif tag in {"required_summon", "summon_hero"}:
            self._summon_count += 1
        elif tag == "merge_heroes":
            # 合成成功 = 拖动机制正常；清空失败记忆，让之前的失败对可重试
            self._failed_merge_pairs.clear()
        elif tag == "main_c_ready":
            # 数量已达标：清空技能选择计数，开始新一轮「选 N 次 → 检查数量」
            self._main_skill_selections = 0
            self._topup_active = False
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
        elif tag == "main_skill_empty_check":
            self._main_skill_empty_checks += 1
        elif tag == "main_skill_page_missing_check":
            self._main_page_missing_checks += 1
        elif tag in {"main_skill_candidate", "main_skill_fallback"}:
            self._awaiting_main_candidates = False
            self._main_page_missing_checks = 0
            self._main_skill_selections += 1
            self._main_skill_selections_total += 1
            self._schedule_next_skill(self._clock())
        elif tag in {"merge_gift_skill_candidate", "merge_gift_skill_fallback"}:
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
        if tag in {"main_skill_candidate", "main_skill_fallback"}:
            # 本次局内选技能未生效：停止等待图标，按随机间隔稍后重新打开技能页
            self._awaiting_main_candidates = False
            self._main_skill_empty_checks = 0
            self._main_page_missing_checks = 0
            self._schedule_next_skill(self._clock())
        elif tag in {
            "opening_skill_candidate",
            "opening_skill_fallback",
            "opening_angel_skill_fallback",
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
        elapsed = 0.0 if self._match_started_at is None else now - self._match_started_at
        if elapsed < self._run_config.skill_late_after_seconds:
            low = self._run_config.skill_early_interval_min
            high = self._run_config.skill_early_interval_max
        else:
            low = self._run_config.skill_late_interval_min
            high = self._run_config.skill_late_interval_max
        self._next_skill_at = now + self._rng.uniform(low, high)

    def _reset_for_next_round(self) -> None:
        self._reset_for_home_reentry()
        # 结算返回后停在合作页面：从合作页面聊天进入下一轮邀请刷新
        self._recruit_step = _RecruitStep.OPEN_COOP_CHAT

    def _reset_for_home_reentry(self) -> None:
        """本局未开始即被取消（被踢回首页/等待超时）：清空对局内进度。

        与结算后重置不同，此时画面在游戏首页，招募入口从首页确认重新开始；
        难度勾选不重复（_difficulty_done_session 保持已置位）。
        """
        self._recruit_step = _RecruitStep.VERIFY_HOME_PAGE
        self._summon_count = 0
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
        self._topup_active = False

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
