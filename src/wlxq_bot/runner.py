"""Runner：加载配置、初始化运行环境、调度任务循环。

依赖方向：CLI -> Runner -> Task Engine(Perception/Action)。
Runner 负责构造 ScreenCapture / Vision / CoopPerception / ActionExecutor /
SafetyGuard / CoopTask，并驱动「截图 -> 识别 -> 状态判断 -> 决策 -> 执行」循环。

循环逻辑放在 _run_loop，接受已构造的依赖，便于用 FakeInput / FakeCapture
做单元测试，不真实点击用户桌面。

输入动作分为立即提交动作和严格动作：前者在发送成功并完成稳定等待后提交内部计数，
后者保留前一帧并在后续截图中验证业务后置条件，验证失败时停止。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wlxq_bot.action.executor import ActionExecutor
from wlxq_bot.action.input import InputController
from wlxq_bot.action.safety import SafetyGuard
from wlxq_bot.assets import TemplatePack
from wlxq_bot.config import (
    DefaultConfig,
    LocalConfig,
    RunConfig,
    TasksConfig,
    parse_coop_difficulties,
)
from wlxq_bot.debug.recorder import DebugRecorder
from wlxq_bot.models import Action, CoopRole, Observation, State
from wlxq_bot.orchestration.coop_grab import CoopGrabCoordinator
from wlxq_bot.perception.coop import _BOARD_WATCH_STATES, CoopPerception
from wlxq_bot.perception.hero_classifier import HeroCellClassifier
from wlxq_bot.perception.screen import (
    ScreenCapture,
    activate_window,
    adjust_window_size,
    get_input_idle_seconds,
    get_window_monitor_resolution,
)
from wlxq_bot.perception.skill_collector import SkillCollector
from wlxq_bot.perception.vision import Vision
from wlxq_bot.tasks.base import Task, TaskContext
from wlxq_bot.tasks.coop import CoopTask
from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

# 模板包根目录
TEMPLATES_ROOT = Path("assets/templates")

# 窗口非前台/最小化时的检查间隔（秒）；日志每 _FG_LOG_EVERY_CHECKS 次提示一次
_FG_CHECK_INTERVAL_SECONDS = 0.5
_FG_LOG_EVERY_CHECKS = 20  # 20 × 0.5s = 每 10 秒一条
# 自动切回前台失败后的重试间隔（秒），避免高频反复激活
_REFOCUS_RETRY_SECONDS = 5.0
# 连续多少轮「识别决策耗时超过截图时效」后触发原地恢复（丢弃超龄决策重新
# 识别，连续恢复有独立预算上限）。识别慢通常是机器负载、省电模式等临时状况；
# 持续超龄说明本机无法在时效内完成识别->动作。3 帧 ≈ 25 秒内即触发恢复，
# 不必苦等 10 帧（实机 2026-09-05 定案：10 次太久）
_MAX_STALE_DECISIONS = 3


# 单次任务最大步数上限，防止无限循环（首版安全网）
@dataclass
class _PendingVerification:
    action: Action
    to_state: State
    before: Observation
    attempts: int = 0


# 技能卡点击类动作：点击未生效可恢复（页面还在，重新决策再选即可），且失败
# 不代表对局结束——游戏仍在自动进行，结算窗口终将出现，重试耗尽也不结束任务
_SKILL_CLICK_TAGS = frozenset(
    {
        "opening_skill_candidate",
        "opening_skill_fallback",
        "opening_angel_skill_fallback",
        "opening_skill_priority",
        "main_skill_candidate",
        "main_skill_fallback",
        "main_skill_priority",
        "merge_gift_skill_candidate",
        "merge_gift_skill_fallback",
        "merge_gift_skill_priority",
    }
)


def _log_merge_verify_failure(
    pending: _PendingVerification,
    observation: Observation,
) -> None:
    """合成验证失败时输出棋盘细节，便于区分拖动落空和识别异常。"""
    action = pending.action
    before_board = pending.before.board
    after_board = observation.board
    if before_board is None or after_board is None:
        logger.warning(
            "合成验证失败详情 frame=%d: 识别无棋盘（before=%s after=%s）",
            observation.frame_id,
            before_board is not None,
            after_board is not None,
        )
        return
    source = next(
        (hero for hero in before_board.heroes if hero.position == action.target),
        None,
    )
    expected_star = source.star_level + 1 if source else None
    source_still = any(hero.position == action.target for hero in after_board.heroes)
    end_hero = next(
        (hero for hero in after_board.heroes if hero.position == action.end),
        None,
    )
    logger.warning(
        "合成验证失败详情 frame=%d: 起点%s英雄仍在=%s 终点格=%s 预期%d星 棋盘英雄数 %d->%d",
        observation.frame_id,
        action.target,
        source_still,
        (f"{end_hero.hero_type}{end_hero.star_level}星" if end_hero else "无英雄"),
        expected_star if expected_star is not None else -1,
        len(before_board.heroes),
        len(after_board.heroes),
    )


@dataclass
class Runner:
    """任务运行调度器。

    Args:
        default_config: configs/default.yaml
        tasks_config: configs/tasks.yaml
        local_config: configs/local.yaml（仅窗口规格和模板包覆盖），可为 None
    """

    default_config: DefaultConfig
    tasks_config: TasksConfig
    local_config: LocalConfig | None = None
    # 测试注入：固定模板包，跳过模板包选择
    _template_pack: TemplatePack | None = field(default=None, repr=False)

    def run(
        self,
        task_name: str,
        main_c: str | None = None,
        start_state: str = "find_coop",
        coop_difficulties: str | None = None,
        skip_difficulty_selection: bool | None = None,
        max_rounds: int | None = None,
    ) -> State:
        """执行指定任务。

        Args:
            task_name: 任务名称，目前只支持 "coop"
            main_c: 主 C 标识，未指定时用 default_main_c
            start_state: 初始状态（State 枚举值字符串），调试用。
                默认 find_coop；传 build_main_c 可跳过抢合作直接测培养闭环。
            coop_difficulties: 命令行难度范围覆盖，例如 ``1-10``；None 使用配置值。
            max_rounds: 命令行局数覆盖；None 使用配置值。仅本次运行生效。
            skip_difficulty_selection: 命令行跳过难度选择覆盖；None 使用配置值。
                游戏本次会话已手动选过难度时置 True：首局招募只跳过勾选难度等级，
                难度弹窗仍开/关一次刷新最新合作邀请。

        Returns:
            任务结束时的最终状态

        Raises:
            ValueError: 不支持的任务名
            RuntimeError: 窗口未找到、模板包缺失或启动检查失败
        """
        if task_name != "coop":
            raise ValueError(f"不支持的任务: {task_name}（目前仅支持 coop）")

        try:
            initial_state = State(start_state)
        except ValueError as exc:
            raise ValueError(
                f"无效的 start_state: {start_state!r}，应为 {', '.join(s.value for s in State)}"
            ) from exc

        mc = main_c or self.default_config.run.default_main_c
        difficulty_levels = (
            parse_coop_difficulties(coop_difficulties)
            if coop_difficulties is not None
            else list(self.default_config.run.coop_difficulties)
        )
        skip_difficulty = (
            self.default_config.run.skip_difficulty_selection
            if skip_difficulty_selection is None
            else skip_difficulty_selection
        )
        run_config = self.default_config.run
        if max_rounds is not None:
            if max_rounds < 1:
                raise ValueError(f"max_rounds 必须大于等于 1，当前 {max_rounds}")
            run_config = run_config.model_copy(update={"max_rounds": max_rounds})
        self._startup_check(mc, initial_state)
        logger.info(
            "启动合作任务，主C=%s，合作难度=%s，跳过难度选择=%s，最大局数=%d",
            mc,
            difficulty_levels,
            skip_difficulty,
            run_config.max_rounds,
        )

        screen = ScreenCapture()
        handle = self._find_game_window(screen)
        if handle is None:
            raise RuntimeError(
                f"未找到游戏窗口，请确认游戏已启动且窗口标题含 「{self._window_title()}」"
            )

        # 激活游戏窗口到前台，给用户切换时间
        if activate_window(handle):
            logger.info("已激活游戏窗口到前台")
        else:
            logger.warning("激活游戏窗口失败，请手动切换到游戏窗口前台")
        # 客户区与本机配置不一致时自动调整（等价 adjust-window），省一次手动操作
        self._ensure_window_size(screen, handle)
        logger.info("3 秒后开始执行，请勿切换窗口")
        time.sleep(3)

        pack = self._load_template_pack(screen, handle)
        vision = Vision()
        debug_recorder = (
            DebugRecorder(
                self.default_config.vision.debug_dir,
                exit_frame_buffer_size=self.default_config.vision.exit_frame_buffer_size,
            )
            if self.default_config.vision.debug
            else None
        )
        hero_cell_classifier = self._load_hero_cell_classifier(mc)
        skill_collector = self._build_skill_collector(run_config)
        title_reader, skill_tiers, new_skill_dir = self._build_skill_priority(mc)
        gold_reader, gold_read_interval = self._build_gold_reader()
        perception = CoopPerception(
            vision,
            pack,
            self.tasks_config,
            CoopRole.HELPER,
            mc,
            debug_recorder=debug_recorder,
            skill_icon_templates=self.default_config.main_c_profiles[mc].skill_icon_templates,
            teammate_skill_icon_templates=run_config.teammate_skill_icon_templates,
            hero_cell_classifier=hero_cell_classifier,
            allowed_heroes={mc, *self.default_config.hero_classifier.lineup_others},
            skill_collector=skill_collector,
            title_reader=title_reader,
            skill_tiers=skill_tiers,
            new_skill_dir=new_skill_dir,
            gold_reader=gold_reader,
            gold_read_interval=gold_read_interval,
        )

        required_heroes = {mc, *self.default_config.hero_classifier.lineup_others}
        missing_model_heroes = sorted(required_heroes - set(perception.available_heroes))
        if missing_model_heroes:
            raise RuntimeError(
                "英雄格模型缺少本局阵容类别: "
                + ", ".join(missing_model_heroes)
                + f"（模型: {self.default_config.main_c_profiles[mc].hero_classifier_model}）"
            )
        self._check_skill_templates(mc, initial_state, pack)
        self._check_home_page_template(initial_state, pack)
        self._check_difficulty_templates(
            difficulty_levels, initial_state, pack, skip_difficulty=skip_difficulty
        )

        safety = SafetyGuard(
            max_failures=self.default_config.safety.max_failures,
            frame_ttl_ms=self.default_config.safety.frame_ttl_ms,
        )
        input_ctrl = InputController()
        executor = ActionExecutor(
            safety,
            input_ctrl,
            min_delay=self.default_config.input.min_delay,
            max_delay=self.default_config.input.max_delay,
            context_validator=screen.validate_context,
        )

        # 复制一份，避免任务侧修改污染已加载的共享任务配置。
        hotspots = dict(self.tasks_config.hotspots)
        skill_roi_name = str(self.tasks_config.skills.get("candidate_roi", "skill_candidates"))
        skill_candidate_roi = self.tasks_config.rois.get(skill_roi_name)
        ctx = TaskContext(
            main_c=mc,
            max_rounds=run_config.max_rounds,
        )
        task = CoopTask(
            ctx,
            run_config,
            CoopRole.HELPER,
            hotspots,
            coop_difficulties=difficulty_levels,
            skill_candidate_roi=skill_candidate_roi,
            skip_difficulty_selection=skip_difficulty,
            main_c_profile=self.default_config.main_c_profiles[mc],
        )
        task.ctx.current_state = initial_state
        logger.info(
            "初始状态=%s，角色=helper，可用英雄=%s",
            task.ctx.current_state.value,
            perception.available_heroes,
        )

        # 抢合作专用执行器：连点 join_coop 的拟人间隔独立配置，不影响
        # 召唤/选技能/点准备等动作的节奏。检查线程用独立的 ScreenCapture。
        grab_executor = ActionExecutor(
            safety,
            InputController(),
            min_delay=self.default_config.run.find_coop_click_delay_min,
            max_delay=self.default_config.run.find_coop_click_delay_max,
            context_validator=screen.validate_context,
        )
        grab_coordinator = CoopGrabCoordinator(
            screen=ScreenCapture(),
            perception=perception,
            grab_executor=grab_executor,
            safety=safety,
            hotspots=hotspots,
            run_config=run_config,
            window_handle=handle,
            debug_recorder=debug_recorder,
        )

        # 启动 Esc 停止热键监听，按下 Esc 后下一轮循环即停止
        if safety.start_esc_listener():
            logger.info("已启动 Esc 停止监听，运行中按 Esc 可停止任务")
        else:
            logger.warning("Esc 停止监听启动失败，可用 Ctrl+C 中断")
        try:
            final = self._run_loop(
                screen,
                perception,
                executor,
                task,
                safety,
                handle,
                grab_coordinator=grab_coordinator,
                debug_recorder=debug_recorder,
            )
        except KeyboardInterrupt:
            # Ctrl+C 用户主动中断：按约定不保存退出帧（清空缓冲后原样上抛）
            if debug_recorder is not None:
                debug_recorder.drain_exit_frames()
            raise
        finally:
            # 循环正常结束或异常崩溃在此落盘退出帧；Esc（stop_requested）是
            # 用户主动停止，Ctrl+C 的缓冲已在 except 清空，两者都不会保存
            self._flush_exit_frames(debug_recorder, task, safety)
            # 技能卡采集收尾：等待写盘队列排空（未启用时为空操作）
            if skill_collector is not None:
                skill_collector.close()
        logger.info("任务结束，最终状态=%s，已完成局数=%d", final.value, task.ctx.round_count)
        return final

    def select_difficulty(self, coop_difficulties: str | None = None) -> State:
        """在用户已打开的难度弹窗中执行一次独立的难度选择能力。

        该入口复用 ``CoopTask`` 的难度识别、点击和滚动逻辑，但不会打开或关闭
        弹窗，也不会继续进入抢合作流程，便于实机单独校验这一段能力。

        Args:
            coop_difficulties: 命令行难度范围覆盖，例如 ``1-10``；None 使用配置值。

        Returns:
            全部目标难度均已点击或在有限重试后跳过时返回 ``State.COMPLETED``；
            窗口、安全检查或动作执行失败时返回当前状态。
        """
        difficulty_levels = (
            parse_coop_difficulties(coop_difficulties)
            if coop_difficulties is not None
            else list(self.default_config.run.coop_difficulties)
        )
        logger.info("启动独立难度选择，合作难度=%s", difficulty_levels)

        screen = ScreenCapture()
        handle = self._find_game_window(screen)
        if handle is None:
            raise RuntimeError(
                f"未找到游戏窗口，请确认游戏已启动且窗口标题含『{self._window_title()}』"
            )

        if activate_window(handle):
            logger.info("已激活游戏窗口到前台")
        else:
            logger.warning("激活游戏窗口失败，请手动切换到游戏窗口前台")
        logger.info("3 秒后开始选择，请保持难度弹窗已打开且勿切换窗口")
        time.sleep(3)

        pack = self._load_template_pack(screen, handle)
        self._check_difficulty_templates(difficulty_levels, State.FIND_COOP, pack)
        debug_recorder = (
            DebugRecorder(
                self.default_config.vision.debug_dir,
                exit_frame_buffer_size=self.default_config.vision.exit_frame_buffer_size,
            )
            if self.default_config.vision.debug
            else None
        )
        perception = CoopPerception(
            Vision(),
            pack,
            self.tasks_config,
            CoopRole.HELPER,
            self.default_config.run.default_main_c,
            debug_recorder=debug_recorder,
        )
        safety = SafetyGuard(
            max_failures=self.default_config.safety.max_failures,
            frame_ttl_ms=self.default_config.safety.frame_ttl_ms,
        )
        executor = ActionExecutor(
            safety,
            InputController(),
            min_delay=self.default_config.input.min_delay,
            max_delay=self.default_config.input.max_delay,
            context_validator=screen.validate_context,
        )
        hotspots = dict(self.tasks_config.hotspots)
        task = CoopTask(
            TaskContext(
                main_c=self.default_config.run.default_main_c,
                current_state=State.FIND_COOP,
                max_rounds=1,
            ),
            self.default_config.run,
            CoopRole.HELPER,
            hotspots,
            coop_difficulties=difficulty_levels,
            difficulty_selection_only=True,
        )

        if safety.start_esc_listener():
            logger.info("已启动 Esc 停止热键监听，运行中按 Esc 可停止能力")
        else:
            logger.warning("Esc 停止热键监听启动失败，可用 Ctrl+C 中断")
        try:
            final = self._run_loop(
                screen,
                perception,
                executor,
                task,
                safety,
                handle,
                max_steps=100,
                debug_recorder=debug_recorder,
            )
        except KeyboardInterrupt:
            # Ctrl+C 用户主动中断：按约定不保存退出帧（清空缓冲后原样上抛）
            if debug_recorder is not None:
                debug_recorder.drain_exit_frames()
            raise
        finally:
            self._flush_exit_frames(debug_recorder, task, safety)
        if final == State.COMPLETED:
            logger.info("独立难度选择完成")
        else:
            logger.warning("独立难度选择未完成，最终状态=%s", final.value)
        return final

    def _run_loop(
        self,
        screen: ScreenCapture,
        perception: CoopPerception,
        executor: ActionExecutor,
        task: Task,
        safety: SafetyGuard,
        window_handle: int,
        max_steps: int | None = None,
        grab_coordinator: CoopGrabCoordinator | None = None,
        debug_recorder: DebugRecorder | None = None,
    ) -> State:
        """调度循环：截图 -> 识别 -> 状态判断 -> 决策 -> 执行。

        纯调度逻辑，依赖通过参数注入，便于用 Fake 组件做单元测试。
        ``max_steps`` 仅用于测试或独立能力覆盖；正式任务默认使用配置中的
        ``run.max_steps_per_round``，并在完成一局后重置计数。
        ``debug_recorder`` 提供时，每张主循环截图进入退出帧缓冲，由调用方
        ``_flush_exit_frames`` 在任务结束时统一落盘（培养阶段识别管线内部的
        多帧截图不进缓冲，缓冲按主循环节奏采样）。
        """
        max_steps_per_round = (
            max_steps if max_steps is not None else self.default_config.run.max_steps_per_round
        )
        if max_steps_per_round < 1:
            raise ValueError("max_steps_per_round 必须大于等于 1")
        round_steps = 0
        tracked_round_count = task.ctx.round_count
        no_fg_count = 0
        stale_decisions = 0  # 连续「识别决策超过截图时效」的轮数
        last_refocus_at = 0.0  # 上次尝试自动切回前台的时刻（失败重试节流）
        pending: _PendingVerification | None = None
        popup_close_retries = 0
        merge_retries = 0
        skill_click_retries = 0
        # 识别出错自动重开：连续无进展的重开次数（验证成功/完成一局清零）
        error_restarts = 0
        error_restart_enabled = self.default_config.run.error_restart_enabled
        error_restart_max = self.default_config.run.error_restart_max_consecutive

        def _recover_or_stop(reason: str) -> bool:
            """保守停止时的原地恢复兜底。返回 True=原地继续循环；False=应停止。

            原地恢复：不重置状态、不回首页，丢弃未验证的动作后直接进入下一帧
            ——状态机每帧按屏幕画面重新识别决策，画面在哪就接着处理哪。
            连续恢复达到预算仍无任何进展（验证成功/完成一局清零）才停止。
            """
            nonlocal error_restarts, pending, round_steps, stale_decisions
            if not error_restart_enabled or error_restarts >= error_restart_max:
                return False
            error_restarts += 1
            logger.warning(
                "%s；原地继续识别（无进展重开 %d/%d，验证成功或完成一局后清零）",
                reason,
                error_restarts,
                error_restart_max,
            )
            pending = None
            round_steps = 0
            stale_decisions = 0
            return True
        while not safety.stop_requested:
            if task.ctx.round_count != tracked_round_count:
                # round_count 在结算返回验证后 +1，新值即刚完成的局号
                # （对局进行中决策日志显示 局=round_count+1）
                logger.info(
                    "第 %d 局完成（%d/%d），单局步数=%d",
                    task.ctx.round_count,
                    task.ctx.round_count,
                    task.ctx.max_rounds,
                    round_steps,
                )
                tracked_round_count = task.ctx.round_count
                round_steps = 0
                error_restarts = 0
            if task.ctx.current_state == State.COMPLETED:
                logger.info(
                    "达到 COMPLETED，任务完成，共完成 %d/%d 局",
                    task.ctx.round_count,
                    task.ctx.max_rounds,
                )
                break
            if round_steps >= max_steps_per_round:
                logger.warning(
                    "达到单局最大步数保险 %d，停止以防无限循环 state=%s round=%d/%d",
                    max_steps_per_round,
                    task.ctx.current_state.value,
                    task.ctx.round_count + 1,
                    task.ctx.max_rounds,
                )
                break
            round_steps += 1

            # 1. 截图
            capture_started = time.perf_counter()
            try:
                ctx, frame = screen.capture(window_handle)
            except Exception as exc:
                logger.error("截图失败: %r", exc)
                if safety.record_failure():
                    logger.error("连续失败达上限，停止")
                    break
                continue
            capture_ms = (time.perf_counter() - capture_started) * 1000

            # 2. 窗口有效性检查
            if ctx.is_minimized or not ctx.is_foreground:
                # 非前台/最小化期间必须挂起（点击会落到当前前台窗口）。
                # 游戏对局自动进行，切回后从当前进度继续；持续超过
                # window_foreground_wait_seconds 才保守停止
                reason = "最小化" if ctx.is_minimized else "非前台"
                no_fg_count += 1
                waited = no_fg_count * _FG_CHECK_INTERVAL_SECONDS
                timeout = self.default_config.run.window_foreground_wait_seconds
                if waited >= timeout:
                    logger.error(
                        "窗口%s已持续 %.0f 秒（上限 %.0f 秒，可用 run.window_foreground_wait_seconds "
                        "调大），停止任务",
                        reason,
                        waited,
                        timeout,
                    )
                    break
                # 系统空闲（用户没在用电脑）时自动把游戏窗口切回前台继续，
                # 不抢用户正在使用时的焦点；激活失败按节流间隔重试
                if self.default_config.run.refocus_when_idle:
                    idle = get_input_idle_seconds()
                    if idle >= self.default_config.run.refocus_idle_seconds and (
                        time.monotonic() - last_refocus_at >= _REFOCUS_RETRY_SECONDS
                    ):
                        last_refocus_at = time.monotonic()
                        if activate_window(window_handle):
                            logger.info(
                                "系统已 %.0f 秒无鼠标/键盘活动，自动切回游戏窗口继续任务",
                                idle,
                            )
                            time.sleep(0.5)  # 等激活生效，下一轮重新校验前台
                            no_fg_count = 0
                            continue
                        logger.warning("系统空闲 %.0f 秒，但游戏窗口激活失败，继续挂起等待", idle)
                # 首次立即提示，之后每 10 秒提示一次，避免高频刷屏
                if no_fg_count == 1 or no_fg_count % _FG_LOG_EVERY_CHECKS == 0:
                    logger.warning(
                        "窗口%s，任务挂起等待切回（已等 %.0f/%.0f 秒，期间不发送输入）",
                        reason,
                        waited,
                        timeout,
                    )
                time.sleep(_FG_CHECK_INTERVAL_SECONDS)
                continue
            no_fg_count = 0  # 恢复前台后重置计数

            # 退出帧缓冲：只保留前台有效截图（挂起/最小化期间的黑帧不冲掉历史）
            if debug_recorder is not None:
                debug_recorder.keep_exit_frame(ctx.frame_id, ctx.captured_at, frame)

            # 3. 识别：培养阶段多帧累积（应对英雄动态漏检），其他单帧。
            # 强制召唤等不以棋盘为门禁的快速阶段（wants_board_watch=False）
            # 退回单帧界面标志识别，避免每步白付多帧识别的耗时
            observe_started = time.perf_counter()
            if task.ctx.current_state in _BOARD_WATCH_STATES and task.wants_board_watch():
                ctx, observation = perception.observe_cultivation(
                    screen,
                    window_handle,
                    ctx,
                    n_frames=self.default_config.run.board_recognition_frames,
                    read_gold=task.wants_gold_read(),
                )
                logger.debug(
                    "frame=%d 多帧累积识别 heroes=%d",
                    ctx.frame_id,
                    len(observation.board.heroes) if observation.board else -1,
                )
            else:
                observation = perception.observe(
                    ctx,
                    frame,
                    task.ctx.current_state,
                    task.observation_mode(),
                    read_gold=task.wants_gold_read(),
                )
            observe_ms = (time.perf_counter() - observe_started) * 1000

            if observation.frame_id != ctx.frame_id:
                logger.error(
                    "识别结果 frame_id=%d 与窗口上下文 frame_id=%d 不一致，停止",
                    observation.frame_id,
                    ctx.frame_id,
                )
                break

            # 严格输入动作使用后续截图验证，不把“输入已发送”当成成功。
            if pending is not None:
                if task.verify_action(pending.action, pending.before, observation):
                    logger.info(
                        "frame=%d 动作后验证通过 tag=%s",
                        ctx.frame_id,
                        pending.action.tag,
                    )
                    error_restarts = 0
                    if (
                        pending.action.tag == "close_popup"
                        or pending.action.tag == "close_double_reward"
                        or pending.action.tag == "close_bonus_popup"
                    ):
                        popup_close_retries = 0
                    elif pending.action.tag == "merge_heroes":
                        merge_retries = 0
                    elif pending.action.tag in _SKILL_CLICK_TAGS:
                        skill_click_retries = 0
                    task.on_action_verified(pending.action, pending.to_state)
                    task.ctx.current_state = pending.to_state
                    pending = None
                else:
                    pending.attempts += 1
                    if pending.attempts >= self.default_config.run.action_verify_frames:
                        tag = pending.action.tag
                        # 击杀奖励弹窗验证失败大概率是关掉后立刻弹了新弹窗；
                        # 合成拖动可能落空（英雄仍在原位）。两者都可恢复，
                        # 在预算内重新决策执行而不是直接结束任务。
                        if tag == "merge_heroes":
                            _log_merge_verify_failure(pending, observation)
                        if (
                            tag in {"close_popup", "close_double_reward", "close_bonus_popup"}
                            and popup_close_retries
                            < self.default_config.run.close_popup_max_retries
                        ):
                            popup_close_retries += 1
                            logger.warning(
                                "%s 动作后验证失败，重新点击关闭 (%d/%d)",
                                tag,
                                popup_close_retries,
                                self.default_config.run.close_popup_max_retries,
                            )
                            pending = None
                            continue
                        if (
                            tag == "merge_heroes"
                            and merge_retries < self.default_config.run.merge_max_retries
                        ):
                            merge_retries += 1
                            logger.warning(
                                "merge_heroes 动作后验证失败，重新决策并重试合成 (%d/%d)",
                                merge_retries,
                                self.default_config.run.merge_max_retries,
                            )
                            pending = None
                            continue
                        if tag in _SKILL_CLICK_TAGS:
                            # 技能点击未生效可恢复：页面还在就重新决策再选。
                            # 重试耗尽也不结束任务——对局仍在自动进行，结算窗口
                            # 终将出现；通知任务放弃本次选择，稍后再试或等对局推进。
                            if (
                                skill_click_retries
                                < self.default_config.run.skill_click_max_retries
                            ):
                                skill_click_retries += 1
                                logger.warning(
                                    "技能点击动作后验证失败，重新决策重试 (%d/%d) tag=%s",
                                    skill_click_retries,
                                    self.default_config.run.skill_click_max_retries,
                                    tag,
                                )
                                pending = None
                                continue
                            logger.warning(
                                "技能点击连续 %d 次未生效，放弃本次选择，等待对局继续 tag=%s",
                                skill_click_retries + 1,
                                tag,
                            )
                            task.on_action_failed(pending.action)
                            pending = None
                            continue
                        if tag == "merge_heroes":
                            # 合成拖动连续未生效（英雄被弹回原位）：不结束任务。
                            # 通知任务记住该失败合成对并跳过，改为召唤新英雄
                            # 改变棋盘——新英雄可能形成新的合法合成对
                            logger.warning(
                                "合成拖动连续 %d 次未生效，放弃该合成对，召唤新英雄改变棋盘 tag=%s",
                                merge_retries + 1,
                                tag,
                            )
                            task.on_action_failed(pending.action)
                            pending = None
                            continue
                        if tag == "summon_hero":
                            # 召唤点击后棋盘始终未变化（多为金币不足，游戏忽略了
                            # 点击）：不结束任务。回补阶段放弃回补回选技能，
                            # 培养阶段等待一段时间后重试（金币随对局恢复）
                            logger.warning(
                                "召唤后棋盘连续 %d 帧未变化（可能金币不足），按任务策略继续 tag=%s",
                                pending.attempts,
                                tag,
                            )
                            task.on_action_failed(pending.action)
                            pending = None
                            continue
                        logger.error(
                            "动作后验证失败，达到 %d 帧上限 tag=%s",
                            pending.attempts,
                            tag,
                        )
                        if not _recover_or_stop(
                            f"动作 {tag} 验证连续 {pending.attempts} 帧失败"
                        ):
                            break
                        continue
                    logger.debug(
                        "动作后状态尚未稳定，等待下一帧 tag=%s (%d/%d)",
                        pending.action.tag,
                        pending.attempts,
                        self.default_config.run.action_verify_frames,
                    )
                    time.sleep(0.2)
                    continue

            # 4. 状态判断
            state = task.determine_state(observation)
            task.ctx.current_state = state
            if task.ctx.current_state not in _BOARD_WATCH_STATES:
                logger.debug("frame=%d state=%s", ctx.frame_id, state.value)

            # 5. 决策
            decision = task.decide_action(observation, ctx)
            if decision is None:
                if not _recover_or_stop(
                    f"状态 {state.value} 无可用动作（识别出错或界面未知）"
                ):
                    logger.info("状态 %s 无可用动作，保守停止", state.value)
                    break
                continue
            action, to_state = decision
            logger.info(
                "frame=%d 局=%d/%d %s -> %s action=%s [%s]",
                ctx.frame_id,
                task.ctx.round_count + 1,
                task.ctx.max_rounds,
                state.value,
                to_state.value,
                action.kind,
                action.reason,
            )

            # 抢合作由双线程协调器执行：连点 join + 并行识别准备按钮。
            # 协调器阻塞主循环，直到发现准备按钮 / 超时 / 窗口失效 / 停止。
            if action.kind == "grab_coop":
                if grab_coordinator is None:
                    logger.error("抢合作协调器未初始化，停止")
                    break
                grab_result = grab_coordinator.run()
                if grab_result.found:
                    logger.info("抢合作成功，回到主循环识别并点击准备按钮")
                    continue
                logger.warning("抢合作未发现准备按钮，停止 reason=%s", grab_result.reason)
                if grab_result.reason == "window_lost":
                    task.ctx.current_state = State.WINDOW_INVALID
                break

            # 6. 执行前校验截图时效：识别和决策本身可能耗时（多模板匹配、
            # 机器负载高/省电模式等），决策所依据的画面超过 frame_ttl 后
            # 不能再执行输入。丢弃本轮重新截图重试——难度弹窗等静态界面
            # 晚几秒重试仍然有效，这不算动作失败，不消耗安全失败预算；
            # 连续多轮超龄（本机持续无法在时效内完成识别->动作）才保守停止。
            age_seconds = time.time() - ctx.captured_at
            if age_seconds * 1000 > safety._frame_ttl_ms:
                stale_decisions += 1
                logger.warning(
                    "frame=%d 截图至动作执行耗时 %.2fs 超过 frame_ttl_ms=%dms，"
                    "丢弃本轮决策重新截图（截图 %.0fms + 识别 %.0fms）（%d/%d）",
                    ctx.frame_id,
                    age_seconds,
                    safety._frame_ttl_ms,
                    capture_ms,
                    observe_ms,
                    stale_decisions,
                    _MAX_STALE_DECISIONS,
                )
                if stale_decisions >= _MAX_STALE_DECISIONS:
                    logger.error(
                        "连续 %d 轮识别决策超过截图时效，本机无法在时效内完成"
                        "识别到动作；可检查电脑负载/电源模式，或调大 safety.frame_ttl_ms",
                        stale_decisions,
                    )
                    if not _recover_or_stop("连续多轮识别决策超过截图时效"):
                        break
                    continue
                continue
            stale_decisions = 0

            # 7. 执行
            result = executor.execute(ctx, action)
            if not result.executed:
                logger.warning("动作执行失败: %s", result.failure_reason)
                if safety.record_failure():
                    logger.error("连续失败达上限，停止")
                    break
                continue

            safety.reset_failures()
            if result.verified or action.verification == "immediate":
                task.on_action_verified(action, to_state)
                task.ctx.current_state = to_state
            else:
                task.ctx.current_state = state
                pending = _PendingVerification(
                    action=action,
                    to_state=to_state,
                    before=observation,
                )
                logger.debug("输入已发送，等待下一帧验证 tag=%s", action.tag)

        return task.ctx.current_state

    @staticmethod
    def _flush_exit_frames(
        debug_recorder: DebugRecorder | None,
        task: Task,
        safety: SafetyGuard,
    ) -> None:
        """任务结束时落盘退出帧缓冲；正常完成（COMPLETED）和用户主动停止不保存。

        由 run()/select_difficulty() 的 finally 调用，循环正常结束、Esc 或异常
        崩溃统一经过这里；Ctrl+C 的缓冲已在调用方 except 中清空，自然不保存。
        保存动作本身失败只记日志，不改变退出结果。
        """
        if debug_recorder is None:
            return
        frames = debug_recorder.drain_exit_frames()
        if not frames:
            return
        if task.ctx.current_state == State.COMPLETED:
            return
        if safety.stop_requested:
            # Esc 停止热键：用户主动退出，不保存退出帧
            return
        try:
            folder = debug_recorder.save_exit_frames(
                frames, state=task.ctx.current_state.value
            )
            logger.warning(
                "任务非正常退出，已保存退出前 %d 帧截图到 %s（状态=%s）",
                len(frames),
                folder,
                task.ctx.current_state.value,
            )
        except Exception as exc:
            # 诊断落盘失败不能反向掩盖原始退出原因
            logger.error("保存退出帧失败: %r", exc)

    # ------------------------------------------------------------------
    # 启动检查与环境初始化
    # ------------------------------------------------------------------

    def _startup_check(self, main_c: str, start_state: State) -> None:
        """启动检查：主C档案存在，正常自动流程必须具备技能图标模板。"""
        profiles = self.default_config.main_c_profiles
        if main_c not in profiles:
            available = " / ".join(profiles.keys()) or "(无)"
            raise RuntimeError(f"主C {main_c} 不在配置档案中，可用: {available}")
        profile = profiles[main_c]
        if not profile.skill_icon_templates:
            if start_state != State.BUILD_MAIN_C:
                raise RuntimeError(
                    f"主C {main_c} 的 skill_icon_templates 为空，不能进入自动对局；"
                    "请采集该主C技能卡上的英雄图标模板并在 main_c_profiles 中配置。"
                    "仅调试培养闭环时可使用 --start-state build_main_c"
                )
            logger.warning(
                "主C %s 的 skill_icon_templates 为空；当前仅以 build_main_c 调试模式运行",
                main_c,
            )
        if not profile.hero_classifier_model.strip():
            raise RuntimeError(f"主C {main_c} 未配置 hero_classifier_model")

    def _load_hero_cell_classifier(self, main_c: str) -> HeroCellClassifier:
        """加载当前主 C 专用 ONNX 模型及其同名 metadata。"""
        model_path = Path(self.default_config.main_c_profiles[main_c].hero_classifier_model)
        try:
            classifier = HeroCellClassifier(model_path)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"英雄格分类模型加载失败: {exc}") from exc
        logger.info(
            "已加载英雄格分类模型 main_c=%s model=%s classes=%d confidence=%.2f margin=%.2f",
            main_c,
            model_path,
            len(classifier.class_names),
            classifier.confidence_threshold,
            classifier.margin_threshold,
        )
        return classifier

    def watch_board(
        self,
        main_c: str,
        *,
        interval: float = 1.0,
        on_result=None,
    ) -> int:
        """循环识别己方棋盘并回调快照，直到回调返回 False 或用户中断。

        与实战共用 ``CoopPerception.observe_cultivation`` 多帧投票路径，用于在
        真实对局中独立验证棋盘识别效果：不执行任何输入动作，只截图与识别。

        Args:
            main_c: 主 C 英文标识，决定加载的模型与阵容类别校验。
            interval: 两轮识别之间的间隔秒数。
            on_result: 每轮回调 ``(轮次, Observation)``；返回 False 停止循环。

        Returns:
            完成的轮次总数。
        """
        from wlxq_bot.perception.screen import ScreenCapture

        screen = ScreenCapture()
        handle = self._find_game_window(screen)
        if handle is None:
            raise RuntimeError(
                f"未找到游戏窗口，请确认游戏已启动且窗口标题含『{self._window_title()}』"
            )
        pack = self._load_template_pack(screen, handle)
        perception = CoopPerception(
            Vision(),
            pack,
            self.tasks_config,
            CoopRole.HELPER,
            main_c,
            hero_cell_classifier=self._load_hero_cell_classifier(main_c),
            allowed_heroes={main_c, *self.default_config.hero_classifier.lineup_others},
        )
        required = {main_c, *self.default_config.hero_classifier.lineup_others}
        missing = sorted(required - set(perception.available_heroes))
        if missing:
            raise RuntimeError("英雄格模型缺少本局阵容类别: " + ", ".join(missing))
        n_frames = self.default_config.run.board_recognition_frames
        safety = SafetyGuard(
            max_failures=self.default_config.safety.max_failures,
            frame_ttl_ms=self.default_config.safety.frame_ttl_ms,
        )
        esc_listener_started = safety.start_esc_listener()
        if esc_listener_started:
            logger.info("按 Esc 停止棋盘识别观察")
        else:
            logger.warning("Esc 监听不可用，仅支持 Ctrl+C 停止")
        logger.info("棋盘识别观察开始 main_c=%s 帧数/轮=%d", main_c, n_frames)
        rounds_done = 0
        try:
            while not safety.stop_requested:
                ctx, _frame = screen.capture(handle)
                _ctx, observation = perception.observe_cultivation(
                    screen,
                    handle,
                    ctx,
                    n_frames=n_frames,
                    require_foreground=False,
                )
                rounds_done += 1
                if on_result is not None and on_result(rounds_done, observation) is False:
                    break
                for _ in range(int(interval / 0.05) + 1):
                    if safety.stop_requested:
                        break
                    time.sleep(0.05)
        except KeyboardInterrupt:
            logger.info("棋盘识别观察被用户中断 rounds=%d", rounds_done)
        logger.info("棋盘识别观察结束 rounds=%d", rounds_done)
        return rounds_done

    def _window_title(self) -> str:
        if self.local_config and self.local_config.window.title:
            return self.local_config.window.title
        return self.default_config.screen.window_title

    def _check_skill_templates(
        self,
        main_c: str,
        start_state: State,
        pack: TemplatePack,
    ) -> None:
        """正常自动流程启动前确认主C技能图标模板真实存在。"""
        if start_state == State.BUILD_MAIN_C:
            return
        icons = self.default_config.main_c_profiles[main_c].skill_icon_templates
        missing = [str(rel) for rel in icons if not pack.resolve_template(str(rel)).is_file()]
        if missing:
            raise RuntimeError(
                "主C技能图标模板缺失: " + ", ".join(missing) + f"（模板包: {pack.root}）"
            )

    def _build_gold_reader(self) -> tuple[Any | None, float]:
        """按标定情况装配金币读取器；三个 ROI 未全部标定时返回 (None, 间隔)。"""
        interval = float(
            self.tasks_config.gold_recognition.get("read_min_interval_seconds", 1.0)
        )
        required = ("gold_current_area", "skill_cost_area", "summon_cost_area")
        if not all(name in self.tasks_config.rois for name in required):
            logger.info("金币感知 ROI 未标定，金币门控不可用")
            return None, interval
        from wlxq_bot.perception.ocr import GoldReader

        return GoldReader(), interval

    def _build_skill_priority(self, mc: str) -> tuple[Any | None, dict[str, int] | None, Path | None]:
        """装配技能标题 OCR 主路径（优先级选卡）三件套。

        Returns:
            (标题识别器, 技能名→档位映射, 新技能记录目录)；主C未配置
            skill_priority、依赖缺失或清单缺失时返回 (None, None, None)，
            技能选择回退原有图标识别+随机流程。
        """
        profile = self.default_config.main_c_profiles[mc]
        if not profile.skill_priority:
            logger.info("主C %s 未配置 skill_priority，技能选择走图标识别流程", mc)
            return None, None, None
        try:
            from wlxq_bot.perception.ocr import TitleReader, ensure_engine
            from wlxq_bot.skill_catalog import compute_skill_tiers, load_skill_name_index

            catalog_path = Path("configs/skills.yaml")
            name_to_hero = load_skill_name_index(catalog_path)
            if not name_to_hero:
                logger.warning("技能清单 %s 为空，标题识别不可用", catalog_path)
                return None, None, None
            skill_tiers = compute_skill_tiers(
                profile.skill_priority, name_to_hero, profile.display_name
            )
            ensure_engine()
            tier8 = sorted(name for name, tier in skill_tiers.items() if tier == 8)
            tier11 = sorted(name for name, tier in skill_tiers.items() if tier == 11)
            logger.info(
                "技能优先级已启用 主C=%s(%s) 第8档=%s 第11档=%s",
                mc,
                profile.display_name,
                tier8,
                tier11,
            )
            new_skill_dir = Path(self.default_config.run.skill_collection.output_dir)
            return TitleReader(), skill_tiers, new_skill_dir
        except (ImportError, RuntimeError, ValueError, OSError) as exc:
            logger.warning("技能标题识别初始化失败，回退图标识别流程: %r", exc)
            return None, None, None

    def _build_skill_collector(self, run_config: RunConfig) -> SkillCollector | None:
        """按统计阶段配置构造技能卡采集器；未启用或初始化失败返回 None。

        采集只在 ``run.skill_collection.enabled`` 打开时进行。运行时采集
        只做裁剪和哈希，英雄归属在离线建册时进行。初始化失败只记日志，
        绝不阻断正常对局。
        """
        cfg = run_config.skill_collection
        if not cfg.enabled:
            logger.info("技能卡采集未启用（run.skill_collection.enabled=false），跳过")
            return None
        try:
            geometry = self.tasks_config.skill_collection
            return SkillCollector(
                output_dir=Path(cfg.output_dir),
                session_label=time.strftime("%Y%m%d_%H%M%S"),
                column_inset_ratio=float(geometry.get("column_inset_ratio", 0.04)),
                top_trim_ratio=float(geometry.get("top_trim_ratio", 0.06)),
                fuse_max_consecutive_failures=cfg.fuse_max_consecutive_failures,
                min_collect_interval_seconds=cfg.min_collect_interval_seconds,
                queue_maxsize=cfg.queue_maxsize,
            )
        except Exception as exc:
            logger.warning("技能卡采集器初始化失败，本次运行不采集: %r", exc)
            return None

    @staticmethod
    def _check_difficulty_templates(
        levels: list[int],
        start_state: State,
        pack: TemplatePack,
        skip_difficulty: bool = False,
    ) -> None:
        """正常招募流程启动前确认所有目标难度模板存在；跳过选择时无需模板。"""
        if skip_difficulty or start_state != State.FIND_COOP:
            return
        missing = [
            level
            for level in levels
            if not pack.resolve_template(f"buttons/coop_difficulty/cai_hong_{level}.png").is_file()
        ]
        if missing:
            raise RuntimeError(
                "合作难度模板缺失: "
                + ", ".join(str(level) for level in missing)
                + f"（模板包: {pack.root}）"
            )

    def _check_home_page_template(self, start_state: State, pack: TemplatePack) -> None:
        """正常合作流程启动前确认首页正向识别模板存在。"""
        if start_state != State.FIND_COOP:
            return
        locator = self.tasks_config.locators.get("home_page_marker")
        template_rel = locator.get("template") if locator else None
        if not template_rel:
            raise RuntimeError("缺少 home_page_marker locator，无法确认游戏首页")
        template_path = pack.resolve_template(str(template_rel))
        if not template_path.is_file():
            raise RuntimeError(f"首页识别模板缺失: {template_rel}（模板包: {pack.root}）")
        threshold = locator.get("threshold")
        if not isinstance(threshold, (int, float)):
            raise RuntimeError("首页识别阈值未标定: locators.home_page_marker.threshold")

    def _find_game_window(self, screen: ScreenCapture) -> int | None:
        title = self._window_title()
        handle = screen.find_window(title)
        if handle is None:
            logger.error("未找到游戏窗口，标题=%s", title)
        else:
            logger.debug("找到游戏窗口，句柄=%s", handle)
        return handle

    def _ensure_window_size(self, screen: ScreenCapture, handle: int) -> None:
        """客户区与本机配置不一致时自动调整窗口尺寸（等价 adjust-window 命令）。

        未配置 local.yaml（无目标尺寸）时不做任何事，模板包按显示器分辨率选择；
        ``run.auto_adjust_window`` 关闭时退回旧行为：报错停止并提示手动执行
        ``wlxq-bot adjust-window``。调整后仍不一致（窗口最小尺寸限制等）则报错停止。
        """
        if self.local_config is None:
            return
        spec = self.local_config.window
        target = (spec.target_client_width, spec.target_client_height)
        info = screen.get_window_info(handle)
        if info.client_size == target:
            return
        if not self.default_config.run.auto_adjust_window:
            raise RuntimeError(
                f"游戏客户区为 {info.client_size[0]}x{info.client_size[1]}，"
                f"与本机配置 {target[0]}x{target[1]} 不一致，且 auto_adjust_window 已关闭；"
                "请先运行 wlxq-bot adjust-window，禁止在未校准尺寸下继续识别和输入"
            )
        logger.warning(
            "游戏客户区 %dx%d 与本机配置 %dx%d 不一致，自动调整窗口尺寸",
            info.client_size[0],
            info.client_size[1],
            target[0],
            target[1],
        )
        adjusted = adjust_window_size(handle, target[0], target[1])
        if adjusted.client_size != target:
            raise RuntimeError(
                f"自动调整后客户区 {adjusted.client_size[0]}x{adjusted.client_size[1]} "
                f"仍与目标 {target[0]}x{target[1]} 不一致（可能是窗口最小尺寸限制），停止任务"
            )
        logger.info("窗口已自动调整到目标客户区 %dx%d", target[0], target[1])

    def _load_template_pack(self, screen: ScreenCapture, handle: int) -> TemplatePack:
        """显式配置优先，否则按游戏窗口所在显示器的物理分辨率加载。"""
        if self._template_pack is not None:
            return self._template_pack

        info = screen.get_window_info(handle)
        if self.local_config is not None:
            expected_size = (
                self.local_config.window.target_client_width,
                self.local_config.window.target_client_height,
            )
            if info.client_size != expected_size:
                raise RuntimeError(
                    f"游戏客户区为 {info.client_size[0]}x{info.client_size[1]}，"
                    f"与本机配置 {expected_size[0]}x{expected_size[1]} 不一致；"
                    "请先运行 wlxq-bot adjust-window，禁止在未校准尺寸下继续识别和输入"
                )
        profile = self.local_config.window.template_pack if self.local_config else ""
        if profile:
            pack_root = TEMPLATES_ROOT / profile
            if not pack_root.is_dir():
                raise RuntimeError(f"本地配置指定的模板包 {profile!r} 不存在：{pack_root}")
            logger.debug("加载显式模板包 %s", profile)
            return TemplatePack(client_size=info.client_size, root=pack_root)

        monitor_w, monitor_h = get_window_monitor_resolution(handle)
        pack_root = TEMPLATES_ROOT / f"{monitor_w}x{monitor_h}"
        if pack_root.is_dir():
            logger.debug(
                "按窗口所在显示器物理分辨率加载模板包 %sx%s",
                monitor_w,
                monitor_h,
            )
            return TemplatePack(client_size=info.client_size, root=pack_root)

        raise RuntimeError(
            f"窗口所在显示器分辨率为 {monitor_w}x{monitor_h}，但模板包不存在：{pack_root}；"
            "请采集该分辨率的模板，或通过 window.template_pack 显式指定"
        )
