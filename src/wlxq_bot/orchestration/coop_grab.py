"""Runner 侧的抢合作双线程协调器。

抢合作阶段需要"尽快连点 join 位置"同时"周期性识别准备按钮"，两者重叠
执行，避免识别准备按钮期间停止点击造成的死区——识别耗时内不点击就可能
错过新邀请（抢合作拼速度，这是选双线程而非单线程批量的根本原因）。

- 抢合作线程：经执行器以拟人间隔（find_coop_click_delay_min/max）连点
  join_coop hotspot，不做任何识别。执行器逐击实时校验前台，失焦/最小化/
  上下文超龄等瞬态失败不累计（失焦期间点击自动暂停，切回后自动恢复）；
  窗口真丢失（句柄失效、几何变化、输入异常）连续失败达上限才报 window_lost。
- 检查线程：用自己的截图实例按 find_coop_check_interval_seconds 周期截图
  并匹配准备按钮；识别到即通知停止。它绝不点击。窗口失焦/最小化时挂起等待
  切回（与主循环同策略：系统空闲时自动切回，持续超过 window_foreground_wait_seconds
  才报 window_lost），期间不识别不点击。前台有效截图进入退出帧缓冲，
  抢合作阶段丢失窗口瞬间的画面也能落盘排查。
- 弹窗关闭（2026-09-14 实机）：连点 join 落在组队大厅会误点自己头像，弹出
  「个人信息」等全屏弹窗挡住准备按钮。检查线程在准备按钮未命中时顺带匹配
  全屏弹窗右上角标准关闭按钮（×，跨弹窗皮肤一致），命中后把匹配位置写给
  抢线程，由抢线程点击关闭——检查线程仍然绝不点击，只是抢线程的点击目标
  从 join 暂时切换为关闭按钮（弹窗出现即已加入成功，暂停连点无损失，反而
  消除连点持续落在弹窗上的风险）。点击后由检查线程下一拍验证：弹窗消失则
  继续找准备按钮；连续多拍仍可见则按 close_popup_max_retries 重试后保守
  停止（popup_unclosable）。已越过招募阶段（见过亮屏/准备/关闭按钮）后
  边缘持续变暗却识别不到标准关闭按钮，视为未知弹窗，保守停止并留证
  （unknown_popup；招募页本身边缘也是暗的，未越过招募阶段不判）。
- 发现准备按钮后两个线程都退出，回到 Runner 主循环，由正常的单线程流程
  重新截图、识别并点击准备按钮（保证 frame_id 校验；点准备绝不并发）。

依赖方向：Runner -> CoopGrabCoordinator -> Perception/Vision/ActionExecutor。
任务状态机不感知本协调器：CoopTask 在 JOIN_COOP 步骤只发出 kind="grab_coop"
的信号动作，由 Runner 据此调用本协调器。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from wlxq_bot.action.executor import ActionExecutor
from wlxq_bot.action.safety import SafetyGuard
from wlxq_bot.config import Hotspot, RunConfig
from wlxq_bot.debug.recorder import DebugRecorder
from wlxq_bot.models import Action, WindowContext
from wlxq_bot.perception.coop import CoopPerception
from wlxq_bot.perception.locator import hotspot_to_client_point
from wlxq_bot.perception.screen import (
    ScreenCapture,
    WindowInfo,
    activate_window,
    get_input_idle_seconds,
    get_window_info,
)
from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

# 抢合作线程连续执行失败达到此次数，认定窗口已真丢失（句柄失效、几何变化、
# 输入异常），保守停止。失焦/最小化/上下文超龄是瞬态失败，不累计。
_GRAB_FAILURE_LIMIT = 3

# 执行器失败原因中的瞬态两类（对应 safety.check_action / screen.validate_context
# 的返回文案）：出现它们时窗口本身没丢，不累计失败、不退出抢合作。
# tests/test_coop_grab.py 有守卫测试保证文案与生产者一致。
_TRANSIENT_FAILURE_REASONS = frozenset({"窗口最小化或非前台", "窗口上下文已超时"})

# 挂起等待切回期间的提示日志间隔（秒）
_FG_LOG_INTERVAL_SECONDS = 10.0
# 自动切回失败后的重试间隔（秒），与 runner 主循环保持一致
_REFOCUS_RETRY_SECONDS = 5.0


def _default_probe_window(window_handle: int) -> WindowInfo | None:
    """读取实时窗口状态；窗口信息不可读（如已关闭）时返回 None。"""
    try:
        return get_window_info(window_handle)
    except Exception:
        return None


@dataclass
class _ForegroundSuspension:
    """挂起等待窗口切回前台的状态（检查线程内使用，恢复前台后 reset）。"""

    since: float | None = None
    last_log: float = 0.0
    last_refocus: float = 0.0

    def reset(self) -> None:
        self.since = None


@dataclass(frozen=True)
class GrabResult:
    """抢合作阶段结果。

    Attributes:
        found: 是否识别到准备按钮（True 表示抢到合作，应进入准备流程）
        reason: 结束原因：ready_found / window_lost / timeout / stopped /
            capture_failed / perception_failed / worker_failed / no_join_hotspot /
            popup_unclosable / unknown_popup；
            window_lost 为连点执行连续失败（窗口真丢失）或失焦/最小化持续
            超过 window_foreground_wait_seconds；
            popup_unclosable 为全屏弹窗关闭按钮重试用尽仍可见；
            unknown_popup 为已越过招募阶段后检测到无标准关闭按钮的未知全屏弹窗
    """

    found: bool
    reason: str


@dataclass(frozen=True)
class _CloseRequest:
    """检查线程识别到弹窗关闭按钮后写给抢线程的点击请求。

    point 为匹配中心（窗口客户区物理像素，与 join 点击同坐标系）；
    frame_id 为识别所依据的帧，供日志排查。
    """

    point: tuple[int, int]
    frame_id: int
    confidence: float


class CoopGrabCoordinator:
    """抢合作双线程协调器。详见模块文档。"""

    def __init__(
        self,
        *,
        screen: ScreenCapture,
        perception: CoopPerception,
        grab_executor: ActionExecutor,
        safety: SafetyGuard,
        hotspots: dict[str, Hotspot],
        run_config: RunConfig,
        window_handle: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        get_idle_seconds: Callable[[], float] = get_input_idle_seconds,
        activate: Callable[[int], bool] = activate_window,
        probe_window: Callable[[int], WindowInfo | None] = _default_probe_window,
        debug_recorder: DebugRecorder | None = None,
    ) -> None:
        self._screen = screen
        self._perception = perception
        self._grab_executor = grab_executor
        self._safety = safety
        self._hotspots = hotspots
        self._run_config = run_config
        self._window_handle = window_handle
        self._clock = clock
        self._sleep = sleep
        self._get_idle_seconds = get_idle_seconds
        self._activate = activate
        self._probe_window = probe_window
        self._debug_recorder = debug_recorder
        self._check_interval = run_config.find_coop_check_interval_seconds
        # 检查线程最新截图上下文，供抢合作线程换算坐标与窗口校验
        self._latest_ctx: WindowContext | None = None
        self._ctx_lock = threading.Lock()
        self._finish_lock = threading.Lock()
        # 弹窗关闭状态（run() 开始时重置，协调器实例跨抢合作段复用）：
        # _close_request 由检查线程写入、抢线程读取；_popup_close_attempts
        # 为关闭按钮连续可见拍数；_passed_recruit 表示已越过招募页（招募页
        # 边缘也是暗的，未越过时不判未知弹窗）；_unknown_popup_streak 为
        # 「边缘暗且无关闭按钮」连续拍数
        self._close_lock = threading.Lock()
        self._close_request: _CloseRequest | None = None
        self._popup_close_attempts = 0
        self._passed_recruit = False
        self._unknown_popup_streak = 0

    def run(self) -> GrabResult:
        """运行抢合作阶段，阻塞直到发现准备按钮、超时、窗口失效或停止。"""
        handle = self._window_handle
        spot = self._hotspots.get("join_coop")
        if spot is None:
            logger.error("join_coop hotspot 缺失，无法抢合作")
            return GrabResult(found=False, reason="no_join_hotspot")

        # 起始截图，给抢合作线程提供初始窗口上下文（后续由检查线程刷新）。
        # 起始即失焦/最小化不在这里快速失败：检查线程的挂起逻辑会等待切回
        # （真实 capture 对最小化窗口直接抛错，走不到这里，按 capture_failed 处理）
        try:
            ctx, _ = self._screen.capture(handle)
        except Exception as exc:
            logger.error("抢合作起始截图失败: %r", exc)
            return GrabResult(found=False, reason="capture_failed")
        with self._ctx_lock:
            self._latest_ctx = ctx
        with self._close_lock:
            self._close_request = None
        self._popup_close_attempts = 0
        self._passed_recruit = False
        self._unknown_popup_streak = 0

        deadline = self._clock() + self._run_config.find_coop_max_duration_seconds
        stop_event = threading.Event()
        holder: list[GrabResult | None] = [None]

        logger.info(
            "开始抢合作：连点 join_coop（间隔 %.2f~%.2fs）并行识别准备按钮（每 %.2fs），最长 %.0fs",
            self._run_config.find_coop_click_delay_min,
            self._run_config.find_coop_click_delay_max,
            self._check_interval,
            self._run_config.find_coop_max_duration_seconds,
        )

        grab_thread = threading.Thread(
            target=self._run_grab_worker,
            args=(spot, stop_event, holder, deadline),
            name="coop-grab",
            daemon=True,
        )
        check_thread = threading.Thread(
            target=self._run_ready_watch_worker,
            args=(stop_event, holder, deadline),
            name="coop-ready-watch",
            daemon=True,
        )
        grab_thread.start()
        check_thread.start()
        grab_thread.join()
        check_thread.join()
        stop_event.set()

        result = holder[0]
        if result is not None:
            logger.info("抢合作结束 reason=%s", result.reason)
            return result
        if self._safety.stop_requested:
            return GrabResult(found=False, reason="stopped")
        return GrabResult(found=False, reason="timeout")

    def _run_grab_worker(
        self,
        spot: Hotspot,
        stop_event: threading.Event,
        holder: list[GrabResult | None],
        deadline: float,
    ) -> None:
        """线程根边界：保留未预料异常，避免工作线程静默退出后等待到超时。"""
        try:
            self._grab_loop(spot, stop_event, holder, deadline)
        except Exception:
            logger.exception("抢合作点击线程异常")
            self._finish(stop_event, holder, GrabResult(False, "worker_failed"))

    def _run_ready_watch_worker(
        self,
        stop_event: threading.Event,
        holder: list[GrabResult | None],
        deadline: float,
    ) -> None:
        """线程根边界：保留未预料异常，避免工作线程静默退出后等待到超时。"""
        try:
            self._ready_watch_loop(stop_event, holder, deadline)
        except Exception:
            logger.exception("抢合作识别线程异常")
            self._finish(stop_event, holder, GrabResult(False, "worker_failed"))

    def _finish(
        self,
        stop_event: threading.Event,
        holder: list[GrabResult | None],
        result: GrabResult,
    ) -> None:
        """线程安全的"首个结论生效"：第一个调用方写入结果并通知停止。"""
        with self._finish_lock:
            if stop_event.is_set():
                return
            holder[0] = result
            stop_event.set()

    def _read_close_request(self) -> _CloseRequest | None:
        """读取检查线程当前写入的弹窗关闭请求（不清除，由检查线程维护生命周期）。"""
        with self._close_lock:
            return self._close_request

    def _grab_loop(
        self,
        spot: Hotspot,
        stop_event: threading.Event,
        holder: list[GrabResult | None],
        deadline: float,
    ) -> None:
        """连点 join_coop，不做识别；窗口真丢失（连续执行失败）才报 window_lost。

        检查线程识别到弹窗关闭按钮时，本线程的点击目标暂时从 join 切换为
        关闭按钮（弹窗出现即已加入成功，暂停连点无损失）；点击成功后等一个
        检查周期再恢复，由检查线程下一拍确认弹窗是否已关。
        """
        consecutive_failures = 0
        while not stop_event.is_set():
            if self._safety.stop_requested:
                self._finish(stop_event, holder, GrabResult(False, "stopped"))
                return
            if self._clock() >= deadline:
                return
            with self._ctx_lock:
                ctx = self._latest_ctx
            if ctx is None:
                self._sleep(0.05)
                continue
            close_req = self._read_close_request()
            if close_req is not None:
                action = Action(
                    kind="click",
                    target=close_req.point,
                    duration=0.08,
                    verification="immediate",
                    tag="close_grab_popup",
                    reason="抢合作期间识别到弹窗关闭按钮，点击关闭",
                )
            else:
                point = hotspot_to_client_point(spot, ctx.client_size)
                action = Action(
                    kind="click",
                    target=point,
                    duration=0.08,
                    verification="immediate",
                    tag="find_coop_click",
                    reason="抢合作连点 join_coop",
                )
            result = self._grab_executor.execute(ctx, action)
            if not result.executed:
                failure = result.failure_reason or "未知失败"
                if failure in _TRANSIENT_FAILURE_REASONS:
                    # 瞬态失败，窗口没丢：失焦/最小化由检查线程的挂起逻辑
                    # 等待切回（空闲自动切回，切回后本线程自动恢复连点）；
                    # 上下文超龄只是检查线程刷新周期偶发超过 frame_ttl，
                    # 等它刷新即可，都不累计失败
                    self._sleep(0.1)
                    continue
                consecutive_failures += 1
                # 非瞬态失败（句柄失效、几何变化、输入异常）连续达到上限，
                # 视为窗口已丢失，保守停止，不再盲点屏幕坐标
                if consecutive_failures >= _GRAB_FAILURE_LIMIT:
                    logger.warning(
                        "抢合作连点连续 %d 次执行失败（最后原因: %s），判定窗口已丢失",
                        consecutive_failures,
                        failure,
                    )
                    self._finish(stop_event, holder, GrabResult(False, "window_lost"))
                    return
                self._sleep(0.1)
                continue
            consecutive_failures = 0
            if close_req is not None:
                logger.info(
                    "抢合作点击弹窗关闭按钮 frame=%d pos=%s conf=%.3f",
                    close_req.frame_id,
                    close_req.point,
                    close_req.confidence,
                )
                # 等检查线程下一拍确认弹窗已关（期间不连点 join、不重复点关闭），
                # 仍可见则检查线程会刷新请求，本线程自然再点一次
                stop_event.wait(self._check_interval)
            # 执行器内部已按 find_coop_click_delay_min/max 做了拟人延迟

    def _suspend_for_foreground(
        self,
        stop_event: threading.Event,
        holder: list[GrabResult | None],
        minimized: bool,
        suspension: _ForegroundSuspension,
    ) -> bool:
        """挂起等待窗口切回前台，与主循环同策略。

        期间不发送输入（点击由执行器逐击实时校验拒绝，不会落到别的窗口），
        系统空闲时自动把游戏窗口切回前台，持续超过 window_foreground_wait_seconds
        才报 window_lost。

        Args:
            stop_event: 结束信号
            holder: 结果槽
            minimized: 是否最小化（否则只是非前台）
            suspension: 挂起计时状态（跨多次调用保留，恢复前台后 reset）

        Returns:
            True: 已等待一个检查周期，继续循环；False: 超时，已写入 window_lost
        """
        reason = "最小化" if minimized else "非前台"
        now = self._clock()
        if suspension.since is None:
            suspension.since = now
            suspension.last_log = now
            logger.warning(
                "抢合作期间窗口%s，暂停连点挂起等待切回（期间不发送输入）", reason
            )
        waited = now - suspension.since
        timeout = self._run_config.window_foreground_wait_seconds
        if waited >= timeout:
            logger.error(
                "抢合作期间窗口%s已持续 %.0f 秒（上限 %.0f 秒，可用 "
                "run.window_foreground_wait_seconds 调大），停止任务",
                reason,
                waited,
                timeout,
            )
            self._finish(stop_event, holder, GrabResult(False, "window_lost"))
            return False
        if now - suspension.last_log >= _FG_LOG_INTERVAL_SECONDS:
            suspension.last_log = now
            logger.warning(
                "窗口%s，抢合作挂起等待切回（已等 %.0f/%.0f 秒，期间不发送输入）",
                reason,
                waited,
                timeout,
            )
        if self._run_config.refocus_when_idle:
            idle = self._get_idle_seconds()
            if (
                idle >= self._run_config.refocus_idle_seconds
                and now - suspension.last_refocus >= _REFOCUS_RETRY_SECONDS
            ):
                suspension.last_refocus = now
                if self._activate(self._window_handle):
                    logger.info(
                        "系统已 %.0f 秒无鼠标/键盘活动，自动切回游戏窗口继续抢合作",
                        idle,
                    )
                else:
                    logger.warning(
                        "系统空闲 %.0f 秒，但游戏窗口激活失败，继续挂起等待", idle
                    )
        stop_event.wait(self._check_interval)
        return True

    def _ready_watch_loop(
        self,
        stop_event: threading.Event,
        holder: list[GrabResult | None],
        deadline: float,
    ) -> None:
        """周期截图并匹配准备按钮；识别到即通知停止。绝不点击。

        准备按钮未命中时顺带匹配全屏弹窗关闭按钮（×）：命中则把匹配位置
        写给抢线程点击关闭，并在后续拍验证关闭结果（详见模块文档）。

        窗口失焦/最小化时挂起等待切回（与主循环同策略），期间不识别不点击；
        持续超过 window_foreground_wait_seconds 才报 window_lost。
        """
        handle = self._window_handle
        capture_failures = 0
        perception_failures = 0
        suspension = _ForegroundSuspension()
        while not stop_event.is_set():
            if self._safety.stop_requested:
                self._finish(stop_event, holder, GrabResult(False, "stopped"))
                return
            if self._clock() >= deadline:
                return
            try:
                ctx, frame = self._screen.capture(handle)
            except Exception as exc:
                # capture 对最小化窗口直接抛错：先探测实时窗口状态，确认是
                # 失焦/最小化则转入挂起等待，其余才算截图失败
                info = self._probe_window(handle)
                if info is not None and (info.is_minimized or not info.is_foreground):
                    if not self._suspend_for_foreground(
                        stop_event, holder, info.is_minimized, suspension
                    ):
                        return
                    continue
                capture_failures += 1
                logger.warning("抢合作检查截图失败 (%d): %r", capture_failures, exc)
                if capture_failures >= _GRAB_FAILURE_LIMIT:
                    self._finish(stop_event, holder, GrabResult(False, "capture_failed"))
                    return
                self._sleep(self._check_interval)
                continue
            capture_failures = 0
            if ctx.is_minimized or not ctx.is_foreground:
                if not self._suspend_for_foreground(
                    stop_event, holder, ctx.is_minimized, suspension
                ):
                    return
                continue
            suspension.reset()
            with self._ctx_lock:
                self._latest_ctx = ctx
            # 前台有效截图进入退出帧缓冲：抢合作阶段丢失窗口瞬间的画面
            # 也能落盘排查（挂起期间的不在前台帧不冲掉历史）
            if self._debug_recorder is not None:
                self._debug_recorder.keep_exit_frame(ctx.frame_id, ctx.captured_at, frame)
            try:
                match = self._perception.match_ready_button(ctx, frame)
            except (OSError, RuntimeError, ValueError) as exc:
                perception_failures += 1
                logger.warning("抢合作准备按钮识别失败 (%d): %r", perception_failures, exc)
                if perception_failures >= _GRAB_FAILURE_LIMIT:
                    self._finish(stop_event, holder, GrabResult(False, "perception_failed"))
                    return
                stop_event.wait(self._check_interval)
                continue
            perception_failures = 0
            if match is not None:
                logger.info(
                    "抢合作识别到准备按钮 frame=%d pos=%s conf=%.3f",
                    ctx.frame_id,
                    match.position,
                    match.confidence,
                )
                self._finish(stop_event, holder, GrabResult(True, "ready_found"))
                return
            if not self._watch_popup(stop_event, holder, ctx, frame):
                return
            stop_event.wait(self._check_interval)

    def _watch_popup(
        self,
        stop_event: threading.Event,
        holder: list[GrabResult | None],
        ctx: WindowContext,
        frame: Any,
    ) -> bool:
        """检查线程的弹窗分支：识别关闭按钮并驱动关闭/未知弹窗保守停止。

        Returns:
            True: 继续下一拍检查；False: 已写入结论并通知停止。
        """
        try:
            close_match = self._perception.match_close_button(ctx, frame)
            modal = self._perception.is_modal_frame(
                frame, self._run_config.modal_edge_brightness
            )
        except (OSError, RuntimeError, ValueError) as exc:
            # 关闭按钮识别失败不中断抢合作：退化为主流程（只找准备按钮）
            logger.warning("抢合作弹窗识别异常，本拍跳过弹窗分支: %r", exc)
            return True
        if close_match is not None:
            # 弹窗可见即已离开招募页；连续可见拍数 = 已发出的关闭点击数
            self._passed_recruit = True
            self._unknown_popup_streak = 0
            self._popup_close_attempts += 1
            # 与 runner 侧同语义：首次点击外再重试 close_popup_max_retries 次
            max_attempts = self._run_config.close_popup_max_retries + 1
            if self._popup_close_attempts > max_attempts:
                logger.error(
                    "抢合作弹窗关闭按钮连续 %d 拍仍可见（首次点击+%d 次重试），判定"
                    "弹窗关不住，保守停止；退出帧缓冲已保留最后画面",
                    self._popup_close_attempts,
                    self._run_config.close_popup_max_retries,
                )
                self._finish(stop_event, holder, GrabResult(False, "popup_unclosable"))
                return False
            request = _CloseRequest(
                point=close_match.position,
                frame_id=ctx.frame_id,
                confidence=close_match.confidence,
            )
            with self._close_lock:
                self._close_request = request
            logger.info(
                "抢合作识别到弹窗关闭按钮 frame=%d pos=%s conf=%.3f，通知点击关闭 (%d/%d)",
                ctx.frame_id,
                close_match.position,
                close_match.confidence,
                self._popup_close_attempts,
                max_attempts,
            )
            return True
        # 关闭按钮不可见：清请求，弹窗已关则抢线程恢复连点
        with self._close_lock:
            self._close_request = None
        self._popup_close_attempts = 0
        if not modal:
            # 边缘亮 = 组队大厅/对局画面，已越过招募页
            self._passed_recruit = True
            self._unknown_popup_streak = 0
            return True
        if not self._passed_recruit:
            # 招募页等全屏面板边缘也是暗的：正常抢的过程中每拍都会走到这里，
            # 未越过招募阶段不判未知弹窗
            return True
        self._unknown_popup_streak += 1
        confirmations = self._run_config.grab_unknown_popup_confirmations
        if self._unknown_popup_streak < confirmations:
            return True
        logger.error(
            "抢合作期间检测到未知全屏弹窗：画面边缘持续变暗（连续 %d 拍）且识别"
            "不到标准关闭按钮，保守停止；退出帧缓冲已保留最后画面，可据此补充模板",
            self._unknown_popup_streak,
        )
        self._finish(stop_event, holder, GrabResult(False, "unknown_popup"))
        return False
