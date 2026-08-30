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
            capture_failed / perception_failed / worker_failed / no_join_hotspot；
            window_lost 为连点执行连续失败（窗口真丢失）或失焦/最小化持续
            超过 window_foreground_wait_seconds
    """

    found: bool
    reason: str


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

    def _grab_loop(
        self,
        spot: Hotspot,
        stop_event: threading.Event,
        holder: list[GrabResult | None],
        deadline: float,
    ) -> None:
        """连点 join_coop，不做识别；窗口真丢失（连续执行失败）才报 window_lost。"""
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
            stop_event.wait(self._check_interval)
