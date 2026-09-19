"""Safety Guard：停止信号、失败次数、窗口状态和动作边界检查。

每个动作执行前必须通过 Safety Guard 检查：
- 全局停止信号是否触发
- 窗口句柄是否有效、是否前台、是否最小化
- 客户区位置、尺寸、DPI 是否与识别时一致
- 截图是否超时（frame_ttl_ms）
- 最终屏幕坐标是否落在客户区范围内
- 连续失败次数是否超限

停止信号由独立监听器设置线程安全标志，
截图、等待和动作执行都要及时检查该信号。
"""

from __future__ import annotations

import threading
import time

from wlxq_bot.models import Action, WindowContext


class SafetyGuard:
    """安全检查器。"""

    def __init__(
        self,
        max_failures: int = 5,
        frame_ttl_ms: int = 500,
        frame_ttl_max_ms: int | None = None,
    ) -> None:
        self._max_failures = max_failures
        self._frame_ttl_base_ms = frame_ttl_ms
        # 慢机自适应上限：识别持续超过基础时效时生效时效翻倍放宽到该值
        # （机器慢只是动作变慢，不应因此停止）；None 或小于基础值 = 不放宽
        self._frame_ttl_max_ms = (
            frame_ttl_ms
            if frame_ttl_max_ms is None
            else max(frame_ttl_max_ms, frame_ttl_ms)
        )
        self._frame_ttl_ms = frame_ttl_ms
        self._stop_flag = threading.Event()
        self._failure_count = 0

    @property
    def frame_ttl_ms(self) -> int:
        """当前生效的截图时效（毫秒），慢机自适应放宽后的值。"""
        return self._frame_ttl_ms

    @property
    def frame_ttl_base_ms(self) -> int:
        """配置的基础截图时效（毫秒），机器恢复后的回落目标。"""
        return self._frame_ttl_base_ms

    @property
    def frame_ttl_max_ms(self) -> int:
        """生效截图时效的放宽上限（毫秒）。"""
        return self._frame_ttl_max_ms

    def relax_frame_ttl(self) -> bool:
        """生效时效翻倍（封顶 frame_ttl_max_ms）。

        识别耗时超过当前时效时由调度循环调用。返回是否成功放宽；
        False 表示已在上限，继续超龄应走丢弃/保守停止路径。
        """
        if self._frame_ttl_ms >= self._frame_ttl_max_ms:
            return False
        self._frame_ttl_ms = min(self._frame_ttl_ms * 2, self._frame_ttl_max_ms)
        return True

    def reset_frame_ttl(self) -> None:
        """生效时效回落到基础值（决策重新在基础时效内新鲜时调用）。"""
        self._frame_ttl_ms = self._frame_ttl_base_ms

    def request_stop(self) -> None:
        """请求停止任务。"""
        self._stop_flag.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop_flag.is_set()

    def start_esc_listener(self) -> bool:
        """启动后台线程监听 Esc 键，按下时调用 request_stop。

        用 GetAsyncKeyState 轮询（pywin32 已装），无需额外依赖。
        任务循环在每轮开头检查 stop_requested，按下 Esc 后下一轮即停止。
        非 Windows 环境返回 False。

        Returns:
            是否成功启动监听
        """
        try:
            import win32api
        except ImportError:
            return False

        VK_ESCAPE = 0x1B

        def _watch() -> None:
            while not self._stop_flag.is_set():
                # GetAsyncKeyState 高位为 1 表示当前按下
                if win32api.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
                    self.request_stop()
                    break
                time.sleep(0.05)

        thread = threading.Thread(target=_watch, daemon=True, name="esc-listener")
        thread.start()
        return True

    def reset_failures(self) -> None:
        self._failure_count = 0

    def record_failure(self) -> bool:
        """记录一次失败，返回是否已达上限。"""
        self._failure_count += 1
        return self._failure_count >= self._max_failures

    def check_action(
        self,
        ctx: WindowContext,
        action: Action,
        now: float,
    ) -> tuple[bool, str]:
        """执行动作前的安全检查。

        Args:
            ctx: 当前窗口上下文
            action: 待执行动作
            now: 当前时间戳

        Returns:
            (是否通过, 失败原因)
        """
        if self.stop_requested:
            return False, "停止信号已触发"

        if ctx.is_minimized or not ctx.is_foreground:
            return False, "窗口最小化或非前台"

        if not ctx.is_valid(now, self._frame_ttl_ms):
            return False, "窗口上下文已超时"

        # 点击/拖动坐标必须在客户区内
        if action.target is not None:
            tx, ty = action.target
            if not ctx.contains(tx, ty):
                return False, f"动作坐标 ({tx}, {ty}) 超出客户区"

        if action.kind == "drag" and action.end is not None:
            ex, ey = action.end
            if not ctx.contains(ex, ey):
                return False, f"拖动终点 ({ex}, {ey}) 超出客户区"

        return True, ""
