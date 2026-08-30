"""Action Executor：动作执行入口。

串联 Safety Guard 和 Input Controller：
1. 执行动作前调用 Safety Guard 检查
2. 检查通过后将客户区坐标转换为屏幕坐标
3. 调用 Input Controller 执行实际输入
4. 记录调试事件
5. 返回 ActionResult

任务代码只通过 Action Executor 执行输入。
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable

from wlxq_bot.action.input import InputController
from wlxq_bot.action.safety import SafetyGuard
from wlxq_bot.models import Action, ActionResult, WindowContext


class ActionExecutor:
    """动作执行器。

    TODO:
        - 接入任务级动作后验证结果
        - 实现调试事件记录
    """

    def __init__(
        self,
        safety: SafetyGuard,
        input_ctrl: InputController,
        min_delay: float = 0.3,
        max_delay: float = 0.8,
        context_validator: Callable[[WindowContext], tuple[bool, str]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._safety = safety
        self._input = input_ctrl
        self._min_delay = min_delay
        self._max_delay = max_delay
        self._context_validator = context_validator
        self._sleep = sleep

    def execute(
        self,
        ctx: WindowContext,
        action: Action,
    ) -> ActionResult:
        """执行一个动作。

        Args:
            ctx: 当前窗口上下文
            action: 待执行动作

        Returns:
            动作执行结果
        """
        if self._context_validator is not None:
            context_ok, context_reason = self._context_validator(ctx)
            if not context_ok:
                return ActionResult(
                    executed=False,
                    verified=False,
                    failure_reason=context_reason,
                )

        now = time.time()
        ok, reason = self._safety.check_action(ctx, action, now)
        if not ok:
            return ActionResult(executed=False, verified=False, failure_reason=reason)

        try:
            self._dispatch(ctx, action)
            # 输入动作后加拟人间隔（显式稳定等待优先）；wait 的 duration 就是
            # 全部等待，不再叠加随机延迟——否则轮询类等待的实际间隔会远大于
            # 配置值（如 0.25 秒轮询实际跑 0.55~1.05 秒，2026-08-21 实机发现）
            if action.kind != "wait":
                delay = (
                    action.post_delay
                    if action.post_delay > 0
                    else random.uniform(self._min_delay, self._max_delay)
                )
                self._sleep(delay)
        except Exception as exc:
            return ActionResult(
                executed=False,
                verified=False,
                failure_reason=f"输入执行异常: {exc!r}",
            )

        # wait 不改变外部界面，可由执行器直接确认完成；输入动作仍须由
        # Runner/Task Engine 使用下一帧 Observation 做业务状态验证。
        if action.kind == "wait":
            return ActionResult(executed=True, verified=True)
        return ActionResult(executed=True, verified=False, failure_reason="等待动作后状态验证")

    def _dispatch(self, ctx: WindowContext, action: Action) -> None:
        """根据动作类型分发到 InputController。"""
        if action.kind == "click" and action.target is not None:
            sx, sy = ctx.client_to_screen(*action.target)
            self._input.click(sx, sy, action.duration or 0.08)

        elif action.kind == "drag" and action.target is not None and action.end is not None:
            start_screen = ctx.client_to_screen(*action.target)
            end_screen = ctx.client_to_screen(*action.end)
            self._input.drag(start_screen, end_screen, action.duration or 0.5)

        elif action.kind == "key" and action.key is not None:
            self._input.press_key(action.key)

        elif action.kind == "wait":
            # duration 就是全部等待时长；显式 0 表示「立即推进」的确认类等待
            self._sleep(action.duration if action.duration is not None else 1.0)

        else:
            raise ValueError(f"未知动作类型或参数不完整: {action}")
