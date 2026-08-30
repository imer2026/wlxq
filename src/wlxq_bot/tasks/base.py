"""Task 基类和状态机基础。

任务设计为状态机而非纯顺序脚本：
1. 截图并生成 WindowContext
2. 识别 Observation
3. 按优先级确定 State
4. 选择该 State 允许的 Transition
5. Safety Guard 执行动作前检查
6. 执行动作
7. 再次截图并验证目标 State
8. 成功则继续，失败则按规则重试或暂停

状态判断优先级固定：
停止/窗口异常 > 阻塞性弹窗 > 当前任务界面 > 普通入口 > UNKNOWN
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from wlxq_bot.models import Action, ActionResult, Observation, State, Transition, WindowContext


@dataclass
class TaskContext:
    """任务运行时上下文。

    贯穿一次任务执行的所有状态判断和动作调度。
    """

    main_c: str = ""
    current_state: State = State.UNKNOWN
    round_count: int = 0
    max_rounds: int = 20
    transitions: list[Transition] = field(default_factory=list)


class Task(ABC):
    """任务状态机基类。

    子类实现 build_transitions 和 determine_state，
    由 Task Engine 驱动执行循环。

    动态动作生成：默认 step 使用 transition.action（静态坐标）。
    需要运行时计算坐标的任务（如拖动棋盘上两个英雄合成）可覆盖
    decide_action，返回 (Action, 目标 State)；Runner 会优先使用它。
    """

    def __init__(self, ctx: TaskContext) -> None:
        self.ctx = ctx
        if not self.ctx.transitions:
            self.ctx.transitions = self.build_transitions()

    @abstractmethod
    def build_transitions(self) -> list[Transition]:
        """构建该任务的状态迁移规则。"""

    @abstractmethod
    def determine_state(self, observation: Observation) -> State:
        """根据识别结果判断当前界面状态。

        实现时遵循固定优先级：
        停止/窗口异常 > 阻塞性弹窗 > 当前任务界面 > 普通入口 > UNKNOWN
        """

    def decide_action(
        self,
        observation: Observation,
        window_ctx: WindowContext | None = None,
    ) -> tuple[Action, State] | None:
        """根据识别结果动态生成动作和目标状态。

        默认返回 None，表示使用 select_transition 选出的静态 transition.action。
        需要运行时计算坐标的子类覆盖此方法。

        Args:
            observation: 当前帧识别结果
            window_ctx: 当前窗口上下文（坐标换算用，可为 None）

        Returns:
            (Action, 目标 State) 或 None（无可用动作，应保守停止）
        """
        return None

    def select_transition(self, state: State) -> Transition | None:
        """选择当前状态下允许执行的迁移。"""
        candidates = [t for t in self.ctx.transitions if t.from_state == state]
        if not candidates:
            return None
        # 首个匹配，子类可覆盖实现优先级
        return candidates[0]

    def observation_mode(self) -> str | None:
        """返回当前步骤需要的专项识别模式；默认只做顶层状态识别。"""
        return None

    def wants_board_watch(self) -> bool:
        """当前步骤是否需要多帧棋盘识别；默认需要。

        不以棋盘识别为门禁的快速阶段（如技能解锁前的强制召唤）可覆盖为
        False，Runner 将退回单帧界面标志识别以加快节奏。
        """
        return True

    def on_action_verified(self, action: Action, to_state: State) -> None:
        """动作经新 Observation 验证后更新任务内部进度。

        默认没有额外状态；需要计数或记录阶段进度的任务可覆盖。
        """
        return None

    def on_action_failed(self, action: Action) -> None:
        """动作重试耗尽、Runner 放弃该动作时通知任务调整内部进度。

        默认没有额外状态；可恢复的动作（如技能卡点击未生效）可覆盖，
        任务不应因此终止——由 Runner 继续驱动主循环。
        """
        return None

    def verify_action(
        self,
        action: Action,
        before: Observation,
        after: Observation,
    ) -> bool:
        """使用新观察验证输入动作的业务后置条件。

        子类必须为需要 ``next_frame`` 验证的动作实现明确规则；默认保守失败。
        """
        return False

    def step(
        self,
        observation: Observation,
        execute_action: object,
    ) -> tuple[Transition | None, ActionResult]:
        """执行一次状态机步进。

        Args:
            observation: 当前帧识别结果
            execute_action: 动作执行回调，签名为 (Action) -> ActionResult

        Returns:
            (执行的 Transition 或 None, ActionResult)
        """
        self.ctx.current_state = self.determine_state(observation)
        transition = self.select_transition(self.ctx.current_state)
        if transition is None:
            return None, ActionResult(
                executed=False,
                verified=False,
                failure_reason=f"状态 {self.ctx.current_state} 无可用迁移",
            )
        result = execute_action(transition.action)
        if result.verified:
            self.on_action_verified(transition.action, transition.to_state)
            self.ctx.current_state = transition.to_state
        return transition, result
