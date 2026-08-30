"""核心数据模型。

定义任务状态机、感知结果和动作执行所需的最小概念集合。
对应 docs/architecture.md「核心概念模型」一节。

所有坐标统一使用客户区物理像素（左上角为原点），
仅在实际输入前通过 WindowContext 转换为屏幕坐标。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# 运行时上下文
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowContext:
    """一次截图、识别和动作执行共同使用的窗口快照。

    Attributes:
        window_handle: 窗口句柄
        client_rect_screen: 客户区在屏幕坐标系中的位置 (x, y, width, height)
        client_size: 客户区物理像素尺寸 (width, height)
        dpi: 当前 DPI 缩放
        monitor_id: 所在显示器标识
        is_foreground: 窗口是否在前台
        is_minimized: 窗口是否最小化
        captured_at: 截图时间戳（秒）
        frame_id: 帧标识，用于关联截图、识别和动作
    """

    window_handle: int
    client_rect_screen: tuple[int, int, int, int]
    client_size: tuple[int, int]
    dpi: int
    monitor_id: str
    is_foreground: bool
    is_minimized: bool
    captured_at: float
    frame_id: int

    def is_valid(self, now: float, ttl_ms: int) -> bool:
        """检查窗口上下文是否仍然有效。

        窗口最小化、非前台或截图已超时均视为无效。
        """
        if self.is_minimized or not self.is_foreground:
            return False
        return (now - self.captured_at) * 1000 <= ttl_ms

    def client_to_screen(self, x: int, y: int) -> tuple[int, int]:
        """客户区坐标转屏幕坐标。"""
        sx, sy, _, _ = self.client_rect_screen
        return (sx + x, sy + y)

    def contains(self, x: int, y: int) -> bool:
        """判断客户区坐标是否落在客户区内。"""
        w, h = self.client_size
        return 0 <= x < w and 0 <= y < h


# ---------------------------------------------------------------------------
# 感知结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchResult:
    """一次模板匹配结果。

    Attributes:
        template_name: 模板名称
        position: 匹配中心在客户区坐标系中的位置 (x, y)
        confidence: 置信度 [0, 1]
        roi_name: 匹配所用的 ROI 名称
        direction: 朝向，仅用于英雄模板，其他场景为 None
    """

    template_name: str
    position: tuple[int, int]
    confidence: float
    roi_name: str | None = None
    direction: str | None = None


@dataclass(frozen=True)
class SkillCandidate:
    """技能候选识别结果。"""

    skill_id: str
    position: tuple[int, int]
    confidence: float
    template_path: str = ""


@dataclass(frozen=True)
class DifficultyCandidate:
    """合作难度候选识别结果。"""

    level: int
    position: tuple[int, int]
    confidence: float
    template_path: str = ""


@dataclass(frozen=True)
class Observation:
    """从一帧或同一稳定窗口上下文的短帧序列得到的识别结果。

    包含当前帧识别到的按钮、弹窗、技能和英雄等信息。
    任务状态机据此判断当前界面状态。

    Attributes:
        frame_id: 关联的帧标识（与 WindowContext.frame_id 一致）
        source_frame_ids: 参与时序融合的帧标识；单帧观察仅包含 frame_id
        matches: 命中的模板匹配结果列表
        raw_data: 附加识别数据，例如界面标志（ready_button_visible / return_button_visible
                  等 bool）和状态诊断信息；任务代码通过键名读取，避免在 Observation
                  上为每个标志单独建字段
        board: 己方棋盘快照，仅在培养主 C 阶段识别棋盘时填充，其他阶段为 None
        difficulty_candidates: 当前难度弹窗内可见的难度候选
    """

    frame_id: int
    source_frame_ids: tuple[int, ...] = field(default_factory=tuple)
    matches: list[MatchResult] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)
    board: BoardSnapshot | None = None
    skill_candidates: list[SkillCandidate] = field(default_factory=list)
    difficulty_candidates: list[DifficultyCandidate] = field(default_factory=list)

    def best_match(self, template_name: str) -> MatchResult | None:
        """返回指定模板置信度最高的匹配。"""
        candidates = [m for m in self.matches if m.template_name == template_name]
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.confidence)

    def flag(self, name: str, default: Any = False) -> Any:
        """读取 raw_data 中的界面标志，不存在时返回 default。"""
        return self.raw_data.get(name, default)


# ---------------------------------------------------------------------------
# 棋盘模型
# ---------------------------------------------------------------------------


class CoopRole(str, Enum):
    """合作模式中的玩家角色。

    对应 game-rules.md「棋盘与合作模式」：发起合作的玩家（initiator）
    位于左下方，帮助合作的玩家（helper）位于右下方。本脚本常规流程是
    抢其他玩家发起的合作并加入，因此角色通常为 helper，己方棋盘在右下方。

    Attributes:
        INITIATOR: 邀请者，己方棋盘位于画面左下（bottom_left_board）
        HELPER: 帮助者，己方棋盘位于画面右下（bottom_right_board）
    """

    INITIATOR = "initiator"
    HELPER = "helper"


# 角色 → 己方棋盘 ROI 配置名映射。
# 对应 game-rules.md：co-op + initiator → bottom_left_board，
# co-op + helper → bottom_right_board。
_BOARD_ROI_NAME: dict[CoopRole, str] = {
    CoopRole.INITIATOR: "bottom_left_board",
    CoopRole.HELPER: "bottom_right_board",
}


def board_roi_name(role: CoopRole) -> str:
    """返回角色对应的己方棋盘 ROI 配置名。

    initiator → bottom_left_board
    helper    → bottom_right_board

    Args:
        role: 合作模式中的玩家角色

    Returns:
        该角色己方棋盘在 tasks.yaml ``rois`` 段中的配置键名
    """
    return _BOARD_ROI_NAME[role]


def parse_hero_template_path(path: str | Path) -> tuple[str, int]:
    """从模板路径反推英雄类型和星级。

    期望路径含 ``heroes/<hero_type>/star<N>/`` 结构。
    例: ``assets/templates/927x1727/heroes/assault/star1/left.png`` → ``("assault", 1)``

    Args:
        path: 模板文件路径

    Returns:
        ``(hero_type, star_level)`` 元组

    Raises:
        ValueError: 路径不含 heroes/ 目录，或星级目录格式不符
    """
    parts = Path(path).parts
    try:
        heroes_idx = parts.index("heroes")
    except ValueError:
        raise ValueError(f"模板路径不含 heroes/ 目录: {path}") from None
    if heroes_idx + 2 >= len(parts):
        raise ValueError(f"模板路径格式不符 heroes/<hero>/star<N>/...: {path}")
    hero_type = parts[heroes_idx + 1]
    star_dir = parts[heroes_idx + 2]
    if not star_dir.startswith("star"):
        raise ValueError(f"星级目录名不以 star 开头: {star_dir}（路径: {path}）")
    try:
        star_level = int(star_dir[len("star") :])
    except ValueError:
        raise ValueError(f"星级目录名格式不符 star<N>: {star_dir}（路径: {path}）") from None
    if not 1 <= star_level <= 4:
        raise ValueError(f"星级超出 1-4 范围: {star_level}（路径: {path}）")
    return hero_type, star_level


@dataclass(frozen=True)
class BoardHero:
    """棋盘上一个英雄。

    hero_type 和 star_level 从模板路径反推（见 parse_hero_template_path），
    不在 MatchResult 里冗余存储，保持 Vision 层纯识别结果与业务模型分层清晰。

    Attributes:
        hero_type: 英雄类型标识，如 ``"assault"`` / ``"monkey"``
        star_level: 星级，1~4
        position: 英雄中心在客户区坐标系中的位置 (x, y)
        confidence: 识别置信度 [0, 1]
        template_path: 命中的模板路径（调试用）
    """

    hero_type: str
    star_level: int
    position: tuple[int, int]
    confidence: float
    template_path: str = ""
    # 格名（如 ``1A``/``4B``，按玩家自身棋盘从外到内 A/B/C）；感知层构建时填充，
    # 供日志和调试使用。旧识别路径可能为空。
    cell_name: str = ""

    @classmethod
    def from_match(
        cls,
        match: MatchResult,
        template_path: str,
    ) -> BoardHero:
        """从 MatchResult 和模板路径构建 BoardHero。

        Args:
            match: 模板匹配结果（提供 position 和 confidence）
            template_path: 命中的模板路径，含 heroes/<hero>/star<N>/ 结构
        """
        hero_type, star_level = parse_hero_template_path(template_path)
        return cls(
            hero_type=hero_type,
            star_level=star_level,
            position=match.position,
            confidence=match.confidence,
            template_path=template_path,
        )


@dataclass(frozen=True)
class BoardCapacity:
    """棋盘格容量状态。

    合作模式不要求视觉区分未开放格与空格；该模型只保留为辅助统计。

    Attributes:
        total_slots: 当前已开放格数
        occupied: 已被英雄占用的格数
    """

    total_slots: int
    occupied: int

    @property
    def available(self) -> int:
        """可用格数 = 已开放 - 已占用。"""
        return self.total_slots - self.occupied


@dataclass(frozen=True)
class MergeCandidate:
    """合法合成对：同类型同星级的两个英雄。

    game-rules.md 合成规则：只有英雄类型相同且星级相同的两个英雄才能合成。

    Attributes:
        hero_a: 第一个英雄
        hero_b: 第二个英雄
        is_main_c: 是否涉及主 C（影响合成优先级，非主 C 优先合成）
    """

    hero_a: BoardHero
    hero_b: BoardHero
    is_main_c: bool = False


@dataclass(frozen=True)
class BoardSnapshot:
    """某一识别时刻的完整棋盘状态。

    一次截图识别后构建，包含棋盘上所有英雄和格子容量。
    合成判断基于此快照，不依赖历史状态。

    Attributes:
        frame_id: 关联的帧标识（与 WindowContext.frame_id 一致）
        heroes: 棋盘上所有识别到的英雄
        capacity: 棋盘格容量状态
        captured_at: 截图时间戳（秒）
        source_frame_ids: 参与时序融合的帧标识；单帧快照仅包含 frame_id
    """

    frame_id: int
    heroes: list[BoardHero]
    capacity: BoardCapacity
    captured_at: float = 0.0
    source_frame_ids: tuple[int, ...] = field(default_factory=tuple)

    def find_merge_candidates(self, main_c: str) -> list[MergeCandidate]:
        """找合法合成对：同类型同星级。

        每个英雄最多参与一次配对。多个候选同时存在时，非主 C 优先
        （game-rules.md：优先合成非主 C 合法对）。

        Args:
            main_c: 当前主 C 的 hero_type，如 ``"assault"``

        Returns:
            合法合成对列表，非主 C 对排前面。
        """
        candidates: list[MergeCandidate] = []
        used: set[int] = set()
        for i, hero_a in enumerate(self.heroes):
            if i in used:
                continue
            for j in range(i + 1, len(self.heroes)):
                if j in used:
                    continue
                hero_b = self.heroes[j]
                if hero_a.hero_type == hero_b.hero_type and hero_a.star_level == hero_b.star_level:
                    is_mc = hero_a.hero_type == main_c
                    candidates.append(
                        MergeCandidate(
                            hero_a=hero_a,
                            hero_b=hero_b,
                            is_main_c=is_mc,
                        )
                    )
                    used.add(i)
                    used.add(j)
                    break
        # 非主 C 优先（is_main_c=False 排前面）
        candidates.sort(key=lambda c: c.is_main_c)
        return candidates


# ---------------------------------------------------------------------------
# 棋盘格子布局
# ---------------------------------------------------------------------------


class BoardCellType(str, Enum):
    """棋盘格子类型。

    Attributes:
        HERO: 英雄位，可召唤/合成/识别
        PET: 宠物位，不出现英雄，识别时跳过
    """

    HERO = "hero"
    PET = "pet"


@dataclass(frozen=True)
class BoardCell:
    """棋盘上一个格子的定义。

    Attributes:
        row: 排号 0-5（排1=0，排6=5，排1在顶部）
        col: 列号 0-2（每个棋盘从自身左侧到右侧编号）
        cell_type: 格子类型（英雄/宠物）
    """

    row: int
    col: int
    cell_type: BoardCellType


@dataclass(frozen=True)
class BoardGridConfig:
    """棋盘格子坐标模型参数（比例坐标）。

    知道一个格子尺寸和棋盘锚点即可推导全部 14 格中心坐标，
    参数由 pick --rect 量测排1/排2 格子后计算得出（见 docs/game-rules.md）。

    Attributes:
        anchor_x_ratio: 棋盘逻辑左上角 x 比例（列0排1格左上角）
        anchor_y_ratio: 棋盘逻辑左上角 y 比例
        col_step_ratio: 列步长比例（格宽+列间距）
        row_step_ratio: 排步长比例（格高+排间距）
        cell_width_ratio: 格子宽度比例
        cell_height_ratio: 格子高度比例
    """

    anchor_x_ratio: float
    anchor_y_ratio: float
    col_step_ratio: float
    row_step_ratio: float
    cell_width_ratio: float
    cell_height_ratio: float

    def cell_center(
        self,
        row: int,
        col: int,
        client_size: tuple[int, int],
    ) -> tuple[int, int]:
        """计算指定格子的中心像素坐标。

        中心 = 锚点 + (列×列步长 + 格宽/2, 排×排步长 + 格高/2)，
        按客户区尺寸换算为像素。

        Args:
            row: 排号 0-5
            col: 列号 0-2
            client_size: 客户区物理像素尺寸 (width, height)

        Returns:
            格子中心像素坐标 (x, y)
        """
        cw, ch = client_size
        x = (self.anchor_x_ratio + col * self.col_step_ratio + self.cell_width_ratio / 2) * cw
        y = (self.anchor_y_ratio + row * self.row_step_ratio + self.cell_height_ratio / 2) * ch
        return (int(x), int(y))


# helper 棋盘 14 格定义（列 0/1/2 = C'/B'/A'，从棋盘左侧到右侧）。
# 对应 game-rules.md 锯齿形布局：排1靠外1格、排2-3靠外2格、排4-5满、
# 排6 两侧为宠物位。helper 排1靠外侧 = 列2(A')。
HELPER_CELLS: tuple[BoardCell, ...] = (
    BoardCell(0, 2, BoardCellType.HERO),  # 排1 A'
    BoardCell(1, 1, BoardCellType.HERO),
    BoardCell(1, 2, BoardCellType.HERO),  # 排2 B',A'
    BoardCell(2, 1, BoardCellType.HERO),
    BoardCell(2, 2, BoardCellType.HERO),  # 排3
    BoardCell(3, 0, BoardCellType.HERO),
    BoardCell(3, 1, BoardCellType.HERO),
    BoardCell(3, 2, BoardCellType.HERO),  # 排4
    BoardCell(4, 0, BoardCellType.HERO),
    BoardCell(4, 1, BoardCellType.HERO),
    BoardCell(4, 2, BoardCellType.HERO),  # 排5
    BoardCell(5, 0, BoardCellType.PET),
    BoardCell(5, 1, BoardCellType.HERO),
    BoardCell(5, 2, BoardCellType.PET),  # 排6
)

# initiator 棋盘镜像（列 0/1/2 = A/B/C，从棋盘左侧到右侧）。
# initiator 排1靠外侧 = 列0(A)。与 helper 沿中间分隔线左右镜像。
INITIATOR_CELLS: tuple[BoardCell, ...] = (
    BoardCell(0, 0, BoardCellType.HERO),  # 排1 A
    BoardCell(1, 0, BoardCellType.HERO),
    BoardCell(1, 1, BoardCellType.HERO),  # 排2 A,B
    BoardCell(2, 0, BoardCellType.HERO),
    BoardCell(2, 1, BoardCellType.HERO),  # 排3
    BoardCell(3, 0, BoardCellType.HERO),
    BoardCell(3, 1, BoardCellType.HERO),
    BoardCell(3, 2, BoardCellType.HERO),  # 排4
    BoardCell(4, 0, BoardCellType.HERO),
    BoardCell(4, 1, BoardCellType.HERO),
    BoardCell(4, 2, BoardCellType.HERO),  # 排5
    BoardCell(5, 0, BoardCellType.PET),
    BoardCell(5, 1, BoardCellType.HERO),
    BoardCell(5, 2, BoardCellType.PET),  # 排6
)


def board_cells_for_role(role: CoopRole) -> tuple[BoardCell, ...]:
    """返回角色对应的己方棋盘格子定义。

    Args:
        role: 合作角色

    Returns:
        该角色己方棋盘的 14 格定义
    """
    if role == CoopRole.HELPER:
        return HELPER_CELLS
    return INITIATOR_CELLS


# ---------------------------------------------------------------------------
# 状态机
# ---------------------------------------------------------------------------


class State(str, Enum):
    """合作任务顶层状态。

    对应 docs/architecture.md「任务执行模型」一节的状态集合。
    """

    UNKNOWN = "unknown"
    FIND_COOP = "find_coop"
    ENTER_MATCH = "enter_match"
    SELECT_OPENING_SKILLS = "select_opening_skills"
    BUILD_MAIN_C = "build_main_c"
    SELECT_MAIN_C_SKILLS = "select_main_c_skills"
    HANDLE_RESULT = "handle_result"
    CLAIM_REWARD = "claim_reward"
    CHECK_ROUND_LIMIT = "check_round_limit"
    COMPLETED = "completed"
    # 阻塞性弹窗
    BLOCKING_DIALOG = "blocking_dialog"
    # 窗口异常
    WINDOW_INVALID = "window_invalid"


@dataclass(frozen=True)
class Action:
    """动作意图。

    Attributes:
        kind: 动作类型 click / drag / key / wait，或由 Runner 路由的复合动作信号
        target: 点击或拖动起点，客户区坐标
        end: 拖动终点，仅 drag 使用
        key: 按键名称，仅 key 使用
        duration: 动作持续时间（秒）
        post_delay: 动作发送成功后的稳定等待时间（秒）；等待完成后才进入下一轮截图
        verification: ``immediate`` 表示输入成功即可提交内部进度，
            ``next_frame`` 表示必须由下一帧业务状态验证。
        tag: 稳定的动作语义标识，供验证通过后更新任务内部状态
        reason: 触发该动作的原因，用于调试记录
    """

    kind: str
    target: tuple[int, int] | None = None
    end: tuple[int, int] | None = None
    key: str | None = None
    duration: float = 0.0
    post_delay: float = 0.0
    verification: Literal["immediate", "next_frame"] = "next_frame"
    tag: str = ""
    reason: str = ""


@dataclass(frozen=True)
class ActionResult:
    """动作执行结果。

    Attributes:
        executed: 动作是否实际执行
        verified: 动作后是否验证进入预期状态
        failure_reason: 失败原因，成功时为空
        evidence: 调试证据路径或描述
    """

    executed: bool
    verified: bool
    failure_reason: str = ""
    evidence: str = ""


@dataclass(frozen=True)
class Transition:
    """状态迁移规则。

    Attributes:
        name: 迁移名称
        from_state: 起始状态
        to_state: 预期目标状态
        action: 要执行的动作
        max_retries: 最大重试次数
        timeout: 超时时间（秒）
    """

    name: str
    from_state: State
    to_state: State
    action: Action
    max_retries: int = 3
    timeout: float = 30.0


# ---------------------------------------------------------------------------
# 调试事件
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DebugEvent:
    """调试记录事件。

    Debug Recorder 通过订阅此类事件记录执行过程，
    不参与业务决策。

    Attributes:
        frame_id: 关联的帧标识
        kind: 事件类型，例如 capture / match / action / state_change / error
        message: 事件描述
        data: 附加数据
        timestamp: 事件时间戳（秒）
    """

    frame_id: int
    kind: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
