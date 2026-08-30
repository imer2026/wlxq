"""Locator：坐标换算、ROI 计算和动作点定位。

统一管理所有定位策略，避免坐标散落在任务代码中。
支持的策略见 docs/architecture.md「定位策略」一节：
- fixed_ratio：按客户区宽高比例计算候选点
- anchor：从客户区边缘或角落按物理像素偏移计算
- template：在指定 ROI 中匹配模板
- template_set：匹配多个模板变体
- manual_calibration：读取用户标定的候选点或 ROI
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wlxq_bot.config import BoardGridParams, Hotspot, RoiConfig
from wlxq_bot.models import (
    BoardCell,
    BoardCellType,
    BoardGridConfig,
    CoopRole,
    MatchResult,
    WindowContext,
    board_cells_for_role,
    board_roi_name,
)

# 英雄模板配置中引用己方棋盘的间接名。
# heroes 段 ``roi: current_board`` 会在运行时按角色解析为实际棋盘 ROI
# （initiator → bottom_left_board, helper → bottom_right_board），
# 避免在英雄配置里硬编码某一个棋盘。
CURRENT_BOARD = "current_board"


def ratio_to_pixel_roi(
    roi: RoiConfig,
    client_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """把比例 ROI 换算为客户区物理像素 ROI。

    配置层用比例坐标定义 ROI（与分辨率无关），运行时按实际客户区
    尺寸换算为像素，供 vision.match_template_set 的 roi 参数使用。

    Args:
        roi: 比例坐标 ROI 配置
        client_size: 客户区物理像素尺寸 (width, height)

    Returns:
        (x, y, width, height) 像素 tuple
    """
    cw, ch = client_size
    x = int(roi.x_ratio * cw)
    y = int(roi.y_ratio * ch)
    w = int(roi.width_ratio * cw)
    h = int(roi.height_ratio * ch)
    return (x, y, w, h)


def roi_column_centers(
    roi: RoiConfig,
    client_size: tuple[int, int],
    columns: int,
) -> list[tuple[int, int]]:
    """把 ROI 等分为若干列，返回每列的中心点击点。

    技能候选区固定为横向三张卡片时，分别取 ROI 宽度的 1/6、3/6、5/6，
    纵向取高度中点，点击会落在三列卡片内部。

    Raises:
        ValueError: 列数、客户区或 ROI 无效。
    """
    if columns <= 0:
        raise ValueError("columns 必须大于 0")
    client_width, client_height = client_size
    if client_width <= 0 or client_height <= 0:
        raise ValueError("客户区宽高必须大于 0")
    if roi.x_ratio + roi.width_ratio > 1.0 or roi.y_ratio + roi.height_ratio > 1.0:
        raise ValueError("ROI 必须完整位于客户区内")
    x, y, width, height = ratio_to_pixel_roi(roi, client_size)
    if width <= 0 or height <= 0:
        raise ValueError("ROI 宽高必须大于 0")
    return [
        (x + int((column + 0.5) * width / columns), y + height // 2) for column in range(columns)
    ]


def hotspot_to_client_point(
    hotspot: Hotspot,
    client_size: tuple[int, int],
) -> tuple[int, int]:
    """把已校验的命名位置换算为客户区物理像素坐标。"""
    client_width, client_height = client_size
    return (
        int(hotspot.x_ratio * client_width),
        int(hotspot.y_ratio * client_height),
    )


def resolve_board_roi(
    role: CoopRole,
    rois: dict[str, RoiConfig],
    client_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """按角色解析己方棋盘的像素 ROI。

    initiator → bottom_left_board，helper → bottom_right_board。
    对应 roi 必须存在于 rois 配置中，否则抛 KeyError。

    Args:
        role: 当前合作角色
        rois: tasks.yaml 的 rois 配置
        client_size: 客户区物理像素尺寸

    Returns:
        己方棋盘像素 ROI (x, y, width, height)

    Raises:
        KeyError: 角色对应的棋盘 ROI 未在配置中定义
    """
    name = board_roi_name(role)
    if name not in rois:
        raise KeyError(f"角色 {role.value} 对应的棋盘 ROI '{name}' 未在 rois 配置中定义")
    return ratio_to_pixel_roi(rois[name], client_size)


def resolve_roi_by_name(
    name: str,
    role: CoopRole,
    rois: dict[str, RoiConfig],
    client_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """按名称解析 ROI，支持 current_board 间接引用。

    英雄模板配置中 roi 可写 ``current_board``，运行时按角色解析为
    实际己方棋盘 ROI；其他名称直接查 rois 配置。这样英雄识别只会在
    指定棋盘 ROI 内匹配，不对整个游戏界面做识别。

    Args:
        name: ROI 名称，可为 ``current_board`` 或 rois 中的键
        role: 当前合作角色（仅 name=current_board 时使用）
        rois: tasks.yaml 的 rois 配置
        client_size: 客户区物理像素尺寸

    Returns:
        像素 ROI (x, y, width, height)

    Raises:
        KeyError: 指定名称的 ROI 未在配置中定义
    """
    if name == CURRENT_BOARD:
        return resolve_board_roi(role, rois, client_size)
    if name not in rois:
        raise KeyError(f"ROI '{name}' 未在 rois 配置中定义")
    return ratio_to_pixel_roi(rois[name], client_size)


def board_grid_for_role(
    role: CoopRole,
    board_params: dict[str, BoardGridParams],
) -> BoardGridConfig:
    """按角色返回己方棋盘坐标模型。

    从 tasks.yaml 的 board 配置加载对应角色的格子坐标参数，
    转为运行时 BoardGridConfig（含 cell_center 方法）。

    Args:
        role: 合作角色
        board_params: tasks.yaml 的 board 配置（key 为 helper/initiator）

    Returns:
        该角色己方棋盘的 BoardGridConfig

    Raises:
        KeyError: 角色对应的棋盘参数未配置
    """
    key = "helper" if role == CoopRole.HELPER else "initiator"
    if key not in board_params:
        raise KeyError(f"角色 {role.value} 对应的棋盘参数 '{key}' 未在 board 配置中定义")
    return BoardGridConfig(**board_params[key].model_dump())


def hero_cells_for_role(role: CoopRole) -> list[BoardCell]:
    """返回角色己方棋盘的英雄位格子（排除宠物位 6A/6C）。

    英雄识别只对这些格子做模板匹配，宠物位跳过。

    Args:
        role: 合作角色

    Returns:
        英雄位格子列表（12 个）
    """
    return [c for c in board_cells_for_role(role) if c.cell_type == BoardCellType.HERO]


def hero_cell_centers(
    grid: BoardGridConfig,
    role: CoopRole,
    client_size: tuple[int, int],
) -> list[tuple[BoardCell, tuple[int, int]]]:
    """返回角色己方棋盘所有英雄位的格子定义和中心像素坐标。

    用于遍历己方棋盘英雄位，在每个格子中心附近做模板匹配或点击。

    Args:
        grid: 棋盘坐标模型（由 board_grid_for_role 得到）
        role: 合作角色
        client_size: 客户区物理像素尺寸 (width, height)

    Returns:
        (BoardCell, (x, y)) 列表，仅含英雄位，中心为客户区像素坐标
    """
    return [
        (cell, grid.cell_center(cell.row, cell.col, client_size))
        for cell in hero_cells_for_role(role)
    ]


def hero_cell_rois(
    grid: BoardGridConfig,
    role: CoopRole,
    client_size: tuple[int, int],
) -> list[tuple[BoardCell, tuple[int, int, int, int]]]:
    """返回 12 个英雄格的客户区像素裁剪范围。

    裁剪尺寸与数据制作使用同一套格子参数和四舍五入规则，保证训练图片与
    正式运行输入来自相同的 ROI 定义。
    """
    client_width, client_height = client_size
    cell_width = max(1, round(grid.cell_width_ratio * client_width))
    cell_height = max(1, round(grid.cell_height_ratio * client_height))
    result: list[tuple[BoardCell, tuple[int, int, int, int]]] = []
    for cell, (center_x, center_y) in hero_cell_centers(grid, role, client_size):
        left = center_x - cell_width // 2
        top = center_y - cell_height // 2
        if (
            left < 0
            or top < 0
            or left + cell_width > client_width
            or top + cell_height > client_height
        ):
            raise ValueError(
                f"格子裁剪范围越界: row={cell.row} col={cell.col} "
                f"roi=({left},{top},{cell_width},{cell_height}) "
                f"client={client_width}x{client_height}"
            )
        result.append((cell, (left, top, cell_width, cell_height)))
    return result


# 格子标识列字母 → 列号。helper 列 0/1/2 = C'/B'/A'，initiator 列 0/1/2 = A/B/C。
COL_MAP_HELPER = {"A": 2, "B": 1, "C": 0}
COL_MAP_INITIATOR = {"A": 0, "B": 1, "C": 2}


def parse_cell_label(label: str, role: CoopRole) -> tuple[int, int]:
    """解析格子标识 '5B' → (row, col)。

    排号 1-6 → row 0-5；列字母 A/B/C 按角色映射到列号。
    helper：A=A'(col2 最外右) B=B'(col1) C=C'(col0 靠中间左)
    initiator：A=A(col0 最外左) B=B(col1) C=C(col2 靠中间右)

    Args:
        label: 格子标识，如 "5B"、"3A"、"6C"
        role: 合作角色，决定列字母到列号的映射

    Returns:
        (row, col) 元组，row 为 0 基排号，col 为 0-2 列号

    Raises:
        ValueError: 标识格式不符、排号越界或列字母非法
    """
    label = label.strip()
    if len(label) < 2 or not label[0].isdigit():
        raise ValueError(f"格子标识格式错误: {label!r}，应为 <排号><列字母> 如 5B")
    row = int(label[0]) - 1
    if not 0 <= row <= 5:
        raise ValueError(f"排号必须在 1-6: {label!r}")
    col_char = label[1:].upper()
    col_map = COL_MAP_HELPER if role == CoopRole.HELPER else COL_MAP_INITIATOR
    if col_char not in col_map:
        raise ValueError(f"列必须是 A/B/C: {label!r}")
    return row, col_map[col_char]


def format_cell_label(cell: BoardCell, role: CoopRole) -> str:
    """把格子模型格式化为简短标签，例如 ``4B``。

    列字母按玩家自身棋盘从外到内使用 A/B/C；helper 的底层列号
    与 initiator 镜像，因此不能直接把 ``cell.col`` 转为字母。
    """
    if not 0 <= cell.row <= 5 or not 0 <= cell.col <= 2:
        raise ValueError(f"格子行列越界: row={cell.row} col={cell.col}")
    col_map = COL_MAP_HELPER if role == CoopRole.HELPER else COL_MAP_INITIATOR
    col_char = next((label for label, col in col_map.items() if col == cell.col), None)
    if col_char is None:
        raise ValueError(f"无法格式化格子列: role={role.value} col={cell.col}")
    return f"{cell.row + 1}{col_char}"


@dataclass(frozen=True)
class ROI:
    """识别区域。

    Attributes:
        x: 区域左上角 x 坐标（客户区坐标系）
        y: 区域左上角 y 坐标
        width: 区域宽度
        height: 区域高度
        relative_to: 参照系 client / anchor / candidate
    """

    x: int
    y: int
    width: int
    height: int
    relative_to: str = "client"

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


class Locator:
    """定位器。

    TODO:
        - 实现 fixed_ratio / anchor 策略
        - 实现 template / template_set 策略与 Vision 协作
        - 实现 ROI 相对坐标换算
        - 实现动作点来源选择（candidate / match_center / match_offset）
    """

    def locate(
        self,
        strategy: str,
        ctx: WindowContext,
        config: dict[str, Any],
        vision: Any | None = None,
        frame: Any | None = None,
    ) -> tuple[tuple[int, int], MatchResult | None]:
        """根据策略计算候选点和可选的匹配验证结果。

        Args:
            strategy: 定位策略名称
            ctx: 当前窗口上下文
            config: 定位器配置
            vision: Vision 实例（template 策略需要）
            frame: 截图帧（template 策略需要）

        Returns:
            (候选点坐标, 匹配结果或 None)
        """
        raise NotImplementedError
