"""Locator ROI 换算与角色解析测试。

验证比例 ROI → 像素 ROI 换算、角色到棋盘的解析，
以及 current_board 间接引用机制。
"""

from __future__ import annotations

import pytest

from wlxq_bot.config import BoardGridParams, Hotspot, RoiConfig
from wlxq_bot.models import (
    BoardCell,
    BoardCellType,
    BoardGridConfig,
    CoopRole,
    board_cells_for_role,
)
from wlxq_bot.perception.locator import (
    CURRENT_BOARD,
    board_grid_for_role,
    format_cell_label,
    hero_cell_centers,
    hero_cells_for_role,
    hotspot_to_client_point,
    parse_cell_label,
    ratio_to_pixel_roi,
    resolve_board_roi,
    resolve_roi_by_name,
    roi_column_centers,
)


def test_hotspot_to_client_point() -> None:
    hotspot = Hotspot(x_ratio=0.25, y_ratio=0.75)

    assert hotspot_to_client_point(hotspot, (1000, 2000)) == (250, 1500)


class TestRatioToPixelRoi:
    """比例 → 像素换算测试。"""

    def test_full_client(self) -> None:
        roi = RoiConfig(x_ratio=0.0, y_ratio=0.0, width_ratio=1.0, height_ratio=1.0)
        assert ratio_to_pixel_roi(roi, (927, 1727)) == (0, 0, 927, 1727)

    def test_half_offset(self) -> None:
        # 右下四分之一区域：0.5*927=463, 0.5*1727=863
        roi = RoiConfig(x_ratio=0.5, y_ratio=0.5, width_ratio=0.5, height_ratio=0.5)
        assert ratio_to_pixel_roi(roi, (927, 1727)) == (463, 863, 463, 863)

    def test_zero_roi_placeholder(self) -> None:
        # 占位值 0.0 换算后为空矩形（标定前不应被使用）
        roi = RoiConfig(x_ratio=0.0, y_ratio=0.0, width_ratio=0.0, height_ratio=0.0)
        assert ratio_to_pixel_roi(roi, (927, 1727)) == (0, 0, 0, 0)

    def test_truncates_to_int(self) -> None:
        # 0.3*927=278.1 → 278，验证 int 截断
        roi = RoiConfig(x_ratio=0.3, y_ratio=0.0, width_ratio=0.3, height_ratio=0.1)
        x, y, w, h = ratio_to_pixel_roi(roi, (927, 1727))
        assert x == 278
        assert w == 278


class TestRoiColumnCenters:
    def test_three_card_centers(self) -> None:
        roi = RoiConfig(
            x_ratio=0.1,
            y_ratio=0.2,
            width_ratio=0.6,
            height_ratio=0.4,
        )

        assert roi_column_centers(roi, (1000, 2000), 3) == [
            (200, 800),
            (400, 800),
            (600, 800),
        ]

    def test_unmarked_roi_rejected(self) -> None:
        roi = RoiConfig(x_ratio=0.0, y_ratio=0.0, width_ratio=0.0, height_ratio=0.0)

        with pytest.raises(ValueError, match="ROI"):
            roi_column_centers(roi, (1000, 2000), 3)

    def test_roi_outside_client_rejected(self) -> None:
        roi = RoiConfig(x_ratio=0.8, y_ratio=0.2, width_ratio=0.3, height_ratio=0.4)

        with pytest.raises(ValueError, match="客户区"):
            roi_column_centers(roi, (1000, 2000), 3)


class TestResolveBoardRoi:
    """角色 → 己方棋盘 ROI 解析测试。"""

    @staticmethod
    def _rois() -> dict[str, RoiConfig]:
        return {
            "bottom_left_board": RoiConfig(
                x_ratio=0.0,
                y_ratio=0.5,
                width_ratio=0.5,
                height_ratio=0.5,
            ),
            "bottom_right_board": RoiConfig(
                x_ratio=0.5,
                y_ratio=0.5,
                width_ratio=0.5,
                height_ratio=0.5,
            ),
        }

    def test_helper_resolves_right_board(self) -> None:
        roi = resolve_board_roi(CoopRole.HELPER, self._rois(), (927, 1727))
        assert roi == (463, 863, 463, 863)

    def test_initiator_resolves_left_board(self) -> None:
        roi = resolve_board_roi(CoopRole.INITIATOR, self._rois(), (927, 1727))
        assert roi == (0, 863, 463, 863)

    def test_missing_board_raises(self) -> None:
        with pytest.raises(KeyError, match="bottom_right_board"):
            resolve_board_roi(CoopRole.HELPER, {}, (927, 1727))

    def test_missing_left_board_raises(self) -> None:
        with pytest.raises(KeyError, match="bottom_left_board"):
            resolve_board_roi(CoopRole.INITIATOR, {}, (927, 1727))


class TestResolveRoiByName:
    """current_board 间接引用与具名 ROI 解析测试。"""

    @staticmethod
    def _rois() -> dict[str, RoiConfig]:
        return {
            "bottom_left_board": RoiConfig(
                x_ratio=0.0,
                y_ratio=0.5,
                width_ratio=0.5,
                height_ratio=0.5,
            ),
            "bottom_right_board": RoiConfig(
                x_ratio=0.5,
                y_ratio=0.5,
                width_ratio=0.5,
                height_ratio=0.5,
            ),
        }

    def test_current_board_helper(self) -> None:
        # helper + current_board → bottom_right_board
        roi = resolve_roi_by_name(
            CURRENT_BOARD,
            CoopRole.HELPER,
            self._rois(),
            (927, 1727),
        )
        assert roi == (463, 863, 463, 863)

    def test_current_board_initiator(self) -> None:
        # initiator + current_board → bottom_left_board
        roi = resolve_roi_by_name(
            CURRENT_BOARD,
            CoopRole.INITIATOR,
            self._rois(),
            (927, 1727),
        )
        assert roi == (0, 863, 463, 863)

    def test_named_roi_direct(self) -> None:
        # 直接用具名 ROI，不经过角色解析
        roi = resolve_roi_by_name(
            "bottom_right_board",
            CoopRole.HELPER,
            self._rois(),
            (927, 1727),
        )
        assert roi == (463, 863, 463, 863)

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            resolve_roi_by_name(
                "nonexistent",
                CoopRole.HELPER,
                self._rois(),
                (927, 1727),
            )


class TestBoardGridConfig:
    """BoardGridConfig 格子坐标计算测试。"""

    def _grid(self) -> BoardGridConfig:
        # helper 棋盘量测参数（923×1723 客户区）
        return BoardGridConfig(
            anchor_x_ratio=0.6229,
            anchor_y_ratio=0.5148,
            col_step_ratio=0.0791,
            row_step_ratio=0.0432,
            cell_width_ratio=0.0737,
            cell_height_ratio=0.0360,
        )

    def test_cell_center_row1_a(self) -> None:
        # 排1 A'格 (row=0, col=2)，量测中心约 (755, 918)
        grid = self._grid()
        cx, cy = grid.cell_center(0, 2, (923, 1723))
        assert abs(cx - 755) <= 2
        assert abs(cy - 918) <= 2

    def test_cell_center_row2_a(self) -> None:
        # 排2 A'格 (row=1, col=2)，量测中心约 (756, 993)
        grid = self._grid()
        cx, cy = grid.cell_center(1, 2, (923, 1723))
        assert abs(cx - 756) <= 2
        assert abs(cy - 993) <= 2

    def test_cell_center_row2_b(self) -> None:
        # 排2 B'格 (row=1, col=1)，量测中心约 (682, 993)
        grid = self._grid()
        cx, cy = grid.cell_center(1, 1, (923, 1723))
        assert abs(cx - 682) <= 2
        assert abs(cy - 993) <= 2


class TestBoardCells:
    """棋盘格子定义测试。"""

    def test_helper_has_14_cells(self) -> None:
        cells = board_cells_for_role(CoopRole.HELPER)
        assert len(cells) == 14

    def test_helper_has_12_hero_2_pet(self) -> None:
        cells = board_cells_for_role(CoopRole.HELPER)
        hero = [c for c in cells if c.cell_type == BoardCellType.HERO]
        pet = [c for c in cells if c.cell_type == BoardCellType.PET]
        assert len(hero) == 12
        assert len(pet) == 2

    def test_pet_cells_at_row6_sides(self) -> None:
        # 宠物位在排6两侧：helper 排6 col0(C') 和 col2(A')
        cells = board_cells_for_role(CoopRole.HELPER)
        pet = [c for c in cells if c.cell_type == BoardCellType.PET]
        assert all(c.row == 5 for c in pet)  # 排6 = row5
        cols = sorted(c.col for c in pet)
        assert cols == [0, 2]  # 两侧

    def test_initiator_mirror(self) -> None:
        # initiator 排1靠外侧 = col0（helper 是 col2），镜像
        helper_r1 = [c for c in board_cells_for_role(CoopRole.HELPER) if c.row == 0]
        initiator_r1 = [c for c in board_cells_for_role(CoopRole.INITIATOR) if c.row == 0]
        assert helper_r1[0].col == 2
        assert initiator_r1[0].col == 0


class TestHeroCellCenters:
    """英雄位中心坐标与量测验证测试。"""

    def _params(self) -> dict[str, BoardGridParams]:
        return {
            "helper": BoardGridParams(
                anchor_x_ratio=0.6229,
                anchor_y_ratio=0.5148,
                col_step_ratio=0.0791,
                row_step_ratio=0.0432,
                cell_width_ratio=0.0737,
                cell_height_ratio=0.0360,
            ),
            "initiator": BoardGridParams(
                anchor_x_ratio=0.1430,
                anchor_y_ratio=0.5141,
                col_step_ratio=0.0791,
                row_step_ratio=0.0432,
                cell_width_ratio=0.0737,
                cell_height_ratio=0.0360,
            ),
        }

    def test_hero_cells_count(self) -> None:
        cells = hero_cells_for_role(CoopRole.HELPER)
        assert len(cells) == 12

    def test_hero_cell_centers_count(self) -> None:
        grid = board_grid_for_role(CoopRole.HELPER, self._params())
        centers = hero_cell_centers(grid, CoopRole.HELPER, (923, 1723))
        assert len(centers) == 12

    def test_centers_match_measured(self) -> None:
        # 验证模型算出的英雄位中心与 pick 量测值吻合（误差 <= 2px）
        grid = board_grid_for_role(CoopRole.HELPER, self._params())
        centers = hero_cell_centers(grid, CoopRole.HELPER, (923, 1723))
        by_rc = {(c.row, c.col): pos for c, pos in centers}

        # 排1 A'(0,2) 量测中心 (755, 918)
        assert abs(by_rc[(0, 2)][0] - 755) <= 2
        assert abs(by_rc[(0, 2)][1] - 918) <= 2
        # 排2 A'(1,2) 量测中心 (756, 993)
        assert abs(by_rc[(1, 2)][0] - 756) <= 2
        assert abs(by_rc[(1, 2)][1] - 993) <= 2
        # 排2 B'(1,1) 量测中心 (682, 993)
        assert abs(by_rc[(1, 1)][0] - 682) <= 2
        assert abs(by_rc[(1, 1)][1] - 993) <= 2
        # 排6 B'(5,1) 量测中心 (683, 1288)，校准排步长
        assert abs(by_rc[(5, 1)][0] - 683) <= 2
        assert abs(by_rc[(5, 1)][1] - 1288) <= 2

    def test_initiator_centers_match_measured(self) -> None:
        # initiator 排6 B(5,1) 量测中心 (240, 1288)
        grid = board_grid_for_role(CoopRole.INITIATOR, self._params())
        centers = hero_cell_centers(grid, CoopRole.INITIATOR, (923, 1723))
        by_rc = {(c.row, c.col): pos for c, pos in centers}
        assert abs(by_rc[(5, 1)][0] - 240) <= 2
        assert abs(by_rc[(5, 1)][1] - 1288) <= 2

    def test_missing_role_raises(self) -> None:
        with pytest.raises(KeyError, match="initiator"):
            board_grid_for_role(CoopRole.INITIATOR, {})


class TestParseCellLabel:
    """格子标识解析测试。"""

    def test_helper_a_maps_col2(self) -> None:
        # helper A=A' 最外右 = col2
        assert parse_cell_label("5A", CoopRole.HELPER) == (4, 2)

    def test_helper_c_maps_col0(self) -> None:
        # helper C=C' 靠中间左 = col0
        assert parse_cell_label("4C", CoopRole.HELPER) == (3, 0)

    def test_initiator_a_maps_col0(self) -> None:
        # initiator A=A 最外左 = col0
        assert parse_cell_label("5A", CoopRole.INITIATOR) == (4, 0)

    def test_initiator_c_maps_col2(self) -> None:
        # initiator C=C 靠中间右 = col2
        assert parse_cell_label("4C", CoopRole.INITIATOR) == (3, 2)

    def test_row1_maps_to_row0(self) -> None:
        assert parse_cell_label("1B", CoopRole.HELPER) == (0, 1)

    def test_row6_maps_to_row5(self) -> None:
        assert parse_cell_label("6B", CoopRole.HELPER) == (5, 1)

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="格式错误"):
            parse_cell_label("X", CoopRole.HELPER)

    def test_invalid_row_raises(self) -> None:
        with pytest.raises(ValueError, match="排号"):
            parse_cell_label("7A", CoopRole.HELPER)

    def test_invalid_col_raises(self) -> None:
        with pytest.raises(ValueError, match="列"):
            parse_cell_label("5D", CoopRole.HELPER)

    def test_format_label_respects_helper_mirror(self) -> None:
        cell = BoardCell(row=3, col=1, cell_type=BoardCellType.HERO)
        assert format_cell_label(cell, CoopRole.HELPER) == "4B"

    def test_format_and_parse_are_symmetric(self) -> None:
        cell = BoardCell(row=4, col=2, cell_type=BoardCellType.HERO)
        label = format_cell_label(cell, CoopRole.HELPER)
        assert parse_cell_label(label, CoopRole.HELPER) == (cell.row, cell.col)
