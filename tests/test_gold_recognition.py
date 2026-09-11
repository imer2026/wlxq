"""金币识别测试：解析器、红色判定、GoldReader 与任务层门控判定。

真实截图集成用例在文件缺失时自动跳过，不依赖真实账号截图入库。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from wlxq_bot.perception.ocr import GoldReader, parse_gold_value, red_ratio
from wlxq_bot.tasks.coop import CoopTask

# 金币感知三区域（927×1727 客户区像素，与 configs/tasks.yaml 的标定一致）
GOLD_ROIS = {
    "current": (268, 1456, 96, 46),
    "skill": (243, 1685, 128, 36),
    "summon": (543, 1685, 118, 36),
}


class TestParseGoldValue:
    def test_plain_number(self) -> None:
        assert parse_gold_value("764") == 764
        assert parse_gold_value("80") == 80

    def test_k_suffix(self) -> None:
        assert parse_gold_value("1.3K") == 1300
        assert parse_gold_value("10.3k") == 10300
        assert parse_gold_value("3.5K") == 3500

    def test_spaces_stripped(self) -> None:
        assert parse_gold_value("1.3 K") == 1300

    def test_invalid_returns_none(self) -> None:
        assert parse_gold_value("") is None
        assert parse_gold_value("金币") is None
        assert parse_gold_value("3.3.3K") is None


class TestRedRatio:
    def _image(self, color: tuple[int, int, int]) -> np.ndarray:
        """深棕底 + 指定颜色文字的合成费用条（粗体大字，接近真实费用数字）。"""
        img = np.full((26, 125, 3), (90, 60, 40), dtype=np.uint8)
        cv2.putText(img, "600", (25, 22), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 5)
        return img

    def test_red_text_detected(self) -> None:
        assert red_ratio(self._image((0, 0, 255))) >= 0.15

    def test_white_text_not_red(self) -> None:
        assert red_ratio(self._image((255, 255, 255))) < 0.15


class TestGoldReader:
    def test_reads_parsed_value_with_injected_engine(self) -> None:
        """注入假引擎：任一预处理变体读到可解析数字即返回。"""

        class FakeEngine:
            def __call__(self, image: np.ndarray):
                box = [[[0, 0], [99, 0], [99, 45], [0, 45]]]
                return ([[box, "441", 0.99]], None)

        reader = GoldReader(engine=FakeEngine())
        frame = np.zeros((1727, 927, 3), dtype=np.uint8)
        info = reader.read(frame, GOLD_ROIS["current"], GOLD_ROIS["skill"], GOLD_ROIS["summon"])
        assert info.current == 441


class TestReadAffordability:
    """只认色模式：红/白判定不触碰 OCR 引擎，数值字段全为 None。"""

    def _image(self, color: tuple[int, int, int]) -> np.ndarray:
        img = np.full((26, 125, 3), (90, 60, 40), dtype=np.uint8)
        cv2.putText(img, "600", (25, 22), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 5)
        return img

    def test_red_cost_unaffordable_without_engine(self) -> None:
        # 不注入引擎：只认色路径一旦触碰 OCR 即失败，证明零 OCR 依赖
        reader = GoldReader()
        info = reader.read_affordability(self._image((0, 0, 255)), (0, 0, 125, 26), (0, 0, 125, 26))
        assert info.skill_red is True
        assert info.current is None
        assert info.skill_cost is None
        assert info.summon_cost is None

    def test_white_cost_affordable(self) -> None:
        reader = GoldReader()
        info = reader.read_affordability(
            self._image((255, 255, 255)), (0, 0, 125, 26), (0, 0, 125, 26)
        )
        assert info.skill_red is False
        assert info.summon_red is False
        assert info.current is None


class TestCanAfford:
    def test_red_means_unaffordable(self) -> None:
        gold = {"current": 9999, "skill_cost": 100, "skill_red": True}
        assert CoopTask._can_afford(gold, "skill") is False

    def test_value_comparison(self) -> None:
        assert CoopTask._can_afford({"current": 682, "skill_cost": 1500}, "skill") is False
        assert CoopTask._can_afford({"current": 682, "skill_cost": 455}, "skill") is True

    def test_missing_values_affordable(self) -> None:
        assert CoopTask._can_afford({"current": None, "skill_cost": None}, "skill") is True
        assert CoopTask._can_afford({}, "skill") is True


# 真实截图回归用例：文件在本地存在时验证识别结果，缺失时跳过
# （截图不入库；期望值来自 2026-09-05 实机四状态样本，人工核对）
REAL_CASES = {
    "screenshot_20260905_015747.png": (1300, 100, 80, False, False),
    "screenshot_20260905_020510.png": (441, 600, 215, True, False),
    "screenshot_20260905_020607.png": (682, 1500, 455, True, False),
    "screenshot_20260905_020635.png": (377, 1800, 515, True, True),
}


@pytest.mark.parametrize("filename,expected", sorted(REAL_CASES.items()))
def test_real_screenshots(filename: str, expected: tuple) -> None:
    path = Path("screenshots/raw") / filename
    if not path.is_file():
        pytest.skip(f"真实截图不存在: {filename}")
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    info = GoldReader().read(
        img, GOLD_ROIS["current"], GOLD_ROIS["skill"], GOLD_ROIS["summon"]
    )
    exp_gold, exp_skill, exp_summon, exp_skill_red, exp_summon_red = expected
    assert info.current == exp_gold, filename
    assert info.skill_cost == exp_skill, filename
    assert info.summon_cost == exp_summon, filename
    assert info.skill_red == exp_skill_red, filename
    assert info.summon_red == exp_summon_red, filename


# 只认色实机回归（2026-09-10 宠物模式实机截图）：红色费用 4 张、白色费用
# 2 张，含边界样本（金币 2.4K vs 费用 2.5K 判红、2.6K vs 2.5K 判白）。
# 文件不入库，缺失时跳过
COLOR_ONLY_CASES = {
    "screenshot_20260910_015954.png": (False, False),
    "screenshot_20260910_020000.png": (True, False),
    "screenshot_20260910_020005.png": (True, False),
    "screenshot_20260910_020009.png": (True, False),
    "screenshot_20260910_020014.png": (True, False),
    "screenshot_20260910_020020.png": (False, False),
    # 短文本红字回归（2026-09-11 实机）：费用涨到 "3K" 后红字像素变少，
    # 旧阈值 0.15 把红色误判为白色（实测占比 0.136）
    "Snipaste_2026-09-11_21-43-16.png": (True, False),
}

# 费用区域客户区比例，与 configs/tasks.yaml 的 skill/summon_cost_area 标定一致
_SKILL_COST_RATIOS = (0.2621, 0.9757, 0.1381, 0.0208)
_SUMMON_COST_RATIOS = (0.5858, 0.9757, 0.1273, 0.0208)


def _ratio_roi(img: np.ndarray, ratios: tuple[float, float, float, float]) -> tuple:
    h, w = img.shape[:2]
    xr, yr, wr, hr = ratios
    return (int(xr * w), int(yr * h), int(wr * w), int(hr * h))


@pytest.mark.parametrize("filename,expected", sorted(COLOR_ONLY_CASES.items()))
def test_color_only_real_screenshots(filename: str, expected: tuple) -> None:
    path = Path("screenshots/raw") / filename
    if not path.is_file():
        pytest.skip(f"真实截图不存在: {filename}")
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    info = GoldReader().read_affordability(
        img,
        _ratio_roi(img, _SKILL_COST_RATIOS),
        _ratio_roi(img, _SUMMON_COST_RATIOS),
    )
    exp_skill_red, exp_summon_red = expected
    assert info.skill_red == exp_skill_red, (
        f"{filename}: skill_red_ratio={info.skill_red_ratio:.3f}"
    )
    assert info.summon_red == exp_summon_red, (
        f"{filename}: summon_red_ratio={info.summon_red_ratio:.3f}"
    )
