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
