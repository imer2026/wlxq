"""结算截图保存器单元测试。

验证按 模式/日期 分目录、按回合+时间命名的落盘规则，
不依赖真实游戏窗口。
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from wlxq_bot.debug.settlement import SettlementScreenshotSaver


def _frame() -> np.ndarray:
    return np.zeros((4, 6, 3), dtype=np.uint8)


def _captured_at(hour: int, minute: int, second: int) -> float:
    """构造当天指定时刻的 epoch 时间戳（本地时区）。"""
    local_midnight = time.strptime("2026-09-06", "%Y-%m-%d")
    return time.mktime(local_midnight) + hour * 3600 + minute * 60 + second


class TestSettlementScreenshotSaver:
    def test_save_writes_png_under_mode_date_folder(self, tmp_path: Path) -> None:
        saver = SettlementScreenshotSaver(tmp_path)

        path = saver.save(
            _frame(),
            mode="coop",
            round_number=3,
            captured_at=_captured_at(14, 32, 1),
        )

        assert path == tmp_path / "coop" / "20260906" / "round03_143201.png"
        assert path.is_file()
        # 落盘的是可回读的合法 PNG
        assert cv2.imread(str(path)) is not None

    def test_save_uses_screenshot_day_not_run_start_day(self, tmp_path: Path) -> None:
        """挂机跨天时按截图时刻归档（游戏次数按天算）。"""
        saver = SettlementScreenshotSaver(tmp_path)

        path = saver.save(
            _frame(),
            mode="pet",
            round_number=12,
            captured_at=_captured_at(0, 5, 9) + 24 * 3600,
        )

        assert path == tmp_path / "pet" / "20260907" / "round12_000509.png"

    def test_save_normalizes_unsafe_mode_name(self, tmp_path: Path) -> None:
        saver = SettlementScreenshotSaver(tmp_path)

        path = saver.save(
            _frame(),
            mode="pet defend!",
            round_number=1,
            captured_at=_captured_at(19, 30, 45),
        )

        assert path == tmp_path / "pet_defend" / "20260906" / "round01_193045.png"

    def test_save_rejects_non_ndarray_frame(self, tmp_path: Path) -> None:
        saver = SettlementScreenshotSaver(tmp_path)

        with pytest.raises(TypeError, match="numpy.ndarray"):
            saver.save(
                "not-a-frame",
                mode="coop",
                round_number=1,
                captured_at=_captured_at(10, 0, 0),
            )
