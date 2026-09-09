"""招募页模式页签行 OCR 识别测试（fake reader，不依赖真实引擎）。"""

from __future__ import annotations

import numpy as np

from wlxq_bot.perception.mode_tab import ModeTabRow, read_mode_tab_rows


class FakeRowReader:
    """TitleReader 协议的 fake：返回预设行（text, confidence, (cx, cy)）。"""

    def __init__(self, lines: list[tuple[str, float, tuple[int, int]]]) -> None:
        self.lines = lines
        self.images: list[np.ndarray] = []

    def read_with_confidence(self, image: np.ndarray):
        self.images.append(image)
        return self.lines


def _frame(h: int = 1727, w: int = 927) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_read_mode_tab_rows_offsets_by_roi() -> None:
    reader = FakeRowReader([("合作模式", 0.71, (124, 65))])
    rows = read_mode_tab_rows(_frame(), roi=(190, 1470, 665, 115), reader=reader)
    assert rows == [ModeTabRow(text="合作模式", confidence=0.71, position=(314, 1535))]
    # 裁剪图尺寸 = ROI 尺寸
    assert reader.images[0].shape[:2] == (115, 665)


def test_read_mode_tab_rows_empty_and_zero_roi() -> None:
    reader = FakeRowReader([("保卫宠物", 0.75, (10, 10))])
    assert read_mode_tab_rows(_frame(), None, reader=reader) == []
    assert read_mode_tab_rows(_frame(), (10, 10, 0, 100), reader=reader) == []
    assert read_mode_tab_rows(_frame(), (10, 10, 100, 0), reader=reader) == []
    assert reader.images == []


def test_read_mode_tab_rows_skips_blank_text() -> None:
    reader = FakeRowReader([("  ", 0.9, (10, 10)), ("共战灭尊", 0.77, (20, 20))])
    rows = read_mode_tab_rows(_frame(), (0, 0, 100, 100), reader=reader)
    assert [row.text for row in rows] == ["共战灭尊"]
