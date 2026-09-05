"""保卫宠物难度行 OCR 识别测试（fake reader，不依赖真实引擎）。"""

from __future__ import annotations

import numpy as np

from wlxq_bot.perception.pet_difficulty import (
    DifficultyRow,
    match_targets,
    read_difficulty_rows,
)


class FakeRowReader:
    """TitleReader 协议的 fake：返回预设行（text, confidence, (cx, cy)）。"""

    def __init__(self, lines: list[tuple[str, float, tuple[int, int]]]) -> None:
        self.lines = lines
        self.images: list[np.ndarray] = []

    def read_with_confidence(self, image: np.ndarray):
        self.images.append(image)
        return self.lines


def _frame(h: int = 1400, w: int = 700) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_read_difficulty_rows_offsets_by_roi() -> None:
    reader = FakeRowReader([("保卫宠物-普通", 0.85, (120, 50))])
    rows = read_difficulty_rows(_frame(), roi=(300, 900, 260, 300), reader=reader)
    assert rows == [DifficultyRow(text="保卫宠物-普通", confidence=0.85, position=(420, 950))]
    # 裁剪图尺寸 = ROI 尺寸
    assert reader.images[0].shape[:2] == (300, 260)


def test_read_difficulty_rows_empty_and_zero_roi() -> None:
    reader = FakeRowReader([("保卫宠物-普通", 0.85, (10, 10))])
    assert read_difficulty_rows(_frame(), None, reader=reader) == []
    assert read_difficulty_rows(_frame(), (10, 10, 0, 100), reader=reader) == []
    assert read_difficulty_rows(_frame(), (10, 10, 100, 0), reader=reader) == []
    assert reader.images == []


def test_read_difficulty_rows_skips_blank_text() -> None:
    reader = FakeRowReader([("  ", 0.9, (10, 10)), ("保卫宠物-中等", 0.7, (20, 20))])
    rows = read_difficulty_rows(_frame(), (0, 0, 100, 100), reader=reader)
    assert [row.text for row in rows] == ["保卫宠物-中等"]


def test_match_targets_substring_and_order() -> None:
    rows = [
        DifficultyRow(text="保卫宠物-困难", confidence=0.8, position=(1, 1)),
        DifficultyRow(text="保卫宠物-普通", confidence=0.9, position=(2, 2)),
        DifficultyRow(text="保卫宠物-中等", confidence=0.7, position=(3, 3)),
    ]
    matched = match_targets(rows, ["普通", "中等"])
    assert [target for target, _ in matched] == ["普通", "中等"]
    assert matched[0][1].position == (2, 2)
    assert matched[1][1].position == (3, 3)


def test_match_targets_skips_missing_and_low_confidence_rows() -> None:
    # match_targets 只做子串匹配，不做置信度过滤（低置信度行由调用方按
    # min_confidence 剔除后再传入，见 CoopPerception 的过滤职责）。
    rows = [
        DifficultyRow(text="保卫宠物-地狱", confidence=0.9, position=(1, 1)),
        DifficultyRow(text="保卫宠物-中等", confidence=0.3, position=(2, 2)),
    ]
    # 「普通」目标缺失 → 不出现在结果；未剔除的低置信度行仍按子串命中
    assert match_targets(rows, ["普通", "中等"]) == [("中等", rows[1])]
    # 调用方剔除低置信度行后 → 无任何命中
    assert match_targets(rows[:1], ["普通", "中等"]) == []
