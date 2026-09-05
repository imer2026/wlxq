"""保卫宠物难度行 OCR 识别。

读取难度弹窗列表区 ROI 内的行文字（「保卫宠物-普通」等），返回文字、
置信度与文字框中心（客户区像素）。只负责识别与文字级子串匹配；ROI 解析
归调用方（CoopPerception 的 _resolve_named_roi），坐标换算归 Locator 体系。
OCR 必须限域：ROI 缺失或零面积一律返回空，禁止全图 OCR 进入主循环
（spec 2026-09-06 §4.4）。离线验证：四行难度全部读对、置信度 0.76~0.85
（素材 buttons/bao_wei_chong_wu/nan_du.png，2026-09-05）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wlxq_bot.perception.ocr import TitleReader
from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

_default_reader: TitleReader | None = None


@dataclass(frozen=True)
class DifficultyRow:
    """难度弹窗内一行识别结果；position 为文字框中心（客户区像素）。"""

    text: str
    confidence: float
    position: tuple[int, int]


def _get_default_reader() -> TitleReader:
    global _default_reader
    if _default_reader is None:
        _default_reader = TitleReader()
    return _default_reader


def read_difficulty_rows(
    frame: Any,
    roi: tuple[int, int, int, int] | None,
    reader: Any | None = None,
) -> list[DifficultyRow]:
    """在 ROI 内做一次 OCR，返回行结果；roi 为 (x, y, w, h) 客户区像素。

    reader 需提供 ``read_with_confidence(image)``（TitleReader 兼容协议），
    坐标相对输入图左上角；测试注入 fake，不依赖真实引擎。
    """
    if roi is None:
        return []
    x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    if w <= 0 or h <= 0:
        return []
    crop = frame[y : y + h, x : x + w]
    lines = (reader or _get_default_reader()).read_with_confidence(crop)
    rows = [
        DifficultyRow(text=str(text).strip(), confidence=float(conf), position=(cx + x, cy + y))
        for text, conf, (cx, cy) in lines
        if str(text).strip()
    ]
    logger.debug("难度行识别 ROI=%s 行=%s", (x, y, w, h), [row.text for row in rows])
    return rows


def match_targets(
    rows: list[DifficultyRow], targets: list[str]
) -> list[tuple[str, DifficultyRow]]:
    """按子串匹配目标难度；targets 顺序即优先顺序，每个目标取置信度最高行。

    目标是行文字的子串（「普通」⊂「保卫宠物-普通」）；普通/中等互不为子串，
    不同目标不会命中同一行。低于调用方置信度过滤的行应在传入前剔除
    （CoopPerception 按 pet_difficulty_recognition.min_confidence 过滤）。
    """
    matched: list[tuple[str, DifficultyRow]] = []
    for target in targets:
        hits = [row for row in rows if target in row.text]
        if hits:
            matched.append((target, max(hits, key=lambda row: row.confidence)))
    return matched
