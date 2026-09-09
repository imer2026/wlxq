"""招募页模式页签行 OCR 识别。

在模式页签行 ROI 内做一次 OCR，返回页签文字（合作模式/保卫宠物/共战灭尊）、
置信度与文字框中心（客户区像素）。三个页签共用一个识别区域、一套识别逻辑，
各模式按模式档案的 mode_tab_text 子串匹配自己的页签；只负责识别与文字级
子串匹配，ROI 解析归调用方（CoopPerception 的 _resolve_named_roi），坐标
换算归 Locator 体系。OCR 必须限域：ROI 缺失或零面积一律返回空，禁止全图
OCR 进入主循环（同 pet_difficulty 约束，spec 2026-09-06 §4.4）。
离线验证：三行页签全部读对、置信度 0.71~0.80（2026-09-08 实机截图）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wlxq_bot.perception.ocr import TitleReader
from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

_default_reader: TitleReader | None = None


@dataclass(frozen=True)
class ModeTabRow:
    """模式页签行内一条识别结果；position 为文字框中心（客户区像素）。"""

    text: str
    confidence: float
    position: tuple[int, int]


def _get_default_reader() -> TitleReader:
    global _default_reader
    if _default_reader is None:
        _default_reader = TitleReader()
    return _default_reader


def read_mode_tab_rows(
    frame: Any,
    roi: tuple[int, int, int, int] | None,
    reader: Any | None = None,
) -> list[ModeTabRow]:
    """在 ROI 内做一次 OCR，返回页签行结果；roi 为 (x, y, w, h) 客户区像素。

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
        ModeTabRow(text=str(text).strip(), confidence=float(conf), position=(cx + x, cy + y))
        for text, conf, (cx, cy) in lines
        if str(text).strip()
    ]
    logger.debug("模式页签行识别 ROI=%s 行=%s", (x, y, w, h), [row.text for row in rows])
    return rows
