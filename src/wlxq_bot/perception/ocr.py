"""技能标题 OCR 识别器与金币读取器。

依赖 rapidocr-onnxruntime，引擎按需懒加载并缓存；识别失败（依赖缺失、
引擎异常）由调用方决定兜底行为。技能标题只识别标题文字；金币读取识别
现有金币与召唤/选技能费用三个小数字区域（含 K 后缀解析与红=买不起判定）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

_engine: Any = None

# 金币/费用数字格式：纯数字或 K 后缀(3.3K)；中间不允许空格(识别结果先拼接)
_GOLD_VALUE_RE = re.compile(r"^\d+(?:\.\d+)?[Kk]?$")
# 红色(买不起)判定阈值：红字帧实测 0.17~0.27,白字帧实测 0.00
_RED_RATIO_THRESHOLD = 0.15
# 小 ROI 的 OCR 预处理放大倍数与边距(边距给检测器留上下文,无边距会漏检)
_GOLD_SCALE = 3
_GOLD_PAD = 30
# 边距填充色：接近按钮底色的棕褐色,避免纯白边框干扰二值化
_GOLD_PAD_COLOR = (120, 90, 60)


def ensure_engine() -> None:
    """预加载 OCR 引擎；依赖缺失时抛 RuntimeError，供启动期探测。"""
    _get_engine()


def _get_engine() -> Any:
    global _engine
    if _engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "技能标题识别需要 rapidocr-onnxruntime：pip install rapidocr-onnxruntime"
            ) from exc
        _engine = RapidOCR()
        logger.info("技能标题 OCR 引擎已加载(rapidocr-onnxruntime)")
    return _engine


class TitleReader:
    """横带标题识别器。

    read() 对输入图做一次 OCR，返回 [(文字, (中心x, 中心y))]，坐标相对
    输入图左上角。输入应为覆盖三张卡标题的水平窄条，一次调用即可同时
    读出三个标题，x 坐标用于区分卡片列。
    """

    def read(self, image: np.ndarray) -> list[tuple[str, tuple[int, int]]]:
        engine = _get_engine()
        result, _ = engine(image)
        lines: list[tuple[str, tuple[int, int]]] = []
        if not result:
            return lines
        for item in result:
            box = np.array(item[0], dtype=float)
            center = (int(box[:, 0].mean()), int(box[:, 1].mean()))
            text = str(item[1]).strip()
            if text:
                lines.append((text, center))
        return lines


@dataclass(frozen=True)
class GoldInfo:
    """一帧的金币感知结果；数值无法解析时为 None。"""

    current: int | None
    skill_cost: int | None
    summon_cost: int | None
    skill_red: bool
    summon_red: bool

    def can_afford_skill(self) -> bool:
        """选技能是否付得起：红色(游戏自身的买不起信号)或数值比较任一判负即不可。"""
        if self.skill_red:
            return False
        if self.current is not None and self.skill_cost is not None:
            return self.current >= self.skill_cost
        return True

    def can_afford_summon(self) -> bool:
        """召唤是否付得起：判定方式同选技能。"""
        if self.summon_red:
            return False
        if self.current is not None and self.summon_cost is not None:
            return self.current >= self.summon_cost
        return True


def parse_gold_value(text: str) -> int | None:
    """解析金币数字：支持 K 后缀(1.3K=1300)；无法解析返回 None。"""
    t = text.replace(" ", "").upper()
    if not _GOLD_VALUE_RE.fullmatch(t):
        return None
    if t.endswith("K"):
        return int(float(t[:-1]) * 1000)
    return int(float(t))


def red_ratio(image: np.ndarray) -> float:
    """高饱和像素中红色的占比：红字费用 ≈0.17~0.27,白字费用 ≈0.00。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    saturated = hsv[hsv[:, 1] > 120]
    if not len(saturated):
        return 0.0
    red = ((saturated[:, 0] < 10) | (saturated[:, 0] > 170)) & (saturated[:, 1] > 120)
    return float(red.sum()) / len(saturated)


class GoldReader:
    """金币读取器：读现有金币与召唤/选技能费用三个小数字区域。

    小 ROI 直接 OCR 会漏检(实机验证)，因此每个区域做 3 倍放大 + 30px
    边距填充，并依次尝试 原图 / R 通道 Otsu 二值化(红字) / G 通道 Otsu
    二值化(白字) 三种预处理，取第一个可解析为数字的结果。
    """

    def __init__(self, engine: Any = None) -> None:
        # engine 可注入用于测试；默认懒加载共享 RapidOCR 引擎
        self._injected_engine = engine

    def read(
        self,
        frame: np.ndarray,
        current_roi: tuple[int, int, int, int],
        skill_roi: tuple[int, int, int, int],
        summon_roi: tuple[int, int, int, int],
    ) -> GoldInfo:
        current_crop = self._crop(frame, current_roi)
        skill_crop = self._crop(frame, skill_roi)
        summon_crop = self._crop(frame, summon_roi)
        current = self._read_value(current_crop)
        skill_cost = self._read_value(skill_crop)
        summon_cost = self._read_value(summon_crop)
        return GoldInfo(
            current=current,
            skill_cost=skill_cost,
            summon_cost=summon_cost,
            skill_red=red_ratio(skill_crop) >= _RED_RATIO_THRESHOLD,
            summon_red=red_ratio(summon_crop) >= _RED_RATIO_THRESHOLD,
        )

    @staticmethod
    def _crop(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = roi
        return frame[y : y + h, x : x + w]

    def _read_value(self, crop: np.ndarray) -> int | None:
        texts: list[str] = []
        for variant in self._variants(crop):
            engine = self._injected_engine if self._injected_engine is not None else _get_engine()
            result, _ = engine(variant)
            if not result:
                continue
            text = "".join(str(item[1]) for item in result)
            value = parse_gold_value(text)
            if value is not None:
                return value
            if text:
                texts.append(text)
        # 无可解析数字时返回首个原始文本，调用方可用于调试日志
        return parse_gold_value(texts[0]) if texts else None

    @staticmethod
    def _variants(crop: np.ndarray) -> list[np.ndarray]:
        big = cv2.resize(
            crop,
            (crop.shape[1] * _GOLD_SCALE, crop.shape[0] * _GOLD_SCALE),
            interpolation=cv2.INTER_CUBIC,
        )
        big = cv2.copyMakeBorder(
            big,
            _GOLD_PAD,
            _GOLD_PAD,
            _GOLD_PAD,
            _GOLD_PAD,
            cv2.BORDER_CONSTANT,
            value=_GOLD_PAD_COLOR,
        )
        channels = (big[:, :, 2], big[:, :, 1])  # R 通道(红字) / G 通道(白字)
        variants = [big]
        for channel in channels:
            _, binary = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
        return variants
