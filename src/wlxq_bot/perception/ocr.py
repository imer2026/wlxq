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

# 免检测整行识别（RapidOCR Global.width_height_ratio，默认 8）：宽高比超此值
# 的图跳过文本检测、整图按一行识别。检测路径会把窄条带按 min_side=736
# 放大后推理（慢速机实测 ~2.8s/条），整行路径只需一次识别（~0.6s）。
# 技能页标题带每列只有标题一行文本，正好适配整行识别。
_LINE_SHAPE_RATIO = 8
# 填充灰：与检测预处理 letterbox 同色，识别模型视其为无内容背景
_PAD_GRAY = 114

# 金币/费用数字格式：纯数字或 K 后缀(3.3K)；中间不允许空格(识别结果先拼接)
_GOLD_VALUE_RE = re.compile(r"^\d+(?:\.\d+)?[Kk]?$")
# 费用数字条里的噪声字符（金币图标边缘被 OCR 读成 '?' 等）：清洗后再解析
_GOLD_NOISE_RE = re.compile(r"[^0-9.Kk]")
# 红色(买不起)判定阈值：红字占比随文本长度变化——"2.5K"实测 0.233、
# "3K"实测 0.136（2026-09-11 实机：短文本红字像素少，0.15 阈值把红色
# 误判为白色导致盲点）；白字实测 ≈0.00。0.05 居中分隔两组
_RED_RATIO_THRESHOLD = 0.05
# 小 ROI 的 OCR 预处理放大倍数与边距(边距给检测器留上下文,无边距会漏检)
_GOLD_SCALE = 3
_GOLD_PAD = 30
# 边距填充色：接近按钮底色的棕褐色,避免纯白边框干扰二值化
_GOLD_PAD_COLOR = (120, 90, 60)


def ensure_engine() -> None:
    """预加载 OCR 引擎；依赖缺失时抛 RuntimeError，供启动期探测。"""
    _get_engine()


def create_ocr_engine() -> Any:
    """创建独立 OCR 引擎实例：后台线程专用，与主线程标题 OCR 隔离。"""
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


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

    read_with_confidence() 对输入图做一次 OCR（检测+识别），返回
    [(文字, 置信度, (中心x, 中心y))]，坐标相对输入图左上角。read() 是丢弃
    置信度的兼容封装。read_line() 把整图当一行识别（免检测快速路径，
    只适合确定单行文本的图）。
    """

    def __init__(self, engine: Any | None = None) -> None:
        # engine 可注入：后台金币线程用独立实例与主线程标题 OCR 隔离；
        # 缺省共享全局懒加载引擎
        self._injected_engine = engine

    def _engine(self) -> Any:
        return self._injected_engine if self._injected_engine is not None else _get_engine()

    def read_with_confidence(self, image: np.ndarray) -> list[tuple[str, float, tuple[int, int]]]:
        engine = self._engine()
        result, _ = engine(image)
        lines: list[tuple[str, float, tuple[int, int]]] = []
        if not result:
            return lines
        for item in result:
            box = np.array(item[0], dtype=float)
            center = (int(box[:, 0].mean()), int(box[:, 1].mean()))
            text = str(item[1]).strip()
            if text:
                lines.append((text, float(item[2]), center))
        return lines

    def read(self, image: np.ndarray) -> list[tuple[str, tuple[int, int]]]:
        return [(text, center) for text, _, center in self.read_with_confidence(image)]

    def read_line(self, image: np.ndarray) -> list[tuple[str, tuple[int, int]]]:
        """整图按单行文本识别（免检测快速路径）。

        图过窄时对称灰色填充到宽高比 > _LINE_SHAPE_RATIO，触发引擎的
        免检测分支（见模块注释）。返回 [(文字, (中心x, 中心y))] 与 read()
        同构（单行，中心即图中心）；无文本返回空列表。输入必须确实是
        单行文本——多行文本整行识别会串读。
        """
        if image is None or image.size == 0:
            return []
        height, width = image.shape[:2]
        target = height * _LINE_SHAPE_RATIO + 2
        if width < target:
            canvas = np.full((height, target, 3), _PAD_GRAY, dtype=np.uint8)
            left = (target - width) // 2
            canvas[:, left : left + width] = image
            image = canvas
            height, width = image.shape[:2]
        engine = self._engine()
        result, _ = engine(image)
        if not result:
            return []
        text = str(result[0][1]).strip()
        if not text:
            return []
        return [(text, (width // 2, height // 2))]


def tight_text_rows(image: np.ndarray) -> np.ndarray:
    """按行水平梯度能量把条带上下收紧到文字实际行。

    整行识别统一按 48px 高归一：不收紧时标题字符被缩到 ~20px、误读率明显
    升高（2026-09-13 离线 120 条带对比：收紧后门控通过率 95%）。找不到
    文字行（纯色带）时原样返回。
    """
    if image is None or image.size == 0:
        return image
    gray = image.mean(axis=2) if image.ndim == 3 else image.astype(float)
    gradient = np.abs(np.diff(gray, axis=1)).sum(axis=1)
    if gradient.size == 0 or gradient.max() == 0:
        return image
    rows = np.where(gradient > gradient.max() * 0.05)[0]
    if rows.size == 0:
        return image
    margin = max(2, image.shape[0] // 20)
    top = max(0, int(rows[0]) - margin)
    bottom = min(image.shape[0], int(rows[-1]) + 1 + margin)
    if bottom - top < 12:
        return image
    return image[top:bottom, :]


@dataclass(frozen=True)
class GoldInfo:
    """一帧的金币感知结果；数值无法解析时为 None。"""

    current: int | None
    skill_cost: int | None
    summon_cost: int | None
    skill_red: bool
    summon_red: bool
    # 只认色模式的判定依据（红色占比实测值），便于实机排查阈值问题
    skill_red_ratio: float = 0.0
    summon_red_ratio: float = 0.0

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


def clean_gold_text(text: str) -> str:
    """剥掉数字/K 后缀以外的噪声字符（如金币图标边缘被 OCR 读成 '?'）。"""
    return _GOLD_NOISE_RE.sub("", text).strip()


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

    def read_affordability(
        self,
        frame: np.ndarray,
        skill_roi: tuple[int, int, int, int],
        summon_roi: tuple[int, int, int, int],
    ) -> GoldInfo:
        """只认色的金币门控：红 = 游戏判买不起，白 = 付得起，不做数字 OCR。

        游戏自身的红色费用文字就是买不起信号（红字占比随文本长度变化：
        "2.5K" 实测 0.233、"3K" 实测 0.136；白字 ≈0.00，阈值 0.05 居中
        分隔），数值比较只是颜色之外的对账。颜色判定是纯 OpenCV 统计
        （毫秒级），替代数字 OCR 每次读取 ~5 秒的开销；数值字段全部为
        None，任务层 _can_afford 对 None 数值视为付得起，门控语义不变。
        """
        skill_ratio = red_ratio(self._crop(frame, skill_roi))
        summon_ratio = red_ratio(self._crop(frame, summon_roi))
        return GoldInfo(
            current=None,
            skill_cost=None,
            summon_cost=None,
            skill_red=skill_ratio >= _RED_RATIO_THRESHOLD,
            summon_red=summon_ratio >= _RED_RATIO_THRESHOLD,
            skill_red_ratio=skill_ratio,
            summon_red_ratio=summon_ratio,
        )

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

    def read_pair(
        self,
        frame: np.ndarray,
        current_roi: tuple[int, int, int, int],
        cost_roi: tuple[int, int, int, int],
        line_reader: TitleReader | None = None,
    ) -> tuple[int | None, int | None]:
        """后台数值读取：read_line 免检测快路径优先（~1s/区域，2026-09-26
        实机），解析不出再回退三变体检测路径（慢但稳）。"""
        current_crop = self._crop(frame, current_roi)
        cost_crop = self._crop(frame, cost_roi)
        current = self._read_value_fast(current_crop, line_reader)
        cost = self._read_value_fast(cost_crop, line_reader)
        return current, cost

    def _read_value_fast(
        self,
        crop: np.ndarray,
        line_reader: TitleReader | None,
    ) -> int | None:
        if line_reader is not None:
            big = cv2.resize(
                crop,
                (crop.shape[1] * _GOLD_SCALE, crop.shape[0] * _GOLD_SCALE),
                interpolation=cv2.INTER_CUBIC,
            )
            for text, _ in line_reader.read_line(big):
                value = parse_gold_value(clean_gold_text(text))
                if value is not None:
                    return value
        return self._read_value(crop)

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


class BackgroundGoldReader:
    """后台金币数值读取器：持有独立 OCR 引擎实例，与主线程标题 OCR 隔离。

    供 GoldCache 后台线程使用；接口收缩为 read_pair 一项。
    """

    def __init__(self, engine: Any | None = None) -> None:
        # 缺省首个 read_pair 时自建独立实例（懒加载，成本后置）；测试可注入假引擎
        self._engine = engine
        self._gold_reader: GoldReader | None = None
        self._line_reader: TitleReader | None = None

    def _ensure(self) -> None:
        if self._gold_reader is None or self._line_reader is None:
            if self._engine is None:
                self._engine = create_ocr_engine()
            self._gold_reader = GoldReader(engine=self._engine)
            self._line_reader = TitleReader(engine=self._engine)

    def read_pair(
        self,
        frame: np.ndarray,
        current_roi: tuple[int, int, int, int],
        cost_roi: tuple[int, int, int, int],
    ) -> tuple[int | None, int | None]:
        self._ensure()
        return self._gold_reader.read_pair(frame, current_roi, cost_roi, self._line_reader)
