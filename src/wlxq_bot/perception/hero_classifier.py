"""棋盘英雄格 ONNX 分类器运行时适配。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_HERO_CLASS_RE = re.compile(r"^(?P<hero>[a-z0-9_]+)_star(?P<star>[1-4])$")


@dataclass(frozen=True)
class HeroCellPrediction:
    """单个格子的分类结果；低置信度结果以 rejected 标记。"""

    class_name: str
    hero_type: str | None
    star_level: int | None
    confidence: float
    margin: float
    rejected: bool
    raw_class_name: str = ""
    second_class_name: str = ""
    second_confidence: float = 0.0
    rejection_reason: str = ""


class HeroCellClassifier:
    """使用 OpenCV DNN 加载训练导出的轻量 ONNX 模型。"""

    def __init__(
        self,
        model_path: Path,
        metadata_path: Path | None = None,
        *,
        net: Any | None = None,
        confidence_threshold: float | None = None,
        margin_threshold: float | None = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._metadata_path = metadata_path or self._model_path.with_suffix(".json")
        metadata = self._load_metadata(self._metadata_path)
        self._class_names = tuple(str(name) for name in metadata["class_names"])
        self._input_size = int(metadata.get("input_size", 96))
        self._mean = np.asarray(metadata.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self._std = np.asarray(metadata.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        self._confidence_threshold = float(
            metadata.get("confidence_threshold", 0.8)
            if confidence_threshold is None
            else confidence_threshold
        )
        self._margin_threshold = float(
            metadata.get("margin_threshold", 0.2) if margin_threshold is None else margin_threshold
        )
        if not self._class_names:
            raise ValueError("分类器 metadata.class_names 不能为空")
        invalid_classes = [
            name
            for name in self._class_names
            if name not in {"empty", "unavailable"} and _HERO_CLASS_RE.fullmatch(name) is None
        ]
        if invalid_classes:
            raise ValueError(f"分类器 metadata 包含无法解析的类别: {', '.join(invalid_classes)}")
        if self._mean.shape != (3,) or self._std.shape != (3,) or np.any(self._std <= 0):
            raise ValueError("分类器 metadata 的 mean/std 必须是 3 个有效通道值")
        if not 0 <= self._confidence_threshold <= 1:
            raise ValueError("分类器 confidence_threshold 必须在 0-1")
        if not 0 <= self._margin_threshold <= 1:
            raise ValueError("分类器 margin_threshold 必须在 0-1")
        if net is None:
            if not self._model_path.is_file():
                raise FileNotFoundError(f"英雄格分类模型不存在: {self._model_path}")
            net = cv2.dnn.readNetFromONNX(str(self._model_path))
        self._net = net

    def predict(self, images: list[np.ndarray]) -> list[HeroCellPrediction]:
        """批量分类 BGR 格子图；空输入直接返回空列表。"""
        if not images:
            return []
        batch = np.stack([self._preprocess(image) for image in images])
        self._net.setInput(batch)
        logits = np.asarray(self._net.forward(), dtype=np.float32)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        if logits.shape != (len(images), len(self._class_names)):
            raise RuntimeError(
                "ONNX 输出尺寸与 metadata 不一致: "
                f"output={logits.shape} images={len(images)} classes={len(self._class_names)}"
            )
        probabilities = self._softmax(logits)
        return [self._prediction(row) for row in probabilities]

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] < 3:
            raise TypeError("英雄格图片必须是至少 3 通道的 numpy.ndarray")
        resized = cv2.resize(
            image[:, :, :3],
            (self._input_size, self._input_size),
            interpolation=cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self._mean) / self._std
        return np.transpose(normalized, (2, 0, 1)).astype(np.float32)

    def _prediction(self, probabilities: np.ndarray) -> HeroCellPrediction:
        order = np.argsort(probabilities)[::-1]
        best_index = int(order[0])
        confidence = float(probabilities[best_index])
        second_index = int(order[1]) if len(order) > 1 else None
        second = float(probabilities[second_index]) if second_index is not None else 0.0
        second_class_name = self._class_names[second_index] if second_index is not None else ""
        margin = confidence - second
        class_name = self._class_names[best_index]
        rejection_reasons: list[str] = []
        if confidence < self._confidence_threshold:
            rejection_reasons.append("low_confidence")
        if margin < self._margin_threshold:
            rejection_reasons.append("low_margin")
        rejected = bool(rejection_reasons)
        hero_type: str | None = None
        star_level: int | None = None
        if not rejected and class_name not in {"empty", "unavailable"}:
            match = _HERO_CLASS_RE.fullmatch(class_name)
            if match is None:  # metadata 在初始化阶段已经校验；保留防御式检查。
                raise RuntimeError(f"无法解析英雄类别名称: {class_name}")
            hero_type = match.group("hero")
            star_level = int(match.group("star"))
        return HeroCellPrediction(
            class_name="unknown" if rejected else class_name,
            hero_type=hero_type,
            star_level=star_level,
            confidence=confidence,
            margin=margin,
            rejected=rejected,
            raw_class_name=class_name,
            second_class_name=second_class_name,
            second_confidence=second,
            rejection_reason="+".join(rejection_reasons),
        )

    @property
    def confidence_threshold(self) -> float:
        """返回本实例实际使用的置信度门槛。"""
        return self._confidence_threshold

    @property
    def margin_threshold(self) -> float:
        """返回本实例实际使用的第一、第二名差值门槛。"""
        return self._margin_threshold

    @property
    def class_names(self) -> tuple[str, ...]:
        """返回模型 metadata 声明的全部精确类别。"""
        return self._class_names

    @property
    def supported_heroes(self) -> frozenset[str]:
        """返回模型能够输出的英雄类型集合。"""
        return frozenset(
            match.group("hero")
            for name in self._class_names
            if (match := _HERO_CLASS_RE.fullmatch(name)) is not None
        )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        values = np.exp(shifted)
        return values / np.sum(values, axis=1, keepdims=True)

    @staticmethod
    def _load_metadata(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"英雄格分类 metadata 不存在: {path}")
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
        if not isinstance(value, dict) or not isinstance(value.get("class_names"), list):
            raise ValueError(f"英雄格分类 metadata 格式错误: {path}")
        return value
