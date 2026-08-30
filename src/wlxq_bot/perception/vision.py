"""Vision：模板匹配、颜色识别、调试标注。

职责：
- 在指定 ROI 中匹配模板，返回置信度和位置
- 支持单模板和模板集（template_set）匹配
- 合并位置相近的重复匹配（NMS）
- 生成调试标注图

约束：
- Vision 只负责识别，不做坐标换算和动作点计算
- 不直接调用 PyAutoGUI 或执行输入
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from wlxq_bot.models import MatchResult


class Vision:
    """图像识别器。

    模板加载带缓存，重复匹配同一模板时不重复读盘。
    Windows 中文路径用 imdecode + np.fromfile 绕过 cv2.imread 限制。
    """

    def __init__(self) -> None:
        self._template_cache: dict[str, np.ndarray] = {}

    def _load_template(self, path: str | Path) -> np.ndarray | None:
        """加载模板图片（带缓存）。

        Windows 上 cv2.imread 读不了中文路径，用 imdecode + np.fromfile 绕过。
        """
        key = str(path)
        if key in self._template_cache:
            return self._template_cache[key]
        try:
            data = np.fromfile(key, dtype=np.uint8)
        except OSError:
            return None
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None
        self._template_cache[key] = img
        return img

    @staticmethod
    def _crop_roi(
        frame: np.ndarray,
        roi: tuple[int, int, int, int] | None,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """裁剪搜索区域，返回 (search_area, offset)。

        offset 是 ROI 左上角在原图坐标系中的偏移，用于把搜索结果坐标还原。
        """
        if roi is None:
            return frame, (0, 0)
        x, y, w, h = roi
        return frame[y : y + h, x : x + w], (x, y)

    @staticmethod
    def _nms(
        candidates: list[tuple[int, int, float, str, int, int]],
        dist: int,
    ) -> list[tuple[int, int, float, str, int, int]]:
        """非极大值抑制：合并距离小于 dist 的候选，保留置信度最高的。

        Args:
            candidates: (x, y, conf, template_path, w, h) 列表，
                        x/y 为匹配左上角坐标
            dist: 合并距离阈值（欧氏距离）

        Returns:
            去重后的候选列表，按置信度降序。
        """
        if not candidates:
            return []
        sorted_cands = sorted(candidates, key=lambda c: -c[2])
        keep: list[tuple[int, int, float, str, int, int]] = []
        taken = [False] * len(sorted_cands)
        for i, cand in enumerate(sorted_cands):
            if taken[i]:
                continue
            keep.append(cand)
            x, y = cand[0], cand[1]
            for j in range(i + 1, len(sorted_cands)):
                if taken[j]:
                    continue
                dx = sorted_cands[j][0] - x
                dy = sorted_cands[j][1] - y
                if (dx * dx + dy * dy) ** 0.5 < dist:
                    taken[j] = True
        return keep

    def match_template(
        self,
        frame: np.ndarray,
        template_path: str,
        roi: tuple[int, int, int, int] | None = None,
        threshold: float = 0.85,
    ) -> MatchResult | None:
        """在帧中匹配单个模板，返回置信度最高的结果。

        Args:
            frame: 截图帧（BGR ndarray）
            template_path: 模板图片路径
            roi: 搜索区域 (x, y, width, height)，None 表示全图
            threshold: 置信度阈值

        Returns:
            匹配结果（位置为匹配中心），未达到阈值返回 None
        """
        template = self._load_template(template_path)
        if template is None:
            return None
        search, offset = self._crop_roi(frame, roi)
        th, tw = template.shape[:2]
        if th > search.shape[0] or tw > search.shape[1]:
            return None
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < threshold:
            return None
        return MatchResult(
            template_name=template_path,
            position=(
                max_loc[0] + offset[0] + tw // 2,
                max_loc[1] + offset[1] + th // 2,
            ),
            confidence=float(max_val),
        )

    def match_template_set(
        self,
        frame: np.ndarray,
        template_paths: list[str],
        roi: tuple[int, int, int, int] | None = None,
        threshold: float = 0.78,
        nms_dist: int | None = None,
    ) -> list[MatchResult]:
        """匹配多个模板变体，返回所有超过阈值的结果（经 NMS 去重）。

        用于英雄等动态对象的识别：同英雄多张模板各匹配各的，
        合并位置相近的重复匹配，返回独立峰值。

        Args:
            frame: 截图帧（BGR ndarray）
            template_paths: 模板文件路径列表
            roi: 搜索区域 (x, y, width, height)，None 表示全图
            threshold: 置信度阈值
            nms_dist: NMS 合并距离，None 则取首个模板宽度的一半

        Returns:
            去重后的匹配结果列表（位置为匹配中心），按置信度降序。
        """
        search, offset = self._crop_roi(frame, roi)
        candidates: list[tuple[int, int, float, str, int, int]] = []
        first_w: int | None = None
        for tp in template_paths:
            template = self._load_template(tp)
            if template is None:
                continue
            th, tw = template.shape[:2]
            if first_w is None:
                first_w = tw
            if th > search.shape[0] or tw > search.shape[1]:
                continue
            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(result >= threshold)
            for x, y in zip(xs, ys, strict=True):
                candidates.append(
                    (
                        int(x) + offset[0],
                        int(y) + offset[1],
                        float(result[y, x]),
                        tp,
                        tw,
                        th,
                    )
                )
        if not candidates:
            return []
        if nms_dist is None:
            nms_dist = max(20, (first_w or 20))
        filtered = self._nms(candidates, nms_dist)
        return [
            MatchResult(
                template_name=tp,
                position=(x + w // 2, y + h // 2),
                confidence=conf,
            )
            for x, y, conf, tp, w, h in filtered
        ]

    def annotate(
        self,
        frame: np.ndarray,
        matches: list[MatchResult],
    ) -> np.ndarray:
        """在帧上标注匹配结果，生成调试图。"""
        annotated = frame.copy()
        for m in matches:
            cx, cy = m.position
            cv2.rectangle(
                annotated,
                (cx - 40, cy - 40),
                (cx + 40, cy + 40),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated,
                f"{m.confidence:.2f}",
                (cx - 20, cy - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        return annotated
