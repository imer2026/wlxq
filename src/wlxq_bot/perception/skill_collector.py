"""技能卡采集器：技能页出现时顺手裁卡存档，供离线建立技能清单。

设计约束（不可阻塞合作主流程）：
- 由 CoopPerception 在技能页识别命中后调用；observe 内部全量捕获异常，
  连续失败达 ``fuse_max_consecutive_failures`` 后自动停用，只记日志
- 写盘走后台守护线程 + 有界队列，生产者入队即返回；队列满丢弃并计数，
  绝不等待、绝不影响识别与决策的主循环节奏
- 不参与业务决策、不产生任何输入、不额外截图（复用主流程已在手的帧）
- 只在统计阶段启用（run.skill_collection.enabled）；英雄技能固定，
  采齐后即可关闭

归属规则：每张技能卡的中部图标区匹配 ``run.skill_collection.hero_icons``
中配置的英雄图标模板，取最高分且过 ``min_icon_confidence`` 者为归属；
都不过阈值的卡按 unknown 归档，离线人工补标。重复卡按感知哈希（aHash）
去重，同一张卡多帧/多局只存一份。
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from wlxq_bot.perception.vision import Vision
from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

# 三等分切列数：技能页固定三张候选卡
_COLUMN_COUNT = 3


@dataclass(frozen=True)
class _CaptureTask:
    """一条待落盘的采集记录。"""

    image_path: Path
    png_bytes: bytes
    meta: dict[str, Any]


class SkillCollector:
    """技能卡采集器。observe 契约：任何情况下都不抛异常、不阻塞。"""

    def __init__(
        self,
        *,
        output_dir: Path,
        hero_icons: dict[str, list[Path]],
        vision: Vision,
        icon_band: tuple[float, float, float, float] = (0.12, 0.28, 0.76, 0.38),
        column_inset_ratio: float = 0.04,
        top_trim_ratio: float = 0.06,
        min_icon_confidence: float = 0.70,
        fuse_max_consecutive_failures: int = 5,
        queue_maxsize: int = 64,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._captures_dir = self._output_dir / "captures"
        self._meta_path = self._output_dir / "meta.jsonl"
        self._hero_icons = {
            hero: [Path(p) for p in paths if Path(p).is_file()]
            for hero, paths in hero_icons.items()
        }
        self._vision = vision
        self._icon_band = icon_band
        self._column_inset_ratio = column_inset_ratio
        self._top_trim_ratio = top_trim_ratio
        self._min_icon_confidence = min_icon_confidence
        self._fuse_max = max(1, fuse_max_consecutive_failures)
        self._queue: queue.Queue[_CaptureTask | None] = queue.Queue(maxsize=queue_maxsize)

        self._known_hashes: set[str] = set()
        self._consecutive_failures = 0
        self._disabled = False
        self._dropped = 0
        self._writer: threading.Thread | None = None

    @property
    def disabled(self) -> bool:
        """采集器是否已熔断停用。"""
        return self._disabled

    @property
    def dropped_count(self) -> int:
        """因写盘队列满而丢弃的采集条数（诊断用）。"""
        return self._dropped

    def observe(
        self,
        frame_id: int,
        frame: Any,
        roi: tuple[int, int, int, int] | None,
        page: str,
    ) -> None:
        """对一帧技能页画面做采集。永不抛异常、永不阻塞。

        Args:
            frame_id: 帧 ID（来自 WindowContext，仅用于追溯）
            frame: 截图帧（BGR ndarray）
            roi: 技能三候选 ROI（客户区像素 x/y/w/h）；None 表示未标定，
                裁剪会失真，直接跳过采集
            page: 出现技能页的任务状态名（如 SELECT_OPENING_SKILLS），仅入档
        """
        if self._disabled or not self._hero_icons:
            return
        try:
            self._observe(frame_id, frame, roi, page)
        except Exception as exc:  # noqa: BLE001 - 采集失败绝不外泄给主流程
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._fuse_max:
                self._disabled = True
                logger.warning(
                    "技能卡采集连续失败 %d 次，已自动停用 last_error=%r",
                    self._consecutive_failures,
                    exc,
                )
            else:
                logger.warning(
                    "技能卡采集失败（连续 %d/%d，不影响对局） frame=%s error=%r",
                    self._consecutive_failures,
                    self._fuse_max,
                    frame_id,
                    exc,
                )

    def close(self, timeout: float = 5.0) -> None:
        """停止写盘线程并等待队列排空；未启动过写盘时为空操作。"""
        if self._writer is None:
            return
        try:
            self._queue.put(None, timeout=timeout)
        except queue.Full:
            logger.warning("技能卡采集写盘队列排空超时，剩余 %d 条未落盘", self._queue.qsize())
            return
        self._writer.join(timeout=timeout)
        self._writer = None

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _observe(
        self,
        frame_id: int,
        frame: Any,
        roi: tuple[int, int, int, int] | None,
        page: str,
    ) -> None:
        if roi is None:
            return
        self._ensure_writer()
        height, width = frame.shape[:2]
        rx, ry, rw, rh = roi
        rw = min(rw, width - rx)
        rh = min(rh, height - ry)
        if rw <= 0 or rh <= 0:
            return
        column_width = rw / _COLUMN_COUNT
        inset = int(column_width * self._column_inset_ratio)
        top_trim = int(rh * self._top_trim_ratio)
        for column in range(_COLUMN_COUNT):
            left = int(rx + column * column_width) + inset
            right = int(rx + (column + 1) * column_width) - inset
            top = ry + top_trim
            bottom = ry + rh
            if right - left < 8 or bottom - top < 8:
                continue
            card = frame[top:bottom, left:right]
            digest = self._ahash(card)
            if digest in self._known_hashes:
                continue
            self._known_hashes.add(digest)
            hero, hero_confidence, icon_template = self._match_hero(card)
            self._enqueue(
                _CaptureTask(
                    image_path=self._captures_dir / f"{digest}.png",
                    png_bytes=cv2.imencode(".png", card)[1].tobytes(),
                    meta={
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "frame_id": frame_id,
                        "page": page,
                        "column": column,
                        "hash": digest,
                        "hero": hero,
                        "hero_confidence": round(hero_confidence, 4),
                        "icon_template": icon_template,
                    },
                )
            )

    def _match_hero(self, card: np.ndarray) -> tuple[str | None, float, str]:
        """在卡内图标区匹配英雄图标模板，返回 (英雄名, 置信度, 模板路径)。"""
        band_height, band_width = card.shape[:2]
        bx, by, bw, bh = self._icon_band
        band = card[
            int(by * band_height) : int((by + bh) * band_height),
            int(bx * band_width) : int((bx + bw) * band_width),
        ]
        if band.size == 0:
            return None, 0.0, ""
        best: tuple[str | None, float, str] = (None, 0.0, "")
        for hero, templates in self._hero_icons.items():
            for template_path in templates:
                match = self._vision.match_template(
                    band, str(template_path), threshold=self._min_icon_confidence
                )
                if match is not None and match.confidence > best[1]:
                    # meta 要写入 JSONL,模板路径统一转 str
                    best = (hero, match.confidence, str(template_path))
        return best

    def _enqueue(self, task: _CaptureTask) -> None:
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            self._dropped += 1
            if self._dropped == 1 or self._dropped % 50 == 0:
                logger.warning(
                    "技能卡采集写盘队列已满，累计丢弃 %d 条（不影响对局）", self._dropped
                )

    def _ensure_writer(self) -> None:
        if self._writer is not None and self._writer.is_alive():
            return
        self._captures_dir.mkdir(parents=True, exist_ok=True)
        self._writer = threading.Thread(
            target=self._writer_loop, name="skill-collector-writer", daemon=True
        )
        self._writer.start()

    def _writer_loop(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                return
            try:
                task.image_path.parent.mkdir(parents=True, exist_ok=True)
                task.image_path.write_bytes(task.png_bytes)
                with self._meta_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(task.meta, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.warning("技能卡采集落盘失败 path=%s error=%r", task.image_path, exc)

    @staticmethod
    def _ahash(image: np.ndarray) -> str:
        """8x8 平均哈希（aHash），用于同卡多帧/多局去重。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        mean = float(small.mean())
        bits = (small > mean).flatten()
        return "".join("1" if bit else "0" for bit in bits)
