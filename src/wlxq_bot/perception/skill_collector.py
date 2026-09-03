"""技能卡采集器：技能页出现时顺手裁卡存档，供离线建立技能清单。

目录结构：卡图按局（run）开始时间分目录，``<输出目录>/<局开始时间>/
序号_时分秒.png``；``meta.jsonl`` 集中在输出根目录，每行记录图片相对
路径、完整哈希、页面与帧号——是否重复采集由内存哈希集合判断，离线建册
（build-skill-catalog）按 meta 哈希去重与追溯。

设计约束（不可侵蚀主循环的截图时效预算）：
- 由 CoopPerception 在「页面标志可见 + 技能卡图标已实际命中（即将点卡）」
  时调用——这是卡片渲染完整的最早时刻，英雄图标必然在画面上；只看任务
  状态会裁到棋盘帧，只看页面标志会裁到卡片/图标未渲染完的过渡帧
- 节流：``min_collect_interval_seconds`` 内只采一次。技能页是静态的，
  采一次就够；动画帧的重复采集毫无价值
- 运行时只做裁剪和 aHash（毫秒级）；PNG 编码在写盘线程完成；
  英雄归属和文字 OCR 全部在离线建册（``build-skill-catalog``）时进行，
  运行时不做任何模板匹配
- 写盘走后台守护线程 + 有界队列，队列满丢弃并计数，绝不等待
- 不参与业务决策、不产生任何输入、不额外截图（复用主流程已在手的帧）
- 只在统计阶段启用（run.skill_collection.enabled）；英雄技能固定，
  采齐后即可关闭
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

# 三等分切列数：技能页固定三张候选卡
_COLUMN_COUNT = 3


@dataclass(frozen=True)
class _CaptureTask:
    """一条待落盘的采集记录；PNG 编码延迟到写盘线程。"""

    image_path: Path
    card: np.ndarray
    meta: dict[str, Any]


class SkillCollector:
    """技能卡采集器。observe 契约：任何情况下都不抛异常、不阻塞。"""

    def __init__(
        self,
        *,
        output_dir: Path,
        session_label: str = "",
        column_inset_ratio: float = 0.04,
        top_trim_ratio: float = 0.06,
        fuse_max_consecutive_failures: int = 5,
        min_collect_interval_seconds: float = 30.0,
        queue_maxsize: int = 64,
    ) -> None:
        self._output_dir = Path(output_dir)
        # 卡图按局（run）开始时间分目录存放；meta.jsonl 仍集中在输出根目录
        self._captures_dir = (
            self._output_dir / session_label if session_label else self._output_dir
        )
        self._session_label = session_label
        self._meta_path = self._output_dir / "meta.jsonl"
        self._column_inset_ratio = column_inset_ratio
        self._top_trim_ratio = top_trim_ratio
        self._fuse_max = max(1, fuse_max_consecutive_failures)
        self._min_interval = max(0.0, min_collect_interval_seconds)
        self._queue: queue.Queue[_CaptureTask | None] = queue.Queue(maxsize=queue_maxsize)

        self._known_hashes: set[str] = set()
        self._consecutive_failures = 0
        self._disabled = False
        self._dropped = 0
        self._seq = 0
        self._last_collect = float("-inf")
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
        if self._disabled:
            return
        # 节流：间隔内直接返回，把运行时增量压到接近零
        now = time.monotonic()
        if now - self._last_collect < self._min_interval:
            return
        try:
            self._last_collect = now
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
            # 文件名 = 序号_时分秒,人能直接读懂;
            # 是否采过(内容去重)由内存里的 aHash 集合判断,完整哈希只记在 meta
            self._seq += 1
            now = datetime.now()
            filename = f"{self._seq:03d}_{now:%H%M%S}.png"
            image_rel = f"{self._session_label}/{filename}" if self._session_label else filename
            self._enqueue(
                _CaptureTask(
                    image_path=self._captures_dir / filename,
                    card=card,
                    meta={
                        "ts": now.isoformat(timespec="seconds"),
                        "frame_id": frame_id,
                        "page": page,
                        "column": column,
                        "hash": digest,
                        "image": image_rel,
                    },
                )
            )

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
                ok, buf = cv2.imencode(".png", task.card)
                if not ok:
                    logger.warning("技能卡 PNG 编码失败 path=%s", task.image_path)
                    continue
                task.image_path.parent.mkdir(parents=True, exist_ok=True)
                task.image_path.write_bytes(buf.tobytes())
                with self._meta_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(task.meta, ensure_ascii=False) + "\n")
            except (OSError, cv2.error) as exc:
                logger.warning("技能卡采集落盘失败 path=%s error=%r", task.image_path, exc)

    @staticmethod
    def _ahash(image: np.ndarray) -> str:
        """8x8 平均哈希（aHash），用于同卡多帧/多局去重。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        mean = float(small.mean())
        bits = (small > mean).flatten()
        return "".join("1" if bit else "0" for bit in bits)
