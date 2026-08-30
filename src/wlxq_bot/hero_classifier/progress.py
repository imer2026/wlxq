"""英雄格离线长任务的节流进度日志。"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable


class ProgressLogger:
    """按时间间隔输出 INFO 进度；阶段开始和结束始终输出。"""

    def __init__(
        self,
        logger: logging.Logger,
        task: str,
        total: int,
        *,
        interval_seconds: float = 5.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if total < 0:
            raise ValueError("进度总数不能小于 0")
        if interval_seconds <= 0:
            raise ValueError("进度日志间隔必须大于 0")
        self._logger = logger
        self._task = task
        self._total = total
        self._interval = interval_seconds
        self._monotonic = monotonic
        self._started_at: float | None = None
        self._last_logged_at: float | None = None

    def start(self, *, detail: str = "") -> None:
        now = self._monotonic()
        self._started_at = now
        self._last_logged_at = now
        suffix = f" {detail}" if detail else ""
        self._logger.info("%s 开始 total=%d%s", self._task, self._total, suffix)

    def update(self, completed: int, *, detail: str = "") -> None:
        now = self._monotonic()
        if self._started_at is None:
            self.start()
            now = self._monotonic()
        if self._last_logged_at is not None and now - self._last_logged_at < self._interval:
            return
        self._last_logged_at = now
        self._log_progress(completed, now, detail)

    def finish(self, completed: int, *, detail: str = "") -> None:
        now = self._monotonic()
        if self._started_at is None:
            self._started_at = now
        elapsed = now - self._started_at
        percent = self._percent(completed)
        suffix = f" {detail}" if detail else ""
        self._logger.info(
            "%s 完成 processed=%d/%d percent=%.1f%% elapsed=%.1fs%s",
            self._task,
            completed,
            self._total,
            percent,
            elapsed,
            suffix,
        )

    def _log_progress(self, completed: int, now: float, detail: str) -> None:
        started_at = self._started_at if self._started_at is not None else now
        suffix = f" {detail}" if detail else ""
        self._logger.info(
            "%s 进度 processed=%d/%d percent=%.1f%% elapsed=%.1fs%s",
            self._task,
            completed,
            self._total,
            self._percent(completed),
            now - started_at,
            suffix,
        )

    def _percent(self, completed: int) -> float:
        if self._total == 0:
            return 100.0
        return min(100.0, max(0.0, completed * 100.0 / self._total))
