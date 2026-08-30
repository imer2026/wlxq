"""棋盘英雄分类素材的完整客户区截图采集器。"""

from __future__ import annotations

import csv
import math
import queue
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from wlxq_bot.models import CoopRole, WindowContext
from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

_CAPTURE_MANIFEST_FIELDS = (
    "round_id",
    "main_c",
    "frame_index",
    "frame_id",
    "captured_at",
    "display_resolution",
    "client_width",
    "client_height",
    "dpi",
    "role",
    "source_screenshot",
    "capture_duration_ms",
    "save_duration_ms",
    "status",
    "failure_reason",
)


@dataclass(frozen=True)
class CaptureRequest:
    """等待异步落盘的一张完整客户区截图。"""

    round_id: str
    frame_index: int
    context: WindowContext
    frame: np.ndarray
    output_path: Path
    capture_duration_ms: float


@dataclass
class CaptureRecord:
    """一张计划截图的采集和保存结果。"""

    round_id: str
    main_c: str
    frame_index: int
    frame_id: int | None = None
    captured_at: float | None = None
    display_resolution: str = ""
    client_width: int | None = None
    client_height: int | None = None
    dpi: int | None = None
    role: str = ""
    source_screenshot: str = ""
    capture_duration_ms: float | None = None
    save_duration_ms: float | None = None
    status: str = "pending"
    failure_reason: str = ""

    def as_csv_row(self) -> dict[str, str | int | float]:
        """转换为可写入 CSV 的扁平记录。"""
        return {
            "round_id": self.round_id,
            "main_c": self.main_c,
            "frame_index": self.frame_index,
            "frame_id": "" if self.frame_id is None else self.frame_id,
            "captured_at": "" if self.captured_at is None else f"{self.captured_at:.6f}",
            "display_resolution": self.display_resolution,
            "client_width": "" if self.client_width is None else self.client_width,
            "client_height": "" if self.client_height is None else self.client_height,
            "dpi": "" if self.dpi is None else self.dpi,
            "role": self.role,
            "source_screenshot": self.source_screenshot,
            "capture_duration_ms": (
                "" if self.capture_duration_ms is None else f"{self.capture_duration_ms:.3f}"
            ),
            "save_duration_ms": (
                "" if self.save_duration_ms is None else f"{self.save_duration_ms:.3f}"
            ),
            "status": self.status,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class CaptureStats:
    """一次完整客户区素材采集的汇总。"""

    expected: int
    captured: int
    saved: int
    failed: int
    dropped: int
    schedule_skipped: int
    max_queue_depth: int
    elapsed_seconds: float
    capture_avg_ms: float
    capture_max_ms: float
    save_avg_ms: float
    save_max_ms: float
    manifest_path: Path


@dataclass
class _WriterState:
    records: dict[int, CaptureRecord]
    save_durations_ms: list[float] = field(default_factory=list)
    saved: int = 0
    failed: int = 0


class HeroFrameCollector:
    """按固定目标时间点采集完整客户区截图，并由后台线程异步保存 PNG。"""

    def __init__(
        self,
        capture_frame: Callable[[], tuple[WindowContext, object]],
        *,
        round_dir: Path,
        round_id: str,
        main_c: str,
        role: CoopRole,
        display_resolution: tuple[int, int],
        interval_seconds: float = 1.0,
        duration_seconds: float = 360.0,
        queue_size: int = 8,
        png_compression: int = 1,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if re.fullmatch(r"\d{12}", round_id) is None:
            raise ValueError("round_id 必须是有效的 YYYYMMDDHHMM 本地时间")
        try:
            parsed_round_id = datetime.strptime(round_id, "%Y%m%d%H%M")
        except ValueError as exc:
            raise ValueError("round_id 必须是有效的 YYYYMMDDHHMM 本地时间") from exc
        if parsed_round_id.strftime("%Y%m%d%H%M") != round_id:
            raise ValueError("round_id 必须是有效的 YYYYMMDDHHMM 本地时间")
        if re.fullmatch(r"[a-z][a-z0-9_]*", main_c) is None:
            raise ValueError("main_c 必须是小写英文标识，可包含数字和下划线")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds 必须大于 0")
        if duration_seconds <= 0:
            raise ValueError("duration_seconds 必须大于 0")
        if queue_size < 1:
            raise ValueError("queue_size 必须大于等于 1")
        if not 0 <= png_compression <= 9:
            raise ValueError("png_compression 必须在 0-9 之间")

        self._capture_frame = capture_frame
        self._round_dir = Path(round_dir)
        self._raw_dir = self._round_dir / "raw"
        self._manifest_path = self._round_dir / "capture_manifest.csv"
        self._round_id = round_id
        self._main_c = main_c
        self._role = role
        self._display_resolution = display_resolution
        self._interval = interval_seconds
        self._duration = duration_seconds
        self._queue_size = queue_size
        self._png_compression = png_compression
        self._monotonic = monotonic
        self._sleep = sleep

    def collect(self) -> CaptureStats:
        """执行采集；任何单帧失败都会记录，队列最终会完整排空。"""
        self._prepare_output()
        expected = max(1, math.ceil(self._duration / self._interval))
        records = {
            index: CaptureRecord(
                round_id=self._round_id,
                main_c=self._main_c,
                frame_index=index,
                display_resolution=f"{self._display_resolution[0]}x{self._display_resolution[1]}",
                role=self._role.value,
            )
            for index in range(1, expected + 1)
        }
        writer_state = _WriterState(records=records)
        save_queue: queue.Queue[CaptureRequest | None] = queue.Queue(maxsize=self._queue_size)
        writer = threading.Thread(
            target=self._writer_loop,
            args=(save_queue, writer_state),
            name="hero-classifier-png-writer",
            daemon=True,
        )
        writer.start()

        started = self._monotonic()
        capture_durations_ms: list[float] = []
        captured = 0
        dropped = 0
        schedule_skipped = 0
        max_queue_depth = 0

        try:
            for frame_index in range(1, expected + 1):
                deadline = started + (frame_index - 1) * self._interval
                now = self._monotonic()
                if now < deadline:
                    self._sleep(deadline - now)
                    now = self._monotonic()
                if now >= deadline + self._interval:
                    record = records[frame_index]
                    record.status = "schedule_skipped"
                    record.failure_reason = "采集处理落后超过一个完整间隔"
                    schedule_skipped += 1
                    continue

                capture_started = perf_counter()
                try:
                    context, raw_frame = self._capture_frame()
                    frame = self._validate_frame(raw_frame)
                    if context.is_minimized or not context.is_foreground:
                        raise RuntimeError("游戏窗口最小化或非前台")
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    duration_ms = (perf_counter() - capture_started) * 1000
                    record = records[frame_index]
                    record.capture_duration_ms = duration_ms
                    record.status = "capture_failed"
                    record.failure_reason = str(exc)
                    capture_durations_ms.append(duration_ms)
                    continue

                duration_ms = (perf_counter() - capture_started) * 1000
                capture_durations_ms.append(duration_ms)
                captured += 1
                output_path = self._raw_dir / f"{self._round_id}_frame{frame_index:06d}.png"
                record = records[frame_index]
                record.frame_id = context.frame_id
                record.captured_at = context.captured_at
                record.client_width = context.client_size[0]
                record.client_height = context.client_size[1]
                record.dpi = context.dpi
                record.source_screenshot = str(output_path.relative_to(self._round_dir)).replace(
                    "\\", "/"
                )
                record.capture_duration_ms = duration_ms
                request = CaptureRequest(
                    round_id=self._round_id,
                    frame_index=frame_index,
                    context=context,
                    frame=frame.copy(),
                    output_path=output_path,
                    capture_duration_ms=duration_ms,
                )
                try:
                    save_queue.put_nowait(request)
                    record.status = "queued"
                    max_queue_depth = max(max_queue_depth, save_queue.qsize())
                except queue.Full:
                    record.status = "queue_dropped"
                    record.failure_reason = f"保存队列已满（上限 {self._queue_size}）"
                    dropped += 1
        finally:
            save_queue.put(None)
            writer.join()

        elapsed = self._monotonic() - started
        self._write_manifest(records)
        failed = sum(
            1
            for record in records.values()
            if record.status not in {"saved", "queue_dropped", "schedule_skipped"}
        )
        return CaptureStats(
            expected=expected,
            captured=captured,
            saved=writer_state.saved,
            failed=failed,
            dropped=dropped,
            schedule_skipped=schedule_skipped,
            max_queue_depth=max_queue_depth,
            elapsed_seconds=elapsed,
            capture_avg_ms=self._average(capture_durations_ms),
            capture_max_ms=max(capture_durations_ms, default=0.0),
            save_avg_ms=self._average(writer_state.save_durations_ms),
            save_max_ms=max(writer_state.save_durations_ms, default=0.0),
            manifest_path=self._manifest_path,
        )

    def _prepare_output(self) -> None:
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        existing = next(self._raw_dir.glob("*.png"), None)
        if existing is not None or self._manifest_path.exists():
            raise FileExistsError(
                f"本局采集目录已有数据，拒绝覆盖: {self._round_dir}；请使用新的对局时间"
            )

    def _writer_loop(
        self,
        save_queue: queue.Queue[CaptureRequest | None],
        state: _WriterState,
    ) -> None:
        while True:
            request = save_queue.get()
            try:
                if request is None:
                    return
                started = perf_counter()
                record = state.records[request.frame_index]
                try:
                    self._write_png(request.output_path, request.frame)
                except (OSError, RuntimeError, cv2.error) as exc:
                    record.status = "save_failed"
                    record.failure_reason = str(exc)
                    state.failed += 1
                else:
                    record.status = "saved"
                    state.saved += 1
                duration_ms = (perf_counter() - started) * 1000
                record.save_duration_ms = duration_ms
                state.save_durations_ms.append(duration_ms)
            finally:
                save_queue.task_done()

    def _write_png(self, path: Path, frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode(
            ".png",
            frame,
            [cv2.IMWRITE_PNG_COMPRESSION, self._png_compression],
        )
        if not ok:
            raise RuntimeError(f"PNG 编码失败: {path}")
        try:
            encoded.tofile(str(path))
        except OSError as exc:
            raise OSError(f"PNG 写入失败: {path}: {exc}") from exc

    def _write_manifest(self, records: dict[int, CaptureRecord]) -> None:
        with self._manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=_CAPTURE_MANIFEST_FIELDS)
            writer.writeheader()
            for frame_index in sorted(records):
                writer.writerow(records[frame_index].as_csv_row())

    @staticmethod
    def _validate_frame(frame: object) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"截图必须是 numpy.ndarray，实际为 {type(frame).__name__}")
        if frame.ndim != 3 or frame.shape[2] not in {3, 4}:
            raise ValueError(f"截图通道异常: shape={frame.shape}")
        return frame[:, :, :3]

    @staticmethod
    def _average(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0
