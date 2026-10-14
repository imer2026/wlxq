"""Debug Recorder 实现。

订阅 DebugEvent，把原始截图、标注图、识别结果和动作日志
保存到 screenshots/debug/ 目录；退出帧目录单独收在
screenshots/debug/exit/ 子目录下并按保留策略清理旧目录。

不参与业务决策，任务状态机不依赖 Debug Recorder。
"""

from __future__ import annotations

import re
import shutil
import time
from collections import deque
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np

from wlxq_bot.models import DebugEvent, MatchResult
from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

# 退出帧：(frame_id, captured_at 时间戳, BGR 图像)
ExitFrame = tuple[int, float, object]

# exit 目录名时间戳：exit_<状态>_YYYYMMDD_HHMMSS[_N] 或 exit_YYYYMMDD_HHMMSS[_N]
_EXIT_DIR_NAME_RE = re.compile(r"^exit(?:_.+)?_(\d{8})_(\d{6})(?:_\d+)?$")


def _parse_exit_dir_date(name: str) -> date | None:
    """从 exit 目录名解析日期；命名不符合约定（含非法日期）返回 None。"""
    match = _EXIT_DIR_NAME_RE.match(name)
    if match is None:
        return None
    try:
        stamp = time.strptime(f"{match.group(1)} {match.group(2)}", "%Y%m%d %H%M%S")
    except ValueError:
        return None
    return date(stamp.tm_year, stamp.tm_mon, stamp.tm_mday)


class DebugRecorder:
    """调试记录器。

    动作日志和事件批量落盘仍待补充；截图和标注图已可保存。
    另维护主循环最近帧的环形缓冲（退出帧），任务非正常退出时批量落盘，
    供事后排查退出原因；不参与业务决策。
    """

    def __init__(
        self,
        debug_dir: str = "screenshots/debug",
        exit_frame_buffer_size: int = 0,
        exit_keep_last: int = 0,
        exit_retention_days: int = 7,
    ) -> None:
        self._debug_dir = Path(debug_dir)
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        self._exit_dir = self._debug_dir / "exit"
        self._exit_keep_last = max(0, exit_keep_last)
        self._exit_retention_days = max(0, exit_retention_days)
        self._events: list[DebugEvent] = []
        self._exit_frames: deque[ExitFrame] = deque(maxlen=exit_frame_buffer_size)
        self._prune_exit_dirs()

    def record(self, event: DebugEvent) -> None:
        """记录一个调试事件。"""
        self._events.append(event)

    def keep_exit_frame(self, frame_id: int, captured_at: float, image: object) -> None:
        """暂存一张主循环截图到退出帧缓冲（超出上限自动淘汰最旧帧）。"""
        self._exit_frames.append((frame_id, captured_at, image))

    def drain_exit_frames(self) -> list[ExitFrame]:
        """取出并清空退出帧缓冲。"""
        frames = list(self._exit_frames)
        self._exit_frames.clear()
        return frames

    def save_frame(self, frame: object, frame_id: int, prefix: str = "frame") -> Path:
        """保存 BGR 截图帧，Windows 中文路径也可写入。"""
        image = self._as_image(frame)
        path = self._new_path(prefix, frame_id)
        self._write_png(path, image)
        return path

    def save_annotated(
        self,
        frame: object,
        matches: list[MatchResult],
        frame_id: int,
        prefix: str = "annotated",
    ) -> Path:
        """保存带匹配位置、置信度标注的截图。"""
        annotated = self._as_image(frame).copy()
        for match in matches:
            cx, cy = match.position
            cv2.rectangle(annotated, (cx - 40, cy - 40), (cx + 40, cy + 40), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{match.confidence:.2f}",
                (cx - 20, cy - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        path = self._new_path(prefix, frame_id)
        self._write_png(path, annotated)
        return path

    def save_annotated_labeled(
        self,
        frame: object,
        items: list[tuple[int, int, str]],
        frame_id: int,
        prefix: str = "annotated",
    ) -> Path:
        """保存带自定义文本标注的截图。

        与 save_annotated 不同，每个标注项可携带任意文本（如难度号:置信度），
        用于难度识别等需要区分具体命中对象的场景。

        Args:
            frame: BGR 截图帧
            items: (cx, cy, text) 列表，在 (cx, cy) 处画框并写 text
            frame_id: 帧标识
            prefix: 文件名前缀
        """
        annotated = self._as_image(frame).copy()
        for cx, cy, text in items:
            cv2.rectangle(annotated, (cx - 60, cy - 35), (cx + 60, cy + 35), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                text,
                (cx - 55, cy - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        path = self._new_path(prefix, frame_id)
        self._write_png(path, annotated)
        return path

    def save_exit_frames(self, frames: list[ExitFrame], *, state: str = "") -> Path:
        """任务非正常退出时，把退出帧缓冲批量落盘到独立文件夹。

        Args:
            frames: (frame_id, captured_at, BGR 图像) 列表，按时间升序；
                非图像项（测试 Fake 截图等）跳过不写
            state: 退出时的任务状态，用于文件夹命名

        Returns:
            本次退出帧文件夹路径（全部帧被跳过时也创建，便于确认发生过退出）
        """
        folder = self._new_exit_dir(state)
        manifest: list[str] = []
        prev_captured_at: float | None = None
        for index, (frame_id, captured_at, image) in enumerate(frames):
            if not isinstance(image, np.ndarray):
                continue
            path = folder / f"{index:02d}_frame_{frame_id}.png"
            self._write_png(path, image)
            delta = 0.0 if prev_captured_at is None else captured_at - prev_captured_at
            manifest.append(
                f"{index:02d} frame_id={frame_id} captured_at="
                + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(captured_at))
                + f".{int(captured_at % 1 * 1000):03d}"
                + f" (+{delta:.3f}s)"
            )
            prev_captured_at = captured_at
        (folder / "frames.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
        return folder

    def _new_exit_dir(self, state: str) -> Path:
        safe_state = re.sub(r"[^A-Za-z0-9._-]+", "_", state).strip("._")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = f"exit_{safe_state}_{stamp}" if safe_state else f"exit_{stamp}"
        folder = self._exit_dir / base
        suffix = 1
        while folder.exists():
            suffix += 1
            folder = self._exit_dir / f"{base}_{suffix}"
        folder.mkdir(parents=True)
        self._prune_exit_dirs()
        return folder

    def _prune_exit_dirs(self) -> None:
        """把顶层遗留的 exit_* 目录挪进 exit/ 子目录，再按保留策略清理。

        保留策略按目录名时间戳判定（与结算截图清理同一风格）：
        超过 ``exit_retention_days`` 天的整目录删除；``exit_keep_last``
        只保留时间戳最新的 N 个；两项任一命中即删，0 表示关闭。
        解析不出时间戳的目录一律不动，单个目录删除失败记 WARNING 跳过。
        """
        self._migrate_legacy_exit_dirs()
        if self._exit_keep_last <= 0 and self._exit_retention_days <= 0:
            return
        if not self._exit_dir.is_dir():
            return
        dated: list[tuple[date, Path]] = []
        for entry in self._exit_dir.iterdir():
            if not entry.is_dir():
                continue
            entry_date = _parse_exit_dir_date(entry.name)
            if entry_date is not None:
                dated.append((entry_date, entry))
        dated.sort(reverse=True)
        cutoff = (
            date.today() - timedelta(days=self._exit_retention_days)
            if self._exit_retention_days > 0
            else None
        )
        removed = 0
        for rank, (entry_date, entry) in enumerate(dated):
            expired = cutoff is not None and entry_date <= cutoff
            overflow = self._exit_keep_last > 0 and rank >= self._exit_keep_last
            if not (expired or overflow):
                continue
            try:
                shutil.rmtree(entry)
            except OSError as exc:
                logger.warning("exit 过期目录删除失败，跳过 %s: %r", entry, exc)
                continue
            removed += 1
        if removed:
            logger.info(
                "exit 目录清理：删除 %d 个，剩余 %d 个（%s）",
                removed,
                len(dated) - removed,
                self._exit_dir,
            )

    def _migrate_legacy_exit_dirs(self) -> None:
        """历史版本把 exit_* 建在 debug 顶层；一次性挪进 exit/ 子目录。"""
        if not self._debug_dir.is_dir():
            return
        moved = 0
        for entry in sorted(self._debug_dir.glob("exit_*")):
            if not entry.is_dir():
                continue
            target = self._exit_dir / entry.name
            if target.exists():
                logger.warning("exit 目录迁移跳过（同名已存在）: %s", entry)
                continue
            try:
                self._exit_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(entry), str(target))
            except OSError as exc:
                logger.warning("exit 目录迁移失败，跳过 %s: %r", entry, exc)
                continue
            moved += 1
        if moved:
            logger.info("exit 目录迁移：已把 %d 个顶层目录挪入 %s", moved, self._exit_dir)

    def _new_path(self, prefix: str, frame_id: int) -> Path:
        timestamp_ms = int(time.time() * 1000)
        safe_prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", prefix).strip("._") or "debug"
        return self._debug_dir / f"{safe_prefix}_{frame_id}_{timestamp_ms}.png"

    @staticmethod
    def _as_image(frame: object) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frame 必须是 numpy.ndarray，实际为 {type(frame).__name__}")
        return frame

    @staticmethod
    def _write_png(path: Path, image: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError(f"PNG 编码失败: {path}")
        encoded.tofile(str(path))

    @property
    def events(self) -> list[DebugEvent]:
        """已记录的事件列表。"""
        return list(self._events)
