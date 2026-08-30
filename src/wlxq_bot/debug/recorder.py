"""Debug Recorder 实现。

订阅 DebugEvent，把原始截图、标注图、识别结果和动作日志
保存到 screenshots/debug/ 目录。

不参与业务决策，任务状态机不依赖 Debug Recorder。
"""

from __future__ import annotations

import re
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from wlxq_bot.models import DebugEvent, MatchResult

# 退出帧：(frame_id, captured_at 时间戳, BGR 图像)
ExitFrame = tuple[int, float, object]


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
    ) -> None:
        self._debug_dir = Path(debug_dir)
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        self._events: list[DebugEvent] = []
        self._exit_frames: deque[ExitFrame] = deque(maxlen=exit_frame_buffer_size)

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
        folder = self._debug_dir / base
        suffix = 1
        while folder.exists():
            suffix += 1
            folder = self._debug_dir / f"{base}_{suffix}"
        folder.mkdir(parents=True)
        return folder

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
