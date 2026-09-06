"""结算截图保存器。

一局结束、点击返回前保存当前结算画面，按 模式/日期 分目录归档，
供玩家事后查看每局结果（游戏次数按天算，目录也按天分）。

只做记录，不参与业务决策；保存失败由调用方记日志后继续主流程。
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import cv2
import numpy as np

DEFAULT_SETTLEMENT_DIR = "screenshots/settlement"


class SettlementScreenshotSaver:
    """按 ``<root>/<模式>/<YYYYMMDD>/roundNN_HHMMSS.png`` 保存结算画面。"""

    def __init__(self, root_dir: str | Path = DEFAULT_SETTLEMENT_DIR) -> None:
        self._root_dir = Path(root_dir)

    def save(
        self,
        frame: object,
        mode: str,
        round_number: int,
        captured_at: float,
    ) -> Path:
        """保存一帧结算画面，返回文件路径。

        Args:
            frame: BGR 截图帧
            mode: 组队对局模式（RunConfig.mode，如 coop/pet），用作一级目录
            round_number: 当前局序号（从 1 起）；返回验证成功前 round_count
                尚未自增，由调用方传 round_count + 1
            captured_at: 该帧的截图时刻（time.time() 秒）；目录按天归档、
                文件名带时刻都取自它，挂机跨天时归到截图当天
        """
        image = self._as_image(frame)
        local_time = time.localtime(captured_at)
        safe_mode = re.sub(r"[^A-Za-z0-9._-]+", "_", mode).strip("._") or "unknown"
        folder = self._root_dir / safe_mode / time.strftime("%Y%m%d", local_time)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"round{round_number:02d}_{time.strftime('%H%M%S', local_time)}.png"
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError(f"PNG 编码失败: {path}")
        encoded.tofile(str(path))
        return path

    @staticmethod
    def _as_image(frame: object) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frame 必须是 numpy.ndarray，实际为 {type(frame).__name__}")
        return frame
