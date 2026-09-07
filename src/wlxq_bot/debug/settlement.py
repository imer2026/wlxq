"""结算截图保存器。

一局结束、点击返回前保存当前结算画面，按 模式/日期 分目录归档，
供玩家事后查看每局结果（游戏次数按天算，目录也按天分）。

只做记录，不参与业务决策；保存失败由调用方记日志后继续主流程。
启动时可调用 cleanup 删除超过保留期的按天目录，只保留最近一周。
"""

from __future__ import annotations

import re
import shutil
import time
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np

from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

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

    def cleanup(self, retention_days: int) -> list[Path]:
        """删除超过保留期的按天目录，返回已删除的目录路径列表。

        扫描 ``<root>/<模式>/<YYYYMMDD>/``：目录名能解析成日期且不晚于
        「今天 - retention_days」的整目录删除，即保留今天往前数
        retention_days 天。目录名不是合法日期（或同名文件）一律不动，
        绝不误删；单个目录删除失败记 WARNING 跳过并继续。
        """
        cutoff = date.today() - timedelta(days=retention_days)
        deleted: list[Path] = []
        if not self._root_dir.is_dir():
            return deleted
        for mode_dir in sorted(self._root_dir.iterdir()):
            if not mode_dir.is_dir():
                continue
            for day_dir in sorted(mode_dir.iterdir()):
                if not day_dir.is_dir():
                    continue
                try:
                    day = time.strptime(day_dir.name, "%Y%m%d")
                except ValueError:
                    continue
                if date(day.tm_year, day.tm_mon, day.tm_mday) > cutoff:
                    continue
                try:
                    shutil.rmtree(day_dir)
                except OSError as exc:
                    logger.warning("结算截图过期目录删除失败，跳过 %s: %r", day_dir, exc)
                    continue
                deleted.append(day_dir)
        return deleted

    @staticmethod
    def _as_image(frame: object) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frame 必须是 numpy.ndarray，实际为 {type(frame).__name__}")
        return frame
