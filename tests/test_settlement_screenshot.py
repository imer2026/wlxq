"""结算截图保存器单元测试。

验证按 模式/日期 分目录、按回合+时间命名的落盘规则，
以及按天目录的过期清理规则；不依赖真实游戏窗口。
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from wlxq_bot.debug.settlement import SettlementScreenshotSaver


def _frame() -> np.ndarray:
    return np.zeros((4, 6, 3), dtype=np.uint8)


def _captured_at(hour: int, minute: int, second: int) -> float:
    """构造当天指定时刻的 epoch 时间戳（本地时区）。"""
    local_midnight = time.strptime("2026-09-06", "%Y-%m-%d")
    return time.mktime(local_midnight) + hour * 3600 + minute * 60 + second


class TestSettlementScreenshotSaver:
    def test_save_writes_png_under_mode_date_folder(self, tmp_path: Path) -> None:
        saver = SettlementScreenshotSaver(tmp_path)

        path = saver.save(
            _frame(),
            mode="coop",
            round_number=3,
            captured_at=_captured_at(14, 32, 1),
        )

        assert path == tmp_path / "coop" / "20260906" / "round03_143201.png"
        assert path.is_file()
        # 落盘的是可回读的合法 PNG
        assert cv2.imread(str(path)) is not None

    def test_save_uses_screenshot_day_not_run_start_day(self, tmp_path: Path) -> None:
        """挂机跨天时按截图时刻归档（游戏次数按天算）。"""
        saver = SettlementScreenshotSaver(tmp_path)

        path = saver.save(
            _frame(),
            mode="pet",
            round_number=12,
            captured_at=_captured_at(0, 5, 9) + 24 * 3600,
        )

        assert path == tmp_path / "pet" / "20260907" / "round12_000509.png"

    def test_save_normalizes_unsafe_mode_name(self, tmp_path: Path) -> None:
        saver = SettlementScreenshotSaver(tmp_path)

        path = saver.save(
            _frame(),
            mode="pet defend!",
            round_number=1,
            captured_at=_captured_at(19, 30, 45),
        )

        assert path == tmp_path / "pet_defend" / "20260906" / "round01_193045.png"

    def test_save_rejects_non_ndarray_frame(self, tmp_path: Path) -> None:
        saver = SettlementScreenshotSaver(tmp_path)

        with pytest.raises(TypeError, match="numpy.ndarray"):
            saver.save(
                "not-a-frame",
                mode="coop",
                round_number=1,
                captured_at=_captured_at(10, 0, 0),
            )


def _day_dir(root: Path, mode: str, day: date) -> Path:
    """造一个按天归档的截图目录，内含一张占位截图。"""
    path = root / mode / day.strftime("%Y%m%d")
    path.mkdir(parents=True, exist_ok=True)
    (path / "round01_120000.png").write_bytes(b"png")
    return path


class TestSettlementCleanup:
    """启动时按天目录的过期清理（保留最近 N 天，含今天）。"""

    def test_deletes_only_dirs_older_than_retention(self, tmp_path: Path) -> None:
        today = date.today()
        kept = [
            _day_dir(tmp_path, "coop", today),
            _day_dir(tmp_path, "coop", today - timedelta(days=1)),
            _day_dir(tmp_path, "pet", today - timedelta(days=6)),
        ]
        expired = [
            _day_dir(tmp_path, "coop", today - timedelta(days=7)),
            _day_dir(tmp_path, "pet", today - timedelta(days=8)),
        ]

        deleted = SettlementScreenshotSaver(tmp_path).cleanup(7)

        assert set(deleted) == set(expired)
        assert all(path.is_dir() for path in kept)
        assert all(not path.exists() for path in expired)

    def test_ignores_unparseable_dir_names_and_files(self, tmp_path: Path) -> None:
        stray = tmp_path / "coop" / "not-a-date"
        stray.mkdir(parents=True)
        (stray / "keep.png").write_bytes(b"x")
        date_named_file = tmp_path / "pet" / "20200101"
        date_named_file.parent.mkdir(parents=True)
        date_named_file.write_text("不是目录，不能删")

        deleted = SettlementScreenshotSaver(tmp_path).cleanup(7)

        assert deleted == []
        assert stray.is_dir() and (stray / "keep.png").exists()
        assert date_named_file.exists()

    def test_missing_root_is_noop(self, tmp_path: Path) -> None:
        deleted = SettlementScreenshotSaver(tmp_path / "nope").cleanup(7)

        assert deleted == []

    def test_deletion_failure_skips_dir_and_continues(self, tmp_path: Path, monkeypatch) -> None:
        import shutil

        locked = _day_dir(tmp_path, "coop", date.today() - timedelta(days=9))
        older = _day_dir(tmp_path, "pet", date.today() - timedelta(days=10))
        real_rmtree = shutil.rmtree

        def flaky_rmtree(path, *args, **kwargs):
            if Path(path) == locked:
                raise OSError("目录被占用")
            return real_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(shutil, "rmtree", flaky_rmtree)

        deleted = SettlementScreenshotSaver(tmp_path).cleanup(7)

        assert deleted == [older]
        assert locked.is_dir() and not older.exists()
