"""DebugRecorder 截图落盘测试。"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np

from wlxq_bot.debug.recorder import DebugRecorder
from wlxq_bot.models import MatchResult


def _make_exit_dir(parent: Path, name: str) -> Path:
    folder = parent / name
    folder.mkdir(parents=True)
    (folder / "frames.txt").write_text("0 frame_id=1\n", encoding="utf-8")
    return folder


def _exit_stamp(day: date) -> str:
    return day.strftime("%Y%m%d") + "_120000"


def test_save_duplicate_skill_annotated_frame(tmp_path) -> None:
    recorder = DebugRecorder(str(tmp_path))
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    matches = [
        MatchResult("skills/assault/a.png", (50, 60), 0.82),
        MatchResult("skills/assault/a.png", (150, 60), 0.94),
    ]

    path = recorder.save_annotated(
        frame,
        matches,
        frame_id=7,
        prefix="skill_duplicate_assault/a",
    )

    assert path.is_file()
    assert path.parent == tmp_path
    assert "/" not in path.name
    saved = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert saved is not None
    assert saved.shape == frame.shape


def test_exit_frames_saved_under_exit_subdir(tmp_path) -> None:
    """退出帧目录改存 <debug_dir>/exit/ 子目录，顶层不再出现 exit_*。"""
    recorder = DebugRecorder(str(tmp_path))

    folder = recorder.save_exit_frames(
        [(1, time.time(), np.zeros((4, 4, 3), dtype=np.uint8))], state="find_coop"
    )

    assert folder.parent == tmp_path / "exit"
    assert folder.name.startswith("exit_find_coop_")
    assert list(tmp_path.glob("exit_*")) == []


def test_migrates_legacy_top_level_exit_dirs(tmp_path) -> None:
    """历史版本建在 debug 顶层的 exit_* 目录在构造时自动挪进 exit/ 子目录。"""
    legacy = _make_exit_dir(tmp_path, "exit_find_coop_20260926_180621")
    stray_file = tmp_path / "exit_looks_like_name.png"
    stray_file.write_bytes(b"x")
    plain_png = tmp_path / "frame_1_123.png"
    plain_png.write_bytes(b"x")

    DebugRecorder(str(tmp_path), exit_retention_days=0)

    assert not legacy.exists()
    assert (tmp_path / "exit" / legacy.name).is_dir()
    # 非 exit 目录（含同名前缀文件）一律不动
    assert stray_file.exists()
    assert plain_png.exists()


def test_retention_days_prunes_old_exit_dirs(tmp_path) -> None:
    """超过 exit_retention_days 的 exit 目录整目录删除，解析失败的不动。"""
    today = date.today()
    old_dir = _make_exit_dir(
        tmp_path / "exit", f"exit_find_coop_{_exit_stamp(today - timedelta(days=8))}"
    )
    recent_dir = _make_exit_dir(
        tmp_path / "exit", f"exit_build_main_c_{_exit_stamp(today - timedelta(days=2))}"
    )
    weird_dir = _make_exit_dir(tmp_path / "exit", "exit_weird_name")

    DebugRecorder(str(tmp_path), exit_retention_days=7)

    assert not old_dir.exists()
    assert recent_dir.exists()
    assert weird_dir.exists()


def test_keep_last_prunes_beyond_newest_n(tmp_path) -> None:
    """exit_keep_last 只保留时间戳最新的 N 个；天数关闭时不按日期删。"""
    oldest = _make_exit_dir(tmp_path / "exit", "exit_a_20260101_120000")
    middle = _make_exit_dir(tmp_path / "exit", "exit_b_20260102_120000")
    newest = _make_exit_dir(tmp_path / "exit", "exit_c_20260103_120000")

    DebugRecorder(str(tmp_path), exit_keep_last=2, exit_retention_days=0)

    assert not oldest.exists()
    assert middle.exists()
    assert newest.exists()


def test_prune_reruns_before_each_new_exit_dir(tmp_path) -> None:
    """每次新建 exit 目录后重跑保留清理，长会话多次退出也不超限。"""
    recorder = DebugRecorder(str(tmp_path), exit_keep_last=1, exit_retention_days=0)

    first = recorder.save_exit_frames(
        [(1, time.time(), np.zeros((4, 4, 3), dtype=np.uint8))], state="a"
    )
    second = recorder.save_exit_frames(
        [(2, time.time(), np.zeros((4, 4, 3), dtype=np.uint8))], state="b"
    )

    assert not first.exists()
    assert second.exists()
    assert len(list((tmp_path / "exit").iterdir())) == 1
