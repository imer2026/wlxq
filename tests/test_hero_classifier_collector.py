"""hero-classifier 完整客户区异步采集测试。"""

from __future__ import annotations

import csv

import cv2
import numpy as np
import pytest

from wlxq_bot.hero_classifier.collector import HeroFrameCollector
from wlxq_bot.models import CoopRole, WindowContext


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def _context(frame_id: int) -> WindowContext:
    return WindowContext(
        window_handle=1,
        client_rect_screen=(10, 20, 120, 240),
        client_size=(120, 240),
        dpi=96,
        monitor_id="DISPLAY1",
        is_foreground=True,
        is_minimized=False,
        captured_at=float(frame_id),
        frame_id=frame_id,
    )


def test_collects_at_fixed_slots_and_writes_manifest(tmp_path) -> None:
    clock = _Clock()
    counter = 0

    def capture():
        nonlocal counter
        counter += 1
        return _context(counter), np.full((240, 120, 3), counter, dtype=np.uint8)

    round_dir = tmp_path / "202608111914"
    stats = HeroFrameCollector(
        capture,
        round_dir=round_dir,
        round_id="202608111914",
        main_c="assault",
        role=CoopRole.HELPER,
        display_resolution=(3000, 2000),
        interval_seconds=1.0,
        duration_seconds=3.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ).collect()

    assert stats.expected == 3
    assert stats.captured == 3
    assert stats.saved == 3
    assert stats.failed == 0
    paths = sorted((round_dir / "raw").glob("*.png"))
    assert [path.name for path in paths] == [
        "202608111914_frame000001.png",
        "202608111914_frame000002.png",
        "202608111914_frame000003.png",
    ]
    decoded = cv2.imdecode(np.fromfile(str(paths[1]), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert int(decoded[0, 0, 0]) == 2
    with stats.manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["status"] for row in rows] == ["saved", "saved", "saved"]
    assert {row["main_c"] for row in rows} == {"assault"}
    assert rows[0]["display_resolution"] == "3000x2000"


def test_refuses_to_overwrite_existing_round(tmp_path) -> None:
    round_dir = tmp_path / "202608111914"
    raw_dir = round_dir / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "existing.png").write_bytes(b"x")

    collector = HeroFrameCollector(
        lambda: (_context(1), np.zeros((240, 120, 3), dtype=np.uint8)),
        round_dir=round_dir,
        round_id="202608111914",
        main_c="assault",
        role=CoopRole.HELPER,
        display_resolution=(3000, 2000),
        duration_seconds=1.0,
    )

    try:
        collector.collect()
    except FileExistsError:
        pass
    else:
        raise AssertionError("已有数据时应拒绝覆盖")


def test_rejects_invalid_timestamp_round_id(tmp_path) -> None:
    with pytest.raises(ValueError, match="有效的 YYYYMMDDHHMM"):
        HeroFrameCollector(
            lambda: (_context(1), np.zeros((240, 120, 3), dtype=np.uint8)),
            round_dir=tmp_path / "202602301200",
            round_id="202602301200",
            main_c="assault",
            role=CoopRole.HELPER,
            display_resolution=(3000, 2000),
        )
