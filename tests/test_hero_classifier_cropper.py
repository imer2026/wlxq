"""hero-classifier 离线 12 格裁剪测试。"""

from __future__ import annotations

import csv
import re

import cv2
import numpy as np

from wlxq_bot.config import BoardGridParams
from wlxq_bot.hero_classifier.cropper import HeroCellCropper
from wlxq_bot.models import CoopRole


def _board_params() -> dict[str, BoardGridParams]:
    return {
        "helper": BoardGridParams(
            anchor_x_ratio=0.55,
            anchor_y_ratio=0.50,
            col_step_ratio=0.10,
            row_step_ratio=0.07,
            cell_width_ratio=0.08,
            cell_height_ratio=0.05,
        )
    }


def _write_png(path, image) -> None:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(str(path))


def test_crops_twelve_cells_and_uses_simple_labels(tmp_path) -> None:
    round_dir = tmp_path / "202608111914"
    raw_dir = round_dir / "raw"
    raw_dir.mkdir(parents=True)
    frame = np.zeros((1000, 500, 3), dtype=np.uint8)
    frame[:, :, 1] = 123
    _write_png(raw_dir / "202608111914_frame000001.png", frame)
    (round_dir / "capture_manifest.csv").write_text(
        "frame_index,status,main_c\n1,saved,assault\n",
        encoding="utf-8-sig",
    )

    stats = HeroCellCropper(
        round_dir=round_dir,
        role=CoopRole.HELPER,
        board_params=_board_params(),
        workers=1,
    ).crop_all()

    assert stats.source_images == 1
    assert stats.expected_crops == 12
    assert stats.written_crops == 12
    names = {path.name for path in (round_dir / "unclassified").rglob("*.png")}
    assert "202608111914_frame000001_1A.png" in names
    assert "202608111914_frame000001_4B.png" in names
    assert "202608111914_frame000001_6B.png" in names
    assert not any(name.endswith("_6A.png") or name.endswith("_6C.png") for name in names)
    # 按格子分子目录
    assert (round_dir / "unclassified" / "4B").is_dir()
    assert (round_dir / "unclassified" / "1A").is_dir()
    with stats.manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 12
    assert {row["main_c"] for row in rows} == {"assault"}
    assert {row["role"] for row in rows} == {"helper"}
    assert (round_dir / "labeled" / "empty" / "plain").is_dir()
    assert (round_dir / "labeled" / "empty" / "effect").is_dir()
    assert (round_dir / "labeled" / "unavailable" / "plain").is_dir()
    assert (round_dir / "labeled" / "unavailable" / "effect").is_dir()
    assert (round_dir / "labeled" / "unknown").is_dir()
    # main_c(assault) 来自 capture_manifest，应预建 assault/star1~4
    for star in (1, 2, 3, 4):
        assert (round_dir / "labeled" / "assault" / f"star{star}").is_dir()


def test_prepare_label_dirs_uses_lineup(tmp_path) -> None:
    round_dir = tmp_path / "202608111914"
    raw_dir = round_dir / "raw"
    raw_dir.mkdir(parents=True)
    frame = np.zeros((1000, 500, 3), dtype=np.uint8)
    _write_png(raw_dir / "202608111914_frame000001.png", frame)
    (round_dir / "capture_manifest.csv").write_text(
        "frame_index,status,main_c\n1,saved,assault\n",
        encoding="utf-8-sig",
    )

    HeroCellCropper(
        round_dir=round_dir,
        role=CoopRole.HELPER,
        board_params=_board_params(),
        workers=1,
        lineup_others=["angel", "snow", "death_knight"],
    ).crop_all()

    # main_c(assault) + lineup_others 都应预建 star1~4
    for hero in ("assault", "angel", "snow", "death_knight"):
        for star in (1, 2, 3, 4):
            assert (round_dir / "labeled" / hero / f"star{star}").is_dir()
    # 未在 lineup 的英雄不应建目录
    assert not (round_dir / "labeled" / "monkey").exists()
    assert not (round_dir / "labeled" / "fox").exists()


def test_prepare_label_dirs_main_c_fallback(tmp_path) -> None:
    round_dir = tmp_path / "202608111914"
    raw_dir = round_dir / "raw"
    raw_dir.mkdir(parents=True)
    frame = np.zeros((1000, 500, 3), dtype=np.uint8)
    _write_png(raw_dir / "202608111914_frame000001.png", frame)
    # 无 capture_manifest.csv（采集被中断的典型情况）

    HeroCellCropper(
        round_dir=round_dir,
        role=CoopRole.HELPER,
        board_params=_board_params(),
        workers=1,
        main_c="assault",
    ).crop_all()

    # 兜底 main_c=assault 应预建 assault/star1~4
    for star in (1, 2, 3, 4):
        assert (round_dir / "labeled" / "assault" / f"star{star}").is_dir()
    # 没传 lineup_others，不应建其他英雄
    assert not (round_dir / "labeled" / "angel").exists()


def test_organize_pools_into_cluster_subdirs(tmp_path) -> None:
    round_dir = tmp_path / "202608111914"
    raw_dir = round_dir / "raw"
    raw_dir.mkdir(parents=True)
    same = np.zeros((1000, 500, 3), dtype=np.uint8)
    same[:, :, 1] = 50
    other = np.full((1000, 500, 3), 200, dtype=np.uint8)
    _write_png(raw_dir / "202608111914_frame000001.png", same)
    _write_png(raw_dir / "202608111914_frame000002.png", same)  # 与 1 相同
    _write_png(raw_dir / "202608111914_frame000003.png", other)  # 不同
    (round_dir / "capture_manifest.csv").write_text(
        "frame_index,status,main_c\n1,saved,assault\n2,saved,assault\n3,saved,assault\n",
        encoding="utf-8-sig",
    )

    stats = HeroCellCropper(
        round_dir=round_dir,
        role=CoopRole.HELPER,
        board_params=_board_params(),
        workers=1,
        organize=True,
        group_threshold=5.0,
    ).crop_all()

    # 跨格归类：整局只有两种视觉状态（same / other），应聚成 2 个簇
    assert stats.distinct_groups == 2
    unclassified = round_dir / "unclassified"
    entries = list(unclassified.iterdir())
    assert entries  # 非空
    # 全部是簇目录（NN_x<size>_c<cells>），没有散装 PNG，也没有遗留的按格目录
    cluster_pattern = re.compile(r"\d{3}_x\d{4}_c\d{2}")
    assert all(p.is_dir() for p in entries)
    assert not any(p.suffix == ".png" for p in entries)
    assert all(cluster_pattern.fullmatch(p.name) for p in entries)
    assert not (unclassified / "4B").exists()  # 按格目录已被清理
    # 簇里确实汇总了全部裁剪图（3 帧 × 12 格 = 36）
    assert sum(len(list(p.glob("*.png"))) for p in entries) == 36
    # manifest 的 crop_path 应已指向簇目录
    with stats.manifest_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    crop_paths_4b = [row["crop_path"] for row in rows if row["cell_label"] == "4B"]
    assert crop_paths_4b and all(cluster_pattern.search(path) for path in crop_paths_4b)
