"""hero-classifier 裁剪图聚类分组测试（按格 group_cell / 跨格 group_files）。"""

from __future__ import annotations

import cv2
import numpy as np

from wlxq_bot.hero_classifier.grouper import (
    frame_index,
    generate_candidates,
    group_cell,
    group_files,
    select_diverse_files,
)


def _write_png(path, image) -> None:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(str(path))


def test_group_cell_groups_identical_frames(tmp_path) -> None:
    cell = tmp_path / "4B"
    cell.mkdir()
    same = np.zeros((64, 68, 3), dtype=np.uint8)
    same[:, :, 1] = 123
    other = np.full((64, 68, 3), 200, dtype=np.uint8)
    _write_png(cell / "r_frame000001_4B.png", same)
    _write_png(cell / "r_frame000002_4B.png", same)  # 与 1 完全相同
    _write_png(cell / "r_frame000005_4B.png", other)  # 不同画面
    _write_png(cell / "r_frame000009_4B.png", same)  # 与 1 完全相同

    groups = group_cell(cell)

    assert len(groups) == 2  # 只有两种不同画面
    assert [len(members) for _, members in groups] == [3, 1]
    # 按首次出现顺序：第一组代表是 frame1，第二组代表是 frame5
    assert frame_index(groups[0][0]) == 1
    assert frame_index(groups[1][0]) == 5


def test_group_cell_empty_dir(tmp_path) -> None:
    cell = tmp_path / "1A"
    cell.mkdir()
    assert group_cell(cell) == []


def test_group_files_pools_across_cells(tmp_path) -> None:
    # 跨格池化：内容一致但分属不同格子的裁剪图应并入同一簇，不同内容分开
    same = np.zeros((64, 68, 3), dtype=np.uint8)
    same[:, :, 1] = 123
    other = np.full((64, 68, 3), dtype=np.uint8, fill_value=200)
    pool = tmp_path / "pool"
    pool.mkdir()
    _write_png(pool / "r_frame000001_1A.png", same)
    _write_png(pool / "r_frame000002_4B.png", same)  # 跨格同状态
    _write_png(pool / "r_frame000003_2A.png", other)  # 不同状态

    groups = group_files(sorted(pool.glob("*.png")))

    assert len(groups) == 2  # 跨格的同内容合并成一簇
    assert sorted(len(members) for _, members in groups) == [1, 2]


def test_select_diverse_files_keeps_visual_extremes(tmp_path) -> None:
    files = []
    for index, value in enumerate((0, 1, 2, 200), start=1):
        path = tmp_path / f"202608111914_frame{index:06d}_4B.png"
        image = np.full((40, 40, 3), value, dtype=np.uint8)
        _write_png(path, image)
        files.append(path)

    selected = select_diverse_files(files, max_count=2)

    assert selected == [files[0], files[3]]


def test_generate_candidates_splits_only_large_primary_clusters(tmp_path) -> None:
    unclassified = tmp_path / "unclassified"
    small = unclassified / "000_x0002_c01"
    large = unclassified / "001_x0004_c01"
    small.mkdir(parents=True)
    large.mkdir(parents=True)
    for directory, values, round_id in (
        (small, (10, 200), "202608111914"),
        (large, (0, 1, 200, 201), "202608112021"),
    ):
        for index, value in enumerate(values, start=1):
            path = directory / f"{round_id}_frame{index:06d}_4B.png"
            _write_png(path, np.full((40, 40, 3), value, dtype=np.uint8))

    stats = generate_candidates(
        unclassified,
        tmp_path / "candidates",
        path_root=tmp_path,
        secondary_trigger=3,
        secondary_threshold=15.0,
        max_per_group=2,
    )

    # 小簇不二分，直接产生一个候选组；大簇按 15 分成暗/亮两个候选组。
    assert stats.primary_clusters == 2
    assert stats.candidate_groups == 3
    assert stats.candidate_images == 6
    assert len(list((tmp_path / "candidates" / small.name).glob("*"))) == 1
    assert len(list((tmp_path / "candidates" / large.name).glob("*"))) == 2
    # candidates 是副本，unclassified 原图完整保留。
    assert len(list(unclassified.rglob("*.png"))) == 6
    assert stats.manifest_path.is_file()
