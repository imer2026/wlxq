"""英雄格数据集多局 import 和标签清单同步测试。"""

from __future__ import annotations

import csv

import cv2
import numpy as np
import pytest

from wlxq_bot.config import BoardGridParams
from wlxq_bot.hero_classifier.dataset import (
    import_rounds,
    select_import_candidates,
    sync_labels,
)
from wlxq_bot.hero_classifier.labels import discover_labeled_samples
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


def _make_round(root, round_id: str, value: int) -> None:
    round_dir = root / "rounds" / round_id
    raw_dir = round_dir / "raw"
    raw_dir.mkdir(parents=True)
    image = np.full((1000, 500, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(str(raw_dir / f"{round_id}_frame000001.png"))
    (round_dir / "capture_manifest.csv").write_text(
        "frame_index,status,main_c,role\n1,saved,assault,helper\n",
        encoding="utf-8-sig",
    )


def _import(root, *, split: str, import_id: str, rounds: list[str]):
    return import_rounds(
        dataset_root=root,
        split=split,
        import_id=import_id,
        round_ids=rounds,
        role=CoopRole.HELPER,
        board_params=_board_params(),
        workers=1,
        group_threshold=5.0,
    )


def test_import_pools_multiple_rounds_without_touching_existing_labels(tmp_path) -> None:
    root = tmp_path / "helper"
    _make_round(root, "202608111914", 20)
    _make_round(root, "202608112021", 200)
    existing = root / "train" / "labeled" / "assault" / "star1" / "keep.png"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"keep")

    stats = _import(
        root,
        split="train",
        import_id="20260812_001",
        rounds=["202608111914", "202608112021"],
    )

    assert stats.source_images == 2
    assert stats.written_crops == 24
    assert stats.distinct_groups == 2
    assert stats.candidate_groups == 2
    assert stats.candidate_images == 20
    assert existing.read_bytes() == b"keep"
    assert (stats.import_dir / "rounds.txt").read_text(encoding="utf-8").splitlines() == [
        "202608111914",
        "202608112021",
    ]
    assert len(list((stats.import_dir / "unclassified").rglob("*.png"))) == 24
    assert len(list((stats.import_dir / "candidates").rglob("*.png"))) == 20
    assert stats.candidate_manifest_path.is_file()
    with stats.candidate_manifest_path.open(encoding="utf-8-sig", newline="") as file:
        candidate_rows = list(csv.DictReader(file))
    assert len(candidate_rows) == 20
    assert all((root / row["source_path"]).is_file() for row in candidate_rows)
    assert all((root / row["candidate_path"]).is_file() for row in candidate_rows)
    with stats.manifest_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert {row["round_id"] for row in rows} == {"202608111914", "202608112021"}
    assert all(row["crop_path"].startswith("train/imports/20260812_001/") for row in rows)
    assert all((root / row["crop_path"]).is_file() for row in rows)


def test_import_skips_processed_round_and_processes_remaining_rounds(tmp_path) -> None:
    root = tmp_path / "helper"
    _make_round(root, "202608111914", 20)
    _make_round(root, "202608112021", 200)
    _import(root, split="train", import_id="first", rounds=["202608111914"])

    stats = _import(
        root,
        split="test",
        import_id="second",
        rounds=["202608111914", "202608112021"],
    )

    assert stats.rounds == ("202608112021",)
    assert [item.round_id for item in stats.skipped_rounds] == ["202608111914"]
    assert stats.skipped_rounds[0].split == "train"
    assert stats.skipped_rounds[0].import_id == "first"
    assert stats.written_crops == 12
    assert stats.candidate_images == 10
    assert (stats.import_dir / "rounds.txt").read_text(encoding="utf-8").splitlines() == [
        "202608112021"
    ]


def test_import_with_only_processed_rounds_does_not_create_empty_import(tmp_path) -> None:
    root = tmp_path / "helper"
    _make_round(root, "202608111914", 20)
    first = _import(root, split="train", import_id="first", rounds=["202608111914"])

    stats = _import(root, split="test", import_id="second", rounds=["202608111914"])

    assert stats.rounds == ()
    assert [item.round_id for item in stats.skipped_rounds] == ["202608111914"]
    assert not stats.import_dir.exists()
    assert first.import_dir.is_dir()


def test_sync_labels_rebuilds_manifest_and_records_unknown(tmp_path) -> None:
    root = tmp_path / "helper"
    _make_round(root, "202608111914", 20)
    stats = _import(root, split="train", import_id="first", rounds=["202608111914"])
    images = sorted((stats.import_dir / "unclassified").rglob("*.png"))
    hero = root / "train" / "labeled" / "assault" / "star1" / images[0].name
    unknown = root / "train" / "labeled" / "unknown" / images[1].name
    hero.parent.mkdir(parents=True, exist_ok=True)
    unknown.parent.mkdir(parents=True, exist_ok=True)
    images[0].replace(hero)
    images[1].replace(unknown)

    first = sync_labels(dataset_root=root, split="train")
    assert first.samples == 2
    assert first.unknown_samples == 1
    with first.manifest_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert {row["class_name"] for row in rows} == {"assault_star1", "unknown"}
    samples = discover_labeled_samples(root / "train")
    assert [(sample.class_name, sample.round_key) for sample in samples] == [
        ("assault_star1", "202608111914")
    ]

    unknown.unlink()
    second = sync_labels(dataset_root=root, split="train")
    assert second.samples == 1
    with second.manifest_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["class_name"] == "assault_star1"


def test_sync_labels_rejects_unregistered_and_duplicate_sources(tmp_path) -> None:
    root = tmp_path / "helper"
    (root / "rounds" / "202608111914").mkdir(parents=True)
    name = "202608111914_frame000001_4B.png"
    unregistered = root / "train" / "labeled" / "unknown" / name
    unregistered.parent.mkdir(parents=True)
    unregistered.write_bytes(b"x")
    with pytest.raises(ValueError, match="尚未登记"):
        sync_labels(dataset_root=root, split="train")

    (root / "train" / "imports" / "first").mkdir(parents=True)
    (root / "train" / "imports" / "first" / "rounds.txt").write_text(
        "202608111914\n", encoding="utf-8"
    )
    duplicate = root / "train" / "labeled" / "empty" / "plain" / name
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(b"x")
    with pytest.raises(ValueError, match="重复标注"):
        sync_labels(dataset_root=root, split="train")


def test_sync_labels_rejects_wrong_filename(tmp_path) -> None:
    root = tmp_path / "helper"
    bad = root / "train" / "labeled" / "unknown" / "renamed.png"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match="文件名不符合约定"):
        sync_labels(dataset_root=root, split="train")


def test_select_import_candidates_supports_existing_import(tmp_path) -> None:
    import_dir = tmp_path / "helper" / "train" / "imports" / "old"
    cluster = import_dir / "unclassified" / "000_x0002_c01"
    cluster.mkdir(parents=True)
    for index, value in enumerate((10, 200), start=1):
        path = cluster / f"202608111914_frame{index:06d}_4B.png"
        ok, encoded = cv2.imencode(".png", np.full((40, 40, 3), value, dtype=np.uint8))
        assert ok
        encoded.tofile(str(path))

    stats = select_import_candidates(import_dir=import_dir)

    assert stats.candidate_groups == 1
    assert stats.candidate_images == 2
    assert len(list((import_dir / "candidates").rglob("*.png"))) == 2
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        select_import_candidates(import_dir=import_dir)
