"""训练后使用模型对 candidates 做离线预分类。"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from wlxq_bot.hero_classifier.suggester import suggest_import_labels
from wlxq_bot.perception.hero_classifier import HeroCellPrediction


class _FakeClassifier:
    confidence_threshold = 0.8
    margin_threshold = 0.2

    def __init__(self, predictions: list[HeroCellPrediction]) -> None:
        self.predictions = predictions
        self.offset = 0

    def predict(self, images: list[np.ndarray]) -> list[HeroCellPrediction]:
        result = self.predictions[self.offset : self.offset + len(images)]
        self.offset += len(images)
        return result


def _prediction(
    raw_class: str,
    *,
    rejected: bool = False,
    reason: str = "",
) -> HeroCellPrediction:
    return HeroCellPrediction(
        class_name="unknown" if rejected else raw_class,
        hero_type=None,
        star_level=None,
        confidence=0.6 if rejected else 0.95,
        margin=0.1 if rejected else 0.8,
        rejected=rejected,
        raw_class_name=raw_class,
        second_class_name="empty",
        second_confidence=0.05,
        rejection_reason=reason,
    )


def _write_png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", np.full((20, 20, 3), value, dtype=np.uint8))
    assert ok
    encoded.tofile(str(path))


def _prepare_import(tmp_path: Path) -> Path:
    import_dir = tmp_path / "train" / "imports" / "batch001"
    candidates = import_dir / "candidates"
    specs = (
        ("000_x0100_c01", "000_x0002", "202608111914", 1),
        ("000_x0100_c01", "000_x0002", "202608111914", 2),
        ("001_x0050_c01", "000_x0002", "202608111915", 3),
        ("001_x0050_c01", "000_x0002", "202608111915", 4),
        ("002_x0030_c01", "000_x0001", "202608111916", 5),
    )
    rows = []
    for order, (primary, secondary, round_id, frame) in enumerate(specs, start=1):
        path = candidates / primary / secondary / f"{round_id}_frame{frame:06d}_4B.png"
        _write_png(path, order)
        rows.append(
            {
                "primary_cluster": primary,
                "secondary_cluster": secondary,
                "source_group_size": "10",
                "selection_order": str(order),
                "round_id": round_id,
                "frame_index": str(frame),
                "cell_label": "4B",
                "source_path": (f"train/imports/batch001/unclassified/{primary}/{path.name}"),
                "candidate_path": path.relative_to(tmp_path).as_posix(),
            }
        )
    with (import_dir / "candidate_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return import_dir


def test_suggest_labels_routes_consistent_mixed_and_rejected_groups(tmp_path) -> None:
    import_dir = _prepare_import(tmp_path)
    classifier = _FakeClassifier(
        [
            _prediction("assault_star2"),
            _prediction("assault_star2"),
            _prediction("angel_star1"),
            _prediction("empty"),
            _prediction("unavailable", rejected=True, reason="low_confidence"),
        ]
    )

    stats = suggest_import_labels(
        import_dir=import_dir,
        model_path=tmp_path / "unused.onnx",
        classifier=classifier,
        batch_size=2,
    )

    assert stats.groups == 3
    assert stats.images == 5
    assert stats.suggested_groups == 1
    assert stats.mixed_groups == 1
    assert stats.low_confidence_groups == 1
    assert len(list((stats.suggested_dir / "assault_star2").rglob("*.png"))) == 2
    assert len(list((stats.suggested_dir / "review" / "mixed_group").rglob("*.png"))) == 2
    assert len(list((stats.suggested_dir / "review" / "low_confidence").rglob("*.png"))) == 1
    assert len(list((import_dir / "candidates").rglob("*.png"))) == 5
    with stats.manifest_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 5
    assert {row["group_decision"] for row in rows} == {
        "suggested",
        "mixed_group",
        "low_confidence",
    }
    assert all(
        row["suggested_path"].startswith("train/imports/batch001/suggested/") for row in rows
    )


def test_suggest_labels_refuses_to_overwrite_existing_output(tmp_path) -> None:
    import_dir = _prepare_import(tmp_path)
    (import_dir / "suggested").mkdir()

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        suggest_import_labels(
            import_dir=import_dir,
            model_path=tmp_path / "unused.onnx",
            classifier=_FakeClassifier([]),
        )


def test_suggest_labels_rejects_manifest_not_matching_candidates(tmp_path) -> None:
    import_dir = _prepare_import(tmp_path)
    _write_png(
        import_dir / "candidates" / "extra" / "000_x0001" / "202608111917_frame000001_4B.png",
        9,
    )

    with pytest.raises(ValueError, match="不一致"):
        suggest_import_labels(
            import_dir=import_dir,
            model_path=tmp_path / "unused.onnx",
            classifier=_FakeClassifier([]),
        )
