"""英雄格人工标签扫描和按整局隔离测试。"""

from __future__ import annotations

import pytest

from wlxq_bot.hero_classifier.labels import (
    LabeledCellSample,
    class_names_for_samples,
    discover_labeled_samples,
    resolve_round_keys,
)
from wlxq_bot.hero_classifier.trainer import (
    TrainingConfig,
    balanced_sample_weights,
    class_sampling_weights,
)


def test_discovers_empty_unavailable_and_hero_labels_but_ignores_unknown(tmp_path) -> None:
    round_dir = tmp_path / "assault" / "3000x2000" / "helper" / "202608111914" / "labeled"
    paths = [
        round_dir / "empty" / "plain" / "a.png",
        round_dir / "empty" / "effect" / "b.png",
        round_dir / "unavailable" / "plain" / "e.png",
        round_dir / "unavailable" / "effect" / "f.png",
        round_dir / "assault" / "star2" / "c.png",
        round_dir / "unknown" / "d.png",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")

    samples = discover_labeled_samples(tmp_path)

    assert len(samples) == 5
    assert class_names_for_samples(samples) == ["empty", "assault_star2", "unavailable"]
    assert {sample.sample_kind for sample in samples} == {
        "empty_plain",
        "empty_effect",
        "unavailable_plain",
        "unavailable_effect",
        "hero",
    }
    assert resolve_round_keys(samples, ["202608111914"]) == {
        "assault/3000x2000/helper/202608111914"
    }


def test_balanced_weights_equalize_class_round_and_empty_kind(tmp_path) -> None:
    samples = []
    specs = [
        ("assault_star1", "202608111914", "hero", 100),
        ("assault_star1", "202608112020", "hero", 10),
        ("monkey_star2", "202608111914", "hero", 5),
        ("empty", "202608111914", "empty_plain", 80),
        ("empty", "202608111914", "empty_effect", 8),
    ]
    for class_name, round_key, sample_kind, count in specs:
        for index in range(count):
            samples.append(
                LabeledCellSample(
                    path=tmp_path / f"{class_name}_{round_key}_{sample_kind}_{index}.png",
                    class_name=class_name,
                    round_key=round_key,
                    sample_kind=sample_kind,
                )
            )

    weights = balanced_sample_weights(samples)

    class_totals: dict[str, float] = {}
    round_totals: dict[tuple[str, str], float] = {}
    kind_totals: dict[tuple[str, str, str], float] = {}
    for sample, weight in zip(samples, weights, strict=True):
        class_totals[sample.class_name] = class_totals.get(sample.class_name, 0.0) + weight
        round_key = (sample.class_name, sample.round_key)
        round_totals[round_key] = round_totals.get(round_key, 0.0) + weight
        kind_key = (sample.class_name, sample.round_key, sample.sample_kind)
        kind_totals[kind_key] = kind_totals.get(kind_key, 0.0) + weight

    assert len({round(value, 8) for value in class_totals.values()}) == 1
    assert round_totals[("assault_star1", "202608111914")] == pytest.approx(
        round_totals[("assault_star1", "202608112020")]
    )
    assert kind_totals[("empty", "202608111914", "empty_plain")] == pytest.approx(
        kind_totals[("empty", "202608111914", "empty_effect")]
    )


def test_business_weights_prioritize_star1_star2_and_main_c(tmp_path) -> None:
    class_names = [
        "assault_star1",
        "angel_star2",
        "assault_star3",
        "angel_star3",
        "assault_star4",
        "angel_star4",
        "empty",
        "unavailable",
    ]
    config = TrainingConfig(main_c="assault")
    expected = {
        "assault_star1": 1.0,
        "angel_star2": 1.0,
        "assault_star3": 0.8,
        "angel_star3": 0.5,
        "assault_star4": 0.3,
        "angel_star4": 0.1,
        "empty": 0.5,
        "unavailable": 0.3,
    }

    class_weights = class_sampling_weights(class_names, config=config)

    assert class_weights == expected
    samples = [
        LabeledCellSample(
            path=tmp_path / f"{class_name}.png",
            class_name=class_name,
            round_key="202608111914",
            sample_kind="hero" if "star" in class_name else f"{class_name}_plain",
        )
        for class_name in class_names
    ]
    sample_weights = balanced_sample_weights(samples, class_weights=class_weights)
    normalized = {
        sample.class_name: weight / sample_weights[0]
        for sample, weight in zip(samples, sample_weights, strict=True)
    }
    assert normalized == pytest.approx(expected)


def test_missing_star4_class_does_not_require_weight_or_sample() -> None:
    class_weights = class_sampling_weights(
        ["assault_star1", "assault_star2", "empty"],
        config=TrainingConfig(main_c="assault"),
    )

    assert class_weights == {
        "assault_star1": 1.0,
        "assault_star2": 1.0,
        "empty": 0.5,
    }


def test_discovers_centralized_split_labels_by_filename_round(tmp_path) -> None:
    labeled = tmp_path / "train" / "labeled"
    paths = [
        labeled / "assault" / "star2" / "202608111914_frame000001_4B.png",
        labeled / "empty" / "plain" / "202608112021_frame000002_3A.png",
        labeled / "unknown" / "202608112021_frame000003_3B.png",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")

    samples = discover_labeled_samples(tmp_path / "train")

    assert {(sample.class_name, sample.round_key) for sample in samples} == {
        ("assault_star2", "202608111914"),
        ("empty", "202608112021"),
    }
