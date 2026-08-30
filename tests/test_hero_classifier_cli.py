"""hero-classifier 命令组注册测试。"""

from __future__ import annotations

from datetime import datetime

import pytest
from typer.testing import CliRunner

from wlxq_bot.cli import app
from wlxq_bot.hero_classifier.cli import (
    _dataset_round_dir,
    _main_c_from_dataset_root,
    _normalize_main_c,
    _resolve_round_id,
)
from wlxq_bot.models import CoopRole


def test_hero_classifier_group_lists_specific_commands() -> None:
    result = CliRunner().invoke(app, ["hero-classifier", "--help"])

    assert result.exit_code == 0, result.output
    for command in (
        "collect",
        "crop",
        "import-rounds",
        "sync-labels",
        "select-candidates",
        "suggest-labels",
        "train",
        "evaluate",
    ):
        assert command in result.output


def test_collect_requires_main_c() -> None:
    result = CliRunner().invoke(app, ["hero-classifier", "collect"])

    assert result.exit_code != 0
    assert "--main-c" in result.output


def test_round_id_defaults_to_minute_timestamp() -> None:
    assert _resolve_round_id(None, now=datetime(2026, 8, 11, 19, 14, 59)) == "202608111914"


def test_round_id_and_main_c_validation() -> None:
    assert _resolve_round_id("202608111914") == "202608111914"
    assert _normalize_main_c(" Assault ") == "assault"
    with pytest.raises(ValueError, match="YYYYMMDDHHMM"):
        _resolve_round_id("202608119999")
    with pytest.raises(ValueError, match="小写英文标识"):
        _normalize_main_c("强袭")


def test_collect_reports_invalid_round_id_without_traceback() -> None:
    result = CliRunner().invoke(
        app,
        [
            "hero-classifier",
            "collect",
            "--main-c",
            "assault",
            "--round-id",
            "202608119999",
        ],
    )

    assert result.exit_code == 2
    assert "采集参数无效" in result.output


def test_collect_round_directory_contains_rounds_level(tmp_path) -> None:
    assert (
        _dataset_round_dir(
            tmp_path,
            main_c="assault",
            display_resolution=(3000, 2000),
            role=CoopRole.HELPER,
            round_id="202608111914",
        )
        == tmp_path / "assault" / "3000x2000" / "helper" / "rounds" / "202608111914"
    )


def test_training_main_c_is_inferred_from_standard_dataset_root(tmp_path) -> None:
    root = tmp_path / "assault" / "3000x2000" / "helper"

    assert _main_c_from_dataset_root(root) == "assault"
    with pytest.raises(ValueError, match="--main-c"):
        _main_c_from_dataset_root(tmp_path / "custom")


def test_import_rounds_reports_skipped_rounds(monkeypatch, tmp_path) -> None:
    from wlxq_bot.hero_classifier.dataset import ImportStats, SkippedRound

    monkeypatch.setattr(
        "wlxq_bot.hero_classifier.dataset.import_rounds",
        lambda **_kwargs: ImportStats(
            split="train",
            import_id="second",
            rounds=("202608112021",),
            skipped_rounds=(SkippedRound("202608111914", "train", "first"),),
            source_images=1,
            written_crops=12,
            distinct_groups=1,
            candidate_groups=1,
            candidate_images=5,
            import_dir=tmp_path / "train" / "imports" / "second",
            manifest_path=tmp_path / "train" / "imports" / "second" / "manifest.csv",
            candidate_manifest_path=tmp_path
            / "train"
            / "imports"
            / "second"
            / "candidate_manifest.csv",
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "hero-classifier",
            "import-rounds",
            str(tmp_path),
            "--split",
            "train",
            "--rounds",
            "202608111914,202608112021",
            "--import-id",
            "second",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "跳过: 202608111914" in result.output
    assert "train/first" in result.output
    assert "新处理局/跳过局: 1/1" in result.output


def test_import_rounds_all_skipped_does_not_report_success(monkeypatch, tmp_path) -> None:
    from wlxq_bot.hero_classifier.dataset import ImportStats, SkippedRound

    import_dir = tmp_path / "train" / "imports" / "second"
    monkeypatch.setattr(
        "wlxq_bot.hero_classifier.dataset.import_rounds",
        lambda **_kwargs: ImportStats(
            split="train",
            import_id="second",
            rounds=(),
            skipped_rounds=(SkippedRound("202608111914", "train", "first"),),
            source_images=0,
            written_crops=0,
            distinct_groups=0,
            candidate_groups=0,
            candidate_images=0,
            import_dir=import_dir,
            manifest_path=import_dir / "manifest.csv",
            candidate_manifest_path=import_dir / "candidate_manifest.csv",
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "hero-classifier",
            "import-rounds",
            str(tmp_path),
            "--split",
            "train",
            "--rounds",
            "202608111914",
            "--import-id",
            "second",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "没有需要处理的新对局" in result.output
    assert "跳过: 202608111914" in result.output
    assert "多局素材导入完成" not in result.output
