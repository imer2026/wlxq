"""棋盘英雄格人工标签扫描与按整局划分。"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

_ROUND_RE = re.compile(r"^\d{12}$")
_STAR_RE = re.compile(r"^star([1-4])$")
_CROP_RE = re.compile(r"^(?P<round>\d{12})_frame\d{6}_[1-6][ABC]\.png$")


@dataclass(frozen=True)
class LabeledCellSample:
    """一张已人工分类的英雄格图片。"""

    path: Path
    class_name: str
    round_key: str
    sample_kind: str


def discover_labeled_samples(root: Path) -> list[LabeledCellSample]:
    """扫描集中 split 或旧式单局 labeled，读取训练样本并忽略 unknown。"""
    root = Path(root).resolve()
    dataset_manifest = root / "dataset_manifest.csv"
    if dataset_manifest.is_file():
        return _samples_from_manifest(dataset_manifest, dataset_root=root.parent)
    samples: list[LabeledCellSample] = []
    centralized = root / "labeled"
    if centralized.is_dir():
        return _samples_in_labeled(centralized, round_key=None)
    for round_dir in sorted(path for path in root.rglob("????????????") if path.is_dir()):
        if _ROUND_RE.fullmatch(round_dir.name) is None:
            continue
        labeled = round_dir / "labeled"
        if not labeled.is_dir():
            continue
        samples.extend(
            _samples_in_labeled(labeled, round_key=round_dir.relative_to(root).as_posix())
        )
    return samples


def _samples_from_manifest(path: Path, *, dataset_root: Path) -> list[LabeledCellSample]:
    samples: list[LabeledCellSample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            class_name = row.get("class_name", "")
            if class_name == "unknown":
                continue
            round_id = row.get("round_id", "")
            image_path = row.get("image_path", "")
            sample_kind = row.get("sample_kind", "")
            if not class_name or _ROUND_RE.fullmatch(round_id) is None or not image_path:
                raise ValueError(f"数据清单包含无效训练记录: {path}")
            image = dataset_root / Path(image_path)
            if not image.is_file():
                raise FileNotFoundError(f"数据清单中的图片不存在，请重新 sync-labels: {image}")
            samples.append(LabeledCellSample(image, class_name, round_id, sample_kind))
    return samples


def _samples_in_labeled(labeled: Path, *, round_key: str | None) -> list[LabeledCellSample]:
    samples: list[LabeledCellSample] = []

    def append(path: Path, class_name: str, sample_kind: str) -> None:
        resolved_round = round_key
        if resolved_round is None:
            match = _CROP_RE.fullmatch(path.name)
            if match is None:
                raise ValueError(f"集中标签图片文件名不符合约定: {path}")
            resolved_round = match.group("round")
        samples.append(LabeledCellSample(path, class_name, resolved_round, sample_kind))

    for category in ("empty", "unavailable"):
        for kind in ("plain", "effect"):
            for path in sorted((labeled / category / kind).glob("*.png")):
                append(path, category, f"{category}_{kind}")
    for hero_dir in sorted(path for path in labeled.iterdir() if path.is_dir()):
        if hero_dir.name in {"empty", "unknown", "unavailable"}:
            continue
        for star_dir in sorted(path for path in hero_dir.iterdir() if path.is_dir()):
            match = _STAR_RE.fullmatch(star_dir.name)
            if match is None:
                continue
            for path in sorted(star_dir.glob("*.png")):
                append(path, f"{hero_dir.name}_star{match.group(1)}", "hero")
    return samples


def class_names_for_samples(samples: list[LabeledCellSample]) -> list[str]:
    """返回稳定类别顺序：empty 在前，其余按名称排序。"""
    names = {sample.class_name for sample in samples}
    return (["empty"] if "empty" in names else []) + sorted(names - {"empty"})


def resolve_round_keys(samples: list[LabeledCellSample], requested: list[str]) -> set[str]:
    """解析完整 round_key 或在无歧义时解析12位对局时间戳。"""
    available = {sample.round_key for sample in samples}
    resolved: set[str] = set()
    for value in requested:
        value = value.strip().replace("\\", "/")
        if not value:
            continue
        if value in available:
            resolved.add(value)
            continue
        matches = {key for key in available if key.rsplit("/", 1)[-1] == value}
        if not matches:
            raise ValueError(f"数据集中不存在局目录: {value}")
        if len(matches) > 1:
            choices = ", ".join(sorted(matches))
            raise ValueError(
                f"对局时间戳 {value} 在多个目录中存在，请使用完整 round_key: {choices}"
            )
        resolved.update(matches)
    return resolved


def samples_for_rounds(
    samples: list[LabeledCellSample],
    round_keys: set[str],
) -> list[LabeledCellSample]:
    """按完整局目录筛选样本。"""
    return [sample for sample in samples if sample.round_key in round_keys]
