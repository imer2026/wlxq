"""英雄格数据集的对局导入、split 隔离和标签清单同步。"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from wlxq_bot.config import BoardGridParams
from wlxq_bot.hero_classifier.cropper import (
    CROP_MANIFEST_FIELDS,
    HeroCellCropper,
    organize_crop_files,
)
from wlxq_bot.hero_classifier.progress import ProgressLogger
from wlxq_bot.models import CoopRole
from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

DatasetSplit = Literal["train", "validation", "test"]
SPLITS: tuple[DatasetSplit, ...] = ("train", "validation", "test")

_ROUND_RE = re.compile(r"^\d{12}$")
_IMPORT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_HERO_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CROP_RE = re.compile(r"^(?P<round>\d{12})_frame(?P<frame>\d{6})_(?P<cell>[1-6][ABC])\.png$")
# helper 视角实际存在的 12 格：含 2B、无 3C（与棋盘格位模型/裁剪产物一致）
_VALID_CELL_LABELS = {
    "1A",
    "2A",
    "2B",
    "3A",
    "3B",
    "4A",
    "4B",
    "4C",
    "5A",
    "5B",
    "5C",
    "6B",
}
_DATASET_MANIFEST_FIELDS = (
    "round_id",
    "split",
    "class_name",
    "sample_kind",
    "cell_label",
    "frame_index",
    "image_path",
)


@dataclass(frozen=True)
class SkippedRound:
    """因已经完成过裁切和聚类而跳过的来源局。"""

    round_id: str
    split: DatasetSplit
    import_id: str


@dataclass(frozen=True)
class ImportStats:
    """一次多局导入的产物汇总。"""

    split: DatasetSplit
    import_id: str
    rounds: tuple[str, ...]
    skipped_rounds: tuple[SkippedRound, ...]
    source_images: int
    written_crops: int
    distinct_groups: int
    candidate_groups: int
    candidate_images: int
    import_dir: Path
    manifest_path: Path
    candidate_manifest_path: Path


@dataclass(frozen=True)
class SyncStats:
    """一次标签清单完整重建的汇总。"""

    split: DatasetSplit
    samples: int
    unknown_samples: int
    manifest_path: Path


def import_rounds(
    *,
    dataset_root: Path,
    split: DatasetSplit,
    import_id: str,
    round_ids: list[str],
    role: CoopRole,
    board_params: dict[str, BoardGridParams],
    workers: int = 4,
    png_compression: int = 1,
    lineup_others: list[str] | None = None,
    main_c: str | None = None,
    group_threshold: float = 35.0,
    candidate_split_trigger: int = 100,
    candidate_group_threshold: float = 15.0,
    candidate_max_per_group: int = 10,
) -> ImportStats:
    """跳过已处理局，把剩余局裁到同一 import，并在全部裁完后联合聚类一次。"""
    root = Path(dataset_root)
    split = validate_split(split)
    import_id = _validate_import_id(import_id)
    requested_rounds = _validate_round_ids(round_ids)
    assignments = registered_rounds(root)
    skipped_rounds: list[SkippedRound] = []
    rounds: list[str] = []
    for round_id in requested_rounds:
        assigned = assignments.get(round_id)
        if assigned is not None:
            assigned_split, assigned_import = assigned
            skipped_rounds.append(
                SkippedRound(
                    round_id=round_id,
                    split=assigned_split,
                    import_id=assigned_import,
                )
            )
            continue
        rounds.append(round_id)

    imports_dir = root / split / "imports"
    import_dir = imports_dir / import_id
    manifest_path = import_dir / "manifest.csv"
    candidate_manifest_path = import_dir / "candidate_manifest.csv"
    if not rounds:
        return ImportStats(
            split=split,
            import_id=import_id,
            rounds=(),
            skipped_rounds=tuple(skipped_rounds),
            source_images=0,
            written_crops=0,
            distinct_groups=0,
            candidate_groups=0,
            candidate_images=0,
            import_dir=import_dir,
            manifest_path=manifest_path,
            candidate_manifest_path=candidate_manifest_path,
        )

    for round_id in rounds:
        round_dir = root / "rounds" / round_id
        if not (round_dir / "raw").is_dir():
            raise FileNotFoundError(f"对局 raw 目录不存在: {round_dir / 'raw'}")

    if import_dir.exists():
        raise FileExistsError(f"import 已存在，拒绝覆盖: {import_dir}")
    working_dir = imports_dir / f".{import_id}.tmp"
    if working_dir.exists():
        raise FileExistsError(f"发现上次未完成的临时 import，请检查后处理: {working_dir}")
    unclassified = working_dir / "unclassified"
    unclassified.mkdir(parents=True)

    all_rows: list[dict[str, str]] = []
    source_images = 0
    import_progress = ProgressLogger(logger, "import 多局裁切", len(rounds))
    import_progress.start(detail=f"split={split} import={import_id}")
    for round_index, round_id in enumerate(rounds, start=1):
        logger.info(
            "import 开始裁切对局 round=%s index=%d/%d",
            round_id,
            round_index,
            len(rounds),
        )
        temporary_manifest = working_dir / f".{round_id}.csv"
        stats = HeroCellCropper(
            round_dir=root / "rounds" / round_id,
            role=role,
            board_params=board_params,
            workers=workers,
            png_compression=png_compression,
            main_c=main_c,
            organize=False,
            output_dir=unclassified,
            manifest_path=temporary_manifest,
            path_root=root,
            prepare_label_dirs=False,
        ).crop_all()
        source_images += stats.source_images
        with temporary_manifest.open("r", encoding="utf-8-sig", newline="") as file:
            all_rows.extend(csv.DictReader(file))
        temporary_manifest.unlink()
        logger.info(
            "import 对局裁切完成 round=%s index=%d/%d source_images=%d crops=%d",
            round_id,
            round_index,
            len(rounds),
            stats.source_images,
            stats.written_crops,
        )
        import_progress.update(
            round_index,
            detail=f"source_images={source_images} crops={len(all_rows)}",
        )
    import_progress.finish(
        len(rounds), detail=f"source_images={source_images} crops={len(all_rows)}"
    )

    move_map, distinct_groups = organize_crop_files(
        unclassified,
        path_root=root,
        threshold=group_threshold,
    )
    for row in all_rows:
        row["crop_path"] = move_map.get(row["crop_path"], row["crop_path"])
        row["crop_path"] = row["crop_path"].replace(
            f"/imports/.{import_id}.tmp/", f"/imports/{import_id}/"
        )
    from wlxq_bot.hero_classifier.grouper import generate_candidates

    candidate_stats = generate_candidates(
        unclassified,
        working_dir / "candidates",
        path_root=root,
        secondary_trigger=candidate_split_trigger,
        secondary_threshold=candidate_group_threshold,
        max_per_group=candidate_max_per_group,
        path_replacement=(f"/imports/.{import_id}.tmp/", f"/imports/{import_id}/"),
    )
    working_manifest_path = working_dir / "manifest.csv"
    _write_csv_atomic(working_manifest_path, CROP_MANIFEST_FIELDS, all_rows)
    (working_dir / "rounds.txt").write_text(
        "".join(f"{item}\n" for item in rounds), encoding="utf-8"
    )
    _prepare_split_labels(
        root / split / "labeled",
        heroes=_import_heroes(all_rows, lineup_others or [], main_c),
    )
    working_dir.replace(import_dir)
    manifest_path = import_dir / "manifest.csv"
    candidate_manifest_path = import_dir / "candidate_manifest.csv"
    return ImportStats(
        split=split,
        import_id=import_id,
        rounds=tuple(rounds),
        skipped_rounds=tuple(skipped_rounds),
        source_images=source_images,
        written_crops=len(all_rows),
        distinct_groups=distinct_groups,
        candidate_groups=candidate_stats.candidate_groups,
        candidate_images=candidate_stats.candidate_images,
        import_dir=import_dir,
        manifest_path=manifest_path,
        candidate_manifest_path=candidate_manifest_path,
    )


def sync_labels(*, dataset_root: Path, split: DatasetSplit) -> SyncStats:
    """扫描 split/labeled 并原子、完整重建 dataset_manifest.csv。"""
    root = Path(dataset_root)
    split = validate_split(split)
    assignments = registered_rounds(root)
    labeled_dir = root / split / "labeled"
    labeled_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    seen_sources: dict[str, Path] = {}
    unknown_samples = 0
    for path in sorted(labeled_dir.rglob("*.png")):
        match = _CROP_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"已标注图片文件名不符合约定: {path}")
        if match.group("cell") not in _VALID_CELL_LABELS:
            raise ValueError(f"已标注图片包含无效 helper 格子标签: {path}")
        round_id = match.group("round")
        assigned = assignments.get(round_id)
        if assigned is None:
            raise ValueError(f"图片来源局尚未登记到任何 split: {path.name}")
        if assigned[0] != split:
            raise ValueError(f"图片来源局 {round_id} 属于 {assigned[0]}，不能放入 {split}")
        if not (root / "rounds" / round_id).is_dir():
            raise ValueError(f"图片来源局目录不存在: {root / 'rounds' / round_id}")
        previous = seen_sources.get(path.name)
        if previous is not None:
            raise ValueError(f"同一来源图被重复标注: {previous} 和 {path}")
        seen_sources[path.name] = path
        class_name, sample_kind = _label_from_path(path.relative_to(labeled_dir))
        unknown_samples += int(class_name == "unknown")
        rows.append(
            {
                "round_id": round_id,
                "split": split,
                "class_name": class_name,
                "sample_kind": sample_kind,
                "cell_label": match.group("cell"),
                "frame_index": int(match.group("frame")),
                "image_path": path.relative_to(root).as_posix(),
            }
        )
    manifest_path = root / split / "dataset_manifest.csv"
    _write_csv_atomic(manifest_path, _DATASET_MANIFEST_FIELDS, rows)
    return SyncStats(split, len(rows), unknown_samples, manifest_path)


def select_import_candidates(
    *,
    import_dir: Path,
    secondary_trigger: int = 100,
    secondary_threshold: float = 15.0,
    max_per_group: int = 10,
):
    """为已经完成一级聚类的历史 import 补生成 candidates，拒绝覆盖已有结果。"""
    from wlxq_bot.hero_classifier.grouper import generate_candidates

    import_dir = Path(import_dir)
    unclassified = import_dir / "unclassified"
    candidates = import_dir / "candidates"
    manifest = import_dir / "candidate_manifest.csv"
    if not unclassified.is_dir():
        raise FileNotFoundError(f"import 缺少 unclassified 目录: {unclassified}")
    if candidates.exists() or manifest.exists():
        raise FileExistsError(f"import 已存在 candidates，拒绝覆盖: {import_dir}")
    try:
        imports_dir = import_dir.parent
        if imports_dir.name != "imports":
            raise ValueError
        dataset_root = imports_dir.parent.parent
        import_dir.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError("import_dir 必须是 <数据组>/<split>/imports/<import_id>") from exc
    return generate_candidates(
        unclassified,
        candidates,
        path_root=dataset_root,
        secondary_trigger=secondary_trigger,
        secondary_threshold=secondary_threshold,
        max_per_group=max_per_group,
    )


def registered_rounds(dataset_root: Path) -> dict[str, tuple[DatasetSplit, str]]:
    """读取所有 imports/*/rounds.txt，并校验每局只属于一个 import 和 split。"""
    root = Path(dataset_root)
    result: dict[str, tuple[DatasetSplit, str]] = {}
    for split in SPLITS:
        imports_dir = root / split / "imports"
        if not imports_dir.is_dir():
            continue
        for rounds_path in sorted(imports_dir.glob("*/rounds.txt")):
            import_id = rounds_path.parent.name
            for round_id in rounds_path.read_text(encoding="utf-8-sig").splitlines():
                round_id = round_id.strip()
                if not round_id:
                    continue
                if _ROUND_RE.fullmatch(round_id) is None:
                    raise ValueError(f"rounds.txt 包含无效局号: {rounds_path}: {round_id}")
                previous = result.get(round_id)
                if previous is not None:
                    raise ValueError(
                        f"对局 {round_id} 被重复登记: {previous[0]}/{previous[1]} 和 {split}/{import_id}"
                    )
                result[round_id] = (split, import_id)
    return result


def validate_split(value: str) -> DatasetSplit:
    if value not in SPLITS:
        raise ValueError(f"split 必须是 {', '.join(SPLITS)} 之一")
    return value  # type: ignore[return-value]


def _validate_import_id(value: str) -> str:
    value = value.strip()
    if _IMPORT_RE.fullmatch(value) is None:
        raise ValueError("import_id 只能包含英文、数字、下划线和连字符")
    return value


def _validate_round_ids(values: list[str]) -> list[str]:
    result = [value.strip() for value in values if value.strip()]
    if not result:
        raise ValueError("至少指定一个来源局")
    if len(result) != len(set(result)):
        raise ValueError("同一个 import 中不能重复指定来源局")
    invalid = [value for value in result if _ROUND_RE.fullmatch(value) is None]
    if invalid:
        raise ValueError(f"局号必须是 YYYYMMDDHHMM: {', '.join(invalid)}")
    return result


def _label_from_path(relative: Path) -> tuple[str, str]:
    parts = relative.parts
    if len(parts) == 2 and parts[0] == "unknown":
        return "unknown", "unknown"
    if len(parts) != 3:
        raise ValueError(f"已标注图片目录层级不符合约定: {relative}")
    category, detail, _name = parts
    if category in {"empty", "unavailable"} and detail in {"plain", "effect"}:
        return category, f"{category}_{detail}"
    if _HERO_RE.fullmatch(category) and re.fullmatch(r"star[1-4]", detail):
        return f"{category}_{detail}", "hero"
    raise ValueError(f"已标注图片目录不符合约定: {relative}")


def _import_heroes(
    rows: list[dict[str, str]], lineup_others: list[str], main_c: str | None
) -> list[str]:
    heroes = [row.get("main_c", "") for row in rows]
    heroes.extend(lineup_others)
    if main_c:
        heroes.append(main_c)
    return sorted({hero for hero in heroes if _HERO_RE.fullmatch(hero)})


def _prepare_split_labels(labeled_dir: Path, *, heroes: list[str]) -> None:
    for category in ("empty", "unavailable"):
        for kind in ("plain", "effect"):
            (labeled_dir / category / kind).mkdir(parents=True, exist_ok=True)
    (labeled_dir / "unknown").mkdir(parents=True, exist_ok=True)
    for hero in heroes:
        for star in range(1, 5):
            (labeled_dir / hero / f"star{star}").mkdir(parents=True, exist_ok=True)


def _write_csv_atomic(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
