"""使用训练后的 ONNX 模型离线预分类 import candidates。"""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from wlxq_bot.hero_classifier.progress import ProgressLogger
from wlxq_bot.perception.hero_classifier import HeroCellClassifier, HeroCellPrediction
from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

_GROUP_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REQUIRED_CANDIDATE_FIELDS = (
    "primary_cluster",
    "secondary_cluster",
    "source_group_size",
    "selection_order",
    "round_id",
    "frame_index",
    "cell_label",
    "source_path",
    "candidate_path",
)
_PREDICTION_MANIFEST_FIELDS = _REQUIRED_CANDIDATE_FIELDS + (
    "model_path",
    "model_sha256",
    "confidence_threshold",
    "margin_threshold",
    "raw_top1_class",
    "raw_top1_confidence",
    "raw_top2_class",
    "raw_top2_confidence",
    "margin",
    "predicted_class",
    "rejected",
    "rejection_reason",
    "group_decision",
    "group_predicted_class",
    "suggested_path",
)


@dataclass(frozen=True)
class SuggestionStats:
    """一次 candidates 离线预分类的汇总。"""

    groups: int
    images: int
    suggested_groups: int
    review_groups: int
    low_confidence_groups: int
    mixed_groups: int
    unknown_groups: int
    suggested_dir: Path
    manifest_path: Path


@dataclass(frozen=True)
class _CandidatePrediction:
    row: dict[str, str]
    path: Path
    prediction: HeroCellPrediction


def suggest_import_labels(
    *,
    import_dir: Path,
    model_path: Path,
    metadata_path: Path | None = None,
    confidence_threshold: float | None = None,
    margin_threshold: float | None = None,
    batch_size: int = 128,
    classifier: Any | None = None,
) -> SuggestionStats:
    """只预测一个 import 的 candidates，按小组整理 suggested 副本。

    本函数不会移动 candidates，也不会写入 split/labeled。已有 suggested 或
    prediction_manifest.csv 时拒绝覆盖，确保人工审核中的结果不会被静默改写。
    """
    if batch_size < 1:
        raise ValueError("batch_size 必须大于等于 1")
    import_dir, dataset_root = _validate_import_dir(import_dir)
    candidates_dir = import_dir / "candidates"
    candidate_manifest = import_dir / "candidate_manifest.csv"
    suggested_dir = import_dir / "suggested"
    manifest_path = import_dir / "prediction_manifest.csv"
    working_dir = import_dir / ".suggested.tmp"
    working_manifest = import_dir / ".prediction_manifest.csv.tmp"
    if not candidates_dir.is_dir():
        raise FileNotFoundError(f"import 缺少 candidates 目录: {candidates_dir}")
    if not candidate_manifest.is_file():
        raise FileNotFoundError(f"import 缺少 candidate_manifest.csv: {candidate_manifest}")
    if suggested_dir.exists() or manifest_path.exists():
        raise FileExistsError(f"import 已存在预分类结果，拒绝覆盖: {import_dir}")
    if working_dir.exists() or working_manifest.exists():
        raise FileExistsError(f"发现上次未完成的预分类临时文件，请检查后处理: {import_dir}")

    rows = _read_candidate_manifest(candidate_manifest)
    candidates = _resolve_candidates(rows, dataset_root, candidates_dir)
    if classifier is None:
        classifier = HeroCellClassifier(
            model_path,
            metadata_path,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
        )
    used_confidence = float(classifier.confidence_threshold)
    used_margin = float(classifier.margin_threshold)
    model_hash = _sha256(Path(model_path)) if Path(model_path).is_file() else ""

    predicted: list[_CandidatePrediction] = []
    progress = ProgressLogger(logger, "candidates 模型预分类", len(candidates))
    progress.start(detail=f"batch_size={batch_size}")
    for offset in range(0, len(candidates), batch_size):
        batch = candidates[offset : offset + batch_size]
        images = [_read_image(path) for _row, path in batch]
        predictions = classifier.predict(images)
        if len(predictions) != len(batch):
            raise RuntimeError(
                f"分类器返回数量不正确: predictions={len(predictions)} images={len(batch)}"
            )
        predicted.extend(
            _CandidatePrediction(row, path, prediction)
            for (row, path), prediction in zip(batch, predictions, strict=True)
        )
        progress.update(min(offset + len(batch), len(candidates)))
    progress.finish(len(candidates))

    grouped: dict[tuple[str, str], list[_CandidatePrediction]] = defaultdict(list)
    for item in predicted:
        grouped[(item.row["primary_cluster"], item.row["secondary_cluster"])].append(item)

    output_rows: list[dict[str, str | float | bool]] = []
    counters = defaultdict(int)
    working_dir.mkdir()
    try:
        for (primary, secondary), items in sorted(grouped.items()):
            decision, group_class = _group_decision(items)
            if decision == "suggested":
                relative_group = Path(group_class) / primary / secondary
                counters["suggested"] += 1
            else:
                relative_group = Path("review") / decision / primary / secondary
                counters[decision] += 1
                counters["review"] += 1
            target_group = working_dir / relative_group
            target_group.mkdir(parents=True)
            for item in items:
                target = target_group / item.path.name
                shutil.copy2(item.path, target)
                prediction = item.prediction
                suggested_relative = (
                    suggested_dir.relative_to(dataset_root) / relative_group / item.path.name
                ).as_posix()
                output_rows.append(
                    {
                        **item.row,
                        "model_path": str(Path(model_path)),
                        "model_sha256": model_hash,
                        "confidence_threshold": used_confidence,
                        "margin_threshold": used_margin,
                        "raw_top1_class": prediction.raw_class_name,
                        "raw_top1_confidence": prediction.confidence,
                        "raw_top2_class": prediction.second_class_name,
                        "raw_top2_confidence": prediction.second_confidence,
                        "margin": prediction.margin,
                        "predicted_class": prediction.class_name,
                        "rejected": prediction.rejected,
                        "rejection_reason": prediction.rejection_reason,
                        "group_decision": decision,
                        "group_predicted_class": group_class,
                        "suggested_path": suggested_relative,
                    }
                )
        _write_prediction_manifest(working_manifest, output_rows)
        working_dir.replace(suggested_dir)
        working_manifest.replace(manifest_path)
    except Exception:
        if working_dir.exists():
            shutil.rmtree(working_dir)
        if working_manifest.exists():
            working_manifest.unlink()
        if suggested_dir.exists() and not manifest_path.exists():
            shutil.rmtree(suggested_dir)
        raise

    return SuggestionStats(
        groups=len(grouped),
        images=len(candidates),
        suggested_groups=counters["suggested"],
        review_groups=counters["review"],
        low_confidence_groups=counters["low_confidence"],
        mixed_groups=counters["mixed_group"],
        unknown_groups=counters["unknown"],
        suggested_dir=suggested_dir,
        manifest_path=manifest_path,
    )


def _validate_import_dir(import_dir: Path) -> tuple[Path, Path]:
    import_dir = Path(import_dir).resolve()
    if import_dir.parent.name != "imports" or import_dir.parent.parent.name not in {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("import_dir 必须是 <数据组>/<split>/imports/<import_id>")
    return import_dir, import_dir.parent.parent.parent


def _read_candidate_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = set(_REQUIRED_CANDIDATE_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"candidate_manifest.csv 缺少字段: {', '.join(sorted(missing))}")
        rows = [
            {field: row.get(field, "") for field in _REQUIRED_CANDIDATE_FIELDS} for row in reader
        ]
    if not rows:
        raise ValueError(f"candidate_manifest.csv 没有候选图片: {path}")
    return rows


def _resolve_candidates(
    rows: list[dict[str, str]], dataset_root: Path, candidates_dir: Path
) -> list[tuple[dict[str, str], Path]]:
    resolved: list[tuple[dict[str, str], Path]] = []
    seen: set[Path] = set()
    for row in rows:
        for field in ("primary_cluster", "secondary_cluster"):
            if _GROUP_COMPONENT_RE.fullmatch(row[field]) is None:
                raise ValueError(f"candidate_manifest.csv 包含无效 {field}: {row[field]!r}")
        candidate_path = row["candidate_path"]
        if not candidate_path:
            raise ValueError("candidate_manifest.csv 包含空 candidate_path")
        path = (dataset_root / Path(candidate_path)).resolve()
        try:
            path.relative_to(candidates_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"候选路径不在当前 import/candidates 中: {candidate_path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"候选图片不存在: {path}")
        if (
            path.parent.name != row["secondary_cluster"]
            or path.parent.parent.name != row["primary_cluster"]
        ):
            raise ValueError(
                "候选路径与清单小组不一致: "
                f"candidate={candidate_path} primary={row['primary_cluster']} "
                f"secondary={row['secondary_cluster']}"
            )
        if path in seen:
            raise ValueError(f"candidate_manifest.csv 重复登记候选图片: {path}")
        seen.add(path)
        resolved.append((row, path))
    actual = {path.resolve() for path in candidates_dir.rglob("*.png")}
    if actual != seen:
        missing = len(actual - seen)
        extra = len(seen - actual)
        raise ValueError(
            "candidates 与 candidate_manifest.csv 不一致: "
            f"未登记图片={missing} 清单额外图片={extra}"
        )
    return resolved


def _group_decision(items: list[_CandidatePrediction]) -> tuple[str, str]:
    raw_classes = {item.prediction.raw_class_name for item in items}
    if "" in raw_classes:
        return "unknown", ""
    if len(raw_classes) > 1:
        return "mixed_group", ""
    group_class = next(iter(raw_classes))
    if any(item.prediction.rejected for item in items):
        return "low_confidence", group_class
    if group_class in {"unknown", ""}:
        return "unknown", ""
    return "suggested", group_class


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"候选图片解码失败: {path}")
    return image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_prediction_manifest(path: Path, rows: list[dict[str, str | float | bool]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_PREDICTION_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
