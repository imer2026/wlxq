"""使用整局隔离数据评估导出的英雄格 ONNX 分类器。"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from wlxq_bot.hero_classifier.labels import (
    discover_labeled_samples,
    resolve_round_keys,
    samples_for_rounds,
)
from wlxq_bot.perception.hero_classifier import HeroCellClassifier


@dataclass(frozen=True)
class EvaluationResult:
    """英雄格分类器整局评估汇总。"""

    samples: int
    accepted: int
    rejected: int
    correct: int
    accuracy_all: float
    accuracy_accepted: float
    report_path: Path
    predictions_path: Path


def evaluate_hero_classifier(
    *,
    dataset_root: Path,
    model_path: Path,
    output_dir: Path,
    rounds: list[str] | None = None,
    split: str = "test",
    batch_size: int = 128,
) -> EvaluationResult:
    """只评估明确指定的整局数据，输出逐图结果和混淆统计。"""
    dataset_root = Path(dataset_root)
    from wlxq_bot.hero_classifier.dataset import validate_split

    split = validate_split(split)
    split_root = dataset_root / split
    if (split_root / "labeled").is_dir():
        manifest = split_root / "dataset_manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"缺少 {split} 标签清单，请先执行 hero-classifier sync-labels: {manifest}"
            )
        samples = discover_labeled_samples(split_root)
        round_keys = {sample.round_key for sample in samples}
    else:
        all_samples = discover_labeled_samples(dataset_root)
        if rounds is None:
            raise ValueError("旧式数据目录必须明确指定评估局")
        round_keys = resolve_round_keys(all_samples, rounds)
        samples = samples_for_rounds(all_samples, round_keys)
    if not samples:
        raise ValueError("指定评估局没有已标注图片")
    if batch_size < 1:
        raise ValueError("batch_size 必须大于等于 1")
    classifier = HeroCellClassifier(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "evaluation_predictions.csv"
    report_path = output_dir / "evaluation_report.json"
    rows: list[dict[str, str | float | bool]] = []
    confusion: dict[str, dict[str, int]] = {}
    accepted = 0
    correct = 0
    accepted_correct = 0

    for offset in range(0, len(samples), batch_size):
        batch_samples = samples[offset : offset + batch_size]
        images = [_read_image(sample.path) for sample in batch_samples]
        predictions = classifier.predict(images)
        for sample, prediction in zip(batch_samples, predictions, strict=True):
            predicted = prediction.class_name
            is_correct = predicted == sample.class_name
            if not prediction.rejected:
                accepted += 1
                accepted_correct += int(is_correct)
            correct += int(is_correct)
            confusion.setdefault(sample.class_name, {})[predicted] = (
                confusion.setdefault(sample.class_name, {}).get(predicted, 0) + 1
            )
            rows.append(
                {
                    "round_key": sample.round_key,
                    "path": str(sample.path),
                    "expected": sample.class_name,
                    "predicted": predicted,
                    "confidence": prediction.confidence,
                    "margin": prediction.margin,
                    "rejected": prediction.rejected,
                    "correct": is_correct,
                }
            )

    with predictions_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    total = len(samples)
    report = {
        "rounds": sorted(round_keys),
        "samples": total,
        "accepted": accepted,
        "rejected": total - accepted,
        "accuracy_all": correct / total,
        "accuracy_accepted": accepted_correct / accepted if accepted else 0.0,
        "confusion": confusion,
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return EvaluationResult(
        samples=total,
        accepted=accepted,
        rejected=total - accepted,
        correct=correct,
        accuracy_all=correct / total,
        accuracy_accepted=accepted_correct / accepted if accepted else 0.0,
        report_path=report_path,
        predictions_path=predictions_path,
    )


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"评估图片解码失败: {path}")
    return image
