"""MobileNetV3-Small 英雄格分类器训练与 ONNX 导出。"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from wlxq_bot.hero_classifier.labels import (
    LabeledCellSample,
    class_names_for_samples,
    discover_labeled_samples,
    resolve_round_keys,
    samples_for_rounds,
)
from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_HERO_CLASS_RE = re.compile(r"^(?P<hero>[a-z][a-z0-9_]*)_star(?P<star>[1-4])$")


@dataclass(frozen=True)
class TrainingConfig:
    """英雄格分类器训练参数。"""

    input_size: int = 96
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.001
    workers: int = 0
    seed: int = 42
    pretrained: bool = True
    confidence_threshold: float = 0.8
    margin_threshold: float = 0.2
    main_c: str = ""
    star1_weight: float = 1.0
    star2_weight: float = 1.0
    main_c_star3_weight: float = 0.8
    other_star3_weight: float = 0.5
    main_c_star4_weight: float = 0.3
    other_star4_weight: float = 0.1
    empty_weight: float = 0.5
    unavailable_weight: float = 0.3


@dataclass(frozen=True)
class TrainingResult:
    """训练完成后的主要产物和最佳验证指标。"""

    checkpoint_path: Path
    onnx_path: Path
    metadata_path: Path
    history_path: Path
    best_epoch: int
    best_validation_accuracy: float
    class_names: tuple[str, ...]


def train_hero_classifier(
    *,
    dataset_root: Path,
    output_dir: Path,
    train_rounds: list[str] | None = None,
    validation_rounds: list[str] | None = None,
    config: TrainingConfig,
) -> TrainingResult:
    """按整局隔离训练和验证，保存最佳权重并导出 ONNX。"""
    torch, nn, data, models, transforms, image_module = _training_dependencies()
    _validate_config(config)
    dataset_root = Path(dataset_root)
    split_layout = (dataset_root / "train" / "labeled").is_dir()
    if split_layout:
        for split in ("train", "validation"):
            manifest = dataset_root / split / "dataset_manifest.csv"
            if not manifest.is_file():
                raise FileNotFoundError(
                    f"缺少 {split} 标签清单，请先执行 hero-classifier sync-labels: {manifest}"
                )
        train_samples = discover_labeled_samples(dataset_root / "train")
        validation_samples = discover_labeled_samples(dataset_root / "validation")
        train_keys = {sample.round_key for sample in train_samples}
        validation_keys = {sample.round_key for sample in validation_samples}
    else:
        all_samples = discover_labeled_samples(dataset_root)
        if not all_samples:
            raise ValueError(f"未找到已标注格子图片: {dataset_root}")
        if train_rounds is None or validation_rounds is None:
            raise ValueError("旧式数据目录必须明确指定训练局和验证局")
        train_keys = resolve_round_keys(all_samples, train_rounds)
        validation_keys = resolve_round_keys(all_samples, validation_rounds)
        train_samples = samples_for_rounds(all_samples, train_keys)
        validation_samples = samples_for_rounds(all_samples, validation_keys)
    overlap = train_keys & validation_keys
    if overlap:
        raise ValueError(f"训练局和验证局重叠: {', '.join(sorted(overlap))}")
    if not train_samples or not validation_samples:
        raise ValueError("训练集和验证集都必须至少包含一张已标注图片")
    class_names = class_names_for_samples(train_samples)
    validation_classes = {sample.class_name for sample in validation_samples}
    missing_train = validation_classes - set(class_names)
    if missing_train:
        raise ValueError(f"验证集包含训练集没有的类别: {', '.join(sorted(missing_train))}")

    _seed_everything(random, torch, config.seed)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    train_transform = transforms.Compose(
        [
            transforms.Resize((config.input_size, config.input_size)),
            transforms.RandomAffine(degrees=5, translate=(0.04, 0.04), scale=(0.75, 1.25)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize((config.input_size, config.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ]
    )

    dataset_type = _build_dataset_type(data.Dataset, image_module)
    train_dataset = dataset_type(train_samples, class_to_index, train_transform)
    validation_dataset = dataset_type(validation_samples, class_to_index, validation_transform)
    class_weights = class_sampling_weights(class_names, config=config)
    sample_weights = balanced_sample_weights(train_samples, class_weights=class_weights)
    sampler = data.WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.workers,
    )
    validation_loader = data.DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
    )

    weights = models.MobileNet_V3_Small_Weights.DEFAULT if config.pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "hero_classifier.pt"
    history_path = output_dir / "training_history.json"
    history: list[dict[str, float | int]] = []
    best_accuracy = -1.0
    best_epoch = 0

    logger.info(
        "英雄格分类器训练开始 train=%d validation=%d classes=%d device=%s",
        len(train_samples),
        len(validation_samples),
        len(class_names),
        device,
    )
    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            torch, model, train_loader, device, criterion, optimizer
        )
        validation_loss, validation_accuracy = _run_epoch(
            torch, model, validation_loader, device, criterion, None
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        logger.info(
            "训练轮次 %d/%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            config.epochs,
            train_loss,
            train_accuracy,
            validation_loss,
            validation_accuracy,
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    onnx_path = output_dir / "hero_classifier.onnx"
    dummy = torch.zeros(1, 3, config.input_size, config.input_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        # torch>=2.6 默认 dynamo 导出器在降级 opset 时经 onnxscript 产出坏图
        # （全部输出同一类、置信度恒 1.0）；老导出器（TorchScript 路径）稳定。
        dynamo=False,
    )
    metadata_path = onnx_path.with_suffix(".json")
    metadata = {
        "architecture": "mobilenet_v3_small",
        "class_names": class_names,
        "input_size": config.input_size,
        "mean": _MEAN,
        "std": _STD,
        "confidence_threshold": config.confidence_threshold,
        "margin_threshold": config.margin_threshold,
        "class_sampling_weights": class_weights,
        "train_rounds": sorted(train_keys),
        "validation_rounds": sorted(validation_keys),
        "training_config": asdict(config),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    logger.info(
        "英雄格分类器训练结束 best_epoch=%d val_acc=%.4f onnx=%s",
        best_epoch,
        best_accuracy,
        onnx_path,
    )
    return TrainingResult(
        checkpoint_path=checkpoint_path,
        onnx_path=onnx_path,
        metadata_path=metadata_path,
        history_path=history_path,
        best_epoch=best_epoch,
        best_validation_accuracy=best_accuracy,
        class_names=tuple(class_names),
    )


def _training_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        from PIL import Image
        from torch import nn
        from torch.utils import data
        from torchvision import models, transforms
    except ImportError as exc:
        raise RuntimeError('缺少英雄格分类训练依赖，请执行 pip install -e ".[train]"') from exc
    return torch, nn, data, models, transforms, Image


def _build_dataset_type(dataset_base: Any, image_module: Any) -> type:
    class CellDataset(dataset_base):
        def __init__(self, samples: list[LabeledCellSample], class_to_index: dict, transform):
            self.samples = samples
            self.class_to_index = class_to_index
            self.transform = transform

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int):
            sample = self.samples[index]
            with image_module.open(sample.path) as image:
                rgb = image.convert("RGB")
                tensor = self.transform(rgb)
            return tensor, self.class_to_index[sample.class_name]

    return CellDataset


def _run_epoch(torch: Any, model: Any, loader: Any, device: Any, criterion: Any, optimizer: Any):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_items += batch_size
    return total_loss / total_items, total_correct / total_items


def _seed_everything(random_module: Any, torch: Any, seed: int) -> None:
    random_module.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_config(config: TrainingConfig) -> None:
    if config.input_size < 32:
        raise ValueError("input_size 必须大于等于 32")
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs 和 batch_size 必须大于等于 1")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate 必须大于 0")
    if config.workers < 0:
        raise ValueError("workers 不能小于 0")
    if not 0 <= config.confidence_threshold <= 1:
        raise ValueError("confidence_threshold 必须在 0-1")
    if not 0 <= config.margin_threshold <= 1:
        raise ValueError("margin_threshold 必须在 0-1")
    if _HERO_CLASS_RE.fullmatch(f"{config.main_c}_star1") is None:
        raise ValueError("main_c 必须是小写英文标识，可包含数字和下划线")
    configured_weights = {
        "star1_weight": config.star1_weight,
        "star2_weight": config.star2_weight,
        "main_c_star3_weight": config.main_c_star3_weight,
        "other_star3_weight": config.other_star3_weight,
        "main_c_star4_weight": config.main_c_star4_weight,
        "other_star4_weight": config.other_star4_weight,
        "empty_weight": config.empty_weight,
        "unavailable_weight": config.unavailable_weight,
    }
    invalid = [name for name, value in configured_weights.items() if value <= 0]
    if invalid:
        raise ValueError(f"训练抽样权重必须大于 0: {', '.join(invalid)}")


def class_sampling_weights(class_names: list[str], *, config: TrainingConfig) -> dict[str, float]:
    """按主 C、星级和负样本类型返回实际存在类别的业务权重。"""
    result: dict[str, float] = {}
    for class_name in class_names:
        if class_name == "empty":
            result[class_name] = config.empty_weight
            continue
        if class_name == "unavailable":
            result[class_name] = config.unavailable_weight
            continue
        match = _HERO_CLASS_RE.fullmatch(class_name)
        if match is None:
            raise ValueError(f"无法为未知训练类别分配抽样权重: {class_name}")
        hero = match.group("hero")
        star = int(match.group("star"))
        if star == 1:
            weight = config.star1_weight
        elif star == 2:
            weight = config.star2_weight
        elif star == 3:
            weight = (
                config.main_c_star3_weight if hero == config.main_c else config.other_star3_weight
            )
        else:
            weight = (
                config.main_c_star4_weight if hero == config.main_c else config.other_star4_weight
            )
        result[class_name] = weight
    return result


def balanced_sample_weights(
    samples: list[LabeledCellSample], *, class_weights: dict[str, float] | None = None
) -> list[float]:
    """按业务类别权重、来源局和样本类型逐层分配训练抽样权重。

    每个类别先获得配置的业务总权重；一个类别内部的各来源局获得相同权重；
    同一类别和来源局中，``empty_plain`` / ``empty_effect`` 等样本类型
    再获得相同权重。最后由组内图片平分该组权重，因此长期停留产生的
    大量相似帧不会压过其他类别、其他局或困难空格负样本。
    """
    if not samples:
        raise ValueError("无法为缺少样本的训练集计算均衡权重")
    class_names = {sample.class_name for sample in samples}
    resolved_class_weights = class_weights or dict.fromkeys(class_names, 1.0)
    missing_weights = class_names - set(resolved_class_weights)
    if missing_weights:
        raise ValueError(f"训练类别缺少抽样权重: {', '.join(sorted(missing_weights))}")
    invalid_weights = [name for name in class_names if resolved_class_weights[name] <= 0]
    if invalid_weights:
        raise ValueError(f"训练类别抽样权重必须大于 0: {', '.join(sorted(invalid_weights))}")
    class_rounds = {(sample.class_name, sample.round_key) for sample in samples}
    subgroup_counts = Counter(
        (sample.class_name, sample.round_key, sample.sample_kind) for sample in samples
    )
    rounds_per_class = Counter(class_name for class_name, _round_key in class_rounds)
    kinds_per_class_round = Counter(
        (class_name, round_key) for class_name, round_key, _sample_kind in subgroup_counts
    )
    total_class_weight = sum(resolved_class_weights[name] for name in class_names)
    return [
        resolved_class_weights[sample.class_name]
        / (
            total_class_weight
            * rounds_per_class[sample.class_name]
            * kinds_per_class_round[(sample.class_name, sample.round_key)]
            * subgroup_counts[(sample.class_name, sample.round_key, sample.sample_kind)]
        )
        for sample in samples
    ]
