"""正式运行时英雄格 ONNX 推理适配测试。"""

from __future__ import annotations

import json

import numpy as np

from wlxq_bot.perception.hero_classifier import HeroCellClassifier


class _FakeNet:
    def __init__(self, logits: np.ndarray) -> None:
        self.logits = logits
        self.input = None

    def setInput(self, value) -> None:
        self.input = value

    def forward(self):
        return self.logits


def _metadata(path) -> None:
    path.write_text(
        json.dumps(
            {
                "class_names": ["empty", "assault_star2", "monkey_star1"],
                "input_size": 32,
                "confidence_threshold": 0.7,
                "margin_threshold": 0.2,
            }
        ),
        encoding="utf-8",
    )


def test_predicts_hero_and_star(tmp_path) -> None:
    metadata = tmp_path / "hero_classifier.json"
    _metadata(metadata)
    net = _FakeNet(np.array([[0.0, 5.0, 1.0]], dtype=np.float32))
    classifier = HeroCellClassifier(tmp_path / "unused.onnx", metadata, net=net)

    prediction = classifier.predict([np.zeros((20, 30, 3), dtype=np.uint8)])[0]

    assert prediction.class_name == "assault_star2"
    assert prediction.hero_type == "assault"
    assert prediction.star_level == 2
    assert not prediction.rejected
    assert net.input.shape == (1, 3, 32, 32)


def test_hero_type_alias_maps_variant_to_base(tmp_path) -> None:
    """变体类别映射为本体（洛姬分身→洛姬）：class_name 保留原值供日志，
    hero_type/star_level 走映射后的本体。"""
    metadata = tmp_path / "hero_classifier.json"
    metadata.write_text(
        json.dumps(
            {
                "class_names": ["empty", "loki_phantom_star2", "loki_star1"],
                "input_size": 32,
                "confidence_threshold": 0.7,
                "margin_threshold": 0.2,
            }
        ),
        encoding="utf-8",
    )
    net = _FakeNet(np.array([[0.0, 5.0, 1.0]], dtype=np.float32))
    classifier = HeroCellClassifier(
        tmp_path / "unused.onnx",
        metadata,
        net=net,
        hero_type_alias={"loki_phantom": "loki"},
    )

    prediction = classifier.predict([np.zeros((20, 30, 3), dtype=np.uint8)])[0]

    assert prediction.class_name == "loki_phantom_star2"
    assert prediction.raw_class_name == "loki_phantom_star2"
    assert prediction.hero_type == "loki"
    assert prediction.star_level == 2


def test_hero_type_alias_unknown_key_is_ignored(tmp_path) -> None:
    """别名键不在模型类别空间时静默不生效（构造时打警告），不改变预测。"""
    metadata = tmp_path / "hero_classifier.json"
    _metadata(metadata)
    net = _FakeNet(np.array([[0.0, 5.0, 1.0]], dtype=np.float32))
    classifier = HeroCellClassifier(
        tmp_path / "unused.onnx",
        metadata,
        net=net,
        hero_type_alias={"phantom_typo": "loki"},
    )

    prediction = classifier.predict([np.zeros((20, 30, 3), dtype=np.uint8)])[0]

    assert prediction.class_name == "assault_star2"
    assert prediction.hero_type == "assault"


def test_rejects_low_margin_prediction(tmp_path) -> None:
    metadata = tmp_path / "hero_classifier.json"
    _metadata(metadata)
    net = _FakeNet(np.array([[0.0, 1.0, 0.9]], dtype=np.float32))
    classifier = HeroCellClassifier(tmp_path / "unused.onnx", metadata, net=net)

    prediction = classifier.predict([np.zeros((20, 30, 3), dtype=np.uint8)])[0]

    assert prediction.class_name == "unknown"
    assert prediction.rejected
    assert prediction.hero_type is None
    assert prediction.star_level is None
    assert prediction.raw_class_name == "assault_star2"
    assert prediction.second_class_name == "monkey_star1"
    assert prediction.rejection_reason == "low_confidence+low_margin"


def test_predicts_unavailable_without_parsing_as_hero(tmp_path) -> None:
    metadata = tmp_path / "hero_classifier.json"
    metadata.write_text(
        json.dumps(
            {
                "class_names": ["empty", "unavailable", "assault_star1"],
                "input_size": 32,
                "confidence_threshold": 0.7,
                "margin_threshold": 0.2,
            }
        ),
        encoding="utf-8",
    )
    net = _FakeNet(np.array([[0.0, 5.0, 1.0]], dtype=np.float32))
    classifier = HeroCellClassifier(tmp_path / "unused.onnx", metadata, net=net)

    prediction = classifier.predict([np.zeros((20, 30, 3), dtype=np.uint8)])[0]

    assert prediction.class_name == "unavailable"
    assert prediction.hero_type is None
    assert prediction.star_level is None
    assert not prediction.rejected


def test_threshold_override_is_used(tmp_path) -> None:
    metadata = tmp_path / "hero_classifier.json"
    _metadata(metadata)
    net = _FakeNet(np.array([[0.0, 1.0, 0.9]], dtype=np.float32))
    classifier = HeroCellClassifier(
        tmp_path / "unused.onnx",
        metadata,
        net=net,
        confidence_threshold=0.3,
        margin_threshold=0.0,
    )

    prediction = classifier.predict([np.zeros((20, 30, 3), dtype=np.uint8)])[0]

    assert prediction.class_name == "assault_star2"
    assert not prediction.rejected
    assert classifier.confidence_threshold == 0.3
    assert classifier.margin_threshold == 0.0


def test_exposes_model_classes_and_supported_heroes(tmp_path) -> None:
    metadata = tmp_path / "hero_classifier.json"
    _metadata(metadata)
    classifier = HeroCellClassifier(
        tmp_path / "unused.onnx",
        metadata,
        net=_FakeNet(np.zeros((1, 3), dtype=np.float32)),
    )

    assert classifier.class_names == ("empty", "assault_star2", "monkey_star1")
    assert classifier.supported_heroes == frozenset({"assault", "monkey"})
