"""SkillCollector 技能卡采集器测试：归属、去重、永不抛异常与熔断、落盘。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from wlxq_bot.perception.skill_collector import SkillCollector, _CaptureTask
from wlxq_bot.perception.vision import Vision

# 与 SkillCollector 默认 icon_band 对应；测试图标放卡片中部，落在带内即可
_ICON_CENTER = (0.50, 0.47)  # 相对卡片宽高的图标中心


def make_card(
    width: int = 280,
    height: int = 560,
    seed: int | None = None,
    noise_seed: int = 0,
) -> np.ndarray:
    """生成一张合成技能卡（噪声底，不同 noise_seed 结构互异、哈希必不同）。

    seed 非 None 时在图标区放一块随机纹理作图标。注意不能用纯色或同构
    渐变底：aHash 只看与均值的相对结构，纯色图哈希全 0、同构渐变哈希
    相同，会被去重逻辑正确地合并。
    """
    rng = np.random.default_rng(noise_seed)
    card = rng.integers(40, 70, size=(height, width, 3), dtype=np.uint8)
    if seed is not None:
        icon_rng = np.random.default_rng(seed)
        icon = icon_rng.integers(0, 255, size=(110, 110, 3), dtype=np.uint8)
        cx, cy = int(width * _ICON_CENTER[0]), int(height * _ICON_CENTER[1])
        card[cy - 55 : cy + 55, cx - 55 : cx + 55] = icon
    return card


def make_frame(cards: list[np.ndarray]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """把卡片横向无缝拼成一帧，返回 (帧, 覆盖全部卡片的 ROI)。"""
    height = cards[0].shape[0]
    width = sum(card.shape[1] for card in cards)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    x = 0
    for card in cards:
        frame[:, x : x + card.shape[1]] = card
        x += card.shape[1]
    return frame, (0, 0, width, height)


def save_template(path: Path, seed: int) -> Path:
    """把与卡内一致的图标纹理存为模板文件。"""
    rng = np.random.default_rng(seed)
    icon = rng.integers(0, 255, size=(110, 110, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", icon)
    assert ok
    path.write_bytes(buf.tobytes())
    return path


def read_meta(meta_path: Path) -> list[dict]:
    lines = meta_path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def make_collector(tmp_path: Path, hero_icons: dict[str, list[Path]], **kwargs) -> SkillCollector:
    return SkillCollector(
        output_dir=tmp_path / "skill_cards",
        hero_icons=hero_icons,
        vision=Vision(),
        **kwargs,
    )


class TestObserve:
    def test_attribution_and_save(self, tmp_path: Path) -> None:
        """带图标的卡归属到英雄，无图标卡归为 unknown，全部落盘。"""
        template = save_template(tmp_path / "tpl_a.png", seed=7)
        # 三张卡噪声底各不相同,确保哈希互异(同卡去重是另一条测试的职责)
        cards = [
            make_card(seed=7, noise_seed=1),
            make_card(noise_seed=2),
            make_card(noise_seed=3),
        ]
        frame, roi = make_frame(cards)
        collector = make_collector(
            tmp_path, {"英雄A": [template]}, fuse_max_consecutive_failures=3
        )

        collector.observe(1, frame, roi, page="SELECT_OPENING_SKILLS")
        collector.close()

        captures = tmp_path / "skill_cards" / "captures"
        saved = sorted(captures.glob("*.png"))
        assert len(saved) == 3
        rows = read_meta(tmp_path / "skill_cards" / "meta.jsonl")
        assert len(rows) == 3
        first = next(row for row in rows if row["column"] == 0)
        assert first["hero"] == "英雄A"
        assert first["hero_confidence"] >= 0.9
        assert first["page"] == "SELECT_OPENING_SKILLS"
        assert first["frame_id"] == 1
        for row in rows:
            if row["column"] != 0:
                assert row["hero"] is None

    def test_dedup_same_frame(self, tmp_path: Path) -> None:
        """同一帧重复采集（多帧重试场景）只落盘一份。"""
        template = save_template(tmp_path / "tpl_a.png", seed=7)
        frame, roi = make_frame(
            [
                make_card(seed=7, noise_seed=1),
                make_card(noise_seed=2),
                make_card(noise_seed=3),
            ]
        )
        collector = make_collector(tmp_path, {"英雄A": [template]})

        collector.observe(1, frame, roi, page="X")
        collector.observe(2, frame, roi, page="X")
        collector.close()

        captures = tmp_path / "skill_cards" / "captures"
        assert len(list(captures.glob("*.png"))) == 3
        assert len(read_meta(tmp_path / "skill_cards" / "meta.jsonl")) == 3

    def test_roi_none_skipped(self, tmp_path: Path) -> None:
        """ROI 未标定（None）时跳过采集，不产生任何文件。"""
        template = save_template(tmp_path / "tpl_a.png", seed=7)
        collector = make_collector(tmp_path, {"英雄A": [template]})
        frame = make_card()

        collector.observe(1, frame, None, page="X")
        collector.close()

        assert not (tmp_path / "skill_cards" / "captures").exists()

    def test_empty_hero_icons_noop(self, tmp_path: Path) -> None:
        """未配置英雄图标时不启动写盘、不产生文件。"""
        collector = make_collector(tmp_path, {})
        frame, roi = make_frame([make_card()])

        collector.observe(1, frame, roi, page="X")
        collector.close()

        assert not (tmp_path / "skill_cards").exists()


class TestNeverRaise:
    def test_internal_error_swallowed_and_fused(self, tmp_path: Path) -> None:
        """内部异常不外泄；连续失败达阈值后自动熔断停用。"""
        template = save_template(tmp_path / "tpl_a.png", seed=7)
        frame, roi = make_frame([make_card(seed=7)])
        collector = make_collector(
            tmp_path, {"英雄A": [template]}, fuse_max_consecutive_failures=3
        )

        with patch.object(SkillCollector, "_ahash", side_effect=RuntimeError("boom")):
            for _ in range(3):
                collector.observe(1, frame, roi, page="X")  # 不应抛异常
        assert collector.disabled is True

        # 熔断后即使异常源消失也不再工作
        collector.observe(2, frame, roi, page="X")
        assert list((tmp_path / "skill_cards" / "captures").glob("*.png")) == []

    def test_queue_full_dropped_not_blocking(self, tmp_path: Path) -> None:
        """写盘队列满时丢弃并计数，入队永不阻塞。"""
        template = save_template(tmp_path / "tpl_a.png", seed=7)
        collector = make_collector(tmp_path, {"英雄A": [template]}, queue_maxsize=8)
        collector._ensure_writer()
        card = make_card()
        ok, buf = cv2.imencode(".png", card)
        assert ok

        for _ in range(200):
            collector._enqueue(
                _CaptureTask(
                    image_path=tmp_path / "x.png",
                    png_bytes=buf.tobytes(),
                    meta={"hash": "x"},
                )
            )
        assert collector.dropped_count > 0
        collector.close()
