"""SkillCollector 技能卡采集器测试：裁剪落盘、去重、节流、永不抛异常与熔断。

运行时采集只做裁剪和哈希；英雄归属在离线建册时进行（见 test_skill_catalog）。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from wlxq_bot.perception.skill_collector import SkillCollector, _CaptureTask

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


def read_meta(meta_path: Path) -> list[dict]:
    lines = meta_path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def make_collector(tmp_path: Path, **kwargs) -> SkillCollector:
    kwargs.setdefault("min_collect_interval_seconds", 0.0)
    kwargs.setdefault("session_label", "20260901_120000")
    return SkillCollector(output_dir=tmp_path / "skill_cards", **kwargs)


class TestObserve:
    def test_capture_and_save(self, tmp_path: Path) -> None:
        """技能页帧裁出三张卡，按局目录落盘，meta 记录相对路径/帧号/页名/列号。"""
        cards = [
            make_card(seed=7, noise_seed=1),
            make_card(noise_seed=2),
            make_card(noise_seed=3),
        ]
        frame, roi = make_frame(cards)
        collector = make_collector(tmp_path, fuse_max_consecutive_failures=3)

        collector.observe(1, frame, roi, page="SELECT_OPENING_SKILLS")
        collector.close()

        session_dir = tmp_path / "skill_cards" / "20260901_120000"
        saved = sorted(session_dir.glob("*.png"))
        assert len(saved) == 3
        # 文件名 = 序号_时分秒,人能直接读懂;序号从 001 递增
        import re

        stems = sorted(p.stem for p in saved)
        assert all(re.fullmatch(r"\d{3}_\d{6}", stem) for stem in stems)
        assert [stem.split("_")[0] for stem in stems] == ["001", "002", "003"]
        rows = read_meta(tmp_path / "skill_cards" / "meta.jsonl")
        assert len(rows) == 3
        assert {row["column"] for row in rows} == {0, 1, 2}
        assert all(row["page"] == "SELECT_OPENING_SKILLS" for row in rows)
        assert all(row["frame_id"] == 1 for row in rows)
        assert all(row["image"].startswith("20260901_120000/") for row in rows)

    def test_dedup_same_frame(self, tmp_path: Path) -> None:
        """同一帧重复采集（多帧重试场景）只落盘一份。"""
        frame, roi = make_frame(
            [
                make_card(seed=7, noise_seed=1),
                make_card(noise_seed=2),
                make_card(noise_seed=3),
            ]
        )
        collector = make_collector(tmp_path)

        collector.observe(1, frame, roi, page="X")
        collector.observe(2, frame, roi, page="X")
        collector.close()

        captures = tmp_path / "skill_cards" / "20260901_120000"
        assert len(list(captures.glob("*.png"))) == 3
        assert len(read_meta(tmp_path / "skill_cards" / "meta.jsonl")) == 3

    def test_roi_none_skipped(self, tmp_path: Path) -> None:
        """ROI 未标定（None）时跳过采集，不产生任何文件。"""
        collector = make_collector(tmp_path)
        frame = make_card()

        collector.observe(1, frame, None, page="X")
        collector.close()

        assert not (tmp_path / "skill_cards" / "20260901_120000").exists()

    def test_throttle_skips_within_interval(self, tmp_path: Path) -> None:
        """节流间隔内的 observe 直接跳过，连新卡也不采集。"""
        collector = make_collector(tmp_path, min_collect_interval_seconds=60.0)
        frame1, roi = make_frame(
            [make_card(seed=7, noise_seed=1), make_card(noise_seed=2), make_card(noise_seed=3)]
        )

        collector.observe(1, frame1, roi, page="X")
        collector.close()
        assert len(read_meta(tmp_path / "skill_cards" / "meta.jsonl")) == 3

        # 间隔内换一批全新卡（哈希必然不同）也应被节流跳过
        frame2, roi2 = make_frame(
            [make_card(seed=8, noise_seed=4), make_card(noise_seed=5), make_card(noise_seed=6)]
        )
        collector.observe(2, frame2, roi2, page="X")
        collector.close()

        assert len(list((tmp_path / "skill_cards" / "20260901_120000").glob("*.png"))) == 3
        assert len(read_meta(tmp_path / "skill_cards" / "meta.jsonl")) == 3


class TestNeverRaise:
    def test_internal_error_swallowed_and_fused(self, tmp_path: Path) -> None:
        """内部异常不外泄；连续失败达阈值后自动熔断停用。"""
        frame, roi = make_frame([make_card(seed=7)])
        collector = make_collector(tmp_path, fuse_max_consecutive_failures=3)

        with patch.object(SkillCollector, "_ahash", side_effect=RuntimeError("boom")):
            for _ in range(3):
                collector.observe(1, frame, roi, page="X")  # 不应抛异常
        assert collector.disabled is True

        # 熔断后即使异常源消失也不再工作
        collector.observe(2, frame, roi, page="X")
        assert list((tmp_path / "skill_cards" / "captures").glob("*.png")) == []

    def test_queue_full_dropped_not_blocking(self, tmp_path: Path) -> None:
        """写盘队列满时丢弃并计数，入队永不阻塞。"""
        collector = make_collector(tmp_path, queue_maxsize=8)
        collector._ensure_writer()
        card = make_card()

        for _ in range(200):
            collector._enqueue(
                _CaptureTask(
                    image_path=tmp_path / "x.png",
                    card=card,
                    meta={"hash": "x"},
                )
            )
        assert collector.dropped_count > 0
        collector.close()
