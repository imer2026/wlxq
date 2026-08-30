"""集成测试：用真实游戏截图验证完整视觉识别链路。

链路：scan_hero_templates → match_template_set → BoardHero.from_match
     → BoardSnapshot → find_merge_candidates

依赖外部文件（文件不存在时自动跳过）：
  - screenshots/raw/Snipaste_2026-08-08_16-16-17.png（真实游戏截图，含3个1星强袭）
  - assets/templates/3000x2000/heroes/assault/star1/强袭.png（1星模板）
  - assets/templates/3000x2000/heroes/assault/star2/2.png（2星模板，目标图里无2星）
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from wlxq_bot.assets import HeroTemplate, find_template_pack
from wlxq_bot.models import BoardCapacity, BoardHero, BoardSnapshot
from wlxq_bot.perception.vision import Vision

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_IMG = PROJECT_ROOT / "screenshots" / "raw" / "Snipaste_2026-08-08_16-16-17.png"
TEMPLATES_ROOT = PROJECT_ROOT / "assets" / "templates"

_SKIP = not TARGET_IMG.exists()


def _imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


@pytest.fixture
def vision() -> Vision:
    return Vision()


@pytest.fixture
def frame() -> np.ndarray:
    img = _imread_unicode(TARGET_IMG)
    assert img is not None, f"无法读取目标图: {TARGET_IMG}"
    return img


@pytest.fixture
def assault_templates() -> list[HeroTemplate]:
    pack = find_template_pack(TEMPLATES_ROOT, 3000, 2000)
    assert pack is not None, "找不到 3000x2000 模板包"
    templates = pack.scan_hero_templates("assault")
    assert len(templates) >= 1, "assault 模板为空"
    return templates


@pytest.mark.skipif(_SKIP, reason=f"目标图不存在: {TARGET_IMG}")
class TestIntegrationVision:
    """用真实游戏截图验证完整链路。"""

    def test_scan_finds_star1_and_star2(self, assault_templates: list[HeroTemplate]) -> None:
        """扫描到 star1 和 star2 模板。"""
        star_levels = {t.star_level for t in assault_templates}
        assert 1 in star_levels
        assert 2 in star_levels

    def test_match_finds_three_assault_star1(
        self,
        vision: Vision,
        frame: np.ndarray,
        assault_templates: list[HeroTemplate],
    ) -> None:
        """匹配到 3 个 1 星强袭（之前验证脚本确认过）。"""
        template_paths = [str(t.path) for t in assault_templates]
        results = vision.match_template_set(frame, template_paths, threshold=0.7)
        # star1 找到 3 个，star2 找到 0 个（目标图无 2 星）
        assert len(results) == 3
        assert all(r.confidence >= 0.7 for r in results)

    def test_build_board_heroes(
        self,
        vision: Vision,
        frame: np.ndarray,
        assault_templates: list[HeroTemplate],
    ) -> None:
        """从 MatchResult 构建 BoardHero，类型和星级正确。"""
        template_paths = [str(t.path) for t in assault_templates]
        results = vision.match_template_set(frame, template_paths, threshold=0.7)
        heroes = [BoardHero.from_match(r, r.template_name) for r in results]

        assert len(heroes) == 3
        # 都是 assault 1 星（star2 模板没匹配到）
        assert all(h.hero_type == "assault" for h in heroes)
        assert all(h.star_level == 1 for h in heroes)
        # 位置各不相同
        positions = {h.position for h in heroes}
        assert len(positions) == 3

    def test_board_snapshot_find_merge_candidates(
        self,
        vision: Vision,
        frame: np.ndarray,
        assault_templates: list[HeroTemplate],
    ) -> None:
        """完整链路：3 个 1 星强袭 → 1 对合成 + 1 个落单。"""
        template_paths = [str(t.path) for t in assault_templates]
        results = vision.match_template_set(frame, template_paths, threshold=0.7)
        heroes = [BoardHero.from_match(r, r.template_name) for r in results]

        snapshot = BoardSnapshot(
            frame_id=1,
            heroes=heroes,
            capacity=BoardCapacity(total_slots=6, occupied=3),
        )

        # main_c = assault → 3 个 1 星 = 1 对合成 + 1 落单
        candidates = snapshot.find_merge_candidates(main_c="assault")
        assert len(candidates) == 1
        assert candidates[0].is_main_c is True
        assert candidates[0].hero_a.hero_type == "assault"
        assert candidates[0].hero_a.star_level == 1

        # main_c = monkey → 同样 1 对，但 is_main_c=False
        candidates = snapshot.find_merge_candidates(main_c="monkey")
        assert len(candidates) == 1
        assert candidates[0].is_main_c is False

    def test_three_heros_equal_spacing(
        self,
        vision: Vision,
        frame: np.ndarray,
        assault_templates: list[HeroTemplate],
    ) -> None:
        """3 个强袭等间距排列（验证棋盘格子特征）。"""
        template_paths = [str(t.path) for t in assault_templates]
        results = vision.match_template_set(frame, template_paths, threshold=0.7)
        sorted_results = sorted(results, key=lambda r: r.position[0])
        xs = [r.position[0] for r in sorted_results]
        gap1 = xs[1] - xs[0]
        gap2 = xs[2] - xs[1]
        assert abs(gap1 - gap2) < 10, f"间距不等: {gap1} vs {gap2}"

    def test_star2_template_no_match(
        self,
        vision: Vision,
        frame: np.ndarray,
        assault_templates: list[HeroTemplate],
    ) -> None:
        """2 星模板在目标图里找不到（目标图只有 1 星强袭）。"""
        star2_templates = [t for t in assault_templates if t.star_level == 2]
        assert len(star2_templates) >= 1
        results = vision.match_template_set(
            frame, [str(t.path) for t in star2_templates], threshold=0.5
        )
        assert len(results) == 0

    def test_full_pipeline_annotated_image(
        self,
        vision: Vision,
        frame: np.ndarray,
        assault_templates: list[HeroTemplate],
    ) -> None:
        """完整链路 + 生成标注图（验证 annotate 可用）。"""
        template_paths = [str(t.path) for t in assault_templates]
        results = vision.match_template_set(frame, template_paths, threshold=0.7)
        annotated = vision.annotate(frame, results)

        # 标注图形状不变
        assert annotated.shape == frame.shape
        # 标注后和原图不同（画了矩形）
        assert not np.array_equal(annotated, frame)
