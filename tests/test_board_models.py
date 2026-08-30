"""棋盘数据模型单元测试。

覆盖 parse_hero_template_path / BoardHero / BoardCapacity /
MergeCandidate / BoardSnapshot（含 find_merge_candidates）。

对应 docs/game-rules.md「最小业务概念」和 docs/architecture.md「核心概念模型」。
"""

from __future__ import annotations

import pytest

from wlxq_bot.models import (
    BoardCapacity,
    BoardHero,
    BoardSnapshot,
    MatchResult,
    parse_hero_template_path,
)

# ---------------------------------------------------------------------------
# parse_hero_template_path
# ---------------------------------------------------------------------------


class TestParseHeroTemplatePath:
    def test_normal_path(self) -> None:
        path = "assets/templates/927x1727/heroes/assault/star1/left.png"
        hero_type, star = parse_hero_template_path(path)
        assert hero_type == "assault"
        assert star == 1

    def test_monkey_star3(self) -> None:
        path = "heroes/monkey/star3/action_02.png"
        hero_type, star = parse_hero_template_path(path)
        assert hero_type == "monkey"
        assert star == 3

    def test_star4(self) -> None:
        path = "templates/heroes/assault/star4/x.png"
        hero_type, star = parse_hero_template_path(path)
        assert hero_type == "assault"
        assert star == 4

    def test_path_object(self) -> None:
        from pathlib import Path

        hero_type, star = parse_hero_template_path(Path("heroes/assault/star2/y.png"))
        assert hero_type == "assault"
        assert star == 2

    def test_no_heroes_dir_raises(self) -> None:
        with pytest.raises(ValueError, match="不含 heroes"):
            parse_hero_template_path("assets/templates/927x1727/assault/star1/x.png")

    def test_star_dir_not_start_with_star_raises(self) -> None:
        with pytest.raises(ValueError, match="不以 star 开头"):
            parse_hero_template_path("heroes/assault/1/x.png")

    def test_star_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="超出 1-4"):
            parse_hero_template_path("heroes/assault/star5/x.png")

    def test_star_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="超出 1-4"):
            parse_hero_template_path("heroes/assault/star0/x.png")

    def test_malformed_star_dir_raises(self) -> None:
        with pytest.raises(ValueError, match="格式不符 star<N>"):
            parse_hero_template_path("heroes/assault/starX/x.png")


# ---------------------------------------------------------------------------
# BoardHero
# ---------------------------------------------------------------------------


class TestBoardHero:
    def test_from_match(self) -> None:
        match = MatchResult(
            template_name="heroes/assault/star1/x.png",
            position=(100, 200),
            confidence=0.92,
        )
        hero = BoardHero.from_match(match, "heroes/assault/star1/x.png")
        assert hero.hero_type == "assault"
        assert hero.star_level == 1
        assert hero.position == (100, 200)
        assert hero.confidence == 0.92
        assert hero.template_path == "heroes/assault/star1/x.png"

    def test_from_match_different_hero(self) -> None:
        match = MatchResult(
            template_name="heroes/monkey/star2/y.png",
            position=(300, 400),
            confidence=0.85,
        )
        hero = BoardHero.from_match(match, "heroes/monkey/star2/y.png")
        assert hero.hero_type == "monkey"
        assert hero.star_level == 2

    def test_frozen(self) -> None:
        """BoardHero 是 frozen dataclass，不可变。"""
        import dataclasses

        hero = BoardHero(
            hero_type="assault",
            star_level=1,
            position=(0, 0),
            confidence=1.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            hero.hero_type = "monkey"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BoardCapacity
# ---------------------------------------------------------------------------


class TestBoardCapacity:
    def test_available(self) -> None:
        cap = BoardCapacity(total_slots=6, occupied=3)
        assert cap.available == 3

    def test_available_zero(self) -> None:
        cap = BoardCapacity(total_slots=6, occupied=6)
        assert cap.available == 0

    def test_available_all_free(self) -> None:
        cap = BoardCapacity(total_slots=8, occupied=0)
        assert cap.available == 8


# ---------------------------------------------------------------------------
# BoardSnapshot.find_merge_candidates
# ---------------------------------------------------------------------------


def _hero(hero_type: str, star: int, x: int, y: int) -> BoardHero:
    """快速构造测试用 BoardHero。"""
    return BoardHero(
        hero_type=hero_type,
        star_level=star,
        position=(x, y),
        confidence=0.9,
        template_path=f"heroes/{hero_type}/star{star}/test.png",
    )


class TestFindMergeCandidates:
    def test_same_type_same_star_pairs(self) -> None:
        """同类型同星级 → 合法对。"""
        snapshot = BoardSnapshot(
            frame_id=1,
            heroes=[
                _hero("assault", 1, 100, 100),
                _hero("assault", 1, 200, 100),
            ],
            capacity=BoardCapacity(total_slots=6, occupied=2),
        )
        candidates = snapshot.find_merge_candidates(main_c="assault")
        assert len(candidates) == 1
        assert candidates[0].hero_a.hero_type == "assault"
        assert candidates[0].is_main_c is True

    def test_different_type_no_pair(self) -> None:
        """不同类型 → 不配对。"""
        snapshot = BoardSnapshot(
            frame_id=1,
            heroes=[
                _hero("assault", 1, 100, 100),
                _hero("monkey", 1, 200, 100),
            ],
            capacity=BoardCapacity(total_slots=6, occupied=2),
        )
        candidates = snapshot.find_merge_candidates(main_c="assault")
        assert len(candidates) == 0

    def test_different_star_no_pair(self) -> None:
        """同类型不同星级 → 不配对。"""
        snapshot = BoardSnapshot(
            frame_id=1,
            heroes=[
                _hero("assault", 1, 100, 100),
                _hero("assault", 2, 200, 100),
            ],
            capacity=BoardCapacity(total_slots=6, occupied=2),
        )
        candidates = snapshot.find_merge_candidates(main_c="assault")
        assert len(candidates) == 0

    def test_non_main_c_first(self) -> None:
        """多个候选同时存在时，非主 C 优先（排前面）。"""
        snapshot = BoardSnapshot(
            frame_id=1,
            heroes=[
                _hero("assault", 1, 100, 100),  # 主C对
                _hero("assault", 1, 200, 100),
                _hero("monkey", 1, 300, 100),  # 非主C对
                _hero("monkey", 1, 400, 100),
            ],
            capacity=BoardCapacity(total_slots=6, occupied=4),
        )
        candidates = snapshot.find_merge_candidates(main_c="assault")
        assert len(candidates) == 2
        # 非主C排前面
        assert candidates[0].is_main_c is False
        assert candidates[0].hero_a.hero_type == "monkey"
        assert candidates[1].is_main_c is True
        assert candidates[1].hero_a.hero_type == "assault"

    def test_already_paired_not_reused(self) -> None:
        """已配对的英雄不重复使用（3个同种只配出1对）。"""
        snapshot = BoardSnapshot(
            frame_id=1,
            heroes=[
                _hero("assault", 1, 100, 100),
                _hero("assault", 1, 200, 100),
                _hero("assault", 1, 300, 100),  # 第三个落单
            ],
            capacity=BoardCapacity(total_slots=6, occupied=3),
        )
        candidates = snapshot.find_merge_candidates(main_c="assault")
        assert len(candidates) == 1

    def test_empty_board(self) -> None:
        """空棋盘返回空列表。"""
        snapshot = BoardSnapshot(
            frame_id=1,
            heroes=[],
            capacity=BoardCapacity(total_slots=6, occupied=0),
        )
        candidates = snapshot.find_merge_candidates(main_c="assault")
        assert candidates == []

    def test_two_pairs_same_type(self) -> None:
        """同类型 4 个英雄 → 配出 2 对。"""
        snapshot = BoardSnapshot(
            frame_id=1,
            heroes=[
                _hero("assault", 1, 100, 100),
                _hero("assault", 1, 200, 100),
                _hero("assault", 1, 300, 100),
                _hero("assault", 1, 400, 100),
            ],
            capacity=BoardCapacity(total_slots=6, occupied=4),
        )
        candidates = snapshot.find_merge_candidates(main_c="monkey")
        assert len(candidates) == 2
        # 都是非主C（main_c=monkey，但英雄是assault）
        assert all(c.is_main_c is False for c in candidates)

    def test_is_main_c_flag(self) -> None:
        """is_main_c 标记正确：英雄类型 == main_c 时为 True。"""
        snapshot = BoardSnapshot(
            frame_id=1,
            heroes=[
                _hero("assault", 1, 100, 100),
                _hero("assault", 1, 200, 100),
            ],
            capacity=BoardCapacity(total_slots=6, occupied=2),
        )
        # main_c 是 assault → is_main_c=True
        candidates = snapshot.find_merge_candidates(main_c="assault")
        assert candidates[0].is_main_c is True

        # main_c 不是 assault → is_main_c=False
        candidates = snapshot.find_merge_candidates(main_c="monkey")
        assert candidates[0].is_main_c is False
