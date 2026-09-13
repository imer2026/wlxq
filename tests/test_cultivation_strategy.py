"""CultivationStrategy 合成选择单元测试（幻天使三级层级，2026-09-13 定案）。

对应 spec docs/superpowers/specs/2026-09-13-angel-phantom-split-design.md §4.5。
"""

from __future__ import annotations

from wlxq_bot.config import RunConfig
from wlxq_bot.models import BoardCapacity, BoardHero, BoardSnapshot
from wlxq_bot.tasks.strategy import CultivationStrategy

WILDCARDS = frozenset({"angel_phantom"})


def _hero(hero_type: str, star: int, x: int, y: int) -> BoardHero:
    return BoardHero(
        hero_type=hero_type,
        star_level=star,
        position=(x, y),
        confidence=0.9,
        template_path=f"heroes/{hero_type}/star{star}/test.png",
    )


def _board(heroes: list[BoardHero], total: int = 12) -> BoardSnapshot:
    return BoardSnapshot(
        frame_id=1,
        heroes=heroes,
        capacity=BoardCapacity(total_slots=total, occupied=len(heroes)),
    )


def _full_board(heroes: list[BoardHero]) -> BoardSnapshot:
    """构造占满棋盘：heroes 之外用互不可合的 4 星填充英雄补足 12 格。"""
    filler = [
        _hero(f"filler{i}", 4, 500 + i * 10, 500) for i in range(12 - len(heroes))
    ]
    return _board([*heroes, *filler], total=12)


def _strategy() -> CultivationStrategy:
    return CultivationStrategy("assault", RunConfig(), None)


class TestSelectMergeWildcard:
    def test_same_type_low_pair_merges_with_space(self) -> None:
        """同类项 1/2 星对：棋盘有空位也立即合并（既有行为）。"""
        board = _board(
            [_hero("snow", 1, 100, 100), _hero("snow", 1, 200, 100)], total=12
        )
        candidates = board.find_merge_candidates("assault")
        pair = _strategy().select_merge(board, candidates, failed_pairs=set())
        assert pair is not None
        assert pair.is_wildcard is False

    def test_wildcard_low_pair_deferred_when_board_not_full(self) -> None:
        """通配 1/2 星对：棋盘有空位 → 不合并（None，落入召唤分支）。"""
        board = _board(
            [_hero("angel_phantom", 1, 100, 100), _hero("snow", 1, 200, 100)],
            total=12,
        )
        candidates = board.find_merge_candidates("assault", wildcard_heroes=WILDCARDS)
        pair = _strategy().select_merge(board, candidates, failed_pairs=set())
        assert pair is None

    def test_wildcard_low_pair_used_when_board_full(self) -> None:
        """通配 1/2 星对：棋盘占满 → 合并。"""
        heroes = [_hero("angel_phantom", 1, 100, 100), _hero("snow", 1, 200, 100)]
        board = _full_board(heroes)
        candidates = board.find_merge_candidates("assault", wildcard_heroes=WILDCARDS)
        pair = _strategy().select_merge(board, candidates, failed_pairs=set())
        assert pair is not None
        assert pair.is_wildcard is True

    def test_wildcard_low_before_3star_when_full(self) -> None:
        """占满时通配 1/2 星对先于 3 星对（不触发赠送技能页）。"""
        heroes = [
            _hero("angel_phantom", 1, 100, 100),
            _hero("snow", 1, 200, 100),
            _hero("angel", 3, 300, 100),
            _hero("angel", 3, 400, 100),
        ]
        board = _full_board(heroes)
        candidates = board.find_merge_candidates("assault", wildcard_heroes=WILDCARDS)
        pair = _strategy().select_merge(board, candidates, failed_pairs=set())
        assert pair is not None
        assert pair.is_wildcard is True
        assert pair.hero_a.star_level == 1

    def test_wildcard_3star_deferred_when_not_full(self) -> None:
        """3 星通配对：有空位时同样召唤避开。"""
        board = _board(
            [
                _hero("angel_phantom", 3, 100, 100),
                _hero("snow", 3, 200, 100),
            ],
            total=12,
        )
        candidates = board.find_merge_candidates("assault", wildcard_heroes=WILDCARDS)
        pair = _strategy().select_merge(board, candidates, failed_pairs=set())
        assert pair is None


class TestSelectMergeBatchWildcard:
    def test_batch_includes_wildcard_when_full(self) -> None:
        """棋盘占满：通配对纳入批次，同类项在前。"""
        heroes = [
            _hero("snow", 1, 100, 100),
            _hero("snow", 1, 200, 100),
            _hero("angel_phantom", 1, 300, 100),
            _hero("death_knight", 1, 400, 100),
        ]
        board = _full_board(heroes)
        candidates = board.find_merge_candidates("assault", wildcard_heroes=WILDCARDS)
        batch = _strategy().select_merge_batch(candidates, failed_pairs=set(), board=board)
        assert len(batch) == 2
        assert batch[0].is_wildcard is False
        assert batch[1].is_wildcard is True

    def test_batch_excludes_wildcard_when_not_full(self) -> None:
        """棋盘未满：批次排除通配对，同类项对照常入批。"""
        heroes = [
            _hero("snow", 1, 100, 100),
            _hero("snow", 1, 200, 100),
            _hero("angel_phantom", 1, 300, 100),
            _hero("death_knight", 1, 400, 100),
        ]
        board = _board(heroes, total=12)
        candidates = board.find_merge_candidates("assault", wildcard_heroes=WILDCARDS)
        batch = _strategy().select_merge_batch(candidates, failed_pairs=set(), board=board)
        assert len(batch) == 1
        assert batch[0].is_wildcard is False

    def test_batch_never_main_c_never_3star(self) -> None:
        """批次既有过滤回归：主 C 对与 3 星对（含通配）不进批次。"""
        heroes = [
            _hero("assault", 1, 100, 100),
            _hero("assault", 1, 200, 100),
            _hero("snow", 3, 300, 100),
            _hero("snow", 3, 400, 100),
        ]
        board = _full_board(heroes)
        candidates = board.find_merge_candidates("assault")
        batch = _strategy().select_merge_batch(candidates, failed_pairs=set(), board=board)
        assert batch == []
