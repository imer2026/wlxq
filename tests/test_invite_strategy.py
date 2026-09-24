"""CultivationStrategy 邀请者模式单元测试（主C猴子，spec 2026-09-21）。

覆盖：阵容门禁、保留数合成、3星避让、变身碰辅助（spec §5.1-§5.3）。
"""

from __future__ import annotations

from wlxq_bot.config import MainCProfile, RunConfig
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
    filler = [
        _hero(f"filler{i}", 4, 500 + i * 10, 500) for i in range(12 - len(heroes))
    ]
    return _board([*heroes, *filler], total=12)


def _monkey_strategy(**overrides: object) -> CultivationStrategy:
    profile = MainCProfile(
        display_name="猴子",
        hero_template_dir="heroes/monkey",
        target_star_override=1,
        forbid_star3_merge=True,
        hero_keep_counts={"angel": 1, "snow": 1, "death_knight": 0},
        required_lineup_heroes=["angel", "snow"],
        transform_sacrifice_order=["death_knight", "snow"],
        **overrides,  # type: ignore[arg-type]
    )
    return CultivationStrategy(
        "monkey", RunConfig(wildcard_merge_heroes=["angel_phantom"]), profile
    )


def _monkey_board() -> BoardSnapshot:
    """门禁达标棋盘：1星猴子 + 1星天使 + 1星雪姬 + 1星死骑。"""
    return _board(
        [
            _hero("monkey", 1, 100, 100),
            _hero("angel", 1, 200, 100),
            _hero("snow", 1, 300, 100),
            _hero("death_knight", 1, 400, 100),
        ]
    )


class TestLineupGate:
    def test_ready_when_lineup_complete(self) -> None:
        assert _monkey_strategy().main_c_ready(_monkey_board()) is True

    def test_not_ready_when_angel_missing(self) -> None:
        board = _board(
            [_hero("monkey", 1, 100, 100), _hero("snow", 1, 300, 100)]
        )
        strategy = _monkey_strategy()
        assert strategy.main_c_ready(board) is False
        assert strategy.lineup_missing(board) == ["angel"]

    def test_not_ready_when_snow_missing(self) -> None:
        board = _board(
            [_hero("monkey", 1, 100, 100), _hero("angel", 1, 200, 100)]
        )
        assert _monkey_strategy().lineup_missing(board) == ["snow"]

    def test_assault_profile_has_no_lineup_gate(self) -> None:
        """强袭档案不配置阵容门禁：仅主C星级决定门禁（现状不变）。"""
        strategy = CultivationStrategy("assault", RunConfig(target_star_level=3), None)
        board = _board([_hero("assault", 3, 100, 100)])
        assert strategy.main_c_ready(board) is True
        assert strategy.required_lineup_heroes == ()

    def test_target_star_override(self) -> None:
        """猴子档案覆盖星级门禁为 1：1 星主C即达标（run 默认 2 不生效）。"""
        board = _board([_hero("monkey", 1, 100, 100)])
        assert _monkey_strategy().target_star_level == 1
        assert _monkey_strategy().has_target_main_c(board) is True


class TestKeepCountMerge:
    def test_three_snow_merge_two_keep_one(self) -> None:
        """3 个 1 星雪姬：可合并其中 2 个（剩 1 满足保留数）。"""
        board = _board(
            [
                _hero("monkey", 1, 100, 100),
                _hero("angel", 1, 200, 100),
                _hero("snow", 1, 300, 100),
                _hero("snow", 1, 400, 100),
                _hero("snow", 1, 500, 100),
            ]
        )
        candidates = board.find_merge_candidates("monkey")
        assert len(candidates) == 1
        pair = _monkey_strategy().select_merge(board, candidates, failed_pairs=set())
        assert pair is not None

    def test_two_snow_cannot_merge_keep_one(self) -> None:
        """只有 2 个雪姬：合并剩 0 跌破保留数 → 不合（返回 None 走召唤）。"""
        board = _board(
            [
                _hero("monkey", 1, 100, 100),
                _hero("angel", 1, 200, 100),
                _hero("snow", 1, 300, 100),
                _hero("snow", 1, 400, 100),
            ]
        )
        candidates = board.find_merge_candidates("monkey")
        assert len(candidates) == 1
        pair = _monkey_strategy().select_merge(board, candidates, failed_pairs=set())
        assert pair is None

    def test_death_knight_pairs_free_to_merge(self) -> None:
        """死骑保留数 0：2 个 1 星死骑可合并（腾格子）。"""
        board = _board(
            [
                _hero("monkey", 1, 100, 100),
                _hero("angel", 1, 200, 100),
                _hero("snow", 1, 300, 100),
                _hero("death_knight", 1, 400, 100),
                _hero("death_knight", 1, 500, 100),
            ]
        )
        candidates = board.find_merge_candidates("monkey")
        pair = _monkey_strategy().select_merge(board, candidates, failed_pairs=set())
        assert pair is not None
        assert pair.hero_a.hero_type == "death_knight"

    def test_wildcard_pair_consumes_only_non_wildcard_side(self) -> None:
        """通配对（幻天使×雪姬）：只检查雪姬侧保留数；仅 1 个雪姬时不合。"""
        board = _board(
            [
                _hero("monkey", 1, 100, 100),
                _hero("angel", 1, 200, 100),
                _hero("snow", 1, 300, 100),
                _hero("angel_phantom", 1, 400, 100),
            ]
        )
        candidates = board.find_merge_candidates("monkey", wildcard_heroes=WILDCARDS)
        assert any(c.is_wildcard for c in candidates)
        pair = _monkey_strategy().select_merge(board, candidates, failed_pairs=set())
        assert pair is None

    def test_wildcard_pair_allowed_with_spare_snow(self) -> None:
        """通配对：雪姬 2 个时花 1 个（通配侧不消耗本类）。"""
        board = _board(
            [
                _hero("monkey", 1, 100, 100),
                _hero("angel", 1, 200, 100),
                _hero("snow", 1, 300, 100),
                _hero("snow", 1, 350, 100),
                _hero("angel_phantom", 1, 400, 100),
            ]
        )
        candidates = board.find_merge_candidates("monkey", wildcard_heroes=WILDCARDS)
        assert any(c.is_wildcard for c in candidates)
        assert _monkey_strategy().select_merge(board, candidates, failed_pairs=set()) is None


class TestForbidStar3Merge:
    def test_two_star_pair_rejected_even_when_full(self) -> None:
        """2星对必产 3 星：占满也不合（强袭的「占满被迫合并」不生效）。"""
        heroes = [
            _hero("monkey", 1, 100, 100),
            _hero("angel", 1, 200, 100),
            _hero("snow", 1, 300, 100),
            _hero("death_knight", 2, 400, 100),
            _hero("death_knight", 2, 500, 100),
        ]
        board = _full_board(heroes)
        candidates = board.find_merge_candidates("monkey")
        assert len(candidates) == 1
        assert _monkey_strategy().select_merge(board, candidates, failed_pairs=set()) is None

    def test_star3_excluded_from_batch(self) -> None:
        """批次合成：2星对被 3 星避让排除。"""
        heroes = [
            _hero("monkey", 1, 100, 100),
            _hero("angel", 1, 200, 100),
            _hero("snow", 1, 300, 100),
            _hero("death_knight", 2, 400, 100),
            _hero("death_knight", 2, 500, 100),
        ]
        board = _full_board(heroes)
        candidates = board.find_merge_candidates("monkey")
        assert _monkey_strategy().select_merge_batch(candidates, set(), board) == []

    def test_assault_forced_merge_unchanged(self) -> None:
        """强袭（未启用 forbid_star3_merge）：占满时 2 星对仍被迫合并（现状）。"""
        heroes = [
            _hero("snow", 2, 100, 100),
            _hero("snow", 2, 200, 100),
        ]
        board = _full_board(heroes)
        candidates = board.find_merge_candidates("assault")
        assert len(candidates) == 1
        assert CultivationStrategy("assault", RunConfig(), None).select_merge(
            board, candidates, failed_pairs=set()
        ) is not None


class TestTransformAssist:
    def test_sacrifice_death_knight_first(self) -> None:
        """卡住出口：牺牲顺序死骑优先。"""
        board = _monkey_board()
        target = _monkey_strategy().select_transform_assist(board, 0, 2)
        assert target is not None
        assert target.hero_type == "death_knight"

    def test_sacrifice_snow_only_when_two(self) -> None:
        """雪姬仅在场 ≥2 才可牺牲：无死骑且只有 1 雪姬时无可牺牲。"""
        board = _board(
            [
                _hero("monkey", 1, 100, 100),
                _hero("angel", 1, 200, 100),
                _hero("snow", 1, 300, 100),
                _hero("filler", 4, 400, 100),
            ]
        )
        assert _monkey_strategy().select_transform_assist(board, 0, 2) is None

    def test_snow_sacrificed_when_count_two(self) -> None:
        """无死骑、雪姬 2 个：牺牲一个雪姬（剩 1 满足保留数）。"""
        board = _board(
            [
                _hero("monkey", 1, 100, 100),
                _hero("angel", 1, 200, 100),
                _hero("snow", 1, 300, 100),
                _hero("snow", 1, 350, 100),
            ]
        )
        target = _monkey_strategy().select_transform_assist(board, 0, 2)
        assert target is not None
        assert target.hero_type == "snow"

    def test_never_sacrifice_when_exhausted(self) -> None:
        """变身次数耗尽：恒 None。"""
        assert _monkey_strategy().select_transform_assist(_monkey_board(), 2, 2) is None

    def test_assault_mode_uses_legacy_transform(self) -> None:
        """强袭（空牺牲名单）：走 3 星非主C碰主C模式（现状）。"""
        strategy = CultivationStrategy("assault", RunConfig(), None)
        board = _board(
            [_hero("snow", 3, 100, 100), _hero("assault", 1, 200, 100)]
        )
        assert strategy.select_transform(board, 0, 2) is not None
        # 猴子策略的 select_transform 在碰辅助模式下不适用（返回 None）
        monkey = _monkey_strategy()
        assert monkey.select_transform(_monkey_board(), 0, 2) is None
