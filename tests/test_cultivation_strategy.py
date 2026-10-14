"""CultivationStrategy 合成选择单元测试（幻天使三级层级，2026-09-13 定案）。

对应 spec docs/superpowers/specs/2026-09-13-angel-phantom-split-design.md §4.5。
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


class TestStar3TopupStage:
    """3星主C培养环节（2026-09-30 用户定案，非强制追加）。

    数量回补到 topup_hero_count 后追加：合成 2 星对碰 3 星（类型随机）+
    变身 3 星非主C 转化；达成 / 变身用尽（整局预算归零即退，2026-10-05
    单门定案）/ 死局即退出。
    """

    def _star3_strategy(self, **profile_fields) -> CultivationStrategy:
        fields = {
            "display_name": "强袭",
            "hero_template_dir": "heroes/assault",
            "avoid_main_c_merge": True,
            "topup_after_skill_selections": 4,
            "topup_hero_count": 4,
            "star3_topup_enabled": True,
        }
        fields.update(profile_fields)
        return CultivationStrategy(
            "assault", RunConfig(), MainCProfile(**fields)
        )

    def _exit(
        self,
        strategy: CultivationStrategy,
        board: BoardSnapshot,
        *,
        transform_used: int = 0,
        failed_targets: set[tuple[str, int, tuple[int, int]]] = frozenset(),
    ) -> tuple[bool, str]:
        candidates = board.find_merge_candidates(
            "assault", wildcard_heroes=WILDCARDS
        )
        return strategy.star3_stage_exit(
            board,
            candidates,
            failed_pairs=set(),
            transform_used=transform_used,
            failed_targets=failed_targets,
        )

    def test_enabled_requires_profile_flag(self) -> None:
        """star3_topup_enabled 只由档案字段决定，无档案/未配置时关闭。"""
        assert CultivationStrategy("assault", RunConfig(), None).star3_topup_enabled is False
        profile = MainCProfile(
            display_name="强袭", hero_template_dir="heroes/assault"
        )
        assert CultivationStrategy("assault", RunConfig(), profile).star3_topup_enabled is False
        profile.star3_topup_enabled = True
        assert CultivationStrategy("assault", RunConfig(), profile).star3_topup_enabled is True

    def test_has_star3_main_c(self) -> None:
        """达成检测：≥3 星主C 在场为 True，低星主C/其他英雄 3 星不算。"""
        strategy = self._star3_strategy()
        assert strategy.has_star3_main_c(_board([_hero("assault", 3, 100, 100)])) is True
        assert strategy.has_star3_main_c(_board([_hero("assault", 2, 100, 100)])) is False
        assert strategy.has_star3_main_c(_board([_hero("snow", 3, 100, 100)])) is False

    def test_exit_on_star3_main_c(self) -> None:
        """达成：场上出现 3 星主C，无论其他条件立即退出。"""
        board = _board(
            [
                _hero("assault", 3, 100, 100),
                _hero("snow", 1, 200, 100),
                _hero("snow", 1, 300, 100),
            ]
        )
        exit_now, reason = self._exit(self._star3_strategy(), board)
        assert exit_now is True
        assert "3星" in reason

    def test_exit_when_transform_exhausted(self) -> None:
        """放弃·变身用尽：整局预算归零即退（2026-10-05 单门定案）。"""
        board = _board([_hero("assault", 2, 100, 100)])
        exit_now, reason = self._exit(
            self._star3_strategy(),
            board,
            transform_used=2,
        )
        assert exit_now is True
        assert "变身" in reason

    def test_exit_when_transform_exhausted_before_entry(self) -> None:
        """进场前预算已用完同样立刻退（2026-10-05 定案，推翻 9-30 环节内
        门槛——实机局 6：空预算留场培养退化成无限召唤）。"""
        board = _board(
            [
                _hero("assault", 2, 100, 100),
                _hero("snow", 1, 200, 100),
                _hero("snow", 1, 300, 100),
            ]
        )
        exit_now, reason = self._exit(
            self._star3_strategy(),
            board,
            transform_used=2,
        )
        assert exit_now is True
        assert "变身" in reason

    def test_continue_with_non_main_pair(self) -> None:
        """有非主C对可合：继续培养。"""
        board = _board(
            [
                _hero("assault", 2, 100, 100),
                _hero("snow", 1, 200, 100),
                _hero("snow", 1, 300, 100),
            ]
        )
        exit_now, _ = self._exit(self._star3_strategy(), board)
        assert exit_now is False

    def test_exit_dead_board(self) -> None:
        """放弃·死局：无非主C对 + 无空位 + 无可变身对象。"""
        # 满员小棋盘（3/3）：主C不同星不成对，death_knight 1星非变身对象
        heroes = [
            _hero("assault", 2, 100, 100),
            _hero("assault", 1, 200, 100),
            _hero("death_knight", 1, 300, 100),
        ]
        board = _board(heroes, total=3)
        exit_now, reason = self._exit(self._star3_strategy(), board)
        assert exit_now is True
        assert "死局" in reason

    def test_continue_dead_board_with_transform_target_and_budget(self) -> None:
        """占满无对但有 3 星非主C + 剩余变身预算：还能变身，不算死局。"""
        heroes = [
            _hero("assault", 2, 100, 100),
            _hero("assault", 1, 200, 100),
            _hero("snow", 3, 300, 100),
        ]
        board = _full_board(heroes)
        exit_now, _ = self._exit(self._star3_strategy(), board)
        assert exit_now is False

    def test_exit_dead_board_when_transform_target_given_up(self) -> None:
        """唯一的可变身对象被按英雄放弃记忆记过（按钮超时等失败）：
        无可变身对象，死局退出，不干等（2026-10-03 记法从整盘签名改按目标）。
        填充用互不同类 2 星（无可合对也够不到变身星级）。"""
        heroes = [
            _hero("assault", 2, 100, 100),
            _hero("assault", 1, 200, 100),
            _hero("snow", 3, 300, 100),
            *[_hero(f"filler{i}", 2, 400 + i * 10, 100) for i in range(9)],
        ]
        board = _board(heroes, total=12)
        exit_now, _ = self._exit(
            self._star3_strategy(), board, failed_targets={("snow", 3, (300, 100))}
        )
        assert exit_now is True

    def test_not_dead_when_other_target_not_given_up(self) -> None:
        """多个可变身对象只记过一个：其余目标仍可变身，不算死局。"""
        heroes = [
            _hero("assault", 2, 100, 100),
            _hero("assault", 1, 200, 100),
            _hero("snow", 3, 300, 100),
            _hero("angel", 3, 400, 100),
            *[_hero(f"filler{i}", 2, 500 + i * 10, 100) for i in range(8)],
        ]
        board = _board(heroes, total=12)
        exit_now, _ = self._exit(
            self._star3_strategy(), board, failed_targets={("snow", 3, (300, 100))}
        )
        assert exit_now is False

    def test_dead_board_ignores_failed_pairs(self) -> None:
        """死局判定排除失败记忆中的对：只剩拖不动的对视作无对。"""
        heroes = [
            _hero("assault", 2, 100, 100),
            _hero("assault", 1, 200, 100),
            _hero("snow", 1, 300, 100),
            _hero("snow", 1, 400, 100),
        ]
        board = _board(heroes, total=4)
        candidates = board.find_merge_candidates("assault")
        failed = {CultivationStrategy.pair_signature(c) for c in candidates}
        exit_now, reason = self._star3_strategy().star3_stage_exit(
            board,
            candidates,
            failed_pairs=failed,
            transform_used=0,
            failed_targets=frozenset(),
        )
        assert exit_now is True
        assert "死局" in reason

    def test_select_merge_star3_blocks_main_pair_without_surplus(self) -> None:
        """star3_stage 无富余（2 个主C）：主C对出局，有空位 → 不合并（召唤）。"""
        board = _board(
            [
                _hero("assault", 1, 100, 100),
                _hero("assault", 1, 200, 100),
            ]
        )
        candidates = board.find_merge_candidates("assault")
        pair = self._star3_strategy().select_merge(
            board, candidates, failed_pairs=set(), star3_stage=True
        )
        assert pair is None

    def test_select_merge_star3_no_forced_main_merge_without_surplus(self) -> None:
        """star3_stage 占满只剩主C对且无富余：也不被迫合并（保 ≥4，走死局退出）。"""
        heroes = [
            _hero("assault", 2, 100, 100),
            _hero("assault", 1, 200, 100),
        ]
        board = _full_board(heroes)
        candidates = board.find_merge_candidates("assault")
        pair = self._star3_strategy().select_merge(
            board, candidates, failed_pairs=set(), star3_stage=True
        )
        assert pair is None

    def test_select_merge_star3_keeps_non_main(self) -> None:
        """star3_stage：非主C对照常可选。"""
        board = _board(
            [
                _hero("assault", 1, 100, 100),
                _hero("snow", 1, 200, 100),
                _hero("snow", 1, 300, 100),
            ]
        )
        candidates = board.find_merge_candidates("assault")
        pair = self._star3_strategy().select_merge(
            board, candidates, failed_pairs=set(), star3_stage=True
        )
        assert pair is not None
        assert pair.is_main_c is False

    def test_select_merge_star3_surplus_main_pair_merges(self) -> None:
        """star3_stage 富余主C对可用（2026-10-01 用户修正：保留 4 个就行）：
        6 个强袭合 1 对最坏剩 4（合成类型随机，产物可能不是强袭）。"""
        heroes = [
            _hero("assault", 2, 100, 100),
            _hero("assault", 2, 200, 100),
            _hero("assault", 1, 300, 100),
            _hero("assault", 1, 400, 100),
            _hero("assault", 1, 500, 100),
            _hero("assault", 1, 600, 100),
        ]
        board = _board(heroes, total=12)  # 有空位也要合（环节合并优先）
        candidates = board.find_merge_candidates("assault")
        pair = self._star3_strategy().select_merge(
            board, candidates, failed_pairs=set(), star3_stage=True
        )
        assert pair is not None
        assert pair.is_main_c is True

    def test_select_merge_star3_surplus_threshold(self) -> None:
        """富余边界（按最坏情况，产物类型随机）：5 个（5-2=3 < 4）不合，
        6 个（6-2=4 ≥ 4）可合。"""
        five = [
            _hero("assault", 1, 100 + i * 10, 100) for i in range(5)
        ]
        board = _board(five, total=12)
        candidates = board.find_merge_candidates("assault")
        assert (
            self._star3_strategy().select_merge(
                board, candidates, failed_pairs=set(), star3_stage=True
            )
            is None
        )
        six = [
            _hero("assault", 1, 100 + i * 10, 100) for i in range(6)
        ]
        board = _board(six, total=12)
        candidates = board.find_merge_candidates("assault")
        assert (
            self._star3_strategy().select_merge(
                board, candidates, failed_pairs=set(), star3_stage=True
            )
            is not None
        )

    def test_exit_not_dead_with_surplus_main_pairs(self) -> None:
        """占满 + 仅剩富余主C对：可合并碰 3 星，不算死局（实机 frame=150）。"""
        heroes = [
            _hero("assault", 2, 100, 100),
            _hero("assault", 2, 200, 100),
            _hero("assault", 2, 300, 100),
            _hero("assault", 1, 400, 100),
            _hero("assault", 1, 500, 100),
            _hero("assault", 1, 600, 100),
            _hero("assault", 1, 700, 100),
            _hero("snow", 2, 800, 100),
            _hero("angel", 1, 900, 100),
        ]
        board = _board(heroes, total=9)
        exit_now, _ = self._exit(self._star3_strategy(), board)
        assert exit_now is False

    def test_phantom_main_pair_consumes_one(self) -> None:
        """幻天使+主C 只消耗 1 只主C（2026-10-02 用户定案）：5 只可合、
        4 只不合——原先按消耗 2 拦，5 只满盘会白白死局。"""
        five = [
            _hero("assault", 1, 100 + i * 10, 100) for i in range(5)
        ] + [_hero("angel_phantom", 1, 400, 100)]
        board = _full_board(five)
        candidates = board.find_merge_candidates("assault", wildcard_heroes=WILDCARDS)
        strategy = self._star3_strategy()
        pair = strategy.select_merge(
            board, candidates, failed_pairs=set(), star3_stage=True
        )
        assert pair is not None and pair.is_wildcard
        exit_now, _ = self._exit(strategy, board)
        assert exit_now is False

        four = [
            _hero("assault", 1, 100 + i * 10, 100) for i in range(4)
        ] + [_hero("angel_phantom", 1, 400, 100)]
        board = _full_board(four)
        candidates = board.find_merge_candidates("assault", wildcard_heroes=WILDCARDS)
        assert (
            strategy.select_merge(
                board, candidates, failed_pairs=set(), star3_stage=True
            )
            is None
        )

    def test_star3_pair_clears_star2_main_c_predicate(self) -> None:
        """清零判据：同类 2 星对=场上恰 2 只；幻天使+主C=恰 1 只；1 星对恒否。"""
        strategy = self._star3_strategy()

        two = _board(
            [_hero("assault", 2, 100, 100), _hero("assault", 2, 200, 100)]
        )
        pair = two.find_merge_candidates("assault")[0]
        assert strategy.star3_pair_clears_star2_main_c(two, pair) is True

        three = _board(
            [
                _hero("assault", 2, 100, 100),
                _hero("assault", 2, 200, 100),
                _hero("assault", 2, 300, 100),
            ]
        )
        pair3 = three.find_merge_candidates("assault")[0]
        assert strategy.star3_pair_clears_star2_main_c(three, pair3) is False

        ones = _board(
            [_hero("assault", 1, 100, 100), _hero("assault", 1, 200, 100)]
        )
        pair1 = ones.find_merge_candidates("assault")[0]
        assert strategy.star3_pair_clears_star2_main_c(ones, pair1) is False

        wildcard = _board(
            [
                _hero("assault", 2, 100, 100),
                _hero("angel_phantom", 2, 200, 100),
            ]
        )
        wpair = wildcard.find_merge_candidates(
            "assault", wildcard_heroes=WILDCARDS
        )[0]
        assert strategy.star3_pair_clears_star2_main_c(wildcard, wpair) is True

    def test_star3_clearing_star2_pair_yields_to_star1(self) -> None:
        """2星主C恰好2只（合成即清零）且有1星对：先合1星对攒（实机 frame=816）。"""
        heroes = [
            _hero("assault", 2, 100, 100),
            _hero("assault", 2, 200, 100),
            _hero("assault", 1, 300, 100),
            _hero("assault", 1, 400, 100),
            _hero("assault", 1, 500, 100),
            _hero("assault", 1, 600, 100),
        ]
        board = _board(heroes, total=12)
        candidates = board.find_merge_candidates("assault")
        pair = self._star3_strategy().select_merge(
            board, candidates, failed_pairs=set(), star3_stage=True
        )
        assert pair is not None
        assert pair.hero_a.star_level == 1 and pair.hero_b.star_level == 1

    def test_star3_star2_pair_preferred_when_not_clearing(self) -> None:
        """场上 ≥3 只 2 星主C：2 星对合成后仍有剩余，优先合（尽快碰 3 星）。"""
        heroes = [
            _hero("assault", 1, 100, 100),
            _hero("assault", 1, 200, 100),
            _hero("assault", 1, 300, 100),
            _hero("assault", 2, 400, 100),
            _hero("assault", 2, 500, 100),
            _hero("assault", 2, 600, 100),
        ]
        board = _board(heroes, total=12)
        candidates = board.find_merge_candidates("assault")
        pair = self._star3_strategy().select_merge(
            board, candidates, failed_pairs=set(), star3_stage=True
        )
        assert pair is not None
        assert pair.hero_a.star_level == 2 and pair.hero_b.star_level == 2

    def test_star3_clearing_pair_still_merges_when_alone(self) -> None:
        """清零的 2 星对无让位对象（1 星对本轮已拖动失败）时照合——让位是排序不是禁止。"""
        heroes = [
            _hero("assault", 2, 100, 100),
            _hero("assault", 2, 200, 100),
            _hero("assault", 1, 300, 100),
            _hero("assault", 1, 400, 100),
            _hero("assault", 1, 500, 100),
            _hero("assault", 1, 600, 100),
        ]
        board = _board(heroes, total=12)
        candidates = board.find_merge_candidates("assault")
        failed = {
            CultivationStrategy.pair_signature(c)
            for c in candidates
            if c.hero_a.star_level == 1
        }
        pair = self._star3_strategy().select_merge(
            board, candidates, failed_pairs=failed, star3_stage=True
        )
        assert pair is not None
        assert pair.hero_a.star_level == 2 and pair.hero_b.star_level == 2

    def test_star3_non_main_pairs_keep_priority_over_reorder(self) -> None:
        """非主C材料对绝对优先，清零让位排序只作用于主C对之间。"""
        heroes = [
            _hero("snow", 1, 100, 100),
            _hero("snow", 1, 200, 100),
            _hero("assault", 2, 300, 100),
            _hero("assault", 2, 400, 100),
            _hero("assault", 1, 500, 100),
            _hero("assault", 1, 600, 100),
            _hero("assault", 1, 700, 100),
            _hero("assault", 1, 800, 100),
        ]
        board = _board(heroes, total=12)
        candidates = board.find_merge_candidates("assault")
        pair = self._star3_strategy().select_merge(
            board, candidates, failed_pairs=set(), star3_stage=True
        )
        assert pair is not None
        assert pair.is_main_c is False


class TestTopupAfterStage:
    """凑 lineup 英雄子环节（3星环节出口追加，2026-10-01 用户定案）。"""

    def _after_strategy(self, **profile_fields) -> CultivationStrategy:
        fields = {
            "display_name": "强袭",
            "hero_template_dir": "heroes/assault",
            "topup_hero_count": 4,
            "star3_topup_enabled": True,
            "star3_topup_after_hero": "fox",
        }
        fields.update(profile_fields)
        return CultivationStrategy("assault", RunConfig(), MainCProfile(**fields))

    def _after_exit(
        self,
        strategy: CultivationStrategy,
        board: BoardSnapshot,
        *,
        transform_usable: bool = False,
        failed_targets: set[tuple[str, int, tuple[int, int]]] = frozenset(),
    ) -> tuple[bool, str]:
        candidates = board.find_merge_candidates("assault", wildcard_heroes=WILDCARDS)
        return strategy.topup_after_stage_exit(
            board,
            candidates,
            failed_pairs=set(),
            transform_usable=transform_usable,
            failed_targets=failed_targets,
        )

    def test_exit_done_when_hero_present(self) -> None:
        """达成：场上 ≥1 只目标英雄（任意星级）即退出。"""
        board = _board(
            [_hero("assault", 3, 100, 100), _hero("fox", 1, 200, 100)]
        )
        exit_now, reason = self._after_exit(self._after_strategy(), board)
        assert exit_now is True
        assert "凑fox" in reason

    def test_continue_with_space(self) -> None:
        """无目标英雄 + 有空位：继续（召唤）。"""
        board = _board(
            [_hero("assault", 3, 100, 100), _hero("angel", 1, 200, 100)]
        )
        exit_now, _ = self._after_exit(self._after_strategy(), board)
        assert exit_now is False

    def test_continue_with_allowed_pair_when_full(self) -> None:
        """满盘但有准入对（天使对）：继续（合成腾格）。"""
        heroes = [
            _hero("assault", 3, 100, 100),
            _hero("assault", 1, 200, 100),
            _hero("assault", 1, 300, 100),
            _hero("assault", 1, 400, 100),
            _hero("angel", 1, 500, 100),
            _hero("angel", 1, 600, 100),
        ]
        board = _full_board(heroes)
        exit_now, _ = self._after_exit(self._after_strategy(), board)
        assert exit_now is False

    def test_surplus_main_pairs_continue_when_full(self) -> None:
        """富余主C对放行（2026-10-02 用户定案）：强袭 6 只继续凑（产物
        3星随机有 1/4 是目标英雄）；5 只不合仍放弃；幻天使+主C 5 只继续。"""
        strategy = self._after_strategy()
        # 强袭 6 只（含 assault2 对，消耗 2 后剩 4）：继续
        six = [
            _hero("assault", 2, 100 + i * 10, 100) for i in range(6)
        ]
        board = _full_board(six)
        exit_now, reason = self._after_exit(strategy, board)
        assert exit_now is False
        # 强袭 5 只：对消耗 2 后剩 3 破底线，无其他准入对 → 放弃
        five = [
            _hero("assault", 2, 100 + i * 10, 100) for i in range(5)
        ] + [_hero("angel", 2, 500, 100)]
        board = _full_board(five)
        exit_now, reason = self._after_exit(strategy, board)
        assert exit_now is True
        assert "放弃凑fox" in reason
        # 幻天使+主C 只消耗 1：5 只继续
        five_phantom = [
            _hero("assault", 1, 100 + i * 10, 100) for i in range(5)
        ] + [_hero("angel_phantom", 1, 500, 100)]
        board = _full_board(five_phantom)
        exit_now, reason = self._after_exit(strategy, board)
        assert exit_now is False

    def test_exit_giveup_when_full_no_pair_no_transform(self) -> None:
        """放弃：满盘 + 无准入对 + 变身不可用。"""
        heroes = [
            _hero("assault", 3, 100, 100),
            _hero("assault", 1, 200, 100),
            _hero("assault", 1, 300, 100),
            _hero("assault", 1, 400, 100),
            _hero("angel", 1, 500, 100),
            _hero("death_knight", 2, 600, 100),
        ]
        board = _full_board(heroes)
        exit_now, reason = self._after_exit(self._after_strategy(), board)
        assert exit_now is True
        assert "放弃凑fox" in reason

    def test_transform_usable_continues_only_with_target(self) -> None:
        """满盘无对 + 变身可用：有目标英雄之外的可变身对象才继续。"""
        strategy = self._after_strategy()
        # 有 3 星死骑（≥ min_star）：可变身碰狐狸，继续
        heroes = [
            _hero("assault", 3, 100, 100),
            _hero("death_knight", 3, 200, 100),
        ]
        board = _full_board(heroes)
        exit_now, reason = self._after_exit(strategy, board, transform_usable=True)
        assert exit_now is False
        assert "变身" in reason
        # 只有 <min_star 的杂兵（无可变身对象；填充用互不同类 2 星，既无对
        # 也够不到变身星级）：放弃
        heroes = [
            _hero("assault", 3, 100, 100),
            _hero("death_knight", 1, 200, 100),
            *[_hero(f"filler{i}", 2, 300 + i * 10, 100) for i in range(10)],
        ]
        board = _board(heroes, total=12)
        exit_now, _ = self._after_exit(strategy, board, transform_usable=True)
        assert exit_now is True

    def test_given_up_target_blocks_transform_continue(self) -> None:
        """按英雄放弃记忆（2026-10-03）：唯一的可变身对象被记过 → 放弃退出，
        不再以「变身可用」继续空转；未记过的目标不受影响。"""
        strategy = self._after_strategy()
        heroes = [
            _hero("assault", 3, 100, 100),
            _hero("death_knight", 3, 200, 100),
            _hero("snow", 3, 300, 100),
            *[_hero(f"filler{i}", 2, 400 + i * 10, 100) for i in range(9)],
        ]
        board = _board(heroes, total=12)
        # 只记 snow3：死骑3 仍可变身 → 继续
        exit_now, reason = self._after_exit(
            strategy,
            board,
            transform_usable=True,
            failed_targets={("snow", 3, (300, 100))},
        )
        assert exit_now is False
        assert "变身" in reason
        # 两个目标都记过：无可变身对象 → 放弃
        exit_now, reason = self._after_exit(
            strategy,
            board,
            transform_usable=True,
            failed_targets={
                ("snow", 3, (300, 100)),
                ("death_knight", 3, (200, 100)),
            },
        )
        assert exit_now is True
        assert "放弃凑fox" in reason

    def test_pair_allowed_excludes_main_c_and_target(self) -> None:
        """准入对过滤：不含目标英雄不合（防合掉）；含主C按消耗数判富余
        （2026-10-02 用户定案：主C+主C ≥6 可合、幻天使+主C ≥5 可合）。"""
        from wlxq_bot.models import MergeCandidate

        strategy = self._after_strategy()

        def _pair(a_type: str, a_star: int, b_type: str, b_star: int) -> MergeCandidate:
            return MergeCandidate(
                hero_a=_hero(a_type, a_star, 100, 100),
                hero_b=_hero(b_type, b_star, 200, 100),
                is_main_c="assault" in (a_type, b_type),
                is_wildcard="angel_phantom" in (a_type, b_type),
            )

        def _board_with(extra: list[BoardHero]) -> BoardSnapshot:
            return _board(extra, total=12)

        # 天使对 / 死骑对：与强袭数量无关，准入
        assert (
            strategy.topup_after_pair_allowed(
                _board_with([]), _pair("angel", 1, "angel", 1)
            )
            is True
        )
        assert (
            strategy.topup_after_pair_allowed(
                _board_with([_hero("death_knight", 1, 300, 100)]),
                _pair("death_knight", 2, "death_knight", 2),
            )
            is True
        )
        # 含目标英雄的对（含通配配对）：不合
        assert (
            strategy.topup_after_pair_allowed(
                _board_with([]), _pair("fox", 1, "fox", 1)
            )
            is False
        )
        assert (
            strategy.topup_after_pair_allowed(
                _board_with([_hero("angel_phantom", 1, 300, 100)]),
                _pair("angel_phantom", 1, "fox", 1),
            )
            is False
        )
        # 主C+主C：消耗 2 → 数量 ≥6 可合，5 只不合（对内英雄在场计入数量）
        at_six = _board_with([_hero("assault", 1, 300, 100) for _ in range(6)])
        assert (
            strategy.topup_after_pair_allowed(
                at_six, _pair("assault", 2, "assault", 2)
            )
            is True
        )
        at_five = _board_with([_hero("assault", 1, 300, 100) for _ in range(5)])
        assert (
            strategy.topup_after_pair_allowed(
                at_five, _pair("assault", 2, "assault", 2)
            )
            is False
        )
        # 幻天使+主C：只消耗 1 → 数量 ≥5 可合，4 只不合
        at_five = _board_with(
            [_hero("assault", 1, 300, 100) for _ in range(5)]
            + [_hero("angel_phantom", 1, 400, 100)]
        )
        assert (
            strategy.topup_after_pair_allowed(
                at_five, _pair("angel_phantom", 1, "assault", 1)
            )
            is True
        )
        at_four = _board_with(
            [_hero("assault", 1, 300, 100) for _ in range(4)]
            + [_hero("angel_phantom", 1, 400, 100)]
        )
        assert (
            strategy.topup_after_pair_allowed(
                at_four, _pair("angel_phantom", 1, "assault", 1)
            )
            is False
        )

    def test_select_transform_excludes_target_hero(self) -> None:
        """变身目标排除：目标英雄在场前的边界——不会把仅有的狐狸变走。"""
        strategy = self._after_strategy()
        board = _board(
            [
                _hero("fox", 3, 100, 100),
                _hero("death_knight", 3, 200, 100),
            ]
        )
        target = strategy.select_transform(board, 0, 2, exclude_heroes=("fox",))
        assert target is not None
        assert target.hero_type == "death_knight"
        # 不排除时按格位顺序取第一个（狐狸）
        target = strategy.select_transform(board, 0, 2)
        assert target is not None
        assert target.hero_type == "fox"

    def test_select_transform_skips_given_up_target(self) -> None:
        """按英雄放弃记忆（2026-10-03 用户定案）：记过的目标本局不再选，
        落到下一个候选；全部记过返回 None（按钮超时是唯一可观测信号——
        皮肤升级的英雄点击永不弹按钮，外观与常规产出无差别）。"""
        strategy = self._after_strategy()
        board = _board(
            [
                _hero("fox", 3, 100, 100),
                _hero("death_knight", 3, 200, 100),
            ]
        )
        target = strategy.select_transform(
            board,
            0,
            2,
            exclude_targets={("fox", 3, (100, 100))},
        )
        assert target is not None
        assert target.hero_type == "death_knight"
        assert (
            strategy.select_transform(
                board,
                0,
                2,
                exclude_targets={
                    ("fox", 3, (100, 100)),
                    ("death_knight", 3, (200, 100)),
                },
            )
            is None
        )
        # 同格换英雄（旧英雄被合掉/升星）键不匹配：新英雄不受旧记录影响
        board2 = _board([_hero("angel", 3, 100, 100)])
        assert (
            strategy.select_transform(
                board2, 0, 2, exclude_targets={("fox", 3, (100, 100))}
            )
            is not None
        )

    def test_after_hero_empty_disables(self) -> None:
        """未配置目标英雄：不启用子环节。"""
        strategy = self._after_strategy(star3_topup_after_hero="")
        assert strategy.star3_topup_after_hero == ""


class TestWildcardGoldWaitException:
    """金币不足例外（2026-10-01 用户定案）：召唤被金币门控拦住时，
    「有空位→召唤不花幻天使」的保留前提不成立，幻天使通配对让位可合。"""

    def test_wildcard_low_pair_used_with_gold_wait_flag(self) -> None:
        """带 allow_wildcard_with_space=True：有空位也返回幻天使对；
        默认（金币可召唤）行为不变——保留幻天使返回 None。"""
        board = _board(
            [_hero("angel_phantom", 1, 100, 100), _hero("snow", 1, 200, 100)],
            total=12,
        )
        candidates = board.find_merge_candidates("assault", wildcard_heroes=WILDCARDS)
        assert (
            _strategy().select_merge(board, candidates, failed_pairs=set()) is None
        )
        pair = _strategy().select_merge(
            board, candidates, failed_pairs=set(), allow_wildcard_with_space=True
        )
        assert pair is not None
        assert pair.is_wildcard is True

    def test_gold_wait_flag_keeps_3star_avoidance(self) -> None:
        """例外只放开幻天使空位保留：仅剩 3 星对时有空位照样不合
        （4 星产物弹赠送技能页的问题与金币无关）。"""
        board = _board(
            [_hero("angel", 3, 100, 100), _hero("angel", 3, 200, 100)],
            total=12,
        )
        candidates = board.find_merge_candidates("assault")
        pair = _strategy().select_merge(
            board, candidates, failed_pairs=set(), allow_wildcard_with_space=True
        )
        assert pair is None


class TestDeadlockTransform:
    """卡死解锁低星变身目标选择（2026-10-01 用户定案）。

    候选池：通配英雄永不出、主C 仅数量 ≥ topup_hero_count+1 入、
    ≥transform_min_star_level 不入；星级在候选内数多数派、平局取 2星、
    某档为 0 落另一档；同档按格位顺序取第一只。
    """

    def _unlock_strategy(self, **profile_fields) -> CultivationStrategy:
        fields = {
            "display_name": "强袭",
            "hero_template_dir": "heroes/assault",
            "topup_hero_count": 4,
            "star3_topup_enabled": True,
        }
        fields.update(profile_fields)
        return CultivationStrategy("assault", RunConfig(), MainCProfile(**fields))

    def _unlock(
        self,
        strategy: CultivationStrategy,
        board: BoardSnapshot,
        *,
        exclude_heroes: tuple[str, ...] = (),
        exclude_targets: set[tuple[str, int, tuple[int, int]]] = frozenset(),
    ) -> BoardHero | None:
        return strategy.select_deadlock_transform(
            board,
            wildcard_heroes=WILDCARDS,
            exclude_heroes=exclude_heroes,
            exclude_targets=exclude_targets,
        )

    def test_frame1199_replica_picks_majority_star(self) -> None:
        """实机 frame 1199 复刻：强袭 4 只不入池，候选 2星 3 只 / 1星
        2 只 → 变 2星，格位序取第一只（死骑）。"""
        board = _board(
            [
                _hero("death_knight", 2, 100, 100),
                _hero("angel", 2, 200, 100),
                _hero("assault", 2, 300, 100),
                _hero("death_knight", 1, 400, 100),
                _hero("assault", 2, 500, 100),
                _hero("assault", 1, 600, 100),
                _hero("assault", 1, 700, 100),
                _hero("fox", 2, 800, 100),
                _hero("fox", 1, 900, 100),
            ]
        )
        target = self._unlock(self._unlock_strategy(), board)
        assert target is not None
        assert target.hero_type == "death_knight"
        assert target.star_level == 2

    def test_wildcard_never_chosen(self) -> None:
        """通配英雄排除：幻天使占多数派也轮不到它，落非通配候选。"""
        board = _board(
            [
                _hero("angel_phantom", 1, 100, 100),
                _hero("angel_phantom", 2, 200, 100),
                _hero("angel", 1, 300, 100),
                _hero("death_knight", 1, 400, 100),
            ]
        )
        target = self._unlock(self._unlock_strategy(), board)
        assert target is not None
        assert target.hero_type != "angel_phantom"
        assert target.star_level == 1

    def test_main_c_gated_by_count(self) -> None:
        """主C 数量 4 不入候选（守底线）；≥5 入候选且按格位序可选中。"""
        at_four = _board(
            [
                _hero("assault", 2, 100, 100),
                _hero("assault", 2, 200, 100),
                _hero("assault", 1, 300, 100),
                _hero("assault", 1, 400, 100),
                _hero("angel", 1, 500, 100),
            ]
        )
        target = self._unlock(self._unlock_strategy(), at_four)
        assert target is not None
        assert target.hero_type == "angel"

        at_five = _board(
            [
                _hero("assault", 1, 100, 100),
                _hero("assault", 1, 200, 100),
                _hero("assault", 1, 300, 100),
                _hero("assault", 1, 400, 100),
                _hero("assault", 1, 500, 100),
                _hero("angel", 1, 600, 100),
            ]
        )
        target = self._unlock(self._unlock_strategy(), at_five)
        assert target is not None
        assert target.hero_type == "assault"

    def test_tie_prefers_2star(self) -> None:
        """候选 1星=2星 平局取 2星：合出 3星随机更接近目标。"""
        board = _board(
            [_hero("angel", 1, 100, 100), _hero("death_knight", 2, 200, 100)]
        )
        target = self._unlock(self._unlock_strategy(), board)
        assert target is not None
        assert target.star_level == 2

    def test_single_tier_falls_to_other(self) -> None:
        """某档为 0：自然落在另一档（此处仅 2星候选）。"""
        board = _board(
            [
                _hero("angel", 2, 100, 100),
                _hero("death_knight", 2, 200, 100),
                _hero("fox", 2, 300, 100),
            ]
        )
        target = self._unlock(self._unlock_strategy(), board)
        assert target is not None
        assert target.star_level == 2

    def test_empty_pool_returns_none(self) -> None:
        """两档全空（只剩通配与数量 4 的主C）：真卡死，无候选。"""
        board = _board(
            [
                _hero("angel_phantom", 1, 100, 100),
                _hero("angel_phantom", 2, 200, 100),
                _hero("assault", 1, 300, 100),
                _hero("assault", 1, 400, 100),
                _hero("assault", 1, 500, 100),
                _hero("assault", 1, 600, 100),
            ]
        )
        assert self._unlock(self._unlock_strategy(), board) is None

    def test_exclude_heroes_respects_target(self) -> None:
        """exclude_heroes 排除子环节目标英雄（狐狸）。"""
        board = _board(
            [_hero("fox", 1, 100, 100), _hero("angel", 1, 200, 100)]
        )
        target = self._unlock(
            self._unlock_strategy(), board, exclude_heroes=("fox",)
        )
        assert target is not None
        assert target.hero_type == "angel"

    def test_exclude_targets_skips_given_up(self) -> None:
        """按英雄放弃记忆（2026-10-03）：解锁候选逐个失败被记后落下一个，
        全部记过返回 None（真卡死）——低星英雄同样可能是皮肤升级产物，
        按钮超时按同一记法排除。"""
        board = _board(
            [
                _hero("death_knight", 2, 100, 100),
                _hero("angel", 2, 200, 100),
                _hero("fox", 2, 300, 100),
            ]
        )
        strategy = self._unlock_strategy()
        # 记过格位序第一只（原会选的死骑）：落到天使
        target = self._unlock(
            strategy, board, exclude_targets={("death_knight", 2, (100, 100))}
        )
        assert target is not None
        assert target.hero_type == "angel"
        # 全部记过：真卡死
        assert (
            self._unlock(
                strategy,
                board,
                exclude_targets={
                    ("death_knight", 2, (100, 100)),
                    ("angel", 2, (200, 100)),
                    ("fox", 2, (300, 100)),
                },
            )
            is None
        )

    def test_high_star_heroes_excluded(self) -> None:
        """≥门槛星级的英雄不入候选（3星死骑归常规变身碰主C管）。"""
        board = _board(
            [_hero("death_knight", 3, 100, 100), _hero("angel", 1, 200, 100)]
        )
        target = self._unlock(self._unlock_strategy(), board)
        assert target is not None
        assert target.hero_type == "angel"
        assert target.star_level == 1

    def test_assist_mode_returns_none(self) -> None:
        """邀请者碰辅助模式（牺牲顺序非空）不适用本例外。"""
        strategy = self._unlock_strategy(transform_sacrifice_order=["death_knight"])
        board = _board([_hero("angel", 1, 100, 100)])
        assert self._unlock(strategy, board) is None
