"""主C培养策略：按主C档案配置的合成、门禁与数量回补决策。

纯决策对象：只依赖棋盘快照（BoardSnapshot）、合成候选和配置，
不接触识别（Vision/Perception）与输入（Action Executor），由
``CoopTask`` 在培养/选技能各决策点调用。依赖方向保持
Task Engine 内部组合，见 docs/architecture.md。

当前策略项（均可按主C档案开关，默认关闭 = 原有通用行为）：
- ``avoid_main_c_merge``：合成尽量不合并主C（保数量）。没有非主C合法对
  且棋盘有空位时召唤新英雄；棋盘占满时才把合并主C对作为腾格子的
  最后手段（2026-08-16 实机规则，当前仅强袭启用）。
- ``topup_after_skill_selections`` / ``topup_hero_count``：局内选满
  N 次技能后回培养阶段，把主C总数补到目标以上再继续选技能。数量补够后
  本局不再复查（2026-09-07 用户策略）；回补阶段棋盘有空位时只召唤不合成，
  占满才合成腾格子。
"""

from __future__ import annotations

from wlxq_bot.config import MainCProfile, ModeProfile, RunConfig
from wlxq_bot.models import BoardHero, BoardSnapshot, MergeCandidate
from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

# 已拖动失败（重试耗尽）的合成对位置签名类型：(起点格, 终点格)
FailedMergePairs = set[tuple[tuple[int, int], tuple[int, int]]]


class CultivationStrategy:
    """单个主 C 的培养策略。

    Args:
        main_c: 主 C 英文标识（如 ``assault``）
        run_config: 运行配置（星级门禁等通用参数）
        profile: 主 C 档案；None 时退化为通用策略（无档案差异行为）
    """

    def __init__(
        self,
        main_c: str,
        run_config: RunConfig,
        profile: MainCProfile | None = None,
    ) -> None:
        self._main_c = main_c
        self._run_config = run_config
        self._profile = profile
        # 通配英雄类型集合（幻天使等）：保留数检查时通配侧不消耗"本类"数量
        self._wildcard_types = frozenset(run_config.wildcard_merge_heroes)

    # ------------------------------------------------------------------
    # 邀请者模式扩展（spec 2026-09-21-initiator-monkey-design.md）
    # ------------------------------------------------------------------

    @property
    def target_star_level(self) -> int:
        """主C门禁星级：档案覆盖优先（target_star_override>0），否则 run 配置。"""
        if self._profile is not None and self._profile.target_star_override > 0:
            return self._profile.target_star_override
        return self._run_config.target_star_level

    @property
    def required_lineup_heroes(self) -> tuple[str, ...]:
        """阵容门禁英雄（每类在场 ≥1 才可转选技能）；空 = 未启用。"""
        if self._profile is None:
            return ()
        return tuple(self._profile.required_lineup_heroes)

    @property
    def _lineup_variant_groups(self) -> dict[str, list[str]]:
        """门禁英雄的形态变体等价表（angel_phantom → angel 等）；空 = 无。"""
        if self._profile is None:
            return {}
        return {
            base: list(variants)
            for base, variants in self._profile.lineup_variant_groups.items()
        }

    @property
    def hero_keep_counts(self) -> dict[str, int]:
        """保留数表（合并后任一输入类型在场数不得低于该值）；空 = 未启用。"""
        if self._profile is None:
            return {}
        return dict(self._profile.hero_keep_counts)

    @property
    def forbid_star3_merge(self) -> bool:
        """3 星软避让是否启用（猴子档案 true：产物星级 ≥3 的合并一律不做）。"""
        return self._profile is not None and self._profile.forbid_star3_merge

    @property
    def transform_sacrifice_order(self) -> tuple[str, ...]:
        """变身碰辅助的牺牲顺序；空 = 强袭的碰主C模式。"""
        if self._profile is None:
            return ()
        return tuple(self._profile.transform_sacrifice_order)

    def lineup_missing(self, board: BoardSnapshot) -> list[str]:
        """返回阵容门禁尚未在场的英雄（按配置顺序）；空列表 = 阵容已齐。

        形态变体等价（lineup_variant_groups）：幻天使是天使的形态
        （game-rules.md），配置后棋盘上有变体即算本体在场。
        """
        present: set[str] = set()
        variant_groups = self._lineup_variant_groups
        for hero in board.heroes:
            t = hero.hero_type
            present.add(t)
            # 变体类型同时记为其本体（angel_phantom → angel）
            for base, variants in variant_groups.items():
                if t in variants:
                    present.add(base)
        return [h for h in self.required_lineup_heroes if h not in present]

    def _hero_type_count(self, board: BoardSnapshot, hero_type: str) -> int:
        return sum(hero.hero_type == hero_type for hero in board.heroes)

    def keep_count_for(self, hero_type: str) -> int:
        """某英雄的保留数；档案未配置时主C默认 1、其他默认 0。"""
        keep = self.hero_keep_counts
        if hero_type in keep:
            return keep[hero_type]
        return 1 if hero_type == self._main_c else 0

    def _merge_allowed(self, board: BoardSnapshot, candidate: MergeCandidate) -> bool:
        """邀请者过滤：3 星避让 + 保留数。强袭（未启用任何新开关）恒 True。

        - 3 星避让：产物星级 = max(输入星级) + 1 ≥ 3 的候选一律拒绝；
        - 保留数：两输入类型各自的「在场数 − 消耗数」低于保留数则拒绝
          （同类对消耗该类 2 个；通配对只消耗非通配一侧 1 个）。
        """
        if self.forbid_star3_merge:
            produced_star = max(candidate.hero_a.star_level, candidate.hero_b.star_level) + 1
            if produced_star >= 3:
                return False
        keep = self.hero_keep_counts
        if not keep:
            return True
        a_type, b_type = candidate.hero_a.hero_type, candidate.hero_b.hero_type
        if a_type == b_type:
            consumed = {a_type: 2}
        else:
            # 通配对（幻天使×其他）：幻天使侧无保留需求，只检查非通配一侧
            wildcard = self._wildcard_types
            a_is_wild = a_type in wildcard
            consumed = {b_type: 1} if a_is_wild else {a_type: 1}
            if not (a_is_wild or b_type in wildcard):
                # 非通配的异类对不存在（合成要求同类），防御性放行走原逻辑
                consumed = {}
        for hero_type, used in consumed.items():
            if self._hero_type_count(board, hero_type) - used < self.keep_count_for(hero_type):
                return False
        return True

    # ------------------------------------------------------------------
    # 培养门禁
    # ------------------------------------------------------------------

    def has_target_main_c(self, board: BoardSnapshot) -> bool:
        """棋盘上是否存在达到目标星级的主 C。"""
        target = self.target_star_level
        return any(
            hero.hero_type == self._main_c and hero.star_level >= target for hero in board.heroes
        )

    def main_c_count(self, board: BoardSnapshot) -> int:
        """棋盘上主 C 英雄总数（不分星级）。"""
        return sum(hero.hero_type == self._main_c for hero in board.heroes)

    def hero_count_satisfied(self, board: BoardSnapshot) -> bool:
        """主C数量是否已达回补目标；未启用数量回补时视为已满足。"""
        if not self.topup_enabled:
            return True
        assert self._profile is not None
        return self.main_c_count(board) >= self._profile.topup_hero_count

    def main_c_ready(
        self,
        board: BoardSnapshot,
        *,
        require_hero_count: bool = False,
    ) -> bool:
        """进入持续选技能阶段的条件。

        首次培养（``require_hero_count=False``）只看目标星级主C是否存在；
        数量回补阶段（选满 N 次技能后回到培养，``require_hero_count=True``）
        还要求主C总数达标（如强袭保证 4 个以上）——首次培养不做数量要求，
        让技能选择尽早开始，数量在回补阶段补足。

        邀请者扩展（spec §5.1）：档案配置 ``required_lineup_heroes`` 时，
        门禁额外要求每类辅助英雄在场 ≥1（如猴子：天使+雪姬都在场）。
        """
        if not self.has_target_main_c(board):
            return False
        if self.required_lineup_heroes and self.lineup_missing(board):
            return False
        if not require_hero_count or not self.topup_enabled:
            return True
        return self.main_c_count(board) >= self._profile.topup_hero_count  # type: ignore[union-attr]

    def ready_reason(self, board: BoardSnapshot, *, require_hero_count: bool = False) -> str:
        """main_c_ready 动作的日志描述。"""
        missing = self.lineup_missing(board)
        if missing:
            return f"达到 {self.target_star_level} 星主C，但阵容缺 {('、'.join(missing))}"
        reason = f"达到 {self.target_star_level} 星主C"
        if self.required_lineup_heroes:
            reason += f" 且阵容齐（{('、'.join(self.required_lineup_heroes))}）"
        if require_hero_count and self.topup_enabled:
            reason += f" 且数量 {self.main_c_count(board)}/{self._profile.topup_hero_count}"  # type: ignore[union-attr]
        return reason

    @property
    def topup_enabled(self) -> bool:
        """是否启用「选满 N 次技能后回培养补数量」。"""
        return self._profile is not None and self._profile.topup_after_skill_selections > 0

    @property
    def skill_selection_cap(self) -> int:
        """局内技能选择总次数上限；0 = 不限制。"""
        return self._profile.skill_selection_cap if self._profile else 0

    def should_topup(self, skill_selections: int) -> bool:
        """局内技能选择次数达到回补阈值时返回 True（未启用恒为 False）。"""
        if not self.topup_enabled:
            return False
        assert self._profile is not None
        return skill_selections >= self._profile.topup_after_skill_selections

    # ------------------------------------------------------------------
    # 变身决策
    # ------------------------------------------------------------------

    def select_transform(
        self,
        board: BoardSnapshot,
        used_count: int,
        max_count: int,
    ) -> BoardHero | None:
        """选择本次变身目标；None 表示不变身。

        强袭模式（默认，``transform_sacrifice_order`` 为空）——设计文档 §4.1：
        条件全部满足才返回目标：
        1. 本局变身次数未用完（used_count < max_count）；
        2. 棋盘存在星级不低于 run.transform_min_star_level 的非主 C 英雄。

        2026-09-17 用户变更：不再要求「尚无达标主 C」——主 C 已达标（含
        回补阶段）时，2 星对合并产出 3 星非主 C 的下一轮识别同样变身碰
        主 C（免费且可能直接提星/凑数量）。

        邀请者碰辅助模式（spec §5.3，``transform_sacrifice_order`` 非空）：
        由调用方在「卡住」（阵容不齐+棋盘满+无安全合成）时调用
        :meth:`select_transform_assist`，本方法不适用（3星避让下不存在
        3星非主C触发条件，恒返回 None 由调用方落空）。

        多个候选取 heroes 列表中的第一个（感知层按棋盘格序构建 heroes，
        即按格位顺序）；返回 None 时调用方落入常规召唤/合成决策。
        """
        if self.transform_sacrifice_order:
            return None
        if used_count >= max_count:
            return None
        min_star = self._run_config.transform_min_star_level
        for hero in board.heroes:
            if hero.hero_type != self._main_c and hero.star_level >= min_star:
                return hero
        return None

    def select_transform_assist(
        self,
        board: BoardSnapshot,
        used_count: int,
        max_count: int,
    ) -> BoardHero | None:
        """变身碰辅助（邀请者卡住出口，spec §5.3）。

        从牺牲顺序（``transform_sacrifice_order``）取第一个满足
        「在场数 > 保留数」的英雄变身：类型必变（本局携带 4 种英雄中除自身
        外随机之一，星级不变），有概率变出阵容缺的辅助。天使/猴子不进
        牺牲名单即永不被变身；雪姬配置保留数 1 时仅在场 ≥2 才可牺牲。

        Returns:
            牺牲英雄；次数耗尽 / 无人可牺牲 / 未启用该模式时 None。
        """
        order = self.transform_sacrifice_order
        if not order or used_count >= max_count:
            return None
        for hero_type in order:
            if self._hero_type_count(board, hero_type) > self.keep_count_for(hero_type):
                for hero in board.heroes:
                    if hero.hero_type == hero_type:
                        return hero
        return None

    # ------------------------------------------------------------------
    # 合成决策
    # ------------------------------------------------------------------

    def select_merge(
        self,
        board: BoardSnapshot,
        candidates: list[MergeCandidate],
        failed_pairs: FailedMergePairs,
    ) -> MergeCandidate | None:
        """从合法合成对中选择本次要拖的一对；None 表示本次不合并（应召唤）。

        决策层级（2026-09-13 幻天使定案，spec §4.5）：
        1. 跳过本轮内已拖动失败的对（英雄被弹回原位的对反复拖没有意义）；
        2. 非主 C 对优先（原有通用规则）；
        3. 只剩主 C 对且档案开启了 avoid_main_c_merge、棋盘有空位时
           不合并主C（保数量），召唤新英雄；棋盘占满时合并主C对腾格子；
        4. 同类项 1/2 星对：立即合并（候选顺序同类项在前）；
        5. 通配 1/2 星对（幻天使）：棋盘有空位时不合并、召唤新英雄
           （保留幻天使作最难凑的兜底）；棋盘占满才合并；
        6. 3 星对（含通配 3 星，合成 4 星会弹赠送技能页）最后手段
           （2026-08-21 用户策略）：棋盘有空位时召唤避开；占满才被迫合并。
        棋盘占满时通配 1/2 星对先于 3 星对（前者不触发赠送技能页）。
        """
        if not candidates:
            return None
        avoid_main_merge = self._profile is not None and self._profile.avoid_main_c_merge
        non_main_pairs = [c for c in candidates if not c.is_main_c]
        if not non_main_pairs and avoid_main_merge and not self._board_full(board):
            logger.info("无非主C合成对且棋盘有空位，不合并主C，召唤新英雄（保主C数量）")
            return None
        pool = non_main_pairs if non_main_pairs else candidates
        unfailed = [c for c in pool if self.pair_signature(c) not in failed_pairs]
        # 邀请者过滤（3星避让+保留数）：猴子档案下强袭的「占满被迫合并」不生效，
        # 不合法的候选直接出局，走召唤/变身/干等出口
        allowed = [c for c in unfailed if self._merge_allowed(board, c)]
        dropped = len(unfailed) - len(allowed)
        if dropped:
            logger.debug("邀请者过滤剔除 %d 个候选（3星避让/保留数）", dropped)
        unfailed = allowed
        same_type_low = [c for c in unfailed if c.hero_a.star_level < 3 and not c.is_wildcard]
        if same_type_low:
            return same_type_low[0]
        if unfailed and not self._board_full(board):
            wildcard_pending = [
                c for c in unfailed if c.is_wildcard and c.hero_a.star_level < 3
            ]
            if wildcard_pending:
                logger.info("棋盘有空位，保留幻天使不合并（留作最难凑兜底），召唤新英雄")
            else:
                logger.info(
                    "合成对仅剩3星（合成4星将弹赠送技能页）且棋盘有空位，召唤新英雄避开"
                )
            return None
        forced_wildcard_low = [
            c for c in unfailed if c.is_wildcard and c.hero_a.star_level < 3
        ]
        if forced_wildcard_low:
            return forced_wildcard_low[0]
        pair = unfailed[0] if unfailed else None
        if pair is None:
            logger.info(
                "候选合成对均已拖动失败 %d 个，召唤新英雄改变棋盘",
                len(failed_pairs),
            )
        return pair

    def select_merge_batch(
        self,
        candidates: list[MergeCandidate],
        failed_pairs: FailedMergePairs,
        board: BoardSnapshot,
    ) -> list[MergeCandidate]:
        """批次合成规划（2026-09-10 用户策略 + 2026-09-13 幻天使定案）。

        把当前快照上能合的对一次排完，只做无争议的合并：
        1. 绝不动主C（主C强袭保留，数量保护；主C对仍由单动作策略在棋盘
           占满等最后手段场景处理）；
        2. 只排 1/2 星对（3星对合成4星会弹赠送技能页，可能中途遮挡棋盘）；
        3. 通配对（幻天使）纳入批次，但棋盘未满时排除——有空位先召唤、
           占满才花幻天使，与单动作策略同一决策层级；
        4. 跳过失败记忆中的对（本局内已确认拖不动的对不反复排）；
        5. 对与对之间格子互不相交（find_merge_candidates 已按英雄去重配对，
           此处按位置再防御性过滤），合并互不冲突，可连续拖完再统一识别。
        """
        board_full = board.capacity.occupied >= board.capacity.total_slots
        selected: list[MergeCandidate] = []
        used: set[tuple[int, int]] = set()
        for candidate in candidates:
            if candidate.is_main_c:
                continue
            if max(candidate.hero_a.star_level, candidate.hero_b.star_level) >= 3:
                continue
            if candidate.is_wildcard and not board_full:
                continue
            if self.pair_signature(candidate) in failed_pairs:
                continue
            if not self._merge_allowed(board, candidate):
                continue
            positions = {candidate.hero_a.position, candidate.hero_b.position}
            if positions & used:
                continue
            used |= positions
            selected.append(candidate)
        return selected

    def _board_full(self, board: BoardSnapshot) -> bool:
        return board.capacity.occupied >= board.capacity.total_slots

    @staticmethod
    def drag_direction(pair: MergeCandidate) -> tuple[BoardHero, BoardHero]:
        """返回 (拖动起点, 拖动终点)：尽量从下往上拖。

        实机规则（2026-08-16）：起点取行号更大（更靠下）的英雄，例如
        1A 和 4B 应从 4B 拖到 1A；同一排没有上下之分，保持原候选顺序。
        """
        if pair.hero_a.position[1] >= pair.hero_b.position[1]:
            return pair.hero_a, pair.hero_b
        return pair.hero_b, pair.hero_a

    @classmethod
    def pair_signature(cls, pair: MergeCandidate) -> tuple[tuple[int, int], tuple[int, int]]:
        """合成对的位置签名：与拖动方向一致（下格在前），供失败记忆比对。"""
        source, dest = cls.drag_direction(pair)
        return (source.position, dest.position)


def build_strategy(
    mode_profile: ModeProfile | None,
    main_c: str,
    run_config: RunConfig,
    profile: MainCProfile | None,
) -> CultivationStrategy:
    """按模式档案构造培养策略（spec §4.2）。

    strategy 字段是阶段化策略（spec §7，如 3星→补4个→无限选）的扩展点；
    当前仅支持 ``default``，未知名直接拒绝，不带病运行。
    """
    if mode_profile is not None and mode_profile.strategy != "default":
        raise ValueError(f"未知策略 {mode_profile.strategy!r}，当前仅支持 'default'")
    return CultivationStrategy(main_c, run_config, profile)
