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

from collections.abc import Collection

from wlxq_bot.config import MainCProfile, ModeProfile, RunConfig
from wlxq_bot.models import BoardHero, BoardSnapshot, MergeCandidate
from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

# 已拖动失败（重试耗尽）的合成对位置签名类型：(起点格, 终点格)
FailedMergePairs = set[tuple[tuple[int, int], tuple[int, int]]]

# 变身放弃记忆的目标英雄键：(英雄类型, 星级, 格位中心像素)。position 由
# 棋盘网格推导（格位中心，确定性），同一格同一英雄跨帧稳定；英雄不挪格，
# 格位即英雄在局内的身份。
TransformGiveupKey = tuple[str, int, tuple[int, int]]


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
    # 3星主C培养环节（数量回补达标后的非强制追加，2026-09-30 用户定案）
    # ------------------------------------------------------------------

    @property
    def star3_topup_enabled(self) -> bool:
        """是否启用「数量达标后追加 3星主C 培养环节」。"""
        return self._profile is not None and self._profile.star3_topup_enabled

    def has_star3_main_c(self, board: BoardSnapshot) -> bool:
        """棋盘上是否存在 ≥3 星主C（3星培养环节的达成条件）。"""
        return any(
            hero.hero_type == self._main_c and hero.star_level >= 3 for hero in board.heroes
        )

    def _has_transform_target(
        self,
        board: BoardSnapshot,
        exclude_targets: Collection[TransformGiveupKey] = (),
    ) -> bool:
        """棋盘上是否存在可变身对象（≥min_star 的非主C，排除已记放弃的英雄）。"""
        min_star = self._run_config.transform_min_star_level
        return any(
            hero.hero_type != self._main_c
            and hero.star_level >= min_star
            and (hero.hero_type, hero.star_level, hero.position) not in exclude_targets
            for hero in board.heroes
        )

    def main_c_pair_surplus_ok(
        self, board: BoardSnapshot, candidate: MergeCandidate
    ) -> bool:
        """含主C对按「对内主C消耗数」判富余（2026-10-02 用户定案）。

        合成类型随机、产物可能不是主C，按消耗后数量仍 ≥ ``topup_hero_count``
        的最坏情况判定：主C+主C 消耗 2（≥ 底线+2 可合，如 6 可合 5 不合）；
        幻天使+主C 只消耗 1（幻天使不是主C，≥ 底线+1 即 ≥5 可合）——与
        低星变身「消耗 1 → ≥ 底线+1 才可变主C」同一底线哲学。
        """
        keep = self._profile.topup_hero_count if self._profile else 0
        consumed = sum(
            hero.hero_type == self._main_c
            for hero in (candidate.hero_a, candidate.hero_b)
        )
        return self.main_c_count(board) - consumed >= keep

    def star3_pair_clears_star2_main_c(
        self, board: BoardSnapshot, candidate: MergeCandidate
    ) -> bool:
        """该对合成后是否把场上 2 星主C清零（2026-10-03 用户定案）。

        判据：对内含 2 星主C，且场上 2 星主C总数恰等于对内 2 星主C数
        （同类 2 星对即「恰好 2 只」，幻天使+主C 对即「恰好 1 只」）。
        供 3 星环节选对排序用——清零的对让位，先合 1 星对攒 2 星主C；
        合成后仍剩 ≥1 只（场上 ≥3 只）的对可优先合，尽快碰 3 星。
        """
        pair_star2 = sum(
            hero.hero_type == self._main_c and hero.star_level == 2
            for hero in (candidate.hero_a, candidate.hero_b)
        )
        if not pair_star2:
            return False
        board_star2 = sum(
            hero.hero_type == self._main_c and hero.star_level == 2
            for hero in board.heroes
        )
        return board_star2 <= pair_star2

    def star3_has_mergeable_pair(
        self,
        board: BoardSnapshot,
        candidates: list[MergeCandidate],
        failed_pairs: FailedMergePairs,
    ) -> bool:
        """3星环节是否存在可用合成对（供合并优先与死局判定共用）。

        非主C对任意可用；含主C对按消耗数判富余（幻天使+主C 消耗 1、
        主C+主C 消耗 2，见 :meth:`main_c_pair_surplus_ok`）。
        """
        return any(
            (
                not candidate.is_main_c
                or self.main_c_pair_surplus_ok(board, candidate)
            )
            and self.pair_signature(candidate) not in failed_pairs
            for candidate in candidates
        )

    def star3_stage_exit(
        self,
        board: BoardSnapshot,
        candidates: list[MergeCandidate],
        failed_pairs: FailedMergePairs,
        *,
        transform_used: int,
        failed_targets: Collection[TransformGiveupKey] = (),
    ) -> tuple[bool, str]:
        """3星培养环节退出判定；返回 (是否退出, 原因)。

        1. 达成：场上出现 ≥3 星主C；
        2. 放弃·变身用尽：整局预算归零即退，不再要求「环节内消耗过变身」
           （2026-10-05 用户定案，推翻 2026-09-30 的环节内门槛——当日实机
           局 6：进场前预算已空仍留场培养，合成路径被数量底线+3星避让锁
           死，70 秒 18 发召唤零产出）。退场后凑 lineup 英雄保证照旧；
        3. 放弃·死局：无可用合成对（非主C对 + 富余主C对，排除失败记忆）
           + 无空位可召唤 + 无可变身对象（3星非主C + 剩余预算 + 未被
           按英雄放弃记忆记过——2026-10-03 记法从整盘签名改按目标英雄，
           记满所有目标等价于变身不可用）。
        """
        if self.has_star3_main_c(board):
            return True, "已达成3星主C"
        max_per_round = self._run_config.transform_max_per_round
        if transform_used >= max_per_round:
            return True, "变身次数用尽仍无3星主C"
        has_pair = self.star3_has_mergeable_pair(board, candidates, failed_pairs)
        has_space = board.capacity.available > 0
        can_transform = (
            transform_used < max_per_round
            and self._has_transform_target(board, failed_targets)
        )
        if not has_pair and not has_space and not can_transform:
            return True, "死局（无对可合/无空位/无可变身对象）"
        return False, "继续培养3星主C"

    # ------------------------------------------------------------------
    # 凑 lineup 英雄子环节（3星环节出口追加，2026-10-01 用户定案）
    # ------------------------------------------------------------------

    @property
    def star3_topup_after_hero(self) -> str:
        """3星环节出口要保证在场的英雄标识；空 = 不启用子环节。"""
        return self._profile.star3_topup_after_hero if self._profile else ""

    def has_hero(self, board: BoardSnapshot, hero: str) -> bool:
        """棋盘上是否存在指定英雄（任意星级）。"""
        return any(h.hero_type == hero for h in board.heroes)

    def topup_after_pair_allowed(
        self, board: BoardSnapshot, candidate: MergeCandidate
    ) -> bool:
        """子环节合成对准入：不含目标英雄，且含主C时按消耗数判富余。

        非主C对任意可用（天使/幻天使/死骑互合，现状）；含主C对 2026-10-02
        用户定案放行富余对——主C+主C 消耗 2（≥ 底线+2 即 ≥6 可合，产物
        3星类型随机有 1/4 直接是目标英雄当场达标）；幻天使+主C 只消耗 1
        （≥ 底线+1 即 ≥5 可合），与环节材料规则同一条判式（见
        :meth:`main_c_pair_surplus_ok`）。目标英雄排除防合掉仅有的目标
        （召唤刚出目标英雄、同帧尚未重新识别的边界）。
        """
        hero = self.star3_topup_after_hero
        if candidate.hero_a.hero_type == hero or candidate.hero_b.hero_type == hero:
            return False
        if not candidate.is_main_c:
            return True
        return self.main_c_pair_surplus_ok(board, candidate)

    def _has_transform_target_excluding(
        self,
        board: BoardSnapshot,
        hero: str,
        exclude_targets: Collection[TransformGiveupKey] = (),
    ) -> bool:
        """是否存在可变身对象（≥min_star 的非主C），再排除指定英雄与已记放弃的。"""
        min_star = self._run_config.transform_min_star_level
        return any(
            h.hero_type not in (self._main_c, hero)
            and h.star_level >= min_star
            and (h.hero_type, h.star_level, h.position) not in exclude_targets
            for h in board.heroes
        )

    def topup_after_stage_exit(
        self,
        board: BoardSnapshot,
        candidates: list[MergeCandidate],
        failed_pairs: FailedMergePairs,
        *,
        transform_usable: bool,
        failed_targets: Collection[TransformGiveupKey] = (),
    ) -> tuple[bool, str]:
        """凑 lineup 英雄子环节退出判定；返回 (是否退出, 原因)。

        1. 达成：场上 ≥1 只目标英雄（任意星级）；
        2. 放弃：棋盘满 + 无准入对（排除失败记忆）+ 无变身可用
           （``transform_usable`` 为调用方综合开关/次数后的结果，变身
           目标检查在此进行——目标不含主C与目标英雄，且未被按英雄
           放弃记忆记过 ``failed_targets``，2026-10-03）。
        """
        hero = self.star3_topup_after_hero
        if self.has_hero(board, hero):
            return True, f"凑{hero}达成"
        has_pair = any(
            self.topup_after_pair_allowed(board, c)
            and self.pair_signature(c) not in failed_pairs
            for c in candidates
        )
        if board.capacity.available > 0 or has_pair:
            return False, f"继续凑{hero}（召唤/合成可用）"
        if transform_usable and self._has_transform_target_excluding(
            board, hero, failed_targets
        ):
            return False, f"继续凑{hero}（变身可用）"
        return True, f"放弃凑{hero}（满盘无对可合无变身）"

    # ------------------------------------------------------------------
    # 变身决策
    # ------------------------------------------------------------------

    def select_transform(
        self,
        board: BoardSnapshot,
        used_count: int,
        max_count: int,
        *,
        exclude_heroes: tuple[str, ...] = (),
        exclude_targets: Collection[TransformGiveupKey] = (),
        min_star_level: int | None = None,
    ) -> BoardHero | None:
        """选择本次变身目标；None 表示不变身。

        ``exclude_heroes``：额外排除的英雄类型（凑 lineup 英雄子环节传目标
        英雄——变身结果类型随机，把仅有的目标英雄变走会直接丢掉达标条件）。
        ``exclude_targets``：按英雄放弃记忆（类型+星级+格位，2026-10-03
        用户定案）——按钮超时等失败的目标本局不再重选；皮肤升级的不可变身
        英雄与常规产出外观无差别，超时是唯一可观测信号。
        ``min_star_level``：目标星级下限覆盖（2026-10-05 用户定案）——缺省
        用 ``run.transform_min_star_level``（一阶段与回补段：只变 3/4 星）；
        3星主C培养环节与凑 lineup 英雄子环节传 1 放开任意星级（低星滚主C
        数量/目标英雄，预算打光即按环节退出离场）。

        强袭模式（默认，``transform_sacrifice_order`` 为空）——设计文档 §4.1：
        条件全部满足才返回目标：
        1. 本局变身次数未用完（used_count < max_count）；
        2. 棋盘存在星级不低于下限的非主 C 英雄。

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
        if min_star_level is not None:
            min_star = min_star_level
        else:
            min_star = self._run_config.transform_min_star_level
        for hero in board.heroes:
            if (
                hero.hero_type != self._main_c
                and hero.star_level >= min_star
                and hero.hero_type not in exclude_heroes
                and (hero.hero_type, hero.star_level, hero.position)
                not in exclude_targets
            ):
                return hero
        return None

    def select_transform_assist(
        self,
        board: BoardSnapshot,
        used_count: int,
        max_count: int,
        *,
        exclude_targets: Collection[TransformGiveupKey] = (),
    ) -> BoardHero | None:
        """变身碰辅助（邀请者卡住出口，spec §5.3）。

        从牺牲顺序（``transform_sacrifice_order``）取第一个满足
        「在场数 > 保留数」的英雄变身：类型必变（本局携带 4 种英雄中除自身
        外随机之一，星级不变），有概率变出阵容缺的辅助。天使/猴子不进
        牺牲名单即永不被变身；雪姬配置保留数 1 时仅在场 ≥2 才可牺牲。
        ``exclude_targets``：按英雄放弃记忆（2026-10-03），同
        :meth:`select_transform`。

        Returns:
            牺牲英雄；次数耗尽 / 无人可牺牲 / 未启用该模式时 None。
        """
        order = self.transform_sacrifice_order
        if not order or used_count >= max_count:
            return None
        for hero_type in order:
            if self._hero_type_count(board, hero_type) > self.keep_count_for(hero_type):
                for hero in board.heroes:
                    if hero.hero_type == hero_type and (
                        hero.hero_type,
                        hero.star_level,
                        hero.position,
                    ) not in exclude_targets:
                        return hero
        return None

    def select_deadlock_transform(
        self,
        board: BoardSnapshot,
        *,
        wildcard_heroes: frozenset[str] = frozenset(),
        exclude_heroes: tuple[str, ...] = (),
        exclude_targets: Collection[TransformGiveupKey] = (),
    ) -> BoardHero | None:
        """卡死解锁的低星变身目标（2026-10-01 用户定案）；None = 无候选。

        仅在满盘 + 无准入合成对 + 无 ≥``transform_min_star_level`` 变身材料
        的卡死判定点调用（3星环节死局出口、凑 lineup 英雄子环节放弃出口），
        常规「变身碰主C」的星级门槛不因此变化。游戏规则：变身对任意星级
        英雄可用，星级不变、类型必变（2026-10-01 实机确认）。

        候选池规则：

        - 通配英雄（幻天使）永不入候选：卡死时它必无同星可合同伴（有则
          早已配对解锁，不会走到卡死点），排除零损失；
        - 主 C 仅当场数量 ≥ ``topup_hero_count``+1 时入候选——变完仍守
          数量底线，与富余主 C 对可合（数量-2 ≥ 底线）同一哲学；
        - ≥ 星级门槛的英雄不入候选（卡死时非主 C 的本就不存在；达成的
          3星主 C 更不能拿来赌），本方法只服务低星例外；
        - ``exclude_heroes``：额外排除的类型（子环节传目标英雄）；
        - ``exclude_targets``：按英雄放弃记忆（2026-10-03），同
          :meth:`select_transform`——解锁候选逐个失败被记后返回 None
          （真卡死），等价于旧的整盘签名抑制且更精确。

        星级在候选内数多数派，平局取 2星（凑出的对合出 3星随机更接近
        目标）；某档为 0 自然落在另一档，两档全空返回 None（真卡死）。
        同档多只按棋盘格位顺序取第一只（与 :meth:`select_transform` 一致）。
        """
        if self.transform_sacrifice_order or self._profile is None:
            return None
        min_star = self._run_config.transform_min_star_level
        main_c_count = self.main_c_count(board)
        main_c_transformable = main_c_count >= self._profile.topup_hero_count + 1
        pool = [
            hero
            for hero in board.heroes
            if hero.star_level < min_star
            and hero.hero_type not in wildcard_heroes
            and hero.hero_type not in exclude_heroes
            and (hero.hero_type != self._main_c or main_c_transformable)
            and (hero.hero_type, hero.star_level, hero.position) not in exclude_targets
        ]
        low_star = [hero for hero in pool if hero.star_level == 1]
        high_star = [hero for hero in pool if hero.star_level == 2]
        if not pool:
            return None
        chosen = high_star if len(high_star) >= len(low_star) else low_star
        return chosen[0]

    # ------------------------------------------------------------------
    # 合成决策
    # ------------------------------------------------------------------

    def select_merge(
        self,
        board: BoardSnapshot,
        candidates: list[MergeCandidate],
        failed_pairs: FailedMergePairs,
        *,
        star3_stage: bool = False,
        allow_wildcard_with_space: bool = False,
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

        ``star3_stage``：3星主C培养环节使用——材料规则（2026-10-01 用户
        修正「保留 4 个就行」、2026-10-02 消耗数化）：非主C对任意可用；
        含主C对按消耗数判富余（幻天使+主C 消耗 1、主C+主C 消耗 2，见
        :meth:`main_c_pair_surplus_ok`）。保留数不变式替代
        avoid_main_c_merge 的数量回避：富余即合（含棋盘有空位），尽快
        合并 2 星对碰 3 星。同类低星对按
        :meth:`_order_star3_same_type_low` 排序（2026-10-03 用户定案）：
        合成后场上 2 星主C清零的对让位到 1 星对之后，先攒后花。

        ``allow_wildcard_with_space``：金币不足例外（2026-10-01 用户定案）——
        召唤被金币门控拦住时「有空位 → 召唤不花幻天使」的前提不成立，
        等金币恢复期间幻天使通配对让位可合（仅放开第 5 层的空位保留；
        3 星对规避与主C保数量照旧）。调用方目前只有等金币合成路径。
        """
        if star3_stage:
            candidates = [
                c
                for c in candidates
                if not c.is_main_c or self.main_c_pair_surplus_ok(board, c)
            ]
            avoid_main_merge = False
        else:
            avoid_main_merge = self._profile is not None and self._profile.avoid_main_c_merge
        if not candidates:
            return None
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
        if star3_stage:
            same_type_low = self._order_star3_same_type_low(board, same_type_low)
        if same_type_low:
            return same_type_low[0]
        wildcard_pending = [
            c for c in unfailed if c.is_wildcard and c.hero_a.star_level < 3
        ]
        if unfailed and not self._board_full(board):
            if wildcard_pending and not allow_wildcard_with_space:
                logger.info("棋盘有空位，保留幻天使不合并（留作最难凑兜底），召唤新英雄")
                return None
            if not wildcard_pending:
                logger.info(
                    "合成对仅剩3星（合成4星将弹赠送技能页）且棋盘有空位，召唤新英雄避开"
                )
                return None
            # 金币不足例外：幻天使对让位，落到下方选中（3星对规避不受影响）
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

    def _order_star3_same_type_low(
        self, board: BoardSnapshot, pairs: list[MergeCandidate]
    ) -> list[MergeCandidate]:
        """3星环节同类低星对排序（2026-10-03 用户定案）。

        同时有 1 星/2 星主C对时：2 星对合成后场上仍剩 ≥1 只 2 星主C
        （场上 ≥3 只）则优先合 2 星对，尽快碰 3 星；恰好清零（场上仅
        2 只）的 2 星对让位到最后，先合 1 星对攒——1 星对合成有 1/4
        直接补出 2 星主C、1/2 配出可用的非主C材料对，不把最后一发
        子弹打光。只有清零对时照合（让位不是禁止）。非主C材料对不受
        影响，保持原序在前。
        """
        non_main = [c for c in pairs if not c.is_main_c]
        main_pairs = [c for c in pairs if c.is_main_c]
        clearing = [
            c for c in main_pairs if self.star3_pair_clears_star2_main_c(board, c)
        ]
        if clearing and len(clearing) != len(main_pairs):
            logger.info("2星主C对合成即清零，让位先合其他同类对（攒2星主C）")
        main_pairs.sort(
            key=lambda c: (
                self.star3_pair_clears_star2_main_c(board, c),
                c.hero_a.star_level != 2,
            )
        )
        return non_main + main_pairs

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
