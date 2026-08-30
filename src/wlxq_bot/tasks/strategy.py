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
  N 次技能后回培养阶段，把主C总数补到目标以上再继续选技能。
"""

from __future__ import annotations

from wlxq_bot.config import MainCProfile, RunConfig
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

    # ------------------------------------------------------------------
    # 培养门禁
    # ------------------------------------------------------------------

    def has_target_main_c(self, board: BoardSnapshot) -> bool:
        """棋盘上是否存在达到目标星级的主 C。"""
        target = self._run_config.target_star_level
        return any(
            hero.hero_type == self._main_c and hero.star_level >= target for hero in board.heroes
        )

    def main_c_count(self, board: BoardSnapshot) -> int:
        """棋盘上主 C 英雄总数（不分星级）。"""
        return sum(hero.hero_type == self._main_c for hero in board.heroes)

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
        """
        if not self.has_target_main_c(board):
            return False
        if not require_hero_count or not self.topup_enabled:
            return True
        return self.main_c_count(board) >= self._profile.topup_hero_count  # type: ignore[union-attr]

    def ready_reason(self, board: BoardSnapshot, *, require_hero_count: bool = False) -> str:
        """main_c_ready 动作的日志描述。"""
        reason = f"达到 {self._run_config.target_star_level} 星主C"
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
    # 合成决策
    # ------------------------------------------------------------------

    def select_merge(
        self,
        board: BoardSnapshot,
        candidates: list[MergeCandidate],
        failed_pairs: FailedMergePairs,
    ) -> MergeCandidate | None:
        """从合法合成对中选择本次要拖的一对；None 表示本次不合并（应召唤）。

        规则（按顺序）：
        1. 跳过本轮内已拖动失败的对（英雄被弹回原位的对反复拖没有意义）；
        2. 非主 C 对优先（原有通用规则）；
        3. 只剩主 C 对且档案开启了 avoid_main_c_merge、棋盘有空位时
           不合并主C（保数量），召唤新英雄；棋盘占满时合并主C对腾格子；
        4. 3 星对（合成 4 星会弹赠送技能页）最后手段（2026-08-21 用户策略）：
           有 1/2 星对时优先合并低星对；只剩 3 星对且棋盘有空位时召唤新英雄
           避开；棋盘占满且无低星对才被迫合并 3 星对（随后进入赠送页确认流程）。
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
        low_pairs = [c for c in unfailed if c.hero_a.star_level < 3]
        if low_pairs:
            return low_pairs[0]
        if unfailed and not self._board_full(board):
            logger.info(
                "合成对仅剩3星（合成4星将弹赠送技能页）且棋盘有空位，召唤新英雄避开"
            )
            return None
        pair = unfailed[0] if unfailed else None
        if pair is None:
            logger.info(
                "候选合成对均已拖动失败 %d 个，召唤新英雄改变棋盘",
                len(failed_pairs),
            )
        return pair

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
