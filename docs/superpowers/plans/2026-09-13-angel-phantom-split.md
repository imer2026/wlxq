# 幻天使分离识别与通配合并 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把幻天使从 angel 识别类别中拆出（angel_phantom，仅 star1~3），并让合成决策支持通配规则（同类项优先、幻天使棋盘占满才用）。

**Architecture:** 数据层在既有 labeled 树内原地迁移（hue 色彩分拣 + 用户人工确认硬门），重训 MobileNetV3-Small ONNX（v5）；代码层加三个配置键（run.wildcard_merge_heroes / hero_classifier.lineup_variants / 模型路径切换），合并候选生成改两阶段（同类项 → 通配兜底），策略层改三级决策层级。识别运行时（HeroCellClassifier）零改动。

**Tech Stack:** Python 3.10+ / Pydantic / OpenCV / torchvision(MobileNetV3-Small, CPU) / pytest

**Spec:** `docs/superpowers/specs/2026-09-13-angel-phantom-split-design.md`（本计划的唯一需求来源，执行前先通读）

## Global Constraints

- 本机是 Windows + Git Bash；CLI 一律用 `python -m wlxq_bot …`（`wlxq-bot` 不在 PATH）。
- Python 3.10+；文件一律 UTF-8；日志一律 `get_logger(__name__)`，禁止裸 `logging.getLogger`。
- 每个任务独立提交，提交信息用任务里给定的消息；`git add` 只加任务列出的文件，禁止 `git add -A`。
- 行号会漂移：**所有编辑用本计划给的内容锚点定位**（Edit 的 old_string 必须与计划中给出的原文逐字一致），不要依赖行号。
- 不改动与本计划无关的代码（特别是 `tasks/coop.py` 里已提交的"变身"阶段机代码，只允许触碰本计划点名的锚点行）。
- 不运行游戏、不真实点击桌面；验证全部用单元测试和离线图片。
- 测试命令统一在仓库根 `D:\data\python\github\wlxq` 执行。
- 计划中标注 ⛔ 的步骤是**人工硬门**：把输出发给用户，等用户明确说"继续"才能往下走。

---

### Task 0: 前置检查（不改任何文件）

- [ ] **Step 1: 确认工作区干净**

Run: `git status --short`
Expected: 无输出（完全干净）。若有输出，**停止**并报告用户，不得继续。

- [ ] **Step 2: 确认分支与依赖**

```bash
git branch --show-current
python -c "import torch, torchvision, cv2; print(torch.__version__, torchvision.__version__, cv2.__version__)"
python -m pytest tests/test_board_models.py -q
```
Expected: 分支为 `feat/pet-defend-mode`；版本号正常打印；测试全部通过（本任务无提交）。

---

### Task 1: models.py —— MergeCandidate.is_wildcard 与两阶段配对

**Files:**
- Modify: `src/wlxq_bot/models.py`（MergeCandidate 类、BoardSnapshot.find_merge_candidates 方法）
- Test: `tests/test_board_models.py`（文件末尾追加新测试类）

**Interfaces:**
- Produces: `MergeCandidate(hero_a, hero_b, is_main_c=False, is_wildcard=False)`；
  `BoardSnapshot.find_merge_candidates(main_c: str, wildcard_heroes: frozenset[str] = frozenset()) -> list[MergeCandidate]`。
  后续任务依赖：`is_wildcard` 字段、默认空集时行为与旧版完全一致。

- [ ] **Step 1: 写失败测试**

在 `tests/test_board_models.py` 文件末尾追加：

```python
class TestFindMergeCandidatesWildcard:
    """幻天使（通配英雄）配对规则（2026-09-13 定案，spec §4.4）。"""

    def test_wildcard_pairs_with_any_same_star(self) -> None:
        """幻天使可与任意同星级英雄配对（标记 is_wildcard）。"""
        snapshot = BoardSnapshot(
            frame_id=1,
            heroes=[
                _hero("angel_phantom", 1, 100, 100),
                _hero("snow", 1, 200, 100),
            ],
            capacity=BoardCapacity(total_slots=6, occupied=2),
        )
        candidates = snapshot.find_merge_candidates(
            main_c="assault", wildcard_heroes=frozenset({"angel_phantom"})
        )
        assert len(candidates) == 1
        assert candidates[0].is_wildcard is True
        assert candidates[0].is_main_c is False
        assert {candidates[0].hero_a.hero_type, candidates[0].hero_b.hero_type} == {
            "angel_phantom",
            "snow",
        }

    def test_wildcard_star_must_match(self) -> None:
        """通配也要求同星级。"""
        snapshot = BoardSnapshot(
            frame_id=1,
            heroes=[
                _hero("angel_phantom", 1, 100, 100),
                _hero("snow", 2, 200, 100),
            ],
            capacity=BoardCapacity(total_slots=6, occupied=2),
        )
        candidates = snapshot.find_merge_candidates(
            main_c="assault", wildcard_heroes=frozenset({"angel_phantom"})
        )
        assert candidates == []

    def test_wildcard_pairs_with_wildcard(self) -> None:
        """幻天使可与另一幻天使配对。"""
        snapshot = BoardSnapshot(
            frame_id=1,
            heroes=[
                _hero("angel_phantom", 2, 100, 100),
                _hero("angel_phantom", 2, 200, 100),
            ],
            capacity=BoardCapacity(total_slots=6, occupied=2),
        )
        candidates = snapshot.find_merge_candidates(
            main_c="assault", wildcard_heroes=frozenset({"angel_phantom"})
        )
        assert len(candidates) == 1
        assert candidates[0].is_wildcard is True

    def test_same_type_preferred_over_wildcard(self) -> None:
        """同类项对优先：两个正常天使先配对，幻天使插不进去。"""
        snapshot = BoardSnapshot(
            frame_id=1,
            heroes=[
                _hero("angel", 1, 100, 100),
                _hero("angel", 1, 200, 100),
                _hero("angel_phantom", 1, 300, 100),
            ],
            capacity=BoardCapacity(total_slots=6, occupied=3),
        )
        candidates = snapshot.find_merge_candidates(
            main_c="assault", wildcard_heroes=frozenset({"angel_phantom"})
        )
        assert len(candidates) == 1
        assert candidates[0].is_wildcard is False
        assert {candidates[0].hero_a.hero_type, candidates[0].hero_b.hero_type} == {"angel"}

    def test_wildcard_prefers_non_wildcard_partner(self) -> None:
        """幻天使优先与非幻英雄配对，第二个幻天使留作后备。"""
        snapshot = BoardSnapshot(
            frame_id=1,
            heroes=[
                _hero("angel_phantom", 1, 100, 100),
                _hero("snow", 1, 200, 100),
                _hero("angel_phantom", 1, 300, 100),
            ],
            capacity=BoardCapacity(total_slots=6, occupied=3),
        )
        candidates = snapshot.find_merge_candidates(
            main_c="assault", wildcard_heroes=frozenset({"angel_phantom"})
        )
        assert len(candidates) == 1
        pair = candidates[0]
        assert "snow" in {pair.hero_a.hero_type, pair.hero_b.hero_type}

    def test_wildcard_with_main_c_marks_main_c(self) -> None:
        """幻天使+主C → is_main_c=True（受 avoid_main_c_merge 保护）。"""
        snapshot = BoardSnapshot(
            frame_id=1,
            heroes=[
                _hero("angel_phantom", 1, 100, 100),
                _hero("assault", 1, 200, 100),
            ],
            capacity=BoardCapacity(total_slots=6, occupied=2),
        )
        candidates = snapshot.find_merge_candidates(
            main_c="assault", wildcard_heroes=frozenset({"angel_phantom"})
        )
        assert len(candidates) == 1
        assert candidates[0].is_main_c is True
        assert candidates[0].is_wildcard is True

    def test_wildcard_order_after_same_type(self) -> None:
        """候选顺序：非主C同类项 → 非主C通配 → 主C。"""
        snapshot = BoardSnapshot(
            frame_id=1,
            heroes=[
                _hero("assault", 1, 100, 100),
                _hero("assault", 1, 200, 100),
                _hero("snow", 1, 300, 100),
                _hero("snow", 1, 400, 100),
                _hero("angel_phantom", 1, 500, 100),
                _hero("death_knight", 1, 600, 100),
            ],
            capacity=BoardCapacity(total_slots=8, occupied=6),
        )
        candidates = snapshot.find_merge_candidates(
            main_c="assault", wildcard_heroes=frozenset({"angel_phantom"})
        )
        assert len(candidates) == 3
        assert candidates[0].hero_a.hero_type == "snow"
        assert candidates[0].is_wildcard is False
        assert candidates[1].is_wildcard is True
        assert candidates[2].is_main_c is True

    def test_default_no_wildcard_behavior_unchanged(self) -> None:
        """不传 wildcard_heroes 时行为与现状一致：幻天使不参与任何配对。"""
        snapshot = BoardSnapshot(
            frame_id=1,
            heroes=[
                _hero("angel_phantom", 1, 100, 100),
                _hero("snow", 1, 200, 100),
                _hero("angel_phantom", 1, 300, 100),
            ],
            capacity=BoardCapacity(total_slots=6, occupied=3),
        )
        candidates = snapshot.find_merge_candidates(main_c="assault")
        assert candidates == []
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_board_models.py -q`
Expected: FAIL（8 个新用例报 `TypeError: unexpected keyword 'wildcard_heroes'` 或 `AttributeError: is_wildcard`），既有用例仍 PASS。

- [ ] **Step 3: 实现——替换 MergeCandidate 类**

在 `src/wlxq_bot/models.py` 中找到下面原文（整块替换）：

```python
@dataclass(frozen=True)
class MergeCandidate:
    """合法合成对：同类型同星级的两个英雄。

    game-rules.md 合成规则：只有英雄类型相同且星级相同的两个英雄才能合成。

    Attributes:
        hero_a: 第一个英雄
        hero_b: 第二个英雄
        is_main_c: 是否涉及主 C（影响合成优先级，非主 C 优先合成）
    """

    hero_a: BoardHero
    hero_b: BoardHero
    is_main_c: bool = False
```

替换为：

```python
@dataclass(frozen=True)
class MergeCandidate:
    """合法合成对：同类型同星级的两个英雄，或通配英雄与任意同星级英雄。

    game-rules.md 合成规则：英雄类型相同且星级相同的两个英雄能合成；
    通配英雄（幻天使，2026-09-13 确认）还可与任意同星级英雄合成，
    合并产物随机。对中任一英雄是主 C 即视为涉及主 C（数量保护）。

    Attributes:
        hero_a: 第一个英雄
        hero_b: 第二个英雄
        is_main_c: 是否涉及主 C（影响合成优先级与数量保护，非主 C 优先合成）
        is_wildcard: 对中含通配英雄（幻天使类赖子；合成产物随机）
    """

    hero_a: BoardHero
    hero_b: BoardHero
    is_main_c: bool = False
    is_wildcard: bool = False
```

- [ ] **Step 4: 实现——替换 find_merge_candidates 方法**

在 `src/wlxq_bot/models.py` 中找到下面原文（整块替换，从 `def find_merge_candidates` 到 `return candidates`）：

```python
    def find_merge_candidates(self, main_c: str) -> list[MergeCandidate]:
        """找合法合成对：同类型同星级。

        每个英雄最多参与一次配对。多个候选同时存在时，非主 C 优先
        （game-rules.md：优先合成非主 C 合法对）。

        Args:
            main_c: 当前主 C 的 hero_type，如 ``"assault"``

        Returns:
            合成对列表，非主 C 对排前面。
        """
        candidates: list[MergeCandidate] = []
        used: set[int] = set()
        for i, hero_a in enumerate(self.heroes):
            if i in used:
                continue
            for j in range(i + 1, len(self.heroes)):
                if j in used:
                    continue
                hero_b = self.heroes[j]
                if hero_a.hero_type == hero_b.hero_type and hero_a.star_level == hero_b.star_level:
                    is_mc = hero_a.hero_type == main_c
                    candidates.append(
                        MergeCandidate(
                            hero_a=hero_a,
                            hero_b=hero_b,
                            is_main_c=is_mc,
                        )
                    )
                    used.add(i)
                    used.add(j)
                    break
        # 非主 C 优先（is_main_c=False 排前面）
        candidates.sort(key=lambda c: c.is_main_c)
        return candidates
```

替换为：

```python
    def find_merge_candidates(
        self,
        main_c: str,
        wildcard_heroes: frozenset[str] = frozenset(),
    ) -> list[MergeCandidate]:
        """找合法合成对：同类项同星级，通配英雄可与任意同星级英雄配对。

        两阶段配对（2026-09-13 幻天使定案，spec §4.4）：先排非通配英雄
        之间的同类项对，再为每个未配对的通配英雄找同星搭档（优先非通配，
        其次另一个通配）。每个英雄最多参与一次配对。多个候选同时存在时
        非主 C 优先（game-rules.md：优先合成非主 C 合法对），同为非主 C
        时同类项对排在通配对前。

        Args:
            main_c: 当前主 C 的 hero_type，如 ``"assault"``
            wildcard_heroes: 通配合并英雄集合（如 ``frozenset({"angel_phantom"})``）；
                空集合 = 仅同类项配对，行为与未引入通配前完全一致。

        Returns:
            合成对列表，非主 C 对排前面。
        """
        candidates: list[MergeCandidate] = []
        used: set[int] = set()

        def _is_main_c_pair(a: BoardHero, b: BoardHero) -> bool:
            return a.hero_type == main_c or b.hero_type == main_c

        # 阶段 1：非通配英雄之间的同类项对
        for i, hero_a in enumerate(self.heroes):
            if i in used or hero_a.hero_type in wildcard_heroes:
                continue
            for j in range(i + 1, len(self.heroes)):
                if j in used:
                    continue
                hero_b = self.heroes[j]
                if hero_b.hero_type in wildcard_heroes:
                    continue
                if hero_a.hero_type == hero_b.hero_type and hero_a.star_level == hero_b.star_level:
                    candidates.append(
                        MergeCandidate(
                            hero_a=hero_a,
                            hero_b=hero_b,
                            is_main_c=_is_main_c_pair(hero_a, hero_b),
                        )
                    )
                    used.add(i)
                    used.add(j)
                    break

        # 阶段 2：通配英雄与任意同星搭档（优先非通配搭档）
        for i, hero_a in enumerate(self.heroes):
            if i in used or hero_a.hero_type not in wildcard_heroes:
                continue
            partner_index: int | None = None
            for prefer_wildcard in (False, True):
                for j, hero_b in enumerate(self.heroes):
                    if j == i or j in used:
                        continue
                    if (hero_b.hero_type in wildcard_heroes) != prefer_wildcard:
                        continue
                    if hero_b.star_level == hero_a.star_level:
                        partner_index = j
                        break
                if partner_index is not None:
                    break
            if partner_index is None:
                continue
            hero_b = self.heroes[partner_index]
            candidates.append(
                MergeCandidate(
                    hero_a=hero_a,
                    hero_b=hero_b,
                    is_main_c=_is_main_c_pair(hero_a, hero_b),
                    is_wildcard=True,
                )
            )
            used.add(i)
            used.add(partner_index)

        # 非主 C 优先（is_main_c=False 排前面）；稳定排序保持同类项在前、通配在后
        candidates.sort(key=lambda c: c.is_main_c)
        return candidates
```

- [ ] **Step 5: 运行确认通过**

Run: `python -m pytest tests/test_board_models.py -v`
Expected: 全部 PASS（含 8 个新用例）。

- [ ] **Step 6: 提交**

```bash
git add src/wlxq_bot/models.py tests/test_board_models.py
git commit -m "feat(models): 合成候选支持通配英雄两阶段配对与 is_wildcard 标记"
```

---

### Task 2: config.py —— wildcard_merge_heroes 与 lineup_variants 配置键

**Files:**
- Modify: `src/wlxq_bot/config.py`（RunConfig 类、HeroClassifierConfig 类、新增模块级校验函数）
- Test: `tests/test_config.py`（追加测试类）

**Interfaces:**
- Produces: `RunConfig.wildcard_merge_heroes: list[str]`（默认 `[]`）；
  `HeroClassifierConfig.lineup_variants: list[str]`（默认 `[]`）。
  后续任务依赖这两个字段名。

- [ ] **Step 1: 写失败测试**

先确认 `tests/test_config.py` 顶部 import 中已有 `DefaultConfig`（第 15 行附近）。检查是否已 import `HeroClassifierConfig`、`RunConfig`、`MainCProfile`、`pytest`、`ValidationError`；缺哪个就在 import 块补哪个（pydantic 的 `ValidationError` 来自 `from pydantic import ValidationError`）。然后在文件末尾追加：

```python
class TestWildcardMergeAndVariants:
    """幻天使通配合并配置键（2026-09-13 spec §4.3）。"""

    def test_lineup_variants_default_and_parse(self) -> None:
        cfg = HeroClassifierConfig()
        assert cfg.lineup_variants == []
        cfg = HeroClassifierConfig(lineup_variants=["angel_phantom"])
        assert cfg.lineup_variants == ["angel_phantom"]

    def test_lineup_variants_reject_invalid(self) -> None:
        with pytest.raises(ValidationError, match="非法英雄标识"):
            HeroClassifierConfig(lineup_variants=["Angel-Phantom"])

    def test_variants_do_not_require_main_c_profile(self) -> None:
        """变体不是主 C 英雄：不在 main_c_profiles 也能通过 lineup 子集校验。"""
        cfg = DefaultConfig(
            main_c_profiles={
                "snow": MainCProfile(display_name="雪姬", hero_template_dir="heroes/snow")
            },
            hero_classifier=HeroClassifierConfig(
                lineup_others=["snow"], lineup_variants=["angel_phantom"]
            ),
        )
        assert "angel_phantom" not in cfg.main_c_profiles

    def test_wildcard_merge_heroes_parse_and_reject(self) -> None:
        run = RunConfig(wildcard_merge_heroes=["angel_phantom"])
        assert run.wildcard_merge_heroes == ["angel_phantom"]
        with pytest.raises(ValidationError, match="非法英雄标识"):
            RunConfig(wildcard_merge_heroes=["9bad"])
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_config.py -q`
Expected: 新用例 FAIL（字段不存在），既有用例 PASS。

- [ ] **Step 3: 实现——新增共享校验函数**

在 `src/wlxq_bot/config.py` 中找到 `class RunConfig(BaseModel):` 这一行，在它**上方**插入：

```python
def _parse_hero_identifier_list(value: Any, label: str) -> list[str]:
    """把配置中的英雄标识列表规范化；非法标识（非小写英文/数字/下划线）报错。"""
    if value is None:
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in value if str(item).strip()]
    for item in items:
        if re.fullmatch(r"[a-z][a-z0-9_]*", item) is None:
            raise ValueError(f"{label} 含非法英雄标识: {item!r}")
    return items


```

- [ ] **Step 4: 实现——RunConfig 加字段**

在 `src/wlxq_bot/config.py` 的 RunConfig 类中找到下面原文：

```python
    # 批次合成：一次棋盘识别上把互不相交的非主C 1/2星合法对一批拖完再统一
    # 识别；主C对与3星对不进批次（仍走单动作策略），拖动未生效的对在批末
    # 按位置签名归因记入失败记忆
    batch_merge_enabled: bool = True
```

在它后面（同缩进）追加：

```python
    # 通配合并英雄（赖子，如幻天使）：可与任意同星级英雄合并；合成优先级
    # 同类项对在前、通配对棋盘占满才用（2026-09-13 用户定案，spec
    # docs/superpowers/specs/2026-09-13-angel-phantom-split-design.md）
    wildcard_merge_heroes: list[str] = Field(default_factory=list)

    @field_validator("wildcard_merge_heroes", mode="before")
    @classmethod
    def _validate_wildcard_merge_heroes(cls, value: Any) -> list[str]:
        return _parse_hero_identifier_list(value, "wildcard_merge_heroes")
```

- [ ] **Step 5: 实现——HeroClassifierConfig 加字段并复用校验**

在 `src/wlxq_bot/config.py` 中把下面原文整块替换：

```python
class HeroClassifierConfig(BaseModel):
    """英雄格分类器数据制作配置（configs/default.yaml）。

    lineup_others 列出主 C 之外、本局固定携带的队友英雄标识；离线裁剪时
    据此与 main_c 一起预建 labeled/<英雄>/star1~4 标注目录，省去人工建目录。
    """

    lineup_others: list[str] = Field(default_factory=list)

    @field_validator("lineup_others", mode="before")
    @classmethod
    def _validate_lineup_others(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",") if item.strip()]
        else:
            items = [str(item).strip() for item in value if str(item).strip()]
        for item in items:
            if re.fullmatch(r"[a-z][a-z0-9_]*", item) is None:
                raise ValueError(f"lineup_others 含非法英雄标识: {item!r}")
        return items
```

替换为：

```python
class HeroClassifierConfig(BaseModel):
    """英雄格分类器数据制作配置（configs/default.yaml）。

    lineup_others 列出主 C 之外、本局固定携带的队友英雄标识；离线裁剪时
    据此与 main_c 一起预建 labeled/<英雄>/star1~4 标注目录，省去人工建目录。

    lineup_variants 列出阵容英雄的棋盘形态变体（如幻天使 angel_phantom）：
    不是独立英雄、不建主 C 档案、永不作为主 C；仅用于放行识别白名单
    （_restrict_prediction），让变体预测结果不被拒为 hero_not_in_lineup。
    """

    lineup_others: list[str] = Field(default_factory=list)
    lineup_variants: list[str] = Field(default_factory=list)

    @field_validator("lineup_others", mode="before")
    @classmethod
    def _validate_lineup_others(cls, value: Any) -> list[str]:
        return _parse_hero_identifier_list(value, "lineup_others")

    @field_validator("lineup_variants", mode="before")
    @classmethod
    def _validate_lineup_variants(cls, value: Any) -> list[str]:
        return _parse_hero_identifier_list(value, "lineup_variants")
```

（注意保持报错文案 `含非法英雄标识` 不变，既有测试依赖此文案。）

- [ ] **Step 6: 运行确认通过**

Run: `python -m pytest tests/test_config.py -q`
Expected: 全部 PASS。

- [ ] **Step 7: 提交**

```bash
git add src/wlxq_bot/config.py tests/test_config.py
git commit -m "feat(config): 新增 wildcard_merge_heroes 与 lineup_variants 配置键"
```

---

### Task 3: strategy.py 三级决策层级 + coop.py 调用点

**Files:**
- Modify: `src/wlxq_bot/tasks/strategy.py`（select_merge、select_merge_batch 两个方法）
- Modify: `src/wlxq_bot/tasks/coop.py`（__init__ 一处、find_merge_candidates 两处、select_merge_batch 一处）
- Create: `tests/test_cultivation_strategy.py`

**Interfaces:**
- Consumes: Task 1 的 `MergeCandidate.is_wildcard` 与 `find_merge_candidates(main_c, wildcard_heroes)`；Task 2 的 `RunConfig.wildcard_merge_heroes`。
- Produces: `CultivationStrategy.select_merge(board, candidates, failed_pairs)`（签名不变，行为加层级）；`select_merge_batch(candidates, failed_pairs, board)`（**新增第三个位置参数 board**，coop.py 唯一调用点同步改）。

- [ ] **Step 1: 新建失败测试 `tests/test_cultivation_strategy.py`**

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_cultivation_strategy.py -q`
Expected: 新用例 FAIL（`is_wildcard` 不存在 / 批次签名不符），`test_same_type_low_pair_merges_with_space` 可能已 PASS。

- [ ] **Step 3: 实现——替换 strategy.py 的 select_merge 方法**

在 `src/wlxq_bot/tasks/strategy.py` 中把下面原文整块替换（从 `def select_merge` 到该方法 `return pair`）：

```python
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
```

替换为：

```python
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
```

- [ ] **Step 4: 实现——替换 strategy.py 的 select_merge_batch 方法**

在 `src/wlxq_bot/tasks/strategy.py` 中把下面原文整块替换：

```python
    def select_merge_batch(
        self,
        candidates: list[MergeCandidate],
        failed_pairs: FailedMergePairs,
    ) -> list[MergeCandidate]:
        """批次合成规划（2026-09-10 用户策略）：把当前快照上能合的对一次排完。

        与单次 select_merge 的取舍不同，批次只做无争议的合并：
        1. 绝不动主C（主C强袭保留，数量保护；主C对仍由单动作策略在棋盘
           占满等最后手段场景处理）；
        2. 只排 1/2 星对（3星对合成4星会弹赠送技能页，可能中途遮挡棋盘）；
        3. 跳过失败记忆中的对（本局内已确认拖不动的对不反复排）；
        4. 对与对之间格子互不相交（find_merge_candidates 已按英雄去重配对，
           此处按位置再防御性过滤），合并互不冲突，可连续拖完再统一识别。
        """
        selected: list[MergeCandidate] = []
        used: set[tuple[int, int]] = set()
        for candidate in candidates:
            if candidate.is_main_c:
                continue
            if max(candidate.hero_a.star_level, candidate.hero_b.star_level) >= 3:
                continue
            if self.pair_signature(candidate) in failed_pairs:
                continue
            positions = {candidate.hero_a.position, candidate.hero_b.position}
            if positions & used:
                continue
            used |= positions
            selected.append(candidate)
        return selected
```

替换为：

```python
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
            positions = {candidate.hero_a.position, candidate.hero_b.position}
            if positions & used:
                continue
            used |= positions
            selected.append(candidate)
        return selected
```

- [ ] **Step 5: 实现——coop.py 四处编辑**

5a. 在 `src/wlxq_bot/tasks/coop.py` 中找到：

```python
        self._strategy = build_strategy(mode_profile, ctx.main_c, run_config, main_c_profile)
```

替换为：

```python
        self._strategy = build_strategy(mode_profile, ctx.main_c, run_config, main_c_profile)
        # 通配合并英雄（幻天使）：来自 run.wildcard_merge_heroes，传给候选生成
        self._wildcard_heroes = frozenset(run_config.wildcard_merge_heroes)
```

5b. 找到培养阶段的调用（上文是 `_batch_summon_action` 的 return）：

```python
            return self._batch_summon_action(observation, window_ctx)
        candidates = board.find_merge_candidates(self.ctx.main_c)
```

替换为：

```python
            return self._batch_summon_action(observation, window_ctx)
        candidates = board.find_merge_candidates(
            self.ctx.main_c, wildcard_heroes=self._wildcard_heroes
        )
```

5c. 找到批次合成调用：

```python
            batch = self._strategy.select_merge_batch(candidates, self._failed_merge_pairs)
```

替换为：

```python
            batch = self._strategy.select_merge_batch(
                candidates, self._failed_merge_pairs, board
            )
```

5d. 找到金币等待合成的调用（下文是 `pair = self._strategy.select_merge`）：

```python
        candidates = board.find_merge_candidates(self.ctx.main_c)
        pair = self._strategy.select_merge(board, candidates, self._failed_merge_pairs)
```

替换为：

```python
        candidates = board.find_merge_candidates(
            self.ctx.main_c, wildcard_heroes=self._wildcard_heroes
        )
        pair = self._strategy.select_merge(board, candidates, self._failed_merge_pairs)
```

- [ ] **Step 6: 运行新测试与相关回归**

```bash
python -m pytest tests/test_cultivation_strategy.py tests/test_board_models.py -q
python -m pytest tests/test_coop_task.py -q
```
Expected: 全部 PASS（coop 任务集成测试不应受影响——默认配置无通配英雄时行为不变）。

- [ ] **Step 7: 提交**

```bash
git add src/wlxq_bot/tasks/strategy.py src/wlxq_bot/tasks/coop.py tests/test_cultivation_strategy.py
git commit -m "feat(task): 合成策略三级层级——幻天使占满才用，批次纳入通配对"
```

---

### Task 4: runner.py —— allowed 集合与 watch 覆盖校验扩展 lineup_variants

**Files:**
- Modify: `src/wlxq_bot/runner.py`（两处 allowed_heroes 构造 + 一处 required）

- [ ] **Step 1: 编辑 run 路径装配（约 L298）**

找到：

```python
            allowed_heroes={mc, *self.default_config.hero_classifier.lineup_others},
```

替换为：

```python
            allowed_heroes={
                mc,
                *self.default_config.hero_classifier.lineup_others,
                *self.default_config.hero_classifier.lineup_variants,
            },
```

- [ ] **Step 2: 编辑 watch 路径（约 L1184-1186）**

找到：

```python
            allowed_heroes={main_c, *self.default_config.hero_classifier.lineup_others},
        )
        required = {main_c, *self.default_config.hero_classifier.lineup_others}
```

替换为：

```python
            allowed_heroes={
                main_c,
                *self.default_config.hero_classifier.lineup_others,
                *self.default_config.hero_classifier.lineup_variants,
            },
        )
        required = {
            main_c,
            *self.default_config.hero_classifier.lineup_others,
            *self.default_config.hero_classifier.lineup_variants,
        }
```

- [ ] **Step 3: 回归**

```bash
python -m pytest tests/test_runner.py tests/test_run_cli.py -q
```
Expected: 全部 PASS。

- [ ] **Step 4: 提交**

```bash
git add src/wlxq_bot/runner.py
git commit -m "feat(runner): 识别白名单与 watch 覆盖校验纳入 lineup_variants"
```

---

### Task 5: configs/default.yaml —— 打开两个配置键

**Files:**
- Modify: `configs/default.yaml`

- [ ] **Step 1: run 段加通配键**

在 `configs/default.yaml` 中找到：

```yaml
  batch_merge_enabled: true
```

在其后插入：

```yaml
  # 通配合并英雄（赖子）：幻天使可与任意同星级英雄合并；同类项对优先、
  # 棋盘占满才使用（2026-09-13 定案，spec 2026-09-13-angel-phantom-split）
  wildcard_merge_heroes: [angel_phantom]
```

- [ ] **Step 2: hero_classifier 段加变体白名单**

在 `configs/default.yaml` 中找到：

```yaml
  lineup_others: [angel, snow, death_knight]
```

在其后插入：

```yaml
  # 阵容英雄的棋盘形态变体：幻天使是天使的形态而非新英雄，不建 main_c_profiles
  # 档案、永不作为主 C；仅放行识别白名单（否则被拒为 hero_not_in_lineup → unknown）
  lineup_variants: [angel_phantom]
```

- [ ] **Step 3: 验证配置加载**

```bash
python -c "
from pathlib import Path
from wlxq_bot.config import load_default_config
cfg = load_default_config(Path('configs/default.yaml'))
print('wildcard:', cfg.run.wildcard_merge_heroes)
print('variants:', cfg.hero_classifier.lineup_variants)
print('lineup:', cfg.hero_classifier.lineup_others)
"
```
Expected: 打印 `wildcard: ['angel_phantom']`、`variants: ['angel_phantom']`、`lineup: ['angel', 'snow', 'death_knight']`。

- [ ] **Step 4: 提交**

```bash
git add configs/default.yaml
git commit -m "feat(config): 启用幻天使通配合并与形态变体白名单"
```

---

### Task 6: 数据重分拣 dry-run —— ⛔ 用户确认门 1

**Files:**
- Create: `scripts/relabel_angel_phantom.py`

- [ ] **Step 1: 新建 `scripts/relabel_angel_phantom.py`（完整内容如下）**

```python
"""幻天使重分拣工具（2026-09-13，spec §4.1，双信号阈值经全量 2176 张校准）。

特征（饱和像素 = s>60 且 v>60，少于 30 像素视为无效）：
  hue_circ    环向 hue 均值（OpenCV 0~179）；正常天使参考≈24.3、幻天使参考≈141.9
  purple_frac hue∈[120,180) 占比；两参考分别 ≈0.003 / ≈0.391

判定（对 angel/star1~3；star4 不动）：
  hue_circ∈[105,180) 且 purple_frac>=0.15  移入 angel_phantom/starN
  hue_circ<80                               留在 angel
  其余（80~104 中间带/紫占比不足）         留在 angel 并列入人工复核清单

用法（仓库根执行）：
  python scripts/relabel_angel_phantom.py dry-run    # 只打印分布
  python scripts/relabel_angel_phantom.py apply      # 执行迁移并写审计 CSV
  python scripts/relabel_angel_phantom.py montage    # 生成人工复核拼图
"""

from __future__ import annotations

import csv
import math
import shutil
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("datasets/hero_classifier/assault/3000x2000/helper")
OUT_DIR = Path("outputs/hero_classifier") / f"relabel_review_{date.today():%Y-%m-%d}"
CIRC_MOVE_MIN = 105.0   # hue_circ >= 该值且 purple_frac 达标 → 迁移
CIRC_KEEP_MAX = 80.0    # hue_circ < 该值 → 留守
PF_MOVE_MIN = 0.15      # 迁移所需的 purple_frac 下限
REFS = {
    "正常天使参考(5B)": ROOT / "train/labeled/angel/star1/202608122001_frame000018_5B.png",
    "幻天使参考(6B)": ROOT / "train/labeled/angel/star1/202608122001_frame000108_6B.png",
}


def features(path: Path) -> tuple[float, float] | None:
    """返回 (hue_circ, purple_frac)；无效样本返回 None。"""
    img = cv2.imread(str(path))
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 1] > 60) & (hsv[..., 2] > 60)
    if int(mask.sum()) < 30:
        return None
    hue = hsv[..., 0][mask].astype(np.float32)
    ang = np.deg2rad(hue * 2.0)
    circ = (math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360.0) / 2.0
    pf = float(((hue >= 120) & (hue < 180)).sum() / mask.sum())
    return float(circ), pf


def collect() -> list[dict]:
    rows: list[dict] = []
    for split in ("train", "validation", "test"):
        base = ROOT / split / "labeled/angel"
        if not base.is_dir():
            continue
        for star_dir in sorted(base.glob("star[1-3]")):
            for path in sorted(star_dir.glob("*.png")):
                feat = features(path)
                rows.append(
                    {
                        "split": split,
                        "star": star_dir.name,
                        "path": path,
                        "circ": None if feat is None else feat[0],
                        "frac": None if feat is None else feat[1],
                    }
                )
    return rows


def decide(circ: float | None, frac: float | None) -> str:
    if circ is None or frac is None:
        return "skip"
    if CIRC_MOVE_MIN <= circ < 180 and frac >= PF_MOVE_MIN:
        return "move"
    if circ < CIRC_KEEP_MAX:
        return "keep"
    return "review"


def cmd_dry_run() -> None:
    for label, ref in REFS.items():
        feat = features(ref)
        print(f"{label}: hue_circ={feat[0]:.1f} purple_frac={feat[1]:.4f}")
    rows = collect()
    print(f"共扫描 {len(rows)} 张（train/validation/test 的 angel/star1~3）")
    for split in ("train", "validation", "test"):
        sub = [r for r in rows if r["split"] == split]
        if not sub:
            continue
        for star in ("star1", "star2", "star3"):
            grp = [r for r in sub if r["star"] == star]
            if not grp:
                continue
            buckets = {"move": 0, "keep": 0, "review": 0, "skip": 0}
            for r in grp:
                buckets[decide(r["circ"], r["frac"])] += 1
            print(
                f"{split}/{star}: 共{len(grp)} 移{buckets['move']} 留{buckets['keep']} "
                f"待复核{buckets['review']} 无效{buckets['skip']}"
            )


def cmd_apply() -> None:
    rows = collect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "relabel_angel_phantom.csv"
    moved = kept = review = 0
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "hue_circ", "purple_frac", "decision", "new_path"])
        for r in rows:
            action = decide(r["circ"], r["frac"])
            circ_text = "" if r["circ"] is None else f"{r['circ']:.1f}"
            frac_text = "" if r["frac"] is None else f"{r['frac']:.4f}"
            new_path = ""
            if action == "move":
                target = (
                    ROOT
                    / r["split"]
                    / "labeled/angel_phantom"
                    / r["star"]
                    / r["path"].name
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(r["path"]), str(target))
                new_path = str(target)
                moved += 1
            elif action == "review":
                review += 1
            else:
                kept += 1
            writer.writerow([str(r["path"]), circ_text, frac_text, action, new_path])
    print(f"迁移 {moved} 张到 angel_phantom；留在 angel {kept} 张；待复核 {review} 张")
    print(f"审计 CSV: {csv_path}")


def _tile(items: list[tuple[str, Path]], cols: int = 8, scale: int = 3) -> np.ndarray:
    cell_w, cell_h = 64 * scale, 68 * scale
    label_h = 22
    rows_n = math.ceil(len(items) / cols)
    sheet = np.full((rows_n * (cell_h + label_h), cols * cell_w, 3), 255, np.uint8)
    for idx, (name, path) in enumerate(items):
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
        r, c = divmod(idx, cols)
        y0, x0 = r * (cell_h + label_h), c * cell_w
        sheet[y0 : y0 + cell_h, x0 : x0 + cell_w] = img
        cv2.putText(
            sheet,
            name[:32],
            (x0 + 2, y0 + cell_h + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return sheet


def _spread(items: list[tuple[str, Path]], limit: int) -> list[tuple[str, Path]]:
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def cmd_montage() -> None:
    out = OUT_DIR / "montage"
    out.mkdir(parents=True, exist_ok=True)
    rows = collect()
    # 迁移全景：angel_phantom 各星级全部
    for split in ("train", "validation", "test"):
        base = ROOT / split / "labeled/angel_phantom"
        if not base.is_dir():
            continue
        for star_dir in sorted(base.glob("star[1-3]")):
            pngs = sorted(star_dir.glob("*.png"))
            if not pngs:
                continue
            sheet = _tile([(p.name, p) for p in pngs])
            cv2.imwrite(str(out / f"moved_{split}_{star_dir.name}.png"), sheet)
            print(f"moved_{split}_{star_dir.name}.png: {len(pngs)} 张")
    # 留守抽样（查漏迁）：各 split/star 留下的图按序抽 48 张
    for split in ("train", "validation", "test"):
        sub = [r for r in rows if r["split"] == split and decide(r["circ"], r["frac"]) == "keep"]
        for star in ("star1", "star2", "star3"):
            grp = [(r["path"].name, r["path"]) for r in sub if r["star"] == star]
            picked = _spread(grp, 48)
            if not picked:
                continue
            sheet = _tile(picked)
            cv2.imwrite(str(out / f"kept_{split}_{star}.png"), sheet)
            print(f"kept_{split}_{star}.png: 抽样 {len(picked)} 张")
    # 中间带复核清单
    review_items = [
        (r["path"].name, r["path"]) for r in rows if decide(r["circ"], r["frac"]) == "review"
    ]
    for i in range(0, len(review_items), 96):
        sheet = _tile(review_items[i : i + 96])
        cv2.imwrite(str(out / f"review_{i // 96:02d}.png"), sheet)
    print(f"review_*.png: 中间带共 {len(review_items)} 张")
    print(f"复核包目录: {OUT_DIR}")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"dry-run", "apply", "montage"}:
        print(__doc__)
        sys.exit(2)
    {"dry-run": cmd_dry_run, "apply": cmd_apply, "montage": cmd_montage}[sys.argv[1]]()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行 dry-run**

Run: `python scripts/relabel_angel_phantom.py dry-run`
Expected: 参考图打印 `正常天使参考(5B): hue_circ=24.3 purple_frac=0.0026`、`幻天使参考(6B): hue_circ=141.9 purple_frac=0.3909`；各 split/star 打印移/留/待复核数量，全量预期约为：train star1 93 / star2 51 / star3 132 迁移，validation 36，test 57，待复核合计约 40（±10 以内属正常）；star4 不在扫描范围。

- [ ] **Step 3: ⛔ 硬门 1——用户确认 dry-run**

把 dry-run 完整输出发给用户，等用户回复"继续"。
用户判断标准：两张参考图分别落在一头一尾（5B 应为留守特征、6B 应为迁移特征）、迁移/留守数量与上述预期同数量级、待复核占比小。分布不干净则停下按用户指示调整阈值（只允许改 `CIRC_MOVE_MIN`/`CIRC_KEEP_MAX`/`PF_MOVE_MIN` 三个常量并重新 dry-run）。

- [ ] **Step 4: 提交脚本**

```bash
git add scripts/relabel_angel_phantom.py
git commit -m "feat(scripts): 幻天使色彩重分拣工具（dry-run/apply/montage）"
```

---

### Task 7: 执行迁移 + 复核包 —— ⛔ 用户确认门 2

**Files:**
- Modify: `datasets/hero_classifier/assault/3000x2000/helper/{train,validation,test}/labeled/angel_phantom/star*/`（新增目录与迁移 PNG）

- [ ] **Step 1: 执行迁移**

Run: `python scripts/relabel_angel_phantom.py apply`
Expected: 打印 `迁移 360 张左右 / 留守 1776 张左右 / 待复核 40 张左右`（±10 属正常）与审计 CSV 路径（`outputs/hero_classifier/relabel_review_<日期>/relabel_angel_phantom.csv`）。

- [ ] **Step 2: 生成复核包**

Run: `python scripts/relabel_angel_phantom.py montage`
Expected: 生成 `moved_*.png`（迁移全景）、`kept_*.png`（留守抽样）、`review_*.png`（中间带）。

- [ ] **Step 3: 自检统计**

```bash
python -c "
from pathlib import Path
root = Path('datasets/hero_classifier/assault/3000x2000/helper')
for split in ('train','validation','test'):
    for hero in ('angel','angel_phantom'):
        for star in ('star1','star2','star3','star4'):
            d = root/split/'labeled'/hero/star
            n = len(list(d.glob('*.png'))) if d.is_dir() else 0
            if n: print(f'{split}/{hero}/{star}: {n}')
"
```
Expected: angel_phantom 只出现在 star1~3（绝无 star4），数量与 dry-run 的 move 数一致。

- [ ] **Step 4: ⛔ 硬门 2——用户人工复核**

把复核包目录路径发给用户，请其逐张查看 `moved_*.png`（查错迁）与 `kept_*.png`（查漏迁）、`review_*.png`（现场改判）。用户发现问题图时按用户指示手工移动文件（`git mv` 或资源管理器移动均可），然后重跑 Step 3 统计给用户复核，直至用户回复"确认"。
**用户确认前不得执行 sync-labels 与训练。**

- [ ] **Step 5: 提交数据迁移**

```bash
git add datasets/hero_classifier/assault/3000x2000/helper
git commit -m "data(hero_classifier): angel 拆分幻天使样本至 angel_phantom（star1~3）"
```
（若 `datasets/` 未被 git 跟踪，`git status` 会无变化——此时跳过提交，在最终报告中注明。）

---

### Task 8: sync-labels + 后台训练

**Files:**
- Modify: `datasets/.../helper/{train,validation,test}/dataset_manifest.csv`（工具重建）

- [ ] **Step 1: 重建三个 split 的清单**

```bash
python -m wlxq_bot hero-classifier sync-labels datasets/hero_classifier/assault/3000x2000/helper --split train
python -m wlxq_bot hero-classifier sync-labels datasets/hero_classifier/assault/3000x2000/helper --split validation
python -m wlxq_bot hero-classifier sync-labels datasets/hero_classifier/assault/3000x2000/helper --split test
```
Expected: 每条打印 `✓ 标签清单已完整重建` 与样本数（train 约 9399、validation 约 918、test 约 470）。

- [ ] **Step 2: 启动后台训练**

```bash
mkdir -p outputs/hero_classifier/assault-helper-v5
python -m wlxq_bot hero-classifier train datasets/hero_classifier/assault/3000x2000/helper -o outputs/hero_classifier/assault-helper-v5 > outputs/hero_classifier/assault-helper-v5/train.log 2>&1 &
```
（在支持后台执行的环境中运行；否则前台执行，预计 30~120 分钟，CPU 训练。）

Expected 完成标志：`outputs/hero_classifier/assault-helper-v5/` 下出现 `hero_classifier.pt`、`hero_classifier.onnx`、`hero_classifier.json`、`training_history.json`，train.log 末尾打印 `✓ 英雄格分类器训练完成`、最佳轮次与最佳验证准确率（参考 v4 基线 0.987）。

- [ ] **Step 3: 校验类别表**

```bash
python -c "
import json
meta = json.load(open('outputs/hero_classifier/assault-helper-v5/hero_classifier.json', encoding='utf-8'))
names = meta['class_names']
print(names)
assert 'angel_phantom_star1' in names and 'angel_phantom_star2' in names and 'angel_phantom_star3' in names
assert 'angel_phantom_star4' not in names
print('类别表 OK')
"
```
Expected: 打印类别列表含 `angel_phantom_star1~3`，无 star4，断言通过。

- [ ] **Step 4: 提交清单变化（如有跟踪）**

```bash
git add datasets/hero_classifier/assault/3000x2000/helper
git commit -m "data(hero_classifier): sync-labels 重建含 angel_phantom 的清单" || echo "datasets 未跟踪，跳过"
```

---

### Task 9: 评估与验收 —— ⛔ 不达标必须停下

- [ ] **Step 1: 运行评估**

```bash
python -m wlxq_bot hero-classifier evaluate datasets/hero_classifier/assault/3000x2000/helper --model outputs/hero_classifier/assault-helper-v5/hero_classifier.onnx -o outputs/hero_classifier/evaluation-v5
```
Expected: 打印样本/接受/拒绝与总体、已接受准确率，报告写到 `outputs/hero_classifier/evaluation-v5/evaluation_report.json`。

- [ ] **Step 2: 验收判定**

```bash
python - <<'EOF'
import json

new = json.load(open("outputs/hero_classifier/evaluation-v5/evaluation_report.json", encoding="utf-8"))
old = json.load(open("outputs/hero_classifier/evaluation/evaluation_report.json", encoding="utf-8"))

def is_phantom(name): return name.startswith("angel_phantom")
def is_angel(name): return name.startswith("angel") and not is_phantom(name)

cross = 0
detail = []
for exp, preds in new["confusion"].items():
    for pred, n in preds.items():
        if (is_angel(exp) and is_phantom(pred)) or (is_phantom(exp) and is_angel(pred)):
            cross += n
            detail.append((exp, pred, n))
print("angel<->angel_phantom 互混:", cross, detail)
print("v5 accuracy_accepted:", round(new["accuracy_accepted"], 4), " 基线:", round(old["accuracy_accepted"], 4))
assert cross <= 3, f"互混 {cross} 张超过验收标准 3 张"
assert new["accuracy_accepted"] >= old["accuracy_accepted"] - 0.005, "已接受准确率回归超过 0.5 个百分点"
print("验收通过")
EOF
```
Expected: 打印 `验收通过`。基线 `old` 约 0.9044（v4 在 test 上的 accuracy_accepted）。
⛔ 若断言失败：**不要切模型**，把混淆明细和准确率发给用户，等待指示。

- [ ] **Step 3: 参考图端到端推理验证**

```bash
python - <<'EOF'
from pathlib import Path
import cv2
from wlxq_bot.perception.hero_classifier import HeroCellClassifier

clf = HeroCellClassifier(Path("outputs/hero_classifier/assault-helper-v5/hero_classifier.onnx"))
base = Path("datasets/hero_classifier/assault/3000x2000/helper/train/labeled/angel/star1")
normal = cv2.imread(str(base / "202608122001_frame000018_5B.png"))
phantom = cv2.imread(str(base / "202608122001_frame000108_6B.png"))
for name, pred in zip(("正常天使", "幻天使"), clf.predict([normal, phantom])):
    print(name, "->", pred.class_name, f"conf={pred.confidence:.3f}", "rejected=", pred.rejected)
EOF
```
Expected: `正常天使 -> angel_star1`、`幻天使 -> angel_phantom_star1`，两者 `rejected= False`。
（注意：这两张参考图在 train 集内，此验证只确认模型学到区分，泛化能力由 Step 1/2 的 test 集保证。）

---

### Task 10: 切换模型配置

**Files:**
- Modify: `configs/default.yaml`（assault 档案模型路径一行）

- [ ] **Step 1: 编辑模型路径**

在 `configs/default.yaml` 中找到（assault 档案下，路径特征是 `assault-helper`）：

```yaml
    hero_classifier_model: outputs/hero_classifier/assault-helper/hero_classifier.onnx
```

替换为：

```yaml
    hero_classifier_model: outputs/hero_classifier/assault-helper-v5/hero_classifier.onnx
```

（⚠️ 文件里有六个相似行，只改路径含 `assault-helper/` 的这一行，angel/monkey/snow 等档案不动。）

- [ ] **Step 2: 全量回归**

Run: `python -m pytest -q`
Expected: 全部 PASS。

- [ ] **Step 3: 提交**

```bash
git add configs/default.yaml
git commit -m "feat(config): 强袭英雄分类器切换 assault-helper-v5（含幻天使类别）"
```

---

### Task 11: 文档同步

**Files:**
- Modify: `docs/game-rules.md`、`docs/coop/TODO.md`、`docs/architecture.md`、`docs/coop/hero-classifier-usage.md`

- [ ] **Step 1: game-rules.md —— 修订合成总则**

找到：

```markdown
只有英雄类型相同且星级相同的两个英雄才能合成。
```

替换为：

```markdown
只有英雄类型相同且星级相同的两个英雄才能合成；唯一例外是幻天使（见「幻天使」一节），它可与任意同星级英雄合成。
```

- [ ] **Step 2: game-rules.md —— 新增幻天使一节**

找到 `## 合成结果随机规则` 标题行，在其**上方**插入：

```markdown
## 幻天使

天使在棋盘上有两种形态：正常天使（金色调）与幻天使（紫色调，赖子型）。已确认规则（用户实机确认 2026-09-13）：

- 召唤、合并均可能出现幻天使；4 星天使必为正常天使；幻天使只有 1~3 星。
- 幻天使可与任意同星级英雄合并，也可与另一幻天使合并。
- 幻天使参与的合并，产物随机：可能是本局任何同高一星英雄，也可能仍是幻天使。
- 幻天使是辅助英雄（形态变体），永远不会当主 C；识别类别为独立标识 `angel_phantom`。
- 3 星幻天使参与合成同样触发 4 星赠送技能页。
- 自动化合成优先级（用户定案 2026-09-13）：同类项对（两个正常天使/两个雪姬等）优先；幻天使对棋盘有空位时先召唤不花幻天使、占满才用（与 3 星对规避同一哲学）。

```

- [ ] **Step 3: game-rules.md —— 合成结果随机规则补幻天使**

找到：

```markdown
合成会消耗两个输入英雄并生成一个高一星英雄。新英雄的类型不继承输入英雄，而是从本局携带的 4 种英雄中随机产生。
```

替换为：

```markdown
合成会消耗两个输入英雄并生成一个高一星英雄。新英雄的类型不继承输入英雄，而是从本局携带的 4 种英雄中随机产生（2026-09-13 确认：幻天使参与的合成结果也可能是幻天使）。
```

- [ ] **Step 4: architecture.md —— 英雄列表补变体说明**

找到：

```markdown
- `fox`：狐狸。
```

替换为：

```markdown
- `fox`：狐狸。

幻天使（`angel_phantom`）不是独立主 C 英雄：它是天使的棋盘形态变体（`hero_classifier.lineup_variants` 白名单放行识别），仅 star1~3，作为通配合并英雄（`run.wildcard_merge_heroes`）参与合成决策，永不作为主 C。
```

- [ ] **Step 5: hero-classifier-usage.md —— 目录结构补变体**

找到：

```text
      angel/star1/
```

替换为：

```text
      angel/star1/
      angel_phantom/star1/
```

（该目录结构示意代码块中 angel/star1/ 之后的 star2~4 行保持原样，不必逐行补全 angel_phantom 的其他星级。）

- [ ] **Step 6: coop/TODO.md —— 追加实机验证项**

在文件末尾追加：

```markdown

## 幻天使（angel_phantom）实机验证（2026-09-13 拆分后待验证）

- [ ] 棋盘日志中出现 `angel_phantom{1,2,3}星@…` 且位置与实际相符（对照参考截图）。
- [ ] 幻天使与同星级英雄拖动合成成功；产物随机不导致流程卡住（动作后重识别正常）。
- [ ] 同类项对与幻天使对并存时先合同类项；棋盘有空位时幻天使不被消耗（先召唤）。
- [ ] 3 星幻天使对在有空位时被规避；占满被迫合成后赠送技能页确认流程正常走通。
- [ ] 幻天使参与的合成产物若是阵容外英雄：该格应识别为 unknown 保守处理（占用统计不受影响），实机观察是否需要专门策略。
- [ ] 幻天使模板未采集（模板备用路径不支持；主链路 ONNX 不受影响），需要时再采集。
```

- [ ] **Step 7: README 检查**

Run: `grep -n "lineup_others\|angel_phantom" README.md`
Expected: 无输出 → README 无需修改。若有输出且涉及阵容描述，按上述文档口径补充后一并提交。

- [ ] **Step 8: 提交**

```bash
git add docs/game-rules.md docs/coop/TODO.md docs/architecture.md docs/coop/hero-classifier-usage.md README.md
git commit -m "docs: 幻天使识别与通配合成规则、实机验证待办同步"
```
（README 无改动时从 add 列表去掉。）

---

### Task 12: 全量验证与收尾

- [ ] **Step 1: 全量测试**

Run: `python -m pytest -q`
Expected: 全部 PASS，无跳过的意外项。

- [ ] **Step 2: 冒烟验证识别链路配置**

```bash
python -c "
from wlxq_bot.perception.hero_classifier import HeroCellClassifier
from pathlib import Path
clf = HeroCellClassifier(Path('outputs/hero_classifier/assault-helper-v5/hero_classifier.onnx'))
print('supported:', sorted(clf.supported_heroes))
assert 'angel_phantom' in clf.supported_heroes
print('OK')
"
```
Expected: supported 列表含 `angel_phantom`，打印 `OK`。

- [ ] **Step 3: 最终报告**

向用户报告：迁移张数（train/validation/test 分列）、v5 最佳验证准确率、test 验收数据（互混数、accuracy_accepted 对比）、已切换配置、全部测试结果、遗留事项（复核清单待二轮精修、实机验证清单见 coop/TODO.md）。不得声称实机可用——实机首跑清单在 spec §5 与 coop/TODO.md 中，需用户实机验证。
