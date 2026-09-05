# 保卫宠物模式支持 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增保卫宠物模式（`run --mode pet`），与合作模式共享状态机骨架，差异通过模式档案（配置）与识别层分派承载。

**Architecture:** 方案 A（模式档案注入）：`configs/default.yaml` 的 `mode_profiles` 存素材名与流程开关；`CoopTask` 注入 `ModeProfile`，页签/聊天/结算三处按档案分派；`CoopPerception` 通过 locator 别名与 OCR 难度行识别切换素材来源，标志名不变（任务代码零改动）；决策差异经 `resolve_main_c_profile` 应用主C覆盖。状态机、抢合作协调器、棋盘识别、技能选卡、金币门控全部不动。

**Tech Stack:** Python 3.10+、Pydantic v2、typer、OpenCV、rapidocr-onnxruntime（OCR）、pytest。

**Spec:** `docs/superpowers/specs/2026-09-06-pet-defend-mode-design.md`（实现时与 spec 一起读；本计划中的差异逐条落层表在 spec §3）

## Global Constraints

- 不带 `--mode` 时（`run.mode: coop`），合作行为与现状唯一差异 = 进招募后多一次「合作模式」页签点击（用户定案，spec §6）。
- 禁止改动：State/Transition、`_INTERFACE_FLAGS_BY_STATE` 门控语义、CoopGrabCoordinator、棋盘识别、技能 OCR 选卡、金币门控、DebugRecorder。
- OCR 必须限域：ROI 缺失或零面积一律返回空，禁止全图 OCR 进入主循环（spec §4.4）。
- OCR 文字置信度与模板匹配置信度语义不同，阈值不得混用（tasks.yaml `pet_difficulty_recognition.min_confidence` 只用于 OCR 行）。
- 业务坐标只进 `configs/tasks.yaml`；运行参数进 `configs/default.yaml`（AGENTS.md）。
- 日志统一 `get_logger(__name__)`；测试不得真实点击桌面；`mode_profile=None` 保持旧行为（既有测试的兼容语义）。
- 测试命令统一在仓库根目录执行：`python -m pytest tests/test_xxx.py -v`（本仓库用 `.venv`，必要时用 `.venv/Scripts/python.exe -m pytest`）。

---

### Task 1: ModeProfile 配置模型与主C覆盖解析

**Files:**
- Modify: `src/wlxq_bot/config.py`
- Modify: `configs/default.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `ModeProfile` / `EntryProfile` / `DifficultyProfile` / `MatchProfile` / `SettlementProfile`（Pydantic 模型，字段见下）；`DefaultConfig.mode_profiles: dict[str, ModeProfile]`；`RunConfig.mode: str`；模块级函数 `resolve_main_c_profile(profiles: dict[str, MainCProfile], mode: ModeProfile, main_c: str) -> MainCProfile`。后续 Task 2~6 全部消费这些名字。

- [ ] **Step 1: 写失败测试**

在 `tests/test_config.py` 末尾追加：

```python
class TestModeProfile:
    """模式档案模型与主C覆盖解析测试。"""

    def _mode(self, **kw) -> ModeProfile:
        base = dict(
            display_name="保卫宠物",
            mode_chat_hotspot="pet_chat",
            entry={"mode_tab_hotspot": "recruit_pet_tab",
                   "difficulty": {"kind": "ocr", "targets": ["普通", "中等"]}},
            match={"ready_button": "pet_ready_button", "leave_team": "pet_leave_team"},
            settlement={"like_chest": False, "return_button": "pet_return_button"},
            main_c_overrides={"assault": {"skill_selection_cap": 0}},
        )
        base.update(kw)
        return ModeProfile(**base)

    def _default(self, mode: str = "pet") -> DefaultConfig:
        return DefaultConfig(
            run={"mode": mode},
            main_c_profiles={"assault": MainCProfile(display_name="强袭", hero_template_dir="heroes/assault")},
            mode_profiles={
                "coop": ModeProfile(
                    mode_chat_hotspot="coop_chat",
                    match={"ready_button": "ready_button", "leave_team": "coop_leave_team"},
                    settlement={"return_button": "return_button"},
                ),
                "pet": self._mode(),
            },
        )

    def test_mode_profile_defaults(self) -> None:
        profile = self._mode()
        assert profile.entry.difficulty.kind == "ocr"
        assert profile.entry.difficulty.targets == ["普通", "中等"]
        assert profile.settlement.like_chest is False
        assert profile.strategy == "default"

    def test_run_mode_must_exist_in_mode_profiles(self) -> None:
        with pytest.raises(ValidationError):
            self._default(mode="not_a_mode")

    def test_ocr_kind_requires_targets(self) -> None:
        profile = self._mode(entry={"mode_tab_hotspot": None, "difficulty": {"kind": "ocr"}})
        with pytest.raises(ValidationError):
            DefaultConfig(
                run={"mode": "pet"},
                main_c_profiles={"assault": MainCProfile(display_name="强袭", hero_template_dir="h")},
                mode_profiles={"pet": profile},
            )

    def test_main_c_overrides_unknown_hero_rejected(self) -> None:
        profile = self._mode(main_c_overrides={"ghost": {"skill_selection_cap": 0}})
        with pytest.raises(ValidationError):
            DefaultConfig(
                run={"mode": "pet"},
                main_c_profiles={"assault": MainCProfile(display_name="强袭", hero_template_dir="h")},
                mode_profiles={"pet": profile},
            )

    def test_main_c_overrides_unknown_field_rejected(self) -> None:
        profile = self._mode(main_c_overrides={"assault": {"no_such_field": 1}})
        with pytest.raises(ValidationError):
            DefaultConfig(
                run={"mode": "pet"},
                main_c_profiles={"assault": MainCProfile(display_name="强袭", hero_template_dir="h")},
                mode_profiles={"pet": profile},
            )

    def test_resolve_main_c_profile_applies_override(self) -> None:
        cfg = self._default()
        resolved = resolve_main_c_profile(
            cfg.main_c_profiles, cfg.mode_profiles["pet"], "assault"
        )
        assert resolved.skill_selection_cap == 0
        # 未覆盖字段继承原档案
        assert resolved.topup_hero_count == cfg.main_c_profiles["assault"].topup_hero_count

    def test_resolve_main_c_profile_without_override_returns_base(self) -> None:
        cfg = self._default()
        resolved = resolve_main_c_profile(
            cfg.main_c_profiles, cfg.mode_profiles["coop"], "assault"
        )
        assert resolved is cfg.main_c_profiles["assault"]

    def test_repo_default_yaml_defines_both_modes(self) -> None:
        """仓库自带 default.yaml 必须定义 coop/pet 两模式且可加载。"""
        cfg = load_default_config(Path("configs/default.yaml"))
        assert set(cfg.mode_profiles) >= {"coop", "pet"}
        assert cfg.run.mode == "coop"
        assert cfg.mode_profiles["pet"].entry.difficulty.kind == "ocr"
        assert cfg.mode_profiles["pet"].main_c_overrides["assault"]["skill_selection_cap"] == 0
```

并在文件头部 `from wlxq_bot.config import (...)` 导入列表中补上：`DefaultConfig, ModeProfile, resolve_main_c_profile`。

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_config.py::TestModeProfile -v`
Expected: FAIL（`ImportError: cannot import name 'ModeProfile'`）

- [ ] **Step 3: 实现 config.py**

`src/wlxq_bot/config.py`：

1. 头部导入改为 `from typing import Any, Literal`。
2. 在 `MainCProfile` 类之后、`SkillCollectionConfig` 之前插入：

```python
class DifficultyProfile(BaseModel):
    """难度识别方式（模式档案）。"""

    kind: Literal["template", "ocr"] = "template"
    # ocr 模式的目标难度（行文字子串，如「普通」⊂「保卫宠物-普通」）；配置顺序即点击顺序
    targets: list[str] = Field(default_factory=list)


class EntryProfile(BaseModel):
    """招募入口差异（模式档案）。"""

    # 进招募后先点的模式页签 hotspot 名（configs/tasks.yaml）；None = 不点
    mode_tab_hotspot: str | None = None
    difficulty: DifficultyProfile = Field(default_factory=DifficultyProfile)


class MatchProfile(BaseModel):
    """对局入口/等待差异（模式档案）：值都是 configs/tasks.yaml 的 locator 名。"""

    ready_button: str
    leave_team: str


class SettlementProfile(BaseModel):
    """结算差异（模式档案）。"""

    # 结算是否点赞并领取宝箱（保卫宠物为 false，只点返回）
    like_chest: bool = True
    return_button: str


class ModeProfile(BaseModel):
    """组队对局模式档案：素材名与流程开关；素材本体在 configs/tasks.yaml。"""

    display_name: str = ""
    # 模式页面聊天按钮（结算返回后 & 退队非首页兜底共用）
    mode_chat_hotspot: str
    entry: EntryProfile = Field(default_factory=EntryProfile)
    match: MatchProfile
    settlement: SettlementProfile
    # 主C档案覆盖：键为主C标识，值为 MainCProfile 字段子集（如 skill_selection_cap）
    main_c_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    strategy: str = "default"


def resolve_main_c_profile(
    profiles: dict[str, MainCProfile], mode: ModeProfile, main_c: str
) -> MainCProfile:
    """按模式档案应用主C覆盖；无覆盖时原样返回共享档案实例。"""
    base = profiles[main_c]
    overrides = mode.main_c_overrides.get(main_c)
    if not overrides:
        return base
    return MainCProfile.model_validate({**base.model_dump(), **overrides})
```

3. `RunConfig` 增加字段（放在 `default_main_c` 之后）：

```python
    # 组队对局模式：coop（合作模式）| pet（保卫宠物）；CLI --mode 可覆盖
    mode: str = "coop"
```

4. `DefaultConfig` 增加字段（`main_c_profiles` 之后）：

```python
    mode_profiles: dict[str, ModeProfile] = Field(default_factory=dict)
```

5. `DefaultConfig` 增加校验器（`_check_lineup_others_subset` 之后）：

```python
    @model_validator(mode="after")
    def _check_mode_profiles(self) -> DefaultConfig:
        if not self.mode_profiles:
            return self
        if self.run.mode not in self.mode_profiles:
            raise ValueError(
                f"run.mode={self.run.mode!r} 不在 mode_profiles 中: {sorted(self.mode_profiles)}"
            )
        for name, profile in self.mode_profiles.items():
            if profile.entry.difficulty.kind == "ocr" and not profile.entry.difficulty.targets:
                raise ValueError(
                    f"mode_profiles.{name}.entry.difficulty.kind=ocr 需要配置 targets"
                )
            unknown_heroes = [
                key for key in profile.main_c_overrides if key not in self.main_c_profiles
            ]
            if unknown_heroes:
                raise ValueError(
                    f"mode_profiles.{name}.main_c_overrides 引用未定义主C: {unknown_heroes}"
                )
            for main_c, overrides in profile.main_c_overrides.items():
                illegal = sorted(set(overrides) - set(MainCProfile.model_fields))
                if illegal:
                    raise ValueError(
                        f"mode_profiles.{name}.main_c_overrides.{main_c} 含未知字段: {illegal}"
                    )
        return self
```

- [ ] **Step 4: 更新 configs/default.yaml**

`run:` 段 `default_main_c: assault` 之后加：

```yaml
  # 组队对局模式：coop（合作模式）| pet（保卫宠物）；CLI --mode 覆盖本次运行
  mode: coop
```

文件末尾追加（对照 spec §4.1；`main_c_overrides` 覆盖强袭技能上限为 0 = 宠物一直选技能到对局结束，2026-09-06 用户确认）：

```yaml
# 组队对局模式档案（方案 A）：存 configs/tasks.yaml 里的 hotspot/locator 名与
# 流程开关；素材本体在 tasks.yaml。详见 docs/superpowers/specs/2026-09-06-pet-defend-mode-design.md
mode_profiles:
  coop:
    display_name: 合作模式
    mode_chat_hotspot: coop_chat
    entry:
      mode_tab_hotspot: recruit_coop_tab
      difficulty:
        kind: template
    match:
      ready_button: ready_button
      leave_team: coop_leave_team
    settlement:
      like_chest: true
      return_button: return_button
    main_c_overrides: {}
    strategy: default

  pet:
    display_name: 保卫宠物
    mode_chat_hotspot: pet_chat
    entry:
      mode_tab_hotspot: recruit_pet_tab
      difficulty:
        kind: ocr
        targets: [普通, 中等]
    match:
      ready_button: pet_ready_button
      leave_team: pet_leave_team
    settlement:
      like_chest: false
      return_button: pet_return_button
    main_c_overrides:
      assault:
        skill_selection_cap: 0
    strategy: default
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_config.py -v`
Expected: 全部 PASS（含既有 TestRunConfig 等）

- [ ] **Step 6: Commit**

```bash
git add src/wlxq_bot/config.py configs/default.yaml tests/test_config.py
git commit -m "feat(config): 模式档案 ModeProfile 与主C覆盖解析（保卫宠物模式 1/7）"
```

---

### Task 2: pet_difficulty.py OCR 难度行识别

**Files:**
- Modify: `src/wlxq_bot/perception/ocr.py`（TitleReader 增加 `read_with_confidence`）
- Create: `src/wlxq_bot/perception/pet_difficulty.py`
- Test: `tests/test_pet_difficulty.py`

**Interfaces:**
- Consumes: `ocr.TitleReader`（新增 `read_with_confidence(image) -> list[tuple[str, float, tuple[int, int]]]`，返回 (文字, 置信度, 中心点)，坐标相对输入图）。
- Produces: `DifficultyRow(text: str, confidence: float, position: tuple[int, int])`（frozen dataclass，position=文字框中心的客户区像素）；`read_difficulty_rows(frame, roi, reader=None) -> list[DifficultyRow]`；`match_targets(rows: list[DifficultyRow], targets: list[str]) -> list[tuple[str, DifficultyRow]]`。Task 3 的 CoopPerception 消费这三个名字；reader 协议 = 提供 `read_with_confidence` 的对象（测试注入 fake）。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_pet_difficulty.py`：

```python
"""保卫宠物难度行 OCR 识别测试（fake reader，不依赖真实引擎）。"""

from __future__ import annotations

import numpy as np

from wlxq_bot.perception.pet_difficulty import (
    DifficultyRow,
    match_targets,
    read_difficulty_rows,
)


class FakeRowReader:
    """TitleReader 协议的 fake：返回预设行（text, confidence, (cx, cy)）。"""

    def __init__(self, lines: list[tuple[str, float, tuple[int, int]]]) -> None:
        self.lines = lines
        self.images: list[np.ndarray] = []

    def read_with_confidence(self, image: np.ndarray):
        self.images.append(image)
        return self.lines


def _frame(h: int = 1400, w: int = 700) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_read_difficulty_rows_offsets_by_roi() -> None:
    reader = FakeRowReader([("保卫宠物-普通", 0.85, (120, 50))])
    rows = read_difficulty_rows(_frame(), roi=(300, 900, 260, 300), reader=reader)
    assert rows == [DifficultyRow(text="保卫宠物-普通", confidence=0.85, position=(420, 950))]
    # 裁剪图尺寸 = ROI 尺寸
    assert reader.images[0].shape[:2] == (300, 260)


def test_read_difficulty_rows_empty_and_zero_roi() -> None:
    reader = FakeRowReader([("保卫宠物-普通", 0.85, (10, 10))])
    assert read_difficulty_rows(_frame(), None, reader=reader) == []
    assert read_difficulty_rows(_frame(), (10, 10, 0, 100), reader=reader) == []
    assert read_difficulty_rows(_frame(), (10, 10, 100, 0), reader=reader) == []
    assert reader.images == []


def test_read_difficulty_rows_skips_blank_text() -> None:
    reader = FakeRowReader([("  ", 0.9, (10, 10)), ("保卫宠物-中等", 0.7, (20, 20))])
    rows = read_difficulty_rows(_frame(), (0, 0, 100, 100), reader=reader)
    assert [row.text for row in rows] == ["保卫宠物-中等"]


def test_match_targets_substring_and_order() -> None:
    rows = [
        DifficultyRow(text="保卫宠物-困难", confidence=0.8, position=(1, 1)),
        DifficultyRow(text="保卫宠物-普通", confidence=0.9, position=(2, 2)),
        DifficultyRow(text="保卫宠物-中等", confidence=0.7, position=(3, 3)),
    ]
    matched = match_targets(rows, ["普通", "中等"])
    assert [target for target, _ in matched] == ["普通", "中等"]
    assert matched[0][1].position == (2, 2)
    assert matched[1][1].position == (3, 3)


def test_match_targets_skips_missing_and_low_confidence_rows() -> None:
    rows = [
        DifficultyRow(text="保卫宠物-地狱", confidence=0.9, position=(1, 1)),
        DifficultyRow(text="保卫宠物-中等", confidence=0.3, position=(2, 2)),
    ]
    assert match_targets(rows, ["普通", "中等"]) == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_pet_difficulty.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'wlxq_bot.perception.pet_difficulty'`）

- [ ] **Step 3: 实现 ocr.py 的 read_with_confidence**

`src/wlxq_bot/perception/ocr.py` 的 `TitleReader` 类改为：

```python
class TitleReader:
    """横带标题识别器。

    read_with_confidence() 对输入图做一次 OCR，返回 [(文字, 置信度, (中心x, 中心y))]，
    坐标相对输入图左上角。read() 是丢弃置信度的兼容封装。
    """

    def read_with_confidence(self, image: np.ndarray) -> list[tuple[str, float, tuple[int, int]]]:
        engine = _get_engine()
        result, _ = engine(image)
        lines: list[tuple[str, float, tuple[int, int]]] = []
        if not result:
            return lines
        for item in result:
            box = np.array(item[0], dtype=float)
            center = (int(box[:, 0].mean()), int(box[:, 1].mean()))
            text = str(item[1]).strip()
            if text:
                lines.append((text, float(item[2]), center))
        return lines

    def read(self, image: np.ndarray) -> list[tuple[str, tuple[int, int]]]:
        return [(text, center) for text, _, center in self.read_with_confidence(image)]
```

- [ ] **Step 4: 实现 pet_difficulty.py**

创建 `src/wlxq_bot/perception/pet_difficulty.py`：

```python
"""保卫宠物难度行 OCR 识别。

读取难度弹窗列表区 ROI 内的行文字（「保卫宠物-普通」等），返回文字、
置信度与文字框中心（客户区像素）。只负责识别与文字级子串匹配；ROI 解析
归调用方（CoopPerception 的 _resolve_named_roi），坐标换算归 Locator 体系。
OCR 必须限域：ROI 缺失或零面积一律返回空，禁止全图 OCR 进入主循环
（spec 2026-09-06 §4.4）。离线验证：四行难度全部读对、置信度 0.76~0.85
（素材 buttons/bao_wei_chong_wu/nan_du.png，2026-09-05）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wlxq_bot.perception.ocr import TitleReader
from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

_default_reader: TitleReader | None = None


@dataclass(frozen=True)
class DifficultyRow:
    """难度弹窗内一行识别结果；position 为文字框中心（客户区像素）。"""

    text: str
    confidence: float
    position: tuple[int, int]


def _get_default_reader() -> TitleReader:
    global _default_reader
    if _default_reader is None:
        _default_reader = TitleReader()
    return _default_reader


def read_difficulty_rows(
    frame: Any,
    roi: tuple[int, int, int, int] | None,
    reader: Any | None = None,
) -> list[DifficultyRow]:
    """在 ROI 内做一次 OCR，返回行结果；roi 为 (x, y, w, h) 客户区像素。

    reader 需提供 ``read_with_confidence(image)``（TitleReader 兼容协议），
    坐标相对输入图左上角；测试注入 fake，不依赖真实引擎。
    """
    if roi is None:
        return []
    x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    if w <= 0 or h <= 0:
        return []
    crop = frame[y : y + h, x : x + w]
    lines = (reader or _get_default_reader()).read_with_confidence(crop)
    rows = [
        DifficultyRow(text=str(text).strip(), confidence=float(conf), position=(cx + x, cy + y))
        for text, conf, (cx, cy) in lines
        if str(text).strip()
    ]
    logger.debug("难度行识别 ROI=%s 行=%s", (x, y, w, h), [row.text for row in rows])
    return rows


def match_targets(
    rows: list[DifficultyRow], targets: list[str]
) -> list[tuple[str, DifficultyRow]]:
    """按子串匹配目标难度；targets 顺序即优先顺序，每个目标取置信度最高行。

    目标是行文字的子串（「普通」⊂「保卫宠物-普通」）；普通/中等互不为子串，
    不同目标不会命中同一行。低于调用方置信度过滤的行应在传入前剔除
    （CoopPerception 按 pet_difficulty_recognition.min_confidence 过滤）。
    """
    matched: list[tuple[str, DifficultyRow]] = []
    for target in targets:
        hits = [row for row in rows if target in row.text]
        if hits:
            matched.append((target, max(hits, key=lambda row: row.confidence)))
    return matched
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_pet_difficulty.py tests/test_gold_recognition.py tests/test_skill_title_fallback.py -v`
Expected: 全部 PASS（ocr.py 改动不破坏既有消费方）

- [ ] **Step 6: Commit**

```bash
git add src/wlxq_bot/perception/ocr.py src/wlxq_bot/perception/pet_difficulty.py tests/test_pet_difficulty.py
git commit -m "feat(perception): 保卫宠物难度行 OCR 识别（保卫宠物模式 2/7）"
```

---

### Task 3: CoopPerception 模式感知接入

**Files:**
- Modify: `src/wlxq_bot/perception/coop.py`
- Modify: `src/wlxq_bot/models.py`（TasksConfig 增加 `pet_difficulty_recognition`）
- Modify: `configs/tasks.yaml`
- Test: `tests/test_coop_perception.py`

**Interfaces:**
- Consumes: Task 2 的 `read_difficulty_rows` / `DifficultyRow`。
- Produces:
  - `CoopPerception.__init__` 新增 kwargs：`difficulty_kind: str = "template"`、`pet_difficulty_targets: list[str] | None = None`、`locator_aliases: dict[str, str] | None = None`、`difficulty_reader: Any | None = None`。
  - OCR 模式下 `observe()` 行为：`observation_mode="coop_difficulty"` / `"difficulty_dialog"` 时不再匹配 `he_zuo_mo_shi` 与 `cai_hong_N`，改为读 `pet_difficulty_recognition.candidate_roi` 的 OCR 行，写入 `raw_data["pet_difficulty_rows"]`（已按 `min_confidence` 过滤）并以行存在性写 `raw_data["difficulty_dialog_visible"]`（标志名不变 → 任务层零改动）。
  - `locator_aliases`：canonical locator 名 → 实际 locator 名（`ready_button` / `return_button` / `coop_leave_team` 三个会被重映射；标志名 `ready_button_visible` / `return_button_visible` / `leave_team_visible` 不变）。
  - `TasksConfig.pet_difficulty_recognition: dict[str, Any]`。

- [ ] **Step 1: 写失败测试**

在 `tests/test_coop_perception.py` 中追加（沿用该文件已有的 `_make_window_ctx` / `_SingleTemplateVision` / `TemplatePack` 写法；`MatchResult(template, position, confidence)` 三参构造）：

```python
class FakeRowReader:
    """pet_difficulty reader 协议的 fake。"""

    def __init__(self, lines: list[tuple[str, float, tuple[int, int]]]) -> None:
        self.lines = lines
        self.images: list = []

    def read_with_confidence(self, image):
        self.images.append(image)
        return self.lines


class TestPetDifficultyOcr:
    """OCR 难度识别模式（保卫宠物）。"""

    def _tasks_cfg(self) -> TasksConfig:
        return TasksConfig(
            rois={
                "pet_difficulty_list": {
                    "relative_to": "client",
                    "x_ratio": 0.0,
                    "y_ratio": 0.0,
                    "width_ratio": 0.5,
                    "height_ratio": 0.5,
                }
            },
            pet_difficulty_recognition={"candidate_roi": "pet_difficulty_list",
                                        "min_confidence": 0.5},
        )

    def _perception(self, reader: FakeRowReader, targets: list[str], tmp_path):
        from wlxq_bot.assets import TemplatePack

        pack_root = tmp_path / "3000x2000"
        pack_root.mkdir(exist_ok=True)
        return CoopPerception(
            _RecordingVision(),
            TemplatePack(client_size=(923, 1723), root=pack_root),
            self._tasks_cfg(),
            CoopRole.HELPER,
            "assault",
            difficulty_kind="ocr",
            pet_difficulty_targets=targets,
            difficulty_reader=reader,
        )

    def _frame(self) -> np.ndarray:
        return np.zeros((1723, 923, 3), dtype=np.uint8)

    def test_ocr_mode_reads_rows_into_raw_data(self, tmp_path) -> None:
        reader = FakeRowReader([
            ("保卫宠物-普通", 0.85, (120, 50)),
            ("保卫宠物-中等", 0.60, (120, 140)),
        ])
        perception = self._perception(reader, ["普通", "中等"], tmp_path)
        observation = perception.observe(
            _make_window_ctx(), self._frame(), observation_mode="coop_difficulty"
        )
        rows = observation.raw_data["pet_difficulty_rows"]
        assert [row.text for row in rows] == ["保卫宠物-普通", "保卫宠物-中等"]
        assert rows[0].confidence == 0.85
        assert observation.raw_data["difficulty_dialog_visible"] is True
        assert observation.difficulty_candidates == []

    def test_ocr_mode_low_confidence_rows_filtered(self, tmp_path) -> None:
        reader = FakeRowReader([("保卫宠物-中等", 0.30, (120, 140))])
        perception = self._perception(reader, ["中等"], tmp_path)
        observation = perception.observe(
            _make_window_ctx(), self._frame(), observation_mode="coop_difficulty"
        )
        assert observation.raw_data["pet_difficulty_rows"] == []
        assert observation.raw_data["difficulty_dialog_visible"] is False

    def test_ocr_dialog_mode_reports_visibility_without_candidates(self, tmp_path) -> None:
        reader = FakeRowReader([("保卫宠物-普通", 0.85, (120, 50))])
        perception = self._perception(reader, ["普通"], tmp_path)
        observation = perception.observe(
            _make_window_ctx(), self._frame(), observation_mode="difficulty_dialog"
        )
        assert observation.raw_data["difficulty_dialog_visible"] is True
        assert "pet_difficulty_rows" in observation.raw_data


class TestLocatorAliases:
    """locator 别名：canonical 名重映射到模式专属 locator，标志名不变。"""

    def test_ready_button_alias_in_grab_mode(self, tmp_path) -> None:
        from wlxq_bot.assets import TemplatePack

        pack_root = tmp_path / "3000x2000"
        (pack_root / "buttons" / "bao_wei_chong_wu").mkdir(parents=True)
        (pack_root / "buttons" / "bao_wei_chong_wu" / "zhun_bei.png").write_bytes(b"t")
        match = MatchResult("zhun_bei.png", (400, 1300), 0.9)
        vision = _SingleTemplateVision("zhun_bei.png", match)
        tasks_cfg = TasksConfig(
            locators={
                "pet_ready_button": {
                    "strategy": "template",
                    "template": "buttons/bao_wei_chong_wu/zhun_bei.png",
                    "threshold": 0.82,
                }
            }
        )
        perception = CoopPerception(
            vision,
            TemplatePack(client_size=(923, 1723), root=pack_root),
            tasks_cfg,
            CoopRole.HELPER,
            "assault",
            locator_aliases={"ready_button": "pet_ready_button"},
        )
        observation = perception.observe(
            _make_window_ctx(), None, hint_state=State.FIND_COOP, observation_mode="coop_grab"
        )
        assert observation.flag("ready_button_visible") is True
        assert observation.raw_data["ready_button_match"] == match

    def test_leave_team_alias_in_opening_state(self, tmp_path) -> None:
        from wlxq_bot.assets import TemplatePack

        pack_root = tmp_path / "3000x2000"
        (pack_root / "buttons" / "bao_wei_chong_wu").mkdir(parents=True)
        (pack_root / "buttons" / "bao_wei_chong_wu" / "tui_dui.png").write_bytes(b"t")
        match = MatchResult("tui_dui.png", (741, 1488), 0.9)
        vision = _SingleTemplateVision("tui_dui.png", match)
        tasks_cfg = TasksConfig(
            locators={
                "pet_leave_team": {
                    "strategy": "template",
                    "template": "buttons/bao_wei_chong_wu/tui_dui.png",
                    "threshold": 0.82,
                }
            }
        )
        perception = CoopPerception(
            vision,
            TemplatePack(client_size=(923, 1723), root=pack_root),
            tasks_cfg,
            CoopRole.HELPER,
            "assault",
            locator_aliases={"coop_leave_team": "pet_leave_team"},
        )
        observation = perception.observe(
            _make_window_ctx(), None, hint_state=State.SELECT_OPENING_SKILLS
        )
        assert observation.flag("leave_team_visible") is True
        assert observation.raw_data["leave_team_match"] == match
```

说明：`_RecordingVision` 未命中模板返回 None（见该类现有实现），隐式断言 OCR 模式下不会调用任何模板匹配；`_make_window_ctx` 为该文件既有 helper。若文件头部未导入 `State`（aliases 用例使用），补齐导入。

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_coop_perception.py::TestPetDifficultyOcr tests/test_coop_perception.py::TestLocatorAliases -v`
Expected: FAIL（`TypeError: CoopPerception.__init__() got an unexpected keyword argument 'difficulty_kind'`）

- [ ] **Step 3: 实现 perception/coop.py**

1. 文件头部导入：`from wlxq_bot.perception.pet_difficulty import read_difficulty_rows`。
2. `__init__`（签名在 `gold_read_interval: float = 1.0` 之后追加 kwargs）：

```python
        difficulty_kind: str = "template",
        pet_difficulty_targets: list[str] | None = None,
        locator_aliases: dict[str, str] | None = None,
        difficulty_reader: Any | None = None,
```

构造体末尾追加：

```python
        # 难度识别方式（模式档案）：template=合作文字模板；ocr=保卫宠物行 OCR
        self._difficulty_kind = difficulty_kind
        self._pet_difficulty_targets = list(pet_difficulty_targets or [])
        # canonical locator 名 → 模式专属 locator 名（ready_button/return_button/
        # coop_leave_team 三个可被重映射；标志名不变，任务层零改动）
        self._locator_aliases = dict(locator_aliases or {})
        self._difficulty_reader = difficulty_reader
```

3. `_detect_interface_flags` 中 `wanted_items` 构造改为别名重映射：

```python
        wanted_items = [
            (self._locator_aliases.get(locator_name, locator_name), flag_name, match_name)
            for locator_name, flag_name, match_name in _INTERFACE_FLAG_LOCATORS
            if locator_name in wanted
        ]
```

退队块（原 `leave_locator = locators.get("coop_leave_team")`）改为：

```python
            leave_locator = locators.get(
                self._locator_aliases.get("coop_leave_team", "coop_leave_team")
            )
```

4. `observe()` 中难度分支改为：

```python
        difficulty_candidates = []
        if observation_mode == "coop_difficulty":
            if self._difficulty_kind == "ocr":
                # 保卫宠物：OCR 读难度行；弹窗开关以行存在性判定（标志名不变）
                self._detect_pet_difficulty_rows(ctx, frame, raw_data)
                raw_data["difficulty_dialog_visible"] = bool(raw_data["pet_difficulty_rows"])
            else:
                self._detect_difficulty_dialog_flag(ctx, frame, raw_data, matches)
                difficulty_candidates = self._detect_difficulty_candidates(ctx, frame, matches)
        elif observation_mode == "difficulty_dialog":
            # 关闭/确认关闭难度弹窗：只判定弹窗开关标识，跳过难度候选匹配
            if self._difficulty_kind == "ocr":
                self._detect_pet_difficulty_rows(ctx, frame, raw_data)
                raw_data["difficulty_dialog_visible"] = bool(raw_data["pet_difficulty_rows"])
            else:
                self._detect_difficulty_dialog_flag(ctx, frame, raw_data, matches)
```

5. 新增方法（放在 `_detect_difficulty_dialog_flag` 之后）：

```python
    def _detect_pet_difficulty_rows(
        self,
        ctx: WindowContext,
        frame: Any,
        raw_data: dict[str, Any],
    ) -> None:
        """OCR 读保卫宠物难度行，写入 pet_difficulty_rows（已按 min_confidence 过滤）。

        ROI 来自 pet_difficulty_recognition.candidate_roi；缺失或零面积时
        read_difficulty_rows 返回空（OCR 必须限域，启动检查会另行为缺失报错）。
        """
        cfg = self._cfg.pet_difficulty_recognition
        roi_name = str(cfg.get("candidate_roi", "pet_difficulty_list"))
        roi = self._resolve_named_roi(roi_name, ctx)
        min_confidence = float(cfg.get("min_confidence", 0.5))
        rows = [
            row
            for row in read_difficulty_rows(frame, roi, reader=self._difficulty_reader)
            if row.confidence >= min_confidence
        ]
        raw_data["pet_difficulty_rows"] = rows
        logger.debug(
            "frame=%d 宠物难度行=%s", ctx.frame_id, [row.text for row in rows]
        )
```

- [ ] **Step 4: models.py 与 tasks.yaml**

`src/wlxq_bot/models.py` 的 `TasksConfig`：

1. 字段区（`difficulty_recognition` 之后）加：

```python
    # 保卫宠物难度弹窗 OCR 识别配置（candidate_roi + min_confidence）
    pet_difficulty_recognition: dict[str, Any] = Field(default_factory=dict)
```

2. `_none_to_empty_dict` 的键元组中加入 `"pet_difficulty_recognition"`。

`configs/tasks.yaml` 的 `locators:` 段末尾（`double_reward_cancel` 之后）追加：

```yaml
  # 【保卫宠物-准备】按钮（抢到保卫宠物后的准备确认；与合作不是同一张图）
  # 阈值 0.82 起步待实机回调；ROI 待标定，标定前退化全图识别
  pet_ready_button:
    strategy: template
    template: buttons/bao_wei_chong_wu/zhun_bei.png
    threshold: 0.82
    action_point:
      source: match_center
  # 【保卫宠物-返回】按钮（保卫宠物结算页；不点赞不领宝箱，点返回即完成本局）
  pet_return_button:
    strategy: template
    template: buttons/bao_wei_chong_wu/fan_hui.png
    threshold: 0.82
    action_point:
      source: match_center
  # 【保卫宠物-退队】按钮（保卫宠物组队大厅；位置与合作的明显不同，
  # 实测点见 hotspots.pet_leave_team；ROI 待标定，标定前退化全图识别）
  pet_leave_team:
    strategy: template
    template: buttons/bao_wei_chong_wu/tui_dui.png
    threshold: 0.82
    action_point:
      source: match_center
```

`rois:` 段 `pet_difficulty_list` 之后追加：

```yaml
  # 保卫宠物难度弹窗 OCR 配置；min_confidence 是 OCR 文字置信度（离线验证
  # 四行 0.76~0.85），语义与模板匹配置信度不同，不得混用
pet_difficulty_recognition:
  candidate_roi: pet_difficulty_list
  min_confidence: 0.5
```

注意：`pet_difficulty_recognition` 是顶层键（与 `difficulty_recognition` 平级），不是 `rois` 的子项——粘贴时不要放进 `rois:` 段内。

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_coop_perception.py tests/test_pet_difficulty.py tests/test_models.py -v`
Expected: 全部 PASS（既有感知测试全部保持绿色 = template 模式行为未变）

- [ ] **Step 6: Commit**

```bash
git add src/wlxq_bot/perception/coop.py src/wlxq_bot/models.py configs/tasks.yaml tests/test_coop_perception.py
git commit -m "feat(perception): 模式感知接入——locator 别名与保卫宠物难度 OCR（保卫宠物模式 3/7）"
```

---

### Task 4: CoopTask 模式注入（页签步骤 + 模式聊天 + 结算分派）

**Files:**
- Modify: `src/wlxq_bot/tasks/coop.py`
- Modify: `src/wlxq_bot/tasks/strategy.py`
- Test: `tests/test_coop_task.py`

**Interfaces:**
- Consumes: Task 1 的 `ModeProfile`。
- Produces: `CoopTask.__init__` 新增 kwarg `mode_profile: ModeProfile | None = None`（None = 旧行为：无页签步骤、聊天硬编码 `coop_chat`、结算含点赞宝箱——既有测试语义不变）。`tasks/strategy.py` 新增 `build_strategy(mode_profile: ModeProfile | None, main_c: str, run_config: RunConfig, profile: MainCProfile | None) -> CultivationStrategy`（`strategy` 字段只允许 `"default"`；未知名抛 `ValueError`——阶段化策略是将来扩展点，spec §7）。新行为标签：`open_mode_tab`（页签点击）、`like_skipped` / `claim_skipped`（结算跳过）。`_RecruitStep.SELECT_MODE_TAB` 新步骤；`open_recruit` 与 `open_coop_chat` 验证后都先进入它（`_after_tab_step` 记录去向：`OPEN_DIFFICULTY_DIALOG` / `OPEN_REFRESH_DIFFICULTY_DIALOG`）。

- [ ] **Step 1: 写失败测试**

`tests/test_coop_task.py`：

1. 头部导入补 `from wlxq_bot.config import ModeProfile`（并入现有 config 导入）；`make_task` 签名追加 `mode_profile: ModeProfile | None = None` 并透传给 `CoopTask(..., mode_profile=mode_profile)`。
2. 策略工厂测试（追加到测试文件难度测试类之后，或并入新测试类）：

```python
    def test_build_strategy_default_and_unknown(self):
        from wlxq_bot.tasks.strategy import CultivationStrategy, build_strategy

        profile = make_mode_profile()
        strategy = build_strategy(profile, "assault", RunConfig(), None)
        assert isinstance(strategy, CultivationStrategy)
        bad = profile.model_copy(update={"strategy": "phased_assault"})
        with pytest.raises(ValueError):
            build_strategy(bad, "assault", RunConfig(), None)
```

3. 模块级 helper（放在 `make_hero` 之后）：

```python
def make_mode_profile(
    mode_tab_hotspot: str | None = "recruit_coop_tab",
    mode_chat_hotspot: str = "coop_chat",
    difficulty_kind: str = "template",
    difficulty_targets: list[str] | None = None,
    like_chest: bool = True,
    ready_button: str = "ready_button",
    leave_team: str = "coop_leave_team",
    return_button: str = "return_button",
) -> ModeProfile:
    return ModeProfile(
        display_name="测试模式",
        mode_chat_hotspot=mode_chat_hotspot,
        entry={
            "mode_tab_hotspot": mode_tab_hotspot,
            "difficulty": {"kind": difficulty_kind, "targets": difficulty_targets or []},
        },
        match={"ready_button": ready_button, "leave_team": leave_team},
        settlement={"like_chest": like_chest, "return_button": return_button},
    )
```

4. 新测试类（追加到文件末尾或难度测试类之后）：

```python
class TestModeProfileDispatch:
    """模式档案注入：页签步骤、模式聊天入口、结算分派。"""

    def _spots_with_tabs(self) -> dict[str, Hotspot]:
        spots = {
            "home_chat": Hotspot(x_ratio=0.1, y_ratio=0.2),
            "open_recruit": Hotspot(x_ratio=0.2, y_ratio=0.3),
            "recruit_coop_tab": Hotspot(x_ratio=0.25, y_ratio=0.35),
            "recruit_pet_tab": Hotspot(x_ratio=0.4, y_ratio=0.35),
            "open_difficulty_dialog": Hotspot(x_ratio=0.3, y_ratio=0.4),
            "difficulty_scroll_start": Hotspot(x_ratio=0.5, y_ratio=0.3),
            "difficulty_scroll_end": Hotspot(x_ratio=0.5, y_ratio=0.7),
            "close_difficulty_dialog": Hotspot(x_ratio=0.4, y_ratio=0.5),
            "coop_chat": Hotspot(x_ratio=0.5, y_ratio=0.6),
            "pet_chat": Hotspot(x_ratio=0.55, y_ratio=0.62),
            "add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96),
            "join_coop": Hotspot(x_ratio=0.6, y_ratio=0.75),
            "select_skill": Hotspot(x_ratio=0.2, y_ratio=0.9),
            "like": Hotspot(x_ratio=0.3, y_ratio=0.8),
            "claim_chest": Hotspot(x_ratio=0.5, y_ratio=0.8),
        }
        return spots

    def test_coop_mode_clicks_mode_tab_before_difficulty_dialog(self):
        task = make_task(
            hotspots=self._spots_with_tabs(),
            mode_profile=make_mode_profile(),  # 合作档案：点 recruit_coop_tab
        )
        task.ctx.current_state = State.FIND_COOP
        tags = []
        for observation in [
            home_observation(frame_id=1),
            Observation(frame_id=2),   # open_recruit
            Observation(frame_id=3),   # open_mode_tab（新增步骤）
            Observation(frame_id=4),   # open_difficulty_dialog
            Observation(frame_id=5),   # settle
            difficulty_observation(2, 1, frame_id=6),
            difficulty_observation(2, 1, frame_id=7),
            difficulty_observation(2, 1, frame_id=8),
        ]:
            decision = task.decide_action(observation, make_window_ctx())
            assert decision is not None
            action, to_state = decision
            tags.append(action.tag)
            task.on_action_verified(action, to_state)
        assert tags == [
            "open_home_chat",
            "open_recruit",
            "open_mode_tab",
            "open_difficulty_dialog",
            "difficulty_open_settle",
            "difficulty_open_hit",
            "difficulty_open_hit",
            "difficulty_open_confirmed",
        ]

    def test_mode_tab_hotspot_missing_stops_conservatively(self):
        # 档案要求点页签但 tasks.yaml 缺该 hotspot：无可用动作 → decide 返回 None
        # （调度层按保守停止计数，绝不跳过页签直接抢）
        task = make_task(mode_profile=make_mode_profile())  # make_task 默认 spots 无 recruit_coop_tab
        task.ctx.current_state = State.FIND_COOP
        d1 = task.decide_action(home_observation(frame_id=1), make_window_ctx())
        assert d1 is not None and d1[0].tag == "open_home_chat"
        task.on_action_verified(*d1)
        d2 = task.decide_action(Observation(frame_id=2), make_window_ctx())
        assert d2 is not None and d2[0].tag == "open_recruit"
        task.on_action_verified(*d2)
        assert task._recruit_step == _RecruitStep.SELECT_MODE_TAB
        assert task.decide_action(Observation(frame_id=3), make_window_ctx()) is None

    def test_next_round_refresh_clicks_mode_tab_after_chat(self):
        task = make_task(
            max_rounds=2,
            hotspots=self._spots_with_tabs(),
            mode_profile=make_mode_profile(),
        )
        task.ctx.current_state = State.CHECK_ROUND_LIMIT
        match = MatchResult("buttons/fan_hui.png", (500, 1000), 0.9)
        before = Observation(
            frame_id=1,
            raw_data={"return_button_visible": True, "return_button_match": match},
        )
        decision = task.decide_action(before, make_window_ctx())
        assert decision is not None
        action, to_state = decision
        assert to_state == State.FIND_COOP
        task.on_action_verified(action, to_state)
        task.ctx.round_count = 1
        task.ctx.current_state = to_state

        chat = task.decide_action(Observation(frame_id=2), make_window_ctx())
        assert chat is not None and chat[0].tag == "open_coop_chat"
        task.on_action_verified(*chat)
        tab = task.decide_action(Observation(frame_id=3), make_window_ctx())
        assert tab is not None and tab[0].tag == "open_mode_tab"
        task.on_action_verified(*tab)
        dialog = task.decide_action(Observation(frame_id=4), make_window_ctx())
        assert dialog is not None and dialog[0].tag == "open_refresh_difficulty_dialog"

    def test_pet_mode_uses_pet_chat_hotspot(self):
        task = make_task(
            hotspots=self._spots_with_tabs(),
            mode_profile=make_mode_profile(
                mode_tab_hotspot="recruit_pet_tab",
                mode_chat_hotspot="pet_chat",
            ),
        )
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.OPEN_COOP_CHAT
        decision = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert decision is not None
        action, _ = decision
        assert action.tag == "open_coop_chat"
        assert action.target == (int(0.55 * 923), int(0.62 * 1723))  # pet_chat 中心

    def test_pet_settlement_skips_like_and_chest(self):
        task = make_task(
            hotspots=self._spots_with_tabs(),
            mode_profile=make_mode_profile(like_chest=False),
        )
        task.ctx.current_state = State.HANDLE_RESULT
        like = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert like is not None
        assert like[0].tag == "like_skipped"
        assert like[1] == State.CLAIM_REWARD
        task.ctx.current_state = State.CLAIM_REWARD
        claim = task.decide_action(Observation(frame_id=2), make_window_ctx())
        assert claim is not None
        assert claim[0].tag == "claim_skipped"
        assert claim[1] == State.CHECK_ROUND_LIMIT

    def test_coop_mode_keeps_like_and_chest(self):
        task = make_task(
            hotspots=self._spots_with_tabs(),
            mode_profile=make_mode_profile(like_chest=True),
        )
        task.ctx.current_state = State.HANDLE_RESULT
        like = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert like is not None and like[0].tag == "like_result"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_coop_task.py::TestModeProfileDispatch -v`
Expected: FAIL（`TypeError: ... unexpected keyword argument 'mode_profile'`）

- [ ] **Step 3: 实现 tasks/coop.py**

1. 导入：`from wlxq_bot.config import ..., ModeProfile`（并入现有 config 导入行）。
2. `_RecruitStep` 枚举 `OPEN_RECRUIT` 之后加：

```python
    SELECT_MODE_TAB = "select_mode_tab"
```

3. `tasks/strategy.py` 末尾新增策略工厂（`RunConfig` 已在该模块导入，`ModeProfile` 需并入 config 导入行）：

```python
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
```

4. `CoopTask.__init__` 签名 kwargs（`skip_difficulty_selection: bool = False` 之后）加 `mode_profile: ModeProfile | None = None,`；`self._strategy = CultivationStrategy(...)`（约 100 行）改为：

```python
        # 主C培养策略（合成/门禁/数量回补）：纯决策对象，状态机只负责调度；
        # 模式档案经 Runner 的 resolve_main_c_profile 应用覆盖后传入 profile
        self._strategy = build_strategy(mode_profile, ctx.main_c, run_config, main_c_profile)
```

（`from wlxq_bot.tasks.strategy import CultivationStrategy, build_strategy` 同步导入行。）构造体 `self._difficulty_done_session = False` 之后加：

```python
        # 模式档案（方案 A）：None = 旧行为（无页签步骤、coop_chat 硬编码、
        # 结算含点赞宝箱）；正式流程由 Runner 按模式注入
        self._mode = mode_profile
        # 页签点击后的去向：首局路径进打开难度弹窗，下一局刷新路径进打开刷新弹窗
        self._after_tab_step = _RecruitStep.OPEN_DIFFICULTY_DIALOG
```

5. `_action_recruit_entry`：

`OPEN_RECRUIT` 分支（原样保留点击）之后、`OPEN_DIFFICULTY_DIALOG` 分支之前插入：

```python
        if self._recruit_step == _RecruitStep.SELECT_MODE_TAB:
            tab_name = self._mode.entry.mode_tab_hotspot if self._mode else None
            if not tab_name:
                # 未配置页签：直接进入页签后的目标步骤（旧行为）
                self._recruit_step = self._after_tab_step
                return self._action_recruit_entry(observation, window_ctx)
            return self._hotspot_click(
                tab_name,
                "open_mode_tab",
                "进入招募：点击模式页签",
                State.FIND_COOP,
                window_ctx,
            )
```

`OPEN_COOP_CHAT` 分支的 hotspot 名改为模式档案解析：

```python
        if self._recruit_step == _RecruitStep.OPEN_COOP_CHAT:
            chat_hotspot = self._mode.mode_chat_hotspot if self._mode else "coop_chat"
            return self._hotspot_click(
                chat_hotspot,
                "open_coop_chat",
                "下一局进入招募：点击模式页面聊天",
                State.FIND_COOP,
                window_ctx,
            )
```

6. `on_action_verified`：

`"open_recruit"` 分支改为：

```python
        elif tag == "open_recruit":
            self._after_tab_step = _RecruitStep.OPEN_DIFFICULTY_DIALOG
            self._recruit_step = _RecruitStep.SELECT_MODE_TAB
```

`"open_coop_chat"` 分支改为：

```python
        elif tag == "open_coop_chat":
            self._after_tab_step = _RecruitStep.OPEN_REFRESH_DIFFICULTY_DIALOG
            self._recruit_step = _RecruitStep.SELECT_MODE_TAB
```

`"open_mode_tab"` 新增分支（放在 `"open_difficulty_dialog"` 分支旁）：

```python
        elif tag == "open_mode_tab":
            self._recruit_step = self._after_tab_step
```

7. `decide_action` 结算分派：`HANDLE_RESULT` / `CLAIM_REWARD` 两分支开头插入跳过逻辑，并加 helper（放在 `_hotspot_click` 附近）：

```python
    def _like_chest_enabled(self) -> bool:
        """结算是否执行点赞与领取宝箱（保卫宠物为 False，只点返回）。"""
        return self._mode is None or self._mode.settlement.like_chest
```

```python
        if state == State.HANDLE_RESULT:
            if not self._like_chest_enabled():
                return (
                    Action(
                        kind="wait",
                        duration=0.3,
                        verification="immediate",
                        tag="like_skipped",
                        reason="保卫宠物结算：跳过点赞",
                    ),
                    State.CLAIM_REWARD,
                )
            return self._hotspot_click(
```

```python
        if state == State.CLAIM_REWARD:
            if not self._like_chest_enabled():
                return (
                    Action(
                        kind="wait",
                        duration=0.3,
                        verification="immediate",
                        tag="claim_skipped",
                        reason="保卫宠物结算：跳过领宝箱",
                    ),
                    State.CHECK_ROUND_LIMIT,
                )
            # 实机确认：宝箱动画未播完时点返回无响应，领取后稳定等待再进入返回判定
            return self._hotspot_click(
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_coop_task.py -v`
Expected: 全部 PASS——既有测试（无 mode_profile）序列不变；`test_missing_entry_hotspot_stops_without_skipping_to_join` 等旧行为用例保持绿色

- [ ] **Step 5: Commit**

```bash
git add src/wlxq_bot/tasks/coop.py tests/test_coop_task.py
git commit -m "feat(task): CoopTask 模式注入——页签步骤/模式聊天/结算分派（保卫宠物模式 4/7）"
```

---

### Task 5: CoopTask 宠物难度选择分支（OCR）

**Files:**
- Modify: `src/wlxq_bot/tasks/coop.py`
- Test: `tests/test_coop_task.py`

**Interfaces:**
- Consumes: Task 4 的 `self._mode`（`entry.difficulty.kind == "ocr"` 分派）、`Observation.raw_data["pet_difficulty_rows"]`（Task 3 产出，元素为 `DifficultyRow`）。
- Produces: `CoopTask.__init__` 新增 kwargs `pet_difficulty_targets: list[str] | None = None`、`pet_difficulty_min_confidence: float = 0.5`；动作标签 `select_pet_difficulty:<目标名>`、`pet_difficulty_miss`、`pet_difficulty_selection_complete`；行持续不可见达 `difficulty_open_fail_misses` 次抛 `RuntimeError` 保守停止。

- [ ] **Step 1: 写失败测试**

`tests/test_coop_task.py` 的 `TestModeProfileDispatch` 类中追加：

```python
    def _pet_task(
        self,
        targets: list[str] | None = None,
        difficulty_selection_only: bool = False,
    ) -> CoopTask:
        return make_task(
            hotspots=self._spots_with_tabs(),
            difficulty_selection_only=difficulty_selection_only,
            mode_profile=make_mode_profile(
                mode_tab_hotspot="recruit_pet_tab",
                mode_chat_hotspot="pet_chat",
                difficulty_kind="ocr",
                difficulty_targets=targets if targets is not None else ["普通", "中等"],
                like_chest=False,
                ready_button="pet_ready_button",
                leave_team="pet_leave_team",
                return_button="pet_return_button",
            ),
            pet_difficulty_targets=targets if targets is not None else ["普通", "中等"],
        )

    @staticmethod
    def _pet_observation(
        rows: list[tuple[str, float]], frame_id: int
    ) -> Observation:
        from wlxq_bot.perception.pet_difficulty import DifficultyRow

        return Observation(
            frame_id=frame_id,
            raw_data={
                "pet_difficulty_rows": [
                    DifficultyRow(text=text, confidence=conf, position=(100 * i, 200 * i))
                    for i, (text, conf) in enumerate(rows, start=1)
                ],
                "difficulty_dialog_visible": True,
            },
        )

    def test_pet_difficulty_clicks_targets_in_configured_order(self):
        task = self._pet_task(["普通", "中等"])
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.SELECT_DIFFICULTIES
        clicks = []
        for i, rows in enumerate(
            [
                [("保卫宠物-普通", 0.85), ("保卫宠物-中等", 0.8)],
                [("保卫宠物-中等", 0.8)],
            ],
            start=1,
        ):
            decision = task.decide_action(
                self._pet_observation(rows, frame_id=i), make_window_ctx()
            )
            assert decision is not None
            action, to_state = decision
            assert to_state == State.FIND_COOP
            clicks.append((action.tag, action.target))
            task.on_action_verified(action, to_state)
        assert clicks == [
            ("select_pet_difficulty:普通", (100, 200)),
            ("select_pet_difficulty:中等", (200, 400)),
        ]
        # 选完 → 进入关闭弹窗步骤
        assert task._recruit_step == _RecruitStep.CLOSE_DIFFICULTY_DIALOG

    def test_pet_difficulty_prefers_higher_confidence_row(self):
        task = self._pet_task(["普通"])
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.SELECT_DIFFICULTIES
        decision = task.decide_action(
            self._pet_observation(
                [("保卫宠物-普通", 0.7), ("保卫宠物-普通", 0.95)], frame_id=1
            ),
            make_window_ctx(),
        )
        assert decision is not None
        assert decision[0].target == (200, 400)  # 置信度 0.95 的行

    def test_pet_difficulty_miss_budget_stops_conservatively(self):
        task = self._pet_task(["普通"])
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.SELECT_DIFFICULTIES
        budget = task._run_config.difficulty_open_fail_misses
        with pytest.raises(RuntimeError, match="难度行持续不可见"):
            for i in range(budget + 1):
                decision = task.decide_action(
                    self._pet_observation([], frame_id=i + 1), make_window_ctx()
                )
                assert decision is not None
                assert decision[0].tag == "pet_difficulty_miss"
                task.on_action_verified(*decision)

    def test_pet_difficulty_selection_only_completes(self):
        task = self._pet_task(["普通"], difficulty_selection_only=True)
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.SELECT_DIFFICULTIES
        decision = task.decide_action(
            self._pet_observation([("保卫宠物-普通", 0.85)], frame_id=1),
            make_window_ctx(),
        )
        assert decision is not None
        task.on_action_verified(*decision)
        final = task.decide_action(
            self._pet_observation([("保卫宠物-普通", 0.85)], frame_id=2),
            make_window_ctx(),
        )
        assert final is not None
        assert final[0].tag == "pet_difficulty_selection_complete"
        assert final[1] == State.COMPLETED

    def test_skip_difficulty_selection_applies_to_pet(self):
        # 已选过难度（_difficulty_done_session）→ 确认打开后直接关闭，不进勾选
        task = self._pet_task(["普通"])
        task._difficulty_done_session = True
        task.ctx.current_state = State.FIND_COOP
        task._recruit_step = _RecruitStep.SELECT_DIFFICULTIES
        task._difficulty_open_is_refresh = False
        # 模拟 difficulty_open_confirmed 的既有分流逻辑（在 on_action_verified 中）
        task._recruit_step = _RecruitStep.CONFIRM_DIFFICULTY_OPEN
        task.on_action_verified(
            Action(kind="wait", duration=0.0, verification="immediate", tag="difficulty_open_confirmed"),
            State.FIND_COOP,
        )
        assert task._recruit_step == _RecruitStep.CLOSE_DIFFICULTY_DIALOG
```

同时更新 `make_task`：签名追加 `pet_difficulty_targets: list[str] | None = None`，透传给 `CoopTask`。

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_coop_task.py::TestModeProfileDispatch -v -k pet`
Expected: FAIL（`TypeError: ... unexpected keyword argument 'pet_difficulty_targets'`）

- [ ] **Step 3: 实现 tasks/coop.py**

1. `__init__` kwargs（`mode_profile` 之后）加：

```python
        pet_difficulty_targets: list[str] | None = None,
        pet_difficulty_min_confidence: float = 0.5,
```

构造体（`self._after_tab_step` 之后）加：

```python
        # 保卫宠物难度选择（OCR 开环）：待选目标按配置顺序点击；行文字子串匹配
        self._remaining_pet_targets = list(pet_difficulty_targets or [])
        self._selected_pet_targets: list[str] = []
        self._pet_difficulty_misses = 0
        self._pet_min_confidence = pet_difficulty_min_confidence
```

2. `_action_recruit_entry` 的 `SELECT_DIFFICULTIES` 分支改为：

```python
        if self._recruit_step == _RecruitStep.SELECT_DIFFICULTIES:
            if self._mode is not None and self._mode.entry.difficulty.kind == "ocr":
                return self._action_select_pet_difficulties(observation)
            return self._action_select_difficulties(observation, window_ctx)
```

（原方法体不变，仅加派发头。）

3. 新增方法（放在 `_action_select_difficulties` 之后）：

```python
    def _action_select_pet_difficulties(
        self, observation: Observation
    ) -> tuple[Action, State] | None:
        """保卫宠物难度勾选（OCR 开环，2026-09-06 定案）。

        行文字来自 CoopPerception 的 pet_difficulty_rows（已按
        pet_difficulty_recognition.min_confidence 过滤）；按配置顺序对每个
        待选目标做行文字子串匹配，命中即点击行文字中心，点击发送即计为已选
        （同合作难度的开环哲学，不识别勾选框）。所有目标不可见时只等待计数，
        连续 difficulty_open_fail_misses 次仍不可见则保守停止——OCR 模式无
        滚动，绝不盲点。
        """
        if not self._remaining_pet_targets:
            self._log_pet_difficulty_summary()
            if self._difficulty_selection_only:
                return (
                    Action(
                        kind="wait",
                        duration=0.0,
                        verification="immediate",
                        tag="pet_difficulty_selection_complete",
                        reason="保卫宠物目标难度已全部点击",
                    ),
                    State.COMPLETED,
                )
            return None

        rows = observation.raw_data.get("pet_difficulty_rows") or []
        for target in self._remaining_pet_targets:
            hits = [row for row in rows if target in row.text]
            if not hits:
                continue
            best = max(hits, key=lambda row: row.confidence)
            if best.confidence < self._pet_min_confidence:
                continue
            logger.info(
                "frame=%d 保卫宠物难度 %s 命中行 %r 置信度 %.2f，点击文字中心",
                observation.frame_id,
                target,
                best.text,
                best.confidence,
            )
            return (
                Action(
                    kind="click",
                    target=best.position,
                    duration=0.08,
                    verification="immediate",
                    tag=f"select_pet_difficulty:{target}",
                    reason=f"选择保卫宠物难度 {target}",
                ),
                State.FIND_COOP,
            )

        if self._pet_difficulty_misses >= self._run_config.difficulty_open_fail_misses:
            raise RuntimeError(
                f"保卫宠物难度行持续不可见（{self._pet_difficulty_misses} 次），保守停止；"
                "请确认难度弹窗已打开且 pet_difficulty_list ROI 标定正确"
            )
        return (
            Action(
                kind="wait",
                duration=0.3,
                verification="immediate",
                tag="pet_difficulty_miss",
                reason="保卫宠物难度行不可见，等待重试",
            ),
            State.FIND_COOP,
        )

    def _log_pet_difficulty_summary(self) -> None:
        """保卫宠物难度选择汇总（仅输出一次）。"""
        if self._difficulty_summary_logged:
            return
        self._difficulty_summary_logged = True
        logger.info(
            "保卫宠物难度选择汇总：选中 %s",
            self._selected_pet_targets,
        )
```

4. `on_action_verified` 新增分支（`select_difficulty:` 分支旁）：

```python
        elif tag.startswith("select_pet_difficulty:"):
            target = tag.partition(":")[2]
            if target in self._remaining_pet_targets:
                self._remaining_pet_targets.remove(target)
            self._selected_pet_targets.append(target)
            self._pet_difficulty_misses = 0
            if not self._remaining_pet_targets:
                self._log_pet_difficulty_summary()
                if not self._difficulty_selection_only:
                    self._recruit_step = _RecruitStep.CLOSE_DIFFICULTY_DIALOG
                    self._difficulty_close_attempts = 0
                    self._difficulty_close_streak = 0
        elif tag == "pet_difficulty_miss":
            self._pet_difficulty_misses += 1
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_coop_task.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/wlxq_bot/tasks/coop.py tests/test_coop_task.py
git commit -m "feat(task): 保卫宠物难度 OCR 勾选分支（保卫宠物模式 5/7）"
```

---

### Task 6: Runner 与 CLI 接线、启动检查

**Files:**
- Modify: `src/wlxq_bot/runner.py`
- Modify: `src/wlxq_bot/cli.py`
- Test: `tests/test_runner.py`、`tests/test_run_cli.py`

**Interfaces:**
- Consumes: Task 1 `ModeProfile` / `resolve_main_c_profile`；Task 3 感知 kwargs；Task 4/5 任务 kwargs。
- Produces:
  - `Runner.run(..., mode: str | None = None)`、`Runner.select_difficulty(..., mode: str | None = None)`（追加在现有签名末尾，None = 用 `run.mode`）。
  - `Runner._resolve_mode_profile(mode: str | None) -> tuple[str, ModeProfile]`。
  - `Runner._check_mode_assets(mode_profile: ModeProfile, pack: TemplatePack, start_state: State) -> None`（缺失 hotspot/locator/模板/阈值、OCR ROI 未标定、OCR 引擎不可用 → RuntimeError）。

- [ ] **Step 1: 写失败测试**

`tests/test_runner.py` 末尾追加（若该文件无 `DefaultConfig`/`TasksConfig` 导入则补充；用纯配置构造 Runner，不碰窗口）：

```python
class TestModeAssets:
    """模式启动检查与模式解析。"""

    def _runner(self, mode_profile: ModeProfile) -> Runner:
        default_cfg = DefaultConfig(
            run={"mode": "pet"},
            main_c_profiles={
                "assault": MainCProfile(display_name="强袭", hero_template_dir="heroes/assault")
            },
            mode_profiles={"pet": mode_profile},
        )
        tasks_cfg = TasksConfig(
            hotspots={
                "pet_chat": {"x_ratio": 0.92, "y_ratio": 0.55},
                "recruit_pet_tab": {"x_ratio": 0.74, "y_ratio": 0.89},
                "join_coop": {"x_ratio": 0.61, "y_ratio": 0.76},
                "home_chat": {"x_ratio": 0.93, "y_ratio": 0.55},
                "open_recruit": {"x_ratio": 0.08, "y_ratio": 0.64},
                "open_difficulty_dialog": {"x_ratio": 0.58, "y_ratio": 0.83},
            },
            locators={
                "pet_ready_button": {
                    "strategy": "template",
                    "template": "buttons/bao_wei_chong_wu/zhun_bei.png",
                    "threshold": 0.82,
                },
                "pet_return_button": {
                    "strategy": "template",
                    "template": "buttons/bao_wei_chong_wu/fan_hui.png",
                    "threshold": 0.82,
                },
                "pet_leave_team": {
                    "strategy": "template",
                    "template": "buttons/bao_wei_chong_wu/tui_dui.png",
                    "threshold": 0.82,
                },
            },
            rois={
                "pet_difficulty_list": {
                    "relative_to": "client",
                    "x_ratio": 0.34,
                    "y_ratio": 0.57,
                    "width_ratio": 0.30,
                    "height_ratio": 0.24,
                }
            },
            pet_difficulty_recognition={"candidate_roi": "pet_difficulty_list",
                                        "min_confidence": 0.5},
        )
        return Runner(default_cfg, tasks_cfg, None)

    def _pet_profile(self) -> ModeProfile:
        return ModeProfile(
            display_name="保卫宠物",
            mode_chat_hotspot="pet_chat",
            entry={
                "mode_tab_hotspot": "recruit_pet_tab",
                "difficulty": {"kind": "ocr", "targets": ["普通", "中等"]},
            },
            match={"ready_button": "pet_ready_button", "leave_team": "pet_leave_team"},
            settlement={"like_chest": False, "return_button": "pet_return_button"},
        )

    def test_resolve_mode_profile_explicit_and_default(self) -> None:
        runner = self._runner(self._pet_profile())
        assert runner._resolve_mode_profile("pet") == ("pet", self._pet_profile())
        assert runner._resolve_mode_profile(None) == ("pet", self._pet_profile())

    def test_check_mode_assets_missing_hotspot_raises(self, tmp_path) -> None:
        runner = self._runner(self._pet_profile())
        runner.tasks_config.hotspots.pop("recruit_pet_tab")
        from wlxq_bot.assets import TemplatePack

        with pytest.raises(RuntimeError, match="recruit_pet_tab"):
            runner._check_mode_assets(
                self._pet_profile(),
                TemplatePack(client_size=(923, 1723), root=tmp_path),
                State.FIND_COOP,
            )

    def test_check_mode_assets_missing_template_raises(self, tmp_path) -> None:
        runner = self._runner(self._pet_profile())
        from wlxq_bot.assets import TemplatePack

        with pytest.raises(RuntimeError, match="zhun_bei"):
            runner._check_mode_assets(
                self._pet_profile(),
                TemplatePack(client_size=(923, 1723), root=tmp_path),
                State.FIND_COOP,
            )

    def test_check_mode_assets_ok_with_templates(self, tmp_path, monkeypatch) -> None:
        # OCR 模式会探测引擎：测试里打桩，避免真实加载 rapidocr
        monkeypatch.setattr("wlxq_bot.runner.ensure_engine", lambda: None)
        runner = self._runner(self._pet_profile())
        from wlxq_bot.assets import TemplatePack

        pack_root = tmp_path / "3000x2000"
        tpl_dir = pack_root / "buttons" / "bao_wei_chong_wu"
        tpl_dir.mkdir(parents=True)
        for name in ("zhun_bei.png", "fan_hui.png", "tui_dui.png"):
            (tpl_dir / name).write_bytes(b"t")
        runner._check_mode_assets(
            self._pet_profile(),
            TemplatePack(client_size=(923, 1723), root=pack_root),
            State.FIND_COOP,
        )

    def test_check_mode_assets_ocr_requires_nonzero_roi(self, tmp_path) -> None:
        runner = self._runner(self._pet_profile())
        runner.tasks_config.rois["pet_difficulty_list"].width_ratio = 0.0
        from wlxq_bot.assets import TemplatePack

        pack_root = tmp_path / "3000x2000"
        tpl_dir = pack_root / "buttons" / "bao_wei_chong_wu"
        tpl_dir.mkdir(parents=True)
        for name in ("zhun_bei.png", "fan_hui.png", "tui_dui.png"):
            (tpl_dir / name).write_bytes(b"t")
        with pytest.raises(RuntimeError, match="pet_difficulty_list"):
            runner._check_mode_assets(
                self._pet_profile(),
                TemplatePack(client_size=(923, 1723), root=pack_root),
                State.FIND_COOP,
            )
```

头部导入补：`import pytest`、`from wlxq_bot.assets import TemplatePack`（若未有）、`from wlxq_bot.config import DefaultConfig, MainCProfile, ModeProfile, TasksConfig`、`from wlxq_bot.models import State`、`from wlxq_bot.runner import Runner`（以文件现有导入为准，缺哪个补哪个）。

`tests/test_run_cli.py`：

1. `_patch_run_env` 的 `fake_run` 签名末尾加 `mode: str | None = None`，函数体内加 `captured["mode"] = mode`。**既有用例的 `captured == {...}` 精确字典断言要同步补 `"mode": None` 键**（如 `test_run_cli_passes_coop_difficulties_override`），否则全量回归会挂。
2. 新增：

```python
def test_run_cli_passes_mode_override(monkeypatch) -> None:
    captured: dict[str, object] = {}
    _patch_run_env(monkeypatch, captured)

    result = CliRunner().invoke(app, ["run", "coop", "--mode", "pet"])

    assert result.exit_code == 0, result.output
    assert captured["mode"] == "pet"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_runner.py::TestModeAssets tests/test_run_cli.py::test_run_cli_passes_mode_override -v`
Expected: FAIL（`AttributeError: 'Runner' object has no attribute '_resolve_mode_profile'` / CLI `mode` 键缺失）

- [ ] **Step 3: 实现 runner.py**

1. 导入补：`from wlxq_bot.config import ModeProfile, resolve_main_c_profile`（并入现有 config 导入）、`from wlxq_bot.perception.ocr import ensure_engine`。
2. `run()` 签名（`max_rounds: int | None = None` 之后）加 `mode: str | None = None,`；`run_config = self.default_config.run` 赋值行之后加：

```python
        mode_name, mode_profile = self._resolve_mode_profile(mode)
        effective_main_c_profile = resolve_main_c_profile(
            self.default_config.main_c_profiles, mode_profile, mc
        )
```

启动日志行加 `模式=%s`（`mode_name` 与 `mode_profile.display_name`）；`_check_difficulty_templates(...)` 调用改为条件门控：

```python
        if mode_profile.entry.difficulty.kind == "template":
            self._check_difficulty_templates(
                difficulty_levels, initial_state, pack, skip_difficulty=skip_difficulty
            )
        self._check_mode_assets(mode_profile, pack, initial_state)
```

3. `CoopPerception(...)` 构造加（`gold_read_interval` 之后）：

```python
            difficulty_kind=mode_profile.entry.difficulty.kind,
            pet_difficulty_targets=(
                mode_profile.entry.difficulty.targets or None
            ),
            locator_aliases={
                "ready_button": mode_profile.match.ready_button,
                "return_button": mode_profile.settlement.return_button,
                "coop_leave_team": mode_profile.match.leave_team,
            },
```

4. `CoopTask(...)` 构造改为：

```python
        task = CoopTask(
            ctx,
            run_config,
            CoopRole.HELPER,
            hotspots,
            coop_difficulties=difficulty_levels,
            skill_candidate_roi=skill_candidate_roi,
            skip_difficulty_selection=skip_difficulty,
            main_c_profile=effective_main_c_profile,
            mode_profile=mode_profile,
            pet_difficulty_targets=mode_profile.entry.difficulty.targets or None,
            pet_difficulty_min_confidence=float(
                self.tasks_config.pet_difficulty_recognition.get("min_confidence", 0.5)
            ),
        )
```

5. 新增方法（放在 `_check_difficulty_templates` 之后）：

```python
    def _resolve_mode_profile(self, mode: str | None) -> tuple[str, ModeProfile]:
        """解析本次运行模式：CLI 参数优先，缺省用 run.mode。"""
        mode_name = mode or self.default_config.run.mode
        profiles = self.default_config.mode_profiles
        if mode_name not in profiles:
            raise ValueError(
                f"未知模式 {mode_name!r}，可用: {sorted(profiles)}；"
                "需在 configs/default.yaml 的 mode_profiles 中定义"
            )
        return mode_name, profiles[mode_name]

    def _check_mode_assets(
        self, mode_profile: ModeProfile, pack: TemplatePack, start_state: State
    ) -> None:
        """按模式档案做启动前资产缺口检查（spec §4.4：缺失即在产生输入前报错）。"""
        if start_state != State.FIND_COOP:
            return
        display = mode_profile.display_name or "未知模式"
        missing_hotspots = [
            name
            for name in (
                mode_profile.mode_chat_hotspot,
                mode_profile.entry.mode_tab_hotspot,
            )
            if name and name not in self.tasks_config.hotspots
        ]
        if missing_hotspots:
            raise RuntimeError(
                f"模式 {display!r} 缺少 hotspot: {missing_hotspots}（configs/tasks.yaml）"
            )
        missing_locators: list[str] = []
        for locator_name in (
            mode_profile.match.ready_button,
            mode_profile.match.leave_team,
            mode_profile.settlement.return_button,
        ):
            locator = self.tasks_config.locators.get(locator_name)
            if not locator:
                missing_locators.append(f"{locator_name}(未定义)")
                continue
            template_rel = locator.get("template")
            if not template_rel or not pack.resolve_template(str(template_rel)).is_file():
                missing_locators.append(f"{locator_name}(模板缺失: {template_rel})")
                continue
            if not isinstance(locator.get("threshold"), (int, float)):
                missing_locators.append(f"{locator_name}(阈值未标定)")
        if missing_locators:
            raise RuntimeError(f"模式 {display!r} 缺少 locator: {missing_locators}")
        difficulty = mode_profile.entry.difficulty
        if difficulty.kind == "ocr":
            roi_name = str(
                self.tasks_config.pet_difficulty_recognition.get(
                    "candidate_roi", "pet_difficulty_list"
                )
            )
            roi = self.tasks_config.rois.get(roi_name)
            if roi is None or roi.width_ratio <= 0 or roi.height_ratio <= 0:
                raise RuntimeError(
                    f"模式 {display!r} 的 OCR 难度识别 ROI 未标定: rois.{roi_name}"
                    "（OCR 必须限域，禁止全图识别）"
                )
            try:
                ensure_engine()
            except RuntimeError as exc:
                raise RuntimeError(
                    f"模式 {display!r} 需要 OCR 引擎: {exc}"
                ) from exc
```

6. `select_difficulty()` 签名加 `mode: str | None = None,`；实现体：

```python
        mode_name, mode_profile = self._resolve_mode_profile(mode)
        logger.info(
            "启动独立难度选择，模式=%s（%s）", mode_name, mode_profile.display_name
        )
        pet_targets = mode_profile.entry.difficulty.targets
        is_ocr = mode_profile.entry.difficulty.kind == "ocr"
```

- 原 `self._check_difficulty_templates(difficulty_levels, State.FIND_COOP, pack)` 改为 `if not is_ocr: self._check_difficulty_templates(...)`，并追加 `self._check_mode_assets(mode_profile, pack, State.FIND_COOP)`。
- `CoopPerception(...)` 构造加与 run() 相同的 `difficulty_kind` / `pet_difficulty_targets` / `locator_aliases` 三个参数（`difficulty_reader` 不传，用真实引擎）。
- `CoopTask(...)` 构造加 `main_c_profile=resolve_main_c_profile(self.default_config.main_c_profiles, mode_profile, self.default_config.run.default_main_c)`、`mode_profile=mode_profile`、`pet_difficulty_targets=pet_targets or None`、`pet_difficulty_min_confidence=float(self.tasks_config.pet_difficulty_recognition.get("min_confidence", 0.5))`。

- [ ] **Step 4: 实现 cli.py**

`run` 命令参数（`start_state` 之前）加：

```python
    mode: str | None = typer.Option(
        None,
        "--mode",
        help="组队对局模式：coop（合作）| pet（保卫宠物）；不传时使用配置 run.mode",
    ),
```

`runner.run(...)` 调用加 `mode=mode,`；`fake_run` 已在测试中改签名，无需改 cli 内部。

`exec_select_difficulty` 参数（`--coop-difficulties` 之后）加同款 `--mode` 选项；该命令体内对 `runner.select_difficulty(...)` 的调用加 `mode=mode,`。注意 `tests/test_exec_cli.py` 若存在 monkeypatch 的 `select_difficulty` fake，同步在其签名末尾补 `mode: str | None = None`。

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_runner.py tests/test_run_cli.py tests/test_exec_cli.py -v`
Expected: 全部 PASS（既有 CLI 测试的 fake_run 已同步签名；若既有用例因签名新增失败，按新签名补 `mode=None` 默认值修复）

- [ ] **Step 6: Commit**

```bash
git add src/wlxq_bot/runner.py src/wlxq_bot/cli.py tests/test_runner.py tests/test_run_cli.py
git commit -m "feat(runner): 模式解析/启动缺口检查/CLI --mode 接线（保卫宠物模式 6/7）"
```

---

### Task 7: 文档同步与全量回归

**Files:**
- Modify: `docs/automation-flow.md`
- Modify: `docs/game-rules.md`
- Modify: `README.md`
- Modify: `docs/todo.md`（记录首跑验证清单）
- Test: 全量 `python -m pytest`

**Interfaces:**
- Consumes: 前 6 个任务的最终行为。
- Produces: 与实现一致的文档；无新代码接口。

- [ ] **Step 1: 更新 docs/automation-flow.md**

- 「进入招募小流程」mermaid 与正文：`OPEN_RECRUIT` 与 `OPEN_COOP_CHAT` 验证后插入「点击模式页签（两模式统一，2026-09-06 定案）」步骤。
- 新增「## 保卫宠物模式」一节：模式档案机制一段 + 宠物全流程（spec §4.3 的流程图文字化）+ OCR 难度勾选规则 + 结算跳过点赞宝箱 + 技能不封顶 + 首跑验证项引用。
- 「启动配置」节补 `run.mode` 与 `--mode`。

- [ ] **Step 2: 更新 docs/game-rules.md**

「棋盘与合作模式」节的入口描述补一句：进入招募后先点一次「合作模式」页签再打开难度弹窗（2026-09-06 定案，两模式统一）。「保卫宠物模式」节核对与实现一致（该节此前已随实机确认逐条写入）。

- [ ] **Step 3: 更新 README.md 与 docs/todo.md**

- README：新增 `wlxq-bot run coop --mode pet` 用法说明与一句模式简介；依赖不变（rapidocr 已是运行时依赖）。
- docs/todo.md：新增「保卫宠物首跑验证清单」小节，搬运 spec §5 的 6 项人工验证（页签重复点击、难度弹窗按钮位置、刷新行为、三模板 ROI/阈值标定、双倍奖励风险、对局标志命中），标注"未完成前不得声称宠物模式实机可用"。

- [ ] **Step 4: 全量回归**

Run: `python -m pytest -v`
Expected: 全部 PASS。任何失败先修复再提交；不跳过失败用例。

- [ ] **Step 5: 链接与术语一致性检查**

人工过一遍三个文档的改动：`docs/automation-flow.md`、`docs/game-rules.md`、`README.md` 中的模式名（保卫宠物）、配置键（`run.mode` / `mode_profiles` / `pet_difficulty_recognition`）、命令示例相互一致。

- [ ] **Step 6: Commit**

```bash
git add docs/automation-flow.md docs/game-rules.md README.md docs/todo.md
git commit -m "docs: 保卫宠物模式用法与流程文档（保卫宠物模式 7/7）"
```

---

## 完成定义（DoD）

1. `python -m pytest` 全绿。
2. 不带 `--mode` 的行为与现状唯一差异 = 合作页签点击（既有测试证明）。
3. `--mode pet` 下：Fake 单元序列覆盖页签→OCR勾选→结算跳过→pet_chat 全链路。
4. 实机首跑按 docs/todo.md 清单执行前，不声称保卫宠物模式实机可用（AGENTS.md 完成标准）。
