# 英雄变身（变身术）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在合作/保卫宠物两种模式的培养循环（BUILD_MAIN_C）中接入英雄变身：3 星非主 C 且无达标主 C 时，点击英雄 → 轮询头顶变身按钮 → 点击 → 验证类型必变，每局最多 2 次、可对同一位置连用。

**Architecture:** 纯决策进 `CultivationStrategy.select_transform`；执行是 `CoopTask._action_build_main_c` 里的任务级阶段机（poll → confirm → verify，settle 是 confirm 与 verify 之间的等待动作），不新增状态机 State；按钮识别是 `CoopPerception.observe` 的专项观察模式 `transform_button`（locator 模板匹配，全图退化 + 任务层窗口校验）；runner 负责把 locator 字典传入任务并在模板缺失时降级禁用。

**Tech Stack:** Python 3.10+ / Pydantic（config）/ OpenCV 模板匹配（复用现有 Vision）/ pytest（fake Vision + fake Observation，不真点桌面）。

**Spec:** `docs/superpowers/specs/2026-09-13-hero-transform-design.md`（本计划的一切决策以 spec 为准；执行者应先通读 spec）

## Global Constraints

- 仓库根目录 `D:\data\python\github\wlxq`，所有命令在此目录执行；Python 环境已就绪（`python -m pytest` 可用）。
- 遵守 `AGENTS.md`：业务坐标/阈值/重试次数必须进配置（不硬编码在任务代码）；每个输入动作有动作后验证；失败保守放弃、绝不结束任务、绝不盲点；日志用 `get_logger(__name__)` 的 `logger`，过程诊断不打循环内高频日志；测试不得真实点击桌面。
- 类型标注跟随现有代码风格（`from __future__ import annotations` 已在各文件头部）。
- 每个 Task 结束前运行该 Task 的测试命令并确认通过后才 commit；提交信息使用仓库现有风格（`feat(scope): 中文描述`）。
- 当前分支为 `feat/pet-defend-mode`，直接在其上提交；不要 push。
- 行为零变化承诺：`transform_enabled` 默认 true，但旧测试全部不传 `transform_button_locator`（构造参数默认 None → 变身禁用），因此现有测试的期望动作序列不受影响。Task 4 完成后必须运行整个 `tests/test_coop_task.py` 确认旧测试全绿。
- 各文件之间的接口名（供跨 Task 对齐）：
  - RunConfig 新字段：`transform_enabled` / `transform_max_per_round` / `transform_min_star_level` / `transform_button_poll_interval_seconds` / `transform_button_appear_timeout_seconds` / `transform_button_reclick_max` / `transform_settle_seconds` / `transform_verify_attempts`。
  - tasks.yaml locator 名：`transform_button`（键：`strategy` / `template` / `threshold` / `action_point` / `search_dx_ratio` / `search_up_ratio` / `search_down_ratio`）。
  - Observation 键：`transform_button_visible`（bool）、`transform_button_match`（MatchResult）。
  - 策略方法：`CultivationStrategy.select_transform(board: BoardSnapshot, used_count: int, max_count: int) -> BoardHero | None`。
  - CoopTask 构造参数：`transform_button_locator: dict[str, Any] | None = None`。
  - 放弃记忆字段：`_transform_skip_signature: tuple | None`（放弃变身时记住棋盘签名，棋盘演化前抑制再次触发——防止「放弃→立刻再触发」死循环，与合成失败记忆同思路）。
  - 动作 tag：`transform_click_hero` / `transform_button_poll` / `transform_reclick_hero` / `transform_click_button` / `transform_reclick_button` / `transform_settle` / `transform_verify_retry` / `transform_verify_wait`。
  - 观察模式字符串：`"transform_button"`。

---

### Task 1: RunConfig 变身参数 + default.yaml

**Files:**
- Modify: `src/wlxq_bot/config.py`（RunConfig，在 `skill_refresh_max_per_round` 字段后插入）
- Modify: `configs/default.yaml`（run 段，在 `merge_gift_fail_misses: 12` 后插入）
- Test: `tests/test_config.py`（追加测试类）

**Interfaces:**
- Consumes: 无（首个 Task）。
- Produces: RunConfig 上的 8 个 transform 字段（名字与默认值见下方代码，后续 Task 直接 `self._run_config.transform_*` 读取）。

- [ ] **Step 1: 写失败测试**

在 `tests/test_config.py` 文件末尾追加（注意该文件已 import `RunConfig`、`pytest`、`ValidationError`，直接用）：

```python
class TestTransformConfig:
    """英雄变身运行参数（设计文档 2026-09-13-hero-transform §4.4）。"""

    def test_transform_defaults(self) -> None:
        run = RunConfig()
        assert run.transform_enabled is True
        assert run.transform_max_per_round == 2
        assert run.transform_min_star_level == 3
        assert run.transform_button_poll_interval_seconds == 0.3
        assert run.transform_button_appear_timeout_seconds == 1.5
        assert run.transform_button_reclick_max == 2
        assert run.transform_settle_seconds == 4.0
        assert run.transform_verify_attempts == 2

    def test_transform_bounds_rejected(self) -> None:
        with pytest.raises(ValidationError, match="transform_max_per_round"):
            RunConfig(transform_max_per_round=-1)
        with pytest.raises(ValidationError, match="transform_min_star_level"):
            RunConfig(transform_min_star_level=0)
        with pytest.raises(ValidationError, match="transform_button_appear_timeout_seconds"):
            RunConfig(transform_button_appear_timeout_seconds=0.0)

    def test_repo_default_yaml_transform_section(self) -> None:
        """仓库 default.yaml 显式定义变身参数且与模型默认一致。"""
        from pathlib import Path

        from wlxq_bot.config import load_default_config

        cfg = load_default_config(Path("configs/default.yaml"))
        assert cfg.run.transform_enabled is True
        assert cfg.run.transform_settle_seconds == 4.0
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_config.py::TestTransformConfig -v`
Expected: FAIL，报 `AttributeError: 'RunConfig' object has no attribute 'transform_enabled'`（或 pydantic 额外字段拒绝错误）。

- [ ] **Step 3: 在 config.py 的 RunConfig 中加字段**

打开 `src/wlxq_bot/config.py`，找到这三行（约 318-320 行）：

```python
    skill_refresh_enabled: bool = True
    skill_refresh_max_per_round: int = Field(default=1, ge=0, le=10)
```

在这两行之后（保持 4 空格缩进）插入：

```python
    # ---- 英雄变身（2026-09-13 用户策略，设计文档 docs/superpowers/specs/2026-09-13-hero-transform-design.md）----
    # 3 星非主 C 且无达标主 C 时变身碰主 C（类型必变、星级不变、每局 2 次、
    # 可对同一位置连用）。按钮点击确认消失后才扣次数；失败路径一律保守放弃。
    transform_enabled: bool = True
    transform_max_per_round: int = Field(default=2, ge=0, le=10)
    # 触发变身的最低星级：仅该星级及以上的非主 C 英雄触发（用户定案仅 3 星）
    transform_min_star_level: int = Field(default=3, ge=1, le=4)
    # 点英雄后按钮轮询：实测约 1 秒内弹出，轮询不发送输入
    transform_button_poll_interval_seconds: float = Field(default=0.3, gt=0.0, le=5.0)
    transform_button_appear_timeout_seconds: float = Field(default=1.5, gt=0.0, le=10.0)
    # 重点击预算：按钮未弹出时重点击英雄 / 按钮未消失时重点按钮（共用）
    transform_button_reclick_max: int = Field(default=2, ge=0, le=10)
    # 点击按钮确认消失后的变身动画稳定等待（实测约 4 秒）
    transform_settle_seconds: float = Field(default=4.0, gt=0.0, le=30.0)
    # 动作后验证轮数：棋盘重识别确认「类型改变且星级不变」，每轮前再等 settle
    transform_verify_attempts: int = Field(default=2, ge=1, le=10)
```

- [ ] **Step 4: 在 default.yaml 的 run 段加同名键**

打开 `configs/default.yaml`，找到（约 199-205 行）：

```yaml
  merge_gift_settle_seconds: 2.0
  merge_gift_poll_interval_seconds: 0.5
  merge_gift_confirm_hits: 2
  merge_gift_fail_misses: 12
```

在这 4 行之后（保持 2 空格缩进）插入：

```yaml
  # 英雄变身（2026-09-13 用户策略，设计见 docs/superpowers/specs/2026-09-13-hero-transform-design.md）：
  # 3 星非主 C 且无达标主 C 时点击英雄 → 轮询头顶变身按钮 → 点击 → 验证
  # 类型必变星级不变；每局 transform_max_per_round 次、可对同一位置连用。
  # 按钮约 1 秒内弹出、无其他点击不消失、点按钮后先消失再变身（实机确认）。
  transform_enabled: true
  transform_max_per_round: 2
  transform_min_star_level: 3
  transform_button_poll_interval_seconds: 0.3
  transform_button_appear_timeout_seconds: 1.5
  transform_button_reclick_max: 2
  transform_settle_seconds: 4.0
  transform_verify_attempts: 2
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_config.py -v`
Expected: 全部 PASS（含原有测试与新增 TestTransformConfig 3 条）。

- [ ] **Step 6: Commit**

```bash
git add src/wlxq_bot/config.py configs/default.yaml tests/test_config.py
git commit -m "feat(config): 新增英雄变身运行参数——每局2次/3星阈值/轮询与验证预算"
```

---

### Task 2: tasks.yaml locator + perception 专项观察模式

**Files:**
- Modify: `configs/tasks.yaml`（locators 段末尾，`pet_leave_team` 条目后）
- Modify: `src/wlxq_bot/perception/coop.py`（`observe` 方法，`challenge_page` 分支后加 `transform_button` 分支）
- Test: `tests/test_coop_perception.py`（追加测试类）

**Interfaces:**
- Consumes: 无。
- Produces: tasks.yaml `locators.transform_button`（dict，含 `search_dx_ratio` / `search_up_ratio` / `search_down_ratio`，供 Task 4/5 的任务层窗口校验）；观察模式 `"transform_button"` → `raw_data["transform_button_visible"]`（bool）与 `raw_data["transform_button_match"]`（MatchResult，未命中时无 match 键、visible=False）。

- [ ] **Step 1: 写失败测试**

在 `tests/test_coop_perception.py` 末尾追加（该文件已有 `MatchResult`、`TasksConfig`、`CoopPerception`、`CoopRole`、`_make_window_ctx`、`_SingleTemplateVision`，直接用）：

```python
class TestTransformButtonObservation:
    """变身按钮只在 transform_button 观察模式识别（设计 §4.2/§4.3）。"""

    def _build(self, tmp_path, vision):
        from wlxq_bot.assets import TemplatePack

        pack_root = tmp_path / "3000x2000"
        buttons_dir = pack_root / "buttons"
        buttons_dir.mkdir(parents=True)
        (buttons_dir / "bian_shen.png").write_bytes(b"template")
        tasks_cfg = TasksConfig(
            locators={
                "transform_button": {
                    "strategy": "template",
                    "template": "buttons/bian_shen.png",
                    "threshold": 0.85,
                }
            }
        )
        return CoopPerception(
            vision,
            TemplatePack(client_size=(3000, 2000), root=pack_root),
            tasks_cfg,
            CoopRole.HELPER,
            "assault",
        )

    def test_detects_transform_button_only_in_requested_mode(self, tmp_path):
        match = MatchResult("bian_shen.png", (689, 986), 0.99)
        vision = _SingleTemplateVision("bian_shen.png", match)
        perception = self._build(tmp_path, vision)

        normal = perception.observe(_make_window_ctx(), None, observation_mode=None)
        transforming = perception.observe(
            _make_window_ctx(), None, observation_mode="transform_button"
        )

        assert "transform_button_visible" not in normal.raw_data
        assert transforming.flag("transform_button_visible") is True
        assert transforming.raw_data["transform_button_match"] == match
        assert len(vision.template_calls) == 1

    def test_missing_template_reports_invisible(self, tmp_path):
        """模板文件缺失时按不可见处理（runner 启动时已降级禁用，双保险）。"""
        vision = _SingleTemplateVision("bian_shen.png", None)
        perception = self._build(tmp_path, vision)

        observation = perception.observe(
            _make_window_ctx(), None, observation_mode="transform_button"
        )

        assert observation.flag("transform_button_visible") is False
        assert "transform_button_match" not in observation.raw_data
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_coop_perception.py::TestTransformButtonObservation -v`
Expected: FAIL——`transform_button_visible` 键不存在（`flag()` 默认 False，第一条断言 `"transform_button_visible" not in normal.raw_data` 可能碰巧通过，但 `transforming.flag(...) is True` 必然 FAIL）。

- [ ] **Step 3: tasks.yaml 加 locator 条目**

打开 `configs/tasks.yaml`，在 locators 段的 `pet_leave_team` 条目（该段最后一个条目）之后插入：

```yaml
  # 英雄变身按钮（2026-09-13 实机采集）：点击英雄后约 1 秒内弹出、悬浮于英雄
  # 头顶（实测水平对齐格子中心±4.5px、按钮底边略压本格顶边）；5/5 实机截图
  # 命中 0.984~0.9998。仅 observation_mode=transform_button（任务层变身轮询
  # 阶段）识别；未配置 roi 时退化全图匹配，任务层再用 search_*_ratio 窗口
  # 校验命中位置是否在目标英雄格上方（窗口由棋盘格尺寸推导：
  # ±1.5 格宽 / 上探 2.5 格高 / 下压 0.4 格高，见设计文档 §4.3）
  transform_button:
    strategy: template
    template: buttons/bian_shen.png
    threshold: 0.85
    action_point:
      source: match_center
    search_dx_ratio: 0.11
    search_up_ratio: 0.093
    search_down_ratio: 0.015
```

插入锚点（old_string 用 pet_leave_team 的完整 6 行，new_string 在其后接上新条目）：

```yaml
  pet_leave_team:
    strategy: template
    template: buttons/bao_wei_chong_wu/tui_dui.png
    threshold: 0.82
    action_point:
      source: match_center
```

- [ ] **Step 4: perception 加观察模式分支**

打开 `src/wlxq_bot/perception/coop.py`，找到 `observe` 方法中这一段（约 356-358 行）：

```python
        elif observation_mode == "challenge_page":
            # 领地入口最后一步：OCR 挑战页按钮文字（页面到达锚点）
            self._detect_challenge_text(ctx, frame, raw_data)
```

在这 3 行之后（与其它 elif 同级、8 空格缩进）插入：

```python
        elif observation_mode == "transform_button":
            # 变身按钮轮询（任务层变身阶段机，仅 poll/confirm 两阶段请求）：
            # 匹配 locator transform_button；模板缺失时 _match_locator_template
            # 返回 None（按不可见处理，runner 启动检查已先行降级禁用）
            transform_locator = self._cfg.locators.get("transform_button")
            transform_match = (
                self._match_locator_template(ctx, frame, transform_locator)
                if transform_locator
                else None
            )
            raw_data["transform_button_visible"] = transform_match is not None
            if transform_match is not None:
                matches.append(transform_match)
                raw_data["transform_button_match"] = transform_match
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_coop_perception.py -v`
Expected: 全部 PASS。

- [ ] **Step 6: Commit**

```bash
git add configs/tasks.yaml src/wlxq_bot/perception/coop.py tests/test_coop_perception.py
git commit -m "feat(perception): 变身按钮专项观察模式 transform_button——locator 模板匹配 + 窗口校验参数入库"
```

---

### Task 3: CultivationStrategy.select_transform 纯决策

**Files:**
- Modify: `src/wlxq_bot/tasks/strategy.py`（`should_topup` 方法后插入新方法）
- Test: `tests/test_coop_task.py`（追加测试类）

**Interfaces:**
- Consumes: `BoardSnapshot` / `BoardHero`（已有）；RunConfig 的 `transform_min_star_level`（Task 1）。
- Produces: `select_transform(board, used_count, max_count) -> BoardHero | None`，Task 4 在 `_action_build_main_c` 中调用。

- [ ] **Step 1: 写失败测试**

在 `tests/test_coop_task.py` 末尾追加（该文件已有 `make_task` / `make_hero` / `make_board`）：

```python
class TestSelectTransform:
    """变身目标纯决策（设计文档 §4.1）。"""

    def test_returns_first_three_star_non_main_c(self):
        task = make_task()
        board = make_board(
            [
                make_hero("snow", star=2, pos=(100, 100)),
                make_hero("angel", star=3, pos=(300, 300)),
                make_hero("death_knight", star=3, pos=(500, 500)),
            ]
        )
        target = task._strategy.select_transform(board, 0, 2)
        assert target is not None
        assert target.hero_type == "angel"
        assert target.star_level == 3

    def test_none_when_main_c_target_reached(self):
        task = make_task()
        board = make_board(
            [
                make_hero("assault", star=2, pos=(100, 100)),
                make_hero("snow", star=3, pos=(300, 300)),
            ]
        )
        assert task._strategy.select_transform(board, 0, 2) is None

    def test_none_when_charges_exhausted(self):
        task = make_task()
        board = make_board([make_hero("snow", star=3, pos=(300, 300))])
        assert task._strategy.select_transform(board, 2, 2) is None

    def test_none_when_below_min_star(self):
        task = make_task()
        board = make_board([make_hero("snow", star=2, pos=(300, 300))])
        assert task._strategy.select_transform(board, 0, 2) is None

    def test_four_star_non_main_c_triggers(self):
        """min_star_level 是下限：4 星非主 C 同样触发（>= 3）。"""
        task = make_task()
        board = make_board([make_hero("snow", star=4, pos=(300, 300))])
        target = task._strategy.select_transform(board, 0, 2)
        assert target is not None
        assert target.star_level == 4

    def test_none_when_board_empty(self):
        task = make_task()
        assert task._strategy.select_transform(make_board([]), 0, 2) is None
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_coop_task.py::TestSelectTransform -v`
Expected: FAIL，`AttributeError: 'CultivationStrategy' object has no attribute 'select_transform'`。

- [ ] **Step 3: 实现 select_transform**

打开 `src/wlxq_bot/tasks/strategy.py`，找到 `should_topup` 方法的最后一行：

```python
        return skill_selections >= self._profile.topup_after_skill_selections
```

在这行之后（保持 4 空格方法缩进）插入：

```python

    # ------------------------------------------------------------------
    # 变身决策
    # ------------------------------------------------------------------

    def select_transform(
        self,
        board: BoardSnapshot,
        used_count: int,
        max_count: int,
    ) -> BoardHero | None:
        """选择本次变身目标；None 表示不变身（设计文档 §4.1）。

        条件（全部满足才返回目标）：
        1. 本局变身次数未用完（used_count < max_count）；
        2. 尚无达标主 C（复用 has_target_main_c）；
        3. 棋盘存在星级不低于 run.transform_min_star_level 的非主 C 英雄。

        多个候选取 heroes 列表中的第一个（感知层按棋盘格序构建 heroes，
        即按格位顺序）；返回 None 时调用方落入常规召唤/合成决策。
        """
        if used_count >= max_count:
            return None
        if self.has_target_main_c(board):
            return None
        min_star = self._run_config.transform_min_star_level
        for hero in board.heroes:
            if hero.hero_type != self._main_c and hero.star_level >= min_star:
                return hero
        return None
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_coop_task.py::TestSelectTransform -v`
Expected: 6 条全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add src/wlxq_bot/tasks/strategy.py tests/test_coop_task.py
git commit -m "feat(task): 变身目标纯决策 select_transform——无达标主C+次数未满+3星非主C"
```

---

### Task 4: CoopTask 变身阶段机

本 Task 是核心。共 8 个插入点，全部给出精确锚文本与完整代码。阶段机只有三个阶段：`poll`（点英雄后轮询按钮）→ `confirm`（点按钮后确认消失）→ `verify`（settle 等待后棋盘验证）；「settle」不是独立阶段，是 confirm 成功时返回的等待动作。

**Files:**
- Modify: `src/wlxq_bot/tasks/coop.py`（8 处）
- Test: `tests/test_coop_task.py`（追加测试类 + 修改 `make_task`）

**Interfaces:**
- Consumes: `select_transform`（Task 3）、观察模式 `"transform_button"`（Task 2）、RunConfig transform 字段（Task 1）。
- Produces: CoopTask 构造参数 `transform_button_locator: dict[str, Any] | None = None`（Task 5 的 runner 传入）；动作 tag 见 Global Constraints。

- [ ] **Step 1: 修改 tests/test_coop_task.py 的 make_task**

找到 `make_task` 签名（约 54-68 行）：

```python
    pet_difficulty_targets: list[str] | None = None,
) -> CoopTask:
```

改为：

```python
    pet_difficulty_targets: list[str] | None = None,
    transform_button_locator: dict | None = None,
) -> CoopTask:
```

再找到 `make_task` 的 return 尾部（约 113-118 行）：

```python
        mode_profile=mode_profile or default_entry_mode_profile(),
        pet_difficulty_targets=pet_difficulty_targets,
    )
```

改为：

```python
        mode_profile=mode_profile or default_entry_mode_profile(),
        pet_difficulty_targets=pet_difficulty_targets,
        transform_button_locator=transform_button_locator,
    )
```

并在文件顶部「构造辅助」注释块之后（`make_window_ctx` 定义之前）加模块级夹具常量：

```python
# 变身按钮 locator 测试夹具（与 configs/tasks.yaml 的 transform_button 同构）
TRANSFORM_LOCATOR = {
    "strategy": "template",
    "template": "buttons/bian_shen.png",
    "threshold": 0.85,
    "action_point": {"source": "match_center"},
    "search_dx_ratio": 0.11,
    "search_up_ratio": 0.093,
    "search_down_ratio": 0.015,
}
```

- [ ] **Step 2: 写失败的阶段机测试**

在 `tests/test_coop_task.py` 末尾追加：

```python
class TestTransformSequence:
    """变身阶段机：poll → confirm → verify（设计文档 §4.2）。

    驱动方式与仓库惯例一致：直接设置任务私有字段 + 构造 Observation 断言动作。
    """

    def _ready_task(self) -> CoopTask:
        task = make_task(transform_button_locator=TRANSFORM_LOCATOR)
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        return task

    def _button_obs(self, match: MatchResult | None) -> Observation:
        raw: dict = {"transform_button_visible": match is not None}
        if match is not None:
            raw["transform_button_match"] = match
        return Observation(frame_id=1, raw_data=raw)

    # ---- 触发 ----

    def test_triggers_click_hero_on_three_star_non_main_c(self):
        task = self._ready_task()
        board = make_board([make_hero("snow", star=3, pos=(100, 100))])
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "click"
        assert decision[0].tag == "transform_click_hero"
        assert decision[0].target == (100, 100)
        assert task._transform_stage == "poll"
        assert task._transform_target is not None
        assert task._transform_target.hero_type == "snow"

    def test_no_trigger_without_locator(self):
        """未接线 locator（如模板缺失降级）时不变身，回落常规决策。"""
        task = make_task()  # 默认不传 locator
        task.ctx.current_state = State.BUILD_MAIN_C
        task._summon_count = task._run_config.minimum_summon_count_before_skills
        board = make_board([make_hero("snow", star=3, pos=(100, 100))])
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag != "transform_click_hero"
        assert task._transform_stage is None

    def test_no_trigger_when_main_c_ready(self):
        task = self._ready_task()
        board = make_board(
            [
                make_hero("assault", star=2, pos=(100, 100)),
                make_hero("snow", star=3, pos=(300, 300)),
            ]
        )
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert task._transform_stage is None

    def test_no_trigger_when_charges_exhausted(self):
        task = self._ready_task()
        task._transform_used_count = 2
        board = make_board([make_hero("snow", star=3, pos=(100, 100))])
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag != "transform_click_hero"

    # ---- poll ----

    def _enter_poll(self, task: CoopTask) -> None:
        board = make_board([make_hero("snow", star=3, pos=(100, 100))])
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None and decision[0].tag == "transform_click_hero"

    def test_poll_clicks_button_when_in_window(self):
        task = self._ready_task()
        self._enter_poll(task)
        match = MatchResult("bian_shen.png", (100, 45), 0.95)
        decision = task.decide_action(self._button_obs(match), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "click"
        assert decision[0].tag == "transform_click_button"
        assert decision[0].target == (100, 45)
        assert task._transform_stage == "confirm"
        assert task._transform_button_pos == (100, 45)

    def test_poll_waits_when_button_missing(self):
        task = self._ready_task()
        self._enter_poll(task)
        decision = task.decide_action(self._button_obs(None), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "wait"
        assert decision[0].tag == "transform_button_poll"

    def test_poll_rejects_match_outside_window(self):
        """窗口外命中（如点击落到了别的英雄上弹出的按钮）不算我们的按钮。"""
        task = self._ready_task()
        self._enter_poll(task)
        match = MatchResult("bian_shen.png", (500, 45), 0.95)
        decision = task.decide_action(self._button_obs(match), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "transform_button_poll"

    def test_poll_timeout_reclicks_hero_then_gives_up(self):
        task = self._ready_task()
        self._enter_poll(task)
        task._transform_poll_deadline = 0.0  # 轮询超时
        board = make_board([make_hero("snow", star=3, pos=(100, 100))])
        obs = Observation(frame_id=1, board=board)
        # 第一次超时：重点击英雄
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "transform_reclick_hero"
        assert task._transform_reclicks == 1
        # 耗尽重点击预算后再超时：放弃，回落常规决策；放弃记忆抑制立刻再触发
        task._transform_reclicks = task._run_config.transform_button_reclick_max
        task._transform_poll_deadline = 0.0
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag not in {
            "transform_click_hero",
            "transform_reclick_hero",
            "transform_button_poll",
        }
        assert task._transform_stage is None
        assert task._transform_used_count == 0
        assert task._transform_skip_signature is not None

    def test_giveup_suppression_lifts_when_board_evolves(self):
        """棋盘演化（如召唤新英雄）后放弃记忆解除，变身可再次触发。"""
        task = self._ready_task()
        self._enter_poll(task)
        task._transform_poll_deadline = 0.0
        task._transform_reclicks = task._run_config.transform_button_reclick_max
        board = make_board([make_hero("snow", star=3, pos=(100, 100))])
        decision = task.decide_action(
            Observation(frame_id=1, board=board), make_window_ctx()
        )
        assert decision is not None
        assert decision[0].tag != "transform_click_hero"
        # 召唤后棋盘多了一个英雄：签名变化，解除抑制
        board2 = make_board(
            [
                make_hero("snow", star=3, pos=(100, 100)),
                make_hero("angel", star=1, pos=(300, 300)),
            ]
        )
        decision = task.decide_action(
            Observation(frame_id=1, board=board2), make_window_ctx()
        )
        assert decision is not None
        assert decision[0].tag == "transform_click_hero"
        assert task._transform_skip_signature is None

    # ---- confirm ----

    def _enter_confirm(self, task: CoopTask) -> None:
        self._enter_poll(task)
        match = MatchResult("bian_shen.png", (100, 45), 0.95)
        decision = task.decide_action(self._button_obs(match), make_window_ctx())
        assert decision is not None and decision[0].tag == "transform_click_button"

    def test_confirm_button_gone_settles_and_deducts(self):
        task = self._ready_task()
        self._enter_confirm(task)
        decision = task.decide_action(self._button_obs(None), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "wait"
        assert decision[0].tag == "transform_settle"
        assert decision[0].duration == task._run_config.transform_settle_seconds
        assert task._transform_stage == "verify"
        assert task._transform_used_count == 1

    def test_confirm_button_still_there_reclicks(self):
        task = self._ready_task()
        self._enter_confirm(task)
        match = MatchResult("bian_shen.png", (105, 48), 0.94)  # 原位附近仍在
        decision = task.decide_action(self._button_obs(match), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "click"
        assert decision[0].tag == "transform_reclick_button"
        assert task._transform_used_count == 0

    def test_confirm_button_moved_counts_as_gone(self):
        """原位按钮消失、别处出现新按钮（如另一次点击残留）按已消失处理。"""
        task = self._ready_task()
        self._enter_confirm(task)
        match = MatchResult("bian_shen.png", (400, 700), 0.94)
        decision = task.decide_action(self._button_obs(match), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "transform_settle"
        assert task._transform_used_count == 1

    def test_confirm_exhausted_gives_up_without_deduct(self):
        task = self._ready_task()
        self._enter_confirm(task)
        task._transform_reclicks = task._run_config.transform_button_reclick_max
        match = MatchResult("bian_shen.png", (100, 45), 0.94)
        board = make_board([make_hero("snow", star=3, pos=(100, 100))])
        obs = Observation(
            frame_id=1,
            board=board,
            raw_data={"transform_button_visible": True, "transform_button_match": match},
        )
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag not in {"transform_click_button", "transform_reclick_button"}
        assert task._transform_stage is None
        assert task._transform_used_count == 0
        assert task._transform_skip_signature is not None

    # ---- verify ----

    def _enter_verify(self, task: CoopTask) -> None:
        self._enter_confirm(task)
        decision = task.decide_action(self._button_obs(None), make_window_ctx())
        assert decision is not None and decision[0].tag == "transform_settle"

    def test_verify_success_retriggers_on_same_position(self):
        """变身出非主C：验证通过后立即对同一位置再次触发（连用，用户定案）。"""
        task = self._ready_task()
        self._enter_verify(task)
        board = make_board([make_hero("angel", star=3, pos=(100, 100))])
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "transform_click_hero"  # 第二次连用
        assert task._transform_stage == "poll"
        assert task._transform_target is not None
        assert task._transform_target.hero_type == "angel"
        assert task._transform_used_count == 1

    def test_verify_main_c_result_goes_to_skills(self):
        """变出主C：验证通过后正常达标，进入选技能。"""
        task = self._ready_task()
        self._enter_verify(task)
        board = make_board([make_hero("assault", star=3, pos=(100, 100))])
        decision = task.decide_action(Observation(frame_id=1, board=board), make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "main_c_ready"
        assert decision[1] == State.SELECT_MAIN_C_SKILLS

    def test_verify_unchanged_type_retries_then_gives_up(self):
        task = self._ready_task()
        self._enter_verify(task)
        board = make_board([make_hero("snow", star=3, pos=(100, 100))])  # 类型未变
        obs = Observation(frame_id=1, board=board)
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag == "transform_verify_retry"
        task._transform_verify_attempts = task._run_config.transform_verify_attempts
        decision = task.decide_action(obs, make_window_ctx())
        assert decision is not None
        assert decision[0].tag not in {
            "transform_verify_retry",
            "transform_click_hero",  # 放弃记忆抑制立刻再触发
        }
        assert task._transform_stage is None
        assert task._transform_skip_signature is not None

    def test_verify_empty_board_waits(self):
        task = self._ready_task()
        self._enter_verify(task)
        decision = task.decide_action(Observation(frame_id=1), make_window_ctx())
        assert decision is not None
        assert decision[0].kind == "wait"
        assert decision[0].tag == "transform_verify_wait"

    # ---- 钩子与重置 ----

    def test_observation_mode_and_board_watch_by_stage(self):
        task = self._ready_task()
        self._enter_poll(task)
        assert task.observation_mode() == "transform_button"
        assert task.wants_board_watch() is False
        task._transform_stage = "verify"
        assert task.observation_mode() is None
        assert task.wants_board_watch() is True

    def test_round_reset_clears_counter(self):
        task = self._ready_task()
        self._enter_verify(task)
        task._transform_used_count = 2
        task._transform_skip_signature = (("snow", 3, (100, 100)),)
        task.on_action_verified(Action(kind="wait", tag="ready_match"), State.ENTER_MATCH)
        assert task._transform_used_count == 0
        assert task._transform_stage is None
        assert task._transform_skip_signature is None
```

- [ ] **Step 3: 运行确认失败**

Run: `python -m pytest tests/test_coop_task.py::TestTransformSequence -v`
Expected: FAIL——`CoopTask.__init__()` 不接受 `transform_button_locator`（TypeError）。

- [ ] **Step 4: coop.py 插入点 1——typing import**

找到：

```python
from typing import Literal
```

改为：

```python
from typing import Any, Literal
```

- [ ] **Step 5: 插入点 2——构造参数**

找到 `__init__` 签名尾部（约 138-140 行）：

```python
        rng: random.Random | None = None,
    ) -> None:
```

改为：

```python
        rng: random.Random | None = None,
        transform_button_locator: dict[str, Any] | None = None,
    ) -> None:
```

- [ ] **Step 6: 插入点 3——状态字段**

找到（约 253-257 行，__init__ 内；注意这段注释使其唯一）：

```python
        # 局内技能页「刷新」：本局已用次数、按钮未识别到的等待重试计数
        # （2026-09-12 用户确认：每局一次，刷新后必选一张；「等卡面换新」由
        # 动作后验证承担——验证帧 OCR 出新卡面即已可识别，不再额外固定等待）
        self._skill_refresh_used_count = 0
        self._skill_refresh_button_waits = 0
```

在其后插入（8 空格缩进）：

```python
        # 变身（2026-09-13 用户策略，设计文档 2026-09-13-hero-transform）：
        # locator 未接线（配置缺失/模板缺失降级）时整分支禁用
        self._transform_locator = transform_button_locator
        # 阶段机：None=未在进行；poll=已点英雄轮询按钮；confirm=已点按钮
        # 确认消失；verify=settle 等待后棋盘验证
        self._transform_stage: str | None = None
        self._transform_target: BoardHero | None = None
        self._transform_button_pos: tuple[int, int] | None = None
        self._transform_reclicks = 0
        self._transform_verify_attempts = 0
        # 本局已用变身次数（按钮确认消失后扣减；重置点与技能刷新一致）
        self._transform_used_count = 0
        self._transform_poll_deadline = 0.0
        # 放弃记忆：放弃变身时的棋盘签名；棋盘未演化前不再触发（防死循环）
        self._transform_skip_signature: tuple | None = None
```

- [ ] **Step 7: 插入点 4——wants_board_watch / observation_mode 钩子**

找到 `wants_board_watch` 内：

```python
        if self._awaiting_merge_gift:
            return False
```

（此 2 行组合在文件中唯一）在其后插入：

```python
        if self._transform_stage is not None:
            # 变身阶段机：poll/confirm 单帧快识别（含按钮专项模式）；
            # verify 恢复多帧棋盘识别做动作后验证
            return self._transform_stage == "verify"
```

找到 `observation_mode` 开头：

```python
    def observation_mode(self) -> str | None:
        if self._recruit_step is _RecruitStep.DETECT_ENTRY_PAGE:
```

在 `def` 行与第一个 `if` 之间插入：

```python
        if self._transform_stage is not None:
            # 变身阶段机接管观察模式（设计 §4.2）：poll/confirm 用按钮专项
            # 模式；verify 恢复默认（无专项模式）。整阶段机拦截而不是只拦
            # poll/confirm——否则 verify 阶段落入下方招募步骤分支（如测试
            # 夹具停在 OPEN_TERRITORY 时返回 territory_nav，2026-09-13 执行
            # 反馈修正）
            if self._transform_stage in ("poll", "confirm"):
                return "transform_button"
            return None
```

- [ ] **Step 8: 插入点 5——阶段机入口与触发检查（_action_build_main_c 两处）**

8a. 阶段机入口——找到（约 2043-2045 行，`_awaiting_merge_gift` 块与赠送页分支之间）：

```python
        # 合成 4 星后系统赠送的技能选择页（实机确认 2026-08-16）：两个 3 星
```

（该注释首行唯一）在这行**之前**插入（8 空格缩进）：

```python
        # ---- 变身阶段机（2026-09-13 用户策略，设计文档 §4.2）----
        # 进行中的变身优先于下方所有常规决策；返回 None 表示本次变身结束
        # （成功/放弃/预算耗尽），清空阶段后落入常规培养决策
        if self._transform_stage is not None:
            stage_action = self._transform_stage_action(observation, window_ctx)
            if stage_action is not None:
                return stage_action

```

8b. 触发检查——找到批次规划注释（约 2200 行）：

```python
        # ---- 批次规划（2026-09-10 用户策略）：识别一次、动作攒批、批后统一识别 ----
```

（唯一）在这行**之前**插入（8 空格缩进）：

```python
        # ---- 变身触发（设计文档 §4.1：无达标主C + 次数未满 + 3星非主C）----
        # 免费且可能直接达标，优先于批次召唤/合成；locator 未接线（模板缺失
        # 降级）或开关关闭时整分支跳过；放弃记忆未解除（棋盘签名未演化）时
        # 抑制触发，防止「放弃 → 立刻再触发」死循环
        if (
            self._transform_skip_signature is not None
            and self._board_signature(board) != self._transform_skip_signature
        ):
            # 棋盘已演化（召唤/合成改变了英雄集合）：解除放弃记忆的抑制
            self._transform_skip_signature = None
        if (
            self._run_config.transform_enabled
            and self._transform_locator is not None
            and self._transform_skip_signature is None
        ):
            transform_target = self._strategy.select_transform(
                board,
                self._transform_used_count,
                self._run_config.transform_max_per_round,
            )
            if transform_target is not None:
                self._transform_stage = "poll"
                self._transform_target = transform_target
                self._transform_button_pos = None
                self._transform_reclicks = 0
                self._transform_verify_attempts = 0
                self._transform_poll_deadline = (
                    self._clock() + self._run_config.transform_button_appear_timeout_seconds
                )
                logger.info(
                    "触发变身：%s%d星 %s 变身碰主C（本局已用 %d/%d 次）",
                    transform_target.hero_type,
                    transform_target.star_level,
                    _hero_location(transform_target),
                    self._transform_used_count,
                    self._run_config.transform_max_per_round,
                )
                return (
                    Action(
                        kind="click",
                        target=transform_target.position,
                        duration=0.08,
                        post_delay=self._run_config.transform_button_poll_interval_seconds,
                        verification="immediate",
                        tag="transform_click_hero",
                        reason="点击3星非主C英雄，弹出变身按钮",
                    ),
                    State.BUILD_MAIN_C,
                )

```

- [ ] **Step 9: 插入点 6——阶段机三个方法 + 窗口校验 + 清空助手**

找到（约 2333 行）：

```python
    def _batch_summon_action(
```

（唯一）在这行**之前**插入以下完整方法（4 空格方法缩进）：

```python
    def _transform_stage_action(
        self,
        observation: Observation,
        window_ctx: WindowContext | None,
    ) -> tuple[Action, State] | None:
        """变身阶段机主体（设计文档 §4.2）：poll → confirm → verify。

        返回 None = 本次变身结束（成功、放弃或预算耗尽），调用方先经
        ``_clear_transform_state`` 清空后落入常规培养决策；任何失败路径都
        不结束任务、不盲点。settle 不是独立阶段：confirm 确认按钮消失后
        直接置 verify 并返回 settle 时长的等待动作。
        """
        cfg = self._run_config
        target = self._transform_target
        assert target is not None  # 阶段存在即有目标（触发处成对设置）

        def _wait(tag: str, duration: float, reason: str) -> tuple[Action, State]:
            return (
                Action(
                    kind="wait",
                    duration=duration,
                    verification="immediate",
                    tag=tag,
                    reason=reason,
                ),
                State.BUILD_MAIN_C,
            )

        if self._transform_stage == "poll":
            match = self._flag_match(observation, "transform_button_match")
            if (
                match is not None
                and self._match_in_transform_window(
                    match.position, target.position, window_ctx
                )
            ):
                self._transform_button_pos = match.position
                self._transform_stage = "confirm"
                return (
                    Action(
                        kind="click",
                        target=match.position,
                        duration=0.08,
                        post_delay=0.3,
                        verification="immediate",
                        tag="transform_click_button",
                        reason=f"点击变身按钮（置信度 {match.confidence:.2f}）",
                    ),
                    State.BUILD_MAIN_C,
                )
            if self._clock() < self._transform_poll_deadline:
                return _wait(
                    "transform_button_poll",
                    cfg.transform_button_poll_interval_seconds,
                    "轮询等待变身按钮弹出",
                )
            if self._transform_reclicks < cfg.transform_button_reclick_max:
                self._transform_reclicks += 1
                self._transform_poll_deadline = (
                    self._clock() + cfg.transform_button_appear_timeout_seconds
                )
                logger.info(
                    "变身按钮未弹出，重点击英雄 %d/%d（按钮约1秒内弹出，仅未弹出时重点击）",
                    self._transform_reclicks,
                    cfg.transform_button_reclick_max,
                )
                return (
                    Action(
                        kind="click",
                        target=target.position,
                        duration=0.08,
                        post_delay=cfg.transform_button_poll_interval_seconds,
                        verification="immediate",
                        tag="transform_reclick_hero",
                        reason="变身按钮未弹出，重点击英雄",
                    ),
                    State.BUILD_MAIN_C,
                )
            logger.warning(
                "变身按钮轮询超时（游戏侧次数已尽或按钮未渲染），放弃本次变身，"
                "不扣次数（计数失步自愈兜底）"
            )
            self._remember_transform_giveup(observation)
            self._clear_transform_state()
            return None

        if self._transform_stage == "confirm":
            match = self._flag_match(observation, "transform_button_match")
            button_tolerance = (
                max(20, int(0.04 * window_ctx.client_size[0]))
                if window_ctx is not None
                else 40
            )
            still_at_pos = (
                match is not None
                and self._transform_button_pos is not None
                and abs(match.position[0] - self._transform_button_pos[0])
                <= button_tolerance
                and abs(match.position[1] - self._transform_button_pos[1])
                <= button_tolerance
            )
            if not still_at_pos:
                # 按钮已消失（原位不在）：变身已开始——扣减本局次数（开环计次，
                # 与技能刷新同哲学），等待动画后棋盘验证
                self._transform_used_count += 1
                self._transform_stage = "verify"
                logger.info(
                    "变身按钮已消失（变身开始），等待动画稳定（本局已用 %d/%d 次）",
                    self._transform_used_count,
                    cfg.transform_max_per_round,
                )
                return _wait(
                    "transform_settle",
                    cfg.transform_settle_seconds,
                    f"变身动画稳定等待（{cfg.transform_settle_seconds:.0f}秒，实测约4秒）",
                )
            if self._transform_reclicks < cfg.transform_button_reclick_max:
                self._transform_reclicks += 1
                return (
                    Action(
                        kind="click",
                        target=self._transform_button_pos,
                        duration=0.08,
                        post_delay=0.3,
                        verification="immediate",
                        tag="transform_reclick_button",
                        reason="变身按钮未消失，重点按钮（点击可能未落上）",
                    ),
                    State.BUILD_MAIN_C,
                )
            logger.warning("变身按钮点击未生效（原位按钮仍在），放弃本次变身，不扣次数")
            self._remember_transform_giveup(observation)
            self._clear_transform_state()
            return None

        # stage == "verify"：多帧棋盘识别已恢复（wants_board_watch 对本阶段
        # 返回 True）；「类型改变且星级不变」才算成功（变身类型必变）
        board = observation.board
        if board is None or not board.heroes:
            return _wait(
                "transform_verify_wait",
                0.5,
                "棋盘识别为空（页面遮挡），等待恢复后验证变身",
            )
        near_tolerance = (
            int(0.03 * window_ctx.client_size[1]) if window_ctx is not None else 52
        )
        nearest = min(
            board.heroes,
            key=lambda hero: (
                (hero.position[0] - target.position[0]) ** 2
                + (hero.position[1] - target.position[1]) ** 2
            ),
        )
        near = (
            abs(nearest.position[0] - target.position[0]) <= near_tolerance
            and abs(nearest.position[1] - target.position[1]) <= near_tolerance
        )
        if (
            near
            and nearest.star_level == target.star_level
            and nearest.hero_type != target.hero_type
        ):
            logger.info(
                "变身验证通过：%s%d星 → %s%d星（同位置星级不变、类型必变）",
                target.hero_type,
                target.star_level,
                nearest.hero_type,
                nearest.star_level,
            )
            self._clear_transform_state()
            return None
        self._transform_verify_attempts += 1
        if self._transform_verify_attempts < cfg.transform_verify_attempts:
            return _wait(
                "transform_verify_retry",
                cfg.transform_settle_seconds,
                (
                    f"变身未确认（第 {self._transform_verify_attempts} 次：类型未变"
                    "或动画未完），再等动画后重识别"
                ),
            )
        logger.warning(
            "变身验证 %d 次未确认（类型未变或英雄缺失），放弃验证，继续常规决策",
            cfg.transform_verify_attempts,
        )
        self._remember_transform_giveup(observation)
        self._clear_transform_state()
        return None

    def _remember_transform_giveup(self, observation: Observation) -> None:
        """放弃变身时记住棋盘签名：棋盘演化前不再触发（防死循环）。

        单帧观察（poll/confirm 阶段）无棋盘快照，退回最近一次多帧识别
        记录的签名（触发时的棋盘）。
        """
        signature = (
            self._board_signature(observation.board)
            if observation.board is not None
            else self._last_board_signature
        )
        if signature is not None:
            self._transform_skip_signature = signature

    def _match_in_transform_window(
        self,
        match_pos: tuple[int, int],
        hero_pos: tuple[int, int],
        window_ctx: WindowContext | None,
    ) -> bool:
        """按钮命中位置是否落在目标英雄的搜索窗口内（设计文档 §4.3）。

        窗口相对英雄格中心：±search_dx_ratio 宽、向上 search_up_ratio 高、
        向下 search_down_ratio 高（比例参数在 tasks.yaml 的
        locators.transform_button，由棋盘格尺寸推导）。窗口外命中（如误点了
        别的英雄弹出的按钮）不算本次目标。window_ctx 缺失时不拦截。
        """
        if window_ctx is None:
            return True
        cfg = self._transform_locator or {}
        cw, ch = window_ctx.client_size
        dx_tol = float(cfg.get("search_dx_ratio", 0.11)) * cw
        up_tol = float(cfg.get("search_up_ratio", 0.093)) * ch
        down_tol = float(cfg.get("search_down_ratio", 0.015)) * ch
        dx = match_pos[0] - hero_pos[0]
        dy = match_pos[1] - hero_pos[1]
        return abs(dx) <= dx_tol and -up_tol <= dy <= down_tol

    def _clear_transform_state(self) -> None:
        """清空变身阶段状态（不重置本局已用次数计数器）。"""
        self._transform_stage = None
        self._transform_target = None
        self._transform_button_pos = None
        self._transform_reclicks = 0
        self._transform_verify_attempts = 0
        self._transform_poll_deadline = 0.0

```

- [ ] **Step 10: 插入点 7——on_action_verified 的 ready_match 重置**

找到（约 2711-2717 行，12 空格缩进使其区别于 _reset_for_home_reentry 中的同名赋值）：

```python
            self._main_skill_selections = 0
            self._main_skill_selections_total = 0
            self._skill_refresh_used_count = 0
            self._skill_refresh_button_waits = 0
            self._topup_active = False
            self._topup_done = False
            self._summon_retry_at = None
```

在 `self._summon_retry_at = None` 之后插入（12 空格缩进）：

```python
            self._transform_used_count = 0
            self._transform_skip_signature = None
            self._clear_transform_state()
```

- [ ] **Step 11: 插入点 8——_reset_for_home_reentry 重置**

找到（约 2902-2907 行，8 空格缩进）：

```python
        self._main_skill_selections = 0
        self._main_skill_selections_total = 0
        self._skill_refresh_used_count = 0
        self._skill_refresh_button_waits = 0
        self._topup_active = False
        self._topup_done = False
```

在 `self._topup_done = False` 之后插入（8 空格缩进）：

```python
        self._transform_used_count = 0
        self._transform_skip_signature = None
        self._clear_transform_state()
```

- [ ] **Step 12: 运行新增测试**

Run: `python -m pytest tests/test_coop_task.py::TestTransformSequence -v`
Expected: 全部 PASS。若个别断言失败，先对照本 Task 的代码逐字核对插入位置与缩进，不要自行改断言迁就实现。

- [ ] **Step 13: 回归旧测试（行为零变化承诺）**

Run: `python -m pytest tests/test_coop_task.py -q`
Expected: 全部 PASS（旧测试未接 locator，变身分支被跳过）。

- [ ] **Step 14: Commit**

```bash
git add src/wlxq_bot/tasks/coop.py tests/test_coop_task.py
git commit -m "feat(task): 培养循环接入变身阶段机——点英雄/轮询按钮/点按钮验证消失/棋盘验证，每局2次可连用"
```

---

### Task 5: runner 接线与模板缺失降级

**Files:**
- Modify: `src/wlxq_bot/runner.py`（run() 主路径的 CoopTask 构造，约 363-390 行）

**Interfaces:**
- Consumes: `CoopTask(transform_button_locator=...)`（Task 4）、`run_config.transform_enabled`（Task 1）、tasks.yaml `locators.transform_button`（Task 2）。
- Produces: 生产路径的变身功能装配；模板缺失时传 None 降级（任务层整分支禁用 + WARNING）。

- [ ] **Step 1: 插入降级检查与传参**

打开 `src/wlxq_bot/runner.py`，找到主 run 路径的任务构造（约 363-368 行；注意 `exec` 路径 550 行还有另一个 `task = CoopTask(`，其第一参数是 `TaskContext(`，不要改那个）：

```python
        task = CoopTask(
            ctx,
            run_config,
            CoopRole.HELPER,
```

在这段之前插入（8 空格缩进）：

```python
        # 变身 locator 接线（设计文档 §4.5）：模板/locator 缺失时降级禁用
        # （WARNING），不阻断整局 run——变身是增益能力
        transform_button_locator = self.tasks_config.locators.get("transform_button")
        if run_config.transform_enabled and transform_button_locator is not None:
            transform_template = str(transform_button_locator.get("template", ""))
            if not pack.resolve_template(transform_template).is_file():
                logger.warning(
                    "变身按钮模板缺失（%s），本局禁用变身功能，不影响其他流程",
                    transform_template,
                )
                transform_button_locator = None

```

再把同一个 CoopTask 调用的尾部：

```python
            entry_max_reclicks=int(
                self.tasks_config.mode_select_recognition.get("entry_max_reclicks", 3)
            ),
        )
        task.ctx.current_state = initial_state
```

改为：

```python
            entry_max_reclicks=int(
                self.tasks_config.mode_select_recognition.get("entry_max_reclicks", 3)
            ),
            transform_button_locator=transform_button_locator,
        )
        task.ctx.current_state = initial_state
```

- [ ] **Step 2: 语法与导入检查**

Run: `python -c "from wlxq_bot.runner import CoopRunner; print('ok')"`
Expected: 输出 `ok`（`logger` 在 runner.py 已有，`pack` 变量在该作用域已存在——它构造 perception 时使用；若 NameError 则说明改到了 exec 路径的构造点，回退重做 Step 1）。

- [ ] **Step 3: 回归**

Run: `python -m pytest tests/test_coop_task.py tests/test_exec_cli.py -q`
Expected: 全部 PASS（runner 变更无专门单测，靠任务层与既有装配测试兜底）。

- [ ] **Step 4: Commit**

```bash
git add src/wlxq_bot/runner.py
git commit -m "feat(runner): 变身 locator 接线——模板缺失降级禁用并 WARNING，不阻断对局"
```

---

### Task 6: 文档同步

**Files:**
- Modify: `docs/game-rules.md`（新增「英雄变身」小节，插在「合成结果随机规则」小节之后、「自动化判断要求」之前）
- Modify: `docs/automation-flow.md`（「主 C 培养循环」小节补变身分支，插在「强袭四阶段策略」段落之后）
- Modify: `docs/todo.md`（追加第 1 排按钮渲染待实机观察项）

**Interfaces:**
- Consumes: 设计文档 §2 的事实表（日期与数值以它为准）。
- Produces: 与实现一致的规则/流程文档。

- [ ] **Step 1: game-rules.md 加规则小节**

在 `docs/game-rules.md` 中找到小节标题：

```markdown
## 自动化判断要求
```

在其**之前**插入：

```markdown
## 英雄变身

英雄有变身能力（实机确认 2026-09-13）。操作流程：点击棋盘上的英雄 → 其头顶悬浮「变身术」按钮 → 点击按钮 → 按钮消失、英雄开始变身 → 等待动画（实测约 4 秒稳定）。

已确认规则：

- 变身结果为本局携带 4 种英雄中**除自身外其他 3 种**的随机之一：类型**必变**，星级不变。例：3 星雪姬变身后是天使/强袭/死骑之一（仍 3 星）。
- 每局有 2 次变身机会，允许对同一位置连续使用。
- 点击英雄后，只要本局还有变身次数，按钮约 1 秒内弹出；按钮弹出后若没有点击其他位置的动作则不会消失；点击变身按钮后按钮先消失、英雄再开始变身。
- 次数耗尽后，点击英雄不再弹出变身按钮（自动化内存计数失步时的自愈依据）。
- 按钮位置是被点英雄所在格的几何函数（实机量测 927×1727 客户区）：水平与本格中心对齐（偏差约 4.5px），垂直悬于本格顶边上方（按钮底边略压顶边约 6px）；按钮模板 `buttons/bian_shen.png` 实机 5/5 截图命中 0.984~0.9998。
- 第 1 排（最上排）英雄的按钮渲染未实机采样（按几何推导悬于棋盘上方中间过道带），失败路径保守兜底，见项目 TODO。

自动化策略（2026-09-13 用户定案）：无达标主 C 且棋盘存在 3 星非主 C 英雄时变身碰主 C（每次 1/3、连用两次 5/9 概率）；第一次未出主 C 立即对同一位置用第二次；主 C 达标后的回补阶段不用变身。合作与保卫宠物两模式都启用。

```

- [ ] **Step 2: automation-flow.md 补流程**

在 `docs/automation-flow.md` 的「主 C 培养循环」小节内，找到段落开头：

```markdown
强袭四阶段策略（档案 `topup_*` 启用，2026-08-16 实机规则；2026-09-07 修订）：
```

在该段落**之前**插入：

```markdown
变身分支（2026-09-13 用户策略，设计文档 `docs/superpowers/specs/2026-09-13-hero-transform-design.md`）：击杀弹窗优先关闭之后、批次召唤/合成之前，若「无达标主 C + 本局变身次数未用完（默认 2 次）+ 棋盘存在 3 星及以上非主 C」（`CultivationStrategy.select_transform`）则进入变身阶段机：点击目标英雄格 → 每 `transform_button_poll_interval_seconds`（0.3 秒）轮询 `transform_button` 专项观察模式匹配变身按钮（全图退化 + 任务层按 `search_*_ratio` 窗口校验命中在目标格上方）→ 未弹出在 `transform_button_reclick_max` 预算内重点击英雄（只对未弹出情形重点击，避免关掉已弹出按钮）→ 命中即点击按钮 → 0.3 秒后复查按钮消失（原位仍在才重点按钮）→ 消失即扣减本局次数并等待 `transform_settle_seconds`（4 秒，实测动画稳定时长）→ 多帧棋盘重识别验证「同位置类型改变且星级不变」。验证通过回到决策循环：变出主 C 即正常达标进选技能；仍是 3 星非主 C 且还有次数则对同一位置立即再用一次（连用）。任何失败（按钮超时/点击未生效/验证耗尽）都只放弃本次、记日志、回常规决策，不结束任务；放弃时记录棋盘签名（放弃记忆），棋盘演化（召唤/合成改变英雄集合）前不再触发变身，防止「放弃→立刻再触发」死循环；轮询超时同时是计数失步的自愈兜底（游戏侧次数已尽则按钮不再弹出）。每局计数在点准备进局/被踢重进/结算重开时重置（与技能刷新同点）；程序中途重启丢计数属已知边界。模板缺失时 runner 启动降级禁用（WARNING）。

```

- [ ] **Step 3: todo.md 加观察项**

打开 `docs/todo.md`，在文件中与实机验证相关的清单末尾追加一行（若无明确清单结构，则在文件末尾新增小节）：

```markdown
- [ ] 变身按钮第 1 排（最上排）渲染待实机观察：按格子几何推导按钮悬于棋盘上方中间过道带，匹配窗口已覆盖；若实机发现第 1 排不弹按钮，失败路径会自动放弃变身（不影响对局），确认后回填 `docs/game-rules.md`。
```

- [ ] **Step 4: 链接与一致性检查**

Run: `python - <<'EOF'
import re, pathlib
for name in ("docs/game-rules.md", "docs/automation-flow.md"):
    text = pathlib.Path(name).read_text(encoding="utf-8")
    assert "变身" in text, name
    for m in re.findall(r"]\(([^)]+\.md[^)]*)\)", text):
        target = pathlib.Path("docs") / m.split("#")[0]
        assert target.exists(), (name, m)
print("docs ok")
EOF`
Expected: 输出 `docs ok`。

- [ ] **Step 5: Commit**

```bash
git add docs/game-rules.md docs/automation-flow.md docs/todo.md
git commit -m "docs: 变身规则与培养循环流程同步——每局2次/类型必变星级不变/轮询与验证预算"
```

---

### Task 7: 全量验证

**Files:** 无新增改动（只验证；若验证暴露问题，修复后重跑本 Task）。

- [ ] **Step 1: 全量测试**

Run: `python -m pytest -q`
Expected: 全部 PASS，无 skip 异常增长。

- [ ] **Step 2: Lint**

Run: `python -m ruff check src tests`
Expected: 无错误（仓库 ruff 已配置；若有报错按提示修复——通常是导入排序或行宽——修复属于本 Task 范围）。

- [ ] **Step 3: 启动装配冒烟（不进对局）**

Run: `python -m wlxq_bot --help`
Expected: 正常输出帮助（CLI 未受影响）。

- [ ] **Step 4: 核对提交历史**

Run: `git log --oneline -8 && git status --short`
Expected: 6 个新提交（Task 1-6 各一），工作区干净。

---

## 执行注意事项（写给执行者）

1. 所有 Edit 的 old_string 锚文本已在步骤中逐字给出；若匹配失败，先 `grep -n` 确认实际文本再重试，**不要凭记忆改锚文本周边的其它内容**。
2. 插入代码的缩进必须与步骤代码块一致（coop.py 方法体 8 空格、方法定义 4 空格；on_action_verified 分支内 12 空格）。
3. 测试失败时优先怀疑插入位置/缩进/漏插入，逐字对照本计划；**禁止修改测试断言来迁就错误实现**（除 Task 4 Step 12 明确列出的对照流程）。
4. 不得提交 `outputs/`、`screenshots/`、`logs/` 等生成物；commit 只加步骤中列出的文件。
5. 实机验证（真实游戏窗口跑一局）不在本计划范围内——完成后由用户实机首跑，观察点已记录在 `docs/todo.md`。
