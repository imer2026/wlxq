# 保卫宠物模式支持 —— 设计文档

日期：2026-09-06
状态：待用户评审
前置事实：[游戏规则知识库 - 保卫宠物模式](../../game-rules.md)（本设计全部实机事实的出处，含标定日期）

## 1. 背景与目标

在现有合作模式自动化之外，新增「保卫宠物」模式支持：以帮打方身份抢加入别人的保卫宠物对局，自动完成「招募 → 抢 → 准备 → 培养 → 选技能 → 结算 → 再抢」循环。

- 默认行为完全不变：不带 `--mode` 参数时走现有合作流程，已有测试与实机行为零变化。
- 两种模式共享状态机骨架、识别基础设施、动作执行链与安全约束。
- 采用**方案 A：模式档案注入**——配置承载素材/名字差异，策略对象承载决策差异，不开任务子类（决策依据见 §3）。

## 2. 已确认实机事实（设计输入，均已核实）

| 事实 | 依据 |
|---|---|
| 入口：聊天 → 招募；招募页有「合作模式 / 保卫宠物」两页签，默认停在合作模式 | 用户实机确认 2026-09-05/06 |
| **两模式统一**：每次进入招募都先点一次自己的模式页签再继续（合作点 `recruit_coop_tab`、宠物点 `recruit_pet_tab`，坐标已标定；幂等设计，2026-09-06 用户定案） | 用户定案 |
| 页签坐标已标定（`recruit_coop_tab` / `recruit_pet_tab`，tasks.yaml） | 2026-09-05 实机 |
| 宠物难度仅 4 档（地狱/困难/中等/普通），一屏放完、无滚动；目标：普通+中等 | 用户确认 |
| 难度行文字「保卫宠物-普通/中等/困难/地狱」OCR 四行全读对（0.76~0.85） | 2026-09-05 离线验证 |
| 难度列表区 ROI 已标定（`pet_difficulty_list`） | 2026-09-05 实机 |
| 抢保卫宠物 = 连点同一 `join_coop` 位置 | 用户确认 |
| 抢到后点准备；准备按钮专属模板 `bao_wei_chong_wu/zhun_bei.png` | 已采集 |
| 组队大厅有【退队】按钮，位置与合作明显不同；专属模板 `bao_wei_chong_wu/tui_dui.png`；实测点 `pet_leave_team` 在合作 ROI 之外 | 2026-09-05 |
| 退队触发逻辑与合作一致（超 `leave_team_after_seconds` 房主未开始即退）；退队后回首页 | 用户确认 2026-09-06 |
| **棋盘位置与合作一致**：排5 瓷砖竖边界差 1~3px、底轨差 3px、12 格网格落位一致 | 2026-09-06 双截图对比 |
| 底部操作栏同位（左下强化按钮 = `select_skill` 位置；右下「召唤」= `add_hero` 位置）；技能页标志仍是页顶「请选择」字样 | 2026-09-06 截图 |
| 结算：不点赞、不领宝箱，只点返回；返回模板 `bao_wei_chong_wu/fan_hui.png` | 用户确认 |
| 结算点返回 → 落在**宠物挑战页面**；聊天按钮 `pet_chat` (0.9234, 0.5541)（与 `coop_chat` 纵向差近 200px，不可互用）；点聊天后按标准流程补点一次招募（幂等，用户定案） | 2026-09-06 |
| 技能选择不限次数（`skill_selection_cap: 0`，代码已支持 0=不限制）；数量回补循环照常 | 用户确认 2026-09-06 |
| 合作保持 9 次技能上限不变 | 设计决策 2026-09-06 |

## 3. 方案比选（已定：方案 A）

- **方案 A：模式档案注入（采用）**。差异本质是「配置 + 少量动作分派 + 策略对象」，状态机与入口子流程只有一份。招募入口是改最勤的代码（8/17、8/19、8/21、8/25 各调过一轮），单份维护是硬要求。
- 方案 B：`PetTask(CoopTask)` 子类。需要整段覆写招募入口子状态机——把最复杂、改最勤的代码复制一份，双份维护风险真实存在。否决。
- 策略差异的承载：`CultivationStrategy` 已存在且由 `CoopTask.__init__` 单点构造（`tasks/coop.py:100`），按模式换对象即可；状态机只向策略提问（`main_c_ready` / `should_topup` / `select_merge` / `skill_selection_cap`），不关心回答者是谁。

### 差异逐条落层表（新增差异先归类再动手）

| # | 差异 | 承载层 | 承载方式 |
|---|---|---|---|
| 1 | 入口页签（两模式统一：进招募先点自己的模式页签） | 配置 | `entry.mode_tab_hotspot` |
| 2 | 难度识别：模板+滚动 vs OCR 免滚动 | 配置+分派 | `difficulty.kind: template / ocr` |
| 3 | 难度弹窗开关确认：标题图 vs「选择难度」收起栏文字 OCR 反向判定 | 配置+分派 | `pet_difficulty_recognition.bar_roi/bar_text` |
| 4 | 准备/返回/退队模板与识别区 | 配置 | locator 名按模式解析 |
| 5 | 结算跳过点赞/宝箱 | 配置 | `settlement.like_chest: false` |
| 6 | 结算返回后的聊天入口（兼退队非首页兜底） | 配置 | `mode_chat_hotspot` |
| 7 | 技能选择不封顶 | 配置 | `main_c_overrides.skill_selection_cap: 0` |
| 8 | 退队后回首页的重抢路径 | 两模式一致 | 复用现有首页探测逻辑 |
| 9 | 棋盘/培养/合成/金币门控/击杀弹窗 | 无差异 | 全部共用 |
| 10 | 未来分阶段打法（如 3星→补4个→无限选） | 策略对象 | `build_strategy` 工厂 + 阶段策略类（**本期只留扩展点，不实现**） |

## 4. 架构设计

### 4.1 配置结构

`configs/default.yaml` 新增顶层 `mode_profiles`，`run.mode` 选择本次模式（CLI `--mode` 覆盖，默认 `coop`）：

```yaml
run:
  mode: coop            # coop | pet；CLI run --mode pet 覆盖

mode_profiles:
  coop:
    display_name: 合作模式
    mode_chat_hotspot: coop_chat      # 模式页面聊天按钮：结算返回后 & 退队非首页兜底
    entry:
      mode_tab_hotspot: recruit_coop_tab  # 进招募先点「合作模式」页签（2026-09-06 定案，两模式统一）
      difficulty:
        kind: template                # 现状：文字模板匹配 + 滚动
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
      mode_tab_hotspot: recruit_pet_tab   # 每次进招募都点（幂等，2026-09-06 定案）
      difficulty:
        kind: ocr
        targets: [普通, 中等]             # 行文字「保卫宠物-普通」按子串匹配
    match:
      ready_button: pet_ready_button
      leave_team: pet_leave_team
    settlement:
      like_chest: false                   # 不点赞、不领宝箱
      return_button: pet_return_button
    main_c_overrides:
      assault:
        skill_selection_cap: 0
    strategy: default
```

要点：

- 档案里存的都是 **tasks.yaml 里的名字**（hotspot/locator/ROI 名），运行时解析——素材差异不产生代码分支。
- `main_c_overrides` 在启动解析主C档案后应用（深合并、只允许覆盖 `MainCProfile` 已有字段），宠物模式借此给强袭覆盖 `skill_selection_cap: 0`；其余策略字段（回补、合成回避、技能优先级）继承。
- `strategy: default` = 现有 `CultivationStrategy`。`build_strategy(mode_profile, ...)` 工厂按名字构造；本期只有 `default`，阶段化策略是将来扩展点（见 §7）。

`configs/tasks.yaml` 新增（名字以本节为准，全部进启动检查；locator 条目沿用现有模式：`strategy: template` + `action_point: match_center`，ROI 未标定时退化全图识别）：

```yaml
locators:
  pet_ready_button:      # buttons/bao_wei_chong_wu/zhun_bei.png，阈值 0.82 起步待实机回调
  pet_return_button:     # buttons/bao_wei_chong_wu/fan_hui.png，同上
  pet_leave_team:        # buttons/bao_wei_chong_wu/tui_dui.png；ROI 待标定（参考 hotspot pet_leave_team 实测点），标定前退化全图识别

pet_difficulty_recognition:
  candidate_roi: pet_difficulty_list
  targets 由 mode_profiles.pet.entry.difficulty.targets 下发
  min_confidence: 0.5    # OCR 文字置信度，语义与模板匹配置信度不同，不混用
```

已有素材（本设计之前已入库）：hotspots `recruit_coop_tab` / `recruit_pet_tab` / `pet_leave_team` / `pet_chat`；ROI `pet_difficulty_list`；模板包 `3000x2000/buttons/bao_wei_chong_wu/{zhun_bei,fan_hui,tui_dui}.png`。

### 4.2 模块改动清单

| 模块 | 改动 |
|---|---|
| `config.py` | 新增 `ModeProfile` / `EntryCfg` / `DifficultyCfg` / `MatchCfg` / `SettlementCfg` 模型；`DefaultConfig.mode_profiles`；`RunConfig.mode` + 校验（必须是 mode_profiles 键）；`TasksConfig.pet_difficulty_recognition` |
| `cli.py` | `run --mode`；`exec select-difficulty --mode`（宠物模式走 OCR 路径，供弹窗打开状态下离线验证） |
| `runner.py` | 解析 mode → ModeProfile（应用 `main_c_overrides` 到主C档案）；按模式装配 CoopTask（注入 ModeProfile）；启动缺口检查按模式取必备 hotspot/locator/模板清单 |
| `tasks/coop.py` | 三处分派：① 招募入口——进招募后先点 `mode_tab_hotspot`（两模式都配置了页签；字段保留可空以备将来默认页签模式），难度选择按 `difficulty.kind` 走模板路径（现状）或 OCR 路径（新）；② 弹窗开关确认——template=标题图（现状），ocr=列表行文字存在性（同一套轮询/连续命中/重试预算参数）；③ 结算——`like_chest=false` 时跳过点赞与宝箱（含 `reward_claim_return_delay` 等待），点返回后用 `mode_chat_hotspot` 进下一局；退队后「非首页」兜底路径同样用 `mode_chat_hotspot`。类名不改（CoopTask 是共享的组队对局任务引擎，改名纯 churn） |
| `perception/pet_difficulty.py`（新文件，避免 coop.py 继续膨胀） | `DifficultyRow`（text / confidence / action_point）与 `read_difficulty_rows(ctx, frame)`：ROI 裁剪 → OCR → 行文字与外接框；`match_targets(rows, targets)`：子串匹配（目标词是行文字的子串，如「普通」⊂「保卫宠物-普通」），返回可点击目标。只负责识别，坐标换算归 Locator 体系（复用 ratio→pixel） |
| `tasks/strategy.py` | `build_strategy(mode_profile, main_c, run_config, main_c_profile)` 工厂；本期只返回 `CultivationStrategy(main_c, run_config, overridden_profile)`。接口契约注释：策略对象必须回答的四个问题（§3），阶段化策略按此接入 |
| 模板包 | 零新增（三张模板已采集入库）；其余分辨率模板包照旧为空，沿用现有模板包选择策略 |

不改动：状态机 State/Transition、识别门控表（`_INTERFACE_FLAGS_BY_STATE`）、抢合作协调器、棋盘识别、技能 OCR 选卡、金币门控、DebugRecorder、英雄分类器。

### 4.3 关键流程（宠物模式）

```text
首局：首页确认 → 首页聊天 → 招募 → 点「保卫宠物」页签 → 打开难度弹窗
  → OCR 读列表区 → 子串匹配「普通」「中等」→ 点击行文字中心（开环，同合作勾选哲学）
  → 关闭弹窗（OCR 行连续不可见确认）→ 双线程连点 join_coop
  → 识别 pet_ready_button → 点击 → 等待开局
对局：与合作完全一致（棋盘/培养/选技能/金币门控/击杀弹窗）；
  差异仅 skill_selection_cap=0（一直选到结束）
退队：等待超 20s 且 pet_leave_team 可见 → 点击 → 按钮消失验证 → 回首页 → 首页聊天重进
结算：识别 pet_return_button → 点击一次 → 消失验证 → 宠物挑战页面
  → 点 pet_chat → 补点一次招募（幂等）→ 点页签 → 开/关难度弹窗刷新邀请 → 再抢
```

弹窗开关确认预算完全复用现有参数（`difficulty_open_settle_seconds` / `difficulty_poll_interval_seconds` / `difficulty_open_confirm_hits` / `difficulty_open_fail_misses` / `difficulty_open_max_reclicks` / `difficulty_close_confirm_frames` / `difficulty_close_max_attempts`），配置 `bar_roi`/`bar_text`（保卫宠物=「选择难度」收起栏文字 OCR）时：栏文字可读 = 已收起，栏被展开面板盖住 = 打开；未配置时按标题图模板正向判定。实机修正（2026-09-06）：最初用 OCR 行存在性判开关，但弹窗关闭后邀请卡标题与难度行同区域，行存在性无法区分开关，实机首跑即踩中，改为收起栏锚点。

### 4.4 错误处理与保守停止

- OCR 读不到目标难度行：不开环盲点。视为「目标不可见」，只等待重读；持续不可见达 `difficulty_open_fail_misses` 预算后直接保守停止整个 run（不重点打开按钮、不走 `error_restart`）。
- OCR 引擎异常/超时：`perception.observe` 不做捕获，异常直接终止任务（比识别失败更保守），不静默跳过。
- `pet_difficulty_list` ROI 缺失或退化：启动检查报错（OCR 必须限域，禁全图 OCR 进主循环）。
- 首局页签点击后画面无变化：难度弹窗打开确认的重试预算会兜住（重点开按钮→保守停止），不做页签状态识别。
- 其余未知界面/验证失败/窗口失效：全部沿用现有保守停止与 `error_restart` 机制。

## 5. 测试计划

1. **单元（纯决策）**：ModeProfile 加载与校验（未知 mode 报错、overrides 深合并、非法字段拒绝）；`build_strategy` 解析与 override 生效（assault cap 9→0）；结算分派（like_chest=false 时不产生 like/claim 动作）；宠物难度目标匹配（子串命中/未命中/多行命中/低置信度拒绝）；入口步骤序列（页签步骤存在性与顺序）。
2. **OCR 离线回归**：以 `bao_wei_chong_wu/nan_du.png` 为固定素材，断言四行文字与框；用 `exec select-difficulty --mode pet` 在实机弹窗打开状态下人工执行验证（不进主循环）。
3. **Fake Input 端到端**：宠物模式单局（Fake 视觉喂帧：首页→招募→难度行→准备→大厅→对局→返回→宠物挑战页），断言动作序列含页签点击、无点赞/宝箱点击、聊天用 `pet_chat`。
4. **合作回归**：现有测试全绿；不带 --mode 的动作序列与现状的差异**仅剩一处**——进招募后多一次「合作模式」页签点击（本次需求引入），结算仍含点赞宝箱，其余不变。
5. **首跑人工验证清单**（实机，跑通前不声称完整可用）：
   - [ ] 两模式的页签重复点击都不清难度勾选（若清了，勾选步骤自愈重勾，记录现象；合作模式为本次新增步骤，重点观察）
   - [ ] `open_difficulty_dialog` 按钮位置在宠物页签下有效（当前按同一位置假设）
   - [ ] 宠物开/关难度弹窗是否刷新邀请（按合作同款策略先做）
   - [ ] `pet_ready_button` / `pet_return_button` / `pet_leave_team` 三模板阈值与 ROI 标定（先全图识别起步，日志记录置信度后回调）
   - [ ] 结算是否存在双倍奖励误触风险（现有结算期双倍弹窗守卫保持启用）
   - [ ] 对局内界面标志（召唤/请选择/击杀弹窗/返回）在宠物局全部按预期命中（跑通一局即验证）

## 6. 非目标（本期不做）

- 合作模式除「进招募先点模式页签」外零改动（技能上限保持 9；是否放开是独立的一行配置决策，另行实机验证）。页签步骤实现时同步更新 `docs/automation-flow.md` 的进入招募小流程与 `docs/game-rules.md`。
- 合作难度 OCR 迁移（本期只建 `pet_difficulty.py` 复用点，迁移另立任务）。
- 分阶段培养策略（`phased` 策略类）——仅保留 `build_strategy` 扩展点与接口契约。
- 发起方（initiator）角色、断线重连、多模式混跑（一局合作一局宠物交替）。

## 7. 未来扩展点

- **阶段化策略**：`strategy: phased_assault` → 阶段序列（如 3星→补4个→无限选）作为策略对象内部状态推进，接口仍是 §3 的四个问题；已与用户对齐两个示例（1星开局即选、3星后再补数量），需求实装时仅需新增策略类并在 mode_profiles 里点名。
- **合作难度 OCR 迁移**：`pet_difficulty.py` 的行识别直接复用，替换 `cai_hong_N` 模板（项目 TODO 既有规划）。
