# 幻天使（angel_phantom）分离识别与通配合并 —— 设计文档

日期：2026-09-13
状态：待用户评审
前置事实：[游戏规则知识库](../../game-rules.md)（本设计实机事实确认后同步写入）；数据普查结论见 §2.1

## 1. 背景与目标

天使在棋盘上有两种形态：正常天使（金色调）与幻天使（紫色调，赖子型英雄）。当前英雄格分类器把两者都识别为 `angel`，导致合并决策无法利用幻天使的通配能力。

目标：

- 分类器把幻天使识别为独立英雄 `angel_phantom`（仅 star1~3）。
- 合并候选生成支持通配规则：幻天使可与任意同星级英雄配对（含另一幻天使），合成产物随机（见 §2 事实 3）。
- 合成优先级：同类项对优先，幻天使对兜底（用户策略 2026-09-13）。
- **不重新采集数据**：现有 `assault/3000x2000/helper` 数据集中两种形态本就混标在 `angel` 目录下，按色彩特征重分拣即可。

非主 C 阵容、召唤、金币门控、技能选择等其余链路零改动。

## 2. 已确认实机事实（设计输入）

| # | 事实 | 依据 |
|---|---|---|
| 1 | 召唤、合并均可能出现幻天使；4 星天使必为正常天使；幻天使只有 1~3 星 | 用户实机确认 2026-09-13 |
| 2 | 幻天使可与任意同星级英雄合并，也可与另一幻天使合并 | 用户实机确认 2026-09-13 |
| 3 | **合并产物随机**：可能是任何英雄，也可能仍是幻天使；不保证是搭档英雄的升星形态 | 用户实机确认 2026-09-13 |
| 4 | 合成优先级：同类项先合（如两个雪姬），幻天使留作最难凑的兜底 | 用户策略 2026-09-13 |
| 5 | 3 星幻天使参与合成同样触发 4 星赠送技能页 | 用户实机确认 2026-09-13 |
| 6 | 天使是辅助英雄，永远不会当主 C | 用户实机确认 2026-09-13 |
| 7 | 视觉差异为主色调：正常天使金色（HSV 环向 hue≈24），幻天使紫色（hue≈142） | 2026-09-13 参考图 OpenCV 普查 |

### 2.1 现有数据普查（2026-09-13，色彩特征扫描）

对 `datasets/hero_classifier/assault/3000x2000/helper` 的 `labeled/angel` 全量扫描（饱和像素 s>60 且 v>60 的环向 hue）：

- hue 直方图双峰：金色簇 10~40 / 紫红簇 130~179，与两形态一一对应。
- 紫色调样本（疑似幻天使）：train star1≈70、star2≈44、star3≈130；validation≈20；test star3≈62。
- **star4 零紫色样本**（train 2 张全金）——与事实 1「4 星必为正常天使」交叉印证。
- star1 另有 38 张 hue=0 的红色调样本（疑似受击/伤害数字干扰），归属待人工复核。
- 参考样本：`train/labeled/angel/star1/202608122001_frame000018_5B.png`（正常）、`…_frame000108_6B.png`（幻）。

## 3. 方案比选（已定：方案 A）

- **方案 A：独立类别重训（采用）**。新建 `angel_phantom/star1~3` 标注目录，重训现有 MobileNetV3-Small ONNX；合并逻辑加通配规则。类别体系天然支持（类别名=目录名，`_HERO_CLASS_RE` 允许 `[a-z0-9_]+`，识别运行时零改动）；金/紫色差对 CNN 极易学习；沿用现有置信度/margin 拒绝与多帧投票机制。
- 方案 B：不重训，在 `angel` 识别结果上叠加紫色调判定。省一次训练，但色彩硬规则受特效/伤害数字/选中高亮干扰大，绕过既有拒绝机制，且把识别逻辑散落两处。否决。
- 方案 C：幻天使走模板匹配。回退旧识别路径，双链路维护。否决。

## 4. 设计

### 4.1 数据层：重分拣（一次性，无新采集）

**分拣特征**：`purple_frac` = 饱和像素（s>60, v>60，少于 30 像素视为无效样本）中 hue∈[120,180) 的占比。

**决策规则**（对 train/validation/test 三个 split 的 `angel/star1~star3` 逐张判定；star4 不动）：

| purple_frac | 动作 |
|---|---|
| ≥ 0.50 | 移入 `angel_phantom/starN` |
| ≤ 0.15 | 留在 `angel` |
| 0.15 ~ 0.50 | 留在 `angel`，列入人工复核清单 |

**执行门槛**：先 dry-run 输出各 split/star 的 purple_frac 分布与两参考图数值；确认分界落在双峰谷底才执行迁移，分布不干净则停下向用户报告。

**审计与可回滚**：所有移动记 CSV（路径、purple_frac、决策）到 `outputs/hero_classifier/relabel_angel_phantom_<日期>.csv`；复核清单拼 montage 存 `outputs/hero_classifier/relabel_review_<日期>/`，供用户事后人工确认（复核结论可驱动二轮迁移，非本期必需）。`rounds/` 原始整局截图不受影响。

迁移后执行 `wlxq-bot hero-classifier sync-labels` 重建 `dataset_manifest.csv`。

### 4.2 模型层：重训与部署

- 训练：`wlxq-bot hero-classifier train datasets/hero_classifier/assault/3000x2000/helper -o outputs/hero_classifier/assault-helper-v5`（默认 epochs 20 / batch 64 / input 96 / 预训练权重；本机 CPU，后台执行）。
- 新类别 `angel_phantom_star1~3` 由目录自动生成；trainer 采样权重、suggest-labels 主动学习闭环对新类别自动生效，trainer 零改动。
- 评估：`wlxq-bot hero-classifier evaluate … --model <v5> --split test`，验收标准（对比 `outputs/hero_classifier/evaluation` 既有报告）：
  1. 混淆矩阵中 `angel_*` 与 `angel_phantom_*` 互混样本 ≤ 3 张（test star3 约 62 张幻样本），且不集中在同一星级；
  2. 已接受样本准确率（accuracy_accepted）不低于现模型 − 0.5 个百分点。
- 部署：达标后将 `main_c_profiles.assault.hero_classifier_model` 指向 v5；回滚 = 改回一行配置。**评估不达标不切换。**

### 4.3 配置层（configs/default.yaml + config.py）

```yaml
run:
  # 新增：通配合并英雄（赖子）。可与任意同星级英雄合并；同类项对优先，通配对兜底。
  wildcard_merge_heroes: [angel_phantom]

main_c_profiles:
  # 新增档案：仅为 lineup 校验与裁剪目录预建服务；永不主 C（事实 6）。
  angel_phantom:
    display_name: 幻天使
    hero_template_dir: heroes/angel_phantom   # 模板未采集（见 §7 TODO），无模板校验，不影响运行
    hero_classifier_model: ""                 # 不作为主 C 加载，留空
    skill_icon_templates: []

hero_classifier:
  # angel_phantom 必须在列，否则 _restrict_prediction 会把幻天使判为 hero_not_in_lineup → unknown
  lineup_others: [angel, angel_phantom, snow, death_knight]
```

- `RunConfig` 新增 `wildcard_merge_heroes: list[str]`（默认空），复用 lineup_others 的小写标识符校验。
- `runner.py` 两处 `allowed_heroes={main_c, *lineup_others}` 构造自动包含 angel_phantom；watch 路径阵容覆盖校验（`runner.py:1186-1189`）按**英雄类型级**比对（`available_heroes`），幻天使只有 star1~3 类不阻塞（已核实代码）。

### 4.4 领域层：合并候选生成（models.py）

`MergeCandidate` 新增 `is_wildcard: bool = False`（对中含通配英雄）。

`BoardSnapshot.find_merge_candidates(main_c, wildcard_heroes=frozenset())` 改为两阶段，**默认空集时行为与现状完全一致**（向后兼容，既有调用与测试不受影响）：

- **阶段 1（同类项）**：现有精确匹配循环原样保留，跳过 `hero_type ∈ wildcard_heroes` 的英雄——非通配英雄之间同类型同星配对。
- **阶段 2（通配兜底）**：按棋盘顺序为每个未配对的通配英雄找第一个未使用的同星搭档——**优先非通配英雄**，无合适非通配搭档才与另一通配英雄配对（事实 2）；标记 `is_wildcard=True`。
- `is_main_c` 语义扩展为「对中任一英雄是主 C」：幻天使+强袭也算主 C 对，受 `avoid_main_c_merge` 保护——合并产物随机（事实 3），不拿主 C 数量去赌。
- 每个英雄最多参与一对（沿用现有 `used` 集合）；稳定排序（`is_main_c`）保持「非主 C 对在前；同为非主 C 时同类项对在通配对前」。

### 4.5 策略层（strategy.py / tasks/coop.py）

- `CultivationStrategy.select_merge`：候选池内显式排序**非通配低星对 → 通配低星对**（不再隐式依赖候选构造顺序）；3 星规避逻辑原样覆盖通配对（棋盘有空位时召唤避开、占满才被迫合、随后走既有赠送技能页确认流程）——事实 5 自然满足。
- `CultivationStrategy.select_merge_batch`：**排除 `is_wildcard` 对**。批次只做无争议合并；通配对的选择（与谁合）是策略决策，留给单动作路径。与现有流程（批次优先、批次为空走单动作）组合后自然实现事实 4：先批次合完同类项 → 批次为空 → 单动作才轮到幻天使对。
- `tasks/coop.py` 两处调用传入配置通配集合：培养阶段 `find_merge_candidates(self.ctx.main_c)`（约 L2214）与金币等待合成（约 L2277）改为 `find_merge_candidates(self.ctx.main_c, wildcard_heroes=frozenset(self._run_config.wildcard_merge_heroes))`。
- 日志约定（docs/logging.md）：通配对被选中合成、被 3 星规避或被批次排除时打 INFO（含幻天使星级与搭档）；候选生成统计打 DEBUG。棋盘识别日志自动带出 `angel_phantom` 类名，无需额外处理。
- 合并产物随机（事实 3）无需专门处理：既有动作后重识别流程天然覆盖。产物若是阵容外英雄，分类器无对应类 → 低置信拒绝 → 该格按 unknown 保守处理（棋盘占用统计不受影响，该英雄不参与后续配对；记入 §7 TODO 观察）。

### 4.6 错误处理与保守行为

- 分拣只迁移高置信紫色样本；中间带一律留在原类并出复核清单——宁可少迁不可错迁。
- 模型评估不达标不切配置，现模型继续服役。
- 通配集合为空（未配置）时所有新逻辑短路，行为与现状一致——配置回滚即功能回滚。
- 识别/合并失败路径（低置信拒绝、拖动失败记忆、棋盘签名变化重识别）全部沿用现有机制，无新增盲操作。

## 5. 测试计划

1. **单元（纯决策）**：
   - `find_merge_candidates`：同类项优先于通配对；通配+任意同星；通配+通配；每个英雄单次使用；星级不匹配不配对；通配+主 C → `is_main_c=True`；`wildcard_heroes` 默认空集时与现状逐例一致（回归）。
   - `select_merge`：同类项与通配对并存时选同类项；仅剩通配对时选通配；3 星通配对在棋盘未满时召唤避开（与同类项 3 星对同待遇）。
   - `select_merge_batch`：通配对被排除。
   - 配置：`lineup_others` 含 angel_phantom 且档案齐全时加载通过；`wildcard_merge_heroes` 非法标识报错。
2. **数据/模型离线验证**：dry-run 分布报告人工过目后迁移；训练后 `evaluate --split test` 按验收标准出报告。
3. **回归**：`python -m pytest` 全绿；既有 coop 任务用例（fake 识别喂 `angel`）不受影响。
4. **首跑人工验证清单**（实机，跑通前不声称完整可用）：
   - [ ] 棋盘日志中 `angel_phantom{1,2,3}星@…` 出现且位置正确（对照参考截图）
   - [ ] 幻天使+雪姬等同星对被拖动合成，产物随机性不导致流程卡住（重识别正常）
   - [ ] 同类项对与幻天使对并存时，先合同类项
   - [ ] 3 星幻天使对在有空位时被避开（召唤优先）
   - [ ] 结算与下一局循环正常

## 6. 非目标（本期不做）

- 复核清单的二轮精修迁移与再训练（montage 人工复核后另行执行）。
- 幻天使出现频率统计、通配对搭档择优（如按场上最少副本的英雄择优）——当前取棋盘顺序第一个可搭档者，简单可预测。
- 其他主 C 数据集（snow/death-knight-helper 等尚不存在）与 angel-helper 模型的幻天使拆分。
- 幻天使模板采集（模板备用路径不支持，主链路 ONNX 不受影响）。
- 合并产物为阵容外英雄时的专门策略（先按 unknown 保守观察）。

## 7. 风险与 TODO（同步记入 docs/coop/TODO.md）

| 风险/未知 | 处置 |
|---|---|
| `angel` 中残留红色调暧昧样本（star1 约 38 张 hue=0）可能混有幻天使 | 留在原类 + montage 复核；确认后二轮迁移（非目标，另期） |
| CPU 训练时长不确定 | 后台执行，与代码改动并行；不阻塞 |
| 新模型既有类别准确率漂移 | evaluate 对比验收，不达标不切配置 |
| 合并产物为阵容外英雄 → 该格 unknown，占用格子且不参与配对 | 保守可接受；实机观察是否需要「清理/忽略」策略（TODO） |
| star4 幻天使类不存在 | 游戏规则不可能出现（事实 1）；若真出现则退化为 unknown，保守 |
| 幻天使模板未采集（`heroes/angel_phantom` 空目录） | 模板路径非主链路；需要时再采集（TODO） |

## 8. 实施顺序（供 writing-plans 展开）

1. 数据重分拣（dry-run → 迁移 → sync-labels）
2. 后台启动训练；并行实施 4.3~4.5 代码与配置改动
3. 单元测试 + 全量回归
4. 模型评估达验收标准 → 切换 `main_c_profiles.assault.hero_classifier_model` 指向 v5
5. 文档同步：game-rules.md（§2 六条事实）、coop/TODO.md（§7）、architecture.md 英雄列表、hero-classifier-usage.md 目录结构、README（如提及 lineup）
