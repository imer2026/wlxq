# 架构设计

本文档记录项目的核心架构、分辨率策略、图像识别边界和任务执行模型。

## 设计目标

- 支持本地窗口截图、图像识别和模拟输入。
- 支持给不同用户使用，但不追求任意分辨率零配置适配。
- 固定 UI 尽量使用窗口相对位置、锚点和 ROI 定位。
- 英雄、技能、弹窗等动态内容必须通过截图识别确认。
- 所有自动化动作都应可观察、可调试、可暂停。

## 运行环境

目标游戏运行形态为微信小游戏。脚本通过 Windows 上可见的微信小游戏窗口进行截图和模拟输入，不修改游戏文件、不注入进程，也不读取网络数据。

已确认的窗口特征：

- 窗口标题：`永远的蔚蓝星球`
- 窗口类名：`Chrome_WidgetWin_0`（基于 Chromium 内核）
- 客户区尺寸：默认竖屏（实测 `927×1727`，9:16 比例）
- DPI：随系统设置，需在程序启动时调用 DPI 感知
- 权限：微信小游戏窗口通常以管理员权限运行，调整窗口大小（`SetWindowPos`）需要在管理员权限终端中执行

窗口查找、截图和坐标转换统一通过 `WindowContext` 和 `ScreenCapture` 抽象隔离，任务逻辑不直接依赖 pywin32 或 MSS。游戏客户区固定为 9:16，任务 hotspot、ROI 和棋盘坐标统一使用客户区比例坐标并保存在可提交的 `configs/tasks.yaml`；由 `pick` 标定一次后作为共享配置使用。本机 `configs/local.yaml`（不提交 git）只保存窗口规格和模板包覆盖，窗口段可由 `wlxq-bot save-window` 生成、`wlxq-bot adjust-window` 应用。

## 总体架构

```text
CLI
 |
Runner
 |
+-- Task Engine
+-- Workflow Coordinator
|   `-- CoopGrabCoordinator
+-- Perception Pipeline
|   +-- Screen Capture
|   +-- Vision
|   `-- Locator
+-- Action Executor
|   +-- Safety Guard
|   `-- Input Controller
`-- Debug Recorder

Runtime Context
+-- WindowContext
`-- Config / Assets
```

上图描述自动任务命令的主依赖链。`screenshot`、`find`、`recognize` 等只读诊断命令可以直接复用 Perception 的公开接口，不需要人为套一层 Runner；`click`、`spam-click`、`move` 等人工输入诊断命令虽然不运行任务状态机，仍必须构造 `Action` 并经过 `Action Executor -> Safety Guard -> Input Controller`，不能直接调用 PyAutoGUI。

`exec` 只是可独立验证业务能力的 CLI 命名空间，不是新的流程编排层。会产生游戏输入的 `exec` 能力必须复用 Runner、任务决策、Perception 和 Action Executor；CLI 只负责参数与结果展示，不得复制正式任务的业务步骤。新增同类能力时优先增加 `exec` 子命令，不继续扩张顶层命令。

各模块职责：

- `CLI`：提供命令入口，例如截图、识别模板、执行任务；可独立验证的业务能力统一收敛到 `exec` 命令组。
- `Runner`：加载配置、初始化运行环境、调度任务循环，并路由任务发出的复合动作信号。
- `Task Engine`：运行任务状态机，根据识别结果选择下一步动作，并处理超时、重试和暂停。任务内部由「状态机调度」与「英雄培养策略」（`tasks/strategy.py` 的 `CultivationStrategy`）组合而成：策略是纯决策对象，只依赖棋盘快照与主 C 档案配置（合成对选择、避让合并、星级/数量门禁、技能次数回补、拖动方向），不接触 Perception 和 Action；新增主 C 差异化策略时改策略类和档案配置，不动状态机流程。
- `Workflow Coordinator`：承载需要并发或跨多个原子动作的 Runner 侧短生命周期编排；例如 `CoopGrabCoordinator` 并行执行抢合作点击和准备按钮识别。它不属于任务状态机，不得绕过 Perception、Action Executor 或 Safety Guard。
- `Perception Pipeline`：把窗口截图转换为可供任务判断的界面状态，不执行输入动作。
- `Screen Capture`：负责获取游戏窗口客户区截图。
- `Vision`：负责模板匹配、颜色识别、图标识别、调试标注。
- `Locator`：基于 `WindowContext` 统一计算候选位置、ROI、模板匹配位置和最终动作点。
- `Action Executor`：在安全检查通过后执行输入，并验证动作结果。
- `Input Controller`：封装鼠标点击、拖动、键盘输入。
- `Safety Guard`：负责停止信号、失败次数、窗口状态和动作边界检查。
- `Debug Recorder`：订阅任务执行过程中的事件，保存原始截图、标注截图、识别结果和动作日志，不参与业务判断。另维护主循环最近帧的环形缓冲（退出帧），任务非正常退出时批量落盘供事后排查，同样不参与业务判断。
- `Hero Classifier Tools`：通过独立的 `hero-classifier` CLI 命名空间完成英雄格完整截图采集、局后裁剪、人工标签扫描、轻量模型训练和整局评估；不属于 Debug Recorder，也不参与任务状态机编排。
- `Hero Cell Classifier`：Perception 层的 ONNX 运行时适配器，把单格 BGR 图片转换为英雄、星级、`empty`、`unavailable` 或 `unknown`；正式合作任务的 12 格棋盘识别只使用该模型，不再回退到英雄模板匹配。
- `Skill Collector`：Perception 层的统计阶段旁路采集器（`perception/skill_collector.py`）。仅在「技能页标志可见且技能卡图标已实际命中（即将点卡）」的帧复用主流程已在手的截图裁卡存档（卡图 + JSONL 元数据）——该时刻卡片必然渲染完整，离线归属才可靠；运行时只做裁剪和 aHash 去重（毫秒级、按 `min_collect_interval_seconds` 节流），PNG 编码在写盘线程完成，队列满丢弃、内部异常只记日志并自动熔断；不参与业务决策、不产生输入、不做任何模板匹配，只在 `run.skill_collection.enabled` 打开时工作。
- `Skill Catalog Builder`：离线建册工具（`skill_catalog.py`，CLI `build-skill-catalog`）。英雄归属（卡图与英雄图标模板匹配）和文字 OCR（首行为技能名、其余行拼接为描述）都在离线完成，按英雄分组合并进 `configs/skills.yaml`，人工补充的 `priority` 等字段保留；英雄技能开局页与合成 4 星赠送页一致，清单按英雄平铺不区分来源。OCR 引擎（`rapidocr-onnxruntime`）只在离线建册时延迟加载。
- `WindowContext`：保存当前窗口句柄、客户区位置和尺寸、DPI、显示器、截图时间等运行时信息。
- `Config / Assets`：保存任务配置、模板图片、阈值、延迟和开关。

模块依赖约束：

- 任务状态机只依赖识别结果和动作执行接口，不直接调用 OpenCV、MSS、pywin32 或 PyAutoGUI。
- 原子输入动作（点击、拖动、按键）统一交给 `Action Executor`；`grab_coop` 这类复合动作信号由 Runner 路由到 `Workflow Coordinator`，协调器内部产生的每个输入仍必须经过 `Action Executor`。
- `Vision` 只负责识别；坐标换算和动作点计算统一由 `Locator` 负责。
- `Input Controller` 只能通过 `Action Executor` 调用，禁止任务代码绕过安全检查直接点击。
- `Debug Recorder` 通过统一事件记录过程，避免日志和截图逻辑散落到各模块。
- 英雄格训练依赖只允许由 `hero_classifier.trainer` 延迟加载；正式运行时的 Perception 只通过 OpenCV DNN 加载 ONNX，不依赖 PyTorch。
- 英雄格训练、验证和测试数据必须按完整对局隔离；数据集、训练缓存和未确认模型属于本地生成物，不提交仓库。训练抽样先按业务优先级给每一个实际存在的类别分配总权重：1 星和 2 星英雄为 `1.0`，主 C/其他英雄 3 星为 `0.8/0.5`，主 C/其他英雄 4 星为 `0.3/0.1`，`empty/unavailable` 为 `0.5/0.3`；类别内部继续按来源局和样本类型均衡。不存在或没有图片的星级类别不进入模型，也不要求补齐。实际类别权重和主 C 写入模型 metadata。

英雄格原始完整截图按 `rounds/<round_id>/` 隔离保存；训练、验证和测试各自使用 `imports/<import_id>/` 登记来源局，同一局只能属于一个 split。创建 import 前必须扫描全部历史 `rounds.txt`：请求中已完成裁切和聚类的局按其原 split/import 提示并跳过，只处理剩余新局；全部为已处理局时不创建空 import。一个 import 内的新局集中裁切并联合聚类，新增 import 不修改旧 import 或 split 共用的 `labeled/`。人工只移动图片，`sync-labels` 扫描真实目录后完整、原子重建 `dataset_manifest.csv`，不要求手工维护清单。

英雄格数据制作中的自动聚类仅是人工标注前的目录整理步骤：它先以平均像素差阈值 35 把同一 import 的裁剪图放入 `unclassified/` 一级簇。随后为所有一级簇生成 `candidates/`：不超过 100 张的簇直接选最多 10 张；超过 100 张的簇以阈值 15 二次细分，每个二级簇选最多 10 张视觉差异尽量大的候选。候选上限可通过 CLI 参数调整。候选使用复制文件，原始裁剪图完整保留，`candidate_manifest.csv` 负责追溯。

聚类和候选生成都不识别英雄或星级、不自动写标签，也不把图片直接加入训练集；只有人工确认并移入 split 共用 `labeled/` 的图片才由训练器读取。候选组内仍允许存在不同标签，标注者可以批量移动一致图片，混合组分别挑选，必要时回到对应 `unclassified/` 一级簇补查。

初始模型通过独立验证后，`hero-classifier suggest-labels` 可以作为纯离线人工辅助工具读取某个 import 的 `candidates/`。它按 `candidate_manifest.csv` 批量执行 ONNX 推理；同一一级簇/二级簇小组内全部候选预测为相同精确类别且每张都通过 confidence、margin 门槛时，复制到 `suggested/<预测类别>/`。组内类别不一致进入 `suggested/review/mixed_group/`，低 confidence 或低 margin 进入 `suggested/review/low_confidence/`，无法使用的预测进入 `suggested/review/unknown/`。逐图 top1、top2、概率、margin、拒绝原因、模型哈希和最终小组决策写入 `prediction_manifest.csv`。

`suggest-labels` 不属于正式棋盘识别，不读取实时游戏窗口，也不参与 `BoardSnapshot`、任务决策或输入动作。它只复制 candidates，禁止改动 `unclassified/`、`candidates/` 或直接写入 `labeled/`；已有 `suggested/` 或 `prediction_manifest.csv` 时拒绝覆盖。人工确认并把 PNG 文件移入 split 共用 `labeled/` 后，仍必须执行 `sync-labels`，模型建议本身永远不作为训练真值。

## 核心概念模型

本项目不建立完整的游戏业务模型，只定义自动化任务运行所需的最小概念：

已经确认的玩法和英雄合成规则统一记录在 [游戏规则知识库](game-rules.md) 中；本文档只描述这些规则如何映射到自动化架构。

| 概念 | 含义 | 类型 |
|------|------|------|
| `Task` | 一个可执行目标，例如领取奖励或选择技能 | 任务模型 |
| `DecisionPolicy` | 根据任务目标和最新识别状态选择召唤、合成或技能等行为 | 任务模型 |
| `State` | 当前识别出的界面状态 | 任务模型 |
| `Observation` | 从单帧或同一稳定窗口上下文内的短帧序列得到的识别结果和置信度 | 任务模型 |
| `Transition` | 某个状态下允许执行的动作、目标状态和失败规则 | 任务模型 |
| `Action` | 点击、拖动、按键、等待，或由 Runner 路由的复合操作信号 | 任务模型 |
| `ActionResult` | 动作是否执行、验证是否成功及失败原因 | 任务模型 |
| `WindowContext` | 截图和动作共同依赖的窗口运行环境 | 技术模型 |
| `Locator` | 候选位置、识别范围和最终动作点的定位规则 | 技术模型 |

概念之间的关系：`Task` 根据 `Observation` 确定 `State`，由 `DecisionPolicy` 结合任务目标选择一个 `Transition` 并产生 `Action`；动作执行后得到 `ActionResult`，再通过新的 `Observation` 验证是否进入目标状态。整个过程共享同一个有效的 `WindowContext`，目标位置由 `Locator` 计算。

涉及棋盘操作的任务还需要从游戏规则知识库读取当前模式、合作角色和己方 `BoardRegion`。英雄召唤、定位和合成都只能在该区域内进行；另一名玩家的棋盘只作为画面背景，不作为动作目标。

首版不建模游戏内部完整的英雄、技能和关卡体系。只有任务确实需要根据属性、组合或关卡规则做决策时，才增加对应模型，避免提前引入无用抽象。

## 运行配置与主 C 选择

稳定的策略参数从配置文件读取，不要求用户每次启动时重复输入：

- `max_rounds`：本次最多完成的合作局数。
- `minimum_summon_count_before_skills`：允许进入局内技能选择前必须完成的最低召唤次数，当前为 5。
- `initial_board_capacity`：开局棋盘可容纳的英雄数量。
- `target_star_level`：主 C 停止召唤和合成的最低星级，当前为 2。
- `summon_recognition_delay_min/max`：召唤后等待英雄落位再识别的最短/最长时间，当前方案为 1～2 秒。
- `board_recognition_frames`：每次主 C 培养决策连续识别的棋盘帧数，默认 10，可按 DEBUG 耗时日志调整。
- `hero_classifier_model`：每个主 C 档案的 ONNX 模型路径；模型旁必须有同名 JSON metadata。
- `coop_difficulties`：首次招募时需要勾选的 1～16 难度范围，命令行参数可以覆盖。
- `skill_icon_templates`：每个主 C 的技能卡英雄图标模板；当前简化策略只判断候选是否属于主 C，不区分同一主 C 的具体技能。
- 安全重试、识别阈值和等待时间。

主 C 使用独立配置档案。当前优先支持：

- `assault`：强袭。
- `monkey`：猴子。
- `angel`：天使。
- `snow`：雪姬。
- `death_knight`：死骑。
- `fox`：狐狸。

格子模型的运行配置结构如下：

```yaml
run:
  max_rounds: 20
  max_steps_per_round: 5000
  minimum_summon_count_before_skills: 5
  initial_board_capacity: 7
  target_star_level: 2
  summon_recognition_delay_min: 1.0
  summon_recognition_delay_max: 2.0
  board_recognition_frames: 10
  default_main_c: assault

main_c_profiles:
  assault:
    display_name: 强袭
    hero_template_dir: heroes/assault
    hero_classifier_model: outputs/hero_classifier/assault-helper/hero_classifier.onnx
    skill_icon_templates:
      - skills/qiang_xi.png

  monkey:
    display_name: 猴子
    hero_template_dir: heroes/monkey
    hero_classifier_model: outputs/hero_classifier/monkey-helper/hero_classifier.onnx
    skill_icon_templates: []

  angel:
    display_name: 天使
    hero_template_dir: heroes/angel
    hero_classifier_model: outputs/hero_classifier/angel-helper/hero_classifier.onnx
    skill_icon_templates: []

  snow:
    display_name: 雪姬
    hero_template_dir: heroes/snow
    hero_classifier_model: outputs/hero_classifier/snow-helper/hero_classifier.onnx
    skill_icon_templates: []

  death_knight:
    display_name: 死骑
    hero_template_dir: heroes/death_knight
    hero_classifier_model: outputs/hero_classifier/death-knight-helper/hero_classifier.onnx
    skill_icon_templates: []

  fox:
    display_name: 狐狸
    hero_template_dir: heroes/fox
    hero_classifier_model: outputs/hero_classifier/fox-helper/hero_classifier.onnx
    skill_icon_templates: []
```

Runner 启动时加载当前主 C 的 ONNX 和同名 metadata，并检查模型是否包含当前主 C 与 `hero_classifier.lineup_others` 中的全部英雄类别。每帧按 Locator 的 12 格 ROI 一次批量推理；同一格在短帧序列中按精确类别多数投票，只有过半类别才进入 `BoardSnapshot`。本局阵容之外的英雄、低 confidence、低 margin、投票并列或不过半都保守视为 `unknown`。

`skill_icon_templates` 为空时启动检查应阻止进入自动对局。当前素材只能识别技能卡上的主 C 英雄图标，同一主 C 的多个具体技能不可区分，因此命中多个主 C 候选时随机选择。连续多帧未命中时，`Locator` 将技能候选 ROI 横向三等分，取三列中心作为三张卡片的候选点击点并随机选择一个；无需额外标定兜底 hotspot。技能候选 ROI 缺失或无效时保守停止。

合作模式已确认初始开放 7 格、每累计召唤 5 次自动开放 1 个新格，并且局内「选择技能」必须在累计召唤至少 5 次后才能使用。目标流程不再批量提交前 7 次召唤：先识别 12 格作为基线，此后每次只召唤 1 个，等待 1～2 秒后再用短帧序列识别全部 12 格。前 5 次以点击成功发送作为计数依据，模型未确认变化也不能阻断下一次召唤；即使提前出现目标主 C 也继续召唤。第一版策略不在这 5 次之间插入合成，以最少动作先完成门禁。第 5 次后只有同时达到主 C 星级条件，才转入局内技能选择。程序不需要通过视觉复核开放格数。

启动时只需要确定本次主 C。命令行显式传入主 C 时直接使用；未传入时可以使用 `default_main_c`，或在检测到多个可用配置档案时提示用户选择。例如：

```powershell
python -m wlxq_bot.cli run coop --main-c assault
python -m wlxq_bot.cli run coop --main-c monkey
```

启动检查必须确认所选主 C 的 ONNX 模型、同名 metadata、当前阵容类别和技能图标模板均可用。主 C 不存在或素材不完整时，应在进入合作前报错停止；运行到技能兜底分支时复用本次识别使用的技能候选 ROI，由 `Locator` 计算三张卡片的中心点击点，ROI 缺失或无效时保守停止。

## WindowContext 与坐标体系

`WindowContext` 是一次截图、识别和动作执行共同使用的窗口快照，至少包含：

```text
window_handle
client_rect_screen
client_size
dpi
monitor_id
is_foreground
is_minimized
captured_at
frame_id
```

坐标统一使用客户区物理像素：左上角为 `(0, 0)`。截图区域、ROI、模板匹配结果都使用客户区坐标；只有实际输入前，才通过 `WindowContext` 转换为屏幕坐标。

### 9:16 通用定位配置约束

游戏客户区的业务布局固定为 9:16。所有业务点击位置、识别区域和棋盘几何参数必须使用相对客户区左上角的比例坐标，统一写入可提交的 `configs/tasks.yaml`：

- `hotspots`：固定按钮、拖动起终点及其他命名动作点。
- `rois`：模板识别、技能候选和界面状态的搜索区域。
- `board`：棋盘锚点、格子尺寸和行列步长。

这些比例坐标在运行时由 `Locator` 结合当前 `WindowContext.client_size` 换算为客户区物理像素。业务代码只能消费结构化定位配置，不得硬编码绝对像素或私自读取本机坐标。

配置边界是强制约束：

- `configs/tasks.yaml` 保存可跨机器复用的业务坐标、ROI、棋盘参数、定位器和素材映射。
- `configs/default.yaml` 只保存运行参数、安全策略和主 C 档案，禁止存放业务坐标。
- `configs/local.yaml` 只保存本机窗口标题、类名、目标客户区尺寸和 `window.template_pack` 覆盖，禁止存放业务 hotspot、ROI 或棋盘参数。
- 新增或调整定位参数时必须通过截图或 `wlxq-bot pick` 实测；未确认的坐标保持缺失并记录 TODO，不得以本机临时值、屏幕绝对坐标或猜测值填充共享配置。

程序启动时应先启用 DPI 感知，再读取窗口位置和客户区大小。每次关键动作必须使用与识别结果引用的最新 `frame_id` 相同的 `WindowContext`，并在点击前重新检查窗口句柄、客户区位置、尺寸和 DPI；如果上下文已经变化，则丢弃动作并重新截图。时序融合必须记录全部 `source_frame_ids`，只允许在窗口元数据全程稳定时聚合；动作点必须重新落到 `Locator` 推导的稳定格子中心，不能复用早期动画帧中的瞬时坐标。

## 技术选型

项目定位为纯命令行工具，不引入 GUI 框架。自动化核心应保持独立，后续即使增加其他入口，也不应让任务逻辑依赖界面层。

推荐技术栈：

- `Python 3.10+`：基础运行环境。
- `Typer`：命令行框架，用于组织 `screenshot`、`find`、`calibrate`、`run` 及 `exec` 能力命令组。
- `Pydantic`：配置模型和校验，避免任务配置、ROI、阈值、模板路径使用裸字典。
- `PyYAML`：读取 YAML 配置文件。
- `OpenCV`：图像识别核心能力，包括模板匹配、ROI 裁剪、颜色识别和调试标注。
- `NumPy`：图像数组处理。
- `MSS`：优先用于截取游戏客户区对应的屏幕区域，性能通常优于 `PyAutoGUI` 截图；窗口被遮挡或最小化时不继续执行任务。
- `Pillow`：图片读写、格式转换和截图兜底。
- `pywin32`：Windows 窗口查找、客户区尺寸读取、窗口移动和坐标转换。
- `PyAutoGUI`：鼠标、键盘模拟输入。
- `pydirectinput`：后续可选替代输入库，用于部分游戏场景下更稳定的输入模拟。
- `Rich`：命令行日志、表格和调试输出。
- `pytest`：单元测试和核心逻辑回归测试。

首期核心组合：

```text
Typer + Pydantic + OpenCV + MSS + pywin32 + PyAutoGUI
```

不建议首期引入：

- GUI 框架，例如 PyQt、Dear PyGui、Electron。
- 复杂任务编排框架。
- 机器学习模型训练框架。

首期重点应放在窗口检测、固定客户区、模板包选择、ROI 识别、调试截图和安全停止机制上。

## 模板包与分辨率策略

项目采用“固定窗口客户区大小 + 按显示器物理分辨率分包 + 可选显式覆盖”的方案。不同物理分辨率下采集的游戏截图像素形态可能不同，OpenCV 模板匹配不能假定模板可以跨分辨率复用，因此模板必须放在采集时对应的 `assets/templates/<宽>x<高>/` 目录中。

默认模板选择依据是**游戏窗口所在显示器**的当前物理分辨率，不是主显示器分辨率，也不是游戏窗口客户区尺寸。运行时先用 `MonitorFromWindow` 确定游戏窗口所在显示器，再读取该显示器物理像素宽高并加载同名模板包。`configs/local.yaml` 的 `window.template_pack` 非空时优先使用，可由 `save-window` 自动写入，也可由用户显式覆盖；指定包或默认分辨率包不存在时立即停止，不尝试其他分辨率。

推荐支持的显示器物理分辨率包括：

- `3000x2000`：3:2 比例，当前开发环境原生面板尺寸，首版模板采集于此。
- `1920x1080`：16:9 横屏分辨率，常见显示器。
- `2560x1440`：2K 横屏分辨率。

运行要求：

- 游戏应运行在窗口化模式。
- 脚本运行期间应固定窗口大小，使用 `wlxq-bot adjust-window` 按本地配置（`configs/local.yaml`）调整到目标客户区尺寸。
- 不建议运行时拖动窗口、切换分辨率或修改游戏内 UI 缩放。
- 微信小游戏窗口通常以管理员权限运行，`adjust-window` 需在管理员权限终端中执行；`inspect` 和 `save-window` 不需要管理员权限。

启动时分别处理窗口客户区调整和模板包选择：

1. 查找游戏窗口并读取其所在显示器和当前客户区大小。
2. 读取 `configs/local.yaml` 中的目标客户区尺寸；不存在时提示先运行 `wlxq-bot save-window` 保存当前窗口规格。
3. 如果客户区尺寸不匹配目标，自动任务启动检查立即停止并提示先执行 `wlxq-bot adjust-window`；调整窗口是显式的独立命令，不在即将输入的任务中隐式改变窗口。
4. 重新启动任务后读取已调整的客户区，客户区尺寸用于坐标换算和 `WindowContext`。
5. 若 `window.template_pack` 非空，加载 `assets/templates/<template_pack>/`，该值优先于自动选择。
6. 否则读取游戏窗口所在显示器的物理分辨率，严格加载 `assets/templates/<宽>x<高>/`。
7. 对应模板包不存在时停止并提示采集，禁止回退到主显示器、客户区同名目录或其他分辨率模板包。

`save-window` 未传 `--template-pack` 时，把游戏窗口所在显示器的物理分辨率写入 `window.template_pack`；传入参数时保存显式覆盖值。微信小游戏窗口的边框和标题栏占用是环境实测信息，不应作为通用常量写入业务代码。

任务运行过程中应定期检查窗口句柄、客户区位置、尺寸和 DPI。一旦发生变化，应暂停任务并重新建立 `WindowContext`。游戏更新后即使仍是最新版本，也可能发生 UI 变化；如果多个核心模板连续失效，应停止任务并提示重新采集或更新模板。

## 模板目录

模板按采集时的显示器物理分辨率组织：

```text
assets/
`-- templates/
    |-- 1920x1080/
    |   |-- buttons/
    |   |-- skills/
    |   `-- heroes/
    |-- 2560x1440/
    |   |-- buttons/
    |   |-- skills/
    |   `-- heroes/
    `-- 3000x2000/
        |-- buttons/
        |   |-- coop_difficulty/         # 合作模式下的难度选择按钮组
        |   |   |-- cai_hong_1.png
        |   |   |-- cai_hong_2.png
        |   |   `-- ...
        |   |-- zhun_bei.png
        |   `-- fan_hui.png
        |-- skills/
        `-- heroes/
```

技能和英雄这类关键识别对象应使用对应分辨率下实采的模板：

```text
assets/templates/1920x1080/skills/fireball.png
assets/templates/2560x1440/skills/fireball.png
```

`buttons/` 既可以平铺独立按钮模板（如 `zhun_bei.png`、`fan_hui.png`），也可以按业务场景建子目录组织一组互斥或相关的按钮（例如 `coop_difficulty/` 下放彩虹层数 1~16 的合作难度按钮）。子目录命名直接对应业务概念，方便后续配置文件中按业务路径引用模板，例如 `buttons/coop_difficulty/cai_hong_2.png`。`TemplatePack.buttons_dir` 不限制子目录结构，扫描和路径解析只要保持相对于模板包根即可。

不建议把运行时缩放模板作为主要识别方案。OpenCV 模板匹配对像素形态比较敏感，缩放会引入插值、抗锯齿和边缘变化，容易降低置信度。模板缩放最多作为辅助兜底；核心识别对象应使用对应分辨率实采模板，或者通过首次校准流程采集用户本机模板。

## 动态英雄格识别

游戏内英雄存在朝向、动作、特效和遮挡变化，正式棋盘识别不再为每个英雄维护运行时模板集。Locator 使用已标定的 12 格参数裁出固定 ROI，`HeroCellClassifier` 将一帧的 12 张格子图一次批量送入 ONNX；训练数据通过多局采集覆盖动态变化。单图先经过 confidence 和 margin 拒识，多帧再按精确类别投票，无法形成过半结论的格子视为 `unknown`，不参与主 C、合成或追加召唤决策。

## 定位策略

定位不应直接散落在任务代码里，应通过 `Locator` 抽象统一管理。

推荐支持以下定位方式：

- `fixed_ratio`：用客户区宽高比例计算一个候选点。
- `anchor`：从客户区边缘或角落按物理像素偏移计算一个候选点。
- `template`：在指定 ROI 中匹配一个模板，返回匹配位置和置信度。
- `template_set`：在指定 ROI 中匹配多个模板变体，返回置信度最高的结果。
- `manual_calibration`：读取用户标定的候选点或 ROI；它是配置来源，不参与运行时识别。

ROI 不是独立定位策略，而是模板识别的搜索范围。每个 ROI 必须声明 `relative_to`，取值为 `client`、`anchor` 或 `candidate`；最终动作点必须明确来自候选点、匹配中心或匹配结果的偏移。

合作模式下英雄识别只在己方棋盘 ROI 内进行，不对整个游戏界面做匹配。当前实现以 `board.helper` / `board.initiator` 的格子模型推导全部英雄位中心，再由 `Locator` 计算包围盒 ROI；旧的 `bottom_left_board` / `bottom_right_board` 仅作为兼容字段保留，不再参与合作棋盘识别。任务只能消费 `BoardSnapshot` 和 Locator 生成的稳定动作点，不能自行换算棋盘绝对坐标。

示例配置：

```yaml
locators:
  claim_reward_button:
    strategy: anchor
    anchor: bottom_right
    offset:
      x: -180
      y: -80
    verify:
      strategy: template
      template: buttons/claim_reward_button.png
      threshold: 0.82
      roi:
        relative_to: candidate
        x: -130
        y: -60
        width: 260
        height: 120
    action_point:
      source: match_center
```

## 截图使用边界

项目不应为了每次点击都做全屏识别，但也不应完全依赖坐标盲点。

建议分工：

- 固定按钮、菜单入口：窗口比例坐标或锚点定位，必要时用小区域截图校验。
- 技能卡槽、英雄卡槽：位置可按窗口比例定位，身份必须截图识别。
- 技能、英雄图标：使用游戏窗口所在显示器对应的模板包或显式覆盖包；识别失败时提示用户补充该分辨率模板。
- 未知界面或异常状态：保存全屏调试截图并暂停，避免继续盲点。

## 技能识别流程

技能选择属于动态内容识别，必须截图确认。

推荐流程：

```text
识别是否处于技能选择界面
 -> 按窗口比例定位技能卡槽区域
 -> 裁剪每个技能卡槽 ROI
 -> 使用当前显示器分辨率对应的模板包识别技能
 -> 根据配置优先级选择目标技能
 -> 点击对应卡槽中心
 -> 点击后截图验证选择界面是否关闭或技能是否生效
```

技能选择配置示例：

```yaml
skills:
  priority:
    - fireball
    - lightning_chain
    - healing_aura

  templates:
    fireball: skills/fireball.png
    lightning_chain: skills/lightning_chain.png
    healing_aura: skills/healing_aura.png
```

## 任务执行模型

任务应设计成状态机，而不是纯顺序脚本。

已经确认的端到端执行顺序见 [自动化执行流程](automation-flow.md)，本节只说明状态机如何承载该流程。

状态机的作用是：先判断“当前处于什么界面状态”，再决定“当前允许执行什么动作”，并检查动作后是否进入预期状态。这样任务可以从弹窗、延迟或识别失败中恢复，而不是假设前面的每一步都已经成功。

首版不需要引入状态机框架，只需定义四类简单对象：

- `Observation`：当前截图中识别到的按钮、弹窗、技能和置信度。
- `State`：根据 Observation 得出的当前界面，例如 `HOME`、`REWARD_DIALOG`、`SKILL_SELECT`、`UNKNOWN`。
- `Transition`：某个状态下允许执行的动作、预期目标状态、超时和最大重试次数。
- `ActionResult`：动作是否执行、验证是否成功、失败原因和调试证据。

最小执行循环：

```text
截图并生成 WindowContext
 -> 识别 Observation
 -> 按优先级确定 State
 -> 选择该 State 允许的 Transition
 -> Safety Guard 执行动作前检查
 -> 执行动作
 -> 再次截图并验证目标 State
 -> 成功则继续，失败则按规则重试或暂停
```

状态判断优先级应固定：停止或窗口异常最高，其次是阻塞性弹窗，再其次是当前任务界面，最后才是普通入口；没有任何状态可信匹配时进入 `UNKNOWN`。同一 Transition 必须声明最大重试次数，避免无限循环；可能重复触发的动作应尽量保证幂等。

合作任务的顶层状态可以保持为以下最小集合：

```text
FIND_COOP
→ ENTER_MATCH
→ SELECT_OPENING_SKILLS
→ BUILD_MAIN_C
→ SELECT_MAIN_C_SKILLS
→ HANDLE_RESULT
→ CLAIM_REWARD
→ CHECK_ROUND_LIMIT
→ FIND_COOP 或 COMPLETED
```

`SELECT_OPENING_SKILLS` 是进入游戏后的检查阶段：点击技能 hotspot 后未出现候选时限次复查，连续为空则进入 `BUILD_MAIN_C`；出现候选时按主 C 技能优先级循环选择。该阶段不设置固定选择次数，但空候选检查和连续识别失败必须有上限。

`BUILD_MAIN_C` 只能在开局技能阶段完成后进入。格子模型先生成一份 12 格基线；之后每召唤 1 个英雄，等待配置的 1～2 秒稳定期，再以短帧序列重新识别全部 12 格。累计召唤不足 5 次时，点击成功发送即计数，识别不稳定不阻断下一次逐次召唤，也不因提前出现 2 星主 C 而转移状态；第一版策略不在这 5 次之间插入合成。累计达到 5 次后，若未发现 `star_level >= 2` 的主 C，则优先执行合法合成，其中非主 C 合成对优先；没有合法合成对时再召唤 1 个英雄。达到 5 次后的决策必须依赖可用棋盘，追加召唤和合成仍验证棋盘变化；只有累计召唤至少 5 次且检测到 2 星、3 星或 4 星主 C，才停止召唤和合成并转入 `SELECT_MAIN_C_SKILLS`。召唤和技能选择使用独立按钮与 Transition，即使它们都消耗金币也不能合并成同一种动作。

`FIND_COOP` 内部包含代码定义的招募子状态机，不增加顶层状态。首次依次执行“正向识别游戏首页 → 首页聊天 → 招募 → 打开难度弹窗 → 从小到大选择配置难度 → 取消难度弹窗”；首页模板缺失或首帧未命中时保守报错，不产生输入。下一局结算返回后依次执行“合作页面聊天 → 打开难度弹窗 → 关闭难度弹窗”，用开关弹窗触发新的合作邀请刷新，但不重复勾选难度；随后进入抢合作阶段（任务发出 `grab_coop` 信号动作，由 `Runner` 的 `CoopGrabCoordinator` 双线程执行：连点 join_coop 并行识别准备按钮）。步骤顺序属于业务逻辑，不允许通过 YAML 重排；共享任务配置只提供各 hotspot 的比例坐标。首页识别仅在首次门禁步骤启用，彩虹难度模板（当前 1～18）也只在首次选择难度或独立难度选择能力中匹配，后续局刷新弹窗时不识别或更改难度。难度滚动使用两个客户区比例 hotspot 计算拖动起终点，因此拖动像素距离随窗口大小变化。准备按钮出现后进入 `ENTER_MATCH`；所谓加载中只是点击准备后的可配置等待，不单独建状态。

`SELECT_MAIN_C_SKILLS` 不读取金币：前期约每 4.5～6 秒、后期约每 9～11 秒点击一次技能 hotspot，再识别候选并选择。`HANDLE_RESULT`、`CLAIM_REWARD`、`CHECK_ROUND_LIMIT` 分别对应点赞、领取宝箱、点击返回并累计局数；点赞和领取宝箱各点击一次后直接推进，只有返回动作需要后续截图验证。

不推荐：

```text
点击 A
点击 B
点击 C
```

推荐：

```text
识别当前界面
 -> 如果看到奖励按钮，点击领取
 -> 如果看到关闭按钮，关闭弹窗
 -> 如果看到技能选择界面，执行技能选择流程
 -> 如果看到主界面入口，进入功能
 -> 如果状态未知，截图、记录、暂停
```

这样可以降低弹窗遮挡、网络延迟、界面变化导致的误点风险。

## 安全策略

- 每个任务都必须有明确的前置界面判断。
- 每次关键点击前应进行截图或 ROI 校验。
- 点击前应确认窗口句柄有效、窗口未最小化且处于前台。
- 点击前应确认客户区位置、尺寸、DPI 与识别时一致，截图未超过允许时效。
- 最终屏幕坐标必须落在当前客户区范围内，否则拒绝执行。
- 关键界面迁移、技能选择、合成、达到 5 次后的追加召唤和返回必须用后续截图验证。高频抢合作可以由独立线程连点 `join_coop`。前 5 次强制召唤逐次提交并等待稳定期，点击成功发送即增加计数，模型识别异常不阻断下一次；达到 5 次后才恢复严格棋盘决策和变化验证。
- 连续识别失败达到阈值后应暂停任务。
- 检测到窗口大小变化后应暂停任务。
- 检测到未知界面时应保存调试截图并暂停。
- 全局停止热键应由独立监听器设置线程安全的停止信号；截图、等待和动作执行都要及时检查该信号。
- 不绕过验证码、风控、登录验证或平台安全机制。

## 可测试性设计

首版保持轻量，只要求以下隔离：

- 为 `Screen Capture` 和 `Input Controller` 定义简单接口，测试时分别替换为本地截图和 Fake Input。
- 保存少量带预期结果的测试截图，用于模板匹配、状态识别和 Locator 坐标回归。
- 状态机测试只输入 Observation，断言得到的 State 和 Transition，不启动真实游戏窗口。
- Action Executor 测试只验证安全检查、坐标范围和动作记录，不实际移动鼠标。
- 保留一个人工执行的端到端冒烟测试，用于验证窗口调整、截图、识别和安全停止完整链路。

## 推荐落地顺序

1. 初始化 Python 包结构。
2. 实现配置加载、窗口查找和 `WindowContext`。
3. 实现窗口尺寸调整、截图和客户区坐标转换。
4. 实现模板包选择和模板匹配。
5. 实现 Locator 抽象。
6. 实现 Debug Recorder 和离线截图回归测试。
7. 实现 Safety Guard、Fake Input 和真实输入适配器。
8. 实现最小 Task Engine 和状态转换测试。
9. 实现技能识别 MVP。
10. 实现第一个完整任务状态机和端到端冒烟测试。
