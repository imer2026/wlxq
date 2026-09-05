# 永远的蔚蓝星球脚本工具集

一个基于 Python 的本地自动化工具集，用于在电脑上通过图像识别、窗口截图和模拟点击辅助执行《永远的蔚蓝星球》中的重复操作。

> 本项目面向学习 Python 自动化、图像识别和桌面控制技术。请自行确认游戏规则、平台条款和账号风险，不建议用于破坏公平性、批量牟利或影响其他玩家体验的场景。

## 项目目标

- 通过截图识别游戏界面元素，例如按钮、图标、弹窗和状态区域。
- 根据识别结果执行鼠标点击、拖动、键盘输入等本地操作。
- 将不同任务封装成可配置、可复用的脚本流程。
- 提供调试工具，方便采集截图、标注模板、验证识别效果。
- 尽量保持低侵入性，不修改游戏文件，不注入进程，不读取网络数据包。

## 技术方案

项目文档：

- [使用指南](docs/usage.md)
- [架构设计](docs/architecture.md)
- [游戏规则知识库](docs/game-rules.md)
- [自动化执行流程](docs/automation-flow.md)
- [项目待办](docs/todo.md)
- [合作主流程待办](docs/coop/TODO.md)
- [英雄格分类器使用手册](docs/coop/hero-classifier-usage.md)
- [日志约定](docs/logging.md)

核心思路：

1. 获取游戏窗口截图。
2. 使用 OpenCV 对截图进行模板匹配、颜色识别或特征检测。
3. 根据识别到的位置和置信度判断当前界面状态。
4. 使用 PyAutoGUI 或同类库模拟鼠标、键盘操作。
5. 通过配置文件组织任务流程、延迟、阈值和开关。

推荐技术栈：

- `Python 3.10+`
- `opencv-python`：图像识别与模板匹配
- `numpy`：图像数组处理
- `pyautogui`：鼠标键盘模拟
- `pillow`：截图和图片处理
- `pydantic`：配置校验
- `pyyaml`：YAML 配置文件
- `rich`：命令行日志输出
- `rapidocr-onnxruntime`：技能标题 OCR（按优先级选卡）与离线建册

## 目录结构

按 [架构设计](docs/architecture.md) 的模块边界组织。Runner 调度 Task Engine；需要并发的短生命周期流程由 Runner 侧 Workflow Coordinator 编排，所有识别和输入仍分别经过 Perception 与 Action Executor。

```text
.
|-- README.md
|-- CLAUDE.md                 # 指向 AGENTS.md
|-- AGENTS.md                 # AI 编码助手工作约定
|-- pyproject.toml            # 包元数据 + ruff + pytest 配置
|-- requirements.txt          # 运行时依赖（与 pyproject.toml 同步）
|-- configs/
|   |-- default.yaml          # 运行参数、安全策略、主C档案
|   |-- local.yaml.template   # 本机窗口规格与模板包覆盖
|   `-- tasks.yaml            # 通用 hotspot、ROI、定位器、技能和英雄模板
|-- assets/
|   `-- templates/
|       |-- 927x1727/          # 旧客户区档案，仅可显式选择
|       |   |-- buttons/
|       |   |-- skills/
|       |   `-- heroes/
|       |       |-- assault/
|       |       |-- monkey/
|       |       |-- angel/
|       |       |-- snow/
|       |       |-- death_knight/
|       |       `-- fox/
|       |-- 1920x1080/        # 16:9 横屏
|       |   |-- buttons/
|       |   |-- skills/
|       |   `-- heroes/
|       |-- 2560x1440/
|       |   |-- buttons/
|       |   |-- skills/
|       |   `-- heroes/
|       `-- 3000x2000/        # 3000x2000 显示器实采模板包
|           |-- buttons/
|           |-- skills/
|           `-- heroes/
|-- screenshots/
|   |-- raw/                  # 原始截图
|   `-- debug/                # 调试标注图
|-- src/
|   `-- wlxq_bot/
|       |-- __init__.py
|       |-- __main__.py       # 支持 python -m wlxq_bot
|       |-- py.typed
|       |-- cli.py            # Typer CLI 入口
|       |-- config.py         # Pydantic 配置模型 + 加载
|       |-- assets.py         # 模板包加载
|       |-- models.py         # 核心数据类
|       |-- runner.py         # Runner: 初始化环境、调度任务
|       |-- orchestration/
|       |   `-- coop_grab.py  # Runner侧抢合作双线程协调器
|       |-- perception/       # Perception Pipeline
|       |   |-- screen.py     # Screen Capture
|       |   |-- vision.py     # Vision: 模板匹配、颜色识别、标注
|       |   |-- locator.py    # Locator: 坐标换算、ROI、动作点
|       |   |-- hero_classifier.py # 英雄格 ONNX 运行时推理适配
|       |   `-- skill_collector.py # 技能卡统计阶段采集（非阻塞旁路）
|       |-- skill_catalog.py  # 技能清单离线建册（OCR 卡图 → configs/skills.yaml）
|       |-- hero_classifier/  # 英雄格素材采集、裁剪、训练和评估
|       |   |-- cli.py        # hero-classifier 采集、导入、同步、训练与评估命令
|       |   |-- collector.py  # 固定时间点截图 + 异步 PNG 落盘
|       |   |-- cropper.py    # 局后离线裁取 12 个英雄格 + 跨格归类
|       |   |-- grouper.py    # 视觉聚类、严格二次细分和多样化 candidates 生成
|       |   |-- dataset.py    # split/import 隔离、多局裁切和标签清单重建
|       |   |-- labels.py     # 人工标签扫描与按整局隔离
|       |   |-- trainer.py    # MobileNetV3-Small 训练 + ONNX 导出
|       |   `-- evaluator.py  # 未参与训练的整局数据评估
|       |-- action/           # Action Executor
|       |   |-- executor.py   # 动作执行入口
|       |   |-- input.py      # Input Controller (PyAutoGUI + FakeInput)
|       |   `-- safety.py     # Safety Guard: 停止信号、边界检查
|       |-- debug/
|       |   `-- recorder.py   # Debug Recorder: 事件订阅
|       |-- tasks/
|       |   |-- base.py       # Task 基类、状态机基础
|       |   `-- coop.py       # 合作任务状态机与主C培养决策链
|       `-- utils/
|           |-- log.py        # Rich 日志初始化
|           `-- time.py       # 随机延迟、超时工具
`-- tests/
    |-- conftest.py
    |-- test_imports.py       # 包导入冒烟测试
    |-- test_config.py        # 配置校验测试
    |-- test_models.py        # 核心数据模型测试
    `-- test_safety.py        # Safety Guard 测试
```

### 配置与工具链

项目使用 `pyproject.toml` 统一管理：

- **包元数据**：`[project]` 段，包含依赖和入口点
- **ruff**：`[tool.ruff]` 段，代码风格和 import 排序
- **pytest**：`[tool.pytest.ini_options]` 段，测试路径和标记
- **coverage**：`[tool.coverage]` 段，覆盖率配置

开发依赖（pytest、ruff 等）通过 `pip install -e ".[dev]"` 安装。
英雄格分类器训练依赖（PyTorch、torchvision、ONNX）独立通过
`pip install -e ".[train]"` 安装；正式运行和离线评估 ONNX 不要求安装 PyTorch。

## 功能规划

### 基础能力

- 截取当前屏幕或指定游戏窗口。
- 保存截图到本地，便于后续制作识别模板。
- 按模板图片在截图中查找目标位置。
- 显示识别置信度、匹配坐标和调试标注图。
- 按目标中心点执行点击。
- 支持点击前后随机延迟，降低误触和过快操作造成的问题。

### 任务能力

后续可以按实际玩法逐步补充：

- 自动关闭常见弹窗。
- 自动领取可见奖励。
- 自动进入指定功能入口。
- 自动执行一组固定日常流程。
- 检测异常状态并暂停，例如识别失败、网络断开、界面不符合预期。

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate
# 开发模式安装（含 dev 依赖：pytest、ruff）
pip install -e ".[dev]"
# 需要训练英雄格分类器时额外安装（正式运行不需要）
pip install -e ".[train]"
# 或仅安装运行时依赖
pip install -r requirements.txt
```

如果使用 PowerShell，激活虚拟环境时可能需要先允许当前用户执行脚本：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 快速开始

安装后支持以下命令（首期部分为占位，随模块完善逐步实现）：

```bash
# 显示版本
wlxq-bot version

# 截取当前游戏窗口，保存到 screenshots/raw/
wlxq-bot screenshot

# 在当前游戏画面中识别指定模板，返回置信度（-t 调识别度 0~1）
wlxq-bot find assets/templates/3000x2000/buttons/coop_difficulty/cai_hong_2.png
wlxq-bot find coop_difficulty/cai_hong_2.png -t 0.7 --show

# 实时识别当前游戏画面，验证模板识别效果
wlxq-bot recognize
wlxq-bot recognize -c heroes --hero assault
wlxq-bot recognize -t 0.7 --show
# 也可识别指定截图文件（离线，无需游戏窗口开着）
wlxq-bot recognize screenshots/raw/shot.png

# 执行合作任务（ONNX 格子模型或技能素材未配置时会在启动检查中阻止正常流程）
# 运行前需停留在游戏首页；程序正向识别首页后才会点击首页聊天
# 棋盘模型路径由 configs/default.yaml 的 main_c_profiles.<主C>.hero_classifier_model 指定；
# 模型旁必须保留训练导出的同名 JSON metadata
wlxq-bot run coop --main-c assault
# 临时覆盖配置中的合作难度范围，依次选择10到1
wlxq-bot run coop --main-c assault --coop-difficulties 1-10
# 游戏本次会话已手动选过难度：首局招募跳过难度弹窗直接抢合作
wlxq-bot run coop --main-c assault --skip-difficulty-selection

# 手动打开难度弹窗后，单独验证难度识别、点击和滚动
wlxq-bot exec select-difficulty --coop-difficulties 1-10

# 技能卡统计阶段采集（一次性）：先在 configs/default.yaml 打开
# run.skill_collection.enabled，正常跑几局合作即可自动采集技能卡到
# datasets/skill_cards/（不参与决策、不阻塞对局）；采齐后记得关掉开关
# 离线 OCR 采集卡图，生成/增量合并技能清单 configs/skills.yaml
# （清单含技能名与描述，按英雄分类；开局页与合成4星赠送页不区分）
wlxq-bot build-skill-catalog --dry-run   # 只看统计不写盘
wlxq-bot build-skill-catalog             # 写入清单

# 英雄格分类器：采集一局完整客户区截图（默认每秒1张、持续360秒）
# 主 C 必填；局目录默认使用命令启动时的本地时间戳
wlxq-bot hero-classifier collect --main-c assault --role helper
# 指定多局创建 train import，集中裁切、联合聚类并生成少量 candidates
wlxq-bot hero-classifier import-rounds datasets/hero_classifier/assault/3000x2000/helper \
  --split train --rounds 202608111914,202608112021 --import-id 20260812_001
# 人工挑图移动到 train/labeled 后，脚本完整重建清单
wlxq-bot hero-classifier sync-labels datasets/hero_classifier/assault/3000x2000/helper --split train
# train/validation 标签准备好后直接按 split 训练；主 C 默认从标准数据目录推导
# 默认业务权重：1/2星=1.0，主C/其他3星=0.8/0.5，
# 主C/其他4星=0.3/0.1，empty/unavailable=0.5/0.3
wlxq-bot hero-classifier train datasets/hero_classifier/assault/3000x2000/helper
# 初始模型验证可用后，对新 import 的 candidates 做离线预分类
wlxq-bot hero-classifier suggest-labels \
  datasets/hero_classifier/assault/3000x2000/helper/train/imports/20260820_001 \
  --model outputs/hero_classifier/model/hero_classifier.onnx
# 使用 test split 的独立整局数据评估
wlxq-bot hero-classifier evaluate datasets/hero_classifier/assault/3000x2000/helper \
  --model outputs/hero_classifier/model/hero_classifier.onnx \
  --split test

# 仅调试主C培养决策链
wlxq-bot run coop --main-c assault --start-state build_main_c

# 也可以用 python -m 调用
python -m wlxq_bot version
```

## 配置示例

`configs/default.yaml`：

```yaml
screen:
  mode: fullscreen
  window_title: "永远的蔚蓝星球"
  screenshot_dir: "screenshots/raw"

vision:
  default_threshold: 0.85
  debug: true
  debug_dir: "screenshots/debug"

input:
  click_duration: 0.08
  # 输入动作后的拟人随机间隔（秒）
  min_delay: 0.8
  max_delay: 1.5

safety:
  stop_hotkey: "esc"
  max_failures: 5

run:
  # 本次最多完成的合作局数
  max_rounds: 20
  # 单局调度循环最终保险，完成一局后自动重置
  max_steps_per_round: 5000
  # 首次招募时按从小到大顺序勾选的合作难度（编号指彩虹难度）
  coop_difficulties: "1-16"
  # 主 C 培养时每批连续识别的棋盘帧数；范围 1～60
  board_recognition_frames: 10
```

使用 `wlxq-bot --debug run coop --main-c assault` 可查看每批棋盘识别的总耗时，以及单帧截图、ONNX 批量推理的平均/最小/最大耗时，再据此调整 `board_recognition_frames`。

## 模板图片规范

- 模板包按游戏窗口所在显示器的物理分辨率分目录；默认只加载完全同名的分辨率目录，`window.template_pack` 可显式覆盖，但禁止跨分辨率回退。
- 模板图片应尽量裁剪到目标元素本身，不要包含过多背景。
- 同一个按钮在不同分辨率、缩放比例或昼夜主题下可能需要多套模板。
- 文件名建议表达清楚用途，例如：
  - `claim_reward_button.png`
  - `close_dialog_x.png`
  - `confirm_button_blue.png`
- 每次新增模板后，应先运行识别测试，确认置信度稳定。

## 开发原则

- 优先做可观察、可调试的自动化，不写黑盒流程。
- 每个任务都应有明确的前置界面判断，避免在错误界面乱点。
- 所有点击动作都应从识别结果推导，尽量避免硬编码坐标。
- 对失败场景保持保守：识别不到就暂停或退出，而不是继续盲点。
- 把任务流程、阈值、延迟放到配置文件中，减少改代码的频率。

## 风险与限制

- 图像识别依赖分辨率、窗口缩放、画面亮度和 UI 状态，环境变化会影响准确率。
- 模拟点击可能误操作，运行前建议先在可控场景中测试。
- 游戏更新后 UI 变化可能导致模板失效。
- 本工具不会绕过验证码、风控、登录验证或平台安全机制。

## 路线图

- [x] 初始化 Python 包结构（src layout，按架构文档模块划分）。
- [x] 增加依赖文件和基础配置（pyproject.toml + configs/）。
- [x] 实现核心数据模型（WindowContext / Observation / State / Action / BoardSnapshot 等）。
- [x] 实现 Safety Guard 安全检查逻辑。
- [x] 实现屏幕截图模块（ScreenCapture + WindowContext）。
- [x] 实现模板匹配模块（Vision：单模板 / 模板集 + NMS）。
- [x] 实现鼠标点击封装和输入前安全门禁（InputController + ActionExecutor）。
- [x] 实现棋盘格子坐标模型，并将正式 12 格英雄识别接入 ONNX 分类模型与多帧投票。
- [x] 实现编排层骨架（Runner + CoopTask 状态机 + `run` 命令）和主C培养决策链。
- [x] 实现任务级动作验证协议；前5次强制召唤逐次发送、点击成功即计数并等待1～2秒识别（识别异常不阻断），第5次后的追加召唤、合成、准备/返回、弹窗和技能选择严格验证。
- [ ] 实现调试截图输出（DebugRecorder 落盘）。
- [ ] 完成通用 hotspot、返回按钮 ROI 等定位参数标定，并实测技能候选 ROI 三列中心点击，执行真实游戏端到端验证。
- [x] 实现 Esc 全局停止监听线程（触发 SafetyGuard.request_stop）。
- [ ] 补充示例模板和端到端冒烟测试。

## 许可证

暂未指定许可证。发布或分享前请先补充明确的开源许可证。
