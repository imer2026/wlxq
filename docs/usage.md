# 使用指南

本文档介绍如何在本地搭建开发环境、运行命令和日常开发。

## 环境要求

- Windows 10/11
- Python 3.10+；普通开发可使用较新版本，MX250 上训练英雄格分类器建议优先使用 Python 3.11，以兼容仍包含 Pascal `sm_61` 的 CUDA 版 PyTorch
- PowerShell（系统自带）

## 首次安装

### 1. 允许 PowerShell 执行脚本

Windows 默认禁止运行 PowerShell 脚本，激活虚拟环境会报错。执行一次以下命令放开权限（仅对当前用户生效，安全）：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

会弹出确认提示，输入 `Y` 回车即可。此设置永久生效，无需重复执行。

> **说明**：`RemoteSigned` 表示本地脚本可直接运行，从网络下载的脚本需要数字签名。这是开发常用安全级别。

### 2. 创建虚拟环境

在项目根目录下创建本地 `.venv`：

```powershell
# 用你机器上的 Python 创建
python -m venv .venv
```

### 3. 安装项目依赖

```powershell
# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 安装项目（含开发依赖：pytest、ruff）
pip install -e ".[dev]"
```

激活成功后，命令行前面会出现 `(.venv)` 标识。

### 4. 验证安装

```powershell
wlxq-bot version
```

输出 `wlxq-bot v0.1.0` 即安装成功。

---

## 日常使用

### 激活虚拟环境

每次打开新的 PowerShell 窗口后，先激活虚拟环境：

```powershell
cd D:\data\python\github\wlxq
.\.venv\Scripts\Activate.ps1
```

激活后 `wlxq-bot`、`python`、`pytest`、`ruff` 等命令直接可用。

退出虚拟环境：

```powershell
deactivate
```

### 不激活直接用完整路径

如果不想激活，也可以用完整路径调用：

```powershell
D:\data\python\github\wlxq\.venv\Scripts\wlxq-bot.exe inspect
```

---

## CLI 命令

所有命令通过 `wlxq-bot` 入口调用。查看帮助：

```powershell
wlxq-bot --help
```

### version：显示版本号

```powershell
wlxq-bot version
```

### inspect：检查游戏窗口信息

用于查看微信小游戏窗口的句柄、标题、类名、客户区尺寸、DPI 和前台状态。这是截图和坐标换算的前置工具。

```powershell
# 按关键字「蔚蓝」模糊查找（默认）
wlxq-bot inspect

# 指定其他关键字
wlxq-bot inspect -k 微信

# 精确匹配窗口标题，显示完整详情
wlxq-bot inspect -t "永远的蔚蓝星球"

# 列出所有可见顶层窗口（排查问题时用）
wlxq-bot inspect --all
```

输出示例：

```
────────── 窗口详情: 永远的蔚蓝星球 ──────────
  句柄          656734
  标题          永远的蔚蓝星球
  类名          Chrome_WidgetWin_0
  进程 ID       4028
  线程 ID       8108
  可见 / 前台 / 最小化  是 / 否 / 否
  DPI           96 (100% 缩放)
  客户区尺寸    924 × 1723 px  非 16:9
  客户区屏幕位置  (2050, 119)  尺寸 924 × 1723
  窗口矩形      (2037, 119) → (2987, 1855)  尺寸 950 × 1736
  边框/标题栏    左 13, 右 13, 上 0, 下 13
  目标尺寸匹配  ✗ 不匹配，最接近 1920x1080
```

**字段说明**：

| 字段 | 含义 |
|------|------|
| 句柄 | Windows 窗口句柄（HWND），程序内部用于操作窗口 |
| 标题 | 窗口标题，配置文件里 `screen.window_title` 用这个 |
| 类名 | 窗口类名，微信小游戏通常是 `Chrome_WidgetWin_0` |
| 客户区尺寸 | 游戏画面实际渲染区域，用于截图和坐标换算 |
| 模板包 | 默认按游戏窗口所在显示器物理分辨率选择，可由 `configs/local.yaml` 显式覆盖 |
| DPI | 显示器缩放比例，96 = 100%，144 = 150% |
| 目标尺寸匹配 | 客户区是否在支持的目标尺寸列表中 |

### screenshot：截取游戏窗口（待实现）

```powershell
wlxq-bot screenshot
```

### find：测试模板识别（待实现）

```powershell
wlxq-bot find assets/templates/924x1723/buttons/example.png
```

### run：执行自动化任务

```powershell
wlxq-bot run coop --main-c assault
# 覆盖 configs/default.yaml 中的 coop_difficulties，本次依次选择10到1
wlxq-bot run coop --main-c assault --coop-difficulties 1-10
# 覆盖最大局数（默认 20），只打 3 局；短参数 -n 等价
wlxq-bot run coop --main-c assault --max-rounds 3
wlxq-bot run coop --main-c assault -n 3
# 游戏本次会话已手动选过难度：首局招募跳过难度弹窗直接抢合作（简写 -d）
wlxq-bot run coop --main-c assault --skip-difficulty-selection
wlxq-bot run coop --main-c assault -d
```

`coop_difficulties` 默认从 `configs/default.yaml` 读取，编号一律指彩虹难度（合作模式-彩虹N层，最高 19；文字模板目前采集到 18），支持 `1-16`、`1-10` 或单个难度；程序统一按从小到大的顺序点击（面板打开停在列表顶部、编号小的先出现，单程点完）。开环点击不复核勾选框，已选过难度的会话请配合 `--skip-difficulty-selection`，否则再次点击会把已选难度取消勾选。难度面板每次打开都定位在普通难度区（第1~12层），脚本会先从下往上滚动进入彩虹区再勾选（2026-08-19 游戏规则）。命令行 `--coop-difficulties` 只覆盖本次运行，不修改配置文件。

`--max-rounds` / `-n` 覆盖 `run.max_rounds`（默认 20），同样只对本次运行生效，不修改配置文件。

`--skip-difficulty-selection` 用于游戏启动后已经手动选过难度的场景：首局“首页聊天 → 招募 → 打开难度弹窗”之后跳过“勾选难度等级”，直接关闭弹窗进入抢合作——弹窗打开再关闭才会刷新出最新合作邀请，因此这一开一关不会被跳过。启动检查也不再要求难度模板。等价的持久配置是 `configs/default.yaml` 的 `run.skip_difficulty_selection: true`。后续局的“合作页聊天 → 开/关难度弹窗刷新邀请”维持原流程。

### exec：独立验证自动化能力

`exec` 是可独立执行能力的统一命令组，避免为每个调试能力增加新的顶层命令。选择难度前，先手动进入招募页面并打开难度弹窗，然后运行：

```powershell
wlxq-bot exec select-difficulty --coop-difficulties 1-10
```

该命令复用正式合作任务中的难度识别、点击和比例滚动逻辑。它只按 `10 → 9 → … → 1` 处理目标难度，完成后保持弹窗打开，不会继续关闭弹窗或抢合作。单个难度识别不到时会在有限滚动后记录并跳过，全部目标处理完仍正常结束；窗口、安全检查或动作执行失败时返回非零退出码。可在命令前加 `--debug` 查看选中与跳过汇总。

### save-window：保存当前窗口尺寸

把当前游戏窗口的客户区尺寸和模板包保存到 `configs/local.yaml`（不提交 git）。后续 `adjust-window` 会读取尺寸配置，Runner 会按 `template_pack` 加载素材。

`run` 启动时若客户区与本机配置不一致，默认会**自动调整**到目标尺寸（`run.auto_adjust_window: true`，等价自动执行 adjust-window）；关闭该开关后退回旧行为——报错停止并提示手动执行 `wlxq-bot adjust-window`。

游戏客户区固定为 9:16，合作流程的命名点击位置统一使用客户区比例坐标，保存在可提交的 `configs/tasks.yaml` 的 `hotspots` 段。用 `pick` 量测后直接更新共享任务配置；未标定的位置保持缺失，相关动作会保守停止。`configs/local.yaml` 只保存本机窗口规格和模板包覆盖。

旧版 `configs/local.yaml` 如果已有 `hotspots`，需将该段迁移到 `configs/tasks.yaml`；Runner 和命名位置诊断命令不再读取本机热点。

未传 `--template-pack` 时，命令读取游戏窗口所在显示器的物理分辨率，并写入同名模板包（例如 `3000x2000`）。传入 `--template-pack` 可显式覆盖。程序不会使用主显示器、客户区尺寸或其他分辨率模板包兜底。

```powershell
# 保存当前窗口实际尺寸作为目标
wlxq-bot save-window

# 指定目标尺寸（覆盖当前值）
wlxq-bot save-window -w 927 -h 1727

# 指定窗口标题
wlxq-bot save-window -t "永远的蔚蓝星球"

# 明确指定当前环境使用的模板包
wlxq-bot save-window --template-pack 3000x2000
```

保存后 `configs/local.yaml` 内容示例：

```yaml
window:
  title: 永远的蔚蓝星球
  class_name: Chrome_WidgetWin_0
  target_client_width: 927
  target_client_height: 1727
  template_pack: 3000x2000
```

### adjust-window：按配置调整窗口尺寸

读取 `configs/local.yaml`，把游戏窗口调整到目标客户区尺寸。每次启动游戏后执行一次，确保截图识别使用同一套模板。

```powershell
# 用默认配置调整
wlxq-bot adjust-window

# 指定配置文件
wlxq-bot adjust-window -c configs/local.yaml
```

> **注意**：微信小游戏窗口通常以管理员权限运行，`adjust-window` 需要在**管理员权限的终端**里执行，否则会报"拒绝访问"。右键点击终端 → "以管理员身份运行"，然后激活 venv 再执行命令。

---

## 开发工作流

### 修改代码

项目使用 editable 模式安装（`pip install -e .`），改了 `src/wlxq_bot/` 下的 Python 文件直接生效，**不需要重新编译或重新安装**。

只有以下情况需要重新执行 `pip install -e ".[dev]"`：

- 修改了 `pyproject.toml`（新增依赖、改入口点）
- 新增了 `src/wlxq_bot/` 下的子目录（需确认有 `__init__.py`）

### 运行测试

```powershell
# 运行全部测试
pytest

# 运行指定测试文件
pytest tests/test_models.py

# 显示详细输出
pytest -v
```

### 代码风格检查

```powershell
# 检查 lint 问题
ruff check src tests

# 自动修复可修复的问题
ruff check --fix src tests

# 格式化代码
ruff format src tests
```

### 提交前检查清单

提交代码前确认：

1. `ruff check src tests` 无错误
2. `ruff format --check src tests` 无需格式化
3. `pytest` 全部通过
4. 改动涉及配置或目录结构时，同步更新 `README.md` 和 `docs/`

---

## 项目目录结构

```
D:\data\python\github\wlxq\
├── .venv\                    # 本地虚拟环境（不提交 git）
├── .gitignore
├── pyproject.toml            # 包元数据 + ruff + pytest 配置
├── requirements.txt          # 运行时依赖
├── configs\
│   ├── default.yaml          # 运行参数、安全策略、主C档案
│   ├── local.yaml.template   # 可复制的本机窗口规格模板
│   └── tasks.yaml            # 通用 hotspot、ROI、任务识别定位器和素材映射
├── assets\
│   └── templates\            # 按分辨率组织的模板图片
├── screenshots\
│   ├── raw\                  # 原始截图（不提交）
│   └── debug\                # 调试标注图（不提交）
├── src\
│   └── wlxq_bot\             # 源码
└── tests\                    # 测试
```

---

## 常见问题

### Q: 提示「无法将 wlxq-bot 项识别为 cmdlet」

虚拟环境没激活。先执行：

```powershell
.\.venv\Scripts\Activate.ps1
```

### Q: 激活时提示「在此系统上禁止运行脚本」

执行一次：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

输入 `Y` 确认。之后无需重复设置。

### Q: inspect 命令找不到游戏窗口

确认游戏窗口已经打开且没有最小化。用 `wlxq-bot inspect --all` 查看所有可见窗口，找到游戏窗口的实际标题后，更新 `configs/default.yaml` 里的 `screen.window_title`。

### Q: 改了代码不生效

确认是 editable 安装：

```powershell
pip show wlxq-bot
```

看 `Location` 是否指向项目 `src` 目录。如果不是，重新执行：

```powershell
pip install -e ".[dev]"
```

### Q: adjust-window 报「拒绝访问」

微信小游戏窗口以管理员权限运行，普通权限的终端无法调整窗口大小。解决方法：

1. 右键点击 PowerShell / Windows Terminal → **以管理员身份运行**
2. 切到项目目录并激活 venv：
   ```powershell
   cd D:\data\python\github\wlxq
   .\.venv\Scripts\Activate.ps1
   ```
3. 再执行：
   ```powershell
   wlxq-bot adjust-window
   ```

`inspect` 和 `save-window` 不需要管理员权限，只有 `adjust-window` 需要。
