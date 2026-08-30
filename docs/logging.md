# 日志约定

本文档记录项目的日志体系、级别定义、命令行开关和代码落地规范。AGENTS.md 只保留最核心的几条规则，详细要求以本文档为准。

## 技术栈与入口

- 使用 Python 标准 `logging` + `rich.logging.RichHandler`，统一在 `wlxq_bot` 命名空间下。
- 入口配置在 `src/wlxq_bot/utils/log.py`：`setup_logging(level)` 初始化根日志器，`get_logger(name)` 获取子日志器。
- CLI 通过 `@app.callback` 在命令开始时调用一次 `setup_logging`。

## 日志级别

| 级别 | 用途 | 何时输出 |
|------|------|---------|
| `ERROR` | 动作执行失败、识别失败导致任务暂停、安全触发、窗口句柄失效等不可继续的情况 | 始终 |
| `WARNING` | 接近阈值、重试、窗口变化可恢复、配置缺失用默认值、坐标被夹紧到客户区边界 | 始终 |
| `INFO` | 正常流程关键节点：命令开始/结束、任务启动、状态迁移、动作完成、截图保存路径 | 始终 |
| `DEBUG` | 详细诊断：窗口句柄值、客户区像素、坐标换算过程、置信度数值、ROI 像素、延迟值、模板路径、`frame_id` 变化 | 仅 `--debug` |

## 命令行开关

`--debug`（简写 `-v`）是全局选项，放在子命令之前：

```bash
wlxq-bot --debug screenshot
wlxq-bot --debug inspect --all
wlxq-bot --debug run coop --main-c assault
```

合作任务的主 C 培养阶段会在每批棋盘识别结束后输出一条 DEBUG 汇总日志，包含配置帧数、截图成功/失败数、整批总耗时，以及单帧截图和模板匹配的平均/最小/最大耗时。例如：

```text
棋盘多帧识别耗时 配置帧数=10 截图成功=10 实际识别=10 截图失败=0 总耗时=8.421s 单帧总耗时(avg/min/max)=0.839/0.801/0.912s 单帧截图平均=0.018s 单帧模板匹配(avg/min/max)=0.821/0.783/0.894s 最终英雄=7 最新frame_id=128
```

帧数通过 `configs/default.yaml` 的 `run.board_recognition_frames` 调整，允许范围为 1～60。日志按批次汇总，不在逐帧循环中刷屏。

培养决策还有两条 INFO 级业务日志：棋盘内容发生变化时输出一次各格英雄汇总（`棋盘识别 frame=N 占用=x/y: assault1星@1A, angel2星@3B, ...`，格名按玩家自身棋盘从外到内 A/B/C，同一签名不重复输出；格名缺失时退回像素坐标）；决定合成时输出拖动动作（`拖动合成非主C对: monkey1星 2A -> 5B`）。关闭击杀奖励弹窗的点击在动作后验证失败时按 `run.close_popup_max_retries` 重新点击，每次重试输出 WARNING，重试耗尽仍失败才以 ERROR 结束任务；合成拖动验证失败时输出棋盘诊断 WARNING（起点英雄是否仍在、终点格星级、英雄数变化）并按 `run.merge_max_retries` 重新决策重试。

合作任务的主循环决策日志带局号：每条 `frame=N 局=X/Y state -> state action=...` 中的 `局=X/Y` 是当前进行中的局（已完成局数 +1 / 最大局数）；每局结算返回验证通过后输出一条 `第 X 局完成（X/Y），单局步数=N`，任务完成时输出 `达到 COMPLETED，任务完成，共完成 X/Y 局`。被踢回首页、退队等本局未开始的情形不增加局数。

「截图 → 识别 → 决策」全程超过 `safety.frame_ttl_ms` 时该轮决策被丢弃、重新截图重试，输出一条带分段耗时的 WARNING（`截图至动作执行耗时 X.XXs 超过 frame_ttl_ms=...（截图 Xms + 识别 Xms）`）；连续多轮超龄才以 ERROR 保守停止。识别耗时随机器负载波动，这条 WARNING 是定位「动作一直被时效拒绝」问题的第一步。

## 退出帧落盘

任务**非正常退出**（保守停止、失败/步数上限、异常崩溃）时，主循环最近 `vision.exit_frame_buffer_size`（默认 20）张截图会批量保存到 `screenshots/debug/exit_<状态>_<时间戳>/`，并输出一条带完整路径的 WARNING（`任务非正常退出，已保存退出前 N 帧截图到 ...`）；正常完成（COMPLETED）和用户主动停止（Esc 热键、Ctrl+C）不保存。文件夹内 `NN_frame_<id>.png` 按时间升序编号，`frames.txt` 记录每帧的 `frame_id`、拍摄时刻和相邻帧间隔，用于定位退出前卡在哪一步。缓冲常驻内存约 N × 4.8MB（927x1727 客户区，20 张约 96MB，只存引用无复制开销）；按主循环和抢合作检查线程的节奏采样（只保留前台有效帧，挂起/最小化期间的黑帧不冲掉历史），培养阶段多帧棋盘识别的内部截图不进缓冲；`vision.debug: false` 时整体关闭。

它通过 `@app.callback` 调用 `setup_logging("DEBUG")`；不带 `--debug` 时为 `INFO`。DEBUG 模式下第三方库（pyautogui、PIL、urllib3 等）仍压在 `WARNING`，避免刷屏。

## 命名空间与获取

- 所有日志器挂在 `wlxq_bot` 根下，通过 `get_logger(__name__)` 获取；禁止用 `logging.getLogger` 直接取无前缀名，以免脱离统一配置。
- `get_logger` 支持短名（`get_logger("screen")`）和完整模块名（`get_logger(__name__)`），均归一到 `wlxq_bot.*`，不会产生 `wlxq_bot.wlxq_bot.xxx` 重复前缀。
- `setup_logging` 关闭 `propagate`（避免冒泡到 root 重复输出），且幂等（多次调用只更新级别，不重复添加 handler）。

```python
from wlxq_bot.utils.log import get_logger, setup_logging

setup_logging("DEBUG")            # 入口处调用一次
logger = get_logger(__name__)     # 模块级，等价于 wlxq_bot.<module>

logger.info("开始截图，目标窗口: %s", window_title)
logger.debug("坐标换算 客户区=%dx%d 比例=(%.4f,%.4f) 屏幕=(%d,%d)", cw, ch, rx, ry, sx, sy)
logger.error("未找到窗口: %s", title)
```

## 必须打日志的位置

每个命令和任务流程的关键节点都要有日志，不要让任何一步成为黑盒：

| 位置 | 级别 | 记录内容 |
|------|------|---------|
| 命令开始 | `INFO` | 目标、关键参数 |
| 窗口查找结果 | `DEBUG` | 句柄、标题、类名 |
| 坐标换算 | `DEBUG` | 客户区像素、比例→像素、屏幕坐标 |
| 截图 | `DEBUG` | `frame_id`、客户区尺寸、句柄 |
| 识别 | `DEBUG` | ROI 像素、模板路径、置信度、匹配位置 |
| 动作 | `INFO`/`DEBUG` | 动作类型、目标坐标、点击前后状态 |
| 失败 | `ERROR`/`WARNING` | 失败原因、上下文（`handle`/`frame_id`/置信度） |
| 命令结束 | `INFO` | 产出路径、计数、耗时 |

新增或修改命令时，按上表补齐日志。

几点约束：

- 高频循环（如 `spam-click` 的每次点击）不要在循环内打日志，否则 DEBUG 下会刷屏数千行；只在循环开始和结束打。
- `hero-classifier import-rounds/select-candidates` 这类可能处理上万张图片的离线长任务例外：必须输出阶段开始/完成，并用统一节流进度日志约每 5 秒输出一次 `processed/total`、百分比、已耗时和阶段指标；禁止逐图输出。Rich 日志前缀必须显示当前本地输出时间。
- 纯决策逻辑（状态机、配置校验）的 DEBUG 日志应在单元测试中可关闭，避免测试输出噪音。
- `version` 这类只打印版本号的命令可以不打日志；`screenshot`、`run`、`click`、`spam-click`、`move` 等执行类命令必须补齐。

## 禁止记录

- 账号、令牌、密码、会话 cookie、支付信息。
- 真实账号截图中的昵称、ID 等可关联到个人的信息；调试截图需先脱敏。

## 与用户面向输出的关系

- `rich.print` / `rprint`：命令的最终结果展示（截图路径、窗口信息表、进度提示），面向终端用户。
- `logger.*`：执行过程诊断信息，面向开发者排查问题。

两者职责不同，不互相替代：`rprint` 输出最终结果，`logger` 记录过程；不要把用户面向的结果只打到日志里，也不要把每一步诊断都 `rprint` 刷屏。同一条信息若既要给用户看、又要留排查记录，分别用 `rprint` 和 `logger` 各打一次即可。

## 实现状态

已按本约定补齐日志的命令：`screenshot`、`inspect`、`save-window`、`adjust-window`、`click`、`spam-click`、`pick`、`move`、`recognize`、`find`、`run`、`exec select-difficulty`，以及 `hero-classifier collect/crop/import-rounds/select-candidates/sync-labels/train/evaluate`。后续新增命令照此模板补。

英雄格离线流程当前可见阶段包括：多局/单局裁切、阈值 35 一级聚类、一级簇结果整理、超过门槛的大簇阈值 15 二次细分、候选视觉描述读取与多样化选择、候选复制和清单完成。快于 5 秒的阶段只显示开始/完成，慢阶段在处理中持续显示进度。
