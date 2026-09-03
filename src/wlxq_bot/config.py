"""配置加载与校验。

使用 Pydantic 模型对 YAML 配置进行结构化校验，避免任务代码传递裸字典。
配置分三层：
- configs/default.yaml：运行参数、安全策略、主 C 档案
- configs/tasks.yaml：任务通用坐标、定位器、ROI、技能和英雄模板
- configs/local.yaml：仅保存本机窗口规格和模板包覆盖

加载时校验召唤后识别等待区间等跨字段约束。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


def parse_coop_difficulties(value: str | list[int]) -> list[int]:
    """解析合作难度范围（编号一律指彩虹难度），统一返回从小到大的难度列表。"""
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d{1,2})(?:\s*-\s*(\d{1,2}))?\s*", value)
        if match is None:
            raise ValueError("coop_difficulties 格式应为 1-16、1-10 或单个难度")
        first = int(match.group(1))
        second = int(match.group(2) or first)
        low, high = sorted((first, second))
        levels = list(range(low, high + 1))
    elif isinstance(value, list):
        levels = sorted(set(value))
    else:
        raise ValueError("coop_difficulties 必须是范围字符串或整数列表")

    if not levels or any(isinstance(level, bool) or not 1 <= level <= 19 for level in levels):
        raise ValueError("coop_difficulties 中的难度必须在 1～19 之间")
    return levels


class ScreenConfig(BaseModel):
    mode: str = "window"
    window_title: str = "永远的蔚蓝星球"
    screenshot_dir: str = "screenshots/raw"
    target_sizes: list[dict[str, int]] = Field(default_factory=list)


class VisionConfig(BaseModel):
    default_threshold: float = 0.85
    debug: bool = True
    debug_dir: str = "screenshots/debug"
    max_consecutive_failures: int = 5
    # 任务非正常退出时落盘的主循环截图张数（环形缓冲）；0 关闭
    exit_frame_buffer_size: int = Field(default=20, ge=0, le=60)


class InputConfig(BaseModel):
    click_duration: float = 0.08
    # 每次输入动作后的拟人随机间隔（秒）：导航类点击（首页聊天/招募/开弹窗等）
    # 连续操作太快不像人（2026-08-21 实机反馈，实机调定为 0.8~1.5）。
    # 抢合作/召唤等有自己的节奏配置，不受这里影响
    min_delay: float = 0.8
    max_delay: float = 1.5
    drag_duration: float = 0.5
    drag_pause: float = 0.2


class SafetyConfig(BaseModel):
    stop_hotkey: str = "esc"
    max_failures: int = 5
    frame_ttl_ms: int = 3000
    no_progress_timeout: int = 30


class MainCProfile(BaseModel):
    display_name: str
    hero_template_dir: str
    # 正式棋盘识别使用的 ONNX 模型；不同主 C 可分别训练，公共英雄样本可复用。
    hero_classifier_model: str = ""
    # 简化版技能识别：主C技能卡上的英雄图标模板（相对模板包路径）。
    # 同一英雄的所有技能卡共用同一张图标，在技能 ROI 匹配它即识别到主C技能。
    skill_icon_templates: list[str] = Field(default_factory=list)
    # 合成时尽量不合并主C：没有非主C合法对且棋盘有空位时优先召唤新英雄，
    # 而不是把两个主C合成一个（保主C数量）。棋盘占满时仍合并主C对腾格子。
    avoid_main_c_merge: bool = False
    # 局内选满该次数技能后（>0 启用），回培养阶段把主C数量补到
    # topup_hero_count 以上，补够后回到选技能。0 = 不启用（默认策略：
    # 2星主C出现后一直选技能到对局结束）
    topup_after_skill_selections: int = Field(default=0, ge=0, le=100)
    topup_hero_count: int = Field(default=0, ge=0, le=12)
    # 局内（不含开局赠送）技能选择的总次数上限：达到后不再花金币选技能，等待
    # 对局结束进结算。0 = 不限制（一直选到对局结束）。
    # 强袭为 4（回补前）+ 5（回补后）= 9（2026-08-16 实机规则）
    skill_selection_cap: int = Field(default=0, ge=0, le=100)


class SkillCollectionConfig(BaseModel):
    """技能卡自动采集（统计阶段）配置。

    采集只在统计阶段打开：英雄技能固定，采齐后即可关闭。运行时只做
    裁剪和哈希（毫秒级、节流），英雄归属与文字 OCR 全部在离线建册
    （``build-skill-catalog``）时进行；采集器不参与任何业务决策，
    内部异常只记日志并自动熔断，写盘走后台线程，绝不阻塞正常对局。
    """

    enabled: bool = False
    # 卡图与元数据输出目录（位于 gitignore 的 datasets/ 下，不提交仓库）
    output_dir: str = "datasets/skill_cards"
    # 采集器连续内部失败达该次数后自动停用（只记日志，不影响主流程）
    fuse_max_consecutive_failures: int = Field(default=5, ge=1, le=100)
    # 两次采集的最小间隔（秒）：技能页是静态的，采一次就够；节流保证采集
    # 开销不会侵蚀主循环的截图时效预算（safety.frame_ttl_ms）
    min_collect_interval_seconds: float = Field(default=30.0, ge=0.0, le=600.0)
    # 写盘队列容量；队列满时丢弃新采集并计数，绝不阻塞主循环
    queue_maxsize: int = Field(default=64, ge=8, le=4096)
    # ---- 以下仅离线建册（build-skill-catalog）使用，运行时不读 ----
    # 英雄名 → 技能卡上的英雄图标模板（相对模板包路径）。
    # 名单外的卡按 unknown 归档，离线人工补标
    hero_icons: dict[str, list[str]] = Field(default_factory=dict)
    # 离线图标匹配置信度低于该值的卡归为 unknown
    min_icon_confidence: float = Field(default=0.70, ge=0.0, le=1.0)


class RunConfig(BaseModel):
    max_rounds: int = 20
    # Runner 的单局最终保险。每当返回验证成功、round_count 增加时重置，
    # 防止异常状态无限循环，但不限制多局任务的累计步数。
    max_steps_per_round: int = Field(default=5000, ge=1, le=100000)
    minimum_summon_count_before_skills: int = Field(default=5, ge=1, le=100)
    initial_board_capacity: int = 7
    target_star_level: int = 2
    summon_recognition_delay_min: float = Field(default=1.0, ge=0.0, le=10.0)
    summon_recognition_delay_max: float = Field(default=2.0, ge=0.0, le=10.0)
    board_recognition_frames: int = Field(default=10, ge=1, le=60)
    default_main_c: str = "assault"
    coop_difficulties: list[int] = Field(default_factory=lambda: list(range(1, 17)))
    # 游戏本次会话已手动选过难度时置 true：首局招募跳过勾选难度等级；
    # 难度弹窗仍开/关一次刷新最新合作邀请
    skip_difficulty_selection: bool = False
    # run 启动时客户区与本机配置（configs/local.yaml）不一致则自动调整窗口尺寸
    # （等价 adjust-window 命令）；关闭后退回旧行为：报错停止并提示手动调整
    auto_adjust_window: bool = True
    # 抢合作双线程：抢合作线程连点 join_coop 的拟人间隔（秒）
    find_coop_click_delay_min: float = Field(default=0.3, ge=0.0, le=10.0)
    find_coop_click_delay_max: float = Field(default=0.5, ge=0.0, le=10.0)
    # 检查线程识别准备按钮的间隔（秒）
    find_coop_check_interval_seconds: float = Field(default=1.0, gt=0.0, le=30.0)
    # 整段抢合作的最长时长（秒），超时未抢到则保守停止，防止无限循环
    find_coop_max_duration_seconds: float = Field(default=300.0, gt=0.0, le=3600.0)
    # 单个目标难度的最大滚动次数（命中或跳过后按需重置）。2026-08-19 起难度
    # 面板每次打开都定位在普通难度区（第1~12层），识别为空=尚未滚入彩虹区，
    # 会直接向更大难度滚动，预算需覆盖滚过普通区进入彩虹区的行程
    difficulty_max_scrolls: int = Field(default=15, ge=1, le=50)
    # 难度弹窗打开确认（2026-08-21 实机定案，2026-08-25 实机调 4→2）：点击打开
    # 按钮后短暂固定等待（等点击生效即可，打开慢由失败预算兜底），再按轮询间隔
    # 识别【合作模式】标题图；连续确认次数达标（间隔×次数 = 连续可见时长，
    # 默认 0.25×2 = 0.5 秒）才算打开。
    # 连续未命中达标（默认 0.25×16 = 4 秒不出现）视为本次点击未生效 → 重点打开
    # 按钮，重试用尽保守停止
    difficulty_open_settle_seconds: float = Field(default=0.5, gt=0.0, le=30.0)
    difficulty_poll_interval_seconds: float = Field(default=0.25, gt=0.0, le=5.0)
    difficulty_open_confirm_hits: int = Field(default=2, ge=1, le=10)
    difficulty_open_fail_misses: int = Field(default=16, ge=1, le=60)
    difficulty_open_max_reclicks: int = Field(default=3, ge=0, le=10)
    # 难度弹窗关闭点击的最大次数：标题图可见说明没关住就再点（实机确认：面板
    # 已收起时再点招募按钮无反应，重复点击无副作用）；达到上限仍可见则保守停止
    # （弹窗挡住 join_coop 会让抢合作无效）
    difficulty_close_max_attempts: int = Field(default=5, ge=1, le=20)
    # 确认难度弹窗关闭所需的连续标题图不可见帧数（打开确认在前，关闭时弹窗
    # 必然已打开过，统一按连续不可见判定，无单帧捷径；2026-08-25 实机调 4→2）
    difficulty_close_confirm_frames: int = Field(default=2, ge=1, le=10)
    # 单次拖拽耗时（秒）。触屏式列表快速拖动带惯性会甩过头，慢拖近似无惯性，
    # 每次只推进一小段，避免从 16 直接滑到 3 这类跳级。
    difficulty_scroll_duration: float = Field(default=5.0, gt=0.0, le=20.0)
    ready_wait_seconds: float = Field(default=3.0, ge=0.0, le=30.0)
    # 技能识别多帧重试：图标在动，连续这么多帧没识别到主C图标才改「随便选一个」
    skill_recognition_frames: int = Field(default=10, ge=1, le=60)
    # 确认开局技能选完所需的连续「页面已关闭」帧数；每次选完一张卡后页面会先
    # 关闭、下一组技能卡再弹出，间隙里召唤按钮会短暂露出，单帧不能判定选完
    opening_exit_confirm_frames: int = Field(default=3, ge=1, le=10)
    # 开局技能最多出现的选择次数（已实机确认最多 3 次，不会再多）；选满后召唤按钮
    # 一出现即结束开局阶段，无需再等退出确认帧
    opening_skill_max_selections: int = Field(default=3, ge=1, le=10)
    # 局内等待主C技能图标时，连续这么多帧【选技能】页面不在才认定页面已关闭
    # （页面可能被击杀奖励弹窗打断关闭），回到定时点选技能的节奏，避免盲点技能卡
    main_skill_page_closed_frames: int = Field(default=3, ge=1, le=10)
    skill_early_interval_min: float = Field(default=4.5, gt=0.0)
    skill_early_interval_max: float = Field(default=6.0, gt=0.0)
    skill_late_interval_min: float = Field(default=9.0, gt=0.0)
    skill_late_interval_max: float = Field(default=11.0, gt=0.0)
    skill_late_after_seconds: float = Field(default=120.0, gt=0.0)
    # 其他队友英雄的技能卡图标（本局阵容 lineup_others 对应）：技能页已打开但
    # 未识别到主C图标时，识别到任一队友图标即确认页面稳定且本组无主C技能卡，
    # 立即随机选一张，不再等满 skill_recognition_frames 帧
    teammate_skill_icon_templates: list[str] = Field(default_factory=list)
    # 技能卡兜底点击前页面至少已出现这么多帧（配合队友图标快捷兜底使用）：
    # 页面刚弹出时图标可能先于卡片渲染出来，点早了会点空（实机 2026-08-16
    # 开局第一组技能卡出现过：首帧即点 → 验证失败重试多花约 12 秒）
    skill_fallback_settle_frames: int = Field(default=1, ge=0, le=10)
    action_verify_frames: int = Field(default=5, ge=1, le=30)
    # 关闭击杀奖励弹窗动作后验证失败时的整动作重试上限；连杀常会关掉后立刻弹新弹窗，
    # 直接退出太激进。重试耗尽仍失败才保守停止。
    close_popup_max_retries: int = Field(default=4, ge=0, le=20)
    # 技能卡点击（开局/局内选技能）验证失败的整动作重试上限；重试耗尽后不结束任务
    # （对局仍在自动进行，结算窗口终将出现），改为通知任务放弃本次选择、等待对局推进
    skill_click_max_retries: int = Field(default=3, ge=0, le=20)
    # 等待开局期间连续这么多帧识别到首页标记，才判定本局被取消/被踢回首页
    # （加载画面可能与首页有短暂相似，单帧命中不能直接判定）
    home_return_confirm_frames: int = Field(default=2, ge=1, le=10)
    # 点准备后等待对局界面的超时（秒）；超时按本局未开始处理，回到招募入口
    # 重新抢合作。实机确认房主可踢出已加入的玩家，被踢后回到首页、对局不开始
    match_start_timeout_seconds: float = Field(default=120.0, gt=0.0, le=3600.0)
    # 点准备后在组队大厅识别到【退队】按钮，且等待房主开始超过该秒数时，
    # 主动点击退队退出本队重新抢合作（房主不开始时干等没有意义）
    leave_team_after_seconds: float = Field(default=20.0, gt=0.0, le=600.0)
    # 窗口非前台/最小化时的最长挂起等待（秒）：期间任务不发送任何输入（点击会
    # 落到当前前台窗口，必须挂起），游戏对局自动进行，切回窗口后从当前进度
    # 继续。超过该时长才保守停止；想临时离开更久可调大
    window_foreground_wait_seconds: float = Field(default=300.0, gt=0.0, le=3600.0)
    # 窗口非前台/最小化挂起期间，系统鼠标/键盘无输入超过该秒数时，自动把
    # 游戏窗口切回前台继续任务（用户在操作电脑时不抢焦点，保持挂起等待）
    refocus_when_idle: bool = True
    refocus_idle_seconds: float = Field(default=20.0, gt=0.0, le=600.0)
    # 领取结算宝箱后、点击返回按钮前的稳定等待（秒）；实机确认点宝箱后立刻点返回
    # 游戏不响应，需等结算动画播完（约 4 秒）再点
    reward_claim_return_delay: float = Field(default=4.0, ge=0.0, le=30.0)
    # 合成拖动验证失败时的重试上限；拖动可能落空（英雄仍在原位），
    # 重新决策后通常直接再拖一次即可。重试耗尽仍失败才保守停止。
    merge_max_retries: int = Field(default=3, ge=0, le=10)
    # 合成拖动的时长（秒）。拖太快时游戏内英雄跟随不及，松开即被弹回原格；
    # 1 秒左右的慢拖成功率稳定（InputController 另在终点停留后再松开）
    merge_drag_duration: float = Field(default=1.0, gt=0.0, le=10.0)
    # 合成4星赠送技能页确认（2026-08-21 实机定案）：被迫合并 3 星对后，先固定
    # 等待页面渲染，再按轮询间隔识别【请选择1个额外技能】提示条；连续 N 次
    # 命中才选技能（半渲染/动画过渡帧不动作）；连续 M 次未命中视为拖动未生效
    # 或对局已结束（页面随结算消失），放弃等待恢复常规决策。等待期间击杀奖励
    # 弹窗仍优先关闭，返回按钮出现即正常转入结算
    merge_gift_settle_seconds: float = Field(default=2.0, gt=0.0, le=30.0)
    merge_gift_poll_interval_seconds: float = Field(default=0.5, gt=0.0, le=5.0)
    merge_gift_confirm_hits: int = Field(default=2, ge=1, le=10)
    merge_gift_fail_misses: int = Field(default=12, ge=1, le=60)
    # 技能解锁前强制召唤阶段（前 minimum_summon_count_before_skills 次）的连点
    # 间隔（秒）：该阶段不以棋盘识别为门禁（点击发送即计数），跳过多帧棋盘识别
    # 快速连点；最后一次保留正常稳定等待后再进入棋盘决策
    forced_summon_interval_seconds: float = Field(default=0.5, gt=0.0, le=10.0)
    # 召唤点击后棋盘始终未变化（多为金币不足，点击被游戏忽略）时的重试等待
    # （秒）：等待后重新尝试召唤。回补阶段的召唤失败不等待，直接放弃回补回选技能
    summon_retry_delay_seconds: float = Field(default=15.0, gt=0.0, le=300.0)
    # 技能卡自动采集（统计阶段）：默认关闭，见 SkillCollectionConfig
    skill_collection: SkillCollectionConfig = Field(default_factory=SkillCollectionConfig)

    @field_validator("coop_difficulties", mode="before")
    @classmethod
    def _parse_coop_difficulties(cls, value: Any) -> list[int]:
        return parse_coop_difficulties(value)

    @model_validator(mode="after")
    def _check_ranges(self) -> RunConfig:
        if self.summon_recognition_delay_min > self.summon_recognition_delay_max:
            raise ValueError("summon_recognition_delay_min 不能大于 summon_recognition_delay_max")
        if self.skill_early_interval_min > self.skill_early_interval_max:
            raise ValueError("skill_early_interval_min 不能大于 skill_early_interval_max")
        if self.skill_late_interval_min > self.skill_late_interval_max:
            raise ValueError("skill_late_interval_min 不能大于 skill_late_interval_max")
        if self.find_coop_click_delay_min > self.find_coop_click_delay_max:
            raise ValueError("find_coop_click_delay_min 不能大于 find_coop_click_delay_max")
        return self


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


class DefaultConfig(BaseModel):
    """configs/default.yaml 的完整模型。"""

    screen: ScreenConfig = Field(default_factory=ScreenConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    main_c_profiles: dict[str, MainCProfile] = Field(default_factory=dict)
    hero_classifier: HeroClassifierConfig = Field(default_factory=HeroClassifierConfig)

    @model_validator(mode="after")
    def _check_lineup_others_subset(self) -> DefaultConfig:
        if self.main_c_profiles and self.hero_classifier.lineup_others:
            known = set(self.main_c_profiles)
            unknown = [h for h in self.hero_classifier.lineup_others if h not in known]
            if unknown:
                raise ValueError(
                    f"hero_classifier.lineup_others 引用了未定义的英雄: {unknown}；"
                    f"已知英雄: {sorted(known)}"
                )
        return self


class RoiConfig(BaseModel):
    """ROI 区域配置（比例坐标）。

    使用客户区比例坐标定义识别区域，运行时按实际客户区尺寸换算为
    物理像素（见 locator.ratio_to_pixel_roi）。英雄识别等动态内容
    只在指定棋盘 ROI 内匹配，不对整个游戏界面做识别。

    relative_to 目前固定为 client（相对客户区左上角），保留字段以兼容
    architecture.md 中 client/anchor/candidate 三种参照系的后续扩展。

    Attributes:
        relative_to: 参照系，目前仅支持 client
        x_ratio: 左上角 x 比例 [0, 1]
        y_ratio: 左上角 y 比例 [0, 1]
        width_ratio: 宽度比例 [0, 1]
        height_ratio: 高度比例 [0, 1]
    """

    relative_to: str = "client"
    x_ratio: float = Field(ge=0.0, le=1.0)
    y_ratio: float = Field(ge=0.0, le=1.0)
    width_ratio: float = Field(ge=0.0, le=1.0)
    height_ratio: float = Field(ge=0.0, le=1.0)


class BoardGridParams(BaseModel):
    """棋盘格子坐标模型参数（配置层）。

    对应 models.BoardGridConfig，由 pick --rect 量测排1/排2 格子后
    计算得出。知道这些参数即可推导全部 14 格中心坐标。

    Attributes:
        anchor_x_ratio: 棋盘逻辑左上角 x 比例（列0排1格左上角）
        anchor_y_ratio: 棋盘逻辑左上角 y 比例
        col_step_ratio: 列步长比例（格宽+列间距）
        row_step_ratio: 排步长比例（格高+排间距）
        cell_width_ratio: 格子宽度比例
        cell_height_ratio: 格子高度比例
    """

    anchor_x_ratio: float
    anchor_y_ratio: float
    col_step_ratio: float
    row_step_ratio: float
    cell_width_ratio: float
    cell_height_ratio: float


class Hotspot(BaseModel):
    """任务命名位置（相对 9:16 游戏客户区的通用比例坐标）。"""

    x_ratio: float = Field(ge=0.0, le=1.0)
    y_ratio: float = Field(ge=0.0, le=1.0)
    description: str = ""


class TasksConfig(BaseModel):
    """configs/tasks.yaml 的完整模型。

    hotspots 已结构化为 Hotspot（通用比例坐标）；
    rois 已结构化为 RoiConfig（比例坐标 + 范围校验）；
    board 已结构化为 BoardGridParams（棋盘格子坐标模型）；
    locators/skills/heroes 暂保持原始字典，随实现推进再逐步结构化。
    """

    tasks: dict[str, dict[str, Any]] = Field(default_factory=dict)
    hotspots: dict[str, Hotspot] = Field(default_factory=dict)
    locators: dict[str, dict[str, Any]] = Field(default_factory=dict)
    rois: dict[str, RoiConfig] = Field(default_factory=dict)
    board: dict[str, BoardGridParams] = Field(default_factory=dict)
    skills: dict[str, Any] = Field(default_factory=dict)
    difficulty_recognition: dict[str, Any] = Field(default_factory=dict)
    # 技能卡采集（统计阶段）的几何参数：比例相对 rois.skill_candidates 裁剪结果
    skill_collection: dict[str, Any] = Field(default_factory=dict)
    heroes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _none_to_empty_dict(cls, data: Any) -> Any:
        """YAML 里空值会被解析为 None，统一转成空字典。"""
        if isinstance(data, dict):
            for key in (
                "tasks",
                "hotspots",
                "locators",
                "rois",
                "board",
                "skills",
                "difficulty_recognition",
                "skill_collection",
                "heroes",
            ):
                if data.get(key) is None:
                    data[key] = {}
        return data


def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 文件为字典。"""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件 {path} 顶层必须是字典，实际为 {type(data).__name__}")
    return data


def load_default_config(path: Path) -> DefaultConfig:
    """加载并校验 configs/default.yaml。"""
    return DefaultConfig(**load_yaml(path))


def load_tasks_config(path: Path) -> TasksConfig:
    """加载并校验 configs/tasks.yaml。"""
    return TasksConfig(**load_yaml(path))


# ---------------------------------------------------------------------------
# 本地窗口配置（configs/local.yaml，不提交 git）
# ---------------------------------------------------------------------------


class WindowSpec(BaseModel):
    """目标窗口规格。

    记录游戏窗口的查找方式和期望的客户区尺寸。
    adjust-window 命令据此调整窗口大小。
    """

    title: str
    class_name: str = ""
    target_client_width: int
    target_client_height: int
    template_pack: str = Field(default="", pattern=r"^[A-Za-z0-9._-]*$")

    @model_validator(mode="after")
    def _check_template_pack_name(self) -> WindowSpec:
        if self.template_pack in {".", ".."}:
            raise ValueError("template_pack 必须是 assets/templates 下的模板包目录名")
        return self


class LocalConfig(BaseModel):
    """configs/local.yaml 的完整模型。

    只存储本机窗口校准结果，不提交 git（已在 .gitignore 中）。
    任务坐标统一存入可提交的 configs/tasks.yaml。
    """

    window: WindowSpec


def load_local_config(path: Path) -> LocalConfig | None:
    """加载 configs/local.yaml，文件不存在时返回 None。"""
    if not path.exists():
        return None
    return LocalConfig(**load_yaml(path))


def save_local_config(path: Path, config: LocalConfig) -> None:
    """保存 configs/local.yaml。"""
    data = config.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
