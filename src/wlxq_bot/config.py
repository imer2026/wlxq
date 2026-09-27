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
from typing import Any, Literal

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
    # 每次输入动作后的拟人随机间隔（秒）：导航类点击（首页聊天/招募/开弹窗、
    # 返回/聊天/难度弹窗等）。2026-09-05 实机调定为 1.0~2.0(区间更宽更像人)。
    # 抢合作/召唤等有自己的节奏配置，不受这里影响
    min_delay: float = 1.0
    max_delay: float = 2.0
    drag_duration: float = 0.5
    drag_pause: float = 0.2


class SafetyConfig(BaseModel):
    stop_hotkey: str = "esc"
    max_failures: int = 5
    frame_ttl_ms: int = 3000
    # 慢机自适应上限：识别耗时持续超过 frame_ttl_ms 时，生效时效翻倍放宽、
    # 封顶该值——机器慢只降低节奏，不触发保守停止；决策重新在基础时效内
    # 新鲜后自动回落。识别比该上限还慢才走「超龄丢弃 → 连续超龄保守停止」
    frame_ttl_max_ms: int = 90000
    no_progress_timeout: int = 30


class LoggingConfig(BaseModel):
    """文件日志配置（configs/default.yaml ``logging`` 段）。

    文件日志比控制台详细：每次命令调用在 ``dir`` 下写一个完整日志文件，
    固定 ``file_level`` 级别（默认 DEBUG），与控制台级别和 ``--debug`` 无关。
    """

    enabled: bool = True
    dir: str = "logs"
    file_level: str = "DEBUG"
    retention_days: int = Field(default=7, ge=1)


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
    # 技能选择优先级（标题 OCR 主路径）：档位号 → 技能名数组，数字越小越优先；
    # 同档多张随机点一张。第 8 档（主C其他技能）与第 11 档（清单剩余技能）
    # 启动时自动计算，配置了会被忽略。未配置时技能选择走原有图标识别流程。
    # 技能名需与 configs/skills.yaml 一致
    skill_priority: dict[int, list[str]] = Field(default_factory=dict)
    # 主C未列入 skill_priority 的技能是否优先于其他英雄技能（第 8 档 vs
    # 第 11 档，强袭需要）；false = 主C未列技能与其余技能同档随机——
    # 猴子模型：只有 天使大/雪姬大 两档 + 其他（2026-09-25 用户定案）
    prefer_main_c_skills: bool = True
    # ---- 邀请者模式（主C猴子）扩展字段；默认值 = 强袭现状行为 ----
    # 主C门禁星级档案化；0 = 回退 run.target_star_level
    target_star_override: int = Field(default=0, ge=0, le=4)
    # 开局召唤次数档案化；0 = 回退 run.minimum_summon_count_before_skills
    opening_summons: int = Field(default=0, ge=0, le=20)
    # 凑阵容阶段每批召唤次数
    summon_batch_size: int = Field(default=3, ge=1, le=10)
    # 3 星软避让：不执行任何产物为 3 星的合并（2星同类对/2星通配对一律不选，
    # 棋盘占满也不合——强袭的「占满被迫合并」兜底不生效）
    forbid_star3_merge: bool = False
    # 保留数合成：合并后任一输入英雄类型的在场数量不得低于该保留数。
    # 键为英雄标识（含主C与通配变体），缺失的英雄默认保留 1（主C）/0（其他）。
    # 空 dict = 不启用该规则（强袭现状）
    hero_keep_counts: dict[str, int] = Field(default_factory=dict)
    # 阵容门禁：这些英雄每类在场 ≥1 才允许转选技能；空 = 只看主C星级（强袭现状）
    required_lineup_heroes: list[str] = Field(default_factory=list)
    # 阵容门禁的形态变体等价：键为 required_lineup_heroes 中的英雄，值为可
    # 替代其满足门禁的形态变体类型。幻天使是天使的形态（game-rules.md），
    # 猴子档案配 {angel: [angel_phantom]}——棋盘上有幻天使即算有天使
    lineup_variant_groups: dict[str, list[str]] = Field(default_factory=dict)
    # 变身碰辅助模式：按此顺序选牺牲英雄（取第一个在场数 > 保留数的）赌变出
    # 缺的辅助；触发条件为阵容不齐 + 棋盘满 + 无安全合成。空 = 强袭的碰主C模式
    transform_sacrifice_order: list[str] = Field(default_factory=list)


class DifficultyProfile(BaseModel):
    """难度识别方式（模式档案）。"""

    kind: Literal["template", "ocr"] = "template"
    # ocr 模式的目标难度（行文字子串，如「普通」⊂「保卫宠物-普通」）；配置顺序即点击顺序
    targets: list[str] = Field(default_factory=list)


class EntryProfile(BaseModel):
    """招募入口差异（模式档案）。"""

    # 进招募后先点的模式页签 hotspot 名（configs/tasks.yaml）；None = 不点。
    # 配置了 mode_tab_text 后流程改走 OCR 定位，本值保留作手动 click 与标定记录
    mode_tab_hotspot: str | None = None
    # 页签 OCR 定位文字（如「合作模式」，为三个页签共用识别区的子串匹配目标）；
    # None = 按固定 hotspot 坐标点击（旧行为）
    mode_tab_text: str | None = None
    # 领地入口（2026-09-09 新流程）：模式选择弹窗内本模式卡片与对应挑战页
    # 按钮的识别配置；ROI 名指向 configs/tasks.yaml rois，文字为 OCR 子串
    # 匹配目标。领地入口的启动检查要求四项全部配置
    mode_card_roi: str | None = None
    mode_card_text: str | None = None
    challenge_roi: str | None = None
    challenge_text: str | None = None
    # 挑战页可能弹出的每周弹窗（保卫宠物=本周Buff，每天首次进入弹出）：
    # roi 指向弹窗确认按钮文字区域，text 为 OCR 子串匹配目标。两项须同时
    # 配置或同时缺省；缺省 = 挑战页确认后直接进招募（旧行为）
    weekly_buff_roi: str | None = None
    weekly_buff_text: str | None = None
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
    """按模式档案应用主C覆盖；无覆盖时原样返回共享档案实例。

    未知 main_c 抛 ValueError（CLI 只捕获 ValueError/RuntimeError，
    不能让裸 KeyError 变成未处理 traceback）。
    """
    if main_c not in profiles:
        raise ValueError(f"主C {main_c!r} 不在配置档案中，可用: {sorted(profiles)}")
    base = profiles[main_c]
    overrides = mode.main_c_overrides.get(main_c)
    if not overrides:
        return base
    return MainCProfile.model_validate({**base.model_dump(), **overrides})


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


class RunConfig(BaseModel):
    max_rounds: int = 20
    # Runner 的单局最终保险。每当返回验证成功、round_count 增加时重置，
    # 防止异常状态无限循环，但不限制多局任务的累计步数。
    max_steps_per_round: int = Field(default=5000, ge=1, le=100000)
    # 开局强制召唤次数（填满开局 7 格）；游戏门禁为累计 5 次解锁选技能，
    # 该值是其上的策略配置（2026-09-10 用户决策改为 7）
    minimum_summon_count_before_skills: int = Field(default=7, ge=1, le=100)
    # 批次召唤（2026-09-10 用户策略）：培养阶段棋盘识别后有空位时，按空格数
    # 规划一批召唤填满；每次点击前做召唤费用红/白金币门控，间隔共用
    # forced_summon_interval_seconds，全部点完并稳定等待后重新识别棋盘
    batch_summon_enabled: bool = True
    # 批次合成：一次棋盘识别上把互不相交的非主C 1/2星合法对一批拖完再统一
    # 识别；主C对与3星对不进批次（仍走单动作策略），拖动未生效的对在批末
    # 按位置签名归因记入失败记忆
    batch_merge_enabled: bool = True
    # 通配合并英雄（赖子，如幻天使）：可与任意同星级英雄合并；合成优先级
    # 同类项对在前、通配对棋盘占满才用（2026-09-13 用户定案，spec
    # docs/superpowers/specs/2026-09-13-angel-phantom-split-design.md）
    wildcard_merge_heroes: list[str] = Field(default_factory=list)

    @field_validator("wildcard_merge_heroes", mode="before")
    @classmethod
    def _validate_wildcard_merge_heroes(cls, value: Any) -> list[str]:
        return _parse_hero_identifier_list(value, "wildcard_merge_heroes")

    initial_board_capacity: int = 7
    target_star_level: int = 2
    summon_recognition_delay_min: float = Field(default=1.0, ge=0.0, le=10.0)
    summon_recognition_delay_max: float = Field(default=2.0, ge=0.0, le=10.0)
    board_recognition_frames: int = Field(default=10, ge=1, le=60)
    default_main_c: str = "assault"
    # 组队对局模式：coop（合作模式）| pet（保卫宠物）；CLI --mode 可覆盖
    mode: str = "coop"
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
    # 抢合作弹窗关闭辅助（2026-09-14 实机标定）：连点误点自己头像会弹出
    # 「个人信息」等全屏弹窗，标准关闭按钮（右上角 ×）跨弹窗皮肤一致
    # （三主题实测 0.99+，普通界面负例 ≤0.67）。modal_edge_brightness 为
    # 全屏模态判定阈值：弹窗打开时游戏把画面两侧边缘压暗（实测 23~31），
    # 普通界面 ≥64；招募页本身也是暗的，故未知弹窗告警仅在已越过招募
    # 阶段（见过亮屏/准备按钮/关闭按钮）后由 grab_unknown_popup_confirmations
    # 连续拍数确认才触发，避免正常抢的过程中误报
    modal_edge_brightness: float = Field(default=45.0, ge=0.0, le=255.0)
    grab_unknown_popup_confirmations: int = Field(default=2, ge=1, le=10)
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
    # 选技能/召唤的金币检查间隔（秒）：每 0.5 秒读一次金币，够就操作，不够就等。
    # 节奏跟着金币走（2026-09-05 定案，取代旧的随机间隔）
    gold_check_interval_seconds: float = Field(default=0.5, gt=0.0, le=60.0)
    # 金币费用红色（游戏自身判买不起）时的等待时长（秒）：等待后重新识别
    # 再判，等击杀回金（2026-09-10 用户定案：红色等 1 秒再查）
    gold_red_wait_seconds: float = Field(default=1.0, gt=0.0, le=60.0)
    # 局内付费选技能流程（点【选技能】开页 / 按档位选卡 / 图标命中选卡 /
    # 兜底选卡）点击后的固定间隔（秒）：替代全局 input.min/max_delay 的
    # 1.0~2.0 随机拟人间隔——2026-09-13 实机日志，「开页→OCR→选卡」一轮
    # 5 秒里 1~2 秒是随机间隔。选卡验证帧为轻量识别、失败重试间隔 0.2 秒，
    # 间隔偏短时页面动画未播完只会多一次廉价重试，不会误点。开局技能页与
    # 4 星赠送页不适用（验证帧需完整 OCR 比对卡面，过早截图反而多等一轮
    # 数秒的完整识别）。0 = 退回全局随机拟人间隔。
    skill_click_post_delay_seconds: float = Field(default=0.5, ge=0.0, le=5.0)
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
    # 局内技能页「刷新」（2026-09-12 用户确认，宠物/合作共用）：三张卡均非主C系
    # 技能（优先级档位全部 ≥9）时点一次刷新换卡，之后必选一张。每局最多
    # max_per_round 次；0 或关闭即禁用该分支。刷新后的「等卡面换新」由动作后
    # 验证承担（验证帧 OCR 出新卡面即证明换卡完成并已可识别），不再额外固定
    # 等待（2026-09-12 用户反馈：验证+固定等待+重复 OCR 太慢）。
    # 开局技能页/天使页/4星赠送页不刷新（按钮仅局内「请选择」页实拍存在）
    skill_refresh_enabled: bool = True
    skill_refresh_max_per_round: int = Field(default=1, ge=0, le=10)
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
    # 刷新按钮未识别到时的限次等待重试（2026-09-12 实机：点开技能页后的首个
    # 决策帧按钮可能尚未渲染完，静态同位置 0.99+ 命中）。每次等待 wait_seconds
    # 后重新识别；连续 wait_max 次仍未命中才放弃刷新、直接选卡。0 = 不等待
    skill_refresh_button_wait_seconds: float = Field(default=1.0, gt=0.0, le=10.0)
    skill_refresh_button_wait_max: int = Field(default=3, ge=0, le=10)
    # 等待开局期间连续这么多帧识别到首页标记，才判定本局被取消/被踢回首页
    # （加载画面可能与首页有短暂相似，单帧命中不能直接判定）
    home_return_confirm_frames: int = Field(default=2, ge=1, le=10)
    # 组队大厅点准备后：准备按钮消失且底部导航连续这么多帧读到「返回」（=
    # 落在领地页），才判定本局被踢/房间解散（2026-09-25 实机：被队长请离后
    # 落点为领地页，toast「你被队长请离了」）；单帧 OCR 误读不能直接判定
    territory_return_confirm_frames: int = Field(default=2, ge=1, le=10)
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
    # 一局结束、点击返回前保存结算画面截图（合作/保卫宠物共用），按
    # 模式/日期归档到 screenshots/settlement/<mode>/<YYYYMMDD>/；
    # 保存失败只记日志不影响任务
    settlement_screenshot: bool = True
    # run 启动时清理超过该天数的按天截图目录（保留最近 N 天，含今天）；
    # 仅 settlement_screenshot 开启时执行
    settlement_screenshot_retention_days: int = Field(default=7, ge=1)
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
    # 间隔（秒）：该阶段不以棋盘识别为门禁（点击发送即计数），跳过多帧棋盘
    # 快速连点；最后一次保留正常稳定等待后再进入棋盘决策。批次召唤共用。
    # 2026-09-26 实机日志：截图+识别固定开销 ≈0.28s（pyautogui.PAUSE 已归
    # 零），0.2 等待对应实际 ≈0.5s/击（此前 0.5 配置实际跑出 ≈1s/击）
    forced_summon_interval_seconds: float = Field(default=0.2, gt=0.0, le=10.0)
    # 召唤点击后棋盘始终未变化（多为金币不足，点击被游戏忽略）时的重试等待
    # （秒）：等待后重新尝试召唤。金币门控（summon_gold_wait）下金币不足时
    # 原地等待恢复，不设次数上限
    summon_retry_delay_seconds: float = Field(default=15.0, gt=0.0, le=300.0)
    # 识别出错自动重开：任务保守停止（无可用动作/验证耗尽/截图时效超限）时，
    # 不退出任务，清空对局进度自动重新开始一局。连续重开达到上限仍无任何
    # 进展（验证成功/完成一局会清零计数）才真正停止，防环境损坏时死循环
    error_restart_enabled: bool = True
    error_restart_max_consecutive: int = Field(default=3, ge=1, le=50)
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
        if self.find_coop_click_delay_min > self.find_coop_click_delay_max:
            raise ValueError("find_coop_click_delay_min 不能大于 find_coop_click_delay_max")
        return self


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


class InviteConfig(BaseModel):
    """邀请者模式（run --invite）配置：组队等待/过滤/踢人循环参数。"""

    # 帮助者关卡进度阈值：低于踢掉请离，不低于留下开局
    min_stage_progress: int = Field(default=475, ge=1)
    # 请离后等房间回空（邀请文字重现）的兜底秒数：回空立即回等待态继续
    # 检查进房 + 轻量邀请（全量大轮回不因请离重置）；超时未回空才重走邀请链
    kick_cooldown_seconds: float = Field(default=60.0, ge=0.0)
    # 等待无人进房的重发间隔秒数：全量邀请链（邀请→一键邀请→好友→
    # 公会招募→一键邀请→关弹窗）的重发周期
    reinvite_after_seconds: float = Field(default=60.0, ge=1.0)
    # 轻量重邀间隔秒数（2026-09-26 用户定案）：两次全量之间每 15 秒走
    # 邀请→一键邀请（弹窗层）→关弹窗 的短链刷新邀请曝光，不走好友页
    light_reinvite_interval_seconds: float = Field(default=15.0, ge=1.0)
    # 连续「空等→重发」轮数上限（帮助者一直没来），超过保守停止
    max_empty_reinvites: int = Field(default=30, ge=1)
    # 连续请离帮助者数量上限（匹配池可能整体不合格），超过保守停止
    max_consecutive_kicks: int = Field(default=5, ge=1)
    # 等待帮助者进房的总时长上限（秒），超过保守停止
    wait_timeout_seconds: float = Field(default=600.0, ge=1.0)
    # 合格帮助者进房后等待其点准备的超时秒数，超时请离重来；0 = 一直等
    ready_timeout_seconds: float = Field(default=60.0, ge=0.0)
    # 点【挑战】开局后的动作验证帧数预算：游戏从点击到切屏实测 ~2.4s
    # （2026-09-26 实机），全局 5 帧（~1.5s）在慢过渡时会误判点击失败
    start_verify_frames: int = Field(default=10, ge=1)


class DefaultConfig(BaseModel):
    """configs/default.yaml 的完整模型。"""

    screen: ScreenConfig = Field(default_factory=ScreenConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    main_c_profiles: dict[str, MainCProfile] = Field(default_factory=dict)
    mode_profiles: dict[str, ModeProfile] = Field(default_factory=dict)
    hero_classifier: HeroClassifierConfig = Field(default_factory=HeroClassifierConfig)
    invite: InviteConfig = Field(default_factory=InviteConfig)

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

    @model_validator(mode="after")
    def _check_mode_profiles(self) -> DefaultConfig:
        if not self.mode_profiles:
            return self
        if self.run.mode not in self.mode_profiles:
            raise ValueError(
                f"run.mode={self.run.mode!r} 不在 mode_profiles 中: {sorted(self.mode_profiles)}"
            )
        for name, profile in self.mode_profiles.items():
            difficulty = profile.entry.difficulty
            if difficulty.kind == "ocr":
                if not difficulty.targets:
                    raise ValueError(
                        f"mode_profiles.{name}.entry.difficulty.kind=ocr 需要配置 targets"
                    )
                # 重复目标会点击同一难度两次；实机确认再次点击已勾选难度
                # 会取消勾选，必须在校验层拒绝
                duplicates = sorted(
                    {t for t in difficulty.targets if difficulty.targets.count(t) > 1}
                )
                if duplicates:
                    raise ValueError(
                        f"mode_profiles.{name}.entry.difficulty.targets "
                        f"含重复项: {duplicates}（再次点击已勾选难度会取消勾选）"
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
    # 保卫宠物难度弹窗 OCR 识别配置（candidate_roi + min_confidence）
    pet_difficulty_recognition: dict[str, Any] = Field(default_factory=dict)
    # 招募页模式页签行 OCR 识别配置（candidate_roi + min_confidence + miss_limit）
    mode_tab_recognition: dict[str, Any] = Field(default_factory=dict)
    # 「模式选择」弹窗 OCR 识别配置（title_roi/title_text + min_confidence +
    # entry_wait_timeout_seconds），领地入口共用
    mode_select_recognition: dict[str, Any] = Field(default_factory=dict)
    # 技能卡采集（统计阶段）的几何参数：比例相对 rois.skill_candidates 裁剪结果
    skill_collection: dict[str, Any] = Field(default_factory=dict)
    # 金币感知（回补召唤/付费选技能门控）配置
    gold_recognition: dict[str, Any] = Field(default_factory=dict)
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
                "pet_difficulty_recognition",
                "mode_tab_recognition",
                "mode_select_recognition",
                "skill_collection",
                "gold_recognition",
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


def load_logging_config(path: Path) -> LoggingConfig:
    """读取 default.yaml 的 ``logging`` 段；文件或段落缺失时返回默认值。

    只读该段而不加载完整 DefaultConfig，保证 CLI 启动时日志系统先于
    业务配置校验就绪；段落存在但字段非法时按配置错误抛出 ValidationError，
    不静默吞掉。
    """
    try:
        data = load_yaml(path)
    except (OSError, ValueError):
        return LoggingConfig()
    section = data.get("logging")
    if section is None:
        return LoggingConfig()
    if not isinstance(section, dict):
        raise ValueError(
            f"配置文件 {path} 的 logging 段必须是字典，实际为 {type(section).__name__}"
        )
    return LoggingConfig(**section)


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
