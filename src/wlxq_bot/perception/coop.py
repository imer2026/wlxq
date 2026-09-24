"""合作任务识别管线。

把窗口截图转换为可供 CoopTask 判断的 Observation：
- 识别界面标志（按钮模板），存入 Observation.raw_data
- 培养主 C 阶段识别己方棋盘英雄，构建 BoardSnapshot 存入 Observation.board

依赖方向：Runner -> CoopPerception -> Vision/Locator/TemplatePack。
任务状态机（CoopTask）只消费 Observation，不直接调用本类，
避免任务代码反向依赖识别实现。

未采集模板的界面标志不会被识别（对应键缺失），CoopTask 据此
保守地返回 UNKNOWN 而不是假装识别。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any
import re

from wlxq_bot.assets import TemplatePack
from wlxq_bot.config import TasksConfig
from wlxq_bot.debug.recorder import DebugRecorder
from wlxq_bot.models import (
    BoardCapacity,
    BoardCell,
    BoardCellType,
    BoardHero,
    BoardSnapshot,
    CoopRole,
    DifficultyCandidate,
    MatchResult,
    Observation,
    SkillCandidate,
    State,
    WindowContext,
)
from wlxq_bot.perception.hero_classifier import HeroCellClassifier, HeroCellPrediction
from wlxq_bot.perception.locator import (
    board_grid_for_role,
    format_cell_label,
    hero_cell_centers,
    hero_cell_rois,
    ratio_to_pixel_roi,
)
from wlxq_bot.perception.mode_tab import ModeTabRow, read_mode_tab_rows
from wlxq_bot.perception.ocr import tight_text_rows
from wlxq_bot.perception.pet_difficulty import read_difficulty_rows
from wlxq_bot.perception.skill_collector import SkillCollector
from wlxq_bot.perception.vision import Vision
from wlxq_bot.skill_catalog import find_fuzzy_match, record_new_skill
from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

# 进入培养阶段后开始识别棋盘的 state 集合。
# 观察这些 state 时，observe 会额外做棋盘英雄匹配并构建 BoardSnapshot。
_BOARD_WATCH_STATES: frozenset[State] = frozenset(
    {
        State.BUILD_MAIN_C,
    }
)

# 结算相关状态：这些状态下额外识别【双倍奖励】确认弹窗
_SETTLEMENT_STATES: frozenset[State] = frozenset(
    {
        State.HANDLE_RESULT,
        State.CLAIM_REWARD,
        State.CHECK_ROUND_LIMIT,
    }
)

# 全局界面标志的定位器三元组（locator 名 / 标志名 / 匹配结果名）
_INTERFACE_FLAG_LOCATORS: tuple[tuple[str, str, str], ...] = (
    ("ready_button", "ready_button_visible", "ready_button_match"),
    ("return_button", "return_button_visible", "return_button_match"),
    ("tan_chuang", "tan_chuang_visible", "tan_chuang_match"),
    # 全屏弹窗右上角标准关闭按钮（×）：抢合作连点误点自己头像弹出「个人信息」
    # 等全屏弹窗（皮肤可变，× 为跨皮肤一致的标准控件，2026-09-14 实机）。
    # 抢合作子步骤识别到即优先关闭；FIND_COOP 回主循环时兜底关闭
    ("close_button", "close_button_visible", "close_button_match"),
    # 同系列技能选满 3 个触发的赠送技能卡片（点击关闭，见 SELECT_MAIN_C_SKILLS）
    ("zeng_song_ji_neng", "zeng_song_ji_neng_visible", "zeng_song_ji_neng_match"),
    # 进入游戏后的两种开局标识：先「选择技能」再「召唤」，二者互斥，
    # 出现其一即确认对局界面已稳定及其开局类型（见 CoopTask._action_opening_skills）。
    ("select_skill_button", "select_skill_button_visible", "select_skill_button_match"),
    # 局内技能页「刷新」按钮：三张卡均无主C系技能时点一次换卡（每局一次，
    # 见 CoopTask 刷新分支）；仅 SELECT_MAIN_C_SKILLS 状态识别，开局/赠送页不识别
    ("skill_refresh_button", "skill_refresh_visible", "skill_refresh_match"),
    # 【请选择1个额外技能】提示条：合成4星赠送技能页的主标识（该页上【选技能】图
    # 不稳定，实机 2026-08-21）；仅 BUILD 状态识别
    ("merge_gift_skill_title", "merge_gift_skill_page_visible", "merge_gift_skill_title_match"),
    ("summon_button", "summon_button_visible", "summon_button_match"),
    # 邀请者开局按钮双态（组队大厅）：帮助者「已准备」态出现 → 邀请者点击
    # 开局；「未准备」态 → 继续等。两态同识别区 invite_start_button_area
    ("invite_start_ready", "invite_start_ready_visible", "invite_start_ready_match"),
    ("invite_start_not_ready", "invite_start_not_ready_visible", "invite_start_not_ready_match"),
)

# 各状态需要识别的全局界面标志。合作只能通过自己点击加入进入，游戏不会主动
# 把玩家拉进对局（实机确认 2026-08-17），因此招募入口（FIND_COOP 的入口子
# 步骤）时游戏未开始、也没发出过任何加入点击，五个标志都不可能出现，全部
# 跳过以压缩单帧识别耗时；抢合作子步骤（observation_mode="coop_grab"）额外
# 查准备按钮——出现即代表抢到，回主循环点准备。对局内各标志随机出现，按
# 状态各取所需；ENTER_MATCH 是组队大厅（对局未开始），击杀奖励弹窗不可能
# 出现，也不查。
_INTERFACE_FLAGS_BY_STATE: dict[State, frozenset[str]] = {
    State.FIND_COOP: frozenset(),
    # 邀请点击链第 7 步关邀请弹窗需要 × 识别门控（盲点坐标会点中【概率】）
    State.INVITE_CLICK: frozenset({"close_button"}),
    # 邀请者等待态：帮助者信息层打开时需要 × 关闭按钮（close_button）
    State.INVITE_WAITING: frozenset({"close_button"}),
    State.INVITE_START: frozenset({"invite_start_ready", "invite_start_not_ready"}),
    State.INVITE_COOLDOWN: frozenset(),
    State.ENTER_MATCH: frozenset({"ready_button"}),
    State.SELECT_OPENING_SKILLS: frozenset(
        {"select_skill_button", "summon_button", "tan_chuang", "return_button"}
    ),
    State.BUILD_MAIN_C: frozenset(
        {
            "select_skill_button",
            "merge_gift_skill_title",
            "summon_button",
            "tan_chuang",
            "return_button",
        }
    ),
    State.SELECT_MAIN_C_SKILLS: frozenset(
        {
            "select_skill_button",
            "skill_refresh_button",
            "tan_chuang",
            "return_button",
            "zeng_song_ji_neng",
        }
    ),
    State.HANDLE_RESULT: frozenset({"return_button"}),
    State.CLAIM_REWARD: frozenset({"return_button"}),
    State.CHECK_ROUND_LIMIT: frozenset({"return_button"}),
}

# 多帧识别的默认帧数。每格按精确类别做多数投票，抵抗动态特效和遮挡。
_DEFAULT_CULTIVATION_FRAMES = 10


class CoopPerception:
    """合作任务识别管线。

    持有 Vision / TemplatePack / HeroCellClassifier / TasksConfig / 角色与主 C，
    提供 observe 方法按当前 hint_state 做对应识别。棋盘 12 格统一由 ONNX
    分类器识别；英雄模板不再参与正式棋盘决策。
    """

    def __init__(
        self,
        vision: Vision,
        template_pack: TemplatePack,
        tasks_config: TasksConfig,
        role: CoopRole,
        main_c: str,
        debug_recorder: DebugRecorder | None = None,
        skill_icon_templates: list[str] | None = None,
        teammate_skill_icon_templates: list[str] | None = None,
        hero_cell_classifier: HeroCellClassifier | None = None,
        allowed_heroes: set[str] | None = None,
        skill_collector: SkillCollector | None = None,
        title_reader: Any | None = None,
        skill_tiers: dict[str, int] | None = None,
        new_skill_dir: Path | None = None,
        gold_reader: Any | None = None,
        gold_read_interval: float = 1.0,
        gold_color_only: bool = False,
        difficulty_kind: str = "template",
        locator_aliases: dict[str, str] | None = None,
        difficulty_reader: Any | None = None,
        territory_nav_roi: str | None = None,
        mode_select_title_roi: str | None = None,
        mode_card_roi: str | None = None,
        challenge_roi: str | None = None,
        weekly_buff_roi: str | None = None,
        mode_select_min_confidence: float = 0.5,
        skill_refresh_enabled: bool = False,
    ) -> None:
        self._vision = vision
        self._pack = template_pack
        self._cfg = tasks_config
        self._role = role
        self._main_c = main_c
        self._debug_recorder = debug_recorder
        self._hero_cell_classifier = hero_cell_classifier
        self._allowed_heroes = frozenset(allowed_heroes or {main_c})
        # 彩虹难度 1～19 文字模板；缺失的编号（当前仅 19）在识别时自动跳过。
        # 不识别行右侧勾选框状态（2026-08-19 实机定案：邀请弹窗会遮挡勾选框
        # 使读数不可靠，而点击行文字几何上永远安全，见 tasks/coop.py 注释；
        # 勾选框模板 gou_xuan/gou_xuan_kong 已采集留档，需要时再启用）
        self._difficulty_template_paths = {
            level: self._pack.resolve_template(f"buttons/coop_difficulty/cai_hong_{level}.png")
            for level in range(1, 20)
        }
        # 主C技能卡上的英雄图标模板（简化版技能识别）：只保留真实存在的文件。
        # 同一英雄所有技能卡共用一张图标，在技能 ROI 匹配它即识别到主C技能。
        self._skill_icon_paths: list[str] = [
            str(self._pack.resolve_template(rel))
            for rel in (skill_icon_templates or [])
            if self._pack.resolve_template(rel).is_file()
        ]
        # 队友英雄技能卡图标：主C图标未命中时匹配它，命中即证明技能页已稳定
        # 且本组没有主C技能卡，任务层可立即随机选卡而不等满识别帧
        self._teammate_skill_paths: list[str] = [
            str(self._pack.resolve_template(rel))
            for rel in (teammate_skill_icon_templates or [])
            if self._pack.resolve_template(rel).is_file()
        ]
        # 技能卡采集器（统计阶段可选）：observe 契约为永不抛异常、永不阻塞，
        # 采集结果只落盘不影响任何业务决策
        self._skill_collector = skill_collector
        # 技能标题 OCR 主路径（可选）：配置了优先级档位时，技能页用一次横带
        # OCR 读出三张卡的技能名，产出 skill_card_options 交给任务层按档位
        # 选卡；标题未读出时回退到图标识别。tier 索引的模糊匹配条目预构建
        self._title_reader = title_reader
        self._skill_tiers = skill_tiers or {}
        self._tier_name_entries = [{"name": name} for name in self._skill_tiers]
        self._new_skill_dir = new_skill_dir
        # 金币感知（按需）：仅在任务层声明需要时读取（回补召唤/付费选技能
        # 门控），最小间隔内复用上次结果；其他状态零开销
        self._gold_reader = gold_reader
        self._gold_read_interval = max(0.0, gold_read_interval)
        # 只认色模式：金币门控只判定费用文字红/白（毫秒级），跳过数字 OCR；
        # 判定足够廉价，每帧新鲜计算，不走进度间隔缓存
        self._gold_color_only = gold_color_only
        self._last_gold_read = float("-inf")
        self._last_gold_info: dict[str, Any] | None = None
        # 难度识别方式（模式档案）：template=合作文字模板；ocr=保卫宠物行 OCR。
        # OCR 目标难度（targets）只由任务层 CoopTask 消费，感知层不持有
        self._difficulty_kind = difficulty_kind
        # canonical locator 名 → 模式专属 locator 名（ready_button/return_button/
        # coop_leave_team 三个可被重映射；标志名不变，任务层零改动）
        self._locator_aliases = dict(locator_aliases or {})
        self._difficulty_reader = difficulty_reader
        # 领地入口（2026-09-09 新流程）识别区：导航文字区共用，模式卡与挑战
        # 按钮区按模式档案传入（rois 名）；OCR 结果只写 raw_data，子串匹配
        # 与点击决策归任务层
        self._territory_nav_roi = territory_nav_roi
        self._mode_select_title_roi = mode_select_title_roi
        self._mode_card_roi = mode_card_roi
        self._challenge_roi = challenge_roi
        # 挑战页每周弹窗按钮识别区（保卫宠物=本周Buff【知道了】）；按模式
        # 档案传入，None = 本模式无此弹窗，不读
        self._weekly_buff_roi = weekly_buff_roi
        self._mode_select_min_confidence = mode_select_min_confidence
        # 局内技能页「刷新」按钮识别开关（RunConfig.skill_refresh_enabled）：
        # 关闭时跳过模板匹配，任务层也不会收到该标志
        self._skill_refresh_enabled = skill_refresh_enabled

    @property
    def available_heroes(self) -> list[str]:
        """当前 ONNX 模型能够识别且本局允许出现的英雄 id。"""
        if self._hero_cell_classifier is None:
            return []
        return sorted(self._hero_cell_classifier.supported_heroes & self._allowed_heroes)

    def observe(
        self,
        ctx: WindowContext,
        frame: Any,
        hint_state: State = State.UNKNOWN,
        observation_mode: str | None = None,
        read_gold: bool = False,
        light: bool = False,
    ) -> Observation:
        """对一帧截图做识别，返回 Observation。

        Args:
            ctx: 当前窗口上下文（提供客户区尺寸和 frame_id）
            frame: 截图帧（BGR ndarray）
            hint_state: 上一轮状态，用于决定是否做棋盘专项识别
            observation_mode: 任务内部步骤要求的专项识别模式
            read_gold: 当前是否需要金币感知（任务层 wants_gold_read 声明）；
                按最小间隔读取，间隔内复用上次结果

        Args:
            light: 轻量模式（动作后验证帧用）：只算界面标志，跳过技能卡
                OCR、金币读取与采集——验证只需要「标志还在不在」

        Returns:
            Observation，界面标志存 raw_data，培养阶段附带 board 快照
        """
        raw_data: dict[str, Any] = {}
        matches: list = []

        # 合作任务首次动作前必须正向识别首页。仅在首页门禁步骤匹配，
        # 避免后续高频循环持续消耗模板识别时间。
        if observation_mode == "home_page":
            locator = self._cfg.locators.get("home_page_marker")
            home_match = self._match_locator_template(ctx, frame, locator) if locator else None
            raw_data["home_page_visible"] = home_match is not None
            if home_match is not None:
                matches.append(home_match)
                raw_data["home_page_match"] = home_match

        # 1. 界面标志识别（按状态门控，见 _INTERFACE_FLAGS_BY_STATE）
        skill_candidates = self._detect_interface_flags(
            ctx,
            frame,
            hint_state,
            raw_data,
            matches,
            observation_mode=observation_mode,
            light=light,
        )
        # 金币感知（按需）：仅任务层声明需要（回补召唤/付费选技能门控）时读取。
        # 技能页/赠送页已打开时金币已支付，读了也无意义——跳过，把帧时间让给标题 OCR
        page_open = any(
            raw_data.get(flag)
            for flag in (
                "select_skill_button_visible",
                "merge_gift_skill_page_visible",
                "tian_shi_kai_ju_visible",
            )
        )
        if self._gold_reader is not None and read_gold and not page_open and (
            not light or self._gold_color_only
        ):
            if self._gold_color_only:
                # 只认色：红/白判定毫秒级，每帧新鲜计算（含轻量验证帧），
                # 无需读取间隔缓存。轻量帧也必须算——关弹窗验证通过后任务层
                # 直接拿该帧决策，缺金币信息会盲点【选技能】（2026-09-10 实机）
                gold = self._read_gold(ctx, frame)
                if gold is not None:
                    raw_data["gold"] = gold
            else:
                now_monotonic = perf_counter()
                if now_monotonic - self._last_gold_read >= self._gold_read_interval:
                    self._last_gold_read = now_monotonic
                    gold = self._read_gold(ctx, frame)
                    if gold is not None:
                        self._last_gold_info = gold
                if (
                    self._last_gold_info is not None
                    and now_monotonic - self._last_gold_read < self._gold_read_interval * 5
                ):
                    raw_data["gold"] = self._last_gold_info
        difficulty_candidates = []
        if observation_mode == "coop_difficulty":
            if self._difficulty_kind == "ocr":
                # 保卫宠物：OCR 读难度行（供勾选点击）；弹窗开关用收起栏判定
                self._detect_pet_difficulty_rows(ctx, frame, raw_data)
                self._detect_pet_difficulty_bar_flag(ctx, frame, raw_data, matches)
            else:
                self._detect_difficulty_dialog_flag(ctx, frame, raw_data, matches)
                difficulty_candidates = self._detect_difficulty_candidates(ctx, frame, matches)
        elif observation_mode == "difficulty_dialog":
            # 关闭/确认关闭难度弹窗：只判定弹窗开关标识，跳过难度候选匹配；
            # 保卫宠物轮询只 OCR 收起栏小区域（控制单帧耗时）
            if self._difficulty_kind == "ocr":
                self._detect_pet_difficulty_bar_flag(ctx, frame, raw_data, matches)
            else:
                self._detect_difficulty_dialog_flag(ctx, frame, raw_data, matches)
        elif observation_mode == "mode_tab":
            # 点模式页签前 OCR 页签行定位文字：三个页签共用一条识别区，
            # 仅进招募后的一两帧识别，不在高频循环内
            self._detect_mode_tab_rows(ctx, frame, raw_data)
        elif observation_mode == "territory_nav":
            # 领地入口第一步：读底部导航领地格文字（领地=未进领地页，
            # 返回=已在领地页），仅入口步骤识别，不在高频循环内
            self._detect_territory_nav_text(ctx, frame, raw_data)
        elif observation_mode == "mode_select":
            # 领地入口：OCR 模式选择弹窗标题（弹窗开关锚点）+ 本模式卡片
            # 文字行（供点击文字框中心）
            self._detect_mode_select_title(ctx, frame, raw_data)
            self._detect_mode_card_rows(ctx, frame, raw_data)
        elif observation_mode == "challenge_page":
            # 领地入口最后一步：OCR 挑战页按钮文字（页面到达锚点）；配置了
            # 每周弹窗识别区（保卫宠物=本周Buff）时同帧读按钮文字行
            self._detect_challenge_text(ctx, frame, raw_data)
            if self._weekly_buff_roi:
                self._detect_weekly_buff_rows(ctx, frame, raw_data)
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
        elif observation_mode == "invite_waiting":
            # 邀请者等待态（spec 2026-09-21 §6）：OCR「邀请」文字（可见=无人
            # 进房）+ 帮助者信息弹层关卡进度行（有人在房时读取）。两者都限域，
            # 单帧只做其一由任务层通过两次 observe 完成（先判定后读进度）
            self._detect_invite_text(ctx, frame, raw_data)
            self._detect_helper_stage_progress(ctx, frame, raw_data)

        # 2. 培养阶段做棋盘英雄识别
        board: BoardSnapshot | None = None
        if hint_state in _BOARD_WATCH_STATES:
            board = self._observe_board(ctx, frame, matches)

        return Observation(
            frame_id=ctx.frame_id,
            source_frame_ids=(ctx.frame_id,),
            matches=matches,
            raw_data=raw_data,
            board=board,
            skill_candidates=skill_candidates,
            difficulty_candidates=difficulty_candidates,
        )

    def _detect_difficulty_candidates(
        self,
        ctx: WindowContext,
        frame: Any,
        matches: list,
    ) -> list[DifficultyCandidate]:
        """识别难度弹窗当前可见的彩虹难度（模板存在的编号，当前 1～18）。

        逐模板独立匹配（2026-08-19 实机定案：不用连续性解码等推断方法，
        每个编号一张实机截图裁剪的文字模板，各匹配各的）；同一行被多个
        相似模板命中时按中心点距离合并、保留置信度最高者。
        """
        cfg = self._cfg.difficulty_recognition
        roi_name = str(cfg.get("candidate_roi", "coop_difficulty_list"))
        roi = self._resolve_named_roi(roi_name, ctx)
        threshold = float(cfg.get("threshold", 0.78))
        merge_dist = int(cfg.get("merge_distance", 30))

        detected: list[DifficultyCandidate] = []
        detected_matches: list = []
        for level, template_path in self._difficulty_template_paths.items():
            if not template_path.is_file():
                continue
            match = self._vision.match_template(
                frame, str(template_path), roi=roi, threshold=threshold
            )
            if match is None:
                continue
            detected.append(
                DifficultyCandidate(
                    level=level,
                    position=match.position,
                    confidence=match.confidence,
                    template_path=str(template_path),
                )
            )
            detected_matches.append(match)

        # 相似数字模板可能命中同一行，只保留该位置置信度最高的难度。
        kept: list[DifficultyCandidate] = []
        kept_matches: list = []
        for candidate, match in sorted(
            zip(detected, detected_matches, strict=True),
            key=lambda item: item[0].confidence,
            reverse=True,
        ):
            if any(
                (
                    (candidate.position[0] - other.position[0]) ** 2
                    + (candidate.position[1] - other.position[1]) ** 2
                )
                ** 0.5
                < merge_dist
                for other in kept
            ):
                continue
            kept.append(candidate)
            kept_matches.append(match)

        matches.extend(kept_matches)
        result = sorted(kept, key=lambda item: item.level, reverse=True)
        logger.debug(
            "frame=%d 难度弹窗可见=%s ROI=%s",
            ctx.frame_id,
            [item.level for item in result],
            roi,
        )

        # 调试记录：保存 ROI 原始裁剪（核对实际像素/边缘裁切）和标注整帧
        # （每个候选标「难度号:置信度」），便于排查 8 识别成 5 这类误判。
        # Debug Recorder 不参与决策，失败仅记 DEBUG 日志。
        if self._debug_recorder is not None:
            try:
                if roi is not None:
                    rx, ry, rw, rh = roi
                    self._debug_recorder.save_frame(
                        frame[ry : ry + rh, rx : rx + rw].copy(),
                        ctx.frame_id,
                        prefix="difficulty_roi",
                    )
                self._debug_recorder.save_annotated_labeled(
                    frame,
                    [
                        (c.position[0], c.position[1], f"{c.level}:{c.confidence:.2f}")
                        for c in result
                    ],
                    ctx.frame_id,
                    prefix="difficulty",
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("难度识别调试截图保存失败: %r", exc)

        return result

    def _detect_interface_flags(
        self,
        ctx: WindowContext,
        frame: Any,
        hint_state: State,
        raw_data: dict[str, Any],
        matches: list,
        *,
        observation_mode: str | None = None,
        light: bool = False,
    ) -> list[SkillCandidate]:
        """按状态识别需要的界面标志按钮，结果写入 raw_data（bool）和 matches。

        只匹配 _INTERFACE_FLAGS_BY_STATE 中当前状态声明的标志；抢合作子步骤
        （observation_mode="coop_grab"）额外查准备按钮和弹窗关闭按钮（连点
        误点自己头像会弹出「个人信息」等全屏弹窗，回主循环时需兜底关闭）。
        识别不到的标志不写入 raw_data，任务层 flag() 读取时按 False 处理。
        """
        locators = self._cfg.locators
        wanted = _INTERFACE_FLAGS_BY_STATE.get(hint_state, frozenset())
        if observation_mode == "coop_grab":
            wanted = wanted | {"ready_button", "close_button"}
        if not self._skill_refresh_enabled:
            wanted = wanted - {"skill_refresh_button"}

        # 当帧需要的标志并行匹配（模板匹配释放全局锁，总耗时≈最慢一个）
        # canonical locator 名先经别名重映射，模式专属 locator 不存在时该标志
        # 按未识别处理（flag() 返回 False），标志名保持不变
        wanted_items = [
            (self._locator_aliases.get(locator_name, locator_name), flag_name, match_name)
            for locator_name, flag_name, match_name in _INTERFACE_FLAG_LOCATORS
            if locator_name in wanted
        ]

        def _match_item(item: tuple[str, str, str]):
            locator = locators.get(item[0])
            return self._match_locator_template(ctx, frame, locator) if locator else None

        with ThreadPoolExecutor(max_workers=4) as pool:
            match_results = list(pool.map(_match_item, wanted_items))

        for (_locator_name, flag_name, match_name), match in zip(
            wanted_items, match_results, strict=True
        ):
            raw_data[flag_name] = match is not None
            if match is not None:
                matches.append(match)
                raw_data[match_name] = match

        # 召唤按钮 OCR 兜底（2026-09-24）：帮助者城墙皮肤会改变底栏按钮配色，
        # 「召唤」文字模板按阈值匹配不上 → 开局类型识别永远等不到召唤标志
        # 而卡死。模板未命中时对该按钮区域做限域 OCR，读到「召唤」即视为
        # 按钮可见（点击点=文字框中心，皮肤主题不影响文字本身）。
        # 仅在开局识别态兜底——BUILD_MAIN_C 没有消费者，挂进去会让培养
        # 主循环每帧多 2.5s OCR（2026-09-24 实机：强制召唤被拖慢到 5s/发）
        if (
            hint_state == State.SELECT_OPENING_SKILLS
            and not raw_data.get("summon_button_visible")
        ):
            rows = self._read_entry_roi_rows("summon_button_area", ctx, frame)
            hit = next((row for row in rows if "召唤" in row.text), None)
            if hit is not None:
                raw_data["summon_button_visible"] = True
                ocr_match = MatchResult(
                    position=hit.position,
                    confidence=hit.confidence,
                    template_name="summon_button_ocr",
                    roi_name="summon_button_area",
                )
                matches.append(ocr_match)
                raw_data["summon_button_match"] = ocr_match
                logger.debug(
                    "frame=%d 召唤按钮模板未命中，OCR 兜底确认（conf=%.2f）",
                    ctx.frame_id,
                    hit.confidence,
                )

        # 赠送技能卡片（同系列技能选满 3 个触发）：仅在选技能阶段识别
        if hint_state == State.SELECT_MAIN_C_SKILLS:
            locator = locators.get("zeng_song_ji_neng")
            match = self._match_locator_template(ctx, frame, locator) if locator else None
            raw_data["zeng_song_ji_neng_visible"] = match is not None
            if match is not None:
                matches.append(match)
                raw_data["zeng_song_ji_neng_match"] = match

        # 天使开局技能页不出现【选技能】图、也没有主C技能图标，只出现专属标识；
        # 仅在开局技能阶段检测，避免其余高频循环额外消耗模板匹配。
        if hint_state == State.SELECT_OPENING_SKILLS:
            locator = locators.get("tian_shi_kai_ju")
            match = self._match_locator_template(ctx, frame, locator) if locator else None
            raw_data["tian_shi_kai_ju_visible"] = match is not None
            if match is not None:
                matches.append(match)
                raw_data["tian_shi_kai_ju_match"] = match
            # 等待开局期间检测是否被踢回首页（实机确认房主可踢出已加入玩家，
            # 被踢后回到首页、对局不开始）；由任务层连续多帧确认后再重新抢合作
            home_locator = locators.get("home_page_marker")
            home_match = (
                self._match_locator_template(ctx, frame, home_locator) if home_locator else None
            )
            raw_data["home_page_visible"] = home_match is not None
            if home_match is not None:
                matches.append(home_match)
                raw_data["home_page_match"] = home_match
            # 组队大厅的【退队】按钮：房主长时间不开始对局时由任务层点击退出
            # （保卫宠物模式经别名重映射到 pet_leave_team，标志名不变）
            leave_locator = locators.get(
                self._locator_aliases.get("coop_leave_team", "coop_leave_team")
            )
            leave_match = (
                self._match_locator_template(ctx, frame, leave_locator) if leave_locator else None
            )
            raw_data["leave_team_visible"] = leave_match is not None
            if leave_match is not None:
                matches.append(leave_match)
                raw_data["leave_team_match"] = leave_match

        if not light:
            if hint_state in {State.SELECT_OPENING_SKILLS, State.SELECT_MAIN_C_SKILLS}:
                return self._detect_skill_candidates(
                    ctx, frame, matches, raw_data, page=hint_state.name
                )
            if hint_state == State.BUILD_MAIN_C and (
                raw_data.get("select_skill_button_visible")
                or raw_data.get("merge_gift_skill_page_visible")
            ):
                # 合成 4 星后系统赠送的技能选择页（实机确认 2026-08-16）：与常规
                # 技能页同一个 3 选 1 界面，会遮住棋盘导致识别为空。页面打开时才做
                # 技能卡识别，不增加常规培养帧的开销；主标识为【请选择1个额外技能】
                # 提示条（2026-08-21 实机确认【选技能】图在该页不稳定），命中任一即识别
                return self._detect_skill_candidates(
                    ctx, frame, matches, raw_data, page=hint_state.name
                )
        # 结算阶段识别【双倍奖励】确认弹窗（实机确认 2026-08-16：结算画面误点
        # 「双倍奖励」后弹出，挡住正常结算）；任务层点击其取消按钮关闭
        if hint_state in _SETTLEMENT_STATES:
            for locator_name, flag_name, match_name in (
                (
                    "double_reward_dialog",
                    "double_reward_dialog_visible",
                    "double_reward_dialog_match",
                ),
                (
                    "double_reward_cancel",
                    "double_reward_cancel_visible",
                    "double_reward_cancel_match",
                ),
            ):
                locator = locators.get(locator_name)
                match = self._match_locator_template(ctx, frame, locator) if locator else None
                raw_data[flag_name] = match is not None
                if match is not None:
                    matches.append(match)
                    raw_data[match_name] = match
        return []

    def _detect_difficulty_dialog_flag(
        self,
        ctx: WindowContext,
        frame: Any,
        raw_data: dict[str, Any],
        matches: list,
    ) -> None:
        """以【合作模式】标题图正向判定难度弹窗是否打开，写入 difficulty_dialog_visible。

        弹窗关闭验证依赖该标志（见 CoopTask._action_close_difficulty_dialog）；
        难度候选（cai_hong_N）只用于勾选难度，不再判定弹窗开关。
        """
        locator = self._cfg.locators.get("he_zuo_mo_shi")
        match = self._match_locator_template(ctx, frame, locator) if locator else None
        raw_data["difficulty_dialog_visible"] = match is not None
        if match is not None:
            matches.append(match)
            raw_data["difficulty_dialog_match"] = match

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

    def _detect_mode_tab_rows(
        self,
        ctx: WindowContext,
        frame: Any,
        raw_data: dict[str, Any],
    ) -> None:
        """OCR 读招募页模式页签行，写入 mode_tab_rows（已按 min_confidence 过滤）。

        ROI 来自 mode_tab_recognition.candidate_roi；缺失或零面积时返回空
        （OCR 必须限域，启动检查会另行为缺失报错）。任务层按模式档案的
        mode_tab_text 子串匹配并点击文字框中心（开环，同难度勾选策略）。
        """
        cfg = self._cfg.mode_tab_recognition
        roi_name = str(cfg.get("candidate_roi", "mode_tab_row_area"))
        roi = self._resolve_named_roi(roi_name, ctx)
        min_confidence = float(cfg.get("min_confidence", 0.5))
        rows = [
            row
            for row in read_mode_tab_rows(frame, roi, reader=self._difficulty_reader)
            if row.confidence >= min_confidence
        ]
        raw_data["mode_tab_rows"] = rows
        logger.debug(
            "frame=%d 模式页签行=%s", ctx.frame_id, [row.text for row in rows]
        )

    def _read_entry_roi_rows(
        self,
        roi_name: str | None,
        ctx: WindowContext,
        frame: Any,
    ) -> list[ModeTabRow]:
        """领地入口共用的限域 OCR：按 rois 名读一个小区域的文字行。

        已按 mode_select_recognition.min_confidence 过滤；ROI 缺失或零面积
        返回空（OCR 必须限域，启动检查会另行为缺失报错）。
        """
        roi = self._resolve_named_roi(roi_name, ctx)
        rows = [
            row
            for row in read_mode_tab_rows(frame, roi, reader=self._difficulty_reader)
            if row.confidence >= self._mode_select_min_confidence
        ]
        return rows

    def _detect_invite_text(
        self,
        ctx: WindowContext,
        frame: Any,
        raw_data: dict[str, Any],
    ) -> None:
        """OCR 组队大厅【邀请】按钮文字区，写 invite_text_visible。

        可见 = 无人进房（继续等）；不可见 = 有人进房或弹窗遮挡（任务层复检）。
        """
        rows = self._read_entry_roi_rows("invite_text", ctx, frame)
        visible = any("邀请" in row.text for row in rows)
        raw_data["invite_text_visible"] = visible
        logger.debug(
            "frame=%d 邀请文字可见=%s rows=%s",
            ctx.frame_id,
            visible,
            [row.text for row in rows],
        )

    def _detect_helper_stage_progress(
        self,
        ctx: WindowContext,
        frame: Any,
        raw_data: dict[str, Any],
    ) -> None:
        """OCR 帮助者信息弹层的关卡进度行（如「401.雷霆之巅」）。

        多段结果合并后取点号前整数（2026-09-21 实机 24 图验证：数字
        24/24 正确；「499.月蚀荒原」可能被拆成两段，合并后不受影响）。
        解析失败写 helper_stage_progress=None（任务层保守处理）。
        """
        rows = self._read_entry_roi_rows("helper_stage_progress", ctx, frame)
        merged = "".join(row.text for row in rows)
        progress: int | None = None
        match = re.search(r"(\d+)", merged)
        if match:
            try:
                progress = int(match.group(1))
            except ValueError:  # pragma: no cover - \d+ 必为合法整数
                progress = None
        raw_data["helper_stage_progress"] = progress
        logger.debug(
            "frame=%d 帮助者关卡进度=%s（raw=%r）",
            ctx.frame_id,
            progress,
            merged,
        )

    def _detect_territory_nav_text(        self,
        ctx: WindowContext,
        frame: Any,
        raw_data: dict[str, Any],
    ) -> None:
        """OCR 读底部导航领地格文字，写入 territory_nav_text（无文字为空串）。

        「领地」=未进领地页（可点），「返回」=已在领地页（跳过点击）。
        """
        rows = self._read_entry_roi_rows(self._territory_nav_roi, ctx, frame)
        raw_data["territory_nav_text"] = rows[0].text if rows else ""
        logger.debug(
            "frame=%d 领地导航文字=%s", ctx.frame_id, raw_data["territory_nav_text"]
        )

    def _detect_mode_select_title(
        self,
        ctx: WindowContext,
        frame: Any,
        raw_data: dict[str, Any],
    ) -> None:
        """OCR 模式选择弹窗标题，按 title_text 子串匹配写 mode_select_visible。

        标题两侧装饰小星星偶被读进文字（「模式选择·」），必须子串匹配；
        弹窗关闭时读空。
        """
        cfg = self._cfg.mode_select_recognition
        title_text = str(cfg.get("title_text", "模式选择"))
        rows = self._read_entry_roi_rows(self._mode_select_title_roi, ctx, frame)
        visible = any(title_text in row.text for row in rows)
        raw_data["mode_select_visible"] = visible
        logger.debug("frame=%d 模式选择标题可见=%s", ctx.frame_id, visible)

    def _detect_mode_card_rows(
        self,
        ctx: WindowContext,
        frame: Any,
        raw_data: dict[str, Any],
    ) -> None:
        """OCR 本模式卡片文字行，写入 mode_card_rows（供任务层点文字框中心）。

        ROI 名按模式档案 entry.mode_card_roi 由 runner 传入。
        """
        rows = self._read_entry_roi_rows(self._mode_card_roi, ctx, frame)
        raw_data["mode_card_rows"] = rows
        logger.debug(
            "frame=%d 模式卡行=%s", ctx.frame_id, [row.text for row in rows]
        )

    def _detect_challenge_text(
        self,
        ctx: WindowContext,
        frame: Any,
        raw_data: dict[str, Any],
    ) -> None:
        """OCR 挑战页按钮文字，写入 challenge_text（无文字为空串）。

        合作挑战页读「开始挑战」、保卫宠物挑战页读「挑战」，到达锚点由
        任务层按模式档案 challenge_text 子串匹配。
        """
        rows = self._read_entry_roi_rows(self._challenge_roi, ctx, frame)
        raw_data["challenge_text"] = rows[0].text if rows else ""
        logger.debug(
            "frame=%d 挑战按钮文字=%s", ctx.frame_id, raw_data["challenge_text"]
        )

    def _detect_weekly_buff_rows(
        self,
        ctx: WindowContext,
        frame: Any,
        raw_data: dict[str, Any],
    ) -> None:
        """OCR 挑战页每周弹窗按钮文字行，写入 weekly_buff_rows（供任务层关闭弹窗）。

        ROI 名按模式档案 entry.weekly_buff_roi 由 runner 传入；保卫宠物=
        本周Buff 弹窗【知道了】按钮（每天首次进入弹出）。无弹窗时读空。
        """
        rows = self._read_entry_roi_rows(self._weekly_buff_roi, ctx, frame)
        raw_data["weekly_buff_rows"] = rows
        logger.debug(
            "frame=%d 每周弹窗按钮行=%s", ctx.frame_id, [row.text for row in rows]
        )

    def _detect_pet_difficulty_bar_flag(
        self,
        ctx: WindowContext,
        frame: Any,
        raw_data: dict[str, Any],
        matches: list,
    ) -> None:
        """以「选择难度」收起栏文字正向判定保卫宠物难度弹窗是否已收起。

        栏文字可见=弹窗已收起（difficulty_dialog_visible=False），栏被展开
        面板盖住=弹窗打开（True）。弹窗关闭后邀请卡标题（保卫宠物-××）与
        难度行同区域，行文字存在性无法区分开关（2026-09-06 实机首跑踩中），
        故开关判定改用收起栏文字；OCR 对皮肤换色免疫，比模板更通用。
        ROI 与文字来自 pet_difficulty_recognition 的 bar_roi / bar_text；
        离线验证（2026-09-06 实机退出帧 20 帧）：可见帧稳定读出「选择难度」
        （conf 0.80），被盖住帧全空，15/15 开关判定一致。
        """
        cfg = self._cfg.pet_difficulty_recognition
        bar_roi_name = str(cfg.get("bar_roi", "pet_difficulty_bar_area"))
        bar_roi = self._resolve_named_roi(bar_roi_name, ctx)
        bar_text = str(cfg.get("bar_text", "选择难度"))
        min_confidence = float(cfg.get("min_confidence", 0.5))
        bar_rows = [
            row
            for row in read_difficulty_rows(frame, bar_roi, reader=self._difficulty_reader)
            if row.confidence >= min_confidence and bar_text in row.text
        ]
        bar_visible = bool(bar_rows)
        raw_data["pet_difficulty_bar_visible"] = bar_visible
        raw_data["difficulty_dialog_visible"] = not bar_visible
        logger.debug(
            "frame=%d 选择难度栏可见=%s", ctx.frame_id, bar_visible
        )

    def _detect_skill_candidates(
        self,
        ctx: WindowContext,
        frame: Any,
        matches: list,
        raw_data: dict[str, Any],
        *,
        page: str = "",
    ) -> list[SkillCandidate]:
        """识别主C技能卡：在技能 ROI 匹配主C的英雄图标模板。

        同一英雄的所有技能卡共用同一张图标，因此匹配到的每个位置都是一张
        主C技能卡。命中多个时由任务层随机选一个；图标在动会单帧漏检，由
        任务层多帧重试。未配置 skill_icon_templates 时返回空（识别不到）。

        主C图标未命中时再匹配队友图标（teammate_skill_icon_templates）：
        命中说明技能页已稳定、本组三张卡都不是主C技能，写入
        ``teammate_skill_visible`` 供任务层立即随机选卡，不等满识别帧。
        """
        roi_name = self._cfg.skills.get("candidate_roi", "skill_candidates")
        roi = self._resolve_named_roi(str(roi_name), ctx)
        threshold = float(self._cfg.skills.get("threshold", 0.82))

        def _collect_on_verified_page() -> None:
            """技能卡采集（统计阶段可选）：在「页面标志可见 + 图标已实际命中」
            的帧裁卡——这正是即将点卡的时刻，卡片必然渲染完整、英雄图标必然
            在画面上，离线归属才不会因图标未渲染而失败。只看任务状态会裁到
            棋盘帧，只看页面标志会裁到卡片未渲染完的过渡帧。
            observe 契约保证不抛异常、不阻塞，失败只记日志。
            """
            if self._skill_collector is None:
                return
            if not any(
                raw_data.get(flag)
                for flag in (
                    "select_skill_button_visible",
                    "merge_gift_skill_page_visible",
                    "tian_shi_kai_ju_visible",
                )
            ):
                return
            self._skill_collector.observe(ctx.frame_id, frame, roi, page=page)

        # 技能标题 OCR 主路径（配置了优先级档位时启用）：一次横带 OCR 读出
        # 三张卡的技能名，产出 skill_card_options 交给任务层按档位选卡，
        # 图标匹配跳过。OCR 未配置/未读到任何标题时走下方图标识别兜底
        if (
            self._title_reader is not None
            and self._skill_tiers
            and roi is not None
            and any(
                raw_data.get(flag)
                for flag in (
                    "select_skill_button_visible",
                    "merge_gift_skill_page_visible",
                    "tian_shi_kai_ju_visible",
                )
            )
        ):
            options = self._read_card_options(ctx, frame, roi, page)
            if options:
                raw_data["skill_card_options"] = options
                return []

        if self._skill_icon_paths:
            skill_matches = self._vision.match_template_set(
                frame,
                self._skill_icon_paths,
                roi=roi,
                threshold=threshold,
            )
            if skill_matches:
                _collect_on_verified_page()
                matches.extend(skill_matches)
                candidates = [
                    SkillCandidate(
                        skill_id=self._main_c,
                        position=match.position,
                        confidence=match.confidence,
                        template_path=match.template_name,
                    )
                    for match in skill_matches
                ]
                if len(skill_matches) > 1:
                    logger.debug(
                        "frame=%d 主C技能图标命中 %d 个位置",
                        ctx.frame_id,
                        len(skill_matches),
                    )
                return candidates
        if self._teammate_skill_paths:
            teammate_matches = self._vision.match_template_set(
                frame,
                self._teammate_skill_paths,
                roi=roi,
                threshold=threshold,
            )
            if teammate_matches:
                _collect_on_verified_page()
                matches.extend(teammate_matches)
                raw_data["teammate_skill_visible"] = True
                raw_data["teammate_skill_match"] = teammate_matches[0]
                logger.debug(
                    "frame=%d 识别到队友技能卡图标 %d 个（本组无主C技能卡）",
                    ctx.frame_id,
                    len(teammate_matches),
                )
        return []

    def _read_title_lines(self, strip: Any) -> list[tuple[str, tuple[int, int]]]:
        """单张标题带 OCR；空图或异常返回空列表，不影响其他列。"""
        if strip is None or strip.size == 0:
            return []
        try:
            return self._title_reader.read(strip)
        except Exception as exc:  # noqa: BLE001 - 单列 OCR 失败跳过该列
            logger.warning("技能标题 OCR 失败（单列跳过） error=%r", exc)
            return []

    def _read_title_lines_fast(
        self, strips: list[Any]
    ) -> list[list[tuple[str, tuple[int, int]]] | None]:
        """免检测整行识别三列标题（快速路径，三线程并发）。

        每列：条带收紧到文字行 → 整行识别（引擎对宽高比 >8 的图跳过文本
        检测，见 perception.ocr 模块注释；单条 ~0.5s，三线程并发实测
        914ms/页，快于拼接整页单读的 1186ms/页——2026-09-13 离线 A/B）。

        逐条门控，返回列表元素语义：
        - list：接受该列结果（含空列表 = 条带无文字：识别置信度过滤与
          检测路径同源，检测同样读不出，不再浪费检测；过渡帧三列全空）
        - None：该列快速结果不可信（读出了名字但按精确或 0.7 模糊都对
          不上清单——整行误读/多行混杂），调用方只对该列回退检测路径
        """
        if self._title_reader is None or self._skill_tiers is None:
            return [None] * len(strips)
        with ThreadPoolExecutor(max_workers=3) as pool:
            return list(pool.map(self._read_strip_fast, strips))

    def _read_strip_fast(
        self, strip: Any
    ) -> list[tuple[str, tuple[int, int]]] | None:
        """单列标题的免检测整行识别 + 门控；None 表示需回退检测路径。"""
        if strip is None or strip.size == 0:
            return None
        try:
            lines = self._title_reader.read_line(tight_text_rows(strip))
        except Exception as exc:  # noqa: BLE001 - 快速路径失败该列退回检测
            logger.warning("技能标题快速识别异常，该列退回检测路径 error=%r", exc)
            return None
        if not lines:
            return []
        name = lines[0][0].strip()
        if name and self._known_title(name):
            return lines
        logger.debug("标题快速识别未能对应清单技能，该列退回检测路径 name=%r", name)
        return None

    def _known_title(self, name: str) -> bool:
        """标题能否按精确或模糊对上清单技能（与 _tier_and_description 同口径）。"""
        if name in self._skill_tiers:
            return True
        return find_fuzzy_match(name, self._tier_name_entries) is not None

    def _read_strips_separately(
        self, strips: list[Any]
    ) -> list[list[tuple[str, tuple[int, int]]]]:
        """逐列分别 OCR（检测路径）：三列并发，总耗时≈最慢一列。"""
        with ThreadPoolExecutor(max_workers=3) as pool:
            return list(pool.map(self._read_title_lines, strips))

    def _read_card_options(
        self,
        ctx: WindowContext,
        frame: Any,
        roi: tuple[int, int, int, int],
        page: str,
    ) -> list[dict[str, Any]]:
        """读出三张卡的技能名并产出档位选项（免检测整行识别优先）。

        主路径（2026-09-13 离线 2176 张采集卡验证 + A/B 对比）：三列标题带
        各自收紧到文字行后走免检测整行识别（引擎对宽高比 >8 的图跳过
        DBNet 检测——检测会把窄条带按 min_side=736 放大 7.5 倍推理，慢速机
        ~2.8s/条，是旧行为单帧识别数秒的主因），三线程并发 ~0.9s/页。
        逐列门控：读出的名字能按精确或 0.7 模糊对上清单才接受；读空视为
        条带无文字（过渡帧三列全空，直接判空不再付检测）；对不上清单的
        列回退逐列检测路径，正确性与旧行为一致，只退速度且回退粒度到列。

        未收录的技能 OCR 整卡补描述并记录到 new_skills.yaml，按第 11 档
        处理。全部列读取失败时返回空，调用方回退图标识别。
        """
        height, width = frame.shape[:2]
        rx, ry, rw, rh = roi
        rw = min(rw, width - rx)
        rh = min(rh, height - ry)
        if rw <= 0 or rh <= 0:
            return []
        column_width = rw / 3
        inset = int(column_width * 0.04)
        band_top = ry + int(rh * 0.06)
        band_bottom = ry + int(rh * 0.22)
        strips = []
        for column in range(3):
            left = int(rx + column * column_width) + inset
            right = int(rx + (column + 1) * column_width) - inset
            strips.append(frame[band_top:band_bottom, max(left, 0) : max(right, 0)])
        column_lines: list[list[tuple[str, tuple[int, int]]] | None] = (
            self._read_title_lines_fast(strips)
        )
        slow_idx = [i for i, lines in enumerate(column_lines) if lines is None]
        if slow_idx:
            logger.debug(
                "标题快速识别 %d 列未确认，仅这些列退回检测路径", len(slow_idx)
            )
            slow = self._read_strips_separately([strips[i] for i in slow_idx])
            for i, lines in zip(slow_idx, slow, strict=True):
                column_lines[i] = lines
        options: list[dict[str, Any]] = []
        for column, lines in enumerate(column_lines):
            if not lines:
                continue
            # 细带内正常只有一行；若多行取最上方（标题）
            name = sorted(lines, key=lambda item: item[1][1])[0][0].strip()
            if not name:
                continue
            tier, known, description = self._tier_and_description(
                frame, rx, ry, rw, rh, column_width, column, name
            )
            option: dict[str, Any] = {
                "column": column,
                "name": name,
                "tier": tier,
                "known": known,
                "page": page,
                "frame_id": ctx.frame_id,
                "position": (int(rx + (column + 0.5) * column_width), int(ry + rh // 2)),
            }
            if description is not None:
                option["description"] = description
            options.append(option)
        if options:
            logger.debug(
                "frame=%d 技能标题 OCR 产出 %s",
                ctx.frame_id,
                [(o["column"], o["name"], o["tier"]) for o in options],
            )
        return options

    def _tier_and_description(
        self,
        frame: Any,
        rx: int,
        ry: int,
        rw: int,
        rh: int,
        column_width: float,
        column: int,
        name: str,
    ) -> tuple[int, bool, str | None]:
        """返回 (档位, 是否清单已知技能, 未收录时补 OCR 的描述)。

        名称精确/模糊命中档位索引即已知；未收录时对整卡 OCR 一次，首行
        之外的文字拼为描述，并记录到 new_skills.yaml，按第 11 档处理。
        """
        if name in self._skill_tiers:
            return self._skill_tiers[name], True, None
        matched = find_fuzzy_match(name, self._tier_name_entries)
        if matched is not None:
            canonical = str(matched["name"])
            logger.debug("技能标题 %r 模糊匹配清单 %r", name, canonical)
            return self._skill_tiers[canonical], True, None
        description = ""
        inset = int(column_width * 0.04)
        left = int(rx + column * column_width) + inset
        right = int(rx + (column + 1) * column_width) - inset
        card = frame[ry : ry + rh, max(left, 0) : max(right, 0)]
        if card.size:
            try:
                card_lines = sorted(self._title_reader.read(card), key=lambda item: item[1][1])
                description = "".join(text for text, _ in card_lines[1:])
            except Exception as exc:  # noqa: BLE001 - 描述补 OCR 失败只影响记录质量
                logger.debug("未收录技能描述 OCR 失败 name=%s error=%r", name, exc)
        if self._new_skill_dir is not None:
            try:
                record_new_skill(self._new_skill_dir, name, description)
            except (OSError, ValueError) as exc:
                logger.warning("新技能记录失败 name=%s error=%r", name, exc)
        return 11, False, description

    def _read_gold(self, ctx: WindowContext, frame: Any) -> dict[str, Any] | None:
        """读取金币门控信息；只认色模式仅判定两个费用区域红/白。

        ROI 未标定时永久停用金币读取。
        """
        current = self._resolve_named_roi("gold_current_area", ctx)
        skill = self._resolve_named_roi("skill_cost_area", ctx)
        summon = self._resolve_named_roi("summon_cost_area", ctx)
        if self._gold_color_only:
            if skill is None or summon is None:
                logger.debug("只认色模式缺费用 ROI，停用金币读取")
                self._gold_reader = None
                return None
            info = self._gold_reader.read_affordability(frame, skill, summon)
            logger.debug(
                "frame=%d 金币只认色 skill红=%s(%.3f) summon红=%s(%.3f)",
                ctx.frame_id,
                info.skill_red,
                info.skill_red_ratio,
                info.summon_red,
                info.summon_red_ratio,
            )
            return {
                "current": None,
                "skill_cost": None,
                "summon_cost": None,
                "skill_red": info.skill_red,
                "summon_red": info.summon_red,
                "skill_red_ratio": info.skill_red_ratio,
                "summon_red_ratio": info.summon_red_ratio,
            }
        if current is None or skill is None or summon is None:
            logger.debug("金币感知 ROI 未标定，停用金币读取")
            self._gold_reader = None
            return None
        info = self._gold_reader.read(frame, current, skill, summon)
        logger.debug(
            "frame=%d 金币读取 current=%s skill=%s(红=%s) summon=%s(红=%s)",
            ctx.frame_id,
            info.current,
            info.skill_cost,
            info.skill_red,
            info.summon_cost,
            info.summon_red,
        )
        return {
            "current": info.current,
            "skill_cost": info.skill_cost,
            "summon_cost": info.summon_cost,
            "skill_red": info.skill_red,
            "summon_red": info.summon_red,
        }

    def match_ready_button(
        self,
        ctx: WindowContext,
        frame: Any,
    ) -> MatchResult | None:
        """仅匹配准备按钮模板，供抢合作检查线程周期调用。

        复用 ready_button locator 配置（模板/阈值/ROI），保证与 observe 中
        识别到的 ready_button 来自同一来源，避免两处阈值/ROI 漂移；
        locator 名同样走别名重映射（保卫宠物模式重映射到 pet_ready_button，
        宠物专属准备按钮与合作不是同一张图）。
        """
        locator = self._cfg.locators.get(
            self._locator_aliases.get("ready_button", "ready_button")
        )
        if not locator:
            return None
        return self._match_locator_template(ctx, frame, locator)

    def match_close_button(
        self,
        ctx: WindowContext,
        frame: Any,
    ) -> MatchResult | None:
        """仅匹配全屏弹窗右上角标准关闭按钮（×），供抢合作检查线程周期调用。

        复用 close_button locator 配置（模板/阈值/ROI），与 observe 中
        close_button 标志同源，避免两处阈值/ROI 漂移。× 是游戏标准控件、
        跨弹窗皮肤一致（2026-09-14 实机三主题 0.99+）；模板未采集的分辨率
        包返回 None，调用方按未识别处理。
        """
        locator = self._cfg.locators.get(
            self._locator_aliases.get("close_button", "close_button")
        )
        if not locator:
            return None
        return self._match_locator_template(ctx, frame, locator)

    def is_modal_frame(self, frame: Any, brightness_threshold: float) -> bool:
        """全屏模态弹窗检测：弹窗打开时游戏把画面两侧边缘压成近乎纯黑。

        2026-09-14 实机标定：个人信息弹窗三主题左右边缘平均亮度 23~31，
        普通界面（大厅/对局/结算）≥64。招募页等全屏面板本身也是暗的，
        因此本方法只说明「有全屏面板」，不区分具体弹窗，调用方需结合
        阶段信息使用。帧不是有效图像（None/形状异常）时保守返回 False。

        Args:
            frame: 截图帧（BGR ndarray）
            brightness_threshold: 边缘平均亮度判定阈值（低于视为有全屏面板）
        """
        if frame is None or not hasattr(frame, "shape") or frame.ndim < 2:
            return False
        height, width = frame.shape[:2]
        if height < 200 or width < 60:
            return False
        y0, y1 = int(height * 0.17), int(height * 0.81)
        margin = max(8, int(width * 0.033))
        left = frame[y0:y1, :margin]
        right = frame[y0:y1, width - margin :]
        try:
            brightness = (float(left.mean()) + float(right.mean())) / 2.0
        except Exception:
            return False
        return brightness < brightness_threshold

    def _match_locator_template(
        self,
        ctx: WindowContext,
        frame: Any,
        locator_cfg: dict[str, Any],
    ):
        """按 locator 配置做单模板匹配，返回 MatchResult 或 None。"""
        template_rel = locator_cfg.get("template")
        if not template_rel:
            return None
        threshold = float(locator_cfg.get("threshold", 0.82))
        roi_name = locator_cfg.get("roi")
        roi = self._resolve_named_roi(roi_name, ctx) if roi_name else None
        template_path = str(self._pack.resolve_template(template_rel))
        if not self._pack.resolve_template(template_rel).is_file():
            return None
        return self._vision.match_template(frame, template_path, roi=roi, threshold=threshold)

    def _resolve_named_roi(
        self,
        name: str | None,
        ctx: WindowContext,
    ) -> tuple[int, int, int, int] | None:
        """按名称从 rois 配置解析像素 ROI，返回 None 表示全图。"""
        if not name:
            return None
        roi_cfg = self._cfg.rois.get(name)
        if roi_cfg is None:
            return None
        pixel = ratio_to_pixel_roi(roi_cfg, ctx.client_size)
        # 0x0 视为未标定，返回 None 走全图
        if pixel[2] <= 0 or pixel[3] <= 0:
            return None
        return pixel

    def _observe_board(
        self,
        ctx: WindowContext,
        frame: Any,
        matches: list,
    ) -> BoardSnapshot:
        """批量分类单帧 12 格并构建 BoardSnapshot。"""
        predictions, centers = self._classify_board_frame(ctx, frame)
        heroes = self._heroes_from_predictions(predictions, centers)
        total_slots = len(predictions) - sum(
            prediction.class_name == "unavailable" for prediction in predictions
        )
        capacity = BoardCapacity(total_slots=total_slots, occupied=len(heroes))
        return BoardSnapshot(
            frame_id=ctx.frame_id,
            heroes=heroes,
            capacity=capacity,
            captured_at=ctx.captured_at,
            source_frame_ids=(ctx.frame_id,),
        )

    def _classify_board_frame(
        self,
        ctx: WindowContext,
        frame: Any,
    ) -> tuple[list[HeroCellPrediction], dict[tuple[int, int], tuple[int, int]]]:
        """按训练裁剪规则切出 12 格，并用一次 ONNX batch 完成分类。"""
        if self._hero_cell_classifier is None:
            raise RuntimeError("合作棋盘未配置英雄格 ONNX 分类器")
        grid = board_grid_for_role(self._role, self._cfg.board)
        crops: list[Any] = []
        centers: dict[tuple[int, int], tuple[int, int]] = {}
        for cell, (left, top, width, height) in hero_cell_rois(
            grid,
            self._role,
            ctx.client_size,
        ):
            crops.append(frame[top : top + height, left : left + width].copy())
            centers[(cell.row, cell.col)] = (left + width // 2, top + height // 2)
        predictions = [
            self._restrict_prediction(prediction)
            for prediction in self._hero_cell_classifier.predict(crops)
        ]
        return predictions, centers

    def _restrict_prediction(self, prediction: HeroCellPrediction) -> HeroCellPrediction:
        """把本局阵容不可能出现的英雄保守拒绝为 unknown。"""
        if prediction.hero_type is None or prediction.hero_type in self._allowed_heroes:
            return prediction
        return replace(
            prediction,
            class_name="unknown",
            hero_type=None,
            star_level=None,
            rejected=True,
            rejection_reason="hero_not_in_lineup",
        )

    def _heroes_from_predictions(
        self,
        predictions: list[HeroCellPrediction],
        centers: dict[tuple[int, int], tuple[int, int]],
    ) -> list[BoardHero]:
        heroes: list[BoardHero] = []
        for cell_key, prediction in zip(centers, predictions, strict=True):
            if prediction.hero_type is None or prediction.star_level is None:
                continue
            row, col = cell_key
            heroes.append(
                BoardHero(
                    hero_type=prediction.hero_type,
                    star_level=prediction.star_level,
                    position=centers[cell_key],
                    confidence=prediction.confidence,
                    template_path="",
                    cell_name=format_cell_label(
                        BoardCell(row, col, BoardCellType.HERO),
                        self._role,
                    ),
                )
            )
        return heroes

    def observe_cultivation(
        self,
        screen: Any,
        handle: int,
        ctx: WindowContext,
        n_frames: int = _DEFAULT_CULTIVATION_FRAMES,
        *,
        require_foreground: bool = True,
        read_gold: bool = False,
    ) -> tuple[WindowContext, Observation]:
        """多帧分类棋盘，并按每格精确类别做多数投票。

        ``unknown``、``empty`` 和 ``unavailable`` 都是有效票。只有某个精确类别
        超过该格有效帧的一半才采纳；并列或不过半时该格保守视为 unknown。

        Args:
            screen: ScreenCapture 实例（用于连续截图）
            handle: 窗口句柄
            ctx: 初始窗口上下文（提供客户区尺寸，窗口可能微移）
            n_frames: 累积帧数
            require_foreground: 是否要求游戏窗口为前台才投票。实战点击前保持
                True（防误点的安全前提）；纯观察类调用（如 exec watch-board）
                传 False——小程序嵌套窗口下前台句柄常是外层容器，内层游戏窗口
                永远判非前台，会导致全部帧被丢弃、棋盘恒为空。最小化检查始终
                保留（最小化时截不到有效画面）。

        Returns:
            (最新窗口上下文, Observation)，Observation.board 为累积快照
        """
        batch_started = perf_counter()
        capture_durations: list[float] = []
        recognition_durations: list[float] = []
        frame_durations: list[float] = []
        capture_failures = 0
        last_capture_error: Exception | None = None

        votes: dict[tuple[int, int], list[HeroCellPrediction]] = defaultdict(list)
        center_by_cell: dict[tuple[int, int], tuple[int, int]] = {}
        latest_ctx = ctx
        latest_frame_id = ctx.frame_id
        latest_frame: Any | None = None
        source_frame_ids: list[int] = []

        for _ in range(n_frames):
            frame_started = perf_counter()
            capture_started = perf_counter()
            try:
                ctx_i, frame_i = screen.capture(handle)
            except (OSError, RuntimeError) as exc:
                capture_failures += 1
                last_capture_error = exc
                continue
            capture_durations.append(perf_counter() - capture_started)
            latest_ctx = ctx_i
            latest_frame_id = ctx_i.frame_id
            if (
                ctx_i.window_handle != ctx.window_handle
                or ctx_i.client_rect_screen != ctx.client_rect_screen
                or ctx_i.client_size != ctx.client_size
                or ctx_i.dpi != ctx.dpi
            ):
                return ctx_i, Observation(
                    frame_id=ctx_i.frame_id,
                    source_frame_ids=(ctx_i.frame_id,),
                    raw_data={"window_invalid": True},
                )
            # 窗口无效则跳过这帧（不影响累积已识别的）
            # 前台检查可按需跳过：小程序嵌套窗口下前台句柄常是外层容器，
            # 内层游戏窗口恒判非前台，观察类调用会因此丢弃全部帧（棋盘恒空）
            if ctx_i.is_minimized or (require_foreground and not ctx_i.is_foreground):
                continue

            latest_frame = frame_i
            source_frame_ids.append(ctx_i.frame_id)
            recognition_started = perf_counter()
            predictions, centers = self._classify_board_frame(ctx_i, frame_i)
            recognition_durations.append(perf_counter() - recognition_started)
            frame_durations.append(perf_counter() - frame_started)
            center_by_cell = centers
            for cell_key, prediction in zip(centers, predictions, strict=True):
                votes[cell_key].append(prediction)

        voted_predictions = {
            cell_key: self._vote_cell(predictions) for cell_key, predictions in votes.items()
        }
        accepted_predictions = {
            cell_key: prediction
            for cell_key, prediction in voted_predictions.items()
            if prediction is not None
        }
        accumulated = self._heroes_from_predictions(
            list(accepted_predictions.values()),
            {cell_key: center_by_cell[cell_key] for cell_key in accepted_predictions},
        )
        unavailable_count = sum(
            prediction is not None and prediction.class_name == "unavailable"
            for prediction in voted_predictions.values()
        )
        total_slots = len(center_by_cell) - unavailable_count
        capacity = BoardCapacity(total_slots=total_slots, occupied=len(accumulated))
        board = BoardSnapshot(
            frame_id=latest_frame_id,
            heroes=accumulated,
            capacity=capacity,
            captured_at=latest_ctx.captured_at,
            source_frame_ids=tuple(source_frame_ids or [latest_frame_id]),
        )
        raw_data: dict[str, Any] = {}
        # 格子中心 -> 格名（如 1A/4B）映射，供观察类调用打印棋盘坐标；
        # 键与 BoardHero.position 同源（都是格子中心像素坐标）
        if center_by_cell:
            cell_by_center = {
                center: cell
                for cell, center in hero_cell_centers(
                    board_grid_for_role(self._role, self._cfg.board),
                    self._role,
                    latest_ctx.client_size,
                )
            }
            raw_data["cell_labels"] = {
                str(center): format_cell_label(cell_by_center[center], self._role)
                for center in center_by_cell.values()
                if center in cell_by_center
            }
        matches: list = []
        if latest_frame is not None:
            self._detect_interface_flags(
                latest_ctx,
                latest_frame,
                State.BUILD_MAIN_C,
                raw_data,
                matches,
            )
            # 金币感知：培养/回补阶段的召唤门控需要金币数据（按最小间隔读取）
            if self._gold_reader is not None:
                gold = self._read_gold(latest_ctx, latest_frame)
                if gold is not None:
                    raw_data["gold"] = gold
        observation = Observation(
            frame_id=latest_frame_id,
            source_frame_ids=tuple(source_frame_ids or [latest_frame_id]),
            matches=matches,
            raw_data=raw_data,
            board=board,
        )
        batch_duration = perf_counter() - batch_started
        frame_avg, frame_min, frame_max = self._duration_stats(frame_durations)
        match_avg, match_min, match_max = self._duration_stats(recognition_durations)
        capture_avg, _, _ = self._duration_stats(capture_durations)
        if capture_failures:
            logger.warning(
                "棋盘多帧识别截图失败汇总 failures=%d/%d last_reason=%r",
                capture_failures,
                n_frames,
                last_capture_error,
            )
        logger.debug(
            "棋盘多帧识别耗时 配置帧数=%d 截图成功=%d 实际识别=%d 截图失败=%d "
            "总耗时=%.3fs 单帧总耗时(avg/min/max)=%.3f/%.3f/%.3fs "
            "单帧截图平均=%.3fs 单帧模型推理(avg/min/max)=%.3f/%.3f/%.3fs "
            "最终英雄=%d 最新frame_id=%d",
            n_frames,
            len(capture_durations),
            len(recognition_durations),
            capture_failures,
            batch_duration,
            frame_avg,
            frame_min,
            frame_max,
            capture_avg,
            match_avg,
            match_min,
            match_max,
            len(accumulated),
            latest_frame_id,
        )
        return latest_ctx, observation

    @staticmethod
    def _vote_cell(predictions: list[HeroCellPrediction]) -> HeroCellPrediction | None:
        """对单格的多帧精确类别投票；不过半时返回 None。"""
        if not predictions:
            return None
        counts = Counter(prediction.class_name for prediction in predictions)
        winner, votes = counts.most_common(1)[0]
        if votes * 2 <= len(predictions):
            return None
        candidates = [prediction for prediction in predictions if prediction.class_name == winner]
        return max(candidates, key=lambda prediction: (prediction.confidence, prediction.margin))

    @staticmethod
    def _duration_stats(durations: list[float]) -> tuple[float, float, float]:
        """返回耗时列表的平均值、最小值和最大值；空列表统一返回 0。"""
        if not durations:
            return (0.0, 0.0, 0.0)
        return (sum(durations) / len(durations), min(durations), max(durations))
