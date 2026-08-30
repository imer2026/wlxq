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
from dataclasses import replace
from time import perf_counter
from typing import Any

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
from wlxq_bot.perception.vision import Vision
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
    # 进入游戏后的两种开局标识：先「选择技能」再「召唤」，二者互斥，
    # 出现其一即确认对局界面已稳定及其开局类型（见 CoopTask._action_opening_skills）。
    ("select_skill_button", "select_skill_button_visible", "select_skill_button_match"),
    # 【请选择1个额外技能】提示条：合成4星赠送技能页的主标识（该页上【选技能】图
    # 不稳定，实机 2026-08-21）；仅 BUILD 状态识别
    ("merge_gift_skill_title", "merge_gift_skill_page_visible", "merge_gift_skill_title_match"),
    ("summon_button", "summon_button_visible", "summon_button_match"),
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
    State.ENTER_MATCH: frozenset({"ready_button"}),
    State.SELECT_OPENING_SKILLS: frozenset({"select_skill_button", "summon_button", "tan_chuang"}),
    State.BUILD_MAIN_C: frozenset(
        {
            "select_skill_button",
            "merge_gift_skill_title",
            "summon_button",
            "tan_chuang",
            "return_button",
        }
    ),
    State.SELECT_MAIN_C_SKILLS: frozenset({"select_skill_button", "tan_chuang", "return_button"}),
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
    ) -> Observation:
        """对一帧截图做识别，返回 Observation。

        Args:
            ctx: 当前窗口上下文（提供客户区尺寸和 frame_id）
            frame: 截图帧（BGR ndarray）
            hint_state: 上一轮状态，用于决定是否做棋盘专项识别
            observation_mode: 任务内部步骤要求的专项识别模式

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
        )
        difficulty_candidates = []
        if observation_mode == "coop_difficulty":
            self._detect_difficulty_dialog_flag(ctx, frame, raw_data, matches)
            difficulty_candidates = self._detect_difficulty_candidates(ctx, frame, matches)
        elif observation_mode == "difficulty_dialog":
            # 关闭/确认关闭难度弹窗：只判定弹窗开关标识，跳过 16 个难度候选
            # 匹配，控制单帧识别耗时（难度候选只用于勾选步骤）
            self._detect_difficulty_dialog_flag(ctx, frame, raw_data, matches)

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
    ) -> list[SkillCandidate]:
        """按状态识别需要的界面标志按钮，结果写入 raw_data（bool）和 matches。

        只匹配 _INTERFACE_FLAGS_BY_STATE 中当前状态声明的标志；抢合作子步骤
        （observation_mode="coop_grab"）额外查准备按钮。识别不到的标志不写入
        raw_data，任务层 flag() 读取时按 False 处理。
        """
        locators = self._cfg.locators
        wanted = _INTERFACE_FLAGS_BY_STATE.get(hint_state, frozenset())
        if observation_mode == "coop_grab":
            wanted = wanted | {"ready_button"}

        for locator_name, flag_name, match_name in _INTERFACE_FLAG_LOCATORS:
            if locator_name not in wanted:
                continue
            locator = locators.get(locator_name)
            match = self._match_locator_template(ctx, frame, locator) if locator else None
            raw_data[flag_name] = match is not None
            if match is not None:
                matches.append(match)
                raw_data[match_name] = match

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
            leave_locator = locators.get("coop_leave_team")
            leave_match = (
                self._match_locator_template(ctx, frame, leave_locator) if leave_locator else None
            )
            raw_data["leave_team_visible"] = leave_match is not None
            if leave_match is not None:
                matches.append(leave_match)
                raw_data["leave_team_match"] = leave_match

        if hint_state in {State.SELECT_OPENING_SKILLS, State.SELECT_MAIN_C_SKILLS}:
            return self._detect_skill_candidates(ctx, frame, matches, raw_data)
        if hint_state == State.BUILD_MAIN_C and (
            raw_data.get("select_skill_button_visible")
            or raw_data.get("merge_gift_skill_page_visible")
        ):
            # 合成 4 星后系统赠送的技能选择页（实机确认 2026-08-16）：与常规
            # 技能页同一个 3 选 1 界面，会遮住棋盘导致识别为空。页面打开时才做
            # 技能卡识别，不增加常规培养帧的开销；主标识为【请选择1个额外技能】
            # 提示条（2026-08-21 实机确认【选技能】图在该页不稳定），命中任一即识别
            return self._detect_skill_candidates(ctx, frame, matches, raw_data)
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

    def _detect_skill_candidates(
        self,
        ctx: WindowContext,
        frame: Any,
        matches: list,
        raw_data: dict[str, Any],
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
        if self._skill_icon_paths:
            skill_matches = self._vision.match_template_set(
                frame,
                self._skill_icon_paths,
                roi=roi,
                threshold=threshold,
            )
            if skill_matches:
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
                matches.extend(teammate_matches)
                raw_data["teammate_skill_visible"] = True
                raw_data["teammate_skill_match"] = teammate_matches[0]
                logger.debug(
                    "frame=%d 识别到队友技能卡图标 %d 个（本组无主C技能卡）",
                    ctx.frame_id,
                    len(teammate_matches),
                )
        return []

    def match_ready_button(
        self,
        ctx: WindowContext,
        frame: Any,
    ) -> MatchResult | None:
        """仅匹配准备按钮模板，供抢合作检查线程周期调用。

        复用 ready_button locator 配置（模板/阈值/ROI），保证与 observe 中
        识别到的 ready_button 来自同一来源，避免两处阈值/ROI 漂移。
        """
        locator = self._cfg.locators.get("ready_button")
        if not locator:
            return None
        return self._match_locator_template(ctx, frame, locator)

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
