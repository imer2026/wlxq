"""CoopPerception 的 ONNX 棋盘分类与多帧投票测试。"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from wlxq_bot.config import BoardGridParams, TasksConfig
from wlxq_bot.models import (
    CoopRole,
    MatchResult,
    State,
    WindowContext,
)
from wlxq_bot.perception.coop import CoopPerception
from wlxq_bot.perception.hero_classifier import HeroCellPrediction

# ---------------------------------------------------------------------------
# 多帧投票（observe_cultivation）
# ---------------------------------------------------------------------------


class FakeVision:
    """假 Vision，按调用顺序返回预设 match_template_set 结果。"""

    def __init__(self, frame_results: list[list[MatchResult]]) -> None:
        self._results = frame_results
        self._call = 0

    def match_template_set(self, frame, template_paths, roi=None, threshold=0.78, nms_dist=None):
        idx = min(self._call, len(self._results) - 1)
        result = self._results[idx]
        self._call += 1
        return result

    def match_template(self, frame, template_path, roi=None, threshold=0.85):
        return None


def _prediction(class_name: str, confidence: float = 0.9) -> HeroCellPrediction:
    hero_type = None
    star_level = None
    if "_star" in class_name:
        hero_type, star = class_name.rsplit("_star", 1)
        star_level = int(star)
    return HeroCellPrediction(
        class_name=class_name,
        hero_type=hero_type,
        star_level=star_level,
        confidence=confidence,
        margin=0.5,
        rejected=class_name == "unknown",
        raw_class_name=class_name,
    )


class FakeCellClassifier:
    supported_heroes = frozenset({"assault", "angel", "snow", "death_knight"})

    def __init__(self, frame_classes: list[list[HeroCellPrediction]]) -> None:
        self._frames = frame_classes
        self._call = 0
        self.batch_shapes: list[tuple[int, ...]] = []

    def predict(self, images):
        assert len(images) == 12
        self.batch_shapes.append(tuple(images[0].shape))
        result = self._frames[min(self._call, len(self._frames) - 1)]
        self._call += 1
        return result


def _board_predictions(first: HeroCellPrediction, rest: str = "empty") -> list[HeroCellPrediction]:
    return [first, *[_prediction(rest) for _ in range(11)]]


class DifficultyVision:
    """按难度模板文件名返回预设匹配。"""

    def __init__(self, matches: dict[int, MatchResult]) -> None:
        self._matches = matches
        self.rois = []

    def match_template(self, frame, template_path, roi=None, threshold=0.85):
        self.rois.append(roi)
        level = int(template_path.rsplit("_", 1)[1].removesuffix(".png"))
        return self._matches.get(level)

    def match_template_set(self, frame, template_paths, roi=None, threshold=0.78, nms_dist=None):
        return []


class HomePageVision:
    """仅对首页标志模板返回固定命中。"""

    def __init__(self, match: MatchResult) -> None:
        self.match = match
        self.template_calls: list[str] = []

    def match_template(self, frame, template_path, roi=None, threshold=0.85):
        self.template_calls.append(template_path)
        return self.match if template_path.endswith("home_page_marker.png") else None

    def match_template_set(self, frame, template_paths, roi=None, threshold=0.78, nms_dist=None):
        return []


class FakeScreen:
    """假截图器，返回固定 WindowContext。"""

    def __init__(self, ctx: WindowContext) -> None:
        self._ctx = ctx
        self.capture_count = 0
        self.last_frame = None

    def capture(self, handle):
        self.capture_count += 1
        self.last_frame = np.zeros((1723, 923, 3), dtype=np.uint8)
        return self._ctx, self.last_frame


def _make_window_ctx() -> WindowContext:
    import time as _time

    return WindowContext(
        window_handle=1,
        client_rect_screen=(0, 0, 923, 1723),
        client_size=(923, 1723),
        dpi=96,
        monitor_id="primary",
        is_foreground=True,
        is_minimized=False,
        captured_at=_time.time(),
        frame_id=1,
    )


def _helper_board_params() -> BoardGridParams:
    """tasks.yaml 的 helper 棋盘格子参数（量测自 923x1723）。"""
    return BoardGridParams(
        anchor_x_ratio=0.6229,
        anchor_y_ratio=0.5148,
        col_step_ratio=0.0791,
        row_step_ratio=0.0432,
        cell_width_ratio=0.0737,
        cell_height_ratio=0.0360,
    )


def _build_perception(tmp_path, classifier) -> CoopPerception:
    """构造 CoopPerception，模板包指向临时目录。"""
    from wlxq_bot.assets import TemplatePack

    pack_root = tmp_path / "3000x2000"
    pack_root.mkdir(parents=True)

    pack = TemplatePack(client_size=(3000, 2000), root=pack_root)
    tasks_cfg = TasksConfig(board={"helper": _helper_board_params()})
    return CoopPerception(
        FakeVision([]),
        pack,
        tasks_cfg,
        CoopRole.HELPER,
        "assault",
        hero_cell_classifier=classifier,
        allowed_heroes={"assault", "angel", "snow", "death_knight"},
    )


class TestObserveCultivation:
    def test_majority_vote_accepts_exact_hero_class(self, tmp_path):
        classifier = FakeCellClassifier(
            [
                _board_predictions(_prediction("assault_star2", 0.80)),
                _board_predictions(_prediction("unknown", 0.70)),
                _board_predictions(_prediction("assault_star2", 0.90)),
            ]
        )
        perception = _build_perception(tmp_path, classifier)
        ctx = _make_window_ctx()
        screen = FakeScreen(ctx)

        latest_ctx, obs = perception.observe_cultivation(screen, 1, ctx, n_frames=3)

        assert obs.board is not None
        assert len(obs.board.heroes) == 1
        assert obs.board.heroes[0].hero_type == "assault"
        assert obs.board.heroes[0].star_level == 2
        assert obs.board.heroes[0].confidence == pytest.approx(0.90)
        # helper 第一个英雄格是排1外侧列（A），识别结果应携带格名
        assert obs.board.heroes[0].cell_name == "1A"
        assert screen.capture_count == 3
        assert classifier.batch_shapes == [(62, 68, 3)] * 3

    def test_logs_batch_and_per_frame_timing_summary(self, tmp_path):
        """每批只打一条汇总日志，包含总耗时和单帧模板匹配耗时。"""
        classifier = FakeCellClassifier([_board_predictions(_prediction("empty"))] * 3)
        perception = _build_perception(tmp_path, classifier)
        ctx = _make_window_ctx()
        screen = FakeScreen(ctx)

        with patch("wlxq_bot.perception.coop.logger.debug") as debug:
            perception.observe_cultivation(screen, 1, ctx, n_frames=3)

        timing_calls = [
            call for call in debug.call_args_list if call.args[0].startswith("棋盘多帧识别耗时")
        ]
        assert len(timing_calls) == 1
        args = timing_calls[0].args
        assert args[1:5] == (3, 3, 3, 0)
        assert "总耗时" in args[0]
        assert "单帧模型推理" in args[0]

    def test_tie_is_rejected_as_unknown(self, tmp_path):
        classifier = FakeCellClassifier(
            [
                _board_predictions(_prediction("assault_star1")),
                _board_predictions(_prediction("angel_star1")),
            ]
        )
        perception = _build_perception(tmp_path, classifier)
        ctx = _make_window_ctx()
        screen = FakeScreen(ctx)

        _, obs = perception.observe_cultivation(screen, 1, ctx, n_frames=2)

        assert obs.board is not None
        assert obs.board.heroes == []

    def test_empty_when_all_frames_miss(self, tmp_path):
        """所有帧都漏检，累积为空棋盘。"""
        classifier = FakeCellClassifier([_board_predictions(_prediction("unknown"))] * 3)
        perception = _build_perception(tmp_path, classifier)
        ctx = _make_window_ctx()
        screen = FakeScreen(ctx)

        _, obs = perception.observe_cultivation(screen, 1, ctx, n_frames=3)

        assert obs.board is not None
        assert len(obs.board.heroes) == 0

    def test_rejects_hero_not_in_current_lineup(self, tmp_path):
        classifier = FakeCellClassifier([_board_predictions(_prediction("monkey_star2"))] * 3)
        perception = _build_perception(tmp_path, classifier)

        _, obs = perception.observe_cultivation(
            FakeScreen(_make_window_ctx()), 1, _make_window_ctx(), n_frames=3
        )

        assert obs.board is not None
        assert obs.board.heroes == []


class RecordingGoldCache:
    """记录 submit 调用的假金币缓存：snapshot 返回固定快照。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def snapshot(self):
        return {"current": 2000, "summon_cost": 320}

    def submit(self, frame, **kwargs):
        self.calls.append({"frame": frame, **kwargs})


def _build_perception_with_gold(tmp_path, classifier, gold_cache):
    """带金币 ROI 标定与注入缓存的 CoopPerception。"""
    from wlxq_bot.assets import TemplatePack
    from wlxq_bot.config import RoiConfig

    pack_root = tmp_path / "3000x2000"
    pack_root.mkdir(parents=True)
    pack = TemplatePack(client_size=(3000, 2000), root=pack_root)
    tasks_cfg = TasksConfig(
        board={"helper": _helper_board_params()},
        rois={
            "gold_current_area": RoiConfig(
                x_ratio=0.30, y_ratio=0.855, width_ratio=0.10, height_ratio=0.02
            ),
            "summon_cost_area": RoiConfig(
                x_ratio=0.62, y_ratio=0.955, width_ratio=0.10, height_ratio=0.02
            ),
        },
    )
    return CoopPerception(
        FakeVision([]),
        pack,
        tasks_cfg,
        CoopRole.HELPER,
        "assault",
        hero_cell_classifier=classifier,
        allowed_heroes={"assault", "angel", "snow", "death_knight"},
        gold_cache=gold_cache,
    )


class TestObserveCultivationGoldFeed:
    def test_multi_frame_path_submits_frame_to_gold_cache(self, tmp_path):
        """回归（2026-09-27 实机）：多帧路径只读快照不喂帧，金币等待循环
        全走本路径时信箱断粮、快照冻结在最后好值 30s。"""
        classifier = FakeCellClassifier([_board_predictions(_prediction("empty"))] * 3)
        gold_cache = RecordingGoldCache()
        perception = _build_perception_with_gold(tmp_path, classifier, gold_cache)
        ctx = _make_window_ctx()
        screen = FakeScreen(ctx)

        latest_ctx, obs = perception.observe_cultivation(screen, 1, ctx, n_frames=3)

        # 每批喂一次（最新帧），不是每帧一次
        assert len(gold_cache.calls) == 1
        call = gold_cache.calls[0]
        assert call["frame"] is screen.last_frame
        assert call["frame_id"] == latest_ctx.frame_id
        assert call["clean"] is True
        assert call["cost_key"] == "summon_cost"
        assert call["current_roi"] == (276, 1473, 92, 34)
        assert obs.raw_data["gold"] == {"current": 2000, "summon_cost": 320}


class TestDifficultyObservation:
    def test_detects_visible_levels_only_in_requested_mode(self, tmp_path):
        from wlxq_bot.assets import TemplatePack

        pack_root = tmp_path / "3000x2000"
        difficulty_dir = pack_root / "buttons" / "coop_difficulty"
        difficulty_dir.mkdir(parents=True)
        for level in (9, 10):
            (difficulty_dir / f"cai_hong_{level}.png").write_bytes(b"template")

        vision = DifficultyVision(
            {
                9: MatchResult("cai_hong_9.png", (300, 500), 0.91),
                10: MatchResult("cai_hong_10.png", (300, 400), 0.93),
            }
        )
        tasks_cfg = TasksConfig(
            rois={
                "coop_difficulty_list": {
                    "x_ratio": 0.2,
                    "y_ratio": 0.3,
                    "width_ratio": 0.4,
                    "height_ratio": 0.5,
                }
            },
            difficulty_recognition={"candidate_roi": "coop_difficulty_list"},
        )
        perception = CoopPerception(
            vision,
            TemplatePack(client_size=(3000, 2000), root=pack_root),
            tasks_cfg,
            CoopRole.HELPER,
            "assault",
        )
        ctx = _make_window_ctx()

        normal = perception.observe(ctx, None, observation_mode=None)
        difficulty = perception.observe(ctx, None, observation_mode="coop_difficulty")

        assert normal.difficulty_candidates == []
        assert [item.level for item in difficulty.difficulty_candidates] == [10, 9]
        assert vision.rois == [(184, 516, 369, 861), (184, 516, 369, 861)]


class TestHomePageObservation:
    def test_detects_home_page_only_in_requested_mode(self, tmp_path):
        from wlxq_bot.assets import TemplatePack

        pack_root = tmp_path / "3000x2000"
        buttons_dir = pack_root / "buttons"
        buttons_dir.mkdir(parents=True)
        (buttons_dir / "home_page_marker.png").write_bytes(b"template")
        match = MatchResult("home_page_marker.png", (100, 200), 0.93)
        vision = HomePageVision(match)
        tasks_cfg = TasksConfig(
            locators={
                "home_page_marker": {
                    "strategy": "template",
                    "template": "buttons/home_page_marker.png",
                    "threshold": 0.82,
                }
            }
        )
        perception = CoopPerception(
            vision,
            TemplatePack(client_size=(3000, 2000), root=pack_root),
            tasks_cfg,
            CoopRole.HELPER,
            "assault",
        )

        normal = perception.observe(_make_window_ctx(), None, observation_mode=None)
        home = perception.observe(_make_window_ctx(), None, observation_mode="home_page")

        assert "home_page_visible" not in normal.raw_data
        assert home.flag("home_page_visible")
        assert home.raw_data["home_page_match"] == match
        assert len(vision.template_calls) == 1


class _SingleTemplateVision:
    """仅对指定模板返回固定命中（或始终未命中）。"""

    def __init__(self, template_name: str, match: MatchResult | None) -> None:
        self._template_name = template_name
        self._match = match
        self.template_calls: list[str] = []

    def match_template(self, frame, template_path, roi=None, threshold=0.85):
        self.template_calls.append(template_path)
        return self._match if template_path.endswith(self._template_name) else None

    def match_template_set(self, frame, template_paths, roi=None, threshold=0.78, nms_dist=None):
        return []


class TestAngelOpeningObservation:
    def test_detects_tian_shi_marker_only_in_opening_state(self, tmp_path):
        """天使开局标识只在开局技能阶段识别，其余状态不产生该标志。"""
        from wlxq_bot.assets import TemplatePack
        from wlxq_bot.models import State

        pack_root = tmp_path / "3000x2000"
        buttons_dir = pack_root / "buttons"
        buttons_dir.mkdir(parents=True)
        (buttons_dir / "tian_shi_kai_ju.png").write_bytes(b"template")
        match = MatchResult("tian_shi_kai_ju.png", (100, 200), 0.93)
        vision = _SingleTemplateVision("tian_shi_kai_ju.png", match)
        tasks_cfg = TasksConfig(
            locators={
                "tian_shi_kai_ju": {
                    "strategy": "template",
                    "template": "buttons/tian_shi_kai_ju.png",
                    "threshold": 0.82,
                }
            }
        )
        perception = CoopPerception(
            vision,
            TemplatePack(client_size=(3000, 2000), root=pack_root),
            tasks_cfg,
            CoopRole.HELPER,
            "assault",
        )

        opening = perception.observe(
            _make_window_ctx(), None, hint_state=State.SELECT_OPENING_SKILLS
        )
        other = perception.observe(_make_window_ctx(), None, hint_state=State.SELECT_MAIN_C_SKILLS)

        assert opening.flag("tian_shi_kai_ju_visible")
        assert opening.raw_data["tian_shi_kai_ju_match"] == match
        assert "tian_shi_kai_ju_visible" not in other.raw_data

    def test_marker_miss_reports_not_visible(self, tmp_path):
        from wlxq_bot.assets import TemplatePack
        from wlxq_bot.models import State

        pack_root = tmp_path / "3000x2000"
        buttons_dir = pack_root / "buttons"
        buttons_dir.mkdir(parents=True)
        (buttons_dir / "tian_shi_kai_ju.png").write_bytes(b"template")
        vision = _SingleTemplateVision("tian_shi_kai_ju.png", None)
        tasks_cfg = TasksConfig(
            locators={
                "tian_shi_kai_ju": {
                    "strategy": "template",
                    "template": "buttons/tian_shi_kai_ju.png",
                    "threshold": 0.82,
                }
            }
        )
        perception = CoopPerception(
            vision,
            TemplatePack(client_size=(3000, 2000), root=pack_root),
            tasks_cfg,
            CoopRole.HELPER,
            "assault",
        )

        opening = perception.observe(
            _make_window_ctx(), None, hint_state=State.SELECT_OPENING_SKILLS
        )

        assert opening.flag("tian_shi_kai_ju_visible") is False


class TestDifficultyDialogFlagObservation:
    def test_dialog_flag_only_in_difficulty_mode(self, tmp_path):
        """【合作模式】标识只在难度弹窗识别模式下检测，命中与否都写入显式标志。"""
        from wlxq_bot.assets import TemplatePack

        pack_root = tmp_path / "3000x2000"
        difficulty_dir = pack_root / "buttons" / "coop_difficulty"
        difficulty_dir.mkdir(parents=True)
        (difficulty_dir / "he_zuo_mo_shi.png").write_bytes(b"template")
        match = MatchResult("he_zuo_mo_shi.png", (100, 200), 0.93)
        vision = _SingleTemplateVision("he_zuo_mo_shi.png", match)
        tasks_cfg = TasksConfig(
            locators={
                "he_zuo_mo_shi": {
                    "strategy": "template",
                    "template": "buttons/coop_difficulty/he_zuo_mo_shi.png",
                    "threshold": 0.82,
                }
            }
        )
        perception = CoopPerception(
            vision,
            TemplatePack(client_size=(3000, 2000), root=pack_root),
            tasks_cfg,
            CoopRole.HELPER,
            "assault",
        )

        difficulty = perception.observe(
            _make_window_ctx(), None, observation_mode="coop_difficulty"
        )
        normal = perception.observe(_make_window_ctx(), None, observation_mode=None)

        assert difficulty.flag("difficulty_dialog_visible")
        assert difficulty.raw_data["difficulty_dialog_match"] == match
        assert "difficulty_dialog_visible" not in normal.raw_data

    def test_difficulty_dialog_mode_skips_candidate_matching(self, tmp_path):
        """关闭弹窗步骤的轻量模式：只识别开关标识，不做 16 个难度候选匹配。"""
        from wlxq_bot.assets import TemplatePack

        pack_root = tmp_path / "3000x2000"
        buttons_dir = pack_root / "buttons"
        difficulty_dir = buttons_dir / "coop_difficulty"
        difficulty_dir.mkdir(parents=True)
        (difficulty_dir / "he_zuo_mo_shi.png").write_bytes(b"template")
        (difficulty_dir / "cai_hong_9.png").write_bytes(b"template")
        match = MatchResult("he_zuo_mo_shi.png", (100, 200), 0.93)
        vision = _SingleTemplateVision("he_zuo_mo_shi.png", match)
        tasks_cfg = TasksConfig(
            locators={
                "he_zuo_mo_shi": {
                    "strategy": "template",
                    "template": "buttons/coop_difficulty/he_zuo_mo_shi.png",
                    "threshold": 0.82,
                }
            }
        )
        perception = CoopPerception(
            vision,
            TemplatePack(client_size=(3000, 2000), root=pack_root),
            tasks_cfg,
            CoopRole.HELPER,
            "assault",
        )

        dialog_only = perception.observe(
            _make_window_ctx(), None, observation_mode="difficulty_dialog"
        )

        assert dialog_only.flag("difficulty_dialog_visible")
        assert dialog_only.raw_data["difficulty_dialog_match"] == match
        assert dialog_only.difficulty_candidates == []
        assert not any("cai_hong" in call for call in vision.template_calls)


class _RecordingVision:
    """记录全部单模板匹配调用（按模板文件名），可对指定文件返回命中。"""

    def __init__(self, hits: dict[str, MatchResult] | None = None) -> None:
        self.template_calls: list[str] = []
        self._hits = hits or {}

    def match_template(self, frame, template_path, roi=None, threshold=0.85):
        name = template_path.replace("\\", "/").rsplit("/", 1)[-1]
        self.template_calls.append(name)
        return self._hits.get(name)

    def match_template_set(self, frame, template_paths, roi=None, threshold=0.78, nms_dist=None):
        return []


_FLAG_LOCATOR_FILES = {
    "ready_button": "zhun_bei.png",
    "return_button": "fan_hui.png",
    "tan_chuang": "tan_chuang.png",
    "select_skill_button": "xuan_ze_ji_neng.png",
    "merge_gift_skill_title": "4_xing_e_wai_ji_neng.png",
    "summon_button": "zhao_huan.png",
    "skill_refresh_button": "shua_xin.png",
    "close_button": "x.png",
    "invite_start_ready": "tiao_zhan_dui_you_zhun_bei.png",
    "invite_start_not_ready": "tiao_zhan_dui_you_wei_zhun_bei.png",
}


def _flag_gated_perception(
    tmp_path, vision, *, skill_refresh_enabled: bool = False, role=CoopRole.HELPER
) -> CoopPerception:
    from wlxq_bot.assets import TemplatePack

    pack_root = tmp_path / "3000x2000"
    buttons_dir = pack_root / "buttons"
    buttons_dir.mkdir(parents=True, exist_ok=True)
    for file in _FLAG_LOCATOR_FILES.values():
        (buttons_dir / file).write_bytes(b"template")
    tasks_cfg = TasksConfig(
        locators={
            name: {"strategy": "template", "template": f"buttons/{file}", "threshold": 0.82}
            for name, file in _FLAG_LOCATOR_FILES.items()
        }
    )
    return CoopPerception(
        vision,
        TemplatePack(client_size=(3000, 2000), root=pack_root),
        tasks_cfg,
        role,
        "assault",
        skill_refresh_enabled=skill_refresh_enabled,
    )


class TestInterfaceFlagGating:
    """全局界面标志按状态门控：入口阶段不查，抢合作只查准备按钮。"""

    def test_recruit_entry_matches_no_interface_flags(self, tmp_path):
        """招募入口子步骤：游戏未开始也没点过加入，五个全局标志都不查。"""
        from wlxq_bot.models import State

        vision = _RecordingVision()
        perception = _flag_gated_perception(tmp_path, vision)

        for mode in (None, "home_page", "difficulty_dialog", "coop_difficulty"):
            raw: dict = {}
            perception._detect_interface_flags(
                _make_window_ctx(), None, State.FIND_COOP, raw, [], observation_mode=mode
            )
        assert vision.template_calls == []
        assert raw == {}

    def test_grab_mode_matches_ready_and_close_buttons(self, tmp_path):
        """抢合作子步骤：额外查准备按钮和弹窗关闭按钮，其余标志仍不查。"""
        from wlxq_bot.models import State

        match = MatchResult("zhun_bei.png", (500, 1000), 0.9)
        vision = _RecordingVision({"zhun_bei.png": match})
        perception = _flag_gated_perception(tmp_path, vision)

        raw: dict = {}
        matches: list = []
        perception._detect_interface_flags(
            _make_window_ctx(),
            None,
            State.FIND_COOP,
            raw,
            matches,
            observation_mode="coop_grab",
        )

        assert sorted(vision.template_calls) == ["x.png", "zhun_bei.png"]
        assert raw["ready_button_visible"] is True
        assert raw["ready_button_match"] == match

    def test_states_match_only_declared_flags(self, tmp_path):
        """各状态只匹配自己声明的标志集合。"""
        from wlxq_bot.models import State

        expectations = {
            State.ENTER_MATCH: {"zhun_bei.png"},
            State.SELECT_OPENING_SKILLS: {
                "xuan_ze_ji_neng.png",
                "zhao_huan.png",
                "tan_chuang.png",
                "fan_hui.png",
            },
            State.BUILD_MAIN_C: {
                "xuan_ze_ji_neng.png",
                "4_xing_e_wai_ji_neng.png",
                "zhao_huan.png",
                "tan_chuang.png",
                "fan_hui.png",
            },
            State.SELECT_MAIN_C_SKILLS: {"xuan_ze_ji_neng.png", "tan_chuang.png", "fan_hui.png"},
            State.HANDLE_RESULT: {"fan_hui.png"},
            State.CHECK_ROUND_LIMIT: {"fan_hui.png"},
        }
        for state, expected in expectations.items():
            vision = _RecordingVision()
            perception = _flag_gated_perception(tmp_path, vision)
            perception._detect_interface_flags(_make_window_ctx(), None, state, {}, [])
            assert set(vision.template_calls) == expected, state

    def test_skill_refresh_flag_gated_by_config(self, tmp_path):
        """刷新按钮标志：默认关闭不匹配；开启后仅局内选技能状态匹配并写出命中。"""
        from wlxq_bot.models import State

        vision_off = _RecordingVision()
        perception_off = _flag_gated_perception(
            tmp_path, vision_off, skill_refresh_enabled=False
        )
        raw_off: dict = {}
        perception_off._detect_interface_flags(
            _make_window_ctx(), None, State.SELECT_MAIN_C_SKILLS, raw_off, []
        )
        assert "shua_xin.png" not in vision_off.template_calls
        assert "skill_refresh_visible" not in raw_off

        vision_on = _RecordingVision(
            {"shua_xin.png": MatchResult("shua_xin.png", (472, 1321), 0.99)}
        )
        perception_on = _flag_gated_perception(
            tmp_path, vision_on, skill_refresh_enabled=True
        )
        raw_on: dict = {}
        perception_on._detect_interface_flags(
            _make_window_ctx(), None, State.SELECT_MAIN_C_SKILLS, raw_on, []
        )
        assert "shua_xin.png" in vision_on.template_calls
        assert raw_on["skill_refresh_visible"] is True
        assert raw_on["skill_refresh_match"].position == (472, 1321)

        # 开局技能状态即使开启也不识别刷新按钮（刷新只在局内「请选择」页）
        vision_opening = _RecordingVision()
        perception_opening = _flag_gated_perception(
            tmp_path, vision_opening, skill_refresh_enabled=True
        )
        raw_opening: dict = {}
        perception_opening._detect_interface_flags(
            _make_window_ctx(), None, State.SELECT_OPENING_SKILLS, raw_opening, []
        )
        assert "shua_xin.png" not in vision_opening.template_calls

    def test_observe_passes_grab_mode_to_flag_detection(self, tmp_path):
        """observe 把 observation_mode 传给标志识别：抢合作模式下准备按钮可命中。"""
        from wlxq_bot.models import State

        vision = _RecordingVision({"zhun_bei.png": MatchResult("zhun_bei.png", (500, 1000), 0.9)})
        perception = _flag_gated_perception(tmp_path, vision)

        observation = perception.observe(
            _make_window_ctx(), None, hint_state=State.FIND_COOP, observation_mode="coop_grab"
        )

        assert sorted(vision.template_calls) == ["x.png", "zhun_bei.png"]
        assert observation.flag("ready_button_visible") is True

    def test_dialog_flag_false_when_marker_missed(self, tmp_path):
        from wlxq_bot.assets import TemplatePack

        pack_root = tmp_path / "3000x2000"
        difficulty_dir = pack_root / "buttons" / "coop_difficulty"
        difficulty_dir.mkdir(parents=True)
        (difficulty_dir / "he_zuo_mo_shi.png").write_bytes(b"template")
        vision = _SingleTemplateVision("he_zuo_mo_shi.png", None)
        tasks_cfg = TasksConfig(
            locators={
                "he_zuo_mo_shi": {
                    "strategy": "template",
                    "template": "buttons/coop_difficulty/he_zuo_mo_shi.png",
                    "threshold": 0.82,
                }
            }
        )
        perception = CoopPerception(
            vision,
            TemplatePack(client_size=(3000, 2000), root=pack_root),
            tasks_cfg,
            CoopRole.HELPER,
            "assault",
        )

        difficulty = perception.observe(
            _make_window_ctx(), None, observation_mode="coop_difficulty"
        )

        assert difficulty.flag("difficulty_dialog_visible") is False


class TestOpeningHomeReturnObservation:
    def test_detects_home_marker_in_opening_state(self, tmp_path):
        """等待开局期间检测首页标志：用于识别本局被取消/被踢回首页。"""
        from wlxq_bot.assets import TemplatePack
        from wlxq_bot.models import State

        pack_root = tmp_path / "3000x2000"
        buttons_dir = pack_root / "buttons"
        buttons_dir.mkdir(parents=True)
        (buttons_dir / "home_page_marker.png").write_bytes(b"template")
        match = MatchResult("home_page_marker.png", (100, 200), 0.93)
        vision = _SingleTemplateVision("home_page_marker.png", match)
        tasks_cfg = TasksConfig(
            locators={
                "home_page_marker": {
                    "strategy": "template",
                    "template": "buttons/home_page_marker.png",
                    "threshold": 0.82,
                }
            }
        )
        perception = CoopPerception(
            vision,
            TemplatePack(client_size=(3000, 2000), root=pack_root),
            tasks_cfg,
            CoopRole.HELPER,
            "assault",
        )

        opening = perception.observe(
            _make_window_ctx(), None, hint_state=State.SELECT_OPENING_SKILLS
        )
        other = perception.observe(_make_window_ctx(), None, hint_state=State.SELECT_MAIN_C_SKILLS)

        assert opening.flag("home_page_visible")
        assert opening.raw_data["home_page_match"] == match
        # 局内等其他状态不做首页匹配，避免高频循环额外消耗模板识别
        assert "home_page_visible" not in other.raw_data

    def test_home_marker_miss_reports_not_visible(self, tmp_path):
        from wlxq_bot.assets import TemplatePack
        from wlxq_bot.models import State

        pack_root = tmp_path / "3000x2000"
        buttons_dir = pack_root / "buttons"
        buttons_dir.mkdir(parents=True)
        (buttons_dir / "home_page_marker.png").write_bytes(b"template")
        vision = _SingleTemplateVision("home_page_marker.png", None)
        tasks_cfg = TasksConfig(
            locators={
                "home_page_marker": {
                    "strategy": "template",
                    "template": "buttons/home_page_marker.png",
                    "threshold": 0.82,
                }
            }
        )
        perception = CoopPerception(
            vision,
            TemplatePack(client_size=(3000, 2000), root=pack_root),
            tasks_cfg,
            CoopRole.HELPER,
            "assault",
        )

        opening = perception.observe(
            _make_window_ctx(), None, hint_state=State.SELECT_OPENING_SKILLS
        )

        assert opening.flag("home_page_visible") is False

    def test_detects_leave_team_marker_in_opening_state(self, tmp_path):
        """组队大厅的【退队】按钮只在开局技能阶段识别，供房主不开始时退出。"""
        from wlxq_bot.assets import TemplatePack
        from wlxq_bot.models import State

        pack_root = tmp_path / "3000x2000"
        buttons_dir = pack_root / "buttons"
        buttons_dir.mkdir(parents=True)
        (buttons_dir / "he_zuo_tui_dui.png").write_bytes(b"template")
        match = MatchResult("he_zuo_tui_dui.png", (300, 1500), 0.91)
        vision = _SingleTemplateVision("he_zuo_tui_dui.png", match)
        tasks_cfg = TasksConfig(
            locators={
                "coop_leave_team": {
                    "strategy": "template",
                    "template": "buttons/he_zuo_tui_dui.png",
                    "threshold": 0.82,
                }
            }
        )
        perception = CoopPerception(
            vision,
            TemplatePack(client_size=(3000, 2000), root=pack_root),
            tasks_cfg,
            CoopRole.HELPER,
            "assault",
        )

        opening = perception.observe(
            _make_window_ctx(), None, hint_state=State.SELECT_OPENING_SKILLS
        )
        other = perception.observe(_make_window_ctx(), None, hint_state=State.SELECT_MAIN_C_SKILLS)

        assert opening.flag("leave_team_visible")
        assert opening.raw_data["leave_team_match"] == match
        assert "leave_team_visible" not in other.raw_data


class _SkillIconVision:
    """技能图标匹配：先匹配主C模板集，再按序匹配队友模板集。"""

    def __init__(
        self,
        main_matches: list[MatchResult],
        teammate_matches: list[MatchResult],
    ) -> None:
        self._results = [main_matches, teammate_matches]
        self._call = 0
        self.template_set_calls: list[tuple[str, ...]] = []

    def match_template_set(self, frame, template_paths, roi=None, threshold=0.78, nms_dist=None):
        self.template_set_calls.append(tuple(template_paths))
        idx = min(self._call, len(self._results) - 1)
        result = self._results[idx]
        self._call += 1
        return result

    def match_template(self, frame, template_path, roi=None, threshold=0.85):
        return None


class TestTeammateSkillIconObservation:
    def _make_perception(self, tmp_path, vision) -> CoopPerception:
        from wlxq_bot.assets import TemplatePack

        pack_root = tmp_path / "3000x2000"
        skills_dir = pack_root / "skills"
        skills_dir.mkdir(parents=True)
        for name in ("qiang_xi.png", "tian_shi.png", "xue_ji.png", "si_qi.png"):
            (skills_dir / name).write_bytes(b"template")
        return CoopPerception(
            vision,
            TemplatePack(client_size=(3000, 2000), root=pack_root),
            TasksConfig(),
            CoopRole.HELPER,
            "assault",
            skill_icon_templates=["skills/qiang_xi.png"],
            teammate_skill_icon_templates=[
                "skills/tian_shi.png",
                "skills/xue_ji.png",
                "skills/si_qi.png",
            ],
        )

    def test_teammate_icon_flag_when_main_c_missed(self, tmp_path):
        """主C图标未命中但识别到队友图标：写入标志供任务层立即随机选卡。"""
        from wlxq_bot.models import State

        vision = _SkillIconVision(
            main_matches=[],
            teammate_matches=[MatchResult("tian_shi.png", (200, 800), 0.9)],
        )
        perception = self._make_perception(tmp_path, vision)

        opening = perception.observe(
            _make_window_ctx(), None, hint_state=State.SELECT_OPENING_SKILLS
        )

        assert opening.skill_candidates == []
        assert opening.flag("teammate_skill_visible")
        assert opening.raw_data["teammate_skill_match"].template_name == "tian_shi.png"

    def test_no_teammate_flag_when_main_c_matched(self, tmp_path):
        """主C图标命中：正常返回主C候选，不再匹配队友图标。"""
        from wlxq_bot.models import State

        vision = _SkillIconVision(
            main_matches=[MatchResult("qiang_xi.png", (300, 800), 0.9)],
            teammate_matches=[MatchResult("tian_shi.png", (200, 800), 0.9)],
        )
        perception = self._make_perception(tmp_path, vision)

        opening = perception.observe(
            _make_window_ctx(), None, hint_state=State.SELECT_OPENING_SKILLS
        )

        assert len(opening.skill_candidates) == 1
        assert opening.skill_candidates[0].skill_id == "assault"
        assert "teammate_skill_visible" not in opening.raw_data
        # 主C命中后不应再调用队友模板集
        assert len(vision.template_set_calls) == 1

    def test_both_missed_reports_nothing(self, tmp_path):
        from wlxq_bot.models import State

        vision = _SkillIconVision(main_matches=[], teammate_matches=[])
        perception = self._make_perception(tmp_path, vision)

        opening = perception.observe(
            _make_window_ctx(), None, hint_state=State.SELECT_OPENING_SKILLS
        )

        assert opening.skill_candidates == []
        assert not opening.flag("teammate_skill_visible")


class _GiftSkillPageVision:
    """合成4星赠送技能页的假 Vision：页面标志可选命中，技能卡图标固定命中。"""

    def __init__(self, page_visible: bool, marker: str = "xuan_ze_ji_neng.png") -> None:
        self._page_visible = page_visible
        self._marker = marker
        self.set_calls = 0

    def match_template(self, frame, template_path, roi=None, threshold=0.85):
        if self._page_visible and template_path.endswith(self._marker):
            return MatchResult(self._marker, (400, 900), 0.9)
        return None

    def match_template_set(self, frame, template_paths, roi=None, threshold=0.78, nms_dist=None):
        self.set_calls += 1
        return [MatchResult("qiang_xi.png", (300, 800), 0.9)]


class TestMergeGiftSkillPageObservation:
    def _make_perception(self, tmp_path, vision) -> CoopPerception:
        from wlxq_bot.assets import TemplatePack

        pack_root = tmp_path / "3000x2000"
        buttons_dir = pack_root / "buttons"
        skills_dir = pack_root / "skills"
        buttons_dir.mkdir(parents=True)
        skills_dir.mkdir(parents=True)
        (buttons_dir / "xuan_ze_ji_neng.png").write_bytes(b"template")
        (skills_dir / "4_xing_e_wai_ji_neng.png").write_bytes(b"template")
        (skills_dir / "qiang_xi.png").write_bytes(b"template")
        return CoopPerception(
            vision,
            TemplatePack(client_size=(3000, 2000), root=pack_root),
            TasksConfig(
                board={"helper": _helper_board_params()},
                locators={
                    "select_skill_button": {
                        "strategy": "template",
                        "template": "buttons/xuan_ze_ji_neng.png",
                        "threshold": 0.82,
                    },
                    "merge_gift_skill_title": {
                        "strategy": "template",
                        "template": "skills/4_xing_e_wai_ji_neng.png",
                        "threshold": 0.82,
                    },
                },
            ),
            CoopRole.HELPER,
            "assault",
            skill_icon_templates=["skills/qiang_xi.png"],
            hero_cell_classifier=FakeCellClassifier([_board_predictions(_prediction("empty"))]),
        )

    def test_detects_skill_candidates_only_when_page_visible(self, tmp_path):
        """培养状态识别到技能页标志时才做技能卡识别（合成4星赠送技能页）。"""
        from wlxq_bot.models import State

        frame = np.zeros((1723, 923, 3), dtype=np.uint8)
        vision = _GiftSkillPageVision(page_visible=True)
        perception = self._make_perception(tmp_path, vision)
        observed = perception.observe(_make_window_ctx(), frame, hint_state=State.BUILD_MAIN_C)

        assert observed.flag("select_skill_button_visible")
        assert len(observed.skill_candidates) == 1
        assert observed.skill_candidates[0].skill_id == "assault"

        # 页面未打开：不做技能卡识别（不增加常规培养帧开销）
        vision._page_visible = False
        closed = perception.observe(_make_window_ctx(), frame, hint_state=State.BUILD_MAIN_C)

        assert closed.flag("select_skill_button_visible") is False
        assert closed.skill_candidates == []
        assert vision.set_calls == 1  # 只在页面打开时做过一次技能卡识别

    def test_detects_skill_candidates_via_merge_gift_title_marker(self, tmp_path):
        """仅【请选择1个额外技能】提示条命中（【选技能】图未命中）也识别技能卡。

        2026-08-21 实机：赠送页等待选择期间【选技能】图可能始终不命中，
        提示条是页面主标识，命中即应触发技能卡识别。
        """
        from wlxq_bot.models import State

        frame = np.zeros((1723, 923, 3), dtype=np.uint8)
        vision = _GiftSkillPageVision(page_visible=True, marker="4_xing_e_wai_ji_neng.png")
        perception = self._make_perception(tmp_path, vision)

        observed = perception.observe(_make_window_ctx(), frame, hint_state=State.BUILD_MAIN_C)

        assert observed.flag("select_skill_button_visible") is False
        assert observed.flag("merge_gift_skill_page_visible")
        assert len(observed.skill_candidates) == 1
        assert observed.skill_candidates[0].skill_id == "assault"


class TestDoubleRewardDialogObservation:
    def test_detects_double_reward_dialog_in_settlement_states(self, tmp_path):
        """【双倍奖励】确认弹窗只在结算阶段识别（挡住结算操作，需点取消）。"""
        from wlxq_bot.assets import TemplatePack
        from wlxq_bot.models import State

        pack_root = tmp_path / "3000x2000"
        buttons_dir = pack_root / "buttons"
        buttons_dir.mkdir(parents=True)
        (buttons_dir / "shuang_bei.png").write_bytes(b"template")
        (buttons_dir / "qu_xiao_shuang_bei.png").write_bytes(b"template")
        dialog_match = MatchResult("shuang_bei.png", (600, 500), 0.9)
        cancel_match = MatchResult("qu_xiao_shuang_bei.png", (300, 1100), 0.9)

        class _DoubleRewardVision:
            def match_template(self, frame, template_path, roi=None, threshold=0.85):
                name = template_path.replace("\\", "/").rsplit("/", 1)[-1]
                if name == "qu_xiao_shuang_bei.png":
                    return cancel_match
                if name == "shuang_bei.png":
                    return dialog_match
                return None

            def match_template_set(
                self, frame, template_paths, roi=None, threshold=0.78, nms_dist=None
            ):
                return []

        tasks_cfg = TasksConfig(
            locators={
                "double_reward_dialog": {
                    "strategy": "template",
                    "template": "buttons/shuang_bei.png",
                    "threshold": 0.82,
                },
                "double_reward_cancel": {
                    "strategy": "template",
                    "template": "buttons/qu_xiao_shuang_bei.png",
                    "threshold": 0.82,
                },
            }
        )
        perception = CoopPerception(
            _DoubleRewardVision(),
            TemplatePack(client_size=(3000, 2000), root=pack_root),
            tasks_cfg,
            CoopRole.HELPER,
            "assault",
        )

        settlement = perception.observe(_make_window_ctx(), None, hint_state=State.HANDLE_RESULT)
        in_match = perception.observe(
            _make_window_ctx(), None, hint_state=State.SELECT_MAIN_C_SKILLS
        )

        assert settlement.flag("double_reward_dialog_visible")
        assert settlement.raw_data["double_reward_cancel_match"] == cancel_match
        # 对局内状态不识别（不影响高频循环）
        assert "double_reward_dialog_visible" not in in_match.raw_data


class FakeRowReader:
    """pet_difficulty reader 协议的 fake。"""

    def __init__(self, lines: list[tuple[str, float, tuple[int, int]]]) -> None:
        self.lines = lines
        self.images: list = []

    def read_with_confidence(self, image):
        self.images.append(image)
        return self.lines


class TestPetDifficultyOcr:
    """OCR 难度识别模式（保卫宠物）。"""

    def _tasks_cfg(self) -> TasksConfig:
        return TasksConfig(
            rois={
                "pet_difficulty_list": {
                    "relative_to": "client",
                    "x_ratio": 0.0,
                    "y_ratio": 0.0,
                    "width_ratio": 0.5,
                    "height_ratio": 0.5,
                }
            },
            pet_difficulty_recognition={"candidate_roi": "pet_difficulty_list",
                                        "min_confidence": 0.5},
        )

    def _perception(self, reader: FakeRowReader, tmp_path):
        from wlxq_bot.assets import TemplatePack

        pack_root = tmp_path / "3000x2000"
        pack_root.mkdir(exist_ok=True)
        return CoopPerception(
            _RecordingVision(),
            TemplatePack(client_size=(923, 1723), root=pack_root),
            self._tasks_cfg(),
            CoopRole.HELPER,
            "assault",
            difficulty_kind="ocr",
            difficulty_reader=reader,
        )

    def _frame(self) -> np.ndarray:
        return np.zeros((1723, 923, 3), dtype=np.uint8)

    def test_ocr_mode_reads_rows_into_raw_data(self, tmp_path) -> None:
        reader = FakeRowReader([
            ("保卫宠物-普通", 0.85, (120, 50)),
            ("保卫宠物-中等", 0.60, (120, 140)),
        ])
        perception = self._perception(reader, tmp_path)
        observation = perception.observe(
            _make_window_ctx(), self._frame(), observation_mode="coop_difficulty"
        )
        rows = observation.raw_data["pet_difficulty_rows"]
        assert [row.text for row in rows] == ["保卫宠物-普通", "保卫宠物-中等"]
        assert rows[0].confidence == 0.85
        assert observation.raw_data["difficulty_dialog_visible"] is True
        assert observation.difficulty_candidates == []

    def test_ocr_mode_low_confidence_rows_filtered(self, tmp_path) -> None:
        reader = FakeRowReader([("保卫宠物-中等", 0.30, (120, 140))])
        perception = self._perception(reader, tmp_path)
        observation = perception.observe(
            _make_window_ctx(), self._frame(), observation_mode="coop_difficulty"
        )
        assert observation.raw_data["pet_difficulty_rows"] == []
        # 低置信度行被过滤后，收起栏锚点也读不到「选择难度」→ 视为弹窗打开
        # （保守：绝不在锚点未确认收起时推进关闭）
        assert observation.raw_data["difficulty_dialog_visible"] is True

    def test_light_mode_skips_skill_candidates_and_title_ocr(self, tmp_path) -> None:
        """light 验证帧只算界面标志：跳过技能卡 OCR（含标题识别）与采集。"""
        reader = FakeRowReader([("保卫宠物-普通", 0.85, (120, 50))])
        perception = self._perception(reader, tmp_path)
        observation = perception.observe(
            _make_window_ctx(), self._frame(),
            hint_state=State.SELECT_MAIN_C_SKILLS, light=True,
        )
        assert observation.skill_candidates == []
        assert reader.images == []  # 技能标题 OCR 未执行
        # 界面标志（含弹窗标识）仍然计算——这是验证帧的本职
        assert "tan_chuang_visible" in observation.raw_data

    def test_ocr_dialog_mode_reports_visibility_without_candidates(self, tmp_path) -> None:
        reader = FakeRowReader([("保卫宠物-普通", 0.85, (120, 50))])
        perception = self._perception(reader, tmp_path)
        observation = perception.observe(
            _make_window_ctx(), self._frame(), observation_mode="difficulty_dialog"
        )
        assert observation.raw_data["difficulty_dialog_visible"] is True
        # 关闭轮询只跑收起栏锚点，不读难度列表（控制单帧耗时）
        assert "pet_difficulty_rows" not in observation.raw_data


class TestLocatorAliases:
    """locator 别名：canonical 名重映射到模式专属 locator，标志名不变。"""

    def test_ready_button_alias_in_grab_mode(self, tmp_path) -> None:
        from wlxq_bot.assets import TemplatePack

        pack_root = tmp_path / "3000x2000"
        (pack_root / "buttons" / "bao_wei_chong_wu").mkdir(parents=True)
        (pack_root / "buttons" / "bao_wei_chong_wu" / "zhun_bei.png").write_bytes(b"t")
        match = MatchResult("zhun_bei.png", (400, 1300), 0.9)
        vision = _SingleTemplateVision("zhun_bei.png", match)
        tasks_cfg = TasksConfig(
            locators={
                "pet_ready_button": {
                    "strategy": "template",
                    "template": "buttons/bao_wei_chong_wu/zhun_bei.png",
                    "threshold": 0.82,
                }
            }
        )
        perception = CoopPerception(
            vision,
            TemplatePack(client_size=(923, 1723), root=pack_root),
            tasks_cfg,
            CoopRole.HELPER,
            "assault",
            locator_aliases={"ready_button": "pet_ready_button"},
        )
        observation = perception.observe(
            _make_window_ctx(), None, hint_state=State.FIND_COOP, observation_mode="coop_grab"
        )
        assert observation.flag("ready_button_visible") is True
        assert observation.raw_data["ready_button_match"] == match

    def test_leave_team_alias_in_opening_state(self, tmp_path) -> None:
        from wlxq_bot.assets import TemplatePack

        pack_root = tmp_path / "3000x2000"
        (pack_root / "buttons" / "bao_wei_chong_wu").mkdir(parents=True)
        (pack_root / "buttons" / "bao_wei_chong_wu" / "tui_dui.png").write_bytes(b"t")
        match = MatchResult("tui_dui.png", (741, 1488), 0.9)
        vision = _SingleTemplateVision("tui_dui.png", match)
        tasks_cfg = TasksConfig(
            locators={
                "pet_leave_team": {
                    "strategy": "template",
                    "template": "buttons/bao_wei_chong_wu/tui_dui.png",
                    "threshold": 0.82,
                }
            }
        )
        perception = CoopPerception(
            vision,
            TemplatePack(client_size=(923, 1723), root=pack_root),
            tasks_cfg,
            CoopRole.HELPER,
            "assault",
            locator_aliases={"coop_leave_team": "pet_leave_team"},
        )
        observation = perception.observe(
            _make_window_ctx(), None, hint_state=State.SELECT_OPENING_SKILLS
        )
        assert observation.flag("leave_team_visible") is True
        assert observation.raw_data["leave_team_match"] == match

    def test_match_ready_button_uses_alias(self, tmp_path) -> None:
        """抢合作检查线程的 match_ready_button 同样走别名重映射：

        保卫宠物模式下 canonical ready_button locator 缺失、只有
        pet_ready_button，方法应命中宠物模板并返回其 MatchResult。
        """
        from wlxq_bot.assets import TemplatePack

        pack_root = tmp_path / "3000x2000"
        (pack_root / "buttons" / "bao_wei_chong_wu").mkdir(parents=True)
        (pack_root / "buttons" / "bao_wei_chong_wu" / "zhun_bei.png").write_bytes(b"t")
        match = MatchResult("zhun_bei.png", (400, 1300), 0.9)
        vision = _SingleTemplateVision("zhun_bei.png", match)
        tasks_cfg = TasksConfig(
            locators={
                "pet_ready_button": {
                    "strategy": "template",
                    "template": "buttons/bao_wei_chong_wu/zhun_bei.png",
                    "threshold": 0.82,
                }
            }
        )
        perception = CoopPerception(
            vision,
            TemplatePack(client_size=(923, 1723), root=pack_root),
            tasks_cfg,
            CoopRole.HELPER,
            "assault",
            locator_aliases={"ready_button": "pet_ready_button"},
        )

        assert perception.match_ready_button(_make_window_ctx(), None) == match
        # 确实匹配的是宠物专属模板（template_calls 记录解析后的绝对路径）
        assert len(vision.template_calls) == 1
        assert vision.template_calls[0].endswith("zhun_bei.png")


class TestPetDifficultyBar:
    """保卫宠物难度弹窗开关的「选择难度」收起栏 OCR 锚点测试（2026-09-06 实机 bug 回归）。

    实机 bug：弹窗关闭后邀请卡标题（保卫宠物-中等）落在难度行同一 ROI，
    行文字存在性把"已关闭"误判为"仍打开"。修复后开关判定改用收起栏文字
    OCR：栏文字可见=已关闭（flag False），栏被展开面板盖住=打开（flag True）。
    """

    def _tasks_cfg(self) -> TasksConfig:
        return TasksConfig(
            rois={
                "pet_difficulty_bar_area": {
                    "relative_to": "client",
                    "x_ratio": 0.0,
                    "y_ratio": 0.0,
                    "width_ratio": 0.5,
                    "height_ratio": 0.125,
                },
                "pet_difficulty_list": {
                    "relative_to": "client",
                    "x_ratio": 0.0,
                    "y_ratio": 0.0,
                    "width_ratio": 0.5,
                    "height_ratio": 0.5,
                },
            },
            pet_difficulty_recognition={
                "candidate_roi": "pet_difficulty_list",
                "min_confidence": 0.5,
                "bar_roi": "pet_difficulty_bar_area",
                "bar_text": "选择难度",
            },
        )

    def _perception(self, tmp_path, reader: FakeBarRowReader):
        from wlxq_bot.assets import TemplatePack

        pack_root = tmp_path / "3000x2000"
        pack_root.mkdir(parents=True, exist_ok=True)
        return CoopPerception(
            _RecordingVision(),
            TemplatePack(client_size=(923, 1723), root=pack_root),
            self._tasks_cfg(),
            CoopRole.HELPER,
            "assault",
            difficulty_kind="ocr",
            difficulty_reader=reader,
        )

    def _frame(self) -> np.ndarray:
        return np.zeros((1723, 923, 3), dtype=np.uint8)


class FakeBarRowReader:
    """按调用顺序返回预设 OCR 结果的 fake：第 1 次调用=难度列表 ROI，第 2 次=收起栏 ROI。

    每项为 (text, confidence, (cx, cy)) 三元组列表；列表耗尽后重复最后一项。
    """

    def __init__(self, calls: list[list[tuple[str, float, tuple[int, int]]]]) -> None:
        self.calls = calls
        self.images: list[np.ndarray] = []

    def read_with_confidence(self, image: np.ndarray):
        self.images.append(image)
        idx = min(len(self.images) - 1, len(self.calls) - 1)
        return self.calls[idx]


class TestPetDifficultyBarAnchor(TestPetDifficultyBar):
    """收起栏 OCR 锚点。"""

    def test_bar_visible_means_dialog_closed_even_with_invite_rows(self, tmp_path) -> None:
        """核心回归：收起栏文字可读 → 已关闭（行文字存在与否不影响开关判定）。"""
        reader = FakeBarRowReader([
            [("选择难度", 0.80, (80, 60))],  # 收起栏 ROI：栏文字可见
        ])
        perception = self._perception(tmp_path, reader)
        observation = perception.observe(
            _make_window_ctx(), np.zeros((1723, 923, 3), dtype=np.uint8),
            observation_mode="difficulty_dialog",
        )
        assert observation.flag("difficulty_dialog_visible") is False
        assert observation.raw_data["pet_difficulty_bar_visible"] is True

    def test_bar_missing_with_invite_rows_means_open(self, tmp_path) -> None:
        """收起栏被盖住 + 行文字存在 → 弹窗打开（关闭轮询的常态判定）。"""
        reader = FakeBarRowReader([
            [("保卫宠物-中等", 0.85, (100, 100))],
            [],  # 栏被面板盖住，OCR 全空
        ])
        perception = self._perception(tmp_path, reader)
        observation = perception.observe(
            _make_window_ctx(), np.zeros((1723, 923, 3), dtype=np.uint8),
            observation_mode="difficulty_dialog",
        )
        assert observation.flag("difficulty_dialog_visible") is True

    def test_close_poll_ocr_targets_bar_roi(self, tmp_path) -> None:
        """关闭轮询只 OCR 收起栏小区域（不跑整个难度列表）。"""
        reader = FakeBarRowReader([[("选择难度", 0.80, (80, 60))]])
        perception = self._perception(tmp_path, reader)
        perception.observe(
            _make_window_ctx(), np.zeros((1723, 923, 3), dtype=np.uint8),
            observation_mode="difficulty_dialog",
        )
        assert len(reader.images) == 1
        # 收起栏 ROI 高 0.125*1723 ≈ 215，远小于难度列表（0.5*1723 ≈ 861）
        assert reader.images[0].shape[0] == int(0.125 * 1723)

    def test_selection_mode_reads_rows_then_bar(self, tmp_path) -> None:
        """勾选模式：第 1 次 OCR 读难度列表（供点击），第 2 次读收起栏（供开关）。"""
        reader = FakeBarRowReader([
            [("保卫宠物-普通", 0.85, (100, 100))],
            [("选择难度", 0.80, (80, 60))],
        ])
        perception = self._perception(tmp_path, reader)
        observation = perception.observe(
            _make_window_ctx(), np.zeros((1723, 923, 3), dtype=np.uint8),
            observation_mode="coop_difficulty",
        )
        rows = observation.raw_data["pet_difficulty_rows"]
        assert [row.text for row in rows] == ["保卫宠物-普通"]
        assert observation.flag("difficulty_dialog_visible") is False
        assert len(reader.images) == 2
        assert reader.images[0].shape[0] == int(0.5 * 1723)
        assert reader.images[1].shape[0] == int(0.125 * 1723)


# ---------------------------------------------------------------------------
# 领地入口观察分支（territory_nav / mode_select / challenge_page）
# ---------------------------------------------------------------------------


class _FakeRowsReader:
    """假 OCR reader：返回固定 (文字, 置信度, 中心) 行，坐标相对输入图。"""

    def __init__(self, rows):
        self.rows = rows

    def read_with_confidence(self, image):
        return self.rows


def _frame() -> np.ndarray:
    """裁剪用零矩阵帧（923×1723 客户区尺寸）。"""
    return np.zeros((1723, 923, 3), dtype=np.uint8)


def _build_entry_perception(tmp_path, reader, weekly_buff_roi: str | None = None) -> CoopPerception:
    """构造带领地入口 ROI 的 CoopPerception（模板包指向临时目录）。"""
    from wlxq_bot.assets import TemplatePack

    pack_root = tmp_path / "3000x2000"
    pack_root.mkdir(parents=True, exist_ok=True)
    pack = TemplatePack(client_size=(3000, 2000), root=pack_root)
    rois = {
        "territory_nav_area": {
            "relative_to": "client", "x_ratio": 0.64,
            "y_ratio": 0.96, "width_ratio": 0.11, "height_ratio": 0.04,
        },
        "mode_select_title_area": {
            "relative_to": "client", "x_ratio": 0.25,
            "y_ratio": 0.16, "width_ratio": 0.49, "height_ratio": 0.06,
        },
        "pet_mode_label_area": {
            "relative_to": "client", "x_ratio": 0.60,
            "y_ratio": 0.65, "width_ratio": 0.16, "height_ratio": 0.06,
        },
        "pet_challenge_start_area": {
            "relative_to": "client", "x_ratio": 0.43,
            "y_ratio": 0.90, "width_ratio": 0.17, "height_ratio": 0.05,
        },
    }
    if weekly_buff_roi:
        rois[weekly_buff_roi] = {
            "relative_to": "client", "x_ratio": 0.39,
            "y_ratio": 0.73, "width_ratio": 0.22, "height_ratio": 0.05,
        }
    tasks_cfg = TasksConfig(
        rois=rois,
        mode_select_recognition={
            "title_roi": "mode_select_title_area",
            "title_text": "模式选择",
            "min_confidence": 0.5,
        },
    )
    return CoopPerception(
        FakeVision([]),
        pack,
        tasks_cfg,
        CoopRole.HELPER,
        "assault",
        difficulty_reader=reader,
        territory_nav_roi="territory_nav_area",
        mode_select_title_roi="mode_select_title_area",
        mode_card_roi="pet_mode_label_area",
        challenge_roi="pet_challenge_start_area",
        weekly_buff_roi=weekly_buff_roi,
        mode_select_min_confidence=0.5,
    )


class TestTerritoryEntryDetection:
    """领地入口三个观察分支：限域 OCR、置信度过滤、子串匹配写 flag。"""

    def test_territory_nav_text_written(self, tmp_path):
        perception = _build_entry_perception(
            tmp_path, _FakeRowsReader([("领地", 0.666, (10, 20))])
        )
        raw: dict = {}
        perception._detect_territory_nav_text(_make_window_ctx(), _frame(), raw)
        assert raw["territory_nav_text"] == "领地"

    def test_territory_nav_empty_when_no_text(self, tmp_path):
        perception = _build_entry_perception(tmp_path, _FakeRowsReader([]))
        raw: dict = {}
        perception._detect_territory_nav_text(_make_window_ctx(), _frame(), raw)
        assert raw["territory_nav_text"] == ""

    def test_territory_nav_probe_reads_nav_and_challenge(self, tmp_path):
        """领地入口探测观察分支（2026-09-25）：导航文字与挑战按钮区同帧
        读，任务层据锚点文字判定已在挑战页、跳过领地导航。"""
        perception = _build_entry_perception(
            tmp_path, _FakeRowsReader([("开始挑战", 0.798, (30, 40))])
        )
        observation = perception.observe(
            _make_window_ctx(),
            _frame(),
            observation_mode="territory_nav_probe",
        )
        # 假 reader 对任何识别区返回同一行：两个区域都各自写出文字
        assert observation.raw_data["territory_nav_text"] == "开始挑战"
        assert observation.raw_data["challenge_text"] == "开始挑战"

    def test_mode_select_title_uses_substring_match(self, tmp_path):
        """标题装饰星星读作「模式选择·」也应命中：必须子串匹配而非全等。"""
        perception = _build_entry_perception(
            tmp_path, _FakeRowsReader([("模式选择·", 0.747, (100, 40))])
        )
        raw: dict = {}
        perception._detect_mode_select_title(_make_window_ctx(), _frame(), raw)
        assert raw["mode_select_visible"] is True

    def test_mode_select_title_false_on_other_text(self, tmp_path):
        perception = _build_entry_perception(
            tmp_path, _FakeRowsReader([("其他文字", 0.9, (100, 40))])
        )
        raw: dict = {}
        perception._detect_mode_select_title(_make_window_ctx(), _frame(), raw)
        assert raw["mode_select_visible"] is False

    def test_mode_card_rows_filtered_by_confidence(self, tmp_path):
        perception = _build_entry_perception(
            tmp_path,
            _FakeRowsReader([("保卫宠物", 0.666, (30, 40)), ("噪声", 0.2, (50, 60))]),
        )
        raw: dict = {}
        perception._detect_mode_card_rows(_make_window_ctx(), _frame(), raw)
        rows = raw["mode_card_rows"]
        assert [row.text for row in rows] == ["保卫宠物"]
        # 文字框中心 = ROI 左上角偏移（0.60*923=553, 0.65*1723=1119）+ 行内坐标
        assert rows[0].position == (553 + 30, 1119 + 40)

    def test_challenge_text_written(self, tmp_path):
        perception = _build_entry_perception(
            tmp_path, _FakeRowsReader([("开始挑战", 0.798, (60, 30))])
        )
        raw: dict = {}
        perception._detect_challenge_text(_make_window_ctx(), _frame(), raw)
        assert raw["challenge_text"] == "开始挑战"

    def test_weekly_buff_rows_written_with_roi_offset(self, tmp_path):
        """每周弹窗按钮行：置信度过滤 + 文字框中心带 ROI 偏移（同模式卡行）。"""
        perception = _build_entry_perception(
            tmp_path,
            _FakeRowsReader([("知道", 0.665, (40, 25)), ("噪声", 0.2, (50, 60))]),
            weekly_buff_roi="pet_weekly_buff_area",
        )
        raw: dict = {}
        perception._detect_weekly_buff_rows(_make_window_ctx(), _frame(), raw)
        rows = raw["weekly_buff_rows"]
        assert [row.text for row in rows] == ["知道"]
        # 文字框中心 = ROI 左上角偏移（0.39*923=359, 0.73*1723=1257）+ 行内坐标
        assert rows[0].position == (359 + 40, 1257 + 25)

    def test_challenge_page_mode_reads_weekly_buff_when_configured(self, tmp_path):
        """observe 分发：challenge_page 模式在配置了识别区时同帧读弹窗按钮行。"""
        class _SequenceRowsReader:
            """按调用次序返回预设行：第 1 次读挑战按钮，第 2 次读弹窗按钮。"""

            def __init__(self, responses):
                self.responses = list(responses)

            def read_with_confidence(self, image):
                return self.responses.pop(0) if self.responses else []

        perception = _build_entry_perception(
            tmp_path,
            _SequenceRowsReader([
                [("挑战", 0.665, (60, 30))],
                [("知道", 0.665, (40, 25))],
            ]),
            weekly_buff_roi="pet_weekly_buff_area",
        )
        obs = perception.observe(
            _make_window_ctx(), _frame(), observation_mode="challenge_page"
        )
        assert obs.raw_data["challenge_text"] == "挑战"
        assert [row.text for row in obs.raw_data["weekly_buff_rows"]] == ["知道"]

    def test_challenge_page_mode_skips_weekly_buff_when_not_configured(self, tmp_path):
        """未配置识别区（合作模式）：不读弹窗行，不写 weekly_buff_rows。"""
        perception = _build_entry_perception(
            tmp_path, _FakeRowsReader([("开始挑战", 0.798, (60, 30))])
        )
        obs = perception.observe(
            _make_window_ctx(), _frame(), observation_mode="challenge_page"
        )
        assert obs.raw_data["challenge_text"] == "开始挑战"
        assert "weekly_buff_rows" not in obs.raw_data

    def test_uncalibrated_roi_degrades_to_empty(self, tmp_path):
        """ROI 未标定（0x0）：OCR 必须限域，退化为读空而不是全图识别。"""
        from wlxq_bot.assets import TemplatePack

        pack_root = tmp_path / "3000x2000"
        pack_root.mkdir(parents=True, exist_ok=True)
        perception = CoopPerception(
            FakeVision([]),
            TemplatePack(client_size=(3000, 2000), root=pack_root),
            TasksConfig(),
            CoopRole.HELPER,
            "assault",
            difficulty_reader=_FakeRowsReader([("领地", 0.9, (1, 1))]),
        )
        raw: dict = {}
        perception._detect_territory_nav_text(_make_window_ctx(), _frame(), raw)
        assert raw["territory_nav_text"] == ""


class TestTransformButtonObservation:
    """变身按钮只在 transform_button 观察模式识别（设计 §4.2/§4.3）。"""

    def _build(self, tmp_path, vision):
        from wlxq_bot.assets import TemplatePack

        pack_root = tmp_path / "3000x2000"
        buttons_dir = pack_root / "buttons"
        buttons_dir.mkdir(parents=True)
        (buttons_dir / "bian_shen.png").write_bytes(b"template")
        tasks_cfg = TasksConfig(
            locators={
                "transform_button": {
                    "strategy": "template",
                    "template": "buttons/bian_shen.png",
                    "threshold": 0.85,
                }
            }
        )
        return CoopPerception(
            vision,
            TemplatePack(client_size=(3000, 2000), root=pack_root),
            tasks_cfg,
            CoopRole.HELPER,
            "assault",
        )

    def test_detects_transform_button_only_in_requested_mode(self, tmp_path):
        match = MatchResult("bian_shen.png", (689, 986), 0.99)
        vision = _SingleTemplateVision("bian_shen.png", match)
        perception = self._build(tmp_path, vision)

        normal = perception.observe(_make_window_ctx(), None, observation_mode=None)
        transforming = perception.observe(
            _make_window_ctx(), None, observation_mode="transform_button"
        )

        assert "transform_button_visible" not in normal.raw_data
        assert transforming.flag("transform_button_visible") is True
        assert transforming.raw_data["transform_button_match"] == match
        assert len(vision.template_calls) == 1

    def test_missing_template_reports_invisible(self, tmp_path):
        """模板文件缺失时按不可见处理（runner 启动时已降级禁用，双保险）。"""
        vision = _SingleTemplateVision("bian_shen.png", None)
        perception = self._build(tmp_path, vision)

        observation = perception.observe(
            _make_window_ctx(), None, observation_mode="transform_button"
        )

        assert observation.flag("transform_button_visible") is False
        assert "transform_button_match" not in observation.raw_data


class TestModalFrameDetection:
    """is_modal_frame：全屏模态弹窗打开时画面两侧边缘被压暗（2026-09-14 实机）。"""

    def test_bright_edges_reported_as_not_modal(self, tmp_path):
        import numpy as np

        perception = _flag_gated_perception(tmp_path, _RecordingVision())
        frame = np.full((1727, 927, 3), 200, dtype=np.uint8)
        assert perception.is_modal_frame(frame, 45.0) is False

    def test_dark_edges_reported_as_modal(self, tmp_path):
        import numpy as np

        perception = _flag_gated_perception(tmp_path, _RecordingVision())
        frame = np.full((1727, 927, 3), 200, dtype=np.uint8)
        frame[:, :30] = 20
        frame[:, -30:] = 20
        assert perception.is_modal_frame(frame, 45.0) is True

    def test_invalid_frame_conservatively_not_modal(self, tmp_path):
        perception = _flag_gated_perception(tmp_path, _RecordingVision())
        assert perception.is_modal_frame(None, 45.0) is False
        assert perception.is_modal_frame("not-a-frame", 45.0) is False


class TestInitiatorInviteButtonFlags:
    """邀请者在开局等待态识别组队大厅双态按钮（挑战被拒回大厅的自愈依据）。

    帮助者流程无此按钮，不识别（避免对局内底栏误匹配白白消耗模板匹配）。
    """

    def test_initiator_opening_state_matches_invite_buttons(self, tmp_path):
        from wlxq_bot.models import State

        vision = _RecordingVision()
        perception = _flag_gated_perception(tmp_path, vision, role=CoopRole.INITIATOR)

        raw: dict = {}
        perception._detect_interface_flags(
            _make_window_ctx(), None, State.SELECT_OPENING_SKILLS, raw, []
        )

        assert "tiao_zhan_dui_you_zhun_bei.png" in vision.template_calls
        assert "tiao_zhan_dui_you_wei_zhun_bei.png" in vision.template_calls
        assert raw["invite_start_ready_visible"] is False
        assert raw["invite_start_not_ready_visible"] is False

    def test_helper_opening_state_skips_invite_buttons(self, tmp_path):
        from wlxq_bot.models import State

        vision = _RecordingVision()
        perception = _flag_gated_perception(tmp_path, vision, role=CoopRole.HELPER)

        raw: dict = {}
        perception._detect_interface_flags(
            _make_window_ctx(), None, State.SELECT_OPENING_SKILLS, raw, []
        )

        assert "tiao_zhan_dui_you_zhun_bei.png" not in vision.template_calls
        assert "tiao_zhan_dui_you_wei_zhun_bei.png" not in vision.template_calls
        assert "invite_start_ready_visible" not in raw
