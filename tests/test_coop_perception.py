"""CoopPerception 的 ONNX 棋盘分类与多帧投票测试。"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from wlxq_bot.config import BoardGridParams, TasksConfig
from wlxq_bot.models import (
    CoopRole,
    MatchResult,
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

    def capture(self, handle):
        self.capture_count += 1
        return self._ctx, np.zeros((1723, 923, 3), dtype=np.uint8)


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
}


def _flag_gated_perception(tmp_path, vision) -> CoopPerception:
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
        CoopRole.HELPER,
        "assault",
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

    def test_grab_mode_matches_only_ready_button(self, tmp_path):
        """抢合作子步骤：额外查准备按钮，其余标志仍不查。"""
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

        assert vision.template_calls == ["zhun_bei.png"]
        assert raw["ready_button_visible"] is True
        assert raw["ready_button_match"] == match

    def test_states_match_only_declared_flags(self, tmp_path):
        """各状态只匹配自己声明的标志集合。"""
        from wlxq_bot.models import State

        expectations = {
            State.ENTER_MATCH: {"zhun_bei.png"},
            State.SELECT_OPENING_SKILLS: {"xuan_ze_ji_neng.png", "zhao_huan.png", "tan_chuang.png"},
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

    def test_observe_passes_grab_mode_to_flag_detection(self, tmp_path):
        """observe 把 observation_mode 传给标志识别：抢合作模式下准备按钮可命中。"""
        from wlxq_bot.models import State

        vision = _RecordingVision({"zhun_bei.png": MatchResult("zhun_bei.png", (500, 1000), 0.9)})
        perception = _flag_gated_perception(tmp_path, vision)

        observation = perception.observe(
            _make_window_ctx(), None, hint_state=State.FIND_COOP, observation_mode="coop_grab"
        )

        assert vision.template_calls == ["zhun_bei.png"]
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
