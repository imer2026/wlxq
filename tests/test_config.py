"""配置加载与校验测试。

验证 Pydantic 模型对 YAML 配置的校验逻辑，
特别是逐次召唤门禁和动作后识别等待区间约束。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from wlxq_bot.config import (
    Hotspot,
    RoiConfig,
    RunConfig,
    WindowSpec,
    load_default_config,
    load_tasks_config,
    parse_coop_difficulties,
)


class TestRunConfig:
    """RunConfig 校验测试。"""

    def test_valid_config(self) -> None:
        cfg = RunConfig(
            max_rounds=10,
            minimum_summon_count_before_skills=5,
            initial_board_capacity=6,
            target_star_level=2,
            default_main_c="assault",
        )
        assert cfg.minimum_summon_count_before_skills == 5
        assert cfg.summon_recognition_delay_min == 1.0
        assert cfg.summon_recognition_delay_max == 2.0
        assert cfg.max_steps_per_round == 5000
        assert cfg.board_recognition_frames == 10
        assert cfg.coop_difficulties == list(range(1, 17))
        assert cfg.difficulty_max_scrolls == 15

    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("1-10", list(range(1, 11))),
            ("10-1", list(range(1, 11))),
            ("8", [8]),
            ([1, 3, 2, 3], [1, 2, 3]),
            ("1-19", list(range(1, 20))),
        ],
    )
    def test_coop_difficulties_are_normalized_ascending(
        self,
        spec: str | list[int],
        expected: list[int],
    ) -> None:
        assert parse_coop_difficulties(spec) == expected
        assert RunConfig(coop_difficulties=spec).coop_difficulties == expected

    @pytest.mark.parametrize("spec", ["", "0-10", "1-20", "a-b"])
    def test_invalid_coop_difficulties_rejected(self, spec: str) -> None:
        with pytest.raises((ValueError, ValidationError), match="coop_difficulties"):
            RunConfig(coop_difficulties=spec)

    @pytest.mark.parametrize("frames", [0, 61])
    def test_board_recognition_frames_out_of_range_rejected(self, frames: int) -> None:
        with pytest.raises(ValidationError, match="board_recognition_frames"):
            RunConfig(board_recognition_frames=frames)

    @pytest.mark.parametrize("steps", [0, 100001])
    def test_max_steps_per_round_out_of_range_rejected(self, steps: int) -> None:
        with pytest.raises(ValidationError, match="max_steps_per_round"):
            RunConfig(max_steps_per_round=steps)

    @pytest.mark.parametrize("interval", [0.0, 31.0])
    def test_find_coop_check_interval_out_of_range_rejected(self, interval: float) -> None:
        with pytest.raises(ValidationError, match="find_coop_check_interval_seconds"):
            RunConfig(find_coop_check_interval_seconds=interval)

    def test_find_coop_click_delay_min_greater_than_max_rejected(self) -> None:
        with pytest.raises(ValidationError, match="find_coop_click_delay_min"):
            RunConfig(find_coop_click_delay_min=0.6, find_coop_click_delay_max=0.5)

    def test_summon_recognition_delay_min_greater_than_max_rejected(self) -> None:
        with pytest.raises(ValidationError, match="summon_recognition_delay_min"):
            RunConfig(
                summon_recognition_delay_min=2.0,
                summon_recognition_delay_max=1.0,
            )


class TestDefaultConfig:
    """DefaultConfig 完整加载测试。"""

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
screen:
  mode: window
  window_title: 测试窗口
vision:
  default_threshold: 0.9
run:
  max_rounds: 5
  minimum_summon_count_before_skills: 3
  initial_board_capacity: 5
main_c_profiles:
  assault:
    display_name: 强袭
    hero_template_dir: heroes/assault
    hero_classifier_model: outputs/hero_classifier/assault-helper/hero_classifier.onnx
    skill_icon_templates: []
"""
        cfg_file = tmp_path / "default.yaml"
        cfg_file.write_text(yaml_content, encoding="utf-8")

        cfg = load_default_config(cfg_file)
        assert cfg.screen.window_title == "测试窗口"
        assert cfg.vision.default_threshold == 0.9
        assert cfg.run.max_rounds == 5
        assert cfg.run.minimum_summon_count_before_skills == 3
        assert "assault" in cfg.main_c_profiles
        assert cfg.main_c_profiles["assault"].display_name == "强袭"
        assert cfg.main_c_profiles["assault"].hero_classifier_model.endswith("hero_classifier.onnx")


class TestRoiConfig:
    """RoiConfig 比例范围校验测试。"""

    def test_valid_roi(self) -> None:
        roi = RoiConfig(x_ratio=0.5, y_ratio=0.6, width_ratio=0.3, height_ratio=0.2)
        assert roi.x_ratio == 0.5
        assert roi.relative_to == "client"

    def test_ratio_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError, match="x_ratio"):
            RoiConfig(x_ratio=1.5, y_ratio=0.0, width_ratio=0.1, height_ratio=0.1)

    def test_negative_ratio_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RoiConfig(x_ratio=-0.1, y_ratio=0.0, width_ratio=0.1, height_ratio=0.1)

    def test_zero_ratio_allowed(self) -> None:
        # 占位值 0.0 应被允许（棋盘坐标待标定时使用）
        roi = RoiConfig(x_ratio=0.0, y_ratio=0.0, width_ratio=0.0, height_ratio=0.0)
        assert roi.width_ratio == 0.0


def test_hotspot_rejects_out_of_client_ratio() -> None:
    with pytest.raises(ValidationError):
        Hotspot(x_ratio=1.1, y_ratio=0.5)


def test_window_spec_rejects_template_pack_path_traversal() -> None:
    with pytest.raises(ValidationError):
        WindowSpec(
            title="game",
            target_client_width=927,
            target_client_height=1727,
            template_pack="../outside",
        )


class TestTasksConfigRois:
    """TasksConfig.rois 结构化加载测试。"""

    def test_rois_structured(self, tmp_path: Path) -> None:
        yaml_content = """
rois:
  bottom_right_board:
    relative_to: client
    x_ratio: 0.5
    y_ratio: 0.5
    width_ratio: 0.4
    height_ratio: 0.3
"""
        cfg_file = tmp_path / "tasks.yaml"
        cfg_file.write_text(yaml_content, encoding="utf-8")
        cfg = load_tasks_config(cfg_file)
        assert "bottom_right_board" in cfg.rois
        assert isinstance(cfg.rois["bottom_right_board"], RoiConfig)
        assert cfg.rois["bottom_right_board"].x_ratio == 0.5

    def test_empty_rois(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "tasks.yaml"
        cfg_file.write_text("tasks: {}\n", encoding="utf-8")
        cfg = load_tasks_config(cfg_file)
        assert cfg.rois == {}

    def test_invalid_ratio_rejected(self, tmp_path: Path) -> None:
        yaml_content = """
rois:
  bad_board:
    x_ratio: 2.0
    y_ratio: 0.0
    width_ratio: 0.1
    height_ratio: 0.1
"""
        cfg_file = tmp_path / "tasks.yaml"
        cfg_file.write_text(yaml_content, encoding="utf-8")
        with pytest.raises(ValidationError, match="x_ratio"):
            load_tasks_config(cfg_file)

    def test_coop_difficulty_settings_are_preserved(self) -> None:
        from wlxq_bot.config import TasksConfig

        cfg = TasksConfig(
            difficulty_recognition={
                "candidate_roi": "coop_difficulty_list",
                "threshold": 0.8,
            }
        )
        assert cfg.difficulty_recognition["threshold"] == 0.8


class TestTasksConfigHotspots:
    def test_hotspots_are_structured_shared_coordinates(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "tasks.yaml"
        cfg_file.write_text(
            """
hotspots:
  home_chat:
    x_ratio: 0.25
    y_ratio: 0.75
    description: 首页聊天
""",
            encoding="utf-8",
        )

        cfg = load_tasks_config(cfg_file)

        assert isinstance(cfg.hotspots["home_chat"], Hotspot)
        assert cfg.hotspots["home_chat"].x_ratio == 0.25
        assert cfg.hotspots["home_chat"].description == "首页聊天"

    def test_empty_hotspots(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "tasks.yaml"
        cfg_file.write_text("hotspots:\n", encoding="utf-8")

        assert load_tasks_config(cfg_file).hotspots == {}

    def test_invalid_hotspot_ratio_rejected(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "tasks.yaml"
        cfg_file.write_text(
            """
hotspots:
  bad:
    x_ratio: -0.1
    y_ratio: 0.5
""",
            encoding="utf-8",
        )

        with pytest.raises(ValidationError, match="x_ratio"):
            load_tasks_config(cfg_file)
