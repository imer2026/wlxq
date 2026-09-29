"""配置加载与校验测试。

验证 Pydantic 模型对 YAML 配置的校验逻辑，
特别是逐次召唤门禁和动作后识别等待区间约束。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from wlxq_bot.config import (
    DefaultConfig,
    HeroClassifierConfig,
    Hotspot,
    MainCProfile,
    ModeProfile,
    RoiConfig,
    RunConfig,
    TasksConfig,
    WindowSpec,
    load_default_config,
    load_tasks_config,
    parse_coop_difficulties,
    resolve_main_c_profile,
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

    def test_settlement_screenshot_defaults_enabled(self) -> None:
        """结算截图默认开启：一局结束点击返回前保存结算画面。"""
        assert RunConfig().settlement_screenshot is True

    def test_settlement_screenshot_retention_days_default_and_bounds(self) -> None:
        """结算截图按天目录默认保留 7 天；保留天数必须 >= 1。"""
        assert RunConfig().settlement_screenshot_retention_days == 7
        with pytest.raises(ValidationError, match="settlement_screenshot_retention_days"):
            RunConfig(settlement_screenshot_retention_days=0)

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


class TestModeProfile:
    """模式档案模型与主C覆盖解析测试。"""

    def _mode(self, **kw) -> ModeProfile:
        base = {
            "display_name": "保卫宠物",
            "mode_chat_hotspot": "pet_chat",
            "entry": {"mode_tab_hotspot": "recruit_pet_tab",
                      "difficulty": {"kind": "ocr", "targets": ["普通", "中等"]}},
            "match": {"ready_button": "pet_ready_button", "leave_team": "pet_leave_team"},
            "settlement": {"like_chest": False, "return_button": "pet_return_button"},
            "main_c_overrides": {"assault": {"skill_selection_cap": 0}},
        }
        base.update(kw)
        return ModeProfile(**base)

    def _default(self, mode: str = "pet") -> DefaultConfig:
        return DefaultConfig(
            run={"mode": mode},
            main_c_profiles={"assault": MainCProfile(display_name="强袭", hero_template_dir="heroes/assault")},
            mode_profiles={
                "coop": ModeProfile(
                    mode_chat_hotspot="coop_chat",
                    match={"ready_button": "ready_button", "leave_team": "coop_leave_team"},
                    settlement={"return_button": "return_button"},
                ),
                "pet": self._mode(),
            },
        )

    def test_mode_profile_defaults(self) -> None:
        profile = self._mode()
        assert profile.entry.difficulty.kind == "ocr"
        assert profile.entry.difficulty.targets == ["普通", "中等"]
        assert profile.settlement.like_chest is False
        assert profile.strategy == "default"

    def test_run_mode_must_exist_in_mode_profiles(self) -> None:
        with pytest.raises(ValidationError):
            self._default(mode="not_a_mode")

    def test_ocr_kind_requires_targets(self) -> None:
        profile = self._mode(entry={"mode_tab_hotspot": None, "difficulty": {"kind": "ocr"}})
        with pytest.raises(ValidationError):
            DefaultConfig(
                run={"mode": "pet"},
                main_c_profiles={"assault": MainCProfile(display_name="强袭", hero_template_dir="h")},
                mode_profiles={"pet": profile},
            )

    def test_main_c_overrides_unknown_hero_rejected(self) -> None:
        profile = self._mode(main_c_overrides={"ghost": {"skill_selection_cap": 0}})
        with pytest.raises(ValidationError):
            DefaultConfig(
                run={"mode": "pet"},
                main_c_profiles={"assault": MainCProfile(display_name="强袭", hero_template_dir="h")},
                mode_profiles={"pet": profile},
            )

    def test_main_c_overrides_unknown_field_rejected(self) -> None:
        profile = self._mode(main_c_overrides={"assault": {"no_such_field": 1}})
        with pytest.raises(ValidationError):
            DefaultConfig(
                run={"mode": "pet"},
                main_c_profiles={"assault": MainCProfile(display_name="强袭", hero_template_dir="h")},
                mode_profiles={"pet": profile},
            )

    def test_resolve_main_c_profile_applies_override(self) -> None:
        cfg = self._default()
        resolved = resolve_main_c_profile(
            cfg.main_c_profiles, cfg.mode_profiles["pet"], "assault"
        )
        assert resolved.skill_selection_cap == 0
        # 未覆盖字段继承原档案
        assert resolved.topup_hero_count == cfg.main_c_profiles["assault"].topup_hero_count

    def test_resolve_main_c_profile_without_override_returns_base(self) -> None:
        cfg = self._default()
        resolved = resolve_main_c_profile(
            cfg.main_c_profiles, cfg.mode_profiles["coop"], "assault"
        )
        assert resolved is cfg.main_c_profiles["assault"]

    def test_resolve_main_c_profile_unknown_main_c_rejected(self) -> None:
        # CLI 只捕获 ValueError/RuntimeError：未知 main_c 必须抛 ValueError
        # 而不是裸 KeyError（会变成未处理 traceback）
        cfg = self._default()
        with pytest.raises(ValueError, match="不在配置档案"):
            resolve_main_c_profile(cfg.main_c_profiles, cfg.mode_profiles["pet"], "ghost")

    def test_ocr_targets_duplicates_rejected(self) -> None:
        # 重复目标会点击同一难度两次，实机确认再次点击已勾选难度会取消勾选
        profile = self._mode(
            entry={"mode_tab_hotspot": "recruit_pet_tab",
                   "difficulty": {"kind": "ocr", "targets": ["普通", "普通"]}},
        )
        with pytest.raises(ValidationError):
            DefaultConfig(
                run={"mode": "pet"},
                main_c_profiles={"assault": MainCProfile(display_name="强袭", hero_template_dir="h")},
                mode_profiles={"pet": profile},
            )

    def test_repo_default_yaml_defines_both_modes(self) -> None:
        """仓库自带 default.yaml 必须定义 coop/pet 两模式且可加载。"""
        cfg = load_default_config(Path("configs/default.yaml"))
        assert set(cfg.mode_profiles) >= {"coop", "pet"}
        assert cfg.run.mode == "coop"
        assert cfg.mode_profiles["pet"].entry.difficulty.kind == "ocr"
        assert cfg.mode_profiles["pet"].main_c_overrides["assault"]["skill_selection_cap"] == 0

    def test_repo_default_yaml_entry_profiles_define_territory_entry(self) -> None:
        """两个模式档案的领地入口字段齐备（卡片文字 + 挑战页锚点）。"""
        from pathlib import Path

        from wlxq_bot.config import load_default_config

        cfg = load_default_config(Path("configs/default.yaml"))
        for mode, card_text, challenge_text in (
            ("coop", "合作", "开始挑战"),
            ("pet", "保卫", "挑战"),
        ):
            entry = cfg.mode_profiles[mode].entry
            assert entry.mode_card_text == card_text, mode
            assert entry.mode_card_roi, mode
            assert entry.challenge_text == challenge_text, mode
            assert entry.challenge_roi, mode

    def test_repo_tasks_yaml_defines_mode_select_recognition(self) -> None:
        """领地入口 OCR 配置段齐备（标题锚点 + 等待上限）。"""
        from pathlib import Path

        from wlxq_bot.config import load_tasks_config

        cfg = load_tasks_config(Path("configs/tasks.yaml"))
        rec = cfg.mode_select_recognition
        assert rec["title_roi"] == "mode_select_title_area"
        assert rec["title_text"] == "模式选择"
        assert rec["min_confidence"] == 0.5
        assert rec["entry_wait_timeout_seconds"] == 60
        assert rec["entry_max_reclicks"] == 3
        for roi_name in (
            "territory_nav_area",
            "coop_mode_label_area",
            "pet_mode_label_area",
            "coop_challenge_start_area",
            "pet_challenge_start_area",
        ):
            roi = cfg.rois[roi_name]
            assert roi.width_ratio > 0 and roi.height_ratio > 0, roi_name
        assert "territory_nav" in cfg.hotspots
        assert "open_mode_select" in cfg.hotspots

    def test_mode_select_recognition_defaults_to_empty_dict(self) -> None:
        """YAML 空值统一转空字典：缺段不报错，由启动缺口检查兜底。"""
        cfg = TasksConfig()
        assert cfg.mode_select_recognition == {}

    def test_repo_tasks_yaml_defines_pet_bar_anchor(self) -> None:
        """仓库 tasks.yaml 必须定义保卫宠物收起栏锚点（弹窗开关的关闭态判据）。"""
        from wlxq_bot.config import load_tasks_config

        cfg = load_tasks_config(Path("configs/tasks.yaml"))
        rec = cfg.pet_difficulty_recognition
        assert rec["bar_roi"] == "pet_difficulty_bar_area"
        assert rec["bar_text"] == "选择难度"
        bar_roi = cfg.rois["pet_difficulty_bar_area"]
        assert bar_roi.width_ratio > 0 and bar_roi.height_ratio > 0


class TestLoggingConfig:
    """文件日志配置（configs/default.yaml ``logging`` 段）测试。"""

    def test_defaults(self) -> None:
        from wlxq_bot.config import LoggingConfig

        cfg = LoggingConfig()
        assert cfg.enabled is True
        assert cfg.dir == "logs"
        assert cfg.file_level == "DEBUG"
        assert cfg.retention_days == 7

    def test_retention_days_must_be_positive(self) -> None:
        from wlxq_bot.config import LoggingConfig

        with pytest.raises(ValidationError, match="retention_days"):
            LoggingConfig(retention_days=0)

    def test_load_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        from wlxq_bot.config import LoggingConfig, load_logging_config

        cfg = load_logging_config(tmp_path / "not_exist.yaml")
        assert cfg == LoggingConfig()

    def test_load_missing_section_returns_defaults(self, tmp_path: Path) -> None:
        from wlxq_bot.config import LoggingConfig, load_logging_config

        path = tmp_path / "default.yaml"
        path.write_text("screen:\n  mode: window\n", encoding="utf-8")
        assert load_logging_config(path) == LoggingConfig()

    def test_load_custom_section(self, tmp_path: Path) -> None:
        from wlxq_bot.config import load_logging_config

        path = tmp_path / "default.yaml"
        path.write_text(
            "logging:\n  enabled: false\n  dir: 'mylogs'\n"
            "  file_level: 'INFO'\n  retention_days: 3\n",
            encoding="utf-8",
        )
        cfg = load_logging_config(path)
        assert cfg.enabled is False
        assert cfg.dir == "mylogs"
        assert cfg.file_level == "INFO"
        assert cfg.retention_days == 3

    def test_load_invalid_section_type_rejected(self, tmp_path: Path) -> None:
        from wlxq_bot.config import load_logging_config

        path = tmp_path / "default.yaml"
        path.write_text("logging: not_a_dict\n", encoding="utf-8")
        with pytest.raises(ValueError, match="logging"):
            load_logging_config(path)

    def test_load_invalid_field_rejected(self, tmp_path: Path) -> None:
        from wlxq_bot.config import load_logging_config

        path = tmp_path / "default.yaml"
        path.write_text("logging:\n  retention_days: 0\n", encoding="utf-8")
        with pytest.raises(ValidationError):
            load_logging_config(path)

    def test_repo_default_yaml_logging_section(self) -> None:
        """仓库自带 default.yaml：文件日志默认开启、DEBUG、保留 7 天。"""
        from wlxq_bot.config import load_logging_config

        cfg = load_logging_config(Path("configs/default.yaml"))
        assert cfg.enabled is True
        assert cfg.file_level == "DEBUG"
        assert cfg.retention_days == 7


class TestTransformConfig:
    """英雄变身运行参数（设计文档 2026-09-13-hero-transform §4.4）。"""

    def test_transform_defaults(self) -> None:
        run = RunConfig()
        assert run.transform_enabled is True
        assert run.transform_max_per_round == 2
        assert run.transform_min_star_level == 3
        assert run.transform_button_poll_interval_seconds == 0.3
        assert run.transform_button_appear_timeout_seconds == 1.5
        assert run.transform_button_reclick_max == 2
        assert run.transform_settle_seconds == 4.0
        assert run.transform_verify_attempts == 2

    def test_transform_bounds_rejected(self) -> None:
        with pytest.raises(ValidationError, match="transform_max_per_round"):
            RunConfig(transform_max_per_round=-1)
        with pytest.raises(ValidationError, match="transform_min_star_level"):
            RunConfig(transform_min_star_level=0)
        with pytest.raises(ValidationError, match="transform_button_appear_timeout_seconds"):
            RunConfig(transform_button_appear_timeout_seconds=0.0)

    def test_repo_default_yaml_transform_section(self) -> None:
        """仓库 default.yaml 显式定义变身参数且与模型默认一致。"""
        from pathlib import Path

        from wlxq_bot.config import load_default_config

        cfg = load_default_config(Path("configs/default.yaml"))
        assert cfg.run.transform_enabled is True
        assert cfg.run.transform_settle_seconds == 4.0


class TestWildcardMergeAndVariants:
    """幻天使通配合并配置键（2026-09-13 spec §4.3）。"""

    def test_lineup_variants_default_and_parse(self) -> None:
        cfg = HeroClassifierConfig()
        assert cfg.lineup_variants == []
        cfg = HeroClassifierConfig(lineup_variants=["angel_phantom"])
        assert cfg.lineup_variants == ["angel_phantom"]

    def test_lineup_variants_reject_invalid(self) -> None:
        with pytest.raises(ValidationError, match="非法英雄标识"):
            HeroClassifierConfig(lineup_variants=["Angel-Phantom"])

    def test_variants_do_not_require_main_c_profile(self) -> None:
        """变体不是主 C 英雄：不在 main_c_profiles 也能通过 lineup 子集校验。"""
        cfg = DefaultConfig(
            main_c_profiles={
                "snow": MainCProfile(display_name="雪姬", hero_template_dir="heroes/snow")
            },
            hero_classifier=HeroClassifierConfig(
                lineup_others=["snow"], lineup_variants=["angel_phantom"]
            ),
        )
        assert "angel_phantom" not in cfg.main_c_profiles

    def test_wildcard_merge_heroes_parse_and_reject(self) -> None:
        run = RunConfig(wildcard_merge_heroes=["angel_phantom"])
        assert run.wildcard_merge_heroes == ["angel_phantom"]
        with pytest.raises(ValidationError, match="非法英雄标识"):
            RunConfig(wildcard_merge_heroes=["9bad"])

    def test_repo_default_yaml_lineup_others_account_split(self) -> None:
        """阵容分叉（2026-09-28）：monkey 档案显式狐狸阵容，全局默认仍是雪姬阵容。

        强袭号没换阵容且其模型无 fox 类——全局若仍是 fox 会让默认 run 启动校验
        直接失败（回归：英雄格模型缺少本局阵容类别: fox）。
        """
        cfg = load_default_config(Path("configs/default.yaml"))
        assert cfg.hero_classifier.lineup_others == ["angel", "snow", "death_knight"]
        assert cfg.main_c_profiles["monkey"].lineup_others == [
            "angel",
            "fox",
            "death_knight",
        ]
        # 未声明档案的 lineup_others 为空，语义 = 继承全局
        assert cfg.main_c_profiles["assault"].lineup_others == []

    def test_main_c_profile_lineup_others_rejects_bad_identifier(self) -> None:
        from wlxq_bot.config import MainCProfile

        with pytest.raises(ValidationError, match="非法英雄标识"):
            MainCProfile(
                display_name="x",
                hero_template_dir="h",
                lineup_others=["9bad"],
            )

    def test_runner_lineup_others_profile_overrides_global(self) -> None:
        """runner 取阵容：档案显式声明优先，未声明继承全局。"""
        from pathlib import Path

        from wlxq_bot.runner import Runner

        cfg = load_default_config(Path("configs/default.yaml"))
        runner = Runner.__new__(Runner)
        runner.default_config = cfg
        assert runner._lineup_others_for("monkey") == ["angel", "fox", "death_knight"]
        assert runner._lineup_others_for("assault") == ["angel", "snow", "death_knight"]
