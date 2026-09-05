"""Runner 调度循环单元测试。

用 Fake 组件（FakeScreen / FakePerception / FakeExecutor）验证循环逻辑，
不真实截图、不真实点击用户桌面。覆盖停止条件、动作执行和失败处理。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from wlxq_bot.action.safety import SafetyGuard
from wlxq_bot.config import DefaultConfig, LocalConfig, TasksConfig, WindowSpec
from wlxq_bot.models import (
    BoardCapacity,
    BoardHero,
    BoardSnapshot,
    Observation,
    State,
    WindowContext,
)
from wlxq_bot.runner import Runner
from wlxq_bot.tasks.coop import CoopTask


def make_window_info():
    from wlxq_bot.perception.screen import WindowInfo

    return WindowInfo(
        handle=1,
        title="永远的蔚蓝星球",
        class_name="Chrome_WidgetWin_0",
        is_visible=True,
        is_minimized=False,
        is_foreground=True,
        window_rect=(100, 100, 1047, 1847),
        client_rect=(110, 110, 927, 1727),
        client_size=(927, 1727),
        dpi=144,
        monitor_id=r"\\.\DISPLAY2",
        monitor_resolution=(2560, 1440),
        process_id=2,
        thread_id=3,
    )


class TemplatePackScreen:
    def get_window_info(self, _handle: int):
        return make_window_info()


# ---------------------------------------------------------------------------
# Fake 组件
# ---------------------------------------------------------------------------


class FakeScreen:
    """假截图器，返回预设 WindowContext 和帧。

    stale_age > 0 时把 captured_at 回拨指定秒数，模拟「识别决策耗时超过
    截图时效」——截图本身成功，但到执行动作时 WindowContext 已超龄。
    """

    def __init__(
        self,
        ctx: WindowContext,
        frame=None,
        raise_on_capture: bool = False,
        stale_age: float = 0.0,
    ) -> None:
        self._ctx = ctx
        self._frame = frame
        self._raise = raise_on_capture
        self._stale_age = stale_age
        self.capture_count = 0

    def capture(self, handle: int):
        self.capture_count += 1
        if self._raise:
            raise RuntimeError("fake capture failure")
        import time as _time

        return (
            replace(
                self._ctx,
                frame_id=self.capture_count,
                captured_at=_time.time() - self._stale_age,
            ),
            self._frame,
        )


class FakePerception:
    """假识别管线，返回预设 Observation。"""

    def __init__(self, observation: Observation, preserve_frame_id: bool = False) -> None:
        self._obs = observation
        self._preserve_frame_id = preserve_frame_id
        self.observe_count = 0
        self.cultivation_n_frames: list[int] = []
        self.observation_modes: list[str | None] = []

    def observe(
        self,
        ctx,
        frame,
        hint_state=State.UNKNOWN,
        observation_mode=None,
        read_gold=False,
    ):
        self.observe_count += 1
        self.observation_modes.append(observation_mode)
        if self._preserve_frame_id:
            return self._obs
        return replace(self._obs, frame_id=ctx.frame_id)

    def observe_cultivation(self, screen, handle, ctx, n_frames=10, read_gold=False):
        """培养阶段多帧累积的 fake：直接返回预设 observation。"""
        self.observe_count += 1
        self.cultivation_n_frames.append(n_frames)
        if self._preserve_frame_id:
            return ctx, self._obs
        return ctx, replace(self._obs, frame_id=ctx.frame_id)


class FakeExecutor:
    """假动作执行器，记录动作，可配置失败。"""

    def __init__(self, fail: bool = False, verified: bool = True) -> None:
        self.actions = []
        self._fail = fail
        self._verified = verified

    def execute(self, ctx, action):
        self.actions.append(action)
        from wlxq_bot.models import ActionResult

        if self._fail:
            return ActionResult(executed=False, verified=False, failure_reason="fake fail")
        return ActionResult(
            executed=True,
            verified=self._verified,
            failure_reason="fake pending verification" if not self._verified else "",
        )


# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------


def make_window_ctx(
    foreground: bool = True,
    minimized: bool = False,
    frame_id: int = 1,
) -> WindowContext:
    import time as _time

    return WindowContext(
        window_handle=1,
        client_rect_screen=(0, 0, 923, 1723),
        client_size=(923, 1723),
        dpi=96,
        monitor_id="primary",
        is_foreground=foreground,
        is_minimized=minimized,
        captured_at=_time.time(),
        frame_id=frame_id,
    )


def make_runner() -> Runner:
    return Runner(
        default_config=DefaultConfig(),
        tasks_config=TasksConfig(),
        local_config=None,
    )


def test_template_pack_defaults_to_game_window_monitor_resolution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pack_root = tmp_path / "2560x1440"
    pack_root.mkdir()
    monkeypatch.setattr("wlxq_bot.runner.TEMPLATES_ROOT", tmp_path)
    monkeypatch.setattr(
        "wlxq_bot.runner.get_window_monitor_resolution",
        lambda handle: (2560, 1440) if handle == 1 else (0, 0),
    )

    pack = make_runner()._load_template_pack(TemplatePackScreen(), 1)

    assert pack.root == pack_root
    assert pack.client_size == (927, 1727)


def test_explicit_template_pack_overrides_monitor_resolution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    explicit_root = tmp_path / "calibrated"
    explicit_root.mkdir()
    monkeypatch.setattr("wlxq_bot.runner.TEMPLATES_ROOT", tmp_path)
    monkeypatch.setattr(
        "wlxq_bot.runner.get_window_monitor_resolution",
        lambda _handle: (_ for _ in ()).throw(AssertionError("不应读取显示器分辨率")),
    )
    local_config = LocalConfig(
        window=WindowSpec(
            title="永远的蔚蓝星球",
            target_client_width=927,
            target_client_height=1727,
            template_pack="calibrated",
        )
    )
    runner = Runner(DefaultConfig(), TasksConfig(), local_config)

    pack = runner._load_template_pack(TemplatePackScreen(), 1)

    assert pack.root == explicit_root


def test_local_client_size_mismatch_stops_before_loading_template_pack(tmp_path) -> None:
    local_config = LocalConfig(
        window=WindowSpec(
            title="永远的蔚蓝星球",
            target_client_width=800,
            target_client_height=1600,
            template_pack="calibrated",
        )
    )
    runner = Runner(DefaultConfig(), TasksConfig(), local_config)

    with pytest.raises(RuntimeError, match="adjust-window"):
        runner._load_template_pack(TemplatePackScreen(), 1)


class _MutableSizeScreen:
    """可变客户区尺寸的假截图器：adjust 成功后更新为目标尺寸。"""

    def __init__(self, client_size: tuple[int, int]) -> None:
        self._client_size = client_size
        self.adjust_calls: list[tuple[int, int, int]] = []

    def get_window_info(self, _handle: int):
        info = make_window_info()
        return replace(info, client_size=self._client_size)


def test_run_auto_adjusts_window_size(monkeypatch) -> None:
    """客户区与本机配置不一致时自动调整（等价 adjust-window），不再要求手动执行。"""
    local_config = LocalConfig(
        window=WindowSpec(
            title="永远的蔚蓝星球",
            target_client_width=927,
            target_client_height=1727,
            template_pack="calibrated",
        )
    )
    runner = Runner(DefaultConfig(), TasksConfig(), local_config)
    screen = _MutableSizeScreen((800, 1600))
    calls: list[tuple[int, int, int]] = []

    def fake_adjust(handle, width, height):
        calls.append((handle, width, height))
        screen._client_size = (width, height)
        return replace(make_window_info(), client_size=(width, height))

    monkeypatch.setattr("wlxq_bot.runner.adjust_window_size", fake_adjust)

    runner._ensure_window_size(screen, 1)

    assert calls == [(1, 927, 1727)]


def test_run_auto_adjust_disabled_stops_with_hint(monkeypatch) -> None:
    """auto_adjust_window 关闭：退回旧行为，报错提示手动 adjust-window。"""
    from wlxq_bot.config import RunConfig

    local_config = LocalConfig(
        window=WindowSpec(
            title="永远的蔚蓝星球",
            target_client_width=927,
            target_client_height=1727,
            template_pack="calibrated",
        )
    )
    runner = Runner(
        DefaultConfig(run=RunConfig(auto_adjust_window=False)),
        TasksConfig(),
        local_config,
    )
    screen = _MutableSizeScreen((800, 1600))
    monkeypatch.setattr(
        "wlxq_bot.runner.adjust_window_size",
        lambda *args: pytest.fail("不应调用窗口调整"),
    )

    with pytest.raises(RuntimeError, match="adjust-window"):
        runner._ensure_window_size(screen, 1)


def test_run_skips_adjust_when_size_matches(monkeypatch) -> None:
    """客户区已一致：不做任何调整。"""
    local_config = LocalConfig(
        window=WindowSpec(
            title="永远的蔚蓝星球",
            target_client_width=927,
            target_client_height=1727,
            template_pack="calibrated",
        )
    )
    runner = Runner(DefaultConfig(), TasksConfig(), local_config)
    screen = _MutableSizeScreen((927, 1727))
    monkeypatch.setattr(
        "wlxq_bot.runner.adjust_window_size",
        lambda *args: pytest.fail("不应调用窗口调整"),
    )

    runner._ensure_window_size(screen, 1)


def test_missing_monitor_resolution_template_pack_stops(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # 即使客户区同名目录存在，也不得用它或其他分辨率兜底。
    (tmp_path / "927x1727").mkdir()
    monkeypatch.setattr("wlxq_bot.runner.TEMPLATES_ROOT", tmp_path)
    monkeypatch.setattr(
        "wlxq_bot.runner.get_window_monitor_resolution",
        lambda _handle: (2560, 1440),
    )

    with pytest.raises(RuntimeError, match="2560x1440"):
        make_runner()._load_template_pack(TemplatePackScreen(), 1)


def make_board(heroes: list[BoardHero]) -> BoardSnapshot:
    return BoardSnapshot(
        frame_id=1,
        heroes=heroes,
        capacity=BoardCapacity(total_slots=12, occupied=len(heroes)),
    )


def test_run_rejects_invalid_max_rounds() -> None:
    """命令行局数覆盖必须 >= 1，且在任何窗口/环境初始化前就拒绝。"""
    runner = Runner(default_config=DefaultConfig(), tasks_config=TasksConfig())

    with pytest.raises(ValueError, match="max_rounds"):
        runner.run("coop", max_rounds=0)


def test_normal_run_requires_skill_icon_templates() -> None:
    from wlxq_bot.config import MainCProfile

    runner = Runner(
        default_config=DefaultConfig(
            main_c_profiles={
                "assault": MainCProfile(
                    display_name="强袭",
                    hero_template_dir="heroes/assault",
                    hero_classifier_model="outputs/assault.onnx",
                    skill_icon_templates=[],
                )
            }
        ),
        tasks_config=TasksConfig(),
    )

    with pytest.raises(RuntimeError, match="skill_icon_templates"):
        runner._startup_check("assault", State.FIND_COOP)

    runner._startup_check("assault", State.BUILD_MAIN_C)


def test_normal_run_requires_skill_template_files(tmp_path) -> None:
    from wlxq_bot.assets import TemplatePack
    from wlxq_bot.config import MainCProfile

    pack_root = tmp_path / "profile"
    (pack_root / "skills").mkdir(parents=True)
    runner = Runner(
        default_config=DefaultConfig(
            main_c_profiles={
                "assault": MainCProfile(
                    display_name="强袭",
                    hero_template_dir="heroes/assault",
                    hero_classifier_model="outputs/assault.onnx",
                    skill_icon_templates=["skills/qiang_xi.png"],
                )
            }
        ),
        tasks_config=TasksConfig(),
    )
    pack = TemplatePack(client_size=(927, 1727), root=pack_root)

    with pytest.raises(RuntimeError, match="qiang_xi"):
        runner._check_skill_templates("assault", State.FIND_COOP, pack)

    (pack_root / "skills" / "qiang_xi.png").write_bytes(b"template")
    runner._check_skill_templates("assault", State.FIND_COOP, pack)


def test_startup_requires_hero_classifier_model() -> None:
    from wlxq_bot.config import MainCProfile

    runner = Runner(
        default_config=DefaultConfig(
            main_c_profiles={
                "assault": MainCProfile(
                    display_name="强袭",
                    hero_template_dir="heroes/assault",
                    hero_classifier_model="",
                    skill_icon_templates=[],
                )
            }
        ),
        tasks_config=TasksConfig(),
    )

    with pytest.raises(RuntimeError, match="hero_classifier_model"):
        runner._startup_check("assault", State.BUILD_MAIN_C)


def test_load_hero_classifier_reports_missing_model(tmp_path) -> None:
    from wlxq_bot.config import MainCProfile

    model_path = tmp_path / "missing.onnx"
    runner = Runner(
        default_config=DefaultConfig(
            main_c_profiles={
                "assault": MainCProfile(
                    display_name="强袭",
                    hero_template_dir="heroes/assault",
                    hero_classifier_model=str(model_path),
                )
            }
        ),
        tasks_config=TasksConfig(),
    )

    with pytest.raises(RuntimeError, match="英雄格分类模型加载失败"):
        runner._load_hero_cell_classifier("assault")


def test_normal_run_requires_requested_difficulty_templates(tmp_path) -> None:
    from wlxq_bot.assets import TemplatePack

    pack_root = tmp_path / "profile"
    difficulty_dir = pack_root / "buttons" / "coop_difficulty"
    difficulty_dir.mkdir(parents=True)
    pack = TemplatePack(client_size=(927, 1727), root=pack_root)

    with pytest.raises(RuntimeError, match="10"):
        Runner._check_difficulty_templates([10, 9], State.FIND_COOP, pack)

    for level in (10, 9):
        (difficulty_dir / f"cai_hong_{level}.png").write_bytes(b"template")
    Runner._check_difficulty_templates([10, 9], State.FIND_COOP, pack)


def test_skip_difficulty_selection_does_not_require_difficulty_templates(tmp_path) -> None:
    from wlxq_bot.assets import TemplatePack

    pack_root = tmp_path / "profile"
    pack_root.mkdir(parents=True)
    pack = TemplatePack(client_size=(927, 1727), root=pack_root)

    # 跳过难度选择时启动检查不要求难度模板存在
    Runner._check_difficulty_templates([10, 9], State.FIND_COOP, pack, skip_difficulty=True)


def test_normal_run_requires_home_page_template(tmp_path) -> None:
    from wlxq_bot.assets import TemplatePack

    pack_root = tmp_path / "profile"
    buttons_dir = pack_root / "buttons"
    buttons_dir.mkdir(parents=True)
    pack = TemplatePack(client_size=(927, 1727), root=pack_root)
    runner = Runner(
        DefaultConfig(),
        TasksConfig(
            locators={
                "home_page_marker": {
                    "strategy": "template",
                    "template": "buttons/home_page_marker.png",
                    "threshold": 0.82,
                }
            }
        ),
    )

    with pytest.raises(RuntimeError, match="首页识别模板缺失"):
        runner._check_home_page_template(State.FIND_COOP, pack)

    (buttons_dir / "home_page_marker.png").write_bytes(b"template")
    runner._check_home_page_template(State.FIND_COOP, pack)
    runner._check_home_page_template(State.BUILD_MAIN_C, pack)


def test_normal_run_requires_calibrated_home_page_threshold(tmp_path) -> None:
    from wlxq_bot.assets import TemplatePack

    pack_root = tmp_path / "profile"
    buttons_dir = pack_root / "buttons"
    buttons_dir.mkdir(parents=True)
    (buttons_dir / "home_page_marker.png").write_bytes(b"template")
    pack = TemplatePack(client_size=(927, 1727), root=pack_root)
    runner = Runner(
        DefaultConfig(),
        TasksConfig(
            locators={
                "home_page_marker": {
                    "strategy": "template",
                    "template": "buttons/home_page_marker.png",
                    "threshold": None,
                }
            }
        ),
    )

    with pytest.raises(RuntimeError, match="首页识别阈值未标定"):
        runner._check_home_page_template(State.FIND_COOP, pack)


# ---------------------------------------------------------------------------
# 停止条件
# ---------------------------------------------------------------------------


class TestLoopStopConditions:
    def test_runner_enables_difficulty_observation_only_for_dialog_step(self):
        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.FIND_COOP
        task = CoopTask(
            ctx,
            RunConfig(),
            CoopRole.HELPER,
            {
                "home_chat": Hotspot(x_ratio=0.1, y_ratio=0.2),
                "open_recruit": Hotspot(x_ratio=0.2, y_ratio=0.3),
                "open_difficulty_dialog": Hotspot(x_ratio=0.3, y_ratio=0.4),
            },
            coop_difficulties=[1],
        )
        perception = FakePerception(Observation(frame_id=1, raw_data={"home_page_visible": True}))

        make_runner()._run_loop(
            FakeScreen(make_window_ctx()),
            perception,
            FakeExecutor(),
            task,
            SafetyGuard(max_failures=3, frame_ttl_ms=5000),
            1,
            max_steps=4,
        )

        # 打开难度弹窗后先进入打开确认步骤：只用标题图轻量识别（difficulty_dialog），
        # 候选识别（coop_difficulty）要到确认打开后的勾选步骤才启用
        assert perception.observation_modes == ["home_page", None, None, "difficulty_dialog"]

    def test_decision_log_includes_round_number(self, caplog):
        """主决策日志带局号；一局完成时输出局号汇总。"""
        import logging

        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole, MatchResult
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        ctx = TaskContext(main_c="assault", max_rounds=2)
        ctx.current_state = State.CHECK_ROUND_LIMIT
        task = CoopTask(
            ctx,
            RunConfig(max_rounds=2),
            CoopRole.HELPER,
            {
                "coop_chat": Hotspot(x_ratio=0.5, y_ratio=0.6),
                "open_difficulty_dialog": Hotspot(x_ratio=0.3, y_ratio=0.4),
            },
        )

        class _ReturnThenGone(FakePerception):
            """第一帧返回按钮可见（决策点击），第二帧消失（验证通过）。"""

            def __init__(self):
                super().__init__(Observation(frame_id=1))
                self._button = Observation(
                    frame_id=1,
                    raw_data={
                        "return_button_visible": True,
                        "return_button_match": MatchResult("buttons/fan_hui.png", (500, 1000), 0.9),
                    },
                )
                self._gone = Observation(frame_id=2)

            def observe(self, ctx, frame, hint_state=State.UNKNOWN, observation_mode=None, read_gold=False):
                self.observe_count += 1
                self.observation_modes.append(observation_mode)
                return self._button if self.observe_count == 1 else self._gone

        with caplog.at_level(logging.INFO, logger="wlxq_bot.runner"):
            make_runner()._run_loop(
                FakeScreen(make_window_ctx()),
                _ReturnThenGone(),
                FakeExecutor(verified=False),
                task,
                SafetyGuard(max_failures=3, frame_ttl_ms=5000),
                1,
                max_steps=5,
            )

        messages = " ".join(record.getMessage() for record in caplog.records)
        # 返回按钮决策发生在第 1 局进行中（round_count=0 → 局=1/2）
        assert "局=1/2 check_round_limit -> find_coop action=click" in messages
        # 返回验证通过、局数 +1 后输出局完成日志
        assert "第 1 局完成（1/2）" in messages

    def test_stops_immediately_on_completed(self):
        """round_count 达上限时 determine_state 返回 COMPLETED，循环立即结束。"""
        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        ctx = TaskContext(main_c="assault", max_rounds=1)
        ctx.round_count = 1
        ctx.current_state = State.CHECK_ROUND_LIMIT
        run_cfg = RunConfig(
            max_rounds=1,
            minimum_summon_count_before_skills=2,
            error_restart_enabled=False,
        )
        task = CoopTask(
            ctx, run_cfg, CoopRole.HELPER, {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)}
        )

        screen = FakeScreen(make_window_ctx())
        perception = FakePerception(Observation(frame_id=1))
        executor = FakeExecutor()
        safety = SafetyGuard(max_failures=3, frame_ttl_ms=5000)

        final = make_runner()._run_loop(screen, perception, executor, task, safety, 1, max_steps=10)
        assert final == State.COMPLETED
        assert len(executor.actions) == 0
        # COMPLETED 在截图→识别后才判定，所以截了 1 次
        assert screen.capture_count == 1

    def test_stops_when_decide_action_none(self):
        """decide_action 返回 None：自动重开关闭时保守停止。"""
        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.UNKNOWN  # decide_action 返回 None
        run_cfg = RunConfig(error_restart_enabled=False)
        task = CoopTask(
            ctx, run_cfg, CoopRole.HELPER, {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)}
        )

        screen = FakeScreen(make_window_ctx())
        perception = FakePerception(Observation(frame_id=1))
        executor = FakeExecutor()
        safety = SafetyGuard(max_failures=3, frame_ttl_ms=5000)

        final = make_runner()._run_loop(screen, perception, executor, task, safety, 1, max_steps=10)
        assert final == State.UNKNOWN
        assert len(executor.actions) == 0

    def test_decide_action_none_recovers_in_place_then_stops_at_budget(self, caplog):
        """回归：decide None 时原地继续识别（默认 3 次），预算耗尽才停止。

        实机 2026-09-05 需求：识别出错不退出，原地重新识别继续对局。
        """
        import logging

        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.UNKNOWN  # decide_action 返回 None
        run_cfg = RunConfig()  # error_restart_enabled=True, max=3
        task = CoopTask(
            ctx, run_cfg, CoopRole.HELPER, {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)}
        )

        screen = FakeScreen(make_window_ctx())
        perception = FakePerception(Observation(frame_id=1))
        executor = FakeExecutor()
        safety = SafetyGuard(max_failures=3, frame_ttl_ms=5000)

        with caplog.at_level(logging.WARNING, logger="wlxq_bot.runner"):
            final = make_runner()._run_loop(
                screen, perception, executor, task, safety, 1, max_steps=100
            )
        assert final == State.UNKNOWN  # 原地恢复不重置状态
        assert len(executor.actions) == 0
        recoveries = sum(
            1 for r in caplog.records if "原地继续识别" in r.getMessage()
        )
        assert recoveries == 3

    def test_stops_on_stop_requested(self):
        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.BUILD_MAIN_C
        task = CoopTask(
            ctx, RunConfig(), CoopRole.HELPER, {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)}
        )

        safety = SafetyGuard(max_failures=3, frame_ttl_ms=5000)
        safety.request_stop()  # 预先触发停止

        screen = FakeScreen(make_window_ctx())
        perception = FakePerception(Observation(frame_id=1))
        executor = FakeExecutor()

        final = make_runner()._run_loop(screen, perception, executor, task, safety, 1, max_steps=10)
        assert final == State.BUILD_MAIN_C
        assert screen.capture_count == 0  # 未进循环

    def test_stops_on_max_steps_per_round(self):
        """单局培养循环持续运行，靠单局步数保险停止。"""
        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.BUILD_MAIN_C
        run_cfg = RunConfig(minimum_summon_count_before_skills=100)  # 永远处于技能解锁前召唤
        task = CoopTask(
            ctx, run_cfg, CoopRole.HELPER, {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)}
        )

        screen = FakeScreen(make_window_ctx())
        # board=None 但技能解锁前识别不阻断召唤，持续 click
        perception = FakePerception(Observation(frame_id=1, board=None))
        executor = FakeExecutor()
        safety = SafetyGuard(max_failures=3, frame_ttl_ms=5000)

        final = make_runner()._run_loop(screen, perception, executor, task, safety, 1, max_steps=5)
        assert final == State.BUILD_MAIN_C
        assert screen.capture_count == 5
        assert len(executor.actions) == 5
        assert all(a.kind == "click" for a in executor.actions)
        # 强制召唤阶段不以棋盘识别为门禁：退回单帧识别，不做多帧棋盘识别
        assert perception.cultivation_n_frames == []

    def test_max_steps_resets_after_round_count_increases(self):
        """每局可分别使用完整步数预算，不把多局步骤累计到同一个上限。"""
        from wlxq_bot.models import Action
        from wlxq_bot.tasks.base import TaskContext

        class TwoRoundTask:
            def __init__(self) -> None:
                self.ctx = TaskContext(
                    main_c="assault",
                    current_state=State.FIND_COOP,
                    max_rounds=2,
                )
                self.actions_in_round = 0

            def observation_mode(self):
                return None

            def wants_gold_read(self):
                return False

            def determine_state(self, observation):
                return self.ctx.current_state

            def decide_action(self, observation, window_ctx=None):
                finishes_round = self.actions_in_round == 1
                finishes_task = finishes_round and self.ctx.round_count + 1 >= self.ctx.max_rounds
                return (
                    Action(kind="wait", duration=0.0, verification="immediate"),
                    State.COMPLETED if finishes_task else State.FIND_COOP,
                )

            def on_action_verified(self, action, to_state):
                self.actions_in_round += 1
                if self.actions_in_round == 2:
                    self.ctx.round_count += 1
                    self.actions_in_round = 0

        task = TwoRoundTask()
        screen = FakeScreen(make_window_ctx())
        executor = FakeExecutor()

        final = make_runner()._run_loop(
            screen,
            FakePerception(Observation(frame_id=1)),
            executor,
            task,
            SafetyGuard(max_failures=3, frame_ttl_ms=5000),
            1,
            max_steps=2,
        )

        assert final == State.COMPLETED
        assert task.ctx.round_count == 2
        assert screen.capture_count == 4
        assert len(executor.actions) == 4

    def test_uses_configured_board_recognition_frames(self):
        """培养阶段把配置的多帧数量传给识别管线。"""
        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        run_cfg = RunConfig(
            minimum_summon_count_before_skills=100,
            board_recognition_frames=3,
        )
        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.BUILD_MAIN_C
        task = CoopTask(
            ctx, run_cfg, CoopRole.HELPER, {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)}
        )
        # 已完成强制召唤：处于棋盘决策阶段才做多帧棋盘识别
        task._summon_count = run_cfg.minimum_summon_count_before_skills
        screen = FakeScreen(make_window_ctx())
        perception = FakePerception(Observation(frame_id=1, board=None))
        executor = FakeExecutor()
        safety = SafetyGuard(max_failures=3, frame_ttl_ms=5000)
        runner = Runner(DefaultConfig(run=run_cfg), TasksConfig(), None)

        runner._run_loop(screen, perception, executor, task, safety, 1, max_steps=1)

        assert perception.cultivation_n_frames == [3]


# ---------------------------------------------------------------------------
# 失败处理
# ---------------------------------------------------------------------------


class TestLoopFailures:
    def test_required_summons_continue_when_board_recognition_is_empty(self):
        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.BUILD_MAIN_C
        task = CoopTask(
            ctx,
            RunConfig(minimum_summon_count_before_skills=2),
            CoopRole.HELPER,
            {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)},
        )
        screen = FakeScreen(make_window_ctx())
        perception = FakePerception(Observation(frame_id=1, board=make_board([])))
        executor = FakeExecutor(verified=False)
        safety = SafetyGuard(max_failures=3, frame_ttl_ms=5000)

        final = make_runner()._run_loop(
            screen,
            perception,
            executor,
            task,
            safety,
            1,
            max_steps=5,
        )

        assert final == State.BUILD_MAIN_C
        # 前置召唤点击成功即计数；识别为空不会让流程卡在 pending 验证。
        assert len(executor.actions) == 5
        assert task._summon_count == 2
        assert [action.tag for action in executor.actions[:2]] == [
            "required_summon",
            "required_summon",
        ]
        assert all(action.kind == "wait" for action in executor.actions[2:])

    def test_stops_when_observation_frame_does_not_match_context(self):
        """识别结果不能冒用其他截图帧的 WindowContext。"""
        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.BUILD_MAIN_C
        task = CoopTask(
            ctx,
            RunConfig(),
            CoopRole.HELPER,
            {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)},
        )
        screen = FakeScreen(make_window_ctx())
        perception = FakePerception(
            Observation(frame_id=999),
            preserve_frame_id=True,
        )
        executor = FakeExecutor()
        safety = SafetyGuard(max_failures=3, frame_ttl_ms=5000)

        final = make_runner()._run_loop(
            screen,
            perception,
            executor,
            task,
            safety,
            1,
            max_steps=3,
        )

        assert final == State.BUILD_MAIN_C
        assert executor.actions == []

    def test_stops_on_capture_failure_limit(self):
        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.FIND_COOP
        task = CoopTask(
            ctx, RunConfig(), CoopRole.HELPER, {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)}
        )

        screen = FakeScreen(make_window_ctx(), raise_on_capture=True)
        perception = FakePerception(Observation(frame_id=1))
        executor = FakeExecutor()
        safety = SafetyGuard(max_failures=3, frame_ttl_ms=5000)

        final = make_runner()._run_loop(screen, perception, executor, task, safety, 1, max_steps=10)
        assert final == State.FIND_COOP
        assert screen.capture_count == 3  # 3 次失败后达上限
        assert len(executor.actions) == 0

    def test_stops_on_executor_failure_limit(self):
        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.FIND_COOP
        task = CoopTask(
            ctx,
            RunConfig(),
            CoopRole.HELPER,
            {
                "add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96),
                "home_chat": Hotspot(x_ratio=0.6, y_ratio=0.75),
            },
        )

        obs = Observation(frame_id=1, raw_data={"home_page_visible": True})
        screen = FakeScreen(make_window_ctx())
        perception = FakePerception(obs)
        executor = FakeExecutor(fail=True)  # 动作执行总失败
        safety = SafetyGuard(max_failures=3, frame_ttl_ms=5000)

        final = make_runner()._run_loop(screen, perception, executor, task, safety, 1, max_steps=10)
        assert final == State.FIND_COOP
        assert len(executor.actions) == 3  # 3 次失败后达上限


# ---------------------------------------------------------------------------
# 识别决策超过截图时效
# ---------------------------------------------------------------------------


class _StaleThenFreshScreen(FakeScreen):
    """前 stale_frames 次截图超龄（模拟识别管线临时变慢），之后恢复新鲜。"""

    def __init__(self, ctx: WindowContext, stale_frames: int) -> None:
        super().__init__(ctx)
        self.stale_frames = stale_frames

    def capture(self, handle: int):
        self._stale_age = 10.0 if self.capture_count < self.stale_frames else 0.0
        return super().capture(handle)


class TestStaleDecision:
    def test_stale_decision_dropped_and_retries_without_failure(self, caplog):
        """超龄决策被丢弃重新截图，不计动作失败，识别恢复后流程继续。"""
        import logging

        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.BUILD_MAIN_C
        task = CoopTask(
            ctx, RunConfig(), CoopRole.HELPER, {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)}
        )
        executor = FakeExecutor()

        with caplog.at_level(logging.WARNING, logger="wlxq_bot.runner"):
            final = make_runner()._run_loop(
                _StaleThenFreshScreen(make_window_ctx(), stale_frames=2),
                FakePerception(Observation(frame_id=1)),
                executor,
                task,
                SafetyGuard(max_failures=3, frame_ttl_ms=3000),
                1,
                max_steps=5,
            )

        assert final == State.BUILD_MAIN_C
        # 前两轮决策超龄被丢弃（未执行输入、未计安全失败），后三轮正常召唤
        assert len(executor.actions) == 3
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "丢弃本轮决策重新截图" in messages
        assert "连续失败达上限" not in messages

    def test_stale_decision_stops_after_limit(self, caplog):
        """截图持续超龄（本机无法在时效内完成识别到动作）：自动重开关闭时保守停止。"""
        import logging

        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.BUILD_MAIN_C
        task = CoopTask(
            ctx,
            RunConfig(error_restart_enabled=False),
            CoopRole.HELPER,
            {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)},
        )
        executor = FakeExecutor()
        screen = FakeScreen(make_window_ctx(), stale_age=10.0)

        with caplog.at_level(logging.WARNING, logger="wlxq_bot.runner"):
            final = make_runner()._run_loop(
                screen,
                FakePerception(Observation(frame_id=1)),
                executor,
                task,
                SafetyGuard(max_failures=3, frame_ttl_ms=3000),
                1,
                max_steps=100,
            )

        assert final == State.BUILD_MAIN_C
        assert executor.actions == []

    def test_stale_decision_recovers_then_stops_at_budget(self, caplog):
        """回归：截图持续超龄时原地继续识别（默认 3 次），预算耗尽才停止。"""
        import logging

        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.BUILD_MAIN_C
        task = CoopTask(
            ctx, RunConfig(), CoopRole.HELPER, {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)}
        )
        executor = FakeExecutor()
        screen = FakeScreen(make_window_ctx(), stale_age=10.0)

        with caplog.at_level(logging.WARNING, logger="wlxq_bot.runner"):
            final = make_runner()._run_loop(
                screen,
                FakePerception(Observation(frame_id=1)),
                executor,
                task,
                SafetyGuard(max_failures=3, frame_ttl_ms=3000),
                1,
                max_steps=100,
            )
        assert final == State.BUILD_MAIN_C
        assert executor.actions == []
        recoveries = sum(
            1 for r in caplog.records if "原地继续识别" in r.getMessage()
        )
        assert recoveries == 3


# ---------------------------------------------------------------------------
# 窗口状态
# ---------------------------------------------------------------------------


class TestLoopWindowState:
    def test_refocus_when_user_idle(self, monkeypatch, caplog):
        """系统空闲超阈值：自动激活游戏窗口切回前台，继续挂起循环不发送输入。"""
        import logging

        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        activated: list[int] = []
        monkeypatch.setattr("wlxq_bot.runner.get_input_idle_seconds", lambda: 10.0)
        monkeypatch.setattr(
            "wlxq_bot.runner.activate_window",
            lambda handle: activated.append(handle) or True,
        )
        runner = Runner(
            default_config=DefaultConfig(
                run=RunConfig(window_foreground_wait_seconds=1.0, refocus_idle_seconds=5.0)
            ),
            tasks_config=TasksConfig(),
        )
        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.BUILD_MAIN_C
        task = CoopTask(
            ctx, RunConfig(), CoopRole.HELPER, {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)}
        )
        executor = FakeExecutor()

        with caplog.at_level(logging.INFO, logger="wlxq_bot.runner"):
            final = runner._run_loop(
                FakeScreen(make_window_ctx(foreground=False)),
                FakePerception(Observation(frame_id=1)),
                executor,
                task,
                SafetyGuard(max_failures=3, frame_ttl_ms=5000),
                1,
                max_steps=10,
            )

        # 空闲触发一次自动切回（窗口仍非前台 → 重试被节流 → 挂起超时停止）
        assert activated == [1]
        assert "自动切回游戏窗口继续任务" in " ".join(r.getMessage() for r in caplog.records)
        assert len(executor.actions) == 0
        assert final == State.BUILD_MAIN_C

    def test_no_refocus_while_user_active(self, monkeypatch):
        """用户正在操作（空闲不足阈值）：不抢焦点，保持挂起等待。"""
        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        activated: list[int] = []
        monkeypatch.setattr("wlxq_bot.runner.get_input_idle_seconds", lambda: 1.0)
        monkeypatch.setattr(
            "wlxq_bot.runner.activate_window",
            lambda handle: activated.append(handle) or True,
        )
        runner = Runner(
            default_config=DefaultConfig(
                run=RunConfig(window_foreground_wait_seconds=1.0, refocus_idle_seconds=5.0)
            ),
            tasks_config=TasksConfig(),
        )
        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.BUILD_MAIN_C
        task = CoopTask(
            ctx, RunConfig(), CoopRole.HELPER, {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)}
        )

        final = runner._run_loop(
            FakeScreen(make_window_ctx(foreground=False)),
            FakePerception(Observation(frame_id=1)),
            FakeExecutor(),
            task,
            SafetyGuard(max_failures=3, frame_ttl_ms=5000),
            1,
            max_steps=10,
        )

        assert activated == []  # 从未抢焦点
        assert final == State.BUILD_MAIN_C

    def test_foreground_wait_times_out_after_configured_seconds(self, caplog):
        """窗口持续非前台：挂起等待，超过配置时长才保守停止（期间不发送输入）。"""
        import logging

        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        runner = Runner(
            default_config=DefaultConfig(
                run=RunConfig(
                    window_foreground_wait_seconds=1.0,
                    refocus_when_idle=False,  # 关闭自动切回，保持本测试封闭
                )
            ),
            tasks_config=TasksConfig(),
        )
        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.BUILD_MAIN_C
        task = CoopTask(
            ctx, RunConfig(), CoopRole.HELPER, {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)}
        )

        screen = FakeScreen(make_window_ctx(foreground=False))
        executor = FakeExecutor()

        with caplog.at_level(logging.WARNING, logger="wlxq_bot.runner"):
            final = runner._run_loop(
                screen,
                FakePerception(Observation(frame_id=1)),
                executor,
                task,
                SafetyGuard(max_failures=3, frame_ttl_ms=5000),
                1,
                max_steps=10,
            )

        # 0.5s × 2 次达到 1.0s 上限 → 停止
        assert final == State.BUILD_MAIN_C
        assert screen.capture_count == 2
        assert len(executor.actions) == 0  # 挂起期间从未发送输入
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "任务挂起等待切回" in messages
        assert "已持续 1 秒" in messages

    def test_skips_when_window_minimized(self):
        """窗口最小化时跳过本轮，不识别不执行。"""
        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.BUILD_MAIN_C
        task = CoopTask(
            ctx, RunConfig(), CoopRole.HELPER, {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)}
        )

        screen = FakeScreen(make_window_ctx(minimized=True))
        perception = FakePerception(Observation(frame_id=1, board=make_board([])))
        executor = FakeExecutor()
        safety = SafetyGuard(max_failures=3, frame_ttl_ms=5000)

        final = make_runner()._run_loop(screen, perception, executor, task, safety, 1, max_steps=3)
        assert final == State.BUILD_MAIN_C
        # 每轮都因最小化跳过，3 步后达上限
        assert screen.capture_count == 3
        assert perception.observe_count == 0  # 从未识别
        assert len(executor.actions) == 0  # 从未执行


# ---------------------------------------------------------------------------
# 完整培养闭环
# ---------------------------------------------------------------------------


class TestCultivationLoop:
    def test_required_summons_then_stop_on_target_main_c(self):
        """完成最低召唤次数后，识别到目标主 C 并进入技能阶段。"""
        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.BUILD_MAIN_C
        run_cfg = RunConfig(minimum_summon_count_before_skills=2, target_star_level=2)
        task = CoopTask(
            ctx, run_cfg, CoopRole.HELPER, {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)}
        )

        # 用可变 perception 模拟棋盘变化
        class MutablePerception:
            def __init__(self):
                self.observe_count = 0
                self.observations = []

            def observe(self, c, f, hint, observation_mode=None, read_gold=False):
                obs = self.observations[min(self.observe_count, len(self.observations) - 1)]
                self.observe_count += 1
                return replace(obs, frame_id=c.frame_id)

            def observe_cultivation(self, screen, handle, ctx, n_frames=10, read_gold=False):
                obs = self.observations[min(self.observe_count, len(self.observations) - 1)]
                self.observe_count += 1
                return ctx, replace(obs, frame_id=ctx.frame_id)

        perception = MutablePerception()
        # 前两次识别不稳定也继续召唤；达到最低次数后用模型结果决策。
        perception.observations = [
            Observation(frame_id=1, board=None),  # 强制召唤 1
            Observation(frame_id=2, board=None),  # 强制召唤 2（完成）
            Observation(frame_id=3, board=make_board([make_hero("assault", star=2)])),  # 达到目标
        ]

        screen = FakeScreen(make_window_ctx())
        executor = FakeExecutor(verified=False)
        safety = SafetyGuard(max_failures=3, frame_ttl_ms=5000)

        # max_steps=4：2 次召唤 click + 1 次停止 wait + 1 次技能阶段 wait
        final = make_runner()._run_loop(screen, perception, executor, task, safety, 1, max_steps=4)
        assert final == State.SELECT_MAIN_C_SKILLS
        assert len(executor.actions) == 4
        assert executor.actions[0].tag == "required_summon"
        assert executor.actions[1].tag == "required_summon"
        assert executor.actions[2].kind == "wait"  # 达到目标星级停止
        assert executor.actions[3].kind == "wait"  # 技能阶段等待对局


def make_hero(hero_type: str = "assault", star: int = 1, pos=(100, 100)) -> BoardHero:
    return BoardHero(hero_type=hero_type, star_level=star, position=pos, confidence=0.9)


# ---------------------------------------------------------------------------
# 关闭击杀奖励弹窗重试
# ---------------------------------------------------------------------------


class _PopupPerception:
    """前 popup_frames 次识别返回带弹窗的 Observation，之后返回无弹窗画面。"""

    def __init__(self, popup_frames: int) -> None:
        from wlxq_bot.models import MatchResult

        self._popup_match = MatchResult("buttons/tan_chuang.png", (400, 900), 0.9)
        self._popup_frames = popup_frames
        self.observe_count = 0

    def observe(self, c, f, hint, observation_mode=None, read_gold=False):
        self.observe_count += 1
        raw = (
            {"tan_chuang_visible": True, "tan_chuang_match": self._popup_match}
            if self.observe_count <= self._popup_frames
            else {}
        )
        return replace(Observation(frame_id=c.frame_id, raw_data=raw), frame_id=c.frame_id)

    def observe_cultivation(self, screen, handle, ctx, n_frames=10, read_gold=False):
        obs = self.observe(ctx, None, State.BUILD_MAIN_C)
        return ctx, obs


def _make_popup_task():
    from wlxq_bot.config import RunConfig
    from wlxq_bot.models import CoopRole
    from wlxq_bot.tasks.base import TaskContext
    from wlxq_bot.tasks.coop import CoopTask

    ctx = TaskContext(main_c="assault", max_rounds=10)
    ctx.current_state = State.BUILD_MAIN_C
    return CoopTask(ctx, RunConfig(), CoopRole.HELPER, {})


class TestClosePopupRetry:
    def test_popup_retry_budget_exhausted_stops(self):
        """弹窗始终关不掉时重试 close_popup_max_retries 次后保守停止。

        自动重开会重置弹窗重试预算，因此本用例关闭自动重开以保持原语义；
        重开行为由 test_stale_decision_recovers_then_stops_at_budget 覆盖。
        """
        from wlxq_bot.config import RunConfig

        runner = Runner(
            default_config=DefaultConfig(
                run=RunConfig(
                    action_verify_frames=1,
                    close_popup_max_retries=4,
                    error_restart_enabled=False,
                )
            ),
            tasks_config=TasksConfig(),
        )
        # 一直返回弹窗画面：验证永远失败
        perception = _PopupPerception(popup_frames=10**6)
        executor = FakeExecutor(verified=False)
        task = _make_popup_task()

        final = runner._run_loop(
            FakeScreen(make_window_ctx()),
            perception,
            executor,
            task,
            SafetyGuard(max_failures=3, frame_ttl_ms=5000),
            1,
            max_steps=100,
        )

        popup_clicks = [a for a in executor.actions if a.tag == "close_popup"]
        # 首次点击 + 4 次重试
        assert len(popup_clicks) == 5
        assert final == State.BUILD_MAIN_C

    def test_popup_recovers_after_retry_and_flow_continues(self):
        """重试后弹窗消失：验证通过、重试计数重置，流程继续不中断。"""
        from wlxq_bot.config import Hotspot, RunConfig

        runner = Runner(
            default_config=DefaultConfig(
                run=RunConfig(action_verify_frames=1, close_popup_max_retries=4)
            ),
            tasks_config=TasksConfig(),
        )
        # 观察序列：弹窗 → 弹窗（首次验证失败，触发重试）→ 无弹窗（验证通过）→ 正常培养
        perception = _PopupPerception(popup_frames=2)
        executor = FakeExecutor(verified=False)
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.BUILD_MAIN_C
        task = CoopTask(
            ctx,
            RunConfig(minimum_summon_count_before_skills=2),
            CoopRole.HELPER,
            {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)},
        )

        final = runner._run_loop(
            FakeScreen(make_window_ctx()),
            perception,
            executor,
            task,
            SafetyGuard(max_failures=3, frame_ttl_ms=5000),
            1,
            max_steps=8,
        )

        popup_clicks = [a for a in executor.actions if a.tag == "close_popup"]
        # 首次点击生效慢一帧：验证失败触发重试清理 pending 后，下一帧弹窗已消失，
        # 任务未中断、直接继续培养流程
        assert len(popup_clicks) == 1
        # 弹窗关闭后流程继续：前置召唤动作正常执行
        assert any(a.tag == "required_summon" for a in executor.actions)
        assert final == State.BUILD_MAIN_C


# ---------------------------------------------------------------------------
# 合成验证失败重试
# ---------------------------------------------------------------------------


class _MergeBoardPerception:
    """按顺序返回预设棋盘 Observation 的假识别管线。"""

    def __init__(self, observations: list[Observation]) -> None:
        self._observations = observations
        self.observe_count = 0

    def _next(self, ctx):
        obs = self._observations[min(self.observe_count, len(self._observations) - 1)]
        self.observe_count += 1
        return replace(obs, frame_id=ctx.frame_id)

    def observe(self, c, f, hint, observation_mode=None, read_gold=False):
        return self._next(c)

    def observe_cultivation(self, screen, handle, ctx, n_frames=10, read_gold=False):
        return ctx, self._next(ctx)


def _merge_task() -> CoopTask:
    from wlxq_bot.config import Hotspot, RunConfig
    from wlxq_bot.models import CoopRole
    from wlxq_bot.tasks.base import TaskContext

    ctx = TaskContext(main_c="assault", max_rounds=10)
    ctx.current_state = State.BUILD_MAIN_C
    task = CoopTask(
        ctx,
        RunConfig(minimum_summon_count_before_skills=1),
        CoopRole.HELPER,
        {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)},
    )
    task._summon_count = 1
    return task


class TestMergeRetry:
    def test_merge_verify_failure_retries_and_recovers(self):
        """合成拖动验证失败后重新决策；棋盘变化后按新棋盘继续培养。"""
        from wlxq_bot.config import RunConfig

        runner = Runner(
            default_config=DefaultConfig(
                run=RunConfig(action_verify_frames=1, merge_max_retries=3)
            ),
            tasks_config=TasksConfig(),
        )
        pair_board = make_board(
            [
                make_hero("snow", star=1, pos=(684, 979)),
                make_hero("snow", star=1, pos=(684, 1201)),
            ]
        )
        # 决策合成（1 帧）→ 验证失败（棋盘没变）→ 重新决策再合成 → 验证成功
        # 从下往上拖：(1201) 是起点，合成结果 2 星落在终点格 (979)
        merged_board = make_board(
            [
                make_hero("snow", star=2, pos=(684, 979)),
            ]
        )
        perception = _MergeBoardPerception(
            [
                Observation(frame_id=1, board=pair_board),
                Observation(frame_id=2, board=pair_board),
                Observation(frame_id=3, board=pair_board),
                Observation(frame_id=4, board=merged_board),
                Observation(frame_id=5, board=merged_board),
            ]
        )
        executor = FakeExecutor(verified=False)
        task = _merge_task()

        final = runner._run_loop(
            FakeScreen(make_window_ctx()),
            perception,
            executor,
            task,
            SafetyGuard(max_failures=3, frame_ttl_ms=5000),
            1,
            max_steps=10,
        )

        merge_drags = [a for a in executor.actions if a.tag == "merge_heroes"]
        # 首次拖动验证失败后重试了一次
        assert len(merge_drags) == 2
        # 合成成功后按新棋盘继续（无合法对 → 召唤）
        assert any(a.tag == "summon_hero" for a in executor.actions)
        assert final == State.BUILD_MAIN_C

    def test_summon_verify_failure_waits_and_does_not_end_task(self):
        """召唤后棋盘始终不变（金币不足）：进入冷却等待，任务不结束。"""
        from wlxq_bot.config import RunConfig

        runner = Runner(
            default_config=DefaultConfig(run=RunConfig(action_verify_frames=1)),
            tasks_config=TasksConfig(),
        )
        # 两个不同类型英雄：无合法对 → 召唤
        board = make_board(
            [
                make_hero("assault", star=1, pos=(684, 979)),
                make_hero("angel", star=1, pos=(684, 1201)),
            ]
        )
        perception = _MergeBoardPerception([Observation(frame_id=1, board=board)])
        executor = FakeExecutor(verified=False)
        task = _merge_task()
        task._clock = lambda: 0.0  # 冻结时间：冷却不结束

        final = runner._run_loop(
            FakeScreen(make_window_ctx()),
            perception,
            executor,
            task,
            SafetyGuard(max_failures=3, frame_ttl_ms=5000),
            1,
            max_steps=6,
        )

        summons = [a for a in executor.actions if a.tag == "summon_hero"]
        # 只点了一次召唤，验证失败后转入冷却等待（而非结束任务）
        assert len(summons) == 1
        assert any(a.tag == "summon_retry_wait" for a in executor.actions)
        assert final == State.BUILD_MAIN_C

    def test_merge_retry_budget_exhausted_falls_back_to_summon(self):
        """合成始终无效：重试耗尽后不结束任务，改召唤新英雄改变棋盘。"""
        from wlxq_bot.config import RunConfig

        runner = Runner(
            default_config=DefaultConfig(
                run=RunConfig(action_verify_frames=1, merge_max_retries=2)
            ),
            tasks_config=TasksConfig(),
        )
        pair_board = make_board(
            [
                make_hero("snow", star=1, pos=(684, 979)),
                make_hero("snow", star=1, pos=(684, 1201)),
            ]
        )
        perception = _MergeBoardPerception([Observation(frame_id=1, board=pair_board)])
        executor = FakeExecutor(verified=False)
        task = _merge_task()

        final = runner._run_loop(
            FakeScreen(make_window_ctx()),
            perception,
            executor,
            task,
            SafetyGuard(max_failures=3, frame_ttl_ms=5000),
            1,
            max_steps=100,
        )

        merge_drags = [a for a in executor.actions if a.tag == "merge_heroes"]
        # 首次拖动 + 2 次重试后放弃该合成对（不再有第 4 次拖动）
        assert len(merge_drags) == 3
        # 放弃后召唤新英雄（棋盘不变 → 召唤验证失败 → 保守停止，但不是合成失败直接结束）
        assert any(a.tag == "summon_hero" for a in executor.actions)
        # 失败对已记录（签名为拖动方向：下格在前），本轮内不再拖同一对
        assert task._failed_merge_pairs == {((684, 1201), (684, 979))}
        assert final == State.BUILD_MAIN_C


# ---------------------------------------------------------------------------
# 技能点击验证失败不结束任务
# ---------------------------------------------------------------------------


class _OpeningSkillPagePerception:
    """始终返回开局技能页面的假识别管线：页面卡住，点击始终无效。"""

    def __init__(self) -> None:
        self.observe_count = 0

    def observe(self, c, f, hint, observation_mode=None, read_gold=False):
        self.observe_count += 1
        from wlxq_bot.models import SkillCandidate

        return Observation(
            frame_id=c.frame_id,
            raw_data={"select_skill_button_visible": True},
            skill_candidates=[SkillCandidate("assault", (100, 100), 0.9)],
        )

    def observe_cultivation(self, screen, handle, ctx, n_frames=10, read_gold=False):
        return ctx, self.observe(ctx, None, None)


class TestSkillClickFailureRecovers:
    def test_opening_skill_click_failure_does_not_end_task(self):
        """技能卡点击始终无效：按预算重试后放弃本次选择并继续等待，任务不中止。"""
        from wlxq_bot.config import RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        runner = Runner(
            default_config=DefaultConfig(
                run=RunConfig(action_verify_frames=1, skill_click_max_retries=2)
            ),
            tasks_config=TasksConfig(),
        )
        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.SELECT_OPENING_SKILLS
        task = CoopTask(ctx, RunConfig(), CoopRole.HELPER, {})
        task._opening_loaded = True
        executor = FakeExecutor(verified=False)

        final = runner._run_loop(
            FakeScreen(make_window_ctx()),
            _OpeningSkillPagePerception(),
            executor,
            task,
            SafetyGuard(max_failures=3, frame_ttl_ms=5000),
            1,
            max_steps=30,
        )

        skill_clicks = [a for a in executor.actions if a.tag == "opening_skill_candidate"]
        # 首次点击 + 2 次重试后放弃本次选择
        assert len(skill_clicks) == 3
        # 放弃后循环仍在继续（等待对局推进），而不是因验证失败结束任务
        assert any(a.tag == "opening_skill_blocked_wait" for a in executor.actions)
        assert task._opening_clicks_blocked is True
        assert final == State.SELECT_OPENING_SKILLS


# ---------------------------------------------------------------------------
# 退出帧落盘（非正常退出排查）
# ---------------------------------------------------------------------------


class TestExitFrameSaving:
    def _forced_summon_task(self):
        """技能解锁前强制召唤的任务：循环只截图+点击，直到步数上限退出。"""
        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        ctx = TaskContext(main_c="assault", max_rounds=10)
        ctx.current_state = State.BUILD_MAIN_C
        return CoopTask(
            ctx,
            RunConfig(minimum_summon_count_before_skills=100),
            CoopRole.HELPER,
            {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)},
        )

    def test_abnormal_exit_saves_last_frames(self, tmp_path):
        """非正常退出：缓冲内最近 N 帧按序落盘并带 manifest，重复 flush 不重复保存。"""
        import numpy as np

        from wlxq_bot.debug.recorder import DebugRecorder

        recorder = DebugRecorder(str(tmp_path), exit_frame_buffer_size=3)
        task = self._forced_summon_task()
        screen = FakeScreen(make_window_ctx(), frame=np.zeros((4, 4, 3), dtype=np.uint8))
        safety = SafetyGuard(max_failures=3, frame_ttl_ms=5000)

        final = make_runner()._run_loop(
            screen,
            FakePerception(Observation(frame_id=1, board=None)),
            FakeExecutor(),
            task,
            safety,
            1,
            max_steps=5,
            debug_recorder=recorder,
        )
        Runner._flush_exit_frames(recorder, task, safety)

        assert final == State.BUILD_MAIN_C
        assert screen.capture_count == 5  # 缓冲上限 3：只保留最后 3 帧
        folders = list(tmp_path.glob("exit_*"))
        assert len(folders) == 1
        assert folders[0].name.startswith("exit_build_main_c_")
        assert [p.name for p in sorted(folders[0].glob("*.png"))] == [
            "00_frame_3.png",
            "01_frame_4.png",
            "02_frame_5.png",
        ]
        manifest = (folders[0] / "frames.txt").read_text(encoding="utf-8")
        assert manifest.count("\n") == 3
        assert "frame_id=5" in manifest
        # 缓冲已清空：重复 flush 不再产生新文件夹
        Runner._flush_exit_frames(recorder, task, safety)
        assert len(list(tmp_path.glob("exit_*"))) == 1

    def test_completed_exit_saves_nothing(self, tmp_path):
        """正常完成（COMPLETED）退出不落盘退出帧。"""
        import numpy as np

        from wlxq_bot.config import Hotspot, RunConfig
        from wlxq_bot.debug.recorder import DebugRecorder
        from wlxq_bot.models import CoopRole
        from wlxq_bot.tasks.base import TaskContext
        from wlxq_bot.tasks.coop import CoopTask

        recorder = DebugRecorder(str(tmp_path), exit_frame_buffer_size=3)
        ctx = TaskContext(main_c="assault", max_rounds=1)
        ctx.round_count = 1
        ctx.current_state = State.CHECK_ROUND_LIMIT
        task = CoopTask(
            ctx,
            RunConfig(max_rounds=1, minimum_summon_count_before_skills=2),
            CoopRole.HELPER,
            {"add_hero": Hotspot(x_ratio=0.7, y_ratio=0.96)},
        )
        safety = SafetyGuard(max_failures=3, frame_ttl_ms=5000)

        final = make_runner()._run_loop(
            FakeScreen(make_window_ctx(), frame=np.zeros((4, 4, 3), dtype=np.uint8)),
            FakePerception(Observation(frame_id=1)),
            FakeExecutor(),
            task,
            safety,
            1,
            max_steps=10,
            debug_recorder=recorder,
        )
        Runner._flush_exit_frames(recorder, task, safety)

        assert final == State.COMPLETED
        assert list(tmp_path.glob("exit_*")) == []

    def test_user_esc_stop_saves_nothing(self, tmp_path):
        """Esc 停止（stop_requested=True）：用户主动退出，不保存退出帧。"""
        import numpy as np

        from wlxq_bot.debug.recorder import DebugRecorder

        recorder = DebugRecorder(str(tmp_path), exit_frame_buffer_size=3)
        recorder.keep_exit_frame(1, 0.0, np.zeros((4, 4, 3), dtype=np.uint8))
        task = self._forced_summon_task()
        safety = SafetyGuard(max_failures=3, frame_ttl_ms=5000)
        safety.request_stop()

        Runner._flush_exit_frames(recorder, task, safety)

        assert list(tmp_path.glob("exit_*")) == []

    def test_flush_without_recorder_or_frames_is_noop(self, tmp_path):
        """无 recorder 或缓冲为空时 flush 直接返回，不报错不落盘。"""
        from wlxq_bot.debug.recorder import DebugRecorder

        task = self._forced_summon_task()
        safety = SafetyGuard(max_failures=3, frame_ttl_ms=5000)
        Runner._flush_exit_frames(None, task, safety)

        recorder = DebugRecorder(str(tmp_path))  # 默认不启用缓冲
        Runner._flush_exit_frames(recorder, task, safety)
        assert list(tmp_path.glob("exit_*")) == []
