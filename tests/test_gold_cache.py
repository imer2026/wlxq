"""GoldCache 后台金币缓存单元测试（假读取器 + 注入时钟 + 真线程）。

覆盖 2026-09-26 重构的门控语义：单槽信箱 latest-wins、脏帧跳过、
解析失败不提交、间隔/事件补读节流、超龄视同缺失放行、连续失败退避。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from wlxq_bot.models import State
from wlxq_bot.perception.gold_cache import GoldCache
from wlxq_bot.perception.ocr import clean_gold_text

_ROI = (1, 2, 3, 4)


class FakeClock:
    """受控单调时钟：线程与测试共享同一时间线，手动推进。"""

    def __init__(self) -> None:
        self.now = 100.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds


@dataclass
class FakePairReader:
    """假数值读取器：按调用序返回预设 (current, cost)；可注入阻塞事件。"""

    results: list[tuple[int | None, int | None]] = field(default_factory=list)
    delay_event: threading.Event | None = None
    release_event: threading.Event | None = None
    frames: list[Any] = field(default_factory=list)
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def read_pair(self, frame, current_roi, cost_roi):
        with self._lock:
            self.calls += 1
            call_index = self.calls
            self.frames.append(frame)
        if self.delay_event is not None:
            self.delay_event.set()
        if self.release_event is not None:
            self.release_event.wait(timeout=2.0)
        if not self.results:
            return None, None
        idx = min(call_index - 1, len(self.results) - 1)
        return self.results[idx]


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _make_cache(
    reader,
    *,
    refresh: float = 2.0,
    stale: float = 30.0,
    read_min: float = 1.0,
    clock: FakeClock | None = None,
) -> GoldCache:
    return GoldCache(
        reader=reader,
        read_min_interval_seconds=read_min,
        refresh_interval_seconds=refresh,
        stale_after_seconds=stale,
        clock=clock or FakeClock(),
        poll_seconds=0.05,
    )


def _submit(cache: GoldCache, frame, *, clean: bool = True, cost_key: str = "summon_cost") -> None:
    cache.submit(
        frame,
        frame_id=1,
        clean=clean,
        current_roi=_ROI,
        cost_roi=_ROI,
        cost_key=cost_key,
    )


class TestGoldCache:
    def test_clean_frame_commits_snapshot(self):
        clock, reader = FakeClock(), FakePairReader(results=[(100, 80)])
        cache = _make_cache(reader, clock=clock)
        _submit(cache, object())
        assert _wait_until(lambda: cache.snapshot() is not None)
        assert cache.snapshot() == {"current": 100, "summon_cost": 80}

    def test_dirty_frame_skips_read(self):
        reader = FakePairReader(results=[(100, 80)])
        cache = _make_cache(reader, clock=FakeClock())
        _submit(cache, object(), clean=False)
        time.sleep(0.3)
        assert reader.calls == 0
        assert cache.snapshot() is None

    def test_interval_throttles_reads(self):
        clock, reader = FakeClock(), FakePairReader(results=[(100, 80), (120, 80)])
        cache = _make_cache(reader, clock=clock, refresh=2.0)
        _submit(cache, object())
        assert _wait_until(lambda: reader.calls == 1)
        # 间隔内的新帧：不触发第二次读取
        clock.advance(1.0)
        _submit(cache, object())
        time.sleep(0.3)
        assert reader.calls == 1
        # 超过间隔：读取
        clock.advance(1.5)
        _submit(cache, object())
        assert _wait_until(lambda: reader.calls == 2)
        assert _wait_until(lambda: cache.snapshot()["current"] == 120)

    def test_invalidate_reads_immediately(self):
        clock, reader = FakeClock(), FakePairReader(results=[(100, 80), (150, 80)])
        cache = _make_cache(reader, clock=clock, refresh=2.0, read_min=0.0)
        _submit(cache, object())
        assert _wait_until(lambda: reader.calls == 1)
        # 花钱失效：不到常规间隔也立即补读
        cache.invalidate()
        clock.advance(0.5)
        _submit(cache, object())
        assert _wait_until(lambda: cache.snapshot()["current"] == 150)

    def test_parse_failure_keeps_previous_cache(self):
        clock = FakeClock()
        reader = FakePairReader(results=[(100, 80), (None, None)])
        cache = _make_cache(reader, clock=clock)
        _submit(cache, object())
        assert _wait_until(lambda: cache.snapshot() is not None)
        clock.advance(5.0)
        _submit(cache, object())
        assert _wait_until(lambda: reader.calls == 2)
        # 读取失败不动缓存：快照保持上一个好值
        assert cache.snapshot() == {"current": 100, "summon_cost": 80}

    def test_stale_snapshot_treated_as_missing(self):
        clock, reader = FakeClock(), FakePairReader(results=[(100, 80)])
        cache = _make_cache(reader, clock=clock, stale=30.0)
        _submit(cache, object())
        assert _wait_until(lambda: cache.snapshot() is not None)
        clock.advance(31.0)
        assert cache.snapshot() is None  # 超龄 → 视同缺失 → 门控放行

    def test_clear_resets_and_forces_refresh(self):
        clock = FakeClock()
        reader = FakePairReader(results=[(100, 80), (200, 80)])
        cache = _make_cache(reader, clock=clock, refresh=2.0, read_min=0.0)
        _submit(cache, object())
        assert _wait_until(lambda: cache.snapshot() is not None)
        cache.clear()
        assert cache.snapshot() is None
        _submit(cache, object())
        assert _wait_until(lambda: cache.snapshot() == {"current": 200, "summon_cost": 80})

    def test_merge_across_cost_keys(self):
        clock = FakeClock()
        reader = FakePairReader(results=[(100, 80), (150, None), (None, 120)])
        cache = _make_cache(reader, clock=clock, read_min=0.0)
        _submit(cache, object(), cost_key="summon_cost")
        assert _wait_until(lambda: cache.snapshot() is not None)
        clock.advance(3.0)
        _submit(cache, object(), cost_key="skill_cost")
        assert _wait_until(lambda: cache.snapshot().get("current") == 150)
        clock.advance(3.0)
        _submit(cache, object(), cost_key="skill_cost")
        assert _wait_until(
            lambda: cache.snapshot().get("skill_cost") == 120
        )
        # 召唤费用字段保留上次读数
        assert cache.snapshot()["summon_cost"] == 80

    def test_consecutive_failures_back_off(self):
        clock = FakeClock()
        reader = FakePairReader(results=[(None, None)])
        cache = _make_cache(reader, clock=clock, refresh=2.0, read_min=0.0)
        _submit(cache, object())
        assert _wait_until(lambda: reader.calls == 1)
        # 失败 1 次 → 间隔翻倍为 4s：3s 后的重试不发生
        clock.advance(3.0)
        _submit(cache, object())
        time.sleep(0.3)
        assert reader.calls == 1
        clock.advance(1.5)
        _submit(cache, object())
        assert _wait_until(lambda: reader.calls == 2)
        # 失败 2 次 → 间隔 8s：4.5s 后的重试不发生，8s 后发生
        clock.advance(4.5)
        _submit(cache, object())
        time.sleep(0.3)
        assert reader.calls == 2
        clock.advance(4.0)
        _submit(cache, object())
        assert _wait_until(lambda: reader.calls == 3)

    def test_latest_frame_wins_when_reader_lags(self):
        start = threading.Event()
        release = threading.Event()
        reader = FakePairReader(results=[(100, 80), (200, 80)])
        reader.delay_event = start
        reader.release_event = release
        clock = FakeClock()
        cache = _make_cache(reader, clock=clock, read_min=0.0)
        frame1, frame2 = object(), object()
        _submit(cache, frame1)
        assert _wait_until(lambda: start.is_set())  # 读者开始处理第 1 帧
        _submit(cache, frame2)  # 读取期间的新帧覆盖信箱
        clock.advance(3.0)  # 让第 2 帧到期（超过刷新间隔）
        release.set()
        assert _wait_until(lambda: reader.calls == 2)
        assert reader.frames[-1] is frame2  # 处理的是最新帧，不是积压的旧帧

    def test_stop_is_idempotent(self):
        cache = _make_cache(FakePairReader(), clock=FakeClock())
        cache.stop()
        cache.stop()


class TestCleanText:
    def test_strips_noise_characters(self):
        assert clean_gold_text("?80") == "80"
        assert clean_gold_text("1 4.7K") == "14.7K"
        assert clean_gold_text("100") == "100"

    def test_all_noise_becomes_empty(self):
        assert clean_gold_text("???") == ""


class TestGateSemantics:
    def test_numeric_compare_via_existing_can_afford(self):
        """缓存快照 dict 直接喂现有 _can_afford：数值比较，缺键放行。"""
        from wlxq_bot.tasks.coop import CoopTask

        assert CoopTask._can_afford({"current": 100, "summon_cost": 80}, "summon")
        assert not CoopTask._can_afford({"current": 50, "summon_cost": 80}, "summon")
        # 缺失（超龄返回 None 后任务层 gold=None 不进门控）；键缺 likewise
        assert CoopTask._can_afford({"current": 100}, "summon")
        assert CoopTask._can_afford({}, "skill")


class TestRunnerSyncHook:
    def test_spend_tag_invalidates_and_round_end_clears(self):
        class Recorder:
            def __init__(self) -> None:
                self.invalidated = 0
                self.cleared = 0

            def invalidate_gold_cache(self) -> None:
                self.invalidated += 1

            def clear_gold_cache(self) -> None:
                self.cleared += 1

        from wlxq_bot.models import State
        from wlxq_bot.runner import Runner

        rec = Recorder()
        Runner._sync_gold_cache(rec, "summon_hero", State.BUILD_MAIN_C)
        assert rec.invalidated == 1
        assert rec.cleared == 0
        Runner._sync_gold_cache(rec, "main_skill_priority", State.FIND_COOP)
        assert rec.invalidated == 2
        assert rec.cleared == 1

    def test_objects_without_cache_methods_are_skipped(self):
        from wlxq_bot.models import State
        from wlxq_bot.runner import Runner

        Runner._sync_gold_cache(object(), "summon_hero", State.BUILD_MAIN_C)  # 不抛异常

class TestSubmitCleanRule:
    """_submit_gold_frame 干净帧判定回归（2026-09-26 04:27 冻结事故）。

    事故：SELECT_MAIN_C_SKILLS 曾要求 select_skill_button（选技能页标题带
    模板，页面开着才可见）可见才算干净——页面关闭（金币不足等待期）反而
    全部判脏，后台停读，缓存冻结在旧余额（1200）直到 30s 超龄。
    """

    def _make_perception_with_cache(self, tmp_path):
        from wlxq_bot.assets import TemplatePack
        from wlxq_bot.config import RoiConfig, TasksConfig
        from wlxq_bot.models import CoopRole
        from wlxq_bot.perception.coop import CoopPerception

        submitted: list[dict] = []

        class FakeCache:
            def submit(self, frame, **kwargs):
                submitted.append(kwargs)

        pack_root = tmp_path / "3000x2000"
        pack_root.mkdir(parents=True, exist_ok=True)
        cfg = TasksConfig()
        cfg.rois["gold_current_area"] = RoiConfig(
            x_ratio=0.28, y_ratio=0.84, width_ratio=0.10, height_ratio=0.02
        )
        cfg.rois["summon_cost_area"] = RoiConfig(
            x_ratio=0.58, y_ratio=0.97, width_ratio=0.12, height_ratio=0.02
        )
        cfg.rois["skill_cost_area"] = RoiConfig(
            x_ratio=0.26, y_ratio=0.97, width_ratio=0.13, height_ratio=0.02
        )
        perception = CoopPerception(
            vision=None,
            template_pack=TemplatePack(client_size=(927, 1727), root=pack_root),
            tasks_config=cfg,
            role=CoopRole.INITIATOR,
            main_c="monkey",
            gold_cache=FakeCache(),
        )
        return perception, submitted

    def _window_ctx(self):
        from wlxq_bot.models import WindowContext

        return WindowContext(
            window_handle=1,
            client_rect_screen=(0, 0, 927, 1727),
            client_size=(927, 1727),
            dpi=96,
            monitor_id="primary",
            is_foreground=True,
            is_minimized=False,
            captured_at=0.0,
            frame_id=1,
        )

    def _frame(self):
        import numpy as np

        return np.zeros((1727, 927, 3), dtype=np.uint8)

    def test_select_state_page_closed_frame_is_clean(self, tmp_path):
        """页面关闭（select_skill 不可见）且无遮挡 → 干净帧，正常提交。"""
        perception, submitted = self._make_perception_with_cache(tmp_path)
        perception._submit_gold_frame(
            self._window_ctx(),
            self._frame(),
            State.SELECT_MAIN_C_SKILLS,
            {"select_skill_button_visible": False},
        )
        assert len(submitted) == 1
        assert submitted[0]["clean"] is True
        assert submitted[0]["cost_key"] == "skill_cost"

    def test_occlusion_flags_make_frame_dirty(self, tmp_path):
        perception, submitted = self._make_perception_with_cache(tmp_path)
        ctx = self._window_ctx()
        for flag in (
            "tan_chuang_visible",
            "zeng_song_ji_neng_visible",
            "merge_gift_skill_page_visible",
        ):
            submitted.clear()
            perception._submit_gold_frame(
                ctx, self._frame(), State.SELECT_MAIN_C_SKILLS, {flag: True}
            )
            # 脏帧仍进信箱（覆盖旧记录），由服务端按 clean=False 跳过读取
            assert len(submitted) == 1, f"{flag} 可见时应提交但标脏"
            assert submitted[0]["clean"] is False, f"{flag} 可见时应判脏帧"

    def test_build_state_submits_with_summon_cost_key(self, tmp_path):
        perception, submitted = self._make_perception_with_cache(tmp_path)
        perception._submit_gold_frame(
            self._window_ctx(), self._frame(), State.BUILD_MAIN_C, {}
        )
        assert len(submitted) == 1
        assert submitted[0]["cost_key"] == "summon_cost"
        assert submitted[0]["clean"] is True

    def test_non_gold_state_not_submitted(self, tmp_path):
        perception, submitted = self._make_perception_with_cache(tmp_path)
        perception._submit_gold_frame(
            self._window_ctx(), self._frame(), State.ENTER_MATCH, {}
        )
        assert submitted == []
