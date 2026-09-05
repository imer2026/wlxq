"""技能标题 OCR 主路径测试：感知层产出档位选项 + 任务层按档位选卡。"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

from test_coop_task import make_task
from wlxq_bot.assets import TemplatePack
from wlxq_bot.config import RoiConfig, TasksConfig
from wlxq_bot.models import Action, CoopRole, Observation, State, WindowContext
from wlxq_bot.perception.coop import CoopPerception
from wlxq_bot.perception.vision import Vision
from wlxq_bot.skill_catalog import compute_skill_tiers, record_new_skill
from wlxq_bot.tasks.coop import CoopTask


class FakeTitleReader:
    """假标题识别器：按调用顺序返回预设响应（每次 read 消费一条）。

    逐列裁剪后每列调用一次 read；未收录技能还会对整卡补一次描述 OCR。
    """

    def __init__(self, responses: list[list[tuple[str, tuple[int, int]]]]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def read(self, image: np.ndarray) -> list[tuple[str, tuple[int, int]]]:
        self.calls += 1
        if not self._responses:
            return []
        return self._responses.pop(0)


def make_frame(width: int = 300, height: int = 600) -> np.ndarray:
    return np.full((height, width, 3), 200, dtype=np.uint8)


def make_ctx(frame_id: int) -> WindowContext:
    return WindowContext(
        window_handle=0,
        client_rect_screen=(0, 0, 300, 600),
        client_size=(300, 600),
        dpi=1.0,
        monitor_id=0,
        frame_id=frame_id,
        captured_at=0.0,
        is_minimized=False,
        is_foreground=True,
    )


def make_perception(
    tmp_path: Path,
    title_reader: FakeTitleReader | None,
    skill_tiers: dict[str, int] | None,
) -> CoopPerception:
    tasks_cfg = TasksConfig(
        rois={
            "skill_candidates": RoiConfig(
                relative_to="client",
                x_ratio=0.0,
                y_ratio=0.0,
                width_ratio=1.0,
                height_ratio=1.0,
            )
        }
    )
    pack = TemplatePack(client_size=(300, 600), root=tmp_path / "pack")
    return CoopPerception(
        Vision(),  # OCR 主路径在图标匹配之前返回,Vision 不会被调用
        pack,
        tasks_cfg,
        CoopRole.HELPER,
        "assault",
        title_reader=title_reader,
        skill_tiers=skill_tiers,
        new_skill_dir=tmp_path / "skill_cards",
    )


SKILL_TIERS = {"天穹支援": 1, "冰雪凤鸣": 11, "血量传承": 11}


class TestPerceptionTitleOcr:
    def test_options_produced_with_tiers(self, tmp_path: Path) -> None:
        """逐列 OCR 读出三张卡标题 → skill_card_options 带列号/档位/点击位置。"""
        reader = FakeTitleReader(
            [
                [("天穹支援", (20, 10))],   # 第 0 列标题
                [("冰雪凤鸣", (20, 10))],   # 第 1 列标题
                [("血量传承", (20, 10))],   # 第 2 列标题
            ]
        )
        perception = make_perception(tmp_path, reader, SKILL_TIERS)
        frame = make_frame()
        raw_data: dict = {"select_skill_button_visible": True}

        result = perception._detect_skill_candidates(
            make_ctx(7), frame, [], raw_data, page="SELECT_MAIN_C_SKILLS"
        )

        assert result == []  # OCR 主路径不走图标识别
        options = raw_data["skill_card_options"]
        assert [o["column"] for o in options] == [0, 1, 2]
        assert [o["name"] for o in options] == ["天穹支援", "冰雪凤鸣", "血量传承"]
        assert [o["tier"] for o in options] == [1, 11, 11]
        assert all(o["known"] for o in options)
        # 点击位置落在各列中心
        assert options[0]["position"][0] == 50
        assert options[2]["position"][0] == 250

    def test_unknown_title_recorded_and_tier11(self, tmp_path: Path) -> None:
        """标题不在清单 → 第 11 档,整卡补描述并记录到 new_skills.yaml。"""
        reader = FakeTitleReader(
            [
                [("新技能甲", (20, 10))],  # 第 0 列标题
                [],                        # 第 0 列整卡描述 OCR（纯色图无文字）
                [],                        # 第 1 列标题（无文字）
                [],                        # 第 2 列标题（无文字）
            ]
        )
        perception = make_perception(tmp_path, reader, dict(SKILL_TIERS))
        frame = make_frame()
        raw_data: dict = {"select_skill_button_visible": True}

        perception._detect_skill_candidates(
            make_ctx(1), frame, [], raw_data, page="X"
        )

        options = raw_data["skill_card_options"]
        assert len(options) == 1
        assert options[0]["name"] == "新技能甲"
        assert options[0]["tier"] == 11
        assert options[0]["known"] is False
        assert options[0]["description"] == ""  # 纯色图 OCR 不到描述,允许为空
        new_file = tmp_path / "skill_cards" / "new_skills.yaml"
        data = yaml.safe_load(new_file.read_text(encoding="utf-8"))
        assert data["new_skills"][0]["name"] == "新技能甲"

    def test_fuzzy_variant_matches_catalog(self, tmp_path: Path) -> None:
        """OCR 误写(圣灵底护)模糊命中清单(圣灵庇护) → 已知技能,不记录。"""
        reader = FakeTitleReader([[("圣灵底护", (20, 10))]])
        perception = make_perception(tmp_path, reader, {"圣灵庇护": 5})
        frame = make_frame()
        raw_data: dict = {"select_skill_button_visible": True}

        perception._detect_skill_candidates(make_ctx(1), frame, [], raw_data, page="X")

        options = raw_data["skill_card_options"]
        assert options[0]["tier"] == 5
        assert options[0]["known"] is True
        assert not (tmp_path / "skill_cards" / "new_skills.yaml").exists()

    def test_no_reader_falls_back_to_icons(self, tmp_path: Path) -> None:
        """未配置识别器时不产 options,走图标兜底(此处仅验证不产 options)。"""
        perception = make_perception(tmp_path, None, dict(SKILL_TIERS))
        frame = make_frame()
        raw_data: dict = {"select_skill_button_visible": True}

        result = perception._detect_skill_candidates(
            make_ctx(1), frame, [], raw_data, page="X"
        )

        assert result == []
        assert "skill_card_options" not in raw_data

    def test_roi_none_skipped(self, tmp_path: Path) -> None:
        """ROI 未标定时跳过 OCR 主路径。"""
        tasks_cfg = TasksConfig()  # 无 skill_candidates ROI
        perception = CoopPerception(
            Vision(),
            TemplatePack(client_size=(300, 600), root=tmp_path / "pack"),
            tasks_cfg,
            CoopRole.HELPER,
            "assault",
            title_reader=FakeTitleReader([("天穹支援", 50)]),
            skill_tiers=dict(SKILL_TIERS),
        )
        frame = make_frame()
        raw_data: dict = {"select_skill_button_visible": True}

        perception._detect_skill_candidates(
            make_ctx(1), frame, [], raw_data, page="X"
        )

        assert "skill_card_options" not in raw_data


class TestComputeSkillTiers:
    def test_tier8_and_11_auto_computed(self) -> None:
        tiers = compute_skill_tiers(
            {1: ["天穹支援"], 9: ["圣羽加持"]},
            {
                "天穹支援": "强袭",
                "粒子增幅": "强袭",
                "分裂强化": "强袭",
                "圣羽加持": "天使",
                "陨星刻印": "天使",
                "职业控制": "雪姬",
            },
            "强袭",
        )
        # 主C未覆盖技能 → 第 8 档;其余 → 第 11 档
        assert tiers["天穹支援"] == 1
        assert tiers["圣羽加持"] == 9
        assert tiers["粒子增幅"] == 8
        assert tiers["分裂强化"] == 8
        assert tiers["圣羽加持"] != 8
        assert tiers["陨星刻印"] == 11
        assert tiers["职业控制"] == 11

    def test_configured_tier_8_11_ignored(self) -> None:
        tiers = compute_skill_tiers(
            {8: ["某技能"], 11: ["另一技能"], 1: ["天穹支援"]},
            {"天穹支援": "强袭", "某技能": "强袭", "另一技能": "死骑"},
            "强袭",
        )
        # 第 8/11 档自动计算:某技能是强袭 → 8;另一技能非强袭 → 11
        assert tiers["某技能"] == 8
        assert tiers["另一技能"] == 11


class TestRecordNewSkill:
    def test_record_and_fuzzy_dedup(self, tmp_path: Path) -> None:
        assert record_new_skill(tmp_path, "新技能甲", "描述A") is True
        # 同名/模糊相似不重复记录
        assert record_new_skill(tmp_path, "新技能甲", "描述A") is False
        data = yaml.safe_load((tmp_path / "new_skills.yaml").read_text(encoding="utf-8"))
        assert len(data["new_skills"]) == 1
        assert data["new_skills"][0]["description"] == "描述A"

    def test_distinct_skills_recorded(self, tmp_path: Path) -> None:
        assert record_new_skill(tmp_path, "冰核爆炸", "爆炸") is True
        assert record_new_skill(tmp_path, "天穹支援", "大招") is True
        data = yaml.safe_load((tmp_path / "new_skills.yaml").read_text(encoding="utf-8"))
        assert len(data["new_skills"]) == 2


class TestSkillPriorityAction:
    def _observation(self, options: list[dict]) -> Observation:
        return Observation(
            frame_id=1,
            source_frame_ids=(1,),
            raw_data={"skill_card_options": options},
        )

    def _fake_task(self) -> CoopTask:
        """只为调用 _skill_priority_action 的最小假任务(self 替身)。"""

        class _FakeTask:
            _rng = random.Random(42)

        return _FakeTask()  # type: ignore[return-value]

    def test_picks_min_tier_randomly(self) -> None:
        options = [
            {"name": "天穹支援", "tier": 1, "position": (50, 300)},
            {"name": "圣羽增持", "tier": 11, "position": (150, 300)},
            {"name": "圣羽加持", "tier": 11, "position": (250, 300)},
        ]
        action, state = CoopTask._skill_priority_action(
            self._fake_task(),  # type: ignore[arg-type]
            self._observation(options),
            "main_skill_priority",
            State.SELECT_MAIN_C_SKILLS,
        )
        assert state == State.SELECT_MAIN_C_SKILLS
        assert action.kind == "click"
        assert action.target == (50, 300)  # 唯一的最小档(第1档)
        assert "第 1 档" in action.reason

    def test_same_tier_random_pool(self) -> None:
        options = [
            {"name": "甲", "tier": 8, "position": (50, 300)},
            {"name": "乙", "tier": 8, "position": (150, 300)},
        ]
        action, _ = CoopTask._skill_priority_action(
            self._fake_task(),  # type: ignore[arg-type]
            self._observation(options),
            "tag",
            State.SELECT_MAIN_C_SKILLS,
        )
        assert action.target in [(50, 300), (150, 300)]  # 同档随机
        assert "第 8 档" in action.reason

    def test_no_options_returns_none(self) -> None:
        result = CoopTask._skill_priority_action(
            self._fake_task(),  # type: ignore[arg-type]
            Observation(frame_id=1, source_frame_ids=(1,)),
            "tag",
            State.SELECT_MAIN_C_SKILLS,
        )
        assert result is None


class TestVerifyActionPriorityTag:
    def test_verify_action_accepts_priority_tag(self) -> None:
        """回归：opening_skill_priority 等新 tag 必须进入验证白名单。

        实机 2026-09-05：新 tag 未注册进 verify_action 的技能点击集合，
        验证走默认保守失败，点击成功也连续 5 帧失败导致保守停止。
        """
        task = make_task()
        before = Observation(
            frame_id=1,
            source_frame_ids=(1,),
            raw_data={"select_skill_button_visible": True},
        )
        after = Observation(
            frame_id=2,
            source_frame_ids=(2,),
            raw_data={"select_skill_button_visible": False},
        )
        for tag in (
            "opening_skill_priority",
            "main_skill_priority",
            "merge_gift_skill_priority",
        ):
            action = Action(
                kind="click",
                target=(150, 300),
                duration=0.08,
                verification="next_frame",
                tag=tag,
            )
            assert task.verify_action(action, before, after) is True, tag

    def test_verify_action_accepts_close_bonus_popup(self) -> None:
        """赠送技能卡片关闭动作：标识消失即验证通过。"""
        task = make_task()
        before = Observation(
            frame_id=1,
            source_frame_ids=(1,),
            raw_data={"zeng_song_ji_neng_visible": True},
        )
        after = Observation(
            frame_id=2,
            source_frame_ids=(2,),
            raw_data={"zeng_song_ji_neng_visible": False},
        )
        action = Action(
            kind="click",
            target=(351, 502),
            duration=0.08,
            verification="next_frame",
            tag="close_bonus_popup",
        )
        assert task.verify_action(action, before, after) is True

    def test_verify_action_accepts_same_page_new_group(self) -> None:
        """回归：开局选完一张后下一组技能页立即弹出(同页换组) → 验证通过。

        实机 2026-09-05：页面未关但卡面(技能名)已变,选卡同样生效;
        若卡面未变(点击未生效)则验证失败交给重试。
        """
        task = make_task()
        before = Observation(
            frame_id=1,
            source_frame_ids=(1,),
            raw_data={
                "select_skill_button_visible": True,
                "skill_card_options": [
                    {"name": "圣羽加持", "tier": 9, "position": (50, 300)},
                    {"name": "灵魂剑气", "tier": 11, "position": (150, 300)},
                ],
            },
        )
        changed = Observation(
            frame_id=2,
            source_frame_ids=(2,),
            raw_data={
                "select_skill_button_visible": True,
                "skill_card_options": [
                    {"name": "天穹支援", "tier": 1, "position": (50, 300)},
                    {"name": "血量传承", "tier": 11, "position": (150, 300)},
                ],
            },
        )
        unchanged = Observation(
            frame_id=3,
            source_frame_ids=(3,),
            raw_data={
                "select_skill_button_visible": True,
                "skill_card_options": [
                    {"name": "圣羽加持", "tier": 9, "position": (50, 300)},
                    {"name": "灵魂剑气", "tier": 11, "position": (150, 300)},
                ],
            },
        )
        action = Action(
            kind="click",
            target=(150, 300),
            duration=0.08,
            verification="next_frame",
            tag="opening_skill_priority",
        )
        assert task.verify_action(action, before, changed) is True
        assert task.verify_action(action, before, unchanged) is False

    def test_last_selection_trusted_immediately(self) -> None:
        """回归：选满档位的最后一次点击直接信任。

        实机 2026-09-05：第 3 张点击后游戏页面停留约 11 秒才关闭,逐帧验证
        全判"未生效"烧完预算,且计数未达 3 导致快速通道失效,收尾拖了 ~19 秒。
        """
        task = make_task()
        task._opening_skill_selections = 2  # 本次点击为第 3 次(最后一次)
        card_options = [{"name": "圣羽加持", "tier": 9, "position": (50, 300)}]
        before = Observation(
            frame_id=1,
            source_frame_ids=(1,),
            raw_data={
                "select_skill_button_visible": True,
                "skill_card_options": list(card_options),
            },
        )
        after = Observation(  # 页面未关、卡面未变(模拟游戏收尾停留)
            frame_id=2,
            source_frame_ids=(2,),
            raw_data={
                "select_skill_button_visible": True,
                "skill_card_options": list(card_options),
            },
        )
        action = Action(
            kind="click",
            target=(50, 300),
            duration=0.08,
            verification="next_frame",
            tag="opening_skill_priority",
        )
        assert task.verify_action(action, before, after) is True
