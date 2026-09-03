"""skill_catalog 离线建册测试：OCR 用 fake 注入，不依赖真实模型。"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from wlxq_bot.skill_catalog import (
    CatalogBuildReport,
    attribute_hero,
    build_catalog,
    extract_card_text,
    light_body_ratio,
    read_capture_meta,
)

_ICON_CENTER = (0.50, 0.45)


def fake_ocr(image: np.ndarray) -> list[tuple[str, float]]:
    """固定三行文本，y 顺序与行号一致。"""
    return [
        ("冰霜新星", 30.0),
        ("对范围内敌人造成120%伤害", 80.0),
        ("并冻结2秒", 110.0),
    ]


def make_card_with_icon(
    width: int = 280,
    height: int = 560,
    icon_seed: int | None = 7,
    light: bool = False,
) -> np.ndarray:
    """生成一张合成卡。

    默认深色噪声底；``light=True`` 生成浅色卡身（近似真技能卡）。
    icon_seed 非 None 时嵌入随机纹理图标（用于归属匹配）。
    """
    if light:
        card = np.full((height, width, 3), 225, dtype=np.uint8)
    else:
        rng = np.random.default_rng(11)
        card = rng.integers(40, 70, size=(height, width, 3), dtype=np.uint8)
    if icon_seed is not None:
        icon_rng = np.random.default_rng(icon_seed)
        icon = icon_rng.integers(0, 255, size=(110, 110, 3), dtype=np.uint8)
        cx, cy = int(width * _ICON_CENTER[0]), int(height * _ICON_CENTER[1])
        card[cy - 55 : cy + 55, cx - 55 : cx + 55] = icon
    return card


def brightness_ocr(image: np.ndarray) -> list[tuple[str, float]]:
    """按亮度分派的 OCR fake：浅色卡返回技能文本，深色垃圾返回数字残片。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.mean() > 150:
        return fake_ocr(image)
    return [("10.0B", 30.0)]


def save_icon_template(path: Path, icon_seed: int = 7) -> Path:
    rng = np.random.default_rng(icon_seed)
    icon = rng.integers(0, 255, size=(110, 110, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", icon)
    assert ok
    path.write_bytes(buf.tobytes())
    return path


class TestExtractCardText:
    def test_first_line_is_name_rest_joined(self) -> None:
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        name, description = extract_card_text(image, fake_ocr)
        assert name == "冰霜新星"
        # 中文行间拼接不加空格
        assert description == "对范围内敌人造成120%伤害并冻结2秒"

    def test_lines_sorted_by_y(self) -> None:
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        name, description = extract_card_text(
            image,
            lambda _: [("死灵战神附带", 100.0), ("最高星", 10.0)],
        )
        assert name == "最高星"
        assert description == "死灵战神附带"

    def test_empty_ocr(self) -> None:
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        name, description = extract_card_text(image, lambda _: [])
        assert name == ""
        assert description == ""


class TestAttributeHero:
    def test_matches_icon_template(self, tmp_path: Path) -> None:
        tpl_a = save_icon_template(tmp_path / "a.png", icon_seed=7)
        tpl_b = save_icon_template(tmp_path / "b.png", icon_seed=8)
        heroes = {"强袭": [tpl_a], "死骑": [tpl_b]}
        card_a = make_card_with_icon(icon_seed=7)
        card_none = make_card_with_icon(icon_seed=None)

        assert attribute_hero(card_a, heroes, 0.70) == "强袭"
        assert attribute_hero(card_none, heroes, 0.70) is None


class TestLightBodyRatio:
    def test_light_card_high_dark_card_low(self) -> None:
        light = make_card_with_icon(light=True, icon_seed=None)
        dark = make_card_with_icon(light=False, icon_seed=None)
        assert light_body_ratio(light) > 0.9
        assert light_body_ratio(dark) < 0.15


class TestBuildCatalog:
    def make_setup(
        self,
        tmp_path: Path,
        rows: list[dict],
        icon_seeds: dict[str, int | None] | None = None,
        kinds: dict[str, str] | None = None,
    ) -> tuple[Path, Path, Path]:
        """kinds: hash -> 'light'|'dark'，默认全部深色噪声卡。"""
        captures = tmp_path / "skill_cards" / "captures"
        captures.mkdir(parents=True, exist_ok=True)
        for row in rows:
            digest = str(row["hash"])
            seed = (icon_seeds or {}).get(digest)
            light = (kinds or {}).get(digest) == "light"
            image = make_card_with_icon(icon_seed=seed, light=light)
            ok, buf = cv2.imencode(".png", image)
            assert ok
            (captures / f"{digest}.png").write_bytes(buf.tobytes())
        meta = tmp_path / "skill_cards" / "meta.jsonl"
        with meta.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        catalog = tmp_path / "configs" / "skills.yaml"
        return captures, meta, catalog

    def test_offline_attribution_groups_by_hero(self, tmp_path: Path) -> None:
        """英雄归属在离线完成：图标命中的进英雄分组。

        ccc 无图标但技能文本与已入库的 冰霜新星 相同 → 判 unchanged,
        既不入英雄分组也不进 unknown。
        """
        tpl_a = save_icon_template(tmp_path / "a.png", icon_seed=7)
        tpl_b = save_icon_template(tmp_path / "b.png", icon_seed=8)
        rows = [
            {"hash": "aaa", "page": "SELECT_OPENING_SKILLS"},
            {"hash": "bbb", "page": "BUILD_MAIN_C"},
            {"hash": "ccc", "page": "X"},
        ]
        captures, meta, catalog = self.make_setup(
            tmp_path,
            rows,
            icon_seeds={"aaa": 7, "bbb": 8, "ccc": None},
            # 深色卡会被结构过滤直接判垃圾,这里验证的是归属分组与 unknown,
            # 三张全部用浅色卡(真卡都是浅色)
            kinds={"aaa": "light", "bbb": "light", "ccc": "light"},
        )

        report = build_catalog(
            captures,
            meta,
            catalog,
            ocr_fn=fake_ocr,
            hero_icons={"强袭": [tpl_a], "死骑": [tpl_b]},
        )

        data = yaml.safe_load(catalog.read_text(encoding="utf-8"))
        assault = data["skills"]["强袭"]
        assert len(assault) == 1
        assert assault[0]["name"] == "冰霜新星"
        assert assault[0]["description"] == "对范围内敌人造成120%伤害并冻结2秒"
        assert data["skills"]["死骑"][0]["name"] == "冰霜新星"
        assert "unknown" not in data["skills"]
        assert isinstance(report, CatalogBuildReport)
        assert report.added == 2
        assert report.unchanged == 1
        assert report.unknown == 0

    def test_fuzzy_merge_ocr_variants(self, tmp_path: Path) -> None:
        """OCR 同名异写(圣灵底护/圣灵庇护)在同一英雄组内被模糊合并。"""
        tpl_a = save_icon_template(tmp_path / "a.png", icon_seed=7)
        rows = [
            {"hash": "aaa"},
            {"hash": "bbb"},
        ]
        captures, meta, catalog = self.make_setup(
            tmp_path,
            rows,
            icon_seeds={"aaa": 7, "bbb": 7},
            kinds={"aaa": "light", "bbb": "light"},
        )

        def variant_ocr(image: np.ndarray) -> list[tuple[str, float]]:
            # 两张卡内容相同,这里模拟两次 OCR 对同一标题的不同误读
            return [("圣灵底护", 30.0), ("召唤物受到赐福时回血30%", 80.0)]

        report = build_catalog(
            captures,
            meta,
            catalog,
            ocr_fn=variant_ocr,
            hero_icons={"天使": [tpl_a]},
        )

        data = yaml.safe_load(catalog.read_text(encoding="utf-8"))
        angel = data["skills"]["天使"]
        assert len(angel) == 1
        assert angel[0]["name"] == "圣灵底护"
        assert report.added == 1
        assert report.unchanged == 1

    def test_garbage_filtered_from_catalog(self, tmp_path: Path) -> None:
        """垃圾裁剪（深色、OCR 无实义文本）被过滤；同文本卡去重后进 unknown。"""
        rows = [
            {"hash": "aaa"},  # 浅色无图标 + 实义 OCR → unknown
            {"hash": "bbb"},  # 同上,但文本与 aaa 相同 → 被 aaa 吸收(unchanged)
            {"hash": "ccc"},  # 深色 + 数字残片 → 过滤
        ]
        captures, meta, catalog = self.make_setup(
            tmp_path,
            rows,
            icon_seeds={"aaa": None, "bbb": None, "ccc": None},
            kinds={"aaa": "light", "bbb": "light", "ccc": "dark"},
        )

        report = build_catalog(captures, meta, catalog, ocr_fn=brightness_ocr)

        data = yaml.safe_load(catalog.read_text(encoding="utf-8"))
        unknown = data["skills"]["unknown"]
        assert [entry["image"] for entry in unknown] == ["aaa.png"]
        assert report.filtered == 1
        assert report.unknown == 1
        assert report.unchanged == 1
        manifest = tmp_path / "skill_cards" / "filtered.txt"
        assert manifest.read_text(encoding="utf-8").splitlines() == ["ccc.png"]

    def test_filter_manifest_not_written_on_dry_run(self, tmp_path: Path) -> None:
        rows = [{"hash": "ccc"}]
        captures, meta, catalog = self.make_setup(tmp_path, rows, icon_seeds={"ccc": None})

        build_catalog(captures, meta, catalog, ocr_fn=brightness_ocr, dry_run=True)

        assert not (tmp_path / "skill_cards" / "filtered.txt").exists()

    def test_unknown_dedup_by_text_and_prune_stale(self, tmp_path: Path) -> None:
        """unknown 按(名称,描述)去重;指向已不存在图片的旧条目被清理。"""
        rows = [{"hash": "aaa"}]
        captures, meta, catalog = self.make_setup(
            tmp_path, rows, icon_seeds={"aaa": None}, kinds={"aaa": "light"}
        )
        catalog.parent.mkdir(parents=True, exist_ok=True)
        catalog.write_text(
            "skills:\n"
            "  unknown:\n"
            "    - name: 冰霜新星\n"
            "      description: 对范围内敌人造成120%伤害并冻结2秒\n"
            "      image: 旧目录/旧图.png\n"          # 图片已不存在 → 清理
            "    - name: 旧技能\n"
            "      description: 只有老图里有的条目\n"
            "      image: 旧目录/旧图2.png\n",      # 同上
            encoding="utf-8",
        )

        report = build_catalog(
            captures, meta, catalog, ocr_fn=fake_ocr, dry_run=False
        )

        data = yaml.safe_load(catalog.read_text(encoding="utf-8"))
        unknown = data["skills"]["unknown"]
        # 同文本旧条目被合并成一条(换成本次有效的 image),失效旧条目被剪掉
        assert len(unknown) == 1
        assert unknown[0]["image"] == "aaa.png"
        assert report.unknown == 1

    def test_rerun_is_idempotent(self, tmp_path: Path) -> None:
        tpl_a = save_icon_template(tmp_path / "a.png", icon_seed=7)
        rows = [{"hash": "aaa"}]
        captures, meta, catalog = self.make_setup(
            tmp_path, rows, icon_seeds={"aaa": 7}, kinds={"aaa": "light"}
        )
        heroes = {"强袭": [tpl_a]}

        build_catalog(captures, meta, catalog, ocr_fn=fake_ocr, hero_icons=heroes)
        report = build_catalog(
            captures, meta, catalog, ocr_fn=fake_ocr, hero_icons=heroes
        )

        assert report.added == 0
        assert report.unchanged == 1
        data = yaml.safe_load(catalog.read_text(encoding="utf-8"))
        assert len(data["skills"]["强袭"]) == 1

    def test_merge_preserves_manual_fields(self, tmp_path: Path) -> None:
        """人工补充的 priority 保留；描述为空的历史条目被补齐。"""
        tpl_a = save_icon_template(tmp_path / "a.png", icon_seed=7)
        rows = [{"hash": "aaa"}]
        captures, meta, catalog = self.make_setup(
            tmp_path, rows, icon_seeds={"aaa": 7}, kinds={"aaa": "light"}
        )
        catalog.parent.mkdir(parents=True, exist_ok=True)
        catalog.write_text(
            "skills:\n"
            "  强袭:\n"
            "    - name: 冰霜新星\n"
            "      description: ''\n"
            "      priority: 1\n"
            "    - name: 已有条目\n"
            "      description: 人工写好的描述\n",
            encoding="utf-8",
        )

        report = build_catalog(
            captures,
            meta,
            catalog,
            ocr_fn=fake_ocr,
            hero_icons={"强袭": [tpl_a]},
        )

        data = yaml.safe_load(catalog.read_text(encoding="utf-8"))
        assault = data["skills"]["强袭"]
        frost = next(entry for entry in assault if entry["name"] == "冰霜新星")
        assert frost["description"] == "对范围内敌人造成120%伤害并冻结2秒"
        assert frost["priority"] == 1
        assert report.updated == 1
        # unchanged 只统计本次采集到的条目;清单中未被采集的「已有条目」不计入
        assert report.unchanged == 0
        assert report.added == 0
        assert len(assault) == 2
        existing = next(entry for entry in assault if entry["name"] == "已有条目")
        assert existing["description"] == "人工写好的描述"

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        rows = [{"hash": "aaa"}]
        captures, meta, catalog = self.make_setup(tmp_path, rows)

        build_catalog(captures, meta, catalog, ocr_fn=fake_ocr, dry_run=True)

        assert not catalog.exists()

    def test_missing_meta_returns_empty_report(self, tmp_path: Path) -> None:
        captures = tmp_path / "captures"
        report = build_catalog(
            captures,
            tmp_path / "missing.jsonl",
            tmp_path / "skills.yaml",
            ocr_fn=fake_ocr,
        )
        assert report.total_captures == 0
        assert not (tmp_path / "skills.yaml").exists()

    def test_missing_image_skipped(self, tmp_path: Path) -> None:
        captures = tmp_path / "skill_cards" / "captures"
        meta = tmp_path / "skill_cards" / "meta.jsonl"
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text('{"hash": "ghost"}\n', encoding="utf-8")
        captures.mkdir(parents=True, exist_ok=True)

        report = build_catalog(captures, meta, tmp_path / "skills.yaml", ocr_fn=fake_ocr)

        assert report.skipped == 1
        assert report.total_captures == 1


class TestReadCaptureMeta:
    def test_dedup_by_hash(self, tmp_path: Path) -> None:
        meta = tmp_path / "meta.jsonl"
        meta.write_text(
            '{"hash": "a"}\n{"hash": "a"}\n{"hash": "b"}\n',
            encoding="utf-8",
        )
        rows = read_capture_meta(meta)
        assert [row["hash"] for row in rows] == ["a", "b"]

    def test_invalid_line_skipped(self, tmp_path: Path) -> None:
        meta = tmp_path / "meta.jsonl"
        meta.write_text('{"hash": "a"}\nnot-json\n\n', encoding="utf-8")
        rows = read_capture_meta(meta)
        assert len(rows) == 1
