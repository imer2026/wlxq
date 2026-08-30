"""skill_catalog 离线建册测试：OCR 用 fake 注入，不依赖真实模型。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from wlxq_bot.skill_catalog import (
    CatalogBuildReport,
    build_catalog,
    extract_card_text,
    read_capture_meta,
)


def fake_ocr(image: np.ndarray) -> list[tuple[str, float]]:
    """固定三行文本，y 顺序与行号一致。"""
    return [
        ("冰霜新星", 30.0),
        ("对范围内敌人造成120%伤害", 80.0),
        ("并冻结2秒", 110.0),
    ]


def write_capture(captures_dir: Path, digest: str) -> Path:
    """写一张合成卡图（内容不影响 OCR，fake 按 y 文本返回）。"""
    captures_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((200, 160, 3), 180, dtype=np.uint8)
    path = captures_dir / f"{digest}.png"
    ok, buf = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(buf.tobytes())
    return path


def write_meta(meta_path: Path, rows: list[dict]) -> Path:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(__import__("json").dumps(row, ensure_ascii=False) + "\n")
    return meta_path


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


class TestBuildCatalog:
    def make_setup(self, tmp_path: Path, rows: list[dict]) -> tuple[Path, Path, Path]:
        captures = tmp_path / "skill_cards" / "captures"
        for row in rows:
            write_capture(captures, str(row["hash"]))
        meta = write_meta(tmp_path / "skill_cards" / "meta.jsonl", rows)
        catalog = tmp_path / "configs" / "skills.yaml"
        return captures, meta, catalog

    def test_build_groups_by_hero(self, tmp_path: Path) -> None:
        rows = [
            {"hash": "aaa", "hero": "强袭", "page": "SELECT_OPENING_SKILLS"},
            {"hash": "bbb", "hero": "死骑", "page": "BUILD_MAIN_C"},
            {"hash": "ccc", "hero": None, "page": "X"},
        ]
        captures, meta, catalog = self.make_setup(tmp_path, rows)

        report = build_catalog(captures, meta, catalog, ocr_fn=fake_ocr)

        data = yaml.safe_load(catalog.read_text(encoding="utf-8"))
        assault = data["skills"]["强袭"]
        assert len(assault) == 1
        assert assault[0]["name"] == "冰霜新星"
        assert assault[0]["description"] == "对范围内敌人造成120%伤害并冻结2秒"
        assert data["skills"]["死骑"][0]["name"] == "冰霜新星"
        unknown = data["skills"]["unknown"]
        assert unknown[0]["image"] == "ccc.png"
        assert unknown[0]["name"] == "冰霜新星"
        assert isinstance(report, CatalogBuildReport)
        assert report.added == 2
        assert report.unknown == 1

    def test_rerun_is_idempotent(self, tmp_path: Path) -> None:
        rows = [{"hash": "aaa", "hero": "强袭"}]
        captures, meta, catalog = self.make_setup(tmp_path, rows)

        build_catalog(captures, meta, catalog, ocr_fn=fake_ocr)
        report = build_catalog(captures, meta, catalog, ocr_fn=fake_ocr)

        assert report.added == 0
        assert report.unchanged == 1
        data = yaml.safe_load(catalog.read_text(encoding="utf-8"))
        assert len(data["skills"]["强袭"]) == 1

    def test_merge_preserves_manual_fields(self, tmp_path: Path) -> None:
        """人工补充的 priority 保留；描述为空的历史条目被补齐。"""
        rows = [{"hash": "aaa", "hero": "强袭"}]
        captures, meta, catalog = self.make_setup(tmp_path, rows)
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

        report = build_catalog(captures, meta, catalog, ocr_fn=fake_ocr)

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
        rows = [{"hash": "aaa", "hero": "强袭"}]
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
        meta = write_meta(tmp_path / "skill_cards" / "meta.jsonl", [{"hash": "ghost", "hero": "强袭"}])
        captures.mkdir(parents=True, exist_ok=True)

        report = build_catalog(captures, meta, tmp_path / "skills.yaml", ocr_fn=fake_ocr)

        assert report.skipped == 1
        assert report.total_captures == 1


class TestReadCaptureMeta:
    def test_dedup_by_hash(self, tmp_path: Path) -> None:
        meta = tmp_path / "meta.jsonl"
        meta.write_text(
            '{"hash": "a", "hero": "强袭"}\n'
            '{"hash": "a", "hero": "强袭"}\n'
            '{"hash": "b", "hero": "死骑"}\n',
            encoding="utf-8",
        )
        rows = read_capture_meta(meta)
        assert [row["hash"] for row in rows] == ["a", "b"]

    def test_invalid_line_skipped(self, tmp_path: Path) -> None:
        meta = tmp_path / "meta.jsonl"
        meta.write_text('{"hash": "a"}\nnot-json\n\n', encoding="utf-8")
        rows = read_capture_meta(meta)
        assert len(rows) == 1
