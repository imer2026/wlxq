"""技能清单离线构建：OCR 统计阶段采集的技能卡，按英雄归类合并进清单 YAML。

用法：``wlxq-bot build-skill-catalog``（见 cli.py）。读取采集器落盘的
``meta.jsonl``（英雄归属在采集时已由图标匹配写入），对每张卡图做 OCR：
最上方一行识别为技能名，其余行按纵向顺序拼接为技能描述。同一英雄下同名
技能只保留一条；人工补充的 ``priority`` 等字段在合并时保留。

英雄技能开局页与合成 4 星赠送页完全一致，清单按英雄平铺，不区分来源页面
（来源只保留在 meta.jsonl 里供追溯）。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

# OCR 函数类型：输入 BGR 图，返回 (文字, 行中心 y) 列表
OcrFn = Callable[[np.ndarray], list[tuple[str, float]]]

# 未识别出英雄的卡的清单分节名
UNKNOWN_SECTION = "unknown"

_CATALOG_HEADER = (
    "# 技能清单：由 `wlxq-bot build-skill-catalog` 从采集卡图 OCR 生成并增量合并。\n"
    "# 结构：skills.<英雄名> = [{name, description, ...}]；人工补充的 priority 等\n"
    "# 字段在合并时保留，但重新生成会丢失本文件头以外的所有注释。\n"
    "# 英雄技能开局页与赠送页一致，清单按英雄平铺，不区分来源页面。\n"
    "# 本清单只做记录与后续技能优先级配置的基础，不影响自动化决策。\n"
)

_ocr_engine: Any = None


def _default_ocr(image: np.ndarray) -> list[tuple[str, float]]:
    """默认 OCR 实现（RapidOCR），引擎按需加载并缓存。"""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "离线建册需要 rapidocr-onnxruntime：pip install rapidocr-onnxruntime"
            ) from exc
        _ocr_engine = RapidOCR()
    result, _ = _ocr_engine(image)
    lines: list[tuple[str, float]] = []
    if result:
        for item in result:
            box = np.array(item[0], dtype=float)
            lines.append((str(item[1]), float(box[:, 1].mean())))
    return lines


@dataclass
class CatalogBuildReport:
    """一次建册的统计结果。"""

    total_captures: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    unknown: int = 0
    skipped: int = 0
    ocr_empty: int = 0
    heroes: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [
            f"采集记录={self.total_captures}",
            f"新增={self.added}",
            f"更新={self.updated}",
            f"保留={self.unchanged}",
            f"unknown={self.unknown}",
            f"跳过={self.skipped}",
            f"OCR为空={self.ocr_empty}",
        ]
        if self.heroes:
            parts.append("按英雄：" + "、".join(f"{h}×{n}" for h, n in sorted(self.heroes.items())))
        return "；".join(parts)


def read_capture_meta(meta_path: Path) -> list[dict[str, Any]]:
    """读取采集元数据 JSONL，按 hash 去重（保留首条）。"""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with meta_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("meta.jsonl 存在无法解析的行，已跳过: %.80s", line)
                continue
            digest = str(row.get("hash", ""))
            if not digest or digest in seen:
                continue
            seen.add(digest)
            rows.append(row)
    return rows


def extract_card_text(image: np.ndarray, ocr_fn: OcrFn) -> tuple[str, str]:
    """OCR 一张技能卡：最上方行为技能名，其余行按纵向顺序拼接为描述。

    中文行间拼接不加空格——跨行断开的词（如 boss 断成 bos/s）由此复原。
    """
    lines = ocr_fn(image)
    if not lines:
        return "", ""
    ordered = sorted(lines, key=lambda item: item[1])
    name = ordered[0][0].strip()
    description = "".join(text.strip() for text, _ in ordered[1:])
    return name, description


def build_catalog(
    captures_dir: Path,
    meta_path: Path,
    catalog_path: Path,
    ocr_fn: OcrFn | None = None,
    dry_run: bool = False,
) -> CatalogBuildReport:
    """从采集目录构建/增量合并技能清单。

    Args:
        captures_dir: 卡图目录（采集器的 ``captures/``）
        meta_path: 采集元数据 JSONL
        catalog_path: 技能清单 YAML 输出路径
        ocr_fn: OCR 实现；None 用内置 RapidOCR
        dry_run: True 只统计不写盘

    Returns:
        CatalogBuildReport，统计各类处理数量。
    """
    ocr_fn = ocr_fn or _default_ocr
    report = CatalogBuildReport()
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    unknown_entries: list[dict[str, Any]] = []

    if not meta_path.is_file():
        logger.warning("未找到采集元数据 %s，请先在统计阶段运行 run 采集技能卡", meta_path)
        return report
    rows = read_capture_meta(meta_path)

    for row in rows:
        report.total_captures += 1
        digest = str(row.get("hash", ""))
        image_path = captures_dir / f"{digest}.png"
        if not digest or not image_path.is_file():
            report.skipped += 1
            continue
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            report.skipped += 1
            continue
        name, description = extract_card_text(image, ocr_fn)
        if not name:
            report.ocr_empty += 1
        hero = row.get("hero") or None
        if hero is None:
            report.unknown += 1
            unknown_entries.append(
                {"image": image_path.name, "name": name, "description": description}
            )
            continue
        entries = grouped.setdefault(str(hero), {})
        if name in entries:
            report.unchanged += 1
        else:
            entries[name] = {"name": name, "description": description}

    existing_skills = _load_existing_skills(catalog_path)

    for hero, entries in grouped.items():
        current = existing_skills.get(hero)
        current = current if isinstance(current, list) else []
        by_name = {
            entry.get("name"): entry
            for entry in current
            if isinstance(entry, dict) and entry.get("name")
        }
        for name, entry in entries.items():
            report.heroes[hero] = report.heroes.get(hero, 0) + 1
            existing_entry = by_name.get(name)
            if existing_entry is None:
                current.append(entry)
                report.added += 1
            elif not existing_entry.get("description") and entry.get("description"):
                # 人工/历史条目描述为空时用本次 OCR 结果补齐，其余字段保留
                existing_entry["description"] = entry["description"]
                report.updated += 1
            else:
                report.unchanged += 1
        existing_skills[hero] = current

    if unknown_entries:
        current_unknown = existing_skills.get(UNKNOWN_SECTION)
        current_unknown = current_unknown if isinstance(current_unknown, list) else []
        known_images = {
            entry.get("image")
            for entry in current_unknown
            if isinstance(entry, dict) and entry.get("image")
        }
        for entry in unknown_entries:
            if entry["image"] not in known_images:
                current_unknown.append(entry)
        existing_skills[UNKNOWN_SECTION] = current_unknown

    if not dry_run:
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        body = yaml.safe_dump(
            {"skills": existing_skills},
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
        catalog_path.write_text(_CATALOG_HEADER + body, encoding="utf-8")
        logger.info(
            "技能清单已写入 path=%s added=%d updated=%d unchanged=%d",
            catalog_path,
            report.added,
            report.updated,
            report.unchanged,
        )
    return report


def _load_existing_skills(catalog_path: Path) -> dict[str, Any]:
    """读取现有清单的 skills 节；文件不存在或为空返回空字典。"""
    if not catalog_path.is_file():
        return {}
    try:
        data = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"技能清单 YAML 无法解析: {catalog_path}") from exc
    skills = data.get("skills") if isinstance(data, dict) else None
    return skills if isinstance(skills, dict) else {}
