"""技能清单离线构建：OCR 统计阶段采集的技能卡，按英雄归类合并进清单 YAML。

用法：``wlxq-bot build-skill-catalog``（见 cli.py）。英雄归属在本模块离线
完成（卡图与英雄图标模板做模板匹配；运行时采集只裁卡、不做任何匹配），
再对每张卡图 OCR：最上方一行识别为技能名，其余行按纵向顺序拼接为技能
描述。同一英雄下同名技能只保留一条；OCR 对同一标题可能产生不同误读
（如 圣灵底护/圣灵庇护），因此组内按名称模糊相似度合并；人工补充的
``priority`` 等字段在合并时保留。

英雄技能开局页与合成 4 星赠送页完全一致，清单按英雄平铺，不区分来源页面
（来源只保留在 meta.jsonl 里供追溯）。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from wlxq_bot.perception.vision import Vision
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
    filtered: int = 0
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
            f"过滤垃圾={self.filtered}",
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


# 垃圾卡过滤（离线建册时执行，不删除原始采集图）。
# 真实技能卡卡身大片浅色：实测已知真卡浅色占比 ≈0.80；棋盘碎片等垃圾
# 普遍 <0.15；300 张真实采集样本呈双峰分布，0.5 位于空档
_LIGHT_RATIO_THRESHOLD = 0.5
# 非 unknown 兜底的最低描述字数：真卡描述远长于此，垃圾 OCR 残片凑不够
_MIN_DESCRIPTION_CHARS = 6
# 技能名模糊合并阈值：OCR 对同一标题的误读对（圣灵底护/圣灵庇护、
# 天穹支援/天弯支援）相似度 0.75；真实不同技能对（冰核增幅/冰核爆炸）
# 相似度 0.5，0.7 位于两者之间
_FUZZY_NAME_THRESHOLD = 0.7


def find_fuzzy_match(
    name: str,
    entries: list[dict[str, Any]],
    threshold: float = _FUZZY_NAME_THRESHOLD,
) -> dict[str, Any] | None:
    """在既有条目里找名称模糊相似的技能（OCR 同名异写归并）。"""
    best, best_ratio = None, 0.0
    for entry in entries:
        existing = str(entry.get("name", "")) if isinstance(entry, dict) else ""
        if not existing or not name:
            continue
        ratio = SequenceMatcher(None, existing, name).ratio()
        if ratio > best_ratio:
            best, best_ratio = entry, ratio
    return best if best_ratio >= threshold else None


def light_body_ratio(image: np.ndarray) -> float:
    """卡身浅色占比：中下部亮度大于 190 的像素比例。

    真实技能卡卡身大片浅色；棋盘、单位特写等垃圾裁剪是深色杂乱内容。
    """
    height, width = image.shape[:2]
    band = image[int(height * 0.25) : int(height * 0.90), :]
    if band.size == 0:
        return 0.0
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    return float((gray > 190).mean())


def resolve_icon_templates(
    templates_root: Path,
    hero_icons: dict[str, list[str]],
) -> dict[str, list[Path]]:
    """把英雄图标相对路径解析为全部分辨率模板包下的真实文件。

    离线建册不知道采集时游戏所在显示器分辨率，因此跨全部模板包解析；
    尺度不符的模板匹配时自然拿不到高分，不影响归属。
    """
    resolved: dict[str, list[Path]] = {}
    templates_root = Path(templates_root)
    for hero, rels in hero_icons.items():
        paths: list[Path] = []
        for rel in rels:
            for pack_dir in sorted(p for p in templates_root.iterdir() if p.is_dir()):
                candidate = pack_dir / rel
                if candidate.is_file():
                    paths.append(candidate)
        if paths:
            resolved[hero] = paths
        else:
            logger.warning("英雄 %s 的图标模板在模板包中均不存在，将全部归为 unknown", hero)
    return resolved


def attribute_hero(
    card: np.ndarray,
    hero_icons: dict[str, list[Path]],
    min_icon_confidence: float,
    vision: Vision | None = None,
) -> str | None:
    """对单张卡图做离线英雄归属：与全部图标模板匹配，取最高分且过阈值者。"""
    if not hero_icons:
        return None
    vision = vision or Vision()
    best_hero, best_confidence = None, 0.0
    for hero, templates in hero_icons.items():
        for template_path in templates:
            match = vision.match_template(card, str(template_path), threshold=min_icon_confidence)
            if match is not None and match.confidence > best_confidence:
                best_hero, best_confidence = hero, match.confidence
    return best_hero


def build_catalog(
    captures_dir: Path,
    meta_path: Path,
    catalog_path: Path,
    ocr_fn: OcrFn | None = None,
    hero_icons: dict[str, list[Path]] | None = None,
    min_icon_confidence: float = 0.70,
    dry_run: bool = False,
) -> CatalogBuildReport:
    """从采集目录构建/增量合并技能清单。

    Args:
        captures_dir: 卡图目录（采集器的 ``captures/``）
        meta_path: 采集元数据 JSONL
        catalog_path: 技能清单 YAML 输出路径
        ocr_fn: OCR 实现；None 用内置 RapidOCR
        hero_icons: 英雄名 → 图标模板路径（已解析为真实文件）；None 或空
            表示不做离线归属，全部按 unknown 归档
        min_icon_confidence: 离线图标匹配置信度阈值
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
    hero_icons = hero_icons or {}
    vision = Vision() if hero_icons else None
    filtered_files: list[str] = []
    existing_skills = _load_existing_skills(catalog_path)

    def _fuzzy_lookup(name: str, hero_key: str | None) -> dict[str, Any] | None:
        """在既有清单/本次已分组技能/待入库 unknown 里找名称模糊相似条目。"""
        candidates: list[dict[str, Any]] = []
        if hero_key is not None:
            keys = [hero_key]
        else:
            keys = sorted(set(list(existing_skills) + list(grouped)) - {UNKNOWN_SECTION})
            candidates.extend(unknown_entries)
        for key in keys:
            current = grouped.get(key, {})
            candidates.extend(current.values())
            existing = existing_skills.get(key)
            if isinstance(existing, list):
                candidates.extend(entry for entry in existing if isinstance(entry, dict))
        return find_fuzzy_match(name, candidates)

    def _absorb(
        matched: dict[str, Any] | None,
        name: str,
        description: str,
    ) -> str:
        """消化一张卡:命中既有条目则按需补空描述,否则返回 'new'。"""
        if matched is None:
            return "new"
        if not matched.get("description") and description:
            matched["description"] = description
            return "updated"
        return "unchanged"

    for row in rows:
        report.total_captures += 1
        digest = str(row.get("hash", ""))
        # 新版 meta 记录相对路径（按局分目录）；旧版退化为按哈希名查找
        rel = str(row.get("image", ""))
        if rel:
            image_path = captures_dir / rel
        else:
            matches = sorted(captures_dir.rglob(f"*{digest}.png"))
            image_path = matches[0] if matches else captures_dir / f"{digest}.png"
        if not digest or not image_path.is_file():
            report.skipped += 1
            continue
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            report.skipped += 1
            continue
        # 廉价结构过滤最先行：真技能卡卡身大片浅色（实测 ≈0.80），棋盘碎片等
        # 垃圾普遍 <0.15（300 样本双峰分布，0.5 位于空档）。深色直接判垃圾，
        # 不做归属匹配也不做 OCR——这是逐张处理的主要提速手段
        if light_body_ratio(image) < _LIGHT_RATIO_THRESHOLD:
            report.filtered += 1
            filtered_files.append(image_path.name)
            continue
        hero = attribute_hero(image, hero_icons, min_icon_confidence, vision)
        name, description = extract_card_text(image, ocr_fn)
        if not name:
            report.ocr_empty += 1
        if hero is not None:
            outcome = _absorb(_fuzzy_lookup(name, str(hero)), name, description)
            if outcome == "new":
                grouped.setdefault(str(hero), {})[name] = {
                    "name": name,
                    "description": description,
                }
                report.added += 1
            elif outcome == "updated":
                report.updated += 1
            else:
                report.unchanged += 1
            continue
        if name and len(description) >= _MIN_DESCRIPTION_CHARS:
            outcome = _absorb(_fuzzy_lookup(name, None), name, description)
            if outcome == "new":
                unknown_entries.append(
                    {
                        "image": image_path.relative_to(captures_dir).as_posix(),
                        "name": name,
                        "description": description,
                    }
                )
            elif outcome == "updated":
                report.updated += 1
            else:
                report.unchanged += 1
            continue
        # 卡身像卡但 OCR 无实义文本(半渲染帧等),不入清单;
        # 原图保留在 captures/,文件名记录到 filtered.txt 供复核捞回
        report.filtered += 1
        filtered_files.append(image_path.name)

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
            elif not existing_entry.get("description") and entry.get("description"):
                # 人工/历史条目描述为空时用本次 OCR 结果补齐，其余字段保留
                existing_entry["description"] = entry["description"]
        existing_skills[hero] = current

    if unknown_entries:
        current_unknown = existing_skills.get(UNKNOWN_SECTION)
        current_unknown = current_unknown if isinstance(current_unknown, list) else []
        # 按 (名称, 描述) 去重 + 名称模糊去重——image 路径会随目录调整变化,
        # 不能作为身份;图纸已不存在的旧条目(如目录迁移前)一并清理
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for entry in current_unknown + unknown_entries:
            if not isinstance(entry, dict):
                continue
            image = str(entry.get("image", ""))
            if image and not (captures_dir / image).is_file():
                continue
            if find_fuzzy_match(str(entry.get("name", "")), list(merged.values())) is not None:
                continue
            key = (str(entry.get("name", "")), str(entry.get("description", "")))
            merged.setdefault(key, entry)
        existing_skills[UNKNOWN_SECTION] = list(merged.values())
    # unknown 计数用合并后的最终条数(待人工确认的独特技能数),比处理过程的
    # 候选数更有意义
    report.unknown = len(existing_skills.get(UNKNOWN_SECTION) or [])

    if not dry_run:
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        body = yaml.safe_dump(
            {"skills": existing_skills},
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
        catalog_path.write_text(_CATALOG_HEADER + body, encoding="utf-8")
        if filtered_files:
            # 过滤清单供人工复核:误杀的真卡可人工搬回并自行补标
            manifest = meta_path.parent / "filtered.txt"
            manifest.write_text("\n".join(filtered_files) + "\n", encoding="utf-8")
            logger.info(
                "垃圾过滤清单已写入 path=%s count=%d", manifest, len(filtered_files)
            )
        logger.info(
            "技能清单已写入 path=%s added=%d updated=%d unchanged=%d filtered=%d",
            catalog_path,
            report.added,
            report.updated,
            report.unchanged,
            report.filtered,
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
