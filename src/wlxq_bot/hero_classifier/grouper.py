"""按图像内容整理待标注裁剪图，便于人工挑选。

裁剪后同一视觉状态（同属空格/同英雄同星级）的多帧，逐像素几乎每帧都因微动画略
有差异，精确哈希无法合并。本模块用「平均像素差 + 容差」的贪心聚类：每帧与已有各
簇的代表帧比，差小于阈值则归入最接近的簇，否则开新簇。聚类只缩小人工浏览范围，
不判断英雄或星级，也不保证簇内标签一致；标注者仍从簇中挑选图片移入 ``labeled/``。

``group_files`` 接收一份显式的裁剪图列表——通常是一整局 12 格的全部裁剪图跨格池
化：标注类本就位置无关（empty / <英雄>+star，不带格子号），跨格聚类可以把分散在
不同格子的相似画面放到一起，减少人工翻找范围。``group_cell`` 是按单格目录聚类的薄
包装，保留作原语。

联系表把每个簇画成一格，标注代表帧号与该簇帧数。簇内标签明确且一致时可以整簇移
入 ``labeled/``；混合簇只挑选标签明确的图片。未标注帧留在 ``unclassified/``，训练
只读 ``labeled/``。
"""

from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wlxq_bot.hero_classifier.progress import ProgressLogger
from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

_FRAME_RE = re.compile(r"_frame(\d+)_")
_CROP_RE = re.compile(r"^(?P<round>\d{12})_frame(?P<frame>\d{6})_(?P<cell>[1-6][ABC])\.png$")
_CANDIDATE_MANIFEST_FIELDS = (
    "primary_cluster",
    "secondary_cluster",
    "source_group_size",
    "selection_order",
    "round_id",
    "frame_index",
    "cell_label",
    "source_path",
    "candidate_path",
)


@dataclass(frozen=True)
class CandidateStats:
    """一个 import 的候选图生成汇总。"""

    primary_clusters: int
    candidate_groups: int
    candidate_images: int
    manifest_path: Path


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"格子图解码失败: {path}")
    return image


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """两张图的全分辨率平均绝对像素差（0-255）。尺寸不同时返回大值以拒绝归并。"""
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def frame_index(path: Path) -> int:
    """从 <round>_frame<NNN>_<cell>.png 解析帧号；无法解析返回 -1。"""
    match = _FRAME_RE.search(path.name)
    return int(match.group(1)) if match else -1


def group_files(
    files: list[Path],
    threshold: float = 35.0,
    *,
    progress_task: str | None = None,
) -> list[tuple[Path, list[Path]]]:
    """对一份裁剪图列表按平均像素差贪心聚类（跨格池化的核心）。

    每帧按列表顺序处理，与已有各簇代表帧比较，归入差最小且低于阈值的簇；都不满足
    则开新簇。相似画面再次出现时可回到原簇。通常传入一整局 12 格的全部裁剪图做跨
    格池化；结果只用于缩小人工浏览范围，不代表簇内已经具有统一标签。

    Args:
        files: 裁剪图路径列表（调用方负责排序，保证代表帧选择确定）。
        threshold: 平均像素差阈值；小于它视为相似画面。单代表帧会随帧漂移，阈值需
            容纳漂移与特效幅度。默认 35.0（按一局 250 帧标定）：太低会把相似画面切成
            大量单人簇，太高会把不同英雄误并——他局动画/特效幅度不同时应重新标定。

    Returns:
        [(代表帧, [簇内所有帧]), ...]，按簇首次出现的顺序。空列表返回 []。
    """
    reps: list[np.ndarray] = []
    rep_paths: list[Path] = []
    clusters: list[list[Path]] = []
    progress = ProgressLogger(logger, progress_task, len(files)) if progress_task else None
    if progress is not None:
        progress.start(detail=f"threshold={threshold:.1f}")
    for processed, path in enumerate(files, start=1):
        img = _read_image(path)
        best_idx = -1
        best_diff = float("inf")
        for index, rep in enumerate(reps):
            diff = _mean_abs_diff(img, rep)
            if diff < best_diff:
                best_diff = diff
                best_idx = index
        if best_idx >= 0 and best_diff < threshold:
            clusters[best_idx].append(path)
        else:
            reps.append(img)
            rep_paths.append(path)
            clusters.append([path])
        if progress is not None:
            progress.update(processed, detail=f"clusters={len(clusters)}")
    if progress is not None:
        progress.finish(len(files), detail=f"clusters={len(clusters)}")
    return [(rep_paths[i], clusters[i]) for i in range(len(reps))]


def group_cell(
    cell_dir: Path,
    threshold: float = 35.0,
) -> list[tuple[Path, list[Path]]]:
    """对 cell_dir 下所有 PNG（单格目录）按平均像素差聚类；``group_files`` 的薄包装。

    Args:
        cell_dir: 单个格子目录（内含 <round>_frame<NNN>_<cell>.png）
        threshold: 见 :func:`group_files`。

    Returns:
        [(代表帧, [簇内所有帧]), ...]，按簇首次出现的顺序。空目录返回 []。
    """
    return group_files(sorted(cell_dir.glob("*.png")), threshold=threshold)


def generate_candidates(
    unclassified_dir: Path,
    candidates_dir: Path,
    *,
    path_root: Path,
    secondary_trigger: int = 100,
    secondary_threshold: float = 15.0,
    max_per_group: int = 10,
    path_replacement: tuple[str, str] | None = None,
) -> CandidateStats:
    """为所有一级簇生成少量多样化候选图，原始裁剪图保持不变。

    不超过 ``secondary_trigger`` 张的一级簇直接作为一个候选组；超过门槛时先以更严格
    的阈值二次细分。每个最终组使用最远优先法选择最多 ``max_per_group`` 张：先取稳定
    的第一张，后续每次选择与已选集合视觉差异最大的图片，减少连续近似帧占满候选。
    """
    if secondary_trigger < 1:
        raise ValueError("secondary_trigger 必须大于等于 1")
    if secondary_threshold < 0:
        raise ValueError("secondary_threshold 不能小于 0")
    if max_per_group < 1:
        raise ValueError("max_per_group 必须大于等于 1")
    unclassified_dir = Path(unclassified_dir)
    candidates_dir = Path(candidates_dir)
    path_root = Path(path_root)
    primary_dirs = sorted(path for path in unclassified_dir.iterdir() if path.is_dir())
    total_images = sum(1 for primary_dir in primary_dirs for _path in primary_dir.glob("*.png"))
    progress = ProgressLogger(logger, "候选生成", total_images)
    progress.start(
        detail=(
            f"primary_clusters={len(primary_dirs)} split_trigger={secondary_trigger} "
            f"secondary_threshold={secondary_threshold:.1f} max_per_group={max_per_group}"
        )
    )
    rows: list[dict[str, str | int]] = []
    candidate_groups = 0
    candidate_images = 0
    processed_images = 0

    for primary_dir in primary_dirs:
        files = sorted(primary_dir.glob("*.png"))
        if not files:
            continue
        if len(files) > secondary_trigger:
            final_groups = group_files(
                files,
                threshold=secondary_threshold,
                progress_task=f"候选二次细分 primary={primary_dir.name}",
            )
            final_groups.sort(key=lambda item: len(item[1]), reverse=True)
        else:
            final_groups = [(files[0], files)]

        primary_candidate_dir = candidates_dir / primary_dir.name
        for secondary_index, (_representative, members) in enumerate(final_groups):
            selected = select_diverse_files(
                members,
                max_count=max_per_group,
                progress_task=(
                    f"候选多样化选择 primary={primary_dir.name} secondary={secondary_index:03d}"
                    if len(members) > secondary_trigger
                    else None
                ),
            )
            secondary_name = f"{secondary_index:03d}_x{len(selected):04d}"
            target_dir = primary_candidate_dir / secondary_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for selection_order, source in enumerate(selected, start=1):
                target = target_dir / source.name
                shutil.copy2(source, target)
                metadata = _crop_metadata(source)
                source_relative = source.relative_to(path_root).as_posix()
                candidate_relative = target.relative_to(path_root).as_posix()
                if path_replacement is not None:
                    source_relative = source_relative.replace(*path_replacement)
                    candidate_relative = candidate_relative.replace(*path_replacement)
                rows.append(
                    {
                        "primary_cluster": primary_dir.name,
                        "secondary_cluster": secondary_name,
                        "source_group_size": len(members),
                        "selection_order": selection_order,
                        "round_id": metadata[0],
                        "frame_index": metadata[1],
                        "cell_label": metadata[2],
                        "source_path": source_relative,
                        "candidate_path": candidate_relative,
                    }
                )
            candidate_groups += 1
            candidate_images += len(selected)
        processed_images += len(files)
        progress.update(
            processed_images,
            detail=(
                f"primary={primary_dir.name} candidate_groups={candidate_groups} "
                f"candidate_images={candidate_images}"
            ),
        )

    manifest_path = candidates_dir.parent / "candidate_manifest.csv"
    _write_candidate_manifest(manifest_path, rows)
    progress.finish(
        processed_images,
        detail=f"candidate_groups={candidate_groups} candidate_images={candidate_images}",
    )
    return CandidateStats(
        primary_clusters=len(primary_dirs),
        candidate_groups=candidate_groups,
        candidate_images=candidate_images,
        manifest_path=manifest_path,
    )


def select_diverse_files(
    files: list[Path],
    *,
    max_count: int = 5,
    progress_task: str | None = None,
) -> list[Path]:
    """确定性地选择最多 max_count 张视觉差异尽量大的图片。"""
    if max_count < 1:
        raise ValueError("max_count 必须大于等于 1")
    ordered = sorted(files)
    if len(ordered) <= max_count:
        return ordered
    # 大簇可能包含上千张图；32x32 描述足以做候选多样性排序，并显著降低内存占用。
    progress = ProgressLogger(logger, progress_task, len(ordered)) if progress_task else None
    if progress is not None:
        progress.start(detail=f"max_candidates={max_count}")
    images: list[np.ndarray] = []
    for processed, path in enumerate(ordered, start=1):
        images.append(cv2.resize(_read_image(path), (32, 32), interpolation=cv2.INTER_AREA))
        if progress is not None:
            progress.update(processed, detail="读取视觉描述")
    selected_indices = [0]
    remaining = set(range(1, len(ordered)))
    while remaining and len(selected_indices) < max_count:
        next_index = max(
            remaining,
            key=lambda index: (
                min(_mean_abs_diff(images[index], images[item]) for item in selected_indices),
                -index,
            ),
        )
        selected_indices.append(next_index)
        remaining.remove(next_index)
    if progress is not None:
        progress.finish(len(ordered), detail=f"selected={len(selected_indices)}")
    return [ordered[index] for index in selected_indices]


def _crop_metadata(path: Path) -> tuple[str, int, str]:
    match = _CROP_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"候选图文件名不符合约定: {path}")
    return match.group("round"), int(match.group("frame")), match.group("cell")


def _write_candidate_manifest(path: Path, rows: list[dict[str, str | int]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_CANDIDATE_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_font(size: int):
    for name in (
        "C:/Windows/Fonts/ariblk.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_contact_sheet(
    cell_label: str,
    groups: list[tuple[Path, list[Path]]],
    out_path: Path,
    tile: int = 150,
    cols: int = 6,
) -> None:
    """生成联系表 PNG：每个簇一格，标注「f<帧号> ×<帧数>」。"""
    if not groups:
        return
    count = len(groups)
    col_count = min(cols, count)
    row_count = (count + col_count - 1) // col_count
    sheet = Image.new("RGB", (col_count * tile, row_count * tile), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    label_font = _load_font(16)
    img_box = tile - 20  # 留出标签高度
    for index, (rep, members) in enumerate(groups):
        row, col = divmod(index, col_count)
        x0 = col * tile + 10
        y0 = row * tile
        try:
            img = Image.open(rep).convert("RGB")
        except OSError:
            continue
        scale = img_box / max(img.size)
        img = img.resize((max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale))))
        sheet.paste(img, (x0, y0))
        prefix = f"{cell_label} " if cell_label else ""
        draw.text(
            (x0, y0 + img_box + 4),
            f"{prefix}f{frame_index(rep)} ×{len(members)}",
            fill=(255, 255, 255),
            font=label_font,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
