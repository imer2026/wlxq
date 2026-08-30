"""从完整客户区截图离线裁取棋盘英雄格。"""

from __future__ import annotations

import csv
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from wlxq_bot.config import BoardGridParams
from wlxq_bot.hero_classifier.progress import ProgressLogger
from wlxq_bot.models import CoopRole
from wlxq_bot.perception.locator import (
    board_grid_for_role,
    format_cell_label,
    hero_cell_rois,
)
from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

_RAW_NAME_RE = re.compile(r"^(?P<round>\d{12})_frame(?P<frame>\d{6})\.png$")
CROP_MANIFEST_FIELDS = (
    "round_id",
    "main_c",
    "frame_index",
    "captured_at",
    "display_resolution",
    "client_width",
    "client_height",
    "dpi",
    "role",
    "cell_label",
    "source_screenshot",
    "crop_path",
    "hero_label",
    "star_label",
    "sample_kind",
)


@dataclass(frozen=True)
class CropStats:
    """一次局后离线裁格汇总。"""

    source_images: int
    expected_crops: int
    written_crops: int
    failed_sources: int
    elapsed_seconds: float
    average_source_ms: float
    manifest_path: Path
    # 开启 organize 时为跨格聚类后的簇数；未开启时为 None
    distinct_groups: int | None = None


@dataclass(frozen=True)
class _CropRow:
    round_id: str
    main_c: str
    frame_index: int
    cell_label: str
    source_screenshot: str
    crop_path: str
    client_width: int
    client_height: int
    captured_at: str = ""
    display_resolution: str = ""
    dpi: str = ""
    role: str = ""

    def as_csv_row(self) -> dict[str, str | int]:
        return {
            "round_id": self.round_id,
            "main_c": self.main_c,
            "frame_index": self.frame_index,
            "captured_at": self.captured_at,
            "display_resolution": self.display_resolution,
            "client_width": self.client_width,
            "client_height": self.client_height,
            "dpi": self.dpi,
            "role": self.role,
            "cell_label": self.cell_label,
            "source_screenshot": self.source_screenshot,
            "crop_path": self.crop_path,
            "hero_label": "",
            "star_label": "",
            "sample_kind": "",
        }


@dataclass(frozen=True)
class _SourceResult:
    rows: tuple[_CropRow, ...]
    duration_ms: float


class HeroCellCropper:
    """复用棋盘格位模型，把一局完整截图裁成 12 个英雄格 PNG。"""

    def __init__(
        self,
        *,
        round_dir: Path,
        role: CoopRole,
        board_params: dict[str, BoardGridParams],
        workers: int = 4,
        png_compression: int = 1,
        lineup_others: list[str] | None = None,
        main_c: str | None = None,
        organize: bool = False,
        group_threshold: float = 35.0,
        output_dir: Path | None = None,
        labeled_dir: Path | None = None,
        manifest_path: Path | None = None,
        path_root: Path | None = None,
        prepare_label_dirs: bool = True,
    ) -> None:
        if workers < 1:
            raise ValueError("workers 必须大于等于 1")
        if not 0 <= png_compression <= 9:
            raise ValueError("png_compression 必须在 0-9 之间")
        self._round_dir = Path(round_dir)
        self._raw_dir = self._round_dir / "raw"
        self._output_dir = (
            Path(output_dir) if output_dir is not None else self._round_dir / "unclassified"
        )
        self._labeled_dir = (
            Path(labeled_dir) if labeled_dir is not None else self._round_dir / "labeled"
        )
        self._manifest_path = (
            Path(manifest_path) if manifest_path is not None else self._round_dir / "manifest.csv"
        )
        self._path_root = Path(path_root) if path_root is not None else self._round_dir
        self._prepare_labels = prepare_label_dirs
        self._role = role
        self._grid = board_grid_for_role(role, board_params)
        self._workers = workers
        self._png_compression = png_compression
        # 主 C 之外、本局固定携带的队友；与 capture_manifest 里的 main_c 合起来预建标注目录
        self._lineup_others = [hero for hero in (lineup_others or []) if hero]
        # capture_manifest 缺失或无 main_c 时的兜底主C（通常取配置 default_main_c）
        self._main_c_fallback = main_c or ""
        # 裁剪后是否跨格池化按状态聚类到子目录（标注类位置无关）；关闭则保留按格布局
        self._organize = organize
        self._group_threshold = group_threshold
        self._capture_metadata = self._load_capture_metadata()

    def crop_all(self) -> CropStats:
        """裁剪本局所有原始截图；已有 manifest 时拒绝重复生成。"""
        if not self._raw_dir.is_dir():
            raise FileNotFoundError(f"本局 raw 目录不存在: {self._raw_dir}")
        if self._manifest_path.exists():
            raise FileExistsError(f"本局已存在 manifest，拒绝重复裁剪: {self._manifest_path}")
        sources = self._source_images()
        if not sources:
            raise FileNotFoundError(f"本局 raw 目录没有有效截图: {self._raw_dir}")

        if self._prepare_labels:
            self._prepare_label_dirs()
        started = perf_counter()
        rows: list[_CropRow] = []
        source_durations: list[float] = []
        failed_sources = 0
        progress = ProgressLogger(logger, f"对局裁切 round={self._round_dir.name}", len(sources))
        progress.start(detail=f"expected_crops={len(sources) * 12} workers={self._workers}")

        with ThreadPoolExecutor(max_workers=self._workers) as executor:
            futures = {executor.submit(self._crop_source, path): path for path in sources}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                except (OSError, RuntimeError, ValueError, cv2.error) as exc:
                    failed_sources += 1
                    logger.error("格子裁剪失败 source=%s reason=%r", path, exc)
                    progress.update(
                        len(source_durations) + failed_sources,
                        detail=f"written_crops={len(rows)} failed_sources={failed_sources}",
                    )
                    continue
                rows.extend(result.rows)
                source_durations.append(result.duration_ms)
                progress.update(
                    len(source_durations) + failed_sources,
                    detail=f"written_crops={len(rows)} failed_sources={failed_sources}",
                )

        if failed_sources:
            progress.finish(
                len(source_durations) + failed_sources,
                detail=f"written_crops={len(rows)} failed_sources={failed_sources}",
            )
            raise RuntimeError(
                f"有 {failed_sources}/{len(sources)} 张完整截图裁剪失败；"
                "已保留成功产物，但未生成 manifest，请检查日志后使用新目录重试"
            )

        distinct_groups: int | None = None
        if self._organize:
            rows, distinct_groups = self._organize_into_groups(rows)

        rows.sort(key=lambda item: (item.frame_index, item.cell_label))
        self._write_manifest(rows)
        progress.finish(len(sources), detail=f"written_crops={len(rows)} failed_sources=0")
        elapsed = perf_counter() - started
        return CropStats(
            source_images=len(sources),
            expected_crops=len(sources) * 12,
            written_crops=len(rows),
            failed_sources=0,
            elapsed_seconds=elapsed,
            average_source_ms=(sum(source_durations) / len(source_durations)),
            manifest_path=self._manifest_path,
            distinct_groups=distinct_groups,
        )

    def _organize_into_groups(self, rows: list[_CropRow]) -> tuple[list[_CropRow], int]:
        """裁剪后跨格池化按状态聚类，把每簇移入子目录，并更新 rows 的 crop_path。

        把 unclassified/ 下所有刚裁好的格子图（位置无关）跨格池化聚类，簇按大小降序
        编号，每簇移入 ``unclassified/<NN>_x{张数}_c{格子数}/``。标注类本就位置无关，
        同状态跨格子会并入同一簇，标注时每簇整批移到 labeled/ 即可。完成后清理空的
        ``unclassified/<格子>/`` 目录。返回 (更新后的 rows, 簇数)。
        """
        move_map, group_count = organize_crop_files(
            self._output_dir,
            path_root=self._path_root,
            threshold=self._group_threshold,
        )

        updated = [
            replace(row, crop_path=move_map.get(row.crop_path, row.crop_path)) for row in rows
        ]
        logger.info(
            "跨格归类完成 distinct_groups=%d threshold=%.1f",
            group_count,
            self._group_threshold,
        )
        return updated, group_count

    def _source_images(self) -> list[Path]:
        sources: list[tuple[int, Path]] = []
        for path in self._raw_dir.glob("*.png"):
            match = _RAW_NAME_RE.fullmatch(path.name)
            if match is None:
                logger.warning("跳过命名不符合约定的完整截图: %s", path)
                continue
            sources.append((int(match.group("frame")), path))
        return [path for _, path in sorted(sources)]

    def _round_main_c(self) -> str:
        """本局主 C：优先 capture_manifest，缺失时用构造时传入的兜底值。"""
        for meta in self._capture_metadata.values():
            main_c = meta.get("main_c", "")
            if main_c:
                return main_c
        return self._main_c_fallback

    def _prepare_label_dirs(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        (self._labeled_dir / "empty" / "plain").mkdir(parents=True, exist_ok=True)
        (self._labeled_dir / "empty" / "effect").mkdir(parents=True, exist_ok=True)
        (self._labeled_dir / "unavailable" / "plain").mkdir(parents=True, exist_ok=True)
        (self._labeled_dir / "unavailable" / "effect").mkdir(parents=True, exist_ok=True)
        (self._labeled_dir / "unknown").mkdir(parents=True, exist_ok=True)
        # 按 lineup（本局 main_c + 配置的固定队友）预建英雄/star 标注目录
        heroes: list[str] = []
        main_c = self._round_main_c()
        if main_c:
            heroes.append(main_c)
        for hero in self._lineup_others:
            if hero not in heroes:
                heroes.append(hero)
        for hero in heroes:
            for star in (1, 2, 3, 4):
                (self._labeled_dir / hero / f"star{star}").mkdir(parents=True, exist_ok=True)

    def _crop_source(self, source_path: Path) -> _SourceResult:
        started = perf_counter()
        match = _RAW_NAME_RE.fullmatch(source_path.name)
        if match is None:
            raise ValueError(f"完整截图文件名不符合约定: {source_path.name}")
        round_id = match.group("round")
        frame_index = int(match.group("frame"))
        frame = self._read_image(source_path)
        height, width = frame.shape[:2]
        rows: list[_CropRow] = []
        metadata = self._capture_metadata.get(frame_index, {})

        for cell, (left, top, cell_width, cell_height) in hero_cell_rois(
            self._grid,
            self._role,
            (width, height),
        ):
            crop = frame[top : top + cell_height, left : left + cell_width].copy()
            cell_label = format_cell_label(cell, self._role)
            crop_name = f"{round_id}_frame{frame_index:06d}_{cell_label}.png"
            # 先按格子落盘；organize 开启时随后跨格池化归类，关闭则保留按格布局
            crop_dir = self._output_dir / cell_label
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = crop_dir / crop_name
            self._write_png(crop_path, crop)
            rows.append(
                _CropRow(
                    round_id=round_id,
                    main_c=metadata.get("main_c", ""),
                    frame_index=frame_index,
                    cell_label=cell_label,
                    source_screenshot=source_path.relative_to(self._path_root).as_posix(),
                    crop_path=crop_path.relative_to(self._path_root).as_posix(),
                    client_width=width,
                    client_height=height,
                    captured_at=metadata.get("captured_at", ""),
                    display_resolution=metadata.get("display_resolution", ""),
                    dpi=metadata.get("dpi", ""),
                    role=metadata.get("role", self._role.value),
                )
            )
        return _SourceResult(tuple(rows), (perf_counter() - started) * 1000)

    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
        except OSError as exc:
            raise OSError(f"读取完整截图失败: {path}: {exc}") from exc
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"完整截图解码失败: {path}")
        return image

    def _write_png(self, path: Path, image: np.ndarray) -> None:
        if path.exists():
            raise FileExistsError(f"裁剪图已存在，拒绝覆盖: {path}")
        ok, encoded = cv2.imencode(
            ".png",
            image,
            [cv2.IMWRITE_PNG_COMPRESSION, self._png_compression],
        )
        if not ok:
            raise RuntimeError(f"格子 PNG 编码失败: {path}")
        encoded.tofile(str(path))

    def _load_capture_metadata(self) -> dict[int, dict[str, str]]:
        path = self._round_dir / "capture_manifest.csv"
        if not path.is_file():
            return {}
        metadata: dict[int, dict[str, str]] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                try:
                    frame_index = int(row.get("frame_index", ""))
                except ValueError:
                    continue
                if row.get("status") == "saved":
                    metadata[frame_index] = row
        return metadata

    def _write_manifest(self, rows: list[_CropRow]) -> None:
        with self._manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CROP_MANIFEST_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row.as_csv_row())


def organize_crop_files(
    output_dir: Path,
    *,
    path_root: Path,
    threshold: float,
) -> tuple[dict[str, str], int]:
    """将一个 import 的全部裁剪图联合聚类，并返回移动前后的相对路径映射。"""
    from wlxq_bot.hero_classifier.grouper import group_files

    output_dir = Path(output_dir)
    path_root = Path(path_root)
    files = sorted(output_dir.rglob("*.png"))
    groups = group_files(files, threshold=threshold, progress_task="一级聚类")
    groups.sort(key=lambda item: len(item[1]), reverse=True)
    move_map: dict[str, str] = {}
    move_progress = ProgressLogger(logger, "一级聚类结果整理", len(files))
    move_progress.start(detail=f"clusters={len(groups)}")
    moved = 0
    for index, (_representative, members) in enumerate(groups):
        cells = sorted({member.parent.name for member in members})
        cluster_dir = output_dir / f"{index:03d}_x{len(members):04d}_c{len(cells):02d}"
        cluster_dir.mkdir(exist_ok=True)
        for member in members:
            new_path = cluster_dir / member.name
            old_relative = member.relative_to(path_root).as_posix()
            shutil.move(str(member), str(new_path))
            move_map[old_relative] = new_path.relative_to(path_root).as_posix()
            moved += 1
            move_progress.update(moved, detail=f"clusters_done={index}/{len(groups)}")

    for cell_dir in [path for path in output_dir.iterdir() if path.is_dir()]:
        if not any(cell_dir.iterdir()):
            cell_dir.rmdir()
    move_progress.finish(moved, detail=f"clusters={len(groups)}")
    return move_map, len(groups)
