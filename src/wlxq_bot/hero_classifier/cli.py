"""``wlxq-bot hero-classifier`` 命令组。"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint

from wlxq_bot.config import load_default_config, load_local_config, load_tasks_config
from wlxq_bot.models import CoopRole
from wlxq_bot.perception.screen import (
    ScreenCapture,
    enable_dpi_awareness,
    get_window_monitor_resolution,
)
from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)

_MAIN_C_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ROUND_ID_FORMAT = "%Y%m%d%H%M"

app = typer.Typer(
    help="采集、裁剪、训练、预分类和评估棋盘英雄格分类器",
    no_args_is_help=True,
)


@app.command("collect")
def collect(
    main_c: Annotated[str, typer.Option("--main-c", help="本局主 C 英文标识，例如 assault")],
    round_id: Annotated[
        str | None,
        typer.Option(
            "--round-id",
            help="对局开始时间 YYYYMMDDHHMM；默认使用命令启动时的本地时间",
        ),
    ] = None,
    role: Annotated[CoopRole, typer.Option("--role", help="己方合作角色")] = CoopRole.HELPER,
    duration: Annotated[float, typer.Option("--duration", min=1.0, help="采集时长（秒）")] = 360.0,
    interval: Annotated[
        float, typer.Option("--interval", min=0.1, help="目标截图间隔（秒）")
    ] = 1.0,
    output_root: Annotated[
        Path, typer.Option("--output-root", help="本地英雄分类数据根目录（不提交 Git）")
    ] = Path("datasets/hero_classifier"),
    title: Annotated[str | None, typer.Option("--title", "-t", help="游戏窗口标题")] = None,
    queue_size: Annotated[
        int, typer.Option("--queue-size", min=1, max=256, help="保存队列上限")
    ] = 8,
    png_compression: Annotated[
        int, typer.Option("--png-compression", min=0, max=9, help="PNG 压缩等级")
    ] = 1,
) -> None:
    """按固定时间点采集一局完整客户区截图，并异步保存 PNG。"""
    from wlxq_bot.hero_classifier.collector import HeroFrameCollector

    try:
        main_c = _normalize_main_c(main_c)
        round_id = _resolve_round_id(round_id)
    except ValueError as exc:
        logger.error("hero-classifier collect 参数无效 reason=%r", exc)
        rprint(f"[red]采集参数无效: {exc}[/red]")
        raise typer.Exit(2) from exc
    enable_dpi_awareness()
    local_config = load_local_config(Path("configs/local.yaml"))
    window_title = title or (local_config.window.title if local_config else None)
    if not window_title:
        rprint("[red]未指定 --title，且 configs/local.yaml 不存在[/red]")
        raise typer.Exit(1)
    screen = ScreenCapture()
    handle = screen.find_window(window_title)
    if not handle:
        logger.error("英雄格素材采集未找到窗口 title=%s", window_title)
        rprint(f"[red]未找到窗口: {window_title}[/red]")
        raise typer.Exit(1)
    display_resolution = get_window_monitor_resolution(handle)
    round_dir = _dataset_round_dir(
        output_root,
        main_c=main_c,
        display_resolution=display_resolution,
        role=role,
        round_id=round_id,
    )
    logger.info(
        "hero-classifier collect 开始 round=%s main_c=%s role=%s "
        "duration=%.1fs interval=%.3fs output=%s",
        round_id,
        main_c,
        role.value,
        duration,
        interval,
        round_dir,
    )
    collector = HeroFrameCollector(
        lambda: screen.capture(handle),
        round_dir=round_dir,
        round_id=round_id,
        main_c=main_c,
        role=role,
        display_resolution=display_resolution,
        interval_seconds=interval,
        duration_seconds=duration,
        queue_size=queue_size,
        png_compression=png_compression,
    )
    try:
        stats = collector.collect()
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        logger.error("hero-classifier collect 失败 reason=%r", exc)
        rprint(f"[red]采集失败: {exc}[/red]")
        raise typer.Exit(1) from exc
    logger.info(
        "hero-classifier collect 结束 expected=%d captured=%d saved=%d failed=%d "
        "dropped=%d schedule_skipped=%d elapsed=%.3fs",
        stats.expected,
        stats.captured,
        stats.saved,
        stats.failed,
        stats.dropped,
        stats.schedule_skipped,
        stats.elapsed_seconds,
    )
    rprint(
        "[green]✓ 完整客户区采集完成[/green]\n"
        f"  目录: {round_dir}\n"
        f"  理论/实际/保存: {stats.expected}/{stats.captured}/{stats.saved}\n"
        f"  失败/队列丢弃/调度跳过: {stats.failed}/{stats.dropped}/{stats.schedule_skipped}\n"
        f"  截图耗时 avg/max: {stats.capture_avg_ms:.1f}/{stats.capture_max_ms:.1f} ms\n"
        f"  保存耗时 avg/max: {stats.save_avg_ms:.1f}/{stats.save_max_ms:.1f} ms\n"
        f"  manifest: {stats.manifest_path}"
    )


@app.command("crop")
def crop(
    round_dir: Annotated[Path, typer.Argument(help="包含 raw/ 的单局目录")],
    role: Annotated[CoopRole, typer.Option("--role", help="己方合作角色")] = CoopRole.HELPER,
    workers: Annotated[int, typer.Option("--workers", min=1, max=16, help="有限并发数")] = 4,
    png_compression: Annotated[
        int, typer.Option("--png-compression", min=0, max=9, help="格子 PNG 压缩等级")
    ] = 1,
    main_c: Annotated[
        str,
        typer.Option(
            "--main-c",
            help="主C英雄标识；capture_manifest 有 main_c 时以它为准，否则用此值（默认取配置 default_main_c）",
        ),
    ] = "",
    organize: Annotated[
        bool,
        typer.Option(
            "--organize/--no-organize",
            help="裁剪后按相似度把每个格子的帧分组到子目录，便于按状态标注",
        ),
    ] = True,
    group_threshold: Annotated[
        float,
        typer.Option(
            "--group-threshold",
            min=0.0,
            help="跨格聚类阈值（平均像素差，小于则视为相似画面）；默认 35.0（按一局 250 帧标定）；只用于整理待标注图片，不会自动贴标签",
        ),
    ] = 35.0,
) -> None:
    """离线读取一局完整截图，每张自动裁出 12 个英雄格；默认裁完跨格池化按状态归类到簇子目录。"""
    from wlxq_bot.hero_classifier.cropper import HeroCellCropper

    logger.info(
        "hero-classifier crop 开始 round_dir=%s role=%s workers=%d",
        round_dir,
        role.value,
        workers,
    )
    try:
        tasks_config = load_tasks_config(Path("configs/tasks.yaml"))
        default_config = load_default_config(Path("configs/default.yaml"))
        lineup_others = list(default_config.hero_classifier.lineup_others)
        resolved_main_c = main_c or default_config.run.default_main_c
        stats = HeroCellCropper(
            round_dir=round_dir,
            role=role,
            board_params=tasks_config.board,
            workers=workers,
            png_compression=png_compression,
            lineup_others=lineup_others,
            main_c=resolved_main_c,
            organize=organize,
            group_threshold=group_threshold,
        ).crop_all()
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        logger.error("hero-classifier crop 失败 reason=%r", exc)
        rprint(f"[red]裁剪失败: {exc}[/red]")
        raise typer.Exit(1) from exc
    logger.info(
        "hero-classifier crop 结束 sources=%d crops=%d elapsed=%.3fs avg_source=%.3fms",
        stats.source_images,
        stats.written_crops,
        stats.elapsed_seconds,
        stats.average_source_ms,
    )
    groups_line = (
        f"  归类簇数: {stats.distinct_groups}\n" if stats.distinct_groups is not None else ""
    )
    rprint(
        "[green]✓ 12 格离线裁剪完成[/green]\n"
        f"  完整截图: {stats.source_images}\n"
        f"  理论/实际格子图: {stats.expected_crops}/{stats.written_crops}\n"
        f"{groups_line}"
        f"  总耗时: {stats.elapsed_seconds:.2f} 秒\n"
        f"  manifest: {stats.manifest_path}\n"
        f"  待分类目录: {round_dir / 'unclassified'}"
    )


@app.command("group")
def group(
    round_dir: Annotated[Path, typer.Argument(help="包含 unclassified/ 的单局目录")],
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            min=0.0,
            help="跨格聚类阈值（平均像素差，小于则视为相似画面）；默认 35.0（按一局 250 帧标定）；只用于整理待标注图片",
        ),
    ] = 35.0,
) -> None:
    """跨格池化聚类 unclassified/ 下所有裁剪图，生成单张 groups/clusters.png 联系表，便于标注前纵观共有哪些状态。"""
    from wlxq_bot.hero_classifier.grouper import group_files, make_contact_sheet

    unclassified = round_dir / "unclassified"
    if not unclassified.is_dir():
        rprint(f"[red]找不到 unclassified 目录: {unclassified}[/red]")
        raise typer.Exit(1)
    logger.info("hero-classifier group 开始 round_dir=%s threshold=%.1f", round_dir, threshold)
    files = sorted(unclassified.rglob("*.png"))
    groups = group_files(files, threshold=threshold)
    groups.sort(key=lambda item: len(item[1]), reverse=True)
    frames = sum(len(members) for _, members in groups)
    out_dir = round_dir / "groups"
    out_path = out_dir / "clusters.png"
    make_contact_sheet("", groups, out_path)
    logger.info(
        "hero-classifier group 结束 distinct=%d frames=%d out=%s",
        len(groups),
        frames,
        out_path,
    )
    rprint(f"[green]✓ 跨格聚类完成[/green] {len(groups)} 簇 / {frames} 帧\n  联系表: {out_path}")


@app.command("import-rounds")
def import_dataset_rounds(
    dataset_root: Annotated[
        Path, typer.Argument(help="数据组根目录，例如 .../assault/3000x2000/helper")
    ],
    split: Annotated[str, typer.Option("--split", help="train、validation 或 test")],
    rounds: Annotated[str, typer.Option("--rounds", help="本批来源局，逗号分隔")],
    import_id: Annotated[str, typer.Option("--import-id", help="本批稳定标识")],
    role: Annotated[CoopRole, typer.Option("--role", help="己方合作角色")] = CoopRole.HELPER,
    workers: Annotated[int, typer.Option("--workers", min=1, max=16)] = 4,
    png_compression: Annotated[int, typer.Option("--png-compression", min=0, max=9)] = 1,
    main_c: Annotated[str, typer.Option("--main-c", help="清单缺失时使用的主 C")] = "",
    group_threshold: Annotated[float, typer.Option("--group-threshold", min=0.0)] = 35.0,
    candidate_split_trigger: Annotated[
        int, typer.Option("--candidate-split-trigger", min=1, help="一级簇超过此数量时二次细分")
    ] = 100,
    candidate_group_threshold: Annotated[
        float,
        typer.Option("--candidate-group-threshold", min=0.0, help="大簇二次细分阈值"),
    ] = 15.0,
    candidate_max_per_group: Annotated[
        int,
        typer.Option("--candidate-max-per-group", min=1, max=100, help="每个最终候选组的图片上限"),
    ] = 10,
) -> None:
    """指定多局创建一个增量 import，集中裁切后联合聚类；不修改既有标签。"""
    from wlxq_bot.hero_classifier.dataset import import_rounds

    logger.info(
        "hero-classifier import-rounds 开始 dataset=%s split=%s import=%s rounds=%s",
        dataset_root,
        split,
        import_id,
        rounds,
    )
    try:
        tasks_config = load_tasks_config(Path("configs/tasks.yaml"))
        default_config = load_default_config(Path("configs/default.yaml"))
        stats = import_rounds(
            dataset_root=dataset_root,
            split=split,
            import_id=import_id,
            round_ids=_csv_values(rounds),
            role=role,
            board_params=tasks_config.board,
            workers=workers,
            png_compression=png_compression,
            lineup_others=list(default_config.hero_classifier.lineup_others),
            main_c=main_c or default_config.run.default_main_c,
            group_threshold=group_threshold,
            candidate_split_trigger=candidate_split_trigger,
            candidate_group_threshold=candidate_group_threshold,
            candidate_max_per_group=candidate_max_per_group,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        logger.error("hero-classifier import-rounds 失败 reason=%r", exc)
        rprint(f"[red]导入失败: {exc}[/red]")
        raise typer.Exit(1) from exc
    logger.info(
        "hero-classifier import-rounds 结束 split=%s import=%s rounds=%d skipped=%d "
        "crops=%d groups=%d candidates=%d",
        stats.split,
        stats.import_id,
        len(stats.rounds),
        len(stats.skipped_rounds),
        stats.written_crops,
        stats.distinct_groups,
        stats.candidate_images,
    )
    skipped_lines = "".join(
        f"  跳过: {item.round_id}（已在 {item.split}/{item.import_id} 完成）\n"
        for item in stats.skipped_rounds
    )
    for item in stats.skipped_rounds:
        logger.warning(
            "hero-classifier import-rounds 跳过已处理对局 round=%s existing_split=%s "
            "existing_import=%s",
            item.round_id,
            item.split,
            item.import_id,
        )
    if not stats.rounds:
        rprint(f"[yellow]没有需要处理的新对局，未创建 import[/yellow]\n{skipped_lines.rstrip()}")
        return
    rprint(
        "[green]✓ 多局素材导入完成[/green]\n"
        f"{skipped_lines}"
        f"  split/import: {stats.split}/{stats.import_id}\n"
        f"  新处理局/跳过局: {len(stats.rounds)}/{len(stats.skipped_rounds)}\n"
        f"  完整截图/格子图: {stats.source_images}/{stats.written_crops}\n"
        f"  联合聚类簇数: {stats.distinct_groups}\n"
        f"  候选组/候选图片: {stats.candidate_groups}/{stats.candidate_images}\n"
        f"  待挑选目录: {stats.import_dir / 'unclassified'}\n"
        f"  人工确认目录: {stats.import_dir / 'candidates'}\n"
        f"  candidate manifest: {stats.candidate_manifest_path}\n"
        f"  manifest: {stats.manifest_path}"
    )


@app.command("sync-labels")
def sync_dataset_labels(
    dataset_root: Annotated[
        Path, typer.Argument(help="数据组根目录，例如 .../assault/3000x2000/helper")
    ],
    split: Annotated[str, typer.Option("--split", help="train、validation 或 test")],
) -> None:
    """扫描人工移动后的 labeled，完整重建该 split 的数据清单。"""
    from wlxq_bot.hero_classifier.dataset import sync_labels

    logger.info("hero-classifier sync-labels 开始 dataset=%s split=%s", dataset_root, split)
    try:
        stats = sync_labels(dataset_root=dataset_root, split=split)
    except (OSError, ValueError) as exc:
        logger.error("hero-classifier sync-labels 失败 reason=%r", exc)
        rprint(f"[red]标签同步失败: {exc}[/red]")
        raise typer.Exit(1) from exc
    logger.info(
        "hero-classifier sync-labels 结束 split=%s samples=%d unknown=%d",
        stats.split,
        stats.samples,
        stats.unknown_samples,
    )
    rprint(
        "[green]✓ 标签清单已完整重建[/green]\n"
        f"  样本/unknown: {stats.samples}/{stats.unknown_samples}\n"
        f"  manifest: {stats.manifest_path}"
    )


@app.command("select-candidates")
def select_candidates(
    import_dir: Annotated[
        Path, typer.Argument(help="已有 import 目录，例如 train/imports/20260812_001")
    ],
    split_trigger: Annotated[
        int, typer.Option("--split-trigger", min=1, help="一级簇超过此数量时二次细分")
    ] = 100,
    threshold: Annotated[
        float, typer.Option("--threshold", min=0.0, help="大簇二次细分阈值")
    ] = 15.0,
    max_per_group: Annotated[
        int, typer.Option("--max-per-group", min=1, max=100, help="每个最终组候选上限")
    ] = 10,
) -> None:
    """为已完成一级聚类的历史 import 补生成 candidates，不重新裁切或聚类。"""
    from wlxq_bot.hero_classifier.dataset import select_import_candidates

    logger.info("hero-classifier select-candidates 开始 import=%s", import_dir)
    try:
        stats = select_import_candidates(
            import_dir=import_dir,
            secondary_trigger=split_trigger,
            secondary_threshold=threshold,
            max_per_group=max_per_group,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        logger.error("hero-classifier select-candidates 失败 reason=%r", exc)
        rprint(f"[red]候选生成失败: {exc}[/red]")
        raise typer.Exit(1) from exc
    logger.info(
        "hero-classifier select-candidates 结束 primary=%d groups=%d images=%d",
        stats.primary_clusters,
        stats.candidate_groups,
        stats.candidate_images,
    )
    rprint(
        "[green]✓ 历史 import 候选生成完成[/green]\n"
        f"  一级簇/候选组/候选图: {stats.primary_clusters}/"
        f"{stats.candidate_groups}/{stats.candidate_images}\n"
        f"  candidates: {import_dir / 'candidates'}\n"
        f"  manifest: {stats.manifest_path}"
    )


@app.command("suggest-labels")
def suggest_labels(
    import_dir: Annotated[
        Path, typer.Argument(help="待预分类 import，例如 train/imports/20260812_001")
    ],
    model: Annotated[Path, typer.Option("--model", help="训练导出的 ONNX 模型")],
    metadata: Annotated[
        Path | None,
        typer.Option("--metadata", help="模型 metadata；默认使用 ONNX 同名 .json"),
    ] = None,
    confidence_threshold: Annotated[
        float | None,
        typer.Option(
            "--confidence-threshold",
            min=0.0,
            max=1.0,
            help="预分类置信度门槛；默认读取模型 metadata",
        ),
    ] = None,
    margin_threshold: Annotated[
        float | None,
        typer.Option(
            "--margin-threshold",
            min=0.0,
            max=1.0,
            help="第一、第二名概率差门槛；默认读取模型 metadata",
        ),
    ] = None,
    batch_size: Annotated[
        int, typer.Option("--batch-size", min=1, max=2048, help="ONNX 批量推理大小")
    ] = 128,
) -> None:
    """用训练后的模型预分类 candidates；只生成副本，不写入 labeled。"""
    from wlxq_bot.hero_classifier.suggester import suggest_import_labels

    logger.info(
        "hero-classifier suggest-labels 开始 import=%s model=%s batch_size=%d",
        import_dir,
        model,
        batch_size,
    )
    try:
        stats = suggest_import_labels(
            import_dir=import_dir,
            model_path=model,
            metadata_path=metadata,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            batch_size=batch_size,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        logger.error("hero-classifier suggest-labels 失败 reason=%r", exc)
        rprint(f"[red]候选预分类失败: {exc}[/red]")
        raise typer.Exit(1) from exc
    logger.info(
        "hero-classifier suggest-labels 结束 groups=%d images=%d suggested=%d review=%d",
        stats.groups,
        stats.images,
        stats.suggested_groups,
        stats.review_groups,
    )
    rprint(
        "[green]✓ candidates 模型预分类完成[/green]\n"
        f"  候选组/候选图: {stats.groups}/{stats.images}\n"
        f"  建议类别组/review 组: {stats.suggested_groups}/{stats.review_groups}\n"
        f"  review 明细（低置信度/混合组/未知）: "
        f"{stats.low_confidence_groups}/{stats.mixed_groups}/{stats.unknown_groups}\n"
        f"  人工审核目录: {stats.suggested_dir}\n"
        f"  prediction manifest: {stats.manifest_path}\n"
        "  注意: suggested 只是模型建议；确认后仍需人工移动到 labeled 并执行 sync-labels"
    )


@app.command("train")
def train(
    dataset_root: Annotated[Path, typer.Argument(help="包含 train/validation 的数据组根目录")],
    train_rounds: Annotated[
        str | None, typer.Option("--train-rounds", help="仅兼容旧目录；训练局，逗号分隔")
    ] = None,
    validation_rounds: Annotated[
        str | None,
        typer.Option("--validation-rounds", help="仅兼容旧目录；验证局，逗号分隔"),
    ] = None,
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="训练产物目录")] = Path(
        "outputs/hero_classifier/model"
    ),
    epochs: Annotated[int, typer.Option("--epochs", min=1, max=1000)] = 20,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1, max=2048)] = 64,
    input_size: Annotated[int, typer.Option("--input-size", min=32, max=512)] = 96,
    learning_rate: Annotated[float, typer.Option("--learning-rate", min=0.000001)] = 0.001,
    workers: Annotated[int, typer.Option("--workers", min=0, max=32)] = 0,
    pretrained: Annotated[bool, typer.Option("--pretrained/--no-pretrained")] = True,
    main_c: Annotated[
        str,
        typer.Option("--main-c", help="本模型主 C；默认从标准数据组目录推导"),
    ] = "",
    star1_weight: Annotated[float, typer.Option("--star1-weight", min=0.000001)] = 1.0,
    star2_weight: Annotated[float, typer.Option("--star2-weight", min=0.000001)] = 1.0,
    main_c_star3_weight: Annotated[
        float, typer.Option("--main-c-star3-weight", min=0.000001)
    ] = 0.8,
    other_star3_weight: Annotated[float, typer.Option("--other-star3-weight", min=0.000001)] = 0.5,
    main_c_star4_weight: Annotated[
        float, typer.Option("--main-c-star4-weight", min=0.000001)
    ] = 0.3,
    other_star4_weight: Annotated[float, typer.Option("--other-star4-weight", min=0.000001)] = 0.1,
    empty_weight: Annotated[float, typer.Option("--empty-weight", min=0.000001)] = 0.5,
    unavailable_weight: Annotated[float, typer.Option("--unavailable-weight", min=0.000001)] = 0.3,
) -> None:
    """按整局隔离数据训练 MobileNetV3-Small，并导出 ONNX。"""
    from wlxq_bot.hero_classifier.trainer import TrainingConfig, train_hero_classifier

    logger.info(
        "hero-classifier train 开始 dataset=%s train_rounds=%s validation_rounds=%s",
        dataset_root,
        train_rounds or "train split",
        validation_rounds or "validation split",
    )
    try:
        resolved_main_c = (
            _normalize_main_c(main_c) if main_c else _main_c_from_dataset_root(dataset_root)
        )
        config = TrainingConfig(
            input_size=input_size,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            workers=workers,
            pretrained=pretrained,
            main_c=resolved_main_c,
            star1_weight=star1_weight,
            star2_weight=star2_weight,
            main_c_star3_weight=main_c_star3_weight,
            other_star3_weight=other_star3_weight,
            main_c_star4_weight=main_c_star4_weight,
            other_star4_weight=other_star4_weight,
            empty_weight=empty_weight,
            unavailable_weight=unavailable_weight,
        )
        result = train_hero_classifier(
            dataset_root=dataset_root,
            output_dir=output_dir,
            train_rounds=_csv_values(train_rounds) if train_rounds else None,
            validation_rounds=_csv_values(validation_rounds) if validation_rounds else None,
            config=config,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("hero-classifier train 失败 reason=%r", exc)
        rprint(f"[red]训练失败: {exc}[/red]")
        raise typer.Exit(1) from exc
    rprint(
        "[green]✓ 英雄格分类器训练完成[/green]\n"
        f"  最佳轮次: {result.best_epoch}\n"
        f"  最佳验证准确率: {result.best_validation_accuracy:.4f}\n"
        f"  ONNX: {result.onnx_path}\n"
        f"  metadata: {result.metadata_path}"
    )


@app.command("evaluate")
def evaluate(
    dataset_root: Annotated[Path, typer.Argument(help="包含 test/ 的数据组根目录")],
    model: Annotated[Path, typer.Option("--model", help="待评估 ONNX 模型")],
    rounds: Annotated[
        str | None, typer.Option("--rounds", help="仅兼容旧目录；评估局，逗号分隔")
    ] = None,
    split: Annotated[str, typer.Option("--split", help="集中数据目录，默认 test")] = "test",
    output_dir: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "outputs/hero_classifier/evaluation"
    ),
    batch_size: Annotated[int, typer.Option("--batch-size", min=1, max=2048)] = 128,
) -> None:
    """在未参与训练的整局数据上评估 ONNX 分类器。"""
    from wlxq_bot.hero_classifier.evaluator import evaluate_hero_classifier

    logger.info(
        "hero-classifier evaluate 开始 dataset=%s model=%s rounds=%s",
        dataset_root,
        model,
        rounds or split,
    )
    try:
        result = evaluate_hero_classifier(
            dataset_root=dataset_root,
            model_path=model,
            output_dir=output_dir,
            rounds=_csv_values(rounds) if rounds else None,
            split=split,
            batch_size=batch_size,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        logger.error("hero-classifier evaluate 失败 reason=%r", exc)
        rprint(f"[red]评估失败: {exc}[/red]")
        raise typer.Exit(1) from exc
    logger.info(
        "hero-classifier evaluate 结束 samples=%d accepted=%d rejected=%d accuracy=%.4f",
        result.samples,
        result.accepted,
        result.rejected,
        result.accuracy_all,
    )
    rprint(
        "[green]✓ 英雄格分类器评估完成[/green]\n"
        f"  样本/接受/拒绝: {result.samples}/{result.accepted}/{result.rejected}\n"
        f"  总体准确率: {result.accuracy_all:.4f}\n"
        f"  已接受样本准确率: {result.accuracy_accepted:.4f}\n"
        f"  报告: {result.report_path}"
    )


def _csv_values(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("至少需要指定一个局目录")
    return values


def _normalize_main_c(value: str) -> str:
    main_c = value.strip().lower()
    if _MAIN_C_RE.fullmatch(main_c) is None:
        raise ValueError("main_c 必须是小写英文标识，可包含数字和下划线")
    return main_c


def _main_c_from_dataset_root(dataset_root: Path) -> str:
    """从 <主C>/<分辨率>/<角色> 标准数据组目录推导主 C。"""
    root = Path(dataset_root)
    if len(root.parents) < 2 or re.fullmatch(r"\d+x\d+", root.parent.name) is None:
        raise ValueError("无法从数据组目录推导主 C，请显式传入 --main-c")
    return _normalize_main_c(root.parent.parent.name)


def _resolve_round_id(value: str | None, *, now: datetime | None = None) -> str:
    round_id = (
        value.strip() if value is not None else (now or datetime.now()).strftime(_ROUND_ID_FORMAT)
    )
    try:
        parsed = datetime.strptime(round_id, _ROUND_ID_FORMAT)
    except ValueError as exc:
        raise ValueError("round_id 必须是有效的 YYYYMMDDHHMM 本地时间") from exc
    if parsed.strftime(_ROUND_ID_FORMAT) != round_id:
        raise ValueError("round_id 必须是有效的 YYYYMMDDHHMM 本地时间")
    return round_id


def _dataset_round_dir(
    output_root: Path,
    *,
    main_c: str,
    display_resolution: tuple[int, int],
    role: CoopRole,
    round_id: str,
) -> Path:
    """返回按主 C、分辨率、角色和 rounds 分层的原始对局目录。"""
    return (
        Path(output_root)
        / main_c
        / f"{display_resolution[0]}x{display_resolution[1]}"
        / role.value
        / "rounds"
        / round_id
    )
