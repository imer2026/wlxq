"""命令行入口。

提供 screenshot / find / recognize / calibrate / run / inspect / save-window /
adjust-window / click / spam-click / pick / move 等命令。
inspect 命令用于检查游戏窗口信息，验证截图前置条件。
save-window 和 adjust-window 用于保存和恢复窗口尺寸。
recognize 命令实时截取当前游戏窗口画面（或识别指定截图），用模板包做模板匹配验证识别效果。
find 命令在当前游戏画面中识别单个指定模板，返回置信度，支持 --threshold 调节识别度。
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console

from wlxq_bot import __version__
from wlxq_bot.config import (
    LocalConfig,
    WindowSpec,
    load_local_config,
    load_tasks_config,
    save_local_config,
)
from wlxq_bot.hero_classifier.cli import app as hero_classifier_app
from wlxq_bot.perception.screen import (
    adjust_window_size,
    enable_dpi_awareness,
    find_window_by_title,
    find_window_smart,
    find_windows_by_keyword,
    get_window_info,
    get_window_monitor_resolution,
    list_windows,
)
from wlxq_bot.utils.log import get_logger, setup_logging

# configs/local.yaml 路径
LOCAL_CONFIG_PATH = Path("configs/local.yaml")
TASKS_CONFIG_PATH = Path("configs/tasks.yaml")

logger = get_logger(__name__)

app = typer.Typer(
    name="wlxq-bot",
    help="《永远的蔚蓝星球》微信小游戏本地自动化工具集",
    no_args_is_help=True,
)
exec_app = typer.Typer(
    help="执行可独立验证的自动化能力",
    no_args_is_help=True,
)
app.add_typer(exec_app, name="exec")
app.add_typer(hero_classifier_app, name="hero-classifier")


def _manual_action_components(jitter: int = 0):
    """为显式人工输入命令构造统一的截图、安全检查和动作执行链。"""
    from wlxq_bot.action.executor import ActionExecutor
    from wlxq_bot.action.input import InputController
    from wlxq_bot.action.safety import SafetyGuard
    from wlxq_bot.perception.screen import ScreenCapture

    screen = ScreenCapture()
    safety = SafetyGuard(max_failures=1, frame_ttl_ms=3000)
    executor = ActionExecutor(
        safety,
        InputController(jitter=jitter),
        min_delay=0.0,
        max_delay=0.0,
        context_validator=screen.validate_context,
    )
    return screen, executor


@app.callback(invoke_without_command=False)
def main_callback(
    ctx: typer.Context,
    debug: bool = typer.Option(
        False,
        "--debug",
        "-v",
        help="输出详细调试日志（窗口句柄、坐标换算、置信度、frame_id 等）",
    ),
) -> None:
    """wlxq-bot 命令行工具。

    全局选项 ``--debug`` 把日志级别调到 DEBUG，输出每个关键步骤的详细数据，
    用于排查识别失败、坐标偏移、窗口变化等问题。默认只输出 INFO 及以上。

    示例::

        wlxq-bot --debug screenshot
        wlxq-bot --debug run coop --main-c assault
    """
    setup_logging("DEBUG" if debug else "INFO")
    logger.debug("日志级别=%s", "DEBUG" if debug else "INFO")


@app.command()
def version() -> None:
    """显示版本号。"""
    rprint(f"[bold]wlxq-bot[/bold] v{__version__}")


@app.command()
def screenshot(
    title: str | None = typer.Option(
        None,
        "--title",
        "-t",
        help="游戏窗口标题，不指定则从 configs/local.yaml 读取",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="输出文件路径，不指定则自动命名到 screenshots/raw/",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="截图后自动用默认图片查看器打开",
    ),
) -> None:
    """截取当前游戏窗口画面，保存为 PNG。

    用于采集模板图片、调试识别效果和保存调试证据。

    示例::

        wlxq-bot screenshot                        # 自动查找窗口并截图
        wlxq-bot screenshot -t "永远的蔚蓝星球"      # 指定窗口标题
        wlxq-bot screenshot -o my_shot.png          # 指定输出路径
        wlxq-bot screenshot --show                  # 截图后自动打开
    """
    enable_dpi_awareness()

    # 确定窗口标题
    window_title = title
    if window_title is None:
        local_config = load_local_config(LOCAL_CONFIG_PATH)
        if local_config is not None:
            window_title = local_config.window.title
            logger.debug("窗口标题来自 configs/local.yaml: %s", window_title)
        else:
            rprint("[red]未指定 --title，且 configs/local.yaml 不存在[/red]")
            rprint("[dim]提示：先运行 wlxq-bot save-window 保存窗口配置[/dim]")
            raise typer.Exit(1)
    else:
        logger.debug("窗口标题来自 --title 参数: %s", window_title)

    logger.info("开始截图，目标窗口: %s", window_title)

    # 查找窗口
    from wlxq_bot.perception.screen import ScreenCapture

    capturer = ScreenCapture()
    handle = capturer.find_window(window_title)
    if not handle:
        logger.error("未找到窗口: %s", window_title)
        rprint(f"[red]未找到窗口: {window_title}[/red]")
        rprint("[dim]提示：用 wlxq-bot inspect --all 查看所有可见窗口[/dim]")
        raise typer.Exit(1)

    logger.debug("已定位窗口句柄: %s", handle)

    # 截图
    try:
        ctx, frame = capturer.capture(handle)
    except RuntimeError as e:
        logger.error("截图失败 handle=%s 原因=%s", handle, e)
        rprint(f"[red]截图失败: {e}[/red]")
        raise typer.Exit(1) from e

    logger.debug(
        "截图完成 frame_id=%s 客户区=%dx%d 句柄=%s",
        ctx.frame_id,
        ctx.client_size[0],
        ctx.client_size[1],
        ctx.window_handle,
    )

    # 确定输出路径
    from datetime import datetime

    if output is None:
        output_dir = Path("screenshots/raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = str(output_dir / f"screenshot_{timestamp}.png")

    # 保存
    # MSS 返回的是 BGRA，需要转 RGB
    import numpy as np
    from PIL import Image

    if isinstance(frame, np.ndarray):
        img = Image.fromarray(frame[..., [2, 1, 0]], mode="RGB")
    else:
        img = Image.fromarray(frame)

    img.save(output)
    logger.info(
        "截图已保存 path=%s size=%dx%d frame_id=%s",
        output,
        ctx.client_size[0],
        ctx.client_size[1],
        ctx.frame_id,
    )

    rprint("[green]✓ 截图已保存[/green]")
    rprint(
        f"  文件: [white]{output}[/white]\n"
        f"  尺寸: [green]{ctx.client_size[0]} × {ctx.client_size[1]}[/green]\n"
        f"  句柄: {ctx.window_handle}\n"
        f"  帧号: {ctx.frame_id}"
    )

    if show:
        import os

        os.startfile(output)  # type: ignore[attr-defined]


def _acquire_frame(image: str | None) -> tuple[object, str]:
    """获取识别用画面。

    image 为 None 时实时截取当前游戏窗口；否则读取指定截图文件。
    返回 (frame_bgr, source_label)。失败时打印错误并 raise typer.Exit(1)。

    实时截图走 ScreenCapture + local.yaml 窗口标题；MSS 返回 BGRA 四通道，
    取前 3 通道转 BGR 给 cv2。离线读取用 np.fromfile + imdecode 绕过中文路径限制。
    """
    import cv2
    import numpy as np

    if image is None:
        enable_dpi_awareness()
        from wlxq_bot.perception.screen import ScreenCapture

        local_config = load_local_config(LOCAL_CONFIG_PATH)
        if local_config is not None:
            window_title = local_config.window.title
            logger.debug("窗口标题来自 configs/local.yaml: %s", window_title)
        else:
            window_title = "永远的蔚蓝星球"
            logger.debug("未读取到 local.yaml，使用默认窗口标题: %s", window_title)

        capturer = ScreenCapture()
        handle = capturer.find_window(window_title)
        if not handle:
            logger.error("未找到窗口: %s", window_title)
            rprint(f"[red]未找到窗口: {window_title}[/red]")
            rprint("[dim]提示：用 wlxq-bot inspect --all 查看所有可见窗口[/dim]")
            raise typer.Exit(1)

        try:
            ctx, raw_frame = capturer.capture(handle)
        except RuntimeError as e:
            logger.error("截图失败 handle=%s 原因=%s", handle, e)
            rprint(f"[red]截图失败: {e}[/red]")
            raise typer.Exit(1) from e

        # MSS 返回 BGRA 四通道，取前 3 通道转 BGR 给 cv2
        if raw_frame.ndim == 3 and raw_frame.shape[2] == 4:
            frame = raw_frame[:, :, :3]
        else:
            frame = raw_frame
        logger.info(
            "实时截图完成 frame_id=%s 客户区=%dx%d",
            ctx.frame_id,
            frame.shape[1],
            frame.shape[0],
        )
        return frame, f"实时帧#{ctx.frame_id}"

    # 离线读取截图文件
    img_path = Path(image)
    if not img_path.is_file():
        logger.error("截图文件不存在: %s", img_path)
        rprint(f"[red]截图文件不存在: {img_path}[/red]")
        raise typer.Exit(1)

    try:
        data = np.fromfile(str(img_path), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError as e:
        logger.error("截图读取失败 path=%s 原因=%s", img_path, e)
        rprint(f"[red]截图读取失败: {e}[/red]")
        raise typer.Exit(1) from e

    if frame is None:
        logger.error("截图解码失败（非有效图片）: %s", img_path)
        rprint(f"[red]无法解码图片: {img_path}[/red]")
        raise typer.Exit(1)

    logger.debug("截图已加载 path=%s 尺寸=%dx%d", img_path, frame.shape[1], frame.shape[0])
    return frame, img_path.name


@app.command()
def find(
    template: str = typer.Argument(..., help="模板图片路径"),
    threshold: float = typer.Option(
        0.85,
        "--threshold",
        "-t",
        help="识别度阈值 0~1，1 为完全匹配；越低越容易匹配但可能误识别",
    ),
    image: str | None = typer.Option(
        None,
        "--image",
        "-i",
        help="指定截图文件离线识别；不指定则实时截取当前游戏窗口",
    ),
    save: str | None = typer.Option(
        None,
        "--save",
        "-s",
        help="标注图保存路径；不指定则自动到 screenshots/debug/",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        help="识别后自动用图片查看器打开标注图",
    ),
) -> None:
    """在当前游戏画面中识别指定模板，报告是否匹配到及置信度。

    实时截取当前游戏窗口画面（或识别指定截图），用模板匹配查找目标，
    返回匹配位置和置信度（0~1，1 为完全匹配）。用于验证单个模板
    是否能在实际游戏画面中被正常识别。

    示例::

        wlxq-bot find assets/templates/3000x2000/buttons/cai_hong_2.png
        wlxq-bot find cai_hong_2.png -t 0.7            # 调低阈值更容易匹配
        wlxq-bot find cai_hong_2.png -t 0.95 --show     # 高阈值 + 打开标注图
        wlxq-bot find cai_hong_2.png --image shot.png   # 离线识别指定截图
    """
    import cv2

    from wlxq_bot.perception.vision import Vision

    logger.info(
        "find 开始 template=%s threshold=%.2f image=%s",
        template,
        threshold,
        image,
    )

    if not 0.0 <= threshold <= 1.0:
        logger.error("threshold 超出范围: %s", threshold)
        rprint(f"[red]--threshold 必须在 0~1 之间，当前 {threshold}[/red]")
        raise typer.Exit(1)

    tpl_path = Path(template)
    if not tpl_path.is_file():
        logger.error("模板文件不存在: %s", tpl_path)
        rprint(f"[red]模板文件不存在: {tpl_path}[/red]")
        raise typer.Exit(1)

    # 1. 获取画面 + 加载模板
    frame, source_label = _acquire_frame(image)
    fh, fw = frame.shape[:2]

    vision = Vision()
    tpl_img = vision._load_template(str(tpl_path))
    if tpl_img is None:
        logger.error("模板加载失败: %s", tpl_path)
        rprint(f"[red]模板加载失败: {tpl_path}[/red]")
        raise typer.Exit(1)

    th, tw = tpl_img.shape[:2]
    logger.debug("模板已加载 path=%s 尺寸=%dx%d 画面尺寸=%dx%d", tpl_path, tw, th, fw, fh)

    rprint(
        f"[bold]模板[/bold] {tpl_path.name}  [green]{tw} × {th}[/green]  "
        f"[bold]画面[/bold] {source_label}  [green]{fw} × {fh}[/green]  "
        f"[bold]阈值[/bold] [yellow]{threshold:.2f}[/yellow]"
    )

    # 2. 模板比画面大时无法匹配
    if th > fh or tw > fw:
        logger.error("模板(%dx%d)大于画面(%dx%d) 无法匹配", tw, th, fw, fh)
        rprint(f"[red]模板({tw}×{th})大于画面({fw}×{fh})，无法匹配[/red]")
        raise typer.Exit(1)

    # 3. 模板匹配（单模板，取最高置信度）
    result = vision.match_template(frame, str(tpl_path), threshold=threshold)

    if result is None:
        # 未达阈值，仍算一下最高置信度供参考
        search = frame
        res = cv2.matchTemplate(search, tpl_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        logger.info(
            "find 未命中 模板=%s 最高置信度=%.3f 阈值=%.2f",
            tpl_path.name,
            float(max_val),
            threshold,
        )
        rprint(
            f"[red]✗ 未识别到[/red]  最高置信度 [yellow]{float(max_val):.3f}[/yellow] "
            f"（未达阈值 {threshold:.2f}）"
        )
        rprint("[dim]可降低 --threshold 重试，或检查模板/画面分辨率是否一致[/dim]")
    else:
        cx, cy = result.position
        logger.info(
            "find 命中 模板=%s 置信度=%.3f 位置=(%d,%d)",
            tpl_path.name,
            result.confidence,
            cx,
            cy,
        )
        rprint(
            f"[green]✓ 已识别到[/green]  置信度 [green]{result.confidence:.3f}[/green]  "
            f"位置 ({cx}, {cy})"
        )

    # 4. 生成标注图：在最高置信度位置画框（无论是否命中）
    annotated = frame.copy()
    if result is not None:
        cx, cy = result.position
    else:
        cx = max_loc[0] + tw // 2
        cy = max_loc[1] + th // 2
    conf = result.confidence if result is not None else float(max_val)
    color = (0, 255, 0) if result is not None else (0, 0, 255)
    cv2.rectangle(annotated, (cx - tw // 2, cy - th // 2), (cx + tw // 2, cy + th // 2), color, 2)
    cv2.putText(
        annotated,
        f"{tpl_path.name} {conf:.3f}",
        (cx - tw // 2, cy - th // 2 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )

    # 5. 保存标注图
    from datetime import datetime

    if save is None:
        debug_dir = Path("screenshots/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save = str(debug_dir / f"find_{tpl_path.stem}_{timestamp}.png")

    save_path = Path(save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    success, buf = cv2.imencode(".png", annotated)
    if success:
        buf.tofile(str(save_path))
        logger.info("标注图已保存 path=%s", save_path)
    else:
        logger.error("标注图保存失败 path=%s", save_path)
        rprint(f"[red]标注图保存失败: {save_path}[/red]")

    rprint(f"标注图 [white]{save_path}[/white]")

    if show:
        import os

        os.startfile(str(save_path))  # type: ignore[attr-defined]


@app.command()
def calibrate() -> None:
    """启动首次校准流程，采集本机模板。

    TODO: 实现交互式模板采集。
    """
    rprint("[yellow]calibrate 命令尚未实现[/yellow]")


@app.command()
def run(
    task: str = typer.Argument("coop", help="任务名称，例如 coop"),
    main_c: str | None = typer.Option(
        None, "--main-c", help="主 C 标识：assault / monkey / angel / snow / death_knight / fox"
    ),
    coop_difficulties: str | None = typer.Option(
        None,
        "--coop-difficulties",
        help="覆盖配置中的合作难度范围，例如 1-16 或 1-10（编号指彩虹难度）；按从小到大依次选择",
    ),
    max_rounds: int | None = typer.Option(
        None,
        "--max-rounds",
        "-n",
        min=1,
        help="覆盖配置中的最大局数（默认 20）；仅本次运行生效，不修改配置文件",
    ),
    skip_difficulty_selection: bool | None = typer.Option(
        None,
        "--skip-difficulty-selection",
        "-d",
        help="跳过合作难度勾选（游戏本次会话已手动选过难度时使用）；"
        "难度弹窗仍打开并关闭一次以刷新最新合作邀请。"
        "不传时使用配置 run.skip_difficulty_selection",
    ),
    start_state: str = typer.Option(
        "find_coop",
        "--start-state",
        "-s",
        help="初始状态（调试用），默认 find_coop；传 build_main_c 可跳过抢合作直接测培养闭环",
    ),
) -> None:
    """执行一个自动化任务。

    目前支持合作任务（coop）。启动后自动寻找合作、进入游戏、培养主C，
    直到完成 max_rounds 局或触发停止条件。运行中可用 Ctrl+C 中断。

    首版培养主C闭环（召唤→识别棋盘→合成→达到目标星级）已可用；
    抢合作、技能选择和结算环节随模板素材补充逐步激活。

    --start-state 用于调试：抢合作等环节模板未采集时，可传 build_main_c
    跳过抢合作，直接在棋盘界面验证培养闭环（需先手动进入游戏到棋盘界面）。

    --skip-difficulty-selection 用于游戏本次会话已手动选过难度的场景：
    首局招募只跳过勾选难度等级，难度弹窗仍打开并关闭一次刷新最新合作邀请。

    示例::

        wlxq-bot run coop --main-c assault
        wlxq-bot run coop --main-c assault --coop-difficulties 1-10
        wlxq-bot run coop --main-c assault --max-rounds 3   # 或简写 -n 3
        wlxq-bot run coop --main-c assault --skip-difficulty-selection   # 或简写 -d
        wlxq-bot --debug run coop --main-c assault
        wlxq-bot run coop --main-c assault -s build_main_c
    """
    from pathlib import Path

    from wlxq_bot.config import load_default_config, load_tasks_config
    from wlxq_bot.runner import Runner

    enable_dpi_awareness()
    logger.info(
        "run 开始 task=%s main_c=%s coop_difficulties=%s skip_difficulty_selection=%s max_rounds=%s",
        task,
        main_c or "(默认)",
        coop_difficulties or "(配置)",
        "未指定" if skip_difficulty_selection is None else skip_difficulty_selection,
        max_rounds if max_rounds is not None else "(配置)",
    )

    try:
        default_cfg = load_default_config(Path("configs/default.yaml"))
        tasks_cfg = load_tasks_config(Path("configs/tasks.yaml"))
    except Exception as exc:
        logger.error("配置加载失败: %r", exc)
        rprint(f"[red]配置加载失败: {exc}[/red]")
        raise typer.Exit(1) from exc

    local_cfg = load_local_config(LOCAL_CONFIG_PATH)
    if local_cfg is None:
        logger.warning("configs/local.yaml 不存在，将使用默认窗口标题和自动模板包选择")
    if not tasks_cfg.hotspots:
        logger.warning("configs/tasks.yaml 的 hotspots 为空，部分动作无法执行")
        rprint("[yellow]提示：configs/tasks.yaml 尚未填写通用 hotspots[/yellow]")

    runner = Runner(default_cfg, tasks_cfg, local_cfg)
    try:
        final = runner.run(
            task,
            main_c,
            start_state=start_state,
            coop_difficulties=coop_difficulties,
            skip_difficulty_selection=skip_difficulty_selection,
            max_rounds=max_rounds,
        )
    except (ValueError, RuntimeError) as exc:
        logger.error("任务执行失败: %r", exc)
        rprint(f"[red]任务执行失败: {exc}[/red]")
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:
        logger.warning("用户中断任务")
        rprint("[yellow]已中断任务[/yellow]")
        raise

    rprint(f"[green]任务结束，最终状态: {final.value}[/green]")


@exec_app.command(name="select-difficulty")
def exec_select_difficulty(
    coop_difficulties: str | None = typer.Option(
        None,
        "--coop-difficulties",
        help="覆盖配置中的合作难度范围，例如 1-16 或 1-10（编号指彩虹难度）；按从小到大依次点击",
    ),
) -> None:
    """在已打开的难度弹窗中，独立验证难度识别、点击和滚动。

    运行前请手动进入招募页面并打开选择难度弹窗。该能力只点击目标难度，
    完成后不会关闭弹窗，也不会继续抢合作。

    示例::

        wlxq-bot exec select-difficulty --coop-difficulties 1-10
        wlxq-bot --debug exec select-difficulty --coop-difficulties 1-16
    """
    from pathlib import Path

    from wlxq_bot.config import load_default_config, load_tasks_config
    from wlxq_bot.models import State
    from wlxq_bot.runner import Runner

    enable_dpi_awareness()
    logger.info(
        "exec select-difficulty 开始 coop_difficulties=%s",
        coop_difficulties or "(配置)",
    )
    try:
        default_cfg = load_default_config(Path("configs/default.yaml"))
        tasks_cfg = load_tasks_config(Path("configs/tasks.yaml"))
    except Exception as exc:
        logger.error("配置加载失败: %r", exc)
        rprint(f"[red]配置加载失败: {exc}[/red]")
        raise typer.Exit(1) from exc

    local_cfg = load_local_config(LOCAL_CONFIG_PATH)
    if not tasks_cfg.hotspots:
        logger.warning("configs/tasks.yaml 的 hotspots 为空，难度列表无法使用滚动 hotspot")
        rprint(
            "[yellow]提示：configs/tasks.yaml 尚未填写通用 hotspots；当前可识别并点击可见难度，"
            "但目标不可见时无法滚动[/yellow]"
        )

    runner = Runner(default_cfg, tasks_cfg, local_cfg)
    try:
        final = runner.select_difficulty(coop_difficulties)
    except (ValueError, RuntimeError) as exc:
        logger.error("独立难度选择失败: %r", exc)
        rprint(f"[red]独立难度选择失败: {exc}[/red]")
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:
        logger.warning("用户中断独立难度选择")
        rprint("[yellow]已中断难度选择[/yellow]")
        raise

    if final != State.COMPLETED:
        logger.warning("exec select-difficulty 未完成，最终状态=%s", final.value)
        rprint("[red]难度选择未完成，请使用 --debug 查看识别或滚动失败原因[/red]")
        raise typer.Exit(1)
    logger.info("exec select-difficulty 结束")
    rprint("[green]目标合作难度已全部点击；弹窗保持打开[/green]")


@exec_app.command(name="watch-board")
def exec_watch_board(
    main_c: str = typer.Option("assault", "--main-c", help="主 C 英文标识，决定加载的分类模型"),
    interval: float = typer.Option(1.0, "--interval", min=0.2, help="两轮识别间隔秒数"),
) -> None:
    """在真实对局中循环识别己方棋盘，实时打印每轮结果，按 Esc 停止。

    与实战共用同一识别路径（多帧截图 + 按格多数投票），不执行任何点击，
    用于独立验证棋盘识别效果。运行前请先进入对局并保持游戏窗口前台。
    终端失焦时 Ctrl+C 可能收不到，因此停止以 Esc 为准（Esc 监听不可用时
    退回 Ctrl+C）。

    示例::

        wlxq-bot exec watch-board --main-c assault
        wlxq-bot --debug exec watch-board --interval 0.5
    """
    from pathlib import Path

    from wlxq_bot.config import load_default_config, load_tasks_config
    from wlxq_bot.runner import Runner

    enable_dpi_awareness()
    logger.info("exec watch-board 开始 main_c=%s interval=%.1fs", main_c, interval)
    try:
        default_cfg = load_default_config(Path("configs/default.yaml"))
        tasks_cfg = load_tasks_config(Path("configs/tasks.yaml"))
    except Exception as exc:
        logger.error("配置加载失败: %r", exc)
        rprint(f"[red]配置加载失败: {exc}[/red]")
        raise typer.Exit(1) from exc

    local_cfg = load_local_config(LOCAL_CONFIG_PATH)
    runner = Runner(default_cfg, tasks_cfg, local_cfg)

    def render(round_no: int, observation) -> bool:
        board = observation.board
        rprint(f"\n[bold]── 第 {round_no} 轮 frame={observation.frame_id} ──[/bold]")
        if board is None or not board.heroes:
            rprint("  [yellow]棋盘为空或识别无结果[/yellow]")
            return True
        # 格名映射：raw_data['cell_labels'] 的键是格子中心坐标的字符串形式，
        # 与 BoardHero.position 同源（都是格子中心像素坐标）
        labels_by_center: dict[str, str] = dict(observation.raw_data.get("cell_labels", {}))
        heroes = sorted(
            board.heroes,
            key=lambda h: (h.position[1], h.position[0]),
        )
        lines = []
        for h in heroes:
            key = f"({h.position[0]}, {h.position[1]})"
            label = labels_by_center.get(key) or labels_by_center.get(str(h.position))
            label = label or f"({h.position[0]},{h.position[1]})"
            lines.append(
                f"  {label:<4} {h.hero_type:<14} {h.star_level}星  conf={h.confidence:.2f}"
            )
        cap = board.capacity
        rprint("\n".join(lines))
        rprint(f"  [dim]占用/总格: {cap.occupied}/{cap.total_slots}[/dim]")
        return True

    try:
        rounds = runner.watch_board(main_c, interval=interval, on_result=render)
    except (ValueError, RuntimeError) as exc:
        logger.error("棋盘识别观察失败: %r", exc)
        rprint(f"[red]棋盘识别观察失败: {exc}[/red]")
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:
        rprint("[yellow]已中断棋盘识别观察[/yellow]")
        raise
    logger.info("exec watch-board 结束 rounds=%d", rounds)
    rprint(f"[green]棋盘识别观察结束，共 {rounds} 轮[/green]")


@app.command()
def inspect(
    keyword: str = typer.Option(
        "蔚蓝",
        "--keyword",
        "-k",
        help="窗口标题或类名关键字，默认「蔚蓝」",
    ),
    title: str | None = typer.Option(
        None,
        "--title",
        "-t",
        help="精确匹配窗口标题（优先于关键字）",
    ),
    all_windows: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="列出所有可见顶层窗口（忽略关键字）",
    ),
) -> None:
    """检查游戏窗口信息：句柄、标题、类名、客户区尺寸、DPI、前台状态。

    用于验证微信小游戏窗口是否可被找到、客户区尺寸是否符合预期、
    DPI 感知是否正常。这是后续截图和坐标换算的基础。

    示例::

        wlxq-bot inspect                    # 用默认关键字「蔚蓝」查找
        wlxq-bot inspect -k 微信            # 换关键字
        wlxq-bot inspect -t "永远的蔚蓝星球"  # 精确标题匹配
        wlxq-bot inspect --all              # 列出所有可见窗口
    """
    enable_dpi_awareness()
    console = Console()

    if all_windows:
        logger.info("inspect --all 列出所有可见顶层窗口")
        windows = list_windows(include_invisible=False)
        logger.debug("列出 %d 个可见顶层窗口", len(windows))
        _print_window_table(console, windows, title="所有可见顶层窗口")
        return

    if title:
        logger.info("inspect 按精确标题查找: %s", title)
        handle = find_window_by_title(title)
        if not handle:
            logger.error("未找到标题精确匹配的窗口: %s", title)
            rprint(f"[red]未找到标题精确匹配的窗口: {title}[/red]")
            return
        info = get_window_info(handle)
        logger.debug(
            "命中窗口 handle=%s 类名=%s 客户区=%dx%d",
            info.handle,
            info.class_name,
            info.client_size[0],
            info.client_size[1],
        )
        _print_window_detail(console, info)
        return

    logger.info("inspect 按关键字查找: %s", keyword)
    rprint(f"[dim]按关键字「{keyword}」查找窗口...[/dim]")
    windows = find_windows_by_keyword(keyword)

    if not windows:
        logger.warning("未找到包含关键字「%s」的窗口", keyword)
        rprint(f"[yellow]未找到包含「{keyword}」的窗口[/yellow]")
        rprint("[dim]提示：用 wlxq-bot inspect --all 查看所有可见窗口[/dim]")
        return

    logger.debug("关键字「%s」命中 %d 个窗口", keyword, len(windows))
    _print_window_table(console, windows, title=f"匹配「{keyword}」的窗口")


def _print_window_table(
    console: Console,
    windows: list,
    title: str,
) -> None:
    """以竖排列表形式打印窗口信息。

    每个窗口一块，不受终端宽度限制。
    """
    console.rule(f"[bold]{title}[/bold]")

    for i, w in enumerate(windows, 1):
        client = f"{w.client_size[0]} × {w.client_size[1]}"
        states = []
        if w.is_foreground:
            states.append("前台")
        if w.is_minimized:
            states.append("最小化")
        elif w.is_visible:
            states.append("可见")
        status = " ".join(states) if states else "—"

        # 判断是否符合支持的目标尺寸
        size_tag = ""
        if w.client_size[0] > 0 and w.client_size[1] > 0:
            if (w.client_size[0], w.client_size[1]) in [
                (927, 1727),
                (1920, 1080),
                (2560, 1440),
                (3000, 2000),
            ]:
                size_tag = " [green]✓目标尺寸[/green]"
            elif abs(w.client_size[0] / w.client_size[1] - 16 / 9) < 0.01:
                size_tag = " [yellow]16:9[/yellow]"
            else:
                size_tag = " [dim]非16:9[/dim]"

        console.print(
            f"[cyan]#{i}[/cyan] 句柄 [bold]{w.handle}[/bold]  "
            f"客户区 [green]{client}[/green]{size_tag}  "
            f"DPI [yellow]{w.dpi}[/yellow]  "
            f"[magenta]{status}[/magenta]  "
            f"PID [dim]{w.process_id}[/dim]"
        )
        console.print(
            f"  标题: [white]{w.title or '(空)'}[/white]\n  类名: [dim]{w.class_name}[/dim]"
        )


def _print_window_detail(console: Console, info) -> None:
    """打印单个窗口的详细信息。"""
    console.rule(f"[bold cyan]窗口详情: {info.title or '(无标题)'}[/bold cyan]")

    client_w, client_h = info.client_size
    cl, ct, cw, ch = info.client_rect
    wl, wt, wr, wb = info.window_rect

    # 尺寸标注
    size_note = ""
    if client_h > 0 and client_w > 0:
        if (client_w, client_h) in [(927, 1727), (1920, 1080), (2560, 1440), (3000, 2000)]:
            size_note = "  [bold green]✓ 目标尺寸[/bold green]"
        elif abs(client_w / client_h - 16 / 9) < 0.01:
            size_note = "  [green]16:9[/green]"
        else:
            size_note = "  [yellow]非 16:9[/yellow]"

    lines = [
        f"  [bold]句柄[/bold]          {info.handle}",
        f"  [bold]标题[/bold]          {info.title or '(空)'}",
        f"  [bold]类名[/bold]          {info.class_name}",
        f"  [bold]进程 ID[/bold]       {info.process_id}",
        f"  [bold]线程 ID[/bold]       {info.thread_id}",
        f"  [bold]可见 / 前台 / 最小化[/bold]  "
        f"{'是' if info.is_visible else '否'} / "
        f"{'是' if info.is_foreground else '否'} / "
        f"{'是' if info.is_minimized else '否'}",
        f"  [bold]DPI[/bold]           {info.dpi} ({info.dpi / 96 * 100:.0f}% 缩放)",
        f"  [bold]客户区尺寸[/bold]    {client_w} × {client_h} px{size_note}",
        f"  [bold]客户区屏幕位置[/bold]  ({cl}, {ct})  尺寸 {cw} × {ch}",
        f"  [bold]窗口矩形[/bold]      ({wl}, {wt}) → ({wr}, {wb})  尺寸 {wr - wl} × {wb - wt}",
        f"  [bold]边框/标题栏[/bold]    左 {cl - wl}, 右 {wr - cl - cw}, "
        f"上 {ct - wt}, 下 {wb - ct - ch}",
    ]

    # 目标尺寸匹配判断
    supported = [(927, 1727), (1920, 1080), (2560, 1440), (3000, 2000)]
    if (client_w, client_h) in supported:
        lines.append(
            f"  [bold]目标尺寸匹配[/bold]  [bold green]✓ 匹配 {client_w}x{client_h}[/bold green]"
        )
    else:
        nearest = min(supported, key=lambda s: abs(s[0] - client_w) + abs(s[1] - client_h))
        lines.append(
            f"  [bold]目标尺寸匹配[/bold]  [yellow]✗ 不匹配，最接近 {nearest[0]}x{nearest[1]}[/yellow]"
        )

    console.print("\n".join(lines))


@app.command(name="save-window")
def save_window(
    title: str = typer.Option(
        "永远的蔚蓝星球",
        "--title",
        "-t",
        help="游戏窗口标题",
    ),
    width: int | None = typer.Option(
        None,
        "--width",
        "-w",
        help="目标客户区宽度，不指定则用当前窗口尺寸",
    ),
    height: int | None = typer.Option(
        None,
        "--height",
        "-h",
        help="目标客户区高度，不指定则用当前窗口尺寸",
    ),
    template_pack: str | None = typer.Option(
        None,
        "--template-pack",
        "-p",
        help="模板包目录名；不指定时使用游戏窗口所在显示器的物理分辨率",
    ),
) -> None:
    """保存当前游戏窗口信息到 configs/local.yaml。

    后续 adjust-window 命令会读取此配置，自动调整窗口到目标尺寸。
    不指定 --width/--height 时，保存当前实际客户区尺寸作为目标。

    示例::

        wlxq-bot save-window                          # 保存当前窗口尺寸
        wlxq-bot save-window -w 924 -h 1723           # 指定目标尺寸
        wlxq-bot save-window -t "永远的蔚蓝星球"       # 指定窗口标题
    """
    enable_dpi_awareness()
    logger.info(
        "save-window 开始 title=%s width=%s height=%s template_pack=%s",
        title,
        width,
        height,
        template_pack,
    )

    handle = find_window_by_title(title)
    if not handle:
        logger.error("未找到窗口: %s", title)
        rprint(f"[red]未找到窗口: {title}[/red]")
        rprint("[dim]提示：用 wlxq-bot inspect --all 查看所有可见窗口[/dim]")
        raise typer.Exit(1)

    info = get_window_info(handle)
    logger.debug(
        "命中窗口 handle=%s 当前客户区=%dx%d",
        info.handle,
        info.client_size[0],
        info.client_size[1],
    )

    target_w = width if width is not None else info.client_size[0]
    target_h = height if height is not None else info.client_size[1]
    if template_pack is None:
        monitor_w, monitor_h = get_window_monitor_resolution(handle)
        template_pack = f"{monitor_w}x{monitor_h}"

    spec = WindowSpec(
        title=info.title,
        class_name=info.class_name,
        target_client_width=target_w,
        target_client_height=target_h,
        template_pack=template_pack,
    )
    local_config = LocalConfig(window=spec)

    save_local_config(LOCAL_CONFIG_PATH, local_config)
    logger.info("窗口配置已保存 path=%s 目标客户区=%dx%d", LOCAL_CONFIG_PATH, target_w, target_h)

    rprint(f"[green]✓ 已保存窗口配置到 {LOCAL_CONFIG_PATH}[/green]")
    rprint(
        f"  标题: [white]{info.title}[/white]\n"
        f"  类名: [dim]{info.class_name}[/dim]\n"
        f"  目标客户区: [green]{target_w} × {target_h}[/green]\n"
        f"  模板包: [green]{template_pack}[/green]"
    )
    if width is None and height is None:
        rprint("[dim]  (使用当前窗口实际尺寸作为目标)[/dim]")


@app.command(name="adjust-window")
def adjust_window_cmd(
    config_path: str = typer.Option(
        "configs/local.yaml",
        "--config",
        "-c",
        help="本地配置文件路径",
    ),
) -> None:
    """按 configs/local.yaml 调整游戏窗口到目标尺寸。

    启动游戏后执行此命令，自动把窗口调整到之前保存的尺寸。
    这样每次截图识别都能使用同一套模板。

    示例::

        wlxq-bot adjust-window                        # 用默认配置调整
        wlxq-bot adjust-window -c configs/local.yaml  # 指定配置路径
    """
    enable_dpi_awareness()

    path = Path(config_path)
    local_config = load_local_config(path)
    if local_config is None:
        logger.error("配置文件不存在: %s", path)
        rprint(f"[red]配置文件不存在: {path}[/red]")
        rprint("[dim]提示：先运行 wlxq-bot save-window 保存窗口配置[/dim]")
        raise typer.Exit(1)

    spec = local_config.window
    logger.info(
        "adjust-window 开始 目标=%s 类名=%s 目标客户区=%dx%d",
        spec.title,
        spec.class_name,
        spec.target_client_width,
        spec.target_client_height,
    )
    rprint(f"[dim]目标窗口: {spec.title}[/dim]")
    rprint(f"[dim]目标客户区: {spec.target_client_width} × {spec.target_client_height}[/dim]")

    handle = find_window_smart(spec.title, spec.class_name)
    if not handle:
        logger.error("未找到窗口: %s 类名=%s", spec.title, spec.class_name)
        rprint(f"[red]未找到窗口: {spec.title}[/red]")
        rprint("[dim]提示：确认游戏已启动，用 wlxq-bot inspect --all 查看可用窗口[/dim]")
        raise typer.Exit(1)

    info = get_window_info(handle)
    logger.debug(
        "命中窗口 handle=%s 当前客户区=%dx%d",
        info.handle,
        info.client_size[0],
        info.client_size[1],
    )
    rprint(f"[dim]当前客户区: {info.client_size[0]} × {info.client_size[1]}[/dim]")

    if info.client_size == (spec.target_client_width, spec.target_client_height):
        logger.info("窗口尺寸已符合目标，无需调整")
        rprint("[green]✓ 窗口尺寸已符合目标，无需调整[/green]")
        return

    if info.is_minimized:
        logger.error("窗口已最小化，无法调整 handle=%s", info.handle)
        rprint("[red]窗口已最小化，请先恢复窗口再调整[/red]")
        raise typer.Exit(1)

    try:
        adjusted = adjust_window_size(
            handle,
            spec.target_client_width,
            spec.target_client_height,
        )
    except RuntimeError as e:
        logger.error("adjust_window_size 失败 handle=%s 原因=%s", info.handle, e)
        rprint(f"[red]调整失败: {e}[/red]")
        raise typer.Exit(1) from e

    logger.info(
        "窗口已调整 调整前=%dx%d 调整后=%dx%d",
        info.client_size[0],
        info.client_size[1],
        adjusted.client_size[0],
        adjusted.client_size[1],
    )
    rprint(
        f"[green]✓ 窗口已调整[/green]\n"
        f"  调整前: {info.client_size[0]} × {info.client_size[1]}\n"
        f"  调整后: [bold green]{adjusted.client_size[0]} × {adjusted.client_size[1]}[/bold green]"
    )

    if adjusted.client_size != (spec.target_client_width, spec.target_client_height):
        logger.warning(
            "实际客户区 %dx%d 与目标 %dx%d 不完全一致",
            adjusted.client_size[0],
            adjusted.client_size[1],
            spec.target_client_width,
            spec.target_client_height,
        )
        rprint(
            f"[yellow]⚠ 实际客户区 {adjusted.client_size[0]}×{adjusted.client_size[1]} "
            f"与目标 {spec.target_client_width}×{spec.target_client_height} 不完全一致[/yellow]"
        )
        rprint("[dim]  可能是窗口最小尺寸限制或边框计算误差导致[/dim]")


@app.command()
def click(
    target: str = typer.Argument(..., help="位置名称（如 join_coop）或 x 比例坐标"),
    y_ratio: float | None = typer.Argument(None, help="y 比例坐标，仅在 target 为数字时使用"),
    count: int = typer.Option(
        1, "--count", "-n", help="点击次数，默认 1 次；>1 时按 --interval 间隔重复点击"
    ),
    interval: float = typer.Option(
        0.4, "--interval", "-i", help="多次点击时相邻两次之间的间隔（秒）"
    ),
    delay: float = typer.Option(
        0.3, "--delay", "-d", help="点击前延迟（秒），给用户切回游戏窗口的时间"
    ),
) -> None:
    """按比例坐标点击游戏窗口。

    支持两种方式：
    1. 用名字：从 configs/tasks.yaml 的 hotspots 读取通用坐标
    2. 用比例：直接传 x y 比例坐标

    默认只点一次。用 ``--count``/``-n`` 可重复点击多次，
    多次之间用 ``--interval``/``-i`` 控制间隔（秒）。

    示例::

        wlxq-bot click join_coop                 # 用名字点击「加入」按钮
        wlxq-bot click 0.6084 0.7597          # 用比例坐标点击
        wlxq-bot click join_coop -d 2            # 延迟2秒后点击
        wlxq-bot click add_hero -n 5          # 连点 5 次增加英雄
        wlxq-bot click add_hero -n 5 -i 0.6   # 连点 5 次，每次间隔 0.6 秒
    """
    enable_dpi_awareness()
    import time

    logger.info(
        "click 开始 target=%s y_ratio=%s count=%d interval=%.2fs delay=%ss",
        target,
        y_ratio,
        count,
        interval,
        delay,
    )

    # 读取窗口配置
    local_config = load_local_config(LOCAL_CONFIG_PATH)
    hotspots = load_tasks_config(TASKS_CONFIG_PATH).hotspots
    if local_config is not None:
        title = local_config.window.title
        class_name = local_config.window.class_name
    else:
        title = "永远的蔚蓝星球"
        class_name = "Chrome_WidgetWin_0"
        logger.debug("未读取到 local.yaml，使用默认窗口标题/类名")

    # 解析 target：名字 or 数字
    hotspot_desc = ""
    try:
        # 尝试解析为数字
        x_ratio_val = float(target)
        if y_ratio is None:
            logger.warning("使用比例坐标时缺少 y")
            rprint("[red]使用比例坐标时需要同时传 x 和 y[/red]")
            rprint("[dim]示例: wlxq-bot click 0.6084 0.7597[/dim]")
            raise typer.Exit(1)
        x_ratio_final = x_ratio_val
        y_ratio_final = y_ratio
    except ValueError:
        # 不是数字，当作 hotspot 名字查找
        if target not in hotspots:
            logger.error("未知位置名称: %s", target)
            rprint(f"[red]未知位置名称: {target}[/red]")
            if hotspots:
                rprint("[dim]可用位置:[/dim]")
                for name, hs in hotspots.items():
                    desc = f" - {hs.description}" if hs.description else ""
                    rprint(f"  {name}  ({hs.x_ratio:.4f}, {hs.y_ratio:.4f}){desc}")
            else:
                rprint("[dim]configs/tasks.yaml 中没有定义 hotspots[/dim]")
            raise typer.Exit(1) from None
        spot = hotspots[target]
        x_ratio_final = spot.x_ratio
        y_ratio_final = spot.y_ratio
        hotspot_desc = f" ({target}" + (f": {spot.description}" if spot.description else "") + ")"
        logger.debug("命中 hotspot %s 比例=(%.4f, %.4f)", target, x_ratio_final, y_ratio_final)

    handle = find_window_smart(title, class_name)
    if not handle:
        logger.error("未找到窗口: %s 类名=%s", title, class_name)
        rprint(f"[red]未找到窗口: {title}[/red]")
        raise typer.Exit(1)

    info = get_window_info(handle)
    if info.is_minimized:
        logger.error("窗口已最小化 handle=%s", info.handle)
        rprint("[red]窗口已最小化，请先恢复窗口[/red]")
        raise typer.Exit(1)

    cl, ct, cw, ch = info.client_rect
    px = int(x_ratio_final * cw)
    py = int(y_ratio_final * ch)
    sx = cl + px
    sy = ct + py
    logger.debug(
        "坐标换算 客户区=%dx%d 比例=(%.4f,%.4f) 像素=(%d,%d) 屏幕=(%d,%d)",
        cw,
        ch,
        x_ratio_final,
        y_ratio_final,
        px,
        py,
        sx,
        sy,
    )

    rprint(f"窗口: {info.title}  客户区: {cw} × {ch}")
    rprint(
        f"比例: ({x_ratio_final:.4f}, {y_ratio_final:.4f})  像素: ({px}, {py})  屏幕: ({sx}, {sy}){hotspot_desc}"
    )

    if delay > 0:
        rprint(f"[yellow]{delay}秒后点击，请勿移动鼠标...[/yellow]")
        time.sleep(delay)

    # 先把窗口设为前台
    if not info.is_foreground:
        logger.debug("窗口不在前台，激活 handle=%s", info.handle)
        rprint("[dim]窗口不在前台，尝试激活...[/dim]")
        import win32con
        import win32gui

        win32gui.ShowWindow(handle, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(handle)
        import time as _time

        _time.sleep(0.3)

    from wlxq_bot.models import Action

    screen, executor = _manual_action_components()
    last_screen_point = (sx, sy)
    for _i in range(count):
        if _i > 0 and interval > 0:
            time.sleep(interval)
        ctx, _frame = screen.capture(handle)
        current_px = int(x_ratio_final * ctx.client_size[0])
        current_py = int(y_ratio_final * ctx.client_size[1])
        result = executor.execute(
            ctx,
            Action(
                kind="click",
                target=(current_px, current_py),
                duration=0.08,
                reason="人工 click 命令",
            ),
        )
        if not result.executed:
            logger.error("点击被安全检查拒绝: %s", result.failure_reason)
            rprint(f"[red]点击被安全检查拒绝: {result.failure_reason}[/red]")
            raise typer.Exit(1)
        last_screen_point = ctx.client_to_screen(current_px, current_py)
        logger.info(
            "已点击 第%d/%d次 frame=%d 屏幕=(%d, %d)",
            _i + 1,
            count,
            ctx.frame_id,
            *last_screen_point,
        )
    rprint(f"[green]✓ 已点击 {count} 次 ({last_screen_point[0]}, {last_screen_point[1]})[/green]")


@app.command(name="spam-click")
def spam_click(
    target: str = typer.Argument(..., help="位置名称（如 join_coop）或 x 比例坐标"),
    y_ratio: float | None = typer.Argument(None, help="y 比例坐标，仅在 target 为数字时使用"),
    min_interval: float = typer.Option(0.15, "--min", help="两次点击最小间隔（秒）"),
    max_interval: float = typer.Option(0.45, "--max", help="两次点击最大间隔（秒）"),
    duration: float = typer.Option(
        0.0, "--duration", "-d", help="持续点击总时长（秒），0 表示按 Esc 停止"
    ),
    jitter: int = typer.Option(5, "--jitter", "-j", help="点击位置随机抖动范围（像素）"),
    burst_chance: float = typer.Option(
        0.15, "--burst", help="连击概率（0~1），模拟人偶尔快速点两下"
    ),
) -> None:
    """模拟人类行为持续点击某个位置。

    随机间隔、位置抖动、偶尔连击，不是机械的固定频率点击。
    用于抢合作等需要持续快速点击的场景。

    示例::

        wlxq-bot spam-click join_coop                  # 持续点击「加入」，按 Esc 停止
        wlxq-bot spam-click join_coop -d 10            # 持续点击10秒后自动停
        wlxq-bot spam-click join_coop --min 0.1 --max 0.3  # 调整间隔范围
        wlxq-bot spam-click 0.6084 0.7597 -d 5     # 用比例坐标，点击5秒
    """
    enable_dpi_awareness()
    logger.info(
        "spam-click 开始 target=%s y_ratio=%s 间隔=%.2f~%.2fs duration=%.1f jitter=%d burst=%.2f",
        target,
        y_ratio,
        min_interval,
        max_interval,
        duration,
        jitter,
        burst_chance,
    )

    # 读取窗口配置
    local_config = load_local_config(LOCAL_CONFIG_PATH)
    hotspots = load_tasks_config(TASKS_CONFIG_PATH).hotspots
    if local_config is not None:
        title = local_config.window.title
        class_name = local_config.window.class_name
    else:
        title = "永远的蔚蓝星球"
        class_name = "Chrome_WidgetWin_0"
        logger.debug("未读取到 local.yaml，使用默认窗口标题/类名")

    # 解析 target
    hotspot_desc = ""
    try:
        x_ratio_val = float(target)
        if y_ratio is None:
            rprint("[red]使用比例坐标时需要同时传 x 和 y[/red]")
            raise typer.Exit(1)
        x_ratio_final = x_ratio_val
        y_ratio_final = y_ratio
    except ValueError:
        if target not in hotspots:
            rprint(f"[red]未知位置名称: {target}[/red]")
            if hotspots:
                for name, hs in hotspots.items():
                    rprint(f"  [dim]{name}  ({hs.x_ratio:.4f}, {hs.y_ratio:.4f})[/dim]")
            raise typer.Exit(1) from None
        spot = hotspots[target]
        x_ratio_final = spot.x_ratio
        y_ratio_final = spot.y_ratio
        hotspot_desc = f" ({target})"

    handle = find_window_smart(title, class_name)
    if not handle:
        rprint(f"[red]未找到窗口: {title}[/red]")
        raise typer.Exit(1)

    info = get_window_info(handle)
    if info.is_minimized:
        rprint("[red]窗口已最小化，请先恢复窗口[/red]")
        raise typer.Exit(1)

    cl, ct, cw, ch = info.client_rect
    px = int(x_ratio_final * cw)
    py = int(y_ratio_final * ch)
    sx = cl + px
    sy = ct + py
    logger.debug(
        "坐标换算 客户区=%dx%d 比例=(%.4f,%.4f) 像素=(%d,%d) 屏幕=(%d,%d)",
        cw,
        ch,
        x_ratio_final,
        y_ratio_final,
        px,
        py,
        sx,
        sy,
    )

    rprint(f"窗口: {info.title}  客户区: {cw} × {ch}")
    rprint(f"目标: ({x_ratio_final:.4f}, {y_ratio_final:.4f}){hotspot_desc}  屏幕: ({sx}, {sy})")
    rprint(f"间隔: {min_interval}~{max_interval}s  抖动: ±{jitter}px  连击概率: {burst_chance:.0%}")

    if duration > 0:
        rprint(f"[yellow]持续 {duration}秒，按 Esc 可提前停止[/yellow]")
    else:
        rprint("[yellow]按 Esc 停止[/yellow]")

    # 激活窗口
    if not info.is_foreground:
        import win32con
        import win32gui

        win32gui.ShowWindow(handle, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(handle)
        import time as _time

        _time.sleep(0.3)

    import ctypes
    import random
    import time

    from wlxq_bot.models import Action

    screen, executor = _manual_action_components(jitter=jitter)

    start_time = time.time()
    click_count = 0

    # Windows 下 Ctrl+C 在后台窗口收不到（控制台被切到后台时键盘事件
    # 不会派发给它），SetConsoleCtrlHandler 也不可靠。
    # 改用 GetAsyncKeyState 读物理按键状态——直接查键盘硬件，
    # 不依赖焦点，不管哪个窗口在前台都能检测到。
    # Esc (VK_ESCAPE = 0x1B) 键小好按，且不会干扰游戏。
    user32 = ctypes.windll.user32
    VK_ESCAPE = 0x1B
    # 循环前先消费掉已按下的 Esc，避免启动即退出
    user32.GetAsyncKeyState(VK_ESCAPE)

    try:
        while True:
            if duration > 0 and (time.time() - start_time) >= duration:
                break

            # 检查 Esc：GetAsyncKeyState 返回值最高位为 1 表示当前正按下
            if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
                break

            ctx, _frame = screen.capture(handle)
            current_px = int(x_ratio_final * ctx.client_size[0])
            current_py = int(y_ratio_final * ctx.client_size[1])
            result = executor.execute(
                ctx,
                Action(
                    kind="click",
                    target=(current_px, current_py),
                    duration=0.08,
                    reason="人工 spam-click 命令",
                ),
            )
            if not result.executed:
                logger.error("持续点击被安全检查拒绝: %s", result.failure_reason)
                break
            click_count += 1

            # 偶尔连击：快速再点一下
            if random.random() < burst_chance:
                _sleep_with_escape(random.uniform(0.05, 0.12), user32, VK_ESCAPE)
                if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
                    break
                ctx, _frame = screen.capture(handle)
                current_px = int(x_ratio_final * ctx.client_size[0])
                current_py = int(y_ratio_final * ctx.client_size[1])
                result = executor.execute(
                    ctx,
                    Action(
                        kind="click",
                        target=(current_px, current_py),
                        duration=0.08,
                        reason="人工 spam-click 连击",
                    ),
                )
                if not result.executed:
                    logger.error("持续点击被安全检查拒绝: %s", result.failure_reason)
                    break
                click_count += 1

            # 随机间隔，分段 sleep 以便快速响应 Esc
            interval = random.uniform(min_interval, max_interval)
            _sleep_with_escape(interval, user32, VK_ESCAPE)
    except KeyboardInterrupt:
        # 兜底：万一用户用 Ctrl+C 且控制台恰好在前台
        pass

    elapsed = time.time() - start_time
    rate = click_count / elapsed if elapsed > 0 else 0
    stopped_by = "Esc" if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000 else "自动"
    logger.info(
        "spam-click 停止 原因=%s 点击=%d次 耗时=%.1fs 平均=%.1f次/秒",
        stopped_by,
        click_count,
        elapsed,
        rate,
    )
    rprint(
        f"[green]✓ 已停止[/green] ({stopped_by})  "
        f"点击 {click_count} 次  耗时 {elapsed:.1f}s  "
        f"平均 {rate:.1f} 次/秒"
    )


def _sleep_with_escape(
    total: float,
    user32: object,
    vk: int,
    step: float = 0.03,
) -> None:
    """分段 sleep，每 step 秒查一次指定虚拟键是否按下。

    用 GetAsyncKeyState 读物理按键状态，不受窗口焦点影响。
    """
    import time

    elapsed = 0.0
    while elapsed < total:
        if user32.GetAsyncKeyState(vk) & 0x8000:
            return
        time.sleep(min(step, total - elapsed))
        elapsed += step


@app.command(name="pick")
def pick_location(
    rect: bool = typer.Option(
        False,
        "--rect",
        "-r",
        help="矩形框选模式：采两个对角点，输出可粘贴进 tasks.yaml 的 ROI 片段",
    ),
) -> None:
    """比例坐标拾取工具。

    默认采点模式：实时读取鼠标位置，计算其在游戏窗口客户区内的比例坐标。
    用于确定按钮位置和 ROI 区域，结果可直接写入配置文件。

    --rect 框选模式：采两个对角点，输出比例矩形 (x_ratio / y_ratio /
    width_ratio / height_ratio)，用于标定棋盘等矩形 ROI，结果可直接
    粘贴进 configs/tasks.yaml 的 rois 段。

    内置参考锚点，方便快速定位：
        - 左上 (0.0, 0.0)    - 上中 (0.5, 0.0)    - 右上 (1.0, 0.0)
        - 左中 (0.0, 0.5)    - 中心 (0.5, 0.5)    - 右中 (1.0, 0.5)
        - 左下 (0.0, 1.0)    - 下中 (0.5, 1.0)    - 右下 (1.0, 1.0)

    操作：
        - 鼠标移动：实时显示当前比例坐标和最近锚点
        - 按 [回车]：采点模式记录坐标 / 框选模式采对角点
        - 按 [q] 或 [Esc]：退出
    """
    enable_dpi_awareness()
    logger.info("pick 开始 模式=%s", "框选(rect)" if rect else "采点")

    # 读取窗口标题
    local_config = load_local_config(LOCAL_CONFIG_PATH)
    if local_config is not None:
        title = local_config.window.title
        class_name = local_config.window.class_name
    else:
        title = "永远的蔚蓝星球"
        class_name = "Chrome_WidgetWin_0"
        logger.debug("未读取到 local.yaml，使用默认窗口标题/类名")

    handle = find_window_smart(title, class_name)
    if not handle:
        logger.error("未找到窗口: %s 类名=%s", title, class_name)
        rprint(f"[red]未找到窗口: {title}[/red]")
        rprint("[dim]提示：用 wlxq-bot inspect --all 查看所有可见窗口[/dim]")
        raise typer.Exit(1)

    info = get_window_info(handle)
    if info.is_minimized:
        logger.error("窗口已最小化 handle=%s", info.handle)
        rprint("[red]窗口已最小化，请先恢复窗口[/red]")
        raise typer.Exit(1)

    cl, ct, cw, ch = info.client_rect
    logger.debug(
        "命中窗口 handle=%s 客户区=%dx%d 左上屏幕=(%d,%d)",
        info.handle,
        cw,
        ch,
        cl,
        ct,
    )
    mode_label = "框选(rect)" if rect else "采点"
    rprint(
        f"[bold]模式: {mode_label}[/bold]  窗口: {info.title}  客户区: [green]{cw} × {ch}[/green]  左上角屏幕坐标: [yellow]({cl}, {ct})[/yellow]"
    )
    if rect:
        rprint(
            "[dim]移动到矩形左上角 → 按[回车]记第一点 → 移到右下角 → 按[回车]输出 ROI → [q]退出[/dim]"
        )
    else:
        rprint("[dim]移动鼠标 → 按[回车]记录坐标 → 按[q]退出[/dim]")
    rprint(
        "[dim]参考锚点: 左上(0,0) 上中(0.5,0) 右上(1,0) 左中(0,0.5) 中心(0.5,0.5) 右中(1,0.5) 左下(0,1) 下中(0.5,1) 右下(1,1)[/dim]"
    )
    rprint("[dim]" + "-" * 50 + "[/dim]")

    # 全局锚点
    anchors = [
        ("左上", 0.0, 0.0),
        ("上中", 0.5, 0.0),
        ("右上", 1.0, 0.0),
        ("左中", 0.0, 0.5),
        ("中心", 0.5, 0.5),
        ("右中", 1.0, 0.5),
        ("左下", 0.0, 1.0),
        ("下中", 0.5, 1.0),
        ("右下", 1.0, 1.0),
    ]

    import ctypes
    import ctypes.wintypes
    import msvcrt
    import sys
    import time

    last_print_time = 0.0
    # 框选模式：第一个对角点（比例坐标），采到第二点后输出矩形并重置
    rect_p1: tuple[float, float] | None = None

    try:
        while True:
            point = ctypes.wintypes.POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
            mx, my = point.x, point.y

            rel_x = mx - cl
            rel_y = my - ct
            in_client = 0 <= rel_x < cw and 0 <= rel_y < ch

            now = time.time()
            if now - last_print_time >= 0.1:
                if in_client:
                    rx = rel_x / cw
                    ry = rel_y / ch
                    if rect and rect_p1 is not None:
                        # 框选模式：显示从 p1 到当前鼠标的预览矩形
                        w_r = abs(rx - rect_p1[0])
                        h_r = abs(ry - rect_p1[1])
                        sys.stdout.write(
                            f"\r  P1=({rect_p1[0]:.4f},{rect_p1[1]:.4f})  "
                            f"预览 w={w_r:.4f} h={h_r:.4f}  "
                        )
                    else:
                        # 找最近的锚点
                        nearest = min(anchors, key=lambda a: (a[1] - rx) ** 2 + (a[2] - ry) ** 2)
                        dist = ((nearest[1] - rx) ** 2 + (nearest[2] - ry) ** 2) ** 0.5
                        anchor_hint = f"  近[{nearest[0]}]" if dist < 0.05 else ""
                        sys.stdout.write(
                            f"\r  比例: ({rx:.4f}, {ry:.4f})  像素: ({rel_x}, {rel_y}){anchor_hint}  "
                        )
                    sys.stdout.flush()
                else:
                    sys.stdout.write(
                        f"\r  (不在客户区) 鼠标=({mx},{my}) 客户区左上=({cl},{ct}) 尺寸={cw}x{ch} rel=({rel_x},{rel_y})  "
                    )
                    sys.stdout.flush()
                last_print_time = now

            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key in (b"\r", b"\n") and in_client:
                    rx = rel_x / cw
                    ry = rel_y / ch
                    if rect:
                        if rect_p1 is None:
                            rect_p1 = (rx, ry)
                            rprint(
                                f"\n  [bold blue]第一点[/bold blue]  "
                                f"({rx:.4f}, {ry:.4f})  像素 ({rel_x}, {rel_y})"
                            )
                            rprint("[dim]移动到右下角按[回车]完成矩形[/dim]")
                        else:
                            x_r = min(rect_p1[0], rx)
                            y_r = min(rect_p1[1], ry)
                            w_r = abs(rx - rect_p1[0])
                            h_r = abs(ry - rect_p1[1])
                            logger.info("ROI 采集 x=%.4f y=%.4f w=%.4f h=%.4f", x_r, y_r, w_r, h_r)
                            rprint(
                                f"\n  [bold green]ROI[/bold green]  "
                                f"x:{x_r:.4f} y:{y_r:.4f} w:{w_r:.4f} h:{h_r:.4f}"
                            )
                            rprint("[dim]可粘贴进 configs/tasks.yaml 的 rois 段：[/dim]")
                            rprint("  [green]relative_to: client[/green]")
                            rprint(f"  [green]x_ratio: {x_r:.4f}[/green]")
                            rprint(f"  [green]y_ratio: {y_r:.4f}[/green]")
                            rprint(f"  [green]width_ratio: {w_r:.4f}[/green]")
                            rprint(f"  [green]height_ratio: {h_r:.4f}[/green]")
                            rect_p1 = None
                            rprint("[dim]可继续采下一个矩形，按[q]退出[/dim]")
                    else:
                        nearest = min(anchors, key=lambda a: (a[1] - rx) ** 2 + (a[2] - ry) ** 2)
                        dist = ((nearest[1] - rx) ** 2 + (nearest[2] - ry) ** 2) ** 0.5
                        anchor_note = f"  (近[{nearest[0]}])" if dist < 0.05 else ""
                        logger.info(
                            "采点 x_ratio=%.4f y_ratio=%.4f pixel=(%d,%d)", rx, ry, rel_x, rel_y
                        )
                        rprint(
                            f"\n  [bold green]位置[/bold green]  "
                            f"x_ratio: {rx:.4f}  y_ratio: {ry:.4f}  "
                            f"pixel: ({rel_x}, {rel_y}){anchor_note}"
                        )
                        rprint("[dim]可粘贴进 configs/tasks.yaml 的 hotspots 段：[/dim]")
                        rprint(f"  [green]x_ratio: {rx:.4f}[/green]")
                        rprint(f"  [green]y_ratio: {ry:.4f}[/green]")
                elif key in (b"q", b"\x1b"):
                    rprint("\n[dim]退出[/dim]")
                    break

            time.sleep(0.02)

    except KeyboardInterrupt:
        rprint("\n[dim]退出[/dim]")


@app.command(name="move")
def move_cell(
    src: str = typer.Argument(..., help="源格子，如 5B"),
    dst: str = typer.Argument(..., help="目标格子，如 3B"),
    role: str = typer.Option("helper", "--role", "-r", help="棋盘：helper / initiator"),
    delay: float = typer.Option(3.0, "--delay", "-d", help="拖动前延迟(秒)，给切窗口时间"),
) -> None:
    """拖动棋盘格子：把源格子的英雄拖到目标格子。

    用于测试合成拖动坐标。格子标识 <排号><列字母>，排 1-6 列 A/B/C。
    helper: A=A'(最外右) B=B'(中) C=C'(靠中间左)。

    示例::

        wlxq-bot move 5B 3B                   # helper 棋盘，5B 拖到 3B
        wlxq-bot move 2A 4C --role initiator  # initiator 棋盘
        wlxq-bot move 5B 3B -d 5              # 延迟 5 秒
    """
    from pathlib import Path

    from wlxq_bot.config import load_tasks_config
    from wlxq_bot.models import CoopRole
    from wlxq_bot.perception.locator import board_grid_for_role, parse_cell_label

    enable_dpi_awareness()
    logger.info("move 开始 src=%s dst=%s role=%s delay=%ss", src, dst, role, delay)

    if role not in ("helper", "initiator"):
        logger.error("未知角色: %s", role)
        rprint(f"[red]未知角色: {role}，应为 helper 或 initiator[/red]")
        raise typer.Exit(1)
    coop_role = CoopRole.HELPER if role == "helper" else CoopRole.INITIATOR

    tasks = load_tasks_config(Path("configs/tasks.yaml"))
    try:
        grid = board_grid_for_role(coop_role, tasks.board)
    except KeyError as e:
        logger.error("棋盘配置缺失 role=%s 原因=%s", role, e)
        rprint(f"[red]棋盘配置缺失: {e}[/red]")
        raise typer.Exit(1) from e

    local_config = load_local_config(LOCAL_CONFIG_PATH)
    title = local_config.window.title if local_config else "永远的蔚蓝星球"
    class_name = local_config.window.class_name if local_config else "Chrome_WidgetWin_0"

    handle = find_window_smart(title, class_name)
    if not handle:
        logger.error("未找到窗口: %s 类名=%s", title, class_name)
        rprint(f"[red]未找到窗口: {title}[/red]")
        raise typer.Exit(1)

    info = get_window_info(handle)
    if info.is_minimized:
        logger.error("窗口已最小化 handle=%s", info.handle)
        rprint("[red]窗口已最小化，请先恢复[/red]")
        raise typer.Exit(1)

    cl, ct = info.client_rect[0], info.client_rect[1]
    cw, ch = info.client_size
    logger.debug("命中窗口 handle=%s 客户区=%dx%d", info.handle, cw, ch)

    try:
        src_row, src_col = parse_cell_label(src, coop_role)
        dst_row, dst_col = parse_cell_label(dst, coop_role)
    except ValueError as e:
        logger.error("格子标识解析失败: %s", e)
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    src_cx, src_cy = grid.cell_center(src_row, src_col, (cw, ch))
    dst_cx, dst_cy = grid.cell_center(dst_row, dst_col, (cw, ch))
    src_sx, src_sy = cl + src_cx, ct + src_cy
    dst_sx, dst_sy = cl + dst_cx, ct + dst_cy
    logger.debug(
        "坐标换算 客户区=%dx%d %s(row%d,col%d)→客户区(%d,%d)→屏幕(%d,%d) %s(row%d,col%d)→客户区(%d,%d)→屏幕(%d,%d)",
        cw,
        ch,
        src,
        src_row,
        src_col,
        src_cx,
        src_cy,
        src_sx,
        src_sy,
        dst,
        dst_row,
        dst_col,
        dst_cx,
        dst_cy,
        dst_sx,
        dst_sy,
    )

    rprint(f"窗口: {info.title}  客户区: {cw} × {ch}  角色: {role}")
    rprint(f"源 {src}: 客户区({src_cx},{src_cy}) 屏幕({src_sx},{src_sy})")
    rprint(f"目标 {dst}: 客户区({dst_cx},{dst_cy}) 屏幕({dst_sx},{dst_sy})")

    if not info.is_foreground:
        import time as _time

        import win32con
        import win32gui

        win32gui.ShowWindow(handle, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(handle)
        _time.sleep(0.3)

    import time

    rprint(f"[yellow]{delay}秒后拖动 {src}→{dst}，请勿移动鼠标...[/yellow]")
    time.sleep(delay)

    from wlxq_bot.models import Action

    screen, executor = _manual_action_components()
    ctx, _frame = screen.capture(handle)
    src_cx, src_cy = grid.cell_center(src_row, src_col, ctx.client_size)
    dst_cx, dst_cy = grid.cell_center(dst_row, dst_col, ctx.client_size)
    result = executor.execute(
        ctx,
        Action(
            kind="drag",
            target=(src_cx, src_cy),
            end=(dst_cx, dst_cy),
            duration=0.5,
            reason="人工 move 命令",
        ),
    )
    if not result.executed:
        logger.error("拖动被安全检查拒绝: %s", result.failure_reason)
        rprint(f"[red]拖动被安全检查拒绝: {result.failure_reason}[/red]")
        raise typer.Exit(1)
    src_sx, src_sy = ctx.client_to_screen(src_cx, src_cy)
    dst_sx, dst_sy = ctx.client_to_screen(dst_cx, dst_cy)
    logger.info(
        "已拖动 %s→%s frame=%d 源屏幕=(%d,%d) 目标屏幕=(%d,%d)",
        src,
        dst,
        ctx.frame_id,
        src_sx,
        src_sy,
        dst_sx,
        dst_sy,
    )
    rprint(f"[green]✓ 已拖动 {src}→{dst}[/green]")


@app.command()
def recognize(
    image: str | None = typer.Argument(
        None,
        help="截图文件路径（PNG）；不指定则实时截取当前游戏窗口画面识别",
    ),
    pack: str | None = typer.Option(
        None,
        "--pack",
        "-p",
        help="指定模板包分辨率（如 3000x2000）；不指定则按画面尺寸自动匹配",
    ),
    category: str = typer.Option(
        "all",
        "--category",
        "-c",
        help="只识别某类模板：buttons / skills / heroes / all（默认 all）",
    ),
    hero: str | None = typer.Option(
        None,
        "--hero",
        help="只识别指定英雄（如 assault），仅在 --category heroes 时生效",
    ),
    threshold: float | None = typer.Option(
        None,
        "--threshold",
        "-t",
        help="覆盖默认阈值；不指定时 buttons/skills=0.85，heroes=0.78",
    ),
    templates_root: str = typer.Option(
        "assets/templates",
        "--templates-root",
        help="模板包根目录路径",
    ),
    save: str | None = typer.Option(
        None,
        "--save",
        "-s",
        help="标注图保存路径；不指定则自动到 screenshots/debug/",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        help="识别后自动用图片查看器打开标注图",
    ),
) -> None:
    """识别当前游戏画面（或指定截图），验证模板识别效果。

    默认实时截取当前游戏窗口画面，用对应分辨率的模板包做模板匹配，
    报告每个模板的命中情况（置信度、位置），并生成标注图。
    传入截图文件路径时改为离线识别该截图，无需游戏窗口开着。

    默认按画面尺寸自动匹配模板包；尺寸无对应模板包时可用 --pack 显式指定。
    默认识别 buttons / skills / heroes 全部模板，可用 --category 缩小范围。

    示例::

        wlxq-bot recognize                          # 实时识别当前游戏画面
        wlxq-bot recognize -c heroes --hero assault # 只识别强袭英雄
        wlxq-bot recognize -t 0.7 --show            # 调阈值并打开标注图
        wlxq-bot recognize screenshots/raw/shot.png # 离线识别指定截图
        wlxq-bot recognize shot.png --pack 3000x2000
    """
    import cv2

    from wlxq_bot.assets import find_template_pack
    from wlxq_bot.perception.vision import Vision

    logger.info(
        "recognize 开始 image=%s pack=%s category=%s hero=%s threshold=%s",
        image,
        pack,
        category,
        hero,
        threshold,
    )

    # 1. 获取画面：不传 image 时实时截取游戏窗口，传了则离线读取截图文件
    frame, source_label = _acquire_frame(image)
    fh, fw = frame.shape[:2]

    # 2. 解析模板包
    root = Path(templates_root)
    if pack:
        try:
            pw_s, ph_s = pack.lower().split("x")
            pack_w, pack_h = int(pw_s), int(ph_s)
        except ValueError:
            logger.error("--pack 格式错误: %s（应为 WxH，如 3000x2000）", pack)
            rprint(f"[red]--pack 格式错误: {pack}（应为 WxH，如 3000x2000）[/red]")
            raise typer.Exit(1) from None
    else:
        pack_w, pack_h = fw, fh

    template_pack = find_template_pack(root, pack_w, pack_h)
    if template_pack is None:
        available: list[str] = []
        if root.is_dir():
            available = sorted(d.name for d in root.iterdir() if d.is_dir() and "x" in d.name)
        logger.error("找不到模板包 %dx%d root=%s", pack_w, pack_h, root)
        rprint(f"[red]找不到 {pack_w}x{pack_h} 模板包（root={root}）[/red]")
        if available:
            rprint(f"[dim]可用模板包: {', '.join(available)}[/dim]")
        rprint("[dim]提示：用 --pack 指定分辨率，或 --templates-root 指定模板根目录[/dim]")
        # 离线模式下，传入图片尺寸偏小很可能是模板图片而非游戏画面截图
        if image is not None and (fw < 500 or fh < 500):
            rprint(f"[yellow]传入图片只有 {fw}×{fh}，看起来像模板图片而非游戏画面截图。[/yellow]")
            rprint(
                "[dim]recognize 识别的是游戏画面；实时识别请直接运行 "
                "「wlxq-bot recognize」，离线识别请传完整游戏窗口截图。[/dim]"
            )
        raise typer.Exit(1)

    logger.debug("模板包 root=%s 尺寸=%dx%d", template_pack.root, pack_w, pack_h)
    rprint(
        f"[bold]画面[/bold] {source_label}  尺寸 [green]{fw} × {fh}[/green]  "
        f"[bold]模板包[/bold] [green]{pack_w}x{pack_h}[/green]"
    )

    # 3. 阈值：buttons/skills 默认 0.85，heroes 默认 0.78；--threshold 覆盖全部
    btn_th = threshold if threshold is not None else 0.85
    hero_th = threshold if threshold is not None else 0.78

    vision = Vision()
    # 标注用结果: (category, label, (x,y), conf, (w,h))
    annotations: list[tuple[str, str, tuple[int, int], float, tuple[int, int]]] = []
    hit_count = 0

    cat = category.lower()
    do_buttons = cat in ("all", "buttons")
    do_skills = cat in ("all", "skills")
    do_heroes = cat in ("all", "heroes")

    # --- buttons ---
    if do_buttons:
        rprint(f"\n[bold cyan]按钮识别[/bold cyan] (阈值 {btn_th:.2f})")
        button_pngs = sorted(template_pack.buttons_dir.glob("*.png"))
        if not button_pngs:
            rprint("  [dim]无按钮模板[/dim]")
        else:
            for tpl in button_pngs:
                m = vision.match_template(frame, str(tpl), threshold=btn_th)
                if m is None:
                    rprint(f"  [red]✗[/red] {tpl.name}")
                    logger.debug("按钮未命中 template=%s", tpl.name)
                else:
                    t = vision._load_template(str(tpl))
                    th, tw = t.shape[:2] if t is not None else (40, 40)
                    annotations.append(("button", tpl.name, m.position, m.confidence, (tw, th)))
                    hit_count += 1
                    rprint(
                        f"  [green]✓[/green] {tpl.name}  "
                        f"置信度 [green]{m.confidence:.3f}[/green]  "
                        f"位置 ({m.position[0]}, {m.position[1]})"
                    )
                    logger.debug(
                        "按钮命中 template=%s conf=%.3f pos=(%d,%d)",
                        tpl.name,
                        m.confidence,
                        m.position[0],
                        m.position[1],
                    )

    # --- skills ---
    if do_skills:
        rprint(f"\n[bold cyan]技能识别[/bold cyan] (阈值 {btn_th:.2f})")
        skill_pngs = sorted(template_pack.skills_dir.glob("*.png"))
        if not skill_pngs:
            rprint("  [dim]无技能模板[/dim]")
        else:
            for tpl in skill_pngs:
                m = vision.match_template(frame, str(tpl), threshold=btn_th)
                if m is None:
                    rprint(f"  [red]✗[/red] {tpl.name}")
                    logger.debug("技能未命中 template=%s", tpl.name)
                else:
                    t = vision._load_template(str(tpl))
                    th, tw = t.shape[:2] if t is not None else (40, 40)
                    annotations.append(("skill", tpl.name, m.position, m.confidence, (tw, th)))
                    hit_count += 1
                    rprint(
                        f"  [green]✓[/green] {tpl.name}  "
                        f"置信度 [green]{m.confidence:.3f}[/green]  "
                        f"位置 ({m.position[0]}, {m.position[1]})"
                    )
                    logger.debug(
                        "技能命中 template=%s conf=%.3f pos=(%d,%d)",
                        tpl.name,
                        m.confidence,
                        m.position[0],
                        m.position[1],
                    )

    # --- heroes ---
    if do_heroes:
        rprint(f"\n[bold cyan]英雄识别[/bold cyan] (阈值 {hero_th:.2f})")
        heroes_root = template_pack.heroes_dir
        if not heroes_root.is_dir():
            rprint("  [dim]无英雄模板目录[/dim]")
        else:
            hero_ids = sorted(d.name for d in heroes_root.iterdir() if d.is_dir())
            if hero:
                hero_ids = [h for h in hero_ids if h == hero]
                if not hero_ids:
                    all_ids = sorted(d.name for d in heroes_root.iterdir() if d.is_dir())
                    rprint(f"  [yellow]未找到英雄: {hero}[/yellow]")
                    rprint(f"[dim]可用英雄: {', '.join(all_ids)}[/dim]")
            for hid in hero_ids:
                templates = template_pack.scan_hero_templates(hid)
                if not templates:
                    rprint(f"  [dim]{hid}: 无模板[/dim]")
                    continue
                paths = [str(t.path) for t in templates]
                # path -> star_level 映射，用于标注
                path_star = {str(t.path): t.star_level for t in templates}
                matches = vision.match_template_set(frame, paths, threshold=hero_th)
                if not matches:
                    rprint(f"  [red]✗[/red] {hid}  [dim](共 {len(templates)} 个模板)[/dim]")
                    logger.debug("英雄未命中 hero=%s 模板数=%d", hid, len(templates))
                    continue
                rprint(f"  [green]✓[/green] {hid}  命中 {len(matches)} 个")
                for i, m in enumerate(matches, 1):
                    star = path_star.get(m.template_name, 0)
                    t = vision._load_template(m.template_name)
                    th, tw = t.shape[:2] if t is not None else (40, 40)
                    star_label = f"★{star}" if star else ""
                    annotations.append(
                        ("hero", f"{hid}{star_label}", m.position, m.confidence, (tw, th))
                    )
                    hit_count += 1
                    rprint(
                        f"     #{i}  {star_label}  "
                        f"置信度 [green]{m.confidence:.3f}[/green]  "
                        f"位置 ({m.position[0]}, {m.position[1]})"
                    )
                    logger.debug(
                        "英雄命中 hero=%s star=%d conf=%.3f pos=(%d,%d)",
                        hid,
                        star,
                        m.confidence,
                        m.position[0],
                        m.position[1],
                    )

    # 4. 生成标注图：按钮绿框、技能蓝框、英雄红框
    annotated = frame.copy()
    color_map = {"button": (0, 255, 0), "skill": (255, 0, 0), "hero": (0, 0, 255)}
    for cat_name, label, (cx, cy), conf, (tw, th) in annotations:
        color = color_map.get(cat_name, (0, 255, 255))
        cv2.rectangle(
            annotated,
            (cx - tw // 2, cy - th // 2),
            (cx + tw // 2, cy + th // 2),
            color,
            2,
        )
        cv2.putText(
            annotated,
            f"{label} {conf:.2f}",
            (cx - tw // 2, cy - th // 2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    # 5. 保存标注图（中文路径用 imencode + tofile）
    from datetime import datetime

    if save is None:
        debug_dir = Path("screenshots/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save = str(debug_dir / f"recognize_{timestamp}.png")

    save_path = Path(save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    success, buf = cv2.imencode(".png", annotated)
    if success:
        buf.tofile(str(save_path))
        logger.info("标注图已保存 path=%s 命中=%d", save_path, hit_count)
    else:
        logger.error("标注图保存失败 path=%s", save_path)
        rprint(f"[red]标注图保存失败: {save_path}[/red]")

    rprint(
        f"\n[bold]汇总[/bold]  命中 [green]{hit_count}[/green] 个  "
        f"标注图 [white]{save_path}[/white]"
    )
    logger.info("recognize 结束 命中=%d 标注图=%s", hit_count, save_path)

    if show:
        import os

        os.startfile(str(save_path))  # type: ignore[attr-defined]


if __name__ == "__main__":
    app()
