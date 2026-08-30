"""棋盘格子工具。

两种模式：

- 拖动模式 ``move <源> <目标>``：把源格子中心的英雄拖到目标格子中心。
  例：``move 5B 3B``
- 画图模式 ``<格子>...``：在截图上红框高亮指定格子，验证坐标。
  例：``5B 3B``

格子标识格式 ``<排号><列字母>``，排 1-6，列 A/B/C：
- helper：A=A'(最外右) B=B'(中) C=C'(靠中间左)
- initiator：A=A(最外左) B=B(中) C=C(靠中间右)

功能已集成到 wlxq-bot move 子命令，本脚本保留作为独立调试入口。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wlxq_bot.config import load_local_config, load_tasks_config
from wlxq_bot.models import CoopRole
from wlxq_bot.perception.locator import board_grid_for_role, parse_cell_label

ROOT = Path(__file__).resolve().parent.parent
TASKS_YAML = ROOT / "configs" / "tasks.yaml"
LOCAL_YAML = ROOT / "configs" / "local.yaml"
DEFAULT_SRC = ROOT / "screenshots" / "raw" / "所有棋盘解锁.png"
DEFAULT_DST = ROOT / "outputs" / "verify_cell.png"


def load_grid(role: CoopRole):
    tasks = load_tasks_config(TASKS_YAML)
    return board_grid_for_role(role, tasks.board)


def find_game_window():
    from wlxq_bot.perception.screen import (
        enable_dpi_awareness,
        find_window_smart,
        get_window_info,
    )

    enable_dpi_awareness()
    local = load_local_config(LOCAL_YAML)
    title = local.window.title if local else "永远的蔚蓝星球"
    class_name = local.window.class_name if local else "Chrome_WidgetWin_0"
    handle = find_window_smart(title, class_name)
    if not handle:
        print(f"未找到窗口: {title}")
        sys.exit(1)
    info = get_window_info(handle)
    if info.is_minimized:
        print("窗口已最小化，请先恢复")
        sys.exit(1)
    return info


def do_annotate(grid, cells, role, src_path, dst_path) -> None:
    """画图验证模式：在截图上红框高亮指定格子。"""
    im = Image.open(src_path)
    cw, ch = im.size
    draw = ImageDraw.Draw(im)

    font = None
    for fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        try:
            font = ImageFont.truetype(fp, 20)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    cell_w = int(grid.cell_width_ratio * cw)
    cell_h = int(grid.cell_height_ratio * ch)

    for cell in cells:
        row, col = parse_cell_label(cell, role)
        cx, cy = grid.cell_center(row, col, (cw, ch))
        x = cx - cell_w // 2
        y = cy - cell_h // 2
        draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(255, 50, 50), width=5)
        draw.text((x + 5, y + 5), f"{cell}→({cx},{cy})", fill=(255, 50, 50), font=font)
        print(f"{cell}: row={row} col={col} 中心=({cx},{cy})")

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    im.save(dst_path)
    print(f"saved {dst_path} ({im.size})")


def do_drag(grid, src_cell, dst_cell, role, delay) -> None:
    """拖动模式：从源格子中心拖到目标格子中心。"""
    import pyautogui

    info = find_game_window()
    cl, ct = info.client_rect[0], info.client_rect[1]
    cw, ch = info.client_size

    src_row, src_col = parse_cell_label(src_cell, role)
    dst_row, dst_col = parse_cell_label(dst_cell, role)
    src_cx, src_cy = grid.cell_center(src_row, src_col, (cw, ch))
    dst_cx, dst_cy = grid.cell_center(dst_row, dst_col, (cw, ch))
    src_sx, src_sy = cl + src_cx, ct + src_cy
    dst_sx, dst_sy = cl + dst_cx, ct + dst_cy

    print(f"窗口: {info.title}  客户区: {cw}×{ch}")
    print(f"源 {src_cell}: 客户区({src_cx},{src_cy}) 屏幕({src_sx},{src_sy})")
    print(f"目标 {dst_cell}: 客户区({dst_cx},{dst_cy}) 屏幕({dst_sx},{dst_sy})")

    if not info.is_foreground:
        import win32con
        import win32gui

        win32gui.ShowWindow(info.handle, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(info.handle)
        time.sleep(0.3)

    print(f"{delay}秒后拖动 {src_cell}→{dst_cell}，请勿移动鼠标...")
    time.sleep(delay)

    pyautogui.moveTo(src_sx, src_sy)
    time.sleep(0.15)
    pyautogui.dragTo(dst_sx, dst_sy, duration=0.5, button="left")
    print(f"✓ 已拖动 {src_cell}→{dst_cell}")


def main() -> None:
    parser = argparse.ArgumentParser(description="棋盘格子工具：move 拖动 / 默认画图验证")
    parser.add_argument("cells", nargs="+", help="move 5B 3B 拖动；或 5B 3B 画图验证")
    parser.add_argument("--role", choices=["helper", "initiator"], default="helper")
    parser.add_argument("--src", default=str(DEFAULT_SRC), help="画图模式源截图路径")
    parser.add_argument("--dst", default=str(DEFAULT_DST), help="画图模式输出路径")
    parser.add_argument("--delay", type=float, default=3.0, help="拖动前延迟(秒)")
    args = parser.parse_args()

    role = CoopRole.HELPER if args.role == "helper" else CoopRole.INITIATOR
    grid = load_grid(role)

    is_move = args.cells[0].lower() == "move"
    if is_move:
        if len(args.cells) < 3:
            parser.error("move 需要 <源格子> <目标格子>，如 move 5B 3B")
        do_drag(grid, args.cells[1], args.cells[2], role, args.delay)
    else:
        do_annotate(grid, args.cells, role, args.src, args.dst)


if __name__ == "__main__":
    main()
