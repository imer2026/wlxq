"""比例坐标拾取工具。

实时读取鼠标位置，计算其在游戏窗口客户区内的比例坐标。
用于确定 ROI 区域和按钮位置。

用法：
    python -m wlxq_bot.tools.locator_picker

操作：
    - 鼠标移动：实时显示当前比例坐标
    - 按回车：打印当前比例坐标到控制台（可复制到配置文件）
    - 按空格：标记 ROI 起点/终点，两次空格后打印 ROI 比例
    - 按 q 或 Esc：退出
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from wlxq_bot.config import load_local_config
from wlxq_bot.perception.screen import (
    enable_dpi_awareness,
    find_window_smart,
    get_window_info,
)


def get_mouse_pos() -> tuple[int, int]:
    """获取鼠标屏幕坐标。"""
    import ctypes

    point = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
    return (point.x, point.y)


def screen_to_client_ratio(
    mouse_x: int,
    mouse_y: int,
    client_left: int,
    client_top: int,
    client_width: int,
    client_height: int,
) -> tuple[float, float] | None:
    """屏幕坐标转客户区比例坐标。

    Returns:
        (ratio_x, ratio_y)，鼠标不在客户区内时返回 None
    """
    rel_x = mouse_x - client_left
    rel_y = mouse_y - client_top
    if rel_x < 0 or rel_y < 0 or rel_x >= client_width or rel_y >= client_height:
        return None
    return (rel_x / client_width, rel_y / client_height)


def main() -> None:
    enable_dpi_awareness()

    # 读取窗口标题
    local_config = load_local_config(Path("configs/local.yaml"))
    if local_config is not None:
        title = local_config.window.title
        class_name = local_config.window.class_name
    else:
        title = "永远的蔚蓝星球"
        class_name = "Chrome_WidgetWin_0"

    # 查找窗口
    handle = find_window_smart(title, class_name)
    if not handle:
        print(f"[错误] 未找到窗口: {title}")
        print("提示：用 wlxq-bot inspect --all 查看所有可见窗口")
        sys.exit(1)

    info = get_window_info(handle)
    if info.is_minimized:
        print("[错误] 窗口已最小化，请先恢复窗口")
        sys.exit(1)

    cl, ct, cw, ch = info.client_rect
    print(f"窗口: {info.title}")
    print(f"客户区: {cw} × {ch}")
    print(f"客户区屏幕位置: ({cl}, {ct})")
    print()
    print("操作说明:")
    print("  移动鼠标到目标位置 → 实时显示比例坐标")
    print("  按 [回车] → 打印当前比例坐标（可复制到配置）")
    print("  按 [空格] → 标记 ROI 起点/终点，两次空格后打印 ROI")
    print("  按 [q] 或 [Esc] → 退出")
    print("-" * 50)

    roi_start: tuple[float, float] | None = None
    last_print_time = 0.0

    try:
        while True:
            mx, my = get_mouse_pos()
            ratio = screen_to_client_ratio(mx, my, cl, ct, cw, ch)

            # 每 0.1 秒打印一次，避免刷屏
            now = time.time()
            if now - last_print_time >= 0.1:
                if ratio is not None:
                    rx, ry = ratio
                    # 转客户区像素坐标
                    px = int(rx * cw)
                    py = int(ry * ch)
                    sys.stdout.write(f"\r  比例: ({rx:.4f}, {ry:.4f})  像素: ({px}, {py})  ")
                    sys.stdout.flush()
                else:
                    sys.stdout.write("\r  (鼠标不在客户区内)                    ")
                    sys.stdout.flush()
                last_print_time = now

            # 检查按键（非阻塞）
            import msvcrt

            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key in (b"\r", b"\n"):
                    if ratio is not None:
                        rx, ry = ratio
                        px = int(rx * cw)
                        py = int(ry * ch)
                        print()
                        print(f"  [位置] ratio_x: {rx:.4f}  ratio_y: {ry:.4f}  pixel: ({px}, {py})")
                        if roi_start is not None:
                            print(f"  [ROI 起点已标记] {roi_start}")
                elif key == b" ":
                    if ratio is not None:
                        if roi_start is None:
                            roi_start = ratio
                            print()
                            print(f"  [ROI 起点] ({ratio[0]:.4f}, {ratio[1]:.4f})")
                        else:
                            rx2, ry2 = ratio
                            rx1, ry1 = roi_start
                            print()
                            print("  [ROI 完成]")
                            print(f"    起点: ({rx1:.4f}, {ry1:.4f})")
                            print(f"    终点: ({rx2:.4f}, {ry2:.4f})")
                            print(f"    宽度比例: {abs(rx2 - rx1):.4f}")
                            print(f"    高度比例: {abs(ry2 - ry1):.4f}")
                            print("    YAML 格式:")
                            print("      relative_to: client")
                            print(f"      x_ratio: {min(rx1, rx2):.4f}")
                            print(f"      y_ratio: {min(ry1, ry2):.4f}")
                            print(f"      width_ratio: {abs(rx2 - rx1):.4f}")
                            print(f"      height_ratio: {abs(ry2 - ry1):.4f}")
                            roi_start = None
                elif key in (b"q", b"\x1b"):  # q 或 Esc
                    print()
                    print("退出")
                    break

            time.sleep(0.02)

    except KeyboardInterrupt:
        print()
        print("退出")


if __name__ == "__main__":
    main()
