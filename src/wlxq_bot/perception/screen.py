"""Screen Capture：获取游戏窗口客户区截图。

职责：
- 查找游戏窗口句柄
- 读取客户区位置和尺寸
- 使用 MSS 截取客户区对应屏幕区域
- 生成 WindowContext 快照

窗口查找使用 pywin32 枚举顶层窗口并按标题模糊匹配。
微信小游戏窗口标题尚未最终确认，因此 find_window 支持关键字模糊匹配，
并可通过 list_windows 列出所有候选窗口辅助排查。
"""

from __future__ import annotations

import ctypes
import time
from collections.abc import Sequence
from dataclasses import dataclass

import win32api
import win32con
import win32gui
import win32process
from mss import mss

from wlxq_bot.models import WindowContext

# ---------------------------------------------------------------------------
# 窗口信息数据类
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowInfo:
    """窗口基本信息，用于检查和调试。

    Attributes:
        handle: 窗口句柄
        title: 窗口标题
        class_name: 窗口类名
        is_visible: 窗口是否可见
        is_minimized: 窗口是否最小化
        is_foreground: 窗口是否在前台
        window_rect: 窗口矩形 (left, top, right, bottom)，屏幕坐标
        client_rect: 客户区矩形 (left, top, width, height)，屏幕坐标
        client_size: 客户区物理像素尺寸 (width, height)
        dpi: 窗口所在显示器 DPI
        monitor_id: 窗口所在显示器设备标识
        monitor_resolution: 窗口所在显示器物理分辨率
        process_id: 进程 ID
        thread_id: 线程 ID
    """

    handle: int
    title: str
    class_name: str
    is_visible: bool
    is_minimized: bool
    is_foreground: bool
    window_rect: tuple[int, int, int, int]
    client_rect: tuple[int, int, int, int]
    client_size: tuple[int, int]
    dpi: int
    monitor_id: str
    monitor_resolution: tuple[int, int]
    process_id: int
    thread_id: int


# ---------------------------------------------------------------------------
# 窗口查找与检查
# ---------------------------------------------------------------------------


def enable_dpi_awareness() -> None:
    """启用进程级 DPI 感知。

    必须在读取窗口位置和客户区尺寸前调用，否则 Windows 会按
    DPI 虚拟化返回缩放后的坐标，导致客户区尺寸不准确。

    优先使用 PerMonitorV2，不支持时降级到 PerMonitor 或 System。
    """
    try:
        import ctypes

        # 尝试 PerMonitorV2 (Windows 10 1703+)
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            return
        except (AttributeError, OSError):
            pass

        # 降级到 PerMonitor (Windows 8.1+)
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
            return
        except (AttributeError, OSError):
            pass

        # 降级到 System DPI Aware (Vista+)
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        # DPI 感知失败不阻塞，但调用方应注意坐标可能不准
        pass


def get_window_dpi(window_handle: int) -> int:
    """获取窗口所在显示器的 DPI。

    Args:
        window_handle: 窗口句柄

    Returns:
        DPI 值（96 表示 100% 缩放）
    """
    try:
        hdc = win32gui.GetDC(window_handle)
        if hdc:
            # LOGPIXELSX = 88
            dpi = win32gui.GetDeviceCaps(hdc, 88)
            win32gui.ReleaseDC(window_handle, hdc)
            if dpi > 0:
                return int(dpi)
    except Exception:
        pass
    return 96


def _get_window_monitor_details(window_handle: int) -> tuple[str, tuple[int, int]]:
    """读取窗口所在显示器的设备标识和物理分辨率。"""
    try:
        monitor = win32api.MonitorFromWindow(
            window_handle,
            win32con.MONITOR_DEFAULTTONEAREST,
        )
        monitor_info = win32api.GetMonitorInfo(monitor)
        monitor_rect = monitor_info["Monitor"]
        monitor_id = str(monitor_info.get("Device", monitor))
        left, top, right, bottom = monitor_rect
        width, height = right - left, bottom - top
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise RuntimeError(f"无法获取窗口 {window_handle} 所在显示器的信息") from exc

    if width <= 0 or height <= 0:
        raise RuntimeError(f"窗口 {window_handle} 所在显示器边界无效: {monitor_rect!r}")
    return monitor_id, (width, height)


def get_window_monitor_resolution(window_handle: int) -> tuple[int, int]:
    """获取指定窗口所在显示器的物理分辨率（宽度, 高度）。

    调用方应先启用 DPI 感知。窗口跨屏时由 Windows 选择与窗口相交面积
    最大的显示器；窗口暂时不在任何显示器上时选择最近的显示器。

    Args:
        window_handle: 窗口句柄

    Returns:
        (width, height) 物理像素

    Raises:
        RuntimeError: 无法读取窗口所在显示器或显示器边界无效
    """
    return _get_window_monitor_details(window_handle)[1]


def get_window_info(window_handle: int) -> WindowInfo:
    """读取窗口详细信息。

    Args:
        window_handle: 窗口句柄

    Returns:
        窗口信息
    """
    title = win32gui.GetWindowText(window_handle)
    class_name = win32gui.GetClassName(window_handle)
    style = win32gui.GetWindowLong(window_handle, win32con.GWL_STYLE)

    is_visible = bool(style & win32con.WS_VISIBLE)
    is_minimized = win32gui.IsIconic(window_handle) != 0
    foreground = win32gui.GetForegroundWindow()
    is_foreground = foreground == window_handle

    # 窗口矩形：屏幕坐标 (left, top, right, bottom)
    window_rect = win32gui.GetWindowRect(window_handle)

    # 客户区矩形：客户区坐标 (0, 0, width, height)
    # ClientToScreen 把客户区左上角转成屏幕坐标，得到屏幕坐标系下的客户区位置
    client_left, client_top = win32gui.ClientToScreen(window_handle, (0, 0))
    client_width = window_rect_to_client_width(window_handle)
    client_height = window_rect_to_client_height(window_handle)
    client_rect_screen = (client_left, client_top, client_width, client_height)
    client_size = (client_width, client_height)

    dpi = get_window_dpi(window_handle)
    monitor_id, monitor_resolution = _get_window_monitor_details(window_handle)

    _, process_id = win32process.GetWindowThreadProcessId(window_handle)
    thread_id_ref: tuple[int, int] = win32process.GetWindowThreadProcessId(window_handle)
    thread_id = thread_id_ref[0]

    return WindowInfo(
        handle=window_handle,
        title=title,
        class_name=class_name,
        is_visible=is_visible,
        is_minimized=is_minimized,
        is_foreground=is_foreground,
        window_rect=window_rect,
        client_rect=client_rect_screen,
        client_size=client_size,
        dpi=dpi,
        monitor_id=monitor_id,
        monitor_resolution=monitor_resolution,
        process_id=process_id,
        thread_id=thread_id,
    )


def window_rect_to_client_width(window_handle: int) -> int:
    """读取窗口客户区宽度。"""
    rect = win32gui.GetClientRect(window_handle)
    # GetClientRect 返回 (0, 0, width, height)
    return rect[2]


def window_rect_to_client_height(window_handle: int) -> int:
    """读取窗口客户区高度。"""
    rect = win32gui.GetClientRect(window_handle)
    return rect[3]


def list_windows(
    keywords: Sequence[str] | None = None,
    include_invisible: bool = False,
) -> list[WindowInfo]:
    """枚举所有顶层窗口，按关键字过滤。

    Args:
        keywords: 标题或类名包含任一关键字的窗口会被保留；
                  None 或空列表表示返回所有可见窗口
        include_invisible: 是否包含不可见窗口

    Returns:
        匹配的窗口列表
    """
    results: list[WindowInfo] = []

    def _enum_proc(handle: int, _lparam: int) -> bool:
        title = win32gui.GetWindowText(handle)
        class_name = win32gui.GetClassName(handle)

        if not title and not include_invisible:
            return True

        style = win32gui.GetWindowLong(handle, win32con.GWL_STYLE)
        is_visible = bool(style & win32con.WS_VISIBLE)
        if not is_visible and not include_invisible:
            return True

        if keywords:
            combined = f"{title} {class_name}".lower()
            if not any(kw.lower() in combined for kw in keywords):
                return True

        try:
            info = get_window_info(handle)
            results.append(info)
        except Exception:
            pass
        return True

    win32gui.EnumWindows(_enum_proc, 0)
    return results


def find_window_by_title(title: str) -> int | None:
    """精确匹配窗口标题，返回句柄。"""
    handle = win32gui.FindWindow(None, title)
    return handle if handle else None


def find_windows_by_keyword(keyword: str) -> list[WindowInfo]:
    """按关键字模糊匹配窗口标题和类名。"""
    return list_windows(keywords=[keyword], include_invisible=False)


def find_window_smart(title: str, class_name: str = "") -> int | None:
    """智能查找窗口句柄。

    先精确匹配标题，失败时按标题关键字模糊匹配。
    如果提供 class_name，多个候选时用它缩小范围。
    """
    handle = find_window_by_title(title)
    if handle:
        return handle

    candidates = find_windows_by_keyword(title)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0].handle
    if class_name:
        for c in candidates:
            if c.class_name == class_name:
                return c.handle
    return None


def activate_window(window_handle: int) -> bool:
    """激活窗口到前台。

    SetForegroundWindow 受 Windows 限制（调用方需为前台进程或满足条件），
    从终端进程调用通常允许激活其他窗口。窗口最小化时先恢复。

    Args:
        window_handle: 窗口句柄

    Returns:
        是否成功激活
    """
    try:
        if win32gui.IsIconic(window_handle):
            win32gui.ShowWindow(window_handle, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(window_handle)
        return True
    except Exception:
        return False


class _LASTINPUTINFO(ctypes.Structure):
    """GetLastInputInfo 用的结构体。"""

    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("dwTime", ctypes.c_uint),
    ]


def get_input_idle_seconds() -> float:
    """返回系统距上一次鼠标/键盘活动的空闲秒数（全系统，含其他程序）。

    用于判断用户当前是否在使用电脑：游戏窗口失焦但系统无输入活动时，
    可以安全地把游戏窗口切回前台继续任务；用户在操作时不抢焦点。
    API 失败时返回 0.0（视为正在活动，不触发自动切回）。
    """
    info = _LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(_LASTINPUTINFO)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return 0.0
    idle_ms = ctypes.windll.kernel32.GetTickCount() - info.dwTime
    return max(0.0, idle_ms / 1000.0)


def adjust_window_size(
    window_handle: int,
    target_client_width: int,
    target_client_height: int,
) -> WindowInfo:
    """调整窗口大小使客户区达到目标尺寸。

    通过 GetWindowRect 计算边框和标题栏占用，再反推窗口矩形，
    最后用 MoveWindow 调整。保持窗口左上角位置不变。

    Args:
        window_handle: 窗口句柄
        target_client_width: 目标客户区宽度（像素）
        target_client_height: 目标客户区高度（像素）

    Returns:
        调整后的窗口信息

    Raises:
        RuntimeError: 窗口最小化或调整失败
    """
    info = get_window_info(window_handle)
    if info.is_minimized:
        raise RuntimeError(f"窗口已最小化，句柄: {window_handle}，请先恢复窗口")

    # 计算边框和标题栏占用
    wl, wt, wr, wb = info.window_rect
    cl, ct, cw, ch = info.client_rect

    border_left = cl - wl
    border_right = wr - cl - cw
    border_top = ct - wt
    border_bottom = wb - ct - ch

    # 新窗口矩形：左上角不变，右下角按目标客户区 + 边框计算
    new_right = wl + border_left + target_client_width + border_right
    new_bottom = wt + border_top + target_client_height + border_bottom

    # SWP_NOZORDER=0x0004 SWP_NOACTIVATE=0x0010
    win32gui.SetWindowPos(
        window_handle,
        0,  # hWndInsertAfter=0，配合 SWP_NOZORDER 保持 z 顺序
        wl,
        wt,
        new_right - wl,
        new_bottom - wt,
        0x0004 | 0x0010,
    )

    # 读取调整后的实际窗口信息
    adjusted = get_window_info(window_handle)
    return adjusted


# ---------------------------------------------------------------------------
# ScreenCapture
# ---------------------------------------------------------------------------


class ScreenCapture:
    """窗口截图器。

    负责查找窗口、读取客户区尺寸、使用 MSS 截取客户区对应屏幕区域，
    并生成 WindowContext 快照。
    """

    def __init__(self) -> None:
        self._frame_counter = 0

    def find_window(self, title: str) -> int | None:
        """根据标题查找窗口句柄。

        先精确匹配，失败时按标题关键字模糊匹配。
        """
        handle = find_window_by_title(title)
        if handle:
            return handle

        candidates = find_windows_by_keyword(title)
        if len(candidates) == 1:
            return candidates[0].handle
        if len(candidates) > 1:
            # 多个候选时不自动选择，由调用方处理
            return None
        return None

    def get_window_info(self, window_handle: int) -> WindowInfo:
        """读取窗口信息。"""
        return get_window_info(window_handle)

    def validate_context(self, expected: WindowContext) -> tuple[bool, str]:
        """在输入前重读窗口元数据，确认识别上下文仍然有效。"""
        try:
            current = get_window_info(expected.window_handle)
        except Exception as exc:
            return False, f"窗口句柄已失效: {exc!r}"

        if current.is_minimized or not current.is_foreground:
            return False, "窗口最小化或非前台"
        if current.client_rect != expected.client_rect_screen:
            return False, "客户区位置或尺寸已变化"
        if current.client_size != expected.client_size:
            return False, "客户区尺寸已变化"
        if current.dpi != expected.dpi:
            return False, "窗口 DPI 已变化"
        if current.monitor_id != expected.monitor_id:
            return False, "窗口所在显示器已变化"
        return True, ""

    def capture(self, window_handle: int) -> tuple[WindowContext, object]:
        """截取窗口客户区，返回 WindowContext 和截图帧。

        Args:
            window_handle: 窗口句柄

        Returns:
            (WindowContext, 截图帧)，截图帧为 numpy ndarray
        """
        info = get_window_info(window_handle)

        if info.is_minimized:
            raise RuntimeError(f"窗口已最小化，句柄: {window_handle}")

        left, top, width, height = info.client_rect
        if width <= 0 or height <= 0:
            raise RuntimeError(f"客户区尺寸异常: {width}x{height}，窗口可能未正确显示")

        with mss() as sct:
            monitor = {
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "mon": 0,
            }
            shot = sct.grab(monitor)
            import numpy as np

            # MSS 返回 BGRA 4 通道，cv2 模板匹配要求与模板（BGR 3 通道）一致，
            # 取前 3 通道转 BGR，否则 matchTemplate 报 type 不匹配
            frame = np.array(shot)[:, :, :3]

        self._frame_counter += 1
        ctx = WindowContext(
            window_handle=window_handle,
            client_rect_screen=(left, top, width, height),
            client_size=(width, height),
            dpi=info.dpi,
            monitor_id=info.monitor_id,
            is_foreground=info.is_foreground,
            is_minimized=info.is_minimized,
            captured_at=time.time(),
            frame_id=self._frame_counter,
        )
        return ctx, frame
