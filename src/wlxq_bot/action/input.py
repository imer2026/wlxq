"""Input Controller：封装鼠标、键盘模拟输入。

只暴露给 Action Executor 调用，任务代码不得直接使用。
使用 PyAutoGUI 实现实际输入，后续可选替换为 pydirectinput。

约束：
- 所有坐标参数为屏幕坐标（已由 WindowContext 转换）
- 点击前后支持随机延迟
- 不做任何业务判断
"""

from __future__ import annotations

import random

import pyautogui


class InputController:
    """输入控制器。

    使用 PyAutoGUI 实现鼠标和键盘输入。
    点击时加入微小随机抖动，模拟人类操作。
    """

    def __init__(self, jitter: int = 3, release_dwell: float = 0.15) -> None:
        """初始化输入控制器。

        Args:
            jitter: 点击位置的随机抖动范围（像素），在目标点周围 ±jitter 范围内随机偏移
            release_dwell: 拖动到达终点后、松开鼠标前的停留秒数。游戏内英雄
                以缓动跟随指针，立即松开会因英雄尚未到位被弹回原格（实机
                确认 2026-08-15：合成拖动连续落空的根因）
        """
        self._jitter = jitter
        self._release_dwell = release_dwell

    def click(self, x: int, y: int, duration: float = 0.08) -> None:
        """在屏幕坐标 (x, y) 处点击。

        加入微小随机抖动，避免每次都精确点击同一个点。
        """
        jx = x + random.randint(-self._jitter, self._jitter)
        jy = y + random.randint(-self._jitter, self._jitter)
        pyautogui.click(jx, jy)

    def drag(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        duration: float = 0.5,
        pause: float = 0.2,
    ) -> None:
        """从 start 拖动到 end。

        序列：指针先落到起点稍停（游戏先注册指针位置）→ 按下 → 插值移动到
        终点 → 停留 ``release_dwell`` 让英雄跟随到位 → 松开。不用
        ``pyautogui.drag`` 一步到位，因为它移动完成立即松开，游戏内英雄
        以缓动跟随指针，来不及到位就会被弹回原格。
        """
        pyautogui.moveTo(start[0], start[1])
        pyautogui.sleep(0.05)
        pyautogui.mouseDown()
        pyautogui.moveTo(end[0], end[1], duration=duration)
        pyautogui.sleep(self._release_dwell)
        pyautogui.mouseUp()
        if pause > 0:
            pyautogui.sleep(pause)

    def press_key(self, key: str) -> None:
        """按键。"""
        pyautogui.press(key)


class FakeInput:
    """测试用 Fake Input，记录所有调用不实际操作。

    自动化测试必须使用 FakeInput，不得真实点击用户桌面。
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def click(self, x: int, y: int, duration: float = 0.08) -> None:
        self.calls.append(("click", str(x), str(y), str(duration)))

    def drag(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        duration: float = 0.5,
        pause: float = 0.2,
    ) -> None:
        self.calls.append(("drag", str(start), str(end), str(duration)))

    def press_key(self, key: str) -> None:
        self.calls.append(("key", key))
