"""InputController 拖动序列单元测试。

monkeypatch pyautogui 记录调用序列，不真实移动用户鼠标。
"""

from __future__ import annotations

from wlxq_bot.action import input as input_module
from wlxq_bot.action.input import InputController


def test_drag_dwells_before_release(monkeypatch):
    """拖动必须：落点起点 → 按下 → 插值到终点 → 停留 → 松开。

    游戏内英雄以缓动跟随指针，到终点立即松开会被弹回原格
    （实机确认 2026-08-15 合成拖动连续落空的根因）。
    """
    calls: list[tuple] = []
    monkeypatch.setattr(
        input_module.pyautogui,
        "moveTo",
        lambda x, y, duration=None: calls.append(("moveTo", x, y, duration)),
    )
    monkeypatch.setattr(input_module.pyautogui, "mouseDown", lambda: calls.append(("mouseDown",)))
    monkeypatch.setattr(input_module.pyautogui, "mouseUp", lambda: calls.append(("mouseUp",)))
    monkeypatch.setattr(
        input_module.pyautogui, "sleep", lambda s: calls.append(("sleep", round(s, 3)))
    )

    controller = InputController(release_dwell=0.15)
    controller.drag((10, 20), (110, 220), duration=1.0, pause=0.0)

    assert calls == [
        ("moveTo", 10, 20, None),  # 指针先落到起点
        ("sleep", 0.05),  # 稍停，游戏先注册指针位置
        ("mouseDown",),
        ("moveTo", 110, 220, 1.0),  # 按住插值移动到终点
        ("sleep", 0.15),  # 终点停留，等英雄跟随到位
        ("mouseUp",),
    ]
