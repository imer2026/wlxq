"""时间相关工具：随机延迟、超时检查。

随机延迟、超时和重试必须可配置、可测试，并设置明确上限。
"""

from __future__ import annotations

import random
import time


def random_delay(min_sec: float, max_sec: float) -> float:
    """生成随机延迟并等待。

    Args:
        min_sec: 最小延迟（秒）
        max_sec: 最大延迟（秒）

    Returns:
        实际等待时间（秒）
    """
    delay = random.uniform(min_sec, max_sec)
    time.sleep(delay)
    return delay


class Timeout:
    """超时检查器。

    用于无进展超时判断，避免界面长时间不变化时无限点击。
    """

    def __init__(self, timeout_sec: float) -> None:
        self._timeout = timeout_sec
        self._start = time.time()

    def reset(self) -> None:
        """重置计时。"""
        self._start = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self._start

    @property
    def expired(self) -> bool:
        return self.elapsed >= self._timeout
