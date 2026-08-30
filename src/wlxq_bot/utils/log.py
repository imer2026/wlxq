"""Rich 日志初始化。

提供统一的日志配置。默认 INFO 级别，`--debug` 时切到 DEBUG。

日志级别约定（详见 AGENTS.md「日志约定」）：

- ``ERROR``：动作执行失败、识别失败导致任务暂停、安全触发、窗口丢失等不可继续的情况。
- ``WARNING``：接近阈值、重试、窗口变化可恢复、配置缺失用默认值、坐标被夹紧。
- ``INFO``：正常流程关键节点：命令开始/结束、任务启动、状态迁移、动作完成、截图保存。
- ``DEBUG``：详细诊断：窗口句柄、客户区像素、坐标换算、置信度数值、ROI、延迟值、frame_id。
  仅在 ``--debug`` 下输出。

日志可包含任务、状态、``frame_id``、置信度、坐标等上下文，
但不得记录账号、令牌或其他敏感信息。
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler

#: 项目日志命名空间根。所有子 logger 都挂在其下。
ROOT_LOGGER_NAME = "wlxq_bot"

#: DEBUG 模式下需要压到 WARNING 及以上的第三方库，避免刷屏。
_NOISY_THIRD_PARTY = (
    "pyautogui",
    "PIL",
    "PIL.PngImagePlugin",
    "urllib3",
    "asyncio",
    "matplotlib",
)


def setup_logging(level: str | int = "INFO") -> logging.Logger:
    """初始化项目根日志器。

    在 CLI 入口（``wlxq-bot --debug``）调用一次即可。重复调用只会更新级别，
    不会重复添加 handler。

    Args:
        level: 日志级别，字符串（``"DEBUG"`` / ``"INFO"`` / …）或整数。

    Returns:
        配置好的项目根日志器（``wlxq_bot``）。
    """
    handler = RichHandler(
        show_time=True,
        omit_repeated_times=False,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        log_time_format="%H:%M:%S",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    root = logging.getLogger(ROOT_LOGGER_NAME)
    # 避免重复添加 handler
    if not any(isinstance(h, RichHandler) for h in root.handlers):
        root.addHandler(handler)
    # 项目日志不向 root 冒泡（root 无 RichHandler，避免重复输出）
    root.propagate = False
    root.setLevel(level)

    # DEBUG 模式下仍把第三方库压到 WARNING，避免 pyautogui / PIL 刷屏。
    third_party_level = logging.WARNING
    for name in _NOISY_THIRD_PARTY:
        logging.getLogger(name).setLevel(third_party_level)

    return root


def get_logger(name: str) -> logging.Logger:
    """获取子日志器。

    支持两种传参：

    - 短名：``get_logger("screen")`` → ``wlxq_bot.screen``
    - 完整模块名：``get_logger(__name__)`` → 若已以 ``wlxq_bot`` 开头则原样使用，
      否则补前缀。避免 ``wlxq_bot.wlxq_bot.xxx`` 这种重复前缀。
    """
    if name.startswith(ROOT_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
