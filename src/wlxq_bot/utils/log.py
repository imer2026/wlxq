"""Rich 日志初始化。

提供统一的日志配置。控制台默认 INFO 级别，`--debug` 时切到 DEBUG；
文件日志默认开启且固定 DEBUG，比控制台更详细（配置见 ``configs/default.yaml``
的 ``logging`` 段，详细说明见 ``docs/logging.md``「文件日志」）。

日志级别约定（详见 AGENTS.md「日志约定」）：

- ``ERROR``：动作执行失败、识别失败导致任务暂停、安全触发、窗口丢失等不可继续的情况。
- ``WARNING``：接近阈值、重试、窗口变化可恢复、配置缺失用默认值、坐标被夹紧。
- ``INFO``：正常流程关键节点：命令开始/结束、任务启动、状态迁移、动作完成、截图保存。
- ``DEBUG``：详细诊断：窗口句柄、客户区像素、坐标换算、置信度数值、ROI、延迟值、frame_id。
  仅在 ``--debug`` 下输出到控制台；文件日志始终记录。

日志可包含任务、状态、``frame_id``、置信度、坐标等上下文，
但不得记录账号、令牌或其他敏感信息。
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

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

#: 文件日志格式：完整日期时间、级别、logger 名、来源模块与行号。
_FILE_FORMAT = (
    "%(asctime)s.%(msecs)03d [%(levelname)-7s] %(name)s (%(module)s:%(lineno)d) %(message)s"
)
_FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: 按天保留的秒数换算基数
_SECONDS_PER_DAY = 86400


def _as_level(value: str | int) -> int:
    """把 ``"INFO"`` / ``"DEBUG"`` 等级别名或整数统一为数值级别。"""
    if isinstance(value, int):
        return value
    return logging.getLevelName(value.upper())


def setup_logging(
    level: str | int = "INFO",
    *,
    log_dir: Path | str | None = None,
    file_level: str | int = "DEBUG",
    retention_days: int = 7,
    argv: Sequence[str] | None = None,
) -> logging.Logger:
    """初始化项目根日志器。

    在 CLI 入口（``wlxq-bot --debug``）调用一次即可。重复调用只会更新级别，
    不会重复添加 handler。

    Args:
        level: 控制台日志级别，字符串（``"DEBUG"`` / ``"INFO"`` / …）或整数。
        log_dir: 文件日志目录；``None`` 表示不启用文件日志。目录不存在会自动创建，
            每次调用写一个 ``wlxq-bot_日期_时间.log``。
        file_level: 文件日志级别，默认 ``DEBUG``——无论控制台级别如何，文件都
            记录完整诊断信息，便于事后定位问题。
        retention_days: 启动时清理超过该天数的旧日志文件（按修改时间，仅删 ``*.log``）。
        argv: 记录到文件首行的命令行；默认取 ``sys.argv``，测试可显式传入。

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
    # 避免重复添加 handler；重复调用时更新既有 handler 的级别
    rich_handler = next(
        (h for h in root.handlers if isinstance(h, RichHandler)),
        None,
    )
    if rich_handler is None:
        rich_handler = handler
        root.addHandler(rich_handler)
    rich_handler.setLevel(level)

    # 项目日志不向 root 冒泡（root 无 RichHandler，避免重复输出）
    root.propagate = False
    # 根日志器必须放行控制台和文件中更详细的那个级别，否则 DEBUG 记录在
    # logger 层就被过滤、到不了文件 handler；控制台级别由 rich_handler 自己把关
    if log_dir is not None:
        root.setLevel(min(_as_level(level), _as_level(file_level)))
    else:
        root.setLevel(level)

    # DEBUG 模式下仍把第三方库压到 WARNING，避免 pyautogui / PIL 刷屏。
    third_party_level = logging.WARNING
    for name in _NOISY_THIRD_PARTY:
        logging.getLogger(name).setLevel(third_party_level)

    # 文件日志：已存在则只更新级别，不存在才创建（保持幂等）。
    file_handler = next(
        (h for h in root.handlers if isinstance(h, logging.FileHandler)),
        None,
    )
    if log_dir is not None:
        if file_handler is None:
            file_handler = _create_file_handler(Path(log_dir), retention_days)
            if file_handler is not None:
                root.addHandler(file_handler)
                command_line = " ".join(sys.argv if argv is None else argv)
                root.info(
                    "文件日志已开启: %s（命令行: %s）", file_handler.baseFilename, command_line
                )
        if file_handler is not None:
            file_handler.setLevel(file_level)

    return root


def _create_file_handler(log_dir: Path, retention_days: int) -> logging.FileHandler | None:
    """创建文件日志 handler；失败只警告，不阻塞命令以控制台模式运行。"""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        _cleanup_old_logs(log_dir, retention_days)
        path = log_dir / f"wlxq-bot_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt=_FILE_FORMAT, datefmt=_FILE_DATE_FORMAT))
        return file_handler
    except OSError as exc:
        get_logger(ROOT_LOGGER_NAME).warning("文件日志初始化失败，本次仅输出到控制台: %s", exc)
        return None


def _cleanup_old_logs(log_dir: Path, retention_days: int) -> int:
    """删除 log_dir 下超过保留期的 ``*.log`` 文件（按修改时间），返回删除数量。"""
    cutoff = time.time() - retention_days * _SECONDS_PER_DAY
    root_logger = get_logger(ROOT_LOGGER_NAME)
    removed = 0
    for path in sorted(log_dir.glob("*.log")):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError as exc:
            root_logger.warning("清理过期日志文件失败: %s（%s）", path, exc)
    if removed:
        root_logger.info(
            "已清理 %d 个过期日志文件（保留最近 %d 天）: %s", removed, retention_days, log_dir
        )
    return removed


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
