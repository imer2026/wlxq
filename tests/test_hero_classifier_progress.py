"""英雄格离线长任务节流进度日志测试。"""

from __future__ import annotations

from wlxq_bot.hero_classifier.progress import ProgressLogger


class _Logger:
    def __init__(self) -> None:
        self.records: list[tuple[str, tuple]] = []

    def info(self, message: str, *args) -> None:
        self.records.append((message, args))


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_progress_logs_start_periodic_update_and_finish() -> None:
    logger = _Logger()
    clock = _Clock()
    progress = ProgressLogger(  # type: ignore[arg-type]
        logger,
        "一级聚类",
        100,
        interval_seconds=5.0,
        monotonic=clock,
    )

    progress.start(detail="threshold=35")
    clock.value = 4.9
    progress.update(20, detail="clusters=3")
    clock.value = 5.0
    progress.update(25, detail="clusters=4")
    clock.value = 10.0
    progress.update(80, detail="clusters=8")
    clock.value = 12.0
    progress.finish(100, detail="clusters=10")

    assert [record[0] for record in logger.records] == [
        "%s 开始 total=%d%s",
        "%s 进度 processed=%d/%d percent=%.1f%% elapsed=%.1fs%s",
        "%s 进度 processed=%d/%d percent=%.1f%% elapsed=%.1fs%s",
        "%s 完成 processed=%d/%d percent=%.1f%% elapsed=%.1fs%s",
    ]
    assert logger.records[1][1][1:4] == (25, 100, 25.0)
    assert logger.records[2][1][1:4] == (80, 100, 80.0)
