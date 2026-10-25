"""后台金币缓存：主循环喂帧，后台线程限频 OCR，门控读缓存快照。

2026-09-26 替代「只认色」金币门控：新城墙皮肤把底栏染成暗红棕，红色
占比统计恒超阈值，召唤/选技能被永久判「买不起」（实机 00:47 日志）。
数值读取与颜色无关；单次读取的耗时（本机实测 ~1s/区域）由后台线程
吸收，主循环只做一次信箱换手（引用赋值，微秒级）。

并发模型：
- 信箱为单槽 latest-wins——门控只关心最新画面，读者落后时旧帧被自动
  覆盖（队列会积压过期帧白白消耗 CPU）。脏帧（弹窗遮挡底栏）同样
  覆盖信箱：读者跳过且不提交，缓存保留最后一个干净帧的读数。
- 读数在后台线程内串行执行；快照换手加锁，门控读侧拿引用后立即释放。
- 读取失败（解析不出数字/引擎异常）不动缓存，只累计连续失败次数并
  指数退避；快照超龄由 snapshot() 返回 None，调用方按「金币缺失不
  拦截」的既有语义放行——本模块不引入任何等待/阻塞。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from wlxq_bot.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class _GoldRequest:
    """信箱记录：一次待读取的干净帧（帧引用 + 已解析的客户区像素 ROI）。"""

    frame: Any
    frame_id: int
    clean: bool
    current_roi: tuple[int, int, int, int]
    cost_roi: tuple[int, int, int, int]
    cost_key: str  # 读数写入快照的键：summon_cost / skill_cost


class GoldCache:
    """金币后台缓存服务。

    Args:
        reader: 数值读取器，接口为 ``read_pair(frame, current_roi, cost_roi)``
            返回 ``(current, cost)``（解析失败为 None）。生产环境用
            ``ocr.build_background_gold_reader()`` 构造独立引擎实例；
            测试注入假读取器。
        read_min_interval_seconds: 两次成功读取的最小间隔，也约束花钱
            事件触发的立即补读。
        refresh_interval_seconds: 常规刷新间隔（干净帧到达且超龄才读）。
        stale_after_seconds: 快照年龄超过此值视为缺失，snapshot() 返回
            None——只兜「后台线程死亡」；调用方按缺失放行。
        clock: 单调时钟，可注入测试。
        poll_seconds: 后台线程轮询信箱的睡眠步长。
    """

    def __init__(
        self,
        reader: Any,
        *,
        read_min_interval_seconds: float = 1.0,
        refresh_interval_seconds: float = 2.0,
        stale_after_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        poll_seconds: float = 0.25,
    ) -> None:
        self._reader = reader
        self._read_min_interval = max(0.0, read_min_interval_seconds)
        self._refresh_interval = max(0.0, refresh_interval_seconds)
        self._stale_after = max(0.0, stale_after_seconds)
        self._clock = clock
        self._poll_seconds = max(0.05, poll_seconds)

        self._mailbox_lock = threading.Lock()
        self._request: _GoldRequest | None = None
        self._cache_lock = threading.Lock()
        self._snapshot: dict[str, Any] | None = None
        self._snapshot_at: float | None = None
        self._snapshot_frame_id = 0
        self._invalidate_pending = False
        self._last_success_at = float("-inf")
        self._last_attempt_at = float("-inf")
        self._consecutive_failures = 0
        self._stop_requested = False
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 主循环接口（非阻塞）
    # ------------------------------------------------------------------

    def submit(
        self,
        frame: Any,
        *,
        frame_id: int,
        clean: bool,
        current_roi: tuple[int, int, int, int],
        cost_roi: tuple[int, int, int, int],
        cost_key: str,
    ) -> None:
        """投递最新帧；覆盖旧记录，绝不排队。首次调用懒启动后台线程。"""
        request = _GoldRequest(
            frame=frame,
            frame_id=frame_id,
            clean=clean,
            current_roi=current_roi,
            cost_roi=cost_roi,
            cost_key=cost_key,
        )
        with self._mailbox_lock:
            self._request = request
        self._ensure_thread()

    def invalidate(self) -> None:
        """花钱事件（召唤落位/技能购买）：下一帧干净帧立即补读。"""
        with self._mailbox_lock:
            self._invalidate_pending = True

    def clear(self) -> None:
        """清空缓存（每局重置）：新局金币与本局无关，首查走缺失放行。"""
        with self._cache_lock:
            self._snapshot = None
            self._snapshot_at = None
        self.invalidate()

    def snapshot(self) -> dict[str, Any] | None:
        """门控用快照：新鲜返回 {current, summon_cost, skill_cost}（键可能
        按状态缺失），超龄/未就绪返回 None（调用方按金币缺失放行）。"""
        with self._cache_lock:
            snapshot = dict(self._snapshot) if self._snapshot is not None else None
            at = self._snapshot_at
        if snapshot is None or at is None:
            return None
        if self._clock() - at > self._stale_after:
            return None
        return snapshot

    def read_now(
        self,
        frame: Any,
        *,
        frame_id: int,
        clean: bool,
        current_roi: tuple[int, int, int, int],
        cost_roi: tuple[int, int, int, int],
        cost_key: str,
    ) -> dict[str, Any] | None:
        """同步现读当前帧（绕过限频与后台线程），读到的值提交进缓存。

        召唤拒判评估专用（2026-10-14 实机三连误判事故）：评估读数必须来自
        点击之后的画面，而后台快照天然滞后一个决策周期——召唤动画窗口内
        的旧值「费用未变化」不构成拒绝证据。调用方在识别帧上调用本方法，
        读数必然晚于上一周期的点击。读取失败返回 None 且不动缓存，调用方
        回退 snapshot() 的既有缺失语义。脏帧（clean=False）直接返回 None：
        遮挡下的解析结果不可信。
        """
        if not clean:
            return None
        try:
            current, cost = self._reader.read_pair(frame, current_roi, cost_roi)
        except Exception as exc:
            logger.debug("frame_id=%d 金币同步现读异常：%r", frame_id, exc)
            return None
        if current is None and cost is None:
            logger.debug("frame_id=%d 金币同步现读失败（解析不出数字）", frame_id)
            return None
        now = self._clock()
        with self._cache_lock:
            values = dict(self._snapshot) if self._snapshot is not None else {}
            if current is not None:
                values["current"] = current
            if cost is not None:
                values[cost_key] = cost
            values["frame_id"] = frame_id
            self._snapshot = values
            self._snapshot_at = now
            self._snapshot_frame_id = frame_id
            self._consecutive_failures = 0
            self._invalidate_pending = False
            self._last_success_at = now
            self._last_attempt_at = now
        return dict(values)

    def stop(self) -> None:
        """请求后台线程退出；幂等。daemon 线程随进程退出兜底。"""
        self._stop_requested = True
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # 后台线程
    # ------------------------------------------------------------------

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._thread_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run, name="gold-cache", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        while not self._stop_requested:
            try:
                self._pump()
            except Exception as exc:  # 后台线程任何异常都不拖垮任务
                logger.warning("金币后台读取异常：%r", exc)
            time.sleep(self._poll_seconds)

    def _pump(self) -> None:
        """单次轮询：取信箱 → 节流判定 → 读取 → 提交快照。可注入时钟单测。"""
        with self._mailbox_lock:
            request = self._request
            self._request = None
        now = self._clock()
        if request is None or not request.clean:
            return
        if now - self._reference_at() < self._required_gap():
            return
        if now - self._last_attempt_at < self._read_min_interval:
            return
        self._last_attempt_at = now
        try:
            current, cost = self._reader.read_pair(
                request.frame, request.current_roi, request.cost_roi
            )
        except Exception as exc:
            self._consecutive_failures += 1
            logger.warning(
                "frame_id=%d 金币后台读取异常（连续 %d 次）：%r",
                request.frame_id,
                self._consecutive_failures,
                exc,
            )
            return
        if current is None and cost is None:
            # 解析不出数字（遮挡/碎字）→ 不提交，缓存保持上一个好值
            self._consecutive_failures += 1
            logger.debug(
                "frame_id=%d 金币后台读取失败（连续 %d 次）",
                request.frame_id,
                self._consecutive_failures,
            )
            return
        with self._cache_lock:
            values = dict(self._snapshot) if self._snapshot is not None else {}
            if current is not None:
                values["current"] = current
            if cost is not None:
                values[request.cost_key] = cost
            self._snapshot = values
            self._snapshot_at = now
            self._snapshot_frame_id = request.frame_id
        self._consecutive_failures = 0
        self._invalidate_pending = False
        self._last_success_at = now
        logger.debug(
            "frame_id=%d 金币后台读取 current=%s %s=%s",
            request.frame_id,
            current,
            request.cost_key,
            cost,
        )

    # 退避：连续失败时拉长刷新间隔（2 的幂，封顶 32 倍），避免对着
    # 长期读不出数字的画面热循环 OCR；花钱失效标记不受退避限制。
    # 锚点取 max(上次成功, 上次尝试)——从未成功过时按上次尝试计算，
    # 否则失败序列的退避形同虚设
    def _required_gap(self) -> float:
        if self._invalidate_pending:
            return 0.0
        backoff = 2 ** min(self._consecutive_failures, 5)
        return self._refresh_interval * backoff

    def _reference_at(self) -> float:
        return max(self._last_success_at, self._last_attempt_at)
