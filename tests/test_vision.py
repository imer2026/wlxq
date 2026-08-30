"""Vision 模板匹配单元测试。

用程序生成的图片测试 match_template / match_template_set / _nms，
不依赖游戏截图，自包含可重复。

模板用「左红右蓝」图案（有水平边缘），避免纯色导致 TM_CCOEFF_NORMED 除零。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from wlxq_bot.models import MatchResult
from wlxq_bot.perception.vision import Vision

# ---------------------------------------------------------------------------
# 辅助：生成测试图片
# ---------------------------------------------------------------------------

PATTERN_W = 30
PATTERN_H = 30


def _make_pattern(w: int = PATTERN_W, h: int = PATTERN_H) -> np.ndarray:
    """生成「左红右蓝」图案模板（有边缘，避免纯色除零）。"""
    t = np.zeros((h, w, 3), dtype=np.uint8)
    t[:, : w // 2] = (0, 0, 255)  # 左红 BGR
    t[:, w // 2 :] = (255, 0, 0)  # 右蓝 BGR
    return t


def _make_frame(width: int = 300, height: int = 300) -> np.ndarray:
    """灰色背景大图。"""
    return np.full((height, width, 3), (50, 50, 50), dtype=np.uint8)


def _place(frame: np.ndarray, x: int, y: int, pattern: np.ndarray) -> None:
    """在大图 (x, y) 位置放置图案。"""
    h, w = pattern.shape[:2]
    frame[y : y + h, x : x + w] = pattern


def _save_png(path: Path, img: np.ndarray) -> None:
    """保存图片（兼容中文路径用 imencode + tofile）。"""
    success, buf = cv2.imencode(".png", img)
    assert success, f"imencode 失败: {path}"
    buf.tofile(str(path))


# ---------------------------------------------------------------------------
# match_template（单模板）
# ---------------------------------------------------------------------------


class TestMatchTemplate:
    def test_single_target(self, tmp_path: Path) -> None:
        """单模板匹配单个目标，返回中心位置和高置信度。"""
        pattern = _make_pattern()
        frame = _make_frame()
        _place(frame, 100, 100, pattern)

        tpl_path = tmp_path / "tpl.png"
        _save_png(tpl_path, pattern)

        vision = Vision()
        result = vision.match_template(frame, str(tpl_path), threshold=0.9)

        assert result is not None
        # 匹配左上角 (100,100)，中心 = 100 + 15 = 115
        assert result.position == (115, 115)
        assert result.confidence >= 0.9

    def test_below_threshold_returns_none(self, tmp_path: Path) -> None:
        """目标不存在时，置信度低于阈值返回 None。"""
        pattern = _make_pattern()
        frame = _make_frame()  # 空背景，无目标

        tpl_path = tmp_path / "tpl.png"
        _save_png(tpl_path, pattern)

        vision = Vision()
        result = vision.match_template(frame, str(tpl_path), threshold=0.9)
        assert result is None

    def test_template_too_large_returns_none(self, tmp_path: Path) -> None:
        """模板比帧大时返回 None。"""
        pattern = _make_pattern(w=400, h=400)
        frame = _make_frame(width=100, height=100)

        tpl_path = tmp_path / "big.png"
        _save_png(tpl_path, pattern)

        vision = Vision()
        result = vision.match_template(frame, str(tpl_path))
        assert result is None

    def test_load_failure_returns_none(self, tmp_path: Path) -> None:
        """模板文件不存在时返回 None。"""
        frame = _make_frame()
        vision = Vision()
        result = vision.match_template(frame, str(tmp_path / "nope.png"))
        assert result is None

    def test_roi_restricts_search(self, tmp_path: Path) -> None:
        """ROI 限定搜索区域，目标在 ROI 外则找不到。"""
        pattern = _make_pattern()
        frame = _make_frame()
        _place(frame, 200, 200, pattern)  # 目标在右下

        tpl_path = tmp_path / "tpl.png"
        _save_png(tpl_path, pattern)

        vision = Vision()
        # ROI 只搜左上 0~100，目标在 200 处，找不到
        result = vision.match_template(frame, str(tpl_path), roi=(0, 0, 100, 100), threshold=0.9)
        assert result is None

        # ROI 包含目标
        result = vision.match_template(
            frame, str(tpl_path), roi=(150, 150, 150, 150), threshold=0.9
        )
        assert result is not None
        # ROI offset 150, 匹配左上角 (200,200) - 150 = (50,50) in ROI
        # 中心 = 50+15+150 = 215
        assert result.position == (215, 215)


# ---------------------------------------------------------------------------
# match_template_set（多模板 + NMS）
# ---------------------------------------------------------------------------


class TestMatchTemplateSet:
    def test_find_multiple_targets(self, tmp_path: Path) -> None:
        """3 个相同目标等间距排列，全部找到。"""
        pattern = _make_pattern()
        frame = _make_frame()
        positions = [(50, 50), (150, 50), (250, 50)]
        for x, y in positions:
            _place(frame, x, y, pattern)

        tpl_path = tmp_path / "tpl.png"
        _save_png(tpl_path, pattern)

        vision = Vision()
        results = vision.match_template_set(frame, [str(tpl_path)], threshold=0.9)

        assert len(results) == 3
        # 中心 = 左上角 + 15
        expected_centers = {(x + 15, y + 15) for x, y in positions}
        actual_centers = {r.position for r in results}
        assert actual_centers == expected_centers

    def test_no_match_returns_empty(self, tmp_path: Path) -> None:
        """无目标时返回空列表。"""
        pattern = _make_pattern()
        frame = _make_frame()

        tpl_path = tmp_path / "tpl.png"
        _save_png(tpl_path, pattern)

        vision = Vision()
        results = vision.match_template_set(frame, [str(tpl_path)], threshold=0.9)
        assert results == []

    def test_multiple_template_variants(self, tmp_path: Path) -> None:
        """两个不同图案的模板，各自匹配各自的目标。"""
        pattern_a = _make_pattern()  # 左红右蓝
        pattern_b = np.zeros((30, 30, 3), dtype=np.uint8)
        pattern_b[:15] = (0, 255, 0)  # 上绿
        pattern_b[15:] = (0, 100, 100)  # 下黄

        frame = _make_frame()
        _place(frame, 50, 50, pattern_a)
        _place(frame, 150, 150, pattern_b)

        tpl_a = tmp_path / "a.png"
        tpl_b = tmp_path / "b.png"
        _save_png(tpl_a, pattern_a)
        _save_png(tpl_b, pattern_b)

        vision = Vision()
        results = vision.match_template_set(frame, [str(tpl_a), str(tpl_b)], threshold=0.9)

        assert len(results) == 2
        centers = {r.position for r in results}
        assert (65, 65) in centers  # pattern_a at (50,50)
        assert (165, 165) in centers  # pattern_b at (150,150)

    def test_nms_merges_nearby(self, tmp_path: Path) -> None:
        """同一目标周围多个高分点被 NMS 合并为 1 个。"""
        pattern = _make_pattern()
        frame = _make_frame()
        _place(frame, 100, 100, pattern)

        tpl_path = tmp_path / "tpl.png"
        _save_png(tpl_path, pattern)

        vision = Vision()
        # 低阈值会捕获目标周围一片高分点
        results = vision.match_template_set(frame, [str(tpl_path)], threshold=0.5)

        # NMS 后应该只有 1 个（同一目标）
        assert len(results) == 1
        assert results[0].position == (115, 115)

    def test_results_sorted_by_confidence(self, tmp_path: Path) -> None:
        """结果按置信度降序。"""
        pattern = _make_pattern()
        frame = _make_frame()
        _place(frame, 50, 50, pattern)
        _place(frame, 150, 50, pattern)

        tpl_path = tmp_path / "tpl.png"
        _save_png(tpl_path, pattern)

        vision = Vision()
        results = vision.match_template_set(frame, [str(tpl_path)], threshold=0.9)

        assert len(results) == 2
        assert results[0].confidence >= results[1].confidence

    def test_custom_nms_dist(self, tmp_path: Path) -> None:
        """自定义 NMS 距离，距离内合并、距离外保留。"""
        pattern = _make_pattern()
        frame = _make_frame()
        # 两个不重叠目标，中心距 50
        _place(frame, 50, 50, pattern)  # 中心 (65,65)
        _place(frame, 100, 50, pattern)  # 中心 (115,65)

        tpl_path = tmp_path / "tpl.png"
        _save_png(tpl_path, pattern)

        vision = Vision()
        # NMS 距离 60 > 50，合并成一个
        merged = vision.match_template_set(frame, [str(tpl_path)], threshold=0.9, nms_dist=60)
        assert len(merged) == 1

        # NMS 距离 40 < 50，保留两个
        separate = vision.match_template_set(frame, [str(tpl_path)], threshold=0.9, nms_dist=40)
        assert len(separate) == 2

    def test_empty_template_paths(self, tmp_path: Path) -> None:
        """空模板列表返回空结果。"""
        frame = _make_frame()
        vision = Vision()
        results = vision.match_template_set(frame, [], threshold=0.9)
        assert results == []


# ---------------------------------------------------------------------------
# _nms（直接测 NMS 逻辑）
# ---------------------------------------------------------------------------


class TestNms:
    def test_merge_nearby(self) -> None:
        """距离内的候选合并，保留置信度最高的。"""
        candidates = [
            (100, 100, 0.9, "a", 30, 30),
            (105, 105, 0.8, "a", 30, 30),  # 距 ~7，合并
        ]
        keep = Vision._nms(candidates, dist=20)
        assert len(keep) == 1
        assert keep[0][2] == 0.9  # 保留高置信度

    def test_keep_far_apart(self) -> None:
        """距离外的候选都保留。"""
        candidates = [
            (100, 100, 0.9, "a", 30, 30),
            (300, 300, 0.8, "a", 30, 30),  # 距 ~283，保留
        ]
        keep = Vision._nms(candidates, dist=20)
        assert len(keep) == 2

    def test_empty(self) -> None:
        assert Vision._nms([], dist=20) == []

    def test_sorted_by_confidence(self) -> None:
        """结果按置信度降序。"""
        candidates = [
            (100, 100, 0.7, "a", 30, 30),
            (300, 300, 0.95, "b", 30, 30),
            (500, 500, 0.8, "c", 30, 30),
        ]
        keep = Vision._nms(candidates, dist=20)
        assert [c[2] for c in keep] == [0.95, 0.8, 0.7]


# ---------------------------------------------------------------------------
# _load_template（缓存 + 中文路径）
# ---------------------------------------------------------------------------


class TestLoadTemplate:
    def test_cache(self, tmp_path: Path) -> None:
        """同一模板第二次加载走缓存。"""
        pattern = _make_pattern()
        tpl_path = tmp_path / "tpl.png"
        _save_png(tpl_path, pattern)

        vision = Vision()
        img1 = vision._load_template(str(tpl_path))
        img2 = vision._load_template(str(tpl_path))

        assert img1 is not None
        assert img2 is not None
        # 缓存返回同一对象
        assert img1 is img2

    def test_chinese_path(self, tmp_path: Path) -> None:
        """中文路径能正常加载。"""
        pattern = _make_pattern()
        cn_path = tmp_path / "强袭.png"
        _save_png(cn_path, pattern)

        vision = Vision()
        img = vision._load_template(str(cn_path))
        assert img is not None
        assert img.shape == (PATTERN_H, PATTERN_W, 3)

    def test_nonexistent_returns_none(self, tmp_path: Path) -> None:
        vision = Vision()
        assert vision._load_template(str(tmp_path / "nope.png")) is None


# ---------------------------------------------------------------------------
# annotate
# ---------------------------------------------------------------------------


class TestAnnotate:
    def test_returns_annotated_frame(self, tmp_path: Path) -> None:
        """annotate 返回标注后的帧（形状不变）。"""
        frame = _make_frame()
        matches = [
            MatchResult(
                template_name="a",
                position=(100, 100),
                confidence=0.9,
            )
        ]
        vision = Vision()
        annotated = vision.annotate(frame, matches)
        assert annotated.shape == frame.shape
        # 标注后帧和原帧不同（画了矩形）
        assert not np.array_equal(annotated, frame)

    def test_empty_matches(self) -> None:
        """空匹配列表返回原帧副本。"""
        frame = _make_frame()
        vision = Vision()
        annotated = vision.annotate(frame, [])
        assert np.array_equal(annotated, frame)
