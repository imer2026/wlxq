"""find 命令单元测试。

用程序生成的图片测试单模板识别命令，不依赖游戏窗口。
自包含可重复。

图案用「左红右蓝」（有水平边缘，避免 TM_CCOEFF_NORMED 除零）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from wlxq_bot.cli import app

runner = CliRunner()

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


def _make_frame(width: int = 200, height: int = 200) -> np.ndarray:
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
# 测试用例
# ---------------------------------------------------------------------------


class TestFindMatch:
    def test_match_found(self, tmp_path: Path) -> None:
        """模板在画面里能匹配到，报告置信度和位置，生成标注图。"""
        pattern = _make_pattern()
        frame = _make_frame()
        _place(frame, 50, 50, pattern)  # 中心 (65,65)

        shot = tmp_path / "shot.png"
        _save_png(shot, frame)
        tpl = tmp_path / "tpl.png"
        _save_png(tpl, pattern)

        annotated = tmp_path / "out.png"
        result = runner.invoke(
            app,
            [
                "find",
                str(tpl),
                "-t",
                "0.9",
                "--image",
                str(shot),
                "--save",
                str(annotated),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "已识别到" in result.output
        assert annotated.exists()

    def test_below_threshold_reports_best_confidence(self, tmp_path: Path) -> None:
        """阈值过高未命中时，仍报告最高置信度供参考。"""
        # 模板是左红右蓝，画面里放不同的图案（上绿下黄），置信度低
        pattern_tpl = _make_pattern()
        pattern_other = np.zeros((30, 30, 3), dtype=np.uint8)
        pattern_other[:15] = (0, 255, 0)  # 上绿 BGR
        pattern_other[15:] = (0, 100, 100)  # 下黄 BGR

        frame = _make_frame()
        _place(frame, 50, 50, pattern_other)

        shot = tmp_path / "shot.png"
        _save_png(shot, frame)
        tpl = tmp_path / "tpl.png"
        _save_png(tpl, pattern_tpl)

        result = runner.invoke(
            app,
            ["find", str(tpl), "-t", "0.9", "--image", str(shot)],
        )

        assert result.exit_code == 0, result.output
        assert "未识别到" in result.output
        assert "最高置信度" in result.output

    def test_high_threshold_still_matches_exact(self, tmp_path: Path) -> None:
        """完全相同的模板+画面，高阈值仍能命中（置信度接近 1）。"""
        pattern = _make_pattern()
        frame = _make_frame()
        _place(frame, 50, 50, pattern)

        shot = tmp_path / "shot.png"
        _save_png(shot, frame)
        tpl = tmp_path / "tpl.png"
        _save_png(tpl, pattern)

        result = runner.invoke(
            app,
            ["find", str(tpl), "-t", "0.99", "--image", str(shot)],
        )

        assert result.exit_code == 0, result.output
        assert "已识别到" in result.output


class TestFindErrors:
    def test_template_not_found(self, tmp_path: Path) -> None:
        """模板文件不存在时退出码 1。"""
        result = runner.invoke(
            app,
            ["find", str(tmp_path / "nope.png")],
        )
        assert result.exit_code == 1
        assert "不存在" in result.output

    def test_bad_threshold(self, tmp_path: Path) -> None:
        """threshold 超出 0~1 时退出码 1。"""
        pattern = _make_pattern()
        tpl = tmp_path / "tpl.png"
        _save_png(tpl, pattern)

        result = runner.invoke(
            app,
            ["find", str(tpl), "-t", "1.5"],
        )
        assert result.exit_code == 1
        assert "0~1" in result.output

    def test_template_larger_than_frame(self, tmp_path: Path) -> None:
        """模板比画面大时退出码 1。"""
        big_pattern = _make_pattern(w=400, h=400)
        tpl = tmp_path / "big.png"
        _save_png(tpl, big_pattern)

        frame = _make_frame(width=100, height=100)
        shot = tmp_path / "shot.png"
        _save_png(shot, frame)

        result = runner.invoke(
            app,
            ["find", str(tpl), "--image", str(shot)],
        )
        assert result.exit_code == 1
        assert "大于画面" in result.output

    def test_image_not_found(self, tmp_path: Path) -> None:
        """--image 指定的截图不存在时退出码 1。"""
        pattern = _make_pattern()
        tpl = tmp_path / "tpl.png"
        _save_png(tpl, pattern)

        result = runner.invoke(
            app,
            ["find", str(tpl), "--image", str(tmp_path / "nope.png")],
        )
        assert result.exit_code == 1
        assert "不存在" in result.output
