"""recognize 命令单元测试。

用程序生成的图片 + 临时模板包测试离线识别命令，不依赖游戏窗口和真实截图。
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


def _build_pack_root(
    root: Path,
    pack_size: tuple[int, int] = (200, 200),
) -> Path:
    """在 root 下创建一个模板包目录结构，返回包根目录。"""
    w, h = pack_size
    pack_dir = root / f"{w}x{h}"
    (pack_dir / "buttons").mkdir(parents=True)
    (pack_dir / "skills").mkdir(parents=True)
    (pack_dir / "heroes" / "assault" / "star1").mkdir(parents=True)
    return pack_dir


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------


class TestRecognizeButton:
    def test_finds_button_match(self, tmp_path: Path) -> None:
        """截图里放了按钮图案，recognize 命中并生成标注图。"""
        root = tmp_path / "templates"
        pack_dir = _build_pack_root(root, (200, 200))

        pattern = _make_pattern()
        frame = _make_frame()
        _place(frame, 50, 50, pattern)  # 按钮放在 (50,50)

        shot = tmp_path / "shot.png"
        _save_png(shot, frame)
        _save_png(pack_dir / "buttons" / "btn.png", pattern)

        annotated = tmp_path / "out.png"
        result = runner.invoke(
            app,
            [
                "recognize",
                str(shot),
                "--templates-root",
                str(root),
                "--pack",
                "200x200",
                "--save",
                str(annotated),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "btn" in result.output
        assert "✓" in result.output
        assert annotated.exists()
        # 标注图尺寸和原图一致
        annotated_img = cv2.imdecode(np.fromfile(str(annotated), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert annotated_img.shape == frame.shape

    def test_button_below_threshold_not_matched(self, tmp_path: Path) -> None:
        """截图里没有目标，高阈值下按钮未命中，但命令仍正常退出。"""
        root = tmp_path / "templates"
        pack_dir = _build_pack_root(root, (200, 200))

        pattern = _make_pattern()
        frame = _make_frame()  # 空背景，无目标

        shot = tmp_path / "shot.png"
        _save_png(shot, frame)
        _save_png(pack_dir / "buttons" / "btn.png", pattern)

        annotated = tmp_path / "out.png"
        result = runner.invoke(
            app,
            [
                "recognize",
                str(shot),
                "--templates-root",
                str(root),
                "--pack",
                "200x200",
                "--threshold",
                "0.95",
                "--save",
                str(annotated),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "✗" in result.output
        assert annotated.exists()


class TestRecognizeHero:
    def test_finds_hero_match(self, tmp_path: Path) -> None:
        """截图里放了英雄图案，按 heroes 分类识别命中。"""
        root = tmp_path / "templates"
        pack_dir = _build_pack_root(root, (200, 200))

        pattern = _make_pattern()
        frame = _make_frame()
        _place(frame, 80, 80, pattern)  # 英雄放在 (80,80)

        shot = tmp_path / "shot.png"
        _save_png(shot, frame)
        # 英雄模板放在 heroes/assault/star1/
        _save_png(pack_dir / "heroes" / "assault" / "star1" / "left.png", pattern)

        annotated = tmp_path / "out.png"
        result = runner.invoke(
            app,
            [
                "recognize",
                str(shot),
                "--templates-root",
                str(root),
                "--pack",
                "200x200",
                "--category",
                "heroes",
                "--threshold",
                "0.7",
                "--save",
                str(annotated),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "assault" in result.output
        assert "命中" in result.output
        assert "★1" in result.output
        assert annotated.exists()

    def test_hero_filter_unknown(self, tmp_path: Path) -> None:
        """--hero 指定不存在的英雄时提示未找到。"""
        root = tmp_path / "templates"
        pack_dir = _build_pack_root(root, (200, 200))

        pattern = _make_pattern()
        frame = _make_frame()
        _place(frame, 80, 80, pattern)

        shot = tmp_path / "shot.png"
        _save_png(shot, frame)
        _save_png(pack_dir / "heroes" / "assault" / "star1" / "left.png", pattern)

        annotated = tmp_path / "out.png"
        result = runner.invoke(
            app,
            [
                "recognize",
                str(shot),
                "--templates-root",
                str(root),
                "--pack",
                "200x200",
                "--category",
                "heroes",
                "--hero",
                "monkey",
                "--save",
                str(annotated),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "未找到英雄" in result.output
        assert "assault" in result.output  # 提示可用英雄


class TestRecognizeErrors:
    def test_image_not_found(self, tmp_path: Path) -> None:
        """截图文件不存在时退出码 1。"""
        result = runner.invoke(
            app,
            [
                "recognize",
                str(tmp_path / "nope.png"),
                "--templates-root",
                str(tmp_path / "templates"),
            ],
        )
        assert result.exit_code == 1
        assert "不存在" in result.output

    def test_no_template_pack(self, tmp_path: Path) -> None:
        """无对应模板包时退出码 1，并提示可用模板包。"""
        root = tmp_path / "templates"
        _build_pack_root(root, (200, 200))  # 只有 200x200

        # 截图尺寸 100x100，无对应模板包
        frame = _make_frame(width=100, height=100)
        shot = tmp_path / "shot.png"
        _save_png(shot, frame)

        result = runner.invoke(
            app,
            [
                "recognize",
                str(shot),
                "--templates-root",
                str(root),
            ],
        )
        assert result.exit_code == 1
        assert "找不到" in result.output
        assert "200x200" in result.output  # 提示可用

    def test_bad_pack_format(self, tmp_path: Path) -> None:
        """--pack 格式错误时退出码 1。"""
        frame = _make_frame()
        shot = tmp_path / "shot.png"
        _save_png(shot, frame)

        result = runner.invoke(
            app,
            [
                "recognize",
                str(shot),
                "--templates-root",
                str(tmp_path / "templates"),
                "--pack",
                "abc",
            ],
        )
        assert result.exit_code == 1
        assert "格式错误" in result.output


class TestRecognizeAutoPack:
    def test_auto_match_by_image_size(self, tmp_path: Path) -> None:
        """不指定 --pack 时按截图尺寸自动匹配模板包。"""
        root = tmp_path / "templates"
        pack_dir = _build_pack_root(root, (200, 200))

        pattern = _make_pattern()
        frame = _make_frame()  # 200x200，与模板包一致
        _place(frame, 50, 50, pattern)

        shot = tmp_path / "shot.png"
        _save_png(shot, frame)
        _save_png(pack_dir / "buttons" / "btn.png", pattern)

        annotated = tmp_path / "out.png"
        result = runner.invoke(
            app,
            [
                "recognize",
                str(shot),
                "--templates-root",
                str(root),
                "--save",
                str(annotated),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "200x200" in result.output
        assert "✓" in result.output
