"""TemplatePack.scan_hero_templates 单元测试。

验证目录扫描：heroes/<hero>/star*/ 下的 *.png 自动发现并解析元数据。
"""

from __future__ import annotations

from pathlib import Path

from wlxq_bot.assets import HeroTemplate, TemplatePack


def _make_pack(tmp_path: Path) -> TemplatePack:
    """在 tmp_path 下建一个模拟模板包。"""
    root = tmp_path / "templates" / "927x1727"
    root.mkdir(parents=True)
    return TemplatePack(client_size=(927, 1727), root=root)


def _touch(root: Path, rel: str) -> None:
    """在模板包内创建空 png 文件。"""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()


class TestScanHeroTemplates:
    def test_scan_assault_multiple_stars(self, tmp_path: Path) -> None:
        """扫描 assault 的 star1（2张）+ star2（1张）= 3 个模板。"""
        pack = _make_pack(tmp_path)
        root = pack.root
        _touch(root, "heroes/assault/star1/a.png")
        _touch(root, "heroes/assault/star1/b.png")
        _touch(root, "heroes/assault/star2/c.png")

        templates = pack.scan_hero_templates("assault")

        assert len(templates) == 3
        # 按星级升序、文件名排序
        assert templates[0].hero_type == "assault"
        assert templates[0].star_level == 1
        assert templates[0].relative_path == "heroes/assault/star1/a.png"
        assert templates[1].star_level == 1
        assert templates[1].relative_path == "heroes/assault/star1/b.png"
        assert templates[2].star_level == 2
        assert templates[2].relative_path == "heroes/assault/star2/c.png"

    def test_scan_monkey(self, tmp_path: Path) -> None:
        pack = _make_pack(tmp_path)
        _touch(pack.root, "heroes/monkey/star3/x.png")

        templates = pack.scan_hero_templates("monkey")

        assert len(templates) == 1
        assert templates[0].hero_type == "monkey"
        assert templates[0].star_level == 3

    def test_nonexistent_hero_returns_empty(self, tmp_path: Path) -> None:
        pack = _make_pack(tmp_path)
        templates = pack.scan_hero_templates("nonexistent")
        assert templates == []

    def test_empty_hero_dir_returns_empty(self, tmp_path: Path) -> None:
        """英雄目录存在但无星级子目录。"""
        pack = _make_pack(tmp_path)
        (pack.root / "heroes" / "assault").mkdir(parents=True)
        templates = pack.scan_hero_templates("assault")
        assert templates == []

    def test_skips_non_star_dirs(self, tmp_path: Path) -> None:
        """不以 star 开头的目录被跳过。"""
        pack = _make_pack(tmp_path)
        _touch(pack.root, "heroes/assault/star1/ok.png")
        _touch(pack.root, "heroes/assault/notes.txt_dir/ignore.png")

        templates = pack.scan_hero_templates("assault")
        assert len(templates) == 1
        assert templates[0].star_level == 1

    def test_skips_non_png_files(self, tmp_path: Path) -> None:
        """非 .png 文件被跳过。"""
        pack = _make_pack(tmp_path)
        _touch(pack.root, "heroes/assault/star1/ok.png")
        _touch(pack.root, "heroes/assault/star1/ignore.txt")
        _touch(pack.root, "heroes/assault/star1/ignore.jpg")

        templates = pack.scan_hero_templates("assault")
        assert len(templates) == 1

    def test_path_is_absolute(self, tmp_path: Path) -> None:
        """返回的 path 是绝对路径。"""
        pack = _make_pack(tmp_path)
        _touch(pack.root, "heroes/assault/star1/a.png")

        templates = pack.scan_hero_templates("assault")
        assert templates[0].path.is_absolute()
        assert templates[0].path.exists()

    def test_all_four_stars(self, tmp_path: Path) -> None:
        """4 个星级都有模板。"""
        pack = _make_pack(tmp_path)
        for star in range(1, 5):
            _touch(pack.root, f"heroes/assault/star{star}/t.png")

        templates = pack.scan_hero_templates("assault")
        assert len(templates) == 4
        assert [t.star_level for t in templates] == [1, 2, 3, 4]

    def test_returns_hero_template_type(self, tmp_path: Path) -> None:
        """返回的是 HeroTemplate 实例。"""
        pack = _make_pack(tmp_path)
        _touch(pack.root, "heroes/assault/star1/a.png")

        templates = pack.scan_hero_templates("assault")
        assert isinstance(templates[0], HeroTemplate)

    def test_arbitrary_filename(self, tmp_path: Path) -> None:
        """文件名任意（中文、数字、Snipaste 默认名等）都能扫描。"""
        pack = _make_pack(tmp_path)
        _touch(pack.root, "heroes/assault/star1/强袭.png")
        _touch(pack.root, "heroes/assault/star1/Snipaste_2026-08-08_16-16-17.png")
        _touch(pack.root, "heroes/assault/star1/2.png")

        templates = pack.scan_hero_templates("assault")
        assert len(templates) == 3
