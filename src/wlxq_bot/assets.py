"""模板包加载。

模板目录结构见 docs/architecture.md「模板目录」一节：
assets/templates/<width>x<height>/{buttons,skills,heroes}/

目录名是游戏窗口所在显示器的物理分辨率；Runner 允许本机配置显式覆盖，
否则严格按窗口所在显示器选择，不跨分辨率回退，也不做运行时缩放。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wlxq_bot.models import parse_hero_template_path


@dataclass(frozen=True)
class HeroTemplate:
    """一个英雄模板文件及其解析后的元数据。

    由 TemplatePack.scan_hero_templates 扫描目录生成，不需要手动构造。

    Attributes:
        path: 模板文件绝对路径
        relative_path: 相对于模板包根的路径（用作 MatchResult.template_name）
        hero_type: 英雄类型标识，从目录名反推
        star_level: 星级 1~4，从 star1~4/ 目录名反推
    """

    path: Path
    relative_path: str
    hero_type: str
    star_level: int


@dataclass(frozen=True)
class TemplatePack:
    """一组同一显示器物理分辨率下采集的模板。

    Attributes:
        client_size: 使用该档案时的实际客户区尺寸
        root: 模板包根目录
    """

    client_size: tuple[int, int]
    root: Path

    @property
    def buttons_dir(self) -> Path:
        return self.root / "buttons"

    @property
    def skills_dir(self) -> Path:
        return self.root / "skills"

    @property
    def heroes_dir(self) -> Path:
        return self.root / "heroes"

    def hero_dir(self, hero_id: str) -> Path:
        """返回某个英雄的模板目录。"""
        return self.heroes_dir / hero_id

    def resolve_template(self, relative_path: str) -> Path:
        """根据相对路径解析模板文件绝对路径。"""
        return self.root / relative_path

    def scan_hero_templates(self, hero_id: str) -> list[HeroTemplate]:
        """扫描某英雄的全部模板（所有星级所有 png）。

        扫描 heroes/<hero_id>/star*/ 目录下的所有 *.png。
        星级目录内文件名任意，只要 *.png 就加载——用户往目录丢图即可扩展
        识别范围，不用改配置。

        Args:
            hero_id: 英雄类型标识，如 "assault"

        Returns:
            该英雄全部模板列表，按星级升序、文件名排序。
            目录不存在或无模板时返回空列表。
        """
        hero_dir = self.hero_dir(hero_id)
        if not hero_dir.is_dir():
            return []
        templates: list[HeroTemplate] = []
        for star_dir in sorted(hero_dir.iterdir()):
            if not star_dir.is_dir():
                continue
            if not star_dir.name.startswith("star"):
                continue
            for png in sorted(star_dir.glob("*.png")):
                relative = png.relative_to(self.root)
                rel_str = str(relative).replace("\\", "/")
                try:
                    hero_type, star_level = parse_hero_template_path(rel_str)
                except ValueError:
                    continue
                templates.append(
                    HeroTemplate(
                        path=png,
                        relative_path=rel_str,
                        hero_type=hero_type,
                        star_level=star_level,
                    )
                )
        return templates


def find_template_pack(
    templates_root: Path,
    client_width: int,
    client_height: int,
) -> TemplatePack | None:
    """根据客户区尺寸查找模板包。

    Args:
        templates_root: assets/templates 目录
        client_width: 客户区宽度
        client_height: 客户区高度

    Returns:
        匹配的 TemplatePack，未找到返回 None
    """
    pack_dir = templates_root / f"{client_width}x{client_height}"
    if pack_dir.is_dir():
        return TemplatePack(
            client_size=(client_width, client_height),
            root=pack_dir,
        )
    return None
