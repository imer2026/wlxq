"""阵容取值与模型覆盖的守卫测试（2026-09-29 两次启动炸裂的回归锁）。

背景：lineup_others 从全局单值改为「档案覆盖 + 全局回退」后，runner 里任何
绕过 `_lineup_others_for` 直读全局的消费点、或任何「模型不含本阵容英雄类别」
的组合，都会在实机启动时以
`RuntimeError: 英雄格模型缺少本局阵容类别: …`
炸掉。这里把这两类风险分别锁死。
"""

from pathlib import Path

from wlxq_bot.config import load_default_config
from wlxq_bot.perception.hero_classifier import HeroCellClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_runner_reads_global_lineup_only_inside_helper() -> None:
    """runner 直读全局 lineup_others 的位置必须只有 _lineup_others_for 内部一处。

    2026-09-28/29 两次实机炸裂的根因都是新增了绕过辅助方法的消费点（run 路径
    校验、run 路径 allowed_heroes 先后漏改）。源码扫描直接锁死这个模式。
    """
    import wlxq_bot.runner as runner_module

    source = (Path(runner_module.__file__).read_text(encoding="utf-8")).splitlines()
    direct_reads = [
        line.strip()
        for line in source
        # 代码读取必经 self.default_config 前缀；docstring/注释里的提法不算
        if "self.default_config.hero_classifier.lineup_others" in line
    ]
    assert len(direct_reads) == 1, (
        "runner 出现了绕过 _lineup_others_for 的全局阵容读取，"
        "会让阵容分叉的主C拿到错误白名单: " + "; ".join(direct_reads)
    )
    # 同样的守卫覆盖 lineup_variants（2026-11-15 档案级变体：猴子 loki_phantom、
    # 强袭 angel_phantom，全局并集会让任一主C的启动类别校验互相炸）
    variant_reads = [
        line.strip()
        for line in source
        if "self.default_config.hero_classifier.lineup_variants" in line
    ]
    assert len(variant_reads) == 1, (
        "runner 出现了绕过 _lineup_variants_for 的全局变体读取，"
        "会让阵容分叉的主C拿到错误白名单: " + "; ".join(variant_reads)
    )
    assert "return profile_variants or" in variant_reads[0]
    assert "return profile_lineup or" in direct_reads[0]


def test_every_configured_model_covers_its_lineup() -> None:
    """每个配了真实模型的主C：阵容+变体类别必须全部被该模型支持。

    在配置装载层提前暴露「模型与阵容不匹配」，而不是等实机 run 启动校验炸裂。
    没有模型文件的档案（fox-helper 等占位）跳过——它们本就不可作主C启动。
    """
    cfg = load_default_config(Path("configs/default.yaml"))
    checked: list[str] = []
    for mc, profile in cfg.main_c_profiles.items():
        model_path = Path(profile.hero_classifier_model)
        if not model_path.is_file():
            continue
        classifier = HeroCellClassifier(model_path)
        lineup = profile.lineup_others or cfg.hero_classifier.lineup_others
        variants = profile.lineup_variants or cfg.hero_classifier.lineup_variants
        allowed = {mc, *lineup, *variants}
        missing = sorted(allowed - set(classifier.supported_heroes))
        assert not missing, f"主C {mc} 的模型 {model_path} 缺少阵容类别: {missing}"
        checked.append(mc)
    # 仓库当前至少应校验强袭(v6)与猴子(v5)两个真实模型
    assert set(checked) >= {"assault", "monkey"}
