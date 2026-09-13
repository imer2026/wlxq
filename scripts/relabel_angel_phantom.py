"""幻天使重分拣工具（2026-09-13，spec §4.1，双信号阈值经全量 2176 张校准）。

特征（饱和像素 = s>60 且 v>60，少于 30 像素视为无效）：
  hue_circ    环向 hue 均值（OpenCV 0~179）；正常天使参考≈24.3、幻天使参考≈141.9
  purple_frac hue∈[120,180) 占比；两参考分别 ≈0.003 / ≈0.391

判定（对 angel/star1~3；star4 不动）：
  hue_circ∈[105,180) 且 purple_frac>=0.15  移入 angel_phantom/starN
  hue_circ<80                               留在 angel
  其余（80~104 中间带/紫占比不足）         留在 angel 并列入人工复核清单

用法（仓库根执行）：
  python scripts/relabel_angel_phantom.py dry-run    # 只打印分布
  python scripts/relabel_angel_phantom.py apply      # 执行迁移并写审计 CSV
  python scripts/relabel_angel_phantom.py montage    # 生成人工复核拼图
"""

from __future__ import annotations

import csv
import math
import shutil
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("datasets/hero_classifier/assault/3000x2000/helper")
OUT_DIR = Path("outputs/hero_classifier") / f"relabel_review_{date.today():%Y-%m-%d}"
CIRC_MOVE_MIN = 105.0   # hue_circ >= 该值且 purple_frac 达标 → 迁移
CIRC_KEEP_MAX = 80.0    # hue_circ < 该值 → 留守
PF_MOVE_MIN = 0.15      # 迁移所需的 purple_frac 下限
REFS = {
    "正常天使参考(5B)": ROOT / "train/labeled/angel/star1/202608122001_frame000018_5B.png",
    "幻天使参考(6B)": ROOT / "train/labeled/angel/star1/202608122001_frame000108_6B.png",
}


def features(path: Path) -> tuple[float, float] | None:
    """返回 (hue_circ, purple_frac)；无效样本返回 None。"""
    img = cv2.imread(str(path))
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 1] > 60) & (hsv[..., 2] > 60)
    if int(mask.sum()) < 30:
        return None
    hue = hsv[..., 0][mask].astype(np.float32)
    ang = np.deg2rad(hue * 2.0)
    circ = (math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360.0) / 2.0
    pf = float(((hue >= 120) & (hue < 180)).sum() / mask.sum())
    return float(circ), pf


def collect() -> list[dict]:
    rows: list[dict] = []
    for split in ("train", "validation", "test"):
        base = ROOT / split / "labeled/angel"
        if not base.is_dir():
            continue
        for star_dir in sorted(base.glob("star[1-3]")):
            for path in sorted(star_dir.glob("*.png")):
                feat = features(path)
                rows.append(
                    {
                        "split": split,
                        "star": star_dir.name,
                        "path": path,
                        "circ": None if feat is None else feat[0],
                        "frac": None if feat is None else feat[1],
                    }
                )
    return rows


def decide(circ: float | None, frac: float | None) -> str:
    if circ is None or frac is None:
        return "skip"
    if CIRC_MOVE_MIN <= circ < 180 and frac >= PF_MOVE_MIN:
        return "move"
    if circ < CIRC_KEEP_MAX:
        return "keep"
    return "review"


def cmd_dry_run() -> None:
    for label, ref in REFS.items():
        feat = features(ref)
        print(f"{label}: hue_circ={feat[0]:.1f} purple_frac={feat[1]:.4f}")
    rows = collect()
    print(f"共扫描 {len(rows)} 张（train/validation/test 的 angel/star1~3）")
    for split in ("train", "validation", "test"):
        sub = [r for r in rows if r["split"] == split]
        if not sub:
            continue
        for star in ("star1", "star2", "star3"):
            grp = [r for r in sub if r["star"] == star]
            if not grp:
                continue
            buckets = {"move": 0, "keep": 0, "review": 0, "skip": 0}
            for r in grp:
                buckets[decide(r["circ"], r["frac"])] += 1
            print(
                f"{split}/{star}: 共{len(grp)} 移{buckets['move']} 留{buckets['keep']} "
                f"待复核{buckets['review']} 无效{buckets['skip']}"
            )


def cmd_apply() -> None:
    rows = collect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "relabel_angel_phantom.csv"
    moved = kept = review = 0
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "hue_circ", "purple_frac", "decision", "new_path"])
        for r in rows:
            action = decide(r["circ"], r["frac"])
            circ_text = "" if r["circ"] is None else f"{r['circ']:.1f}"
            frac_text = "" if r["frac"] is None else f"{r['frac']:.4f}"
            new_path = ""
            if action == "move":
                target = (
                    ROOT
                    / r["split"]
                    / "labeled/angel_phantom"
                    / r["star"]
                    / r["path"].name
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(r["path"]), str(target))
                new_path = str(target)
                moved += 1
            elif action == "review":
                review += 1
            else:
                kept += 1
            writer.writerow([str(r["path"]), circ_text, frac_text, action, new_path])
    print(f"迁移 {moved} 张到 angel_phantom；留在 angel {kept} 张；待复核 {review} 张")
    print(f"审计 CSV: {csv_path}")


def _tile(items: list[tuple[str, Path]], cols: int = 8, scale: int = 3) -> np.ndarray:
    cell_w, cell_h = 64 * scale, 68 * scale
    label_h = 22
    rows_n = math.ceil(len(items) / cols)
    sheet = np.full((rows_n * (cell_h + label_h), cols * cell_w, 3), 255, np.uint8)
    for idx, (name, path) in enumerate(items):
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
        r, c = divmod(idx, cols)
        y0, x0 = r * (cell_h + label_h), c * cell_w
        sheet[y0 : y0 + cell_h, x0 : x0 + cell_w] = img
        cv2.putText(
            sheet,
            name[:32],
            (x0 + 2, y0 + cell_h + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return sheet


def _spread(items: list[tuple[str, Path]], limit: int) -> list[tuple[str, Path]]:
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def cmd_montage() -> None:
    out = OUT_DIR / "montage"
    out.mkdir(parents=True, exist_ok=True)
    rows = collect()
    # 迁移全景：angel_phantom 各星级全部
    for split in ("train", "validation", "test"):
        base = ROOT / split / "labeled/angel_phantom"
        if not base.is_dir():
            continue
        for star_dir in sorted(base.glob("star[1-3]")):
            pngs = sorted(star_dir.glob("*.png"))
            if not pngs:
                continue
            sheet = _tile([(p.name, p) for p in pngs])
            cv2.imwrite(str(out / f"moved_{split}_{star_dir.name}.png"), sheet)
            print(f"moved_{split}_{star_dir.name}.png: {len(pngs)} 张")
    # 留守抽样（查漏迁）：各 split/star 留下的图按序抽 48 张
    for split in ("train", "validation", "test"):
        sub = [r for r in rows if r["split"] == split and decide(r["circ"], r["frac"]) == "keep"]
        for star in ("star1", "star2", "star3"):
            grp = [(r["path"].name, r["path"]) for r in sub if r["star"] == star]
            picked = _spread(grp, 48)
            if not picked:
                continue
            sheet = _tile(picked)
            cv2.imwrite(str(out / f"kept_{split}_{star}.png"), sheet)
            print(f"kept_{split}_{star}.png: 抽样 {len(picked)} 张")
    # 中间带复核清单
    review_items = [
        (r["path"].name, r["path"]) for r in rows if decide(r["circ"], r["frac"]) == "review"
    ]
    for i in range(0, len(review_items), 96):
        sheet = _tile(review_items[i : i + 96])
        cv2.imwrite(str(out / f"review_{i // 96:02d}.png"), sheet)
    print(f"review_*.png: 中间带共 {len(review_items)} 张")
    print(f"复核包目录: {OUT_DIR}")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"dry-run", "apply", "montage"}:
        print(__doc__)
        sys.exit(2)
    {"dry-run": cmd_dry_run, "apply": cmd_apply, "montage": cmd_montage}[sys.argv[1]]()


if __name__ == "__main__":
    main()
