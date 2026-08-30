"""验证：用用户裁好的1星强袭模板，在整张游戏截图里找多个强袭。

输入：
  - 模板: assets/templates/3000x2000/heroes/assault/强袭.png
  - 目标: assets/templates/3000x2000/heroes/assault/Snipaste_2026-08-08_16-16-17.png

输出：outputs/verify/assault_annotated.png（标注图）+ 控制台报告
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "assets" / "templates" / "3000x2000" / "heroes" / "assault" / "强袭.png"
TARGET = ROOT / "assets" / "templates" / "3000x2000" / "heroes" / "assault" / "Snipaste_2026-08-08_16-16-17.png"
OUT = ROOT / "outputs" / "verify"
OUT.mkdir(parents=True, exist_ok=True)


def imread_unicode(path: Path) -> np.ndarray | None:
    """cv2.imread 在 Windows 上读不了中文路径，用 imdecode + fromfile 绕过。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def nms(
    cands: list[tuple[int, int, float]],
    dist: int,
) -> list[tuple[int, int, float]]:
    """非极大值抑制：合并距离小于 dist 的点，保留置信度最高的。"""
    if not cands:
        return []
    s = sorted(cands, key=lambda c: -c[2])
    keep: list[tuple[int, int, float]] = []
    taken = [False] * len(s)
    for i, (x, y, c) in enumerate(s):
        if taken[i]:
            continue
        keep.append((x, y, c))
        for j in range(i + 1, len(s)):
            if taken[j]:
                continue
            dx = s[j][0] - x
            dy = s[j][1] - y
            if (dx * dx + dy * dy) ** 0.5 < dist:
                taken[j] = True
    return keep


def main() -> None:
    template = imread_unicode(TEMPLATE)
    target = imread_unicode(TARGET)
    if template is None:
        raise SystemExit(f"无法读取模板: {TEMPLATE}")
    if target is None:
        raise SystemExit(f"无法读取目标: {TARGET}")

    th, tw = template.shape[:2]
    H, W = target.shape[:2]
    print(f"模板: {tw}x{th}")
    print(f"目标: {W}x{H}")
    if th > H or tw > W:
        raise SystemExit("模板比目标图大，无法 matchTemplate")

    result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
    print(f"结果矩阵: {result.shape}  min={result.min():.3f}  max={result.max():.3f}")
    print()
    print("各阈值下候选点数（看置信度分布，选合适阈值）：")
    for thr in [0.6, 0.7, 0.8, 0.9, 0.95]:
        ys, xs = np.where(result >= thr)
        print(f"  阈值 {thr}: {len(xs)} 个候选点")

    # 多尺度诊断：模板可能和目标分辨率不匹配，放大模板试
    print()
    print("多尺度诊断（模板原始尺寸可能和目标里的强袭不一致）：")
    best_scale = 1.0
    best_max = result.max()
    for scale in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        new_w = int(tw * scale)
        new_h = int(th * scale)
        if new_w > W or new_h > H:
            print(f"  scale {scale}: {new_w}x{new_h} 超过目标图，跳过")
            continue
        scaled = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
        r = cv2.matchTemplate(target, scaled, cv2.TM_CCOEFF_NORMED)
        mark = "  <-- 当前最佳" if r.max() > best_max else ""
        print(f"  scale {scale}: 模板 {new_w}x{new_h}  max={r.max():.3f}{mark}")
        if r.max() > best_max:
            best_max = r.max()
            best_scale = scale
    print(f"\n最佳缩放: {best_scale}x (模板->{int(tw*best_scale)}x{int(th*best_scale)}) 最高置信度={best_max:.3f}")
    if best_scale != 1.0:
        print("→ 模板原始尺寸和目标里强袭的实际尺寸不匹配，需要按目标分辨率重新裁模板")

    # 用 0.7 阈值 + NMS
    threshold = 0.7
    ys, xs = np.where(result >= threshold)
    candidates = [(int(x), int(y), float(result[y, x])) for y, x in zip(ys, xs)]
    # NMS 距离：模板宽度的一半（同英雄不会重叠到小于半身）
    nms_dist = max(20, tw // 2)
    filtered = nms(candidates, nms_dist)
    print(f"\nNMS(dist={nms_dist}) 后独立峰值: {len(filtered)} 个")
    print("（你说图里有好几个1星强袭，看找到的数量对不对）\n")
    for i, (x, y, c) in enumerate(filtered):
        cx = x + tw // 2
        cy = y + th // 2
        print(f"  #{i + 1}: 匹配左上=({x},{y}) 中心=({cx},{cy}) 置信度={c:.3f}")

    # 标注图
    annotated = target.copy()
    for x, y, c in filtered:
        cv2.rectangle(annotated, (x, y), (x + tw, y + th), (0, 255, 0), 3)
        cv2.putText(
            annotated, f"{c:.2f}", (x, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
    cv2.imwrite(str(OUT / "assault_annotated.png"), annotated)
    cv2.imwrite(str(OUT / "assault_template.png"), template)
    print(f"\n标注图: {OUT / 'assault_annotated.png'}")
    print(f"模板副本: {OUT / 'assault_template.png'}")

    # 颜色诊断：在目标图里找紫色强袭区域，输出位置和尺寸
    # 帮你定位强袭实际在哪、多大，方便重新裁模板
    print("\n" + "=" * 60)
    print("颜色诊断：目标图里的紫色区域（强袭机甲主色）")
    print("=" * 60)
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    # 紫色 HSV 范围（强袭机甲紫色调）
    mask = cv2.inRange(hsv, np.array([125, 40, 40]), np.array([165, 255, 255]))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    purple_regions = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 500:  # 过滤小噪点，只留大块紫色
            purple_regions.append((x, y, w, h, area))
    purple_regions.sort(key=lambda r: -r[4])  # 按面积降序
    print(f"找到 {len(purple_regions)} 个紫色区域（面积>500）：\n")
    for i, (x, y, w, h, area) in enumerate(purple_regions[:15]):
        print(f"  #{i + 1}: 位置=({x},{y}) 尺寸={w}x{h} 面积={area}")
    print(f"\n→ 这些就是强袭在目标图里的实际位置和尺寸")
    print(f"→ 你现在的模板是 {tw}x{th}，对比上面紫色区域的尺寸，看差多少")
    print(f"→ 建议：直接从目标图里某个紫色区域裁完整英雄主体做模板")

    # 自动验证：用最大紫色区域当模板，证明 OpenCV 多峰值机制本身能用
    if purple_regions and result.max() < 0.5:
        print("\n" + "=" * 60)
        print("自动验证：用最大紫色区域当模板重新匹配")
        print("=" * 60)
        px, py, pw, ph, _ = purple_regions[0]
        auto_template = target[py:py + ph, px:px + pw].copy()
        print(f"自动模板: 从 ({px},{py}) 裁 {pw}x{ph}")
        cv2.imwrite(str(OUT / "auto_template.png"), auto_template)

        auto_result = cv2.matchTemplate(target, auto_template, cv2.TM_CCOEFF_NORMED)
        print(f"max={auto_result.max():.3f}")
        print()
        for thr in [0.6, 0.7, 0.8, 0.9]:
            ys, xs = np.where(auto_result >= thr)
            print(f"  阈值 {thr}: {len(xs)} 候选点")

        threshold = 0.7
        ys, xs = np.where(auto_result >= threshold)
        candidates = [(int(x), int(y), float(auto_result[y, x])) for y, x in zip(ys, xs)]
        nms_dist = max(20, pw // 2)
        filtered = nms(candidates, nms_dist)
        print(f"\nNMS(dist={nms_dist}) 后独立峰值: {len(filtered)} 个")
        for i, (x, y, c) in enumerate(filtered):
            print(f"  #{i + 1}: ({x},{y})~({x + pw},{y + ph}) 置信度={c:.3f}")

        auto_annotated = target.copy()
        for x, y, c in filtered:
            cv2.rectangle(auto_annotated, (x, y), (x + pw, y + ph), (0, 255, 0), 3)
            cv2.putText(
                auto_annotated, f"{c:.2f}", (x, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
        cv2.imwrite(str(OUT / "auto_annotated.png"), auto_annotated)
        print(f"\n自动标注图: {OUT / 'auto_annotated.png'}")
        print(f"自动模板:   {OUT / 'auto_template.png'}")

    # 定位诊断：模板对的，但匹配低——看"最像模板"的位置在哪
    print("\n" + "=" * 60)
    print("定位诊断：用你的模板找目标图里'最像'的位置")
    print("=" * 60)
    print(f"全局最高置信度: {result.max():.3f}")
    print("\nTop 10 最像模板的位置：")
    flat = result.flatten()
    top_n = 10
    top_indices = np.argpartition(flat, -top_n)[-top_n:]
    top_list = []
    for idx in top_indices:
        y, x = divmod(int(idx), result.shape[1])
        top_list.append((x, y, float(result[y, x])))
    top_list.sort(key=lambda t: -t[2])
    for i, (x, y, c) in enumerate(top_list):
        near = ""
        for j, (px, py, pw, ph, _) in enumerate(purple_regions):
            if px - tw <= x <= px + pw and py - th <= y <= py + ph:
                near = f" [在紫色区域#{j + 1} ({px},{py}) {pw}x{ph} 内]"
        print(f"  #{i + 1}: 匹配左上=({x},{y}) 中心=({x + tw // 2},{y + th // 2}) conf={c:.3f}{near}")
    print(f"\n→ 看这些位置是不是强袭所在的位置")
    print(f"→ 如果是强袭位置但 conf 仍低（如 0.2-0.4），说明朝向/动作/背景和模板不同")
    print(f"→ 这就验证了 architecture.md 说的：同一英雄要采朝左/朝右/动作多张模板")


if __name__ == "__main__":
    main()