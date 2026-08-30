"""离线验证 OpenCV 多峰值检测。

目标：用 screenshots/raw 里招募大厅截图里的多个强袭头像，
验证 OpenCV 能找到多个相同模板（机制验证，非棋盘英雄验证）。

不在真实窗口上操作，不修改游戏，纯离线跑。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "screenshots" / "raw" / "screenshot_20260807_205013.png"
OUT = PROJECT_ROOT / "outputs" / "verify"
OUT.mkdir(parents=True, exist_ok=True)


def nms(
    cands: list[tuple[int, int, float]],
    dist: int,
) -> list[tuple[int, int, float]]:
    """简单的非极大值抑制：合并距离小于 dist 的点，保留置信度最高的。"""
    if not cands:
        return []
    sorted_cands = sorted(cands, key=lambda c: -c[2])
    keep: list[tuple[int, int, float]] = []
    taken = [False] * len(sorted_cands)
    for i, (x, y, c) in enumerate(sorted_cands):
        if taken[i]:
            continue
        keep.append((x, y, c))
        for j in range(i + 1, len(sorted_cands)):
            if taken[j]:
                continue
            dx = sorted_cands[j][0] - x
            dy = sorted_cands[j][1] - y
            if (dx * dx + dy * dy) ** 0.5 < dist:
                taken[j] = True
    return keep


def main() -> None:
    img = cv2.imread(str(RAW))
    if img is None:
        raise SystemExit(f"无法读取 {RAW}")
    H, W = img.shape[:2]
    print(f"图像: W={W} H={H}")

    # 相对坐标：从中部招募条"美味鱼头"那条的第 2 个强袭头像裁模板
    # 目测：x≈225/927=0.243, y≈590/1727=0.342，模板约 80x80
    cx_r, cy_r = 0.243, 0.342
    tw_r, th_r = 0.085, 0.05
    cx, cy = int(cx_r * W), int(cy_r * H)
    tw, th = int(tw_r * W), int(th_r * H)
    x1 = max(0, cx - tw // 2)
    y1 = max(0, cy - th // 2)
    x2 = x1 + tw
    y2 = y1 + th
    template = img[y1:y2, x1:x2].copy()
    print(f"模板: ({x1},{y1})~({x2},{y2}) 尺寸 {template.shape}")
    cv2.imwrite(str(OUT / "template.png"), template)

    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    print(f"结果矩阵: {result.shape}  min={result.min():.3f} max={result.max():.3f}")

    # 阈值化
    threshold = 0.7
    ys, xs = np.where(result >= threshold)
    candidates = [(int(x), int(y), float(result[y, x])) for y, x in zip(ys, xs)]
    print(f"\n阈值 {threshold} 候选点数: {len(candidates)}")

    # NMS：头像间距约 0.16*W，NMS 距离取 0.04*W（远小于间距）
    nms_dist = max(20, int(0.04 * W))
    filtered = nms(candidates, nms_dist)
    print(f"NMS(dist={nms_dist}) 后独立峰值: {len(filtered)} 个")
    print("图里强袭出现位置: 中部招募条第2、下方招募条第1、第一张招募条第4\n")
    for i, (x, y, c) in enumerate(filtered):
        cx_p = x + template.shape[1] // 2
        cy_p = y + template.shape[0] // 2
        print(f"  #{i + 1}: 中心=({cx_p},{cy_p}) 置信度={c:.3f}")

    annotated = img.copy()
    for x, y, c in filtered:
        cx_p = x + template.shape[1] // 2
        cy_p = y + template.shape[0] // 2
        cv2.rectangle(
            annotated, (cx_p - 40, cy_p - 40), (cx_p + 40, cy_p + 40), (0, 255, 0), 3,
        )
        cv2.putText(
            annotated, f"{c:.2f}", (cx_p - 20, cy_p - 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    cv2.imwrite(str(OUT / "annotated.png"), annotated)
    print(f"\n标注图: {OUT / 'annotated.png'}")
    print(f"模板:   {OUT / 'template.png'}")


if __name__ == "__main__":
    main()