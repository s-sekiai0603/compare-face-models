"""切り出し画像の有効/無効判定（別人の骨格が含まれていないか）。"""
from __future__ import annotations

import numpy as np

KPT_CONF_MIN = 0.3        # この信頼度以上の骨格点だけを「見えている」とみなす
FOREIGN_MIN_POINTS = 3    # 他人物の骨格点がクロップ内にこの点数以上あれば別人の骨格を含むと判定
SAME_BOX_TOL = 0.5        # 対象人物自身の検出boxとみなす座標の許容差(px)


def foreign_skeleton_reason(own_box, all_boxes: np.ndarray | None, all_kpts: np.ndarray | None,
                            crop_xyxy: tuple[int, int, int, int]) -> str | None:
    """切り出し範囲に別人の骨格点が FOREIGN_MIN_POINTS 点以上含まれるなら無効理由を返す。

    all_boxes/all_kpts は同一フレームの全検出（対象人物自身を含む）。
    対象人物自身の検出は own_box との一致で除外する。
    """
    if all_boxes is None or all_kpts is None or len(all_boxes) == 0:
        return None
    x1, y1, x2, y2 = crop_xyxy
    own = np.asarray(own_box, dtype=np.float32)
    worst = 0
    for box, kpt in zip(all_boxes, all_kpts):
        if np.allclose(box, own, atol=SAME_BOX_TOL):
            continue  # 対象人物自身
        inside = ((kpt[:, 2] >= KPT_CONF_MIN) & (kpt[:, 0] >= x1) & (kpt[:, 0] < x2)
                  & (kpt[:, 1] >= y1) & (kpt[:, 1] < y2))
        worst = max(worst, int(inside.sum()))
    if worst >= FOREIGN_MIN_POINTS:
        return f"別人の骨格を含む（他の人物の骨格点 {worst} 点が切り出し範囲内）"
    return None
