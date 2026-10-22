"""切り出し画像の有効/無効判定。

1. 別人の骨格を含む: 切り出し範囲に他の人物の骨格点が FOREIGN_MIN_POINTS 点以上入る（reid 版から引き継ぎ）。
2. 顔枠に本人の鼻点が無い: 顔検出枠が本人の顔に付いていない画像（隣の人物の顔、後頭部や衣服の誤検出）を検出する。
   本人の鼻のキーポイントが顔枠から顔幅の FACE_BOX_TOL 倍より離れていれば無効。鼻が見えていない（信頼度 < NOSE_CONF_MIN）
   画像は判定しない。PoC7 の検証で、人が「別人の顔」と判断した 2 枚は枠外距離が顔幅の 1.0 倍と 7.8 倍、本人と確認した
   21 枚は全て枠内だった。許容を 0.1 倍にすると小さい顔（幅 10〜15px）のキーポイント誤差（2〜4px）まで拾って 3,000 枚
   近くが無効になったため、0.5 倍にしている。
抽出時に両方を適用する。既存の抽出結果には revalidate_json（CLI の revalidate）で 2 を後から適用できる。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KPT_CONF_MIN = 0.3        # この信頼度以上の骨格点だけを「見えている」とみなす
FOREIGN_MIN_POINTS = 3    # 他人物の骨格点がクロップ内にこの点数以上あれば別人の骨格を含むと判定
SAME_BOX_TOL = 0.5        # 対象人物自身の検出boxとみなす座標の許容差(px)
NOSE_IDX = 0              # COCO 17 点の鼻
NOSE_CONF_MIN = 0.15      # 顔ゲート（face_crop.KP_CONF_TH）と同じ。これ未満なら鼻が見えていないとして判定しない
FACE_BOX_TOL = 0.50       # 顔枠の幅に対する許容割合（これより離れていれば顔枠は本人の顔ではないとみなす）
NOSE_RULE_PREFIX = "顔枠に本人の鼻点が無い"


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


def _kpts_array(kpts) -> np.ndarray | None:
    """(17, 3) 配列、または JSON の [{x, y, conf}, ...] を (N, 3) 配列に揃える。"""
    if kpts is None:
        return None
    if isinstance(kpts, (list, tuple)) and kpts and isinstance(kpts[0], dict):
        return np.array([[k.get("x", 0), k.get("y", 0), k.get("conf", 0)] for k in kpts], np.float32)
    arr = np.asarray(kpts, np.float32)
    return arr if arr.ndim == 2 and arr.shape[1] >= 3 else None


def face_box_mismatch_reason(kpts, face_bbox_xyxy) -> str | None:
    """本人の鼻のキーポイントが顔枠に入っていなければ無効理由を返す（隣の人物の顔を切り出した疑い）。"""
    arr = _kpts_array(kpts)
    if arr is None or len(arr) <= NOSE_IDX or face_bbox_xyxy is None:
        return None
    x, y, c = (float(v) for v in arr[NOSE_IDX][:3])
    if c < NOSE_CONF_MIN:
        return None
    x1, y1, x2, y2 = (float(v) for v in face_bbox_xyxy)
    tol = max(0.0, (x2 - x1) * FACE_BOX_TOL)
    if x1 - tol <= x <= x2 + tol and y1 - tol <= y <= y2 + tol:
        return None
    dist = max(x1 - x, x - x2, y1 - y, y - y2, 0.0)
    return (f"{NOSE_RULE_PREFIX}（鼻 ({x:.0f},{y:.0f}) が顔枠 [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] から {dist:.0f}px 離れている。"
            "別人の顔、または後頭部などの誤検出の可能性）")


def revalidate_json(path: Path) -> tuple[int, int]:
    """追跡ID の JSON に鼻点ルールを後から適用する。他の理由で無効の画像（別人の骨格、手動除外）は触らない。
    以前の鼻点ルールで無効になった画像は判定し直す（しきい値を変えたときに戻せる）。
    戻り値: (判定した画像数, 新たに無効にした画像数 - 解除した画像数)。変更があればファイルを書き戻す。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    checked = changed = 0
    dirty = False
    for im in data.get("images", []):
        was_nose_rule = str(im.get("invalid_reason", "")).startswith(NOSE_RULE_PREFIX)
        if im.get("invalid") and not was_nose_rule:
            continue
        checked += 1
        reason = face_box_mismatch_reason(im.get("keypoints"), im.get("face_bbox_xyxy"))
        if reason and not im.get("invalid"):
            changed += 1
        elif not reason and was_nose_rule:
            changed -= 1
        if reason != (im.get("invalid_reason") or None) or bool(reason) != bool(im.get("invalid")):
            im["invalid"] = reason is not None
            im["invalid_reason"] = reason or ""
            dirty = True
    if dirty:
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return checked, changed


def revalidate_tree(root: Path) -> tuple[int, int, int]:
    """root 配下の全追跡ID JSON（<追跡ID>/<追跡ID>.json）に revalidate_json を適用する。
    戻り値: (JSON 数, 判定した画像数, 新たに無効にした画像数)。"""
    n_json = checked = newly = 0
    for p in sorted(Path(root).rglob("*.json")):
        if p.parent.name != p.stem:
            continue
        n_json += 1
        c, n = revalidate_json(p)
        checked += c
        newly += n
    return n_json, checked, newly
