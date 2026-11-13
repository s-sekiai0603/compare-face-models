"""カメラからの距離範囲の確認（距離範囲マッピング）。

カメラ設定フォルダの <カメラID>.json（射影変換設定抽出画面の出力）からカメラ直下の地面点を求め、カメラ画像（ファイル名 = カメラID）の
うち「カメラ直下から指定距離(m)以内」の地面に写る範囲を半透明で塗って保存する。塗る対象は魚眼（歪み）補正後の画像
（設定ファイルの座標系。λ = 0 なら元画像のまま）。距離の算出は core/camera/distance.py。"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..camera.distance import CameraDistance, IMAGE_SUFFIXES
from ..common.imageio import imread, imwrite

FILL_COLOR_BGR = (0, 200, 0)   # 塗りつぶしの色
FILL_ALPHA = 0.4               # 塗りつぶしの不透明度
EDGE_COLOR_BGR = (0, 120, 0)   # 境界線の色
OUTPUT_SUFFIX = "_distance"


def render_distance_range(image_bgr: np.ndarray, distance_map: np.ndarray, max_distance_m: float) -> np.ndarray:
    """画像のうち距離 <= max_distance_m の画素を半透明で塗り、境界線と凡例を描く。distance_map は画像と同じ大きさ（NaN は塗らない）。"""
    with np.errstate(invalid="ignore"):
        mask = distance_map <= max_distance_m
    out = image_bgr.copy()
    if mask.any():
        fill = np.empty_like(out)
        fill[:] = FILL_COLOR_BGR
        blended = (out.astype(np.float32) * (1 - FILL_ALPHA) + fill.astype(np.float32) * FILL_ALPHA).astype(np.uint8)
        out[mask] = blended[mask]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, EDGE_COLOR_BGR, 2, cv2.LINE_AA)
    return _draw_legend(out, max_distance_m)


def _draw_legend(img: np.ndarray, max_distance_m: float) -> np.ndarray:
    """右上に「塗った範囲 = カメラから N m 以内」の凡例を描く（小さい画像には描かない）。"""
    h, w = img.shape[:2]
    if w < 320 or h < 160:
        return img
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.6
    label = f"within {max_distance_m:g} m from camera"
    tw = cv2.getTextSize(label, font, scale, 1)[0][0]
    box = 18
    x1 = w - w // 40
    x0, y0 = x1 - tw - box - 24, h // 40
    cv2.rectangle(img, (x0, y0), (x1, y0 + 34), (255, 255, 255), -1)
    cv2.rectangle(img, (x0 + 8, y0 + 8), (x0 + 8 + box, y0 + 8 + box), FILL_COLOR_BGR, -1)
    cv2.putText(img, label, (x0 + 8 + box + 8, y0 + 24), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def run_distance_range_map(config_dir: Path, camera_dir: Path, output_dir: Path, max_distance_m: float,
                           progress: Callable[[str], None] | None = None) -> tuple[list[Path], list[str]]:
    """カメラ画像ごとに距離範囲を塗った画像を output_dir へ保存する。戻り値は (出力ファイル, スキップ・注意の説明)。"""
    say = progress or (lambda _msg: None)
    if max_distance_m <= 0:
        raise ValueError("距離は 0 より大きい値を指定してください。")
    images = sorted(p for p in camera_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"カメラ画像が見つかりません: {camera_dir}")
    cams = CameraDistance(config_dir, camera_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    notes: list[str] = []
    for i, path in enumerate(images, 1):
        say(f"距離範囲を塗り中 ({i}/{len(images)}): {path.name}")
        img = imread(path)
        if img is None:
            notes.append(f"{path.name}: 画像を読み込めないためスキップ")
            continue
        res = cams.corrected_distance_map(path.stem)
        if res is None:
            notes.append(f"{path.name}: {cams.errors.get(path.stem, '距離を算出できません')}ためスキップ")
            continue
        dist, dmap = res
        if not dist.is_identity:
            img = dist.correct_image(img)
            notes.append(f"{path.name}: 魚眼補正して塗りました（λ={dist.lam:.4f}）")
        out = output_dir / f"{path.stem}{OUTPUT_SUFFIX}.png"
        if not imwrite(out, render_distance_range(img, dmap, max_distance_m)):
            notes.append(f"{path.name}: 保存に失敗: {out}")
            continue
        outputs.append(out)
    return outputs, notes
