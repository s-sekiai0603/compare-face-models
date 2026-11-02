"""指定顔サイズ以上で見える画角範囲の確認（足元ヒートマップ）。

比較対象フォルダ（出力先ルート\\日時\\カメラID\\映像名\\追跡ID\\<追跡ID>.json）の各 json から、顔 bbox の高さが指定 px 以上の画像の
足元位置（foot_xy。フレーム座標）をカメラごとに集め、カメラ画像（cam1.png など。ファイル名 = カメラID）にヒートマップで重ねて保存する。
ヒートマップは pose-estimation-app の滞在ヒートマップと同じ手順（グリッド集計 → ぼかし → 青→赤の色相 → 値を透過度にして合成）。"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..common.imageio import imread, imwrite
from .gallery import _bbox_side, in_excluded_dir, in_query_sets_dir

HEATMAP_GRID_SIZE = 10       # 足元を集計するグリッドの大きさ(px)
HEATMAP_MAX_ALPHA = 0.85     # 最も濃い所の不透明度
HEATMAP_MIN_ALPHA = 0.35     # 足元が 1 点でもある所の最低の不透明度（薄すぎて見えなくなるのを防ぐ）
HEATMAP_GAMMA = 0.5          # 1.0 未満で低い値を持ち上げる
HEATMAP_GAUSSIAN_SIGMA = 20  # ぼかしのシグマ(px)
HEATMAP_PEAK_PERCENTILE = 95  # 足元がある範囲の密度のこの百分位を赤とする（1 か所の密集に他が埋もれないようにする）
LEGEND_AREA_PX = 50          # 凡例の件数の単位: 周囲 この px 四方 あたりの足元の件数
HEATMAP_PRESENCE = 0.02      # 密度がこの値（赤を 1 とした割合）未満の所は色を付けない
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
OUTPUT_SUFFIX = "_heatmap"

Points = dict[str, list[tuple[float, float]]]


def _face_height(im: dict) -> float | None:
    h = im.get("face_height")
    return float(h) if h is not None else _bbox_side(im.get("face_bbox_xyxy"), 1)


def collect_foot_points(gallery_dir: Path, min_face_height: float) -> Points:
    """比較対象フォルダの json から、顔 bbox の高さが min_face_height 以上の画像の足元位置をカメラID（小文字）ごとに集める。
    old フォルダの画像と検索元セット置き場は除く（比較対象と同じ扱い）。"""
    points: Points = defaultdict(list)
    for jp in sorted(gallery_dir.rglob("*.json")):
        track_dir = jp.parent
        if jp.stem != track_dir.name or in_query_sets_dir(jp, gallery_dir) or in_excluded_dir(jp, gallery_dir):
            continue
        camera_dir = track_dir.parent.parent  # 追跡ID → 映像名 → カメラID
        if camera_dir == track_dir.parent:
            continue
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for im in data.get("images", []) if isinstance(data, dict) else []:
            h = _face_height(im)
            foot = im.get("foot_xy")
            if h is None or h < min_face_height or not foot or len(foot) < 2:
                continue
            points[camera_dir.name.lower()].append((float(foot[0]), float(foot[1])))
    return dict(points)


def make_heatmap(shape: tuple[int, int], points: list[tuple[float, float]]) -> tuple[np.ndarray, float]:
    """((高さ, 幅) の 0〜1 のヒートマップ, 赤(=1)に当たる件数)。件数は周囲 LEGEND_AREA_PX 四方あたりの足元の数。点が無ければ全部 0。"""
    height, width = shape
    grid_h, grid_w = height // HEATMAP_GRID_SIZE + 1, width // HEATMAP_GRID_SIZE + 1
    hist = np.zeros((grid_h, grid_w), dtype=np.float32)
    for x, y in points:
        gx, gy = int(x // HEATMAP_GRID_SIZE), int(y // HEATMAP_GRID_SIZE)
        if 0 <= gx < grid_w and 0 <= gy < grid_h:
            hist[gy, gx] += 1.0
    heat = cv2.resize(hist, (width, height), interpolation=cv2.INTER_LINEAR)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=HEATMAP_GAUSSIAN_SIGMA)
    present = heat[heat > heat.max() * 0.01] if heat.max() > 0 else heat
    peak = float(np.percentile(present, HEATMAP_PEAK_PERCENTILE)) if present.size else 0.0
    # heat の値は「グリッド 1 マス（HEATMAP_GRID_SIZE 四方）あたりの件数」をぼかしたもの。面積比で凡例の単位に換算する
    peak_count = peak * (LEGEND_AREA_PX / HEATMAP_GRID_SIZE) ** 2
    return (np.clip(heat / peak, 0.0, 1.0) if peak > 0 else heat), peak_count


def render_heatmap(image_bgr: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    """画像に足元ヒートマップを透過で重ねる。色は青（少ない）→シアン→緑→黄→赤（多い）。点が無ければ元画像のまま。"""
    if not points:
        return image_bgr.copy()
    heat, peak_count = make_heatmap(image_bgr.shape[:2], points)
    hue = (120 * (1.0 - heat)).astype(np.uint8)  # OpenCV の hue は 0-180。青=120、赤=0
    hsv = np.stack([hue, np.full_like(hue, 255), np.full_like(hue, 255)], axis=-1)
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
    strength = np.power(heat, HEATMAP_GAMMA)
    alpha = HEATMAP_MIN_ALPHA + (HEATMAP_MAX_ALPHA - HEATMAP_MIN_ALPHA) * strength
    alpha = np.where(heat >= HEATMAP_PRESENCE, alpha, strength * HEATMAP_MIN_ALPHA / HEATMAP_PRESENCE ** HEATMAP_GAMMA)
    alpha = np.clip(alpha, 0.0, HEATMAP_MAX_ALPHA).astype(np.float32)[..., None]
    out = (image_bgr.astype(np.float32) * (1 - alpha) + color * alpha).astype(np.uint8)
    return _draw_legend(out, peak_count, len(points))


def _draw_legend(img: np.ndarray, peak_count: float, total: int) -> np.ndarray:
    """右上に色の凡例を描く。色バーの下に、色ごとの件数のめやす（周囲 LEGEND_AREA_PX 四方あたりの足元の数。右端は「以上」）と総件数。"""
    h, w = img.shape[:2]
    if w < 480 or h < 240:  # 凡例が入りきらない小さい画像には描かない
        return img
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.45
    bw, bh = max(w // 6, 120), max(h // 60, 10)
    x0, y0 = w - bw - w // 40, h // 40 + 40
    ramp = np.linspace(0.0, 1.0, bw, dtype=np.float32)
    hue = np.tile((120 * (1.0 - ramp)).astype(np.uint8), (bh, 1))
    bar = cv2.cvtColor(np.stack([hue, np.full_like(hue, 255), np.full_like(hue, 255)], axis=-1), cv2.COLOR_HSV2BGR)
    cv2.rectangle(img, (x0 - 8, y0 - 40), (x0 + bw + 8, y0 + bh + 30), (255, 255, 255), -1)
    cv2.putText(img, f"points per {LEGEND_AREA_PX}x{LEGEND_AREA_PX}px (n={total})", (x0, y0 - 22), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
    img[y0:y0 + bh, x0:x0 + bw] = bar
    top = max(round(peak_count), 1)
    for frac, label in ((0.0, "0"), (0.5, f"{max(round(peak_count / 2), 1)}"), (1.0, f"{top}+")):
        tx = x0 + int(frac * (bw - 1))
        cv2.line(img, (tx, y0 + bh), (tx, y0 + bh + 4), (0, 0, 0), 1)
        size = cv2.getTextSize(label, font, scale, 1)[0][0]
        cv2.putText(img, label, (min(max(tx - size // 2, x0 - 4), x0 + bw + 4 - size), y0 + bh + 20), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def run_face_size_map(gallery_dir: Path, camera_dir: Path, output_dir: Path, min_face_height: float,
                      progress: Callable[[str], None] | None = None) -> tuple[list[Path], list[str]]:
    """カメラ画像ごとにヒートマップ画像を output_dir へ保存する。戻り値は (出力ファイル, スキップ・注意の説明)。"""
    say = progress or (lambda _msg: None)
    images = sorted(p for p in camera_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"カメラ画像が見つかりません: {camera_dir}")
    say("比較対象の json を読み込み中...")
    points = collect_foot_points(gallery_dir, min_face_height)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    notes: list[str] = []
    for i, path in enumerate(images, 1):
        pts = points.get(path.stem.lower())
        if pts is None:
            notes.append(f"{path.name}: 対応するカメラID（{path.stem}）の足元データが無いためスキップ")
            continue
        say(f"プロット中 ({i}/{len(images)}): {path.name}（{len(pts)} 点）")
        img = imread(path)
        if img is None:
            notes.append(f"{path.name}: 画像を読み込めないためスキップ")
            continue
        out = output_dir / f"{path.stem}{OUTPUT_SUFFIX}.png"
        if not imwrite(out, render_heatmap(img, pts)):
            notes.append(f"{path.name}: 保存に失敗: {out}")
            continue
        outputs.append(out)
    return outputs, notes
