"""レンズの放射歪み（樽型・糸巻き型）の補正。

除算モデル `x_u = x_d / (1 + λ·r_d²)` を使う（座標は画像中心を原点に、画像の長辺の半分で割った正規化座標。r は原点からの距離）。
焦点距離を前提にしないので、焦点距離の推定（projective.py）と循環しない。λ < 0 が樽型（端が外へ膨らむ）、λ > 0 が糸巻き型。
歪みの中心は画像中心と仮定する。

補正後の座標系 = 補正後の画像のピクセル座標。画像サイズは元画像と同じで、補正後の内容が枠に収まるよう中心から縮尺 k で縮める
（k は λ と画像サイズから決まる。縮尺は幾何（射影変換・消失点・距離）に影響しない）。
GUI に依存しない。"""
from __future__ import annotations

import cv2
import numpy as np

from .projective import ProjectionError

MIN_LINE_POINTS = 3     # 歪み補正線 1 本あたりの最小点数
LAMBDA_SCAN_STEPS = 801  # λ の粗い探索の分割数
EDGE_SAMPLES = 64       # 補正後の内容の広がりを調べる、画像の辺上のサンプル数
MIN_DENOMINATOR = 1e-3


def lambda_range(size: tuple[int, int]) -> tuple[float, float]:
    """この画像サイズで使える λ の範囲（画像の隅でも 1 + λ·r² が正で、逆変換が実数になる範囲）。"""
    w, h = size
    s = max(w, h) / 2.0
    r2_max = ((w / 2.0) ** 2 + (h / 2.0) ** 2) / (s * s)
    return -0.9 / r2_max, 0.2 / r2_max


def _undistort_norm(pts_n: np.ndarray, lam: float) -> np.ndarray:
    r2 = np.sum(pts_n * pts_n, axis=-1, keepdims=True)
    return pts_n / np.maximum(1.0 + lam * r2, MIN_DENOMINATOR)


def _distort_norm(pts_n: np.ndarray, lam: float) -> np.ndarray:
    """_undistort_norm の逆。r_u = r_d / (1 + λ·r_d²) を r_d について解く（r_d = 0 付近で連続な方の根）。"""
    if lam == 0.0:
        return pts_n.copy()
    ru = np.sqrt(np.sum(pts_n * pts_n, axis=-1, keepdims=True))
    disc = np.maximum(1.0 - 4.0 * lam * ru * ru, 0.0)
    safe = np.where(ru > 1e-12, ru, 1.0)
    rd = np.where(ru > 1e-12, (1.0 - np.sqrt(disc)) / (2.0 * lam * safe), 1.0)
    scale = np.where(ru > 1e-12, rd / safe, 1.0)
    return pts_n * scale


class Distortion:
    """歪み係数 λ と画像サイズから、元画像 ⇄ 補正後の画像の座標変換と、補正後の画像の生成を行う。"""

    def __init__(self, lam: float, size: tuple[int, int]):
        self.lam = float(lam)
        self.size = (int(size[0]), int(size[1]))
        self.center = np.array([size[0] / 2.0, size[1] / 2.0])
        self.norm = max(size) / 2.0
        self.scale = self._fit_scale()

    @property
    def is_identity(self) -> bool:
        return self.lam == 0.0

    def _fit_scale(self) -> float:
        """補正後の内容（元画像の全域）が枠に収まるよう、中心からの縮尺を決める（拡大はしない）。"""
        w, h = self.size
        t = np.linspace(0.0, 1.0, EDGE_SAMPLES)
        edge = np.concatenate([
            np.stack([t * w, np.zeros_like(t)], axis=1), np.stack([t * w, np.full_like(t, h)], axis=1),
            np.stack([np.zeros_like(t), t * h], axis=1), np.stack([np.full_like(t, w), t * h], axis=1)])
        u = _undistort_norm((edge - self.center) / self.norm, self.lam) * self.norm
        ext_x, ext_y = float(np.max(np.abs(u[:, 0]))), float(np.max(np.abs(u[:, 1])))
        return min(1.0, (w / 2.0) / max(ext_x, 1e-9), (h / 2.0) / max(ext_y, 1e-9))

    def to_corrected(self, pts) -> np.ndarray:
        """元画像のピクセル座標 → 補正後の画像のピクセル座標。pts は (..., 2)。"""
        p = np.asarray(pts, dtype=np.float64)
        u = _undistort_norm((p - self.center) / self.norm, self.lam)
        return self.center + u * self.norm * self.scale

    def to_original(self, pts) -> np.ndarray:
        """補正後の画像のピクセル座標 → 元画像のピクセル座標。"""
        p = np.asarray(pts, dtype=np.float64)
        d = _distort_norm((p - self.center) / (self.norm * self.scale), self.lam)
        return self.center + d * self.norm

    def correct_image(self, img: np.ndarray) -> np.ndarray:
        """元画像 → 補正後の画像（同じ大きさ。元画像の外は黒）。"""
        if self.is_identity:
            return img
        return Remapper(self).apply(img)


class Remapper:
    """補正後の画像を作る remap のマップを 1 回だけ作って使い回す（映像の全フレームを補正するとき用。correct_image と同じ結果）。"""

    def __init__(self, dist: Distortion):
        self.identity = dist.is_identity
        self.size = dist.size
        if self.identity:
            self.map1 = self.map2 = None
            return
        w, h = dist.size
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float64) + 0.5, np.arange(h, dtype=np.float64) + 0.5)
        src = dist.to_original(np.stack([xs, ys], axis=-1)) - 0.5
        self.map1, self.map2 = src[..., 0].astype(np.float32), src[..., 1].astype(np.float32)

    def apply(self, img: np.ndarray) -> np.ndarray:
        if self.identity:
            return img
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _line_cost(pts_u: np.ndarray) -> float:
    """点列の直線からのずれ = 共分散の最小固有値 / 固有値の和（縮尺に依らない。一直線なら 0）。"""
    c = pts_u - pts_u.mean(axis=0)
    w = np.linalg.eigvalsh(c.T @ c)
    return float(w[0] / max(w.sum(), 1e-12))


def estimate_lambda(lines, size: tuple[int, int]) -> float:
    """現実ではまっすぐな線（元画像の座標で、1 本 3 点以上の点列）のリストから、補正後に最も直線に近づく λ を求める。"""
    w, h = size
    center = np.array([w / 2.0, h / 2.0])
    norm = max(size) / 2.0
    valid = [(np.asarray(line, dtype=np.float64).reshape(-1, 2) - center) / norm
             for line in lines if len(line) >= MIN_LINE_POINTS]
    if not valid:
        raise ProjectionError(f"歪み補正線は 1 本あたり {MIN_LINE_POINTS} 点以上の点が必要です。")

    def cost(lam: float) -> float:
        return float(np.mean([_line_cost(_undistort_norm(p, lam)) for p in valid]))

    lo, hi = lambda_range(size)
    grid = np.linspace(lo, hi, LAMBDA_SCAN_STEPS)
    costs = [cost(float(v)) for v in grid]
    i = int(np.argmin(costs))
    if i == 0 or i == len(grid) - 1:
        raise ProjectionError("歪みが大きすぎて推定できません（補正できる範囲の端になりました）。線の点の取り方を見直してください。")
    a, b = float(grid[i - 1]), float(grid[i + 1])
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):  # 黄金分割法
        c, d = b - phi * (b - a), a + phi * (b - a)
        if cost(c) < cost(d):
            b = d
        else:
            a = c
    return (a + b) / 2.0
