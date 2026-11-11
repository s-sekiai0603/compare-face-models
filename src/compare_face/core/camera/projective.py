"""カメラ画像の射影変換設定（画像座標 → 地面の現実座標[m]）とカメラ設置座標の算出。

画像上の 4 点（現実では長方形になる点。クリック順に一周）と長方形の 2 辺の長さから、
  - 画像 → 地面（m）の射影変換 H
  - カメラ設置座標 = カメラ直下の地面点が画像に写る位置（nadir。画像の外になり得る）
を求める。nadir は長方形の 2 組の対辺の消失点から求める（主点 = 画像中心・正方画素と仮定して焦点距離を推定）。
距離は「カメラ直下の地面点から対象点までの地面上の距離（m）」。GUI に依存しない。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MIN_W = 1e-3  # 消失点の同次座標 w を 4 点の広がり² で割った値がこれ未満なら対辺が平行（消失点が無限遠）とみなす


class ProjectionError(ValueError):
    """4 点・辺長からカメラ位置を算出できない（点の並び・退化した配置）。"""


@dataclass(frozen=True)
class CameraProjection:
    homography: np.ndarray          # 画像 → 地面(m) 3x3
    camera_xy: tuple[float, float] | None  # カメラ設置座標（画像座標系、画像外になり得る）。垂直補正後は無限遠なので None
    focal_px: float                 # 推定焦点距離(px)
    height_m: float                 # 推定カメラ高さ(m)
    vanishing_points: tuple = ()    # 長方形の 2 組の対辺の消失点 ((x, y), (x, y))。結ぶ直線が地平線
    camera_ground: tuple[float, float] | None = None  # カメラ直下の地面点の地面座標(m)（距離の基準）

    def ground_xy(self, xy: tuple[float, float]) -> tuple[float, float]:
        p = self.homography @ np.array([xy[0], xy[1], 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def distance_m(self, xy: tuple[float, float]) -> float:
        """カメラ設置地点（直下）から画像座標 xy までの地面上の距離(m)。"""
        gx, gy = self.ground_xy(xy)
        cx, cy = self.camera_ground if self.camera_ground is not None else self.ground_xy(self.camera_xy)
        return float(np.hypot(gx - cx, gy - cy))

    def slant_distance_m(self, xy: tuple[float, float]) -> float:
        """カメラ本体（地面から推定高さの位置）から画像座標 xy の地面点までの直線距離(m)。"""
        return float(np.hypot(self.distance_m(xy), self.height_m))


def resolve_sides(long_m: float, short_m: float, first_side: str) -> tuple[float, float]:
    """辺 p0→p1 の長さ a と辺 p1→p2 の長さ b を返す。first_side は 'long' か 'short'（p0→p1 の辺がどちらか）。"""
    return (long_m, short_m) if first_side == "long" else (short_m, long_m)


def _intersect(a1, a2, b1, b2) -> np.ndarray:
    """直線 a1-a2 と b1-b2 の交点（同次座標）。"""
    la = np.cross([a1[0], a1[1], 1.0], [a2[0], a2[1], 1.0])
    lb = np.cross([b1[0], b1[1], 1.0], [b2[0], b2[1], 1.0])
    return np.cross(la, lb)


def compute_projection(points, side_a: float, side_b: float, image_size: tuple[int, int],
                       focal_px: float | None = None, vertical=None) -> CameraProjection:
    """points: 画像座標 4 点（長方形を一周する順）。side_a = p0→p1 の長さ(m)、side_b = p1→p2 の長さ(m)。
    image_size: (幅, 高さ)。主点を画像中心とみなす。focal_px を与えると焦点距離を推定せずその値を使う（設置座標のドラッグ補正・垂直補正後・保存済みの設定の焦点距離用）。
    vertical: 鉛直（下向き）方向のカメラ座標ベクトル。与えると長方形の消失点から求めず、これをカメラ直下の方向とする
    （垂直補正後の画像用。(0, 1, 0) なら設置座標は画像の下方向の無限遠で camera_xy は None になる）。"""
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    if side_a <= 0 or side_b <= 0:
        raise ProjectionError("辺の長さは 0 より大きい値を指定してください。")
    cross = []
    for i in range(4):
        e1, e2 = pts[(i + 1) % 4] - pts[i], pts[(i + 2) % 4] - pts[(i + 1) % 4]
        cross.append(e1[0] * e2[1] - e1[1] * e2[0])
    if not (all(c > 0 for c in cross) or all(c < 0 for c in cross)):
        raise ProjectionError("4 点が凸四角形になっていません。長方形を一周する順にクリックしてください。")

    world = np.array([[0, 0], [side_a, 0], [side_a, side_b], [0, side_b]], dtype=np.float32)
    h = cv2.getPerspectiveTransform(pts.astype(np.float32), world).astype(np.float64)

    # 同次座標のスケールを揃えてから無限遠判定する（直線の係数は座標の大きさに比例するため）
    v1 = _intersect(pts[0], pts[1], pts[3], pts[2])  # 辺 p0p1 と p3p2（長さ a の組）の消失点
    v2 = _intersect(pts[1], pts[2], pts[0], pts[3])  # 辺 p1p2 と p0p3（長さ b の組）の消失点
    span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1.0)
    if abs(v1[2]) < MIN_W * span * span or abs(v2[2]) < MIN_W * span * span:
        raise ProjectionError("対辺がほぼ平行に写っており、カメラ位置を算出できません。台形が強く出る 4 点を選んでください。")
    c = np.array([image_size[0] / 2.0, image_size[1] / 2.0])
    q1, q2 = v1[:2] / v1[2], v2[:2] / v2[2]
    if focal_px is None:
        f2 = -float((q1 - c) @ (q2 - c))
        if f2 <= 0:
            raise ProjectionError("焦点距離を推定できません（4 点の取り方を変えてください）。")
        f = float(np.sqrt(f2))
    else:
        f = float(focal_px)
    k = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1.0]])
    k_inv = np.linalg.inv(k)

    d1 = k_inv @ np.array([q1[0], q1[1], 1.0])
    d2 = k_inv @ np.array([q2[0], q2[1], 1.0])
    if vertical is None:
        n = np.cross(d1 / np.linalg.norm(d1), d2 / np.linalg.norm(d2))  # 地面の法線（カメラ座標）
    else:
        n = np.asarray(vertical, dtype=np.float64)
    n /= np.linalg.norm(n)
    rays = [k_inv @ np.array([x, y, 1.0]) for x, y in pts]
    if n @ np.mean(rays, axis=0) < 0:
        if vertical is not None:
            raise ProjectionError("4 点が地平線より上にあります。地面上の長方形の 4 点を指定してください。")
        n = -n  # カメラから地面へ向かう向きにそろえる
    nadir = k @ n
    if abs(nadir[2]) < 1e-6:
        if vertical is None:
            raise ProjectionError("カメラが水平に近く、設置座標が無限遠になります。")
        camera_xy = None  # 垂直補正後: カメラ直下は画像の無限遠（地面座標には写せる）
    else:
        camera_xy = (float(nadir[0] / nadir[2]), float(nadir[1] / nadir[2]))
    g = h @ nadir  # 同次座標のまま地面に写す（無限遠でもよい）
    if abs(g[2]) < 1e-12:
        raise ProjectionError("カメラ直下の地面座標を求められません（4 点の取り方を変えてください）。")
    camera_ground = (float(g[0] / g[2]), float(g[1] / g[2]))

    # カメラ高さ: 地面までの距離 1 としたときの 3D 辺長と実際の辺長(m)の比
    p3d = [r / (n @ r) for r in rays]
    ratios = [side_a / np.linalg.norm(p3d[1] - p3d[0]), side_b / np.linalg.norm(p3d[2] - p3d[1]),
              side_a / np.linalg.norm(p3d[2] - p3d[3]), side_b / np.linalg.norm(p3d[3] - p3d[0])]
    return CameraProjection(h, camera_xy, f, float(np.mean(ratios)),
                            ((float(q1[0]), float(q1[1])), (float(q2[0]), float(q2[1]))), camera_ground)


def clip_line_to_rect(p1, p2, width: float, height: float):
    """2 点 p1, p2 を通る直線を、画像の矩形 [0,width]x[0,height] で切り取った線分 ((x, y), (x, y))。交わらなければ None。"""
    x1, y1 = p1
    dx, dy = p2[0] - x1, p2[1] - y1
    if dx == 0 and dy == 0:
        return None
    t0, t1 = -np.inf, np.inf
    for p, q in ((-dx, x1), (dx, width - x1), (-dy, y1), (dy, height - y1)):  # Liang-Barsky（直線は無限に延びる）
        if p == 0:
            if q < 0:
                return None
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
    if t0 > t1:
        return None
    return (x1 + t0 * dx, y1 + t0 * dy), (x1 + t1 * dx, y1 + t1 * dy)


NADIR_SCAN_STEPS = 120          # 焦点距離の粗い探索の分割数
NADIR_SCAN_RANGE = (0.2, 5.0)   # 探索する焦点距離の範囲（4 点から推定した焦点距離に対する倍率）


def focal_for_nadir(points, side_a: float, side_b: float, image_size: tuple[int, int],
                    target_xy: tuple[float, float]) -> float:
    """カメラ設置座標（カメラ直下が写る位置）を target_xy の近くへ動かすための焦点距離。
    設置座標は焦点距離を変えると画像上の曲線（ほぼ縦方向）の上を動くので、target_xy をその曲線へ投影した（最も近い）点の焦点距離を返す。
    4 点から推定した焦点距離を中心に NADIR_SCAN_RANGE 倍の範囲で探す。垂直補正後（設置座標が無限遠）には使えない。"""
    base = compute_projection(points, side_a, side_b, image_size)
    tx, ty = float(target_xy[0]), float(target_xy[1])

    def cost(f: float) -> float:
        try:
            xy = compute_projection(points, side_a, side_b, image_size, focal_px=f).camera_xy
        except ProjectionError:
            return float("inf")
        return float("inf") if xy is None else (xy[0] - tx) ** 2 + (xy[1] - ty) ** 2

    fs = base.focal_px * np.geomspace(NADIR_SCAN_RANGE[0], NADIR_SCAN_RANGE[1], NADIR_SCAN_STEPS)
    costs = [cost(float(f)) for f in fs]
    best = int(np.argmin(costs))
    if not np.isfinite(costs[best]):
        raise ProjectionError("カメラ設置座標を求められません。4 点の取り方を見直してください。")
    lo, hi = float(fs[max(best - 1, 0)]), float(fs[min(best + 1, len(fs) - 1)])
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(40):  # 黄金分割法
        c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
        if cost(c) < cost(d):
            hi = d
        else:
            lo = c
    return (lo + hi) / 2


# ---- 設定ファイル（<画像名>.json） ----
def save_config(path: Path, image_name: str, points, long_m: float, short_m: float, first_side: str,
                camera_xy: tuple[float, float] | None, focal_px: float | None = None,
                distortion: dict | None = None, rectification: dict | None = None) -> None:
    """設定ファイルを書く。focal_px は、カメラ設置座標をドラッグして補正した（focal_for_nadir）ときだけ記録する（calibration.focal_px。
    距離の算出は 4 点から推定する代わりにこの焦点距離を使う）。
    distortion = {"lambda", "scale", "lines"} はレンズ歪みを補正したときだけ記録する。補正したときは points / camera_xy は
    「補正後の画像」の座標（元画像の点は distortion.py の変換で補正後の座標に直してから使う）。lines は元画像の座標。
    rectification = {"lines", "focal_px"} は垂直補正（vertical.py）をしたときだけ記録する。lines は垂直線（元画像の座標。
    歪み補正後に垂直補正する）、focal_px は垂直補正に使った焦点距離。垂直補正したときは points は垂直補正後の画像の座標、
    camera_xy は無限遠なので null。"""
    data = {
        "image": image_name,
        "points": [[float(x), float(y)] for x, y in points],
        "long_side_m": float(long_m),
        "short_side_m": float(short_m),
        "first_side": first_side,  # points[0]→points[1] の辺が long / short のどちらか
        "camera_xy": None if camera_xy is None else [float(camera_xy[0]), float(camera_xy[1])],
    }
    if rectification is not None:
        data["rectification"] = {
            "lines": [[[float(x), float(y)] for x, y in line] for line in rectification["lines"]],
            "focal_px": float(rectification["focal_px"]),
        }
    if distortion is not None:
        data["distortion"] = {
            "lambda": float(distortion["lambda"]), "scale": float(distortion["scale"]),
            "lines": [[[float(x), float(y)] for x, y in line] for line in distortion["lines"]],
        }
    if focal_px is not None:
        data["calibration"] = {"focal_px": float(focal_px)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(path: Path) -> dict | None:
    """読めない・形式が違うときは None。"""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        pts = [(float(x), float(y)) for x, y in d["points"]]
        if len(pts) != 4 or d["first_side"] not in ("long", "short"):
            return None
        cal = d.get("calibration")   # カメラ設置座標をドラッグして補正したときの焦点距離（古い版の実測補正で保存したものも同じ形）
        calibration = None
        if cal is not None:
            calibration = {"focal_px": float(cal["focal_px"])}
        dist = d.get("distortion")
        distortion = None
        if dist is not None:
            distortion = {"lambda": float(dist["lambda"]), "scale": float(dist["scale"]),
                          "lines": [[(float(x), float(y)) for x, y in line] for line in dist["lines"]]}
        rect = d.get("rectification")
        rectification = None
        if rect is not None:
            rectification = {"lines": [tuple((float(x), float(y)) for x, y in line) for line in rect["lines"]],
                             "focal_px": float(rect["focal_px"])}
        cam = d["camera_xy"]
        return {"calibration": calibration, "distortion": distortion, "rectification": rectification, "points": pts, "long_side_m": float(d["long_side_m"]), "short_side_m": float(d["short_side_m"]),
                "first_side": d["first_side"], "camera_xy": None if cam is None else tuple(float(v) for v in cam)}
    except (OSError, ValueError, KeyError, TypeError):
        return None
