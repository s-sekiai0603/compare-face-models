"""建物の垂直線による垂直補正（キーストーン補正）。GUI に依存しない。

カメラが傾いていると、現実では垂直な線（建物の壁の角など）は画像の下の消失点（カメラ直下）に向かって収束し、
上に行くほど広がって見える。画像上の垂直線を数本指定してその消失点を求め、「カメラを水平に向け直したら」の画像
（H = S·K·R·K⁻¹。垂直線がすべて画像の縦方向と平行になる）に変換する。主点 = 画像中心・正方画素と仮定し、焦点距離は
呼び出し側（長方形から求めた値や実測で補正した値）が与える。

補正後の画像ではカメラ直下の地面点が画像の下方向の無限遠になる（設置座標は画像座標では表せない）。
地面上の距離は射影変換 H だけで決まる（core/camera/projective.py の vertical 引数）。座標は補正前の画像の座標を
「補正前」、補正後の画像の座標を「補正後」と呼ぶ。縮尺 S は補正後の画像に全体が収まるように決める（1 以下。小さくなりすぎるときは MIN_RECT_SCALE で止め、端を切る）。"""
from __future__ import annotations

import cv2
import numpy as np

from .projective import ProjectionError

MIN_VERTICAL_LINES = 2
MIN_DOWN_Y = 0.2         # 鉛直方向（カメラ座標・正規化）の下向き成分がこれ未満なら、画像の縦方向から外れすぎとして補正しない
MIN_FRONT_Z = 0.05       # 回転後に画像の端がカメラの前方（z）にこれ以上無いと補正できない
MIN_RECT_SCALE = 0.5     # 縮尺の下限（全体を収めると小さくなりすぎるときは、画像の端を切って解像度を保つ）
MIN_SECOND_SV = 1e-4     # 垂直線がすべて同じ線のとき（消失点が決まらない）の判定


def vertical_vanishing_point(lines, size: tuple[int, int]) -> np.ndarray:
    """垂直線（各線は 2 点 ((x, y), (x, y))）がすべて通る点 = 鉛直方向の消失点を、同次座標 (x, y, w) で返す。
    無限遠（垂直線が画像上で平行）でもよい。各線までの距離の二乗和を最小にする（画像中心・長辺で正規化して解く）。"""
    if len(lines) < MIN_VERTICAL_LINES:
        raise ProjectionError(f"垂直線は {MIN_VERTICAL_LINES} 本以上指定してください。")
    w, h = size
    cx, cy, s = w / 2.0, h / 2.0, float(max(w, h))
    rows = []
    for p1, p2 in lines:
        a = np.array([(p1[0] - cx) / s, (p1[1] - cy) / s, 1.0])
        b = np.array([(p2[0] - cx) / s, (p2[1] - cy) / s, 1.0])
        line = np.cross(a, b)
        norm = float(np.hypot(line[0], line[1]))
        if norm < 1e-9:
            raise ProjectionError("垂直線の 2 点が同じ位置です。離れた 2 点を指定してください。")
        rows.append(line / norm)
    mat = np.array(rows)
    _, sv, vt = np.linalg.svd(mat)
    if sv[1] < MIN_SECOND_SV:
        raise ProjectionError("垂直線がすべて同じ線です。別の壁の角など、位置の違う垂直線を指定してください。")
    t_inv = np.array([[s, 0.0, cx], [0.0, s, cy], [0.0, 0.0, 1.0]])
    return t_inv @ vt[-1]


def line_tilt_deg(p1, p2) -> float:
    """2 点を結ぶ線の、画像の縦方向からの傾き（度）。"""
    return float(np.degrees(np.arctan2(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))))


class VerticalRectification:
    """補正前の画像 ⇄ 垂直補正後の画像の座標変換と画像の変換。"""

    vertical = (0.0, 1.0, 0.0)  # 補正後のカメラ座標での鉛直（下向き）方向

    def __init__(self, vp, source_focal_px: float, size: tuple[int, int], min_scale: float = MIN_RECT_SCALE):
        if source_focal_px <= 0:
            raise ProjectionError("焦点距離が正ではありません。")
        w, h = size
        cx, cy = w / 2.0, h / 2.0
        f = float(source_focal_px)
        d = np.array([(vp[0] - cx * vp[2]) / f, (vp[1] - cy * vp[2]) / f, vp[2]], dtype=np.float64)
        norm = float(np.linalg.norm(d))
        if norm <= 0:
            raise ProjectionError("垂直線の消失点を求められません。")
        d /= norm
        if d[1] < 0:
            d = -d  # 画像の下向きに揃える（消失点は ±d のどちらでも同じ点）
        if d[1] < MIN_DOWN_Y:
            raise ProjectionError("垂直線の消失点が画像の縦方向から大きく外れています（カメラが真下に近い向きの可能性）。"
                                  "垂直線を見直してください。")
        e = np.array([0.0, 1.0, 0.0])
        axis = np.cross(d, e)
        s2 = float(axis @ axis)
        if s2 < 1e-24:
            rot = np.eye(3)
        else:
            a_x = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
            rot = np.eye(3) + a_x + a_x @ a_x * ((1.0 - float(d @ e)) / s2)
        k = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
        h_rot = k @ rot @ np.linalg.inv(k)
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64) @ h_rot.T
        if np.any(corners[:, 2] < MIN_FRONT_Z):
            raise ProjectionError("垂直補正すると画像の端がカメラの後ろ側になり、補正できません（垂直線か焦点距離を見直してください）。")
        u = corners[:, :2] / corners[:, 2:3] - np.array([cx, cy])
        fit = min(1.0, cx / max(float(np.abs(u[:, 0]).max()), 1e-9), cy / max(float(np.abs(u[:, 1]).max()), 1e-9))
        scale = max(fit, min_scale)
        self.cropped = scale > fit  # True なら画像の端が補正後の画像からはみ出して切れている
        s_mat = np.array([[scale, 0.0, cx * (1 - scale)], [0.0, scale, cy * (1 - scale)], [0.0, 0.0, 1.0]])
        self.size = (int(w), int(h))
        self.source_focal_px = f           # 補正前の画像で使った焦点距離
        self.scale = float(scale)
        self.focal_px = f * self.scale     # 補正後の画像の焦点距離（縮尺 S をかけた値）
        self.matrix = s_mat @ h_rot        # 補正前 → 補正後
        self._inverse = np.linalg.inv(self.matrix)

    @staticmethod
    def _apply(mat: np.ndarray, pts) -> np.ndarray:
        arr = np.asarray(pts, dtype=np.float64)
        flat = arr.reshape(-1, 2)
        q = np.hstack([flat, np.ones((len(flat), 1))]) @ mat.T
        return (q[:, :2] / q[:, 2:3]).reshape(arr.shape)

    def to_rectified(self, pts) -> np.ndarray:
        """補正前の座標 → 補正後の座標。"""
        return self._apply(self.matrix, pts)

    def to_unrectified(self, pts) -> np.ndarray:
        """補正後の座標 → 補正前の座標。"""
        return self._apply(self._inverse, pts)

    def rectify_image(self, img: np.ndarray) -> np.ndarray:
        """画像を補正後の画像にする（同じ大きさ。範囲外は黒）。"""
        return cv2.warpPerspective(img, self.matrix, self.size, flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
