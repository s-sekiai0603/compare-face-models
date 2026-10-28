"""顔クロップの生成。premises-security の顔識別と同じ判定・前処理を移植したもの。

1人物・1フレームの流れ:
  1. 顔ゲート（骨格キーポイント）: 鼻+片目以上が見えている / 下向きの顔でない / 両目間隔が極端に狭くない
  2. 顔幅に応じた適応マージンで顔 box を拡張（inflate）
  3. 小さい顔（短辺 < 96px）は周辺 ROI を長辺 800px に拡大してから切り出す
  4. 切り出し（品質ガード: 切り出し box の短辺が 60px 未満は premises-security では棄却）
切り出した画像はリサイズせずに保存し、特徴量抽出時に各モデルの入力サイズへ整える。

顔ゲートと品質ガードの判定結果（採用基準）は JSON に記録するだけで、本ツールは基準に外れた顔も保存する。
基準に外れた画像を使わない判断はレポート時に行う（core/report/gate.py。既定は premises-security と同じ基準）。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# --- キーポイント（COCO-17）と顔ゲート（premises-security config.KP_CONF_TH 等） ---
KP_CONF_TH = 0.15
IDX_NOSE, IDX_L_EYE, IDX_R_EYE, IDX_L_EAR, IDX_R_EAR = 0, 1, 2, 3, 4
FACE_GATE_RELAXED = True     # True: 鼻+片目以上で採用（斜め〜軽い横顔も拾う）
MIN_EYE_RATIO = 0.05         # eye_dist/顔幅 がこれ未満は棄却
FRONTAL_MIN_RATIO = 0.25     # eye_dist/顔幅 がこれ以上なら正面とみなす
# --- 適応マージン（顔幅に応じて bbox の拡大率を変える） ---
FACE_WIDTH_SMALL = 40.0
FACE_WIDTH_MEDIUM = 80.0
INFLATE_MARGIN_SMALL = 0.30
INFLATE_MARGIN_MEDIUM = 0.20
INFLATE_MARGIN_LARGE = 0.10
# --- 小顔前処理・品質ガード（AdaFaceEmbedder 相当） ---
SMALL_FACE_MIN_SIDE = 96
SMALL_FACE_TARGET_LONG_SIDE = 800
MIN_CROP_SIDE = 60


@dataclass
class FaceGate:
    ok: bool
    reason: str = ""
    face_visible: bool = False   # 骨格ルール（鼻+片目が見える / 下向きでない）を通ったか。両目間隔比の判定はこの後
    frontal: bool = False
    face_width: float = 0.0
    eye_ratio: float | None = None
    nose_conf: float | None = None
    l_eye_conf: float | None = None
    r_eye_conf: float | None = None


def evaluate_face_gate(face_box, kpts) -> FaceGate:
    """顔 box と人物の骨格キーポイント(17,3)から、顔として採用できるかを判定する。"""
    box = np.asarray(face_box, np.float32).reshape(-1)[:4]
    face_w = max(1.0, float(box[2] - box[0]))
    k = None if kpts is None else np.asarray(kpts, np.float32)
    if k is None or k.ndim != 2 or k.shape[0] < 5 or k.shape[1] < 3:
        return FaceGate(False, "骨格キーポイントなし", face_width=face_w)
    conf = k[:, 2]
    nose, le, re = float(conf[IDX_NOSE]), float(conf[IDX_L_EYE]), float(conf[IDX_R_EYE])
    g = FaceGate(False, face_width=face_w, nose_conf=nose, l_eye_conf=le, r_eye_conf=re)
    nose_ok, l_ok, r_ok = nose >= KP_CONF_TH, le >= KP_CONF_TH, re >= KP_CONF_TH
    if FACE_GATE_RELAXED:
        face_ok = nose_ok and (l_ok or r_ok)
    else:
        face_ok = nose_ok and l_ok and r_ok and (conf[IDX_L_EAR] >= KP_CONF_TH or conf[IDX_R_EAR] >= KP_CONF_TH)
    if not face_ok:
        g.reason = "鼻または目が見えない"
        return g
    l_ear_down = conf[IDX_L_EAR] >= KP_CONF_TH and l_ok and k[IDX_L_EAR, 1] < k[IDX_L_EYE, 1]
    r_ear_down = conf[IDX_R_EAR] >= KP_CONF_TH and r_ok and k[IDX_R_EAR, 1] < k[IDX_R_EYE, 1]
    if l_ear_down or r_ear_down:
        g.reason = "下向きの顔（耳が目より上）"
        return g
    g.face_visible = True
    if l_ok and r_ok:
        eye_dist = float(np.linalg.norm(k[IDX_L_EYE, :2] - k[IDX_R_EYE, :2]))
        # JSON に記録する精度（小数 4 桁）に丸めてから判定する。レポート時の再判定（report/gate.py）と結果を一致させるため
        # （premises-security との差は丸め誤差の範囲）
        g.eye_ratio = round(eye_dist / face_w, 4)
        if g.eye_ratio < MIN_EYE_RATIO:
            g.reason = "両目の間隔が狭すぎる（横顔）"
            return g
        g.frontal = g.eye_ratio >= FRONTAL_MIN_RATIO
    g.ok = True
    return g


def adaptive_margin(face_w: float) -> float:
    if face_w < FACE_WIDTH_SMALL:
        return INFLATE_MARGIN_SMALL
    if face_w < FACE_WIDTH_MEDIUM:
        return INFLATE_MARGIN_MEDIUM
    return INFLATE_MARGIN_LARGE


def inflate_box(box, frame_hw, margin: float) -> np.ndarray:
    """box を全方向に margin 割合だけ広げ、画像内にクリップする（最低 1px を確保）。"""
    h, w = int(frame_hw[0]), int(frame_hw[1])
    b = np.asarray(box, np.float32).reshape(-1)[:4].copy()
    bw, bh = b[2] - b[0], b[3] - b[1]
    b[0] -= bw * margin
    b[1] -= bh * margin
    b[2] += bw * margin
    b[3] += bh * margin
    b[0] = np.clip(b[0], 0, max(0, w - 1))
    b[1] = np.clip(b[1], 0, max(0, h - 1))
    b[2] = np.clip(b[2], 1, max(1, w))
    b[3] = np.clip(b[3], 1, max(1, h))
    b[2] = max(b[2], b[0] + 1)
    b[3] = max(b[3], b[1] + 1)
    return b


def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    return max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)


def prepare_small_face_roi(frame: np.ndarray, box) -> tuple[np.ndarray, np.ndarray]:
    """短辺が SMALL_FACE_MIN_SIDE 未満の顔は、周辺 ROI を切り出して長辺 SMALL_FACE_TARGET_LONG_SIDE に
    拡大する。戻り値は (切り出し元画像, その画像座標系での顔 box)。大きい顔は入力をそのまま返す。"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, np.asarray(box, np.float32).reshape(-1)[:4])
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    orig = np.array([x1, y1, x2, y2], np.float32)
    min_side = min(bw, bh)
    if min_side >= SMALL_FACE_MIN_SIDE:
        return frame, orig
    scale = SMALL_FACE_MIN_SIDE / max(min_side, 1e-3)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = bw * scale / 2, bh * scale / 2
    rx1, ry1, rx2, ry2 = _clip_box(int(round(cx - hw)), int(round(cy - hh)),
                                   int(round(cx + hw)), int(round(cy + hh)), w, h)
    if rx2 <= rx1 + 1 or ry2 <= ry1 + 1:
        return frame, orig
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return frame, orig
    hr, wr = roi.shape[:2]
    scale_long = SMALL_FACE_TARGET_LONG_SIDE / float(max(hr, wr))
    if abs(scale_long - 1.0) < 1e-3:
        roi_r, sx, sy = roi, 1.0, 1.0
    else:
        nw, nh = max(1, round(wr * scale_long)), max(1, round(hr * scale_long))
        roi_r = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_LINEAR)
        sx, sy = nw / wr, nh / hr
    box_local = np.array([(x1 - rx1) * sx, (y1 - ry1) * sy, (x2 - rx1) * sx, (y2 - ry1) * sy], np.float32)
    return roi_r, box_local


def crop_min_side(box_local) -> float:
    """切り出し box の短辺(px)。品質ガード（MIN_CROP_SIDE）の判定値で、小顔 ROI 拡大後の座標で測る。
    JSON に記録する精度（小数 1 桁）に丸める（レポート時の再判定と結果を一致させるため）。"""
    x1, y1, x2, y2 = map(float, np.asarray(box_local, np.float32).reshape(-1)[:4])
    return round(min(max(1.0, x2 - x1), max(1.0, y2 - y1)), 1)


def crop_guard_reason(min_side: float, min_crop_side: float = MIN_CROP_SIDE) -> str:
    """品質ガードに外れる理由（短辺 < min_crop_side）。通れば空文字。"""
    return f"切り出しが小さい（短辺 {min_side:.0f}px < {min_crop_side:g}px）" if min_side < min_crop_side else ""


def crop_face(frame_local: np.ndarray, box_local) -> np.ndarray | None:
    """box を切り出す（AdaFaceEmbedder._crop_with_margin の margin=0 相当。リサイズはしない）。
    範囲が空なら None。短辺の品質ガードは掛けない（crop_min_side で測り、呼び出し側が判定する）。"""
    h, w = frame_local.shape[:2]
    x1, y1, x2, y2 = map(float, np.asarray(box_local, np.float32).reshape(-1)[:4])
    x1, y1, x2, y2 = _clip_box(int(x1), int(y1), int(x2), int(y2), w, h)
    crop = frame_local[y1:y2 + 1, x1:x2 + 1]
    return crop if crop.size else None


@dataclass
class FaceCrop:
    image: np.ndarray | None   # 切り出し画像（範囲が空なら None）
    inflated: np.ndarray       # 適応マージンで拡張した box（フレーム座標）
    min_side: float            # 切り出し box の短辺(px)。小顔 ROI 拡大後の座標（品質ガードの判定値）

    @property
    def guard_ok(self) -> bool:
        """品質ガード（短辺 >= MIN_CROP_SIDE）。premises-security ではこれに外れると出力しない。"""
        return self.min_side >= MIN_CROP_SIDE


def make_face_crop(frame: np.ndarray, face_box) -> FaceCrop:
    """検出顔 box(フレーム座標) → 適応マージン拡張 → 小顔 ROI 拡大 → 切り出し。品質ガードの判定値も返す。"""
    box = np.asarray(face_box, np.float32).reshape(-1)[:4]
    inflated = inflate_box(box, frame.shape[:2], adaptive_margin(float(box[2] - box[0])))
    frame_local, box_local = prepare_small_face_roi(frame, inflated)
    return FaceCrop(crop_face(frame_local, box_local), inflated, crop_min_side(box_local))


def quality_score(face_conf: float, face_w: float) -> float:
    """品質スコア = 顔検出信頼度 + min(10, 顔幅/10)（premises-security の quality と同じ）。"""
    return float(face_conf) + min(10.0, float(face_w) / 10.0)


def attach_faces(person_boxes, face_boxes, face_scores) -> list[int | None]:
    """各人物に顔を紐付ける。顔 box の中心が人物 box 内にある顔を、その人物の顔とする。
    1つの顔中心が複数の人物 box に含まれる場合は面積が最小の人物に付け、
    1人物に複数の顔が付く場合は検出信頼度が最大の顔を採用する。
    戻り値は人物ごとの顔インデックス（無ければ None）。"""
    n = len(person_boxes)
    out: list[int | None] = [None] * n
    if n == 0 or len(face_boxes) == 0:
        return out
    pb = np.asarray(person_boxes, np.float32).reshape(-1, 4)
    fb = np.asarray(face_boxes, np.float32).reshape(-1, 4)
    areas = (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1])
    best = [-1.0] * n
    for fi, (f, s) in enumerate(zip(fb, face_scores)):
        cx, cy = (f[0] + f[2]) / 2, (f[1] + f[3]) / 2
        inside = np.where((pb[:, 0] <= cx) & (cx <= pb[:, 2]) & (pb[:, 1] <= cy) & (cy <= pb[:, 3]))[0]
        if len(inside) == 0:
            continue
        pi = int(inside[np.argmin(areas[inside])])
        if float(s) > best[pi]:
            best[pi] = float(s)
            out[pi] = fi
    return out
