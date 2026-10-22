"""InsightFace 2d106det.onnx（106 点ランドマーク）→ 5 点 → ArcFace テンプレートへのアフィン変換。

premises-security core/detection/face_align.py の移植。重みは models/landmark/2d106det.onnx（無ければ insightface の
buffalo_l パックから自動取得）。
"""
from __future__ import annotations

import cv2
import numpy as np

from ..common.weights import LANDMARK_DIR
from .base import ARCFACE_TEMPLATE_5PTS, AlignResult, FaceAligner, WeightsUnavailable, register, template_5pts  # noqa: F401
from .insightface_pack import ensure_buffalo_file

# 2d106det の 106 点 → 5 点（InsightFace の公開インデックス）
_LEFT_EYE_INDICES = list(range(33, 43))
_RIGHT_EYE_INDICES = list(range(87, 97))
_NOSE_TIP_INDEX = 86
_LEFT_MOUTH_INDEX = 52
_RIGHT_MOUTH_INDEX = 61
BBOX_MARGIN_RATIO = 0.10  # 5 点がクロップ外に出ることを許容する割合


def extract_5pts_from_106(lmk106: np.ndarray) -> np.ndarray:
    """(106, 2) -> ArcFace 順の (5, 2)。"""
    lmk106 = np.asarray(lmk106, np.float32)
    return np.array([
        lmk106[_LEFT_EYE_INDICES].mean(axis=0), lmk106[_RIGHT_EYE_INDICES].mean(axis=0),
        lmk106[_NOSE_TIP_INDEX], lmk106[_LEFT_MOUTH_INDEX], lmk106[_RIGHT_MOUTH_INDEX],
    ], dtype=np.float32)


def estimate_affine_5pts(src_5pts: np.ndarray, out_size: int) -> np.ndarray | None:
    """5 点（クロップ座標系）を ArcFace テンプレートへ合わせる相似変換 (2, 3)。失敗時は None。"""
    src = np.asarray(src_5pts, np.float32).reshape(5, 2)
    if not np.all(np.isfinite(src)):
        return None
    try:
        m, _ = cv2.estimateAffinePartial2D(src, template_5pts(out_size), method=cv2.LMEDS,
                                           ransacReprojThreshold=3.0, maxIters=2000,
                                           confidence=0.99, refineIters=10)
    except cv2.error:
        return None
    if m is None or m.shape != (2, 3):
        return None
    return m.astype(np.float32)


def transform_points(pts: np.ndarray, m: np.ndarray) -> np.ndarray:
    """(N, 2) の点に 2x3 アフィン行列を適用する。"""
    p = np.asarray(pts, np.float32).reshape(-1, 2)
    return (p @ m[:, :2].T + m[:, 2]).astype(np.float32)


def align_from_5pts(face_bgr: np.ndarray, src_5pts: np.ndarray, out_size: int) -> np.ndarray | None:
    """5 点でアラインした画像のみを返す。"""
    r = align_and_transform(face_bgr, src_5pts, out_size)
    return None if r is None else r.image


def align_and_transform(face_bgr: np.ndarray, src_5pts: np.ndarray, out_size: int) -> AlignResult | None:
    """5 点でアラインし、アライン後画像内の 5 点座標も返す。"""
    m = estimate_affine_5pts(src_5pts, out_size)
    if m is None:
        return None
    img = cv2.warpAffine(face_bgr, m, (out_size, out_size), borderValue=(0, 0, 0))
    return AlignResult(img, transform_points(src_5pts, m))


def points_within(pts: np.ndarray, w: int, h: int, margin_ratio: float = BBOX_MARGIN_RATIO) -> bool:
    """全ての点がクロップ内（幅・高さの margin_ratio だけ外側まで許容）に収まるか。"""
    mx, my = w * margin_ratio, h * margin_ratio
    return bool(np.all((pts[:, 0] >= -mx) & (pts[:, 0] <= w + mx) & (pts[:, 1] >= -my) & (pts[:, 1] <= h + my)))


@register
class InsightFace2d106(FaceAligner):
    name = "InsightFace 2d106det"
    description = "106 点ランドマーク(2d106det.onnx) → 5 点 → ArcFace テンプレートへアライン"
    weights_file = "2d106det.onnx"

    def __init__(self):
        self.session = None
        self.input_name = ""
        self.input_size = 192

    def weights_path(self):
        return LANDMARK_DIR / self.weights_file

    def weights_ready(self) -> bool:
        return self.weights_path().is_file()

    def manual_instructions(self) -> str:
        return (
            f"{self.name} のモデル {self.weights_file} が見つかりません。画面を開いたときに自動取得します（buffalo_l.zip 約 290MB）。\n"
            "手動で用意する場合は https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip を展開し、"
            f"{self.weights_file} を次の場所に配置してください。\n   {self.weights_path()}"
        )

    def download_weights(self, log=print) -> None:
        if not self.weights_ready():
            ensure_buffalo_file(self.weights_file, log)
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())

    def load(self) -> None:
        if self.session is not None:
            return
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())
        import onnxruntime as ort
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(str(self.weights_path()), providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        if isinstance(inp.shape[-1], int):
            self.input_size = int(inp.shape[-1])
        print(f"[推論環境] 顔ランドマーク: ONNX Runtime {ort.__version__} providers={self.session.get_providers()}"
              f" / モデル: {self.weights_path().name} (入力 {self.input_size}x{self.input_size})", flush=True)

    def predict_106(self, face_bgr: np.ndarray) -> np.ndarray | None:
        """BGR 顔画像 -> 106 点 (106, 2)（入力画像の座標系）。失敗時は None。"""
        self.load()
        h, w = face_bgr.shape[:2]
        s = self.input_size
        resized = cv2.resize(face_bgr, (s, s), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32)[np.newaxis, ...]  # [0, 255] のまま
        raw = np.asarray(self.session.run(None, {self.input_name: tensor})[0]).flatten()
        if raw.size < 212:
            return None
        lmk = raw[:212].reshape(106, 2).astype(np.float32)
        lmk = (lmk + 1.0) * (s / 2.0)  # [-1, 1] -> [0, s]
        lmk[:, 0] *= w / float(s)
        lmk[:, 1] *= h / float(s)
        return lmk

    def align_with_landmarks(self, face_bgr: np.ndarray, out_size: int) -> AlignResult | None:
        if face_bgr is None or face_bgr.size == 0:
            return None
        lmk = self.predict_106(face_bgr)
        if lmk is None:
            return None
        pts = extract_5pts_from_106(lmk)
        if not np.all(np.isfinite(pts)):
            return None
        h, w = face_bgr.shape[:2]
        if not points_within(pts, w, h):
            return None  # 5 点がクロップの外（耳だけ等）
        return align_and_transform(face_bgr, pts, out_size)

    def align(self, face_bgr: np.ndarray, out_size: int) -> np.ndarray | None:
        r = self.align_with_landmarks(face_bgr, out_size)
        return None if r is None else r.image
