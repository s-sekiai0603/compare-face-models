"""CVLFace 付属の Differentiable Face Aligner（DFA, RetinaFace-MobileNet0.25）。

minchul/cvlface_DFA_mobilenet を models/landmark/cvlface_DFA_mobilenet/ に取得し、transformers を使わず直接読み込む。
入力の顔クロップから 5 点を検出し、ArcFace テンプレートへ相似変換した 112x112 画像と、
その画像内の 5 点座標を返す。KP-RPE モデルの学習時と同じアライナなので、KP-RPE のキーポイント源として推奨。
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from ..common.cvlface_repo import (
    describe_device, download_repo, import_package, is_downloaded, load_conf, local_dir,
    manual_instructions, torch_device,
)
from ..common.weights import LANDMARK_DIR
from .base import AlignResult, FaceAligner, WeightsUnavailable, register

REPO_ID = "minchul/cvlface_DFA_mobilenet"
PACKAGE = "aligners"
DFA_OUT_SIZE = 112   # DFA の出力サイズ（model.yaml の output_size）
# 顔検出スコアの下限。監視映像の小さい顔（15〜40px）では正しく検出できていても 0.1〜0.4 程度に
# 出る（実測: 51枚で中央値 0.41、最小 0.06）ため、既定ではスコアで棄却せず、
# ランドマークの幾何的な妥当性（landmarks_plausible）だけで失敗を判定する。
MIN_SCORE = 0.0
PLAUSIBLE_MARGIN = 0.25  # ランドマークが画像外へ出てよい割合（画像サイズ比）


def landmarks_plausible(pts: np.ndarray, size: int, margin: float = PLAUSIBLE_MARGIN) -> bool:
    """5 点が顔として妥当な配置か（有限 / 画像の近傍にある / 左目が右目より左 / 目が口より上）。"""
    p = np.asarray(pts, np.float32).reshape(-1, 2)
    if p.shape[0] != 5 or not np.all(np.isfinite(p)):
        return False
    lo, hi = -margin * size, (1.0 + margin) * size
    if not np.all((p >= lo) & (p <= hi)):
        return False
    l_eye, r_eye, nose, l_mouth, r_mouth = p
    return bool(l_eye[0] < r_eye[0] and l_mouth[0] < r_mouth[0]
                and max(l_eye[1], r_eye[1]) < min(l_mouth[1], r_mouth[1])
                and max(l_eye[1], r_eye[1]) < nose[1] < min(l_mouth[1], r_mouth[1]) + 0.1 * size)


def to_input_tensor(torch, face_bgr: np.ndarray):
    """BGR uint8 -> (1, 3, H, W) RGB float32 [-1, 1]。"""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - 0.5) / 0.5).transpose(2, 0, 1)).unsqueeze(0)


def to_bgr_uint8(aligned_chw: np.ndarray) -> np.ndarray:
    """(3, H, W) RGB [-1, 1] -> (H, W, 3) BGR uint8。"""
    rgb = np.clip((aligned_chw.transpose(1, 2, 0) + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@register
class CVLFaceDFA(FaceAligner):
    name = "CVLFace DFA"
    description = ("CVLFace 付属の Differentiable Face Aligner（RetinaFace-MobileNet で 5 点検出 → ArcFace テンプレートへ"
                   "アライン）。KP-RPE の学習時と同じアライナ。初回は Hugging Face から自動取得")

    def __init__(self):
        self.model = None
        self.device = "cpu"

    def weights_ready(self) -> bool:
        return is_downloaded(REPO_ID, PACKAGE, LANDMARK_DIR)

    def manual_instructions(self) -> str:
        return manual_instructions(self.name, REPO_ID, LANDMARK_DIR)

    def download_weights(self, log: Callable[[str], None] = print) -> None:
        if not self.weights_ready():
            download_repo(REPO_ID, log, LANDMARK_DIR)
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())
        import torch
        d = local_dir(REPO_ID, LANDMARK_DIR)
        pkg = import_package(d / PACKAGE, f"cvlface_{d.name}_{PACKAGE}")
        conf = load_conf(d / "pretrained_model" / "model.yaml")
        conf.start_from = ""  # 学習用の初期重みパスは使わない（model.pt を読む）
        conf.freeze = True
        model = pkg.get_aligner(conf)
        model.load_state_dict_from_path(str(d / "pretrained_model" / "model.pt"))
        self.device = torch_device()
        self.model = model.eval().to(self.device)
        print(f"[推論環境] 顔ランドマーク: デバイス: {describe_device()} / フレームワーク: PyTorch"
              f" / torch {torch.__version__} / モデル: {self.name} ({REPO_ID})", flush=True)

    def run(self, face_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """(アライン済み 112x112 BGR, 5 点ピクセル座標 (5,2), 顔検出スコア)。"""
        import torch
        self.load()
        with torch.no_grad():
            x = to_input_tensor(torch, face_bgr).to(self.device)
            aligned_x, _orig_ldmks, aligned_ldmks, score, _thetas, _bbox = self.model(x)
        img = to_bgr_uint8(aligned_x[0].float().cpu().numpy())
        pts = aligned_ldmks[0].float().cpu().numpy().reshape(5, 2) * np.array([img.shape[1], img.shape[0]], np.float32)
        return img, pts.astype(np.float32), float(score.flatten()[0])

    def align_with_landmarks(self, face_bgr: np.ndarray, out_size: int) -> AlignResult | None:
        if face_bgr is None or face_bgr.size == 0:
            return None
        img, pts, score = self.run(face_bgr)
        if score < MIN_SCORE or not landmarks_plausible(pts, img.shape[0]):
            return None
        if out_size != img.shape[0]:
            scale = float(out_size) / img.shape[0]
            img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
            pts = pts * scale
        return AlignResult(img, pts)

    def align(self, face_bgr: np.ndarray, out_size: int) -> np.ndarray | None:
        r = self.align_with_landmarks(face_bgr, out_size)
        return None if r is None else r.image
