"""FSPFM（特徴空間での姿勢正面化, ICIP 2025, EPFL）による顔特徴量抽出。

元コード・配布: https://github.com/nikolaystanishev/feature-space-pose-frontalization（重みは Google Drive）。
  入力: ArcFace テンプレートでアラインした 112x112 RGB、(x/255 - 0.5)/0.5。
  姿勢: 同じアライン済み画像を 224x224 にして HopeNet（300W-LP, alpha 2）に入れ、分類層手前の 2048 次元を使う
        （元コードの pose_estimation.py と同じ。論文の説明と異なり、オイラー角そのものではない）。
  出力: FSPFM 出力 512 次元を L2 正規化。TTA は元リポジトリの評価既定（add_flip=False）に合わせて無し。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..pose.hopenet import HeadPoseEstimator
from ..common.weights import FEATURE_DIR
from .base import FaceFeatureExtractor, WeightsUnavailable, register

FSPFM_DIR = FEATURE_DIR / "fspfm"
INPUT_SIZE = 112
EMBED_BATCH = 64
POSE_VARIANT = "alpha2"  # 元コードが使う HopeNet の重み
REPO_URL = "https://github.com/nikolaystanishev/feature-space-pose-frontalization"
MIN_FILE_SIZE = 500_000_000  # 配布 .pt は 588MB（stage two）/ 761MB（stage one。optimizer 込み）


def preprocess(face_bgr: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """BGR 顔画像 -> (3, size, size) float32（RGB, [-1, 1]）。"""
    if face_bgr.shape[0] != size or face_bgr.shape[1] != size:
        face_bgr = cv2.resize(face_bgr, (size, size), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - 0.5) / 0.5).transpose(2, 0, 1)


class _FSPFMBase(FaceFeatureExtractor):
    """FSPFM 系の共通実装。派生クラスは重みファイル・Google Drive ID・正面化の有無を定義する。"""

    weights_file = ""
    gdrive_id = ""
    weights_env = ""
    use_frontalization = True
    input_size = INPUT_SIZE
    embed_dim = 512
    needs_landmarks = False

    def __init__(self):
        self.backbone = None
        self.frontalization = None
        self.pose = HeadPoseEstimator(POSE_VARIANT) if self.use_frontalization else None
        self.device = "cpu"

    def weights_path(self) -> Path:
        env = os.environ.get(self.weights_env) if self.weights_env else None
        return Path(env) if env else FSPFM_DIR / self.weights_file

    def weights_ready(self) -> bool:
        return self.weights_path().is_file() and (self.pose is None or self.pose.weights_ready())

    def manual_instructions(self) -> str:
        msg = (
            f"{self.name} の重み {self.weights_file} が見つかりません。画面を開いたときに Google Drive から自動取得します"
            "（約 0.6〜0.8GB、ネットワーク接続が必要。Google Drive の制限で失敗することがあります）。\n"
            f"手動で用意する場合は {REPO_URL} の README にあるチェックポイント"
            f"（https://drive.google.com/file/d/{self.gdrive_id}/view）をダウンロードし、次の場所に配置してください。\n"
            f"   {self.weights_path()}"
        )
        if self.pose is not None and not self.pose.weights_ready():
            msg += "\n\n" + self.pose.manual_instructions()
        return msg

    def download_weights(self, log: Callable[[str], None] = print) -> None:
        dst = self.weights_path()
        if not dst.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            log(f"{self.name} の重みを Google Drive からダウンロードしています: {dst.name}")
            try:
                import gdown
                tmp = dst.with_suffix(dst.suffix + ".part")
                out = gdown.download(id=self.gdrive_id, output=str(tmp), quiet=True)
                if not out or not tmp.is_file() or tmp.stat().st_size < MIN_FILE_SIZE:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError("ダウンロード結果が不正です（サイズ不足）")
                tmp.replace(dst)
            except Exception as e:
                raise WeightsUnavailable(f"{self.manual_instructions()}\n\n（自動取得エラー: {type(e).__name__}: {e}）") from e
        if self.pose is not None:
            self.pose.download_weights(log)

    def load(self) -> None:
        if self.backbone is not None:
            return
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())
        import torch
        from .fspfm_net import FeatureSpacePoseFrontalizationModule, ResnetIR, load_fspfm_state_dicts
        sd_backbone, sd_front = load_fspfm_state_dicts(str(self.weights_path()))
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        backbone = ResnetIR(depth=50, dropout_ratio=0.4, feature_dim=512, out_h=7, out_w=7)
        backbone.load_state_dict(sd_backbone, strict=True)
        self.backbone = backbone.eval().to(self.device)
        if self.use_frontalization:
            front = FeatureSpacePoseFrontalizationModule(2048, 512)
            front.load_state_dict(sd_front, strict=True)
            self.frontalization = front.eval().to(self.device)
            self.pose.load()
        dev = f"GPU ({self.device}, {torch.cuda.get_device_name(0)})" if self.device != "cpu" else "CPU"
        print(f"[推論環境] 顔特徴量抽出: デバイス: {dev} / フレームワーク: PyTorch (FP32) / torch {torch.__version__}"
              f" / モデル: {self.name} / 重み: {self.weights_path().name}"
              f" / 姿勢正面化: {'あり (HopeNet ' + POSE_VARIANT + ')' if self.use_frontalization else 'なし'}", flush=True)

    def embed_batch(self, faces_bgr: list[np.ndarray],
                    landmarks: list[np.ndarray | None] | None = None) -> np.ndarray:
        import torch
        self.load()  # landmarks は使わない
        feats = []
        with torch.no_grad():
            for i in range(0, len(faces_bgr), EMBED_BATCH):
                chunk = faces_bgr[i:i + EMBED_BATCH]
                x = torch.from_numpy(np.stack([preprocess(f, self.input_size) for f in chunk])).to(self.device)
                f = self.backbone(x)
                if self.use_frontalization:
                    pose = torch.from_numpy(self.pose.embed(chunk)).to(self.device)
                    f = self.frontalization(f, pose)
                f = torch.nn.functional.normalize(f.float(), dim=1)
                feats.append(f.cpu().numpy().astype(np.float32))
        return np.concatenate(feats) if feats else np.zeros((0, self.embed_dim), np.float32)


@register
class FSPFMFineTuned(_FSPFMBase):
    name = "FSPFM ResNet50 (正面化あり, 微調整済)"
    description = ("FSPFM: ResNet50-IR（MS1M）+ 特徴空間の姿勢正面化モジュール（HopeNet の姿勢埋め込みを入力）。"
                   "stage two（正面/横顔ペアで微調整済）。CPLFW 93.85。初回は Google Drive から自動取得（約 590MB）")
    weights_file = "fspfm_stage2.pt"
    gdrive_id = "10QcM-HHBlmArC6S-SJmBa6M9V5QqVCcV"
    weights_env = "COMPARE_FACE_FSPFM_WEIGHTS"


@register
class FSPFMPreTrained(_FSPFMBase):
    name = "FSPFM ResNet50 (正面化あり, 事前学習のみ)"
    description = ("FSPFM の stage one（MS1M + MS1M-Pose-Sim で backbone と正面化モジュールを同時学習）。CPLFW 93.65。"
                   "初回は Google Drive から自動取得（約 760MB）")
    weights_file = "fspfm_stage1.pt"
    gdrive_id = "1gOui-e9vyH9ijPx-Ua5R3IO92t8wP2XE"
    weights_env = "COMPARE_FACE_FSPFM_STAGE1_WEIGHTS"
    hidden = True   # 微調整済と比べて精度が低いので画面には出さない（CLI の --model では使える）


# 注意: backbone の出力だけを使う「正面化なし」は登録していない。FSPFM は backbone と正面化モジュールを
# 一体で学習しており（ArcFace ヘッドは正面化モジュールの出力に付く）、backbone 単体の出力は識別用に
# 学習されていないため比較対象として意味がない（実測: 本人 0.73 に対し他人の最大 0.63 と分離が悪い）。
