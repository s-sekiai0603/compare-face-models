"""モデル重みの置き場所と、重みが無いときの例外（依存なしの共通モジュール）。

models/ の構成（すべて Git 管理外）:
  detection/   人物検出 (YOLOv8-pose) と顔検出 (YOLOv8-Face) の重み・TensorRT エンジン
  landmark/    顔ランドマーク（2d106det.onnx, det_10g.onnx, retinaface/, mtcnn/, cvlface_DFA_mobilenet/）
  feature/     顔特徴量（adaface/, cvlface/, lafs/, fspfm/）
  pose/        頭部姿勢（hopenet/, 6drepnet/）
評価用の検証セット（insightface 形式の lfw.bin, cfp_fp.bin 等。Git 管理外）は dataset/ に置く。
"""
from __future__ import annotations

from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[4] / "models"
DETECTION_DIR = MODELS_DIR / "detection"
LANDMARK_DIR = MODELS_DIR / "landmark"
FEATURE_DIR = MODELS_DIR / "feature"
POSE_DIR = MODELS_DIR / "pose"
DATASET_DIR = MODELS_DIR.parent / "dataset"  # 検証セット（.bin）。eval コマンドの既定の探索先


class WeightsUnavailable(RuntimeError):
    """重みを取得できない。メッセージにユーザー向けの配置手順を含める。"""
