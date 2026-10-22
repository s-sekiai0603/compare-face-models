"""頭部姿勢モデルのレジストリ。レポート画面の「頭部姿勢」選択肢（名前 -> 推定器を作る関数。None は表示しない）。
推定器は HeadPoseEstimator（HopeNet）と同じインターフェース（name / weights_path / weights_ready / manual_instructions /
download_weights / load / angles）を持つ。"""
from .hopenet import VARIANTS, HeadPoseEstimator, decode_angles  # noqa: F401
from .sixdrepnet import SixDRepNetEstimator  # noqa: F401

POSE_OPTIONS = {
    "なし": None,
    "HopeNet (robust)": lambda: HeadPoseEstimator("robust_alpha1"),
    "HopeNet (alpha2)": lambda: HeadPoseEstimator("alpha2"),
    "6DRepNet": lambda: SixDRepNetEstimator(),
}


def get_pose_estimator(name: str):
    if name not in POSE_OPTIONS:
        raise KeyError(f"未登録の頭部姿勢モデル: {name}（登録済み: {', '.join(POSE_OPTIONS)}）")
    factory = POSE_OPTIONS[name]
    return None if factory is None else factory()
