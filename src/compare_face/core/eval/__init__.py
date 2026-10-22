"""検証セット（insightface 形式 .bin）による 1:1 照合精度の評価（CLI の eval コマンド）。"""
from .pipeline import EvalRow, run_eval  # noqa: F401
from .verification import VerificationResult, VerificationSet, evaluate_pairs, fuse_flip, load_bin  # noqa: F401
