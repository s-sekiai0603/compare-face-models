"""insightface 形式の検証セット（lfw.bin, cfp_fp.bin 等）の読込と、10 分割交差検証による 1:1 照合精度。

.bin は pickle で (画像バイト列のリスト, 同一人物フラグのリスト) を持つ。画像は 2 枚で 1 ペア（偶数番目と奇数番目）で、
112x112 の ArcFace テンプレートでアライン済み（insightface の Dataset Zoo が配布するもの）。
精度の計算手順は insightface の recognition/arcface_torch/eval/verification.py（calculate_roc）と同じ:
  - 特徴量を L2 正規化し、ペアの二乗ユークリッド距離（= 2 - 2cos）を取る
  - しきい値 0.00〜3.99（0.01 刻み）、10 分割（シャッフルなし）。9 分割で正解率が最大のしきい値を選び、
    残り 1 分割の正解率を測る。10 回の平均と標準偏差を報告する（論文の LFW / CFP-FP 等の数値と同じ定義）
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

THRESHOLDS = np.arange(0.0, 4.0, 0.01, dtype=np.float64)  # 二乗ユークリッド距離のしきい値（insightface と同じ範囲・刻み）
N_FOLDS = 10
# 検証セット側の整列失敗（横顔のランドマーク誤検出で肌や背景だけが切り出された画像）の判定。
# 目〜口が入る中央領域（縦 31〜89%、横 22〜80%）のグレースケール標準偏差がこの値未満なら「平坦」とみなす。
# 顔が写っていればまつ毛・鼻孔・口で 20 以上になる。
FLAT_STD_THRESHOLD = 15.0
FLAT_ROI = (0.31, 0.89, 0.22, 0.80)  # y0, y1, x0, x1（割合）

# .bin の pickle に含まれてよいオブジェクト（numpy 配列のみ。任意コード実行を防ぐ）
_ALLOWED_GLOBALS = {
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("_codecs", "encode"),  # protocol 2 の bytes（配列データ）の復元に使われる
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"検証セットに想定外のオブジェクトが含まれています: {module}.{name}")


@dataclass
class VerificationSet:
    name: str                  # ファイル名の stem（lfw, cfp_fp, ...）
    images: list[np.ndarray]   # BGR 画像。[2i] と [2i+1] が i 番目のペア
    issame: np.ndarray         # (ペア数,) bool。同一人物なら True

    @property
    def n_pairs(self) -> int:
        return len(self.issame)

    def flat_mask(self) -> np.ndarray:
        """(枚数,) bool。True は整列失敗（平坦）とみなす画像（結果はキャッシュ）。"""
        if getattr(self, "_flat", None) is None:
            self._flat = flat_image_mask(self.images)
        return self._flat

    def valid_pair_mask(self) -> np.ndarray:
        """(ペア数,) bool。両方の画像が平坦でないペア。"""
        flat = self.flat_mask()
        return ~(flat[0::2] | flat[1::2])


def decode_image(buf) -> np.ndarray:
    """JPEG 等のバイト列（bytes または uint8 配列）-> BGR 画像。"""
    raw = buf if isinstance(buf, (bytes, bytearray)) else np.asarray(buf, np.uint8).tobytes()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("検証セットの画像をデコードできません")
    return img


def load_bin(path: Path, limit_pairs: int = 0) -> VerificationSet:
    """insightface 形式の .bin を読み込む。limit_pairs > 0 なら等間隔に選んだそのペア数だけ使う（動作確認用）。"""
    path = Path(path)
    with open(path, "rb") as f:
        bins, issame = _RestrictedUnpickler(f, encoding="bytes").load()
    issame = np.asarray(list(issame), bool)
    if len(bins) != 2 * len(issame):
        raise ValueError(f"{path.name}: 画像数 {len(bins)} がペア数 {len(issame)} の 2 倍ではありません")
    if 0 < limit_pairs < len(issame):
        # 先頭から取ると同一人物ペアだけになる（.bin は分割ごとに同一→別人の順）ので、全体から等間隔に選ぶ
        keep = np.unique(np.linspace(0, len(issame) - 1, limit_pairs).round().astype(int))
        issame = issame[keep]
        bins = [bins[2 * i + j] for i in keep for j in (0, 1)]
    return VerificationSet(path.stem, [decode_image(b) for b in bins], issame)


def flat_image_mask(images: list[np.ndarray], threshold: float = FLAT_STD_THRESHOLD) -> np.ndarray:
    """顔の中央領域がほぼ一色の画像（検証セット側の整列失敗）を True にする。"""
    out = np.zeros(len(images), bool)
    for i, im in enumerate(images):
        h, w = im.shape[:2]
        roi = im[int(h * FLAT_ROI[0]):int(h * FLAT_ROI[1]), int(w * FLAT_ROI[2]):int(w * FLAT_ROI[3])]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        out[i] = float(gray.std()) < threshold
    return out


def select_pairs(emb: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """ペア順の特徴量 (2N, D) から keep (N,) が True のペアだけを取り出す。"""
    return emb.reshape(-1, 2, emb.shape[1])[np.asarray(keep, bool)].reshape(-1, emb.shape[1])


def normalize_rows(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, np.float64)
    return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)


def pair_distances(emb: np.ndarray) -> np.ndarray:
    """(2N, D) の特徴量 -> (N,) ペアごとの二乗ユークリッド距離（L2 正規化後。= 2 - 2cos）。"""
    e = normalize_rows(emb)
    return np.sum(np.square(e[0::2] - e[1::2]), axis=1)


def k_fold_indices(n: int, k: int) -> list[np.ndarray]:
    """sklearn の KFold(shuffle=False) と同じ分割（先頭 n % k 個の分割が 1 件多い）。"""
    return np.array_split(np.arange(n), k)


@dataclass
class VerificationResult:
    accuracy: float            # 10 分割の平均正解率（0〜1）
    accuracy_std: float        # 分割間の標準偏差
    threshold: float           # 選ばれたしきい値（二乗ユークリッド距離）の平均
    fold_accuracies: list[float]

    @property
    def cos_threshold(self) -> float:
        """同じしきい値をコサイン類似度で表したもの（sim >= この値なら同一人物）。"""
        return 1.0 - self.threshold / 2.0


def evaluate_pairs(emb: np.ndarray, issame: np.ndarray, n_folds: int = N_FOLDS,
                   thresholds: np.ndarray = THRESHOLDS) -> VerificationResult:
    """ペア順に並んだ特徴量 (2N, D) と同一人物フラグ (N,) から 10 分割交差検証の正解率を求める。"""
    issame = np.asarray(issame, bool)
    dist = pair_distances(emb)
    if dist.shape != issame.shape:
        raise ValueError(f"特徴量 {len(emb)} 件に対しペア数 {len(issame)} が合いません")
    if len(issame) < n_folds:
        raise ValueError(f"ペア数 {len(issame)} が分割数 {n_folds} より少ないため評価できません")
    folds = k_fold_indices(len(issame), n_folds)
    accs, ths = [], []
    for i, test in enumerate(folds):
        train = np.concatenate([f for j, f in enumerate(folds) if j != i])
        train_acc = np.mean((dist[train][None, :] < thresholds[:, None]) == issame[train][None, :], axis=1)
        best = float(thresholds[int(np.argmax(train_acc))])  # 同率なら小さいしきい値（insightface の argmax と同じ）
        accs.append(float(np.mean((dist[test] < best) == issame[test])))
        ths.append(best)
    return VerificationResult(float(np.mean(accs)), float(np.std(accs)), float(np.mean(ths)), accs)


def fuse_flip(emb: np.ndarray, emb_flipped: np.ndarray) -> np.ndarray:
    """元画像と左右反転画像の特徴量を足して L2 正規化（insightface / AdaFace の flip test と同じ融合）。"""
    return normalize_rows(np.asarray(emb, np.float64) + np.asarray(emb_flipped, np.float64)).astype(np.float32)
