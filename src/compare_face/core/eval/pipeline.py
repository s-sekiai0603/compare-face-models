"""検証セット（.bin）で「顔特徴量モデル × 顔ランドマークモデル」の 1:1 照合精度を測る（CLI の eval コマンド本体）。

手順（データセットごと）:
  1. .bin を読み込む（アライン済み 112x112 画像とペアの同一人物フラグ）
  2. ランドマークモデルで各画像をアラインし直す（「なし」はリサイズのみ。.bin は整列済みなので通常は「なし」で評価する。
     検出できなかった画像はレポートと同様にリサイズで代替し、件数を記録する）
  3. 各特徴量モデルで元画像と左右反転画像の特徴量を取り、
     - 融合あり: 2 つを足して L2 正規化（insightface / AdaFace の flip test。論文の公表値と同じ条件）
     - 融合なし: 元画像だけ（当ツールのレポートと同じ条件。モデル内部の TTA はそのまま）
     の両方で 10 分割交差検証の正解率を出す。検証セット側の整列失敗（平坦画像）を含むペアを除いた精度も併記する
     （.bin の版によって横顔の整列失敗が含まれ、どのモデルでも照合できないため公表値との差の切り分けに使う）
  4. 行ごとに CSV（UTF-8 BOM 付き、Excel でそのまま開ける）へ保存し直す（途中で止めても結果が残る）
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..common.face_template import mirror_5pts
from ..common.weights import DATASET_DIR
from ..features import get_extractor
from ..landmarks import get_aligner
from .published import published_accuracy
from .verification import N_FOLDS, VerificationSet, evaluate_pairs, fuse_flip, load_bin, select_pairs

CHUNK_SIZE = 256


@dataclass
class AlignedSet:
    faces: list[np.ndarray]                # アライン後（またはリサイズ）の画像
    landmarks: list[np.ndarray | None]     # アライン後画像内の 5 点（無ければ None）
    align_failed: int                      # 検出できずリサイズで代替した件数


@dataclass
class EvalRow:
    dataset: str
    model: str
    landmark: str
    n_pairs: int
    accuracy: float               # flip 融合あり（%）
    accuracy_std: float           # 同、10 分割の標準偏差（%）
    accuracy_single: float        # flip 融合なし（%）
    accuracy_valid: float         # flip 融合あり、整列失敗（平坦画像）を含むペアを除外（%）。除外対象が無ければ accuracy と同じ
    flat_images: int              # 平坦画像の枚数（ペアに並んだ延べ枚数）
    flat_pairs: int               # 平坦画像を含むペア数
    cos_threshold: float          # 融合ありで選ばれたしきい値（コサイン類似度）
    align_failed: int
    landmark_missing: int | None  # needs_landmarks のモデルのみ
    published: float | None       # 公表値（%）
    seconds: float


CSV_COLUMNS = [
    ("dataset", "データセット"), ("model", "顔特徴量モデル"), ("landmark", "顔ランドマークモデル"), ("n_pairs", "ペア数"),
    ("accuracy", "精度 flip融合あり(%)"), ("accuracy_std", "標準偏差(%)"), ("accuracy_single", "精度 flip融合なし(%)"),
    ("accuracy_valid", "精度 整列失敗ペア除外(%)"), ("flat_images", "整列失敗画像(枚)"), ("flat_pairs", "整列失敗ペア"),
    ("cos_threshold", "しきい値(cos)"), ("align_failed", "アライン失敗"), ("landmark_missing", "ランドマーク未取得"),
    ("published", "公表値(%)"), ("seconds", "処理時間(秒)"),
]


def default_bins() -> list[Path]:
    """dataset/ 配下の .bin（名前順）。"""
    return sorted(DATASET_DIR.glob("*.bin")) if DATASET_DIR.is_dir() else []


def default_csv_path() -> Path:
    return DATASET_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}.csv"


def align_set(aligner, images: list[np.ndarray], size: int,
              progress: Callable[[int, int], None] | None = None) -> AlignedSet:
    faces, landmarks, failed = [], [], 0
    for i, img in enumerate(images):
        res = aligner.align_with_landmarks(img, size)
        if res is None:
            failed += 1
            faces.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC))
            landmarks.append(None)
        else:
            faces.append(res.image)
            landmarks.append(res.landmarks)
        if progress and (i + 1) % 1000 == 0:
            progress(i + 1, len(images))
    return AlignedSet(faces, landmarks, failed)


def embed_with_flip(extractor, aligned: AlignedSet) -> tuple[np.ndarray, np.ndarray]:
    """元画像と左右反転画像の特徴量（各 L2 正規化済み）。反転画像には左右反転した 5 点を渡す。"""
    out_a, out_b = [], []
    for i in range(0, len(aligned.faces), CHUNK_SIZE):
        faces = aligned.faces[i:i + CHUNK_SIZE]
        lms = aligned.landmarks[i:i + CHUNK_SIZE]
        flipped = [cv2.flip(f, 1) for f in faces]
        if extractor.needs_landmarks:
            lms_m = [None if lm is None else mirror_5pts(lm, f.shape[1]) for lm, f in zip(lms, faces)]
            out_a.append(extractor.embed_batch(faces, landmarks=lms))
            out_b.append(extractor.embed_batch(flipped, landmarks=lms_m))
        else:
            out_a.append(extractor.embed_batch(faces))
            out_b.append(extractor.embed_batch(flipped))
    return np.concatenate(out_a), np.concatenate(out_b)


def evaluate_set(extractor, aligned: AlignedSet, ds: VerificationSet, landmark_name: str) -> EvalRow:
    t0 = time.perf_counter()
    emb, emb_flip = embed_with_flip(extractor, aligned)
    fused_emb = fuse_flip(emb, emb_flip)
    fused = evaluate_pairs(fused_emb, ds.issame)
    single = evaluate_pairs(emb, ds.issame)
    keep = ds.valid_pair_mask()
    n_flat_pairs = int((~keep).sum())
    if n_flat_pairs and keep.sum() >= N_FOLDS:
        valid = evaluate_pairs(select_pairs(fused_emb, keep), ds.issame[keep]).accuracy * 100
    else:
        valid = fused.accuracy * 100
    missing = sum(1 for lm in aligned.landmarks if lm is None) if extractor.needs_landmarks else None
    return EvalRow(ds.name, extractor.name, landmark_name, ds.n_pairs, fused.accuracy * 100, fused.accuracy_std * 100,
                   single.accuracy * 100, valid, int(ds.flat_mask().sum()), n_flat_pairs, fused.cos_threshold,
                   aligned.align_failed, missing, published_accuracy(extractor.name, ds.name), time.perf_counter() - t0)


def format_row(row: EvalRow) -> str:
    pub = "-" if row.published is None else f"{row.published:.2f}"
    extra = f" / アライン失敗 {row.align_failed}" if row.align_failed else ""
    if row.landmark_missing:
        extra += f" / ランドマーク未取得 {row.landmark_missing}"
    flat = f" / 整列失敗ペア {row.flat_pairs} 件除外で {row.accuracy_valid:.2f}%" if row.flat_pairs else ""
    return (f"[{row.dataset}] {row.model} × {row.landmark}: 精度 {row.accuracy:.2f}±{row.accuracy_std:.2f}%"
            f"（flip 融合なし {row.accuracy_single:.2f}% / 公表値 {pub} / しきい値 cos {row.cos_threshold:.3f}）"
            f"{flat}{extra} / {row.seconds:.0f} 秒")


def _cell(value) -> str | int:
    if value is None:
        return ""
    return f"{value:.3f}" if isinstance(value, float) else value


def write_csv(rows: list[EvalRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([label for _, label in CSV_COLUMNS])
        for r in rows:
            w.writerow([_cell(getattr(r, key)) for key, _ in CSV_COLUMNS])


def _ensure_ready(obj, echo) -> None:
    if not obj.weights_ready():
        obj.download_weights(log=echo)
    obj.load()


def run_eval(bin_paths: list[Path], model_names: list[str], landmark_names: list[str], out_csv: Path,
             limit_pairs: int = 0, echo: Callable[[str], None] = print) -> list[EvalRow]:
    """全組み合わせを評価し、結果行を返す（CSV は 1 行終わるごとに保存する）。"""
    if not bin_paths:
        raise ValueError(f"検証セット（.bin）がありません。{DATASET_DIR} に置くか --bin で指定してください")
    extractors = {}  # モデルは一度だけ読み込む
    rows: list[EvalRow] = []
    for bin_path in bin_paths:
        t0 = time.perf_counter()
        ds = load_bin(bin_path, limit_pairs)
        echo(f"[{ds.name}] {bin_path}: {ds.n_pairs} ペア / {len(ds.images)} 枚を読込（{time.perf_counter() - t0:.1f} 秒）"
             f" / 整列失敗（平坦）画像 {int(ds.flat_mask().sum())} 枚, それを含むペア {int((~ds.valid_pair_mask()).sum())} 件")
        for landmark_name in landmark_names:
            aligner = get_aligner(landmark_name)
            _ensure_ready(aligner, echo)
            aligned_by_size: dict[int, AlignedSet] = {}
            for model_name in model_names:
                if model_name not in extractors:
                    extractors[model_name] = get_extractor(model_name)
                    _ensure_ready(extractors[model_name], echo)
                extractor = extractors[model_name]
                if extractor.input_size not in aligned_by_size:
                    t1 = time.perf_counter()
                    aligned_by_size[extractor.input_size] = align_set(
                        aligner, ds.images, extractor.input_size,
                        lambda i, n: echo(f"[{ds.name}] {landmark_name}: アライン中 {i}/{n}"))
                    a = aligned_by_size[extractor.input_size]
                    echo(f"[{ds.name}] {landmark_name}: アライン完了（失敗 {a.align_failed} 件 / "
                         f"{time.perf_counter() - t1:.0f} 秒）")
                row = evaluate_set(extractor, aligned_by_size[extractor.input_size], ds, landmark_name)
                rows.append(row)
                echo(format_row(row))
                write_csv(rows, out_csv)
    return rows
