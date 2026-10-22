"""クエリ画像の時間グループ化と特徴量マージ（追跡ID指定時）。"""
from __future__ import annotations

import numpy as np


def assign_groups(times: list[float], unit_sec: float) -> list[int]:
    """時刻(秒)の早い順に、先頭を起点として unit_sec 秒ごとの区間でグループ化する。

    区間は [t0 + k*unit, t0 + (k+1)*unit)。画像が1枚も無い区間は飛ばし、
    グループIDは画像のある区間に対して 1 から連番で採番する。
    戻り値は入力と同じ順序の、各画像のグループID。
    """
    if not times:
        return []
    if unit_sec <= 0:
        raise ValueError("マージ単位秒数は 0 より大きい値を指定してください")
    t0 = min(times)
    window = [int((t - t0) // unit_sec) for t in times]
    ids = {w: i + 1 for i, w in enumerate(sorted(set(window)))}
    return [ids[w] for w in window]


def merge_embeddings(emb: np.ndarray, group_ids: list[int]) -> np.ndarray:
    """グループごとに特徴量を平均し、L2 正規化する。行はグループID昇順（ID1が0行目）。"""
    n_groups = max(group_ids) if group_ids else 0
    out = np.zeros((n_groups, emb.shape[1] if emb.ndim == 2 else 0), np.float32)
    for gid in range(1, n_groups + 1):
        rows = [i for i, g in enumerate(group_ids) if g == gid]
        m = emb[rows].mean(axis=0)
        norm = np.linalg.norm(m)
        out[gid - 1] = m / norm if norm > 0 else m
    return out
