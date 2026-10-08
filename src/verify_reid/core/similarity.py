"""コサイン類似度・順位付け・追跡ID集約。特徴量は L2 正規化済みを前提とする。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRID_MAX_ITEMS = 500


@dataclass
class Pair:
    q_idx: int
    g_idx: int
    sim: float


def similarity_matrix(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    if len(q) == 0 or len(g) == 0:
        return np.zeros((len(q), len(g)), np.float32)
    return (q @ g.T).astype(np.float32)


def rank_pairs(sim: np.ndarray, limit: int = GRID_MAX_ITEMS) -> list[Pair]:
    """全ペアを類似度降順に並べて上位 limit 件。"""
    if sim.size == 0:
        return []
    flat = sim.ravel()
    k = min(limit, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k] if k < flat.size else np.arange(flat.size)
    idx = idx[np.argsort(-flat[idx], kind="stable")]
    g_n = sim.shape[1]
    return [Pair(int(i // g_n), int(i % g_n), float(flat[i])) for i in idx]


def aggregate_by_track(sim: np.ndarray, track_keys: list, limit: int = GRID_MAX_ITEMS) -> list[Pair]:
    """追跡IDごとに最大類似度の1ペアだけを残して降順に並べる（max集約）。"""
    best: dict = {}
    if sim.size:
        q_best = sim.argmax(axis=0)  # 各比較画像に対する最良クエリ
        for g_idx, key in enumerate(track_keys):
            s = float(sim[q_best[g_idx], g_idx])
            if key not in best or s > best[key].sim:
                best[key] = Pair(int(q_best[g_idx]), g_idx, s)
    return sorted(best.values(), key=lambda p: -p.sim)[:limit]


def competition_ranks(sims: list[float]) -> list[int]:
    """標準競技順位（小数4桁に丸めた同値は同順位）。入力は降順前提。"""
    ranks: list[int] = []
    prev = None
    rank = 0
    for i, s in enumerate(sims, start=1):
        r = round(s, 4)
        if r != prev:
            rank = i
            prev = r
        ranks.append(rank)
    return ranks
