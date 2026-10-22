"""比較対象の顔特徴量キャッシュ。

同じ比較対象フォルダに対して、同じ顔特徴量モデル × 顔ランドマークモデルでクエリだけ変えてレポートを出すとき、
最も時間のかかる「比較対象のアライン + 特徴量抽出」を省く。保存先は比較対象フォルダ直下の
.embedding_cache/<モデル名>_<ランドマーク名>_<キー>.npz。画像ごとに比較対象フォルダからの相対パス・ファイルサイズ・
更新時刻を記録し、3 つとも一致した画像だけ再利用する（追加・差し替えされた画像は計算し直して追記）。
特徴量の値には影響しない（キャッシュ無しと同じ結果になる）。前処理やアラインの実装を変えて特徴量が変わるときは
CACHE_SCHEMA を上げて古いキャッシュを無効化すること。
variant は同じモデル × ランドマークでも特徴量が変わる前処理の区別（"clahe" = アライン後に CLAHE を掛ける）。
variant ごとに別ファイル・別キーになる。
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

CACHE_DIR_NAME = ".embedding_cache"
CACHE_SCHEMA = "1"


def _safe(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")[:40] or "x"


class EmbeddingCache:
    """モデル × ランドマーク 1 組ぶんのキャッシュ。load() -> get()/put() -> save() の順に使う。"""

    def __init__(self, gallery_dir: Path, model_name: str, landmark_name: str, variant: str = ""):
        self.gallery_dir = Path(gallery_dir)
        self.variant = variant
        self.key = (f"{model_name}|{landmark_name}|{variant}|schema={CACHE_SCHEMA}" if variant
                    else f"{model_name}|{landmark_name}|schema={CACHE_SCHEMA}")
        digest = hashlib.sha1(self.key.encode("utf-8")).hexdigest()[:12]
        stem = f"{_safe(model_name)}_{_safe(landmark_name)}" + (f"_{_safe(variant)}" if variant else "")
        self.path = self.gallery_dir / CACHE_DIR_NAME / f"{stem}_{digest}.npz"
        self._entries: dict[str, tuple[int, float, np.ndarray, bool, bool]] = {}  # rel -> (size, mtime, emb, failed, missing)
        self.hits = 0
        self.misses = 0
        self._dirty = False

    def _rel(self, face_path: Path) -> str:
        try:
            return Path(face_path).resolve().relative_to(self.gallery_dir.resolve()).as_posix()
        except ValueError:
            return Path(face_path).resolve().as_posix()

    def load(self) -> int:
        """保存済みキャッシュを読み込み、件数を返す（無い・壊れている・キーが違うときは 0）。"""
        self._entries.clear()
        if not self.path.is_file():
            return 0
        try:
            with np.load(self.path, allow_pickle=False) as z:
                if str(z["key"]) != self.key:
                    return 0
                paths, sizes, mtimes = z["paths"], z["sizes"], z["mtimes"]
                emb, failed, missing = z["emb"], z["align_failed"], z["landmark_missing"]
        except Exception:  # noqa: BLE001  壊れたキャッシュは無視して作り直す
            return 0
        for i in range(len(paths)):
            self._entries[str(paths[i])] = (int(sizes[i]), float(mtimes[i]), emb[i], bool(failed[i]), bool(missing[i]))
        return len(self._entries)

    def get(self, face_path: Path) -> tuple[np.ndarray, bool, bool] | None:
        """(特徴量, アライン失敗, ランドマーク未取得)。無い・ファイルが変わっているときは None。"""
        hit = self._entries.get(self._rel(face_path))
        if hit is None:
            self.misses += 1
            return None
        try:
            st = Path(face_path).stat()
        except OSError:
            self.misses += 1
            return None
        if st.st_size != hit[0] or abs(st.st_mtime - hit[1]) > 1e-6:
            self.misses += 1
            return None
        self.hits += 1
        return hit[2], hit[3], hit[4]

    def put(self, face_path: Path, emb: np.ndarray, align_failed: bool, landmark_missing: bool) -> None:
        try:
            st = Path(face_path).stat()
        except OSError:
            return
        self._entries[self._rel(face_path)] = (st.st_size, st.st_mtime, np.asarray(emb, np.float32), align_failed,
                                               landmark_missing)
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        items = sorted(self._entries.items())
        dim = len(items[0][1][2]) if items else 0
        tmp = self.path.with_name(self.path.stem + ".part.npz")  # np.savez は拡張子 .npz を要求する
        np.savez(tmp, key=np.array(self.key), paths=np.array([k for k, _ in items]),
                 sizes=np.array([v[0] for _, v in items], np.int64), mtimes=np.array([v[1] for _, v in items], np.float64),
                 emb=np.stack([v[2] for _, v in items]) if items else np.zeros((0, dim), np.float32),
                 align_failed=np.array([v[3] for _, v in items], bool),
                 landmark_missing=np.array([v[4] for _, v in items], bool))
        tmp.replace(self.path)
        self._dirty = False

    def __len__(self) -> int:
        return len(self._entries)
