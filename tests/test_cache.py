"""比較対象の特徴量キャッシュ（core/report/cache.py）と embed_entries との連携のテスト。GPU・重み不要。"""
import os

import cv2
import numpy as np

from compare_face.core.features.base import FaceFeatureExtractor
from compare_face.core.landmarks import get_aligner
from compare_face.core.report.cache import CACHE_DIR_NAME, EmbeddingCache
from compare_face.core.report.gallery import scan_gallery
from compare_face.core.report.pipeline import embed_entries


class CountingExtractor(FaceFeatureExtractor):
    """平均色を特徴量にし、embed_batch に渡された枚数を数える。"""
    name = "テスト用カウント"
    embed_dim = 3
    calls = 0

    def weights_ready(self):
        return True

    def load(self):
        pass

    def embed_batch(self, faces_bgr, landmarks=None):
        CountingExtractor.calls += len(faces_bgr)
        feats = np.stack([f.reshape(-1, 3).mean(0) for f in faces_bgr]).astype(np.float32) + 1.0
        return feats / np.linalg.norm(feats, axis=1, keepdims=True)


def _make_gallery(root, n_tracks=3, n_images=2):
    for t in range(1, n_tracks + 1):
        d = root / "run" / "Cam1" / "v" / str(t)
        d.mkdir(parents=True)
        for i in range(n_images):
            cv2.imwrite(str(d / f"20260311_1227{t}{i}_000_face.jpg"), np.full((40, 40, 3), 40 * t + i, np.uint8))
    return scan_gallery(root)


def test_cache_roundtrip_and_reuse(tmp_path):
    entries = _make_gallery(tmp_path)
    ext, al = CountingExtractor(), get_aligner("なし")
    CountingExtractor.calls = 0
    cache = EmbeddingCache(tmp_path, ext.name, al.name)
    assert cache.load() == 0 and not cache.path.exists()
    emb1, _ = embed_entries(ext, al, entries, cache=cache)
    cache.save()
    assert CountingExtractor.calls == 6 and (cache.hits, cache.misses) == (0, 6)
    assert cache.path.parent.name == CACHE_DIR_NAME and cache.path.is_file()
    # 2 回目は全件キャッシュから（embed_batch は呼ばれない）。順序を変えても結果は一致
    cache2 = EmbeddingCache(tmp_path, ext.name, al.name)
    assert cache2.load() == 6
    CountingExtractor.calls = 0
    emb2, stats = embed_entries(ext, al, list(reversed(entries)), cache=cache2)
    assert CountingExtractor.calls == 0 and (cache2.hits, cache2.misses) == (6, 0)
    assert np.allclose(emb2[::-1], emb1) and stats.align_failed == 0
    # キャッシュ無しと同じ値
    emb3, _ = embed_entries(ext, al, entries)
    assert np.allclose(emb3, emb1)


def test_cache_detects_changed_and_new_files(tmp_path):
    entries = _make_gallery(tmp_path)
    ext, al = CountingExtractor(), get_aligner("なし")
    cache = EmbeddingCache(tmp_path, ext.name, al.name)
    embed_entries(ext, al, entries, cache=cache)
    cache.save()
    # 1 枚を差し替え（内容・更新時刻が変わる）、1 枚を追加
    changed = entries[0].face_path
    cv2.imwrite(str(changed), np.full((40, 40, 3), 200, np.uint8))
    st = os.stat(changed)
    os.utime(changed, (st.st_atime, st.st_mtime + 10))  # mtime の分解能に依存しないよう明示的に進める
    new_dir = tmp_path / "run" / "Cam1" / "v" / "9"
    new_dir.mkdir()
    cv2.imwrite(str(new_dir / "20260311_122799_000_face.jpg"), np.full((40, 40, 3), 10, np.uint8))
    entries2 = scan_gallery(tmp_path)
    cache2 = EmbeddingCache(tmp_path, ext.name, al.name)
    cache2.load()
    CountingExtractor.calls = 0
    emb, _ = embed_entries(ext, al, entries2, cache=cache2)
    assert CountingExtractor.calls == 2 and (cache2.hits, cache2.misses) == (5, 2)
    cache2.save()
    assert EmbeddingCache(tmp_path, ext.name, al.name).load() == 7
    # 差し替えた画像の特徴量は新しい内容で計算されている
    i = [e.face_path for e in entries2].index(changed)
    direct, _ = embed_entries(ext, al, [entries2[i]])
    assert np.allclose(emb[i], direct[0])


def test_cache_key_separates_models_and_ignores_other_schema(tmp_path):
    entries = _make_gallery(tmp_path, n_tracks=1)
    ext, al = CountingExtractor(), get_aligner("なし")
    c1 = EmbeddingCache(tmp_path, ext.name, al.name)
    embed_entries(ext, al, entries, cache=c1)
    c1.save()
    c_other = EmbeddingCache(tmp_path, "別モデル", al.name)
    assert c_other.path != c1.path and c_other.load() == 0
    # キーが違うファイルを同じパスに置いても読まない
    broken = EmbeddingCache(tmp_path, ext.name, al.name)
    np.savez(broken.path, key=np.array("other"), paths=np.array(["a"]), sizes=np.array([1]), mtimes=np.array([1.0]),
             emb=np.zeros((1, 3), np.float32), align_failed=np.array([False]), landmark_missing=np.array([False]))
    assert broken.load() == 0
    broken.path.write_bytes(b"garbage")
    assert broken.load() == 0


def test_cache_records_align_failure_flags(tmp_path):
    entries = _make_gallery(tmp_path, n_tracks=1, n_images=2)

    class FailFirst(type(get_aligner("なし"))):
        n = 0

        def align(self, img, size):
            FailFirst.n += 1
            return None if FailFirst.n == 1 else super().align(img, size)

    ext = CountingExtractor()
    cache = EmbeddingCache(tmp_path, ext.name, "fail")
    _, stats = embed_entries(ext, FailFirst(), entries, cache=cache)
    cache.save()
    assert stats.align_failed == 1
    cache2 = EmbeddingCache(tmp_path, ext.name, "fail")
    cache2.load()
    _, stats2 = embed_entries(ext, FailFirst(), entries, cache=cache2)
    assert stats2.align_failed == 1 and cache2.hits == 2  # 再利用時も失敗件数を引き継ぐ
