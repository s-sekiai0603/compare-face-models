"""eval コマンド（insightface 形式 .bin の 1:1 照合精度）のテスト。GPU・重み不要。"""
import os
import pickle

import cv2
import numpy as np
import pytest

from compare_face.cli import build_parser, main
from compare_face.core.common.face_template import mirror_5pts
from compare_face.core.eval.pipeline import AlignedSet, EvalRow, align_set, embed_with_flip, run_eval, write_csv
from compare_face.core.eval.published import PUBLISHED, SOURCES
from compare_face.core.eval.verification import (
    THRESHOLDS, evaluate_pairs, flat_image_mask, fuse_flip, k_fold_indices, load_bin, pair_distances, select_pairs,
)
from compare_face.core.features import EXTRACTORS
from compare_face.core.features.base import FaceFeatureExtractor
from compare_face.core.landmarks import get_aligner


def _write_bin(path, n_pairs=20, size=112, flat_images=()):
    """同一人物ペアは同じ色、別人ペアは違う色の画像で .bin を作る（Python 2 由来の pickle と同じ protocol 2）。
    flat_images に指定した画像番号は一色（整列失敗相当）にする。"""
    rng = np.random.default_rng(0)
    bases = [(220, 30, 30), (30, 220, 30), (30, 30, 220)]  # 3 人分の「顔色」。方向が離れているので平均色で区別できる
    bins, issame = [], []
    for i in range(n_pairs):
        same = i % 2 == 0
        c1 = bases[i % 3]
        c2 = c1 if same else bases[(i + 1) % 3]
        for j, c in enumerate((c1, c2)):
            img = np.full((size, size, 3), c, np.uint8)
            img[:, : size // 2] = np.minimum(np.array(c) + 20, 255)  # 左右非対称にして反転の効果が出るようにする
            if 2 * i + j not in flat_images:
                noise = rng.integers(-40, 41, (size, size, 1))
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)  # 平坦画像と区別できる質感
            ok, buf = cv2.imencode(".jpg", img)
            assert ok
            bins.append(buf.reshape(-1))
        issame.append(same)
    with open(path, "wb") as f:
        pickle.dump((bins, issame), f, protocol=2)
    return path


class ColorExtractor(FaceFeatureExtractor):
    """テスト用: 画像の平均色を特徴量にする（重み不要）。"""
    name = "テスト用平均色"
    embed_dim = 3
    hidden = True

    def weights_ready(self):
        return True

    def load(self):
        pass

    def embed_batch(self, faces_bgr, landmarks=None):
        feats = np.stack([f.reshape(-1, 3).mean(0) for f in faces_bgr]).astype(np.float32) + 1.0
        return feats / np.linalg.norm(feats, axis=1, keepdims=True)


class LandmarkRecorder(ColorExtractor):
    name = "テスト用ランドマーク記録"
    needs_landmarks = True
    received = []

    def embed_batch(self, faces_bgr, landmarks=None):
        LandmarkRecorder.received.append(landmarks)
        return super().embed_batch(faces_bgr)


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setitem(EXTRACTORS, ColorExtractor.name, ColorExtractor)
    monkeypatch.setitem(EXTRACTORS, LandmarkRecorder.name, LandmarkRecorder)
    LandmarkRecorder.received.clear()


# ---- .bin の読込 ----
def test_load_bin_decodes_pairs(tmp_path):
    ds = load_bin(_write_bin(tmp_path / "cfp_fp.bin", n_pairs=12))
    assert ds.name == "cfp_fp" and ds.n_pairs == 12 and len(ds.images) == 24
    assert ds.images[0].shape == (112, 112, 3) and ds.images[0].dtype == np.uint8
    assert ds.issame.tolist() == [True, False] * 6
    limited = load_bin(tmp_path / "cfp_fp.bin", limit_pairs=5)  # 等間隔: ペア 0, 3, 6, 8, 11
    assert limited.n_pairs == 5 and len(limited.images) == 10
    assert limited.issame.tolist() == [True, False, True, True, False]
    assert np.array_equal(limited.images[2], ds.images[6]) and np.array_equal(limited.images[3], ds.images[7])
    assert load_bin(tmp_path / "cfp_fp.bin", limit_pairs=100).n_pairs == 12


def test_load_bin_rejects_pickle_with_code(tmp_path):
    with open(tmp_path / "bad.bin", "wb") as f:
        pickle.dump(([os.system], [True]), f)
    with pytest.raises(pickle.UnpicklingError):
        load_bin(tmp_path / "bad.bin")


def test_load_bin_rejects_mismatched_counts(tmp_path):
    with open(tmp_path / "bad.bin", "wb") as f:
        pickle.dump(([np.zeros(3, np.uint8)] * 3, [True]), f)
    with pytest.raises(ValueError):
        load_bin(tmp_path / "bad.bin")


# ---- 精度計算 ----
def test_k_fold_matches_sklearn_layout():
    folds = k_fold_indices(23, 10)
    assert [len(f) for f in folds] == [3, 3, 3, 2, 2, 2, 2, 2, 2, 2]
    assert np.concatenate(folds).tolist() == list(range(23))


def test_pair_distances_is_two_minus_two_cos():
    emb = np.array([[1, 0], [0, 1], [1, 0], [1, 0], [1, 0], [-1, 0]], np.float32)
    assert np.allclose(pair_distances(emb), [2.0, 0.0, 4.0])
    assert THRESHOLDS[0] == 0.0 and abs(THRESHOLDS[-1] - 3.99) < 1e-9 and len(THRESHOLDS) == 400


def test_evaluate_pairs_perfect_and_random():
    rng = np.random.default_rng(1)
    n = 50
    issame = np.array([i % 2 == 0 for i in range(n)])
    emb = []
    for same in issame:
        a = rng.normal(size=8)
        emb += [a, a + rng.normal(size=8) * 0.01 if same else -a]
    r = evaluate_pairs(np.array(emb), issame)
    assert r.accuracy == 1.0 and r.accuracy_std == 0.0 and len(r.fold_accuracies) == 10
    assert 0.0 < r.cos_threshold < 1.0 and abs(r.cos_threshold - (1 - r.threshold / 2)) < 1e-9
    r2 = evaluate_pairs(rng.normal(size=(2 * n, 8)), issame)
    assert 0.0 <= r2.accuracy <= 1.0
    with pytest.raises(ValueError):
        evaluate_pairs(np.ones((8, 4)), np.ones(4, bool))  # 枚数がペア数の 2 倍でない
    with pytest.raises(ValueError):
        evaluate_pairs(np.ones((8, 4)), np.ones(4, bool)[:4])  # ペア数が分割数より少ない


def test_flat_image_mask_and_select_pairs():
    rng = np.random.default_rng(0)
    flat = np.full((112, 112, 3), 120, np.uint8)
    textured = rng.integers(0, 256, (112, 112, 3)).astype(np.uint8)
    edge_only = flat.copy()
    edge_only[:20, :, :] = 0  # 中央領域の外だけ変化 -> 平坦扱い
    assert flat_image_mask([flat, textured, edge_only]).tolist() == [True, False, True]
    emb = np.arange(12, dtype=np.float32).reshape(6, 2)
    assert select_pairs(emb, np.array([True, False, True])).tolist() == [[0, 1], [2, 3], [8, 9], [10, 11]]


def test_fuse_flip_normalizes_sum():
    a = np.array([[3.0, 0.0]], np.float32)
    b = np.array([[0.0, 4.0]], np.float32)
    assert np.allclose(fuse_flip(a, b), [[0.6, 0.8]])


def test_mirror_5pts_swaps_sides():
    pts = np.array([[30, 50], [80, 50], [55, 70], [40, 90], [70, 90]], np.float32)
    m = mirror_5pts(pts, 112)
    assert np.allclose(m, [[31, 50], [81, 50], [56, 70], [41, 90], [71, 90]])
    assert np.allclose(mirror_5pts(m, 112), pts)


# ---- パイプライン ----
def test_align_set_counts_failures():
    class FailEvery3(type(get_aligner("なし"))):
        calls = 0

        def align(self, img, size):
            FailEvery3.calls += 1
            return None if FailEvery3.calls % 3 == 0 else super().align(img, size)

    images = [np.zeros((100, 90, 3), np.uint8)] * 6
    a = align_set(FailEvery3(), images, 112)
    assert a.align_failed == 2
    assert all(f.shape == (112, 112, 3) for f in a.faces) and all(lm is None for lm in a.landmarks)


def test_embed_with_flip_mirrors_landmarks(registry):
    img = np.zeros((112, 112, 3), np.uint8)
    pts = np.array([[30, 50], [80, 50], [55, 70], [40, 90], [70, 90]], np.float32)
    aligned = AlignedSet([img, img], [pts, None], 0)
    emb, emb_flip = embed_with_flip(LandmarkRecorder(), aligned)
    assert emb.shape == emb_flip.shape == (2, 3)
    assert len(LandmarkRecorder.received) == 2
    assert np.allclose(LandmarkRecorder.received[0][0], pts) and LandmarkRecorder.received[0][1] is None
    assert np.allclose(LandmarkRecorder.received[1][0], mirror_5pts(pts, 112)) and LandmarkRecorder.received[1][1] is None


def test_run_eval_writes_csv_and_perfect_accuracy(tmp_path, registry):
    bin_path = _write_bin(tmp_path / "lfw.bin", n_pairs=30)
    out = tmp_path / "result.csv"
    logs = []
    rows = run_eval([bin_path], [ColorExtractor.name], ["なし"], out, echo=logs.append)
    assert len(rows) == 1
    r = rows[0]
    assert (r.dataset, r.model, r.landmark, r.n_pairs) == ("lfw", ColorExtractor.name, "なし", 30)
    assert r.accuracy == 100.0 and r.accuracy_single == 100.0 and r.align_failed == 0 and r.landmark_missing is None
    assert r.published is None and (r.accuracy_valid, r.flat_images, r.flat_pairs) == (100.0, 0, 0)
    text = out.read_text(encoding="utf-8-sig").splitlines()
    assert text[0].startswith("データセット,顔特徴量モデル,顔ランドマークモデル,ペア数,精度 flip融合あり(%),標準偏差(%),"
                              "精度 flip融合なし(%),精度 整列失敗ペア除外(%),整列失敗画像(枚),整列失敗ペア,")
    assert text[1].startswith(f"lfw,{ColorExtractor.name},なし,30,100.000,0.000,100.000,100.000,0,0,")
    assert any("精度 100.00±0.00%" in m for m in logs)


def test_run_eval_excludes_flat_pairs(tmp_path, registry):
    # 画像 1（ペア 0 の片側）と画像 6, 7（ペア 3 の両側）を平坦にする -> 整列失敗画像 3 枚 / ペア 2 件
    bin_path = _write_bin(tmp_path / "cfp_fp.bin", n_pairs=30, flat_images=(1, 6, 7))
    rows = run_eval([bin_path], [ColorExtractor.name], ["なし"], tmp_path / "r.csv", echo=lambda m: None)
    r = rows[0]
    assert (r.flat_images, r.flat_pairs) == (3, 2)
    assert r.accuracy_valid == 100.0 and r.accuracy <= 100.0


def test_run_eval_requires_bins(tmp_path):
    with pytest.raises(ValueError):
        run_eval([], ["AdaFace"], ["なし"], tmp_path / "x.csv")


def test_write_csv_blank_for_none(tmp_path):
    write_csv([EvalRow("lfw", "m", "l", 10, 99.5, 0.1, 99.0, 99.8, 2, 1, 0.3, 0, None, None, 1.5)], tmp_path / "r.csv")
    line = tmp_path.joinpath("r.csv").read_text(encoding="utf-8-sig").splitlines()[1]
    assert line == "lfw,m,l,10,99.500,0.100,99.000,99.800,2,1,0.300,0,,,1.500"


# ---- CLI ----
def test_eval_parser_defaults():
    args = build_parser().parse_args(["eval"])
    assert (args.bin, args.model, args.landmark, args.out, args.limit) == (None, None, None, None, 0)
    args = build_parser().parse_args(["eval", "--bin", "a.bin", "--bin", "b.bin", "--model", "AdaFace", "--limit", "50"])
    assert args.bin == ["a.bin", "b.bin"] and args.model == ["AdaFace"] and args.limit == 50


def test_eval_command_end_to_end(tmp_path, registry, capsys):
    bin_path = _write_bin(tmp_path / "agedb_30.bin", n_pairs=20)
    out = tmp_path / "eval.csv"
    assert main(["eval", "--bin", str(bin_path), "--model", ColorExtractor.name, "--out", str(out), "--limit", "12"]) == 0
    assert out.is_file()
    assert "12 ペア" in capsys.readouterr().out
    assert main(["eval", "--bin", str(tmp_path / "missing.bin"), "--model", ColorExtractor.name, "--out", str(out)]) == 1


def test_published_values_refer_to_registered_models():
    for model, dataset in PUBLISHED:
        assert model in EXTRACTORS, model
        assert dataset in {"lfw", "cfp_fp", "cplfw", "calfw", "agedb_30"}
        assert 90.0 < PUBLISHED[(model, dataset)] <= 100.0
    assert set(SOURCES) == {m for m, _ in PUBLISHED}
