import subprocess
from pathlib import Path

import numpy as np
import pytest

from compare_face.core.common import cvlface_repo as loader
from compare_face.core.features import EXTRACTORS, WeightsUnavailable
from compare_face.core.features.cvlface import mirror_keypoints, normalize_keypoints, preprocess
from compare_face.core.landmarks import ALIGNERS, AlignResult, get_aligner, template_5pts
from compare_face.core.landmarks.cvlface_dfa import to_bgr_uint8
from compare_face.core.landmarks.insightface_2d106 import (
    ARCFACE_TEMPLATE_5PTS, align_and_transform, estimate_affine_5pts, transform_points,
)
from compare_face.core.report.pipeline import EmbedStats, embed_entries

KPRPE = "CVLFace KP-RPE ViT-B (WebFace12M)"


# ---- レジストリ ----
def test_cvlface_models_registered():
    assert KPRPE in EXTRACTORS and "CVLFace KP-RPE ViT-B (WebFace4M)" in EXTRACTORS
    ex = EXTRACTORS[KPRPE]()
    assert ex.needs_landmarks and ex.input_size == 112 and ex.embed_dim == 512
    assert ex.repo_id == "minchul/cvlface_adaface_vit_base_kprpe_webface12m"
    assert ex.local_dir() == loader.CVLFACE_FEATURE_DIR / "cvlface_adaface_vit_base_kprpe_webface12m"
    assert not EXTRACTORS["AdaFace"]().needs_landmarks
    assert "CVLFace DFA" in ALIGNERS


def test_cvlface_manual_instructions_mention_repo():
    ex = EXTRACTORS["CVLFace KP-RPE ViT-B (WebFace4M)"]()
    if ex.weights_ready():
        pytest.skip("WebFace4M モデルが取得済み")
    msg = ex.manual_instructions()
    assert "minchul/cvlface_adaface_vit_base_kprpe_webface4m" in msg and "download" in msg
    with pytest.raises(WeightsUnavailable):
        ex.load()


# ---- 前処理・キーポイント ----
def test_cvlface_preprocess_shape_and_range():
    x = preprocess(np.zeros((40, 30, 3), np.uint8))
    assert x.shape == (3, 112, 112) and x.dtype == np.float32 and float(x.min()) == -1.0
    assert float(preprocess(np.full((112, 112, 3), 255, np.uint8)).max()) == 1.0


def test_normalize_keypoints_uses_template_when_missing():
    kp, missing = normalize_keypoints([np.array([[0, 0], [112, 0], [56, 56], [0, 112], [112, 112]], np.float32), None], 2)
    assert kp.shape == (2, 5, 2) and missing == 1
    assert np.allclose(kp[0], [[0, 0], [1, 0], [0.5, 0.5], [0, 1], [1, 1]])
    assert np.allclose(kp[1], template_5pts(112) / 112.0)
    kp2, missing2 = normalize_keypoints(None, 3)
    assert kp2.shape == (3, 5, 2) and missing2 == 3


def test_mirror_keypoints_swaps_left_right():
    kp = np.array([[[0.3, 0.4], [0.7, 0.4], [0.5, 0.6], [0.35, 0.8], [0.65, 0.8]]], np.float32)
    m = mirror_keypoints(kp)
    assert np.allclose(m[0, 0], [0.3, 0.4]) and np.allclose(m[0, 1], [0.7, 0.4])  # 右目を反転 → 左目の位置
    assert np.allclose(m[0, 2], [0.5, 0.6]) and np.allclose(m[0, 3], [0.35, 0.8])
    assert np.allclose(mirror_keypoints(m), kp)  # 2回で元に戻る


# ---- ランドマークモデル共通 ----
def test_noalign_has_no_landmarks():
    r = get_aligner("なし").align_with_landmarks(np.zeros((50, 60, 3), np.uint8), 112)
    assert isinstance(r, AlignResult) and r.image.shape == (112, 112, 3) and r.landmarks is None


def test_2d106_align_and_transform_returns_landmarks():
    img = np.random.randint(0, 255, (112, 112, 3), np.uint8)
    r = align_and_transform(img, ARCFACE_TEMPLATE_5PTS, 112)
    assert r.image.shape == (112, 112, 3) and np.allclose(r.landmarks, ARCFACE_TEMPLATE_5PTS, atol=0.5)
    m = estimate_affine_5pts(ARCFACE_TEMPLATE_5PTS * 2, 224)
    assert np.allclose(transform_points(ARCFACE_TEMPLATE_5PTS * 2, m), ARCFACE_TEMPLATE_5PTS * 2, atol=0.5)
    assert estimate_affine_5pts(np.full((5, 2), np.nan, np.float32), 112) is None


def test_dfa_to_bgr_uint8():
    chw = np.zeros((3, 4, 4), np.float32)
    chw[0] = 1.0   # R 最大
    chw[2] = -1.0  # B 最小
    bgr = to_bgr_uint8(chw)
    assert bgr.shape == (4, 4, 3) and bgr.dtype == np.uint8
    assert bgr[0, 0].tolist() == [0, 127, 255]


def test_dfa_weights_state_matches_files():
    from compare_face.core.common.weights import LANDMARK_DIR
    al = ALIGNERS["CVLFace DFA"]()
    assert al.weights_ready() == loader.is_downloaded("minchul/cvlface_DFA_mobilenet", "aligners", LANDMARK_DIR)
    assert loader.local_dir("minchul/cvlface_DFA_mobilenet", LANDMARK_DIR) == LANDMARK_DIR / "cvlface_DFA_mobilenet"
    if not al.weights_ready():
        assert "cvlface_DFA_mobilenet" in al.manual_instructions()


def test_import_package_under_unique_name(tmp_path):
    pkg = tmp_path / "models"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("from .sub import VALUE\n\ndef get_model(conf):\n    return VALUE\n", encoding="utf-8")
    (pkg / "sub" / "__init__.py").write_text("from .inner import VALUE\n", encoding="utf-8")
    (pkg / "sub" / "inner.py").write_text("VALUE = 'ok'\n", encoding="utf-8")
    mod = loader.import_package(pkg, "cvlface_test_models")
    assert mod.get_model(None) == "ok"
    assert loader.import_package(pkg, "cvlface_test_models") is mod  # 2回目は再利用
    with pytest.raises(FileNotFoundError):
        loader.import_package(tmp_path / "missing", "cvlface_test_missing")


def test_without_extension_build_blocks_setup_py():
    with loader.without_extension_build():
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(["python", "setup.py", "install"])
    assert subprocess.check_call is not None


def test_is_downloaded_requires_weights_and_package(tmp_path):
    repo = "minchul/some_model"
    assert not loader.is_downloaded(repo, "models", tmp_path)
    d = loader.local_dir(repo, tmp_path)
    assert d == tmp_path / "some_model"
    (d / "pretrained_model").mkdir(parents=True)
    (d / "models").mkdir()
    (d / "pretrained_model" / "model.pt").write_bytes(b"x")
    (d / "pretrained_model" / "model.yaml").write_text("name: base")
    (d / "models" / "__init__.py").write_text("")
    assert loader.is_downloaded(repo, "models", tmp_path)
    assert loader.local_dir(repo).parent == loader.CVLFACE_FEATURE_DIR


class _FakeAligner:
    def __init__(self, with_landmarks):
        self.with_landmarks = with_landmarks

    def align_with_landmarks(self, img, size):
        if img.shape[0] == 1:  # 1px 高さの画像は失敗扱い
            return None
        return AlignResult(np.zeros((size, size, 3), np.uint8), template_5pts(size) if self.with_landmarks else None)


class _FakeExtractor:
    input_size = 112
    embed_dim = 4
    needs_landmarks = True

    def __init__(self):
        self.received = []

    def embed_batch(self, faces, landmarks=None):
        self.received.append(landmarks)
        return np.ones((len(faces), 4), np.float32)


def _entries(tmp_path, shapes):
    import cv2
    from compare_face.core.report.gallery import GalleryEntry
    out = []
    for i, shape in enumerate(shapes):
        p = tmp_path / f"2026_{i}_face.jpg"
        cv2.imwrite(str(p), np.zeros(shape, np.uint8))
        out.append(GalleryEntry(p, None, tmp_path, "C", "v", "1", f"2026_{i}"))
    return out


def test_embed_entries_passes_landmarks_and_counts(tmp_path):
    entries = _entries(tmp_path, [(50, 50, 3), (1, 50, 3), (60, 40, 3)])
    ex = _FakeExtractor()
    emb, stats = embed_entries(ex, _FakeAligner(with_landmarks=True), entries)
    assert emb.shape == (3, 4)
    assert stats == EmbedStats(align_failed=1, landmark_missing=1)  # 失敗した1枚はランドマークも無い
    lms = ex.received[0]
    assert lms[0] is not None and lms[1] is None and lms[2] is not None
    ex2 = _FakeExtractor()
    _, stats2 = embed_entries(ex2, _FakeAligner(with_landmarks=False), entries)
    assert stats2 == EmbedStats(align_failed=1, landmark_missing=3)


def test_dfa_landmarks_plausible():
    from compare_face.core.landmarks.cvlface_dfa import landmarks_plausible
    good = np.array([[38, 52], [74, 52], [56, 72], [42, 92], [71, 92]], np.float32)
    assert landmarks_plausible(good, 112)
    assert not landmarks_plausible(good[[1, 0, 2, 3, 4]], 112)          # 左右の目が逆
    assert not landmarks_plausible(good[[3, 4, 2, 0, 1]], 112)          # 目と口が上下逆
    assert not landmarks_plausible(good + 200, 112)                     # 画像の外
    assert not landmarks_plausible(np.full((5, 2), np.nan, np.float32), 112)
    assert not landmarks_plausible(good[:4], 112)


def test_hidden_models_are_not_visible_but_still_usable():
    from compare_face.core.features import EXTRACTORS, get_extractor, visible_extractors
    vis = visible_extractors()
    assert "CVLFace KP-RPE ViT-B (WebFace12M)" in vis and "AdaFace" in vis
    assert "CVLFace KP-RPE ViT-B (WebFace4M)" not in vis and "Part fViT (WebFace4M, 事前学習なし)" not in vis
    assert get_extractor("CVLFace KP-RPE ViT-B (WebFace4M)").hidden and len(vis) == len(EXTRACTORS) - 3
