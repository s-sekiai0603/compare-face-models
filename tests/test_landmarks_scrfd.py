import numpy as np
import pytest

from compare_face.core.common.weights import DETECTION_DIR, FEATURE_DIR, LANDMARK_DIR, MODELS_DIR, POSE_DIR
from compare_face.core.features import EXTRACTORS, visible_extractors
from compare_face.core.features.adaface import ADAFACE_DIR, apply_clahe, preprocess
from compare_face.core.landmarks import ALIGNERS
from compare_face.core.landmarks.base import ARCFACE_TEMPLATE_5PTS
from compare_face.core.landmarks.insightface_scrfd import distance2bbox, distance2kps, estimate_norm, nms


def test_models_dir_layout():
    assert MODELS_DIR.name == "models" and MODELS_DIR.parent.is_dir()   # リポジトリ直下の models/（フォルダ名には依存しない）
    assert DETECTION_DIR == MODELS_DIR / "detection" and LANDMARK_DIR == MODELS_DIR / "landmark"
    assert FEATURE_DIR == MODELS_DIR / "feature" and POSE_DIR == MODELS_DIR / "pose"
    assert ADAFACE_DIR == FEATURE_DIR / "adaface"


def test_aligner_order_and_scrfd_registered():
    assert list(ALIGNERS)[:3] == ["なし", "InsightFace 2d106det", "InsightFace SCRFD (det_10g)"] and list(ALIGNERS)[-1] == "CVLFace DFA"
    al = ALIGNERS["InsightFace SCRFD (det_10g)"]()
    assert al.weights_path() == LANDMARK_DIR / "det_10g.onnx"
    assert "buffalo_l" in al.manual_instructions()


def test_hidden_extractors_and_clahe_preprocess():
    assert "AdaFace" in visible_extractors() and not EXTRACTORS["AdaFace"].hidden
    for hidden_name in ("FSPFM ResNet50 (正面化あり, 事前学習のみ)", "CVLFace KP-RPE ViT-B (WebFace4M)"):
        assert EXTRACTORS[hidden_name].hidden and hidden_name not in visible_extractors()
    assert not any(n.startswith("AdaFace (CLAHE") for n in EXTRACTORS)   # 互換版は削除済み（--clahe で代替）
    img = np.random.randint(0, 255, (112, 112, 3), np.uint8)
    assert np.array_equal(preprocess(img), preprocess(img, clahe=False))
    assert not np.array_equal(preprocess(img), preprocess(img, clahe=True))
    assert apply_clahe(img).shape == img.shape


def test_scrfd_decode_helpers():
    pts = np.array([[10.0, 20.0], [30.0, 40.0]], np.float32)
    bbox = distance2bbox(pts, np.array([[1, 2, 3, 4], [5, 6, 7, 8]], np.float32))
    assert bbox.tolist() == [[9, 18, 13, 24], [25, 34, 37, 48]]
    kps = distance2kps(pts, np.array([[1, 1, 2, 2, 3, 3, 4, 4, 5, 5]] * 2, np.float32))
    assert kps.shape == (2, 10) and kps[0].tolist() == [11, 21, 12, 22, 13, 23, 14, 24, 15, 25]
    dets = np.array([[0, 0, 10, 10, 0.9], [1, 1, 11, 11, 0.8], [50, 50, 60, 60, 0.7]], np.float32)
    assert nms(dets, 0.4) == [0, 2]


def test_scrfd_estimate_norm_identity_on_template():
    m = estimate_norm(ARCFACE_TEMPLATE_5PTS, 112)
    assert m.shape == (2, 3) and np.allclose(m, [[1, 0, 0], [0, 1, 0]], atol=1e-4)
    m2 = estimate_norm(ARCFACE_TEMPLATE_5PTS * 2, 112)  # 2 倍に拡大した点 -> 1/2 に縮小する変換
    assert np.allclose(m2[:, :2], [[0.5, 0], [0, 0.5]], atol=1e-4)


def test_scrfd_real_model_detects_synthetic_face_or_skips():
    al = ALIGNERS["InsightFace SCRFD (det_10g)"]()
    if not al.weights_ready():
        pytest.skip("det_10g.onnx が未配置")
    al.load()
    dets, kpss = al.detector.detect(np.zeros((200, 200, 3), np.uint8))  # 真っ黒 → 顔なし
    assert dets.shape[1] == 5 and kpss.shape[1:] == (5, 2) and len(dets) == 0
    assert al.align_with_landmarks(np.zeros((200, 200, 3), np.uint8), 112) is None
