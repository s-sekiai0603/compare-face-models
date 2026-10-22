"""論文どおりの検出器（RetinaFace-R50, MTCNN）の構造・補助関数・重み読込のテスト。"""
import numpy as np
import pytest
import torch

from compare_face.core.common.weights import LANDMARK_DIR
from compare_face.core.landmarks import ALIGNERS, get_aligner
from compare_face.core.landmarks.mtcnn import MTCNN_DIR, WEIGHT_FILES, pad_image
from compare_face.core.landmarks.mtcnn_net import calibrate_box, convert_to_square, nms as mtcnn_nms
from compare_face.core.landmarks.retinaface_r50 import (
    RETINAFACE_DIR, WEIGHTS_FILE, build_retinaface_r50, decode_boxes, decode_landmarks, load_retinaface_state_dict,
    pad_for_detection, prior_boxes,
)

RETINA = "RetinaFace-R50 (Pytorch_Retinaface)"
MTCNN = "MTCNN (AdaFace 公式)"


def test_aligner_registry_order():
    assert list(ALIGNERS) == ["なし", "InsightFace 2d106det", "InsightFace SCRFD (det_10g)", RETINA, MTCNN, "CVLFace DFA"]
    assert get_aligner(RETINA).weights_path() == RETINAFACE_DIR / WEIGHTS_FILE == LANDMARK_DIR / "retinaface" / WEIGHTS_FILE
    assert get_aligner(MTCNN).weights_dir() == MTCNN_DIR == LANDMARK_DIR / "mtcnn"
    assert "facexlib" in get_aligner(RETINA).manual_instructions() and "AdaFace" in get_aligner(MTCNN).manual_instructions()


# ---- RetinaFace ----
def test_retinaface_prior_boxes_and_decode():
    priors = prior_boxes(64, 64)
    # stride 8: 8x8x2, stride 16: 4x4x2, stride 32: 2x2x2
    assert priors.shape == (8 * 8 * 2 + 4 * 4 * 2 + 2 * 2 * 2, 4)
    assert np.allclose(priors[0], [0.5 * 8 / 64, 0.5 * 8 / 64, 16 / 64, 16 / 64])
    zero = np.zeros((len(priors), 4), np.float32)
    boxes = decode_boxes(zero, priors)  # オフセット 0 -> prior をそのまま xyxy にしたもの
    assert np.allclose(boxes[0], [priors[0, 0] - priors[0, 2] / 2, priors[0, 1] - priors[0, 3] / 2,
                                  priors[0, 0] + priors[0, 2] / 2, priors[0, 1] + priors[0, 3] / 2])
    lm = decode_landmarks(np.zeros((len(priors), 10), np.float32), priors)
    assert lm.shape == (len(priors), 10) and np.allclose(lm[0, :2], priors[0, :2]) and np.allclose(lm[0, 8:], priors[0, :2])


def test_retinaface_pad_for_detection():
    img = np.zeros((100, 200, 3), np.uint8)
    padded, scale, px, py = pad_for_detection(img, fraction=0.5, max_side=10000)
    assert padded.shape == (200, 400, 3) and scale == 1.0 and (px, py) == (100, 50)
    padded2, scale2, _, _ = pad_for_detection(img, fraction=0.5, max_side=200)
    assert max(padded2.shape[:2]) == 200 and abs(scale2 - 0.5) < 1e-6


def test_retinaface_structure_matches_checkpoint_keys():
    model = build_retinaface_r50()
    keys = set(model.state_dict().keys())
    assert "body.conv1.weight" in keys and "fpn.output1.0.weight" in keys and "ssh1.conv3X3.0.weight" in keys
    assert "ClassHead.2.conv1x1.bias" in keys and "LandmarkHead.0.conv1x1.weight" in keys and "body.fc.weight" not in keys
    assert model.state_dict()["LandmarkHead.0.conv1x1.weight"].shape == (20, 256, 1, 1)
    with torch.no_grad():
        bbox, cls, ldm = model.eval()(torch.zeros(1, 3, 64, 64))
    n = len(prior_boxes(64, 64))
    assert bbox.shape == (1, n, 4) and cls.shape == (1, n, 2) and ldm.shape == (1, n, 10)
    assert torch.allclose(cls.sum(-1), torch.ones(1, n))


def test_retinaface_real_weights_load_strictly_or_skip():
    al = get_aligner(RETINA)
    if not al.weights_ready():
        pytest.skip("RetinaFace の重みが未配置")
    sd = load_retinaface_state_dict(torch, al.weights_path())
    assert not any(k.startswith("module.") for k in sd)
    build_retinaface_r50().load_state_dict(sd, strict=True)
    al.load()
    assert al.align_with_landmarks(np.zeros((160, 160, 3), np.uint8), 112) is None  # 真っ黒 → 顔なし


# ---- MTCNN ----
def test_mtcnn_box_helpers():
    boxes = np.array([[0, 0, 9, 19, 0.9], [100, 100, 119, 119, 0.5]], np.float32)
    sq = convert_to_square(boxes.copy())
    assert np.allclose(sq[0], [-5, 0, 14, 19, 0]) and np.allclose(sq[1, :4], [100, 100, 119, 119])
    cal = calibrate_box(boxes.copy(), np.array([[0.1, 0.1, 0.1, 0.1], [0, 0, 0, 0]], np.float32))
    assert np.allclose(cal[0, :4], [1, 2, 10, 21]) and np.allclose(cal[1, :4], boxes[1, :4])
    dets = np.array([[0, 0, 10, 10, 0.9], [1, 1, 11, 11, 0.8], [50, 50, 60, 60, 0.7]], np.float32)
    assert sorted(mtcnn_nms(dets, 0.5)) == [0, 2]
    padded, px, py = pad_image(np.zeros((40, 80, 3), np.uint8), 0.25)
    assert padded.shape == (60, 120, 3) and (px, py) == (20, 10)


def test_mtcnn_real_weights_load_or_skip():
    al = get_aligner(MTCNN)
    if not al.weights_ready():
        pytest.skip("MTCNN の重みが未配置")
    assert all((MTCNN_DIR / f).is_file() for f in WEIGHT_FILES)
    al.load()
    assert al.align_with_landmarks(np.zeros((160, 160, 3), np.uint8), 112) is None  # 真っ黒 → 顔なし
