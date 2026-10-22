import numpy as np
import pytest
import torch

from compare_face.core.features import EXTRACTORS, WeightsUnavailable
from compare_face.core.features.fspfm import FSPFM_DIR, preprocess as fspfm_preprocess
from compare_face.core.features.fspfm_net import (
    FeatureSpacePoseFrontalizationModule, ResnetIR, load_fspfm_state_dicts,
)
from compare_face.core.pose import POSE_OPTIONS, HeadPoseEstimator, decode_angles, get_pose_estimator
from compare_face.core.pose.hopenet import VARIANTS, build_hopenet, preprocess as pose_preprocess

FINE = "FSPFM ResNet50 (正面化あり, 微調整済)"
PRE = "FSPFM ResNet50 (正面化あり, 事前学習のみ)"


# ---- レジストリ ----
def test_fspfm_models_registered():
    assert {FINE, PRE} <= set(EXTRACTORS) and not any("正面化なし" in n for n in EXTRACTORS)
    ex = EXTRACTORS[FINE]()
    assert ex.use_frontalization and ex.pose is not None and ex.pose.variant == "alpha2"
    assert ex.embed_dim == 512 and ex.input_size == 112 and not ex.needs_landmarks
    assert ex.weights_path() == FSPFM_DIR / "fspfm_stage2.pt"
    assert "10QcM-HHBlmArC6S-SJmBa6M9V5QqVCcV" in ex.manual_instructions()


def test_pose_options_and_estimator():
    assert "なし" in POSE_OPTIONS and get_pose_estimator("なし") is None
    est = get_pose_estimator("HopeNet (robust)")
    assert isinstance(est, HeadPoseEstimator) and est.variant == "robust_alpha1"
    assert est.weights_path().name == VARIANTS["robust_alpha1"]
    with pytest.raises(KeyError):
        get_pose_estimator("存在しない")
    with pytest.raises(KeyError):
        HeadPoseEstimator("unknown")


# ---- 前処理・角度復号 ----
def test_preprocess_ranges():
    x = fspfm_preprocess(np.zeros((40, 30, 3), np.uint8))
    assert x.shape == (3, 112, 112) and float(x.min()) == -1.0
    p = pose_preprocess(np.full((50, 50, 3), 255, np.uint8))
    assert p.shape == (3, 224, 224)
    assert np.allclose(p[:, 0, 0], (1.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225]), atol=1e-5)


def test_decode_angles_expected_value():
    logits = np.full((2, 66), -1e9, np.float32)
    logits[0, 33] = 0.0           # ビン 33 -> 33*3-99 = 0 度
    logits[1, 0] = 0.0            # ビン 0 -> -99 度
    assert np.allclose(decode_angles(logits), [0.0, -99.0])
    logits2 = np.zeros((1, 66), np.float32)  # 一様 -> 期待値 32.5 -> -1.5 度
    assert np.allclose(decode_angles(logits2), [-1.5])


# ---- ネットワーク構造（キー互換） ----
def test_hopenet_structure_matches_official_keys():
    m = build_hopenet()
    keys = m.state_dict().keys()
    # 配布 .pkl は旧 PyTorch 製で BatchNorm の num_batches_tracked を含まない（読込時に自動補完される）
    assert len([k for k in keys if not k.endswith("num_batches_tracked")]) == 273
    assert "fc_yaw.weight" in keys and "fc_finetune.weight" in keys
    assert m.state_dict()["fc_yaw.weight"].shape == (66, 2048) and m.state_dict()["fc_finetune.weight"].shape == (3, 2051)
    with torch.no_grad():
        yaw, pitch, roll, f = m.eval()(torch.zeros(1, 3, 224, 224))
    assert yaw.shape == (1, 66) and f.shape == (1, 2048)


def test_fspfm_structure_and_forward():
    bb = ResnetIR().eval()
    keys = bb.state_dict().keys()
    assert len(keys) == 349 and bb.state_dict()["output_layer.3.weight"].shape == (512, 25088)
    front = FeatureSpacePoseFrontalizationModule().eval()
    assert len(front.state_dict()) == 62 and front.state_dict()["pose_mapping1.fc1.weight"].shape == (512, 2048)
    with torch.no_grad():
        f = bb(torch.zeros(2, 3, 112, 112))
        out = front(f, torch.zeros(2, 2048))
    assert f.shape == (2, 512) and out.shape == (2, 512)


def test_load_fspfm_state_dicts_splits_and_drops_head(tmp_path):
    sd = {"backbone.input_layer.0.weight": torch.zeros(1), "frontalization_module.pose_mapping1.fc1.bias": torch.zeros(1),
          "head.weight": torch.zeros(2, 2)}
    p = tmp_path / "ck.pt"
    torch.save({"state_dict": sd, "epoch": 1}, p)
    bb, fr = load_fspfm_state_dicts(str(p))
    assert list(bb) == ["input_layer.0.weight"] and list(fr) == ["pose_mapping1.fc1.bias"]


def test_fspfm_real_checkpoints_load_strictly():
    ex = EXTRACTORS[FINE]()
    if not ex.weights_path().is_file():
        with pytest.raises(WeightsUnavailable):
            ex.load()
        pytest.skip("FSPFM の重みが未配置")
    bb_sd, fr_sd = load_fspfm_state_dicts(str(ex.weights_path()))
    ResnetIR().load_state_dict(bb_sd, strict=True)
    FeatureSpacePoseFrontalizationModule().load_state_dict(fr_sd, strict=True)


def test_hopenet_real_weights_load_strictly():
    est = HeadPoseEstimator("alpha2")
    if not est.weights_ready():
        pytest.skip("HopeNet の重みが未配置")
    build_hopenet().load_state_dict(torch.load(str(est.weights_path()), map_location="cpu", weights_only=True), strict=True)


# ---- レポートの情報欄 ----
def test_info_text_shows_pose(tmp_path):
    from pathlib import Path
    from compare_face.core.report.gallery import GalleryEntry
    from compare_face.core.report.excel import _info_text
    e = GalleryEntry(Path("a_face.jpg"), None, tmp_path, "C", "v", "1", "20260311_120000_000")
    assert "姿勢" not in _info_text(e, "1位", 0.5)
    e.pose = (12.4, -3.2, 0.4)
    assert "姿勢 yaw:+12° pitch:-3° roll:+0°" in _info_text(e, "1位", 0.5)
