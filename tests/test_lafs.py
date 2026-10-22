import numpy as np
import pytest
import torch

from compare_face.core.features import EXTRACTORS, WeightsUnavailable
from compare_face.core.features.lafs import LAFS_DIR, preprocess
from compare_face.core.features.lafs_net import PartFViT, extract_patches, load_partfvit_state_dict

LAFS = "LAFS Part fViT (WebFace4M)"
BASELINE = "Part fViT (WebFace4M, 事前学習なし)"


def test_lafs_models_registered():
    assert LAFS in EXTRACTORS and BASELINE in EXTRACTORS
    ex = EXTRACTORS[LAFS]()
    assert not ex.needs_landmarks and ex.input_size == 112 and ex.embed_dim == 1536
    assert ex.weights_path() == LAFS_DIR / "lafs_partfvit_webface4m.pth"
    assert "1BUYm2Bcgp8ZRlBcwOZxiJtWiQAvK2Ujy" in ex.manual_instructions()


def test_lafs_preprocess_range():
    x = preprocess(np.zeros((50, 40, 3), np.uint8))
    assert x.shape == (3, 112, 112) and float(x.min()) == -0.5
    assert abs(float(preprocess(np.full((112, 112, 3), 255, np.uint8)).max()) - 0.5) < 1e-6


def test_extract_patches_layout():
    # 画像の各画素に位置を書き込み、ランドマーク中心のパッチが正しい場所から取られることを確認する
    img = torch.arange(112 * 112, dtype=torch.float32).view(1, 1, 112, 112)  # 画素値 = y*112 + x
    row, p = 14, 8
    # 各パッチの中心を格子に置く。align_corners=False では座標 v が画素 v-0.5 を指すため +0.5 して画素中心に合わせる
    centers = torch.tensor([[x * p + p / 2 + 0.5, y * p + p / 2 + 0.5] for y in range(row) for x in range(row)],
                           dtype=torch.float32)[None]  # (1, 196, 2) = (x, y)
    out = extract_patches(img, centers, p)
    assert out.shape == (1, 1, 112, 112)
    # 元実装どおり、パッチ内では行が x 方向・列が y 方向に対応する（転置された並び）: patch[r, c] = img[y=c, x=r]
    patch0 = out[0, 0, :p, :p]
    expect = (torch.arange(p)[None, :] * 112 + torch.arange(p)[:, None]).float()
    assert torch.allclose(patch0, expect, atol=1e-2)  # 浮動小数の誤差（1e-4 程度）を許容
    # 2番目のパッチ（x 方向に 8px 隣）は x が 8 増える
    assert torch.allclose(out[0, 0, :p, p:2 * p], expect + 8, atol=1e-2)
    # 2行目のパッチ（y 方向に 8px 隣）は y が 8 増える（値で 8*112）
    assert torch.allclose(out[0, 0, p:2 * p, :p], expect + 8 * 112, atol=1e-2)


def test_partfvit_forward_shape_and_keys():
    torch.manual_seed(0)
    model = PartFViT().eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 112, 112) - 0.5)
    assert out.shape == (2, 768)
    keys = set(model.state_dict().keys())
    assert "stn.features.15.conv.8.running_var" in keys and "transformer.layers.11.1.fn.fn.net.3.bias" in keys
    assert "output_layer.1.weight" in keys and model.state_dict()["output_layer.1.weight"].shape == (392, 160)
    assert model.state_dict()["transformer.layers.0.0.fn.fn.to_qkv.weight"].shape == (2112, 768)
    assert len(keys) == 432  # 配布チェックポイントの 433 キーから loss.weight を除いた数


def test_load_partfvit_state_dict_strips_prefix(tmp_path):
    sd = {"module.cls_token": torch.zeros(1, 1, 768), "module.loss.weight": torch.zeros(3, 768)}
    p = tmp_path / "x.pth"
    torch.save(sd, p)
    loaded = load_partfvit_state_dict(str(p))
    assert list(loaded) == ["cls_token"]


def test_lafs_real_checkpoint_loads_strictly():
    ex = EXTRACTORS[LAFS]()
    if not ex.weights_ready():
        with pytest.raises(WeightsUnavailable):
            ex.load()
        pytest.skip("LAFS の重みが未配置")
    model = PartFViT()
    model.load_state_dict(load_partfvit_state_dict(str(ex.weights_path())), strict=True)
