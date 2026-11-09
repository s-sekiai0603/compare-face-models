"""レンズ歪みの補正（core/camera/distortion.py）。GUI・モデル重みなしで通る。"""
import numpy as np
import pytest

from compare_face.core.camera.distortion import Distortion, estimate_lambda, lambda_range
from compare_face.core.camera.projective import ProjectionError

SIZE = (1920, 1080)


def test_roundtrip_and_identity():
    rng = np.random.default_rng(0)
    pts = rng.uniform([0, 0], SIZE, size=(50, 2))
    for lam in (-0.3, -0.05, 0.0, 0.05):
        d = Distortion(lam, SIZE)
        assert d.to_original(d.to_corrected(pts)) == pytest.approx(pts, abs=1e-6)
    assert Distortion(0.0, SIZE).scale == 1.0
    assert Distortion(0.0, SIZE).is_identity


def test_barrel_pushes_outward_and_scale_fits():
    d = Distortion(-0.3, SIZE)
    assert d.scale < 1.0
    corners = np.array([[0, 0], [1920, 0], [0, 1080], [1920, 1080]], dtype=float)
    out = d.to_corrected(corners)
    assert np.all(out[:, 0] >= -1e-6) and np.all(out[:, 0] <= SIZE[0] + 1e-6)
    assert np.all(out[:, 1] >= -1e-6) and np.all(out[:, 1] <= SIZE[1] + 1e-6)
    # 中心は動かない
    assert d.to_corrected([960.0, 540.0]) == pytest.approx([960.0, 540.0])


def _straight_lines():
    """補正後の座標系での、まっすぐな線（画像の端寄りと斜め）。"""
    lines = []
    for y in (150.0, 900.0):
        lines.append([(x, y) for x in np.linspace(150, 1770, 12)])
    for x in (200.0, 1700.0):
        lines.append([(x, y) for y in np.linspace(100, 980, 10)])
    lines.append([(x, 100 + 0.4 * x) for x in np.linspace(200, 1500, 10)])
    return lines


@pytest.mark.parametrize("true_lam", [-0.25, -0.1, 0.05])
def test_estimate_lambda_recovers_true_value(true_lam):
    d = Distortion(true_lam, SIZE)
    distorted = [d.to_original(np.array(line)) for line in _straight_lines()]
    assert estimate_lambda(distorted, SIZE) == pytest.approx(true_lam, abs=0.005)


def test_estimate_lambda_with_noise_and_few_lines():
    d = Distortion(-0.2, SIZE)
    rng = np.random.default_rng(1)
    lines = [d.to_original(np.array(line)) + rng.normal(0, 0.7, (len(line), 2)) for line in _straight_lines()[:2]]
    assert estimate_lambda(lines, SIZE) == pytest.approx(-0.2, abs=0.03)


def test_estimate_lambda_errors():
    with pytest.raises(ProjectionError):
        estimate_lambda([[(0, 0), (10, 10)]], SIZE)  # 2 点しかない
    with pytest.raises(ProjectionError):
        estimate_lambda([], SIZE)
    lo, hi = lambda_range(SIZE)
    assert lo < 0 < hi


def test_correct_image_makes_curved_line_straight():
    d = Distortion(-0.25, SIZE)
    # 補正後に y = 200 の水平線になる線を、元画像に曲げて描いた画像を補正し直す
    img = np.zeros((SIZE[1], SIZE[0]), dtype=np.uint8)
    xs = np.linspace(100, 1820, 4000)
    curve = d.to_original(np.stack([xs, np.full_like(xs, 200.0)], axis=1))
    for x, y in curve:
        if 0 <= x < SIZE[0] and 0 <= y < SIZE[1]:
            img[int(y), int(x)] = 255
    out = d.correct_image(img)
    assert out.shape == img.shape
    rows = np.nonzero(out[:, 400:1500].max(axis=1))[0]
    assert rows.min() >= 197 and rows.max() <= 203
    assert Distortion(0.0, SIZE).correct_image(img) is img
