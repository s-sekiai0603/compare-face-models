"""6DRepNet（頭部姿勢）の移植テスト。重み無しで通る。重みがあれば strict 読込と推論も確認する。"""
import numpy as np
import pytest

from compare_face.core.pose import POSE_OPTIONS, SixDRepNetEstimator, get_pose_estimator
from compare_face.core.pose.sixdrepnet import (
    WEIGHTS_FILE, build_sixdrepnet, euler_deg_from_matrix, preprocess, rotation_matrix_from_6d,
)


def test_registry_has_sixdrepnet():
    est = get_pose_estimator("6DRepNet")
    assert isinstance(est, SixDRepNetEstimator) and est.weights_path().name == WEIGHTS_FILE
    assert "HopeNet (robust)" in POSE_OPTIONS and POSE_OPTIONS["なし"] is None
    assert "6DRepNet" in est.name and "150MB" in est.manual_instructions()


def test_network_matches_official_state_dict_layout():
    """公式 deploy 版 6DRepNet_300W_LP_AFLW2000.pth と同じキー・形状（58 キー、RepVGG-B1g2 の groups=2 配置）。"""
    sd = build_sixdrepnet().state_dict()
    assert len(sd) == 58
    assert tuple(sd["layer0.rbr_reparam.weight"].shape) == (64, 3, 3, 3)
    assert tuple(sd["layer1.0.rbr_reparam.weight"].shape) == (128, 64, 3, 3)
    assert tuple(sd["layer1.1.rbr_reparam.weight"].shape) == (128, 64, 3, 3)  # groups=2（入力 128 を 2 分割）
    assert [tuple(sd[f"layer3.{i}.rbr_reparam.weight"].shape)[1] for i in range(4)] == [256, 256, 512, 256]
    assert tuple(sd["layer4.0.rbr_reparam.weight"].shape) == (2048, 512, 3, 3)
    assert tuple(sd["linear_reg.weight"].shape) == (6, 2048)
    assert all(k.endswith((".rbr_reparam.weight", ".rbr_reparam.bias")) or k.startswith("linear_reg.") for k in sd)


def test_preprocess_short_side_resize_and_center_crop():
    x = preprocess(np.full((60, 40, 3), 255, np.uint8))  # 縦長: 幅 40 -> 224、高さ 336 -> 中央 224
    assert x.shape == (3, 224, 224)
    assert np.allclose(x[:, 0, 0], (1.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225]), atol=1e-5)
    img = np.zeros((224, 448, 3), np.uint8)
    img[:, 112:336] = 200  # 中央だけ明るい横長画像 -> 中央切り出しで全面が明るくなる
    y = preprocess(img)
    assert y.shape == (3, 224, 224) and float(y.min()) > 0


def test_rotation_6d_to_euler():
    # 単位行列に対応する 6D 表現 -> 全角度 0
    r = rotation_matrix_from_6d(np.array([[1, 0, 0, 0, 1, 0]], np.float32))
    assert np.allclose(r[0], np.eye(3), atol=1e-6)
    assert np.allclose(euler_deg_from_matrix(r), 0.0, atol=1e-5)
    # y 軸まわりに +30 度回した行列 -> yaw = 30 度（元コードのオイラー角の定義）
    t = np.radians(30.0)
    ry = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    ang = euler_deg_from_matrix(ry[None])
    assert np.allclose(ang[0], [30.0, 0.0, 0.0], atol=1e-4)
    # 正規直交化: 直交していない 6D でも回転行列（直交・行列式 1）になる
    r2 = rotation_matrix_from_6d(np.array([[2.0, 0.5, 0.1, 0.3, 1.0, -0.2]]))
    assert np.allclose(r2[0] @ r2[0].T, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(r2[0]), 1.0)


def test_loose_face_crop_geometry():
    from compare_face.core.pose.sixdrepnet import loose_face_crop
    body = np.zeros((200, 100, 3), np.uint8)
    body[:, :, 0] = np.arange(100, dtype=np.uint8)[None, :]  # 列番号を青に入れて位置を確認できるようにする
    # 全身枠 (50, 100)-(150, 300)、顔枠 (70, 120)-(110, 160) -> 全身画像内では (20, 20)-(60, 60)、幅高さ 40
    crop = loose_face_crop(body, [50, 100, 150, 300], [70, 120, 110, 160])
    assert crop.shape == (int(40 * (1 + 0.6 + 0.25)), int(40 * 1.5), 3)  # 高さ 20-24..60+10=74, 幅 10..70
    assert int(crop[0, 0, 0]) == 10 and int(crop[0, -1, 0]) == 69
    # 全身画像の外に出る分は端を複製する（左上の顔）
    crop2 = loose_face_crop(body, [50, 100, 150, 300], [50, 100, 90, 140])
    assert crop2.shape[1] == 60 and int(crop2[0, 0, 0]) == 0
    assert loose_face_crop(None, [0, 0, 1, 1], [0, 0, 1, 1]) is None
    assert loose_face_crop(body, [0, 0, 1, 1], [5, 5, 5, 5]) is None


def test_estimate_poses_uses_loose_crop_for_sixdrepnet(tmp_path):
    import cv2
    from compare_face.core.report.gallery import GalleryEntry
    from compare_face.core.report.pipeline import estimate_poses
    face = tmp_path / "20260311_120000_000_face.jpg"
    body = tmp_path / "20260311_120000_000_body.jpg"
    cv2.imwrite(str(face), np.full((40, 40, 3), 100, np.uint8))
    cv2.imwrite(str(body), np.full((200, 100, 3), 100, np.uint8))
    e = GalleryEntry(face_path=face, body_path=body, track_dir=tmp_path, camera_id="Cam1", video_name="v", track_id="1",
                     ts="20260311_120000_000", face_bbox_xyxy=[70, 120, 110, 160], person_bbox_xyxy=[50, 100, 150, 300])
    sizes = []

    class Fake:
        prefers_loose_crop = True

        def angles(self, imgs):
            sizes.extend(i.shape[:2] for i in imgs)
            return np.zeros((len(imgs), 3), np.float32)

    estimate_poses(Fake(), [e])
    assert sizes == [(74, 60)] and e.pose == (0.0, 0.0, 0.0)
    e.person_bbox_xyxy = None  # 枠が無ければ顔切り出しをそのまま使う
    sizes.clear()
    estimate_poses(Fake(), [e])
    assert sizes == [(40, 40)]

    class Plain:  # HopeNet 相当（余白広めを求めない）
        def angles(self, imgs):
            sizes.extend(i.shape[:2] for i in imgs)
            return np.zeros((len(imgs), 3), np.float32)

    e.person_bbox_xyxy = [50, 100, 150, 300]
    sizes.clear()
    estimate_poses(Plain(), [e])
    assert sizes == [(40, 40)]


def test_strict_load_and_inference_if_weights_present():
    est = SixDRepNetEstimator()
    if not est.weights_ready():
        pytest.skip("6DRepNet の重みが無い")
    est.load()  # strict=True
    ang = est.angles([np.full((64, 64, 3), 128, np.uint8), np.zeros((50, 70, 3), np.uint8)])
    assert ang.shape == (2, 3) and np.all(np.abs(ang) <= 180.0)
