"""RetinaFace（ResNet-50, Pytorch_Retinaface）による 5 点検出と、insightface の norm_crop と同じアライン。

MS1M-RetinaFace（FSPFM の学習データ）や WebFace4M/12M（AdaFace, LAFS, CVLFace の学習データ）は RetinaFace の
5 点で整列されている。ここでは広く使われる PyTorch 再実装 https://github.com/biubug6/Pytorch_Retinaface の
ResNet-50 版（WIDER FACE 学習, 5 点出力）を推論部分だけ移植し、重みは facexlib の GitHub Releases
（Pytorch_Retinaface と同一の Resnet50_Final.pth）から自動取得する。
"""
from __future__ import annotations

from math import ceil
from typing import Callable

import cv2
import numpy as np

from ..common.weights import LANDMARK_DIR
from .base import AlignResult, FaceAligner, WeightsUnavailable, register
from .insightface_scrfd import estimate_norm, nms

RETINAFACE_DIR = LANDMARK_DIR / "retinaface"
WEIGHTS_FILE = "detection_Resnet50_Final.pth"
WEIGHTS_URL = "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"
MIN_SIZES = ((16, 32), (64, 128), (256, 512))
STEPS = (8, 16, 32)
VARIANCES = (0.1, 0.2)
MEAN_BGR = (104.0, 117.0, 123.0)
CONF_THRESH = 0.6         # 元コードの可視化しきい値 vis_thres
CONF_THRESH_RETRY = 0.05  # 見つからないとき（入力は顔と確定した切り出し）
NMS_THRESH = 0.4
TOP_K = 5000
KEEP_TOP_K = 750
PAD_FRACTION = 0.5        # 顔が全面を占める切り出しの周囲に、各辺この割合のグレー余白を足してから検出する
MAX_SIDE = 1024           # 余白込みの長辺がこれを超えたら縮小（計算量の上限）
PAD_FILL = 114


def build_retinaface_r50():
    """Pytorch_Retinaface の RetinaFace(cfg_re50, phase='test') と同じキー構造のモジュール。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    from torchvision.models._utils import IntermediateLayerGetter

    def conv_bn(inp, oup, stride=1, leaky=0.0):
        return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup),
                             nn.LeakyReLU(negative_slope=leaky, inplace=True))

    def conv_bn_no_relu(inp, oup, stride):
        return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup))

    def conv_bn1x1(inp, oup, stride, leaky=0.0):
        return nn.Sequential(nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False), nn.BatchNorm2d(oup),
                             nn.LeakyReLU(negative_slope=leaky, inplace=True))

    class SSH(nn.Module):
        def __init__(self, in_channel, out_channel):
            super().__init__()
            leaky = 0.1 if out_channel <= 64 else 0.0
            self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)
            self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
            self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)
            self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
            self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        def forward(self, x):
            c3 = self.conv3X3(x)
            c5_1 = self.conv5X5_1(x)
            c5 = self.conv5X5_2(c5_1)
            c7 = self.conv7x7_3(self.conv7X7_2(c5_1))
            return F.relu(torch.cat([c3, c5, c7], dim=1))

    class FPN(nn.Module):
        def __init__(self, in_channels_list, out_channels):
            super().__init__()
            leaky = 0.1 if out_channels <= 64 else 0.0
            self.output1 = conv_bn1x1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
            self.output2 = conv_bn1x1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
            self.output3 = conv_bn1x1(in_channels_list[2], out_channels, stride=1, leaky=leaky)
            self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
            self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

        def forward(self, inputs):
            inputs = list(inputs.values())
            o1, o2, o3 = self.output1(inputs[0]), self.output2(inputs[1]), self.output3(inputs[2])
            o2 = self.merge2(o2 + F.interpolate(o3, size=o2.shape[2:], mode="nearest"))
            o1 = self.merge1(o1 + F.interpolate(o2, size=o1.shape[2:], mode="nearest"))
            return [o1, o2, o3]

    class Head(nn.Module):
        def __init__(self, inchannels, num_anchors, out_per_anchor):
            super().__init__()
            self.out_per_anchor = out_per_anchor
            self.conv1x1 = nn.Conv2d(inchannels, num_anchors * out_per_anchor, kernel_size=(1, 1), stride=1, padding=0)

        def forward(self, x):
            out = self.conv1x1(x).permute(0, 2, 3, 1).contiguous()
            return out.view(out.shape[0], -1, self.out_per_anchor)

    class RetinaFace(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = torchvision.models.resnet50(weights=None)
            self.body = IntermediateLayerGetter(backbone, {"layer2": 1, "layer3": 2, "layer4": 3})
            in_channels_list, out_channels = [512, 1024, 2048], 256
            self.fpn = FPN(in_channels_list, out_channels)
            self.ssh1 = SSH(out_channels, out_channels)
            self.ssh2 = SSH(out_channels, out_channels)
            self.ssh3 = SSH(out_channels, out_channels)
            self.ClassHead = nn.ModuleList([Head(out_channels, 2, 2) for _ in range(3)])
            self.BboxHead = nn.ModuleList([Head(out_channels, 2, 4) for _ in range(3)])
            self.LandmarkHead = nn.ModuleList([Head(out_channels, 2, 10) for _ in range(3)])

        def forward(self, x):
            fpn = self.fpn(self.body(x))
            feats = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]
            bbox = torch.cat([self.BboxHead[i](f) for i, f in enumerate(feats)], dim=1)
            cls = torch.cat([self.ClassHead[i](f) for i, f in enumerate(feats)], dim=1)
            ldm = torch.cat([self.LandmarkHead[i](f) for i, f in enumerate(feats)], dim=1)
            return bbox, F.softmax(cls, dim=-1), ldm

    return RetinaFace()


def prior_boxes(image_h: int, image_w: int) -> np.ndarray:
    """(N, 4) の [cx, cy, w, h]（画像サイズで正規化）。元コードの PriorBox と同じ並び
    （ストライド順 → 行 → 列 → min_size 順）。numpy でまとめて生成する（元コードの Python ループは 1 枚 20ms 超かかる）。"""
    parts = []
    for k, step in enumerate(STEPS):
        fh, fw = ceil(image_h / step), ceil(image_w / step)
        ii, jj = np.meshgrid(np.arange(fh), np.arange(fw), indexing="ij")  # (fh, fw): 行 → 列の順
        cx = ((jj + 0.5) * step / image_w).reshape(-1)
        cy = ((ii + 0.5) * step / image_h).reshape(-1)
        sizes = np.asarray(MIN_SIZES[k], np.float64)
        n_cell, n_size = len(cx), len(sizes)
        block = np.empty((n_cell, n_size, 4), np.float64)
        block[:, :, 0] = cx[:, None]
        block[:, :, 1] = cy[:, None]
        block[:, :, 2] = (sizes / image_w)[None, :]
        block[:, :, 3] = (sizes / image_h)[None, :]
        parts.append(block.reshape(-1, 4))
    return np.concatenate(parts).astype(np.float32)


def decode_boxes(loc: np.ndarray, priors: np.ndarray) -> np.ndarray:
    boxes = np.concatenate([priors[:, :2] + loc[:, :2] * VARIANCES[0] * priors[:, 2:],
                            priors[:, 2:] * np.exp(loc[:, 2:] * VARIANCES[1])], axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landmarks(pre: np.ndarray, priors: np.ndarray) -> np.ndarray:
    return np.concatenate([priors[:, :2] + pre[:, 2 * i:2 * i + 2] * VARIANCES[0] * priors[:, 2:] for i in range(5)], axis=1)


def load_retinaface_state_dict(torch, path) -> dict:
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}


class RetinaFaceDetector:
    def __init__(self, weights_path):
        import torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = build_retinaface_r50()
        model.load_state_dict(load_retinaface_state_dict(torch, weights_path), strict=True)
        self.model = model.eval().to(self.device)

    def detect(self, img_bgr: np.ndarray, conf_thresh: float = CONF_THRESH) -> tuple[np.ndarray, np.ndarray]:
        """-> (dets (N,5)=[x1,y1,x2,y2,score], landmarks (N,5,2))。入力画像の座標系（元コード detect.py と同じ手順）。"""
        import torch
        h, w = img_bgr.shape[:2]
        x = (img_bgr.astype(np.float32) - np.array(MEAN_BGR, np.float32)).transpose(2, 0, 1)[None]
        with torch.no_grad():
            loc, conf, ldm = self.model(torch.from_numpy(np.ascontiguousarray(x)).to(self.device))
        priors = prior_boxes(h, w)
        boxes = decode_boxes(loc[0].cpu().numpy(), priors) * np.array([w, h, w, h], np.float32)
        scores = conf[0].cpu().numpy()[:, 1]
        landms = decode_landmarks(ldm[0].cpu().numpy(), priors) * np.array([w, h] * 5, np.float32)
        inds = np.where(scores > conf_thresh)[0]
        boxes, landms, scores = boxes[inds], landms[inds], scores[inds]
        order = scores.argsort()[::-1][:TOP_K]
        boxes, landms, scores = boxes[order], landms[order], scores[order]
        if len(scores) == 0:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        dets = np.hstack((boxes, scores[:, None])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)[:KEEP_TOP_K]
        return dets[keep], landms[keep].reshape(-1, 5, 2)


def pad_for_detection(img_bgr: np.ndarray, fraction: float = PAD_FRACTION,
                      max_side: int = MAX_SIDE) -> tuple[np.ndarray, float, int, int]:
    """周囲にグレー余白を足し、長辺が max_side を超えれば縮小する。戻り値 (画像, 縮小率, 余白x, 余白y)。"""
    h, w = img_bgr.shape[:2]
    pad_x, pad_y = int(w * fraction), int(h * fraction)
    padded = cv2.copyMakeBorder(img_bgr, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(PAD_FILL,) * 3)
    scale = 1.0
    long_side = max(padded.shape[:2])
    if long_side > max_side:
        scale = max_side / long_side
        padded = cv2.resize(padded, (max(1, int(padded.shape[1] * scale)), max(1, int(padded.shape[0] * scale))))
    return padded, scale, pad_x, pad_y


@register
class RetinaFaceR50(FaceAligner):
    name = "RetinaFace-R50 (Pytorch_Retinaface)"
    description = ("RetinaFace ResNet-50（WIDER FACE 学習）で 5 点を検出し、insightface の norm_crop と同じ相似変換でアライン。"
                   "MS1M-RetinaFace / WebFace の整列に使われた RetinaFace の PyTorch 再実装。初回は facexlib の配布重み（109MB）を自動取得")

    def __init__(self):
        self.detector: RetinaFaceDetector | None = None

    def weights_path(self):
        return RETINAFACE_DIR / WEIGHTS_FILE

    def weights_ready(self) -> bool:
        return self.weights_path().is_file()

    def manual_instructions(self) -> str:
        return (
            f"{self.name} の重み {WEIGHTS_FILE} が見つかりません。画面を開いたときに自動取得します（約 109MB）。\n"
            f"手動で用意する場合は {WEIGHTS_URL} をダウンロードし、次の場所に配置してください。\n   {self.weights_path()}"
        )

    def download_weights(self, log: Callable[[str], None] = print) -> None:
        if self.weights_ready():
            return
        import urllib.request
        self.weights_path().parent.mkdir(parents=True, exist_ok=True)
        log(f"{self.name} の重みをダウンロードしています: {WEIGHTS_URL}")
        tmp = self.weights_path().with_suffix(".pth.part")
        try:
            urllib.request.urlretrieve(WEIGHTS_URL, str(tmp))
            if tmp.stat().st_size < 100_000_000:
                raise RuntimeError("ダウンロード結果が不正です（サイズ不足）")
            tmp.replace(self.weights_path())
        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise WeightsUnavailable(f"{self.manual_instructions()}\n\n（自動取得エラー: {type(e).__name__}: {e}）") from e

    def load(self) -> None:
        if self.detector is not None:
            return
        if not self.weights_ready():
            raise WeightsUnavailable(self.manual_instructions())
        import torch
        self.detector = RetinaFaceDetector(self.weights_path())
        dev = f"GPU ({self.detector.device}, {torch.cuda.get_device_name(0)})" if self.detector.device != "cpu" else "CPU"
        print(f"[推論環境] 顔ランドマーク: デバイス: {dev} / フレームワーク: PyTorch / モデル: {self.name} ({WEIGHTS_FILE})", flush=True)

    def detect_main_face(self, face_bgr: np.ndarray) -> np.ndarray | None:
        """切り出し内で最もスコアの高い顔の 5 点（切り出し座標）。見つからなければ None。"""
        self.load()
        padded, scale, pad_x, pad_y = pad_for_detection(face_bgr)
        dets, landms = self.detector.detect(padded, CONF_THRESH)
        if len(dets) == 0:
            dets, landms = self.detector.detect(padded, CONF_THRESH_RETRY)
        if len(dets) == 0:
            return None
        pts = landms[int(np.argmax(dets[:, 4]))] / scale - np.array([pad_x, pad_y], np.float32)
        return pts.astype(np.float32)

    def align_with_landmarks(self, face_bgr: np.ndarray, out_size: int) -> AlignResult | None:
        if face_bgr is None or face_bgr.size == 0:
            return None
        pts = self.detect_main_face(face_bgr)
        if pts is None or not np.all(np.isfinite(pts)):
            return None
        m = estimate_norm(pts, out_size)
        img = cv2.warpAffine(face_bgr, m, (out_size, out_size), borderValue=0.0)
        return AlignResult(img, (pts @ m[:, :2].T + m[:, 2]).astype(np.float32))

    def align(self, face_bgr: np.ndarray, out_size: int) -> np.ndarray | None:
        r = self.align_with_landmarks(face_bgr, out_size)
        return None if r is None else r.image
