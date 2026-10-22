"""MTCNN（Zhang et al. 2016）の PyTorch 実装の移植。

元コード: AdaFace 公式リポジトリ同梱の face_alignment/mtcnn_pytorch（TropComplique/mtcnn-pytorch 由来。
重みは原著の Caffe モデルを変換した pnet/rnet/onet.npy）。P-Net / R-Net / O-Net の 3 段で顔 box と 5 点を出す。
入力は PIL の RGB 画像（元コードどおり PIL でリサイズする）。
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
from PIL import Image


def build_nets(weights_dir):
    """(PNet, RNet, ONet) を .npy の重みを読み込んで返す。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Flatten(nn.Module):
        def forward(self, x):
            x = x.transpose(3, 2).contiguous()  # Caffe の並びに合わせる
            return x.view(x.size(0), -1)

    class PNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(OrderedDict([
                ("conv1", nn.Conv2d(3, 10, 3, 1)), ("prelu1", nn.PReLU(10)), ("pool1", nn.MaxPool2d(2, 2, ceil_mode=True)),
                ("conv2", nn.Conv2d(10, 16, 3, 1)), ("prelu2", nn.PReLU(16)),
                ("conv3", nn.Conv2d(16, 32, 3, 1)), ("prelu3", nn.PReLU(32)),
            ]))
            self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
            self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

        def forward(self, x):
            x = self.features(x)
            return self.conv4_2(x), F.softmax(self.conv4_1(x), dim=-1)

    class RNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(OrderedDict([
                ("conv1", nn.Conv2d(3, 28, 3, 1)), ("prelu1", nn.PReLU(28)), ("pool1", nn.MaxPool2d(3, 2, ceil_mode=True)),
                ("conv2", nn.Conv2d(28, 48, 3, 1)), ("prelu2", nn.PReLU(48)), ("pool2", nn.MaxPool2d(3, 2, ceil_mode=True)),
                ("conv3", nn.Conv2d(48, 64, 2, 1)), ("prelu3", nn.PReLU(64)),
                ("flatten", Flatten()), ("conv4", nn.Linear(576, 128)), ("prelu4", nn.PReLU(128)),
            ]))
            self.conv5_1 = nn.Linear(128, 2)
            self.conv5_2 = nn.Linear(128, 4)

        def forward(self, x):
            x = self.features(x)
            return self.conv5_2(x), F.softmax(self.conv5_1(x), dim=-1)

    class ONet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(OrderedDict([
                ("conv1", nn.Conv2d(3, 32, 3, 1)), ("prelu1", nn.PReLU(32)), ("pool1", nn.MaxPool2d(3, 2, ceil_mode=True)),
                ("conv2", nn.Conv2d(32, 64, 3, 1)), ("prelu2", nn.PReLU(64)), ("pool2", nn.MaxPool2d(3, 2, ceil_mode=True)),
                ("conv3", nn.Conv2d(64, 64, 3, 1)), ("prelu3", nn.PReLU(64)), ("pool3", nn.MaxPool2d(2, 2, ceil_mode=True)),
                ("conv4", nn.Conv2d(64, 128, 2, 1)), ("prelu4", nn.PReLU(128)),
                ("flatten", Flatten()), ("conv5", nn.Linear(1152, 256)), ("drop5", nn.Dropout(0.25)), ("prelu5", nn.PReLU(256)),
            ]))
            self.conv6_1 = nn.Linear(256, 2)
            self.conv6_2 = nn.Linear(256, 4)
            self.conv6_3 = nn.Linear(256, 10)

        def forward(self, x):
            x = self.features(x)
            return self.conv6_3(x), self.conv6_2(x), F.softmax(self.conv6_1(x), dim=-1)

    nets = []
    for cls, fname in ((PNet, "pnet.npy"), (RNet, "rnet.npy"), (ONet, "onet.npy")):
        net = cls()
        weights = np.load(str(weights_dir / fname), allow_pickle=True)[()]
        for n, p in net.named_parameters():
            p.data = torch.FloatTensor(weights[n])
        nets.append(net.eval())
    return tuple(nets)


# ---------------------------------------------------------------- box utils（元コード box_utils.py）
def nms(boxes: np.ndarray, overlap_threshold: float = 0.5, mode: str = "union") -> list[int]:
    if len(boxes) == 0:
        return []
    pick = []
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]
    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    ids = np.argsort(score)
    while len(ids) > 0:
        last = len(ids) - 1
        i = ids[last]
        pick.append(int(i))
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])
        inter = np.maximum(0.0, ix2 - ix1 + 1.0) * np.maximum(0.0, iy2 - iy1 + 1.0)
        if mode == "min":
            overlap = inter / np.minimum(area[i], area[ids[:last]])
        else:
            overlap = inter / (area[i] + area[ids[:last]] - inter)
        ids = np.delete(ids, np.concatenate([[last], np.where(overlap > overlap_threshold)[0]]))
    return pick


def convert_to_square(bboxes: np.ndarray) -> np.ndarray:
    square = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h, w = y2 - y1 + 1.0, x2 - x1 + 1.0
    max_side = np.maximum(h, w)
    square[:, 0] = x1 + w * 0.5 - max_side * 0.5
    square[:, 1] = y1 + h * 0.5 - max_side * 0.5
    square[:, 2] = square[:, 0] + max_side - 1.0
    square[:, 3] = square[:, 1] + max_side - 1.0
    return square


def calibrate_box(bboxes: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = np.expand_dims(x2 - x1 + 1.0, 1)
    h = np.expand_dims(y2 - y1 + 1.0, 1)
    bboxes[:, 0:4] = bboxes[:, 0:4] + np.hstack([w, h, w, h]) * offsets
    return bboxes


def _preprocess(img: np.ndarray) -> np.ndarray:
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    return (img - 127.5) * 0.0078125


def _correct_bboxes(bboxes, width, height):
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w, h = x2 - x1 + 1.0, y2 - y1 + 1.0
    n = bboxes.shape[0]
    x, y, ex, ey = x1, y1, x2, y2
    dx, dy = np.zeros((n,)), np.zeros((n,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0
    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0
    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0
    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0
    return [i.astype("int32") for i in (dy, edy, dx, edx, y, ey, x, ex, w, h)]


def get_image_boxes(bounding_boxes: np.ndarray, img: Image.Image, size: int = 24) -> np.ndarray:
    num_boxes = len(bounding_boxes)
    width, height = img.size
    dy, edy, dx, edx, y, ey, x, ex, w, h = _correct_bboxes(bounding_boxes, width, height)
    img_boxes = np.zeros((num_boxes, 3, size, size), "float32")
    img_array = np.asarray(img, "uint8")
    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3), "uint8")
        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = img_array[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]
        img_box = np.asarray(Image.fromarray(img_box).resize((size, size), Image.BILINEAR), "float32")
        img_boxes[i, :, :, :] = _preprocess(img_box)
    return img_boxes


def _generate_bboxes(probs, offsets, scale, threshold):
    stride, cell_size = 2, 12
    inds = np.where(probs > threshold)
    if inds[0].size == 0:
        return np.array([])
    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]
    return np.vstack([
        np.round((stride * inds[1] + 1.0) / scale), np.round((stride * inds[0] + 1.0) / scale),
        np.round((stride * inds[1] + 1.0 + cell_size) / scale), np.round((stride * inds[0] + 1.0 + cell_size) / scale),
        score, offsets,
    ]).T


def run_first_stage(image: Image.Image, net, scale: float, threshold: float, device):
    import torch
    width, height = image.size
    sw, sh = math.ceil(width * scale), math.ceil(height * scale)
    img = np.asarray(image.resize((sw, sh), Image.BILINEAR), "float32")
    with torch.no_grad():
        output = net(torch.FloatTensor(_preprocess(img)).to(device))
    probs = output[1].cpu().numpy()[0, 1, :, :]
    offsets = output[0].cpu().numpy()
    boxes = _generate_bboxes(probs, offsets, scale, threshold)
    if len(boxes) == 0:
        return None
    return boxes[nms(boxes[:, 0:5], overlap_threshold=0.5)]


class MTCNNDetector:
    """AdaFace の face_alignment/mtcnn.py と同じ 3 段検出（既定値もそのまま）。"""

    def __init__(self, weights_dir, min_face_size: float = 20.0, thresholds=(0.6, 0.7, 0.9),
                 nms_thresholds=(0.7, 0.7, 0.7), factor: float = 0.85):
        import torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pnet, self.rnet, self.onet = (n.to(self.device) for n in build_nets(weights_dir))
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.nms_thresholds = nms_thresholds
        self.factor = factor

    def detect_faces(self, image: Image.Image, thresholds=None) -> tuple[np.ndarray, np.ndarray]:
        """-> (boxes (N,5)=[x1,y1,x2,y2,score], landmarks (N,10)=[x1..x5, y1..y5])。スコア降順。
        thresholds を渡すと 3 段のしきい値をその呼び出しだけ上書きする。"""
        import torch
        thresholds = self.thresholds if thresholds is None else thresholds
        width, height = image.size
        min_length = min(height, width)
        min_detection_size = 12
        m = min_detection_size / self.min_face_size
        min_length *= m
        scales, factor_count = [], 0
        while min_length > min_detection_size:
            scales.append(m * self.factor ** factor_count)
            min_length *= self.factor
            factor_count += 1
        empty = (np.zeros((0, 5), np.float32), np.zeros((0, 10), np.float32))
        bounding_boxes = [b for b in (run_first_stage(image, self.pnet, s, thresholds[0], self.device) for s in scales)
                          if b is not None]
        if not bounding_boxes:
            return empty
        with torch.no_grad():
            bounding_boxes = np.vstack(bounding_boxes)
            bounding_boxes = bounding_boxes[nms(bounding_boxes[:, 0:5], self.nms_thresholds[0])]
            bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
            img_boxes = get_image_boxes(bounding_boxes, image, size=24)
            if len(img_boxes) == 0:
                return empty
            output = self.rnet(torch.FloatTensor(img_boxes).to(self.device))
            offsets, probs = output[0].cpu().numpy(), output[1].cpu().numpy()
            keep = np.where(probs[:, 1] > thresholds[1])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]
            keep = nms(bounding_boxes, self.nms_thresholds[1])
            bounding_boxes = calibrate_box(bounding_boxes[keep], offsets[keep])
            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
            img_boxes = get_image_boxes(bounding_boxes, image, size=48)
            if len(img_boxes) == 0:
                return empty
            output = self.onet(torch.FloatTensor(img_boxes).to(self.device))
            landmarks, offsets, probs = output[0].cpu().numpy(), output[1].cpu().numpy(), output[2].cpu().numpy()
            keep = np.where(probs[:, 1] > thresholds[2])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets, landmarks = offsets[keep], landmarks[keep]
            width_ = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
            height_ = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
            xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
            landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width_, 1) * landmarks[:, 0:5]
            landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height_, 1) * landmarks[:, 5:10]
            bounding_boxes = calibrate_box(bounding_boxes, offsets)
            keep = nms(bounding_boxes, self.nms_thresholds[2], mode="min")
            bounding_boxes, landmarks = bounding_boxes[keep], landmarks[keep]
        if len(bounding_boxes) == 0:
            return empty
        return bounding_boxes.astype(np.float32), landmarks.astype(np.float32)
