"""FSPFM（Pose-invariant face recognition via feature-space pose frontalization, ICIP 2025）のネットワーク定義の移植。

元コード: https://github.com/nikolaystanishev/feature-space-pose-frontalization
  core/model/resnet.py（FaceX-Zoo 由来の ResNet-IR）, core/model/frontalization.py（FSPFM）。
推論に必要な部分を state_dict のキー構造が一致するように移植した（ArcFace の分類ヘッドは省略）。

構成: ResNet50-IR（112x112 入力, 512 次元）が顔特徴量を出し、HopeNet の 2048 次元埋め込み（姿勢）を
2 本の FC ブロックで 512 次元へ写像して、特徴量の残差ブロック出力と掛け合わせる:
  out = pose1 * res1(f) + pose2 * res2(f) + f
"""
from __future__ import annotations

from collections import namedtuple

import torch
from torch import nn


# ---------------------------------------------------------------- ResNet-IR（FaceX-Zoo）
class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class BottleneckIR(nn.Module):
    def __init__(self, in_channel: int, depth: int, stride: int):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                                                nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
        )

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


_Block = namedtuple("Block", ["in_channel", "depth", "stride"])


def _get_block(in_channel: int, depth: int, num_units: int, stride: int = 2) -> list[_Block]:
    return [_Block(in_channel, depth, stride)] + [_Block(depth, depth, 1) for _ in range(num_units - 1)]


_BLOCKS = {
    50: [_get_block(64, 64, 3), _get_block(64, 128, 4), _get_block(128, 256, 14), _get_block(256, 512, 3)],
    100: [_get_block(64, 64, 3), _get_block(64, 128, 13), _get_block(128, 256, 30), _get_block(256, 512, 3)],
    152: [_get_block(64, 64, 3), _get_block(64, 128, 8), _get_block(128, 256, 36), _get_block(256, 512, 3)],
}


class ResnetIR(nn.Module):
    """FaceX-Zoo の Resnet(mode='ir')。112x112 入力 -> feature_dim。"""

    def __init__(self, depth: int = 50, dropout_ratio: float = 0.4, feature_dim: int = 512, out_h: int = 7, out_w: int = 7):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64))
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512), nn.Dropout(dropout_ratio), Flatten(),
                                          nn.Linear(512 * out_h * out_w, feature_dim), nn.BatchNorm1d(feature_dim))
        self.body = nn.Sequential(*[BottleneckIR(b.in_channel, b.depth, b.stride) for blk in _BLOCKS[depth] for b in blk])

    def forward(self, x):
        return self.output_layer(self.body(self.input_layer(x)))


# ---------------------------------------------------------------- FSPFM
class PoseFCBlock(nn.Module):
    def __init__(self, input_feature_dim: int, feature_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_feature_dim, feature_dim)
        self.bn1 = nn.BatchNorm1d(feature_dim)
        self.relu1 = nn.PReLU()
        self.fc2 = nn.Linear(feature_dim, feature_dim)
        self.bn2 = nn.BatchNorm1d(feature_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(self.tanh(x))))
        return self.tanh(self.bn2(self.fc2(x)))


class ResidualFCBlock(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, feature_dim)
        self.bn1 = nn.BatchNorm1d(feature_dim)
        self.relu1 = nn.PReLU()
        self.fc2 = nn.Linear(feature_dim, feature_dim)
        self.bn2 = nn.BatchNorm1d(feature_dim)
        self.relu2 = nn.PReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))
        return self.relu2(self.bn2(self.fc2(x)))


class FeatureSpacePoseFrontalizationModule(nn.Module):
    def __init__(self, pose_input_feat_dim: int = 2048, feature_dim: int = 512):
        super().__init__()
        self.pose_mapping1 = PoseFCBlock(pose_input_feat_dim, feature_dim)
        self.pose_mapping2 = PoseFCBlock(pose_input_feat_dim, feature_dim)
        self.residual_block1 = ResidualFCBlock(feature_dim)
        self.residual_block2 = ResidualFCBlock(feature_dim)

    def forward(self, features, pose):
        return (self.pose_mapping1(pose) * self.residual_block1(features)
                + self.pose_mapping2(pose) * self.residual_block2(features) + features)


def load_fspfm_state_dicts(path: str) -> tuple[dict, dict]:
    """配布チェックポイント（{'state_dict', 'epoch', 'optimizer'}）から (backbone, frontalization_module) の
    state_dict を取り出す。ArcFace ヘッド（head.weight, 93431 クラス）は推論に不要なので捨てる。"""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    backbone = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    front = {k[len("frontalization_module."):]: v for k, v in sd.items() if k.startswith("frontalization_module.")}
    return backbone, front
