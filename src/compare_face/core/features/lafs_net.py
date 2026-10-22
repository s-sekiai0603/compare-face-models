"""Part fViT（LAFS, CVPR 2024）のネットワーク定義の移植。

元コード: https://github.com/szlbiubiubiu/LAFS_CVPR2024 の face_pre_pro/ViT_face.py（ViT_face_landmark_patch8,
with_land=True）と face_pre_pro/mobilenet.py（MobileNetV3_backbone）。推論に必要な部分だけを、
state_dict のキー構造が一致するように移植した（学習用の損失・DropPath・可視化は省略）。

構成: MobileNetV3-Large（stn）が画像から 196 点（14x14）のランドマークを予測し、各点を中心に 8x8 の
パッチを grid_sample で切り出して ViT-Base（dim 768, depth 12, heads 11）のトークンにする。
出力は cls トークンを LayerNorm した 768 次元（未正規化）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- MobileNetV3 backbone（ランドマーク予測）
def _conv_bn(inp: int, oup: int, stride: int, nlin_layer=nn.ReLU) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nlin_layer(inplace=True))


class Hswish(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class Hsigmoid(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class SEModule(nn.Module):
    def __init__(self, channel: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), Hsigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel: int):
        super().__init__()

    def forward(self, x):
        return x


class MobileBottleneck(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int, stride: int, exp: int, se: bool = False, nl: str = "RE"):
        super().__init__()
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup
        nlin_layer = nn.ReLU if nl == "RE" else Hswish
        se_layer = SEModule if se else Identity
        self.conv = nn.Sequential(
            nn.Conv2d(inp, exp, 1, 1, 0, bias=False), nn.BatchNorm2d(exp), nlin_layer(inplace=True),
            nn.Conv2d(exp, exp, kernel, stride, padding, groups=exp, bias=False), nn.BatchNorm2d(exp),
            se_layer(exp), nlin_layer(inplace=True),
            nn.Conv2d(exp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


_MOBILENET_LARGE = [
    # kernel, exp, out, SE, 活性化, stride
    [3, 16, 16, False, "RE", 1], [3, 64, 24, False, "RE", 2], [3, 72, 24, False, "RE", 1],
    [5, 72, 40, True, "RE", 2], [5, 120, 40, True, "RE", 1], [5, 120, 40, True, "RE", 1],
    [3, 240, 80, False, "HS", 2], [3, 200, 80, False, "HS", 1], [3, 184, 80, False, "HS", 1],
    [3, 184, 80, False, "HS", 1], [3, 480, 112, True, "HS", 1], [3, 672, 112, True, "HS", 1],
    [5, 672, 160, True, "HS", 2], [5, 960, 160, True, "HS", 1], [5, 960, 160, True, "HS", 1],
]


class MobileNetV3Backbone(nn.Module):
    """MobileNetV3-Large の特徴抽出部（最終出力 160 チャネル）。"""

    def __init__(self):
        super().__init__()
        layers = [_conv_bn(3, 16, 2, nlin_layer=Hswish)]
        inp = 16
        for k, exp, c, se, nl, s in _MOBILENET_LARGE:
            layers.append(MobileBottleneck(inp, c, k, s, exp, se, nl))
            inp = c
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)


# ---------------------------------------------------------------- Transformer（vit-pytorch 系）
class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5  # 元実装どおり（dim_head ではなく dim で割る）
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads
        q, k, v = (t.view(b, n, h, -1).transpose(1, 2) for t in self.to_qkv(x).chunk(3, dim=-1))  # b h n d
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([Residual(PreNorm(dim, Attention(dim, heads, dim_head, dropout))),
                           Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout)))])
            for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = ff(attn(x))
        return x


# ---------------------------------------------------------------- パッチ抽出と Part fViT 本体
def extract_patches(imgs: torch.Tensor, landmarks: torch.Tensor, patch: int) -> torch.Tensor:
    """各ランドマークを中心に patch x patch の領域を grid_sample で切り出し、(B, C, row*patch, row*patch) に並べる
    （元実装 extract_patches_pytorch_gridsample と同じ座標計算。ランドマークごとのループをまとめて 1 回でサンプリング）。"""
    b, c, img_size, _ = imgs.shape
    n = landmarks.shape[1]
    row = int(math.sqrt(n))
    half = patch / 2
    ar = torch.arange(-half, half, device=landmarks.device, dtype=landmarks.dtype)
    gy, gx = torch.meshgrid(ar, ar, indexing="ij")
    sampling_grid = torch.stack([gy, gx], dim=-1)  # (patch, patch, 2) 元実装の transpose 後と同じ並び
    grid = (sampling_grid[None, None] + landmarks[:, :, None, None, :]) / (img_size * 0.5) - 1  # (B, N, p, p, 2)
    patches = F.grid_sample(imgs, grid.reshape(b, n * patch, patch, 2), align_corners=False)  # (B, C, N*p, p)
    patches = patches.reshape(b, c, row, row, patch, patch).permute(0, 1, 2, 4, 3, 5)
    return patches.reshape(b, c, row * patch, row * patch)


class PartFViT(nn.Module):
    """Part fViT（ランドマーク予測 CNN 内蔵の ViT-Base）。forward は 768 次元の未正規化特徴量を返す。"""

    def __init__(self, image_size: int = 112, patch_size: int = 8, dim: int = 768, depth: int = 12,
                 heads: int = 11, mlp_dim: int = 2048, dim_head: int = 64, channels: int = 3):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.row_num = int(math.sqrt(num_patches))
        self.stn = MobileNetV3Backbone()
        self.output_layer = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(160, num_patches * 2))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(channels * patch_size ** 2, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(0.0)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim))

    def predict_landmarks(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 112, 112) -> (B, 196, 2) のランドマーク（ピクセル座標 [0, 111]）。"""
        theta = self.stn(x).mean(dim=(-2, -1))
        theta = self.output_layer(theta)
        t_max = theta.max(dim=1, keepdim=True)[0]
        t_min = theta.min(dim=1, keepdim=True)[0]
        theta = (theta - t_min) / (t_max - t_min) * (self.image_size - 1)
        return theta.view(-1, self.num_patches, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        p = self.patch_size
        landmarks = self.predict_landmarks(x)
        patches = extract_patches(x, landmarks, p)  # (B, C, row*p, row*p)
        row = self.row_num
        tokens = patches.view(b, -1, row, p, row, p).permute(0, 2, 4, 3, 5, 1).reshape(b, row * row, -1)  # b (h w) (p1 p2 c)
        tokens = self.patch_to_embedding(tokens)
        tokens = torch.cat((self.cls_token.expand(b, -1, -1), tokens), dim=1)
        tokens = tokens + self.pos_embedding[:, :tokens.shape[1]]
        tokens = self.dropout(tokens)
        tokens = self.transformer(tokens)
        return self.mlp_head(tokens[:, 0])


def load_partfvit_state_dict(path: str) -> dict:
    """LAFS 配布チェックポイント（DataParallel で保存、CosFace の分類重みを含む）から backbone の state_dict を取り出す。"""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("loss."):
            continue  # 学習時の分類層（205990 クラス）は推論に不要
        out[k] = v
    return out
