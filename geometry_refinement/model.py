"""Predict language-conditioned grasp centers and poses on a stride-4 grid.

Use `build` to initialize the network with the training vocabulary size, `loss`
to supervise it from grasp annotations, and `decode` to produce oriented
rectangles. At 160 x 160 pixels the grid has 40 x 40 cells.

在步长为 4 的网格上，根据指令预测抓取中心和姿态。
`build` 根据训练词表大小初始化网络，`loss` 使用抓取标注进行监督，
`decode` 将输出转换为带角度的矩形。输入为 160 × 160 像素时，输出网格为 40 × 40。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _conv(in_channels, out_channels, stride=1):
    """Use a 3 x 3 convolution, eight-group normalization, and SiLU.

    使用 3 × 3 卷积、8 组归一化及 SiLU 激活。
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
    )


class DenseGrasp(nn.Module):
    """Fuse fine and coarse visual features conditioned on one prompt per image.

    给每张图像配一条指令，用指令调制并融合细尺度和粗尺度视觉特征。
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.text_context = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.text_projection = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 96), nn.SiLU())

        self.stem = nn.Sequential(_conv(3, 32, stride=2), _conv(32, 48, stride=2))
        self.coarse = nn.Sequential(
            _conv(48, 64, stride=2),
            _conv(64, 96, stride=2),
            _conv(96, 96),
        )
        self.fine_film = nn.Linear(96, 2 * 48)
        self.coarse_film = nn.Linear(96, 2 * 96)
        self.coarse_projection = nn.Conv2d(96, 48, 1, bias=False)
        self.fusion = nn.Sequential(_conv(48, 48), _conv(48, 48))
        self.quality = nn.Conv2d(48, 1, 1)
        self.geometry = nn.Conv2d(48, 6, 1)
        # Start with a low center probability and a small opening. The near-zero
        # FiLM parameters keep both image branches close to identity conditioning.
        # 中心概率和开口尺寸从较小值起步。接近零的 FiLM 参数让两路图像特征
        # 在训练初期近似保持原值。
        nn.init.constant_(self.quality.bias, math.log(0.05 / 0.95))
        nn.init.constant_(self.geometry.bias[2:4], math.log(0.20 / 0.80))
        for film in (self.fine_film, self.coarse_film):
            nn.init.normal_(film.weight, std=0.01)
            nn.init.zeros_(film.bias)

    @staticmethod
    def _condition(features, parameters):
        """Broadcast a [B, 2C] vector as per-channel scale and bias over [B, C, H, W].

        把 [B, 2C] 向量拆成逐通道缩放系数和偏置，广播到 [B, C, H, W]。
        """
        scale, bias = parameters[:, :, None, None].chunk(2, dim=1)
        return features * (1.0 + scale) + bias

    def forward(self, images, tokens):
        """Map RGB [B, 3, H, W] in [0, 1] and token IDs [B, T] to dense predictions.

        Return center logits [B, H/4, W/4] and raw pose channels
        [B, 6, H/4, W/4] for image sizes divisible by four.

        输入像素位于 [0, 1] 的 RGB 张量 [B, 3, H, W] 和词元 ID [B, T]。
        当图像尺寸能被 4 整除时，返回中心分数的 logits [B, H/4, W/4]
        和原始姿态通道 [B, 6, H/4, W/4]。
        """
        # Convolve along the token sequence and pool only the non-padding positions.
        # 沿词元序列执行卷积，池化时只计入非填充位置。
        token_mask = tokens.ne(0).unsqueeze(1)
        words = self.embedding(tokens).transpose(1, 2)
        words = words + F.silu(self.text_context(words))
        pooled = (words * token_mask).sum(-1) / token_mask.sum(-1).clamp(min=1)
        text = self.text_projection(pooled)

        # At 160 x 160, the stem gives [B, 48, 40, 40] and the coarse branch
        # gives [B, 96, 10, 10]. Project and resize the latter before adding it.
        # 输入为 160 × 160 时，浅层分支得到 [B, 48, 40, 40]，粗尺度分支得到
        # [B, 96, 10, 10]。后者先投影并上采样，再与细特征相加。
        fine = self.stem(images * 2.0 - 1.0)
        fine = self._condition(fine, self.fine_film(text))
        coarse = self._condition(self.coarse(fine), self.coarse_film(text))
        coarse = F.interpolate(
            self.coarse_projection(coarse), size=fine.shape[-2:], mode="bilinear", align_corners=False
        )
        features = self.fusion(fine + coarse)
        return self.quality(features).squeeze(1), self.geometry(features)


def build(vocab_size):
    """Initialize the model using the vocabulary dimension saved by training.

    使用训练时确定的词表维度初始化模型。
    """
    return DenseGrasp(vocab_size)


def _grid(quality):
    """Return [H, W, 2] cell coordinates in (x, y) order and the (W, H) scale.

    返回按 (x, y) 排列的网格坐标 [H, W, 2]，以及缩放尺度 (W, H)。
    """
    height, width = quality.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=quality.device, dtype=torch.float32),
        torch.arange(width, device=quality.device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack((x, y), dim=-1), quality.new_tensor((width, height), dtype=torch.float32)


def _unpack_geometry(geometry):
    """Read the six channels as two local offsets, two sizes, and a 2D direction.

    Offsets are in cell units and sizes in units of the normalized image. The
    last pair is trained to represent cosine and sine of twice the angle.

    六个通道依次表示两个局部偏移、两个尺寸和两维方向。偏移以网格为单位，
    尺寸以归一化后的图像为单位。最后两个值分别拟合两倍角的余弦和正弦。
    """
    geometry = geometry.float().movedim(1, -1)
    return geometry[..., :2].tanh() * 0.5, geometry[..., 2:4].sigmoid(), geometry[..., 4:]


def loss(output, target, valid):
    """Supervise centers and positive-cell geometry from annotated rectangles.

    `target` is [B, G, 5] with normalized center and size and angle in radians.
    The boolean `valid` mask [B, G] distinguishes annotations from padding.

    用抓取矩形标注监督中心分数，并在正样本网格上回归矩形几何。
    `target` 为 [B, G, 5]，含归一化中心、宽高和弧度角。布尔张量
    `valid` 为 [B, G]，标记哪些元素是实际标注。
    """
    quality, geometry = output
    quality = quality.float()
    grid, scale = _grid(quality)

    # Keep one masked row so the reductions also work for an empty target set.
    # 空标注集用一条无效记录占位，使后续沿标注维度的归约仍可执行。
    if target.shape[1] == 0:
        target = target.new_zeros((target.shape[0], 1, 5))
        valid = valid.new_zeros((valid.shape[0], 1))
    target = torch.where(valid[..., None], target.float(), 0.0)
    centers = target[..., :2].clamp(0.0, 1.0) * scale
    cells = torch.minimum(centers.floor(), scale - 1.0)
    x_dist = grid[None, None, ..., 0] - cells[..., 0, None, None]
    y_dist = grid[None, None, ..., 1] - cells[..., 1, None, None]
    distance = x_dist.square() + y_dist.square()
    distance.masked_fill_(~valid[..., None, None], torch.inf)

    # Break same-cell ties by choosing the annotation closest to the cell center.
    # 多条标注落入同一网格时，匹配中心最接近该格中心的标注。
    residual = (centers - cells - 0.5).square().sum(-1)
    nearest = (distance + residual[..., None, None]).argmin(dim=1, keepdim=True)
    nearest_distance = distance.gather(1, nearest).squeeze(1)
    positive = nearest_distance == 0.0
    heatmap = (-nearest_distance / (2 * 1.25**2)).exp()
    count = positive.sum().clamp(min=1)

    # Focal-style loss downweights easy cells and negatives near a labeled center.
    # 焦点损失降低易分类网格以及标注中心附近负样本网格的权重。
    probability = quality.sigmoid()
    positive_loss = F.softplus(-quality) * (1.0 - probability).square()
    negative_loss = F.softplus(quality) * probability.square() * (1.0 - heatmap).pow(4)
    center_loss = torch.where(positive, positive_loss, negative_loss).sum() / count

    batch, height, width = quality.shape
    indices = nearest.reshape(batch, height * width, 1).expand(-1, -1, 5)
    matched = target.gather(1, indices).reshape(batch, height, width, 5)
    wanted_offset = matched[..., :2].clamp(0.0, 1.0) * scale - grid - 0.5
    wanted_angle = matched[..., 4] * 2.0
    wanted_direction = torch.stack((wanted_angle.cos(), wanted_angle.sin()), dim=-1)

    # Smooth L1 compares normalized geometry at positive cells. Doubling the angle
    # makes parallel-jaw poses at angles separated by pi equivalent.
    # Smooth L1 只对正样本网格的几何参数计入最终损失。使用两倍角可将
    # 相差 π 的两个平行夹爪方向视为同一姿态。
    offset, size, direction = _unpack_geometry(geometry)
    center_offset_loss = F.smooth_l1_loss(offset, wanted_offset, reduction="none", beta=0.1)
    size_loss = F.smooth_l1_loss(size, matched[..., 2:4], reduction="none", beta=0.05)
    angle_loss = F.smooth_l1_loss(direction, wanted_direction, reduction="none", beta=0.2)
    pose_loss = center_offset_loss + 4.0 * size_loss + angle_loss
    return center_loss + (pose_loss.sum(-1) * positive).sum() / count


def decode(output, topk=5):
    """Return up to K ranked rectangles [B, K, 5] in normalized coordinates.

    The score favors 3 x 3 local maxima, with a softer penalty for other cells.

    返回最多 K 个按分数排序的归一化抓取矩形 [B, K, 5]。
    保留 3 × 3 局部峰值的原分数，对其余位置降权。
    """
    quality, geometry = output
    grid, scale = _grid(quality)
    offset, size, direction = _unpack_geometry(geometry)
    center = ((grid + 0.5 + offset) / scale).clamp(0.0, 1.0)
    angle = 0.5 * torch.atan2(direction[..., 1], direction[..., 0])
    poses = torch.cat((center, size, angle.unsqueeze(-1)), dim=-1).flatten(1, 2)

    probability = quality.float().sigmoid()
    local_maximum = F.max_pool2d(probability.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    scores = torch.where(probability == local_maximum, probability, probability * 0.2).flatten(1)
    indices = scores.topk(min(topk, scores.shape[1]), dim=1).indices
    return poses.gather(1, indices[..., None].expand(-1, -1, 5))
