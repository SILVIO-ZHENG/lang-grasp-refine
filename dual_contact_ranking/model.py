"""Predict pairs of gripper contacts from an RGB scene and a grasp instruction.

The dense heads propose an opening and orientation at each image cell. The
decoder converts the contact pair into the shared five-value grasp rectangle.

从场景 RGB 图像和抓取指令预测夹爪的两个接触点。

密集预测头为每个网格位置提供抓取开口和方向。解码器把接触点对转换成与其他模型
共用的五参数抓取矩形。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _conv(in_channels, out_channels, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=dilation, dilation=dilation, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
    )


def _grid(heatmap):
    """Return cell coordinates in (x, y) order and their (width, height) scale.

    返回按 (x, y) 排列的网格坐标，以及对应的 (宽, 高) 归一化尺度。
    """
    height, width = heatmap.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=heatmap.device, dtype=torch.float32),
        torch.arange(width, device=heatmap.device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack((x, y), dim=-1), heatmap.new_tensor((width, height), dtype=torch.float32)


class ContactGrasp(nn.Module):
    """Fuse instruction features with a fine image map and a coarse context map.

    将指令特征融合到图像细节特征图和低分辨率上下文特征图中。
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.text = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.word_context = nn.Conv1d(64, 64, 3, padding=1)
        self.text_query = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 64), nn.SiLU())

        self.detail = nn.Sequential(_conv(3, 32, stride=2), _conv(32, 48, stride=2))
        self.context = nn.Sequential(_conv(48, 64, stride=2), _conv(64, 64, dilation=2))
        self.visual_key = nn.Conv2d(64, 64, 1, bias=False)
        self.context_projection = nn.Conv2d(64, 48, 1)
        self.fusion = nn.Sequential(_conv(48, 48), _conv(48, 48))

        self.center = nn.Conv2d(48, 1, 1)
        self.endpoints = nn.Conv2d(48, 4, 1)
        self.thickness = nn.Conv2d(48, 1, 1)
        nn.init.constant_(self.center.bias, math.log(0.05 / 0.95))
        nn.init.constant_(self.thickness.bias, math.log(0.14 / 0.86))
        nn.init.normal_(self.endpoints.weight, std=0.01)
        with torch.no_grad():
            self.endpoints.bias.copy_(torch.tensor([-0.08, 0.0, 0.08, 0.0]))

    def forward(self, images, tokens):
        """Map RGB [B,3,H,W] in [0,1] and padded token IDs [B,L] to dense proposals.

        A 160-by-160 input produces 40-by-40 maps of center logits and normalized
        side lengths, plus contacts [B,40,40,2,2] indexed by point and (x, y).

        输入为取值在 [0,1] 的 RGB 张量 [B,3,H,W] 和补齐后的词元编号 [B,L]。

        160×160 的图像输出 40×40 的中心 logits 和归一化边长图，以及
        [B,40,40,2,2] 的接触点张量，最后两维依次表示点的编号和 (x, y)。
        """
        words = self.text(tokens).transpose(1, 2)
        words = words + F.silu(self.word_context(words))
        # Pool the contextualized word embeddings over non-padding positions.
        # 只对非补齐位置的上下文词向量做平均池化。
        present = tokens.ne(0).unsqueeze(1)
        query = self.text_query((words * present).sum(-1) / present.sum(-1).clamp_min(1))

        detail = self.detail(images * 2.0 - 1.0)
        context = self.context(detail)
        query = query[..., None, None]
        # A scaled image-key/text-query dot product gates each coarse cell.
        # Project the gated context to 48 channels before adding it to the detail map.
        # 对图像键和文本查询做缩放点积，得到控制各粗网格的门控权重。
        # 将门控后的上下文投影到 48 个通道，再与细节特征图相加。
        relevance = (self.visual_key(context) * query).sum(1, keepdim=True) * 0.125
        context = self.context_projection(context * relevance.sigmoid() + query)
        detail = self.fusion(detail + F.interpolate(context, detail.shape[-2:], mode="bilinear", align_corners=False))

        heatmap = self.center(detail).squeeze(1)
        cells, scale = _grid(heatmap)
        raw = self.endpoints(detail).float().movedim(1, -1).unflatten(-1, (2, 2))
        raw_midpoint = raw.mean(dim=-2, keepdim=True)
        # Keep the contact midpoint inside its grid cell. The endpoints can lie
        # outside that cell for wider openings. These four channels encode two points.
        # 把接触点中点限制在当前网格内。夹爪开口较宽时，端点可以跨出该网格。
        # 这四个通道表示两个点的坐标。
        midpoint = (cells.unsqueeze(-2) + 0.5 + 0.5 * raw_midpoint.tanh()) / scale
        contacts = midpoint + 0.65 * (raw - raw_midpoint).tanh()
        return heatmap, contacts, self.thickness(detail).squeeze(1).float().sigmoid()


def build(vocab_size):
    return ContactGrasp(vocab_size)


def loss(output, target, valid):
    """Combine focal center confidence with point, midpoint, and side-length losses.

    Targets have shape [B,N,5] and store normalized (x, y, width, height) plus
    angle in radians. The Boolean [B,N] mask marks annotations for each instruction.

    综合中心置信度的焦点损失，以及接触点、中点和边长的回归损失。

    目标张量 [B,N,5] 包含归一化的 (x, y, 宽, 高) 和以弧度表示的角度。
    布尔掩码 [B,N] 标记每条指令对应的有效标注。
    """
    logits, contacts, thickness = output
    logits = logits.float()
    grid, scale = _grid(logits)
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
    # Choose the closest annotated cell, using the subcell offset to break ties.
    # Exact center cells are positives. The Gaussian target downweights nearby negatives.
    # 对每个网格选择最近的标注网格，距离相同时由标注在格内的偏移打破平局。
    # 标注中心所在网格是正样本。高斯目标降低其邻近负样本的权重。
    residual = (centers - cells - 0.5).square().sum(-1)
    nearest = (distance + residual[..., None, None]).argmin(dim=1, keepdim=True)
    nearest_distance = distance.gather(1, nearest).squeeze(1)
    positive = nearest_distance == 0.0
    heatmap = (-nearest_distance / (2.0 * 1.25**2)).exp()
    count = positive.sum().clamp_min(1)

    probability = logits.sigmoid()
    positive_loss = F.softplus(-logits) * (1.0 - probability).square()
    negative_loss = F.softplus(logits) * probability.square() * (1.0 - heatmap).pow(4)
    confidence_loss = torch.where(positive, positive_loss, negative_loss).sum() / count

    batch, height, width = logits.shape
    indices = nearest.reshape(batch, -1, 1).expand(-1, -1, 5)
    matched = target.gather(1, indices).reshape(batch, height, width, 5)
    center = matched[..., :2]
    theta = matched[..., 4]
    direction = torch.stack((theta.cos(), theta.sin()), dim=-1)
    half_width = 0.5 * matched[..., 2:3] * direction
    desired = torch.stack((center - half_width, center + half_width), dim=-2)

    # The two jaws have no assigned order. Take the smaller Smooth L1 error
    # across both endpoint assignments and apply geometry loss at positive cells.
    # 两个夹爪端点没有固定顺序。比较两种点对匹配的 Smooth L1 误差，
    # 取较小值，并且只在正样本网格计算几何回归损失。
    error = F.smooth_l1_loss(contacts, desired, beta=0.025, reduction="none").sum(dim=(-1, -2))
    swapped = F.smooth_l1_loss(contacts, desired.flip(-2), beta=0.025, reduction="none").sum(dim=(-1, -2))
    contact_loss = torch.minimum(error, swapped)
    center_loss = F.smooth_l1_loss(contacts.mean(-2) * scale, center * scale, beta=0.1, reduction="none").sum(-1)
    thickness_loss = F.smooth_l1_loss(thickness, matched[..., 3], beta=0.05, reduction="none")
    regression = center_loss + 4.0 * contact_loss + 4.0 * thickness_loss
    return confidence_loss + (regression * positive).sum() / count


def decode(output, topk=5):
    """Return [B,K,5] rectangles ranked by local center confidence.

    Use the pair midpoint, distance, and direction for the center, opening width,
    and angle. The thickness head supplies the other side. Allow up to `topk` poses.

    按中心置信度返回 [B,K,5] 的抓取矩形。

    接触点对的中点、距离和方向分别给出中心、开口宽度和角度。另一条边长由
    边长预测头给出。最多保留 `topk` 个候选。
    """
    logits, contacts, thickness = output
    center = contacts.mean(-2).clamp(0.0, 1.0)
    width_vector = contacts[..., 1, :] - contacts[..., 0, :]
    width = width_vector.norm(dim=-1).clamp_min(1e-4)
    angle = torch.atan2(width_vector[..., 1], width_vector[..., 0])
    angle = (angle + math.pi / 2).remainder(math.pi) - math.pi / 2
    poses = torch.cat((center, torch.stack((width, thickness, angle), dim=-1)), dim=-1).flatten(1, 2)

    probability = logits.float().sigmoid()
    # Prefer 3-by-3 local peaks while leaving other cells eligible at a lower score.
    # 优先选择 3×3 邻域峰值，其余网格保留为低分候选。
    peaks = F.max_pool2d(probability.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    scores = torch.where(probability == peaks, probability, probability * 0.2).flatten(1)
    indices = scores.topk(min(topk, scores.shape[1]), dim=1).indices
    return poses.gather(1, indices[..., None].expand(-1, -1, 5))
