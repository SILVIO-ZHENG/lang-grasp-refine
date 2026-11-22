"""Predict language-conditioned grasp centers and poses on a stride-4 grid."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _conv(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
    )


class DenseGrasp(nn.Module):
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
        nn.init.constant_(self.quality.bias, math.log(0.05 / 0.95))
        nn.init.constant_(self.geometry.bias[2:4], math.log(0.20 / 0.80))
        for film in (self.fine_film, self.coarse_film):
            nn.init.normal_(film.weight, std=0.01)
            nn.init.zeros_(film.bias)

    @staticmethod
    def _condition(features, parameters):
        scale, bias = parameters[:, :, None, None].chunk(2, dim=1)
        return features * (1.0 + scale) + bias

    def forward(self, images, tokens):
        token_mask = tokens.ne(0).unsqueeze(1)
        words = self.embedding(tokens).transpose(1, 2)
        words = words + F.silu(self.text_context(words))
        pooled = (words * token_mask).sum(-1) / token_mask.sum(-1).clamp(min=1)
        text = self.text_projection(pooled)

        fine = self.stem(images * 2.0 - 1.0)
        fine = self._condition(fine, self.fine_film(text))
        coarse = self._condition(self.coarse(fine), self.coarse_film(text))
        coarse = F.interpolate(
            self.coarse_projection(coarse), size=fine.shape[-2:], mode="bilinear", align_corners=False
        )
        features = self.fusion(fine + coarse)
        return self.quality(features).squeeze(1), self.geometry(features)


def build(vocab_size):
    return DenseGrasp(vocab_size)


def _grid(quality):
    height, width = quality.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=quality.device, dtype=torch.float32),
        torch.arange(width, device=quality.device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack((x, y), dim=-1), quality.new_tensor((width, height), dtype=torch.float32)


def _unpack_geometry(geometry):
    geometry = geometry.float().movedim(1, -1)
    return geometry[..., :2].tanh() * 0.5, geometry[..., 2:4].sigmoid(), geometry[..., 4:]


def loss(output, target, valid):
    quality, geometry = output
    quality = quality.float()
    grid, scale = _grid(quality)

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
    residual = (centers - cells - 0.5).square().sum(-1)
    nearest = (distance + residual[..., None, None]).argmin(dim=1, keepdim=True)
    nearest_distance = distance.gather(1, nearest).squeeze(1)
    positive = nearest_distance == 0.0
    heatmap = (-nearest_distance / (2 * 1.25**2)).exp()
    count = positive.sum().clamp(min=1)

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

    offset, size, direction = _unpack_geometry(geometry)
    center_offset_loss = F.smooth_l1_loss(offset, wanted_offset, reduction="none", beta=0.1)
    size_loss = F.smooth_l1_loss(size, matched[..., 2:4], reduction="none", beta=0.05)
    angle_loss = F.smooth_l1_loss(direction, wanted_direction, reduction="none", beta=0.2)
    pose_loss = center_offset_loss + 4.0 * size_loss + angle_loss
    return center_loss + (pose_loss.sum(-1) * positive).sum() / count


def decode(output, topk=5):
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
