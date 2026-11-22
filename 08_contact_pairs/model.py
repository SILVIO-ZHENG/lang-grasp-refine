"""Dense language-guided prediction of opposite gripper contact points."""

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
    height, width = heatmap.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=heatmap.device, dtype=torch.float32),
        torch.arange(width, device=heatmap.device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack((x, y), dim=-1), heatmap.new_tensor((width, height), dtype=torch.float32)


class ContactGrasp(nn.Module):
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
        words = self.text(tokens).transpose(1, 2)
        words = words + F.silu(self.word_context(words))
        present = tokens.ne(0).unsqueeze(1)
        query = self.text_query((words * present).sum(-1) / present.sum(-1).clamp_min(1))

        detail = self.detail(images * 2.0 - 1.0)
        context = self.context(detail)
        query = query[..., None, None]
        relevance = (self.visual_key(context) * query).sum(1, keepdim=True) * 0.125
        context = self.context_projection(context * relevance.sigmoid() + query)
        detail = self.fusion(detail + F.interpolate(context, detail.shape[-2:], mode="bilinear", align_corners=False))

        heatmap = self.center(detail).squeeze(1)
        cells, scale = _grid(heatmap)
        raw = self.endpoints(detail).float().movedim(1, -1).unflatten(-1, (2, 2))
        raw_midpoint = raw.mean(dim=-2, keepdim=True)
        # Bound the mean to its cell but let contacts extend across the image.
        # The four output channels are supervised only through the two points.
        midpoint = (cells.unsqueeze(-2) + 0.5 + 0.5 * raw_midpoint.tanh()) / scale
        contacts = midpoint + 0.65 * (raw - raw_midpoint).tanh()
        return heatmap, contacts, self.thickness(detail).squeeze(1).float().sigmoid()


def build(vocab_size):
    return ContactGrasp(vocab_size)


def loss(output, target, valid):
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

    error = F.smooth_l1_loss(contacts, desired, beta=0.025, reduction="none").sum(dim=(-1, -2))
    swapped = F.smooth_l1_loss(contacts, desired.flip(-2), beta=0.025, reduction="none").sum(dim=(-1, -2))
    contact_loss = torch.minimum(error, swapped)
    center_loss = F.smooth_l1_loss(contacts.mean(-2) * scale, center * scale, beta=0.1, reduction="none").sum(-1)
    thickness_loss = F.smooth_l1_loss(thickness, matched[..., 3], beta=0.05, reduction="none")
    regression = center_loss + 4.0 * contact_loss + 4.0 * thickness_loss
    return confidence_loss + (regression * positive).sum() / count


def decode(output, topk=5):
    logits, contacts, thickness = output
    center = contacts.mean(-2).clamp(0.0, 1.0)
    width_vector = contacts[..., 1, :] - contacts[..., 0, :]
    width = width_vector.norm(dim=-1).clamp_min(1e-4)
    angle = torch.atan2(width_vector[..., 1], width_vector[..., 0])
    angle = (angle + math.pi / 2).remainder(math.pi) - math.pi / 2
    poses = torch.cat((center, torch.stack((width, thickness, angle), dim=-1)), dim=-1).flatten(1, 2)

    probability = logits.float().sigmoid()
    peaks = F.max_pool2d(probability.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    scores = torch.where(probability == peaks, probability, probability * 0.2).flatten(1)
    indices = scores.topk(min(topk, scores.shape[1]), dim=1).indices
    return poses.gather(1, indices[..., None].expand(-1, -1, 5))
