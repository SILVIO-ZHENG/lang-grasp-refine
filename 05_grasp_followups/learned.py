"""Train-only listwise grasp scoring and bounded pose refinement."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


BOUNDS = (0.075, 0.075, math.log(1.5), math.log(1.5), math.pi / 8)
BATCH = 96
EPOCHS = 14


def _wrap(theta):
    return (theta + math.pi / 2) % math.pi - math.pi / 2


def apply_delta(boxes, delta):
    """Apply normalized center, log-scale and clockwise angle residuals."""
    if delta is None:
        return boxes
    if isinstance(boxes, torch.Tensor):
        delta = torch.as_tensor(delta, dtype=boxes.dtype, device=boxes.device)
        xy = (boxes[..., :2] + delta[..., :2]).clamp(0, 1)
        wh = (boxes[..., 2:4] * delta[..., 2:4].exp()).clamp(1e-3, 1)
        theta = _wrap(boxes[..., 4:] + delta[..., 4:])
        return torch.cat((xy, wh, theta), dim=-1)
    boxes, delta = np.asarray(boxes), np.asarray(delta)
    xy = np.clip(boxes[..., :2] + delta[..., :2], 0, 1)
    wh = np.clip(boxes[..., 2:4] * np.exp(delta[..., 2:4]), 1e-3, 1)
    theta = _wrap(boxes[..., 4:] + delta[..., 4:])
    return np.concatenate((xy, wh, theta), axis=-1)


def _input(features, boxes):
    theta = 2 * boxes[..., 4:]
    return torch.cat((features, boxes[..., :4], theta.sin(), theta.cos()), dim=-1)


class CandidateModel(nn.Module):
    def __init__(self, feature_dim, refine):
        super().__init__()
        self.feature_dim = feature_dim
        self.refine = refine
        self.register_buffer("mean", torch.zeros(feature_dim + 6))
        self.register_buffer("std", torch.ones(feature_dim + 6))
        self.register_buffer("calibration", torch.tensor([1., 0.]))
        self.register_buffer("bounds", torch.tensor(BOUNDS), persistent=False)
        self.encode = nn.Sequential(nn.Linear(feature_dim + 6, 64), nn.LayerNorm(64), nn.SiLU(), nn.Linear(64, 48))
        self.context = nn.Sequential(nn.Linear(144, 64), nn.LayerNorm(64), nn.SiLU())
        self.score = nn.Linear(64, 1)
        self.residual = nn.Linear(64, 5) if refine else None
        if self.residual is not None:
            nn.init.zeros_(self.residual.weight)
            nn.init.zeros_(self.residual.bias)

    def forward(self, features, boxes):
        values = ((_input(features, boxes) - self.mean) / self.std).clamp(-8, 8)
        local = self.encode(values)
        relative = local - local.mean(dim=1, keepdim=True)
        peak = local.amax(dim=1, keepdim=True).expand_as(local)
        contextual = self.context(torch.cat((local, relative, peak), dim=-1))
        logits = self.score(contextual).squeeze(-1)
        delta = None if self.residual is None else self.residual(contextual).tanh() * self.bounds
        return logits, delta


def checkpoint_learned(model):
    return {"feature_dim": model.feature_dim, "refine": model.refine,
            "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}}


def restore_learned(checkpoint, device):
    with torch.random.fork_rng(devices=[]):
        model = CandidateModel(checkpoint["feature_dim"], checkpoint["refine"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _corners(boxes):
    signs = boxes.new_tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    x, y = (signs * boxes[:, None, 2:4] * 0.5).unbind(dim=-1)
    cosine, sine = boxes[:, 4:5].cos(), boxes[:, 4:5].sin()
    return torch.stack((x * cosine - y * sine, x * sine + y * cosine), dim=-1) + boxes[:, None, :2]


def _inside(points, boxes):
    x, y = (points - boxes[:, None, :2]).unbind(dim=-1)
    cosine, sine = boxes[:, 4:5].cos(), boxes[:, 4:5].sin()
    return (((x * cosine + y * sine).abs() <= boxes[:, 2:3] * 0.5 + 1e-7)
            & ((y * cosine - x * sine).abs() <= boxes[:, 3:4] * 0.5 + 1e-7))


def _oriented_iou(a, b):
    """Exact aligned oriented IoU via corners and valid edge intersections."""
    p, q = _corners(a), _corners(b)
    u = (p.roll(-1, dims=1) - p)[:, :, None, :]
    v = (q.roll(-1, dims=1) - q)[:, None, :, :]
    gap = q[:, None, :, :] - p[:, :, None, :]
    denominator = _cross(u, v)
    crossing = denominator.abs() > 1e-10
    divisor = torch.where(crossing, denominator, 1)
    t, s = _cross(gap, v) / divisor, _cross(gap, u) / divisor
    crossing = crossing & (t >= 0) & (t <= 1) & (s >= 0) & (s <= 1)
    hits = (p[:, :, None, :] + t.unsqueeze(-1) * u).flatten(start_dim=1, end_dim=2)
    points = torch.cat((p, q, hits), dim=1)
    valid = torch.cat((_inside(p, b), _inside(q, a), crossing.flatten(start_dim=1)), dim=1)
    count = valid.sum(dim=1, keepdim=True)
    centroid = torch.where(valid.unsqueeze(-1), points, 0).sum(dim=1, keepdim=True) / count.clamp_min(1).unsqueeze(-1)
    rays = points - centroid
    order = torch.atan2(rays[..., 1], rays[..., 0]).masked_fill(~valid, 4).argsort(dim=1)
    sorted_rays = rays.gather(1, order.unsqueeze(-1).expand(-1, -1, 2))
    index = torch.arange(points.shape[1], device=points.device)[None, :]
    following = torch.where(index + 1 < count, index + 1, 0)
    next_rays = sorted_rays.gather(1, following.unsqueeze(-1).expand(-1, -1, 2))
    intersection = (_cross(sorted_rays, next_rays) * (index < count)).sum(dim=1).abs() * 0.5
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - intersection
    return (intersection / union.clamp_min(1e-8)).clamp(0, 1)


def _prepare(data, labels):
    values = {name: torch.as_tensor(data[name], dtype=torch.float32, device="cpu") for name in ("features", "boxes")}
    x, boxes = values["features"], values["boxes"]
    if x.ndim != 3 or boxes.shape != (*x.shape[:2], 5) or x.shape[1] != 24 or x.shape[2] == 0:
        raise ValueError("Expected features [N,24,D] and boxes [N,24,5]")
    if not x.isfinite().all() or not boxes.isfinite().all() or (boxes[..., 2:4] <= 0).any():
        raise ValueError("Non-finite features or invalid candidate geometry")
    if not labels:
        return values
    correct = torch.as_tensor(data["correct"], dtype=torch.float32, device="cpu")
    quality = torch.as_tensor(data.get("quality", correct), dtype=torch.float32, device="cpu")
    if correct.shape != x.shape[:2] or quality.shape != x.shape[:2] or not ((correct == 0) | (correct == 1)).all():
        raise ValueError("Expected correct and quality [N,24], with binary correctness")
    if not quality.isfinite().all() or ((quality < 0) | (quality > 1)).any():
        raise ValueError("Expected finite quality in [0,1]")
    target = torch.as_tensor(data.get("target", torch.zeros_like(boxes)), dtype=torch.float32, device="cpu")
    if target.shape != boxes.shape:
        raise ValueError("Expected matching target and candidate box shapes")
    usable = target.isfinite().all(dim=-1) & (target[..., 2:4] > 0).all(dim=-1)
    regress = torch.as_tensor(data.get("regress", usable), dtype=torch.bool, device="cpu")
    if regress.shape != x.shape[:2] or (regress & ~usable).any():
        raise ValueError("A positive regression mask requires finite target geometry")
    target = torch.where(regress.unsqueeze(-1), target, boxes)
    values.update(correct=correct, quality=quality, target=target, regress=regress)
    return values


def _batch(data, index, device):
    return {name: value[index].to(device) for name, value in data.items()}


@torch.no_grad()
def _refined_supervision(data, delta):
    mask = data["regress"]
    prediction = apply_delta(data["boxes"][mask], delta[mask])
    target = data["target"][mask]
    angle = _wrap(prediction[:, 4] - target[:, 4]).abs()
    iou = _oriented_iou(prediction, target)
    correct = data["correct"].clone()
    correct[mask] = ((iou > 0.25) & (angle < math.pi / 6)).float()
    quality = data["quality"].clone()
    quality[mask] = iou * (1 - angle / (math.pi / 3)).clamp(0, 1)
    return correct, quality


def _train_objective(model, data, logits, delta):
    correct, quality = data["correct"], data["quality"]
    regression = logits.new_zeros(())
    if delta is not None:
        correct, quality = _refined_supervision(data, delta)
        target, boxes = data["target"], data["boxes"]
        xy = target[..., :2] - boxes[..., :2]
        scale = (target[..., 2:4] / boxes[..., 2:4]).log()
        angle = _wrap(target[..., 4:] - boxes[..., 4:])
        desired = (torch.cat((xy, scale, angle), dim=-1) / model.bounds).clamp(-1, 1)
        error = F.smooth_l1_loss(delta / model.bounds, desired, beta=0.2, reduction="none").mean(dim=-1)
        mask = data["regress"]
        regression = (error * mask).sum() / mask.sum().clamp_min(1)
        regression = regression + 0.12 * ((delta / model.bounds).square().mean(dim=-1) * ~mask).mean()

    # Train a calibrated success score alongside ranking likely useful grasps.
    mass = (quality.square() + correct * 0.5)
    mass_sum = mass.sum(dim=1)
    per_list = -(mass * logits.log_softmax(dim=1)).sum(dim=1) / mass_sum.clamp_min(1e-8)
    listwise = per_list.sum() / (mass_sum > 0).sum().clamp_min(1)
    binary = F.binary_cross_entropy_with_logits(logits, correct)
    return binary + 0.6 * listwise + 3 * regression, correct


@torch.no_grad()
def _validation(model, values, device):
    loss = 0.0
    logits_out, labels_out = [], []
    count = values["boxes"].shape[0]
    for start in range(0, count, BATCH):
        data = _batch(values, slice(start, start + BATCH), device)
        logits, delta = model(data["features"], data["boxes"])
        objective, correct = _train_objective(model, data, logits, delta)
        loss += len(logits) * float(objective)
        logits_out.append(logits.cpu())
        labels_out.append(correct.cpu())
    logits, correct = torch.cat(logits_out), torch.cat(labels_out)
    top1 = correct.gather(1, logits.argmax(dim=1, keepdim=True)).mean()
    return loss / count, float(top1), logits, correct


def _fit_calibration(model, logits, correct):
    if not 0 < correct.mean() < 1:
        return {"scale": 1.0, "bias": 0.0, "reason": "validation_has_one_label"}
    fit = torch.zeros(2, requires_grad=True)
    optimizer = torch.optim.LBFGS([fit], max_iter=25, line_search_fn="strong_wolfe")

    def objective():
        optimizer.zero_grad()
        scale = fit[0].clamp(-2, 2).exp()
        loss = F.binary_cross_entropy_with_logits(scale * logits + fit[1], correct) + 1e-4 * fit.square().sum()
        loss.backward()
        return loss

    optimizer.step(objective)
    with torch.no_grad():
        scale, bias = fit[0].clamp(-2, 2).exp(), fit[1]
        model.calibration.copy_(torch.stack((scale, bias)).to(model.calibration))
        after = F.binary_cross_entropy_with_logits(scale * logits + bias, correct)
        return {"scale": float(scale), "bias": float(bias), "validation_bce": float(after)}


def _fit(train, validation, device, seed, refine):
    x = _input(train["features"], train["boxes"])
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = CandidateModel(train["features"].shape[-1], refine)
    model.mean.copy_(x.mean(dim=(0, 1)))
    model.std.copy_(x.std(dim=(0, 1), unbiased=False).clamp_min(1e-4))
    del x
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.0002)
    generator = torch.Generator().manual_seed(seed)
    count = train["boxes"].shape[0]
    best_loss, best_epoch, best = math.inf, 0, None
    epochs = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for index in torch.randperm(count, generator=generator).split(BATCH):
            data = _batch(train, index, device)
            optimizer.zero_grad(set_to_none=True)
            logits, delta = model(data["features"], data["boxes"])
            objective, _ = _train_objective(model, data, logits, delta)
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0, error_if_nonfinite=True)
            optimizer.step()
            train_loss += len(index) * float(objective.detach())
        scheduler.step()
        model.eval()
        val_loss, top1, _, _ = _validation(model, validation, device)
        epochs.append({"epoch": epoch, "train_loss": train_loss / count, "validation_loss": val_loss,
                       "validation_top1_matched_target": top1})
        if val_loss < best_loss:
            best_loss, best_epoch, best = val_loss, epoch, checkpoint_learned(model)
        if epoch - best_epoch >= 4:
            break
    if best is None:
        raise RuntimeError("Training produced no finite validation checkpoint")
    model = restore_learned(best, device)
    _, _, logits, labels = _validation(model, validation, device)
    calibration = _fit_calibration(model, logits, labels)
    return model, {"best_epoch": best_epoch, "epochs": epochs, "calibration": calibration}


def train_learned(train, validation, device, seed=121):
    """Fit on supplied train rows and choose checkpoints and calibration on validation."""
    train, validation = _prepare(train, True), _prepare(validation, True)
    if not len(train["boxes"]) or not len(validation["boxes"]) or not train["regress"].any():
        raise ValueError("Expected nonempty train and validation sets, with nearby targets in training")
    if train["features"].shape[-1] != validation["features"].shape[-1]:
        raise ValueError("Train and validation feature dimensions differ")
    ranker, rank_history = _fit(train, validation, device, seed, refine=False)
    refiner, refine_history = _fit(train, validation, device, seed + 1, refine=True)
    return ranker, refiner, {"ranker": rank_history, "refiner": refine_history}


@torch.inference_mode()
def predict_learned(model, data, device):
    values = _prepare(data, False)
    model.to(device).eval()
    scores, residuals = [], []
    count = values["boxes"].shape[0]
    for start in range(0, count, 256):
        batch = _batch(values, slice(start, start + 256), device)
        logits, delta = model(batch["features"], batch["boxes"])
        scores.append((logits * model.calibration[0] + model.calibration[1]).cpu())
        if delta is not None:
            residuals.append(delta.cpu())
    logits = torch.cat(scores) if scores else values["boxes"].new_empty((0, 24))
    if not model.refine:
        return logits.numpy(), None
    delta = torch.cat(residuals) if residuals else values["boxes"].new_empty((0, 24, 5))
    return logits.numpy(), delta.numpy()
