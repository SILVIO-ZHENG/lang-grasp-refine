"""Learn to correct and score a list of grasp candidates.

The refiner predicts bounded corrections to the center, size, and angle of
each box and learns which corrected boxes should come first. Use validation
candidates to select the checkpoint and calibrate the learned scores.

学习修正一组抓取候选并为它们评分。

细化模型预测每个框的有界中心、尺寸和角度修正，并学习哪些框在细化后应排在前面。
使用验证集候选选择权重并校准分数。
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from geometry_refinement.refinement import apply_delta, wrap_angle as _wrap


BOUNDS = (0.075, 0.075, math.log(1.5), math.log(1.5), math.pi / 8)
BATCH = 96
EPOCHS = 14


def _input(features, boxes):
    # The doubled angle makes the geometry invariant to a half turn of the jaws.
    # 使用二倍角的正弦和余弦，使夹爪旋转半周时几何表示保持一致。
    theta = 2 * boxes[..., 4:]
    return torch.cat((features, boxes[..., :4], theta.sin(), theta.cos()), dim=-1)


class CandidateModel(nn.Module):
    """Score candidates jointly and predict five correction values per pose.

    A 155-value descriptor and six geometry values give 161 inputs per candidate
    in the reported experiments. The encoder maps them to 48 channels. The
    context MLP reads the local embedding, deviation from the list mean, and
    the elementwise list maximum.

    联合处理一组抓取候选，为每个候选评分并输出五维位姿修正。

    报告中的实验将 155 维特征和六维几何编码拼成每个候选的 161 维输入。编码器
    将其映射到 48 维。上下文 MLP 读取候选自身的编码、它相对列表均值的偏差，
    以及整组候选按维度取出的最大值。
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.register_buffer("mean", torch.zeros(feature_dim + 6))
        self.register_buffer("std", torch.ones(feature_dim + 6))
        self.register_buffer("calibration", torch.tensor([1., 0.]))
        self.register_buffer("bounds", torch.tensor(BOUNDS), persistent=False)
        self.encode = nn.Sequential(nn.Linear(feature_dim + 6, 64), nn.LayerNorm(64), nn.SiLU(), nn.Linear(64, 48))
        self.context = nn.Sequential(nn.Linear(144, 64), nn.LayerNorm(64), nn.SiLU())
        self.score = nn.Linear(64, 1)
        self.residual = nn.Linear(64, 5)
        # Start the refiner with zero corrections, preserving the input poses.
        # 让细化头从零修正开始，初始输出保持候选位姿不变。
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, features, boxes):
        """Map [B,K,D] features and [B,K,5] poses to raw scores [B,K].

        The second return value is [B,K,5] pose corrections. Calibration is
        applied separately during prediction.

        将特征 [B,K,D] 和位姿 [B,K,5] 映射为原始分数 [B,K]。

        第二项输出是 [B,K,5] 的位姿修正量。分数校准在推理时单独应用。
        """
        values = ((_input(features, boxes) - self.mean) / self.std).clamp(-8, 8)
        local = self.encode(values)
        relative = local - local.mean(dim=1, keepdim=True)
        peak = local.amax(dim=1, keepdim=True).expand_as(local)
        contextual = self.context(torch.cat((local, relative, peak), dim=-1))
        logits = self.score(contextual).squeeze(-1)
        delta = self.residual(contextual).tanh() * self.bounds
        return logits, delta


def checkpoint_learned(model):
    """Snapshot weights, feature statistics, and calibration on the CPU.

    在 CPU 上保存权重、特征统计量及校准参数的副本。
    """
    return {"feature_dim": model.feature_dim, "refine": True,
            "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}}


def restore_learned(checkpoint, device):
    if checkpoint.get("refine") is not True:
        raise ValueError("Expected a geometry-refiner checkpoint")
    with torch.random.fork_rng(devices=[]):
        model = CandidateModel(checkpoint["feature_dim"])
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
    """Compute rotated IoU for aligned pairs of boxes indexed by row.

    Form the intersection polygon from corners inside the other box and
    valid edge crossings. Order points around their centroid to compute area.

    按行配对计算旋转矩形的交并比。

    交集多边形由落在另一框内的角点和有效的边交点构成。将点按其相对质心的
    方位排序后计算面积。
    """
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
    """Validate a bank of 24 candidates per instruction and optional targets.

    检查候选库中每条指令的 24 个候选，以及可选的监督标签与匹配目标。
    """
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
    """Recompute success and quality against the matched annotation after correction.

    This updates score targets for candidates with usable matches. The IoU and
    success label calculations do not pass gradients to the geometry head.

    对应用修正后的候选框，按其匹配标注重新计算成功标签和质量分数。

    只更新有有效匹配的候选。IoU 和成功标签的计算不会向几何头反传梯度。
    """
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
    """Train candidate success, ranking within each list, and pose corrections.

    训练每个候选的成功分数、同一列表内的排序以及位姿修正量。
    """
    correct, quality = _refined_supervision(data, delta)
    target, boxes = data["target"], data["boxes"]
    # Supervise normalized residuals on matched boxes. A smaller penalty
    # discourages unnecessary movement of candidates without usable targets.
    # 对匹配框监督归一化残差。对缺少有效匹配的候选，用较小的惩罚约束无依据的移动。
    xy = target[..., :2] - boxes[..., :2]
    scale = (target[..., 2:4] / boxes[..., 2:4]).log()
    angle = _wrap(target[..., 4:] - boxes[..., 4:])
    desired = (torch.cat((xy, scale, angle), dim=-1) / model.bounds).clamp(-1, 1)
    error = F.smooth_l1_loss(delta / model.bounds, desired, beta=0.2, reduction="none").mean(dim=-1)
    mask = data["regress"]
    regression = (error * mask).sum() / mask.sum().clamp_min(1)
    regression = regression + 0.12 * ((delta / model.bounds).square().mean(dim=-1) * ~mask).mean()

    # Binary cross-entropy learns per-candidate success. Listwise cross-entropy
    # places score mass on strong candidates across the full instruction list.
    # 二元交叉熵让模型学习每个候选是否成功。列表式交叉熵使分数集中到同一指令
    # 下表现较好的候选上。
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
    """Fit a positive scale and a bias using the held-out validation logits.

    The positive scale preserves ordering within this head while calibrating
    its score for fusion with other terms at inference time.

    在保留的验证集 logits 上拟合正的缩放系数和偏置。

    正的缩放保留该预测头内部的排序，同时调整分数尺度，供推理时与其他分数融合。
    """
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


def _fit(train, validation, device, seed):
    """Fit one head and restore the checkpoint with the lowest validation loss.

    训练一个预测头，并恢复验证损失最低时保存的权重。
    """
    x = _input(train["features"], train["boxes"])
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = CandidateModel(train["features"].shape[-1])
    # Compute normalizer statistics from training proposals only.
    # 特征归一化的统计量只从训练候选计算。
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


def train_refiner(train, validation, device, seed=121):
    """Fit the refiner and return its weights and validation history.

    Use separate labeled sets for training and validation, each with
    24 proposals per instruction.

    训练细化模型，并返回权重和验证记录。

    分别使用带标签的训练集和验证集。每条指令提供 24 个候选。
    """
    train, validation = _prepare(train, True), _prepare(validation, True)
    if not len(train["boxes"]) or not len(validation["boxes"]) or not train["regress"].any():
        raise ValueError("Expected nonempty train and validation sets, with nearby targets in training")
    if train["features"].shape[-1] != validation["features"].shape[-1]:
        raise ValueError("Train and validation feature dimensions differ")
    # Keep the seed offset from the reported run for the same initialization.
    # 保留实验使用的随机种子偏移，使初始化与原实验一致。
    return _fit(train, validation, device, seed + 1)


@torch.inference_mode()
def predict_learned(model, data, device):
    """Return calibrated NumPy scores and pose corrections in proposal order.

    按原候选顺序返回校准后的 NumPy 分数和位姿修正量。
    """
    values = _prepare(data, False)
    model.to(device).eval()
    scores, residuals = [], []
    count = values["boxes"].shape[0]
    for start in range(0, count, 256):
        batch = _batch(values, slice(start, start + 256), device)
        logits, delta = model(batch["features"], batch["boxes"])
        scores.append((logits * model.calibration[0] + model.calibration[1]).cpu())
        residuals.append(delta.cpu())
    logits = torch.cat(scores) if scores else values["boxes"].new_empty((0, 24))
    delta = torch.cat(residuals) if residuals else values["boxes"].new_empty((0, 24, 5))
    return logits.numpy(), delta.numpy()
