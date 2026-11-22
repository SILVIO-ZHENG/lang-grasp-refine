"""Candidate appearance, grounding features, and supervision used by both variants."""

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from run import region_scores
from run_experiment import SIZE, cv_rect


K = 24


def angles(first, second):
    return np.abs((first - second + np.pi / 2) % np.pi - np.pi / 2)


def supervised_targets(poses, truth):
    delta = np.linalg.norm(poses[:, None, :2] - truth[None, :, :2], axis=-1)
    theta = angles(poses[:, None, 4], truth[None, :, 4])
    rad_candidate = np.linalg.norm(poses[:, 2:4], axis=-1) / 2
    rad_truth = np.linalg.norm(truth[:, 2:4], axis=-1) / 2
    iou = np.zeros_like(delta)
    candidate_ids, target_ids = np.nonzero(delta < rad_candidate[:, None] + rad_truth)
    for i, j in zip(candidate_ids, target_ids, strict=True):
        _, polygon = cv2.rotatedRectangleIntersection(cv_rect(poses[i]), cv_rect(truth[j]))
        if polygon is None:
            continue
        area = cv2.contourArea(polygon)
        union = (poses[i, 2] * poses[i, 3] + truth[j, 2] * truth[j, 3]) * SIZE**2 - area
        if union > 0:
            iou[i, j] = min(1.0, area / union)
    hit = ((iou > 0.25) & (theta < np.pi / 6)).any(axis=1)
    quality = iou * np.exp(-0.5 * (theta / 0.40) ** 2)
    close = np.exp(-0.5 * (delta / 0.055) ** 2 - 0.5 * (theta / 0.65) ** 2)
    nearest = np.argmax(quality + 0.25 * close, axis=1)
    indices = np.arange(len(poses))
    regress = (delta[indices, nearest] < 0.09) & (theta[indices, nearest] < np.pi / 3)
    return hit, quality.max(axis=1), truth[nearest], regress


def candidate_rgb(images, boxes):
    axis = torch.linspace(-0.7, 0.7, 7, device=images.device)
    v, u = torch.meshgrid(axis, axis, indexing='ij')
    x, y, width, height, theta = boxes[:, :, None, None].unbind(-1)
    xx = x + width * u * theta.cos() - height * v * theta.sin()
    yy = y + width * u * theta.sin() + height * v * theta.cos()
    grid = 2 * torch.stack((xx, yy), dim=-1).flatten(1, 2) - 1
    patches = F.grid_sample(images, grid, align_corners=False, padding_mode='border')
    return patches.reshape(images.shape[0], 3, boxes.shape[1], 49).transpose(1, 2).flatten(2).cpu().numpy()


def feature_vector(boxes, regions, rgb):
    obj, part, zoom, *_ = region_scores(boxes, regions)
    image_features = rgb - rgb.mean(axis=1, keepdims=True)
    best_object = np.zeros(5, dtype=np.float32)
    if len(regions['object_boxes']):
        best_object = regions['object_boxes'][0]
    center = (best_object[:2] + best_object[2:4]) / 2
    size = (best_object[2:4] - best_object[:2]).clip(min=0.05)
    object_relative = (boxes[:, :2] - center) / size
    side_ratio = boxes[:, 2:4] / size
    rank = np.log1p(np.arange(len(boxes), dtype=np.float32)) / np.log(1 + K)
    semantic = np.stack((obj, part, zoom, rank), axis=-1)
    return np.concatenate((image_features, object_relative, side_ratio, semantic), axis=-1).astype(np.float32)


def agreement(first, second):
    if not len(second):
        return np.zeros(len(first))
    distance = np.linalg.norm(first[:, None, :2] - second[None, :, :2], axis=-1)
    theta = angles(first[:, None, 4], second[None, :, 4])
    return np.exp(-0.5 * (distance / 0.07) ** 2 - 0.5 * (theta / 0.46) ** 2).max(axis=1)
