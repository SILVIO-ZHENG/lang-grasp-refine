"""Build features and annotation targets for the learned candidate heads.

Each candidate carries the appearance around its gripper footprint, geometry
relative to the detected object, grounding scores, and its proposal rank.

为学习式候选预测头构建输入特征与标注目标。

每个候选包含夹爪覆盖区域的外观、相对检测目标的几何信息、定位分数及初始名次。
"""

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from grounded_baseline import region_scores
from run_experiment import SIZE, cv_rect


K = 24


def angles(first, second):
    """Return the smallest angular difference for parallel jaws, modulo pi.

    按平行夹爪的半周对称性，返回两个角度之间的最小夹角。
    """
    return np.abs((first - second + np.pi / 2) % np.pi - np.pi / 2)


def supervised_targets(poses, truth):
    """Label each candidate against the instruction's nonempty set of grasps.

    `poses` [K,5] and `truth` [N,5] use normalized center and size with an angle
    in radians. Return binary hits [K], soft quality [K], matched boxes [K,5],
    and a Boolean mask [K] indicating where refiner supervision is appropriate.

    用指令对应的一组有效抓取标注为每个候选生成训练目标，标注至少有一个。

    `poses` [K,5] 和 `truth` [N,5] 均采用归一化中心和尺寸，角度单位为弧度。
    输出成功标签 [K]、连续质量分数 [K]、匹配矩形 [K,5]，以及标记是否参与
    细化回归的布尔掩码 [K]。
    """
    delta = np.linalg.norm(poses[:, None, :2] - truth[None, :, :2], axis=-1)
    theta = angles(poses[:, None, 4], truth[None, :, 4])
    rad_candidate = np.linalg.norm(poses[:, 2:4], axis=-1) / 2
    rad_truth = np.linalg.norm(truth[:, 2:4], axis=-1) / 2
    iou = np.zeros_like(delta)
    # Skip the rotated intersection test for disjoint enclosing circles.
    # 两个外接圆不相交时，跳过旋转矩形的求交计算。
    candidate_ids, target_ids = np.nonzero(delta < rad_candidate[:, None] + rad_truth)
    for i, j in zip(candidate_ids, target_ids, strict=True):
        _, polygon = cv2.rotatedRectangleIntersection(cv_rect(poses[i]), cv_rect(truth[j]))
        if polygon is None:
            continue
        area = cv2.contourArea(polygon)
        union = (poses[i, 2] * poses[i, 3] + truth[j, 2] * truth[j, 3]) * SIZE**2 - area
        if union > 0:
            iou[i, j] = min(1.0, area / union)
    # Any matching annotation can supply a success label. Regression uses one
    # compatible target, chosen by overlap, angle, and center proximity.
    # 任意一条标注满足阈值即可产生成功标签。回归只使用一个匹配目标，
    # 由重叠程度、角度差和中心距离共同选出。
    hit = ((iou > 0.25) & (theta < np.pi / 6)).any(axis=1)
    quality = iou * np.exp(-0.5 * (theta / 0.40) ** 2)
    close = np.exp(-0.5 * (delta / 0.055) ** 2 - 0.5 * (theta / 0.65) ** 2)
    nearest = np.argmax(quality + 0.25 * close, axis=1)
    indices = np.arange(len(poses))
    regress = (delta[indices, nearest] < 0.09) & (theta[indices, nearest] < np.pi / 3)
    return hit, quality.max(axis=1), truth[nearest], regress


def candidate_rgb(images, boxes):
    """Sample each rotated proposal as flattened 7-by-7 RGB evidence.

    Read images [B,3,H,W] and boxes [B,K,5]. Rotate each sampling grid with its
    box, with a small margin beyond the rectangle. Return a NumPy array [B,K,147].

    在每个旋转候选框附近采样 7×7 的 RGB 网格并展平。

    输入为图像 [B,3,H,W] 与抓取框 [B,K,5]。采样网格按框角度旋转，覆盖框外
    的少量区域，输出 NumPy 数组 [B,K,147]。
    """
    axis = torch.linspace(-0.7, 0.7, 7, device=images.device)
    v, u = torch.meshgrid(axis, axis, indexing='ij')
    x, y, width, height, theta = boxes[:, :, None, None].unbind(-1)
    xx = x + width * u * theta.cos() - height * v * theta.sin()
    yy = y + width * u * theta.sin() + height * v * theta.cos()
    grid = 2 * torch.stack((xx, yy), dim=-1).flatten(1, 2) - 1
    patches = F.grid_sample(images, grid, align_corners=False, padding_mode='border')
    return patches.reshape(images.shape[0], 3, boxes.shape[1], 49).transpose(1, 2).flatten(2).cpu().numpy()


def feature_vector(boxes, regions, rgb):
    """Return [K,155] features for one image and one instruction.

    The descriptor combines 147 RGB samples, four object-relative geometry
    values, and the object, part, zoomed-part, and proposal-rank signals.

    为一张图像和一条指令构建 [K,155] 的候选特征。

    特征由 147 个 RGB 采样值、四个相对目标框的几何量，以及物体定位、部位定位、
    放大部位定位和初始候选名次四个分数组成。
    """
    obj, part, zoom, *_ = region_scores(boxes, regions)
    # Center each appearance feature across the candidates for this instruction.
    # 对这条指令下的同一外观特征，以所有候选的均值做中心化。
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
    """For each first-list box, score its closest match in the second list.

    用中心距离和角度差衡量两组候选的一致性，为第一组的每个候选返回最佳匹配分数。
    """
    if not len(second):
        return np.zeros(len(first))
    distance = np.linalg.norm(first[:, None, :2] - second[None, :, :2], axis=-1)
    theta = angles(first[:, None, 4], second[None, :, 4])
    return np.exp(-0.5 * (distance / 0.07) ** 2 - 0.5 * (theta / 0.46) ** 2).max(axis=1)
