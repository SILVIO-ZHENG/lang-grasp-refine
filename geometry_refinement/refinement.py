"""Apply the geometry refiner's five predicted residuals to a candidate box.

将几何细化头预测的五维残差应用到候选抓取框。
"""

import math

import numpy as np
import torch


def wrap_angle(theta):
    """Wrap a parallel-jaw orientation to [-pi/2, pi/2).

    将平行夹爪的朝向归一到 [-pi/2, pi/2)。
    """
    return (theta + math.pi / 2) % math.pi - math.pi / 2


def apply_delta(boxes, delta):
    """Decode five residuals while preserving all leading box dimensions.

    Boxes end in [cx, cy, width, height, angle]. The two center offsets use
    square-image units, the side offsets are log multipliers, and the angle
    offset is in radians. Tensor inputs retain gradients for the refiner
    loss. NumPy inputs follow the same equations for offline scoring.

    将五维残差解码成新的抓取框，保留输入的所有前导维度。

    抓取框最后一维为 [中心 x、中心 y、宽、高、角度]。中心偏移以方形图像边长
    为单位，两条边使用对数尺度修正，角度偏移用弧度表示。张量输入保留细化损失
    的梯度，离线评分的 NumPy 输入使用同一组公式。
    """
    if delta is None:
        return boxes
    if isinstance(boxes, torch.Tensor):
        delta = torch.as_tensor(delta, dtype=boxes.dtype, device=boxes.device)
        xy = (boxes[..., :2] + delta[..., :2]).clamp(0, 1)
        wh = (boxes[..., 2:4] * delta[..., 2:4].exp()).clamp(1e-3, 1)
        theta = wrap_angle(boxes[..., 4:] + delta[..., 4:])
        return torch.cat((xy, wh, theta), dim=-1)
    boxes, delta = np.asarray(boxes), np.asarray(delta)
    xy = np.clip(boxes[..., :2] + delta[..., :2], 0, 1)
    wh = np.clip(boxes[..., 2:4] * np.exp(delta[..., 2:4]), 1e-3, 1)
    theta = wrap_angle(boxes[..., 4:] + delta[..., 4:])
    return np.concatenate((xy, wh, theta), axis=-1)
