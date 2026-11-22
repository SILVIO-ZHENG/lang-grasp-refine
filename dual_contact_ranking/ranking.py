"""Merge the refined heatmap and dual-contact scores with fixed weights.

用已选定的权重合并细化热图与双接触点分支的分数。
"""

import numpy as np


def merge_scores(refined_scores, heatmap_rank_scores, contact_scores, rank_weight, contact_bias):
    """Return 24 refined heatmap scores followed by 24 contact scores.

    The learned rank scores correspond to the original heatmap candidates,
    whose ordering is preserved by refinement. Contact proposals receive
    grounding scores and a branch offset. Validation chooses the two
    scalars before test evaluation.

    返回 24 个细化热图候选的分数，随后是 24 个接触点候选的分数。

    学到的排序分数对应原始热图候选。细化会修改抓取框几何，保留候选顺序。
    接触点候选使用定位分数和独立的分支偏置。两个标量在验证集上确定后，
    固定用于测试集评测。
    """
    return np.concatenate((
        refined_scores + rank_weight * heatmap_rank_scores,
        contact_scores + contact_bias,
    ), axis=-1)
