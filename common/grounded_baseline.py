"""Rescore a fixed grasp pool with open-vocabulary object and part grounding.

Load trained proposer weights and a frozen Grounding DINO snapshot.
Select score weights on validation, then evaluate the fixed settings and
export ranked predictions for the requested test splits.

使用开放词汇的物体与部位定位，对同一组抓取候选重新打分。
加载训练好的候选生成器和冻结的 Grounding DINO 快照。权重只用验证集选取，
随后用固定配置评估指定测试集并导出排序结果。
"""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from evaluate_seen_unseen import harmonic, load_checkpoint
from run_experiment import GraspDataset, SIZE, correct, normalized, overlay, read_rows, safe_json

from grounding import Grounder


OBJECT_WEIGHTS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
PART_WEIGHTS = (0.0, 0.5, 1.0, 2.0, 4.0)


class RankingDataset(GraspDataset):
    """Provide separate proposer and grounding images of the same padded scene.

    对同一个补边场景，分别生成候选模型和定位模型所需分辨率的图像。
    """

    def __getitem__(self, index):
        """Return two CHW images in [0, 1], prompt token IDs, and the row index.

        The proposer uses SIZE (160 in this release). Grounding uses 416.
        Padding comes before resizing so both use the same normalized frame.

        返回两张取值 [0, 1] 的 CHW 图像、指令词元 ID 和原始行号。
        候选模型使用 SIZE（此版本为 160），定位模型使用 416。先统一补成
        正方形再分别缩放，使两路共享归一化坐标。
        """
        with Image.open(self.images / self.rows[index]['image']) as source:
            image = source.convert('RGB')
        width, height = image.size
        if width != height:
            side = max(width, height)
            square = Image.new('RGB', (side, side))
            square.paste(image, ((side - width) // 2, (side - height) // 2))
            image = square
        pixels = []
        for resolution in (SIZE, 416):
            resized = image.resize((resolution, resolution), Image.Resampling.BILINEAR)
            pixels.append(torch.from_numpy(np.array(resized).transpose(2, 0, 1).copy()).float() / 255.0)
        proposal_image, grounding_image = pixels
        return proposal_image, grounding_image, self.tokens[index], index


def support(points, bounds):
    """Return soft spatial support [K, N] between K points and N regions.

    A point inside a region gets one. Outside, the response decays with
    distance measured relative to the box width and height, with a size floor
    to keep very small boxes from creating extreme penalties.

    计算 K 个点相对于 N 个区域的软支持度 [K, N]。框内得分为 1，
    框外随相对于框宽高的距离衰减。框尺寸设有下限，避免极小框导致过强惩罚。
    """
    lower, upper = bounds[:, :2], bounds[:, 2:4]
    extension = np.maximum(lower - points[:, None], points[:, None] - upper).clip(min=0.0)
    distance = np.linalg.norm(extension / (upper - lower).clip(min=0.03), axis=-1)
    return np.exp(-0.5 * (distance / 0.22) ** 2)


def localized_part_score(centers, obj, part):
    """Score each center using part boxes linked to a plausible object box.

    A part needs at least 70% containment and may cover at most 80% of its
    supporting object. Combine object and part confidences by geometric mean.

    利用与物体框匹配的部位框计算每个抓取中心的得分。部位框至少有
    70% 面积落入物体框，且面积不超过物体框的 80%。取物体和部位
    置信度的几何平均。
    """
    if not len(part):
        return np.zeros(len(centers), dtype=np.float32), False
    lower = np.maximum(part[:, None, :2], obj[None, :, :2])
    upper = np.minimum(part[:, None, 2:4], obj[None, :, 2:4])
    intersection = (upper - lower).clip(min=0).prod(axis=-1)
    part_area = (part[:, 2:4] - part[:, :2]).prod(axis=-1)
    object_area = (obj[:, 2:4] - obj[:, :2]).prod(axis=-1)
    contained = (intersection >= 0.7 * part_area[:, None]) & (part_area[:, None] <= 0.8 * object_area)
    # Require a predicted object to support the part evidence.
    # 部位检测必须得到至少一个预测物体框的支持。
    confidence = np.sqrt((contained * obj[:, 4]).max(axis=1) * part[:, 4])
    score = (confidence * support(centers, part)).max(axis=1)
    return score, bool(confidence.any())


def region_scores(poses, grounding):
    """Produce object, full-image part, and zoomed part evidence for [K, 5] poses.

    Grounding flags indicate which cues were available for the instruction.
    With no object detection, all cue scores are zero for all candidates.

    为 [K, 5] 中的 K 个抓取姿态分别计算物体、全图部位和放大部位得分。
    同时输出各类线索是否可用。没有检测到物体时，所有候选的三类分数均为零。
    """
    centers = poses[:, :2]
    obj = grounding['object_boxes']
    object_score = np.zeros(len(poses), dtype=np.float32)
    if not len(obj):
        return object_score, object_score, object_score, False, False, False

    # The center provides the semantic match. Gripper contacts supply a weak
    # second cue against very wide rectangles crossing onto nearby objects.
    # 以抓取中心是否落在目标附近为主，夹爪两端提供较弱的辅助线索，
    # 减少开口过宽、跨到相邻物体上的抓取框。
    axis = np.stack((np.cos(poses[:, 4]), np.sin(poses[:, 4])), axis=-1)
    contact_delta = 0.45 * poses[:, 2:3] * axis
    center_support = support(centers, obj)
    contact_support = (support(centers - contact_delta, obj) + support(centers + contact_delta, obj)) / 2.0
    object_score = (obj[:, 4] * center_support * (0.8 + 0.2 * contact_support)).max(axis=1)

    part_score, has_part = localized_part_score(centers, obj, grounding['part_boxes'])
    zoom_score, has_zoom = localized_part_score(centers, obj, grounding.get('zoom_part_boxes', ()))
    return object_score, part_score, zoom_score, True, has_part, has_zoom


@torch.inference_mode()
def collect(rows, images, method, model, vocabulary, grounder, args, group):
    """Decode a pool per instruction and cache evidence and annotation matches.

    Each record holds grasp candidates [K, 5] and separate [K]
    arrays for annotation hits and each grounding cue. Annotation hits are
    computed for evaluation. Only the grounding cues and original order score
    a candidate at inference. The annotation used for the preview is the first
    one, while hit checking considers all valid annotations in the row.

    为每条指令解码一组候选，缓存其定位线索和标注匹配结果。每条记录含
    [K, 5] 候选、[K] 命中标记和各类 [K] 定位分数。匹配标注只用于评估，
    推理打分只用定位线索和初始排序。可视化展示第一条标注，命中判定使用
    当前指令下的全部有效标注。
    """
    dataset = RankingDataset(rows, images, vocabulary)
    loader = DataLoader(dataset, batch_size=args.batch, num_workers=2, pin_memory=True)
    records = []
    started = time.perf_counter()
    for image, grounding_image, tokens, indices in loader:
        data = [rows[index] for index in indices.tolist()]
        poses = method.decode(model(image.cuda(non_blocking=True), tokens.cuda(non_blocking=True)), topk=args.pool)
        regions = grounder.predict(grounding_image, [row['prompt'] for row in data])
        if len(regions) != len(data):
            raise ValueError('Grounder batch size does not match proposal batch')
        for row, candidates, grounding in zip(data, poses.float().cpu().numpy(), regions):
            truth = normalized(row['grasps'], row.get('width', 416), row.get('height', 416))
            hits = np.asarray([correct(candidate, truth) for candidate in candidates], dtype=bool)
            scores = region_scores(candidates, grounding)
            records.append((row, candidates, truth[0], hits, *scores))
        if len(records) % 128 < args.batch or len(records) == len(rows):
            print(f'{group} {len(records)}/{len(rows)} in {time.perf_counter() - started:.1f}s', flush=True)
    return records


def metrics(records, weights, predictions=False):
    """Evaluate fixed weights on a nonempty group of cached instructions.

    Top-1 and Top-5 count instructions with at least one valid choice among
    the selected ranks. The pool oracle checks every candidate and is unchanged
    by permutations. Optional exports keep the new and original first choices.

    用固定权重评估一组非空的指令记录。Top-1 和 Top-5 分别统计第一名和
    前五名中至少有一个有效候选的指令。候选覆盖率检查整个池，顺序变化
    不影响它。导出结果时同时保留重排前后的第一选择。
    """
    top1 = top5 = oracle = object_detected = part_detected = zoom_part_detected = 0
    changed = recovered = spoiled = 0
    items = []
    object_weight, part_weight, zoom_weight = weights
    for row, poses, target, hits, obj, part, zoom_part, has_obj, has_part, has_zoom in records:
        # Keep the original rank as a prior. With zero grounding weights, this
        # recovers the proposer's order and provides the matching baseline.
        # 以初始名次为先验。所有定位权重为零时，得分恢复候选原始顺序，
        # 对应同一候选池上的基线。
        prior = -np.log1p(np.arange(len(poses)))
        scores = prior + object_weight * obj + part_weight * part + zoom_weight * zoom_part
        ranks = np.argsort(-scores, kind='stable')
        first = ranks[0]
        top1 += bool(hits[first])
        top5 += bool(hits[ranks[:5]].any())
        oracle += bool(hits.any())
        object_detected += has_obj
        part_detected += has_part
        zoom_part_detected += has_zoom
        changed += bool(first != 0)
        recovered += bool(hits[first] and not hits[0])
        spoiled += bool(hits[0] and not hits[first])
        if predictions:
            items.append({'scene_id': row['scene_id'], 'target_id': row['target_id'], 'image': row['image'],
                          'prompt': row['prompt'], 'gt': target.tolist(), 'prediction': poses[first].tolist(),
                          'baseline_prediction': poses[0].tolist(), 'success': bool(hits[first]),
                          'baseline_success': bool(hits[0]), 'grounded_object': has_obj,
                          'grounded_part': has_part, 'grounded_zoom_part': has_zoom,
                          'selected_proposal': int(first)})
    n = len(records)
    return {'n': n, 'top1': top1/n, 'top5': top5/n, 'oracle_pool': oracle/n,
            'object_detected': object_detected, 'part_detected': part_detected,
            'zoom_part_detected': zoom_part_detected, 'changed': changed,
            'recovered': recovered, 'spoiled': spoiled}, items


def select_weights(records, candidates):
    """Choose validation weights by Top-1, Top-5, and then lighter weights.

    按验证集 Top-1、Top-5 的顺序选权重；同分时优先用更轻的定位权重。
    """
    def score(weights):
        result, _ = metrics(records, weights)
        return result['top1'], result['top5'], -sum(weights), -weights[1] - weights[2]

    return max(candidates, key=score)


def paired_switches(records, predictions):
    """Check whether two different prompts on one scene select distinct grasps.

    A scene qualifies if its two prompts refer to different objects. Both
    predictions must succeed on their own prompts. Each prediction is also checked
    against the other prompt's annotations, where a match fails the switch test.

    检查同一场景的两条指令是否分别选到了各自的抓取。场景须恰有两条
    指令，且指向不同物体。两条预测都须各自命中，再交叉检查另一条指令
    的标注。交叉命中表示这次切换没有区分开目标。
    """
    by_scene = {}
    for (row, *_), item in zip(records, predictions):
        by_scene.setdefault(row['scene_id'], []).append((row, item))
    successes = pairs = 0
    for examples in by_scene.values():
        if len(examples) != 2:
            continue
        (row_a, item_a), (row_b, item_b) = examples
        if row_a['object_id'] == row_b['object_id'] or row_a['prompt'] == row_b['prompt']:
            continue
        pairs += 2
        if item_a['success'] and item_b['success']:
            truth_a = normalized(row_a['grasps'], row_a['width'], row_a['height'])
            truth_b = normalized(row_b['grasps'], row_b['width'], row_b['height'])
            successes += not correct(np.asarray(item_b['prediction']), truth_a)
            successes += not correct(np.asarray(item_a['prediction']), truth_b)
    return {'strict_switch_successes': successes, 'strict_switch_queries': pairs}


def main():
    """Load local checkpoints, tune validation weights, and write test reports.

    读取本地检查点，选取重排权重，并生成测试报告。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--proposer', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--grounder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pool', type=int, default=24)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--val-limit', type=int, default=512)
    parser.add_argument('--validation-split', type=Path)
    parser.add_argument('--zoom-part', action='store_true')
    parser.add_argument('--test-groups', nargs='+', default=['test_seen', 'test_unseen'])
    args = parser.parse_args()
    if min(args.pool, args.batch, args.val_limit) <= 0:
        parser.error('Pool, batch and validation limit must be positive')
    cv2.setNumThreads(0)
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    # The proposer run provides best.pt and its matching vocabulary.json.
    # Grounding uses a separate, already-downloaded model on the same CUDA GPU.
    # 候选模型目录包含 best.pt 及配套的 vocabulary.json。定位使用另一个
    # 已下载的模型，两个模型放在同一张 CUDA 显卡上。
    device = torch.device('cuda:0')
    method, model, vocabulary, epoch = load_checkpoint(args.proposer, args.run, device)
    grounder = Grounder(args.grounder, device, zoom_parts=args.zoom_part)
    validation = read_rows(args.validation_split or args.splits / 'val.jsonl')
    # Fix the subset seed so the validation search can be reproduced.
    # 固定验证子集的采样种子，便于复现权重选择。
    rng = np.random.default_rng(121)
    indices = rng.choice(len(validation), min(len(validation), args.val_limit), replace=False)
    validation = collect([validation[index] for index in indices], args.images, method, model,
                         vocabulary, grounder, args, 'val')

    object_only = select_weights(validation, [(value, 0.0, 0.0) for value in OBJECT_WEIGHTS])
    object_and_part = select_weights(validation, [(obj, part, 0.0) for obj in OBJECT_WEIGHTS for part in PART_WEIGHTS])
    variants = {'original_order': (0.0, 0.0, 0.0), 'object_only': object_only, 'object_and_part': object_and_part}
    if args.zoom_part:
        variants['object_and_zoom_part'] = select_weights(
            validation, [(obj, 0.0, part) for obj in OBJECT_WEIGHTS for part in PART_WEIGHTS]
        )
    report = {'proposer': str(args.run), 'proposer_epoch': epoch, 'pool': args.pool,
              'grounder': str(args.grounder), 'grounding_rgb_size': 416, 'weights': variants,
              'validation': {key: metrics(validation, value)[0] for key, value in variants.items()},
              'test': {}}
    # Save validation results first and refresh the partial report per test group.
    # 先保存验证结果，每评估完一个测试组就更新一次阶段报告。
    safe_json(args.output / 'metrics.partial.json', report)
    for group in args.test_groups:
        rows = read_rows(args.splits / f'{group}.jsonl')
        records = collect(rows, args.images, method, model, vocabulary, grounder, args, group)
        report['test'][group] = {}
        for name, weights in variants.items():
            result, predictions = metrics(records, weights, predictions=True)
            result.update(paired_switches(records, predictions))
            report['test'][group][name] = result
            with (args.output / f'{group}_{name}.jsonl').open('w', encoding='utf-8') as output:
                for prediction in predictions:
                    output.write(json.dumps(prediction, ensure_ascii=False) + '\n')
            overlay(predictions, args.images, args.output / f'{group}_{name}.jpg')
        safe_json(args.output / 'metrics.partial.json', report)
    if {'test_seen', 'test_unseen'} <= report['test'].keys():
        report['harmonic_top1'] = {name: harmonic(report['test']['test_seen'][name]['top1'],
                                                 report['test']['test_unseen'][name]['top1']) for name in variants}
    safe_json(args.output / 'metrics.json', report)
    print('GROUNDED_RERANK', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
