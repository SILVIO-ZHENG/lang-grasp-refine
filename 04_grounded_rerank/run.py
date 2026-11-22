"""Rescore a fixed grasp pool with open-vocabulary object and part grounding."""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from evaluate_seen_unseen import harmonic, load_checkpoint
from run_experiment import GraspDataset, SIZE, correct, normalized, overlay, read_rows, safe_json

from grounding import Grounder


OBJECT_WEIGHTS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
PART_WEIGHTS = (0.0, 0.5, 1.0, 2.0, 4.0)


class RankingDataset(GraspDataset):
    def __getitem__(self, index):
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
    """Downweight points just outside a predicted region without hard rejection."""
    lower, upper = bounds[:, :2], bounds[:, 2:4]
    extension = np.maximum(lower - points[:, None], points[:, None] - upper).clip(min=0.0)
    distance = np.linalg.norm(extension / (upper - lower).clip(min=0.03), axis=-1)
    return np.exp(-0.5 * (distance / 0.22) ** 2)


def localized_part_score(centers, obj, part):
    if not len(part):
        return np.zeros(len(centers), dtype=np.float32), False
    lower = np.maximum(part[:, None, :2], obj[None, :, :2])
    upper = np.minimum(part[:, None, 2:4], obj[None, :, 2:4])
    intersection = (upper - lower).clip(min=0).prod(axis=-1)
    part_area = (part[:, 2:4] - part[:, :2]).prod(axis=-1)
    object_area = (obj[:, 2:4] - obj[:, :2]).prod(axis=-1)
    contained = (intersection >= 0.7 * part_area[:, None]) & (part_area[:, None] <= 0.8 * object_area)
    # Require a predicted object to support the part evidence.
    confidence = np.sqrt((contained * obj[:, 4]).max(axis=1) * part[:, 4])
    score = (confidence * support(centers, part)).max(axis=1)
    return score, bool(confidence.any())


def region_scores(poses, grounding):
    centers = poses[:, :2]
    obj = grounding['object_boxes']
    object_score = np.zeros(len(poses), dtype=np.float32)
    if not len(obj):
        return object_score, object_score, object_score, False, False, False

    # The center provides the semantic match. Gripper contacts supply a weak
    # second cue against very wide rectangles crossing onto nearby objects.
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
    top1 = top5 = oracle = object_detected = part_detected = zoom_part_detected = 0
    changed = recovered = spoiled = 0
    items = []
    object_weight, part_weight, zoom_weight = weights
    for row, poses, target, hits, obj, part, zoom_part, has_obj, has_part, has_zoom in records:
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
    def score(weights):
        result, _ = metrics(records, weights)
        return result['top1'], result['top5'], -sum(weights), -weights[1] - weights[2]

    return max(candidates, key=score)


def paired_switches(records, predictions):
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
    device = torch.device('cuda:0')
    method, model, vocabulary, epoch = load_checkpoint(args.proposer, args.run, device)
    grounder = Grounder(args.grounder, device, zoom_parts=args.zoom_part)
    validation = read_rows(args.validation_split or args.splits / 'val.jsonl')
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
