"""Shared candidate generation and scoring for the two learned 2-D variants."""

import itertools
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from panels import ROOT, load_panel_shard, read_regions, shard_path

sys.path[:0] = [str(ROOT / name) for name in ('05_grasp_followups', '04_grounded_rerank', 'common')]
from candidate_features import K, agreement, candidate_rgb, feature_vector, supervised_targets
from evaluate_seen_unseen import load_checkpoint
from learned import apply_delta, predict_learned, restore_learned
from run import OBJECT_WEIGHTS, PART_WEIGHTS, region_scores
from run_experiment import SIZE, correct, normalized, tokenize


MODEL_NAMES = ('01_film_dense', '08_contact_pairs')
VARIANTS = ('grounded_zoom', 'geometry_refinement', 'refined_heatmap_contact')
COEFFICIENTS = (0.15, 0.35, 0.65, 1.0, 1.6)
SHRINKS = (0.3, 0.6, 1.0)
REFINED_RANK = (0.0, 0.35, 0.8)
CONTACT_BIAS = (-3.5, -2.5, -1.5, -0.5)


class Images(Dataset):
    def __init__(self, rows, root):
        self.rows = rows
        self.root = root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.root / self.rows[index]['image']) as source:
            image = source.convert('RGB')
        width, height = image.size
        if width != height:
            side = max(width, height)
            square = Image.new('RGB', (side, side))
            square.paste(image, ((side - width) // 2, (side - height) // 2))
            image = square
        resized = image.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(resized).transpose(2, 0, 1).copy()).float() / 255.0, index, width, height


def geometry(boxes, regions):
    obj, _, zoom, *_ = region_scores(boxes, regions)
    return np.stack((obj, zoom), axis=-1)


def grounded(values, weights):
    return -np.log1p(np.arange(len(values))) + values @ np.asarray(weights)


def get_hits(boxes, truth):
    return np.fromiter((correct(box, truth) for box in boxes), dtype=bool, count=len(boxes))


class ProposalBank:
    def __init__(self, root, tag, device, fit=None):
        self.device = torch.device(device)
        self.base, self.aux = (load_checkpoint(root / name / 'model.py', root / name / tag, device)
                               for name in MODEL_NAMES)
        self.selection = None
        if fit is not None:
            self.attach_fit(*fit)

    def attach_fit(self, settings, folder):
        self.selection = settings
        self.ranker, self.refiner = (
            restore_learned(torch.load(folder / name, map_location='cpu', weights_only=True), self.device)
            for name in ('learned_ranking.pt', 'pose_refiner.pt')
        )

    def _proposals(self, model_info, images, prompts):
        method, model, vocabulary, _ = model_info
        words = torch.stack([tokenize(prompt, vocabulary) for prompt in prompts]).to(self.device)
        boxes = method.decode(model(images, words), topk=K).float()
        if boxes.shape != (len(images), K, 5):
            raise ValueError(f'Expected one grasp pool [{len(images)},{K},5], got {tuple(boxes.shape)}')
        return boxes

    def raw(self, images, prompts, regions, include_aux=True):
        with torch.inference_mode():
            images = images.to(self.device, non_blocking=True)
            dense = self._proposals(self.base, images, prompts)
            patches = candidate_rgb(images, dense)
            dense = dense.cpu().numpy()
            aux = self._proposals(self.aux, images, prompts).cpu().numpy() if include_aux else None
            features = np.stack([feature_vector(box, detections, rgb)
                                 for box, detections, rgb in zip(dense, regions, patches, strict=True)])
        return images, dense, aux, features

    def learned(self, dense, features):
        values = {'boxes': dense, 'features': features}
        rank, _ = predict_learned(self.ranker, values, self.device)
        refine, delta = predict_learned(self.refiner, values, self.device)
        return np.clip(rank, -6, 6), np.clip(refine, -6, 6), delta

    def results(self, images, prompts, regions):
        _, dense, aux, features = self.raw(images, prompts, regions)
        rank, refine, delta = self.learned(dense, features)
        weights = self.selection['grounded_zoom']['weights']
        refine_weight, shrink = self.selection['geometry_refinement']['weights']
        rank_weight, contact_bias = self.selection['refined_heatmap_contact']['weights']
        refined = apply_delta(dense, delta * shrink)
        combo = np.concatenate((refined, aux), axis=1)
        for i, detection in enumerate(regions):
            before = grounded(geometry(dense[i], detection), weights)
            after = grounded(geometry(refined[i], detection), weights) + refine_weight * refine[i]
            contact = grounded(geometry(aux[i], detection), weights) + contact_bias
            combined = np.concatenate((after + rank_weight * rank[i], contact))
            yield dense[i], before, refined[i], after, combo[i], combined


def batches(args, rows, regions):
    loader = DataLoader(Images(rows, args.images), batch_size=args.batch, num_workers=args.loader_workers,
                        pin_memory=args.device.startswith('cuda'))
    for images, indices, widths, heights in loader:
        ids = indices.tolist()
        sample = [rows[i] for i in ids]
        detected = [regions[i] for i in ids]
        for row, width, height in zip(sample, widths.tolist(), heights.tolist(), strict=True):
            if (row.get('width', width), row.get('height', height)) != (width, height):
                raise ValueError(f'Grasp labels use different dimensions from the image: {row["image"]}')
        truths = [normalized(row['grasps'], width, height)
                  for row, width, height in zip(sample, widths.tolist(), heights.tolist(), strict=True)]
        yield images, sample, [row['prompt'] for row in sample], detected, truths


def fitting_records(args, bank, data, digest, group, detector=None):
    validation = group == 'val'
    data_shards = data['groups'][group]['shards']
    for index in range(len(data_shards)):
        rows = load_panel_shard(args.panels, data, group, index)
        path = shard_path(args.cache, group, index)
        detections, detector_id = read_regions(path, rows, digest, detector)
        detector = detector_id
        for images, _, prompts, regions, truths in batches(args, rows, detections):
            _, dense, aux, features = bank.raw(images, prompts, regions, include_aux=validation)
            targets = [supervised_targets(poses, truth) for poses, truth in zip(dense, truths, strict=True)]
            correct, quality, matched, regresses = [np.stack(values) for values in zip(*targets, strict=True)]
            dataset = {'features': features, 'boxes': dense, 'correct': correct, 'quality': quality,
                       'target': matched, 'regress': regresses}
            yield dataset, (aux, regions, truths) if validation else None, detector_id
        print('HEAD_INPUT', group, index, len(data_shards), len(rows), flush=True)


def score_order(scores, allowed=None):
    order = np.argsort(-scores, kind='stable') if allowed is None else allowed[np.argsort(-scores[allowed], kind='stable')]
    return order


def measure(boxes, hits, order, suppress=False):
    top = int(order[0])
    chosen = order[:5]
    if suppress:
        chosen = []
        for index in order:
            if chosen and agreement(boxes[index:index + 1], boxes[chosen])[0] > 0.94:
                continue
            chosen.append(int(index))
            if len(chosen) == 5:
                break
    return int(hits[top]), int(hits[chosen].any()), int(hits.any()), top


def _best_grid(configs, scorer):
    return max(configs, key=lambda config: (scorer(config), -sum(map(abs, config))))


def _top1(hits, values):
    return int(np.take_along_axis(hits, values.argmax(axis=1, keepdims=True), axis=1).sum())


def _validation_metrics(boxes, hits, rows, suppress=False):
    candidates = (measure(poses, correct, score_order(scores), suppress)
                  for poses, correct, scores in zip(boxes, hits, rows, strict=True))
    first, first_five, covered, _ = map(sum, zip(*candidates, strict=True))
    return metrics(len(boxes), {'top1_successes': first, 'top5_successes': first_five, 'oracle_successes': covered})


def metrics(n, counts):
    return {'n': n, **counts, 'top1': counts['top1_successes'] / n,
            'top5': counts['top5_successes'] / n, 'oracle_pool': counts['oracle_successes'] / n}


def select(bank, validation, aux, regions, truths):
    dense = validation['boxes']
    dense_hits = validation['correct']
    dense_geo = np.stack([geometry(boxes, detected) for boxes, detected in zip(dense, regions, strict=True)])
    aux_geo = np.stack([geometry(boxes, detected) for boxes, detected in zip(aux, regions, strict=True)])
    rank, refine, delta = bank.learned(dense, validation['features'])
    prior = -np.log1p(np.arange(K))
    base = _best_grid(itertools.product((*OBJECT_WEIGHTS[1:], 12.0), PART_WEIGHTS[1:]),
                      lambda weights: _top1(dense_hits, prior + dense_geo @ weights))
    refine_candidates = {}
    for shrink in SHRINKS:
        boxes = apply_delta(dense, delta * shrink)
        hits = np.stack([get_hits(poses, truth) for poses, truth in zip(boxes, truths, strict=True)])
        scores = np.stack([grounded(geometry(poses, detected), base)
                           for poses, detected in zip(boxes, regions, strict=True)])
        refine_candidates[shrink] = boxes, hits, scores
    scale, shrink = _best_grid(itertools.product(COEFFICIENTS, SHRINKS), lambda config: _top1(
        refine_candidates[config[1]][1], refine_candidates[config[1]][2] + config[0] * refine))
    boxes, hits, score = refine_candidates[shrink]
    score = score + scale * refine
    aux_hits = np.stack([get_hits(poses, truth) for poses, truth in zip(aux, truths, strict=True)])
    aux_score = prior + aux_geo @ base
    combo_boxes = np.concatenate((boxes, aux), axis=1)
    combo_hits = np.concatenate((hits, aux_hits), axis=1)

    def combine(config):
        rank_weight, bias = config
        return np.concatenate((score + rank_weight * rank, aux_score + bias), axis=1)

    combo = _best_grid(itertools.product(REFINED_RANK, CONTACT_BIAS),
                       lambda config: _top1(combo_hits, combine(config)))
    return {
        'grounded_zoom': {'weights': base, 'val': _validation_metrics(dense, dense_hits, prior + dense_geo @ base)},
        'geometry_refinement': {'weights': (scale, shrink), 'val': _validation_metrics(boxes, hits, score)},
        'refined_heatmap_contact': {'weights': combo, 'val': _validation_metrics(combo_boxes, combo_hits, combine(combo), True)},
    }
