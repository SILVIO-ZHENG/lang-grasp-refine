"""Shared candidate generation and scoring for the two learned 2-D variants."""

# Generate proposals with frozen backbones, construct supervision for the two
# candidate heads, and reuse the same scoring rules in validation and test.
# 用冻结的主模型生成候选，为两个候选头构造监督，并在验证与测试中复用同一套评分规则。

import itertools
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from panels import ROOT, load_panel_shard, read_regions, shard_path

# Resolve the shared code from the release tree so the entry point can run
# directly without installing the repository as a Python package.
# 从发布目录加载共享代码，允许直接运行入口脚本，无需先安装为 Python 包。
sys.path[:0] = [str(ROOT), str(ROOT / 'common')]
from candidate_features import K, agreement, candidate_rgb, feature_vector, supervised_targets
from dual_contact_ranking.ranking import merge_scores
from evaluate_seen_unseen import load_checkpoint
from geometry_refinement.refinement import apply_delta
from grounded_baseline import OBJECT_WEIGHTS, PART_WEIGHTS, region_scores
from learned import predict_learned, restore_learned
from run_experiment import SIZE, correct, normalized, tokenize


MODEL_NAMES = ('01_film_dense', '08_contact_pairs')
MODEL_DIRECTORIES = dict(zip(MODEL_NAMES, ('geometry_refinement', 'dual_contact_ranking'), strict=True))
VARIANTS = ('grounded_zoom', 'geometry_refinement', 'refined_heatmap_contact')

# Search these score weights and residual scales on validation Top-1 only.
# 这些评分权重和残差缩放系数仅使用验证集 Top-1 进行选择。
COEFFICIENTS = (0.15, 0.35, 0.65, 1.0, 1.6)
SHRINKS = (0.3, 0.6, 1.0)
REFINED_RANK = (0.0, 0.35, 0.8)
CONTACT_BIAS = (-3.5, -2.5, -1.5, -0.5)


def backbone_paths(root, method, tag):
    """Resolve source and weights while retaining the archive's method IDs.

    A benchmark root may use the current method folders or the original
    directory names. The fingerprint checks the actual source bytes,
    checkpoint, and vocabulary from the chosen directory.

    定位源码和权重，同时保留实验归档中的方法标识。

    实验根目录可使用现在的方案文件夹，也可使用历史目录名。指纹核验所选目录中
    实际读取的源码、检查点与词表。
    """
    directory = root / MODEL_DIRECTORIES[method]
    if not (directory / 'model.py').is_file():
        directory = root / method
    return directory / 'model.py', directory / tag


class Images(Dataset):
    # Each sample belongs to one instruction. Preserve its row index and
    # original image dimensions to align cached regions and grasp labels later.
    # 每条样本对应一条指令。保留行号和原图尺寸，供后续对齐缓存区域与抓取标注。
    def __init__(self, rows, root):
        self.rows = rows
        self.root = root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.root / self.rows[index]['image']) as source:
            image = source.convert('RGB')
        width, height = image.size
        # Match backbone training by padding to a centered square before
        # resizing to [3, SIZE, SIZE] and scaling RGB values to [0, 1].
        # 与主模型训练保持一致，先居中补成方图，再缩放为 [3, SIZE, SIZE]，
        # 转为数值在 [0, 1] 的 RGB 浮点张量。
        if width != height:
            side = max(width, height)
            square = Image.new('RGB', (side, side))
            square.paste(image, ((side - width) // 2, (side - height) // 2))
            image = square
        resized = image.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(resized).transpose(2, 0, 1).copy()).float() / 255.0, index, width, height


def geometry(boxes, regions):
    # Scoring uses object support and part support found inside the object crop.
    # 候选评分使用物体区域支持度，以及在物体裁剪图中找到的部位支持度。
    obj, _, zoom, *_ = region_scores(boxes, regions)
    return np.stack((obj, zoom), axis=-1)


def grounded(values, weights):
    # The logarithmic prior preserves the generator's ordering when grounding
    # gives equal evidence. Later candidates receive a larger penalty.
    # 当定位证据相同时，对数先验保留生成器原有顺序，靠后的候选受到更大的惩罚。
    return -np.log1p(np.arange(len(values))) + values @ np.asarray(weights)


def get_hits(boxes, truth):
    # A box is correct if at least one annotation has rotated IoU > 0.25
    # and its angle distance modulo pi is < 30 degrees.
    # 至少有一个标注与该候选同时满足旋转框 IoU > 0.25、模 π 的角度差 < 30°，
    # 该候选才记为成功。
    return np.fromiter((correct(box, truth) for box in boxes), dtype=bool, count=len(boxes))


class ProposalBank:
    # The two checkpoint tuples hold the model definition, evaluation-mode
    # weights, vocabulary, and selected epoch for each proposal branch.
    # 两路检查点各自保存模型定义、评估模式下的权重、词表和选中的训练轮次。
    def __init__(self, root, tag, device, fit=None):
        self.device = torch.device(device)
        self.base, self.aux = (load_checkpoint(*backbone_paths(root, name, tag), device)
                               for name in MODEL_NAMES)
        self.selection = None
        if fit is not None:
            self.attach_fit(*fit)

    def attach_fit(self, settings, folder):
        # Load the validation settings and trained heads as a matched pair.
        # 同时加载配套的验证集评分配置和已训练的候选头。
        self.selection = settings
        self.ranker, self.refiner = (
            restore_learned(torch.load(folder / name, map_location='cpu', weights_only=True), self.device)
            for name in ('learned_ranking.pt', 'pose_refiner.pt')
        )

    def _proposals(self, model_info, images, prompts):
        # Tokenize with this backbone's own vocabulary. Both decoders return
        # the same normalized pose format [B, 24, 5] even though the contact
        # model predicts endpoints before decoding.
        # 使用当前主模型自己的词表编码指令。两路解码后统一为 [B, 24, 5] 的
        # 归一化抓取框，双接触点模型在解码前输出的是端点。
        method, model, vocabulary, _ = model_info
        words = torch.stack([tokenize(prompt, vocabulary) for prompt in prompts]).to(self.device)
        boxes = method.decode(model(images, words), topk=K).float()
        if boxes.shape != (len(images), K, 5):
            raise ValueError(f'Expected one grasp pool [{len(images)},{K},5], got {tuple(boxes.shape)}')
        return boxes

    def raw(self, images, prompts, regions, include_aux=True):
        # Extract features in inference mode to keep both backbone outputs
        # fixed during head training. The feature vector uses RGB sampled at
        # each proposal, grounding evidence, and proposal metadata.
        # 在推理模式提取特征，在候选头训练时固定两路主模型的输出。特征来自候选周围的 RGB
        # 采样、定位结果和候选自身的信息。
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
        # Both heads read the original heatmap boxes. The ranker returns a
        # score, and the refiner returns a quality score plus five pose edits.
        # 两个头都读取原始热图抓取框。排序头输出分数，细化头输出质量分数和五维位姿修正。
        values = {'boxes': dense, 'features': features}
        rank, _ = predict_learned(self.ranker, values, self.device)
        refine, delta = predict_learned(self.refiner, values, self.device)
        return np.clip(rank, -6, 6), np.clip(refine, -6, 6), delta

    def results(self, images, prompts, regions):
        # Return one aligned candidate and score list per method for each
        # instruction. The first K entries of the combined list remain the
        # refined heatmap boxes so their scores and labels can be reused.
        # 为每条指令返回三种方法各自对齐的候选和分数。合并列表的前 K 项仍是
        # 细化后的热图框，因此能复用这部分分数和标注匹配结果。
        _, dense, aux, features = self.raw(images, prompts, regions)
        rank, refine, delta = self.learned(dense, features)
        weights = self.selection['grounded_zoom']['weights']
        refine_weight, shrink = self.selection['geometry_refinement']['weights']
        rank_weight, contact_bias = self.selection['refined_heatmap_contact']['weights']
        refined = apply_delta(dense, delta * shrink)
        combo = np.concatenate((refined, aux), axis=1)
        for i, detection in enumerate(regions):
            # Recompute grounding after moving a box. Only the refined
            # heatmap branch gets the extra learned rank score; contact boxes
            # instead receive a validation-selected branch offset.
            # 抓取框移动后重新计算定位支持度。额外排序分数只加在细化热图分支，
            # 双接触点分支则使用验证集选出的分数偏置。
            before = grounded(geometry(dense[i], detection), weights)
            after = grounded(geometry(refined[i], detection), weights) + refine_weight * refine[i]
            contact = grounded(geometry(aux[i], detection), weights)
            combined = merge_scores(after, rank[i], contact, rank_weight, contact_bias)
            yield dense[i], before, refined[i], after, combo[i], combined


def batches(args, rows, regions):
    # Preserve instruction order while loading images in parallel. Normalize
    # labels with the same square padding as the image so coordinates agree.
    # 并行读取图像时保留指令顺序。标签按照图像的同一套方形补边规则归一化，
    # 保证候选与标注在同一个坐标系。
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
    # Build head-learning records from the requested training or validation
    # shard. Contact proposals are needed only when validating the joint ranking.
    # 读取指定的训练集或验证集分片来构造候选头样本。双接触点候选仅用于验证联合排序。
    validation = group == 'val'
    data_shards = data['groups'][group]['shards']
    for index in range(len(data_shards)):
        rows = load_panel_shard(args.panels, data, group, index)
        path = shard_path(args.cache, group, index)
        detections, detector_id = read_regions(path, rows, digest, detector)
        # Carry the detector identity forward so later shards must match it.
        # 记录当前检测器指纹，后续所有分片都需要与它一致。
        detector = detector_id
        for images, _, prompts, regions, truths in batches(args, rows, detections):
            _, dense, aux, features = bank.raw(images, prompts, regions, include_aux=validation)
            # Supervise success and quality for each heatmap proposal. Supply
            # a matched rectangle and a mask that enables pose regression
            # only for candidates close enough to their chosen label.
            # 为每个热图候选构造成功标签和质量标签，并给出匹配框及回归掩码。
            # 只有足够接近所选标注的候选才参与位姿回归。
            targets = [supervised_targets(poses, truth) for poses, truth in zip(dense, truths, strict=True)]
            correct, quality, matched, regresses = [np.stack(values) for values in zip(*targets, strict=True)]
            dataset = {'features': features, 'boxes': dense, 'correct': correct, 'quality': quality,
                       'target': matched, 'regress': regresses}
            yield dataset, (aux, regions, truths) if validation else None, detector_id
        print('HEAD_INPUT', group, index, len(data_shards), len(rows), flush=True)


def score_order(scores, allowed=None):
    # Sort by decreasing score and keep the input order for ties. When a
    # shortlist is supplied, return indices into the original candidate list.
    # 按分数降序排列，同分时保留输入顺序。若提供候选子集，则返回原始列表中的索引。
    order = np.argsort(-scores, kind='stable') if allowed is None else allowed[np.argsort(-scores[allowed], kind='stable')]
    return order


def measure(boxes, hits, order, suppress=False):
    # Count first-choice success, any success among the selected five, and
    # coverage of the full pool. Only the Top-5 selection can suppress
    # proposals that have very similar centers and angles.
    # 分别统计首选是否成功、选出的五项中是否有成功框，以及整个候选池是否覆盖
    # 有效抓取。相近中心与角度的候选只会在 Top-5 选择时被压制。
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
    # Prefer the more conservative weight setting when success counts tie.
    # 成功次数相同时，优先选取权重绝对值总和更小的配置。
    return max(configs, key=lambda config: (scorer(config), -sum(map(abs, config))))


def _top1(hits, values):
    # Look up the success flag at each instruction's highest-scoring box.
    # 对每条指令，取分数最高候选对应的成功标签，然后加总。
    return int(np.take_along_axis(hits, values.argmax(axis=1, keepdims=True), axis=1).sum())


def _validation_metrics(boxes, hits, rows, suppress=False):
    # Reuse the same selection and hit counting as the test benchmark.
    # 复用测试评测的候选选择和成功统计规则。
    candidates = (measure(poses, correct, score_order(scores), suppress)
                  for poses, correct, scores in zip(boxes, hits, rows, strict=True))
    first, first_five, covered, _ = map(sum, zip(*candidates, strict=True))
    return metrics(len(boxes), {'top1_successes': first, 'top5_successes': first_five, 'oracle_successes': covered})


def metrics(n, counts):
    # Use the number of instructions as the denominator for all three rates.
    # 三个比率都以指令数为分母。
    return {'n': n, **counts, 'top1': counts['top1_successes'] / n,
            'top5': counts['top5_successes'] / n, 'oracle_pool': counts['oracle_successes'] / n}


def select(bank, validation, aux, regions, truths):
    # Tune in stages on validation. Freeze the baseline weights before
    # choosing refinement settings, then freeze those while merging branches.
    # 在验证集分阶段调参。固定基线权重后选择细化配置，固定细化配置后再合并两路。
    dense = validation['boxes']
    dense_hits = validation['correct']
    dense_geo = np.stack([geometry(boxes, detected) for boxes, detected in zip(dense, regions, strict=True)])
    aux_geo = np.stack([geometry(boxes, detected) for boxes, detected in zip(aux, regions, strict=True)])
    rank, refine, delta = bank.learned(dense, validation['features'])
    prior = -np.log1p(np.arange(K))
    base = _best_grid(itertools.product((*OBJECT_WEIGHTS[1:], 12.0), PART_WEIGHTS[1:]),
                      lambda weights: _top1(dense_hits, prior + dense_geo @ weights))
    # Moving proposals changes which annotations they match. Recompute hits
    # for the boxes produced at each residual scale.
    # 移动抓取框会改变它与标注的匹配情况。每种残差缩放都要用修正后的抓取框重新判断是否成功。
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
        # The rank weight affects the refined heatmap branch. A constant
        # offset calibrates contact scores against it and preserves their order.
        # 排序权重用于第一路。偏置校准双接触点分支与第一路的相对分数，
        # 不改变双接触点分支内部的候选顺序。
        rank_weight, bias = config
        return merge_scores(score, rank, aux_score, rank_weight, bias)

    combo = _best_grid(itertools.product(REFINED_RANK, CONTACT_BIAS),
                       lambda config: _top1(combo_hits, combine(config)))
    return {
        'grounded_zoom': {'weights': base, 'val': _validation_metrics(dense, dense_hits, prior + dense_geo @ base)},
        'geometry_refinement': {'weights': (scale, shrink), 'val': _validation_metrics(boxes, hits, score)},
        'refined_heatmap_contact': {'weights': combo, 'val': _validation_metrics(combo_boxes, combo_hits, combine(combo), True)},
    }
