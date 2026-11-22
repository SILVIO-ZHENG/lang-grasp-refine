"""Fit candidate heads and evaluate the language-guided 2-D grasp variants."""

# The workflow fits the two candidate heads, evaluates the three methods on
# test shards, and aggregates their results after validating every shard.
# 按顺序训练两个候选头，在测试分片上评测三种方法，然后核验所有分片并汇总结果。

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from panels import detector_identity, load_panel_shard, manifest, positive, read_regions, sha256, shard_path
from panels import TEST_GROUPS, write_json, write_lines
from pipeline import (MODEL_NAMES, VARIANTS, ProposalBank, backbone_paths, batches, fitting_records, get_hits,
                      measure, metrics, score_order, select)

import cv2
import numpy as np
import torch

from evaluate_seen_unseen import harmonic
from learned import checkpoint_learned, train_learned


HEAD_FILES = ('learned_ranking.pt', 'pose_refiner.pt')
SELECTION = 'validation_selection.json'
VERSION = 1


def read_json(path):
    # Receipts and manifests use UTF-8 so instruction text survives serialization.
    # 完成记录和清单统一使用 UTF-8，完整保留指令文本。
    return json.loads(path.read_text(encoding='utf-8'))


def checkpoints(root, tag):
    # Fingerprint source, weights, and vocabulary separately because each can
    # affect the candidate pool produced by a checkpoint.
    # 分别记录模型源码、权重和词表的指纹，三者都可能改变检查点生成的候选池。
    fingerprints = {}
    for method in MODEL_NAMES:
        source, run = backbone_paths(root, method, tag)
        fingerprints[method] = {
            'model.py': sha256(source),
            f'{tag}/best.pt': sha256(run / 'best.pt'),
            f'{tag}/vocabulary.json': sha256(run / 'vocabulary.json'),
        }
    return fingerprints


def load_fit(folder, root=None, tag=None, panel_digest=None):
    # Validate the fitted heads and their recorded inputs before reuse.
    # Callers that load backbone weights also verify the backbone hashes.
    # 复用候选头前，先校验权重及其记录的输入。需要加载主模型时还会核验主模型指纹。
    path = folder / SELECTION
    config = read_json(path)
    if config['version'] != VERSION or (panel_digest is not None and config['panel_sha256'] != panel_digest):
        raise ValueError(f'Head training was completed for a different panel: {path}')
    if root is not None and checkpoints(root, tag) != config['backbones']:
        raise ValueError(f'Backbone model files differ from the head training configuration: {path}')
    if any(sha256(folder / name) != digest for name, digest in config['heads'].items()):
        raise ValueError(f'Learned head weights changed: {path}')
    return config, sha256(path)


def _load_head_panel(args, bank, data, digest, group, detector):
    # Concatenate supervised records along the instruction axis. The
    # validation-only contact boxes, regions, and labels keep that same order.
    # 沿指令轴合并监督样本。验证专用的接触点候选、定位区域和标注保持同样的顺序。
    pieces = []
    aux, regions, truths = [], [], []
    for values, extra, detector in fitting_records(args, bank, data, digest, group, detector):
        pieces.append(values)
        if extra is not None:
            aux.append(extra[0])
            regions.extend(extra[1])
            truths.extend(extra[2])
    combined = {key: np.concatenate([part[key] for part in pieces], axis=0) for key in pieces[0]}
    return combined, (np.concatenate(aux), regions, truths) if aux else None, detector


def fit(args):
    # A finished fit can be reused when the panel, checkpoints, and seed agree.
    # The validation receipt is written after the checkpoints and history.
    # 数据清单、检查点和随机种子一致时，可复用已完成的训练。
    # 验证配置在候选头权重和训练历史写好后才落盘。
    data, digest = manifest(args.panels)
    if (args.output / SELECTION).is_file():
        previous, _ = load_fit(args.output, args.root, args.tag, digest)
        if previous['seed'] != args.seed:
            raise ValueError('Existing head training used a different random seed')
        print('FIT_ALREADY_COMPLETE', args.output, flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    base_sha = checkpoints(args.root, args.tag)
    bank = ProposalBank(args.root, args.tag, args.device)
    # Training supplies the head supervision. Validation selects head
    # checkpoints, calibrates scores, and chooses the final scoring weights.
    # 训练集提供候选头监督。验证集用来选择检查点、校准分数和选择最终评分权重。
    train_set, _, detector = _load_head_panel(args, bank, data, digest, 'train', None)
    validation, info, detector = _load_head_panel(args, bank, data, digest, 'val', detector)
    bank.ranker, bank.refiner, history = train_learned(train_set, validation, args.device, seed=args.seed)
    report = select(bank, validation, *info)
    if checkpoints(args.root, args.tag) != base_sha:
        raise ValueError('Backbone files changed during head training')
    # Stage each checkpoint first so an interrupted write cannot replace a
    # usable file with a partial checkpoint.
    # 先写入临时文件，避免保存中断时用不完整权重替换可用的检查点。
    for name, head in zip(HEAD_FILES, (bank.ranker, bank.refiner), strict=True):
        path = args.output / name
        stage = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
        torch.save(checkpoint_learned(head), stage)
        stage.replace(path)
    write_json(args.output / 'learner_history.json', history)
    result = {'version': VERSION, 'panel_sha256': digest, 'detector': detector,
              'backbones': base_sha, 'backbone_epochs': dict(zip(MODEL_NAMES, (bank.base[3], bank.aux[3]))),
              'heads': {name: sha256(args.output / name) for name in HEAD_FILES}, 'seed': args.seed,
              'refiner_head_train_rows': len(train_set['boxes']), 'selection_val_rows': len(validation['boxes']),
              'selection': report}
    write_json(args.output / SELECTION, result)
    print('FIT_COMPLETE', json.dumps(result), flush=True)


def _evaluation_signature(panel_digest, fit_digest):
    # Bind result shards to the exact panel and fitted-head configuration.
    # 将评测分片绑定到对应的数据清单和候选头训练配置。
    return {'version': VERSION, 'panel_sha256': panel_digest, 'fit_sha256': fit_digest}


def _eval_rows(args, bank, rows, detections, totals):
    # Compare all three methods on the same instruction and annotations.
    # Test annotations are used to measure already-scored proposals.
    # 在同一条指令和同一组标注上比较三种方法。
    # 测试标注只用于评测已完成打分的候选。
    for images, source, prompts, regions, truths in batches(args, rows, detections):
        stream = bank.results(images, prompts, regions)
        for row, truth, result in zip(source, truths, stream, strict=True):
            raw, base_score, refined, refine_score, combo, scores = result
            raw_hits = get_hits(raw, truth)
            refined_hits = get_hits(refined, truth)
            # The combined pool begins with the refined boxes, so only the
            # appended contact boxes require additional ground-truth matching.
            # 合并候选池以细化后的热图框开头，因此只需对附加的接触点候选再匹配标注。
            combo_hits = np.concatenate((refined_hits, get_hits(combo[len(refined):], truth)))
            groups = [(raw, raw_hits, score_order(base_score), False),
                      (refined, refined_hits, score_order(refine_score), False),
                      (combo, combo_hits, score_order(scores), True)]
            predictions = {}
            for name, (boxes, hits, order, distinct) in zip(VARIANTS, groups, strict=True):
                top1, top5, oracle, top = measure(boxes, hits, order, distinct)
                totals[name].update(top1_successes=top1, top5_successes=top5, oracle_successes=oracle)
                predictions[name] = {'success': bool(top1), 'prediction': boxes[top].tolist()}
            # Keep one reference box for inspection. The metrics above compare
            # with all valid annotations for the instruction.
            # 留下一个参考抓取框供检查。上面的指标使用这条指令的全部有效标注。
            yield {key: row[key] for key in ('scene_id', 'target_id', 'image', 'prompt')} | {
                'gt': truth[0].tolist(), 'methods': predictions}


def _verify_receipt(path, sign, n, predictions):
    # Reuse a completed shard only when its run identity, expected row count,
    # and saved prediction bytes all match the receipt.
    # 只有运行标识、预期行数和预测文件内容均通过核验时，才复用已完成的分片。
    receipt = read_json(path)
    if any(receipt[key] != value for key, value in sign.items()) or receipt['rows'] != n:
        raise ValueError(f'Evaluation shard was written for a different run: {path}')
    if sha256(predictions) != receipt['predictions_sha256']:
        raise ValueError(f'Evaluation predictions changed: {predictions}')
    return receipt


def evaluate(args):
    # Workers own disjoint shard indices by stride. Initialize the models only
    # when this worker finds a shard that still needs evaluation.
    # 各 worker 按固定步长处理互不重叠的分片。找到尚未评测的分片后才加载模型。
    data, panel_digest = manifest(args.panels)
    fit_settings, fit_digest = load_fit(args.fit, args.root, args.tag, panel_digest)
    sign = _evaluation_signature(panel_digest, fit_digest)
    bank = None
    for group in args.groups:
        shards = data['groups'][group]['shards']
        for index in range(args.worker, len(shards), args.workers):
            output = shard_path(args.output, group, index)
            receipt = output.with_suffix('.json')
            if receipt.is_file():
                _verify_receipt(receipt, sign, shards[index]['rows'], output)
                continue
            if bank is None:
                bank = ProposalBank(args.root, args.tag, args.device, (fit_settings['selection'], args.fit))
            rows = load_panel_shard(args.panels, data, group, index)
            cache = shard_path(args.cache, group, index)
            detections, _ = read_regions(cache, rows, panel_digest, fit_settings['detector'])
            totals = {name: Counter() for name in VARIANTS}
            # Iterating the prediction generator also updates the counts.
            # Commit the receipt only after the prediction file is complete.
            # 消费预测生成器的同时更新统计。预测文件完整写入后才保存完成记录。
            write_lines(output, _eval_rows(args, bank, rows, detections, totals))
            result = {**sign, 'rows': len(rows), 'counts': totals,
                      'predictions_sha256': sha256(output)}
            write_json(receipt, result)
            print('EVALUATED', group, index, len(rows), flush=True)


def summarize(args):
    # Add integer successes across all shards before dividing by total
    # instructions. Averaging shard rates would give a short shard extra weight.
    # 先累加所有分片的成功次数，再除以总指令数，避免直接平均各分片的比例时
    # 给短分片过高权重。
    data, panel_digest = manifest(args.panels)
    fit_settings, fit_digest = load_fit(args.fit, panel_digest=panel_digest)
    sign = _evaluation_signature(panel_digest, fit_digest)
    variants = {name: {} for name in VARIANTS}
    for group in TEST_GROUPS:
        total, counts = 0, {name: Counter() for name in VARIANTS}
        for index, expected in enumerate(data['groups'][group]['shards']):
            predictions = shard_path(args.evaluation, group, index)
            receipt = _verify_receipt(predictions.with_suffix('.json'), sign, expected['rows'], predictions)
            total += receipt['rows']
            for name in VARIANTS:
                counts[name].update(receipt['counts'][name])
        # Report results only after every eligible test instruction was covered.
        # 确认所有符合条件的测试指令都已评测，再发布汇总指标。
        if total != data['groups'][group]['rows'] or total != data['groups'][group]['input_rows']:
            raise ValueError(f'Evaluation did not cover every eligible {group} instruction')
        for name in VARIANTS:
            variants[name][group] = metrics(total, counts[name]) | {'scenes': data['groups'][group]['scenes']}
    for value in variants.values():
        # H summarizes Seen and Unseen with equal weight for the two groups.
        # H 对 Seen 和 Unseen 两组的成功率取调和平均，给两组同等权重。
        for top in ('top1', 'top5'):
            value[f'h_{top}'] = harmonic(value['test_seen'][top], value['test_unseen'][top])
    report = {**sign, 'metric': 'rotated IoU > 0.25 and angle modulo pi < 30 degrees, per instruction',
              'groups': {name: {key: count for key, count in value.items() if key != 'shards'}
                         for name, value in data['groups'].items()},
              'refiner_train_rows': fit_settings['refiner_head_train_rows'],
              'validation_settings': fit_settings['selection'],
              'offline_2d': variants}
    write_json(args.output, report)
    print('SUMMARY_COMPLETE', json.dumps(variants), flush=True)


def main():
    # Keep fit, evaluation workers, and summary independent so the last two
    # stages can run separately or resume from committed artifacts.
    # 将训练、分片评测和汇总设为独立命令，评测和汇总可分别执行，
    # 也可基于已经写好的产物继续。
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    def configure(name, function, paths, proposals=False, workers=False):
        command = commands.add_parser(name)
        command.set_defaults(function=function)
        for path in paths.split():
            command.add_argument(f'--{path}', required=True, type=Path)
        if proposals:
            command.add_argument('--tag', required=True)
            command.add_argument('--device', default='cuda:0')
            command.add_argument('--batch', type=positive, default=16)
        if workers:
            command.add_argument('--loader-workers', type=int, default=2)
        return command

    train_parser = configure('fit', fit, 'root panels cache images output', proposals=True, workers=True)
    train_parser.add_argument('--seed', type=int, default=121)
    ev_parser = configure('evaluate', evaluate, 'root fit panels cache images output', True, True)
    ev_parser.add_argument('--worker', type=int, default=0)
    ev_parser.add_argument('--workers', type=positive, default=1)
    ev_parser.add_argument('--groups', choices=TEST_GROUPS, nargs='+', default=TEST_GROUPS)
    configure('summarize', summarize, 'panels fit evaluation output')
    args = parser.parse_args()
    if args.command in ('fit', 'evaluate') and (
            args.tag in ('', '.', '..') or Path(args.tag).name != args.tag):
        parser.error('Expected a plain backbone run tag')
    if getattr(args, 'loader_workers', 0) < 0:
        parser.error('Invalid loader worker count')
    if args.command == 'evaluate' and not 0 <= args.worker < args.workers:
        parser.error('Invalid evaluation worker index')
    if args.command != 'summarize':
        torch.set_num_threads(2)
        cv2.setNumThreads(0)
    args.function(args)


if __name__ == '__main__':
    main()
