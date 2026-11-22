"""Evaluate one trained checkpoint on Seen and Unseen test instructions.

Read the test groups supplied by the split directory, report them separately,
and combine their Top-1 and Top-5 success rates with a harmonic mean.

用同一个训练检查点评估 Seen 和 Unseen 两组测试指令。分别报告两组的
结果，并计算 Top-1 与 Top-5 成功率的调和平均。
"""

import argparse
import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader

from run_experiment import GraspDataset, evaluate, load_method, overlay, read_rows, safe_json


def harmonic(seen, unseen):
    """Combine two rates as 2xy/(x+y), with zero for two zero rates.

    用 2xy/(x+y) 合并两组成功率。两个输入都为零时返回零。
    """
    return 2 * seen * unseen / (seen + unseen) if seen + unseen else 0.0


def load_checkpoint(method_path, run, device):
    """Rebuild the original architecture and vocabulary before loading best.pt.

    Use the run's saved vocabulary to reproduce the embedding dimensions and
    token IDs. The three extra IDs are reserved by the training tokenizer.
    Return the module, inference-ready model, vocabulary, and selected epoch.

    加载 best.pt 前先恢复原模型结构和词表。使用训练目录中的词表还原
    嵌入层维度及词元 ID，另加训练分词器预留的三个 ID。返回模型模块、
    评估模式下的模型、词表和选中的训练轮次。
    """
    method = load_method(method_path)
    vocabulary = json.loads((run / 'vocabulary.json').read_text(encoding='utf-8'))
    model = method.build(len(vocabulary) + 3).to(device)
    checkpoint = torch.load(run / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    return method, model.eval(), vocabulary, checkpoint['epoch']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    args = parser.parse_args()
    device = torch.device('cuda:0')
    method, model, vocabulary, epoch = load_checkpoint(args.method, args.run, device)
    torch.set_num_threads(2)
    cv2.setNumThreads(0)
    report = {'best_epoch': epoch, 'protocol': 'LGD released object IDs, scene-disjoint training, ++ instructions'}
    # Use the same checkpoint for both precomputed groups. Reuse the evaluator
    # with prompt-switch diagnostics enabled for each group.
    # 两组预先划好的测试集共用同一个检查点。对每组均启用指令切换诊断。
    for group in ('seen', 'unseen'):
        rows = read_rows(args.splits / f'test_{group}.jsonl')
        data = GraspDataset(rows, args.images, vocabulary)
        loader = DataLoader(data, batch_size=48, num_workers=2, pin_memory=True)
        metrics, predictions = evaluate(method, model, loader, device, counterfactual=True)
        report[group] = metrics
        with (args.run / f'{group}_predictions.jsonl').open('w', encoding='utf-8') as output:
            for prediction in predictions:
                output.write(json.dumps(prediction, ensure_ascii=False) + '\n')
        overlay(predictions, args.images, args.run / f'{group}_qualitative.jpg')
    for key in ('top1', 'top5'):
        report[f'h_{key}'] = harmonic(report['seen'][key], report['unseen'][key])
    safe_json(args.run / 'seen_unseen_metrics.json', report)
    print('SEEN_UNSEEN', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
