"""Evaluate a trained method on the two released LGD test groups."""

import argparse
import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader

from run_experiment import GraspDataset, evaluate, load_method, overlay, read_rows, safe_json


def harmonic(seen, unseen):
    return 2 * seen * unseen / (seen + unseen) if seen + unseen else 0.0


def load_checkpoint(method_path, run, device):
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
