"""Freeze split panels and cache detector output in independently resumable shards."""

import argparse
import hashlib
import heapq
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('train', 'val', 'test_seen', 'test_unseen')
TEST_GROUPS = GROUPS[2:]
PANEL_VERSION = 1
REGION_NAMES = ('object_boxes', 'part_boxes', 'zoom_part_boxes')


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def json_lines(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    temp.replace(path)


def sha256(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def row_key(row):
    return row['scene_id'], row['target_id'], row['prompt']


def shard_path(folder, group, index, suffix='.jsonl'):
    return folder / group / f'{index:06d}{suffix}'


def _sample_score(group, row):
    encoded = json.dumps((217 if group == 'train' else 121, *row_key(row)), ensure_ascii=False)
    return int.from_bytes(hashlib.blake2b(encoded.encode(), digest_size=8).digest(), 'big')


def prepare(args):
    expected = {'version': PANEL_VERSION, 'shard_size': args.shard_size,
                'train_count': None if args.all_train else args.train_count,
                'val_count': args.val_count,
                'sources': {group: sha256(args.splits / f'{group}.jsonl') for group in GROUPS}}
    manifest_path = args.output / 'manifest.json'
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if any(previous[key] != value for key, value in expected.items()):
            raise ValueError('Existing panel was prepared with a different configuration. Use a new output directory.')
        print('PANELS_READY', json.dumps(previous), flush=True)
        return
    completed = {}
    scenes = {}
    for group in GROUPS:
        limit = expected.get(f'{group}_count')
        sample, buffer, shards = [], [], []
        n = n_scenes = 0

        def flush():
            path = shard_path(args.output, group, len(shards))
            write_lines(path, buffer)
            shards.append({'rows': len(buffer), 'sha256': sha256(path)})
            buffer.clear()

        for row in json_lines(args.splits / f'{group}.jsonl'):
            scene, _, _ = row_key(row)
            owner = scenes.setdefault(scene, (group, n))
            if owner[0] != group:
                raise ValueError(f'Scene overlap between {owner[0]} and {group}: {scene}')
            n_scenes += owner[1] == n
            if not row['image'] or not row['grasps']:
                raise ValueError(f'Missing image name or grasp annotation for {row_key(row)}')
            if limit is None:
                buffer.append(row)
                if len(buffer) == args.shard_size:
                    flush()
            else:
                item = (-_sample_score(group, row), -n, row)
                if len(sample) < limit:
                    heapq.heappush(sample, item)
                elif item > sample[0]:
                    heapq.heapreplace(sample, item)
            n += 1
        included_scenes = n_scenes
        if limit is not None:
            included_scenes = len({row['scene_id'] for _, _, row in sample})
            for _, _, row in sorted(sample, key=lambda item: -item[1]):
                buffer.append(row)
                if len(buffer) == args.shard_size:
                    flush()
        if buffer:
            flush()
        if not shards:
            raise ValueError(f'No usable rows in {group}')
        completed[group] = {'input_rows': n, 'input_scenes': n_scenes,
                            'rows': sum(shard['rows'] for shard in shards),
                            'scenes': included_scenes, 'shards': shards}
        print('PANEL', group, n, n_scenes, completed[group]['rows'], len(shards), flush=True)
    if any(sha256(args.splits / f'{group}.jsonl') != expected['sources'][group] for group in GROUPS):
        raise RuntimeError('Input splits changed while preparing the panels')
    write_json(manifest_path, {**expected, 'groups': completed})


def manifest(panels):
    path = panels / 'manifest.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    if data['version'] != PANEL_VERSION:
        raise ValueError(f'Unsupported panel version: {data["version"]}')
    return data, sha256(path)


def load_panel_shard(panels, data, group, index):
    path = shard_path(panels, group, index)
    expected = data['groups'][group]['shards'][index]
    if sha256(path) != expected['sha256']:
        raise ValueError(f'Panel shard changed: {path}')
    rows = list(json_lines(path))
    if len(rows) != expected['rows']:
        raise ValueError(f'Unexpected panel shard size: {path}')
    return rows


def read_regions(path, rows, panel_digest, detector=None, *, require_receipt=True):
    import numpy as np

    result = []
    stream = json_lines(path)
    header = next(stream, None)
    if header is None or header['panel_sha256'] != panel_digest or header['rows'] != len(rows):
        raise ValueError(f'Detection cache does not correspond to the panel: {path}')
    if detector is not None and header['detector'] != detector:
        raise ValueError(f'Detection cache model changed: {path}')
    receipt = path.with_suffix('.json')
    if receipt.is_file():
        expected = json.loads(receipt.read_text(encoding='utf-8'))
        if expected != {**header, 'cache_sha256': sha256(path)}:
            raise ValueError(f'Detection cache receipt does not match its shard: {path}')
    elif require_receipt:
        raise FileNotFoundError(f'Missing committed detection cache receipt: {receipt}')
    for row, item in zip(rows, stream, strict=True):
        if tuple(item['key']) != row_key(row):
            raise ValueError(f'Detection cache row order changed: {path}')
        values = {key: np.asarray(item['regions'][key], dtype=np.float32).reshape(-1, 5)
                  for key in REGION_NAMES}
        if any(not np.isfinite(value).all() for value in values.values()):
            raise ValueError(f'Detection cache contains nonfinite values: {path}')
        result.append(values)
    return result, header['detector']


def detector_identity(model):
    core = ('config.json', 'model.safetensors')
    processor = ('preprocessor_config.json', 'processor_config.json', 'special_tokens_map.json',
                 'tokenizer_config.json', 'tokenizer.json', 'vocab.txt', 'added_tokens.json')
    assets = {name: sha256(model / name) for name in core}
    assets.update({name: sha256(model / name) for name in processor if (model / name).is_file()})
    return {'assets': assets, 'zoom_parts': True, 'threshold': 0.25}


def cache(args):
    import torch
    from torch.utils.data import DataLoader

    sys.path[:0] = [str(ROOT / '04_grounded_rerank'), str(ROOT / 'common')]
    from grounding import Grounder
    from run import RankingDataset

    data, digest = manifest(args.panels)
    torch.set_num_threads(2)
    model = args.model.resolve(strict=True)
    detector_id = detector_identity(model)
    detector = None
    started = time.monotonic()
    for group in args.groups:
        shards = data['groups'][group]['shards']
        for index in range(args.worker, len(shards), args.workers):
            rows = load_panel_shard(args.panels, data, group, index)
            destination = shard_path(args.output, group, index)
            receipt = destination.with_suffix('.json')
            header = {'panel_sha256': digest, 'rows': len(rows), 'detector': detector_id}
            if destination.is_file():
                read_regions(destination, rows, digest, detector_id, require_receipt=False)
                if not receipt.is_file():
                    write_json(receipt, {**header, 'cache_sha256': sha256(destination)})
                continue
            if detector is None:
                detector = Grounder(model, args.device, batch_size=args.batch, zoom_parts=True)
            loader = DataLoader(RankingDataset(rows, args.images, {}), batch_size=args.batch,
                                num_workers=args.loader_workers, pin_memory=False)

            def detected_rows():
                yield header
                for _, images, _, positions in loader:
                    batch = [rows[position] for position in positions.tolist()]
                    regions = detector.predict(images, [row['prompt'] for row in batch])
                    for row, detected in zip(batch, regions, strict=True):
                        yield {'key': row_key(row), 'regions': {key: detected[key].tolist() for key in REGION_NAMES}}

            write_lines(destination, detected_rows())
            write_json(receipt, {**header, 'cache_sha256': sha256(destination)})
            print('CACHED', group, index, len(rows), f'{time.monotonic() - started:.0f}s', flush=True)


def verify_cache(args):
    data, digest = manifest(args.panels)
    detector = detector_identity(args.model) if args.model else None
    status = {}
    for group in args.groups:
        ready = rows_verified = 0
        shards = data['groups'][group]['shards']
        for index in range(len(shards)):
            path = shard_path(args.cache, group, index)
            if not path.is_file() or not path.with_suffix('.json').is_file():
                continue
            rows = load_panel_shard(args.panels, data, group, index)
            _, detector = read_regions(path, rows, digest, detector)
            ready += 1
            rows_verified += len(rows)
        status[group] = {'shards_verified': ready, 'shards_total': len(shards), 'rows_verified': rows_verified}
    print('CACHE_VERIFIED', json.dumps(status), flush=True)
    if not args.allow_missing and any(row['shards_verified'] != row['shards_total'] for row in status.values()):
        raise RuntimeError('Some detection cache shards have no complete receipt')


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Expected a positive integer')
    return number


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    plan = subs.add_parser('prepare')
    for name in ('splits', 'output'):
        plan.add_argument(f'--{name}', type=Path, required=True)
    plan.add_argument('--shard-size', type=positive, default=512)
    plan.add_argument('--train-count', type=positive, default=8192)
    plan.add_argument('--all-train', action='store_true', help='Use all eligible training rows')
    plan.add_argument('--val-count', type=positive, default=1024)
    detector = subs.add_parser('cache')
    for name in ('panels', 'images', 'model', 'output'):
        detector.add_argument(f'--{name}', type=Path, required=True)
    detector.add_argument('--groups', choices=GROUPS, nargs='+', default=GROUPS)
    detector.add_argument('--worker', type=int, default=0)
    detector.add_argument('--workers', type=positive, default=1)
    detector.add_argument('--batch', type=positive, default=4)
    detector.add_argument('--loader-workers', type=int, default=1)
    detector.add_argument('--device', default='cuda:0')
    verifier = subs.add_parser('verify-cache')
    verifier.add_argument('--panels', type=Path, required=True)
    verifier.add_argument('--cache', type=Path, required=True)
    verifier.add_argument('--model', type=Path)
    verifier.add_argument('--groups', nargs='+', choices=GROUPS, default=GROUPS)
    verifier.add_argument('--allow-missing', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'verify-cache':
        verify_cache(args)
    elif not 0 <= args.worker < args.workers or args.loader_workers < 0:
        parser.error('Invalid worker index or loader count')
    else:
        cache(args)


if __name__ == '__main__':
    main()
