"""Train a grasp candidate model on a scene-disjoint indexed JSONL dataset."""

import argparse
from array import array
from contextlib import contextmanager, nullcontext
from collections import Counter
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import random
import re
import time

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, default_collate

from run_experiment import MAX_TARGETS, SIZE, WORD, correct, load_method, normalized, safe_json, tokenize


SPLITS = ("val", "test_seen", "test_unseen", "train")
PREDICTION_KEYS = ("scene_id", "prompt", "target_id", "image")
DIRECTION = re.compile(r"\b(?:left|right|clockwise|counterclockwise)\b", re.I)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("method", "images", "splits", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=121)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--shuffle-block", type=int, default=8192)
    parser.add_argument("--save-every-steps", type=int, default=2500)
    parser.add_argument("--log-every-steps", type=int, default=250)
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience in epochs; 0 disables")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-test", action="store_true", help="Skip final evaluation of the best checkpoint")
    args = parser.parse_args()
    positive = (args.epochs, args.batch, args.lr, args.shuffle_block, args.save_every_steps, args.log_every_steps)
    if not all(value > 0 and math.isfinite(value) for value in positive) or min(args.workers, args.patience) < 0:
        parser.error("Expected positive training parameters and nonnegative workers and patience")
    return args


@contextmanager
def run_lock(path):
    """Prevent concurrent resume processes from overwriting each other's state."""
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def source_info(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_indexes(args):
    """Store one file offset per row and compute the vocabulary from training only."""
    root = args.output / "indexes"
    root.mkdir(exist_ok=True)
    manifest = root / "manifest.json"
    vocabulary_path = args.output / "vocabulary.json"
    sources = {name: args.splits / f"{name}.jsonl" for name in SPLITS}
    signature = {name: source_info(path) for name, path in sources.items()}
    if manifest.is_file():
        index = json.loads(manifest.read_text(encoding="utf-8"))
        for name in SPLITS:
            info = index[name]
            if any(info[key] != value for key, value in signature[name].items()):
                raise ValueError(f"Indexed source changed since the initial run: {sources[name]}")
            if (root / f"{name}.u64").stat().st_size != 8 * (info["rows"] + 1):
                raise ValueError(f"Unexpected index length for {name}")
        return index, json.loads(vocabulary_path.read_text(encoding="utf-8"))

    holdout_scenes = set()
    index = {}
    words = Counter()
    for name, source in sources.items():
        offset = count = 0
        digest = hashlib.sha256()
        offsets = array("Q")
        subset_scenes = set()
        target = root / f"{name}.u64"
        temporary = target.with_suffix(".tmp")
        with source.open("rb", buffering=1024 * 1024) as data, temporary.open("wb") as out:
            for line in data:
                digest.update(line)
                start = offset
                offset += len(line)
                if line.isspace():
                    continue
                row = json.loads(line)
                scene = row["scene_id"]
                if scene in holdout_scenes:
                    raise ValueError(f"Shared scene detected in {name}: {scene}")
                if name == "train":
                    words.update(word.lower() for word in WORD.findall(row["prompt"]))
                else:
                    subset_scenes.add(scene)
                offsets.append(start)
                count += 1
                if len(offsets) == 65536:
                    offsets.tofile(out)
                    offsets = array("Q")
            offsets.append(offset)
            offsets.tofile(out)
        if not count or source_info(source) != signature[name]:
            raise ValueError(f"Empty or changing JSONL source: {source}")
        temporary.replace(target)
        holdout_scenes.update(subset_scenes)
        index[name] = {**signature[name], "rows": count, "sha256": digest.hexdigest()}
        print(f"INDEX {name} {count:,} rows", flush=True)
    vocab = {word: i for i, (word, _) in enumerate(sorted(words.items(), key=lambda item: (-item[1], item[0]))[:8189], 3)}
    safe_json(vocabulary_path, vocab)
    safe_json(manifest, index)
    return index, vocab


class IndexedDataset(Dataset):
    def __init__(self, source, offsets, rows, images, vocabulary, *, train=False, seed=121):
        self.source = source
        self.offsets = offsets
        self.rows = rows
        self.images = images
        self.vocabulary = vocabulary
        self.train = train
        self.seed = seed
        self._data = self._starts = None

    def __len__(self):
        return self.rows

    def __getstate__(self):
        return dict(vars(self), _data=None, _starts=None)

    def __getitem__(self, index):
        epoch, index = index if self.train else (0, index)
        if self._data is None:
            with self.source.open("rb") as source:
                self._data = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
            self._starts = np.memmap(self.offsets, dtype=np.uint64, mode="r")
        row = json.loads(self._data[int(self._starts[index]) : int(self._starts[index + 1])])
        with Image.open(self.images / row["image"]) as source:
            rgb = source.convert("RGB")
        width, height = rgb.size
        boxes = normalized(row["grasps"], width, height)
        if width != height:
            side = max(width, height)
            square = Image.new("RGB", (side, side))
            square.paste(rgb, ((side - width) // 2, (side - height) // 2))
            rgb = square
        rgb = rgb.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        if self.train and not DIRECTION.search(row["prompt"]):
            key = f"{self.seed}:{epoch}:{index}".encode("ascii")
            if hashlib.blake2b(key, digest_size=1).digest()[0] < 128:
                rgb = ImageOps.mirror(rgb)
                boxes[:, 0] = 1.0 - boxes[:, 0]
                boxes[:, 4] = -boxes[:, 4]
        image = torch.from_numpy(np.asarray(rgb).transpose(2, 0, 1).copy()).float() / 255.0
        prompt = tokenize(row["prompt"], self.vocabulary)
        targets = torch.zeros(MAX_TARGETS, 5)
        count = min(MAX_TARGETS, len(boxes))
        targets[:count] = torch.from_numpy(boxes[:count])
        valid = torch.arange(MAX_TARGETS) < count
        values = image, prompt, targets, valid
        if self.train:
            return values
        return values, boxes, {key: row[key] for key in PREDICTION_KEYS}


class LocalitySampler(Sampler):
    """Shuffle both row blocks and rows within each block with bounded memory."""

    def __init__(self, length, block_size, seed):
        self.length, self.block_size, self.seed = length, block_size, seed
        self.epoch, self.skip = 1, 0

    def __len__(self):
        return max(0, self.length - self.skip)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        blocks = math.ceil(self.length / self.block_size)
        remaining_skip = self.skip
        for block in torch.randperm(blocks, generator=generator).tolist():
            first = block * self.block_size
            size = min(self.block_size, self.length - first)
            indices = torch.randperm(size, generator=generator).tolist()
            for offset in indices[remaining_skip:]:
                yield self.epoch, first + offset
            remaining_skip = max(0, remaining_skip - size)


def collate_eval(batch):
    values, truth, metadata = zip(*batch, strict=True)
    return default_collate(values), truth, metadata


@torch.inference_mode()
def evaluate(method, net, loader, device, predictions=None):
    net.eval()
    count = top1 = top5 = 0
    total_loss = infer_seconds = 0.0
    use_cuda = device.type == "cuda"
    temp = predictions.with_suffix(".tmp") if predictions is not None else None
    with temp.open("w", encoding="utf-8") if temp else nullcontext(None) as stream:
        for values, truth, metadata in loader:
            images, tokens, targets, valid = (tensor.to(device, non_blocking=use_cuda) for tensor in values)
            if use_cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            output = net(images, tokens)
            poses = method.decode(output, topk=5)
            if use_cuda:
                torch.cuda.synchronize(device)
            infer_seconds += time.perf_counter() - start
            total_loss += float(method.loss(output, targets, valid)) * len(images)
            count += len(images)
            for candidate, gt, row in zip(poses.float().cpu().numpy(), truth, metadata, strict=True):
                hit = correct(candidate[0], gt)
                top1 += hit
                top5 += hit or any(correct(box, gt) for box in candidate[1:])
                if stream is not None:
                    result = {**row, "success": bool(hit), "gt": gt[0].tolist(), "prediction": candidate[0].tolist()}
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    if temp:
        temp.replace(predictions)
    timing_name = "gpu_batch_inference_ms_per_image" if use_cuda else "cpu_batch_inference_ms_per_image"
    return {"n": count, "top1": float(top1 / count), "top5": float(top5 / count), "loss": total_loss / count,
            timing_name: 1000.0 * infer_seconds / count}


def save_checkpoint(path, payload):
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def rng_state(device):
    name, keys, position, has_gaussian, gaussian = np.random.get_state()
    return {"python": random.getstate(), "numpy": (name, keys.tolist(), position, has_gaussian, gaussian),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    name, keys, position, has_gaussian, gaussian = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gaussian, gaussian))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def fit(args, method, model, loaders, sampler, device, training_signature):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.08 * args.lr)
    best_path = args.output / "best.pt"
    last_path = args.output / "last.pt"
    history_path = args.output / "history.json"
    state = {"epoch": 1, "batch": 0, "steps": 0, "loss_sum": 0.0, "examples": 0, "elapsed_s": 0.0,
             "best": (-1.0, -1.0, -math.inf), "stale": 0, "stopped": False, "history": []}
    if args.resume and last_path.is_file():
        resumed = torch.load(last_path, map_location="cpu", weights_only=True)
        if resumed["signature"] != training_signature:
            raise ValueError("Resume configuration differs from the saved training configuration")
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        restore_rng(resumed["rng"], device)
        state = resumed["state"]
        safe_json(history_path, state["history"])
    elapsed = state["elapsed_s"]
    start = time.perf_counter()

    def checkpoint():
        state["elapsed_s"] = elapsed + time.perf_counter() - start
        save_checkpoint(last_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                    "scheduler": scheduler.state_dict(), "rng": rng_state(device),
                                    "signature": training_signature, "state": state})

    if not last_path.is_file():
        checkpoint()
    for epoch in range(state["epoch"], args.epochs + 1):
        if state["stopped"]:
            break
        model.train()
        sampler.epoch = epoch
        sampler.skip = state["examples"]
        for values in loaders["train"]:
            image, tokens, targets, valid = (tensor.to(device, non_blocking=device.type == "cuda") for tensor in values)
            optimizer.zero_grad(set_to_none=True)
            loss = method.loss(model(image, tokens), targets, valid)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss at epoch {epoch}, batch {state['batch'] + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0, error_if_nonfinite=True)
            optimizer.step()
            state["batch"] += 1
            state["steps"] += 1
            state["examples"] += len(image)
            state["loss_sum"] += float(loss.detach()) * len(image)
            if state["steps"] % args.log_every_steps == 0:
                print(f"TRAIN epoch={epoch} step={state['steps']} rows={state['examples']}/{sampler.length} "
                      f"loss={state['loss_sum'] / state['examples']:.5f}", flush=True)
            if state["steps"] % args.save_every_steps == 0:
                checkpoint()

        validation = evaluate(method, model, loaders["val"], device)
        record = {"epoch": epoch, "train_loss": state["loss_sum"] / state["examples"], "validation": validation,
                  "elapsed_s": elapsed + time.perf_counter() - start, "optimizer_steps": state["steps"]}
        state["history"].append(record)
        score = validation["top1"], validation["top5"], -validation["loss"]
        if score > state["best"]:
            state["best"], state["stale"] = score, 0
            save_checkpoint(best_path, {"model": model.state_dict(), "epoch": epoch, "validation": validation})
        else:
            state["stale"] += 1
        scheduler.step()
        state.update(epoch=epoch + 1, batch=0, examples=0, loss_sum=0.0,
                     stopped=bool(args.patience and state["stale"] >= args.patience))
        checkpoint()
        safe_json(history_path, state["history"])
        print(json.dumps(record), flush=True)
    return state


def main():
    args = arguments()
    if args.resume and not args.output.is_dir():
        raise FileNotFoundError(f"Cannot resume a missing run: {args.output}")
    args.output.mkdir(parents=True, mode=0o700, exist_ok=args.resume)
    with run_lock(args.output / ".run.lock"):
        train(args)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    cv2.setNumThreads(0)
    if args.device == "cuda" and torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one CUDA GPU for each model run")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    index, vocabulary = build_indexes(args)
    datasets = {name: IndexedDataset(args.splits / f"{name}.jsonl", args.output / "indexes" / f"{name}.u64",
                                     info["rows"], args.images, vocabulary, train=name == "train", seed=args.seed)
                for name, info in index.items()}
    sampler = LocalitySampler(len(datasets["train"]), args.shuffle_block, args.seed)
    loaders = {}
    for name, data in datasets.items():
        training = name == "train"
        loaders[name] = DataLoader(data, batch_size=args.batch, sampler=sampler if training else None,
                                   collate_fn=None if training else collate_eval, num_workers=args.workers,
                                   pin_memory=device.type == "cuda", persistent_workers=training and args.workers > 0,
                                   generator=torch.Generator().manual_seed(args.seed))
    signature = {key: getattr(args, key) for key in ("epochs", "batch", "seed", "lr", "shuffle_block", "patience", "device")}
    signature.update(method=str(args.method.resolve()), method_sha256=hashlib.sha256(args.method.read_bytes()).hexdigest(),
                     images=str(args.images.resolve()), data={name: info["sha256"] for name, info in index.items()})
    sizes = {name: len(data) for name, data in datasets.items()}
    sizes["test"] = sizes["test_seen"] + sizes["test_unseen"]
    config_path = args.output / "config.json"
    if not config_path.is_file():
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(torch_version=torch.__version__, gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                      image_size=SIZE, max_targets=MAX_TARGETS, data_sizes=sizes, training_signature=signature,
                      logical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        safe_json(config_path, config)
    elif json.loads(config_path.read_text(encoding="utf-8"))["training_signature"] != signature:
        raise ValueError("Resume configuration differs from the initial run")
    method = load_method(args.method)
    model = method.build(len(vocabulary) + 3).to(device)
    state = fit(args, method, model, loaders, sampler, device, signature)
    best = torch.load(args.output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    test = {}
    if not args.skip_test:
        for group in ("seen", "unseen"):
            test[group] = evaluate(method, model, loaders[f"test_{group}"], device,
                                   predictions=args.output / f"{group}_predictions.jsonl")
        result = {"best_epoch": best["epoch"], "protocol": "fixed indexed scene splits, rotation IoU/angle evaluation", **test}
        for key in ("top1", "top5"):
            seen, unseen = test["seen"][key], test["unseen"][key]
            result[f"h_{key}"] = 2 * seen * unseen / (seen + unseen) if seen + unseen else 0.0
        safe_json(args.output / "seen_unseen_metrics.json", result)
    report = {"best_epoch": best["epoch"], "best_validation": best["validation"], "test": test,
              "epochs_completed": len(state["history"]), "optimizer_steps": state["steps"],
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "training_seconds": state["elapsed_s"]}
    safe_json(args.output / "metrics.json", report)
    print("COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
