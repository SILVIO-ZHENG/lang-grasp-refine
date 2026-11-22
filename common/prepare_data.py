"""Pair Grasp-Anything RGB with Grasp-Anything++ instructions and rectangles.

The official ++ IDs use `<scene SHA256>_<object index>_<part index>`. The RGB
images come from the original dataset. See the official ++ dataset README and
LGD's `utils/dataset_processing/grasp.py` for the source formats.

将 Grasp-Anything 的 RGB 图像与 Grasp-Anything++ 的抓取指令和矩形标注配对。

官方 ++ ID 由 `<场景 SHA256>_<物体索引>_<部位索引>` 组成。RGB 图像来自原版
数据集。源数据格式见官方 ++ 数据集说明和 LGD 的 `utils/dataset_processing/grasp.py`。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import pickle
import re
import shutil
import struct
import tempfile
import zipfile
import zlib


PP_REVISION = "9b30c841d4f04e4a62e9f7abb3b503337a6200a7"
ID_PATTERN = re.compile(r"([0-9a-f]{64})_(\d+)_(\d+)")
MAX_INSTRUCTION_BYTES = 64 * 1024
MAX_TENSOR_BYTES = 2 * 1024 * 1024
ZIP_HEADER = struct.Struct("<4s5H3I2H")
MIB = 1024**2


class PrimitivePickle(pickle.Unpickler):
    """Decode primitive annotation data with class and external lookups disabled.

    读取基础类型标注数据，禁用类解析和外部持久化引用。
    """

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Global {module}.{name} is not permitted in an instruction")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("Persistent pickle references are not permitted")


def clean_instruction(payload: bytes) -> str:
    """Decode an official instruction string and normalize its whitespace.

    解码官方指令字符串，并统一空白字符。
    """
    obj = PrimitivePickle(io.BytesIO(payload)).load()
    if not isinstance(obj, str):
        raise ValueError(f"Expected official Grasp-Anything++ instruction string, got {type(obj).__name__}")
    text = " ".join(obj.split())
    if not text or len(text) > 1000:
        raise ValueError("Instruction is empty or exceeds the length limit")
    return text


def annotation_key(name: str, suffix: str) -> str | None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.suffix != suffix:
        return None
    return path.stem if ID_PATTERN.fullmatch(path.stem) else None


class DirectorySource:
    def __init__(self, root: Path, suffix: str, max_size: int):
        self.root, self.max_size = root, max_size
        self.records = {}
        # Official archive components are flat; allow an enclosing directory.
        # 官方归档组件是平铺的，同时兼容外面多包一层目录的情况。
        for directory, children, files in os.walk(root, followlinks=False):
            children[:] = [d for d in children if d not in {"__MACOSX", ".git"}]
            for name in files:
                key = annotation_key(name, suffix)
                if key is None:
                    continue
                path = Path(directory) / name
                if path.is_symlink() or key in self.records:
                    raise ValueError(f"Symlink or duplicate annotation: {path}")
                self.records[key] = path

    def read(self, key: str) -> bytes:
        with self.records[key].open("rb") as stream:
            value = stream.read(self.max_size + 1)
        if len(value) > self.max_size:
            raise ValueError(f"Oversized annotation {key}")
        return value


class ZipSource:
    def __init__(self, file: Path, suffix: str, max_size: int, stack: ExitStack):
        self.file = stack.enter_context(zipfile.ZipFile(file))
        self.max_size = max_size
        self.records = {}
        for info in self.file.infolist():
            key = annotation_key(info.filename, suffix)
            if key is None:
                continue
            if key in self.records:
                raise ValueError(f"Duplicate annotation {key} in {file}")
            self.records[key] = info

    def read(self, key: str) -> bytes:
        info = self.records[key]
        if info.file_size > self.max_size:
            raise ValueError(f"Oversized annotation {key}")
        return self.file.read(info)


class PrefixZipSource:
    """Index complete local ZIP members in a byte-contiguous official prefix.

    This permits early, explicitly labeled pilot splits during the full archive
    download. The file may continue to grow; indexing uses one length snapshot
    and reading verifies the member's advertised length and CRC32. This does
    not substitute for verifying the completed archive against its official hash.

    为官方 ZIP 连续前缀中已完整下载的成员建立索引。

    这样可在完整下载前生成明确标注的试验划分。索引使用文件长度快照，读取时校验
    成员的声明长度和 CRC32。完整归档仍须按官方哈希单独核验。
    """

    def __init__(self, file: Path, suffix: str, max_size: int, stack: ExitStack):
        self.file = stack.enter_context(file.open("rb"))
        self.max_size = max_size
        self.scanned_bytes = 0
        self.snapshot_bytes = os.fstat(self.file.fileno()).st_size
        self.records = {}
        while self.file.tell() + ZIP_HEADER.size <= self.snapshot_bytes:
            header = self.file.read(ZIP_HEADER.size)
            magic, _, flags, method, _, _, crc, compressed, size, n, extra = ZIP_HEADER.unpack(header)
            if magic in (b"PK\x01\x02", b"PK\x05\x06", b"PK\x06\x06"):
                break
            if magic != b"PK\x03\x04" or flags & 9:
                raise ValueError(f"Invalid, encrypted, or descriptor-based local ZIP header in {file}")
            if self.file.tell() + n + extra > self.snapshot_bytes:
                break
            name = self.file.read(n).decode("utf-8" if flags & 0x800 else "cp437")
            extras = self.file.read(extra)
            if compressed == 0xFFFFFFFF or size == 0xFFFFFFFF:
                compressed, size = self._zip64_sizes(extras, compressed, size)
            offset = self.file.tell()
            if offset + compressed > self.snapshot_bytes:
                break
            key = annotation_key(name, suffix)
            if key is not None:
                if key in self.records:
                    raise ValueError(f"Duplicate annotation {key} in {file}")
                self.records[key] = offset, compressed, size, method, crc
            self.file.seek(compressed, os.SEEK_CUR)
            self.scanned_bytes = self.file.tell()

    @staticmethod
    def _zip64_sizes(extras: bytes, compressed: int, size: int) -> tuple[int, int]:
        start = 0
        while start + 4 <= len(extras):
            kind, length = struct.unpack_from("<HH", extras, start)
            start += 4
            value = extras[start:start + length]
            if kind == 1:
                if size == 0xFFFFFFFF:
                    size, = struct.unpack_from("<Q", value)
                    value = value[8:]
                if compressed == 0xFFFFFFFF:
                    compressed, = struct.unpack_from("<Q", value)
                return compressed, size
            start += length
        raise ValueError("Missing required ZIP64 size data")

    def read(self, key: str) -> bytes:
        offset, compressed, size, method, crc = self.records[key]
        if size > self.max_size or compressed > self.max_size + 1024:
            raise ValueError(f"Oversized annotation {key}")
        self.file.seek(offset)
        body = self.file.read(compressed)
        if len(body) != compressed:
            raise ValueError(f"ZIP prefix changed or was truncated for {key}")
        if method == zipfile.ZIP_STORED:
            value = body
        elif method == zipfile.ZIP_DEFLATED:
            decoder = zlib.decompressobj(-15)
            value = decoder.decompress(body, self.max_size + 1)
            if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
                raise ValueError(f"Invalid raw DEFLATE member for {key}")
        else:
            raise ValueError(f"Unsupported compression method {method} for {key}")
        if len(value) != size or zlib.crc32(value) != crc:
            raise ValueError(f"ZIP member checksum or length mismatch for {key}")
        return value


def open_archive_source(file: Path, suffix: str, limit: int, stack: ExitStack, accept_prefix: bool):
    if zipfile.is_zipfile(file):
        return ZipSource(file, suffix, limit, stack)
    if accept_prefix:
        return PrefixZipSource(file, suffix, limit, stack)
    raise ValueError(f"Expected a complete ZIP: {file}; use --accept-prefix explicitly for a pilot")


def load_grasps(payload: bytes, side: int, limit: int) -> list[list[float]]:
    """Convert official N-by-6 tensors into validated five-value rectangles.

    Keep center and size in image pixels and wrap angles in radians modulo pi.
    A zero limit keeps every valid rectangle.

    将官方 N×6 张量转换成经过检查的五参数抓取矩形。

    中心和尺寸保留像素单位，角度转换为弧度并按半周取模。`limit=0` 时保留所有
    有效矩形。
    """
    import torch

    # Modern Torch archives plus weights_only avoids arbitrary object loading.
    # 只读取现代 Torch 归档，并启用 weights_only，避免加载任意对象。
    if not payload.startswith(b"PK\x03\x04"):
        raise ValueError("Expected a modern ZIP-based tensor archive")
    labels = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2 or labels.shape[1] != 6 or len(labels) > 10000:
        raise ValueError("Expected N×6 grasp tensor [score,x,y,w,h,theta_deg]")
    result = []
    for _, x, y, w, h, theta in labels.tolist():
        if not all(math.isfinite(v) for v in (x, y, w, h, theta)):
            continue
        if not (0 <= x < side and 0 <= y < side and 0 < w <= side and 0 < h <= side):
            continue
        angle = (math.radians(theta) + math.pi / 2) % math.pi - math.pi / 2
        result.append([float(x), float(y), float(w), float(h), angle])
        if limit and len(result) == limit:
            break
    return result


def stable_hash(scene: str, seed: int, salt: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"{salt}:{seed}:{scene}".encode(), digest_size=8).digest(), "big")


def split_name(scene: str, seed: int) -> str:
    """Assign all rows of one scene to the same 80/10/10 hash bucket.

    按场景哈希以 80/10/10 的比例分桶。同一场景的全部记录进入同一划分。
    """
    block = stable_hash(scene, seed, "split") % 100
    return "train" if block < 80 else "val" if block < 90 else "test"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True, help="Grasp-Anything's image folder, not its parent")
    p.add_argument("--pp", type=Path, help="Extracted Grasp-Anything++ dataset directory")
    p.add_argument("--instructions-zip", type=Path, help="Complete official instruction ZIP or pilot prefix")
    p.add_argument("--positive-zip", type=Path, help="Complete official positive ZIP or pilot prefix")
    p.add_argument("--accept-prefix", action="store_true", help="Read complete CRC-checked members from an incomplete ZIP prefix")
    p.add_argument("--out", type=Path, required=True, help="Output split directory, which must not exist yet")
    p.add_argument("--max-scenes", type=int, default=2000, help="0 includes all joined scenes")
    p.add_argument("--max-grasps", type=int, default=0, help="0 keeps all valid ground-truth rectangles")
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--image-size", type=int, default=416)
    a = p.parse_args(argv)
    if bool(a.pp) == bool(a.instructions_zip or a.positive_zip) or (not a.pp and not (a.instructions_zip and a.positive_zip)):
        p.error("pass --pp, or both --instructions-zip and --positive-zip")
    if a.max_scenes < 0 or a.max_grasps < 0 or a.image_size <= 0:
        p.error("limits must be nonnegative, and --image-size positive")
    return a


def describe_source(source, path):
    summary = {"kind": type(source).__name__, "path": str(path.resolve()), "members": len(source.records)}
    if isinstance(source, PrefixZipSource):
        summary.update(snapshot_bytes=source.snapshot_bytes, scanned_bytes=source.scanned_bytes)
    return summary


def load_pilot_receipt(instructions: Path, positives: Path):
    """If present beside both archives, verify the pilot receipt and local hashes.

    如果两个归档的同一目录中有试验采样记录，核验其来源及两个本地归档的哈希。
    """
    if instructions.parent.resolve() != positives.parent.resolve():
        return None
    receipt_path = instructions.parent / "range_pilot_receipt.json"
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("kind") != "official_pinned_zip_range_pilot" or receipt.get("revision") != PP_REVISION:
        raise ValueError("Unexpected annotation sample provenance")
    for path in (instructions, positives):
        expected = receipt.get("repacked_archives", {}).get(path.name, {})
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(MIB):
                digest.update(block)
        if (path.stat().st_size, digest.hexdigest()) != (expected.get("bytes"), expected.get("sha256")):
            raise ValueError(f"Pilot receipt checksum mismatch for {path.name}")
    return receipt


def prepare(a):
    """Write scene-disjoint splits from extracted annotations or paired ZIP files.

    Keep targets that have both an instruction and valid rectangles, and require
    the original RGB image for each retained scene. Output JSONL and a manifest.

    从已解压的标注或配对 ZIP 文件生成按场景隔离的训练、验证和测试划分。

    只保留同时拥有指令与有效矩形的目标，每个保留场景还需有原版 RGB 图像。
    输出 JSONL 与数据清单。
    """
    if not a.images.is_dir() or a.out.exists():
        raise ValueError("The image directory must exist and output must be a new directory")
    a.out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    scratch = Path(tempfile.mkdtemp(prefix=f".{a.out.name}-", dir=a.out.parent))
    tally = Counter()
    scenes_per_split = Counter()
    rows_per_split = Counter()
    rows_multitarget = Counter()
    pilot_receipt = None
    try:
        with ExitStack() as stack:
            if a.pp:
                instructions_path = a.pp / "grasp_instructions"
                positive_path = a.pp / "grasp_label_positive"
                if not positive_path.is_dir():
                    positive_path = a.pp / "positive_grasp"
                if not (instructions_path.is_dir() and positive_path.is_dir()):
                    raise ValueError("Expected extracted ++ instruction and positive grasp folders")
                instructions = DirectorySource(instructions_path, ".pkl", MAX_INSTRUCTION_BYTES)
                positives = DirectorySource(positive_path, ".pt", MAX_TENSOR_BYTES)
            else:
                instructions_path, positive_path = a.instructions_zip, a.positive_zip
                pilot_receipt = load_pilot_receipt(instructions_path, positive_path)
                instructions = open_archive_source(instructions_path, ".pkl", MAX_INSTRUCTION_BYTES, stack, a.accept_prefix)
                positives = open_archive_source(positive_path, ".pt", MAX_TENSOR_BYTES, stack, a.accept_prefix)
            by_scene = defaultdict(list)
            for target in instructions.records.keys() & positives.records.keys():
                scene, object_id, part_id = ID_PATTERN.fullmatch(target).groups()
                by_scene[scene].append((int(object_id), int(part_id), target))
            tally["matched_targets"] = sum(map(len, by_scene.values()))
            tally["matched_scenes"] = len(by_scene)
            sinks = {k: stack.enter_context((scratch / f"{k}.jsonl").open("w", encoding="utf-8"))
                     for k in ("train", "val", "test")}
            # Separate the sampling hash from the split hash so limiting scenes
            # does not select by the order used for train/validation assignment.
            # 分开使用抽样哈希和划分哈希，限制场景数量时不依赖训练与验证的分桶顺序。
            seq = sorted(by_scene, key=lambda s: stable_hash(s, a.seed, "selection"))
            for scene in seq:
                if a.max_scenes and sum(scenes_per_split.values()) == a.max_scenes:
                    break
                image = f"{scene}.jpg"
                if not (a.images / image).is_file():
                    tally["missing_rgb_scenes"] += 1
                    continue
                examples = []
                for object_id, part_id, target in sorted(by_scene[scene]):
                    try:
                        prompt = clean_instruction(instructions.read(target))
                        grasps = load_grasps(positives.read(target), a.image_size, a.max_grasps)
                    except (ValueError, RuntimeError, EOFError, pickle.UnpicklingError, zipfile.BadZipFile, zlib.error):
                        tally["unreadable_targets"] += 1
                        if tally["unreadable_targets"] > 50 and not sum(rows_per_split.values()):
                            raise
                        continue
                    if not grasps:
                        tally["no_valid_grasps"] += 1
                        continue
                    examples.append(dict(scene_id=scene, image=image, prompt=prompt, grasps=grasps,
                                         target_id=target, object_id=object_id, part_id=part_id,
                                         width=a.image_size, height=a.image_size))
                if not examples:
                    continue
                split = split_name(scene, a.seed)
                scenes_per_split[split] += 1
                multi = len({e["object_id"] for e in examples}) > 1
                rows_multitarget[split] += len(examples) if multi else 0
                for r in examples:
                    r["same_scene_other_targets"] = [
                        {"target_id": x["target_id"], "prompt": x["prompt"]}
                        for x in examples if x["object_id"] != r["object_id"] and x["prompt"] != r["prompt"]
                    ][:3]
                    sinks[split].write(json.dumps(r, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n")
                    rows_per_split[split] += 1
                if sum(scenes_per_split.values()) % 1000 == 0:
                    print(f"Matched {sum(scenes_per_split.values())} usable scenes", flush=True)
            if any(rows_per_split[k] == 0 for k in sinks):
                raise ValueError(f"Insufficient joined scenes for disjoint train/val/test: {dict(rows_per_split)}; index more of the archives")
            manifest = {
                "dataset": "airvlab/Grasp-Anything-pp", "revision": PP_REVISION,
                "image_root": str(a.images.resolve()), "seed": a.seed, "scene_counts": dict(scenes_per_split),
                "row_counts": dict(rows_per_split), "rows_from_multi_object_scenes": dict(rows_multitarget),
                "screening": dict(tally), "annotations": {
                    "instructions": describe_source(instructions, instructions_path),
                    "positive_grasps": describe_source(positives, positive_path),
                },
                "angle_convention": "radians, width-axis direction in image coordinates (+x right, +y down), modulo pi",
                "split": "80/10/10 by seeded scene hash; independent seeded selection of scenes",
            }
            if pilot_receipt is not None:
                manifest["range_pilot_receipt"] = pilot_receipt
        (scratch / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        scratch.rename(a.out)
        print(json.dumps(manifest, ensure_ascii=False), flush=True)
    finally:
        if scratch.is_dir() and not scratch.is_symlink() and scratch.resolve().parent == a.out.parent.resolve():
            shutil.rmtree(scratch)


if __name__ == "__main__":
    os.umask(0o077)
    prepare(parse_args())
