"""Join the author-released LGD object lists with pinned official ++ annotations.

Use the published Seen and Unseen object IDs for evaluation. Report coverage
and missing objects when the public RGB and annotation files cannot supply all
of those targets.

把 LGD 作者发布的物体列表与固定版本的官方 Grasp-Anything++ 标注配对。

评测沿用作者发布的 Seen 和 Unseen 物体 ID。公开图像或标注缺失时，在清单中
记录覆盖范围和缺失的物体。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import struct
import tempfile
import zipfile
import zlib

from PIL import Image

from prepare_data import (
    ID_PATTERN, MAX_INSTRUCTION_BYTES, MAX_TENSOR_BYTES, PP_REVISION, ZIP_HEADER,
    PrefixZipSource, PrimitivePickle, clean_instruction, load_grasps,
    load_pilot_receipt, stable_hash,
)
from sample_pp_annotations import ARCHIVES, manifest_for


AUTHOR_COMMIT = "501904dc5c30c8fad58d90f7e4f49b4095f11eb0"
AUTHOR_URL = f"https://github.com/Fsoft-AIC/LGD/tree/{AUTHOR_COMMIT}/split/grasp-anything%2B%2B"
AUTHOR_LISTS = {
    "train_seen": ("train/seen.obj", 14516, "6b6ddee6ad2bc19b8fdfebcc61ab58900fbe384f498dd38d1c4478728e7ac10d"),
    "test_seen": ("test/seen.obj", 573, "37e929120751d99d9abb48a5fdb9d4a3033ab5aae67caec30b8cb49086fc10ee"),
    "test_unseen": ("test/unseen.obj", 230, "94d175613ed506cbe6265e58b31f87658ee86f30a232f640167fa90ecab8db92"),
}
OBJECT_ID = re.compile(r"[0-9a-f]{64}_[0-9]+")
PARTITIONS = ("train", "val", "test_seen", "test_unseen")
EOCD = struct.Struct("<4s4H2IH")
ZIP64_LOC = struct.Struct("<4sIQI")
ZIP64_EOCD = struct.Struct("<4sQ2H2I4Q")
DIRECTORY_HEADER = struct.Struct("<4s6H3I5H2I")


def official_objects(root):
    """Verify the author's object-list hashes, expected counts, and disjoint IDs.

    校验作者发布的物体列表哈希、预期数量，并确认各列表的物体 ID 互不重叠。
    """
    sets = {}
    for split, (rel, count, expected_sha) in AUTHOR_LISTS.items():
        data = (root / rel).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_sha:
            raise ValueError(f"Incorrect author split SHA256 for {rel}")
        ids = PrimitivePickle(io.BytesIO(data)).load()
        if not isinstance(ids, list) or len(ids) != count or any(
            not isinstance(value, str) or OBJECT_ID.fullmatch(value) is None for value in ids
        ) or len(set(ids)) != count:
            raise ValueError(f"Unexpected object IDs in author split {rel}")
        sets[split] = set(ids)
    if (sets["train_seen"] & sets["test_seen"] or sets["train_seen"] & sets["test_unseen"]
            or sets["test_seen"] & sets["test_unseen"]):
        raise ValueError("Unexpected author object-ID overlap")
    return sets


def scene_id(object_id):
    return object_id.rsplit("_", 1)[0]


def scene_partitions(official, seed):
    """Preserve the published test membership and separate the remaining scenes.

    Remove training objects whose scenes appear in either test split. Select
    validation scenes by seeded hash and assign the remaining objects to train.

    保留作者发布的测试物体归属，再对其余场景划分训练和验证。

    训练列表里出现在任一测试场景中的物体会被剔除。按带种子的哈希选出验证场景，
    其余物体划入训练集。
    """
    seen_scenes = {scene_id(key) for key in official["test_seen"]}
    unseen_scenes = {scene_id(key) for key in official["test_unseen"]}
    if seen_scenes & unseen_scenes:
        raise ValueError("Unexpected scene overlap between published seen and unseen evaluation")
    held_out = seen_scenes | unseen_scenes
    removed = {key for key in official["train_seen"] if scene_id(key) in held_out}
    partitions = {"train": set(), "val": set(),
                  "test_seen": official["test_seen"], "test_unseen": official["test_unseen"]}
    for key in official["train_seen"] - removed:
        scene = scene_id(key)
        part = "val" if stable_hash(scene, seed, "lgd_official_train_val") % 10 == 0 else "train"
        partitions[part].add(key)
    return partitions, removed


def central_directory(stream):
    size = stream.seek(0, os.SEEK_END)
    stream.seek(max(0, size - 65535 - EOCD.size))
    base = stream.tell()
    tail = stream.read()
    end = tail.rfind(b"PK\x05\x06")
    if end < 0 or end + EOCD.size > len(tail):
        raise ValueError("Missing the ZIP end record")
    _, disk, directory_disk, count_disk, count, directory_size, offset, comment = EOCD.unpack_from(tail, end)
    if disk or directory_disk or count_disk != count or end + EOCD.size + comment != len(tail):
        raise ValueError("Unexpected split ZIP or malformed ZIP end record")
    end_offset = base + end
    if count == 0xFFFF or directory_size == 0xFFFFFFFF or offset == 0xFFFFFFFF:
        stream.seek(end_offset - ZIP64_LOC.size)
        magic, record_disk, ptr, disk_total = ZIP64_LOC.unpack(stream.read(ZIP64_LOC.size))
        if magic != b"PK\x06\x07" or record_disk or disk_total != 1:
            raise ValueError("Invalid ZIP64 locator")
        stream.seek(ptr)
        magic, _, _, _, disk, directory_disk, count_disk, count, directory_size, offset = ZIP64_EOCD.unpack(
            stream.read(ZIP64_EOCD.size))
        if magic != b"PK\x06\x06" or disk or directory_disk or count_disk != count:
            raise ValueError("Invalid ZIP64 end record")
        end_offset = ptr
    if offset + directory_size > end_offset or count * DIRECTORY_HEADER.size > directory_size:
        raise ValueError("Central directory exceeds ZIP bounds")
    return count, offset, directory_size


class SelectedZipSource(PrefixZipSource):
    """Index ZIP records only for object IDs in the selected author lists.

    只为选定作者列表中的物体建立 ZIP 成员索引，无需在内存中保存全部标注记录。
    """

    def __init__(self, file, suffix, limit, wanted, stack):
        self.file = stack.enter_context(file.open("rb", buffering=1024 * 1024))
        self.max_size = limit
        self.records = {}
        wanted_bytes = {key.encode("ascii") for key in wanted}
        expected_suffix = suffix.encode("ascii")
        count, offset, directory_size = central_directory(self.file)
        self.file.seek(offset)
        hits = {}
        for _ in range(count):
            header = DIRECTORY_HEADER.unpack(self.file.read(DIRECTORY_HEADER.size))
            magic, _, _, flags, method, _, _, crc, compressed, size, n, extra, comment, disk, _, _, local = header
            if magic != b"PK\x01\x02":
                raise ValueError(f"Malformed ZIP central entry in {file}")
            name = self.file.read(n)
            self.file.seek(extra + comment, os.SEEK_CUR)
            basename = name.rsplit(b"/", 1)[-1]
            if not basename.endswith(expected_suffix):
                continue
            stem = basename[:-len(expected_suffix)]
            if stem.rsplit(b"_", 1)[0] not in wanted_bytes:
                continue
            key = stem.decode("ascii")
            if ID_PATTERN.fullmatch(key) is None:
                continue
            if key in hits or flags & 1 or disk or local >= offset or compressed > limit + 1024 or size > limit:
                raise ValueError(f"Invalid selected member {key} in {file}")
            hits[key] = (local, compressed, size, method, crc, name)
        if self.file.tell() != offset + directory_size:
            raise ValueError(f"Unexpected bytes after central entries in {file}")
        for key, record in sorted(hits.items(), key=lambda item: item[1][0]):
            local, compressed, size, method, crc, name = record
            self.file.seek(local)
            magic, _, flags, local_method, _, _, _, _, _, n, extra = ZIP_HEADER.unpack(self.file.read(ZIP_HEADER.size))
            if magic != b"PK\x03\x04" or flags & 1 or local_method != method or self.file.read(n) != name:
                raise ValueError(f"Local ZIP header does not match central index for {key}")
            data_offset = self.file.tell() + extra
            if data_offset + compressed > offset:
                raise ValueError(f"Selected ZIP member crosses directory boundary for {key}")
            self.records[key] = data_offset, compressed, size, method, crc


def annotation_sources(a):
    """Check full archives against pinned hashes or require a partial-pilot receipt.

    完整归档按固定哈希核验。使用部分归档试验时，则要求提供匹配的采样记录。
    """
    paths = {"grasp_instructions.zip": a.instructions_zip, "grasp_label_positive.zip": a.positive_zip}
    if a.allow_partial:
        receipt = load_pilot_receipt(a.instructions_zip, a.positive_zip)
        if receipt is None:
            raise ValueError("Partial archives require the original matching range_pilot_receipt.json")
        return {"kind": "official_range_sample", "receipt": receipt}
    checked = {}
    for name, path in paths.items():
        expected_bytes, expected_sha = ARCHIVES[name]
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"Expected the complete pinned source {name}; wrong archive size")
        observed = manifest_for(path)
        if observed["sha256"] != expected_sha:
            raise ValueError(f"Complete pinned archive SHA256 mismatch for {path}")
        checked[name] = observed
    return {"kind": "verified_full_official_archives", "archives": checked}


def image_is_available(images, name):
    path = images / name
    try:
        with Image.open(path) as image:
            return image.format == "JPEG" and image.size == (416, 416)
    except (OSError, ValueError):
        return False


def official_rows(a, instructions, positives, object_to_split):
    """Join usable RGB, instructions, and positive grasp targets by part ID.

    Return rows and recovered object IDs per partition, plus screening reasons
    for missing or invalid inputs. Check that the resulting scenes are disjoint.

    按部位 ID 配对可用的 RGB 图像、抓取指令与正例抓取标注。

    返回各划分中的记录、成功配对的物体 ID 和输入缺失或无效的原因，并检查各划分
    之间没有场景交叉。
    """
    by_scene = defaultdict(list)
    image_status = {}
    reasons = {part: defaultdict(set) for part in PARTITIONS}
    for key in sorted(instructions.records.keys() | positives.records.keys()):
        object_key = key.rsplit("_", 1)[0]
        split = object_to_split[object_key]
        if key not in instructions.records or key not in positives.records:
            reasons[split]["only_one_annotation_source"].add(object_key)
            continue
        scene, object_id, part_id = ID_PATTERN.fullmatch(key).groups()
        filename = f"{scene}.jpg"
        if scene not in image_status:
            image_status[scene] = image_is_available(a.images, filename)
        if not image_status[scene]:
            reasons[split]["missing_or_invalid_rgb"].add(object_key)
            continue
        try:
            prompt = clean_instruction(instructions.read(key))
            grasps = load_grasps(positives.read(key), 416, 0)
        except (ValueError, RuntimeError, EOFError, pickle.UnpicklingError, zipfile.BadZipFile, zlib.error):
            reasons[split]["unreadable_annotation"].add(object_key)
            continue
        if not grasps:
            reasons[split]["no_valid_grasps"].add(object_key)
            continue
        by_scene[scene].append((split, object_key, {
            "scene_id": scene, "image": filename, "prompt": prompt, "grasps": grasps,
            "target_id": key, "object_id": int(object_id), "part_id": int(part_id), "width": 416, "height": 416,
        }))
    rows = {part: [] for part in PARTITIONS}
    found = {part: set() for part in PARTITIONS}
    for scene in sorted(by_scene):
        group = by_scene[scene]
        if len({split for split, _, _ in group}) != 1:
            raise ValueError(f"Split leakage for {scene}")
        for split, object_key, row in group:
            row["same_scene_other_targets"] = [
                {"target_id": peer["target_id"], "prompt": peer["prompt"]}
                for _, other, peer in group if other != object_key and peer["prompt"] != row["prompt"]
            ][:3]
            rows[split].append(row)
            found[split].add(object_key)
    return rows, found, reasons


def write_rows(path, rows):
    with path.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def build(a):
    """Write four split manifests, a combined test file, and a coverage audit.

    写出四个划分的 JSONL、合并的测试文件，以及标注覆盖率核查结果。
    """
    if not a.images.is_dir() or a.out.exists():
        raise ValueError("The RGB directory must exist and the output directory must be new")
    official = official_objects(a.author_splits)
    partitions, removed = scene_partitions(official, a.seed)
    object_to_split = {key: part for part, ids in partitions.items() for key in ids}
    sources = annotation_sources(a)
    with ExitStack() as stack:
        instructions = SelectedZipSource(a.instructions_zip, ".pkl", MAX_INSTRUCTION_BYTES, object_to_split, stack)
        positives = SelectedZipSource(a.positive_zip, ".pt", MAX_TENSOR_BYTES, object_to_split, stack)
        rows, found, reasons = official_rows(a, instructions, positives, object_to_split)
        available_annotation_objects = {
            key.rsplit("_", 1)[0] for key in instructions.records.keys() | positives.records.keys()
        }
    coverage = {}
    missing = {}
    for part in PARTITIONS:
        missing[part] = sorted(partitions[part] - found[part])
        missing_without_annotations = len(partitions[part] - available_annotation_objects)
        coverage[part] = {
            "expected_objects_after_leakage_filter": len(partitions[part]),
            "expected_scenes_after_leakage_filter": len({scene_id(key) for key in partitions[part]}),
            "usable_objects": len(found[part]), "usable_rows": len(rows[part]),
            "usable_scenes": len({row["scene_id"] for row in rows[part]}),
            "missing_objects": len(missing[part]), "objects_absent_from_both_annotation_sources": missing_without_annotations,
            "objects_with_at_least_one_affected_part_by_reason": {reason: len(ids) for reason, ids in reasons[part].items()},
        }
    status = ("official_archive_sample" if a.allow_partial else
              "incomplete_public_release_membership" if any(missing.values()) else "complete_public_release_membership")
    manifest = {
        "protocol": "LGD author-released object membership with scene-disjoint training and validation",
        "status": status, "all_partitions_nonempty": all(rows.values()), "full_paper_run": False,
        "author_release": {"url": AUTHOR_URL, "commit": AUTHOR_COMMIT, "archives": {
            part: {"path": rel, "objects": count, "sha256": sha,
                   "scenes": len({scene_id(key) for key in official[part]})}
            for part, (rel, count, sha) in AUTHOR_LISTS.items()}},
        "annotation_revision": PP_REVISION, "annotation_sources": sources, "images": str(a.images.resolve()),
        "language_source": "pinned Grasp-Anything++ per-part instructions",
        "grasp_geometry": "cx,cy,width,height in 416px image coordinates and clockwise pi-periodic radians",
        "validation_rule": {"source": "eligible released train_seen scenes", "selection": "scene hash modulo 10 equals 0", "seed": a.seed},
        "removed_released_train_objects_for_eval_scene_overlap": len(removed),
        "removed_released_train_scenes_for_eval_scene_overlap": len({scene_id(key) for key in removed}),
        "eval_split_membership": {"test_seen": "author test/seen.obj", "test_unseen": "author test/unseen.obj"},
        "test_jsonl_is_union": True, "coverage": coverage,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f".{a.out.name}-", dir=a.out.parent))
    try:
        for part in PARTITIONS:
            write_rows(scratch / f"{part}.jsonl", rows[part])
        write_rows(scratch / "test.jsonl", rows["test_seen"] + rows["test_unseen"])
        (scratch / "missing_author_objects.json").write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
        (scratch / "excluded_train_objects.txt").write_text("\n".join(sorted(removed)) + "\n", encoding="ascii")
        (scratch / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        scratch.rename(a.out)
    finally:
        if scratch.is_dir() and not scratch.is_symlink() and scratch.resolve().parent == a.out.parent.resolve():
            shutil.rmtree(scratch)
    print(json.dumps(manifest), flush=True)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--instructions-zip", type=Path, required=True)
    parser.add_argument("--positive-zip", type=Path, required=True)
    parser.add_argument("--author-splits", type=Path, default=Path(__file__).with_name("lgd_release_501904d"))
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--allow-partial", action="store_true", help="Accept explicitly receipt-verified range samples")
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
