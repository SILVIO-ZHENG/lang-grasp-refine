"""Publish a small, reproducible matched pilot from pinned official ZIP ranges.

Downloads a byte-contiguous prefix of each archive, checks response bounds and
each selected member's ZIP CRC, then repacks only matched target IDs. The pilot
receipt deliberately distinguishes this from full archive SHA-256 verification.

从固定版本的官方 ZIP 文件中抽取可复现的小规模标注样本。

下载各归档开头连续的字节范围，校验返回范围和所选成员的 ZIP CRC，然后只打包
指令与正例标注都齐全的目标。采样记录注明部分归档的核验范围，完整归档需要
单独进行 SHA-256 核验。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile

from prepare_data import (
    ID_PATTERN, MAX_INSTRUCTION_BYTES, MAX_TENSOR_BYTES, MIB, PP_REVISION,
    PrefixZipSource, ZipSource, clean_instruction, stable_hash,
)


ARCHIVES = {
    "grasp_instructions.zip": (1544210262, "afd0b539d70551a0bf9cb589c0d37230ec5029c4d7b4db470122665b3b8f9a6a"),
    "grasp_label_positive.zip": (3949367278, "3a43236f123b0abc3c6a6fe0d250f63de45f8ebf69b5647154c297500acafd4f"),
}


class HTTPSOnlyRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        if target.scheme != "https" or target.username or target.password:
            raise ValueError("Refusing an unexpected redirect for the public dataset")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def official_url(endpoint, name):
    return f"https://{endpoint}/datasets/airvlab/Grasp-Anything-pp/resolve/{PP_REVISION}/{name}?download=true"


def read_chunk(endpoint, name, start, stop, chunk):
    """Read one pinned HTTP byte range and publish it after checking its bounds.

    读取固定版本归档的一个 HTTP 字节范围，检查返回边界和实际长度后再保存。
    """
    size = stop - start
    if chunk.is_file() and chunk.stat().st_size == size:
        return chunk
    total = ARCHIVES[name][0]
    wanted_range = f"bytes {start}-{stop-1}/{total}"
    req = Request(official_url(endpoint, name), headers={
        "Accept-Encoding": "identity", "Range": f"bytes={start}-{stop-1}"})
    temporary = chunk.with_suffix(".inprogress")
    for attempt in range(4):
        try:
            with build_opener(HTTPSOnlyRedirect).open(req, timeout=25) as response:
                if response.status != 206 or response.headers.get("Content-Range") != wanted_range:
                    raise ValueError(f"Incorrect HTTP range for {name} [{start},{stop})")
                with temporary.open("wb") as sink:
                    remaining = size
                    while remaining:
                        data = response.read(min(MIB, remaining))
                        if not data:
                            raise OSError(f"Short range body for {name} [{start},{stop})")
                        sink.write(data)
                        remaining -= len(data)
                    if response.read(1):
                        raise OSError(f"Overlength range body for {name} [{start},{stop})")
                temporary.replace(chunk)
            return chunk
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("Unreachable retry loop")


def manifest_for(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(MIB):
            digest.update(block)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def select_targets(instructions, positives, excluded, paired_only, max_scenes, seed):
    """Sample scenes containing targets present in both annotation sources.

    Exclude every scene in the supplied history. In paired mode, require at
    least two object IDs and two distinct instruction texts in a scene.

    从两份标注都含有可匹配目标的场景中抽样。

    排除历史列表中的所有场景。开启配对模式时，每个场景至少包含两个不同的
    物体 ID 及两条不同的指令文本。
    """
    by_scene = {}
    for target in instructions.records.keys() & positives.records.keys():
        scene, object_id, _ = ID_PATTERN.fullmatch(target).groups()
        if scene not in excluded:
            by_scene.setdefault(scene, []).append((object_id, target))
    available_scenes = len(by_scene)
    if paired_only:
        eligible = {}
        for scene, candidates in by_scene.items():
            if len({object_id for object_id, _ in candidates}) < 2:
                continue
            prompts = {clean_instruction(instructions.read(target)) for _, target in candidates}
            if len(prompts) >= 2:
                eligible[scene] = candidates
        by_scene = eligible
    if len(by_scene) < (1 if paired_only else 25):
        raise ValueError(f"Only {len(by_scene)} eligible scenes, increase prefix sizes")
    scenes = sorted(by_scene, key=lambda scene: stable_hash(scene, seed, "pilot"))[:max_scenes]
    return available_scenes, len(by_scene), scenes, sorted(key for scene in scenes for _, key in by_scene[scene])


def sample(a):
    """Repack selected instruction and grasp records into matched pilot ZIPs.

    The receipt stores the source revision, sampling seed, excluded scenes,
    and checksums of the newly written archives.

    把选中的指令和抓取标注重新打包为相互匹配的试验 ZIP 文件。

    采样记录保存源版本、随机种子、排除场景的数量和输出归档的校验值。
    """
    if a.out.exists():
        raise ValueError("Pilot output directory must not already exist")
    a.work.mkdir(parents=True, exist_ok=True, mode=0o700)
    a.out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    want = dict(zip(ARCHIVES, (a.instruction_mib * MIB, a.positive_mib * MIB)))
    if not all(0 < n < ARCHIVES[name][0] for name, n in want.items()):
        raise ValueError("Prefix sizes must be positive and less than the full archive")
    downloaded = 0
    parts = {}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {}
        for name, length in want.items():
            parts[name] = []
            for start in range(0, length, 4 * MIB):
                stop = min(length, start + 4 * MIB)
                file = a.work / f"{PP_REVISION[:7]}-{name}-{start}-{stop}.range"
                parts[name].append(file)
                futures[pool.submit(read_chunk, a.endpoint, name, start, stop, file)] = stop - start
        for future in as_completed(futures):
            future.result()
            downloaded += futures[future]
            if downloaded % (16 * MIB) == 0 or downloaded == sum(want.values()):
                print(f"Official prefix ranges available: {downloaded / MIB:.0f} MiB", flush=True)
    prefixes = {}
    # Range requests finish out of order. Concatenate by byte offset so the
    # prefix can be parsed as consecutive ZIP local members.
    # 并发范围请求的完成顺序可能不同。按字节偏移依次拼接，才能连续解析 ZIP 本地成员。
    for name, file_list in parts.items():
        prefix = a.work / (name + ".prefix")
        with prefix.open("wb") as sink:
            for file in file_list:
                with file.open("rb") as source:
                    shutil.copyfileobj(source, sink, MIB)
        prefixes[name] = prefix
    scratch = Path(tempfile.mkdtemp(prefix=f".{a.out.name}-", dir=a.out.parent))
    try:
        with ExitStack() as stack:
            instructions = PrefixZipSource(prefixes["grasp_instructions.zip"], ".pkl", MAX_INSTRUCTION_BYTES, stack)
            positives = PrefixZipSource(prefixes["grasp_label_positive.zip"], ".pt", MAX_TENSOR_BYTES, stack)
            excluded = set()
            if a.exclude_instruction_zip:
                earlier = ZipSource(a.exclude_instruction_zip, ".pkl", MAX_INSTRUCTION_BYTES, stack)
                excluded = {ID_PATTERN.fullmatch(target)[1] for target in earlier.records}
            available, eligible, scenes, keep = select_targets(
                instructions, positives, excluded, a.paired_only, a.max_scenes, a.seed)
            print(f"Matched {available} unexcluded scenes and selected {len(keep)} targets across {len(scenes)} scenes", flush=True)
            for name, reader, prefix, ext in (
                ("grasp_instructions.zip", instructions, "grasp_instructions", ".pkl"),
                ("grasp_label_positive.zip", positives, "grasp_label_positive", ".pt"),
            ):
                with zipfile.ZipFile(scratch/name, "w", compression=zipfile.ZIP_STORED) as destination:
                    for key in keep:
                        destination.writestr(f"{prefix}/{key}{ext}", reader.read(key))
            receipt = {
                "kind": "official_pinned_zip_range_pilot",
                "repository": "https://huggingface.co/datasets/airvlab/Grasp-Anything-pp",
                "revision": PP_REVISION, "endpoint": a.endpoint, "seed": a.seed,
                "scenes": len(scenes), "targets": len(keep), "matched_scenes_before_sampling": available,
                "eligible_scenes": eligible, "paired_only": a.paired_only, "excluded_scenes": len(excluded),
                "excluded_instruction_archive": manifest_for(a.exclude_instruction_zip) if a.exclude_instruction_zip else None,
                "source_archives": {name: dict(total_bytes=meta[0], official_sha256=meta[1],
                                              received_prefix_bytes=want[name], scope="CRC-verified members from partial official archive")
                                    for name, meta in ARCHIVES.items()},
                "repacked_archives": {name: manifest_for(scratch/name) for name in ARCHIVES},
            }
        (scratch/"range_pilot_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
        scratch.rename(a.out)
        print(json.dumps(receipt), flush=True)
    finally:
        if scratch.is_dir() and not scratch.is_symlink() and scratch.resolve().parent == a.out.parent.resolve():
            shutil.rmtree(scratch)


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--instruction-mib", type=int, default=40)
    p.add_argument("--positive-mib", type=int, default=80)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-scenes", type=int, default=12000)
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--paired-only", action="store_true", help="Keep scenes with distinct object IDs and instructions")
    p.add_argument("--exclude-instruction-zip", type=Path, help="Exclude all scene IDs in a previous instruction archive")
    p.add_argument("--endpoint", choices=("huggingface.co", "hf-mirror.com"), default="huggingface.co")
    args = p.parse_args()
    if args.workers <= 0 or args.max_scenes < 25:
        p.error("--workers must be positive and --max-scenes at least 25")
    sample(args)
