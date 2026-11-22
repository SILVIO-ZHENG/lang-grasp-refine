"""Build scene-disjoint Grasp-Anything++ splits from pinned public assets.

Index annotations and object categories, cross-check instruction targets,
seal the split plan, then write train, validation, Seen and Unseen records.

从固定版本的公开数据构建按场景隔离的 Grasp-Anything++ 划分。

先为标注和物体类别建索引，再核对指令目标、冻结划分计划，最后写出训练集、
验证集、Seen 测试集和 Unseen 测试集。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing, contextmanager
import hashlib
import io
from itertools import groupby, islice
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import re
import sqlite3
import time
import unicodedata
import zipfile
import zlib

from prepare_data import (
    MAX_INSTRUCTION_BYTES, MAX_TENSOR_BYTES, PP_REVISION, PrimitivePickle, ZIP_HEADER,
    clean_instruction, load_grasps, stable_hash,
)
from prepare_official_split import DIRECTORY_HEADER, EOCD, ZIP64_EOCD, ZIP64_LOC, image_is_available
from sample_pp_annotations import ARCHIVES, manifest_for


ORIGINAL_REVISION = "6b4917455d0910c23f936a6c7a968418ba518897"
ORIGINAL_DESCRIPTIONS = (343076702, "357b26882c9a620aa05dce94bf0cd5b7a5970fd9a0c1e540bc56239f6588452b")
EXPECTED_PARTS = 4412383
EXPECTED_RGB = 994860
EXPECTED_MATCHED_SCENES = 809789
SCENE_STEM = re.compile(rb"[0-9a-f]{64}")
PART_STEM = re.compile(rb"([0-9a-f]{64})_(0|[1-9][0-9]*)_(0|[1-9][0-9]*)")
BATCH = 16384
VERSION = 1


def report(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False, allow_nan=False), flush=True)


def publish_json(path, data):
    temp = path.with_suffix(path.suffix + ".in_progress")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def normalized_label(label):
    label = unicodedata.normalize("NFKC", label).casefold().replace("_", " ").replace("-", " ")
    label = " ".join(label.strip(" \t\r\n.!?;,\"'").split())
    return re.sub(r"^(a|an|the) ", "", label)


def load_taxonomy(path):
    """Index normalized LVIS names and synonyms, marking ambiguous aliases as -1.

    为标准化的 LVIS 类别名和同义词建立索引。一个名称对应多个类别时标记为 -1。
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("categories")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Taxonomy must be an LVIS categories list or object with a categories list")
    categories, alias = {}, defaultdict(set)
    for row in raw:
        category_id, name = row["id"], row["name"]
        synonyms = row.get("synonyms", [])
        if type(category_id) is not int or category_id <= 0 or category_id in categories or not isinstance(name, str):
            raise ValueError("Invalid or repeated LVIS category ID/name")
        if not isinstance(synonyms, list) or not all(isinstance(word, str) for word in synonyms):
            raise ValueError(f"Invalid LVIS synonyms for {name}")
        categories[category_id] = (name, str(row.get("synset", "")))
        for term in [name, *synonyms]:
            clean = normalized_label(term)
            if clean:
                alias[clean].add(category_id)
    return categories, {word: next(iter(ids)) if len(ids) == 1 else -1 for word, ids in alias.items()}


def central_directory(file):
    """Read single-volume ZIP directory bounds, including the pinned legacy locator.

    读取单卷 ZIP 的目录边界，并兼容固定版本归档中的旧式定位器。
    """
    size = file.seek(0, os.SEEK_END)
    file.seek(max(0, size - 65535 - EOCD.size))
    base, tail = file.tell(), file.read()
    end = tail.rfind(b"PK\x05\x06")
    if end < 0 or end + EOCD.size > len(tail):
        raise ValueError("Missing ZIP end record")
    _, disk, directory_disk, count_disk, count, length, start, comment = EOCD.unpack_from(tail, end)
    if disk != 0 or directory_disk != 0 or count_disk != count or end + EOCD.size + comment != len(tail):
        raise ValueError("Expected a bounded single-volume ZIP end record")
    records_end = base + end
    locator_signature = end >= ZIP64_LOC.size and tail[end - ZIP64_LOC.size: end - ZIP64_LOC.size + 4] == b"PK\x06\x07"
    if locator_signature or count == 0xFFFF or length == 0xFFFFFFFF or start == 0xFFFFFFFF:
        locator_at = records_end - ZIP64_LOC.size
        file.seek(locator_at)
        magic, locator_disk, pointer, total_disks = ZIP64_LOC.unpack(file.read(ZIP64_LOC.size))
        # The original release's Zip64 locator writes zero here. Both disks in
        # its other end records are zero, matching Python's single-volume check.
        # 原始发行版的 Zip64 定位器在这里写入零。其他结尾记录中的磁盘编号也均为零，
        # 与 Python 的单卷校验规则一致。
        if magic != b"PK\x06\x07" or locator_disk != 0 or total_disks not in (0, 1) or pointer + ZIP64_EOCD.size > locator_at:
            raise ValueError("Expected a bounded single-volume ZIP64 locator")
        file.seek(pointer)
        magic, record_size, _, _, disk, directory_disk, count_disk, count, length, start = ZIP64_EOCD.unpack(
            file.read(ZIP64_EOCD.size))
        if (magic != b"PK\x06\x06" or disk != 0 or directory_disk != 0 or count_disk != count
                or record_size < ZIP64_EOCD.size - 12 or pointer + 12 + record_size != locator_at):
            raise ValueError("Expected a bounded single-volume ZIP64 end record")
        records_end = pointer
    if start + length > records_end or count * DIRECTORY_HEADER.size > length:
        raise ValueError("ZIP central directory exceeds archive bounds")
    return count, start, length


def zip_members(path, suffix, pattern):
    """Stream trusted single-volume ZIP directory metadata without a ZipInfo list.

    流式读取已核验的单卷 ZIP 目录元数据，无需创建完整的 ZipInfo 列表。
    """
    with path.open("rb", buffering=1024 * 1024) as file:
        count, start, length = central_directory(file)
        file.seek(start)
        for _ in range(count):
            entry = DIRECTORY_HEADER.unpack(file.read(DIRECTORY_HEADER.size))
            magic, _, _, flags, method, _, _, crc, packed, size, n, extra, comment, disk, _, _, local = entry
            if magic != b"PK\x01\x02" or flags & 1 or disk or local >= start:
                raise ValueError(f"Unexpected central entry in {path}")
            name = file.read(n).rsplit(b"/", 1)[-1]
            file.seek(extra + comment, os.SEEK_CUR)
            if not name:
                continue
            match = pattern.fullmatch(name[:-len(suffix)]) if name.endswith(suffix) else None
            if match is None:
                raise ValueError(f"Unexpected data filename {name!r} in {path}")
            yield match, (local, packed, size, method, crc)
        if file.tell() != start + length:
            raise ValueError(f"Invalid central directory bounds in {path}")


def payload(file, info, limit, expected_basename):
    local, packed, size, method, crc = info
    if size > limit or packed > limit + 1024:
        raise ValueError("ZIP member exceeds bounded annotation size")
    file.seek(local)
    magic, _, flags, local_method, _, _, _, _, _, name_size, extra = ZIP_HEADER.unpack(file.read(ZIP_HEADER.size))
    name = file.read(name_size).rsplit(b"/", 1)[-1]
    if magic != b"PK\x03\x04" or flags & 1 or local_method != method or name != expected_basename:
        raise ValueError("ZIP metadata does not match its local header")
    file.seek(extra, os.SEEK_CUR)
    body = file.read(packed)
    if len(body) != packed:
        raise ValueError("Short ZIP member")
    if method == zipfile.ZIP_STORED:
        value = body
    elif method == zipfile.ZIP_DEFLATED:
        decoder = zlib.decompressobj(-15)
        value = decoder.decompress(body, limit + 1)
        if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
            raise ValueError("Invalid DEFLATE member")
    else:
        raise ValueError(f"Unsupported ZIP compression {method}")
    if len(value) != size or zlib.crc32(value) != crc:
        raise ValueError("Unexpected ZIP member size or CRC")
    return value


def sha_verified(path, expected):
    size, digest = expected
    if path.stat().st_size != size:
        raise ValueError(f"Expected pinned archive size for {path}")
    observed = manifest_for(path)
    if observed["sha256"] != digest:
        raise ValueError(f"Expected pinned archive SHA-256 for {path}")
    report("archive_verified", archive=str(path), **observed)
    return observed


def row_groups(items, size=BATCH):
    block = []
    for item in items:
        block.append(item)
        if len(block) == size:
            yield block
            block = []
    if block:
        yield block


def open_work(work):
    conn = sqlite3.connect(work / "inventory.sqlite", timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-131072")
    conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS steps (name TEXT PRIMARY KEY)")
    return conn


def step(conn, name, action):
    """Each committed index stage survives interruption. Partial stages roll back.

    已提交的索引阶段在中断后保留。未完成的阶段会回滚。
    """
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT 1 FROM steps WHERE name=?", (name,)).fetchone():
            report("reuse_step", step=name)
            return
        started = time.monotonic()
        report("start_step", step=name)
        action()
        conn.execute("INSERT INTO steps VALUES (?)", (name,))
    report("commit_step", step=name, seconds=round(time.monotonic() - started, 2))


def source_identity(a):
    paths = {"instructions": a.instructions_zip, "positives": a.positive_zip,
             "scene_descriptions": a.scene_descriptions, "taxonomy": a.taxonomy}
    data = {"version": VERSION, "images": str(a.images.resolve())}
    for name, path in paths.items():
        stat = path.stat()
        data[name] = {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return data


def rgb_scene_ids(images):
    with os.scandir(images) as directory:
        return {bytes.fromhex(entry.name[:-4]) for entry in directory
                if entry.name.endswith(".jpg") and SCENE_STEM.fullmatch(entry.name[:-4].encode("utf-8"))}


def description_rows(path, rgb, aliases):
    with path.open("rb", buffering=1024 * 1024) as content:
        for match, info in zip_members(path, b".pkl", SCENE_STEM):
            name = match[0]
            scene = bytes.fromhex(name.decode("ascii"))
            try:
                raw = payload(content, info, MAX_INSTRUCTION_BYTES, name + b".pkl")
                value = PrimitivePickle(io.BytesIO(raw)).load()
                if (not isinstance(value, tuple) or len(value) != 2 or not isinstance(value[0], str)
                        or not isinstance(value[1], list) or not all(isinstance(query, str) for query in value[1])):
                    raise ValueError("Scene description schema differs from the expected tuple")
                labels = [aliases.get(normalized_label(query), 0) for query in value[1]]
                yield scene, scene in rgb, len(labels), labels.count(0), labels.count(-1), 0, labels
            except (ValueError, pickle.UnpicklingError, EOFError, zlib.error):
                yield scene, scene in rgb, 0, 0, 0, 1, []


def index_descriptions(conn, a, categories, aliases):
    """Index original RGB scene IDs and map each ordered object query to LVIS.

    为原版 RGB 场景 ID 建索引，并把场景中的每个物体查询按原有次序映射到 LVIS 类别。
    """
    rgb = rgb_scene_ids(a.images)
    if len(rgb) != EXPECTED_RGB:
        raise ValueError(f"Expected {EXPECTED_RGB:,} original RGB scene filenames, found {len(rgb):,}")
    conn.execute("CREATE TABLE descriptions (scene BLOB PRIMARY KEY, rgb INTEGER, queries INTEGER, unknown INTEGER, ambiguous INTEGER, invalid INTEGER) WITHOUT ROWID")
    conn.execute("CREATE TABLE query_labels (scene BLOB, obj INTEGER, category INTEGER, PRIMARY KEY (scene,obj)) WITHOUT ROWID")
    conn.execute("CREATE TABLE categories (category INTEGER PRIMARY KEY, name TEXT, synset TEXT, objects INTEGER DEFAULT 0, tier TEXT DEFAULT 'unobserved')")
    conn.executemany("INSERT INTO categories(category,name,synset) VALUES (?,?,?)",
                     ((i, name, synset) for i, (name, synset) in categories.items()))
    total = 0
    for block in row_groups(description_rows(a.scene_descriptions, rgb, aliases)):
        conn.executemany("INSERT INTO descriptions VALUES (?,?,?,?,?,?)", (row[:6] for row in block))
        conn.executemany("INSERT INTO query_labels VALUES (?,?,?)",
                         ((row[0], obj, category) for row in block for obj, category in enumerate(row[6]) if category > 0))
        total += len(block)
        if total % (BATCH * 8) == 0:
            report("description_progress", scenes=total)
    report("description_indexed", scenes=total, rgb=len(rgb), aliases=len(aliases))


def index_annotations(conn, table, path, suffix):
    """Index archive offsets by (scene, object, part) without extracting payloads.

    按（场景、物体、部位）记录每条标注的归档偏移，暂不解压实际内容。
    """
    conn.execute(f"CREATE TABLE {table} (scene BLOB,obj INTEGER,part INTEGER,local INTEGER,packed INTEGER,size INTEGER,method INTEGER,crc INTEGER)")
    values = ((bytes.fromhex(match[1].decode("ascii")), int(match[2]), int(match[3]), *info)
              for match, info in zip_members(path, suffix, PART_STEM))
    total = 0
    for block in row_groups(values):
        conn.executemany(f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?,?)", block)
        total += len(block)
        if total % (BATCH * 32) == 0:
            report("annotation_index_progress", source=table, members=total)
    if total != EXPECTED_PARTS:
        raise ValueError(f"Unexpected {table} member count: {total:,}")
    report("building_annotation_key_index", source=table, members=total)
    conn.execute(f"CREATE UNIQUE INDEX {table}_key ON {table}(scene,obj,part)")


def index_policy(conn):
    """Join paired target annotations and initialize Base/New by category frequency.

    配对同一目标的指令和正例抓取，并按类别频次初始化 Base 和 New 类别组。
    """
    conn.execute("""CREATE TABLE objects AS
        SELECT i.scene, i.obj, count(*) AS parts FROM instructions i INDEXED BY instructions_key
        JOIN positives p INDEXED BY positives_key USING(scene,obj,part) GROUP BY i.scene,i.obj""")
    conn.execute("CREATE UNIQUE INDEX object_key ON objects(scene,obj)")
    parts, scene_count = conn.execute("SELECT sum(parts),count(DISTINCT scene) FROM objects").fetchone()
    if (parts, scene_count) != (EXPECTED_PARTS, EXPECTED_MATCHED_SCENES):
        raise ValueError(f"Pin mismatch in matched counts: parts={parts}, scenes={scene_count}")
    conn.execute("""CREATE TABLE density AS SELECT q.category,count(*) AS objects
        FROM objects o JOIN query_labels q USING(scene,obj) JOIN descriptions d USING(scene)
        WHERE d.rgb=1 GROUP BY q.category""")
    ranks = list(conn.execute("SELECT category,objects FROM density ORDER BY objects DESC,category"))
    cutoff = math.floor(0.70 * len(ranks))
    if cutoff < 1 or len(ranks) <= cutoff:
        raise ValueError("No feasible Base/New grouping given observed categories")
    conn.executemany("UPDATE categories SET objects=?,tier=? WHERE category=?",
                     ((count, "base" if index < cutoff else "new", cid) for index, (cid, count) in enumerate(ranks)))
    refresh_scene_policy(conn)


def refresh_scene_policy(conn):
    """Record all scene queries as well as paired target counts for each tier.

    记录场景里全部已知查询的类别组，同时统计各类别组内带配对标注的目标数量。
    """
    conn.execute("DROP TABLE IF EXISTS query_tiers")
    conn.execute("DROP TABLE IF EXISTS scene_policy")
    conn.execute("""CREATE TABLE query_tiers AS SELECT q.scene,
        sum(c.tier='new') AS new_queries, sum(c.tier='unobserved') AS unobserved_queries
        FROM query_labels q JOIN categories c USING(category) GROUP BY q.scene""")
    conn.execute("CREATE UNIQUE INDEX query_tier_scene ON query_tiers(scene)")
    conn.execute("""CREATE TABLE scene_policy AS SELECT o.scene, coalesce(d.rgb,0) AS rgb,
        coalesce(d.invalid,1) AS invalid_description, coalesce(d.unknown,0) AS unknown_queries,
        coalesce(d.ambiguous,0) AS ambiguous_queries, coalesce(t.unobserved_queries,0) AS unobserved_queries,
        coalesce(t.new_queries,0) AS new_queries, count(*) AS objects, sum(o.parts) AS parts,
        sum(CASE WHEN c.tier='base' THEN o.parts ELSE 0 END) AS base_parts,
        sum(CASE WHEN c.tier='new' THEN o.parts ELSE 0 END) AS new_parts,
        sum(c.tier IS NULL OR c.tier='unobserved') AS unknown_target_objects
        FROM objects o LEFT JOIN descriptions d USING(scene) LEFT JOIN query_tiers t USING(scene)
        LEFT JOIN query_labels q USING(scene,obj) LEFT JOIN categories c USING(category) GROUP BY o.scene""")
    conn.execute("CREATE UNIQUE INDEX scene_policy_key ON scene_policy(scene)")


STRICT = "rgb=1 AND invalid_description=0 AND unknown_queries=0 AND ambiguous_queries=0 AND unobserved_queries=0 AND unknown_target_objects=0"
BASE = "new_queries=0 AND base_parts>0"
NEW = "new_queries>0 AND new_parts>0"


def inventory_summary(conn):
    def summarize(where):
        return dict(zip(("scenes", "parts", "base_parts", "new_parts"), conn.execute(
            f"SELECT count(*),coalesce(sum(parts),0),coalesce(sum(base_parts),0),coalesce(sum(new_parts),0) FROM scene_policy WHERE {where}"
        ).fetchone()))

    assignment = conn.execute("SELECT value FROM metadata WHERE key='category_assignment'").fetchone()
    policy = json.loads(assignment[0]) if assignment else None
    return {
        "version": VERSION,
        "class_source": "Unique normalized exact aliases of original per-scene object queries to pinned LVIS categories",
        "category_rank": (policy["methodology"] if policy else
                          "Distinct paired target objects in original RGB scenes; descending count then ascending LVIS ID; floor(0.70*n) Base"),
        "category_assignment": policy,
        "full_paper_membership_recovered": False,
        "coverage": {"all_matched": summarize("1"), "missing_or_unusable_description": summarize("invalid_description=1"),
                     "unknown_or_ambiguous_scene_queries": summarize("unknown_queries>0 OR ambiguous_queries>0 OR unobserved_queries>0"),
                     "any_known_new": summarize("new_queries>0"), "strict_base": summarize(f"{STRICT} AND {BASE}"),
                     "strict_new": summarize(f"{STRICT} AND {NEW}"), "base_including_unknown_context": summarize(f"rgb=1 AND {BASE}")},
        "categories": [dict(id=i, name=name, synset=synset, objects=count, tier=tier) for i, name, synset, count, tier
                       in conn.execute("SELECT category,name,synset,objects,tier FROM categories ORDER BY objects DESC,category")],
    }


def inventory(a):
    """Verify source archives and build a resumable SQLite metadata index.

    核验源归档，并建立可续建的 SQLite 元数据索引。
    """
    a.work.mkdir(parents=True, exist_ok=True, mode=0o700)
    sources = source_identity(a)
    categories, aliases = load_taxonomy(a.taxonomy)
    with closing(open_work(a.work)) as conn:
        def verify():
            verified = {"instructions": sha_verified(a.instructions_zip, ARCHIVES["grasp_instructions.zip"]),
                        "positives": sha_verified(a.positive_zip, ARCHIVES["grasp_label_positive.zip"]),
                        "scene_descriptions": sha_verified(a.scene_descriptions, ORIGINAL_DESCRIPTIONS),
                        "taxonomy": manifest_for(a.taxonomy)}
            conn.executemany("INSERT INTO metadata VALUES (?,?)", (("sources", json.dumps(sources)), ("checksums", json.dumps(verified))))

        existing = conn.execute("SELECT value FROM metadata WHERE key='sources'").fetchone()
        if existing and json.loads(existing[0]) != sources:
            raise ValueError("Inputs changed since inventory started; choose a fresh work directory")
        step(conn, "verify_sources", verify)
        step(conn, "index_descriptions", lambda: index_descriptions(conn, a, categories, aliases))
        step(conn, "index_instructions", lambda: index_annotations(conn, "instructions", a.instructions_zip, b".pkl"))
        step(conn, "index_positives", lambda: index_annotations(conn, "positives", a.positive_zip, b".pt"))
        step(conn, "index_class_policy", lambda: index_policy(conn))
        summary = inventory_summary(conn)
        summary.update(original_revision=ORIGINAL_REVISION, pp_revision=PP_REVISION, inputs=sources,
                       checksums=json.loads(conn.execute("SELECT value FROM metadata WHERE key='checksums'").fetchone()[0]))
        publish_json(a.work / "inventory.json", summary)
        report("inventory_complete", destination=str(a.work), coverage=summary["coverage"])


ACTION = re.compile(r"^(?:please )?(?:take hold of|pick up|give me|grasps?|holds?|grips?|grabs?|lifts?|fetch|takes?|get) +(.+)$")
PART_CLAUSE = re.compile(r" (?:by|at|on|from|using) (?:its|their|the|a|an) ")


def classify_instruction(prompt, aliases):
    """Resolve the explicit object phrase through a unique alias, or return zero.

    按唯一的类别别名识别指令明确提到的目标物体。无法确定时返回 0。
    """
    clean = normalized_label(prompt)
    action = ACTION.fullmatch(clean)
    if action is None:
        return 0
    noun_phrase = PART_CLAUSE.split(action[1], maxsplit=1)[0]
    return max(0, aliases.get(normalized_label(noun_phrase), 0))


def verified_work_sources(conn):
    complete = conn.execute("SELECT 1 FROM steps WHERE name='index_class_policy'").fetchone()
    if complete is None:
        raise ValueError("Run inventory first")
    settings = json.loads(conn.execute("SELECT value FROM metadata WHERE key='sources'").fetchone()[0])
    for name in ("instructions", "positives", "scene_descriptions", "taxonomy"):
        spec = settings[name]
        observed = Path(spec["path"]).stat()
        if (observed.st_size, observed.st_mtime_ns) != (spec["bytes"], spec["mtime_ns"]):
            raise ValueError(f"{name} changed since archive verification; use a fresh inventory")
    return settings


def ascii_target(scene, obj, part):
    return f"{scene.hex()}_{obj}_{part}"


def alignment_rows(conn, source, aliases):
    cursor = conn.execute("""SELECT i.scene,i.obj,i.part,i.local,i.packed,i.size,i.method,i.crc,q.category
        FROM instructions i NOT INDEXED LEFT JOIN query_labels q USING(scene,obj) ORDER BY i.rowid""")
    with source.open("rb", buffering=1024 * 1024) as content:
        for scene, obj, part, local, packed, size, method, crc, query in cursor:
            if query is None:
                yield scene, obj, part, 0
                continue
            basename = ascii_target(scene, obj, part).encode("ascii") + b".pkl"
            try:
                prompt = clean_instruction(payload(content, (local, packed, size, method, crc), MAX_INSTRUCTION_BYTES, basename))
            except (ValueError, EOFError, pickle.UnpicklingError, zlib.error):
                yield scene, obj, part, -2
                continue
            classified = classify_instruction(prompt, aliases)
            yield scene, obj, part, 0 if classified == 0 else 1 if classified == query else -1


def index_alignment(conn, source, aliases):
    conn.execute("CREATE TABLE prompt_checks (scene BLOB,obj INTEGER,part INTEGER,status INTEGER)")
    total = 0
    for block in row_groups(alignment_rows(conn, source, aliases)):
        conn.executemany("INSERT INTO prompt_checks VALUES (?,?,?,?)", block)
        total += len(block)
        if total % (BATCH * 32) == 0:
            report("prompt_alignment_progress", parts=total)
    if total != EXPECTED_PARTS:
        raise ValueError(f"Expected {EXPECTED_PARTS:,} prompt alignment rows, got {total:,}")
    conn.execute("CREATE UNIQUE INDEX prompt_check_key ON prompt_checks(scene,obj,part)")
    refresh_prompt_scene(conn)


def refresh_prompt_scene(conn):
    conn.execute("DROP TABLE IF EXISTS prompt_scene")
    conn.execute("""CREATE TABLE prompt_scene AS SELECT a.scene,
        sum(a.status=-1) AS conflicts, sum(a.status=0) AS unresolved, sum(a.status=-2) AS invalid,
        sum(CASE WHEN a.status=1 AND c.tier='base' THEN 1 ELSE 0 END) AS base_confirmed,
        sum(CASE WHEN a.status=1 AND c.tier='new' THEN 1 ELSE 0 END) AS new_confirmed
        FROM prompt_checks a INDEXED BY prompt_check_key LEFT JOIN query_labels q USING(scene,obj)
        LEFT JOIN categories c USING(category) GROUP BY a.scene""")
    conn.execute("CREATE UNIQUE INDEX prompt_scene_key ON prompt_scene(scene)")


def crosscheck(a):
    """Compare instruction subjects with the original query at the same object index.

    将抓取指令中的目标物体与原版场景中同一物体索引上的查询进行核对。
    """
    with closing(open_work(a.work)) as conn:
        sources = verified_work_sources(conn)
        _, aliases = load_taxonomy(Path(sources["taxonomy"]["path"]))
        step(conn, "index_prompt_alignment", lambda: index_alignment(conn, Path(sources["instructions"]["path"]), aliases))
        statuses = {-2: "invalid", -1: "conflicting", 0: "unresolved", 1: "confirmed"}
        counts = {statuses[i]: n for i, n in conn.execute("SELECT status,count(*) FROM prompt_checks GROUP BY status")}
        scenes = dict(zip(("with_conflicts", "with_confirmed_base", "with_confirmed_new"), conn.execute(
            "SELECT sum(conflicts>0),sum(base_confirmed>0),sum(new_confirmed>0) FROM prompt_scene").fetchone()))
        publish_json(a.work / "prompt_validation.json", {"counts": counts, "scenes": scenes,
                     "rule": "Classify only explicit instruction subjects by unique normalized LVIS aliases. Compare to original query at object index."})
        report("prompt_alignment_complete", parts=counts, scenes=scenes)


HISTORY_ROLES = {"train.jsonl": False, "val.jsonl": True, "test.jsonl": True,
                 "test_seen.jsonl": True, "test_unseen.jsonl": True}


def history_scenes(path):
    with path.open(encoding="utf-8") as records:
        for line in records:
            line = line.strip()
            if not line:
                continue
            scene = json.loads(line)["scene_id"] if line.startswith("{") else line
            if not isinstance(scene, str) or not SCENE_STEM.fullmatch(scene.encode("ascii")):
                raise ValueError(f"Invalid scene identifier in history file {path}")
            yield bytes.fromhex(scene)


def history_paths(a):
    files = {path.resolve(): False for path in a.prior_used}
    files.update({path.resolve(): True for path in a.protect})
    for folder in a.history_dir:
        if not (folder / "train.jsonl").is_file() or not (folder / "val.jsonl").is_file() or not any(
            (folder / name).is_file() for name in ("test.jsonl", "test_seen.jsonl", "test_unseen.jsonl")
        ):
            raise ValueError(f"Expected train, val and test split manifests in history directory {folder}")
        for rel, protected in HISTORY_ROLES.items():
            path = (folder / rel).resolve()
            if path.is_file():
                files[path] = files.get(path, False) or protected
    return files


def load_history(files):
    """Return all prior scenes, protected scenes, and checksums of their manifests.

    返回历史使用过的场景、禁止纳入任何新划分的受保护场景，以及历史清单的校验值。
    """
    seen, protect, official = set(), set(), []
    for path, protected in sorted(files.items()):
        ids = set(history_scenes(path))
        if not ids:
            raise ValueError(f"History file has no scene IDs: {path}")
        seen.update(ids)
        if protected:
            protect.update(ids)
        official.append({"path": str(path), "protected": protected, "scenes": len(ids), **manifest_for(path)})
    return seen, protect, official


def initialize_history_index(conn, previously_used, protected):
    conn.execute("CREATE TEMP TABLE history (scene BLOB PRIMARY KEY, protected INTEGER) WITHOUT ROWID")
    conn.executemany("INSERT INTO history VALUES (?,?)", ((s, int(s in protected)) for s in previously_used))


def candidate_count(conn, tier, require_alignment):
    policy = BASE if tier == "base" else NEW
    alignment = f"AND a.conflicts=0 AND a.{tier}_confirmed>0" if require_alignment else ""
    alignment_join = "JOIN prompt_scene a USING(scene)" if require_alignment else ""
    row_count = f"a.{tier}_confirmed" if require_alignment else f"{tier}_parts"
    count, rows = conn.execute(f"""SELECT count(*), coalesce(sum({row_count}),0)
        FROM scene_policy {alignment_join} LEFT JOIN history h USING(scene)
        WHERE h.scene IS NULL AND {STRICT} AND {policy} {alignment}""").fetchone()
    return {"scenes": count, "part_rows_upper_bound": rows}


def category_assignment(conn):
    row = conn.execute("SELECT value FROM metadata WHERE key='category_assignment'").fetchone()
    return json.loads(row[0]) if row else None


def build_membership(conn, categories, strict_train):
    """Pack exact category-to-scene memberships and training row counts in bitsets.

    Evaluation queries and confirmed targets are tracked separately so a New
    object can exclude a scene from Base even if it has no confirmed grasp there.

    用位集合保存精确的类别与场景对应关系及训练记录数量。

    分开记录评测场景里出现的物体与经核实的目标。只要场景含 New 类物体，即使没有
    对应的已核实抓取标注，该场景也不能进入 Base 组。
    """
    train_scope = STRICT if strict_train else "rgb=1 AND invalid_description=0"
    conn.execute(f"""CREATE TEMP TABLE balance_scope AS SELECT p.scene,
        (h.scene IS NULL AND {STRICT} AND a.base_confirmed+a.new_confirmed>0) AS use_eval,
        (coalesce(h.protected,0)=0 AND {train_scope} AND base_parts+new_parts-a.invalid>0) AS use_train,
        base_parts+new_parts-a.invalid AS rows
        FROM scene_policy p JOIN prompt_scene a USING(scene) LEFT JOIN history h USING(scene) WHERE a.conflicts=0""")
    conn.execute("CREATE UNIQUE INDEX balance_scope_key ON balance_scope(scene)")
    nbytes = (EXPECTED_MATCHED_SCENES + 7) // 8
    eval_queries, eval_targets, train_queries = [{cid: bytearray(nbytes) for cid in categories} for _ in range(3)]
    planes = {}
    cursor = conn.execute("""SELECT s.scene,s.use_eval,s.use_train,s.rows,q.category,
        CASE WHEN s.use_eval THEN EXISTS (SELECT 1 FROM prompt_checks a INDEXED BY prompt_check_key
             WHERE a.scene=q.scene AND a.obj=q.obj AND a.status=1) ELSE 0 END AS target
        FROM balance_scope s INDEXED BY balance_scope_key JOIN query_labels q USING(scene)
        WHERE s.use_eval OR s.use_train ORDER BY s.scene,q.obj""")
    eval_n = train_n = train_rows = 0

    def put(vector, idx):
        vector[idx // 8] |= 1 << (idx % 8)

    for _, block in groupby(cursor, key=lambda row: row[0]):
        group = list(block)
        _, use_eval, use_train, row_weight, _, _ = group[0]
        queries = {row[4] for row in group if row[4] in eval_queries}
        if use_eval:
            targets = {row[4] for row in group if row[5] and row[4] in eval_targets}
            if not targets:
                raise ValueError("Evaluation membership has no confirmed target")
            for cid in queries:
                put(eval_queries[cid], eval_n)
            for cid in targets:
                put(eval_targets[cid], eval_n)
            eval_n += 1
        if use_train:
            for cid in queries:
                put(train_queries[cid], train_n)
            train_rows += row_weight
            for shift in range(row_weight.bit_length()):
                if row_weight & (1 << shift):
                    if shift not in planes:
                        planes[shift] = bytearray(nbytes)
                    put(planes[shift], train_n)
            train_n += 1
    report("exact_scene_memberships", eval_scenes=eval_n, train_scenes=train_n, train_target_rows=train_rows)
    return (
        {cid: tuple(int.from_bytes(a[cid], "little") for a in (eval_queries, eval_targets, train_queries)) for cid in categories},
        {shift: int.from_bytes(value, "little") for shift, value in planes.items()},
        (eval_n, train_n, train_rows),
    )


def seeded_strata(ranked, count, n_new):
    n = len(ranked)
    count = min(n, count)
    strata = [ranked[i * n // count:(i + 1) * n // count] for i in range(count)]
    quotas = [len(group) * n_new // n for group in strata]
    order = sorted(range(count), key=lambda i: (-(len(strata[i]) * n_new % n), i))
    for i in order[:n_new - sum(quotas)]:
        quotas[i] += 1
    return list(zip(strata, quotas))


def seeded_category_search(a, ranked, membership):
    """Pick reserved categories with seeded sampling within frequency strata.

    Prefer assignments that meet the held-out scene quotas and reserve margin.
    Among equally feasible assignments, prefer more eligible training rows.

    在不同频次分层中按固定随机种子选取保留类别。

    优先满足留出场景的数量与备用余量。可行性相同时，优先保留更多训练记录。
    """
    masks, planes, (eval_n, train_n, train_rows) = membership
    n_new = len(ranked) - math.floor(0.70 * len(ranked))
    strata = seeded_strata(ranked, a.strata, n_new)
    min_base = a.val_scenes + a.seen_scenes
    min_new = a.unseen_scenes
    reserve_base = a.val_scenes + math.ceil(a.seen_scenes * a.reserve_factor)
    reserve_new = math.ceil(a.unseen_scenes * a.reserve_factor)
    rng = random.Random(a.seed)
    cached = {}

    def score(new):
        if new in cached:
            return cached[new]
        eval_blocked = eval_targets = train_blocked = 0
        for cid in new:
            query, target, training = masks[cid]
            eval_blocked |= query
            eval_targets |= target
            train_blocked |= training
        base_eval = eval_n - eval_blocked.bit_count()
        new_eval = eval_targets.bit_count()
        eligible_train = train_n - train_blocked.bit_count()
        eligible_train_rows = train_rows - sum((train_blocked & plane).bit_count() << shift for shift, plane in planes.items())
        quota = base_eval >= min_base and new_eval >= min_new
        padding = base_eval >= reserve_base and new_eval >= reserve_new
        status = 2 if padding else int(quota)
        shortfall = min(base_eval / min_base, new_eval / min_new) if not quota else 1
        key = (status, shortfall, eligible_train_rows, eligible_train, min(base_eval - a.val_scenes, new_eval))
        result = (key, {"fresh_base_scenes": base_eval, "fresh_unseen_scenes": new_eval,
                        "eligible_train_scenes_before_new_holdout": eligible_train,
                        "eligible_train_rows_before_new_holdout": eligible_train_rows})
        cached[new] = result
        return result

    best = None
    for _ in range(a.trials):
        new = frozenset(cid for group, quota in strata for cid in rng.sample(group, quota))
        if best is None or score(new)[0] > score(best)[0]:
            best = new
    movable = [group for group, quota in strata if 0 < quota < len(group)]
    if movable:
        for _ in range(a.refine_trials):
            group = rng.choice(movable)
            old = rng.choice([cid for cid in group if cid in best])
            replacement = rng.choice([cid for cid in group if cid not in best])
            proposal = (best - {old}) | {replacement}
            if score(proposal)[0] > score(best)[0]:
                best = proposal
    score_key, counts = score(best)
    return {
        "methodology": ("Seeded frequency-stratified class assignment over distinct paired target object count ranks. "
                        "Choose among random assignments and within-stratum swaps using exact metadata scene feasibility and Base training coverage."),
        "seed": a.seed, "random_trials": a.trials, "within_stratum_swap_trials": a.refine_trials,
        "unique_assignments_evaluated": len(cached), "observed_classes": len(ranked),
        "base_classes": len(ranked) - n_new, "new_classes": n_new,
        "new_category_ids": sorted(best),
        "strata": [{"category_ids": group, "new_count": quota} for group, quota in strata],
        "required_scenes": {"base_val_seen_combined": min_base, "unseen": min_new},
        "reserve_target_scenes": {"base_val_seen_combined": reserve_base, "unseen": reserve_new},
        "quota_feasible": score_key[0] > 0, "meets_reserve_targets": score_key[0] == 2,
        "selected_coverage": counts, "strict_train_context": a.strict_train,
    }


def rebalance(a):
    if (min(a.val_scenes, a.seen_scenes, a.unseen_scenes, a.trials, a.strata) < 1 or a.refine_trials < 0
            or not math.isfinite(a.reserve_factor) or a.reserve_factor < 1):
        raise ValueError("Invalid category rebalance quotas or search configuration")
    used, protected, receipts = load_history(history_paths(a))
    with closing(open_work(a.work)) as conn:
        sources = verified_work_sources(conn)
        if conn.execute("SELECT 1 FROM steps WHERE name='index_prompt_alignment'").fetchone() is None:
            raise ValueError("Run crosscheck before the seeded class assignment")
        config = dict(seed=a.seed, strata=a.strata, trials=a.trials, refine_trials=a.refine_trials, reserve_factor=a.reserve_factor,
                      val_scenes=a.val_scenes, seen_scenes=a.seen_scenes, unseen_scenes=a.unseen_scenes, strict_train=a.strict_train)
        if (previous := category_assignment(conn)) is not None:
            if previous["settings"] != config or previous["history_files"] != receipts:
                raise ValueError("This inventory already has a different category assignment")
            assignment = previous
        else:
            if (a.work / "plan.json").is_file():
                raise ValueError("A plan is already sealed for the original category assignment")
            initial = a.work / "inventory.json"
            if initial.is_file() and not (a.work / "inventory_frequency_top70.json").exists():
                publish_json(a.work / "inventory_frequency_top70.json", json.loads(initial.read_text(encoding="utf-8")))
            old = a.work / "feasibility.json"
            if old.is_file() and not (a.work / "feasibility_frequency_top70.json").exists():
                publish_json(a.work / "feasibility_frequency_top70.json", json.loads(old.read_text(encoding="utf-8")))
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                initialize_history_index(conn, used, protected)
                ranked = [cid for cid, in conn.execute("SELECT category FROM categories WHERE objects>0 ORDER BY objects DESC,category")]
                membership = build_membership(conn, ranked, a.strict_train)
                assignment = seeded_category_search(a, ranked, membership)
                assignment.update(settings=config, history_files=receipts, prior_scene_counts={"used": len(used), "protected": len(protected)})
                report("category_assignment_candidate", **assignment["selected_coverage"],
                       quota_feasible=assignment["quota_feasible"], meets_reserve_targets=assignment["meets_reserve_targets"])
                publish_json(a.work / "rebalance.json", assignment)
                if not assignment["quota_feasible"]:
                    raise SystemExit(2)
                conn.execute("UPDATE categories SET tier='base' WHERE objects>0")
                conn.executemany("UPDATE categories SET tier='new' WHERE category=?", ((i,) for i in assignment["new_category_ids"]))
                refresh_scene_policy(conn)
                refresh_prompt_scene(conn)
                actual = {tier: candidate_count(conn, tier, True)["scenes"] for tier in ("base", "new")}
                expected = assignment["selected_coverage"]
                if actual != {"base": expected["fresh_base_scenes"], "new": expected["fresh_unseen_scenes"]}:
                    raise ValueError(f"SQL and packed-bitset scene eligibility disagree: {actual}")
                conn.execute("INSERT INTO metadata VALUES ('category_assignment',?)", (json.dumps(assignment),))
        summary = inventory_summary(conn)
        summary.update(original_revision=ORIGINAL_REVISION, pp_revision=PP_REVISION, inputs=sources,
                       checksums=json.loads(conn.execute("SELECT value FROM metadata WHERE key='checksums'").fetchone()[0]))
        publish_json(a.work / "inventory.json", summary)
        publish_json(a.work / "rebalance.json", assignment)
        report("category_assignment_sealed", **assignment["selected_coverage"], classes=assignment["observed_classes"],
               base=assignment["base_classes"], new=assignment["new_classes"], reserve_met=assignment["meets_reserve_targets"])


def plan(a):
    """Seal quotas and source history after checking scene coverage and alignment.

    核查场景数量和指令对齐状态后，冻结各划分的目标数量与历史来源。
    """
    if min(a.val_scenes, a.seen_scenes, a.unseen_scenes) < 1:
        raise ValueError("All three holdouts require a positive scene quota")
    used, protected, files = load_history(history_paths(a))
    with closing(open_work(a.work)) as conn:
        verified_work_sources(conn)
        initialize_history_index(conn, used, protected)
        alignment = bool(conn.execute("SELECT 1 FROM steps WHERE name='index_prompt_alignment'").fetchone())
        available = {tier: candidate_count(conn, tier, alignment) for tier in ("base", "new")}
        requests = {"val": a.val_scenes, "test_seen": a.seen_scenes, "test_unseen": a.unseen_scenes}
        assignment = category_assignment(conn)
        if assignment is not None:
            planned = assignment["settings"]
            given = dict(seed=a.seed, val_scenes=a.val_scenes, seen_scenes=a.seen_scenes,
                         unseen_scenes=a.unseen_scenes, strict_train=a.strict_train)
            if any(planned[key] != value for key, value in given.items()) or assignment["history_files"] != files:
                raise ValueError("Plan options and history must match the selected category assignment")
        fits = available["base"]["scenes"] >= a.val_scenes + a.seen_scenes and available["new"]["scenes"] >= a.unseen_scenes
        status = "ready" if fits and alignment else "requires_prompt_alignment" if fits else "insufficient_eligible_scenes"
        config = {
            "version": VERSION, "status": status, "seed": a.seed, "required_unique_scenes": requests,
            "candidate_coverage": available, "prompt_alignment_complete": alignment,
            "historical_scene_counts": {"excluded_from_new_val_and_tests": len(used), "excluded_from_all_splits": len(protected)},
            "history_files": files, "strict_train_context": a.strict_train,
            "category_assignment": assignment,
            "category_provenance": "Public object-query LVIS alias mapping; original full Table 2 assignments unavailable",
            "validation_policy": "Base only, fully resolved query labels, no known New query, only confirmed instructions",
            "unseen_policy": "At least one known New target; mixed scenes kept out of Base cohorts; only confirmed New rows scored",
            "unresolved_or_conflict_policy": "Any conflicting instruction or unresolved object query invalidates whole held-out scene; unconfirmed instruction rows omitted",
        }
        publish_json(a.work / "feasibility.json", config)
        report("split_feasibility", status=status, coverage=available, required=requests, used=len(used), protected=len(protected))
        if status == "ready":
            path = a.work / "plan.json"
            if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != config:
                raise ValueError("This work directory has a different sealed plan")
            publish_json(path, config)
        else:
            raise SystemExit(2)


def candidate_scenes(conn, tier, seed):
    """Return reproducibly ordered, fresh, fully resolved held-out candidates.

    返回无历史使用记录、查询已完全解析的留出候选场景，并以可复现的顺序排列。
    """
    policy = BASE if tier == "base" else NEW
    ids = [scene for scene, in conn.execute(f"""SELECT p.scene FROM scene_policy p
        JOIN prompt_scene a USING(scene) LEFT JOIN history h USING(scene)
        WHERE h.scene IS NULL AND {STRICT} AND {policy} AND a.conflicts=0 AND a.{tier}_confirmed>0""")]
    ids.sort(key=lambda scene: (stable_hash(scene.hex(), seed, f"grasp_full_{tier}_eval_v1"), scene))
    return ids


def training_scenes(conn, strict):
    """Return Base scenes without prior holdouts, known New queries, or conflicts.

    返回 Base 场景，排除历史留出场景、含已知 New 类物体的场景以及指令冲突场景。
    """
    scope = STRICT if strict else "rgb=1 AND invalid_description=0"
    return [scene for scene, in conn.execute(f"""SELECT p.scene FROM scene_policy p
        JOIN prompt_scene a USING(scene) LEFT JOIN history h USING(scene)
        WHERE coalesce(h.protected,0)=0 AND {scope} AND {BASE} AND a.conflicts=0 ORDER BY p.scene""")]


def scene_rows(conn, scene, split, images, instructions, positives, screening):
    """Decode valid part-level instruction and grasp rows for one eligible scene.

    Unseen emits confirmed New targets. The other partitions emit Base targets.
    Only training may keep an unresolved instruction with a known Base target.

    为一个符合条件的场景解码有效的部位级指令和抓取记录。

    Unseen 只输出经核实的 New 目标，其余划分只输出 Base 目标。只有训练集可以保留
    指令未能核实但目标属于已知 Base 类别的记录。
    """
    scene_hex = scene.hex()
    image = f"{scene_hex}.jpg"
    if not image_is_available(images, image):
        screening["unreadable_or_wrong_size_rgb_scenes"] += 1
        return []
    tier = "new" if split == "test_unseen" else "base"
    allowed = "a.status IN (0,1)" if split == "train" else "a.status=1"
    records = conn.execute(f"""SELECT i.obj,i.part,i.local,i.packed,i.size,i.method,i.crc,
        p.local,p.packed,p.size,p.method,p.crc
        FROM instructions i INDEXED BY instructions_key JOIN positives p INDEXED BY positives_key USING(scene,obj,part)
        JOIN query_labels q USING(scene,obj) JOIN categories c USING(category) JOIN prompt_checks a USING(scene,obj,part)
        WHERE i.scene=? AND c.tier=? AND {allowed} ORDER BY i.obj,i.part""", (scene, tier))
    rows = []
    for obj, part, *metadata in records:
        target_id = ascii_target(scene, obj, part)
        target = target_id.encode("ascii")
        try:
            prompt = clean_instruction(payload(instructions, metadata[:5], MAX_INSTRUCTION_BYTES, target + b".pkl"))
            grasps = load_grasps(payload(positives, metadata[5:], MAX_TENSOR_BYTES, target + b".pt"), 416, 0)
        except (ValueError, RuntimeError, EOFError, pickle.UnpicklingError, zipfile.BadZipFile, zlib.error):
            screening["unreadable_target_rows"] += 1
            continue
        if not grasps:
            screening["rows_with_no_valid_grasps"] += 1
            continue
        rows.append({"scene_id": scene_hex, "image": image, "prompt": prompt, "grasps": grasps,
                     "target_id": target_id, "object_id": obj, "part_id": part, "width": 416, "height": 416})
    for row in rows:
        row["same_scene_other_targets"] = [
            {"target_id": other["target_id"], "prompt": other["prompt"]}
            for other in rows if other["object_id"] != row["object_id"] and other["prompt"] != row["prompt"]
        ][:3]
    return rows


@contextmanager
def materialization_lock(state):
    lock = state / "lock"
    with lock.open("a+b") as file:
        if os.name == "nt":
            import msvcrt
            if file.tell() == 0:
                file.write(b"\0")
                file.flush()
            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


_READER = None


def initialize_reader(database, sources):
    import torch
    torch.set_num_threads(1)
    conn = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    conn.execute("PRAGMA query_only=ON")
    global _READER
    _READER = (conn, Path(sources["images"]), Path(sources["instructions"]["path"]).open("rb"),
               Path(sources["positives"]["path"]).open("rb"))


def decode_scene_group(split, scenes):
    conn, images, instructions, positives = _READER
    result = []
    for cursor, scene in scenes:
        screening = Counter()
        rows = scene_rows(conn, scene, split, images, instructions, positives, screening) if scene is not None else None
        result.append((cursor, rows, screening))
    return result


def decoded_candidates(split, candidates, taken, start, conn, images, instructions, positives, executor, workers):
    scenes = ((cursor, None if scene in taken else scene)
              for cursor, scene in enumerate(islice(candidates, start, None), start=start))
    if executor is None:
        for cursor, scene in scenes:
            screening = Counter()
            rows = scene_rows(conn, scene, split, images, instructions, positives, screening) if scene is not None else None
            yield cursor, rows, screening
        return
    pending = deque()
    groups = iter(row_groups(scenes, size=32))
    try:
        for group in islice(groups, workers * 2):
            pending.append(executor.submit(decode_scene_group, split, group))
        while pending:
            result = pending.popleft().result()
            group = next(groups, None)
            if group is not None:
                pending.append(executor.submit(decode_scene_group, split, group))
            yield from result
    finally:
        for future in pending:
            future.cancel()


def write_partition(split, candidates, taken, goal, conn, sources, stage, state, executor=None, workers=1):
    """Stream one JSONL partition and resume from its committed byte offset.

    A scene counts toward the quota when at least one usable row is written.

    流式写入一个 JSONL 划分，并从上次提交的字节偏移续写。

    一个场景至少写入一条有效记录后，才计入场景配额。
    """
    progress_file = state / f"{split}.json"
    path = stage / f"{split}.jsonl"
    progress = json.loads(progress_file.read_text(encoding="utf-8")) if progress_file.is_file() else {
        "cursor": 0, "scenes": 0, "rows": 0, "bytes": 0, "screening": {}, "complete": False,
    }
    if progress["complete"]:
        if path.stat().st_size != progress["bytes"]:
            raise ValueError(f"Completed partition has unexpected length: {path}")
        return progress
    if progress["cursor"] > len(candidates) or (progress["bytes"] and not path.is_file()):
        raise ValueError(f"Partition checkpoint disagrees with its data: {split}")
    screening = Counter(progress["screening"])
    images = Path(sources["images"])
    begun = time.monotonic()
    with path.open("r+b" if path.is_file() else "w+b") as sink, \
            Path(sources["instructions"]["path"]).open("rb") as instructions, \
            Path(sources["positives"]["path"]).open("rb") as positives:
        if sink.seek(0, os.SEEK_END) < progress["bytes"]:
            raise ValueError(f"Partition is shorter than its checkpoint: {path}")
        sink.truncate(progress["bytes"])
        sink.seek(progress["bytes"])

        def commit(complete=False):
            sink.flush()
            os.fsync(sink.fileno())
            progress.update(bytes=sink.tell(), screening=dict(screening), complete=complete)
            publish_json(progress_file, progress)
            report("materialize_progress", split=split, candidate_cursor=progress["cursor"],
                   scenes=progress["scenes"], rows=progress["rows"], required=goal,
                   seconds=round(time.monotonic() - begun, 2), complete=complete)

        decoded = decoded_candidates(split, candidates, taken, progress["cursor"], conn, images,
                                     instructions, positives, executor, workers)
        with closing(decoded):
            if goal is None or progress["scenes"] < goal:
                for cursor, rows, local_screening in decoded:
                    screening.update(local_screening)
                    if rows:
                        for row in rows:
                            sink.write((json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
                        progress["scenes"] += 1
                        progress["rows"] += len(rows)
                    elif rows is not None:
                        screening["scenes_without_usable_rows"] += 1
                    progress["cursor"] = cursor + 1
                    if progress["cursor"] % 1024 == 0:
                        commit()
                    if goal is not None and progress["scenes"] >= goal:
                        break
        if goal is not None and progress["scenes"] != goal:
            commit()
            raise ValueError(f"{split} exhausted its verified candidate pool before reaching its requested {goal:,} distinct scenes")
        if progress["scenes"] == 0:
            commit()
            raise ValueError(f"{split} has no usable scenes")
        commit(complete=True)
    return progress


def read_sealed_plan(work):
    path = work / "plan.json"
    if not path.is_file():
        raise ValueError("No feasible sealed plan. Run inventory, crosscheck and plan first.")
    data = path.read_bytes()
    config = json.loads(data)
    if config.get("version") != VERSION or config.get("status") != "ready" or not config.get("prompt_alignment_complete"):
        raise ValueError("Expected an alignment-verified sealed plan")
    return config, hashlib.sha256(data).hexdigest()


def materialize(a):
    """Publish the sealed split only after all scene-disjointness checks pass.

    Fill validation and both test partitions first. Training then consumes the
    eligible remainder. The manifest records source and split checksums.

    在通过各划分的场景隔离检查后，再发布冻结的划分。

    先分配验证集和两个测试集，再用其余符合条件的场景构建训练集。清单记录源数据
    和输出划分的校验值。
    """
    workers = a.workers
    if workers < 1:
        raise ValueError("Expected at least one materialization worker")
    config, digest = read_sealed_plan(a.work)
    out = a.out.resolve()
    if out.exists():
        manifest = out / "manifest.json"
        if manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")).get("plan_sha256") == digest:
            report("reuse_published_split", destination=str(out))
            return
        raise ValueError("Output exists and does not match the sealed plan")
    used, protected, receipts = load_history({Path(r["path"]): r["protected"] for r in config["history_files"]})
    if receipts != config["history_files"]:
        raise ValueError("A prior split manifest changed after the plan was sealed")
    out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    stage = out.parent / f".{out.name}.staging"
    state_suffix = hashlib.sha256(os.fsencode(str(out))).hexdigest()[:20]
    state = a.work / f"materialize-{state_suffix}"
    state.mkdir(exist_ok=True, mode=0o700)
    with materialization_lock(state), closing(open_work(a.work)) as conn:
        tag = state / "plan_digest.json"
        if tag.is_file() and json.loads(tag.read_text(encoding="utf-8")) != digest:
            raise ValueError("Existing checkpoint has a different plan")
        if not tag.is_file() and stage.exists():
            raise ValueError("Existing stage has no matching checkpoint")
        publish_json(tag, digest)
        stage.mkdir(exist_ok=True, mode=0o700)
        sources = verified_work_sources(conn)
        initialize_history_index(conn, used, protected)
        base = candidate_scenes(conn, "base", config["seed"])
        new = candidate_scenes(conn, "new", config["seed"])
        train = training_scenes(conn, config["strict_train_context"])
        taken = set(used)
        progress = {}
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=initialize_reader, initargs=(a.work / "inventory.sqlite", sources)) if workers > 1 else None
        try:
            for split, candidates in (("val", base), ("test_seen", base), ("test_unseen", new), ("train", train)):
                goal = config["required_unique_scenes"].get(split)
                forbidden = taken if goal is not None else protected | (taken - used)
                progress[split] = write_partition(split, candidates, forbidden, goal, conn, sources, stage, state, pool, workers)
                selected = set(history_scenes(stage / f"{split}.jsonl"))
                if len(selected) != progress[split]["scenes"] or selected & forbidden:
                    raise ValueError(f"Scene selection or disjointness audit failed for {split}")
                taken.update(selected)
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        inventory_data = json.loads((a.work / "inventory.json").read_text(encoding="utf-8"))
        manifest = {
            "status": "complete", "protocol": "Deterministic independent scene splits using public query-to-LVIS aliases",
            "plan_sha256": digest, "plan": config, "original_revision": ORIGINAL_REVISION, "pp_revision": PP_REVISION,
            "images": sources["images"], "source_checksums": inventory_data["checksums"],
            "category_rank": inventory_data["category_rank"], "categories": inventory_data["categories"],
            "train_policy": ("Only matched Base targets. Exclude every scene with a known New query, conflicting instruction, "
                             "prior holdout or newly sealed holdout. Unknown query context may remain unless strict_train_context is true."),
            "angle_convention": "Clockwise pi-periodic radians in 416px image coordinates",
            "scene_counts": {part: stats["scenes"] for part, stats in progress.items()},
            "row_counts": {part: stats["rows"] for part, stats in progress.items()},
            "screening": {part: stats["screening"] for part, stats in progress.items()},
            "split_checksums": {part: manifest_for(stage / f"{part}.jsonl") for part in progress},
        }
        publish_json(stage / "manifest.json", manifest)
        stage.rename(out)
        report("split_published", destination=str(out), scene_counts=manifest["scene_counts"], row_counts=manifest["row_counts"])


def split_options(parser):
    parser.add_argument("--work", type=Path, required=True)
    for flag in ("--history-dir", "--prior-used", "--protect"):
        parser.add_argument(flag, type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--val-scenes", type=int, default=40000)
    parser.add_argument("--seen-scenes", type=int, default=40000)
    parser.add_argument("--unseen-scenes", type=int, default=40000)
    parser.add_argument("--strict-train", action="store_true", help="Also reject unresolved object-query context from training scenes")


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True, dest="stage")
    inv = commands.add_parser("inventory", help="Verify and index the complete pinned corpus with category coverage")
    for field in ("images", "scene-descriptions", "instructions-zip", "positive-zip", "taxonomy", "work"):
        inv.add_argument("--" + field, type=Path, required=True)
    inv.set_defaults(action=inventory)
    align = commands.add_parser("crosscheck", help="Validate explicit ++ instruction objects against original object queries")
    align.add_argument("--work", type=Path, required=True)
    align.set_defaults(action=crosscheck)
    balance = commands.add_parser("rebalance", help="Select a seeded class-stratified assignment that meets the scene quotas")
    split_options(balance)
    balance.add_argument("--strata", type=int, default=10)
    balance.add_argument("--trials", type=int, default=256)
    balance.add_argument("--refine-trials", type=int, default=256)
    balance.add_argument("--reserve-factor", type=float, default=1.375)
    balance.set_defaults(action=rebalance)
    planner = commands.add_parser("plan", help="Check exact unique-scene quotas and seal eligible split policy")
    split_options(planner)
    planner.set_defaults(action=plan)
    export = commands.add_parser("materialize", help="Validate 416px RGB and positive poses, then atomically publish sealed JSONL partitions")
    export.add_argument("--work", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--workers", type=int, default=1, help="Bounded ordered CPU readers for RGB and label validation")
    export.set_defaults(action=materialize)
    a = parser.parse_args()
    a.action(a)


if __name__ == "__main__":
    main()
