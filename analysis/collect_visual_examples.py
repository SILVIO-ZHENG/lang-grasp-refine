"""Reproduce the candidate traces behind the paper's qualitative examples."""

# Replay only eligible examples with the frozen 2D checkpoints. Use the
# same ranking pipeline and compare the replayed boxes with saved Top-1s.
# 只用冻结的二维检查点重跑符合条件的样例。复用同一套排序流程，再与保存的首选抓取框核对。

import argparse
import json
import math
from pathlib import Path
import re

from collect_paired_counts import METHODS, read_receipt


ROOT = Path(__file__).resolve().parents[1]
BASELINE, GEOMETRY, CONTACTS = METHODS
BEFORE_AFTER = {GEOMETRY: (BASELINE, GEOMETRY), CONTACTS: (GEOMETRY, CONTACTS)}


def target_name(prompt):
    # This display heuristic diversifies the examples by the final word of
    # the object phrase. It has no role in grounding or evaluation.
    # 这个展示用规则取物体短语的末词，让样例覆盖不同物体。它不参与定位或评测。
    prefix = re.split(r"\s+(?:on|by|at|from|with)\s+", prompt.lower(), maxsplit=1)[0]
    words = prefix.rsplit(maxsplit=1)
    return words[-1].strip(".,") if words else ""


def angle_error(pose, annotation):
    # A parallel-jaw grasp is periodic over 180 degrees.
    # 平行夹爪的抓取角度以 180° 为周期。
    return math.degrees(abs((pose[4] - annotation[4] + math.pi / 2) % math.pi - math.pi / 2))


def run(args):
    # Load inference dependencies for the replay command. Argument help and
    # standalone receipt auditing use only the standard library.
    # 执行重放时才加载推理依赖。查看参数帮助或单独核验评测记录只需标准库。
    import cv2
    import numpy as np
    import torch

    from panels import json_lines, load_panel_shard, manifest, read_regions, row_key, shard_path
    from pipeline import Images, ProposalBank, score_order
    from run_2d import load_fit
    from run_experiment import SIZE, correct, cv_rect, tokenize

    torch.set_num_threads(2)
    cv2.setNumThreads(0)
    data, digest = manifest(args.panels)
    fit, fit_digest = load_fit(args.fit, args.root, args.tag, digest)
    bank = ProposalBank(args.root, args.tag, args.device, fit=(fit["selection"], args.fit))
    used = set()

    def overlap(pose, gt):
        # Use the same image scale and OpenCV rectangle representation as
        # the benchmark to report IoU for the displayed reference annotation.
        # 使用与评测一致的图像尺度和 OpenCV 旋转框表示，计算图中参考标注的 IoU。
        _, points = cv2.rotatedRectangleIntersection(cv_rect(pose), cv_rect(gt))
        if points is None:
            return 0.0
        intersection = cv2.contourArea(points)
        union = (pose[2] * pose[3] + gt[2] * gt[3]) * SIZE**2 - intersection
        return intersection / union if union > 0 else 0.0

    def center_scores(model_data, images, prompts):
        # Render the branch's predicted center probabilities, before any
        # pose decoding or final score, in the paper's center-map panel.
        # 论文的中心热图面板显示当前分支在位姿解码和最终打分前预测的中心概率。
        _, model, vocabulary, _ = model_data
        words = torch.stack([tokenize(prompt, vocabulary) for prompt in prompts]).to(bank.device)
        with torch.inference_mode():
            return model(images.to(bank.device), words)[0].sigmoid().cpu().numpy()

    def candidates_for_shard(group, index, panel_rows, collected, represented):
        path = shard_path(args.evaluation, group, index)
        read_receipt(path, digest, len(panel_rows), fit_digest)
        candidates = []
        for offset, (source, row) in enumerate(zip(panel_rows, json_lines(path), strict=True)):
            if row_key(source) != row_key(row) or source["image"] != row["image"]:
                raise ValueError(f"Prediction rows do not match the panel: {path}")
            if row["image"] in used or len(row["prompt"]) > 44 or min(row["gt"][2:4]) < 0.032:
                continue
            for kind, (before, after) in BEFORE_AFTER.items():
                if len(collected[kind]) == args.examples or target_name(row["prompt"]) in represented[kind]:
                    continue
                first, second = (row["methods"][name] for name in (before, after))
                truth = np.asarray([row["gt"]], dtype=np.float32)
                box = np.asarray(second["prediction"])
                # Choose a miss-to-hit example whose single displayed label
                # also confirms the hit. Keep poses away from image edges.
                # 选择从失败变为成功、且图中这一条标注也能证实成功的样例，
                # 同时让抓取框远离图像边缘。
                if first["success"] or not second["success"] or not correct(box, truth):
                    continue
                if min(box[:2]) < 0.08 or max(box[:2]) > 0.92:
                    continue
                if kind == GEOMETRY:
                    distance = np.linalg.norm(np.subtract(first["prediction"][:2], box[:2]))
                    if not 0.017 < distance < 0.15:
                        continue
                candidates.append((offset, kind, row))
        return candidates

    def collect(group):
        collected = {kind: [] for kind in BEFORE_AFTER}
        represented = {kind: set() for kind in BEFORE_AFTER}
        group_shards = data["groups"][group]["shards"]
        for index in range(min(args.shards, len(group_shards))):
            panel_rows = load_panel_shard(args.panels, data, group, index)
            candidates = candidates_for_shard(group, index, panel_rows, collected, represented)
            if not candidates:
                continue
            cached, _ = read_regions(shard_path(args.cache, group, index), panel_rows, digest, fit["detector"])
            for start in range(0, len(candidates), args.batch):
                batch = [(offset, kind, row) for offset, kind, row in candidates[start:start + args.batch]
                         if len(collected[kind]) < args.examples and row["image"] not in used]
                if not batch:
                    continue
                rows = [row for _, _, row in batch]
                dataset = Images(rows, args.images)
                images = torch.stack([dataset[i][0] for i in range(len(batch))])
                prompts = [row["prompt"] for row in rows]
                actual = bank.results(images, prompts, [cached[offset] for offset, _, _ in batch])
                maps = {kind: center_scores(model, images, prompts) for kind, model in
                        ((GEOMETRY, bank.base), (CONTACTS, bank.aux)) if any(k == kind for _, k, _ in batch)}

                for member, ((_, kind, row), trace) in enumerate(zip(batch, actual, strict=True)):
                    target = target_name(row["prompt"])
                    if len(collected[kind]) >= args.examples or row["image"] in used or target in represented[kind]:
                        continue
                    raw, baseline_score, refined, refine_score, combined, combo_score = trace
                    baseline_id = int(score_order(baseline_score)[0])
                    refine_id = int(score_order(refine_score)[0])
                    combo_id = int(score_order(combo_score)[0])
                    before_id, after_id = BEFORE_AFTER[kind]
                    before, after = (row["methods"][name]["prediction"] for name in (before_id, after_id))

                    # The geometry case must preserve the winning heatmap
                    # index. The contact case must select an appended box.
                    # 几何细化样例的首选必须仍是同一个热图候选。接触点样例的首选
                    # 必须来自附加的接触点候选。
                    if kind == GEOMETRY:
                        if baseline_id != refine_id:
                            continue
                        replayed = (raw[baseline_id], refined[refine_id])
                        winner = refine_id
                    else:
                        if combo_id < len(refined):
                            continue
                        replayed = (refined[refine_id], combined[combo_id])
                        winner = combo_id
                    if not np.allclose(replayed, (before, after), atol=1e-4, rtol=1e-5):
                        continue
                    gt = row["gt"]
                    collected[kind].append({
                        "image": row["image"], "prompt": row["prompt"], "groundTruth": gt,
                        "before": before, "after": after, "map": maps[kind][member].round(5).tolist(),
                        "beforeSuccess": False, "afterSuccess": True,
                        "beforeIoU": overlap(before, gt), "afterIoU": overlap(after, gt),
                        "beforeAngle": angle_error(before, gt), "afterAngle": angle_error(after, gt),
                        "sameCandidate": kind == GEOMETRY, "winnerIndex": winner,
                    })
                    represented[kind].add(target)
                    used.add(row["image"])
            if all(len(cases) == args.examples for cases in collected.values()):
                return collected
        counts = {kind: len(cases) for kind, cases in collected.items()}
        raise RuntimeError(f"Too few matching examples in the requested shards for {group}: {counts}")

    groups = {group: collect(group) for group in args.groups}
    print(json.dumps(groups, separators=(",", ":")))


def main():
    from panels import TEST_GROUPS, positive

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Directory containing the two backbone runs")
    for path in ("panels", "cache", "evaluation", "fit", "images"):
        parser.add_argument(f"--{path}", required=True, type=Path)
    parser.add_argument("--tag", required=True, help="Backbone run directory name")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--groups", nargs="+", choices=TEST_GROUPS, default=TEST_GROUPS)
    parser.add_argument("--examples", type=positive, default=2, help="Number of examples per method and split")
    parser.add_argument("--shards", type=positive, default=24, help="Maximum shards scanned per split")
    parser.add_argument("--batch", type=positive, default=16)
    args = parser.parse_args()
    if args.tag in ("", ".", "..") or Path(args.tag).name != args.tag:
        parser.error("Expected a plain backbone run tag")
    run(args)


if __name__ == "__main__":
    main()
