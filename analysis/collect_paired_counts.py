"""Audit saved 2D grasp evaluations and count per-instruction Top-1 changes."""

# Work from archived predictions and receipts with the standard library.
# Each prediction contains the first choice. Top-5 and pool coverage counts
# come from the receipts written during the full evaluation.
# 使用标准库分析已保存的预测与完成记录。每条预测保存首选抓取框。
# Top-5 和候选池覆盖次数读取自完整评测时写入的完成记录。

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "12_full_scale"))
from panels import TEST_GROUPS, json_lines, load_panel_shard, manifest, row_key, sha256, shard_path


# These keys match the archived evaluation schema and identify the same
# methods even if the implementation directories are renamed.
# 这些键与已保存的评测格式一致。即使实现目录改名，它们仍指向同样的方法。
METHODS = ("grounded_zoom", "geometry_refinement")
TRANSITIONS = ("00", "01", "10", "11")
RECEIPT_VERSION = 1
COUNTS = ("top1_successes", "top5_successes", "oracle_successes")


def read_receipt(predictions, panel_digest, n_rows, fit_digest):
    # Apply the same receipt version, run identity, row count and file-digest
    # checks used when resuming the evaluator.
    # 按评测续跑时的规则，核验完成记录版本、运行标识、行数和预测文件指纹。
    path = predictions.with_suffix(".json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (receipt["version"] != RECEIPT_VERSION or receipt["panel_sha256"] != panel_digest
            or receipt["rows"] != n_rows):
        raise ValueError(f"Evaluation receipt does not match the panel shard: {path}")
    if fit_digest is not None and receipt["fit_sha256"] != fit_digest:
        raise ValueError(f"Evaluation receipts refer to different fitted heads: {path}")
    if sha256(predictions) != receipt["predictions_sha256"]:
        raise ValueError(f"Evaluation predictions do not match their receipt: {predictions}")
    return receipt


def count_shard(panel_rows, predictions, transitions):
    # Pair outcomes within one instruction. A configuration change can both
    # gain previously missed grasps (01) and lose previously correct ones (10).
    # 对同一条指令的结果做配对统计。更换配置既可能把失败变成成功（01），
    # 也可能把原本的成功变成失败（10）。
    successes = Counter()
    for source, row in zip(panel_rows, json_lines(predictions), strict=True):
        if row_key(source) != row_key(row) or source["image"] != row["image"]:
            raise ValueError(f"Evaluation row does not match its panel instruction: {predictions}")
        outcomes = {method: row["methods"][method]["success"] for method in METHODS}
        if any(type(hit) is not bool for hit in outcomes.values()):
            raise ValueError(f"Expected boolean success flags: {predictions}")
        successes.update(method for method, hit in outcomes.items() if hit)
        before, after = (int(outcomes[method]) for method in METHODS)
        transitions[f"{before}{after}"] += 1
    return successes


def audit(panels, evaluation):
    # Walk the manifest's complete plan. This catches a missing final shard
    # as well as missing artifacts between two existing shard indices.
    # 按清单列出的完整计划读取。既能发现中间缺失的产物，也能发现末尾缺失的分片。
    data, panel_digest = manifest(panels)
    fit_digest = None
    methods = {method: {} for method in METHODS}
    transitions = {}
    for group in TEST_GROUPS:
        panel = data["groups"][group]
        expected = panel["shards"]
        totals = {method: Counter() for method in METHODS}
        transitions[group] = {"refinement": dict.fromkeys(TRANSITIONS, 0)}
        n_rows = 0
        for index, shard in enumerate(expected):
            panel_rows = load_panel_shard(panels, data, group, index)
            path = shard_path(evaluation, group, index)
            receipt = read_receipt(path, panel_digest, shard["rows"], fit_digest)
            fit_digest = receipt["fit_sha256"]
            if not isinstance(fit_digest, str) or not fit_digest:
                raise ValueError(f"Missing fitted-head digest: {path.with_suffix('.json')}")

            observed = count_shard(panel_rows, path, transitions[group]["refinement"])
            for method, total in totals.items():
                values = receipt["counts"][method]
                top1, top5, pool = (values[key] for key in COUNTS)
                # Independently count Top-1 from every prediction row. A
                # valid first choice is also counted in Top-5 and coverage.
                # 从每条预测独立复算 Top-1。首选成功时，Top-5 与候选池覆盖也应记为成功。
                if (any(type(value) is not int for value in (top1, top5, pool))
                        or top1 != observed[method] or not 0 <= top1 <= top5 <= pool <= len(panel_rows)):
                    raise ValueError(f"Invalid or inconsistent success counts for {method}: {path}")
                total.update({key: values[key] for key in COUNTS})
            n_rows += len(panel_rows)

        # Match the benchmark's full-coverage check. Use the number of
        # instructions as the denominator for each rate.
        # 按评测流程核验完整覆盖，三项比率均以实际指令数为分母。
        if not n_rows or n_rows != panel["rows"] or n_rows != panel["input_rows"]:
            raise ValueError(f"Incomplete benchmark coverage for {group}")
        for method, counts in totals.items():
            methods[method][group] = {
                "n": n_rows,
                "scenes": panel["scenes"],
                **counts,
                "top1": counts["top1_successes"] / n_rows,
                "top5": counts["top5_successes"] / n_rows,
                "oracle_pool": counts["oracle_successes"] / n_rows,
            }
    # Compute H from the two split rates after the shard totals are known.
    # 分片汇总后，用两个测试组的成功率计算调和平均 H。
    for values in methods.values():
        for key in ("top1", "top5"):
            seen, unseen = (values[group][key] for group in TEST_GROUPS)
            values[f"h_{key}"] = 2 * seen * unseen / (seen + unseen) if seen + unseen else 0.0
    return {
        "panel_sha256": panel_digest,
        "fit_sha256": fit_digest,
        "offline_2d": methods,
        "transitions": transitions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", required=True, type=Path, help="Directory containing the panel manifest and shards")
    parser.add_argument("--evaluation", required=True, type=Path, help="Directory containing evaluation predictions and receipts")
    args = parser.parse_args()
    print(json.dumps(audit(args.panels, args.evaluation), indent=2))


if __name__ == "__main__":
    main()
