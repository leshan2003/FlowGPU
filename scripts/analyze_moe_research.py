"""Analyze and plot a run_moe_research result bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row, name: str) -> float:
    return float(row[name])


def grouped_means(
    rows: Iterable[dict],
    group_fields: Tuple[str, ...],
    value: str,
) -> Dict[tuple, float]:
    values = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        values[key].append(as_float(row, value))
    return {key: mean(items) for key, items in values.items()}


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def analyze(bundle: Path) -> dict:
    static = read_csv(bundle / "static_capacity.csv")
    community = read_csv(bundle / "community.csv")
    node_batch = read_csv(bundle / "node_batch.csv")
    skew = read_csv(bundle / "skew.csv")
    ranked_full = read_csv(bundle / "ranked_full.csv")
    refresh = read_csv(bundle / "refresh.csv")
    seeds = read_csv(bundle / "seeds.csv")
    seed_ci = read_csv(bundle / "seed_ci.csv")
    sensitivity = read_csv(bundle / "sensitivity.csv")

    summary = {
        "row_counts": {
            "static_capacity": len(static),
            "community": len(community),
            "node_batch": len(node_batch),
            "skew": len(skew),
            "ranked_full": len(ranked_full),
            "refresh": len(refresh),
            "seeds": len(seeds),
            "sensitivity": len(sensitivity),
        },
        "static": {},
        "community": {},
        "node_batch": {},
        "skew": {},
        "ranked_full": {},
        "refresh": {},
        "seed_ci": {},
        "sensitivity": {},
    }

    if static:
        policy_rows = defaultdict(list)
        for row in static:
            if row["suite"] == "capacity" and row["policy"] != "unique-balanced":
                policy_rows[row["policy"]].append(row)
        policy_summary = {}
        for policy, rows in policy_rows.items():
            speeds = [as_float(row, "speedup_vs_gpu") for row in rows]
            policy_summary[policy] = {
                "runs": len(rows),
                "mean_speedup": mean(speeds),
                "min_speedup": min(speeds),
                "max_speedup": max(speeds),
            }
        summary["static"]["policies"] = policy_summary

        paired = defaultdict(dict)
        wins = defaultdict(int)
        for row in static:
            if row["suite"] != "capacity":
                continue
            key = (row["nodes"], row["capacity_experts_per_chip"])
            paired[key][row["policy"]] = as_float(row, "speedup_vs_gpu")
        for key, values in paired.items():
            if values:
                best = max(values.values())
                winners = [name for name, value in values.items() if value == best]
                if len(winners) == 1:
                    wins[winners[0]] += 1
                else:
                    wins["tie"] += 1
        summary["static"]["paired_wins"] = dict(wins)

        fixed_capacity = 4
        plt.figure(figsize=(7, 4.5))
        for policy in (
            "coverage-first", "replica-first", "joint-marginal",
            "ranked-0.25",
        ):
            rows = sorted(
                (
                    row for row in static
                    if row["suite"] == "capacity" and
                    row["policy"] == policy and
                    int(row["capacity_experts_per_chip"]) == fixed_capacity
                ),
                key=lambda row: int(row["nodes"]),
            )
            if rows:
                plt.plot(
                    [int(row["nodes"]) for row in rows],
                    [as_float(row, "speedup_vs_gpu") for row in rows],
                    marker="o",
                    label=policy,
                )
        plt.xlabel("brain nodes")
        plt.ylabel("speedup vs GPU-only")
        plt.title("Capacity frontier (C=4)")
        plt.legend()
        savefig(bundle / "static_capacity.png")

    if community:
        grouped = defaultdict(list)
        for row in community:
            scenario = row["scenario"]
            cluster = "cluster" if "cluster=2.0" in scenario else "no-cluster"
            coact = "coact-high" if "coact=2.0" in scenario else "coact-low"
            grouped[(coact, cluster)].append(
                as_float(row, "speedup_vs_gpu")
            )
        summary["community"]["groups"] = {
            "%s/%s" % key: mean(values)
            for key, values in grouped.items()
        }
        labels = sorted(summary["community"]["groups"])
        plt.figure(figsize=(8, 4.5))
        plt.bar(
            labels,
            [summary["community"]["groups"][label] for label in labels],
        )
        plt.ylabel("speedup vs GPU-only")
        plt.title("Controlled co-activation and clustering")
        plt.xticks(rotation=20, ha="right")
        savefig(bundle / "community.png")

    if node_batch:
        subset = [
            row for row in node_batch
            if row["policy"] == "coverage-first"
        ]
        grouped = defaultdict(list)
        for row in subset:
            batch = int(row["scenario"].split("=")[1].split(",")[0])
            grouped[(int(row["nodes"]), batch)].append(
                as_float(row, "speedup_vs_gpu")
            )
        plt.figure(figsize=(7, 4.5))
        for nodes in sorted({key[0] for key in grouped}):
            points = sorted(
                (batch, mean(values))
                for (node_count, batch), values in grouped.items()
                if node_count == nodes
            )
            if not points:
                continue
            plt.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                label="nodes=%d" % nodes,
            )
        plt.xscale("log", base=2)
        plt.xlabel("evaluation batch")
        plt.ylabel("speedup vs GPU-only")
        plt.title("Coverage policy: node and batch sensitivity")
        plt.legend()
        savefig(bundle / "node_batch.png")
        summary["node_batch"]["coverage_by_nodes_batch"] = {
            "%d/%d" % key: mean(values)
            for key, values in grouped.items()
        }

    if skew:
        grouped = defaultdict(list)
        for row in skew:
            exponent = float(row["scenario"].split("=")[1])
            grouped[(exponent, row["policy"])].append(
                as_float(row, "speedup_vs_gpu")
            )
        plt.figure(figsize=(7, 4.5))
        for policy in sorted({key[1] for key in grouped}):
            points = sorted(
                (exponent, mean(values))
                for (exponent, policy_name), values in grouped.items()
                if policy_name == policy
            )
            if points:
                plt.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o",
                    label=policy,
                )
        summary["skew"]["by_policy"] = {
            "%s/exponent=%s" % key: mean(values)
            for key, values in grouped.items()
        }
        plt.xlabel("Zipf exponent")
        plt.ylabel("speedup vs GPU-only")
        plt.title("Coverage versus replication under routing skew")
        plt.legend()
        savefig(bundle / "skew.png")

    if ranked_full:
        grouped = defaultdict(list)
        for row in ranked_full:
            grouped[(
                row["scenario"],
                int(row["nodes"]),
                int(row["capacity_experts_per_chip"]),
            )].append(as_float(row, "speedup_vs_gpu"))
        summary["ranked_full"] = {
            "%s/nodes=%d/C=%d" % key: mean(values)
            for key, values in grouped.items()
        }

    if refresh:
        grouped = defaultdict(list)
        for row in refresh:
            grouped[row["policy"]].append((
                int(row["epoch"]), as_float(row, "cumulative_time_s")
            ))
        plt.figure(figsize=(7, 4.5))
        for policy, values in sorted(grouped.items()):
            points = sorted(values)
            plt.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                label=policy,
            )
        summary["refresh"]["final"] = {
            policy: sorted(values)[-1][1]
            for policy, values in grouped.items()
        }
        plt.xlabel("epoch")
        plt.ylabel("cumulative projected time (s)")
        plt.title("Dynamic local refresh comparison")
        plt.legend(fontsize=8)
        savefig(bundle / "refresh.png")

    if seed_ci:
        summary["seed_ci"] = seed_ci
    if sensitivity:
        grouped = defaultdict(list)
        for row in sensitivity:
            key = (
                row["policy"],
                float(row["bridge_scale"]),
                float(row["brain_latency_scale"]),
            )
            grouped[key].append(as_float(row, "speedup_vs_gpu"))
        summary["sensitivity"] = {
            "%s/bridge=%.2f/brain=%.2f" % key: mean(values)
            for key, values in grouped.items()
        }

    (bundle / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    args = parser.parse_args()
    bundle = Path(args.bundle).resolve()
    summary = analyze(bundle)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
