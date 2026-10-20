"""Run a controlled C=1 clustering stress test.

The Stage 0 microbenchmark can place all three clustering policies in the
same layout. This runner creates a discriminating case with fixed resources:
three nodes, four C=1 chips per node, nine experts, and three disjoint
three-expert hyperedges. Only the expert-to-community permutation changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_moe_stage0 import (  # noqa: E402
    file_hash,
    make_brain_chips,
    make_cluster,
    make_gpu_chips,
    make_kernels,
    projected_provenance,
)
from flowgpu.sim_v2.demand import profile_demand  # noqa: E402
from flowgpu.sim_v2.hardware_profiles import (  # noqa: E402
    load_fabric_profile,
    load_hardware_profile,
)
from flowgpu.sim_v2.placement import (  # noqa: E402
    coactivation_clustered,
    full_hyperedge_cut,
    hyperedge_clustered,
    primary_coactivation_cut,
    unique_balanced,
)
from flowgpu.sim_v2.research import execute_hbm_fallback_plan  # noqa: E402
from flowgpu.sim_v2.routing import RoutingTrace  # noqa: E402
from flowgpu.sim_v2.schema import (  # noqa: E402
    ExpertKey,
    ExpertSpec,
    RoutingRecord,
)


def write_csv(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plan_fingerprint(plan) -> str:
    locations = sorted(
        (str(replica.expert), replica.chip_id)
        for replica in plan.replicas
    )
    return hashlib.sha256(repr(locations).encode("utf-8")).hexdigest()


def run_case(seed: int, permutation: bool):
    provenance = projected_provenance("C=1 clustering stress")
    brain_profile = load_hardware_profile("lynxi_hp300_proj")
    bridge_profile = load_fabric_profile("pcie5_x16")
    gpu_profile = load_hardware_profile("h100_sxm")
    cluster = make_cluster(3, 4, brain_profile, bridge_profile)
    chips = make_brain_chips(
        cluster,
        max_experts=1,
        count_per_node=4,
        reserve=0.0,
        profile=brain_profile,
    )
    gpu = make_gpu_chips(profile=gpu_profile)
    expert_ids = list(range(9))
    if permutation:
        random.Random(seed).shuffle(expert_ids)
        communities = (
            tuple(expert_ids[0:3]),
            tuple(expert_ids[3:6]),
            tuple(expert_ids[6:9]),
        )
    else:
        # Interleaved communities match the three home nodes but conflict with
        # the load-balanced key order, so clustering has a real move to make.
        communities = ((0, 3, 6), (1, 4, 7), (2, 5, 8))
    experts = tuple(
        ExpertSpec(ExpertKey(0, expert_id), 100, provenance)
        for expert_id in range(9)
    )
    records = []
    for token in range(30):
        community = token % len(communities)
        records.append(RoutingRecord(
            request_id=token // 3,
            token_id=token,
            step=token,
            layer_id=0,
            home_node=cluster.ingress_nodes[community],
            experts=communities[community],
            gate_weights=(1 / 3,) * 3,
            task_family="community%d" % community,
            stage="decode",
            timestamp_s=float(token),
            window="stress",
        ))
    trace = RoutingTrace(records, provenance, "evaluation")
    demand = profile_demand(RoutingTrace(
        records,
        provenance,
        "placement",
    ))
    expected = {
        expert.key: float(demand.global_for(expert.key))
        for expert in experts
    }
    plans = (
        ("unique-balanced", unique_balanced(experts, chips, expected)),
        ("pair-clustered", coactivation_clustered(experts, chips, demand)),
        ("full-hyperedge", hyperedge_clustered(experts, chips, demand)),
    )
    kernels = make_kernels(experts, gpu, chips, "fp8")
    rows = []
    baseline_completion = None
    for strategy, plan in plans:
        execution = execute_hbm_fallback_plan(
            plan,
            trace,
            cluster,
            [chip.chip_id for chip in gpu],
            kernels,
            hidden_bytes=1,
        )
        if baseline_completion is None:
            baseline_completion = execution.completion_s
        rows.append({
            "case": "aligned" if not permutation else "permutation",
            "seed": seed,
            "strategy": strategy,
            "completion_s": execution.completion_s,
            "speedup_vs_unique": (
                baseline_completion / execution.completion_s
                if execution.completion_s > 0 else 1.0
            ),
            "node_cut": primary_coactivation_cut(plan, demand, "node"),
            "chip_cut": primary_coactivation_cut(plan, demand, "chip"),
            "full_hyperedge_cut": full_hyperedge_cut(plan, demand),
            "local_assignment_fraction": execution.local_assignment_fraction,
            "plan_fingerprint": plan_fingerprint(plan),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--permutations", type=int, default=20)
    args = parser.parse_args()
    if args.permutations <= 0:
        raise SystemExit("--permutations must be positive")
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit("output directory already exists: %s" % output)
    temp = output.parent / (output.name + ".tmp")
    if temp.exists():
        raise SystemExit("temporary output directory already exists: %s" % temp)
    temp.mkdir(parents=True)

    rows = run_case(-1, permutation=False)
    for seed in range(args.permutations):
        rows.extend(run_case(seed, permutation=True))
    write_csv(temp / "raw.csv", rows)

    aggregate = []
    for strategy in ("unique-balanced", "pair-clustered", "full-hyperedge"):
        selected = [row for row in rows if row["strategy"] == strategy]
        speeds = [row["speedup_vs_unique"] for row in selected]
        aggregate.append({
            "strategy": strategy,
            "runs": len(selected),
            "speedup_mean": statistics.mean(speeds),
            "speedup_min": min(speeds),
            "speedup_max": max(speeds),
            "completion_mean_s": statistics.mean(
                row["completion_s"] for row in selected
            ),
            "node_cut_mean": statistics.mean(
                row["node_cut"] for row in selected
            ),
            "local_fraction_mean": statistics.mean(
                row["local_assignment_fraction"]
                for row in selected
            ),
        })
    write_csv(temp / "aggregate.csv", aggregate)

    unique_rows = [
        row for row in rows
        if row["case"] == "permutation" and row["strategy"] == "unique-balanced"
    ]
    pair_rows = [
        row for row in rows
        if row["case"] == "permutation" and row["strategy"] == "pair-clustered"
    ]
    full_rows = [
        row for row in rows
        if row["case"] == "permutation" and row["strategy"] == "full-hyperedge"
    ]
    pair_identical = sum(
        left["plan_fingerprint"] == right["plan_fingerprint"]
        for left, right in zip(unique_rows, pair_rows)
    )
    full_improved = sum(
        row["speedup_vs_unique"] > 1.0 + 1e-12 for row in full_rows
    )
    full_worse = sum(
        row["speedup_vs_unique"] < 1.0 - 1e-12 for row in full_rows
    )
    aligned = {
        row["strategy"]: {
            key: row[key]
            for key in (
                "completion_s",
                "speedup_vs_unique",
                "node_cut",
                "full_hyperedge_cut",
                "local_assignment_fraction",
            )
        }
        for row in rows if row["case"] == "aligned"
    }
    full_speeds = [row["speedup_vs_unique"] for row in full_rows]
    summary = {
        "schema_version": 1,
        "kind": "moe-c1-clustering-stress",
        "controlled_variables": {
            "c_max": 1,
            "nodes": 3,
            "chips_per_node": 4,
            "experts": 9,
            "hyperedge_size": 3,
            "evaluation_tokens": 30,
            "expert_weight_bytes": 100,
            "hidden_bytes": 1,
        },
        "varied_variable": (
            "expert-to-community permutation; alignment case is the identity "
            "permutation with home nodes matching communities"
        ),
        "baseline": "unique-balanced with identical C=1 resources and trace",
        "aligned": aligned,
        "permutations": args.permutations,
        "pair_identical_to_unique": pair_identical,
        "full_improved_cases": full_improved,
        "full_worse_cases": full_worse,
        "full_speedup_mean": statistics.mean(full_speeds),
        "full_speedup_min": min(full_speeds),
        "full_speedup_max": max(full_speeds),
        "claim_boundary": (
            "controlled C=1 clustering stress with synthetic routing and "
            "projected hardware; not a full-model performance claim"
        ),
    }
    (temp / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    source_paths = [
        Path(__file__),
        SCRIPTS / "run_moe_stage0.py",
        ROOT / "flowgpu" / "units.py",
        *sorted((ROOT / "flowgpu" / "sim_v2").glob("*.py")),
    ]
    manifest = {
        "schema_version": 1,
        "kind": "moe-c1-clustering-stress",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {
            str(path.resolve()): file_hash(path)
            for path in source_paths
        },
        "claim_boundary": summary["claim_boundary"],
    }
    (temp / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp.rename(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
