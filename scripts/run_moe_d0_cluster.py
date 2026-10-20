"""Run a D0 clustering grid with controlled hot-set and copy budgets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_moe_d0 import make_case_cluster  # noqa: E402
from run_moe_stage0 import (  # noqa: E402
    effective_dtype,
    file_hash,
    make_brain_chips,
    make_gpu_chips,
    make_kernels,
    model_case,
    runtime_profiles,
    task_trace,
)
from flowgpu.sim_v2.demand import profile_demand  # noqa: E402
from flowgpu.sim_v2.placement import (  # noqa: E402
    coactivation_clustered,
    full_hyperedge_cut,
    hyperedge_clustered,
    primary_coactivation_cut,
    unique_balanced,
)
from flowgpu.sim_v2.replicas import ReplicaPlan  # noqa: E402
from flowgpu.sim_v2.research import (  # noqa: E402
    TaskPhase,
    execute_hbm_fallback_plan,
)
from flowgpu.sim_v2.schema import ReplicaSpec  # noqa: E402
from flowgpu.sim_v2.serving import confidence_interval  # noqa: E402


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def overlay_plan(gpu_plan, clustered_hot):
    replicas = list(gpu_plan.replicas)
    replicas.extend(
        ReplicaSpec(
            "brain-cluster-" + replica.replica_id,
            replica.expert,
            replica.chip_id,
            False,
        )
        for replica in clustered_hot.replicas
    )
    return ReplicaPlan(
        experts=gpu_plan.experts,
        chips=tuple(gpu_plan.chips) + tuple(clustered_hot.chips),
        replicas=replicas,
        strategy="D0-" + clustered_hot.strategy,
    )


def run_seed(args, seed):
    geometry, experts, used_layers = model_case(args)
    gpu_profile, brain_profile, bridge_profile = runtime_profiles(args)
    gpu = make_gpu_chips(count=16, profile=gpu_profile)
    rows = []
    for node_count in args.nodes:
        cluster = make_case_cluster(
            node_count,
            brain_profile,
            bridge_profile,
            1,
        )
        for request_count in args.batches:
            planning = task_trace(
                max(2, args.tokens),
                (TaskPhase("d0-cluster", request_count, {
                    "code": 0.5,
                    "math": 0.5,
                }),),
                used_layers,
                geometry.n_experts,
                geometry.top_k,
                cluster.ingress_nodes,
                seed,
                "placement",
                geometry.first_moe_layer,
                args.zipf_exponent,
            )
            evaluation = task_trace(
                max(2, args.tokens),
                (TaskPhase("d0-cluster", request_count, {
                    "code": 0.5,
                    "math": 0.5,
                }),),
                used_layers,
                geometry.n_experts,
                geometry.top_k,
                cluster.ingress_nodes,
                seed + 1,
                "evaluation",
                geometry.first_moe_layer,
                args.zipf_exponent,
            )
            demand = profile_demand(planning.trace)
            expected = {
                expert.key: float(demand.global_for(expert.key))
                for expert in experts
            }
            experts_by_layer = {}
            for expert in experts:
                experts_by_layer.setdefault(
                    expert.key.layer_id, []
                ).append(expert)
            for layer_experts in experts_by_layer.values():
                layer_experts.sort(key=lambda expert: (
                    -expected[expert.key] / expert.weight_bytes,
                    expert.key,
                ))
            ordered_experts = tuple(
                expert
                for layer_id in sorted(experts_by_layer)
                for expert in experts_by_layer[layer_id][:2]
            )
            for c_max in args.c_values:
                brain = make_brain_chips(
                    cluster,
                    max_experts=c_max,
                    count_per_node=8,
                    profile=brain_profile,
                )
                effective_copies = min(
                    args.copies,
                    len(brain) * c_max,
                    len(experts),
                )
                hot_experts = ordered_experts[:effective_copies]
                if len(hot_experts) < effective_copies:
                    selected = set(expert.key for expert in hot_experts)
                    hot_experts += tuple(
                        expert
                        for expert in sorted(
                            experts,
                            key=lambda expert: (
                                -expected[expert.key] /
                                expert.weight_bytes,
                                expert.key,
                            ),
                        )
                        if expert.key not in selected
                    )[:effective_copies - len(hot_experts)]
                gpu_plan = unique_balanced(experts, gpu, expected)
                kernels = make_kernels(
                    experts,
                    gpu,
                    brain,
                    geometry.dtype,
                )
                gpu_execution = execute_hbm_fallback_plan(
                    gpu_plan,
                    evaluation.trace,
                    cluster,
                    [chip.chip_id for chip in gpu],
                    kernels,
                    7168,
                    dispatch_mode="gpu-direct",
                )
                plans = (
                    ("unique", unique_balanced(
                        hot_experts,
                        brain,
                        {expert.key: expected[expert.key]
                         for expert in hot_experts},
                    )),
                    ("pair", coactivation_clustered(
                        hot_experts,
                        brain,
                        demand,
                    )),
                    ("full-hyperedge", hyperedge_clustered(
                        hot_experts,
                        brain,
                        demand,
                    )),
                )
                for strategy, hot_plan in plans:
                    plan = overlay_plan(gpu_plan, hot_plan)
                    execution = execute_hbm_fallback_plan(
                        plan,
                        evaluation.trace,
                        cluster,
                        [chip.chip_id for chip in gpu],
                        kernels,
                        7168,
                        dispatch_mode="gpu-direct",
                        dispatch_policy="brain-first",
                    )
                    brain_per_chip = {}
                    for replica in hot_plan.replicas:
                        brain_per_chip[replica.chip_id] = (
                            brain_per_chip.get(replica.chip_id, 0) + 1
                        )
                    bridge_bytes = sum(
                        value
                        for layer in execution.layers
                        for link_id, value in layer.link_bytes.items()
                        if link_id.startswith("bridge:")
                    )
                    rows.append({
                        "seed": seed,
                        "nodes": node_count,
                        "batch": request_count,
                        "c_max": c_max,
                        "effective_copies": effective_copies,
                        "strategy": strategy,
                        "completion_s": execution.completion_s,
                        "speedup_vs_gpu": (
                            gpu_execution.completion_s /
                            execution.completion_s
                        ),
                        "node_cut": primary_coactivation_cut(
                            hot_plan, demand, "node"
                        ),
                        "chip_cut": primary_coactivation_cut(
                            hot_plan, demand, "chip"
                        ),
                        "full_hyperedge_cut": full_hyperedge_cut(
                            hot_plan, demand
                        ),
                        "local_assignment_fraction":
                            execution.local_assignment_fraction,
                        "used_brain_chips": len(brain_per_chip),
                        "max_experts_per_brain_chip": max(
                            brain_per_chip.values(), default=0
                        ),
                        "bridge_bytes": bridge_bytes,
                    })
    return rows


def _run_seed_job(payload):
    local, seed = payload
    return run_seed(local, seed)


def run_seeds(local, seeds, jobs):
    """Run independent seeds in worker processes while preserving row order."""

    workers = min(max(1, jobs), len(seeds))
    if workers == 1:
        return [
            row
            for seed in seeds
            for row in run_seed(local, seed)
        ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        per_seed = executor.map(
            _run_seed_job,
            ((local, seed) for seed in seeds),
        )
        return [row for rows in per_seed for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="mixtral_8x7b")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seeds", default="101,102,103,104,105")
    parser.add_argument("--nodes", default="1,2,4")
    parser.add_argument("--batches", default="1,4,16")
    parser.add_argument("--c-values", default="1,2,4")
    parser.add_argument("--copies", type=int, default=24)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--zipf-exponent", type=float, default=1.3)
    parser.add_argument("--layer-limit", type=int, default=0)
    parser.add_argument("--gpu-profile", default="h100_sxm")
    parser.add_argument("--brain-profile", default="lynxi_hp300_proj")
    parser.add_argument("--bridge-fabric", default="pcie5_x16")
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(5, os.cpu_count() or 1),
        help="parallel seed workers; use 1 for serial execution",
    )
    args = parser.parse_args()
    args.seeds = tuple(
        int(value) for value in args.seeds.split(",") if value
    )
    args.nodes = tuple(
        int(value) for value in args.nodes.split(",") if value
    )
    args.batches = tuple(
        int(value) for value in args.batches.split(",") if value
    )
    args.c_values = tuple(
        int(value) for value in args.c_values.split(",") if value
    )
    if not all((
        args.seeds, args.nodes, args.batches, args.c_values,
    )):
        raise SystemExit("all experiment axes must be non-empty")
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit("output directory already exists: %s" % output)
    temp = output.parent / (output.name + ".tmp")
    if temp.exists():
        raise SystemExit("temporary output directory already exists: %s" % temp)
    temp.mkdir(parents=True)
    local = argparse.Namespace(**vars(args))
    local.requests = 2
    raw = run_seeds(local, args.seeds, args.jobs)
    write_csv(temp / "raw.csv", raw)
    grouped = {}
    for row in raw:
        key = (
            row["nodes"],
            row["batch"],
            row["c_max"],
            row["strategy"],
        )
        grouped.setdefault(key, []).append(row)
    aggregate = []
    for (nodes, batch, c_max, strategy), values in sorted(
        grouped.items()
    ):
        speeds = [row["speedup_vs_gpu"] for row in values]
        ci = confidence_interval(speeds, lower_bound=0.0)
        aggregate.append({
            "nodes": nodes,
            "batch": batch,
            "c_max": c_max,
            "strategy": strategy,
            "runs": len(values),
            "effective_copies": values[0]["effective_copies"],
            "speedup_mean": ci.mean,
            "speedup_low95": ci.low95,
            "speedup_high95": ci.high95,
            "node_cut_mean": statistics.mean(
                row["node_cut"] for row in values
            ),
            "chip_cut_mean": statistics.mean(
                row["chip_cut"] for row in values
            ),
            "local_fraction_mean": statistics.mean(
                row["local_assignment_fraction"] for row in values
            ),
            "used_brain_chips_mean": statistics.mean(
                row["used_brain_chips"] for row in values
            ),
            "max_experts_per_brain_chip_mean": statistics.mean(
                row["max_experts_per_brain_chip"] for row in values
            ),
            "bridge_bytes_mean": statistics.mean(
                row["bridge_bytes"] for row in values
            ),
        })
    write_csv(temp / "aggregate.csv", aggregate)
    contrasts = {}
    for nodes in args.nodes:
        for batch in args.batches:
            for c_max in args.c_values:
                by_strategy = {
                    row["strategy"]: row
                    for row in aggregate
                    if row["nodes"] == nodes
                    and row["batch"] == batch
                    and row["c_max"] == c_max
                }
                unique = by_strategy["unique"]
                for strategy in ("pair", "full-hyperedge"):
                    candidate = by_strategy[strategy]
                    contrasts["%d|%d|%d|%s" % (
                        nodes, batch, c_max, strategy
                    )] = (
                        unique["speedup_mean"] /
                        candidate["speedup_mean"]
                        if candidate["speedup_mean"] else 0.0
                    )
    summary = {
        "schema_version": 1,
        "kind": "moe-d0-clustering",
        "model": args.model,
        "dtype": effective_dtype(local),
        "seeds": list(args.seeds),
        "execution_jobs": args.jobs,
        "nodes": list(args.nodes),
        "batches": list(args.batches),
        "c_values": list(args.c_values),
        "copy_budget": args.copies,
        "layer_limit": args.layer_limit,
        "dispatch_mode": "D0-shared",
        "dispatch_policy": "brain-first",
        "strategies": ["unique", "pair", "full-hyperedge"],
        "controlled_boundary": (
            "same GPU plan, hot set, effective copy count, trace, node and C "
            "within a compared cell; only Brain placement changes"
        ),
        "claim_boundary": (
            "D0 synthetic/projected clustering screen; not real trace or "
            "measured-hardware evidence"
        ),
    }
    (temp / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    source_paths = [
        Path(__file__),
        SCRIPTS / "run_moe_d0.py",
        SCRIPTS / "run_moe_stage0.py",
        ROOT / "flowgpu" / "units.py",
        *sorted((ROOT / "flowgpu" / "sim_v2").glob("*.py")),
    ]
    manifest = {
        "schema_version": 1,
        "kind": "moe-d0-clustering",
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
