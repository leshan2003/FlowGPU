"""Run controlled D0/D1 and M2/M3 MoE dispatch experiments."""

from __future__ import annotations

import argparse
import csv
import json
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
from flowgpu.sim_v2.placement import unique_balanced  # noqa: E402
from flowgpu.sim_v2.research import (  # noqa: E402
    build_h_exclusive_plan,
    execute_hbm_fallback_plan,
    TaskPhase,
)
from flowgpu.sim_v2.replicas import ReplicaPlan  # noqa: E402
from flowgpu.sim_v2.schema import ReplicaSpec  # noqa: E402
from flowgpu.sim_v2.serving import confidence_interval  # noqa: E402
from flowgpu.sim_v2.topology import (  # noqa: E402
    LinkProfile,
    build_hierarchical_cluster,
)


def make_case_cluster(
    nodes,
    brain_profile,
    bridge_profile,
    gpu_ingress_count,
):
    scale_up = LinkProfile(
        brain_profile.inter_chip_bandwidth_bytes_per_s,
        brain_profile.inter_chip_latency_s,
        0.85,
        2e-12,
        brain_profile.provenance,
    )
    scale_out = LinkProfile(
        brain_profile.inter_board_bandwidth_bytes_per_s,
        brain_profile.inter_board_latency_s,
        0.85,
        5e-12,
        brain_profile.provenance,
    )
    bridge = LinkProfile(
        bridge_profile.bandwidth_bytes_per_s,
        bridge_profile.latency_s,
        bridge_profile.efficiency,
        bridge_profile.energy_per_byte,
        bridge_profile.provenance,
    )
    return build_hierarchical_cluster(
        n_nodes=nodes,
        chips_per_node=8,
        ingress_node_indices=range(nodes),
        scale_up=scale_up,
        scale_out=scale_out,
        bridge=bridge,
        gpu_ingress_count=gpu_ingress_count,
    )


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def overlay_from_exclusive(exclusive_plan, gpu_plan):
    """Keep M2 hot locations and add non-primary HBM copies for hot experts."""

    hot_experts = {
        replica.expert
        for replica in exclusive_plan.replicas
        if exclusive_plan.chip(replica.chip_id).role.startswith("brain")
    }
    replicas = list(exclusive_plan.replicas)
    for replica in gpu_plan.replicas:
        if replica.expert not in hot_experts:
            continue
        replicas.append(ReplicaSpec(
            "gpu-overlay-" + replica.replica_id,
            replica.expert,
            replica.chip_id,
            False,
        ))
    return ReplicaPlan(
        experts=exclusive_plan.experts,
        chips=exclusive_plan.chips,
        replicas=replicas,
        strategy="M3-overlay-same-locations",
    )


def run_seed(args, seed):
    local = argparse.Namespace(**vars(args))
    local.seed = seed
    geometry, experts, used_layers = model_case(local)
    gpu_profile, brain_profile, bridge_profile = runtime_profiles(local)
    rows = []
    variants = (
        ("D1-shared", "gpu-to-home", 1),
        ("D0-shared", "gpu-direct", 1),
    )
    for node_count in args.nodes:
        cluster_by_gpu_count = {
            1: make_case_cluster(
                node_count,
                brain_profile,
                bridge_profile,
                1,
            )
        }
        gpu = make_gpu_chips(count=16, profile=gpu_profile)
        for request_count in args.batches:
            planning = task_trace(
                max(2, args.tokens),
                (TaskPhase("d0", request_count, {
                    "code": 0.5,
                    "math": 0.5,
                }),),
                used_layers,
                geometry.n_experts,
                geometry.top_k,
                cluster_by_gpu_count[1].ingress_nodes,
                seed,
                "placement",
                geometry.first_moe_layer,
                args.zipf_exponent,
            )
            evaluation = task_trace(
                max(2, args.tokens),
                (TaskPhase("d0", request_count, {
                    "code": 0.5,
                    "math": 0.5,
                }),),
                used_layers,
                geometry.n_experts,
                geometry.top_k,
                cluster_by_gpu_count[1].ingress_nodes,
                seed + 1,
                "evaluation",
                geometry.first_moe_layer,
                args.zipf_exponent,
            )
            demand = profile_demand(planning.trace)
            for variant, dispatch_mode, gpu_count in variants:
                cluster = cluster_by_gpu_count[gpu_count]
                brain = make_brain_chips(
                    cluster,
                    max_experts=args.c_max,
                    count_per_node=8,
                    profile=brain_profile,
                )
                kernels = make_kernels(
                    experts,
                    gpu,
                    brain,
                    geometry.dtype,
                )
                expected = {
                    expert.key: float(demand.global_for(expert.key))
                    for expert in experts
                }
                gpu_only_plan = unique_balanced(experts, gpu, expected)
                m2_plan = build_h_exclusive_plan(
                        experts,
                        gpu,
                        brain,
                        demand,
                        max_brain_bytes=(
                            args.copies * geometry.expert_weight_bytes
                        ),
                        max_brain_copies=args.copies,
                    )
                overlay_plan = overlay_from_exclusive(
                    m2_plan,
                    gpu_only_plan,
                )
                plan_runs = (
                    ("GPU-only", gpu_only_plan, "earliest_finish"),
                    (
                        "M2-tier-exclusive",
                        m2_plan,
                        "earliest_finish",
                    ),
                    ("M3-overlay", overlay_plan, "earliest_finish"),
                    ("M3-overlay", overlay_plan, "least_loaded"),
                    ("M3-overlay", overlay_plan, "nearest"),
                    ("M3-overlay", overlay_plan, "brain-first"),
                    ("M3-overlay", overlay_plan, "split"),
                )
                execution_by_plan = {}
                for name, plan, dispatch_policy in plan_runs:
                    execution_by_plan[(name, dispatch_policy)] = (
                        execute_hbm_fallback_plan(
                            plan,
                            evaluation.trace,
                            cluster,
                            [chip.chip_id for chip in gpu],
                            kernels,
                            7168,
                            dispatch_mode=dispatch_mode,
                            dispatch_policy=dispatch_policy,
                        )
                    )
                gpu_only = execution_by_plan[
                    ("GPU-only", "earliest_finish")
                ]
                m2 = execution_by_plan[
                    ("M2-tier-exclusive", "earliest_finish")
                ]
                for name, plan, dispatch_policy in plan_runs:
                    execution = execution_by_plan[(name, dispatch_policy)]
                    gpu_busy = 0.0
                    brain_busy = 0.0
                    for layer in execution.layers:
                        for replica_execution in layer.replica_executions:
                            role = plan.chip(
                                replica_execution.chip_id
                            ).role
                            busy = (
                                replica_execution.compute_finish_s -
                                replica_execution.compute_start_s
                            )
                            if role == "gpu":
                                gpu_busy += busy
                            elif role.startswith("brain"):
                                brain_busy += busy
                    gpu_count = sum(
                        chip.role == "gpu" for chip in plan.chips
                    )
                    brain_count = sum(
                        chip.role.startswith("brain")
                        for chip in plan.chips
                    )
                    bridge_bytes = sum(
                        nbytes
                        for layer in execution.layers
                        for link_id, nbytes in layer.link_bytes.items()
                        if link_id.startswith("bridge:")
                    )
                    scaleout_bytes = sum(
                        nbytes
                        for layer in execution.layers
                        for link_id, nbytes in layer.link_bytes.items()
                        if link_id.startswith("scaleout:")
                    )
                    assignments = (
                        execution.gpu_assignments +
                        execution.brain_assignments
                    )
                    rows.append({
                        "seed": seed,
                        "nodes": node_count,
                        "batch": request_count,
                        "variant": variant,
                        "dispatch_mode": dispatch_mode,
                        "strategy": name,
                        "dispatch_policy": dispatch_policy,
                        "completion_s": execution.completion_s,
                        "speedup_vs_gpu": (
                            gpu_only.completion_s / execution.completion_s
                        ),
                        "m3_speedup_vs_m2": (
                            m2.completion_s / execution.completion_s
                        ),
                        "gpu_assignments": execution.gpu_assignments,
                        "brain_assignments": execution.brain_assignments,
                        "brain_replicas": execution.brain_replicas,
                        "local_assignment_fraction":
                            execution.local_assignment_fraction,
                        "gpu_assignment_fraction": (
                            execution.gpu_assignments / assignments
                            if assignments else 0.0
                        ),
                        "gpu_busy_fraction": (
                            gpu_busy / (gpu_count * execution.completion_s)
                            if gpu_count and execution.completion_s > 0
                            else 0.0
                        ),
                        "brain_busy_fraction": (
                            brain_busy /
                            (brain_count * execution.completion_s)
                            if brain_count and execution.completion_s > 0
                            else 0.0
                        ),
                        "bridge_bytes": bridge_bytes,
                        "scaleout_bytes": scaleout_bytes,
                    })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="mixtral_8x7b")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--c-max", type=int, default=2)
    parser.add_argument("--copies", type=int, default=8)
    parser.add_argument("--seeds", default="101,102,103,104,105")
    parser.add_argument("--batches", default="1,2,4")
    parser.add_argument("--nodes", default="1,2,4")
    parser.add_argument("--zipf-exponent", type=float, default=1.3)
    parser.add_argument("--layer-limit", type=int, default=0)
    parser.add_argument("--gpu-profile", default="h100_sxm")
    parser.add_argument("--brain-profile", default="lynxi_hp300_proj")
    parser.add_argument("--bridge-fabric", default="pcie5_x16")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit("output directory already exists: %s" % output)
    temp = output.parent / (output.name + ".tmp")
    if temp.exists():
        raise SystemExit("temporary output directory already exists: %s" % temp)
    temp.mkdir(parents=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    batches = tuple(int(value) for value in args.batches.split(",") if value)
    nodes = tuple(int(value) for value in args.nodes.split(",") if value)
    if not seeds:
        raise SystemExit("at least one seed is required")
    if not batches:
        raise SystemExit("at least one batch is required")
    if not nodes:
        raise SystemExit("at least one node count is required")
    local = argparse.Namespace(**vars(args))
    local.requests = 2
    local.hot_experts = args.copies
    local.replicas = args.copies
    local.batches = batches
    local.nodes = nodes
    raw = []
    for seed in seeds:
        raw.extend(run_seed(local, seed))
    write_csv(temp / "raw.csv", raw)
    grouped = {}
    for row in raw:
        key = (
            row["variant"],
            row["nodes"],
            row["batch"],
            row["strategy"],
            row["dispatch_policy"],
        )
        grouped.setdefault(key, []).append(row)
    aggregate = []
    for (variant, node_count, batch, strategy, dispatch_policy), values in sorted(
        grouped.items()
    ):
        speedups = [row["speedup_vs_gpu"] for row in values]
        ci = confidence_interval(speedups, lower_bound=0.0)
        aggregate.append({
            "variant": variant,
            "nodes": node_count,
            "batch": batch,
            "strategy": strategy,
            "dispatch_policy": dispatch_policy,
            "runs": len(values),
            "speedup_mean": ci.mean,
            "speedup_low95": ci.low95,
            "speedup_high95": ci.high95,
            "gpu_assignment_fraction": statistics.mean(
                row["gpu_assignment_fraction"] for row in values
            ),
            "gpu_busy_fraction": statistics.mean(
                row["gpu_busy_fraction"] for row in values
            ),
            "brain_busy_fraction": statistics.mean(
                row["brain_busy_fraction"] for row in values
            ),
            "bridge_bytes_mean": statistics.mean(
                row["bridge_bytes"] for row in values
            ),
            "scaleout_bytes_mean": statistics.mean(
                row["scaleout_bytes"] for row in values
            ),
        })
    write_csv(temp / "aggregate.csv", aggregate)
    d0_d1_contrasts = {}
    for variant, dispatch_policy in sorted({
        (row["variant"], row["dispatch_policy"])
        for row in raw
        if row["strategy"] == "M3-overlay"
    }):
        if variant != "D0-shared":
            continue
        for strategy in ("M2-tier-exclusive", "M3-overlay"):
            d1 = {
                (row["seed"], row["nodes"], row["batch"]):
                    row["completion_s"]
                for row in raw
                if row["variant"] == "D1-shared"
                and row["strategy"] == strategy
                and row["dispatch_policy"] == dispatch_policy
            }
            d0 = {
                (row["seed"], row["nodes"], row["batch"]):
                    row["completion_s"]
                for row in raw
                if row["variant"] == "D0-shared"
                and row["strategy"] == strategy
                and row["dispatch_policy"] == dispatch_policy
            }
            values = [d1[key] / d0[key] for key in d0 if key in d1]
            if values:
                ci = confidence_interval(values, lower_bound=0.0)
                d0_d1_contrasts["%s|%s" % (
                    strategy, dispatch_policy
                )] = {
                    "ratio": "D1 completion / D0 completion",
                    "mean": ci.mean,
                    "low95": ci.low95,
                    "high95": ci.high95,
                }
    m3_m2_contrasts = {}
    for variant in ("D0-shared", "D1-shared"):
        for dispatch_policy in (
            "brain-first",
            "earliest_finish",
            "least_loaded",
            "nearest",
            "split",
        ):
            m2 = {
                (row["seed"], row["nodes"], row["batch"]):
                    row["completion_s"]
                for row in raw
                if row["variant"] == variant
                and row["strategy"] == "M2-tier-exclusive"
            }
            m3 = {
                (row["seed"], row["nodes"], row["batch"]):
                    row["completion_s"]
                for row in raw
                if row["variant"] == variant
                and row["strategy"] == "M3-overlay"
                and row["dispatch_policy"] == dispatch_policy
            }
            values = [m2[key] / m3[key] for key in m3 if key in m2]
            if values:
                ci = confidence_interval(values, lower_bound=0.0)
                m3_m2_contrasts["%s|%s" % (
                    variant, dispatch_policy
                )] = {
                    "ratio": "M2 completion / M3 completion",
                    "mean": ci.mean,
                    "low95": ci.low95,
                    "high95": ci.high95,
                }
    summary = {
        "schema_version": 1,
        "kind": "moe-d0-d1-dispatch",
        "model": args.model,
        "dtype": effective_dtype(local),
        "seeds": list(seeds),
        "tokens": args.tokens,
        "c_max": args.c_max,
        "copies": args.copies,
        "zipf_exponent": args.zipf_exponent,
        "layer_limit": args.layer_limit,
        "nodes": list(nodes),
        "batches": list(batches),
        "variants": ["D1-shared", "D0-shared"],
        "not_included": (
            "D0-per-gpu requires request-to-GPU placement and full/partitioned "
            "expert replicas; the shared-endpoint experiment cannot represent it"
        ),
        "policies": [
            "earliest_finish",
            "least_loaded",
            "nearest",
            "brain-first",
            "split",
        ],
        "d0_vs_d1": d0_d1_contrasts,
        "m3_vs_m2": m3_m2_contrasts,
        "claim_boundary": (
            "D0/D1 static event-model comparison with synthetic routing and "
            "projected hardware; no multicast or local-reduce claim"
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
        "kind": "moe-d0-d1-dispatch",
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
