"""Run a full-model M1 Brain-DRAM/SRAM B-stage comparison."""

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
from flowgpu.sim_v2.placement import unique_balanced  # noqa: E402
from flowgpu.sim_v2.research import (  # noqa: E402
    TaskPhase,
    build_b_stage_layout,
    execute_b_stage_layer,
    execute_hbm_fallback_plan,
)
from flowgpu.sim_v2.serving import confidence_interval  # noqa: E402


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def required_brain_nodes(geometry, brain_profile):
    usable = max(1, int(brain_profile.dram_capacity_bytes * 0.8))
    return max(1, int(
        (geometry.total_weight_bytes + usable - 1) // usable
    ))


def run_seed(args, seed, batch):
    local = argparse.Namespace(**vars(args))
    local.seed = seed
    local.requests = batch
    geometry, experts, used_layers = model_case(local)
    gpu_profile, brain_profile, bridge_profile = runtime_profiles(local)
    m1_nodes = required_brain_nodes(geometry, brain_profile)
    brain_nodes = max(args.brain_nodes, m1_nodes)
    cluster = make_case_cluster(
        brain_nodes,
        brain_profile,
        bridge_profile,
        1,
    )
    gpu = make_gpu_chips(count=16, profile=gpu_profile)
    brain = make_brain_chips(
        cluster,
        max_experts=args.c_max,
        count_per_node=8,
        profile=brain_profile,
    )
    planning = task_trace(
        max(2, args.tokens),
        (TaskPhase("m1", batch, {
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
        (TaskPhase("m1", batch, {
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
    kernels = make_kernels(experts, gpu, brain, geometry.dtype)
    dram_bw = {
        node: brain_profile.dram_bandwidth_bytes_per_s
        for node in cluster.ingress_nodes
    }
    dram_capacity = {
        node: int(brain_profile.dram_capacity_bytes * 0.8)
        for node in cluster.ingress_nodes
    }
    layout = build_b_stage_layout(
        experts,
        brain,
        demand,
        dram_bw,
        dram_capacity,
        dram_latency_s=brain_profile.dram_latency_s,
        max_hot_bytes=(
            args.hot_experts * geometry.expert_weight_bytes
        ),
        max_hot_experts=args.hot_experts,
    )
    by_layer = {}
    for assignment in evaluation.trace.assignments():
        by_layer.setdefault(assignment.layer_id, []).append(assignment)
    total_time = 0.0
    transfer_bytes = 0
    transfer_time = 0.0
    hot_fraction_sum = 0.0
    for layer_id in sorted(by_layer):
        result = execute_b_stage_layer(
            layout,
            by_layer[layer_id],
            kernels,
        )
        total_time += result.completion_s
        transfer_bytes += result.transfer_bytes
        transfer_time += result.transfer_time_s
        hot_fraction_sum += (
            result.hot_assignment_fraction * result.assignments
        )
    assignments = sum(len(items) for items in by_layer.values())
    gpu_plan = unique_balanced(experts, gpu, {
        expert.key: float(demand.global_for(expert.key))
        for expert in experts
    })
    gpu_execution = execute_hbm_fallback_plan(
        gpu_plan,
        evaluation.trace,
        cluster,
        [chip.chip_id for chip in gpu],
        kernels,
        7168,
        dispatch_mode="gpu-direct",
    )
    return [{
        "seed": seed,
        "batch": batch,
        "model": args.model,
        "brain_nodes": brain_nodes,
        "used_moe_layers": used_layers,
        "m1_completion_s": total_time,
        "gpu_only_completion_s": gpu_execution.completion_s,
        "speedup_vs_gpu": (
            gpu_execution.completion_s / total_time
        ),
        "dram_transfer_bytes": transfer_bytes,
        "dram_transfer_time_s": transfer_time,
        "hot_assignment_fraction": (
            hot_fraction_sum / assignments if assignments else 0.0
        ),
        "assignments": assignments,
    }]


def _run_job(payload):
    return run_seed(*payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="mixtral_8x7b")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seeds", default="101,102,103,104,105")
    parser.add_argument("--batches", default="1,4")
    parser.add_argument("--brain-nodes", type=int, default=1)
    parser.add_argument("--hot-experts", type=int, default=8)
    parser.add_argument("--c-max", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--zipf-exponent", type=float, default=1.3)
    parser.add_argument("--layer-limit", type=int, default=0)
    parser.add_argument("--gpu-profile", default="h100_sxm")
    parser.add_argument("--brain-profile", default="lynxi_hp300_proj")
    parser.add_argument("--bridge-fabric", default="pcie5_x16")
    parser.add_argument("--jobs", type=int, default=min(5, os.cpu_count() or 1))
    args = parser.parse_args()
    args.seeds = tuple(int(v) for v in args.seeds.split(",") if v)
    args.batches = tuple(int(v) for v in args.batches.split(",") if v)
    if not args.seeds or not args.batches:
        raise SystemExit("seeds and batches cannot be empty")
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit("output directory already exists: %s" % output)
    temp = output.parent / (output.name + ".tmp")
    if temp.exists():
        raise SystemExit("temporary output directory already exists: %s" % temp)
    temp.mkdir(parents=True)
    jobs = min(max(1, args.jobs), len(args.seeds) * len(args.batches))
    payloads = [
        (args, seed, batch)
        for seed in args.seeds
        for batch in args.batches
    ]
    if jobs == 1:
        raw = [row for payload in payloads for row in run_seed(*payload)]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            raw = [
                row
                for rows in executor.map(_run_job, payloads)
                for row in rows
            ]
    write_csv(temp / "raw.csv", raw)
    aggregate = []
    for batch in args.batches:
        values = [row for row in raw if row["batch"] == batch]
        ci = confidence_interval(
            [row["speedup_vs_gpu"] for row in values],
            lower_bound=0.0,
        )
        aggregate.append({
            "batch": batch,
            "runs": len(values),
            "m1_completion_mean_s": statistics.mean(
                row["m1_completion_s"] for row in values
            ),
            "gpu_only_completion_mean_s": statistics.mean(
                row["gpu_only_completion_s"] for row in values
            ),
            "speedup_mean": ci.mean,
            "speedup_low95": ci.low95,
            "speedup_high95": ci.high95,
            "hot_assignment_fraction_mean": statistics.mean(
                row["hot_assignment_fraction"] for row in values
            ),
            "dram_transfer_bytes_mean": statistics.mean(
                row["dram_transfer_bytes"] for row in values
            ),
        })
    write_csv(temp / "aggregate.csv", aggregate)
    summary = {
        "schema_version": 1,
        "kind": "moe-m1-full",
        "model": args.model,
        "dtype": effective_dtype(argparse.Namespace(
            model=args.model, dtype=args.dtype
        )),
        "seeds": list(args.seeds),
        "batches": list(args.batches),
        "brain_nodes_min": args.brain_nodes,
        "hot_experts": args.hot_experts,
        "c_max": args.c_max,
        "layer_limit": args.layer_limit,
        "claim_boundary": (
            "M1 full-model grouped B-stage with synthetic routing and "
            "projected hardware; no GPU fallback"
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
    (temp / "manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "kind": "moe-m1-full",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": {
                str(path.resolve()): file_hash(path)
                for path in source_paths
            },
            "claim_boundary": summary["claim_boundary"],
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp.rename(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
