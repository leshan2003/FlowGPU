"""Compare D0 shared and per-GPU owner dispatch with valid GPU replicas."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_moe_d0 import make_case_cluster, overlay_from_exclusive  # noqa: E402
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
from flowgpu.sim_v2.replicas import ReplicaPlan  # noqa: E402
from flowgpu.sim_v2.research import (  # noqa: E402
    build_h_exclusive_plan,
    execute_hbm_fallback_plan,
    TaskPhase,
)
from flowgpu.sim_v2.schema import ReplicaSpec  # noqa: E402
from flowgpu.sim_v2.serving import confidence_interval  # noqa: E402


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replicated_gpu_plan(experts, gpu):
    replicas = []
    for expert in experts:
        for index, chip in enumerate(gpu):
            replicas.append(ReplicaSpec(
                "gpu-replica-L%d-E%d-%s" % (
                    expert.key.layer_id,
                    expert.key.expert_id,
                    chip.chip_id,
                ),
                expert.key,
                chip.chip_id,
                index == 0,
            ))
    return ReplicaPlan(tuple(experts), tuple(gpu), replicas, "gpu-replicated")


def replicated_overlay_plan(exclusive_plan, experts, gpu):
    hot = {
        replica.expert
        for replica in exclusive_plan.replicas
        if exclusive_plan.chip(replica.chip_id).role.startswith("brain")
    }
    replicas = [
        replica for replica in exclusive_plan.replicas
        if replica.chip_id not in {chip.chip_id for chip in gpu}
    ]
    for expert in experts:
        for index, chip in enumerate(gpu):
            replicas.append(ReplicaSpec(
                "gpu-replica-L%d-E%d-%s" % (
                    expert.key.layer_id,
                    expert.key.expert_id,
                    chip.chip_id,
                ),
                expert.key,
                chip.chip_id,
                expert.key not in hot and index == 0,
            ))
    return ReplicaPlan(
        tuple(experts),
        tuple(gpu) + tuple(
            chip for chip in exclusive_plan.chips
            if chip.chip_id not in {item.chip_id for item in gpu}
        ),
        replicas,
        "D0-per-gpu-overlay",
    )


def activity_metrics(execution, plan):
    gpu_busy = 0.0
    brain_busy = 0.0
    for layer in execution.layers:
        for replica in layer.replica_executions:
            busy = (
                replica.compute_finish_s - replica.compute_start_s
            )
            role = plan.chip(replica.chip_id).role
            if role == "gpu":
                gpu_busy += busy
            elif role.startswith("brain"):
                brain_busy += busy
    gpu_count = sum(chip.role == "gpu" for chip in plan.chips)
    brain_count = sum(
        chip.role.startswith("brain") for chip in plan.chips
    )
    return (
        gpu_busy / (gpu_count * execution.completion_s)
        if gpu_count and execution.completion_s else 0.0,
        brain_busy / (brain_count * execution.completion_s)
        if brain_count and execution.completion_s else 0.0,
    )


def run_seed(args, seed):
    geometry, experts, used_layers = model_case(args)
    gpu_profile, brain_profile, bridge_profile = runtime_profiles(args)
    gpu = make_gpu_chips(count=args.gpus, profile=gpu_profile)
    clusters = {
        count: make_case_cluster(
            args.brain_nodes,
            brain_profile,
            bridge_profile,
            count,
        )
        for count in (1, args.gpus)
    }
    shared_cluster = clusters[1]
    planning = task_trace(
        max(2, args.tokens),
        (TaskPhase("d0-pergpu", args.batch, {
            "code": 0.5,
            "math": 0.5,
        }),),
        used_layers,
        geometry.n_experts,
        geometry.top_k,
        shared_cluster.ingress_nodes,
        seed,
        "placement",
        geometry.first_moe_layer,
        args.zipf_exponent,
    )
    evaluation = task_trace(
        max(2, args.tokens),
        (TaskPhase("d0-pergpu", args.batch, {
            "code": 0.5,
            "math": 0.5,
        }),),
        used_layers,
        geometry.n_experts,
        geometry.top_k,
        shared_cluster.ingress_nodes,
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
    rows = []
    variants = (
        ("D0-shared-minimal", "gpu-direct", 1, False),
        ("D0-shared-replicated", "gpu-direct", 1, True),
        ("D0-per-gpu", "gpu-direct", args.gpus, True),
    )
    for name, mode, gpu_count, replicated in variants:
        cluster = clusters[gpu_count]
        brain = make_brain_chips(
            cluster,
            max_experts=args.c_max,
            count_per_node=8,
            profile=brain_profile,
        )
        kernels = make_kernels(experts, gpu, brain, geometry.dtype)
        gpu_plan = (
            replicated_gpu_plan(experts, gpu)
            if replicated else
            unique_balanced(experts, gpu, expected)
        )
        exclusive = build_h_exclusive_plan(
            experts,
            gpu,
            brain,
            demand,
            max_brain_bytes=args.copies * geometry.expert_weight_bytes,
            max_brain_copies=args.copies,
        )
        plan = (
            replicated_overlay_plan(exclusive, experts, gpu)
            if replicated else
            overlay_from_exclusive(exclusive, gpu_plan)
        )
        gpu_execution = execute_hbm_fallback_plan(
            gpu_plan,
            evaluation.trace,
            cluster,
            [chip.chip_id for chip in gpu],
            kernels,
            7168,
            dispatch_mode=mode,
        )
        execution = execute_hbm_fallback_plan(
            plan,
            evaluation.trace,
            cluster,
            [chip.chip_id for chip in gpu],
            kernels,
            7168,
            dispatch_mode=mode,
            dispatch_policy="brain-first",
        )
        gpu_busy, brain_busy = activity_metrics(execution, plan)
        rows.append({
            "seed": seed,
            "gpus": args.gpus,
            "brain_nodes": args.brain_nodes,
            "batch": args.batch,
            "variant": name,
            "completion_s": execution.completion_s,
            "speedup_vs_gpu": (
                gpu_execution.completion_s / execution.completion_s
            ),
            "gpu_assignments": execution.gpu_assignments,
            "brain_assignments": execution.brain_assignments,
            "gpu_busy_fraction": gpu_busy,
            "brain_busy_fraction": brain_busy,
            "bridge_bytes": sum(
                value
                for layer in execution.layers
                for link_id, value in layer.link_bytes.items()
                if link_id.startswith("bridge:")
            ),
        })
    return rows


def _run_job(payload):
    return run_seed(*payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="mixtral_8x7b")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seeds", default="101,102,103")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--brain-nodes", type=int, default=2)
    parser.add_argument("--batches", default="1,4,16")
    parser.add_argument("--copies", type=int, default=8)
    parser.add_argument("--c-max", type=int, default=2)
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
    if args.gpus <= 0 or args.brain_nodes <= 0:
        raise SystemExit("gpus and brain-nodes must be positive")
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit("output directory already exists: %s" % output)
    temp = output.parent / (output.name + ".tmp")
    if temp.exists():
        raise SystemExit("temporary output directory already exists: %s" % temp)
    temp.mkdir(parents=True)
    raw = []
    jobs = min(max(1, args.jobs), len(args.seeds))
    if jobs == 1:
        for seed in args.seeds:
            for batch in args.batches:
                local = argparse.Namespace(**vars(args))
                local.batch = batch
                raw.extend(run_seed(local, seed))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = []
            for seed in args.seeds:
                for batch in args.batches:
                    local = argparse.Namespace(**vars(args))
                    local.batch = batch
                    futures.append((seed, batch, executor.submit(
                        _run_job, (local, seed)
                    )))
            for _, _, future in futures:
                raw.extend(future.result())
    write_csv(temp / "raw.csv", raw)
    grouped = {}
    for row in raw:
        key = (row["variant"], row["gpus"], row["brain_nodes"], row["batch"])
        grouped.setdefault(key, []).append(row)
    aggregate = []
    for key, values in sorted(grouped.items()):
        variant, gpus, brain_nodes, batch = key
        ci = confidence_interval(
            [row["speedup_vs_gpu"] for row in values],
            lower_bound=0.0,
        )
        aggregate.append({
            "variant": variant,
            "gpus": gpus,
            "brain_nodes": brain_nodes,
            "batch": batch,
            "runs": len(values),
            "speedup_mean": ci.mean,
            "speedup_low95": ci.low95,
            "speedup_high95": ci.high95,
            "gpu_busy_fraction": statistics.mean(
                row["gpu_busy_fraction"] for row in values
            ),
            "brain_busy_fraction": statistics.mean(
                row["brain_busy_fraction"] for row in values
            ),
            "bridge_bytes_mean": statistics.mean(
                row["bridge_bytes"] for row in values
            ),
        })
    write_csv(temp / "aggregate.csv", aggregate)
    summary = {
        "schema_version": 1,
        "kind": "moe-d0-per-gpu",
        "model": args.model,
        "dtype": effective_dtype(argparse.Namespace(
            model=args.model, dtype=args.dtype
        )),
        "seeds": list(args.seeds),
        "gpus": args.gpus,
        "brain_nodes": args.brain_nodes,
        "batches": list(args.batches),
        "copies": args.copies,
        "c_max": args.c_max,
        "layer_limit": args.layer_limit,
        "variant_semantics": {
            "D0-shared-minimal": (
                "one GPU ingress endpoint; one GPU copy per expert"
            ),
            "D0-shared-replicated": (
                "one GPU ingress endpoint; all experts replicated on every GPU"
            ),
            "D0-per-gpu": (
                "one ingress endpoint per GPU; request owner GPU has all experts"
            ),
        },
        "claim_boundary": (
            "D0 synthetic/projected per-GPU ownership test; Mixtral only"
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
            "kind": "moe-d0-per-gpu",
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
