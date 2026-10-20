"""Run fixed-budget EPLB experiments for the D0 MoE overlay."""

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
from flowgpu.sim_v2.eplb import (  # noqa: E402
    build_eplb_plan,
    execution_load_balance,
    rebalance_brain_replicas,
)
from flowgpu.sim_v2.placement import unique_balanced  # noqa: E402
from flowgpu.sim_v2.research import (  # noqa: E402
    TaskPhase,
    build_hbm_fallback_plan,
    execute_hbm_fallback_plan,
)
from flowgpu.sim_v2.serving import confidence_interval  # noqa: E402


def write_csv(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _brain_stats(plan):
    replicas = tuple(
        replica for replica in plan.replicas
        if plan.chip(replica.chip_id).role.startswith("brain")
    )
    copies = {}
    for replica in replicas:
        copies[replica.expert] = copies.get(replica.expert, 0) + 1
    return {
        "brain_copies": len(replicas),
        "brain_unique_experts": len(copies),
        "max_copies_per_expert": max(copies.values(), default=0),
        "redundant_copies": len(replicas) - len(copies),
        "used_brain_chips": len({
            replica.chip_id for replica in replicas
        }),
    }


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
            phases = (
                TaskPhase("eplb", request_count, {
                    "code": 0.5,
                    "math": 0.5,
                }),
            )
            planning = task_trace(
                max(2, args.tokens),
                phases,
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
                phases,
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
            brain = make_brain_chips(
                cluster,
                max_experts=args.c_max,
                count_per_node=8,
                profile=brain_profile,
            )
            budget = min(
                args.copies,
                len(brain) * args.c_max,
            )
            gpu_plan = unique_balanced(experts, gpu, expected)
            coverage = build_hbm_fallback_plan(
                experts,
                gpu,
                brain,
                demand,
                "coverage-first",
                max_brain_replicas=budget,
            )
            brain_ids = tuple(chip.chip_id for chip in brain)
            plans = [
                ("coverage", coverage),
                (
                    "coverage-rebalanced",
                    rebalance_brain_replicas(
                        coverage, planning.trace, brain_ids
                    ),
                ),
            ]
            for fraction in args.redundancy_fractions:
                plans.append((
                    "eplb-%.3f" % fraction,
                    build_eplb_plan(
                        experts,
                        gpu,
                        brain,
                        planning.trace,
                        budget,
                        redundancy_fraction=fraction,
                    ),
                ))
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
            brain_nodes = tuple(chip.node_id for chip in brain)
            for strategy, plan in plans:
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
                balance = execution_load_balance(
                    execution, plan, brain_nodes
                )
                stats = _brain_stats(plan)
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
                    "c_max": args.c_max,
                    "copy_budget": budget,
                    "strategy": strategy,
                    "completion_s": execution.completion_s,
                    "speedup_vs_gpu": (
                        gpu_execution.completion_s /
                        execution.completion_s
                    ),
                    "node_lb": balance.node_lb,
                    "chip_lb": balance.chip_lb,
                    "lb_steps": balance.steps,
                    "brain_assignment_fraction": (
                        execution.brain_assignments /
                        execution.assignments
                    ),
                    "local_assignment_fraction":
                        execution.local_assignment_fraction,
                    "bridge_bytes": bridge_bytes,
                    "network_energy_j": execution.network_energy_j,
                    **stats,
                })
    return rows


def _run_seed_job(payload):
    local, seed = payload
    return run_seed(local, seed)


def run_seeds(local, seeds, jobs):
    workers = min(max(1, jobs), len(seeds))
    if workers == 1:
        return [row for seed in seeds for row in run_seed(local, seed)]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        per_seed = executor.map(
            _run_seed_job,
            ((local, seed) for seed in seeds),
        )
        return [row for rows in per_seed for row in rows]


def paired_contrasts(raw, reference):
    left = {
        (row["seed"], row["nodes"], row["batch"], row["strategy"]):
            float(row["completion_s"])
        for row in raw
    }
    families = {
        (
            row["seed"],
            row["nodes"],
            row["batch"],
            row["strategy"],
        )
        for row in raw
    }
    output = {}
    for seed, nodes, batch, strategy in sorted(families):
        base = left.get((seed, nodes, batch, reference))
        candidate = left.get((seed, nodes, batch, strategy))
        if base is None or candidate is None:
            continue
        output.setdefault((nodes, batch, strategy), []).append(
            base / candidate
        )
    return {
        "%d|%d|%s" % key: {
            "ratio": "%s completion / %s completion" % (
                reference, key[2]
            ),
            "mean": ci.mean,
            "low95": ci.low95,
            "high95": ci.high95,
        }
        for key, values in output.items()
        for ci in (confidence_interval(values, lower_bound=0.0),)
    }


def aggregate_rows(raw):
    grouped = {}
    for row in raw:
        grouped.setdefault((
            row["nodes"],
            row["batch"],
            row["c_max"],
            row["copy_budget"],
            row["strategy"],
        ), []).append(row)
    aggregate = []
    for key, values in sorted(grouped.items()):
        nodes, batch, c_max, budget, strategy = key
        ci = confidence_interval(
            [float(row["speedup_vs_gpu"]) for row in values],
            lower_bound=0.0,
        )
        aggregate.append({
            "nodes": nodes,
            "batch": batch,
            "c_max": c_max,
            "copy_budget": budget,
            "strategy": strategy,
            "runs": len(values),
            "speedup_vs_gpu_mean": ci.mean,
            "speedup_vs_gpu_low95": ci.low95,
            "speedup_vs_gpu_high95": ci.high95,
            "completion_s_mean": statistics.mean(
                float(row["completion_s"]) for row in values
            ),
            "node_lb_mean": statistics.mean(
                float(row["node_lb"]) for row in values
            ),
            "chip_lb_mean": statistics.mean(
                float(row["chip_lb"]) for row in values
            ),
            "brain_assignment_fraction_mean": statistics.mean(
                float(row["brain_assignment_fraction"]) for row in values
            ),
            "bridge_bytes_mean": statistics.mean(
                int(row["bridge_bytes"]) for row in values
            ),
            "brain_copies_mean": statistics.mean(
                int(row["brain_copies"]) for row in values
            ),
            "brain_unique_experts_mean": statistics.mean(
                int(row["brain_unique_experts"]) for row in values
            ),
            "redundant_copies_mean": statistics.mean(
                int(row["redundant_copies"]) for row in values
            ),
            "max_copies_per_expert_mean": statistics.mean(
                int(row["max_copies_per_expert"]) for row in values
            ),
        })
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="mixtral_8x7b")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seeds", default="101,102,103,104,105")
    parser.add_argument("--nodes", default="1,2,4")
    parser.add_argument("--batches", default="1,16")
    parser.add_argument("--c-max", type=int, default=2)
    parser.add_argument("--copies", type=int, default=32)
    parser.add_argument(
        "--redundancy-fractions",
        default="0.250,0.500",
        help="comma-separated Brain copy-budget fractions reserved "
             "for duplicate hot experts",
    )
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
    args.redundancy_fractions = tuple(
        float(value)
        for value in args.redundancy_fractions.split(",")
        if value
    )
    if not all((
        args.seeds,
        args.nodes,
        args.batches,
        args.redundancy_fractions,
    )):
        raise SystemExit("all experiment axes must be non-empty")
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    if any(not 0 <= value < 1 for value in args.redundancy_fractions):
        raise SystemExit("redundancy fractions must be in [0, 1)")

    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit("output directory already exists: %s" % output)
    temp = output.parent / (output.name + ".tmp")
    if temp.exists():
        raise SystemExit("temporary output directory already exists: %s" % temp)
    temp.mkdir(parents=True)
    local = argparse.Namespace(**vars(args))
    raw = run_seeds(local, args.seeds, args.jobs)
    write_csv(temp / "raw.csv", raw)
    write_csv(temp / "aggregate.csv", aggregate_rows(raw))

    summary = {
        "schema_version": 1,
        "kind": "moe-d0-eplb",
        "model": args.model,
        "dtype": effective_dtype(local),
        "seeds": list(args.seeds),
        "execution_jobs": args.jobs,
        "nodes": list(args.nodes),
        "batches": list(args.batches),
        "c_max": args.c_max,
        "copy_budget": args.copies,
        "redundancy_fractions": list(args.redundancy_fractions),
        "layer_limit": args.layer_limit,
        "dispatch_mode": "D0-shared",
        "dispatch_policy": "brain-first",
        "strategies": [
            "coverage",
            "coverage-rebalanced",
            *[
                "eplb-%.3f" % value
                for value in args.redundancy_fractions
            ],
        ],
        "controlled_boundary": (
            "within each seed/node/batch cell all strategies share the GPU "
            "plan, planning/evaluation split, effective Brain copy budget, "
            "C, kernels and evaluation trace"
        ),
        "eplb_semantics": (
            "coverage reserves 1-r of the fixed copy budget for distinct "
            "experts; the remaining r is assigned progressively to covered "
            "experts by peak-load reduction, then Brain copies are moved to "
            "reduce mean per-step node/chip max/mean load"
        ),
        "contrast_vs_coverage": paired_contrasts(raw, "coverage"),
        "contrast_vs_coverage_rebalanced": paired_contrasts(
            raw, "coverage-rebalanced"
        ),
        "claim_boundary": (
            "D0 synthetic/projected EPLB screen; not a real request-ordered "
            "router trace and not measured KA200 hardware evidence"
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
        "kind": "moe-d0-eplb",
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
