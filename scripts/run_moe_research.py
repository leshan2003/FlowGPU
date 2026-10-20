"""Run the frozen MoE research matrix on the v2 detailed simulator.

This runner is intentionally independent of the strict production YAML schema.
It uses explicit provenance and writes a self-describing result bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowgpu.sim_v2.demand import DemandProfile, profile_demand
from flowgpu.sim_v2.kernel_model import ExpertKernelCurve, KernelSample
from flowgpu.sim_v2.research import (
    PlanExecution,
    TaskAwareTrace,
    build_hbm_fallback_plan,
    execute_hbm_fallback_plan,
    generate_task_community_trace,
    hybrid_replica_plan,
    migration_bytes,
    plan_replica_locations,
)
from flowgpu.sim_v2.routing import RoutingTrace
from flowgpu.sim_v2.serving import confidence_interval
from flowgpu.sim_v2.schema import (
    ChipSpec,
    Confidence,
    ExpertKey,
    ExpertSpec,
    Provenance,
)
from flowgpu.sim_v2.topology import (
    ClusterTopology,
    LinkProfile,
    build_hierarchical_cluster,
)


EXPERT_BYTES = 3 * 7168 * 2048
GPU_COUNT = 16
GPU_BYTES = 80_000_000_000
GPU_RESERVE = 0.20
CHIPS_PER_NODE = 8


def provenance() -> Provenance:
    return Provenance(
        Confidence.DERIVED,
        "DeepSeek-V3 public geometry; FP8 expert weight size",
    )


def links(
    fast: bool = False,
    bridge_scale: float = 1.0,
) -> Tuple[LinkProfile, LinkProfile, LinkProfile]:
    p = provenance()
    scale_up = LinkProfile(
        200e9 if fast else 90e9,
        4e-7,
        0.85,
        2e-12,
        p,
    )
    scale_out = LinkProfile(
        100e9 if fast else 39e9,
        3e-6 if fast else 7e-6,
        0.85,
        5e-12,
        p,
    )
    bridge = LinkProfile(
        51.2e9 * bridge_scale,
        4e-7,
        0.80,
        8e-12,
        p,
    )
    return scale_up, scale_out, bridge


def make_cluster(
    nodes: int,
    chips_per_node: int,
    fast: bool = False,
    concentrated_ingress: int = 0,
    bridge_scale: float = 1.0,
) -> ClusterTopology:
    scale_up, scale_out, bridge = links(fast, bridge_scale)
    ingress = (
        range(min(nodes, concentrated_ingress))
        if concentrated_ingress else range(nodes)
    )
    return build_hierarchical_cluster(
        n_nodes=nodes,
        chips_per_node=chips_per_node,
        ingress_node_indices=ingress,
        scale_up=scale_up,
        scale_out=scale_out,
        bridge=bridge,
    )


def make_chips(
    cluster: ClusterTopology,
    capacity: int,
    reserve: float,
    count_per_node: int,
) -> Tuple[ChipSpec, ...]:
    p = provenance()
    chips = []
    node_ids = sorted({endpoint.node_id for endpoint in
                       cluster.topology.endpoints.values()
                       if endpoint.kind == "node_switch"})
    for node_id in node_ids:
        for chip_index in range(count_per_node):
            chip_id = "%s:chip%d" % (node_id, chip_index)
            chips.append(ChipSpec(
                chip_id=chip_id,
                node_id=node_id,
                sram_capacity_bytes=capacity,
                runtime_reserve_fraction=reserve,
                fixed_reserved_bytes=0,
                provenance=p,
            ))
    return tuple(chips)


def make_gpu_chips() -> Tuple[ChipSpec, ...]:
    p = provenance()
    return tuple(
        ChipSpec(
            chip_id="gpu%d" % index,
            node_id="gpu-tier",
            sram_capacity_bytes=GPU_BYTES,
            runtime_reserve_fraction=GPU_RESERVE,
            fixed_reserved_bytes=0,
            provenance=p,
        )
        for index in range(GPU_COUNT)
    )


def make_experts(n_layers: int, n_experts: int) -> Tuple[ExpertSpec, ...]:
    p = provenance()
    return tuple(
        ExpertSpec(
            ExpertKey(layer, expert),
            EXPERT_BYTES,
            p,
        )
        for layer in range(n_layers)
        for expert in range(n_experts)
    )


def make_kernels(
    experts: Iterable[ExpertSpec],
    gpu_chips: Iterable[ChipSpec],
    brain_chips: Iterable[ChipSpec],
    brain_scale: float = 1.0,
) -> Mapping:
    p = Provenance(
        Confidence.PROJECTED,
        "hardware-relative synthetic expert curves; not KA200 measured",
    )
    m_values = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    gpu_us = (20, 21, 23, 30, 40, 55, 80, 150, 300, 600, 1200)
    brain_us = (
        8, 8.5, 9, 11, 14, 20, 30, 50, 85, 160, 320
    )
    gpu_curve = ExpertKernelCurve(
        "deepseek-v3-gpu",
        "fp8",
        1,
        "hbm",
        [
            KernelSample(m, value * 1e-6)
            for m, value in zip(m_values, gpu_us)
        ],
        p,
    )
    brain_curve = ExpertKernelCurve(
        "deepseek-v3-brain",
        "fp8",
        1,
        "sram",
        [
            KernelSample(m, value * 1e-6 * brain_scale)
            for m, value in zip(m_values, brain_us)
        ],
        p,
    )
    models = {}
    for expert in experts:
        for chip in gpu_chips:
            models[(expert.key, chip.chip_id)] = gpu_curve
        for chip in brain_chips:
            models[(expert.key, chip.chip_id)] = brain_curve
    return models


def normalized_l1(
    previous: Mapping[ExpertKey, int],
    current: Mapping[ExpertKey, int],
) -> float:
    keys = set(previous) | set(current)
    p_total = sum(previous.values())
    c_total = sum(current.values())
    if not keys or p_total == 0 or c_total == 0:
        return 0.0
    return 0.5 * sum(
        abs(previous.get(key, 0) / p_total -
            current.get(key, 0) / c_total)
        for key in keys
    )


def demand_for_plan(trace: TaskAwareTrace) -> DemandProfile:
    return profile_demand(trace.trace)


def expected(demand: DemandProfile, experts: Iterable[ExpertSpec]):
    return {
        expert.key: float(demand.global_for(expert.key))
        for expert in experts
    }


def gpu_only_plan(
    experts: Iterable[ExpertSpec],
    gpu_chips: Iterable[ChipSpec],
    demand: DemandProfile,
):
    from flowgpu.sim_v2.placement import unique_balanced

    experts = tuple(experts)
    plan = unique_balanced(
        experts, tuple(gpu_chips), expected(demand, experts)
    )
    return plan


@dataclass(frozen=True)
class StaticRow:
    suite: str
    scenario: str
    policy: str
    nodes: int
    chips_per_node: int
    capacity_experts_per_chip: int
    brain_capacity_ratio: float
    brain_replicas: int
    brain_unique_experts: int
    completion_s: float
    tokens_per_s: float
    tokens_per_s_per_node: float
    gpu_assignment_fraction: float
    local_assignment_fraction: float
    network_energy_j: float
    speedup_vs_gpu: float


@dataclass(frozen=True)
class RefreshRow:
    policy: str
    epoch: int
    task_mix: str
    drift_l1: float
    triggered: bool
    adopted: bool
    epoch_steps: int
    migration_copies: int
    migration_bytes: int
    migration_time_s: float
    current_completion_s: float
    candidate_completion_s: float
    selected_completion_s: float
    projected_epoch_time_s: float
    cumulative_time_s: float
    brain_replicas: int
    brain_unique_experts: int


@dataclass(frozen=True)
class SeedRow:
    suite: str
    placement_seed: int
    evaluation_seed: int
    policy: str
    nodes: int
    capacity_experts_per_chip: int
    completion_s: float
    gpu_completion_s: float
    speedup_vs_gpu: float
    local_assignment_fraction: float


@dataclass(frozen=True)
class SeedCIRow:
    suite: str
    policy: str
    nodes: int
    capacity_experts_per_chip: int
    runs: int
    speedup_mean: float
    speedup_low95: float
    speedup_high95: float
    conclusion: str


@dataclass(frozen=True)
class SensitivityRow:
    suite: str
    policy: str
    bridge_scale: float
    brain_latency_scale: float
    completion_s: float
    speedup_vs_gpu: float
    gpu_assignment_fraction: float
    local_assignment_fraction: float


def _execution_row(
    suite: str,
    scenario: str,
    nodes: int,
    capacity: int,
    total_expert_bytes: int,
    execution: PlanExecution,
    gpu_time: float,
) -> StaticRow:
    brain_bytes = execution.brain_replicas * EXPERT_BYTES
    return StaticRow(
        suite=suite,
        scenario=scenario,
        policy=execution.strategy,
        nodes=nodes,
        chips_per_node=CHIPS_PER_NODE,
        capacity_experts_per_chip=capacity,
        brain_capacity_ratio=brain_bytes / total_expert_bytes,
        brain_replicas=execution.brain_replicas,
        brain_unique_experts=execution.brain_unique_experts,
        completion_s=execution.completion_s,
        tokens_per_s=execution.tokens_per_s,
        tokens_per_s_per_node=execution.tokens_per_s / max(1, nodes),
        gpu_assignment_fraction=(
            execution.gpu_assignments / max(1, execution.assignments)
        ),
        local_assignment_fraction=execution.local_assignment_fraction,
        network_energy_j=execution.network_energy_j,
        speedup_vs_gpu=gpu_time / execution.completion_s,
    )


def run_static_suite(
    quick: bool,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[list, list]:
    n_layers, n_experts, top_k = (8, 64, 4) if quick else (58, 256, 8)
    experts = make_experts(n_layers, n_experts)
    gpu = make_gpu_chips()
    rows = []
    details = []
    # The detailed path model is intentionally capped at 512 brain chips for
    # this study. Larger topologies entered a multi-GB, hours-long regime in
    # the current Python flow scheduler and are recorded as a limitation.
    node_grid = (2, 4, 8, 16) if quick else (4, 8, 16, 32, 64)
    policies = (
        "coverage-first", "replica-first", "joint-marginal",
    )
    placement_tokens = 16 if quick else 8
    evaluation_tokens = placement_tokens
    for nodes in node_grid:
        print("static: nodes=%d" % nodes, flush=True)
        cluster = make_cluster(nodes, CHIPS_PER_NODE)
        brain = make_chips(
            cluster,
            capacity=max(1, EXPERT_BYTES) * 14,
            reserve=0.0,
            count_per_node=CHIPS_PER_NODE,
        )
        kernels = make_kernels(experts, gpu, brain)
        placement = generate_task_community_trace(
            n_tokens=placement_tokens,
            n_layers=n_layers,
            n_experts=n_experts,
            top_k=top_k,
            home_nodes=cluster.ingress_nodes,
            task_weights={"code": 0.6, "math": 0.4},
            seed=101,
            provenance=provenance(),
            split="placement",
            community_count=8,
            community_strength=1.0,
            node_affinity_strength=0.30,
        )
        evaluation = generate_task_community_trace(
            n_tokens=evaluation_tokens,
            n_layers=n_layers,
            n_experts=n_experts,
            top_k=top_k,
            home_nodes=cluster.ingress_nodes,
            task_weights={"code": 0.6, "math": 0.4},
            seed=211,
            provenance=provenance(),
            split="evaluation",
            community_count=8,
            community_strength=1.0,
            node_affinity_strength=0.30,
        )
        demand = demand_for_plan(placement)
        base = gpu_only_plan(experts, gpu, demand)
        base_exec = execute_hbm_fallback_plan(
            base,
            evaluation.trace,
            cluster,
            [chip.chip_id for chip in gpu],
            kernels,
            7168,
        )
        rows.append(_execution_row(
            "capacity", "gpu-only", nodes, 0,
            len(experts) * EXPERT_BYTES, base_exec,
            base_exec.completion_s,
        ))
        for capacity in (1, 2, 4, 8, 14):
            print(
                "static: nodes=%d C=%d" % (nodes, capacity),
                flush=True,
            )
            brain = make_chips(
                cluster,
                capacity=capacity * EXPERT_BYTES,
                reserve=0.0,
                count_per_node=CHIPS_PER_NODE,
            )
            kernels = make_kernels(experts, gpu, brain)
            max_copies = nodes * CHIPS_PER_NODE * capacity
            for policy in policies:
                plan = build_hbm_fallback_plan(
                    experts,
                    gpu,
                    brain,
                    demand,
                    policy,
                    max_brain_replicas=max_copies,
                    coactivation_weight=0.5,
                )
                result = execute_hbm_fallback_plan(
                    plan,
                    evaluation.trace,
                    cluster,
                    [chip.chip_id for chip in gpu],
                    kernels,
                    7168,
                )
                row = _execution_row(
                    "capacity", "policy-grid", nodes, capacity,
                    len(experts) * EXPERT_BYTES, result,
                    base_exec.completion_s,
                )
                rows.append(row)
                if (
                    nodes == node_grid[len(node_grid) // 2] and
                    capacity in (2, 8)
                ):
                    details.append({
                        "suite": "capacity",
                        "nodes": nodes,
                        "capacity": capacity,
                        "policy": policy,
                        "entry": asdict(row),
                        "critical_paths": [
                            {
                                "layer_id": layer.layer_id,
                                "assignment_id":
                                    layer.critical_path.assignment_id,
                                "replica_id":
                                    layer.critical_path.replica_id,
                                "dispatch_finish_s":
                                    layer.critical_path.dispatch_finish_s,
                                "compute_finish_s":
                                    layer.critical_path.compute_finish_s,
                                "combine_finish_s":
                                    layer.critical_path.combine_finish_s,
                            }
                            for layer in result.layers
                        ],
                    })
            if checkpoint_path is not None:
                write_csv(checkpoint_path, rows)
    return rows, details


def run_community_suite(quick: bool) -> Tuple[list, list]:
    # Mechanism isolation uses a fixed 8-layer, 64-expert subsystem.
    n_layers = 8
    n_experts = 64
    experts = make_experts(n_layers, n_experts)
    rows = []
    details = []
    evaluation_tokens = 32
    for community_strength in (0.0, 2.0):
        for capacity in (2, 4, 8):
            print(
                "community: coact=%.1f C=%d" % (
                    community_strength, capacity
                ),
                flush=True,
            )
            copies = n_layers * n_experts // 2
            chips_needed = math.ceil(copies / capacity)
            nodes = max(1, math.ceil(chips_needed / CHIPS_PER_NODE))
            cluster = make_cluster(nodes, CHIPS_PER_NODE)
            brain = make_chips(
                cluster,
                capacity=capacity * EXPERT_BYTES,
                reserve=0.0,
                count_per_node=CHIPS_PER_NODE,
            )
            gpu = make_gpu_chips()
            kernels = make_kernels(experts, gpu, brain)
            placement = generate_task_community_trace(
                evaluation_tokens,
                n_layers, n_experts, 4, cluster.ingress_nodes,
                {"code": 0.5, "math": 0.5}, 311, provenance(),
                "placement", community_count=8,
                community_strength=community_strength,
                node_affinity_strength=0.0,
            )
            evaluation = generate_task_community_trace(
                evaluation_tokens,
                n_layers, n_experts, 4, cluster.ingress_nodes,
                {"code": 0.5, "math": 0.5}, 421, provenance(),
                "evaluation", community_count=8,
                community_strength=community_strength,
                node_affinity_strength=0.0,
            )
            demand = demand_for_plan(placement)
            base = gpu_only_plan(experts, gpu, demand)
            base_exec = execute_hbm_fallback_plan(
                base, evaluation.trace, cluster,
                [chip.chip_id for chip in gpu], kernels, 7168,
            )
            for cluster_weight in (0.0, 2.0):
                plan = build_hbm_fallback_plan(
                    experts, gpu, brain, demand, "joint-marginal",
                    max_brain_replicas=copies,
                    coactivation_weight=cluster_weight,
                )
                result = execute_hbm_fallback_plan(
                    plan, evaluation.trace, cluster,
                    [chip.chip_id for chip in gpu], kernels, 7168,
                )
                row = _execution_row(
                    "community",
                    "coact=%.1f,cluster=%.1f" % (
                        community_strength, cluster_weight
                    ),
                    nodes, capacity, len(experts) * EXPERT_BYTES,
                    result, base_exec.completion_s,
                )
                rows.append(row)
                details.append({
                    "suite": "community",
                    "community_strength": community_strength,
                    "cluster_weight": cluster_weight,
                    "capacity": capacity,
                    "entry": asdict(row),
                    "coactivation_edges":
                        len(demand.coactivation),
                })
    return rows, details


def migration_penalty_for_plan(
    current,
    new,
    expert_bytes: int,
    bridge_bandwidth: float = 51.2e9,
    latency_s: float = 4e-7,
):
    """Return total bytes and parallel-bridge wall time for changed copies."""

    current_locations = plan_replica_locations(current)
    new_locations = plan_replica_locations(new)
    added = new_locations - current_locations
    by_node = {}
    for _, chip_id in added:
        node_id = new.chip(chip_id).node_id
        by_node[node_id] = by_node.get(node_id, 0) + expert_bytes
    nbytes = len(added) * expert_bytes
    if not by_node:
        return 0, 0.0
    return nbytes, (
        max(by_node.values()) / bridge_bandwidth + latency_s
    )


def run_refresh_suite(quick: bool) -> Tuple[list, list]:
    n_layers = 8
    n_experts = 64
    capacity = 4
    copies = 256
    nodes = 8
    epoch_mixes = (
        (("code", 0.9), ("math", 0.1)),
        (("code", 0.7), ("math", 0.3)),
        (("code", 0.3), ("math", 0.7)),
        (("code", 0.1), ("math", 0.9)),
        (("code", 0.8), ("math", 0.2)),
        (("code", 0.35), ("math", 0.65)),
    )
    if quick:
        epoch_mixes = epoch_mixes[:4]
    cluster = make_cluster(nodes, CHIPS_PER_NODE)
    brain = make_chips(
        cluster,
        capacity=capacity * EXPERT_BYTES,
        reserve=0.0,
        count_per_node=CHIPS_PER_NODE,
    )
    gpu = make_gpu_chips()
    experts = make_experts(n_layers, n_experts)
    kernels = make_kernels(experts, gpu, brain)
    epochs = []
    evaluation_tokens = 32 if quick else 64
    local_copy_budget = 1 if quick else 4
    epoch_steps = 1024 if quick else 4096
    for index, mix in enumerate(epoch_mixes):
        task_weights = dict(mix)
        planning = generate_task_community_trace(
            32, n_layers, n_experts, 4, cluster.ingress_nodes,
            task_weights, 1000 + index, provenance(), "placement",
            community_count=8, community_strength=1.5,
            node_affinity_strength=0.20,
        )
        evaluation = generate_task_community_trace(
            evaluation_tokens,
            n_layers, n_experts, 4, cluster.ingress_nodes,
            task_weights, 2000 + index, provenance(), "evaluation",
            community_count=8, community_strength=1.5,
            node_affinity_strength=0.20,
        )
        epochs.append((mix, planning, evaluation))

    policies = (
        "never", "global-if-profitable", "local-if-profitable",
        "drift-local", "task-boundary-local",
    )
    first_demand = demand_for_plan(epochs[0][1])
    initial = build_hbm_fallback_plan(
        experts, gpu, brain, first_demand, "joint-marginal",
        max_brain_replicas=copies, coactivation_weight=0.5,
    )
    current_by_policy = {policy: initial for policy in policies}
    cumulative = {policy: 0.0 for policy in policies}
    rows = []
    details = []
    previous_expected = dict(first_demand.global_assignments)
    previous_mix = tuple(epochs[0][0])
    gpu_ids = [chip.chip_id for chip in gpu]
    for epoch_index, (mix, planning, evaluation) in enumerate(epochs):
        demand = demand_for_plan(planning)
        drift = normalized_l1(
            previous_expected, dict(demand.global_assignments)
        )
        candidate = build_hbm_fallback_plan(
            experts, gpu, brain, demand, "joint-marginal",
            max_brain_replicas=copies, coactivation_weight=0.5,
        )
        execution_cache = {}

        def evaluate(plan):
            signature = plan_replica_locations(plan)
            if signature not in execution_cache:
                execution_cache[signature] = execute_hbm_fallback_plan(
                    plan, evaluation.trace, cluster, gpu_ids, kernels, 7168
                )
            return execution_cache[signature]

        candidate_exec = evaluate(candidate)
        mix_key = tuple(mix)
        for policy in policies:
            current = current_by_policy[policy]
            current_exec = evaluate(current)
            local = hybrid_replica_plan(
                current, candidate, demand,
                max_added_copies=local_copy_budget,
                strategy="local-incremental",
            )
            local_exec = evaluate(local)
            candidate_copies = migration_bytes(
                current, candidate, EXPERT_BYTES
            ) // EXPERT_BYTES
            local_copies = migration_bytes(
                current, local, EXPERT_BYTES
            ) // EXPERT_BYTES
            candidate_bytes, candidate_penalty = migration_penalty_for_plan(
                current, candidate, EXPERT_BYTES
            )
            local_bytes, local_penalty = migration_penalty_for_plan(
                current, local, EXPERT_BYTES
            )
            triggered = True
            adopted = False
            chosen = current
            chosen_exec = current_exec
            chosen_copies = 0
            chosen_bytes = 0
            chosen_penalty = 0.0
            if policy == "never":
                triggered = False
            elif policy == "global-if-profitable":
                if (
                    candidate_exec.completion_s * epoch_steps +
                    candidate_penalty
                ) < current_exec.completion_s * epoch_steps:
                    adopted = True
                    chosen = candidate
                    chosen_exec = candidate_exec
                    chosen_copies = candidate_copies
                    chosen_bytes = candidate_bytes
                    chosen_penalty = candidate_penalty
            elif policy == "local-if-profitable":
                if (
                    local_exec.completion_s * epoch_steps +
                    local_penalty
                ) < current_exec.completion_s * epoch_steps:
                    adopted = True
                    chosen = local
                    chosen_exec = local_exec
                    chosen_copies = local_copies
                    chosen_bytes = local_bytes
                    chosen_penalty = local_penalty
            elif policy == "drift-local":
                triggered = drift >= 0.10
                if triggered and (
                    local_exec.completion_s * epoch_steps +
                    local_penalty
                ) < current_exec.completion_s * epoch_steps:
                    adopted = True
                    chosen = local
                    chosen_exec = local_exec
                    chosen_copies = local_copies
                    chosen_bytes = local_bytes
                    chosen_penalty = local_penalty
            elif policy == "task-boundary-local":
                triggered = mix_key != previous_mix
                if triggered and (
                    local_exec.completion_s * epoch_steps +
                    local_penalty
                ) < current_exec.completion_s * epoch_steps:
                    adopted = True
                    chosen = local
                    chosen_exec = local_exec
                    chosen_copies = local_copies
                    chosen_bytes = local_bytes
                    chosen_penalty = local_penalty
            cumulative[policy] += (
                chosen_exec.completion_s * epoch_steps + chosen_penalty
            )
            row = RefreshRow(
                policy=policy,
                epoch=epoch_index,
                task_mix=json.dumps(dict(mix), sort_keys=True),
                drift_l1=drift,
                triggered=triggered,
                adopted=adopted,
                epoch_steps=epoch_steps,
                migration_copies=chosen_copies,
                migration_bytes=chosen_bytes,
                migration_time_s=chosen_penalty,
                current_completion_s=current_exec.completion_s,
                candidate_completion_s=candidate_exec.completion_s,
                selected_completion_s=chosen_exec.completion_s,
                projected_epoch_time_s=(
                    chosen_exec.completion_s * epoch_steps +
                    chosen_penalty
                ),
                cumulative_time_s=cumulative[policy],
                brain_replicas=chosen_exec.brain_replicas,
                brain_unique_experts=chosen_exec.brain_unique_experts,
            )
            rows.append(row)
            details.append({
                "policy": policy,
                "epoch": epoch_index,
                "entry": asdict(row),
                "local_candidate_completion_s": local_exec.completion_s,
                "local_migration_copies": local_copies,
            })
            current_by_policy[policy] = chosen
        previous_expected = dict(demand.global_assignments)
        previous_mix = mix_key
    return rows, details


def run_node_batch_suite(quick: bool) -> list:
    n_layers, n_experts = 8, 64
    nodes_grid = (2, 4, 8)
    batch_grid = (1, 4, 16, 64)
    capacity = 4
    copies = n_layers * n_experts // 2
    experts = make_experts(n_layers, n_experts)
    gpu = make_gpu_chips()
    rows = []
    for nodes in nodes_grid:
        print("node_batch: nodes=%d" % nodes, flush=True)
        cluster = make_cluster(nodes, CHIPS_PER_NODE)
        brain = make_chips(
            cluster, capacity * EXPERT_BYTES, 0.0, CHIPS_PER_NODE
        )
        kernels = make_kernels(experts, gpu, brain)
        placement = generate_task_community_trace(
            32, n_layers, n_experts, 4, cluster.ingress_nodes,
            {"code": 0.5, "math": 0.5}, 700 + nodes, provenance(),
            "placement", community_count=8,
            community_strength=1.0, node_affinity_strength=0.3,
        )
        demand = demand_for_plan(placement)
        base = gpu_only_plan(experts, gpu, demand)
        plans = [
            build_hbm_fallback_plan(
                experts, gpu, brain, demand, policy,
                max_brain_replicas=copies,
                coactivation_weight=0.5,
            )
            for policy in ("coverage-first", "joint-marginal")
        ]
        for batch in batch_grid:
            evaluation = generate_task_community_trace(
                batch, n_layers, n_experts, 4, cluster.ingress_nodes,
                {"code": 0.5, "math": 0.5}, 900 + nodes * 10 + batch,
                provenance(), "evaluation", community_count=8,
                community_strength=1.0, node_affinity_strength=0.3,
            )
            base_exec = execute_hbm_fallback_plan(
                base, evaluation.trace, cluster,
                [chip.chip_id for chip in gpu], kernels, 7168,
            )
            rows.append(_execution_row(
                "node_batch", "batch=%d,gpu-only" % batch,
                nodes, 0, len(experts) * EXPERT_BYTES,
                base_exec, base_exec.completion_s,
            ))
            for plan in plans:
                result = execute_hbm_fallback_plan(
                    plan, evaluation.trace, cluster,
                    [chip.chip_id for chip in gpu], kernels, 7168,
                )
                rows.append(_execution_row(
                    "node_batch", "batch=%d" % batch,
                    nodes, capacity, len(experts) * EXPERT_BYTES,
                    result, base_exec.completion_s,
                ))
    return rows


def run_skew_suite(quick: bool) -> list:
    n_layers, n_experts = 8, 64
    nodes = 4
    capacity = 4
    copies = n_layers * n_experts // 2
    experts = make_experts(n_layers, n_experts)
    cluster = make_cluster(nodes, CHIPS_PER_NODE)
    gpu = make_gpu_chips()
    brain = make_chips(
        cluster, capacity * EXPERT_BYTES, 0.0, CHIPS_PER_NODE
    )
    rows = []
    for exponent in (0.2, 1.0, 2.0, 3.0):
        print("skew: exponent=%.1f" % exponent, flush=True)
        placement = generate_task_community_trace(
            24, n_layers, n_experts, 4, cluster.ingress_nodes,
            {"code": 0.6, "math": 0.4}, 1701 + int(exponent * 10),
            provenance(), "placement", community_count=8,
            community_strength=2.0, node_affinity_strength=0.1,
            zipf_exponent=exponent,
        )
        evaluation = generate_task_community_trace(
            24, n_layers, n_experts, 4, cluster.ingress_nodes,
            {"code": 0.6, "math": 0.4}, 1801 + int(exponent * 10),
            provenance(), "evaluation", community_count=8,
            community_strength=2.0, node_affinity_strength=0.1,
            zipf_exponent=exponent,
        )
        demand = demand_for_plan(placement)
        kernels = make_kernels(experts, gpu, brain)
        base = gpu_only_plan(experts, gpu, demand)
        base_exec = execute_hbm_fallback_plan(
            base, evaluation.trace, cluster,
            [chip.chip_id for chip in gpu], kernels, 7168,
        )
        for policy in (
            "coverage-first", "replica-first", "joint-marginal",
            "ranked-0.10", "ranked-0.25", "ranked-0.50",
            "ranked-0.75", "ranked-0.90", "ranked-adaptive",
        ):
            plan = build_hbm_fallback_plan(
                experts, gpu, brain, demand, policy,
                max_brain_replicas=copies,
            )
            result = execute_hbm_fallback_plan(
                plan, evaluation.trace, cluster,
                [chip.chip_id for chip in gpu], kernels, 7168,
            )
            rows.append(_execution_row(
                "skew", "zipf=%.1f" % exponent,
                nodes, capacity, len(experts) * EXPERT_BYTES,
                result, base_exec.completion_s,
            ))
    return rows


def run_ranked_full_suite(quick: bool) -> list:
    """Selected full-model points for the gradual ranked policy."""

    if quick:
        return []
    n_layers, n_experts, top_k = 58, 256, 8
    experts = make_experts(n_layers, n_experts)
    gpu = make_gpu_chips()
    rows = []
    for nodes in (16, 32, 64):
        cluster = make_cluster(nodes, CHIPS_PER_NODE)
        placement = generate_task_community_trace(
            8, n_layers, n_experts, top_k, cluster.ingress_nodes,
            {"code": 0.6, "math": 0.4}, 101,
            provenance(), "placement", community_count=8,
            community_strength=1.0, node_affinity_strength=0.3,
        )
        evaluation = generate_task_community_trace(
            8, n_layers, n_experts, top_k, cluster.ingress_nodes,
            {"code": 0.6, "math": 0.4}, 211,
            provenance(), "evaluation", community_count=8,
            community_strength=1.0, node_affinity_strength=0.3,
        )
        demand = demand_for_plan(placement)
        for capacity in (4, 14):
            brain = make_chips(
                cluster, capacity * EXPERT_BYTES, 0.0, CHIPS_PER_NODE
            )
            kernels = make_kernels(experts, gpu, brain)
            base = gpu_only_plan(experts, gpu, demand)
            base_exec = execute_hbm_fallback_plan(
                base, evaluation.trace, cluster,
                [chip.chip_id for chip in gpu], kernels, 7168,
            )
            for policy in (
                "ranked-0.25", "ranked-0.50", "ranked-0.75",
                "ranked-adaptive",
            ):
                print(
                    "ranked_full: nodes=%d C=%d policy=%s" % (
                        nodes, capacity, policy
                    ),
                    flush=True,
                )
                plan = build_hbm_fallback_plan(
                    experts, gpu, brain, demand,
                    policy,
                    max_brain_replicas=nodes * CHIPS_PER_NODE * capacity,
                    coactivation_weight=0.5,
                )
                result = execute_hbm_fallback_plan(
                    plan, evaluation.trace, cluster,
                    [chip.chip_id for chip in gpu], kernels, 7168,
                )
                rows.append(_execution_row(
                    "ranked_full", policy,
                    nodes, capacity, len(experts) * EXPERT_BYTES,
                    result, base_exec.completion_s,
                ))
    return rows


def run_seed_suite(quick: bool) -> Tuple[list, list]:
    n_layers, n_experts = 8, 64
    nodes = 8
    capacity = 4
    copies = n_layers * n_experts // 2
    experts = make_experts(n_layers, n_experts)
    cluster = make_cluster(nodes, CHIPS_PER_NODE)
    gpu = make_gpu_chips()
    brain = make_chips(
        cluster, capacity * EXPERT_BYTES, 0.0, CHIPS_PER_NODE
    )
    kernels = make_kernels(experts, gpu, brain)
    seed_pairs = ((11, 17), (19, 29), (31, 37))
    rows = []
    collected = {}
    for placement_seed, evaluation_seed in seed_pairs:
        print(
            "seeds: placement=%d evaluation=%d" % (
                placement_seed, evaluation_seed
            ),
            flush=True,
        )
        placement = generate_task_community_trace(
            16 if quick else 8, n_layers, n_experts,
            4 if quick else 8, cluster.ingress_nodes,
            {"code": 0.6, "math": 0.4}, placement_seed,
            provenance(), "placement", community_count=8,
            community_strength=1.0, node_affinity_strength=0.3,
        )
        evaluation = generate_task_community_trace(
            16 if quick else 8, n_layers, n_experts,
            4 if quick else 8, cluster.ingress_nodes,
            {"code": 0.6, "math": 0.4}, evaluation_seed,
            provenance(), "evaluation", community_count=8,
            community_strength=1.0, node_affinity_strength=0.3,
        )
        demand = demand_for_plan(placement)
        base = gpu_only_plan(experts, gpu, demand)
        base_exec = execute_hbm_fallback_plan(
            base, evaluation.trace, cluster,
            [chip.chip_id for chip in gpu], kernels, 7168,
        )
        for policy in ("coverage-first", "joint-marginal"):
            plan = build_hbm_fallback_plan(
                experts, gpu, brain, demand, policy,
                max_brain_replicas=copies,
                coactivation_weight=0.5,
            )
            result = execute_hbm_fallback_plan(
                plan, evaluation.trace, cluster,
                [chip.chip_id for chip in gpu], kernels, 7168,
            )
            row = SeedRow(
                suite="seed_sensitivity",
                placement_seed=placement_seed,
                evaluation_seed=evaluation_seed,
                policy=policy,
                nodes=nodes,
                capacity_experts_per_chip=capacity,
                completion_s=result.completion_s,
                gpu_completion_s=base_exec.completion_s,
                speedup_vs_gpu=(
                    base_exec.completion_s / result.completion_s
                ),
                local_assignment_fraction=(
                    result.local_assignment_fraction
                ),
            )
            rows.append(row)
            collected.setdefault(policy, []).append(row.speedup_vs_gpu)
    ci_rows = []
    for policy, values in sorted(collected.items()):
        ci = confidence_interval(values, lower_bound=0.0)
        if ci.low95 > 1.0:
            conclusion = "supported"
        elif ci.high95 <= 1.0:
            conclusion = "unsupported"
        else:
            conclusion = "inconclusive"
        ci_rows.append(SeedCIRow(
            suite="seed_sensitivity",
            policy=policy,
            nodes=nodes,
            capacity_experts_per_chip=capacity,
            runs=len(values),
            speedup_mean=ci.mean,
            speedup_low95=ci.low95,
            speedup_high95=ci.high95,
            conclusion=conclusion,
        ))
    return rows, ci_rows


def run_sensitivity_suite(quick: bool) -> list:
    n_layers, n_experts = 8, 64
    nodes = 8
    capacity = 4
    copies = n_layers * n_experts // 2
    experts = make_experts(n_layers, n_experts)
    gpu = make_gpu_chips()
    rows = []
    for bridge_scale in (0.25, 1.0, 4.0):
        print("sensitivity: bridge=%.2f" % bridge_scale, flush=True)
        cluster = make_cluster(
            nodes, CHIPS_PER_NODE, bridge_scale=bridge_scale
        )
        brain = make_chips(
            cluster, capacity * EXPERT_BYTES, 0.0, CHIPS_PER_NODE
        )
        placement = generate_task_community_trace(
            16 if quick else 8, n_layers, n_experts,
            4 if quick else 8, cluster.ingress_nodes,
            {"code": 0.5, "math": 0.5}, 1301, provenance(),
            "placement", community_count=8,
            community_strength=2.0, node_affinity_strength=0.3,
        )
        evaluation = generate_task_community_trace(
            16 if quick else 8, n_layers, n_experts,
            4 if quick else 8, cluster.ingress_nodes,
            {"code": 0.5, "math": 0.5}, 1501, provenance(),
            "evaluation", community_count=8,
            community_strength=2.0, node_affinity_strength=0.3,
        )
        demand = demand_for_plan(placement)
        for brain_latency_scale in (0.5, 1.0, 2.0):
            kernels = make_kernels(
                experts, gpu, brain,
                brain_scale=brain_latency_scale,
            )
            base = gpu_only_plan(experts, gpu, demand)
            base_exec = execute_hbm_fallback_plan(
                base, evaluation.trace, cluster,
                [chip.chip_id for chip in gpu], kernels, 7168,
            )
            for policy in ("coverage-first", "joint-marginal"):
                plan = build_hbm_fallback_plan(
                    experts, gpu, brain, demand, policy,
                    max_brain_replicas=copies,
                    coactivation_weight=0.5,
                )
                result = execute_hbm_fallback_plan(
                    plan, evaluation.trace, cluster,
                    [chip.chip_id for chip in gpu], kernels, 7168,
                )
                rows.append(SensitivityRow(
                    suite="sensitivity",
                    policy=policy,
                    bridge_scale=bridge_scale,
                    brain_latency_scale=brain_latency_scale,
                    completion_s=result.completion_s,
                    speedup_vs_gpu=(
                        base_exec.completion_s / result.completion_s
                    ),
                    gpu_assignment_fraction=(
                        result.gpu_assignments /
                        max(1, result.assignments)
                    ),
                    local_assignment_fraction=(
                        result.local_assignment_fraction
                    ),
                ))
    return rows


def write_csv(path: Path, rows: Iterable) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=("quick", "full"), default="full"
    )
    parser.add_argument(
        "--components",
        default=(
            "static,community,node_batch,skew,"
            "ranked_full,refresh,seeds,sensitivity"
        ),
        help="comma-separated experiment components",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit("output directory already exists: %s" % output)
    output.mkdir(parents=True)
    quick = args.suite == "quick"

    components = {
        item.strip() for item in args.components.split(",") if item.strip()
    }
    allowed = {
        "static", "community", "node_batch", "skew", "refresh",
        "seeds", "sensitivity", "ranked_full",
    }
    unknown = components - allowed
    if unknown:
        raise SystemExit("unknown components: %s" % sorted(unknown))
    result_rows = {}
    details_all = []
    if "static" in components:
        checkpoint = output / "static_capacity.partial.csv"
        rows, details = run_static_suite(quick, checkpoint)
        write_csv(output / "static_capacity.csv", rows)
        if checkpoint.exists():
            checkpoint.unlink()
        details_all.extend(details)
        result_rows["static_capacity"] = len(rows)
    if "community" in components:
        rows, details = run_community_suite(quick)
        write_csv(output / "community.csv", rows)
        details_all.extend(details)
        result_rows["community"] = len(rows)
    if "node_batch" in components:
        rows = run_node_batch_suite(quick)
        write_csv(output / "node_batch.csv", rows)
        result_rows["node_batch"] = len(rows)
    if "skew" in components:
        rows = run_skew_suite(quick)
        write_csv(output / "skew.csv", rows)
        result_rows["skew"] = len(rows)
    if "ranked_full" in components:
        rows = run_ranked_full_suite(quick)
        write_csv(output / "ranked_full.csv", rows)
        result_rows["ranked_full"] = len(rows)
    if "refresh" in components:
        rows, details = run_refresh_suite(quick)
        write_csv(output / "refresh.csv", rows)
        details_all.extend(details)
        result_rows["refresh"] = len(rows)
    if "seeds" in components:
        rows, ci_rows = run_seed_suite(quick)
        write_csv(output / "seeds.csv", rows)
        write_csv(output / "seed_ci.csv", ci_rows)
        result_rows["seeds"] = len(rows)
        result_rows["seed_ci"] = len(ci_rows)
    if "sensitivity" in components:
        rows = run_sensitivity_suite(quick)
        write_csv(output / "sensitivity.csv", rows)
        result_rows["sensitivity"] = len(rows)
    (output / "details.json").write_text(
        json.dumps(details_all, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    source_files = [
        Path(__file__),
        ROOT / "flowgpu" / "sim_v2" / "research.py",
        ROOT / "flowgpu" / "sim_v2" / "tiering.py",
        ROOT / "flowgpu" / "sim_v2" / "layer_engine.py",
        ROOT / "flowgpu" / "sim_v2" / "detailed_network.py",
    ]
    manifest = {
        "schema_version": 1,
        "suite": args.suite,
        "components": sorted(components),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": sys.platform,
        "expert_bytes": EXPERT_BYTES,
        "gpu_count": GPU_COUNT,
        "gpu_bytes_per_device": GPU_BYTES,
        "gpu_reserve_fraction": GPU_RESERVE,
        "source_sha256": {
            str(path.resolve()): file_hash(path)
            for path in source_files
        },
        "result_rows": result_rows,
        "claim_boundary": (
            "paper-constrained synthetic routing and projected hardware; "
            "not KA200 or trace measurement"
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
