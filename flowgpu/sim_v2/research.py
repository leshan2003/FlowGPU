"""Research helpers for paper-constrained MoE placement experiments.

The functions in this module are intentionally separate from the strict CLI
schema. They provide controlled routing generators, an HBM-fallback planner
that can compare coverage and replication under the same capacity, and a
small evaluation adapter over the existing v2 event engine.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Tuple

from .demand import DemandProfile, profile_demand
from .dispatcher import Dispatcher
from .layer_engine import MoELayerEngine, MoELayerResult
from .placement import unique_balanced
from .replicas import ReplicaPlan
from .routing import (
    RoutingTrace,
    _weighted_sample_without_replacement,
    zipf_expert_weights,
)
from .schema import (
    ChipSpec,
    ExpertKey,
    ExpertSpec,
    Provenance,
    ReplicaSpec,
    RoutingRecord,
    RuntimeCapabilities,
    ValidationError,
)
from .topology import ClusterTopology


@dataclass(frozen=True)
class TaskAwareTrace:
    trace: RoutingTrace
    task_by_token: Mapping[int, str]


@dataclass(frozen=True)
class PlanExecution:
    strategy: str
    completion_s: float
    evaluation_tokens: int
    assignments: int
    gpu_assignments: int
    brain_assignments: int
    local_assignment_fraction: float
    network_energy_j: float
    brain_unique_experts: int
    brain_replicas: int
    layers: Tuple[MoELayerResult, ...]

    @property
    def tokens_per_s(self) -> float:
        return self.evaluation_tokens / self.completion_s


def generate_task_community_trace(
    n_tokens: int,
    n_layers: int,
    n_experts: int,
    top_k: int,
    home_nodes: Iterable[str],
    task_weights: Mapping[str, float],
    seed: int,
    provenance: Provenance,
    split: str,
    community_count: int = 8,
    community_strength: float = 1.0,
    node_affinity_strength: float = 0.0,
    zipf_exponent: float = 0.8,
) -> TaskAwareTrace:
    """Generate task-conditioned top-k routing with community co-activation.

    This is a controlled synthetic generator, not a replacement for a real
    router trace. Task labels influence a configurable community boost so the
    strict marginal popularity can be held similar while co-activation changes.
    """

    homes = tuple(home_nodes)
    tasks = tuple(sorted(task_weights))
    if min(n_tokens, n_layers, n_experts, top_k) <= 0:
        raise ValidationError("trace dimensions must be positive")
    if top_k > n_experts:
        raise ValidationError("top_k cannot exceed n_experts")
    if not homes or len(set(homes)) != len(homes):
        raise ValidationError("home nodes must be non-empty and unique")
    if not tasks:
        raise ValidationError("tasks cannot be empty")
    weights = tuple(float(task_weights[task]) for task in tasks)
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValidationError("task weights must be finite and non-negative")
    if sum(weights) <= 0:
        raise ValidationError("task weights must contain a positive value")
    if community_count <= 0 or community_count > n_experts:
        raise ValidationError("invalid community_count")
    if not math.isfinite(community_strength) or community_strength < 0:
        raise ValidationError("community_strength must be finite and >= 0")
    if (not math.isfinite(node_affinity_strength) or
            node_affinity_strength < 0):
        raise ValidationError(
            "node_affinity_strength must be finite and >= 0"
        )

    rng = random.Random(seed)
    base = zipf_expert_weights(n_experts, zipf_exponent)
    community = tuple(
        rng.randrange(community_count) for _ in range(n_experts)
    )
    preferred = {}
    preferred_count = max(
        1, community_count // max(1, len(tasks) * 2)
    )
    for task_index, task in enumerate(tasks):
        start = (task_index * preferred_count) % community_count
        preferred[task] = frozenset(
            (start + offset) % community_count
            for offset in range(preferred_count)
        )

    records = []
    task_by_token = {}
    equal_weight = 1.0 / top_k
    for token_id in range(n_tokens):
        target = rng.random() * sum(weights)
        running = 0.0
        task = tasks[-1]
        for name, value in zip(tasks, weights):
            running += value
            if target < running:
                task = name
                break
        task_by_token[token_id] = task
        home = homes[token_id % len(homes)]
        home_rank = homes.index(home)
        for layer_id in range(n_layers):
            local = []
            for expert_id, weight in enumerate(base):
                value = weight
                if community[expert_id] in preferred[task]:
                    value *= 1.0 + community_strength
                if node_affinity_strength:
                    # Deterministic node affinity, varied by layer/task.
                    preferred_node = (
                        expert_id + layer_id + tasks.index(task)
                    ) % len(homes)
                    if preferred_node == home_rank:
                        value *= 1.0 + node_affinity_strength
                local.append(value)
            experts = _weighted_sample_without_replacement(
                rng, tuple(local), top_k
            )
            records.append(RoutingRecord(
                request_id=0,
                token_id=token_id,
                step=token_id,
                layer_id=layer_id,
                home_node=home,
                experts=experts,
                gate_weights=(equal_weight,) * top_k,
            ))
    return TaskAwareTrace(
        trace=RoutingTrace(records, provenance, split),
        task_by_token=MappingProxyType(dict(task_by_token)),
    )


def _copy_allowed(
    key: ExpertKey,
    chip: ChipSpec,
    size: int,
    used: Mapping[str, int],
    locations: set,
) -> bool:
    return (
        (key, chip.chip_id) not in locations and
        used[chip.chip_id] + size <= chip.usable_weight_bytes
    )


def _coactivation_neighbors(
    demand: DemandProfile,
    per_expert: int = 24,
) -> Mapping[ExpertKey, Tuple[Tuple[ExpertKey, int], ...]]:
    affinity = {}
    for hyperedge, weight in demand.coactivation.items():
        for left in hyperedge:
            for right in hyperedge:
                if left == right:
                    continue
                pair = (left, right)
                affinity[pair] = affinity.get(pair, 0) + weight
    grouped = {}
    for (left, right), value in affinity.items():
        grouped.setdefault(left, []).append((right, value))
    return {
        key: tuple(sorted(values, key=lambda item: (-item[1], item[0]))[
            :per_expert
        ])
        for key, values in grouped.items()
    }


def adaptive_replica_share(expected: Mapping[ExpertKey, float]) -> float:
    """Choose a replication reserve from top-decile demand concentration."""

    values = sorted(
        (float(value) for value in expected.values() if value > 0),
        reverse=True,
    )
    total = sum(values)
    if total <= 0:
        return 0.10
    top_count = max(1, int(math.ceil(len(expected) * 0.10)))
    top_share = sum(values[:top_count]) / total
    if top_share >= 0.85:
        return 0.50
    if top_share >= 0.60:
        return 0.25
    return 0.0


def build_hbm_fallback_plan(
    experts: Iterable[ExpertSpec],
    gpu_chips: Iterable[ChipSpec],
    brain_chips: Iterable[ChipSpec],
    demand: DemandProfile,
    policy: str,
    max_brain_replicas: Optional[int] = None,
    coactivation_weight: float = 0.0,
    replica_share: float = 0.25,
) -> ReplicaPlan:
    """Build a GPU-primary plan with capacity-limited brain-side replicas.

    ``coverage-first``, ``replica-first`` and ``joint-marginal`` all consume
    the same GPU and brain capacity. The joint policy directly compares the
    next copy of each expert, including when ``brain_capacity < one unique
    copy of every expert``.
    """

    requested_policy = policy
    adaptive = policy == "ranked-adaptive"
    if adaptive:
        policy = "ranked-replication"
    elif policy.startswith("ranked-"):
        try:
            replica_share = float(policy.split("-", 1)[1])
        except ValueError as exc:
            raise ValidationError(
                "ranked policy must be ranked-<fraction>"
            ) from exc
        policy = "ranked-replication"
    if policy not in (
        "coverage-first", "replica-first", "joint-marginal",
        "ranked-replication",
    ):
        raise ValidationError("unknown HBM-fallback policy: %s" % policy)
    if not math.isfinite(replica_share) or not 0 <= replica_share < 1:
        raise ValidationError("replica_share must be in [0, 1)")
    expert_list = tuple(experts)
    gpu = tuple(gpu_chips)
    brain = tuple(brain_chips)
    if not expert_list or not gpu or not brain:
        raise ValidationError("tiered research plan needs experts/chips")
    if ({chip.chip_id for chip in gpu} &
            {chip.chip_id for chip in brain}):
        raise ValidationError("GPU and brain chip ids must differ")

    expected = {
        expert.key: float(demand.global_for(expert.key))
        for expert in expert_list
    }
    if adaptive:
        replica_share = adaptive_replica_share(expected)
    base = unique_balanced(expert_list, gpu, expected)
    replicas = list(base.replicas)
    brains_by_node: Dict[str, list] = {}
    for chip in brain:
        brains_by_node.setdefault(chip.node_id, []).append(chip)
    for values in brains_by_node.values():
        values.sort(key=lambda item: item.chip_id)
    used = {chip.chip_id: 0 for chip in brain}
    locations = set()
    node_residents = {node_id: set() for node_id in brains_by_node}
    node_copy_count = {}
    neighbors = _coactivation_neighbors(demand)
    ordered = tuple(sorted(
        expert_list,
        key=lambda expert: (
            -expected[expert.key] / expert.weight_bytes,
            expert.key,
        ),
    ))
    physical_limit = (
        sum(chip.usable_weight_bytes for chip in brain) //
        min(expert.weight_bytes for expert in expert_list)
    )
    limit = (
        min(max_brain_replicas, physical_limit)
        if max_brain_replicas is not None
        else physical_limit
    )
    if limit < 0:
        raise ValidationError("max_brain_replicas cannot be negative")

    def place(expert):
        feasible = []
        for node_id, chips in brains_by_node.items():
            demand_value = demand.node_for(node_id, expert.key)
            copies = node_copy_count.get((node_id, expert.key), 0)
            affinity = 0
            if coactivation_weight:
                for neighbor, weight in neighbors.get(expert.key, ()):
                    if neighbor in node_residents[node_id]:
                        affinity += weight
            for chip in chips:
                if _copy_allowed(
                        expert.key, chip, expert.weight_bytes,
                        used, locations):
                    feasible.append((
                        -(
                            demand_value / (copies + 1) +
                            coactivation_weight * affinity
                        ),
                        used[chip.chip_id] / chip.usable_weight_bytes,
                        chip.chip_id,
                        chip,
                        node_id,
                    ))
        if not feasible:
            return False
        _, _, _, chip, node_id = min(feasible)
        replicas.append(ReplicaSpec(
            replica_id="brain%d-L%d-E%d-%s" % (
                len(replicas), expert.key.layer_id,
                expert.key.expert_id, chip.chip_id,
            ),
            expert=expert.key,
            chip_id=chip.chip_id,
            is_primary=False,
        ))
        locations.add((expert.key, chip.chip_id))
        used[chip.chip_id] += expert.weight_bytes
        node_residents[node_id].add(expert.key)
        pair = (node_id, expert.key)
        node_copy_count[pair] = node_copy_count.get(pair, 0) + 1
        return True

    placed = 0
    if policy == "coverage-first":
        for expert in ordered:
            if placed >= limit:
                break
            if place(expert):
                placed += 1
        while placed < limit:
            progress = False
            for expert in ordered:
                if placed >= limit:
                    break
                if place(expert):
                    placed += 1
                    progress = True
            if not progress:
                break
    elif policy == "replica-first":
        for expert in ordered:
            while placed < limit and place(expert):
                placed += 1
            if placed >= limit:
                break
    elif policy == "ranked-replication":
        coverage_limit = int(math.floor(limit * (1.0 - replica_share)))
        for expert in ordered:
            if placed >= coverage_limit:
                break
            if place(expert):
                placed += 1
        queue = []
        copies = {expert.key: 0 for expert in expert_list}
        for expert in ordered:
            copies[expert.key] = sum(
                1 for located, _ in locations
                if located == expert.key
            )
            if copies[expert.key] and expected[expert.key] > 0:
                current = copies[expert.key]
                value = expected[expert.key] / (
                    current * (current + 1)
                )
                heapq.heappush(
                    queue, (-value, expert.key, expert)
                )
        while queue and placed < limit:
            _, key, expert = heapq.heappop(queue)
            if not place(expert):
                continue
            placed += 1
            copies[key] += 1
            current = copies[key]
            value = expected[key] / (current * (current + 1))
            if value > 0:
                heapq.heappush(
                    queue, (-value, key, expert)
                )
    else:
        queue = []
        copies = {expert.key: 0 for expert in expert_list}
        for expert in ordered:
            value = expected[expert.key]
            if value > 0:
                heapq.heappush(
                    queue, (-value, expert.key, expert)
                )
        while queue and placed < limit:
            _, key, expert = heapq.heappop(queue)
            if not place(expert):
                continue
            placed += 1
            copies[key] += 1
            current = copies[key]
            value = expected[key] / (current * (current + 1))
            if value > 0:
                heapq.heappush(
                    queue, (-value, key, expert)
                )
    return ReplicaPlan(
        experts=expert_list,
        chips=gpu + brain,
        replicas=replicas,
        strategy=(
            requested_policy if requested_policy.startswith("ranked-")
            else policy
        ),
    )


def execute_hbm_fallback_plan(
    plan: ReplicaPlan,
    trace: RoutingTrace,
    cluster: ClusterTopology,
    gpu_chip_ids: Iterable[str],
    kernel_models: Mapping,
    hidden_bytes: int,
    capabilities: RuntimeCapabilities = RuntimeCapabilities(),
) -> PlanExecution:
    """Run one held-out trace through the existing detailed v2 event engine."""

    if trace.split not in ("evaluation", "oracle"):
        raise ValidationError("execution requires a held-out trace")
    gpu_ids = frozenset(gpu_chip_ids)
    chip_endpoints = dict(cluster.chip_endpoints)
    for chip_id in gpu_ids:
        chip_endpoints[chip_id] = cluster.gpu_endpoint
    dispatcher = Dispatcher(
        replica_plan=plan,
        topology=cluster.topology,
        home_endpoints=cluster.home_endpoints,
        chip_endpoints=chip_endpoints,
        kernel_models=kernel_models,
        hidden_bytes=hidden_bytes,
        policy="earliest_finish",
        ingress_source_endpoint=cluster.gpu_endpoint,
    )
    engine = MoELayerEngine(
        dispatcher=dispatcher,
        capabilities=capabilities,
    )
    by_layer = {}
    for assignment in trace.assignments():
        by_layer.setdefault(assignment.layer_id, []).append(assignment)
    now_s = 0.0
    layer_results = []
    for layer_id in sorted(by_layer):
        result = engine.run(by_layer[layer_id], start_time_s=now_s)
        layer_results.append(result)
        now_s = result.completion_s
    assignments = sum(item.assignment_count for item in layer_results)
    gpu_assignments = sum(
        execution.assignment_count
        for layer in layer_results
        for execution in layer.replica_executions
        if execution.chip_id in gpu_ids
    )
    brain_unique = len({
        replica.expert for replica in plan.replicas
        if not replica.is_primary
    })
    return PlanExecution(
        strategy=plan.strategy,
        completion_s=now_s,
        evaluation_tokens=len(trace.records) // max(1, trace.top_k),
        assignments=assignments,
        gpu_assignments=gpu_assignments,
        brain_assignments=assignments - gpu_assignments,
        local_assignment_fraction=(
            sum(
                item.local_assignment_fraction * item.assignment_count
                for item in layer_results
            ) / assignments
        ),
        network_energy_j=sum(
            item.network_energy_j for item in layer_results
        ),
        brain_unique_experts=brain_unique,
        brain_replicas=sum(
            not replica.is_primary for replica in plan.replicas
        ),
        layers=tuple(layer_results),
    )


def plan_replica_locations(plan: ReplicaPlan) -> frozenset:
    return frozenset(
        (replica.expert, replica.chip_id)
        for replica in plan.replicas
        if not replica.is_primary
    )


def migration_bytes(old: ReplicaPlan, new: ReplicaPlan, bytes_per_expert: int) -> int:
    old_locations = plan_replica_locations(old)
    new_locations = plan_replica_locations(new)
    return len(new_locations - old_locations) * bytes_per_expert


def hybrid_replica_plan(
    current: ReplicaPlan,
    candidate: ReplicaPlan,
    demand: DemandProfile,
    max_added_copies: int,
    strategy: str = "local-incremental",
) -> ReplicaPlan:
    """Retain unchanged copies and add the highest-demand candidate copies.

    Copies present only in ``current`` are released, which keeps the result a
    subset of the capacity-feasible candidate plan.
    """

    if max_added_copies < 0:
        raise ValidationError("max_added_copies cannot be negative")
    current_locations = plan_replica_locations(current)
    unchanged = [
        replica for replica in candidate.replicas
        if not replica.is_primary and
        (replica.expert, replica.chip_id) in current_locations
    ]
    additions = [
        replica for replica in candidate.replicas
        if not replica.is_primary and
        (replica.expert, replica.chip_id) not in current_locations
    ]
    additions.sort(key=lambda replica: (
        -demand.global_for(replica.expert),
        replica.expert,
        replica.chip_id,
    ))
    replicas = [
        replica for replica in current.replicas
        if replica.is_primary
    ]
    replicas.extend(unchanged)
    replicas.extend(additions[:max_added_copies])
    return ReplicaPlan(
        experts=current.experts,
        chips=current.chips,
        replicas=replicas,
        strategy=strategy,
    )
