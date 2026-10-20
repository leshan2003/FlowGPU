"""Research helpers for paper-constrained MoE placement experiments.

The functions in this module are intentionally separate from the strict CLI
schema. They provide controlled routing generators, an HBM-fallback planner
that can compare coverage and replication under the same capacity, and a
small evaluation adapter over the existing v2 event engine.
"""

from __future__ import annotations

import heapq
import itertools
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
    Assignment,
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
class TaskPhase:
    name: str
    n_requests: int
    task_weights: Mapping[str, float]


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


@dataclass(frozen=True)
class BStageLayout:
    experts: Tuple[ExpertSpec, ...]
    brain_chips: Tuple[ChipSpec, ...]
    hot_chip: Mapping[ExpertKey, str]
    dram_node: Mapping[ExpertKey, str]
    staging_chip: Mapping[ExpertKey, str]
    dram_bandwidth_bytes_per_s: Mapping[str, float]
    dram_latency_s: float
    hot_bytes: int
    strategy: str = "b-stage"


@dataclass(frozen=True)
class BStageExecution:
    completion_s: float
    transfer_bytes: int
    transfer_time_s: float
    hot_assignment_fraction: float
    assignments: int


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


def generate_ordered_task_trace(
    tokens_per_request: int,
    n_layers: int,
    n_experts: int,
    top_k: int,
    home_nodes: Iterable[str],
    phases: Iterable[TaskPhase],
    seed: int,
    provenance: Provenance,
    split: str,
    community_count: int = 8,
    community_strength: float = 1.0,
    node_affinity_strength: float = 0.0,
    zipf_exponent: float = 0.8,
    first_layer: int = 0,
) -> TaskAwareTrace:
    """Generate a request-ordered task trajectory with stable communities.

    A task is sampled once per request, so all tokens in that request retain
    the same family. The expert-community mapping is generated once for the
    whole trajectory rather than re-randomized at every phase boundary.
    """

    if tokens_per_request <= 0:
        raise ValidationError("tokens_per_request must be positive")
    if first_layer < 0:
        raise ValidationError("first_layer cannot be negative")
    if min(n_layers, n_experts, top_k) <= 0:
        raise ValidationError("trace dimensions must be positive")
    if top_k > n_experts:
        raise ValidationError("top_k cannot exceed n_experts")
    homes = tuple(home_nodes)
    if not homes or len(set(homes)) != len(homes):
        raise ValidationError("home nodes must be non-empty and unique")
    phase_list = tuple(phases)
    if not phase_list or any(
        phase.n_requests <= 0 for phase in phase_list
    ):
        raise ValidationError("task phases must contain positive requests")
    task_names = sorted({
        task for phase in phase_list for task in phase.task_weights
    })
    if not task_names:
        raise ValidationError("task phases must define task weights")
    for phase in phase_list:
        if not phase.name:
            raise ValidationError("task phase name must be non-empty")
        if set(phase.task_weights) != set(task_names):
            raise ValidationError(
                "all task phases must use the same task-family key set"
            )
        values = tuple(float(value) for value in phase.task_weights.values())
        if sum(values) <= 0 or any(
            not math.isfinite(value) or value < 0 for value in values
        ):
            raise ValidationError("task phase weights must be valid")
    if community_count <= 0 or community_count > n_experts:
        raise ValidationError("invalid community_count")
    if (
        not math.isfinite(community_strength) or
        community_strength < 0
    ):
        raise ValidationError("community_strength must be finite and >= 0")
    if (
        not math.isfinite(node_affinity_strength) or
        node_affinity_strength < 0
    ):
        raise ValidationError(
            "node_affinity_strength must be finite and >= 0"
        )

    rng = random.Random(seed)
    base = zipf_expert_weights(n_experts, zipf_exponent)
    community = tuple(
        rng.randrange(community_count) for _ in range(n_experts)
    )
    preferred_count = max(
        1, community_count // max(1, len(task_names) * 2)
    )
    preferred = {}
    for task_index, task in enumerate(task_names):
        start = (task_index * preferred_count) % community_count
        preferred[task] = frozenset(
            (start + offset) % community_count
            for offset in range(preferred_count)
        )

    records = []
    task_by_token = {}
    request_id = 0
    global_token_id = 0
    equal_weight = 1.0 / top_k
    for phase in phase_list:
        tasks = tuple(sorted(phase.task_weights))
        phase_weights = tuple(
            float(phase.task_weights[task]) for task in tasks
        )
        total_weight = sum(phase_weights)
        for _ in range(phase.n_requests):
            target = rng.random() * total_weight
            running = 0.0
            task = tasks[-1]
            for name, value in zip(tasks, phase_weights):
                running += value
                if target < running:
                    task = name
                    break
            for _token in range(tokens_per_request):
                home = homes[global_token_id % len(homes)]
                home_rank = homes.index(home)
                task_by_token[global_token_id] = task
                for local_layer_id in range(n_layers):
                    layer_id = first_layer + local_layer_id
                    local = []
                    for expert_id, weight in enumerate(base):
                        value = weight
                        if community[expert_id] in preferred[task]:
                            value *= 1.0 + community_strength
                        if node_affinity_strength:
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
                        request_id=request_id,
                        token_id=global_token_id,
                        step=global_token_id,
                        layer_id=layer_id,
                        home_node=home,
                        experts=experts,
                        gate_weights=(equal_weight,) * top_k,
                        task_family=task,
                        stage="decode",
                        timestamp_s=float(global_token_id),
                        window=phase.name,
                    ))
                global_token_id += 1
            request_id += 1
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
) -> Mapping[ExpertKey, Tuple[Tuple[ExpertKey, float], ...]]:
    affinity = {}
    for hyperedge, weight in demand.coactivation.items():
        if len(hyperedge) <= 1:
            continue
        normalized = weight / float(len(hyperedge) - 1)
        for left, right in itertools.combinations(hyperedge, 2):
            affinity[(left, right)] = (
                affinity.get((left, right), 0) + normalized
            )
            affinity[(right, left)] = (
                affinity.get((right, left), 0) + normalized
            )
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


def build_marginal_event_plan(
    experts: Iterable[ExpertSpec],
    gpu_chips: Iterable[ChipSpec],
    brain_chips: Iterable[ChipSpec],
    demand: DemandProfile,
    max_brain_bytes: Optional[int] = None,
    max_brain_replicas: Optional[int] = None,
    evaluate_plan=None,
    candidate_limit: int = 64,
    coactivation_weight: float = 0.5,
) -> ReplicaPlan:
    """Spend a brain byte budget by repeated first/extra-copy competition.

    Candidate generation is deterministic. If ``evaluate_plan`` is supplied,
    only the top surrogate candidates are replayed through the event model and
    the lowest strictly improving objective is accepted. Without a callback,
    the normalized demand/locality proxy is used; such a result must be
    labelled as a surrogate plan by the caller.
    """

    expert_list = tuple(experts)
    gpu = tuple(gpu_chips)
    brain = tuple(brain_chips)
    if not expert_list or not gpu or not brain:
        raise ValidationError("marginal planning needs experts and chips")
    if ({chip.chip_id for chip in gpu} &
            {chip.chip_id for chip in brain}):
        raise ValidationError("GPU and brain chip ids must differ")
    if max_brain_bytes is not None and max_brain_bytes < 0:
        raise ValidationError("max_brain_bytes cannot be negative")
    if max_brain_replicas is not None and max_brain_replicas < 0:
        raise ValidationError("max_brain_replicas cannot be negative")
    if candidate_limit <= 0:
        raise ValidationError("candidate_limit must be positive")
    if (
        not math.isfinite(coactivation_weight) or
        coactivation_weight < 0
    ):
        raise ValidationError(
            "coactivation_weight must be finite and non-negative"
        )

    expected = {
        expert.key: float(demand.global_for(expert.key))
        for expert in expert_list
    }
    base = unique_balanced(expert_list, gpu, expected)
    replicas = list(base.replicas)
    brain_by_node = {}
    for chip in brain:
        brain_by_node.setdefault(chip.node_id, []).append(chip)
    for values in brain_by_node.values():
        values.sort(key=lambda item: item.chip_id)

    used_bytes = {chip.chip_id: 0 for chip in brain}
    chip_copy_count = {chip.chip_id: 0 for chip in brain}
    locations = set()
    node_copy_count = {}
    node_residents = {node_id: set() for node_id in brain_by_node}
    edge_membership = {}
    for hyperedge, weight in demand.coactivation.items():
        for key in hyperedge:
            edge_membership.setdefault(key, []).append(
                (hyperedge, weight)
            )

    def affinity(key, node_id):
        total = 0.0
        for hyperedge, weight in edge_membership.get(key, ()):
            overlap = sum(
                other in node_residents[node_id]
                for other in hyperedge if other != key
            )
            if overlap:
                total += weight * overlap / max(1, len(hyperedge) - 1)
        return total

    def best_candidate(expert):
        candidates = []
        for node_id, node_chips in brain_by_node.items():
            node_copies = node_copy_count.get((node_id, expert.key), 0)
            local_demand = demand.node_for(node_id, expert.key)
            for chip in node_chips:
                if (expert.key, chip.chip_id) in locations:
                    continue
                if (
                    used_bytes[chip.chip_id] + expert.weight_bytes >
                    chip.usable_weight_bytes
                ):
                    continue
                if (
                    chip.max_experts is not None and
                    chip_copy_count[chip.chip_id] + 1 > chip.max_experts
                ):
                    continue
                score = (
                    local_demand / (node_copies + 1) +
                    coactivation_weight * affinity(expert.key, node_id)
                )
                candidates.append((
                    score,
                    -used_bytes[chip.chip_id] / chip.usable_weight_bytes,
                    chip.chip_id,
                    chip,
                    node_id,
                ))
        if not candidates:
            return None
        return max(candidates, key=lambda item: (
            item[0], item[1], item[2]
        ))

    if evaluate_plan is None:
        current_value = None
    else:
        current_value = float(evaluate_plan(ReplicaPlan(
            expert_list, gpu + brain, replicas, "marginal-event"
        )))
        if not math.isfinite(current_value):
            raise ValidationError("event objective must return finite values")

    placed_bytes = 0
    placed_copies = 0
    while True:
        if (
            max_brain_replicas is not None and
            placed_copies >= max_brain_replicas
        ):
            break
        candidates = []
        for expert in expert_list:
            if expected[expert.key] <= 0:
                continue
            if (
                max_brain_bytes is not None and
                placed_bytes + expert.weight_bytes > max_brain_bytes
            ):
                continue
            best = best_candidate(expert)
            if best is None:
                continue
            score, load_tiebreak, chip_id, chip, node_id = best
            if score <= 0:
                continue
            candidates.append((
                score / expert.weight_bytes,
                load_tiebreak,
                expert.key,
                chip_id,
                expert,
                chip,
                node_id,
            ))
        if not candidates:
            break
        candidates.sort(reverse=True)
        shortlisted = candidates[:candidate_limit]

        chosen = shortlisted[0]
        chosen_value = None
        if evaluate_plan is not None:
            chosen = None
            for candidate in shortlisted:
                expert = candidate[4]
                chip = candidate[5]
                trial_replicas = replicas + [ReplicaSpec(
                    replica_id="marginal%d-L%d-E%d-%s" % (
                        len(replicas),
                        expert.key.layer_id,
                        expert.key.expert_id,
                        chip.chip_id,
                    ),
                    expert=expert.key,
                    chip_id=chip.chip_id,
                    is_primary=False,
                )]
                trial = ReplicaPlan(
                    expert_list, gpu + brain, trial_replicas,
                    "marginal-event",
                )
                value = float(evaluate_plan(trial))
                if not math.isfinite(value):
                    raise ValidationError(
                        "event objective must return finite values"
                    )
                if value < current_value - 1e-12 and (
                    chosen is None or
                    (value, candidate[2], candidate[3]) <
                    (chosen_value, chosen[2], chosen[3])
                ):
                    chosen = candidate
                    chosen_value = value
            if chosen is None:
                break
            current_value = chosen_value

        _, _, _, chip_id, expert, chip, node_id = chosen
        replicas.append(ReplicaSpec(
            replica_id="marginal%d-L%d-E%d-%s" % (
                len(replicas),
                expert.key.layer_id,
                expert.key.expert_id,
                chip.chip_id,
            ),
            expert=expert.key,
            chip_id=chip.chip_id,
            is_primary=False,
        ))
        locations.add((expert.key, chip.chip_id))
        used_bytes[chip.chip_id] += expert.weight_bytes
        chip_copy_count[chip.chip_id] += 1
        placed_bytes += expert.weight_bytes
        placed_copies += 1
        pair = (node_id, expert.key)
        node_copy_count[pair] = node_copy_count.get(pair, 0) + 1
        node_residents[node_id].add(expert.key)

    return ReplicaPlan(
        expert_list, gpu + brain, replicas, "marginal-event"
    )


def build_h_exclusive_plan(
    experts: Iterable[ExpertSpec],
    gpu_chips: Iterable[ChipSpec],
    brain_chips: Iterable[ChipSpec],
    demand: DemandProfile,
    max_brain_bytes: Optional[int] = None,
    max_brain_copies: Optional[int] = None,
) -> ReplicaPlan:
    """Place each expert on exactly one execution tier.

    Hot experts receive a Brain-SRAM primary and lose their GPU/HBM copy;
    all remaining experts retain one GPU/HBM primary. There is no fallback.
    """

    expert_list = tuple(experts)
    gpu = tuple(gpu_chips)
    brain = tuple(brain_chips)
    if not expert_list or not gpu or not brain:
        raise ValidationError("H-exclusive planning needs experts and chips")
    if any(chip.role != "gpu" for chip in gpu):
        raise ValidationError("H-exclusive GPU chips must have role=gpu")
    if any(chip.role != "brain_sram" for chip in brain):
        raise ValidationError(
            "H-exclusive brain chips must have role=brain_sram"
        )
    if max_brain_bytes is not None and max_brain_bytes < 0:
        raise ValidationError("max_brain_bytes cannot be negative")
    if max_brain_copies is not None and max_brain_copies < 0:
        raise ValidationError("max_brain_copies cannot be negative")

    expected = {
        expert.key: float(demand.global_for(expert.key))
        for expert in expert_list
    }
    brain_by_node = {}
    for chip in brain:
        brain_by_node.setdefault(chip.node_id, []).append(chip)
    for values in brain_by_node.values():
        values.sort(key=lambda item: item.chip_id)
    used = {chip.chip_id: 0 for chip in brain}
    counts = {chip.chip_id: 0 for chip in brain}
    node_experts = {node_id: set() for node_id in brain_by_node}
    hot_chip = {}
    hot_bytes = 0
    ordered = sorted(
        expert_list,
        key=lambda expert: (
            -expected[expert.key] / expert.weight_bytes,
            expert.key,
        ),
    )
    for expert in ordered:
        if max_brain_copies is not None and len(hot_chip) >= max_brain_copies:
            break
        if (
            max_brain_bytes is not None and
            hot_bytes + expert.weight_bytes > max_brain_bytes
        ):
            continue
        feasible = []
        for node_id, node_chips in brain_by_node.items():
            locality = demand.node_for(node_id, expert.key)
            for chip in node_chips:
                if (
                    used[chip.chip_id] + expert.weight_bytes >
                    chip.usable_weight_bytes
                ):
                    continue
                if (
                    chip.max_experts is not None and
                    counts[chip.chip_id] + 1 > chip.max_experts
                ):
                    continue
                feasible.append((
                    -locality,
                    -int(expert.key in node_experts[node_id]),
                    used[chip.chip_id] / chip.usable_weight_bytes,
                    chip.chip_id,
                    chip,
                    node_id,
                ))
        if not feasible:
            continue
        _, _, _, chip_id, chip, node_id = min(feasible)
        hot_chip[expert.key] = chip_id
        hot_bytes += expert.weight_bytes
        used[chip_id] += expert.weight_bytes
        counts[chip_id] += 1
        node_experts[node_id].add(expert.key)

    cold_experts = tuple(
        expert for expert in expert_list
        if expert.key not in hot_chip
    )
    if cold_experts:
        base = unique_balanced(
            cold_experts,
            gpu,
            {expert.key: expected[expert.key] for expert in cold_experts},
        )
        replicas = list(base.replicas)
    else:
        replicas = []
    for expert in expert_list:
        chip_id = hot_chip.get(expert.key)
        if chip_id is not None:
            replicas.append(ReplicaSpec(
                "h-brain-L%d-E%d-%s" % (
                    expert.key.layer_id,
                    expert.key.expert_id,
                    chip_id,
                ),
                expert.key,
                chip_id,
                True,
            ))
    return ReplicaPlan(
        expert_list, gpu + brain, replicas, "h-exclusive"
    )


def build_b_stage_layout(
    experts: Iterable[ExpertSpec],
    brain_chips: Iterable[ChipSpec],
    demand: DemandProfile,
    dram_bandwidth_bytes_per_s: Mapping[str, float],
    dram_capacity_bytes_per_node: Mapping[str, int],
    dram_latency_s: float = 900e-9,
    max_hot_bytes: Optional[int] = None,
    max_hot_experts: Optional[int] = None,
) -> BStageLayout:
    """Build a B-stage layout: all experts in DRAM, hot subset resident SRAM."""

    expert_list = tuple(experts)
    chips = tuple(sorted(brain_chips, key=lambda chip: chip.chip_id))
    if not expert_list or not chips:
        raise ValidationError("B-stage layout needs experts and brain chips")
    if any(chip.role != "brain_sram" for chip in chips):
        raise ValidationError("B-stage execution chips must be brain_sram")
    nodes = sorted({chip.node_id for chip in chips})
    if not nodes:
        raise ValidationError("B-stage layout has no brain nodes")
    for node_id in nodes:
        if node_id not in dram_bandwidth_bytes_per_s:
            raise ValidationError(
                "missing DRAM bandwidth for node %s" % node_id
            )
        if node_id not in dram_capacity_bytes_per_node:
            raise ValidationError(
                "missing DRAM capacity for node %s" % node_id
            )
        if dram_bandwidth_bytes_per_s[node_id] <= 0:
            raise ValidationError("DRAM bandwidth must be positive")
        if dram_capacity_bytes_per_node[node_id] < 0:
            raise ValidationError("DRAM capacity cannot be negative")
    if not math.isfinite(dram_latency_s) or dram_latency_s < 0:
        raise ValidationError("DRAM latency must be finite and non-negative")
    if max_hot_bytes is not None and max_hot_bytes < 0:
        raise ValidationError("max_hot_bytes cannot be negative")
    if max_hot_experts is not None and max_hot_experts < 0:
        raise ValidationError("max_hot_experts cannot be negative")

    dram_node = {}
    dram_bytes = {node_id: 0 for node_id in nodes}
    for expert in sorted(expert_list, key=lambda value: value.key):
        feasible = [
            node_id for node_id in nodes
            if (
                dram_bytes[node_id] + expert.weight_bytes <=
                dram_capacity_bytes_per_node[node_id]
            )
        ]
        if not feasible:
            raise ValidationError(
                "no B-stage DRAM node can hold expert %r" %
                (expert.key,)
            )
        node_id = min(
            feasible,
            key=lambda value: (
                -demand.node_for(value, expert.key),
                (
                    dram_bytes[value] /
                    dram_capacity_bytes_per_node[value]
                    if dram_capacity_bytes_per_node[value] > 0 else 0.0
                ),
                nodes.index(value),
            ),
        )
        dram_node[expert.key] = node_id
        dram_bytes[node_id] += expert.weight_bytes

    chips_by_node = {
        node_id: tuple(
            chip for chip in chips if chip.node_id == node_id
        )
        for node_id in nodes
    }
    used = {chip.chip_id: 0 for chip in chips}
    counts = {chip.chip_id: 0 for chip in chips}
    hot_chip = {}
    hot_bytes = 0
    ordered = sorted(
        expert_list,
        key=lambda expert: (
            -demand.global_for(expert.key) / expert.weight_bytes,
            expert.key,
        ),
    )
    for expert in ordered:
        if (
            max_hot_experts is not None and
            len(hot_chip) >= max_hot_experts
        ):
            break
        if (
            max_hot_bytes is not None and
            hot_bytes + expert.weight_bytes > max_hot_bytes
        ):
            continue
        preferred_node = dram_node[expert.key]
        candidates = []
        for chip in chips:
            if (
                used[chip.chip_id] + expert.weight_bytes >
                chip.usable_weight_bytes
            ):
                continue
            if (
                chip.max_experts is not None and
                counts[chip.chip_id] + 1 > chip.max_experts
            ):
                continue
            candidates.append((
                chip.node_id != preferred_node,
                used[chip.chip_id] / chip.usable_weight_bytes,
                chip.chip_id,
                chip,
            ))
        if not candidates:
            continue
        _, _, chip_id, chip = min(candidates)
        hot_chip[expert.key] = chip_id
        hot_bytes += expert.weight_bytes
        used[chip_id] += expert.weight_bytes
        counts[chip_id] += 1

    staging_chip = {}
    staging_load = {chip.chip_id: 0.0 for chip in chips}
    for expert in expert_list:
        if expert.key in hot_chip:
            continue
        preferred_node = dram_node[expert.key]
        candidates = []
        for chip in chips:
            if chip.usable_weight_bytes < expert.weight_bytes:
                continue
            candidates.append((
                chip.node_id != preferred_node,
                staging_load[chip.chip_id],
                chip.chip_id,
                chip,
            ))
        if not candidates:
            raise ValidationError(
                "no SRAM chip can stage expert %r" % (expert.key,)
            )
        _, _, chip_id, chip = min(candidates)
        staging_chip[expert.key] = chip_id
        staging_load[chip_id] += demand.global_for(expert.key)

    return BStageLayout(
        experts=expert_list,
        brain_chips=chips,
        hot_chip=MappingProxyType(dict(hot_chip)),
        dram_node=MappingProxyType(dict(dram_node)),
        staging_chip=MappingProxyType(dict(staging_chip)),
        dram_bandwidth_bytes_per_s=MappingProxyType(dict(
            dram_bandwidth_bytes_per_s
        )),
        dram_latency_s=dram_latency_s,
        hot_bytes=hot_bytes,
    )


def execute_b_stage_layer(
    layout: BStageLayout,
    assignments: Iterable[Assignment],
    kernel_models: Mapping,
) -> BStageExecution:
    """Execute one layer with grouped expert coalescing and DRAM staging.

    Cold experts are fetched at most once per expert per layer invocation.
    The staging uses the selected chip queue, then performs the kernel there.
    """

    grouped = {}
    for assignment in assignments:
        grouped[assignment.expert] = grouped.get(
            assignment.expert, 0
        ) + 1
    expert_map = {expert.key: expert for expert in layout.experts}
    chip_map = {chip.chip_id: chip for chip in layout.brain_chips}
    finish = {chip.chip_id: 0.0 for chip in layout.brain_chips}
    transfer_bytes = 0
    transfer_time = 0.0
    hot_assignments = 0
    total_assignments = sum(grouped.values())
    for key, count in sorted(grouped.items()):
        expert = expert_map.get(key)
        if expert is None:
            raise ValidationError("B-stage assignment references unknown expert")
        chip_id = layout.hot_chip.get(key)
        staged = chip_id is None
        if staged:
            chip_id = layout.staging_chip.get(key)
            if chip_id is None:
                raise ValidationError("B-stage missing staging chip")
            node_id = layout.dram_node[key]
            load_s = (
                expert.weight_bytes /
                layout.dram_bandwidth_bytes_per_s[node_id] +
                layout.dram_latency_s
            )
            transfer_bytes += expert.weight_bytes
            transfer_time += load_s
        else:
            load_s = 0.0
            hot_assignments += count
        try:
            kernel = kernel_models[(key, chip_id)]
        except KeyError as exc:
            raise ValidationError(
                "B-stage missing kernel for %r on %s" % (key, chip_id)
            ) from exc
        compute_s = kernel.predict(count).latency_s
        finish[chip_id] += load_s + compute_s
    return BStageExecution(
        completion_s=max(finish.values(), default=0.0),
        transfer_bytes=transfer_bytes,
        transfer_time_s=transfer_time,
        hot_assignment_fraction=(
            hot_assignments / total_assignments
            if total_assignments else 0.0
        ),
        assignments=total_assignments,
    )


def build_plan_dispatcher(
    plan: ReplicaPlan,
    cluster: ClusterTopology,
    gpu_chip_ids: Iterable[str],
    kernel_models: Mapping,
    hidden_bytes: int,
    capabilities: RuntimeCapabilities = RuntimeCapabilities(),
    policy: str = "earliest_finish",
) -> Dispatcher:
    """Build the common dispatcher used by execution and serving."""

    gpu_ids = frozenset(gpu_chip_ids)
    chip_endpoints = dict(cluster.chip_endpoints)
    for chip_id in gpu_ids:
        chip_endpoints[chip_id] = cluster.gpu_endpoint
    return Dispatcher(
        replica_plan=plan,
        topology=cluster.topology,
        home_endpoints=cluster.home_endpoints,
        chip_endpoints=chip_endpoints,
        kernel_models=kernel_models,
        hidden_bytes=hidden_bytes,
        policy=policy,
        ingress_source_endpoint=cluster.gpu_endpoint,
    )


def execute_hbm_fallback_plan(
    plan: ReplicaPlan,
    trace: RoutingTrace,
    cluster: ClusterTopology,
    gpu_chip_ids: Iterable[str],
    kernel_models: Mapping,
    hidden_bytes: int,
    capabilities: RuntimeCapabilities = RuntimeCapabilities(),
    allow_placement_trace: bool = False,
) -> PlanExecution:
    """Run one held-out trace through the existing detailed v2 event engine."""

    if (
        trace.split not in ("evaluation", "oracle") and
        not allow_placement_trace
    ):
        raise ValidationError("execution requires a held-out trace")
    gpu_ids = frozenset(gpu_chip_ids)
    dispatcher = build_plan_dispatcher(
        plan, cluster, gpu_ids, kernel_models, hidden_bytes,
        capabilities=capabilities,
        policy="earliest_finish",
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
    brain_replicas = tuple(
        replica for replica in plan.replicas
        if plan.chip(replica.chip_id).role in (
            "brain_sram", "brain_dram"
        )
    )
    brain_unique = len({
        replica.expert for replica in brain_replicas
    })
    return PlanExecution(
        strategy=plan.strategy,
        completion_s=now_s,
        evaluation_tokens=len({
            record.token_id for record in trace.records
        }),
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
        brain_replicas=len(brain_replicas),
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
