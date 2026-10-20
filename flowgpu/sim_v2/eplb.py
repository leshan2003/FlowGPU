"""Step-aware expert-parallel load balancing for the MoE v2 simulator."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Tuple

from .placement import unique_balanced
from .replicas import ReplicaPlan
from .routing import RoutingTrace
from .schema import ExpertKey, ReplicaSpec, ValidationError


@dataclass(frozen=True)
class LoadBalanceMetrics:
    node_lb: float
    chip_lb: float
    steps: int


def assignment_counts(
    trace: RoutingTrace,
) -> Mapping[Tuple[int, int, ExpertKey], int]:
    cached = getattr(trace, "_eplb_assignment_counts", None)
    if cached is not None:
        return cached
    counts = {}
    for record in trace.records:
        for expert_id in record.experts:
            key = (record.step, record.layer_id, ExpertKey(
                record.layer_id,
                expert_id,
            ))
            counts[key] = counts.get(key, 0) + 1
    cached = MappingProxyType(dict(counts))
    trace._eplb_assignment_counts = cached
    return cached


def _ratio(loads: Iterable[float]) -> float:
    values = tuple(float(value) for value in loads)
    if not values or sum(values) <= 0:
        return 1.0
    mean = sum(values) / len(values)
    return max(values) / mean if mean > 0 else 1.0


def placement_load_balance(
    plan: ReplicaPlan,
    trace: RoutingTrace,
    brain_chip_ids: Iterable[str],
) -> LoadBalanceMetrics:
    """Estimate per-step node/chip imbalance for Brain replicas.

    If an expert has replicas on multiple nodes/chips, each copy receives an
    equal share. This is a placement proxy, not an event-model measurement.
    """

    brain_ids = frozenset(brain_chip_ids)
    replicas_by_expert = {}
    for replica in plan.replicas:
        if replica.chip_id not in brain_ids:
            continue
        replicas_by_expert.setdefault(replica.expert, []).append(replica)
    counts = assignment_counts(trace)
    by_time = {}
    for (step, layer, expert), value in counts.items():
        by_time.setdefault((step, layer), []).append((expert, value))
    node_values = []
    chip_values = []
    for (step, layer), items in by_time.items():
        node_load = {
            node_id: 0.0
            for node_id in sorted({
                plan.chip(chip_id).node_id
                for chip_id in brain_ids
            })
        }
        chip_load = {chip_id: 0.0 for chip_id in brain_ids}
        for expert, count in items:
            replicas = replicas_by_expert.get(expert, ())
            if not replicas:
                continue
            for replica in replicas:
                share = count / len(replicas)
                chip_load[replica.chip_id] += share
                node_load[plan.chip(replica.chip_id).node_id] += share
        node_values.append(_ratio(node_load.values()))
        chip_values.append(_ratio(chip_load.values()))
    return LoadBalanceMetrics(
        node_lb=(
            sum(node_values) / len(node_values)
            if node_values else 1.0
        ),
        chip_lb=(
            sum(chip_values) / len(chip_values)
            if chip_values else 1.0
        ),
        steps=len(by_time),
    )


def allocate_eplb_replicas(
    trace: RoutingTrace,
    experts: Iterable[ExpertKey],
    total_copies: int,
    redundancy_fraction: float = 0.25,
    max_copies_per_expert: Optional[int] = None,
) -> Mapping[ExpertKey, int]:
    """Allocate a fixed Brain budget between coverage and redundant copies.

    The first ``1 - redundancy_fraction`` of the budget covers distinct
    experts in decreasing observed-load order. Remaining copies are assigned
    progressively to the covered experts whose next copy has the largest
    peak-load reduction. This deliberately permits duplication before full
    unique coverage, matching the ranked/graduated replication model.
    """

    if trace.split == "evaluation":
        raise ValidationError(
            "evaluation trace cannot be used for EPLB allocation"
        )
    if total_copies < 0:
        raise ValidationError("total_copies cannot be negative")
    if (
        not math.isfinite(redundancy_fraction) or
        not 0 <= redundancy_fraction < 1
    ):
        raise ValidationError("redundancy_fraction must be in [0, 1)")
    if max_copies_per_expert is not None and max_copies_per_expert < 1:
        raise ValidationError("max_copies_per_expert must be positive")
    expert_keys = tuple(dict.fromkeys(experts))
    if total_copies == 0:
        return {}
    if not expert_keys:
        raise ValidationError("EPLB allocation needs at least one expert")

    total_load = {key: 0.0 for key in expert_keys}
    peak_load = {key: 0.0 for key in expert_keys}
    active_bins = {key: 0 for key in expert_keys}
    for _bin, values in _loads_by_bin(trace).items():
        for key, value in values.items():
            if key not in total_load:
                continue
            total_load[key] += value
            peak_load[key] = max(peak_load[key], float(value))
            active_bins[key] += 1

    ordered = tuple(sorted(
        expert_keys,
        key=lambda key: (
            -total_load[key],
            -peak_load[key],
            key,
        ),
    ))
    if max_copies_per_expert is None:
        max_copies = None
        feasible_budget = total_copies
    else:
        max_copies = max_copies_per_expert
        feasible_budget = sum(min(max_copies, total_copies)
                              for _ in expert_keys[:total_copies])
    if total_copies > feasible_budget:
        raise ValidationError(
            "total_copies exceeds EPLB per-expert copy capacity"
        )

    coverage_budget = int(math.floor(total_copies * (
        1.0 - redundancy_fraction
    )))
    coverage_budget = min(
        len(expert_keys),
        max(1, coverage_budget),
        total_copies,
    )
    selected = ordered[:coverage_budget]
    counts = {key: 1 for key in selected}
    remaining = total_copies - len(selected)
    queue = []
    for key in selected:
        next_load = peak_load[key] if peak_load[key] > 0 else total_load[key]
        heapq.heappush(
            queue,
            (-next_load / (counts[key] * (counts[key] + 1)), key),
        )
    while remaining:
        if not queue:
            raise ValidationError("EPLB could not allocate all requested copies")
        _, key = heapq.heappop(queue)
        if max_copies is not None and counts[key] >= max_copies:
            continue
        counts[key] += 1
        remaining -= 1
        next_load = peak_load[key] if peak_load[key] > 0 else total_load[key]
        heapq.heappush(
            queue,
            (-next_load / (counts[key] * (counts[key] + 1)), key),
        )
    return counts


def build_eplb_plan(
    experts,
    gpu_chips,
    brain_chips,
    trace: RoutingTrace,
    total_brain_copies: int,
    redundancy_fraction: float = 0.25,
    max_passes: int = 4,
) -> ReplicaPlan:
    """Build a GPU-primary overlay plan with DeepInfer-style EPLB layout."""

    expert_list = tuple(experts)
    gpu = tuple(gpu_chips)
    brain = tuple(brain_chips)
    if not expert_list or not gpu or not brain:
        raise ValidationError("EPLB planning needs experts and chips")
    if ({chip.chip_id for chip in gpu} &
            {chip.chip_id for chip in brain}):
        raise ValidationError("GPU and brain chip ids must differ")

    loads_by_bin = _loads_by_bin(trace)
    expected = {expert.key: 0.0 for expert in expert_list}
    for values in loads_by_bin.values():
        for key, value in values.items():
            if key in expected:
                expected[key] += value
    gpu_base = unique_balanced(expert_list, gpu, expected)
    counts = allocate_eplb_replicas(
        trace,
        (expert.key for expert in expert_list),
        total_brain_copies,
        redundancy_fraction=redundancy_fraction,
        max_copies_per_expert=len(brain),
    )
    expert_map = {expert.key: expert for expert in expert_list}
    chip_map = {chip.chip_id: chip for chip in brain}
    used = {chip_id: 0 for chip_id in chip_map}
    residents = {chip_id: 0 for chip_id in chip_map}
    locations = set()
    node_residents = {
        node_id: set() for node_id in {chip.node_id for chip in brain}
    }
    projected_chip = {chip_id: 0.0 for chip_id in chip_map}
    projected_node = {
        node_id: 0.0 for node_id in {chip.node_id for chip in brain}
    }
    replicas = list(gpu_base.replicas)
    peak = {
        key: max(
            (values.get(key, 0) for values in loads_by_bin.values()),
            default=0,
        )
        for key in counts
    }
    ordered = sorted(
        counts,
        key=lambda key: (-expected[key], -peak[key], key),
    )
    for key in ordered:
        expert = expert_map[key]
        copies = counts[key]
        for copy_index in range(copies):
            feasible = []
            for chip_id, chip in chip_map.items():
                if (key, chip_id) in locations:
                    continue
                if used[chip_id] + expert.weight_bytes > (
                    chip.usable_weight_bytes
                ):
                    continue
                if (
                    chip.max_experts is not None and
                    residents[chip_id] + 1 > chip.max_experts
                ):
                    continue
                share = (
                    float(peak[key]) / copies
                    if peak[key] > 0 else
                    expected[key] / max(1.0, float(copies))
                )
                feasible.append((
                    chip.node_id in node_residents and copy_index > 0,
                    key in node_residents[chip.node_id],
                    projected_node[chip.node_id] + share,
                    projected_chip[chip_id] + share,
                    used[chip_id] / chip.usable_weight_bytes,
                    chip_id,
                ))
            if not feasible:
                raise ValidationError(
                    "EPLB has no feasible Brain chip for expert %r" % (key,)
                )
            _, _, node_value, chip_value, _, chip_id = min(feasible)
            chip = chip_map[chip_id]
            replicas.append(ReplicaSpec(
                "eplb-%d-L%d-E%d-%s" % (
                    len(replicas),
                    key.layer_id,
                    key.expert_id,
                    chip_id,
                ),
                key,
                chip_id,
                False,
            ))
            locations.add((key, chip_id))
            used[chip_id] += expert.weight_bytes
            residents[chip_id] += 1
            node_residents[chip.node_id].add(key)
            projected_chip[chip_id] = chip_value
            projected_node[chip.node_id] = node_value

    plan = ReplicaPlan(
        experts=expert_list,
        chips=gpu + brain,
        replicas=replicas,
        strategy="eplb-redundant",
    )
    return rebalance_brain_replicas(
        plan,
        trace,
        (chip.chip_id for chip in brain),
        max_passes=max_passes,
    )


def _loads_by_bin(
    trace: RoutingTrace,
) -> Mapping[Tuple[int, int], Mapping[ExpertKey, int]]:
    cached = getattr(trace, "_eplb_loads_by_bin", None)
    if cached is not None:
        return cached
    by_bin = {}
    for (step, layer, key), count in assignment_counts(trace).items():
        by_bin.setdefault((step, layer), {})[key] = count
    cached = MappingProxyType({
        key: MappingProxyType(dict(value))
        for key, value in by_bin.items()
    })
    trace._eplb_loads_by_bin = cached
    return cached


def rebalance_brain_replicas(
    plan: ReplicaPlan,
    trace: RoutingTrace,
    brain_chip_ids: Iterable[str],
    max_passes: int = 4,
) -> ReplicaPlan:
    """Move Brain replicas to reduce step-level node/chip imbalance.

    The expert-copy multiset is preserved exactly. GPU replicas are untouched.
    Moves respect chip bytes, max-experts, and per-chip duplicate constraints.
    """

    if max_passes < 0:
        raise ValidationError("max_passes cannot be negative")
    if trace.split == "evaluation":
        raise ValidationError(
            "evaluation trace cannot be used for EPLB rebalancing"
        )
    brain_ids = tuple(sorted(brain_chip_ids))
    brain_set = frozenset(brain_ids)
    chip_map = {chip.chip_id: chip for chip in plan.chips}
    gpu_replicas = [
        replica for replica in plan.replicas
        if replica.chip_id not in brain_set
    ]
    brain_replicas = [
        replica for replica in plan.replicas
        if replica.chip_id in brain_set
    ]
    if not brain_replicas:
        return plan

    used = {chip_id: 0 for chip_id in brain_ids}
    counts = {chip_id: 0 for chip_id in brain_ids}
    experts_on_chip = {chip_id: set() for chip_id in brain_ids}
    expert_bytes = {
        expert.key: expert.weight_bytes for expert in plan.experts
    }
    for replica in brain_replicas:
        weight = expert_bytes[replica.expert]
        used[replica.chip_id] += weight
        counts[replica.chip_id] += 1
        experts_on_chip[replica.chip_id].add(replica.expert)

    current = list(brain_replicas)
    current_metrics = placement_load_balance(
        ReplicaPlan(
            plan.experts,
            plan.chips,
            gpu_replicas + current,
            plan.strategy,
        ),
        trace,
        brain_ids,
    )
    for _ in range(max_passes):
        aggregate_chip_loads = _aggregate_chip_loads(
            trace, current, brain_ids
        )
        mean_chip_load = (
            sum(aggregate_chip_loads.values()) / len(brain_ids)
        )
        source_indices = [
            index for index, replica in enumerate(current)
            if aggregate_chip_loads[replica.chip_id] > mean_chip_load
        ]
        target_ids = [
            chip_id for chip_id in brain_ids
            if aggregate_chip_loads[chip_id] < mean_chip_load
        ]
        if not source_indices or not target_ids:
            break
        best = None
        for replica_index in source_indices:
            replica = current[replica_index]
            source = replica.chip_id
            for target in target_ids:
                if target == source:
                    continue
                target_chip = chip_map[target]
                if replica.expert in experts_on_chip[target]:
                    continue
                if (
                    target_chip.max_experts is not None and
                    len(experts_on_chip[target]) + 1 >
                    target_chip.max_experts
                ):
                    continue
                weight = expert_bytes[replica.expert]
                if used[target] + weight > target_chip.usable_weight_bytes:
                    continue
                moved = ReplicaSpec(
                    replica.replica_id,
                    replica.expert,
                    target,
                    replica.is_primary,
                )
                candidate = list(current)
                candidate[replica_index] = moved
                metrics = placement_load_balance(
                    ReplicaPlan(
                        plan.experts,
                        plan.chips,
                        gpu_replicas + candidate,
                        plan.strategy,
                    ),
                    trace,
                    brain_ids,
                )
                score = (
                    metrics.node_lb + 0.25 * metrics.chip_lb,
                    metrics.node_lb,
                    metrics.chip_lb,
                    target,
                )
                if best is None or score < best[0]:
                    best = (
                        score,
                        replica_index,
                        source,
                        target,
                        moved,
                        metrics,
                    )
        if best is None:
            break
        score, index, source, target, moved, metrics = best
        if score[0] >= (
            current_metrics.node_lb + 0.25 * current_metrics.chip_lb
        ) - 1e-12:
            break
        experts_on_chip[source].remove(moved.expert)
        counts[source] -= 1
        used[source] -= expert_bytes[moved.expert]
        experts_on_chip[target].add(moved.expert)
        counts[target] += 1
        used[target] += expert_bytes[moved.expert]
        current[index] = moved
        current_metrics = metrics

    for _ in range(max_passes):
        aggregate_chip_loads = _aggregate_chip_loads(
            trace, current, brain_ids
        )
        mean_chip_load = (
            sum(aggregate_chip_loads.values()) / len(brain_ids)
        )
        high_indices = [
            index for index, replica in enumerate(current)
            if aggregate_chip_loads[replica.chip_id] > mean_chip_load
        ]
        low_indices = [
            index for index, replica in enumerate(current)
            if aggregate_chip_loads[replica.chip_id] < mean_chip_load
        ]
        if not high_indices or not low_indices:
            break
        best_swap = None
        for left_index in high_indices:
            left = current[left_index]
            for right_index in low_indices:
                if right_index == left_index:
                    continue
                right = current[right_index]
                if left.expert == right.expert:
                    continue
                left_chip_id = left.chip_id
                right_chip_id = right.chip_id
                if left_chip_id == right_chip_id:
                    continue
                if right.expert in experts_on_chip[left_chip_id]:
                    continue
                if left.expert in experts_on_chip[right_chip_id]:
                    continue
                left_weight = expert_bytes[left.expert]
                right_weight = expert_bytes[right.expert]
                if (
                    used[left_chip_id] - left_weight + right_weight >
                    chip_map[left_chip_id].usable_weight_bytes
                ):
                    continue
                if (
                    used[right_chip_id] - right_weight + left_weight >
                    chip_map[right_chip_id].usable_weight_bytes
                ):
                    continue
                moved_left = ReplicaSpec(
                    left.replica_id,
                    left.expert,
                    right_chip_id,
                    left.is_primary,
                )
                moved_right = ReplicaSpec(
                    right.replica_id,
                    right.expert,
                    left_chip_id,
                    right.is_primary,
                )
                candidate = list(current)
                candidate[left_index] = moved_left
                candidate[right_index] = moved_right
                metrics = placement_load_balance(
                    ReplicaPlan(
                        plan.experts,
                        plan.chips,
                        gpu_replicas + candidate,
                        plan.strategy,
                    ),
                    trace,
                    brain_ids,
                )
                score = (
                    metrics.node_lb + 0.25 * metrics.chip_lb,
                    metrics.node_lb,
                    metrics.chip_lb,
                    left_chip_id,
                    right_chip_id,
                )
                if best_swap is None or score < best_swap[0]:
                    best_swap = (
                        score, left_index, right_index,
                        moved_left, moved_right, metrics,
                    )
        if best_swap is None:
            break
        score, left_index, right_index, moved_left, moved_right, metrics = (
            best_swap
        )
        if score[0] >= (
            current_metrics.node_lb + 0.25 * current_metrics.chip_lb
        ) - 1e-12:
            break
        left = current[left_index]
        right = current[right_index]
        experts_on_chip[left.chip_id].remove(left.expert)
        experts_on_chip[left.chip_id].add(right.expert)
        experts_on_chip[right.chip_id].remove(right.expert)
        experts_on_chip[right.chip_id].add(left.expert)
        used[left.chip_id] = (
            used[left.chip_id] - expert_bytes[left.expert] +
            expert_bytes[right.expert]
        )
        used[right.chip_id] = (
            used[right.chip_id] - expert_bytes[right.expert] +
            expert_bytes[left.expert]
        )
        current[left_index] = moved_left
        current[right_index] = moved_right
        current_metrics = metrics

    return ReplicaPlan(
        experts=plan.experts,
        chips=plan.chips,
        replicas=gpu_replicas + current,
        strategy="eplb-rebalanced",
    )


def _aggregate_chip_loads(
    trace: RoutingTrace,
    replicas,
    brain_chip_ids,
) -> Mapping[str, float]:
    """Aggregate planning load under equal sharing across expert copies."""

    chip_ids = tuple(brain_chip_ids)
    loads = {chip_id: 0.0 for chip_id in chip_ids}
    replicas_by_expert = {}
    for replica in replicas:
        replicas_by_expert.setdefault(replica.expert, []).append(replica)
    for key, count in assignment_counts(trace).items():
        expert = key[2]
        copies = replicas_by_expert.get(expert, ())
        if copies:
            share = count / len(copies)
            for replica in copies:
                loads[replica.chip_id] += share
    return loads


def execution_load_balance(
    execution,
    plan: ReplicaPlan,
    brain_node_ids: Iterable[str],
) -> LoadBalanceMetrics:
    """Measure event-model assignment imbalance over all Brain nodes/chips."""

    node_ids = tuple(sorted(brain_node_ids))
    chip_ids = tuple(sorted(
        chip.chip_id for chip in plan.chips
        if chip.node_id in node_ids and chip.role.startswith("brain")
    ))
    node_values = []
    chip_values = []
    for layer in execution.layers:
        node_load = {node_id: 0.0 for node_id in node_ids}
        chip_load = {chip_id: 0.0 for chip_id in chip_ids}
        for item in layer.replica_executions:
            chip = plan.chip(item.chip_id)
            if chip.node_id not in node_ids or not chip.role.startswith("brain"):
                continue
            node_load[chip.node_id] += item.assignment_count
            chip_load[chip.chip_id] += item.assignment_count
        node_values.append(_ratio(node_load.values()))
        chip_values.append(_ratio(chip_load.values()))
    return LoadBalanceMetrics(
        node_lb=(
            sum(node_values) / len(node_values)
            if node_values else 1.0
        ),
        chip_lb=(
            sum(chip_values) / len(chip_values)
            if chip_values else 1.0
        ),
        steps=len(execution.layers),
    )
