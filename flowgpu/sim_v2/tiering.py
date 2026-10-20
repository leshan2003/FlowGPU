"""GPU--brain expert tiering plans for R < 1 and excess-capacity studies."""

from __future__ import annotations

import heapq
from typing import Dict, Iterable, Mapping, Optional, Tuple

from .replicas import ReplicaPlan
from .schema import (ChipSpec, ExpertKey, ExpertSpec, ReplicaSpec,
                     ValidationError)


def _gpu_primary(key: ExpertKey, gpu_chip_id: str) -> ReplicaSpec:
    return ReplicaSpec(
        replica_id="gpu-primary-L%d-%sE%d" % (
            key.layer_id, "S" if key.shared else "R", key.expert_id
        ),
        expert=key,
        chip_id=gpu_chip_id,
        is_primary=True,
    )


def _brain_replica(
    key: ExpertKey, chip_id: str, index: int
) -> ReplicaSpec:
    return ReplicaSpec(
        replica_id="brain%d-L%d-%sE%d-%s" % (
            index, key.layer_id,
            "S" if key.shared else "R",
            key.expert_id, chip_id,
        ),
        expert=key,
        chip_id=chip_id,
        is_primary=False,
    )


def make_gpu_tier_chip(
    experts: Iterable[ExpertSpec],
    gpu_chip_id: str,
    provenance,
    capacity_bytes: Optional[int] = None,
) -> ChipSpec:
    """Create the explicit GPU/HBM residency device used by tiered plans."""

    expert_list = tuple(experts)
    required = sum(expert.weight_bytes for expert in expert_list)
    if not expert_list:
        raise ValidationError("GPU tier needs experts")
    if capacity_bytes is None:
        capacity_bytes = required
    if capacity_bytes < required:
        raise ValidationError(
            "GPU tier cannot hold one fallback copy of every expert"
        )
    return ChipSpec(
        chip_id=gpu_chip_id,
        node_id="gpu-tier",
        sram_capacity_bytes=capacity_bytes,
        runtime_reserve_fraction=0.0,
        fixed_reserved_bytes=0,
        provenance=provenance,
    )


def tiered_expert_plan(
    experts: Iterable[ExpertSpec],
    brain_chips: Iterable[ChipSpec],
    gpu_chip: ChipSpec,
    expected_assignments: Mapping[ExpertKey, float],
    policy: str,
    max_brain_replicas: Optional[int] = None,
    node_assignments: Optional[
        Mapping[Tuple[str, ExpertKey], float]
    ] = None,
) -> ReplicaPlan:
    """Keep a GPU fallback for every expert and spend brain SRAM by policy.

    ``coverage-first`` tries one copy of every expert before adding extra
    replicas. ``replica-first`` exhausts the hottest experts first.
    ``joint-marginal`` compares the next copy of every expert globally.

    The joint allocator uses ``d / (n * (n + 1))`` as the marginal load
    balancing proxy for the next copy, where ``d`` is placement-window demand
    and ``n`` is the number of existing copies. This is a planning heuristic;
    downstream event simulation remains the source of performance results.

    Legacy names ``unique-first`` and ``hot-replica-first`` alias
    ``coverage-first`` and ``replica-first`` respectively.
    """

    expert_list = tuple(experts)
    brain = tuple(brain_chips)
    if not expert_list or not brain:
        raise ValidationError("tiered plan needs experts and brain chips")
    if gpu_chip.chip_id in {chip.chip_id for chip in brain}:
        raise ValidationError("GPU and brain chip ids must be distinct")
    requested_policy = policy
    policy = {
        "unique-first": "coverage-first",
        "hot-replica-first": "replica-first",
    }.get(policy, policy)
    if policy not in ("coverage-first", "replica-first", "joint-marginal"):
        raise ValidationError("unknown tiering policy: %s" % policy)
    demand = dict(expected_assignments)
    node_demand = dict(node_assignments or {})
    if any(
        key not in {expert.key for expert in expert_list} or
        value < 0
        for key, value in demand.items()
    ):
        raise ValidationError("invalid tiering expected assignments")
    if any(
        key[1] not in {expert.key for expert in expert_list} or
        value < 0
        for key, value in node_demand.items()
    ):
        raise ValidationError("invalid tiered node assignments")
    if max_brain_replicas is not None and max_brain_replicas < 0:
        raise ValidationError("max_brain_replicas cannot be negative")

    replicas = [
        _gpu_primary(expert.key, gpu_chip.chip_id)
        for expert in expert_list
    ]
    used = {chip.chip_id: 0 for chip in brain}
    node_copy_count: Dict[Tuple[str, ExpertKey], int] = {}
    locations = set()
    ordered = tuple(sorted(
        expert_list,
        key=lambda expert: (
            -demand.get(expert.key, 0.0) / expert.weight_bytes,
            expert.key,
        ),
    ))
    limit = (
        max_brain_replicas
        if max_brain_replicas is not None
        else sum(
            chip.usable_weight_bytes for chip in brain
        ) // min(expert.weight_bytes for expert in expert_list)
    )

    def add_one(expert):
        def score(chip):
            node_copies = node_copy_count.get(
                (chip.node_id, expert.key), 0
            )
            local_demand = node_demand.get(
                (chip.node_id, expert.key), 0.0
            )
            return (
                -(local_demand / (node_copies + 1)),
                used[chip.chip_id] / chip.usable_weight_bytes,
                chip.chip_id,
            )

        feasible = [
            chip for chip in brain
            if (expert.key, chip.chip_id) not in locations and
            used[chip.chip_id] + expert.weight_bytes <=
            chip.usable_weight_bytes
        ]
        if not feasible:
            return False
        chip = min(feasible, key=score)
        replicas.append(_brain_replica(
            expert.key, chip.chip_id, len(replicas)
        ))
        locations.add((expert.key, chip.chip_id))
        used[chip.chip_id] += expert.weight_bytes
        pair = (chip.node_id, expert.key)
        node_copy_count[pair] = node_copy_count.get(pair, 0) + 1
        return True

    placed = 0
    if policy == "coverage-first":
        for expert in ordered:
            if placed >= limit:
                break
            if add_one(expert):
                placed += 1
        while placed < limit:
            progress = False
            for expert in ordered:
                if placed >= limit:
                    break
                if add_one(expert):
                    placed += 1
                    progress = True
            if not progress:
                break
    else:
        if policy == "replica-first":
            for expert in ordered:
                while placed < limit and add_one(expert):
                    placed += 1
                if placed >= limit:
                    break
        else:
            queue = []
            copies = {expert.key: 0 for expert in expert_list}
            for expert in ordered:
                value = demand.get(expert.key, 0.0)
                if value > 0:
                    heapq.heappush(
                        queue,
                        (-value, expert.key, expert, 0),
                    )
            while queue and placed < limit:
                _, key, expert, _ = heapq.heappop(queue)
                if not add_one(expert):
                    continue
                placed += 1
                copies[key] += 1
                current = copies[key]
                value = demand.get(key, 0.0) / (
                    current * (current + 1)
                )
                if value > 0:
                    heapq.heappush(
                        queue,
                        (-value, key, expert, current),
                    )

    return ReplicaPlan(
        experts=expert_list,
        chips=brain + (gpu_chip,),
        replicas=replicas,
        strategy="tiered-" + requested_policy,
    )
