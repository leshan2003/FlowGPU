"""Initial v2 expert placement strategies.

Only the primary-only ``unique-balanced`` baseline is implemented here.  It is
kept independent of v1 placement because v1 maps operations to device pools,
whereas v2 places complete physical expert replicas on concrete chips.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Callable, Dict, Iterable, Mapping, Optional

from .demand import DemandProfile
from .replicas import ReplicaPlan
from .schema import (ChipSpec, ExpertKey, ExpertSpec, ReplicaSpec,
                     ValidationError)


def _replica_id(key: ExpertKey) -> str:
    kind = "S" if key.shared else "R"
    return "primary-L%d-%sE%d" % (key.layer_id, kind, key.expert_id)


def unique_balanced(
    experts: Iterable[ExpertSpec],
    chips: Iterable[ChipSpec],
    expected_assignments: Optional[Mapping[ExpertKey, float]] = None,
) -> ReplicaPlan:
    """Place one primary per expert while balancing predicted chip load.

    Heavier-demand and larger experts are placed first.  Among feasible chips,
    the deterministic objective is predicted load, then SRAM utilization, then
    chip id.  This is a baseline heuristic, not the small-topology oracle.
    """

    expert_list = tuple(experts)
    chip_list = tuple(chips)
    if not expert_list or not chip_list:
        raise ValidationError("unique-balanced needs experts and chips")
    demand = dict(expected_assignments or {})
    for key, value in demand.items():
        if value < 0:
            raise ValidationError(
                "expected assignment load cannot be negative for %r" % (key,)
            )

    used = {chip.chip_id: 0 for chip in chip_list}
    load = {chip.chip_id: 0.0 for chip in chip_list}
    replicas = []
    ordered = sorted(
        expert_list,
        key=lambda expert: (
            -demand.get(expert.key, 0.0),
            -expert.weight_bytes,
            expert.key,
        ),
    )
    for expert in ordered:
        feasible = [
            chip for chip in chip_list
            if used[chip.chip_id] + expert.weight_bytes
            <= chip.usable_weight_bytes
        ]
        if not feasible:
            raise ValidationError(
                "unique-balanced cannot fit expert %r (%d bytes)" %
                (expert.key, expert.weight_bytes)
            )
        chip = min(
            feasible,
            key=lambda candidate: (
                load[candidate.chip_id],
                used[candidate.chip_id] /
                candidate.usable_weight_bytes,
                candidate.chip_id,
            ),
        )
        replicas.append(ReplicaSpec(
            replica_id=_replica_id(expert.key),
            expert=expert.key,
            chip_id=chip.chip_id,
            is_primary=True,
        ))
        used[chip.chip_id] += expert.weight_bytes
        load[chip.chip_id] += demand.get(expert.key, 0.0)

    return ReplicaPlan(
        experts=expert_list,
        chips=chip_list,
        replicas=replicas,
        strategy="unique-balanced",
    )


def capacity_ratio(
    experts: Iterable[ExpertSpec],
    chips: Iterable[ChipSpec],
) -> float:
    unique_weight_bytes = sum(expert.weight_bytes for expert in experts)
    if unique_weight_bytes <= 0:
        raise ValidationError("capacity ratio needs positive expert weights")
    return (
        sum(chip.usable_weight_bytes for chip in chips) /
        unique_weight_bytes
    )


def _replica_with_index(
    key: ExpertKey, chip_id: str, index: int
) -> ReplicaSpec:
    kind = "S" if key.shared else "R"
    return ReplicaSpec(
        replica_id="replica%d-L%d-%sE%d-%s" % (
            index, key.layer_id, kind, key.expert_id, chip_id
        ),
        expert=key,
        chip_id=chip_id,
        is_primary=False,
    )


def _initial_state(
    experts: tuple,
    chips: tuple,
    demand: DemandProfile,
):
    expected = {
        expert.key: demand.global_for(expert.key)
        for expert in experts
    }
    base = unique_balanced(experts, chips, expected)
    used = {
        chip_id: usage.weight_bytes
        for chip_id, usage in base.capacity_usage.items()
    }
    locations = {
        (replica.expert, replica.chip_id)
        for replica in base.replicas
    }
    expert_map = {expert.key: expert for expert in experts}
    chip_map = {chip.chip_id: chip for chip in chips}
    return base, used, locations, expert_map, chip_map


def popularity_replicated(
    experts: Iterable[ExpertSpec],
    chips: Iterable[ChipSpec],
    demand: DemandProfile,
    max_total_replicas: Optional[int] = None,
) -> ReplicaPlan:
    """Global-popularity replication baseline.

    Candidate value is the idealized reduction in assignments per copy divided
    by expert bytes.  It intentionally ignores home-node locality, providing a
    clean contrast with :func:`ingress_local`.
    """

    expert_list = tuple(experts)
    chip_list = tuple(chips)
    base, used, locations, expert_map, chip_map = _initial_state(
        expert_list, chip_list, demand
    )
    if max_total_replicas is None:
        max_total_replicas = sum(
            chip.usable_weight_bytes for chip in chip_list
        ) // min(expert.weight_bytes for expert in expert_list)
    if max_total_replicas < len(base.replicas):
        raise ValidationError(
            "max_total_replicas is below required primary count"
        )
    replicas = list(base.replicas)
    copies = {expert.key: 1 for expert in expert_list}
    predicted_chip_load = {chip.chip_id: 0.0 for chip in chip_list}
    for replica in base.replicas:
        predicted_chip_load[replica.chip_id] += (
            demand.global_for(replica.expert)
        )

    while len(replicas) < max_total_replicas:
        candidates = []
        for expert in expert_list:
            current_copies = copies[expert.key]
            marginal = (
                demand.global_for(expert.key) /
                (current_copies * (current_copies + 1))
            )
            if marginal <= 0:
                continue
            for chip in chip_list:
                if (expert.key, chip.chip_id) in locations:
                    continue
                if (used[chip.chip_id] + expert.weight_bytes >
                        chip.usable_weight_bytes):
                    continue
                score_per_byte = marginal / expert.weight_bytes
                candidates.append((
                    -score_per_byte,
                    predicted_chip_load[chip.chip_id],
                    used[chip.chip_id] / chip.usable_weight_bytes,
                    expert.key,
                    chip.chip_id,
                    expert,
                    chip,
                ))
        if not candidates:
            break
        _, _, _, _, _, expert, chip = min(candidates)
        replica = _replica_with_index(
            expert.key, chip.chip_id, len(replicas)
        )
        replicas.append(replica)
        locations.add((expert.key, chip.chip_id))
        used[chip.chip_id] += expert.weight_bytes
        old_copies = copies[expert.key]
        copies[expert.key] += 1
        predicted_chip_load[chip.chip_id] += (
            demand.global_for(expert.key) / (old_copies + 1)
        )

    return ReplicaPlan(
        expert_list, chip_list, replicas, "popularity-replicated"
    )


def ingress_local(
    experts: Iterable[ExpertSpec],
    chips: Iterable[ChipSpec],
    demand: DemandProfile,
    max_total_replicas: Optional[int] = None,
) -> ReplicaPlan:
    """Replicate experts into nodes that originate demand for them.

    This strategy adds at most one copy of a given expert per node.  Therefore
    every added copy represents a locality opportunity; same-node extra copies
    for pure compute balancing are left to a separate strategy/ablation.
    """

    expert_list = tuple(experts)
    chip_list = tuple(chips)
    base, used, locations, expert_map, chip_map = _initial_state(
        expert_list, chip_list, demand
    )
    if max_total_replicas is None:
        max_total_replicas = sum(
            chip.usable_weight_bytes for chip in chip_list
        ) // min(expert.weight_bytes for expert in expert_list)
    if max_total_replicas < len(base.replicas):
        raise ValidationError(
            "max_total_replicas is below required primary count"
        )
    replicas = list(base.replicas)
    node_chips: Dict[str, list] = {}
    for chip in chip_list:
        node_chips.setdefault(chip.node_id, []).append(chip)
    for values in node_chips.values():
        values.sort(key=lambda chip: chip.chip_id)

    while len(replicas) < max_total_replicas:
        resident_nodes = {
            (replica.expert, chip_map[replica.chip_id].node_id)
            for replica in replicas
        }
        candidates = []
        for (node_id, key), count in demand.node_assignments.items():
            if count <= 0 or key not in expert_map:
                continue
            if (key, node_id) in resident_nodes:
                continue
            expert = expert_map[key]
            for chip in node_chips.get(node_id, ()):
                if (key, chip.chip_id) in locations:
                    continue
                if (used[chip.chip_id] + expert.weight_bytes >
                        chip.usable_weight_bytes):
                    continue
                candidates.append((
                    -(count / expert.weight_bytes),
                    used[chip.chip_id] / chip.usable_weight_bytes,
                    key,
                    chip.chip_id,
                    count,
                    expert,
                    chip,
                ))
        if not candidates:
            break
        _, _, _, _, _, expert, chip = min(candidates)
        replica = _replica_with_index(
            expert.key, chip.chip_id, len(replicas)
        )
        replicas.append(replica)
        locations.add((expert.key, chip.chip_id))
        used[chip.chip_id] += expert.weight_bytes

    return ReplicaPlan(expert_list, chip_list, replicas, "ingress-local")


def node_load_replicated(
    experts: Iterable[ExpertSpec],
    chips: Iterable[ChipSpec],
    demand: DemandProfile,
    max_total_replicas: Optional[int] = None,
) -> ReplicaPlan:
    """Add same-node hot replicas only after locality copies are exhausted.

    Phase one reuses :func:`ingress_local`: the first copy of an expert in a
    node is valued as a communication-locality opportunity.  Phase two may
    add that expert to other chips in the same node.  Those later copies are
    valued only by the idealized reduction in assignments per local copy,
    ``d / (c * (c + 1))``.  This cleanly separates locality replication from
    compute-load replication and never duplicates an expert on one chip.
    """

    expert_list = tuple(experts)
    chip_list = tuple(chips)
    if not expert_list or not chip_list:
        raise ValidationError(
            "node-load-replicated needs experts and chips"
        )
    if max_total_replicas is None:
        max_total_replicas = sum(
            chip.usable_weight_bytes for chip in chip_list
        ) // min(expert.weight_bytes for expert in expert_list)
    base = ingress_local(
        expert_list,
        chip_list,
        demand,
        max_total_replicas=max_total_replicas,
    )
    replicas = list(base.replicas)
    used = {
        chip_id: usage.weight_bytes
        for chip_id, usage in base.capacity_usage.items()
    }
    expert_map = {expert.key: expert for expert in expert_list}
    chip_map = {chip.chip_id: chip for chip in chip_list}
    locations = {
        (replica.expert, replica.chip_id)
        for replica in replicas
    }

    while len(replicas) < max_total_replicas:
        node_copy_count = {}
        for replica in replicas:
            node_id = chip_map[replica.chip_id].node_id
            pair = (node_id, replica.expert)
            node_copy_count[pair] = node_copy_count.get(pair, 0) + 1
        candidates = []
        for (node_id, key), count in demand.node_assignments.items():
            copies = node_copy_count.get((node_id, key), 0)
            # A zero here would be a remaining locality opportunity.  The
            # ingress-local phase should have consumed it unless infeasible.
            if count <= 0 or copies <= 0 or key not in expert_map:
                continue
            expert = expert_map[key]
            marginal = count / (copies * (copies + 1))
            for chip in chip_list:
                if chip.node_id != node_id:
                    continue
                if (key, chip.chip_id) in locations:
                    continue
                if (used[chip.chip_id] + expert.weight_bytes >
                        chip.usable_weight_bytes):
                    continue
                candidates.append((
                    -(marginal / expert.weight_bytes),
                    used[chip.chip_id] / chip.usable_weight_bytes,
                    key,
                    chip.chip_id,
                    expert,
                    chip,
                ))
        if not candidates:
            break
        _, _, _, _, expert, chip = min(candidates)
        replica = _replica_with_index(
            expert.key, chip.chip_id, len(replicas)
        )
        replicas.append(replica)
        locations.add((expert.key, chip.chip_id))
        used[chip.chip_id] += expert.weight_bytes

    return ReplicaPlan(
        expert_list, chip_list, replicas, "node-load-replicated"
    )


def layer_packed(
    experts: Iterable[ExpertSpec],
    chips: Iterable[ChipSpec],
) -> ReplicaPlan:
    """Pack same-layer primary replicas before opening another chip."""

    expert_list = tuple(experts)
    chip_list = tuple(chips)
    if not expert_list or not chip_list:
        raise ValidationError("layer-packed needs experts and chips")
    used = {chip.chip_id: 0 for chip in chip_list}
    replicas = []
    current_chip_by_layer = {}
    ordered = sorted(
        expert_list,
        key=lambda expert: (
            expert.key.layer_id,
            expert.key.shared,
            expert.key.expert_id,
        ),
    )
    for expert in ordered:
        candidates = [
            chip for chip in chip_list
            if used[chip.chip_id] + expert.weight_bytes <=
            chip.usable_weight_bytes
        ]
        if not candidates:
            raise ValidationError(
                "layer-packed cannot fit expert %r" % (expert.key,)
            )
        preferred = current_chip_by_layer.get(expert.key.layer_id)
        chip = next(
            (candidate for candidate in candidates
             if candidate.chip_id == preferred),
            None,
        )
        if chip is None:
            # Prefer the fullest feasible chip to minimize the number opened.
            chip = min(
                candidates,
                key=lambda candidate: (
                    -(used[candidate.chip_id] /
                      candidate.usable_weight_bytes),
                    candidate.chip_id,
                ),
            )
        current_chip_by_layer[expert.key.layer_id] = chip.chip_id
        used[chip.chip_id] += expert.weight_bytes
        replicas.append(ReplicaSpec(
            _replica_id(expert.key), expert.key, chip.chip_id, True
        ))
    return ReplicaPlan(
        expert_list, chip_list, replicas, "layer-packed"
    )


def _pair_affinity(demand: DemandProfile) -> Dict[tuple, int]:
    affinity = {}
    for hyperedge, weight in demand.coactivation.items():
        for left, right in itertools.combinations(hyperedge, 2):
            key = (left, right) if left < right else (right, left)
            affinity[key] = affinity.get(key, 0) + weight
    return affinity


def coactivation_clustered(
    experts: Iterable[ExpertSpec],
    chips: Iterable[ChipSpec],
    demand: DemandProfile,
) -> ReplicaPlan:
    """Place one primary per expert to greedily reduce coactivation cuts."""

    expert_list = tuple(experts)
    chip_list = tuple(chips)
    if not expert_list or not chip_list:
        raise ValidationError(
            "coactivation-clustered needs experts and chips"
        )
    expert_map = {expert.key: expert for expert in expert_list}
    affinity = _pair_affinity(demand)
    weighted_degree = {expert.key: 0 for expert in expert_list}
    for (left, right), value in affinity.items():
        if left in weighted_degree:
            weighted_degree[left] += value
        if right in weighted_degree:
            weighted_degree[right] += value
    ordered = sorted(
        expert_list,
        key=lambda expert: (
            -weighted_degree[expert.key],
            -demand.global_for(expert.key),
            expert.key,
        ),
    )
    used = {chip.chip_id: 0 for chip in chip_list}
    residents = {chip.chip_id: [] for chip in chip_list}
    replicas = []
    predicted_load = {chip.chip_id: 0 for chip in chip_list}
    for expert in ordered:
        feasible = [
            chip for chip in chip_list
            if used[chip.chip_id] + expert.weight_bytes <=
            chip.usable_weight_bytes
        ]
        if not feasible:
            raise ValidationError(
                "coactivation-clustered cannot fit expert %r" %
                (expert.key,)
            )

        def score(chip):
            affinity_gain = 0
            for resident in residents[chip.chip_id]:
                pair = (
                    (resident, expert.key)
                    if resident < expert.key
                    else (expert.key, resident)
                )
                affinity_gain += affinity.get(pair, 0)
            return (
                -affinity_gain,
                predicted_load[chip.chip_id],
                used[chip.chip_id] / chip.usable_weight_bytes,
                chip.chip_id,
            )

        chip = min(feasible, key=score)
        used[chip.chip_id] += expert.weight_bytes
        residents[chip.chip_id].append(expert.key)
        predicted_load[chip.chip_id] += demand.global_for(expert.key)
        replicas.append(ReplicaSpec(
            _replica_id(expert.key), expert.key, chip.chip_id, True
        ))
    return ReplicaPlan(
        expert_list, chip_list, replicas, "coactivation-clustered"
    )


def primary_coactivation_cut(
    plan: ReplicaPlan,
    demand: DemandProfile,
    locality_domain: str = "chip",
) -> int:
    """Weighted hyperedge cut of primary replicas."""

    if locality_domain not in ("chip", "node"):
        raise ValidationError("locality_domain must be chip or node")
    location = {}
    for replica in plan.replicas:
        if not replica.is_primary:
            continue
        if locality_domain == "chip":
            location[replica.expert] = replica.chip_id
        else:
            location[replica.expert] = plan.chip(replica.chip_id).node_id
    cut = 0
    for hyperedge, weight in demand.coactivation.items():
        known = [location[key] for key in hyperedge if key in location]
        if len(known) != len(hyperedge):
            continue
        cut += weight * max(0, len(set(known)) - 1)
    return cut


def _hyperedge_cut_vector(
    plan: ReplicaPlan,
    demand: DemandProfile,
):
    chip_locations = {}
    node_locations = {}
    for replica in plan.replicas:
        if not replica.is_primary:
            continue
        chip_locations[replica.expert] = replica.chip_id
        node_locations[replica.expert] = plan.chip(
            replica.chip_id
        ).node_id
    chip_cut = 0
    node_cut = 0
    for hyperedge, weight in demand.coactivation.items():
        if not all(key in chip_locations for key in hyperedge):
            continue
        chip_cut += weight * (
            len({chip_locations[key] for key in hyperedge}) - 1
        )
        node_cut += weight * (
            len({node_locations[key] for key in hyperedge}) - 1
        )
    return node_cut, chip_cut


def full_hyperedge_cut(
    plan: ReplicaPlan,
    demand: DemandProfile,
    node_weight: float = 1.0,
    chip_weight: float = 0.25,
) -> float:
    """Weighted full-hyperedge objective; unlike pair expansion, each
    hyperedge is charged once for its realized node/chip fanout."""

    if node_weight < 0 or chip_weight < 0:
        raise ValidationError("hyperedge cut weights cannot be negative")
    node_cut, chip_cut = _hyperedge_cut_vector(plan, demand)
    return node_weight * node_cut + chip_weight * chip_cut


def hyperedge_clustered(
    experts: Iterable[ExpertSpec],
    chips: Iterable[ChipSpec],
    demand: DemandProfile,
    node_weight: float = 1.0,
    chip_weight: float = 0.25,
    load_weight: float = 0.05,
    max_passes: int = 4,
) -> ReplicaPlan:
    """Greedily improve a balanced placement on the full-hyperedge cut.

    The baseline point is ``unique_balanced``. Each move must remain
    capacity/C-feasible, must not increase the busiest-chip predicted load,
    and must strictly reduce a lexicographic cut-plus-load objective.
    Hyperedges are charged by realized fanout rather than expanded pairs.
    """

    expert_list = tuple(experts)
    chip_list = tuple(chips)
    if not expert_list or not chip_list:
        raise ValidationError("hyperedge_clustered needs experts and chips")
    if max_passes < 0:
        raise ValidationError("max_passes cannot be negative")
    if load_weight < 0 or not math.isfinite(load_weight):
        raise ValidationError("load_weight must be finite and non-negative")

    expected = {
        expert.key: float(demand.global_for(expert.key))
        for expert in expert_list
    }
    current = unique_balanced(expert_list, chip_list, expected)
    if not demand.coactivation:
        return ReplicaPlan(
            expert_list, chip_list, current.replicas,
            "full-hyperedge-clustered",
        )

    chip_load = {chip.chip_id: 0.0 for chip in chip_list}
    for replica in current.replicas:
        chip_load[replica.chip_id] += expected[replica.expert]
    baseline_max_load = max(chip_load.values(), default=0.0)
    edges_by_expert = defaultdict(list)
    for hyperedge, weight in demand.coactivation.items():
        for key in hyperedge:
            edges_by_expert[key].append((hyperedge, weight))

    def node_cut_for_edges(location, edges):
        total = 0
        for hyperedge, weight in edges:
            if all(key in location for key in hyperedge):
                total += weight * (
                    len({location[key] for key in hyperedge}) - 1
                )
        return total

    def chip_cut_for_edges(chip_location, edges):
        total = 0
        for hyperedge, weight in edges:
            if all(key in chip_location for key in hyperedge):
                total += weight * (
                    len({chip_location[key] for key in hyperedge}) - 1
                )
        return total

    for _ in range(max_passes):
        locations = {
            replica.expert: replica.chip_id
            for replica in current.replicas if replica.is_primary
        }
        node_locations = {
            key: current.chip(chip_id).node_id
            for key, chip_id in locations.items()
        }
        used_bytes = {
            chip_id: usage.weight_bytes
            for chip_id, usage in current.capacity_usage.items()
        }
        counts = {
            chip_id: usage.resident_experts
            for chip_id, usage in current.capacity_usage.items()
        }
        best = None
        for expert in sorted(expert_list, key=lambda value: value.key):
            source = locations[expert.key]
            edges = edges_by_expert.get(expert.key, ())
            old_node = node_cut_for_edges(node_locations, edges)
            old_chip = chip_cut_for_edges(locations, edges)
            for target in sorted(chip_list, key=lambda value: value.chip_id):
                if target.chip_id == source:
                    continue
                if (
                    used_bytes[target.chip_id] + expert.weight_bytes >
                    target.usable_weight_bytes
                ):
                    continue
                if (
                    target.max_experts is not None and
                    counts[target.chip_id] + 1 > target.max_experts
                ):
                    continue
                node_locations[expert.key] = target.node_id
                locations[expert.key] = target.chip_id
                new_node = node_cut_for_edges(node_locations, edges)
                new_chip = chip_cut_for_edges(locations, edges)
                node_locations[expert.key] = current.chip(source).node_id
                locations[expert.key] = source
                cut_delta = (
                    node_weight * (new_node - old_node) +
                    chip_weight * (new_chip - old_chip)
                )
                old_source_load = chip_load[source]
                old_target_load = chip_load[target.chip_id]
                new_source_load = old_source_load - expected[expert.key]
                new_target_load = old_target_load + expected[expert.key]
                if new_target_load > baseline_max_load + 1e-12:
                    continue
                load_delta = load_weight * (
                    new_source_load * new_source_load +
                    new_target_load * new_target_load -
                    old_source_load * old_source_load -
                    old_target_load * old_target_load
                )
                objective_delta = cut_delta + load_delta
                candidate_key = (
                    objective_delta,
                    expert.key,
                    source,
                    target.chip_id,
                )
                if objective_delta < -1e-12 and (
                    best is None or candidate_key < best[0]
                ):
                    best = (
                        candidate_key, expert, source, target,
                        cut_delta, load_delta,
                        new_source_load, new_target_load,
                    )
        if best is None:
            break
        _, expert, source, target, cut_delta, load_delta, src_load, dst_load = best
        moved = []
        for replica in current.replicas:
            if replica.is_primary and replica.expert == expert.key:
                moved.append(ReplicaSpec(
                    _replica_id(expert.key),
                    expert.key,
                    target.chip_id,
                    True,
                ))
            else:
                moved.append(replica)
        candidate = ReplicaPlan(
            expert_list, chip_list, moved, "full-hyperedge-clustered"
        )
        current = candidate
        chip_load[source] = src_load
        chip_load[target.chip_id] = dst_load

    return ReplicaPlan(
        expert_list, chip_list, current.replicas,
        "full-hyperedge-clustered",
    )


def joint_greedy(
    primary_plan: ReplicaPlan,
    objective: Callable[[ReplicaPlan], float],
    max_total_replicas: int,
    min_improvement: float = 0.0,
) -> ReplicaPlan:
    """Add replicas using the caller's complete event-model objective.

    ``objective`` must be a deterministic scalar to minimize, normally layer
    completion or a declared latency/energy composite evaluated on the
    placement window.  Every candidate is reconstructed and capacity-checked.
    """

    if max_total_replicas < len(primary_plan.replicas):
        raise ValidationError(
            "max_total_replicas is below primary replica count"
        )
    if min_improvement < 0 or not math.isfinite(min_improvement):
        raise ValidationError("min_improvement must be finite and >= 0")
    experts = primary_plan.experts
    chips = primary_plan.chips
    replicas = list(primary_plan.replicas)
    current = ReplicaPlan(
        experts, chips, replicas, "joint-greedy"
    )
    current_value = float(objective(current))
    if not math.isfinite(current_value):
        raise ValidationError("joint objective must return finite values")

    while len(replicas) < max_total_replicas:
        locations = {
            (replica.expert, replica.chip_id) for replica in replicas
        }
        candidates = []
        for expert in sorted(experts, key=lambda item: item.key):
            for chip in sorted(chips, key=lambda item: item.chip_id):
                if (expert.key, chip.chip_id) in locations:
                    continue
                candidate_replica = _replica_with_index(
                    expert.key, chip.chip_id, len(replicas)
                )
                try:
                    candidate_plan = ReplicaPlan(
                        experts, chips,
                        replicas + [candidate_replica],
                        "joint-greedy",
                    )
                except ValidationError:
                    continue
                value = float(objective(candidate_plan))
                if not math.isfinite(value):
                    raise ValidationError(
                        "joint objective must return finite values"
                    )
                candidates.append((
                    value, expert.key, chip.chip_id,
                    candidate_replica, candidate_plan,
                ))
        if not candidates:
            break
        value, _, _, replica, candidate_plan = min(candidates)
        improvement = current_value - value
        if improvement <= min_improvement:
            break
        replicas.append(replica)
        current = candidate_plan
        current_value = value
    return current
