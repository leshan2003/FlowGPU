"""Replica selection policies for a static v2 placement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

from .network_model import Flow, estimate_unloaded
from .replicas import ReplicaPlan
from .schema import Assignment, ExpertKey, ReplicaSpec, ValidationError
from .topology import Topology


DISPATCH_POLICIES = frozenset((
    "nearest",
    "least_loaded",
    "earliest_finish",
    "brain-first",
    "split",
))
DISPATCH_MODES = frozenset(("gpu-direct", "gpu-to-home", "home-resident"))


@dataclass(frozen=True)
class AssignmentBinding:
    assignment: Assignment
    replica: ReplicaSpec
    dispatch_s: float
    combine_s: float
    predicted_batch_m: int
    predicted_finish_s: float


@dataclass(frozen=True)
class ReplicaBatch:
    replica: ReplicaSpec
    assignments: Tuple[Assignment, ...]
    dispatch_arrival_s: float
    compute_start_s: float
    compute_finish_s: float
    kernel_warning: str = ""


@dataclass(frozen=True)
class DispatchPlan:
    policy: str
    bindings: Tuple[AssignmentBinding, ...]
    batches: Tuple[ReplicaBatch, ...]

    @property
    def assignment_count(self) -> int:
        return len(self.bindings)


class Dispatcher:
    """Select replicas without moving or inventing expert weights.

    The dispatcher batches all assignments bound to the same replica in the
    current scheduling window and evaluates calibrated ``t_expert(M)``.
    Chip-wide compute contention is handled by the later event engine; this
    class models replica queue availability and network distance.
    """

    def __init__(
        self,
        replica_plan: ReplicaPlan,
        topology: Topology,
        home_endpoints: Mapping[str, str],
        chip_endpoints: Mapping[str, str],
        kernel_models: Mapping[ExpertKey, object],
        hidden_bytes: int,
        policy: str = "earliest_finish",
        ingress_source_endpoint: str = None,
        dispatch_mode: str = "gpu-to-home",
        gpu_ingress_endpoints: Iterable[str] = (),
        gpu_endpoint_ids: Iterable[str] = (),
    ) -> None:
        if policy not in DISPATCH_POLICIES:
            raise ValidationError("unknown dispatcher policy: %s" % policy)
        if dispatch_mode not in DISPATCH_MODES:
            raise ValidationError("unknown dispatch mode: %s" % dispatch_mode)
        if hidden_bytes <= 0:
            raise ValidationError("hidden_bytes must be positive")
        self.replica_plan = replica_plan
        self.topology = topology
        self.home_endpoints = dict(home_endpoints)
        self.chip_endpoints = dict(chip_endpoints)
        self.kernel_models = dict(kernel_models)
        self.hidden_bytes = hidden_bytes
        self.policy = policy
        self.ingress_source_endpoint = ingress_source_endpoint
        self.dispatch_mode = dispatch_mode
        self.gpu_ingress_endpoints = tuple(gpu_ingress_endpoints)
        self.gpu_endpoint_ids = frozenset(gpu_endpoint_ids)
        if (ingress_source_endpoint is not None and
                ingress_source_endpoint not in topology.endpoints):
            raise ValidationError(
                "ingress source endpoint is not in topology: %s" %
                ingress_source_endpoint
            )
        for chip in replica_plan.chips:
            if chip.chip_id not in self.chip_endpoints:
                raise ValidationError(
                    "missing topology endpoint for chip %s" % chip.chip_id
                )

    def kernel_model(self, expert: ExpertKey, replica: ReplicaSpec):
        model = self.kernel_models.get((expert, replica.chip_id))
        if model is None:
            model = self.kernel_models.get(expert)
        if model is None:
            raise ValidationError(
                "missing kernel model for expert %r on chip %s" %
                (expert, replica.chip_id)
            )
        return model

    def _communication(
        self, assignment: Assignment, replica: ReplicaSpec
    ) -> Tuple[float, float]:
        try:
            home = self.home_endpoints[assignment.home_node]
        except KeyError as exc:
            raise ValidationError(
                "no ingress endpoint for home node %s" %
                assignment.home_node
            ) from exc
        chip = self.chip_endpoints[replica.chip_id]
        source = self._ingress_source(assignment)
        if (
            self.dispatch_mode == "gpu-direct" and
            chip in self.gpu_endpoint_ids
        ):
            # A GPU-resident expert must execute on the ingress GPU that owns
            # the activation. Remote GPU replicas are not silently selected.
            return (
                (0.0, 0.0)
                if source is not None and chip == source
                else (math.inf, math.inf)
            )
        if (source is not None and chip == source):
            # The activation and GPU-resident fallback expert share the GPU
            # endpoint; routing through home and back would double-count the
            # bridge and create an artificial loop.
            return 0.0, 0.0
        if self.dispatch_mode == "gpu-direct":
            if source is None:
                raise ValidationError(
                    "gpu-direct dispatch requires an ingress source"
                )
            dispatch_local = estimate_unloaded(
                self.topology,
                Flow(
                    "dispatch:%s:%s" %
                    (assignment.assignment_id, replica.replica_id),
                    source, chip, self.hidden_bytes,
                ),
            )
            combine_local = estimate_unloaded(
                self.topology,
                Flow(
                    "combine:%s:%s" %
                    (assignment.assignment_id, replica.replica_id),
                    chip, source, self.hidden_bytes,
                    traffic_class="combine",
                ),
            )
            return dispatch_local.duration_s, combine_local.duration_s
        dispatch_local = estimate_unloaded(
            self.topology,
            Flow(
                "dispatch:%s:%s" %
                (assignment.assignment_id, replica.replica_id),
                home, chip, self.hidden_bytes,
            ),
        )
        combine_local = estimate_unloaded(
            self.topology,
            Flow(
                "combine:%s:%s" %
                (assignment.assignment_id, replica.replica_id),
                chip, home, self.hidden_bytes,
                traffic_class="combine",
            ),
        )
        dispatch_s = dispatch_local.duration_s
        combine_s = combine_local.duration_s
        if (self.dispatch_mode == "gpu-to-home" and
                source is not None):
            # Force the route through the record's home/ingress node.  A plain
            # shortest path from the GPU would be free to choose a different
            # ingress when several exist, destroying node-conditional demand.
            bridge_out = estimate_unloaded(
                self.topology,
                Flow(
                    "ingress:%s" % assignment.assignment_id,
                    source, home, self.hidden_bytes,
                    traffic_class="bridge-dispatch",
                ),
            )
            bridge_back = estimate_unloaded(
                self.topology,
                Flow(
                    "egress:%s" % assignment.assignment_id,
                    home, source, self.hidden_bytes,
                    traffic_class="bridge-combine",
                ),
            )
            dispatch_s += bridge_out.duration_s
            combine_s += bridge_back.duration_s
        return dispatch_s, combine_s

    def _ingress_source(self, assignment: Assignment):
        if self.gpu_ingress_endpoints:
            index = int(assignment.request_id) % len(
                self.gpu_ingress_endpoints
            )
            return self.gpu_ingress_endpoints[index]
        return self.ingress_source_endpoint

    def flow_endpoints(self, assignment: Assignment, replica: ReplicaSpec):
        """Return (source, destination, waypoints) for dispatch.

        Combine is the reverse path with the same waypoints.
        """

        home = self.home_endpoints[assignment.home_node]
        chip = self.chip_endpoints[replica.chip_id]
        source = self._ingress_source(assignment)
        if self.dispatch_mode == "home-resident":
            return home, chip, tuple()
        if source is None:
            return home, chip, tuple()
        if source == chip:
            return source, chip, tuple()
        if self.dispatch_mode == "gpu-direct":
            return source, chip, tuple()
        return source, chip, (home,)

    def plan(
        self,
        assignments: Iterable[Assignment],
        replica_ready_s: Mapping[str, float] = None,
    ) -> DispatchPlan:
        items = tuple(assignments)
        if len({item.assignment_id for item in items}) != len(items):
            raise ValidationError("assignment ids must be unique")
        ready = dict(replica_ready_s or {})
        assigned: Dict[str, list] = {
            replica.replica_id: [] for replica in self.replica_plan.replicas
        }
        max_arrival = {replica_id: 0.0 for replica_id in assigned}
        provisional = []

        for assignment in sorted(
            items,
            key=lambda item: (
                item.step, item.layer_id, item.token_id, item.assignment_id
            ),
        ):
            candidates = []
            for replica in self.replica_plan.replicas_for(
                    assignment.expert):
                model = self.kernel_model(
                    assignment.expert, replica
                )
                dispatch_s, combine_s = self._communication(
                    assignment, replica
                )
                count = len(assigned[replica.replica_id]) + 1
                prediction = model.predict(count)
                arrival = dispatch_s
                finish = (
                    max(
                        ready.get(replica.replica_id, 0.0),
                        max(max_arrival[replica.replica_id], arrival),
                    )
                    + prediction.latency_s
                    + combine_s
                )
                if self.policy == "nearest":
                    score = (
                        dispatch_s + combine_s,
                        len(assigned[replica.replica_id]),
                        finish,
                        replica.replica_id,
                    )
                elif self.policy == "brain-first":
                    is_brain = self.replica_plan.chip(
                        replica.chip_id
                    ).role.startswith("brain")
                    score = (
                        0 if is_brain else 1,
                        finish,
                        dispatch_s + combine_s,
                        replica.replica_id,
                    )
                elif self.policy == "split":
                    score = (
                        len(assigned[replica.replica_id]),
                        finish,
                        dispatch_s + combine_s,
                        replica.replica_id,
                    )
                elif self.policy == "least_loaded":
                    score = (
                        len(assigned[replica.replica_id]),
                        ready.get(replica.replica_id, 0.0),
                        dispatch_s + combine_s,
                        replica.replica_id,
                    )
                else:
                    score = (
                        finish,
                        dispatch_s + combine_s,
                        len(assigned[replica.replica_id]),
                        replica.replica_id,
                    )
                candidates.append(
                    (score, replica, dispatch_s, combine_s, finish)
                )
            _, replica, dispatch_s, combine_s, finish = min(
                candidates, key=lambda candidate: candidate[0]
            )
            assigned[replica.replica_id].append(assignment)
            max_arrival[replica.replica_id] = max(
                max_arrival[replica.replica_id], dispatch_s
            )
            provisional.append((
                assignment, replica, dispatch_s, combine_s, finish
            ))

        batches = []
        final_times = {}
        warnings = {}
        for replica in sorted(
                self.replica_plan.replicas,
                key=lambda item: item.replica_id,
        ):
            batch_assignments = tuple(assigned[replica.replica_id])
            if not batch_assignments:
                continue
            model = self.kernel_model(replica.expert, replica)
            prediction = model.predict(len(batch_assignments))
            start = max(
                ready.get(replica.replica_id, 0.0),
                max_arrival[replica.replica_id],
            )
            finish = start + prediction.latency_s
            final_times[replica.replica_id] = finish
            warnings[replica.replica_id] = prediction.warning or ""
            batches.append(ReplicaBatch(
                replica=replica,
                assignments=batch_assignments,
                dispatch_arrival_s=max_arrival[replica.replica_id],
                compute_start_s=start,
                compute_finish_s=finish,
                kernel_warning=prediction.warning or "",
            ))

        bindings = tuple(AssignmentBinding(
            assignment=assignment,
            replica=replica,
            dispatch_s=dispatch_s,
            combine_s=combine_s,
            predicted_batch_m=len(assigned[replica.replica_id]),
            predicted_finish_s=(
                final_times[replica.replica_id] + combine_s
            ),
        ) for assignment, replica, dispatch_s, combine_s, _ in provisional)
        if len(bindings) != len(items):
            raise RuntimeError("dispatcher lost assignments")
        return DispatchPlan(self.policy, bindings, tuple(batches))
