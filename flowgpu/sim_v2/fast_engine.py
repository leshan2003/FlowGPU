"""Explicit fast approximation and fast/detailed consistency checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

from .dispatcher import Dispatcher
from .layer_engine import MoELayerResult
from .schema import Assignment, RuntimeCapabilities, ValidationError


@dataclass(frozen=True)
class FastReplicaExecution:
    replica_id: str
    chip_id: str
    assignment_count: int
    dispatch_arrival_s: float
    compute_start_s: float
    compute_finish_s: float


@dataclass(frozen=True)
class FastLayerResult:
    layer_id: int
    completion_s: float
    assignment_count: int
    replica_executions: Tuple[FastReplicaExecution, ...]
    logical_dispatch_bytes: int
    logical_combine_bytes: int
    local_assignment_fraction: float
    warnings: Tuple[str, ...]


@dataclass(frozen=True)
class FastDetailedComparison:
    baseline_fast_s: float
    candidate_fast_s: float
    baseline_detailed_s: float
    candidate_detailed_s: float
    fast_speedup: float
    detailed_speedup: float
    direction_consistent: bool
    warning: str


class FastMoELayerEngine:
    """Uncontended-path screening model.

    Fast mode uses the same placement, dispatcher, topology routes, and kernel
    curves as detailed mode, but ignores network contention and overlap among
    flows.  It keeps one compute queue per chip.  Capability-aware
    dedup/reduce/multicast is deliberately excluded because aggregating those
    flows without the event DAG can reverse conclusions.
    """

    def __init__(
        self,
        dispatcher: Dispatcher,
        capabilities: RuntimeCapabilities = RuntimeCapabilities(),
    ) -> None:
        if any((
            capabilities.dispatch_dedup_chip,
            capabilities.dispatch_dedup_node,
            capabilities.combine_reduce_chip,
            capabilities.combine_reduce_node,
            capabilities.hardware_multicast,
        )):
            raise ValidationError(
                "fast mode does not approximate communication capabilities; "
                "use detailed mode"
            )
        self.dispatcher = dispatcher

    def run(
        self,
        assignments: Iterable[Assignment],
        replica_ready_s: Mapping[str, float] = None,
    ) -> FastLayerResult:
        items = tuple(assignments)
        if not items:
            raise ValidationError("fast layer window must contain assignments")
        layers = {assignment.layer_id for assignment in items}
        if len(layers) != 1:
            raise ValidationError("fast layer accepts exactly one layer")
        plan = self.dispatcher.plan(items, replica_ready_s or {})
        bindings_by_replica: Dict[str, list] = {}
        for binding in plan.bindings:
            bindings_by_replica.setdefault(
                binding.replica.replica_id, []
            ).append(binding)
        ready_by_chip: Dict[str, list] = {}
        warnings = {
            "fast mode ignores network contention and flow overlap"
        }
        for replica_id, bindings in bindings_by_replica.items():
            replica = bindings[0].replica
            arrival = max(binding.dispatch_s for binding in bindings)
            ready_by_chip.setdefault(replica.chip_id, []).append(
                (arrival, replica_id, bindings)
            )
        executions = {}
        combine_finish = {}
        for chip_id in sorted(ready_by_chip):
            chip_ready = 0.0
            for arrival, replica_id, bindings in sorted(
                    ready_by_chip[chip_id],
                    key=lambda item: (item[0], item[1])):
                model = self.dispatcher.kernel_models[
                    bindings[0].replica.expert
                ]
                prediction = model.predict(len(bindings))
                if prediction.warning:
                    warnings.add(prediction.warning)
                start = max(chip_ready, arrival)
                finish = start + prediction.latency_s
                chip_ready = finish
                executions[replica_id] = FastReplicaExecution(
                    replica_id=replica_id,
                    chip_id=chip_id,
                    assignment_count=len(bindings),
                    dispatch_arrival_s=arrival,
                    compute_start_s=start,
                    compute_finish_s=finish,
                )
                for binding in bindings:
                    combine_finish[binding.assignment.assignment_id] = (
                        finish + binding.combine_s
                    )
        if len(combine_finish) != len(items):
            raise RuntimeError("fast layer lost assignments")
        local = sum(
            self.dispatcher.replica_plan.chip(
                binding.replica.chip_id
            ).node_id == binding.assignment.home_node
            for binding in plan.bindings
        )
        return FastLayerResult(
            layer_id=next(iter(layers)),
            completion_s=max(combine_finish.values()),
            assignment_count=len(items),
            replica_executions=tuple(
                executions[key] for key in sorted(executions)
            ),
            logical_dispatch_bytes=(
                len(items) * self.dispatcher.hidden_bytes
            ),
            logical_combine_bytes=(
                len(items) * self.dispatcher.hidden_bytes
            ),
            local_assignment_fraction=local / len(items),
            warnings=tuple(sorted(warnings)),
        )


def compare_fast_detailed_direction(
    baseline_fast: FastLayerResult,
    candidate_fast: FastLayerResult,
    baseline_detailed: MoELayerResult,
    candidate_detailed: MoELayerResult,
    tolerance: float = 1e-12,
) -> FastDetailedComparison:
    """Detect whether fast and detailed disagree on which strategy wins."""

    times = (
        baseline_fast.completion_s, candidate_fast.completion_s,
        baseline_detailed.completion_s, candidate_detailed.completion_s,
    )
    if any(not math.isfinite(value) or value <= 0 for value in times):
        raise ValidationError("comparison times must be finite and positive")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValidationError("comparison tolerance must be finite and >= 0")
    fast_delta = candidate_fast.completion_s - baseline_fast.completion_s
    detailed_delta = (
        candidate_detailed.completion_s -
        baseline_detailed.completion_s
    )

    def direction(delta):
        if abs(delta) <= tolerance:
            return 0
        return 1 if delta > 0 else -1

    consistent = direction(fast_delta) == direction(detailed_delta)
    warning = ""
    if not consistent:
        warning = (
            "fast/detailed conclusion direction differs; detailed coverage "
            "must be expanded and fast mode cannot support the claim"
        )
    return FastDetailedComparison(
        baseline_fast_s=baseline_fast.completion_s,
        candidate_fast_s=candidate_fast.completion_s,
        baseline_detailed_s=baseline_detailed.completion_s,
        candidate_detailed_s=candidate_detailed.completion_s,
        fast_speedup=(
            baseline_fast.completion_s / candidate_fast.completion_s
        ),
        detailed_speedup=(
            baseline_detailed.completion_s /
            candidate_detailed.completion_s
        ),
        direction_consistent=consistent,
        warning=warning,
    )

