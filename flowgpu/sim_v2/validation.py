"""Mandatory Tier-1 conservation checks for v2 execution results."""

from __future__ import annotations

from collections import Counter

from .layer_engine import MoELayerResult
from .schema import ValidationError


def _close(left: float, right: float, scale: float = 1.0) -> bool:
    tolerance = 1e-12 * max(scale, abs(left), abs(right))
    return abs(left - right) <= tolerance


def validate_layer_result(result: MoELayerResult) -> None:
    """Raise when a layer result cannot be reconstructed from its events."""

    if result.assignment_count <= 0:
        raise ValidationError("layer result has no assignments")
    executed = sum(
        execution.assignment_count
        for execution in result.replica_executions
    )
    if executed != result.assignment_count:
        raise ValidationError(
            "assignment execution conservation failed: %d != %d" %
            (executed, result.assignment_count)
        )
    for execution in result.replica_executions:
        if execution.dispatch_arrival_s > execution.compute_start_s:
            raise ValidationError("compute starts before dispatch arrival")
        if execution.compute_start_s > execution.compute_finish_s:
            raise ValidationError("compute finish precedes start")
        if not _close(
            execution.compute_queue_s,
            execution.compute_start_s - execution.dispatch_arrival_s,
        ):
            raise ValidationError("compute queue attribution mismatch")

    flows = tuple(result.flow_results.values())
    dispatch = tuple(
        flow for flow in flows if flow.flow.traffic_class == "dispatch"
    )
    combine = tuple(
        flow for flow in flows if flow.flow.traffic_class == "combine"
    )
    if len(dispatch) != result.dispatch_flow_count:
        raise ValidationError("dispatch flow count mismatch")
    if len(combine) != result.combine_flow_count:
        raise ValidationError("combine flow count mismatch")
    if sum(flow.flow.nbytes for flow in dispatch) != \
            result.physical_dispatch_bytes:
        raise ValidationError("physical dispatch byte conservation failed")
    if sum(flow.flow.nbytes for flow in combine) != \
            result.physical_combine_bytes:
        raise ValidationError("physical combine byte conservation failed")
    # Hierarchical node-aware execution may represent scale-out and local
    # fanout as separate flows, so the sum of flow payloads can exceed logical
    # assignment bytes.  Savings must be compared per network tier rather than
    # assuming one end-to-end physical flow per logical assignment.

    reconstructed = Counter()
    for flow in flows:
        if flow.finish_s < flow.flow.earliest_start_s:
            raise ValidationError("flow finished before it was released")
        for link in flow.route.links:
            reconstructed[link.link_id] += flow.flow.nbytes
    all_links = set(reconstructed) | set(result.link_bytes)
    for link_id in all_links:
        if reconstructed[link_id] != result.link_bytes.get(link_id, 0):
            raise ValidationError(
                "link byte conservation failed on %s: %d != %d" %
                (link_id, reconstructed[link_id],
                 result.link_bytes.get(link_id, 0))
            )

    energy = sum(flow.energy_j for flow in flows)
    if not _close(energy, result.network_energy_j):
        raise ValidationError("network energy attribution mismatch")
    contention = sum(flow.contention_delay_s for flow in flows)
    if not _close(contention, result.network_contention_s):
        raise ValidationError("network contention attribution mismatch")
    if not _close(
        result.critical_path.combine_finish_s, result.completion_s
    ):
        raise ValidationError(
            "critical path does not terminate at layer completion"
        )
