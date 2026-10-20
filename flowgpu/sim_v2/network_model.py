"""Flow primitives over the explicit v2 topology.

``estimate_unloaded`` is an unloaded lower-bound estimate.
``SerializedPathScheduler`` is a deterministic, conservative reference that
serializes flows sharing any path link.  It is useful for invariants and small
hand checks, but is intentionally not presented as the eventual detailed
fluid-sharing network model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

from .schema import ValidationError
from .topology import Route, Topology


@dataclass(frozen=True)
class Flow:
    flow_id: str
    src: str
    dst: str
    nbytes: int
    earliest_start_s: float = 0.0
    traffic_class: str = "dispatch"
    waypoints: Tuple[str, ...] = tuple()
    route_link_ids: Tuple[str, ...] = tuple()

    def __post_init__(self) -> None:
        if not self.flow_id or not self.src or not self.dst:
            raise ValidationError("flow id and endpoints must be non-empty")
        if self.nbytes < 0:
            raise ValidationError("flow bytes cannot be negative")
        if (not math.isfinite(self.earliest_start_s) or
                self.earliest_start_s < 0):
            raise ValidationError("flow earliest start must be finite and >= 0")
        if not self.traffic_class:
            raise ValidationError("flow traffic_class must be non-empty")
        if any(not waypoint for waypoint in self.waypoints):
            raise ValidationError("flow waypoints must be non-empty ids")
        if any(not link_id for link_id in self.route_link_ids):
            raise ValidationError("explicit route link ids must be non-empty")
        if self.waypoints and self.route_link_ids:
            raise ValidationError(
                "flow cannot use both waypoints and an explicit route"
            )


def resolve_flow_route(
    topology: Topology,
    flow: Flow,
    metric: str,
) -> Route:
    if flow.route_link_ids:
        return topology.explicit_route(
            flow.src, flow.dst, flow.route_link_ids
        )
    return topology.route_through(
        flow.src, flow.dst, flow.waypoints, metric
    )


@dataclass(frozen=True)
class FlowResult:
    flow: Flow
    route: Route
    start_s: float
    finish_s: float
    queue_s: float
    serialization_s: float
    propagation_s: float
    energy_j: float

    @property
    def duration_s(self) -> float:
        return self.finish_s - self.start_s


def estimate_unloaded(
    topology: Topology, flow: Flow, metric: str = "latency"
) -> FlowResult:
    route = resolve_flow_route(topology, flow, metric)
    serialization = (
        0.0 if not route.links or flow.nbytes == 0
        else flow.nbytes / route.bottleneck_bandwidth
    )
    propagation = route.latency_s
    finish = flow.earliest_start_s + propagation + serialization
    return FlowResult(
        flow=flow,
        route=route,
        start_s=flow.earliest_start_s,
        finish_s=finish,
        queue_s=0.0,
        serialization_s=serialization,
        propagation_s=propagation,
        energy_j=flow.nbytes * route.energy_per_byte,
    )


class SerializedPathScheduler:
    """Deterministic path-reservation baseline with auditable byte counters."""

    model_name = "serialized-path-reference"
    is_detailed = False

    def __init__(self, topology: Topology, metric: str = "latency") -> None:
        if metric not in ("latency", "hops"):
            raise ValidationError("routing metric must be latency or hops")
        self.topology = topology
        self.metric = metric
        self._link_ready: Dict[str, float] = {
            link_id: 0.0 for link_id in topology.links
        }
        self._link_bytes: Dict[str, int] = {
            link_id: 0 for link_id in topology.links
        }

    @property
    def link_bytes(self) -> Dict[str, int]:
        return dict(self._link_bytes)

    def schedule(self, flow: Flow) -> FlowResult:
        route = resolve_flow_route(self.topology, flow, self.metric)
        ready = max(
            [flow.earliest_start_s] +
            [self._link_ready[link.link_id] for link in route.links]
        )
        serialization = (
            0.0 if not route.links or flow.nbytes == 0
            else flow.nbytes / route.bottleneck_bandwidth
        )
        propagation = route.latency_s
        finish = ready + propagation + serialization
        for link in route.links:
            self._link_ready[link.link_id] = finish
            self._link_bytes[link.link_id] += flow.nbytes
        return FlowResult(
            flow=flow,
            route=route,
            start_s=ready,
            finish_s=finish,
            queue_s=ready - flow.earliest_start_s,
            serialization_s=serialization,
            propagation_s=propagation,
            energy_j=flow.nbytes * route.energy_per_byte,
        )
