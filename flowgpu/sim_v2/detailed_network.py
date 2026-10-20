"""Event-driven max-min fair flow sharing for Simulator v2.

The model is fluid rather than packet-level.  Whenever a flow starts or
finishes, rates are recomputed with progressive filling across all resources
used by its path:

* directed full-duplex link bandwidth;
* shared half-duplex link-group bandwidth;
* optional source injection bandwidth;
* optional destination receive bandwidth;
* optional per-directed-link ``max_inflight`` admission.

Propagation latency is paid before a flow becomes eligible for bandwidth.
This module is the detailed network foundation; switch buffers and packet
head-of-line blocking are intentionally not approximated silently.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Set, Tuple

from .network_model import Flow, resolve_flow_route
from .schema import ValidationError
from .topology import Route, Topology


_EPS_TIME = 1e-15
_EPS_BYTES = 1e-9


def _next_positive_float(value: float) -> float:
    """Return the next representable positive double above ``value``."""

    if hasattr(math, "nextafter"):
        return math.nextafter(value, math.inf)
    if value < 0 or not math.isfinite(value):
        raise ValidationError("next-float helper requires a finite value >= 0")
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return struct.unpack(">d", struct.pack(">Q", bits + 1))[0]


@dataclass(frozen=True)
class DetailedFlowResult:
    flow: Flow
    route: Route
    release_s: float
    first_service_s: float
    finish_s: float
    unloaded_serialization_s: float
    propagation_s: float
    contention_delay_s: float
    energy_j: float

    @property
    def total_time_s(self) -> float:
        return self.finish_s - self.flow.earliest_start_s


@dataclass
class _FlowState:
    flow: Flow
    route: Route
    resources: Tuple[str, ...]
    release_s: float
    remaining_bytes: float
    first_service_s: Optional[float] = None
    finish_s: Optional[float] = None
    status: str = "pending"


class FluidNetwork:
    """Incremental max-min fair network state for integration with compute."""

    model_name = "max-min-fluid"
    is_detailed = True

    def __init__(self, topology: Topology, metric: str = "latency") -> None:
        if metric not in ("latency", "hops"):
            raise ValidationError("routing metric must be latency or hops")
        self.topology = topology
        self.metric = metric
        self.now_s = 0.0
        self._states: Dict[str, _FlowState] = {}
        self._results: Dict[str, DetailedFlowResult] = {}
        self._active: Set[str] = set()
        self._resource_capacity = self._build_resource_capacities()
        self._link_bytes = {
            link_id: 0 for link_id in topology.links
        }
        self._inflight_counts = {
            link_id: 0 for link_id, link in topology.links.items()
            if link.max_inflight is not None
        }

    def _build_resource_capacities(self) -> Dict[str, float]:
        capacities = {}
        for link in self.topology.links.values():
            if link.duplex == "half":
                resource = "half:" + link.duplex_group
            else:
                resource = "link:" + link.link_id
            capacity = link.effective_bandwidth
            if resource in capacities:
                capacities[resource] = min(capacities[resource], capacity)
            else:
                capacities[resource] = capacity
        for endpoint in self.topology.endpoints.values():
            if endpoint.injection_bandwidth_bytes_per_s is not None:
                capacities["inject:" + endpoint.endpoint_id] = (
                    endpoint.injection_bandwidth_bytes_per_s
                )
            if endpoint.receive_bandwidth_bytes_per_s is not None:
                capacities["receive:" + endpoint.endpoint_id] = (
                    endpoint.receive_bandwidth_bytes_per_s
                )
        return capacities

    @property
    def results(self) -> Mapping[str, DetailedFlowResult]:
        return dict(self._results)

    @property
    def link_bytes(self) -> Mapping[str, int]:
        return dict(self._link_bytes)

    @property
    def unfinished_count(self) -> int:
        return len(self._states) - len(self._results)

    def _resources_for(self, flow: Flow, route: Route) -> Tuple[str, ...]:
        resources = []
        seen = set()
        for link in route.links:
            if link.duplex == "half":
                resource = "half:" + link.duplex_group
            else:
                resource = "link:" + link.link_id
            if resource not in seen:
                resources.append(resource)
                seen.add(resource)
        source = self.topology.endpoints[flow.src]
        destination = self.topology.endpoints[flow.dst]
        if source.injection_bandwidth_bytes_per_s is not None:
            resources.append("inject:" + flow.src)
        if destination.receive_bandwidth_bytes_per_s is not None:
            resources.append("receive:" + flow.dst)
        return tuple(resources)

    def add_flow(self, flow: Flow) -> None:
        if flow.flow_id in self._states:
            raise ValidationError("duplicate flow id: %s" % flow.flow_id)
        if flow.earliest_start_s < self.now_s - _EPS_TIME:
            raise ValidationError(
                "new flow starts before current network time"
            )
        route = resolve_flow_route(self.topology, flow, self.metric)
        release = flow.earliest_start_s + route.latency_s
        state = _FlowState(
            flow=flow,
            route=route,
            resources=self._resources_for(flow, route),
            release_s=release,
            remaining_bytes=float(flow.nbytes),
        )
        self._states[flow.flow_id] = state
        for link in route.links:
            self._link_bytes[link.link_id] += flow.nbytes
        self._activate_due()

    def _complete(self, state: _FlowState, finish_s: float) -> None:
        if state.flow.flow_id in self._results:
            return
        was_active = state.status == "active"
        state.status = "complete"
        state.finish_s = finish_s
        self._active.discard(state.flow.flow_id)
        if was_active:
            for link in state.route.links:
                if link.max_inflight is not None:
                    self._inflight_counts[link.link_id] -= 1
                    if self._inflight_counts[link.link_id] < 0:
                        raise RuntimeError("negative max_inflight occupancy")
        if state.first_service_s is None:
            state.first_service_s = finish_s
        bottleneck = state.route.bottleneck_bandwidth
        unloaded = (
            0.0 if not state.route.links or state.flow.nbytes == 0
            else state.flow.nbytes / bottleneck
        )
        ideal = state.flow.earliest_start_s + state.route.latency_s + unloaded
        contention = max(0.0, finish_s - ideal)
        self._results[state.flow.flow_id] = DetailedFlowResult(
            flow=state.flow,
            route=state.route,
            release_s=state.release_s,
            first_service_s=state.first_service_s,
            finish_s=finish_s,
            unloaded_serialization_s=unloaded,
            propagation_s=state.route.latency_s,
            contention_delay_s=contention,
            energy_j=(
                state.flow.nbytes * state.route.energy_per_byte
            ),
        )

    def _activate_due(self) -> None:
        ordered = sorted(
            self._states,
            key=lambda flow_id: (
                self._states[flow_id].release_s, flow_id
            ),
        )
        for flow_id in ordered:
            state = self._states[flow_id]
            if (state.status == "pending" and
                    state.release_s <= self.now_s + _EPS_TIME):
                if state.remaining_bytes <= _EPS_BYTES or not state.resources:
                    self._complete(state, max(self.now_s, state.release_s))
                else:
                    limited_links = [
                        link for link in state.route.links
                        if link.max_inflight is not None
                    ]
                    if any(
                        self._inflight_counts[link.link_id] >=
                        link.max_inflight
                        for link in limited_links
                    ):
                        continue
                    state.status = "active"
                    self._active.add(flow_id)
                    for link in limited_links:
                        self._inflight_counts[link.link_id] += 1

    def _rates(self) -> Dict[str, float]:
        """Compute unweighted max-min fair rates by progressive filling."""

        rates = {flow_id: 0.0 for flow_id in self._active}
        unfrozen = set(self._active)
        residual = dict(self._resource_capacity)
        while unfrozen:
            counts = {
                resource: sum(
                    resource in self._states[flow_id].resources
                    for flow_id in unfrozen
                )
                for resource in residual
            }
            candidates = [
                (residual[resource] / count, resource)
                for resource, count in counts.items()
                if count > 0
            ]
            if not candidates:
                raise RuntimeError("active flow has no bandwidth resource")
            delta, limiting_resource = min(candidates)
            if delta < -_EPS_BYTES:
                raise RuntimeError("negative residual network capacity")
            delta = max(0.0, delta)
            for flow_id in unfrozen:
                rates[flow_id] += delta
            for resource, count in counts.items():
                if count:
                    residual[resource] = max(
                        0.0, residual[resource] - delta * count
                    )
            saturated = {
                resource for resource, count in counts.items()
                if count and residual[resource] <= _EPS_BYTES
            }
            frozen = {
                flow_id for flow_id in unfrozen
                if any(resource in saturated
                       for resource in self._states[flow_id].resources)
            }
            if not frozen:
                frozen = {
                    flow_id for flow_id in unfrozen
                    if limiting_resource in
                    self._states[flow_id].resources
                }
            unfrozen -= frozen
        return rates

    def next_event_time(self) -> float:
        self._activate_due()
        next_release = min(
            (
                state.release_s for state in self._states.values()
                if (state.status == "pending" and
                    state.release_s > self.now_s + _EPS_TIME)
            ),
            default=math.inf,
        )
        next_finish = math.inf
        if self._active:
            rates = self._rates()
            for flow_id in self._active:
                rate = rates[flow_id]
                if rate <= 0:
                    raise RuntimeError("active flow received zero fair-share rate")
                state = self._states[flow_id]
                finish = self.now_s + state.remaining_bytes / rate
                # A tiny serialization interval can fall below one binary64
                # ULP at a non-zero absolute time. Returning now_s would make
                # the event loop retry forever without consuming any bytes.
                if finish <= self.now_s:
                    finish = _next_positive_float(self.now_s)
                next_finish = min(
                    next_finish,
                    finish,
                )
        return min(next_release, next_finish)

    def advance_to(self, timestamp_s: float) -> Tuple[DetailedFlowResult, ...]:
        if not math.isfinite(timestamp_s) or timestamp_s < self.now_s:
            raise ValidationError(
                "network advance target must be finite and monotonic"
            )
        next_internal = self.next_event_time()
        if timestamp_s > next_internal + _EPS_TIME:
            raise ValidationError(
                "cannot advance across an unprocessed network event"
            )
        rates = self._rates() if self._active else {}
        delta = timestamp_s - self.now_s
        # The rounded absolute comparison detects sub-ULP completion events;
        # the duration comparison handles the ordinary case. This avoids a
        # scale-dependent artificial epsilon.
        finishing = {
            flow_id
            for flow_id, rate in rates.items()
            if (
                rate > 0 and (
                    self._states[flow_id].remaining_bytes / rate <= delta or
                    self.now_s +
                    self._states[flow_id].remaining_bytes / rate
                    <= timestamp_s
                )
            )
        }
        for flow_id, rate in rates.items():
            state = self._states[flow_id]
            if rate > 0 and state.first_service_s is None:
                state.first_service_s = self.now_s
            if flow_id in finishing:
                state.remaining_bytes = 0.0
            else:
                state.remaining_bytes = max(
                    0.0, state.remaining_bytes - rate * delta
                )
        self.now_s = timestamp_s
        completed = []
        for flow_id in sorted(tuple(self._active)):
            state = self._states[flow_id]
            if state.remaining_bytes <= _EPS_BYTES:
                self._complete(state, self.now_s)
                completed.append(self._results[flow_id])
        before = set(self._results)
        self._activate_due()
        for flow_id in sorted(set(self._results) - before):
            completed.append(self._results[flow_id])
        completed.sort(key=lambda result: result.flow.flow_id)
        return tuple(completed)

    def run(self) -> Mapping[str, DetailedFlowResult]:
        while self.unfinished_count:
            event_time = self.next_event_time()
            if not math.isfinite(event_time):
                raise RuntimeError("network deadlock: unfinished flows remain")
            self.advance_to(event_time)
        return self.results


def simulate_flows(
    topology: Topology,
    flows: Iterable[Flow],
    metric: str = "latency",
) -> Tuple[Mapping[str, DetailedFlowResult], Mapping[str, int]]:
    network = FluidNetwork(topology, metric)
    for flow in sorted(
        flows, key=lambda item: (item.earliest_start_s, item.flow_id)
    ):
        network.add_flow(flow)
    return network.run(), network.link_bytes
