"""Explicit directed topology and deterministic static routing."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Tuple

from .schema import Provenance, ValidationError


ENDPOINT_KINDS = frozenset((
    "gpu", "brain_chip", "local_switch", "node_switch", "fabric_switch"
))


class UnreachableError(ValidationError):
    """Raised when no directed path connects two topology endpoints."""


@dataclass(frozen=True)
class Endpoint:
    endpoint_id: str
    kind: str
    node_id: Optional[str] = None
    chip_id: Optional[str] = None
    injection_bandwidth_bytes_per_s: Optional[float] = None
    receive_bandwidth_bytes_per_s: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.endpoint_id:
            raise ValidationError("endpoint_id must be non-empty")
        if self.kind not in ENDPOINT_KINDS:
            raise ValidationError("unsupported endpoint kind: %s" % self.kind)
        if self.kind == "brain_chip" and not self.chip_id:
            raise ValidationError("brain_chip endpoint requires chip_id")
        for name, value in (
            ("injection", self.injection_bandwidth_bytes_per_s),
            ("receive", self.receive_bandwidth_bytes_per_s),
        ):
            if value is not None and (
                    not math.isfinite(value) or value <= 0):
                raise ValidationError(
                    "%s bandwidth must be finite and positive" % name
                )


@dataclass(frozen=True)
class LinkSpec:
    link_id: str
    src: str
    dst: str
    bandwidth_bytes_per_s: float
    latency_s: float
    efficiency: float
    energy_per_byte: float
    duplex: str
    provenance: Provenance
    max_inflight: Optional[int] = None
    duplex_group: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.link_id or not self.src or not self.dst:
            raise ValidationError("link id and endpoints must be non-empty")
        if (not math.isfinite(self.bandwidth_bytes_per_s) or
                self.bandwidth_bytes_per_s <= 0):
            raise ValidationError("link bandwidth must be finite and positive")
        if not math.isfinite(self.latency_s) or self.latency_s < 0:
            raise ValidationError("link latency must be finite and >= 0")
        if not 0 < self.efficiency <= 1:
            raise ValidationError("link efficiency must be in (0, 1]")
        if not math.isfinite(self.energy_per_byte) or self.energy_per_byte < 0:
            raise ValidationError("link energy_per_byte must be finite and >= 0")
        if self.duplex not in ("full", "half"):
            raise ValidationError("link duplex must be 'full' or 'half'")
        if self.max_inflight is not None and self.max_inflight <= 0:
            raise ValidationError("max_inflight must be positive when set")
        if self.duplex == "half" and not self.duplex_group:
            raise ValidationError(
                "half-duplex link requires a shared duplex_group"
            )

    @property
    def effective_bandwidth(self) -> float:
        return self.bandwidth_bytes_per_s * self.efficiency


@dataclass(frozen=True)
class Route:
    src: str
    dst: str
    links: Tuple[LinkSpec, ...]
    metric: str

    @property
    def hop_count(self) -> int:
        return len(self.links)

    @property
    def latency_s(self) -> float:
        return sum(link.latency_s for link in self.links)

    @property
    def bottleneck_bandwidth(self) -> float:
        if not self.links:
            return math.inf
        return min(link.effective_bandwidth for link in self.links)

    @property
    def energy_per_byte(self) -> float:
        return sum(link.energy_per_byte for link in self.links)


class Topology:
    """Validated explicit graph.

    Links are directed even when their physical interface is full duplex.
    Callers must supply both directions when traffic can travel both ways.
    """

    def __init__(
        self,
        endpoints: Iterable[Endpoint],
        links: Iterable[LinkSpec],
    ) -> None:
        endpoint_map = self._unique(
            ((endpoint.endpoint_id, endpoint) for endpoint in endpoints),
            "endpoint",
        )
        link_map = self._unique(
            ((link.link_id, link) for link in links), "link"
        )
        if not endpoint_map:
            raise ValidationError("topology must contain endpoints")
        adjacency: Dict[str, list] = {
            endpoint_id: [] for endpoint_id in endpoint_map
        }
        for link in link_map.values():
            if link.src not in endpoint_map or link.dst not in endpoint_map:
                raise ValidationError(
                    "link %s references an unknown endpoint" % link.link_id
                )
            if link.src == link.dst:
                raise ValidationError(
                    "self-loop link is not allowed: %s" % link.link_id
                )
            adjacency[link.src].append(link)
        for value in adjacency.values():
            value.sort(key=lambda link: link.link_id)
        self._endpoints = MappingProxyType(endpoint_map)
        self._links = MappingProxyType(link_map)
        self._adjacency = MappingProxyType({
            key: tuple(value) for key, value in adjacency.items()
        })

    @staticmethod
    def _unique(items, kind: str) -> dict:
        out = {}
        for key, value in items:
            if key in out:
                raise ValidationError("duplicate %s id: %s" % (kind, key))
            out[key] = value
        return out

    @property
    def endpoints(self) -> Mapping[str, Endpoint]:
        return self._endpoints

    @property
    def links(self) -> Mapping[str, LinkSpec]:
        return self._links

    def route(self, src: str, dst: str, metric: str = "latency") -> Route:
        if src not in self._endpoints or dst not in self._endpoints:
            raise ValidationError("route endpoint is not in topology")
        if metric not in ("latency", "hops"):
            raise ValidationError("routing metric must be latency or hops")
        if src == dst:
            return Route(src, dst, tuple(), metric)

        # Key order makes equal-cost routes deterministic by hop count and
        # lexicographic link-id sequence.
        start_key = (0.0, 0, tuple())
        best = {src: start_key}
        heap = [(start_key, src, tuple())]
        while heap:
            key, current, path = heapq.heappop(heap)
            if key != best.get(current):
                continue
            if current == dst:
                return Route(src, dst, path, metric)
            for link in self._adjacency[current]:
                edge_cost = link.latency_s if metric == "latency" else 1.0
                new_path = path + (link,)
                new_key = (
                    key[0] + edge_cost,
                    key[1] + 1,
                    key[2] + (link.link_id,),
                )
                if link.dst not in best or new_key < best[link.dst]:
                    best[link.dst] = new_key
                    heapq.heappush(heap, (new_key, link.dst, new_path))
        raise UnreachableError("no directed path from %s to %s" % (src, dst))

    def route_through(
        self,
        src: str,
        dst: str,
        waypoints: Iterable[str],
        metric: str = "latency",
    ) -> Route:
        """Route through ordered mandatory waypoints."""

        points = (src,) + tuple(waypoints) + (dst,)
        links = []
        for left, right in zip(points, points[1:]):
            links.extend(self.route(left, right, metric).links)
        return Route(src, dst, tuple(links), metric)

    def explicit_route(
        self,
        src: str,
        dst: str,
        link_ids: Iterable[str],
    ) -> Route:
        """Validate and return an explicitly selected directed path."""

        if src not in self._endpoints or dst not in self._endpoints:
            raise ValidationError("explicit route endpoint is not in topology")
        links = []
        current = src
        for link_id in link_ids:
            try:
                link = self._links[link_id]
            except KeyError as exc:
                raise ValidationError(
                    "explicit route references unknown link %s" % link_id
                ) from exc
            if link.src != current:
                raise ValidationError(
                    "explicit route is not contiguous at link %s" % link_id
                )
            links.append(link)
            current = link.dst
        if current != dst:
            raise ValidationError(
                "explicit route terminates at %s, expected %s" %
                (current, dst)
            )
        return Route(src, dst, tuple(links), "explicit")


@dataclass(frozen=True)
class LinkProfile:
    bandwidth_bytes_per_s: float
    latency_s: float
    efficiency: float
    energy_per_byte: float
    provenance: Provenance
    duplex: str = "full"

    def make(
        self,
        link_id: str,
        src: str,
        dst: str,
        duplex_group: Optional[str] = None,
    ) -> LinkSpec:
        return LinkSpec(
            link_id=link_id,
            src=src,
            dst=dst,
            bandwidth_bytes_per_s=self.bandwidth_bytes_per_s,
            latency_s=self.latency_s,
            efficiency=self.efficiency,
            energy_per_byte=self.energy_per_byte,
            duplex=self.duplex,
            provenance=self.provenance,
            duplex_group=duplex_group,
        )


@dataclass(frozen=True)
class ClusterTopology:
    topology: Topology
    home_endpoints: Mapping[str, str]
    chip_endpoints: Mapping[str, str]
    gpu_endpoint: str
    ingress_nodes: Tuple[str, ...]
    gpu_endpoints: Tuple[str, ...] = ()


def build_hierarchical_cluster(
    n_nodes: int,
    chips_per_node: int,
    ingress_node_indices: Iterable[int],
    scale_up: LinkProfile,
    scale_out: LinkProfile,
    bridge: LinkProfile,
    gpu_ingress_count: int = 1,
) -> ClusterTopology:
    """Expand a regular cluster into an inspectable directed graph.

    The GPU/bridge endpoint connects only to the declared ingress node
    switches.  Token ``home_node`` maps to a concrete node-switch endpoint;
    remote expert service crosses the central fabric when ``n_nodes > 1``.
    """

    if n_nodes <= 0 or chips_per_node <= 0:
        raise ValidationError("node and chips-per-node counts must be positive")
    if gpu_ingress_count <= 0:
        raise ValidationError("gpu ingress count must be positive")
    ingress_indices = tuple(sorted(set(ingress_node_indices)))
    if not ingress_indices:
        raise ValidationError("at least one ingress node is required")
    if any(index < 0 or index >= n_nodes for index in ingress_indices):
        raise ValidationError("ingress node index is outside cluster")

    gpu_endpoints = tuple(
        "gpu%d" % index for index in range(gpu_ingress_count)
    )
    endpoints = [Endpoint(endpoint, "gpu") for endpoint in gpu_endpoints]
    links = []
    if n_nodes > 1:
        endpoints.append(Endpoint("fabric0", "fabric_switch"))
    home_endpoints = {}
    chip_endpoints = {}

    def add_pair(prefix, left, right, profile):
        duplex_group = prefix if profile.duplex == "half" else None
        links.append(profile.make(
            prefix + ":fwd", left, right, duplex_group
        ))
        links.append(profile.make(
            prefix + ":rev", right, left, duplex_group
        ))

    for node_index in range(n_nodes):
        node_id = "node%d" % node_index
        switch_id = node_id + ":switch"
        endpoints.append(Endpoint(
            switch_id, "node_switch", node_id=node_id
        ))
        home_endpoints[node_id] = switch_id
        if n_nodes > 1:
            add_pair(
                "scaleout:%s" % node_id,
                switch_id, "fabric0", scale_out,
            )
        if node_index in ingress_indices:
            for gpu_index, gpu_endpoint in enumerate(gpu_endpoints):
                prefix = (
                    "bridge:%s" % node_id
                    if gpu_ingress_count == 1
                    else "bridge:gpu%d:%s" % (gpu_index, node_id)
                )
                add_pair(prefix, gpu_endpoint, switch_id, bridge)
        for chip_index in range(chips_per_node):
            chip_id = "%s:chip%d" % (node_id, chip_index)
            endpoint_id = chip_id
            endpoints.append(Endpoint(
                endpoint_id, "brain_chip",
                node_id=node_id, chip_id=chip_id,
            ))
            chip_endpoints[chip_id] = endpoint_id
            add_pair(
                "scaleup:%s" % chip_id,
                switch_id, endpoint_id, scale_up,
            )

    ingress_nodes = tuple("node%d" % index for index in ingress_indices)
    return ClusterTopology(
        topology=Topology(endpoints, links),
        home_endpoints=MappingProxyType(home_endpoints),
        chip_endpoints=MappingProxyType(chip_endpoints),
        gpu_endpoint="gpu0",
        ingress_nodes=ingress_nodes,
        gpu_endpoints=gpu_endpoints,
    )
