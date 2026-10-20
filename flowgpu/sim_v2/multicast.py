"""Explicit store-and-forward multicast trees over the detailed network."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Tuple

from .detailed_network import DetailedFlowResult, FluidNetwork
from .network_model import Flow
from .schema import ValidationError
from .topology import LinkSpec, Topology


@dataclass(frozen=True)
class MulticastTree:
    source: str
    destinations: Tuple[str, ...]
    links: Tuple[LinkSpec, ...]
    root_link_ids: Tuple[str, ...]
    children: Mapping[str, Tuple[str, ...]]


@dataclass(frozen=True)
class MulticastResult:
    tree: MulticastTree
    completion_s: float
    destination_finish_s: Mapping[str, float]
    flow_results: Mapping[str, DetailedFlowResult]
    link_bytes: Mapping[str, int]
    energy_j: float


def build_multicast_tree(
    topology: Topology,
    source: str,
    destinations: Iterable[str],
    metric: str = "latency",
) -> MulticastTree:
    """Union deterministic shortest paths into a directed arborescence."""

    destination_tuple = tuple(sorted(set(destinations)))
    if not destination_tuple:
        raise ValidationError("multicast requires at least one destination")
    if source not in topology.endpoints:
        raise ValidationError("multicast source is not in topology")
    if any(destination not in topology.endpoints
           for destination in destination_tuple):
        raise ValidationError("multicast destination is not in topology")

    link_map = {}
    parent_by_endpoint = {}
    for destination in destination_tuple:
        route = topology.route(source, destination, metric)
        for link in route.links:
            previous = parent_by_endpoint.get(link.dst)
            if previous is not None and previous != link.link_id:
                raise ValidationError(
                    "shortest-path union is not a multicast tree at %s" %
                    link.dst
                )
            parent_by_endpoint[link.dst] = link.link_id
            link_map[link.link_id] = link
    children: Dict[str, list] = {
        link_id: [] for link_id in link_map
    }
    roots = []
    for link_id, link in link_map.items():
        parent = parent_by_endpoint.get(link.src)
        if link.src == source:
            roots.append(link_id)
        elif parent is None:
            raise ValidationError(
                "multicast tree edge %s is disconnected" % link_id
            )
        else:
            children[parent].append(link_id)
    for value in children.values():
        value.sort()
    roots.sort()
    return MulticastTree(
        source=source,
        destinations=destination_tuple,
        links=tuple(link_map[key] for key in sorted(link_map)),
        root_link_ids=tuple(roots),
        children=MappingProxyType({
            key: tuple(value) for key, value in children.items()
        }),
    )


def simulate_multicast(
    topology: Topology,
    multicast_id: str,
    source: str,
    destinations: Iterable[str],
    nbytes: int,
    earliest_start_s: float = 0.0,
    metric: str = "latency",
    traffic_class: str = "multicast",
) -> MulticastResult:
    """Run one multicast tree with explicit per-edge contention."""

    if not multicast_id:
        raise ValidationError("multicast_id must be non-empty")
    if nbytes < 0:
        raise ValidationError("multicast bytes cannot be negative")
    tree = build_multicast_tree(
        topology, source, destinations, metric
    )
    link_by_id = {link.link_id: link for link in tree.links}
    network = FluidNetwork(topology, metric)
    flow_to_link = {}
    launched = set()
    destination_finish = {
        destination: earliest_start_s
        for destination in tree.destinations
        if destination == source
    }

    def launch(link_id, release_s):
        if link_id in launched:
            return
        link = link_by_id[link_id]
        flow_id = "%s:%s" % (multicast_id, link_id)
        network.add_flow(Flow(
            flow_id=flow_id,
            src=link.src,
            dst=link.dst,
            nbytes=nbytes,
            earliest_start_s=release_s,
            traffic_class=traffic_class,
            route_link_ids=(link_id,),
        ))
        flow_to_link[flow_id] = link_id
        launched.add(link_id)

    for root in tree.root_link_ids:
        launch(root, earliest_start_s)
    while network.unfinished_count:
        event_time = network.next_event_time()
        if not math.isfinite(event_time):
            raise RuntimeError("multicast network deadlock")
        for result in network.advance_to(event_time):
            link_id = flow_to_link[result.flow.flow_id]
            link = link_by_id[link_id]
            if link.dst in tree.destinations:
                destination_finish[link.dst] = result.finish_s
            for child in tree.children[link_id]:
                launch(child, result.finish_s)
    if set(destination_finish) != set(tree.destinations):
        raise RuntimeError("multicast did not reach every destination")
    results = network.results
    return MulticastResult(
        tree=tree,
        completion_s=max(destination_finish.values()),
        destination_finish_s=MappingProxyType(destination_finish),
        flow_results=results,
        link_bytes=network.link_bytes,
        energy_j=sum(result.energy_j for result in results.values()),
    )

