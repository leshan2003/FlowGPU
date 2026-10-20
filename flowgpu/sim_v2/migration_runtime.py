"""Shared-bandwidth replay for background expert migration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple

from .detailed_network import FluidNetwork
from .network_model import Flow
from .refresh import MigrationCopy as MigrationCopySpec
from .refresh import MigrationPlan
from .schema import ValidationError
from .topology import ClusterTopology


@dataclass(frozen=True)
class SharedMigrationResult:
    serving_completion_s: float
    shared_serving_completion_s: float
    migration_completion_s: float
    combined_completion_s: float
    serving_delay_s: float
    migration_energy_j: float
    migration_flows: int
    migration_bytes: int


def _compact_flows(flows: Iterable[Flow]):
    grouped = {}
    for flow in flows:
        key = (
            flow.src,
            flow.dst,
            flow.earliest_start_s,
            flow.traffic_class,
            flow.waypoints,
            flow.route_link_ids,
        )
        grouped[key] = grouped.get(key, 0) + flow.nbytes
    compact = []
    for index, (key, nbytes) in enumerate(sorted(
        grouped.items(), key=lambda item: item[0]
    )):
        src, dst, start, traffic_class, waypoints, route_ids = key
        compact.append(Flow(
            flow_id="compact:%d:%s" % (index, traffic_class),
            src=src,
            dst=dst,
            nbytes=nbytes,
            earliest_start_s=start,
            traffic_class=traffic_class,
            waypoints=waypoints,
            route_link_ids=route_ids,
        ))
    return tuple(compact)


def build_migration_flows(
    migration: MigrationPlan,
    cluster: ClusterTopology,
    gpu_chip_ids: Iterable[str],
    start_s: float = 0.0,
) -> Tuple[Flow, ...]:
    gpu_ids = frozenset(gpu_chip_ids)
    endpoints = dict(cluster.chip_endpoints)
    for chip_id in gpu_ids:
        endpoints[chip_id] = cluster.gpu_endpoint
    flows = []
    for index, copy in enumerate(migration.additions):
        try:
            source = (
                cluster.gpu_endpoint
                if copy.source_chip_id == cluster.gpu_endpoint
                else endpoints[copy.source_chip_id]
            )
            target = endpoints[copy.target_chip_id]
        except KeyError as exc:
            raise ValidationError(
                "migration endpoint is not present in topology"
            ) from exc
        flows.append(Flow(
            flow_id="migration:%d:%d:%d" % (
                index, copy.expert.layer_id, copy.expert.expert_id
            ),
            src=source,
            dst=target,
            nbytes=copy.nbytes,
            earliest_start_s=start_s,
            traffic_class="migration",
        ))
    return tuple(flows)


def replay_with_shared_migration(
    cluster: ClusterTopology,
    serving_flows: Iterable[Flow],
    migration: MigrationPlan,
    gpu_chip_ids: Iterable[str],
    migration_start_s: float = 0.0,
) -> SharedMigrationResult:
    existing = _compact_flows(serving_flows)
    migration_flows = build_migration_flows(
        migration,
        cluster,
        gpu_chip_ids,
        start_s=migration_start_s,
    )
    if len({flow.flow_id for flow in existing}) != len(existing):
        raise ValidationError("serving flow ids must be unique")
    if len({flow.flow_id for flow in migration_flows}) != len(
        migration_flows
    ):
        raise ValidationError("migration flow ids must be unique")

    baseline = FluidNetwork(cluster.topology)
    for flow in sorted(existing, key=lambda item: (
        item.earliest_start_s, item.flow_id
    )):
        baseline.add_flow(flow)
    baseline_results = baseline.run()

    shared = FluidNetwork(cluster.topology)
    for flow in sorted(
        existing + migration_flows,
        key=lambda item: (item.earliest_start_s, item.flow_id),
    ):
        shared.add_flow(flow)
    shared_results = shared.run()

    serving_finish = max(
        (result.finish_s for result in baseline_results.values()),
        default=0.0,
    )
    shared_serving_finish = max(
        (
            shared_results[flow.flow_id].finish_s
            for flow in existing
        ),
        default=0.0,
    )
    serving_delay = max(
        (
            shared_results[flow.flow_id].finish_s
            - baseline_results[flow.flow_id].finish_s
            for flow in existing
        ),
        default=0.0,
    )
    combined_finish = max(
        (result.finish_s for result in shared_results.values()),
        default=0.0,
    )
    migration_finish = max(
        (
            result.finish_s for flow_id, result in shared_results.items()
            if flow_id.startswith("migration:")
        ),
        default=0.0,
    )
    migration_energy = sum(
        result.energy_j for flow_id, result in shared_results.items()
        if flow_id.startswith("migration:")
    )
    return SharedMigrationResult(
        serving_completion_s=serving_finish,
        shared_serving_completion_s=shared_serving_finish,
        migration_completion_s=migration_finish,
        combined_completion_s=combined_finish,
        serving_delay_s=max(0.0, serving_delay),
        migration_energy_j=migration_energy,
        migration_flows=len(migration_flows),
        migration_bytes=migration.total_bytes,
    )
