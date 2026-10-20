"""Versioned placement changes and source-aware migration planning."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Tuple

from .replicas import ReplicaPlan
from .schema import ExpertKey, ExpertSpec, ValidationError


@dataclass(frozen=True)
class MigrationCopy:
    expert: ExpertKey
    source_chip_id: str
    target_chip_id: str
    target_node_id: str
    nbytes: int


@dataclass(frozen=True)
class ReplicaRemoval:
    expert: ExpertKey
    chip_id: str
    node_id: str
    nbytes: int


@dataclass(frozen=True)
class MigrationPlan:
    additions: Tuple[MigrationCopy, ...]
    removals: Tuple[ReplicaRemoval, ...]

    @property
    def bytes_by_target_node(self) -> Mapping[str, int]:
        values = {}
        for copy in self.additions:
            values[copy.target_node_id] = (
                values.get(copy.target_node_id, 0) + copy.nbytes
            )
        return MappingProxyType(values)

    @property
    def total_bytes(self) -> int:
        return sum(copy.nbytes for copy in self.additions)


def build_migration_plan(
    old: ReplicaPlan,
    new: ReplicaPlan,
    experts: Iterable[ExpertSpec],
) -> MigrationPlan:
    """Diff replica locations and choose a deterministic copy source.

    Copying from an already resident brain replica is preferred over the GPU
    fallback to avoid crossing the bridge. Source and target nodes are both
    retained in the operation for flow-level contention modeling.
    """

    expert_map = {expert.key: expert for expert in experts}
    old_by_expert: Dict[ExpertKey, list] = {}
    for replica in old.replicas:
        old_by_expert.setdefault(replica.expert, []).append(replica)
    old_locations = {
        (replica.expert, replica.chip_id) for replica in old.replicas
    }
    new_locations = {
        (replica.expert, replica.chip_id) for replica in new.replicas
    }
    additions = []
    for replica in sorted(
        new.replicas,
        key=lambda item: (item.expert, item.chip_id),
    ):
        location = (replica.expert, replica.chip_id)
        if location in old_locations:
            continue
        expert = expert_map.get(replica.expert)
        if expert is None:
            raise ValidationError("migration references unknown expert")
        source_candidates = old_by_expert.get(replica.expert, ())
        brain_sources = [
            item for item in source_candidates
            if old.chip(item.chip_id).role in (
                "brain_sram", "brain_dram"
            )
        ]
        source = (
            min(brain_sources, key=lambda item: item.chip_id)
            if brain_sources else
            min(source_candidates, key=lambda item: item.chip_id)
            if source_candidates else None
        )
        if source is None:
            raise ValidationError(
                "cannot migrate expert with no source replica"
            )
        target_chip = new.chip(replica.chip_id)
        additions.append(MigrationCopy(
            expert=replica.expert,
            source_chip_id=source.chip_id,
            target_chip_id=target_chip.chip_id,
            target_node_id=target_chip.node_id,
            nbytes=expert.weight_bytes,
        ))

    removals = []
    for replica in sorted(
        old.replicas,
        key=lambda item: (item.expert, item.chip_id),
    ):
        location = (replica.expert, replica.chip_id)
        if location in new_locations:
            continue
        expert = expert_map.get(replica.expert)
        if expert is None:
            raise ValidationError("removal references unknown expert")
        chip = old.chip(replica.chip_id)
        removals.append(ReplicaRemoval(
            expert=replica.expert,
            chip_id=replica.chip_id,
            node_id=chip.node_id,
            nbytes=expert.weight_bytes,
        ))
    if any(
        key not in {
            replica.expert for replica in new.replicas
        }
        for key in expert_map
    ):
        raise ValidationError("new placement is missing an expert replica")
    return MigrationPlan(tuple(additions), tuple(removals))


@dataclass(frozen=True)
class MigrationTiming:
    bytes_by_target_node: Mapping[str, int]
    completion_s_by_target_node: Mapping[str, float]
    completion_s: float


def estimate_migration_timing(
    plan: MigrationPlan,
    target_node_bandwidth_bytes_per_s: Mapping[str, float],
    reservation_fraction: float = 1.0,
    fixed_latency_s: float = 0.0,
) -> MigrationTiming:
    if not 0 < reservation_fraction <= 1:
        raise ValidationError("reservation_fraction must be in (0, 1]")
    if fixed_latency_s < 0:
        raise ValidationError("fixed_latency_s cannot be negative")
    completion = {}
    for node_id, nbytes in plan.bytes_by_target_node.items():
        try:
            bandwidth = target_node_bandwidth_bytes_per_s[node_id]
        except KeyError as exc:
            raise ValidationError(
                "missing target bandwidth for node %s" % node_id
            ) from exc
        if bandwidth <= 0:
            raise ValidationError("target bandwidth must be positive")
        completion[node_id] = (
            fixed_latency_s +
            nbytes / (bandwidth * reservation_fraction)
        )
    return MigrationTiming(
        bytes_by_target_node=plan.bytes_by_target_node,
        completion_s_by_target_node=MappingProxyType(completion),
        completion_s=max(completion.values(), default=0.0),
    )


@dataclass(frozen=True)
class PlacementVersion:
    version_id: str
    plan: ReplicaPlan
    ready: bool = False


class VersionedPlacement:
    """Reference-safe activation and retirement of placement versions."""

    def __init__(self, initial_plan: ReplicaPlan, version_id: str = "v0"):
        if not version_id:
            raise ValidationError("initial version id must be non-empty")
        self._versions = {
            version_id: PlacementVersion(version_id, initial_plan, True)
        }
        self._active = version_id
        self._pending = None
        self._retiring = None
        self._request_refs = {}
        self._version_refs = {version_id: 0}

    @property
    def active_version(self) -> str:
        return self._active

    @property
    def pending_version(self) -> Optional[str]:
        return self._pending

    @property
    def retiring_version(self) -> Optional[str]:
        return self._retiring

    def begin(self, candidate: ReplicaPlan, version_id: str) -> None:
        if self._pending is not None:
            raise ValidationError("a placement migration is already pending")
        if not version_id or version_id in self._versions:
            raise ValidationError("placement version id must be new")
        self._versions[version_id] = PlacementVersion(
            version_id, candidate, False
        )
        self._version_refs[version_id] = 0
        self._pending = version_id

    def mark_ready(self, version_id: str) -> None:
        if version_id != self._pending:
            raise ValidationError("only the pending version can become ready")
        version = self._versions[version_id]
        self._versions[version_id] = PlacementVersion(
            version.version_id, version.plan, True
        )

    def activate(self, version_id: str) -> None:
        if version_id != self._pending:
            raise ValidationError("only the pending version can activate")
        version = self._versions[version_id]
        if not version.ready:
            raise ValidationError("pending version is not fully ready")
        if self._retiring is not None:
            raise ValidationError("previous retiring version is still active")
        self._retiring = self._active
        self._active = version_id
        self._pending = None

    def bind_request(self, request_id: int) -> str:
        version_id = self._active
        if request_id in self._request_refs:
            bound_version, count = self._request_refs[request_id]
            if bound_version != version_id:
                raise ValidationError(
                    "one request cannot span active placement versions"
                )
            self._request_refs[request_id] = (bound_version, count + 1)
        else:
            self._request_refs[request_id] = (version_id, 1)
        self._version_refs[version_id] += 1
        return version_id

    def release_request(self, request_id: int) -> None:
        if request_id not in self._request_refs:
            raise ValidationError("request has no active placement reference")
        version_id, count = self._request_refs[request_id]
        self._version_refs[version_id] -= 1
        if count == 1:
            del self._request_refs[request_id]
        else:
            self._request_refs[request_id] = (version_id, count - 1)
        if (
            self._retiring is not None and
            self._version_refs.get(self._retiring, 0) == 0
        ):
            retired = self._retiring
            self._retiring = None
            if retired != self._active:
                self._versions.pop(retired, None)
                self._version_refs.pop(retired, None)

    def plan(self, version_id: str) -> ReplicaPlan:
        try:
            return self._versions[version_id].plan
        except KeyError as exc:
            raise ValidationError("unknown placement version") from exc
