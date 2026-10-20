"""Expert replica inventory and capacity validation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Tuple

from .schema import (Assignment, ChipSpec, ExpertKey, ExpertSpec, ReplicaSpec,
                     ValidationError)


@dataclass(frozen=True)
class CapacityUsage:
    chip_id: str
    weight_bytes: int
    usable_weight_bytes: int

    @property
    def utilization(self) -> float:
        return self.weight_bytes / self.usable_weight_bytes

    @property
    def free_bytes(self) -> int:
        return self.usable_weight_bytes - self.weight_bytes


class ReplicaPlan:
    """Validated, read-only expert-to-chip replica placement."""

    def __init__(
        self,
        experts: Iterable[ExpertSpec],
        chips: Iterable[ChipSpec],
        replicas: Iterable[ReplicaSpec],
        strategy: str,
    ) -> None:
        if not strategy:
            raise ValidationError("placement strategy must be non-empty")
        self.strategy = strategy
        self.experts = tuple(experts)
        self.chips = tuple(chips)
        self.replicas = tuple(replicas)

        expert_map = self._unique_map(
            ((expert.key, expert) for expert in self.experts), "expert"
        )
        chip_map = self._unique_map(
            ((chip.chip_id, chip) for chip in self.chips), "chip"
        )
        replica_map = self._unique_map(
            ((replica.replica_id, replica) for replica in self.replicas),
            "replica",
        )
        if not expert_map:
            raise ValidationError("replica plan must contain experts")
        if not chip_map:
            raise ValidationError("replica plan must contain chips")

        by_expert: Dict[ExpertKey, list] = {
            key: [] for key in expert_map
        }
        seen_locations = set()
        used = {chip_id: 0 for chip_id in chip_map}
        for replica in self.replicas:
            if replica.expert not in expert_map:
                raise ValidationError(
                    "replica %s references unknown expert %r" %
                    (replica.replica_id, replica.expert)
                )
            if replica.chip_id not in chip_map:
                raise ValidationError(
                    "replica %s references unknown chip %s" %
                    (replica.replica_id, replica.chip_id)
                )
            location = (replica.expert, replica.chip_id)
            if location in seen_locations:
                raise ValidationError(
                    "expert %r is duplicated on chip %s" % location
                )
            seen_locations.add(location)
            by_expert[replica.expert].append(replica)
            used[replica.chip_id] += expert_map[
                replica.expert
            ].weight_bytes

        for key, placed in by_expert.items():
            if not placed:
                raise ValidationError(
                    "expert %r has no resident primary replica" % (key,)
                )
            primary_count = sum(replica.is_primary for replica in placed)
            if primary_count != 1:
                raise ValidationError(
                    "expert %r must have exactly one primary; got %d" %
                    (key, primary_count)
                )

        usage = {}
        for chip_id, weight_bytes in used.items():
            chip = chip_map[chip_id]
            if weight_bytes > chip.usable_weight_bytes:
                raise ValidationError(
                    "chip %s capacity exceeded: %d > %d usable bytes" %
                    (chip_id, weight_bytes, chip.usable_weight_bytes)
                )
            usage[chip_id] = CapacityUsage(
                chip_id, weight_bytes, chip.usable_weight_bytes
            )

        self._expert_map = MappingProxyType(expert_map)
        self._chip_map = MappingProxyType(chip_map)
        self._replica_map = MappingProxyType(replica_map)
        self._by_expert = MappingProxyType({
            key: tuple(value) for key, value in by_expert.items()
        })
        self._usage = MappingProxyType(usage)

    @staticmethod
    def _unique_map(items, kind: str) -> dict:
        out = {}
        for key, value in items:
            if key in out:
                raise ValidationError("duplicate %s identifier: %r" %
                                      (kind, key))
            out[key] = value
        return out

    @property
    def capacity_usage(self) -> Mapping[str, CapacityUsage]:
        return self._usage

    def replicas_for(self, expert: ExpertKey) -> Tuple[ReplicaSpec, ...]:
        if expert not in self._by_expert:
            raise ValidationError("unknown expert: %r" % (expert,))
        return self._by_expert[expert]

    def replica(self, replica_id: str) -> ReplicaSpec:
        try:
            return self._replica_map[replica_id]
        except KeyError as exc:
            raise ValidationError(
                "assignment references unknown replica %s" % replica_id
            ) from exc

    def chip(self, chip_id: str) -> ChipSpec:
        try:
            return self._chip_map[chip_id]
        except KeyError as exc:
            raise ValidationError("unknown chip %s" % chip_id) from exc

    def validate_binding(
        self, assignment: Assignment, replica_id: str
    ) -> ReplicaSpec:
        replica = self.replica(replica_id)
        if replica.expert != assignment.expert:
            raise ValidationError(
                "assignment %s for %r cannot bind to replica %s for %r" %
                (assignment.assignment_id, assignment.expert,
                 replica_id, replica.expert)
            )
        return replica

