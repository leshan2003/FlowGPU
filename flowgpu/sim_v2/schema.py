"""Core immutable data structures for FlowGPU Simulator v2.

The v2 simulator deliberately does not reuse v1's ``shards`` abstraction.
Experts, complete physical replicas, routed assignments, and chips are
represented separately so capacity and routing conservation can be checked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class ValidationError(ValueError):
    """Raised when a v2 input violates a model invariant."""


class Confidence(str, Enum):
    """Evidence strength, ordered from weakest to strongest."""

    PROJECTED = "projected"
    DERIVED = "derived"
    PAPER = "paper"
    VENDOR = "vendor"
    MEASURED = "measured"

    @property
    def rank(self) -> int:
        return {
            Confidence.PROJECTED: 0,
            Confidence.DERIVED: 1,
            Confidence.PAPER: 2,
            Confidence.VENDOR: 3,
            Confidence.MEASURED: 4,
        }[self]


@dataclass(frozen=True)
class Provenance:
    confidence: Confidence
    source: str
    artifact: Optional[str] = None
    measured_at: Optional[str] = None
    machine_revision: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.confidence, str):
            try:
                object.__setattr__(self, "confidence",
                                   Confidence(self.confidence))
            except ValueError as exc:
                raise ValidationError(
                    "unknown provenance confidence: %r" % self.confidence
                ) from exc
        if not self.source or not self.source.strip():
            raise ValidationError("provenance source must be non-empty")
        if self.confidence is Confidence.MEASURED and not self.artifact:
            raise ValidationError(
                "measured provenance must reference a raw artifact"
            )


@dataclass(frozen=True, order=True)
class ExpertKey:
    layer_id: int
    expert_id: int
    shared: bool = False

    def __post_init__(self) -> None:
        if self.layer_id < 0 or self.expert_id < 0:
            raise ValidationError("layer_id and expert_id must be non-negative")


@dataclass(frozen=True)
class ExpertSpec:
    key: ExpertKey
    weight_bytes: int
    provenance: Provenance

    def __post_init__(self) -> None:
        if self.weight_bytes <= 0:
            raise ValidationError("expert weight_bytes must be positive")


@dataclass(frozen=True)
class ChipSpec:
    chip_id: str
    node_id: str
    sram_capacity_bytes: int
    runtime_reserve_fraction: float = 0.15
    fixed_reserved_bytes: int = 0
    provenance: Optional[Provenance] = None

    def __post_init__(self) -> None:
        if not self.chip_id or not self.node_id:
            raise ValidationError("chip_id and node_id must be non-empty")
        if self.sram_capacity_bytes <= 0:
            raise ValidationError("chip SRAM capacity must be positive")
        if not 0.0 <= self.runtime_reserve_fraction < 1.0:
            raise ValidationError(
                "runtime_reserve_fraction must be in [0, 1)"
            )
        if self.fixed_reserved_bytes < 0:
            raise ValidationError("fixed_reserved_bytes cannot be negative")
        if self.fixed_reserved_bytes >= self.sram_capacity_bytes:
            raise ValidationError(
                "fixed reserved bytes leave no usable chip SRAM"
            )
        if self.usable_weight_bytes <= 0:
            raise ValidationError("chip has no SRAM available for weights")

    @property
    def usable_weight_bytes(self) -> int:
        after_fraction = math.floor(
            self.sram_capacity_bytes *
            (1.0 - self.runtime_reserve_fraction)
        )
        return after_fraction - self.fixed_reserved_bytes


@dataclass(frozen=True)
class ReplicaSpec:
    replica_id: str
    expert: ExpertKey
    chip_id: str
    is_primary: bool

    def __post_init__(self) -> None:
        if not self.replica_id or not self.chip_id:
            raise ValidationError("replica_id and chip_id must be non-empty")


@dataclass(frozen=True)
class Assignment:
    assignment_id: str
    request_id: int
    token_id: int
    step: int
    layer_id: int
    expert_id: int
    gate_weight: float
    home_node: str
    shared: bool = False

    def __post_init__(self) -> None:
        if not self.assignment_id or not self.home_node:
            raise ValidationError(
                "assignment_id and home_node must be non-empty"
            )
        if min(self.request_id, self.token_id, self.step,
               self.layer_id, self.expert_id) < 0:
            raise ValidationError("assignment integer fields must be >= 0")
        if not math.isfinite(self.gate_weight) or self.gate_weight < 0:
            raise ValidationError("gate_weight must be finite and non-negative")

    @property
    def expert(self) -> ExpertKey:
        return ExpertKey(self.layer_id, self.expert_id, self.shared)


@dataclass(frozen=True)
class RoutingRecord:
    """One token's routed experts at one layer."""

    request_id: int
    token_id: int
    step: int
    layer_id: int
    home_node: str
    experts: Tuple[int, ...]
    gate_weights: Tuple[float, ...]

    def __post_init__(self) -> None:
        if min(self.request_id, self.token_id, self.step, self.layer_id) < 0:
            raise ValidationError("routing record integer fields must be >= 0")
        if not self.home_node:
            raise ValidationError("routing record home_node must be non-empty")
        if not self.experts:
            raise ValidationError("routing record must select at least one expert")
        if len(self.experts) != len(self.gate_weights):
            raise ValidationError(
                "experts and gate_weights must have equal length"
            )
        if len(set(self.experts)) != len(self.experts):
            raise ValidationError("top-k experts must be unique per token")
        if any(expert < 0 for expert in self.experts):
            raise ValidationError("expert ids must be non-negative")
        if any((not math.isfinite(weight) or weight < 0)
               for weight in self.gate_weights):
            raise ValidationError(
                "gate weights must be finite and non-negative"
            )

    @property
    def top_k(self) -> int:
        return len(self.experts)

    def assignments(self) -> Tuple[Assignment, ...]:
        out = []
        for rank, (expert_id, weight) in enumerate(
                zip(self.experts, self.gate_weights)):
            assignment_id = "%d:%d:%d:%d:%d" % (
                self.request_id, self.token_id, self.step,
                self.layer_id, rank
            )
            out.append(Assignment(
                assignment_id=assignment_id,
                request_id=self.request_id,
                token_id=self.token_id,
                step=self.step,
                layer_id=self.layer_id,
                expert_id=expert_id,
                gate_weight=weight,
                home_node=self.home_node,
            ))
        return tuple(out)


@dataclass(frozen=True)
class RuntimeCapabilities:
    dispatch_dedup_chip: bool = False
    dispatch_dedup_node: bool = False
    combine_reduce_chip: bool = False
    combine_reduce_node: bool = False
    hardware_multicast: bool = False

