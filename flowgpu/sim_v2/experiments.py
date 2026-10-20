"""Fair node-count sweep construction for Simulator v2.

This module builds experiment cases; it deliberately does not run placement
or execution.  Keeping construction separate makes the changing hardware
quantity (node count) and the fixed per-node quantities auditable before an
expensive sweep starts.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from .placement import capacity_ratio
from .routing import (RoutingTrace, generate_uniform_trace,
                      generate_weighted_trace, load_routing_trace_jsonl)
from .schema import ChipSpec, ExpertSpec, Provenance, ValidationError
from .topology import ClusterTopology, LinkProfile, build_hierarchical_cluster


INGRESS_MODES = frozenset(("distributed", "concentrated"))
SCALING_MODES = frozenset(("strong", "weak"))


def classify_capacity_regime(ratio: float, tolerance: float = 1e-12) -> str:
    """Classify aggregate capacity without claiming fragmentation feasibility."""

    if not math.isfinite(ratio) or ratio <= 0:
        raise ValidationError("capacity ratio must be finite and positive")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValidationError(
            "capacity regime tolerance must be finite and >= 0"
        )
    if ratio < 1.0 - tolerance:
        return "spill-required"
    if ratio <= 1.0 + tolerance:
        return "unique-fit-boundary"
    return "replication-capable"


@dataclass(frozen=True)
class NodeSweepSpec:
    """The controlled variables for a node-count scaling study.

    ``concentrated_ingress_count`` is a cap.  Thus a one-node case has one
    ingress even when the requested concentrated count is larger.
    """

    node_counts: Tuple[int, ...]
    chips_per_node: int
    chip_sram_capacity_bytes: int
    chip_provenance: Provenance
    runtime_reserve_fraction: float = 0.15
    fixed_reserved_bytes: int = 0
    ingress_modes: Tuple[str, ...] = ("distributed", "concentrated")
    concentrated_ingress_count: int = 1

    def __post_init__(self) -> None:
        counts = tuple(self.node_counts)
        modes = tuple(self.ingress_modes)
        if not counts or any(
                not isinstance(count, int) or isinstance(count, bool) or
                count <= 0 for count in counts):
            raise ValidationError(
                "node sweep counts must be positive integers"
            )
        if len(set(counts)) != len(counts):
            raise ValidationError("node sweep counts must be unique")
        if self.chips_per_node <= 0:
            raise ValidationError("chips_per_node must be positive")
        if self.chip_sram_capacity_bytes <= 0:
            raise ValidationError(
                "chip_sram_capacity_bytes must be positive"
            )
        if (not math.isfinite(self.runtime_reserve_fraction) or
                not 0 <= self.runtime_reserve_fraction < 1):
            raise ValidationError(
                "runtime_reserve_fraction must be in [0, 1)"
            )
        if self.fixed_reserved_bytes < 0:
            raise ValidationError("fixed_reserved_bytes cannot be negative")
        if not modes or len(set(modes)) != len(modes):
            raise ValidationError(
                "ingress_modes must be non-empty and unique"
            )
        unknown = set(modes) - INGRESS_MODES
        if unknown:
            raise ValidationError(
                "unknown ingress mode: %s" % sorted(unknown)[0]
            )
        if self.concentrated_ingress_count <= 0:
            raise ValidationError(
                "concentrated_ingress_count must be positive"
            )
        # Canonical ordering makes case ids and output paths reproducible.
        object.__setattr__(self, "node_counts", tuple(sorted(counts)))
        object.__setattr__(self, "ingress_modes", tuple(sorted(modes)))


@dataclass(frozen=True)
class NodeSweepCase:
    case_id: str
    node_count: int
    chips_per_node: int
    ingress_mode: str
    ingress_nodes: Tuple[str, ...]
    routing_home_nodes: Tuple[str, ...]
    chips: Tuple[ChipSpec, ...]
    cluster: ClusterTopology
    aggregate_usable_weight_bytes: int
    capacity_ratio: Optional[float]
    capacity_regime: str
    per_node_hardware_fingerprint: str
    bridge_link_pairs: int

    def fairness_metadata(self) -> dict:
        """Return JSON-safe scaling and fairness fields for result configs."""

        return {
            "case_id": self.case_id,
            "node_count": self.node_count,
            "chips_per_node": self.chips_per_node,
            "total_chips": len(self.chips),
            "ingress_mode": self.ingress_mode,
            "ingress_nodes": list(self.ingress_nodes),
            "routing_home_nodes": list(self.routing_home_nodes),
            "bridge_link_pairs": self.bridge_link_pairs,
            "aggregate_usable_weight_bytes":
                self.aggregate_usable_weight_bytes,
            "capacity_ratio": self.capacity_ratio,
            "capacity_regime": self.capacity_regime,
            "per_node_hardware_fingerprint":
                self.per_node_hardware_fingerprint,
        }


@dataclass(frozen=True)
class SweepWorkloadSpec:
    """Synthetic trace controls shared by every node-count case.

    In ``strong`` scaling, placement/evaluation token counts are fixed across
    node counts.  In ``weak`` scaling, they are interpreted as tokens per
    cluster node and multiplied by the case's node count.
    """

    n_layers: int
    n_experts: int
    top_k: int
    placement_tokens: int
    evaluation_tokens: int
    placement_seed: int
    evaluation_seed: int
    provenance: Provenance
    scaling_mode: str = "strong"
    routing_model: str = "uniform"
    placement_expert_weights: Tuple[float, ...] = ()
    evaluation_expert_weights: Tuple[float, ...] = ()
    placement_trace_template: Optional[str] = None
    evaluation_trace_template: Optional[str] = None

    def __post_init__(self) -> None:
        if min(
            self.n_layers, self.n_experts, self.top_k,
            self.placement_tokens, self.evaluation_tokens,
        ) <= 0:
            raise ValidationError(
                "sweep workload dimensions and token counts must be positive"
            )
        if self.top_k > self.n_experts:
            raise ValidationError("sweep workload top_k exceeds n_experts")
        if min(self.placement_seed, self.evaluation_seed) < 0:
            raise ValidationError("sweep workload seeds must be non-negative")
        if self.placement_seed == self.evaluation_seed:
            raise ValidationError(
                "placement and evaluation seeds must differ"
            )
        if self.scaling_mode not in SCALING_MODES:
            raise ValidationError(
                "scaling_mode must be strong or weak"
            )
        if self.routing_model not in ("uniform", "weighted", "trace"):
            raise ValidationError(
                "routing_model must be uniform, weighted, or trace"
            )
        placement_weights = tuple(self.placement_expert_weights)
        evaluation_weights = tuple(self.evaluation_expert_weights)
        if self.routing_model == "uniform":
            if placement_weights or evaluation_weights:
                raise ValidationError(
                    "uniform routing cannot declare expert weights"
                )
        elif self.routing_model == "weighted":
            if (len(placement_weights) != self.n_experts or
                    len(evaluation_weights) != self.n_experts):
                raise ValidationError(
                    "weighted routing needs one placement/evaluation "
                    "weight per expert"
                )
            for weights in (placement_weights, evaluation_weights):
                if any(not math.isfinite(value) or value < 0
                       for value in weights):
                    raise ValidationError(
                        "routing weights must be finite and non-negative"
                    )
                if sum(value > 0 for value in weights) < self.top_k:
                    raise ValidationError(
                        "routing weights need at least top_k positives"
                    )
        else:
            if placement_weights or evaluation_weights:
                raise ValidationError(
                    "trace routing cannot declare expert weights"
                )
            if (not self.placement_trace_template or
                    not self.evaluation_trace_template):
                raise ValidationError(
                    "trace routing needs placement/evaluation templates"
                )
            if (self.placement_trace_template ==
                    self.evaluation_trace_template):
                raise ValidationError(
                    "placement and evaluation traces must differ"
                )
        object.__setattr__(
            self, "placement_expert_weights", placement_weights
        )
        object.__setattr__(
            self, "evaluation_expert_weights", evaluation_weights
        )


@dataclass(frozen=True)
class SweepTracePair:
    case_id: str
    scaling_mode: str
    placement: RoutingTrace
    evaluation: RoutingTrace
    placement_token_count: int
    evaluation_token_count: int


def build_case_trace_pair(
    case: NodeSweepCase,
    workload: SweepWorkloadSpec,
) -> SweepTracePair:
    """Create independent placement and held-out evaluation traces."""

    multiplier = case.node_count if workload.scaling_mode == "weak" else 1
    placement_tokens = workload.placement_tokens * multiplier
    evaluation_tokens = workload.evaluation_tokens * multiplier
    if workload.routing_model == "uniform":
        placement = generate_uniform_trace(
            n_tokens=placement_tokens,
            n_layers=workload.n_layers,
            n_experts=workload.n_experts,
            top_k=workload.top_k,
            home_nodes=case.routing_home_nodes,
            seed=workload.placement_seed,
            provenance=workload.provenance,
            split="placement",
        )
        evaluation = generate_uniform_trace(
            n_tokens=evaluation_tokens,
            n_layers=workload.n_layers,
            n_experts=workload.n_experts,
            top_k=workload.top_k,
            home_nodes=case.routing_home_nodes,
            seed=workload.evaluation_seed,
            provenance=workload.provenance,
            split="evaluation",
        )
    elif workload.routing_model == "weighted":
        placement = generate_weighted_trace(
            n_tokens=placement_tokens,
            n_layers=workload.n_layers,
            expert_weights=workload.placement_expert_weights,
            top_k=workload.top_k,
            home_nodes=case.routing_home_nodes,
            seed=workload.placement_seed,
            provenance=workload.provenance,
            split="placement",
        )
        evaluation = generate_weighted_trace(
            n_tokens=evaluation_tokens,
            n_layers=workload.n_layers,
            expert_weights=workload.evaluation_expert_weights,
            top_k=workload.top_k,
            home_nodes=case.routing_home_nodes,
            seed=workload.evaluation_seed,
            provenance=workload.provenance,
            split="evaluation",
        )
    else:
        try:
            placement_path = workload.placement_trace_template.format(
                case_id=case.case_id
            )
            evaluation_path = workload.evaluation_trace_template.format(
                case_id=case.case_id
            )
        except (KeyError, ValueError) as exc:
            raise ValidationError(
                "invalid routing trace template; only {case_id} is supported"
            ) from exc
        placement = load_routing_trace_jsonl(
            placement_path, workload.provenance, "placement"
        )
        evaluation = load_routing_trace_jsonl(
            evaluation_path, workload.provenance, "evaluation"
        )
        for name, trace, expected_tokens in (
            ("placement", placement, placement_tokens),
            ("evaluation", evaluation, evaluation_tokens),
        ):
            identities = {
                (record.request_id, record.token_id, record.step)
                for record in trace.records
            }
            layers = {record.layer_id for record in trace.records}
            homes = {record.home_node for record in trace.records}
            if len(identities) != expected_tokens:
                raise ValidationError(
                    "%s trace token count does not match workload/case" %
                    name
                )
            if layers != set(range(workload.n_layers)):
                raise ValidationError(
                    "%s trace layers do not match model" % name
                )
            if trace.top_k != workload.top_k:
                raise ValidationError(
                    "%s trace top_k does not match model" % name
                )
            if not homes.issubset(set(case.routing_home_nodes)):
                raise ValidationError(
                    "%s trace has a home node outside this sweep case" %
                    name
                )
            if any(
                expert_id >= workload.n_experts
                for record in trace.records
                for expert_id in record.experts
            ):
                raise ValidationError(
                    "%s trace expert id exceeds model" % name
                )
    return SweepTracePair(
        case_id=case.case_id,
        scaling_mode=workload.scaling_mode,
        placement=placement,
        evaluation=evaluation,
        placement_token_count=placement_tokens,
        evaluation_token_count=evaluation_tokens,
    )


def _provenance_dict(provenance: Provenance) -> dict:
    return {
        "confidence": provenance.confidence.value,
        "source": provenance.source,
        "artifact": provenance.artifact,
        "measured_at": provenance.measured_at,
        "machine_revision": provenance.machine_revision,
    }


def _profile_dict(profile: LinkProfile) -> dict:
    return {
        "bandwidth_bytes_per_s": profile.bandwidth_bytes_per_s,
        "latency_s": profile.latency_s,
        "efficiency": profile.efficiency,
        "energy_per_byte": profile.energy_per_byte,
        "duplex": profile.duplex,
        "provenance": _provenance_dict(profile.provenance),
    }


def _per_node_hardware_fingerprint(
    spec: NodeSweepSpec,
    scale_up: LinkProfile,
    scale_out: LinkProfile,
    bridge: LinkProfile,
) -> str:
    payload = {
        "chips_per_node": spec.chips_per_node,
        "chip_sram_capacity_bytes": spec.chip_sram_capacity_bytes,
        "runtime_reserve_fraction": spec.runtime_reserve_fraction,
        "fixed_reserved_bytes": spec.fixed_reserved_bytes,
        "chip_provenance": _provenance_dict(spec.chip_provenance),
        "scale_up": _profile_dict(scale_up),
        "scale_out": _profile_dict(scale_out),
        "bridge": _profile_dict(bridge),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_node_count_sweep(
    spec: NodeSweepSpec,
    scale_up: LinkProfile,
    scale_out: LinkProfile,
    bridge: LinkProfile,
    experts: Iterable[ExpertSpec] = (),
) -> Tuple[NodeSweepCase, ...]:
    """Build distributed- and concentrated-ingress scaling cases.

    Distributed ingress gives every node a GPU bridge and permits every node
    to be a token home.  Concentrated ingress connects only a fixed prefix of
    nodes and restricts token homes to that prefix.  The two ingress modes are
    separate hardware scenarios and must not be pooled as repetitions.
    """

    expert_list = tuple(experts)
    fingerprint = _per_node_hardware_fingerprint(
        spec, scale_up, scale_out, bridge
    )
    cases = []
    for node_count in spec.node_counts:
        chips = tuple(
            ChipSpec(
                chip_id="node%d:chip%d" % (node_index, chip_index),
                node_id="node%d" % node_index,
                sram_capacity_bytes=spec.chip_sram_capacity_bytes,
                runtime_reserve_fraction=spec.runtime_reserve_fraction,
                fixed_reserved_bytes=spec.fixed_reserved_bytes,
                provenance=spec.chip_provenance,
            )
            for node_index in range(node_count)
            for chip_index in range(spec.chips_per_node)
        )
        aggregate_usable = sum(
            chip.usable_weight_bytes for chip in chips
        )
        ratio = (
            capacity_ratio(expert_list, chips)
            if expert_list else None
        )
        for mode in spec.ingress_modes:
            if mode == "distributed":
                ingress_indices = tuple(range(node_count))
            else:
                ingress_indices = tuple(range(min(
                    node_count, spec.concentrated_ingress_count
                )))
            cluster = build_hierarchical_cluster(
                n_nodes=node_count,
                chips_per_node=spec.chips_per_node,
                ingress_node_indices=ingress_indices,
                scale_up=scale_up,
                scale_out=scale_out,
                bridge=bridge,
            )
            # Only bridged nodes are home nodes in the concentrated scenario;
            # this avoids inventing free token ingress on unbridged nodes.
            home_nodes = cluster.ingress_nodes
            case_id = "nodes-%d__chips-%d__ingress-%s-%d" % (
                node_count,
                spec.chips_per_node,
                mode,
                len(cluster.ingress_nodes),
            )
            cases.append(NodeSweepCase(
                case_id=case_id,
                node_count=node_count,
                chips_per_node=spec.chips_per_node,
                ingress_mode=mode,
                ingress_nodes=cluster.ingress_nodes,
                routing_home_nodes=home_nodes,
                chips=chips,
                cluster=cluster,
                aggregate_usable_weight_bytes=aggregate_usable,
                capacity_ratio=ratio,
                capacity_regime=(
                    classify_capacity_regime(ratio)
                    if ratio is not None else "unknown"
                ),
                per_node_hardware_fingerprint=fingerprint,
                bridge_link_pairs=len(cluster.ingress_nodes),
            ))
    return tuple(cases)
