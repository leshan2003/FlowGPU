"""Placement-window demand statistics for expert planning."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple

from .routing import RoutingTrace
from .schema import ExpertKey, ValidationError


@dataclass(frozen=True)
class DemandProfile:
    """Observed routed-assignment demand, including node conditionals."""

    global_assignments: Mapping[ExpertKey, int]
    node_assignments: Mapping[Tuple[str, ExpertKey], int]
    coactivation: Mapping[Tuple[ExpertKey, ...], int]
    records: int
    assignments: int
    source_split: str

    def global_for(self, expert: ExpertKey) -> int:
        return self.global_assignments.get(expert, 0)

    def node_for(self, node_id: str, expert: ExpertKey) -> int:
        return self.node_assignments.get((node_id, expert), 0)


def profile_demand(
    trace: RoutingTrace,
    allow_oracle: bool = False,
) -> DemandProfile:
    """Build ``lambda[layer,node,expert]`` from a placement-only window.

    Evaluation traces are rejected.  An oracle trace is accepted only with an
    explicit flag, so future-aware placement cannot silently enter a result.
    """

    if trace.split == "evaluation":
        raise ValidationError(
            "evaluation trace cannot be used for replica placement"
        )
    if trace.split == "oracle" and not allow_oracle:
        raise ValidationError(
            "oracle trace requires allow_oracle=True and must be labelled"
        )

    global_counts = Counter()
    node_counts = Counter()
    hyperedges = Counter()
    for record in trace.records:
        keys = tuple(sorted(
            ExpertKey(record.layer_id, expert_id)
            for expert_id in record.experts
        ))
        hyperedges[keys] += 1
        for key in keys:
            global_counts[key] += 1
            node_counts[(record.home_node, key)] += 1
    assignments = sum(global_counts.values())
    expected = len(trace.records) * trace.top_k
    if assignments != expected:
        raise RuntimeError(
            "demand conservation failed: %d != %d" %
            (assignments, expected)
        )
    return DemandProfile(
        global_assignments=MappingProxyType(dict(global_counts)),
        node_assignments=MappingProxyType(dict(node_counts)),
        coactivation=MappingProxyType(dict(hyperedges)),
        records=len(trace.records),
        assignments=assignments,
        source_split=trace.split,
    )

