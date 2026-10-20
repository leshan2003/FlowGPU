"""Routing trace validation and deterministic synthetic generators."""

from __future__ import annotations

import random
import math
import json
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Tuple

from .schema import (Assignment, Provenance, RoutingRecord, ValidationError)


@dataclass(frozen=True)
class RoutingSummary:
    records: int
    assignments: int
    top_k: int
    home_nodes: int
    distinct_experts: int
    max_expert_load: int


class RoutingTrace:
    def __init__(
        self,
        records: Iterable[RoutingRecord],
        provenance: Provenance,
        split: str,
    ) -> None:
        if split not in ("placement", "evaluation", "oracle"):
            raise ValidationError(
                "trace split must be placement, evaluation, or oracle"
            )
        self.records = tuple(records)
        self.provenance = provenance
        self.split = split
        if not self.records:
            raise ValidationError("routing trace must contain records")
        identities = set()
        top_k = self.records[0].top_k
        for record in self.records:
            identity = (
                record.request_id, record.token_id,
                record.step, record.layer_id,
            )
            if identity in identities:
                raise ValidationError(
                    "duplicate token-layer routing record: %r" % (identity,)
                )
            identities.add(identity)
            if record.top_k != top_k:
                raise ValidationError(
                    "a trace must use a stable routed top_k"
                )
        self.top_k = top_k

    def assignments(self) -> Tuple[Assignment, ...]:
        assignments = tuple(
            assignment
            for record in self.records
            for assignment in record.assignments()
        )
        expected = len(self.records) * self.top_k
        if len(assignments) != expected:
            raise RuntimeError(
                "assignment conservation failed: %d != %d" %
                (len(assignments), expected)
            )
        if len({item.assignment_id for item in assignments}) != len(assignments):
            raise RuntimeError("generated assignment ids are not unique")
        return assignments

    def summary(self) -> RoutingSummary:
        counts = Counter(
            (record.layer_id, expert)
            for record in self.records
            for expert in record.experts
        )
        return RoutingSummary(
            records=len(self.records),
            assignments=len(self.records) * self.top_k,
            top_k=self.top_k,
            home_nodes=len({record.home_node for record in self.records}),
            distinct_experts=len(counts),
            max_expert_load=max(counts.values()),
        )


def generate_uniform_trace(
    n_tokens: int,
    n_layers: int,
    n_experts: int,
    top_k: int,
    home_nodes: Iterable[str],
    seed: int,
    provenance: Provenance,
    split: str,
) -> RoutingTrace:
    """Generate uniform top-k routing without replacement.

    Token home nodes are assigned round-robin so ingress scope is explicit and
    exactly balanced when ``n_tokens`` is divisible by the ingress count.
    """

    homes = tuple(home_nodes)
    if min(n_tokens, n_layers, n_experts, top_k) <= 0:
        raise ValidationError("trace dimensions must be positive")
    if top_k > n_experts:
        raise ValidationError("top_k cannot exceed n_experts")
    if not homes or any(not home for home in homes):
        raise ValidationError("home_nodes must contain non-empty node ids")
    if len(set(homes)) != len(homes):
        raise ValidationError("home_nodes must be unique")

    rng = random.Random(seed)
    records = []
    equal_weight = 1.0 / top_k
    for token_id in range(n_tokens):
        home = homes[token_id % len(homes)]
        for layer_id in range(n_layers):
            experts = tuple(rng.sample(range(n_experts), top_k))
            records.append(RoutingRecord(
                request_id=0,
                token_id=token_id,
                step=token_id,
                layer_id=layer_id,
                home_node=home,
                experts=experts,
                gate_weights=(equal_weight,) * top_k,
            ))
    return RoutingTrace(records, provenance, split)


def zipf_expert_weights(
    n_experts: int,
    exponent: float,
) -> Tuple[float, ...]:
    """Return normalized rank weights for a declared synthetic skew."""

    if n_experts <= 0:
        raise ValidationError("Zipf n_experts must be positive")
    if not math.isfinite(exponent) or exponent < 0:
        raise ValidationError(
            "Zipf exponent must be finite and non-negative"
        )
    raw = tuple(
        1.0 / ((rank + 1) ** exponent)
        for rank in range(n_experts)
    )
    total = sum(raw)
    return tuple(value / total for value in raw)


def _weighted_sample_without_replacement(rng, weights, count):
    remaining = list(enumerate(weights))
    selected = []
    for _ in range(count):
        total = sum(weight for _, weight in remaining)
        if total <= 0:
            raise ValidationError(
                "fewer than top_k experts have positive routing weight"
            )
        target = rng.random() * total
        cumulative = 0.0
        chosen_index = len(remaining) - 1
        for index, (_, weight) in enumerate(remaining):
            cumulative += weight
            if target < cumulative:
                chosen_index = index
                break
        expert_id, _ = remaining.pop(chosen_index)
        selected.append(expert_id)
    return tuple(selected)


def generate_weighted_trace(
    n_tokens: int,
    n_layers: int,
    expert_weights: Iterable[float],
    top_k: int,
    home_nodes: Iterable[str],
    seed: int,
    provenance: Provenance,
    split: str,
    node_expert_multipliers=None,
) -> RoutingTrace:
    """Generate weighted top-k routing with optional node affinity.

    ``node_expert_multipliers[(node_id, expert_id)]`` scales the global
    expert weight for that home node. Missing pairs use multiplier 1.0.
    Selection is without replacement, so one token never routes twice to the
    same expert.
    """

    weights = tuple(float(value) for value in expert_weights)
    homes = tuple(home_nodes)
    multipliers = dict(node_expert_multipliers or {})
    if min(n_tokens, n_layers, top_k) <= 0 or not weights:
        raise ValidationError(
            "weighted trace dimensions and expert weights must be positive"
        )
    if top_k > len(weights):
        raise ValidationError("top_k cannot exceed weighted experts")
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValidationError(
            "expert routing weights must be finite and non-negative"
        )
    if sum(value > 0 for value in weights) < top_k:
        raise ValidationError(
            "fewer than top_k experts have positive routing weight"
        )
    if not homes or any(not home for home in homes):
        raise ValidationError("home_nodes must contain non-empty node ids")
    if len(set(homes)) != len(homes):
        raise ValidationError("home_nodes must be unique")
    for (node_id, expert_id), multiplier in multipliers.items():
        if node_id not in homes:
            raise ValidationError(
                "node affinity references an unknown home node"
            )
        if (not isinstance(expert_id, int) or expert_id < 0 or
                expert_id >= len(weights)):
            raise ValidationError(
                "node affinity references an unknown expert"
            )
        if not math.isfinite(multiplier) or multiplier < 0:
            raise ValidationError(
                "node affinity multipliers must be finite and non-negative"
            )

    rng = random.Random(seed)
    records = []
    gate_weights = (1.0 / top_k,) * top_k
    for token_id in range(n_tokens):
        home = homes[token_id % len(homes)]
        local_weights = tuple(
            weight * multipliers.get((home, expert_id), 1.0)
            for expert_id, weight in enumerate(weights)
        )
        if sum(value > 0 for value in local_weights) < top_k:
            raise ValidationError(
                "node affinity leaves fewer than top_k positive experts "
                "for %s" % home
            )
        for layer_id in range(n_layers):
            experts = _weighted_sample_without_replacement(
                rng, local_weights, top_k
            )
            records.append(RoutingRecord(
                request_id=0,
                token_id=token_id,
                step=token_id,
                layer_id=layer_id,
                home_node=home,
                experts=experts,
                gate_weights=gate_weights,
            ))
    return RoutingTrace(records, provenance, split)


def load_routing_trace_jsonl(
    path,
    provenance: Provenance,
    split: str,
) -> RoutingTrace:
    """Load a strict real/synthetic router trace without extra dependencies."""

    trace_path = Path(path).resolve()
    if not trace_path.is_file():
        raise ValidationError(
            "routing trace is not a file: %s" % trace_path
        )
    records = []
    required = {
        "request_id", "token_id", "step", "layer_id",
        "home_node", "experts", "gate_weights",
    }

    def strict_int(value, field, line_number):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValidationError(
                "routing trace line %d field %s must be an integer" %
                (line_number, field)
            )
        return value

    try:
        with trace_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValidationError(
                        "routing trace line %d is not an object" %
                        line_number
                    )
                missing = required - set(item)
                unknown = set(item) - required
                if missing:
                    raise ValidationError(
                        "routing trace line %d missing %s" %
                        (line_number, sorted(missing)[0])
                    )
                if unknown:
                    raise ValidationError(
                        "routing trace line %d unknown field %s" %
                        (line_number, sorted(unknown)[0])
                    )
                if (not isinstance(item["experts"], list) or
                        not isinstance(item["gate_weights"], list)):
                    raise ValidationError(
                        "routing trace experts/gate_weights must be arrays"
                    )
                if not isinstance(item["home_node"], str):
                    raise ValidationError(
                        "routing trace home_node must be a string"
                    )
                for value in item["gate_weights"]:
                    if (not isinstance(value, (int, float)) or
                            isinstance(value, bool)):
                        raise ValidationError(
                            "routing trace gate weights must be numeric"
                        )
                records.append(RoutingRecord(
                    request_id=strict_int(
                        item["request_id"], "request_id", line_number
                    ),
                    token_id=strict_int(
                        item["token_id"], "token_id", line_number
                    ),
                    step=strict_int(
                        item["step"], "step", line_number
                    ),
                    layer_id=strict_int(
                        item["layer_id"], "layer_id", line_number
                    ),
                    home_node=item["home_node"],
                    experts=tuple(
                        strict_int(value, "experts", line_number)
                        for value in item["experts"]
                    ),
                    gate_weights=tuple(
                        float(value) for value in item["gate_weights"]
                    ),
                ))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(
            "cannot parse routing trace: %s" % trace_path
        ) from exc
    if not records:
        raise ValidationError("routing trace is empty")
    return RoutingTrace(records, provenance, split)
