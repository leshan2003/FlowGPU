"""Provenance-aware system energy accounting for detailed v2 results."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from .layer_engine import MoELayerResult
from .schema import Confidence, Provenance, ValidationError


@dataclass(frozen=True)
class ChipPowerProfile:
    """Per-chip base and incremental compute power.

    ``idle_power_w`` is paid for every provisioned chip throughout the
    observation window. ``compute_dynamic_power_w`` is the incremental power
    paid while that chip's compute engine is busy, not total active power.
    """

    idle_power_w: float
    compute_dynamic_power_w: float
    provenance: Provenance

    def __post_init__(self) -> None:
        for name, value in (
            ("idle_power_w", self.idle_power_w),
            ("compute_dynamic_power_w", self.compute_dynamic_power_w),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValidationError(
                    "%s must be finite and non-negative" % name
                )


@dataclass(frozen=True)
class EnergyBreakdown:
    observation_start_s: float
    observation_finish_s: float
    output_tokens: int
    provisioned_chips: int
    chip_busy_s: float
    compute_dynamic_j: float
    idle_j: float
    network_j: float
    total_j: float
    energy_per_token_j: float
    decisive_confidence: Confidence
    provenance_counts: Mapping[str, int]


def _merged_duration(intervals) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    start, finish = ordered[0]
    total = 0.0
    for next_start, next_finish in ordered[1:]:
        if next_start <= finish:
            finish = max(finish, next_finish)
        else:
            total += finish - start
            start, finish = next_start, next_finish
    return total + finish - start


def account_system_energy(
    layer_results: Iterable[MoELayerResult],
    chip_power_profiles: Mapping[str, ChipPowerProfile],
    output_tokens: int,
    observation_start_s: float = 0.0,
    observation_finish_s: Optional[float] = None,
) -> EnergyBreakdown:
    """Account compute, idle, and network energy without invented defaults.

    The mapping defines all provisioned chips, including inactive chips.  This
    is intentional: node-count studies must pay idle energy for hardware added
    to the case.  Overlapping execution intervals on a chip are unioned before
    dynamic energy is charged.
    """

    results = tuple(layer_results)
    profiles = dict(chip_power_profiles)
    if not results:
        raise ValidationError("energy accounting needs layer results")
    if not profiles:
        raise ValidationError(
            "energy accounting needs explicit chip power profiles"
        )
    if output_tokens <= 0:
        raise ValidationError("energy output_tokens must be positive")
    if (not math.isfinite(observation_start_s) or
            observation_start_s < 0):
        raise ValidationError(
            "energy observation_start_s must be finite and >= 0"
        )
    if observation_finish_s is None:
        observation_finish_s = max(
            result.completion_s for result in results
        )
    if (not math.isfinite(observation_finish_s) or
            observation_finish_s < observation_start_s):
        raise ValidationError("invalid energy observation window")

    intervals = {chip_id: [] for chip_id in profiles}
    link_provenance = []
    for result in results:
        if result.completion_s > observation_finish_s + 1e-12:
            raise ValidationError(
                "layer completion exceeds energy observation window"
            )
        for execution in result.replica_executions:
            if execution.chip_id not in profiles:
                raise ValidationError(
                    "missing power profile for active chip %s" %
                    execution.chip_id
                )
            if (execution.compute_start_s < observation_start_s - 1e-12 or
                    execution.compute_finish_s >
                    observation_finish_s + 1e-12):
                raise ValidationError(
                    "compute interval is outside energy observation window"
                )
            intervals[execution.chip_id].append((
                execution.compute_start_s,
                execution.compute_finish_s,
            ))
        for flow in result.flow_results.values():
            link_provenance.extend(
                link.provenance for link in flow.route.links
            )

    busy_by_chip = {
        chip_id: _merged_duration(values)
        for chip_id, values in intervals.items()
    }
    compute_dynamic = sum(
        busy_by_chip[chip_id] *
        profiles[chip_id].compute_dynamic_power_w
        for chip_id in profiles
    )
    window_s = observation_finish_s - observation_start_s
    idle = window_s * sum(
        profile.idle_power_w for profile in profiles.values()
    )
    network = sum(result.network_energy_j for result in results)
    total = compute_dynamic + idle + network

    provenances = [
        profile.provenance for profile in profiles.values()
    ] + link_provenance
    counts = Counter(
        provenance.confidence.value for provenance in provenances
    )
    decisive = min(
        (provenance.confidence for provenance in provenances),
        key=lambda confidence: confidence.rank,
    )
    return EnergyBreakdown(
        observation_start_s=observation_start_s,
        observation_finish_s=observation_finish_s,
        output_tokens=output_tokens,
        provisioned_chips=len(profiles),
        chip_busy_s=sum(busy_by_chip.values()),
        compute_dynamic_j=compute_dynamic,
        idle_j=idle,
        network_j=network,
        total_j=total,
        energy_per_token_j=total / output_tokens,
        decisive_confidence=decisive,
        provenance_counts=dict(sorted(counts.items())),
    )
