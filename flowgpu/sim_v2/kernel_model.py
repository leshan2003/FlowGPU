"""Calibrated expert-kernel latency models for grouped token execution."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from .schema import Provenance, ValidationError


@dataclass(frozen=True)
class KernelSample:
    m: int
    latency_s: float
    p95_s: Optional[float] = None
    repetitions: Optional[int] = None

    def __post_init__(self) -> None:
        if self.m <= 0:
            raise ValidationError("kernel sample M must be positive")
        if not math.isfinite(self.latency_s) or self.latency_s <= 0:
            raise ValidationError("kernel latency must be finite and positive")
        if self.p95_s is not None and self.p95_s < self.latency_s:
            raise ValidationError("kernel p95 cannot be below central latency")
        if self.repetitions is not None and self.repetitions <= 0:
            raise ValidationError("kernel repetitions must be positive")


@dataclass(frozen=True)
class KernelPrediction:
    m: int
    latency_s: float
    extrapolated: bool
    bracket_m: Tuple[int, int]
    provenance: Provenance
    warning: Optional[str] = None


class ExpertKernelCurve:
    """Piecewise-linear ``t_expert(M)`` curve from benchmark samples.

    Linear interpolation is deliberately transparent.  Extrapolation is
    allowed so sweeps can finish, but it is surfaced in every prediction and
    must be propagated to the run warnings/result contract.
    """

    def __init__(
        self,
        expert_shape: str,
        dtype: str,
        concurrent_experts: int,
        residency: str,
        samples: Iterable[KernelSample],
        provenance: Provenance,
    ) -> None:
        if not expert_shape or not dtype or not residency:
            raise ValidationError(
                "kernel shape, dtype, and residency must be non-empty"
            )
        if concurrent_experts <= 0:
            raise ValidationError("concurrent_experts must be positive")
        ordered = tuple(sorted(samples, key=lambda sample: sample.m))
        if not ordered:
            raise ValidationError("kernel curve needs at least one sample")
        if len({sample.m for sample in ordered}) != len(ordered):
            raise ValidationError("kernel curve has duplicate M samples")
        previous = 0.0
        for sample in ordered:
            if sample.latency_s < previous:
                raise ValidationError(
                    "kernel latency must be non-decreasing with M"
                )
            previous = sample.latency_s
        self.expert_shape = expert_shape
        self.dtype = dtype
        self.concurrent_experts = concurrent_experts
        self.residency = residency
        self.samples = ordered
        self.provenance = provenance
        self._ms = tuple(sample.m for sample in ordered)

    def predict(self, m: int) -> KernelPrediction:
        if m <= 0:
            raise ValidationError("kernel prediction M must be positive")
        index = bisect.bisect_left(self._ms, m)
        if index < len(self.samples) and self.samples[index].m == m:
            sample = self.samples[index]
            return KernelPrediction(
                m, sample.latency_s, False, (m, m), self.provenance
            )
        if len(self.samples) == 1:
            sample = self.samples[0]
            # With one point, constant time is less misleading than inventing
            # linear scaling.  It remains explicitly marked extrapolated.
            return KernelPrediction(
                m=m,
                latency_s=sample.latency_s,
                extrapolated=True,
                bracket_m=(sample.m, sample.m),
                provenance=self.provenance,
                warning=(
                    "kernel M=%d is outside the single measured point M=%d" %
                    (m, sample.m)
                ),
            )
        if index == 0:
            lower, upper = self.samples[0], self.samples[1]
            extrapolated = True
        elif index == len(self.samples):
            lower, upper = self.samples[-2], self.samples[-1]
            extrapolated = True
        else:
            lower, upper = self.samples[index - 1], self.samples[index]
            extrapolated = False
        slope = (
            (upper.latency_s - lower.latency_s) /
            (upper.m - lower.m)
        )
        predicted = lower.latency_s + slope * (m - lower.m)
        # A startup-dominated fit can have a zero slope; never permit a
        # non-positive result when extrapolating below the first point.
        predicted = max(predicted, min(lower.latency_s, upper.latency_s) * 0.1)
        warning = None
        if extrapolated:
            warning = (
                "kernel M=%d is outside calibrated range [%d, %d]" %
                (m, self._ms[0], self._ms[-1])
            )
        return KernelPrediction(
            m=m,
            latency_s=predicted,
            extrapolated=extrapolated,
            bracket_m=(lower.m, upper.m),
            provenance=self.provenance,
            warning=warning,
        )


class AnalyticExpertKernel:
    """Explicitly derived fallback when no benchmark curve exists."""

    def __init__(
        self,
        flops_per_token: float,
        weight_bytes: int,
        peak_flops_per_s: float,
        sram_bandwidth_bytes_per_s: float,
        max_efficiency: float,
        saturation_m: int,
        launch_overhead_s: float,
        provenance: Provenance,
    ) -> None:
        values = (
            flops_per_token, weight_bytes, peak_flops_per_s,
            sram_bandwidth_bytes_per_s, max_efficiency,
            saturation_m, launch_overhead_s,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValidationError("analytic kernel parameters must be finite")
        if (flops_per_token <= 0 or weight_bytes <= 0 or
                peak_flops_per_s <= 0 or sram_bandwidth_bytes_per_s <= 0 or
                not 0 < max_efficiency <= 1 or saturation_m <= 0 or
                launch_overhead_s < 0):
            raise ValidationError("invalid analytic kernel parameter")
        self.flops_per_token = flops_per_token
        self.weight_bytes = weight_bytes
        self.peak_flops_per_s = peak_flops_per_s
        self.sram_bandwidth_bytes_per_s = sram_bandwidth_bytes_per_s
        self.max_efficiency = max_efficiency
        self.saturation_m = saturation_m
        self.launch_overhead_s = launch_overhead_s
        self.provenance = provenance

    def predict(self, m: int) -> KernelPrediction:
        if m <= 0:
            raise ValidationError("kernel prediction M must be positive")
        fill = min(1.0, m / self.saturation_m)
        efficiency = self.max_efficiency * fill
        compute_s = (
            self.flops_per_token * m /
            (self.peak_flops_per_s * efficiency)
        )
        # Resident expert weights are read once per grouped invocation in this
        # conservative fallback; activation traffic requires calibrated data.
        memory_s = self.weight_bytes / self.sram_bandwidth_bytes_per_s
        latency = self.launch_overhead_s + max(compute_s, memory_s)
        return KernelPrediction(
            m=m,
            latency_s=latency,
            extrapolated=True,
            bracket_m=(m, m),
            provenance=self.provenance,
            warning="analytic expert-kernel fallback; no calibrated curve",
        )

