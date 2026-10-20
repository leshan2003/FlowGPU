"""Paired block statistics and drift detection for MoE timelines."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Sequence, Tuple

from .schema import ValidationError


@dataclass(frozen=True)
class BlockBootstrapCI:
    ratio: float
    low95: float
    high95: float
    blocks: int
    resamples: int


def paired_block_bootstrap_ci(
    blocks: Iterable[Tuple[float, float]],
    resamples: int = 10000,
    seed: int = 0,
) -> BlockBootstrapCI:
    """Bootstrap aggregate candidate/baseline ratio by resampling blocks."""

    values = tuple((float(left), float(right)) for left, right in blocks)
    if not values:
        raise ValidationError("block bootstrap requires paired blocks")
    if any(
        not math.isfinite(left) or not math.isfinite(right) or left < 0 or right < 0
        for left, right in values
    ):
        raise ValidationError("block values must be finite and non-negative")
    if resamples <= 0:
        raise ValidationError("resamples must be positive")
    total_baseline = sum(left for left, _ in values)
    if total_baseline <= 0:
        raise ValidationError("baseline block total must be positive")
    rng = random.Random(seed)
    ratios = []
    for _ in range(resamples):
        baseline = 0.0
        candidate = 0.0
        for _ in range(len(values)):
            left, right = values[rng.randrange(len(values))]
            baseline += left
            candidate += right
        ratios.append(candidate / baseline if baseline > 0 else 0.0)
    ratios.sort()
    low_index = max(0, int(math.floor(0.025 * (resamples - 1))))
    high_index = min(resamples - 1, int(math.ceil(
        0.975 * (resamples - 1)
    )))
    return BlockBootstrapCI(
        ratio=sum(right for _, right in values) / total_baseline,
        low95=ratios[low_index],
        high95=ratios[high_index],
        blocks=len(values),
        resamples=resamples,
    )


def required_paired_blocks(
    standard_deviation: float,
    target_effect: float,
    alpha_z: float = 1.959963984540054,
    power_z: float = 0.8416212335729143,
    minimum: int = 20,
) -> int:
    """Approximate two-sided paired-sample size for a target mean effect."""

    if (
        not math.isfinite(standard_deviation) or
        standard_deviation < 0 or
        not math.isfinite(target_effect) or
        target_effect <= 0
    ):
        raise ValidationError(
            "standard deviation/effect must be finite and non-negative"
        )
    if minimum <= 0:
        raise ValidationError("minimum must be positive")
    if standard_deviation == 0:
        return minimum
    return max(
        minimum,
        int(math.ceil(
            ((alpha_z + power_z) * standard_deviation / target_effect) ** 2
        )),
    )


def normalized_l1(
    previous: Mapping[Hashable, float],
    current: Mapping[Hashable, float],
) -> float:
    keys = set(previous) | set(current)
    previous_total = sum(max(0.0, float(value)) for value in previous.values())
    current_total = sum(max(0.0, float(value)) for value in current.values())
    if not keys or previous_total <= 0 or current_total <= 0:
        return 0.0
    return 0.5 * sum(
        abs(
            max(0.0, float(previous.get(key, 0.0))) / previous_total -
            max(0.0, float(current.get(key, 0.0))) / current_total
        )
        for key in keys
    )


@dataclass(frozen=True)
class DriftDecision:
    window: int
    drift_l1: float
    triggered: bool
    dwell_windows: int


class HysteresisDriftDetector:
    """Threshold crossing with minimum residency between triggers."""

    def __init__(
        self,
        threshold: float,
        release_threshold: float,
        min_dwell_windows: int = 1,
    ) -> None:
        if (
            not math.isfinite(threshold) or
            not math.isfinite(release_threshold) or
            threshold <= 0 or
            release_threshold < 0 or
            release_threshold > threshold
        ):
            raise ValidationError("invalid drift thresholds")
        if min_dwell_windows <= 0:
            raise ValidationError("min_dwell_windows must be positive")
        self.threshold = threshold
        self.release_threshold = release_threshold
        self.min_dwell_windows = min_dwell_windows
        self._previous = None
        self._last_trigger_window = None
        self._armed = True

    def observe(
        self,
        window: int,
        distribution: Mapping[Hashable, float],
    ) -> DriftDecision:
        if window < 0:
            raise ValidationError("window must be non-negative")
        if self._previous is None:
            self._previous = dict(distribution)
            return DriftDecision(window, 0.0, False, 0)
        drift = normalized_l1(self._previous, distribution)
        self._previous = dict(distribution)
        if drift <= self.release_threshold:
            self._armed = True
        dwell = (
            window - self._last_trigger_window
            if self._last_trigger_window is not None else window
        )
        triggered = (
            self._armed and drift >= self.threshold and
            dwell >= self.min_dwell_windows
        )
        if triggered:
            self._armed = False
            self._last_trigger_window = window
        return DriftDecision(window, drift, triggered, dwell)
