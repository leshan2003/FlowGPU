"""Decode-only synchronous continuous batching on the v2 MoE layer engine."""

from __future__ import annotations

import math
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple

from .layer_engine import MoELayerEngine, MoELayerResult
from .schema import Assignment, ValidationError


@dataclass(frozen=True)
class DecodeRequest:
    request_id: int
    ready_s: float
    output_tokens: int

    def __post_init__(self) -> None:
        if self.request_id < 0:
            raise ValidationError("request_id must be non-negative")
        if not math.isfinite(self.ready_s) or self.ready_s < 0:
            raise ValidationError("request ready_s must be finite and >= 0")
        if self.output_tokens <= 0:
            raise ValidationError("request output_tokens must be positive")


@dataclass(frozen=True)
class AdmissionPolicy:
    """Iteration-boundary cap on active plus admitted waiting requests."""

    max_outstanding_requests: int

    def __post_init__(self) -> None:
        if self.max_outstanding_requests <= 0:
            raise ValidationError(
                "max_outstanding_requests must be positive"
            )


@dataclass(frozen=True)
class ResourceBlockWindow:
    """External compute reservation, such as a projected prefill chunk."""

    start_s: float
    finish_s: float
    source: str = "prefill"

    def __post_init__(self) -> None:
        if (not math.isfinite(self.start_s) or
                not math.isfinite(self.finish_s) or
                self.start_s < 0 or self.finish_s <= self.start_s):
            raise ValidationError("invalid resource block window")
        if not self.source:
            raise ValidationError("resource block source must be non-empty")


@dataclass(frozen=True)
class RouteChoice:
    home_node: str
    experts: Tuple[int, ...]
    gate_weights: Tuple[float, ...]


class UniformServingRouter:
    """Stateless deterministic uniform top-k router for serving tests."""

    def __init__(
        self,
        n_experts: int,
        top_k: int,
        home_nodes: Iterable[str],
        seed: int,
    ) -> None:
        self.home_nodes = tuple(home_nodes)
        if n_experts <= 0 or top_k <= 0 or top_k > n_experts:
            raise ValidationError("invalid serving router expert/top-k counts")
        if not self.home_nodes or any(not node for node in self.home_nodes):
            raise ValidationError("serving router needs non-empty home nodes")
        if len(set(self.home_nodes)) != len(self.home_nodes):
            raise ValidationError("serving router home nodes must be unique")
        if seed < 0:
            raise ValidationError("serving router seed must be non-negative")
        self.n_experts = n_experts
        self.top_k = top_k
        self.seed = seed

    def route(
        self, request_id: int, token_index: int, layer_id: int
    ) -> RouteChoice:
        mixed_seed = (
            self.seed * 1_000_003 +
            request_id * 97_409 +
            token_index * 65_537 +
            layer_id * 8_191
        )
        rng = random.Random(mixed_seed)
        experts = tuple(rng.sample(range(self.n_experts), self.top_k))
        return RouteChoice(
            home_node=self.home_nodes[
                request_id % len(self.home_nodes)
            ],
            experts=experts,
            gate_weights=(1.0 / self.top_k,) * self.top_k,
        )


class WeightedServingRouter:
    """Stateless deterministic weighted top-k router for serving."""

    def __init__(
        self,
        expert_weights: Iterable[float],
        top_k: int,
        home_nodes: Iterable[str],
        seed: int,
        node_expert_multipliers=None,
    ) -> None:
        self.expert_weights = tuple(
            float(value) for value in expert_weights
        )
        self.home_nodes = tuple(home_nodes)
        self.multipliers = dict(node_expert_multipliers or {})
        if (not self.expert_weights or top_k <= 0 or
                top_k > len(self.expert_weights)):
            raise ValidationError("invalid weighted serving top-k")
        if any(not math.isfinite(value) or value < 0
               for value in self.expert_weights):
            raise ValidationError(
                "weighted serving weights must be finite and non-negative"
            )
        if sum(value > 0 for value in self.expert_weights) < top_k:
            raise ValidationError(
                "weighted serving needs top_k positive experts"
            )
        if not self.home_nodes or len(set(self.home_nodes)) != \
                len(self.home_nodes):
            raise ValidationError(
                "weighted serving home nodes must be non-empty and unique"
            )
        if seed < 0:
            raise ValidationError(
                "weighted serving seed must be non-negative"
            )
        for (node_id, expert_id), multiplier in self.multipliers.items():
            if (node_id not in self.home_nodes or
                    not isinstance(expert_id, int) or
                    not 0 <= expert_id < len(self.expert_weights)):
                raise ValidationError(
                    "invalid weighted serving node affinity key"
                )
            if not math.isfinite(multiplier) or multiplier < 0:
                raise ValidationError(
                    "weighted serving affinity must be non-negative"
                )
        self.top_k = top_k
        self.seed = seed

    def route(
        self, request_id: int, token_index: int, layer_id: int
    ) -> RouteChoice:
        home = self.home_nodes[request_id % len(self.home_nodes)]
        weights = [
            value * self.multipliers.get((home, expert_id), 1.0)
            for expert_id, value in enumerate(self.expert_weights)
        ]
        if sum(value > 0 for value in weights) < self.top_k:
            raise ValidationError(
                "serving affinity leaves fewer than top_k experts"
            )
        rng = random.Random(
            self.seed * 1_000_003 +
            request_id * 97_409 +
            token_index * 65_537 +
            layer_id * 8_191
        )
        remaining = list(enumerate(weights))
        selected = []
        for _ in range(self.top_k):
            total = sum(value for _, value in remaining)
            target = rng.random() * total
            cumulative = 0.0
            chosen = len(remaining) - 1
            for index, (_, value) in enumerate(remaining):
                cumulative += value
                if target < cumulative:
                    chosen = index
                    break
            expert_id, _ = remaining.pop(chosen)
            selected.append(expert_id)
        return RouteChoice(
            home_node=home,
            experts=tuple(selected),
            gate_weights=(1.0 / self.top_k,) * self.top_k,
        )


@dataclass(frozen=True)
class RequestServingResult:
    request_id: int
    ready_s: float
    token_finish_s: Tuple[float, ...]

    @property
    def finish_s(self) -> float:
        return self.token_finish_s[-1]

    @property
    def ttft_s(self) -> float:
        return self.token_finish_s[0] - self.ready_s

    @property
    def mean_tpot_s(self) -> float:
        if len(self.token_finish_s) <= 1:
            return 0.0
        gaps = [
            right - left
            for left, right in zip(
                self.token_finish_s, self.token_finish_s[1:]
            )
        ]
        return sum(gaps) / len(gaps)


@dataclass(frozen=True)
class ServingResult:
    requests: Tuple[RequestServingResult, ...]
    layer_results: Tuple[MoELayerResult, ...]
    iteration_batch_sizes: Tuple[int, ...]
    start_s: float
    finish_s: float
    output_tokens: int
    output_throughput: float
    network_energy_j: float
    offered_requests: int
    rejected_request_ids: Tuple[int, ...]
    prefill_blocked_s: float
    system_energy_j: Optional[float] = None

    @property
    def admission_fraction(self) -> float:
        return len(self.requests) / self.offered_requests

    def ttft_percentile(self, percentile: float) -> float:
        return _percentile(
            [request.ttft_s for request in self.requests], percentile
        )

    def tpot_percentile(self, percentile: float) -> float:
        values = [
            request.mean_tpot_s for request in self.requests
            if len(request.token_finish_s) > 1
        ]
        return _percentile(values, percentile) if values else 0.0

    def slo_attainment(
        self,
        ttft_s: float,
        tpot_s: float,
    ) -> float:
        if ttft_s <= 0 or tpot_s < 0:
            raise ValidationError("invalid serving SLO thresholds")
        attained = sum(
            request.ttft_s <= ttft_s and
            request.mean_tpot_s <= tpot_s
            for request in self.requests
        )
        # Rejected requests are offered requests that did not attain the SLO.
        return attained / self.offered_requests

    def slo_goodput_per_s(self, ttft_s: float, tpot_s: float) -> float:
        if ttft_s <= 0 or tpot_s < 0:
            raise ValidationError("invalid serving SLO thresholds")
        attained = sum(
            request.ttft_s <= ttft_s and
            request.mean_tpot_s <= tpot_s
            for request in self.requests
        )
        wall_time = self.finish_s - self.start_s
        return attained / wall_time if wall_time > 0 else math.inf


@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    low95: float
    high95: float
    samples: int
    method: str


@dataclass(frozen=True)
class ServingAggregate:
    throughput: ConfidenceInterval
    mean_ttft_s: ConfidenceInterval
    mean_tpot_s: ConfidenceInterval
    slo_attainment: ConfidenceInterval
    slo_goodput_per_s: ConfidenceInterval
    admission_fraction: ConfidenceInterval
    network_energy_per_token_j: ConfidenceInterval
    system_energy_per_token_j: Optional[ConfidenceInterval]

    @property
    def energy_per_token_j(self) -> ConfidenceInterval:
        """Compatibility view; system energy wins when explicitly modeled."""

        return (
            self.system_energy_per_token_j
            if self.system_energy_per_token_j is not None
            else self.network_energy_per_token_j
        )


@dataclass(frozen=True)
class SLOCapacityPoint:
    offered_rate_per_s: float
    aggregate: ServingAggregate
    lower_bound_meets_target: bool


@dataclass(frozen=True)
class SLOCapacityResult:
    capacity_rate_per_s: float
    target_attainment: float
    points: Tuple[SLOCapacityPoint, ...]
    monotonic_attainment: bool
    warning: str


@dataclass
class _RequestState:
    spec: DecodeRequest
    generated: int
    token_finish: list


def _percentile(values, percentile):
    if not values:
        raise ValidationError("cannot compute percentile of empty values")
    if not 0 <= percentile <= 100:
        raise ValidationError("percentile must be in [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return (
        ordered[lower] * (1 - fraction) +
        ordered[upper] * fraction
    )


class ContinuousBatchingSimulator:
    """Synchronous iteration-boundary continuous batching.

    This is decode-only: ``ready_s`` is the time prefill has completed.  New
    requests join at the next decode-iteration boundary when capacity exists.
    Each active request emits one token per iteration and traverses all MoE
    layers sequentially.  Layer pipelining and prefill interference remain
    outside this MVP and are not implied by the term continuous batching.
    """

    def __init__(
        self,
        layer_engine: MoELayerEngine,
        router,
        n_layers: int,
        max_batch: int,
        admission_policy: AdmissionPolicy = None,
        compute_block_windows: Iterable[ResourceBlockWindow] = (),
        chip_power_profiles: Mapping = None,
    ) -> None:
        if n_layers <= 0 or max_batch <= 0:
            raise ValidationError(
                "n_layers and max_batch must be positive"
            )
        if not hasattr(router, "route"):
            raise ValidationError("serving router must provide route()")
        self.layer_engine = layer_engine
        self.router = router
        self.n_layers = n_layers
        self.max_batch = max_batch
        self.admission_policy = admission_policy
        windows = tuple(sorted(
            compute_block_windows,
            key=lambda item: (item.start_s, item.finish_s, item.source),
        ))
        previous_finish = -math.inf
        for window in windows:
            if window.start_s < previous_finish:
                raise ValidationError(
                    "resource block windows must not overlap"
                )
            previous_finish = window.finish_s
        self.compute_block_windows = windows
        self.chip_power_profiles = (
            dict(chip_power_profiles)
            if chip_power_profiles is not None else None
        )
        if self.chip_power_profiles is not None and \
                not self.chip_power_profiles:
            raise ValidationError(
                "chip_power_profiles cannot be an empty mapping"
            )

    def _advance_past_blocks(self, now_s):
        blocked = 0.0
        for window in self.compute_block_windows:
            if window.start_s <= now_s < window.finish_s:
                blocked += window.finish_s - now_s
                now_s = window.finish_s
        return now_s, blocked

    def run(self, requests: Iterable[DecodeRequest]) -> ServingResult:
        specs = tuple(sorted(
            requests, key=lambda item: (item.ready_s, item.request_id)
        ))
        if not specs:
            raise ValidationError("serving simulation needs requests")
        if len({spec.request_id for spec in specs}) != len(specs):
            raise ValidationError("serving request ids must be unique")
        future = list(specs)
        waiting = []
        active = []
        completed = []
        rejected = []
        layer_results = []
        batch_sizes = []
        now_s = specs[0].ready_s
        prefill_blocked_s = 0.0

        while future or waiting or active:
            if (not active and not waiting and future and
                    future[0].ready_s > now_s):
                now_s = future[0].ready_s
            now_s, blocked = self._advance_past_blocks(now_s)
            prefill_blocked_s += blocked
            # Admission is intentionally evaluated only at iteration
            # boundaries. All arrivals observed by this boundary are decided
            # before filling newly free batch slots.
            while future and future[0].ready_s <= now_s:
                spec = future.pop(0)
                outstanding = len(active) + len(waiting)
                if (self.admission_policy is not None and
                        outstanding >=
                        self.admission_policy.max_outstanding_requests):
                    rejected.append(spec.request_id)
                    continue
                waiting.append(_RequestState(
                    spec=spec,
                    generated=0,
                    token_finish=[],
                ))
            while waiting and len(active) < self.max_batch:
                active.append(waiting.pop(0))
            if not active:
                continue
            active.sort(key=lambda state: state.spec.request_id)
            batch_sizes.append(len(active))
            for layer_id in range(self.n_layers):
                now_s, blocked = self._advance_past_blocks(now_s)
                prefill_blocked_s += blocked
                assignments = []
                for state in active:
                    choice = self.router.route(
                        state.spec.request_id,
                        state.generated,
                        layer_id,
                    )
                    if (len(choice.experts) !=
                            len(choice.gate_weights)):
                        raise ValidationError(
                            "serving route expert/weight length mismatch"
                        )
                    for rank, (expert_id, weight) in enumerate(zip(
                            choice.experts, choice.gate_weights)):
                        assignments.append(Assignment(
                            assignment_id="r%d:o%d:l%d:k%d" % (
                                state.spec.request_id,
                                state.generated,
                                layer_id,
                                rank,
                            ),
                            request_id=state.spec.request_id,
                            token_id=state.generated,
                            step=state.generated,
                            layer_id=layer_id,
                            expert_id=expert_id,
                            gate_weight=weight,
                            home_node=choice.home_node,
                        ))
                layer_result = self.layer_engine.run(
                    assignments, start_time_s=now_s
                )
                layer_results.append(layer_result)
                now_s = layer_result.completion_s
            survivors = []
            for state in active:
                state.token_finish.append(now_s)
                state.generated += 1
                if state.generated == state.spec.output_tokens:
                    completed.append(state)
                else:
                    survivors.append(state)
            active = survivors

        request_results = tuple(sorted(
            (
                RequestServingResult(
                    state.spec.request_id,
                    state.spec.ready_s,
                    tuple(state.token_finish),
                )
                for state in completed
            ),
            key=lambda result: result.request_id,
        ))
        start_s = min(spec.ready_s for spec in specs)
        finish_s = max(result.finish_s for result in request_results)
        output_tokens = sum(
            len(result.token_finish_s) for result in request_results
        )
        wall_time = finish_s - start_s
        throughput = (
            output_tokens / wall_time if wall_time > 0 else math.inf
        )
        network_energy_j = sum(
            result.network_energy_j for result in layer_results
        )
        system_energy_j = None
        if self.chip_power_profiles is not None:
            from .energy import account_system_energy
            system_energy_j = account_system_energy(
                layer_results=layer_results,
                chip_power_profiles=self.chip_power_profiles,
                output_tokens=output_tokens,
                observation_start_s=start_s,
                observation_finish_s=finish_s,
            ).total_j
        return ServingResult(
            requests=request_results,
            layer_results=tuple(layer_results),
            iteration_batch_sizes=tuple(batch_sizes),
            start_s=start_s,
            finish_s=finish_s,
            output_tokens=output_tokens,
            output_throughput=throughput,
            network_energy_j=network_energy_j,
            offered_requests=len(specs),
            rejected_request_ids=tuple(sorted(rejected)),
            prefill_blocked_s=prefill_blocked_s,
            system_energy_j=system_energy_j,
        )


def generate_poisson_requests(
    n_requests: int,
    arrival_rate_per_s: float,
    output_tokens,
    seed: int,
    start_s: float = 0.0,
) -> Tuple[DecodeRequest, ...]:
    """Generate reproducible Poisson decode-ready arrivals."""

    if n_requests <= 0:
        raise ValidationError("n_requests must be positive")
    if (not math.isfinite(arrival_rate_per_s) or
            arrival_rate_per_s <= 0):
        raise ValidationError("arrival rate must be finite and positive")
    if seed < 0:
        raise ValidationError("Poisson seed must be non-negative")
    if not math.isfinite(start_s) or start_s < 0:
        raise ValidationError("Poisson start_s must be finite and >= 0")
    if isinstance(output_tokens, int):
        if output_tokens <= 0:
            raise ValidationError("output_tokens must be positive")
        output_range = (output_tokens, output_tokens)
    else:
        try:
            low, high = output_tokens
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "output_tokens must be int or (low, high)"
            ) from exc
        if (not isinstance(low, int) or not isinstance(high, int) or
                low <= 0 or high < low):
            raise ValidationError("invalid output token range")
        output_range = (low, high)
    rng = random.Random(seed)
    now_s = start_s
    requests = []
    for request_id in range(n_requests):
        now_s += rng.expovariate(arrival_rate_per_s)
        length = rng.randint(output_range[0], output_range[1])
        requests.append(DecodeRequest(request_id, now_s, length))
    return tuple(requests)


def load_decode_request_trace(path) -> Tuple[DecodeRequest, ...]:
    """Load JSONL records with request_id, ready_s, and output_tokens."""

    trace_path = Path(path).resolve()
    if not trace_path.is_file():
        raise ValidationError(
            "decode request trace is not a file: %s" % trace_path
        )
    requests = []
    try:
        with trace_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValidationError(
                        "trace line %d is not an object" % line_number
                    )
                missing = [
                    field for field in (
                        "request_id", "ready_s", "output_tokens"
                    )
                    if field not in record
                ]
                if missing:
                    raise ValidationError(
                        "trace line %d missing: %s" %
                        (line_number, ",".join(missing))
                    )
                requests.append(DecodeRequest(
                    int(record["request_id"]),
                    float(record["ready_s"]),
                    int(record["output_tokens"]),
                ))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(
            "cannot parse decode request trace: %s" % trace_path
        ) from exc
    if not requests:
        raise ValidationError("decode request trace is empty")
    if len({request.request_id for request in requests}) != len(requests):
        raise ValidationError("decode request trace has duplicate ids")
    return tuple(sorted(
        requests, key=lambda request: (request.ready_s, request.request_id)
    ))


_T_CRITICAL_95 = (
    None,
    12.706, 4.303, 3.182, 2.776, 2.571,
    2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131,
    2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060,
    2.056, 2.052, 2.048, 2.045, 2.042,
)


def _confidence_interval(
    values,
    lower_bound=None,
    upper_bound=None,
) -> ConfidenceInterval:
    if not values:
        raise ValidationError("confidence interval needs samples")
    if any(not math.isfinite(value) for value in values):
        raise ValidationError("confidence interval samples must be finite")
    mean = statistics.mean(values)
    if len(values) == 1:
        return ConfidenceInterval(
            mean, mean, mean, 1, "single-sample-no-uncertainty"
        )
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    degrees = len(values) - 1
    if degrees <= 30:
        critical = _T_CRITICAL_95[degrees]
    else:
        # Accurate asymptotic expansion around the two-sided normal 97.5%
        # quantile; remains slightly wider than 1.96 for finite df.
        critical = (
            1.959964 +
            2.372272 / degrees +
            2.822 / (degrees * degrees)
        )
    margin = critical * standard_error
    low = mean - margin
    high = mean + margin
    if lower_bound is not None:
        low = max(lower_bound, low)
    if upper_bound is not None:
        high = min(upper_bound, high)
    return ConfidenceInterval(
        mean, low, high, len(values), "student-t-95"
    )


def confidence_interval(
    values,
    lower_bound=None,
    upper_bound=None,
) -> ConfidenceInterval:
    """Public Student-t interval helper used by cross-seed reports."""

    return _confidence_interval(
        tuple(values), lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def aggregate_serving_results(
    results: Iterable[ServingResult],
    slo_ttft_s: float,
    slo_tpot_s: float,
) -> ServingAggregate:
    """Aggregate independent seeds with normal-approximation 95% CIs."""

    runs = tuple(results)
    if not runs:
        raise ValidationError("serving aggregate needs runs")
    if slo_ttft_s <= 0 or slo_tpot_s < 0:
        raise ValidationError("invalid aggregate SLO thresholds")
    throughput = [run.output_throughput for run in runs]
    mean_ttft = [
        statistics.mean(request.ttft_s for request in run.requests)
        for run in runs
    ]
    mean_tpot = [
        statistics.mean(
            request.mean_tpot_s for request in run.requests
        )
        for run in runs
    ]
    attainment = [
        run.slo_attainment(slo_ttft_s, slo_tpot_s) for run in runs
    ]
    slo_goodput = [
        run.slo_goodput_per_s(slo_ttft_s, slo_tpot_s) for run in runs
    ]
    admission_fraction = [run.admission_fraction for run in runs]
    network_energy_per_token = [
        run.network_energy_j / run.output_tokens for run in runs
    ]
    has_system_energy = [
        run.system_energy_j is not None for run in runs
    ]
    if any(has_system_energy) and not all(has_system_energy):
        raise ValidationError(
            "serving aggregate mixes system-energy and network-only runs"
        )
    system_energy_per_token = (
        [
            run.system_energy_j / run.output_tokens
            for run in runs
        ]
        if all(has_system_energy) else None
    )
    return ServingAggregate(
        throughput=_confidence_interval(throughput, lower_bound=0.0),
        mean_ttft_s=_confidence_interval(mean_ttft, lower_bound=0.0),
        mean_tpot_s=_confidence_interval(mean_tpot, lower_bound=0.0),
        slo_attainment=_confidence_interval(
            attainment, lower_bound=0.0, upper_bound=1.0
        ),
        slo_goodput_per_s=_confidence_interval(
            slo_goodput, lower_bound=0.0
        ),
        admission_fraction=_confidence_interval(
            admission_fraction, lower_bound=0.0, upper_bound=1.0
        ),
        network_energy_per_token_j=_confidence_interval(
            network_energy_per_token, lower_bound=0.0
        ),
        system_energy_per_token_j=(
            _confidence_interval(
                system_energy_per_token, lower_bound=0.0
            )
            if system_energy_per_token is not None else None
        ),
    )


def evaluate_slo_capacity(
    simulator: ContinuousBatchingSimulator,
    offered_rates_per_s: Iterable[float],
    seeds: Iterable[int],
    n_requests: int,
    output_tokens,
    slo_ttft_s: float,
    slo_tpot_s: float,
    target_attainment: float = 0.99,
) -> SLOCapacityResult:
    """Evaluate a declared load grid and conservatively select SLO capacity."""

    rates = tuple(sorted(set(float(rate) for rate in offered_rates_per_s)))
    seed_values = tuple(sorted(set(int(seed) for seed in seeds)))
    if not rates or any(not math.isfinite(rate) or rate <= 0
                        for rate in rates):
        raise ValidationError("capacity search rates must be positive")
    if not seed_values or any(seed < 0 for seed in seed_values):
        raise ValidationError("capacity search seeds must be non-negative")
    if n_requests <= 0:
        raise ValidationError("capacity search n_requests must be positive")
    if not 0 < target_attainment <= 1:
        raise ValidationError("target attainment must be in (0, 1]")

    points = []
    for rate in rates:
        runs = []
        for seed in seed_values:
            requests = generate_poisson_requests(
                n_requests=n_requests,
                arrival_rate_per_s=rate,
                output_tokens=output_tokens,
                seed=seed,
            )
            runs.append(simulator.run(requests))
        aggregate = aggregate_serving_results(
            runs, slo_ttft_s, slo_tpot_s
        )
        points.append(SLOCapacityPoint(
            offered_rate_per_s=rate,
            aggregate=aggregate,
            lower_bound_meets_target=(
                aggregate.slo_attainment.low95 >= target_attainment
            ),
        ))

    attainment_means = [
        point.aggregate.slo_attainment.mean for point in points
    ]
    monotonic = all(
        right <= left + 1e-12
        for left, right in zip(
            attainment_means, attainment_means[1:]
        )
    )
    qualifying_prefix = []
    for point in points:
        if not point.lower_bound_meets_target:
            break
        qualifying_prefix.append(point.offered_rate_per_s)
    has_later_qualifying_point = any(
        point.lower_bound_meets_target
        for point in points[len(qualifying_prefix) + 1:]
    )
    warning = ""
    if not monotonic:
        warning = (
            "SLO attainment is non-monotonic across the tested load grid; "
            "increase request count/seeds before claiming capacity"
        )
    if has_later_qualifying_point:
        warning = (
            warning + ("; " if warning else "") +
            "non-contiguous qualifying points were excluded from capacity"
        )
    if not qualifying_prefix:
        capacity = 0.0
        warning = (
            warning + ("; " if warning else "") +
            "the smallest tested load does not satisfy the "
            "lower-confidence SLO target"
        )
    else:
        capacity = qualifying_prefix[-1]
        if capacity == rates[-1]:
            warning = (
                warning + ("; " if warning else "") +
                "capacity is right-censored by the largest tested load"
            )
    return SLOCapacityResult(
        capacity_rate_per_s=capacity,
        target_attainment=target_attainment,
        points=tuple(points),
        monotonic_attainment=monotonic,
        warning=warning,
    )
