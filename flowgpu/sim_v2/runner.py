"""End-to-end fixed-window strategy comparison for node-sweep cases."""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import replace
from typing import Mapping, Optional, Tuple

from .demand import profile_demand
from .dispatcher import Dispatcher
from .experiments import (NodeSweepCase, SweepTracePair, SweepWorkloadSpec,
                          build_case_trace_pair)
from .layer_engine import MoELayerEngine, MoELayerResult
from .placement import (
    capacity_ratio,
    coactivation_clustered,
    ingress_local,
    layer_packed,
    node_load_replicated,
    popularity_replicated,
    unique_balanced,
)
from .replicas import ReplicaPlan
from .schema import (ExpertKey, ExpertSpec, RuntimeCapabilities,
                     ValidationError)
from .serving import (
    AdmissionPolicy,
    ConfidenceInterval,
    ContinuousBatchingSimulator,
    SLOCapacityResult,
    UniformServingRouter,
    WeightedServingRouter,
    confidence_interval,
    evaluate_slo_capacity,
)
from .tiering import make_gpu_tier_chip, tiered_expert_plan


FIXED_WINDOW_STRATEGIES = (
    "unique-balanced",
    "layer-packed",
    "popularity-replicated",
    "ingress-local",
    "node-load-replicated",
    "coactivation-clustered",
)


@dataclass(frozen=True)
class FixedWindowStrategyResult:
    strategy: str
    replica_plan: ReplicaPlan
    layer_results: Tuple[MoELayerResult, ...]
    completion_s: float
    assignment_count: int
    network_energy_j: float
    local_assignment_fraction: float


@dataclass(frozen=True)
class FixedWindowComparison:
    case_id: str
    ingress_semantics: str
    dispatcher_policy: str
    evaluation_split: str
    strategies: Tuple[FixedWindowStrategyResult, ...]

    def result_for(self, strategy: str) -> FixedWindowStrategyResult:
        for result in self.strategies:
            if result.strategy == strategy:
                return result
        raise KeyError(strategy)


@dataclass(frozen=True)
class NodeSweepMetricPoint:
    case_id: str
    node_count: int
    ingress_mode: str
    strategy: str
    capacity_ratio: float
    resident_replicas: int
    completion_s: float
    tokens_per_s: float
    tokens_per_s_per_node: float
    speedup_vs_unique: float
    local_assignment_fraction: float
    network_energy_j: float


@dataclass(frozen=True)
class NodeSweepComparisonResult:
    scaling_mode: str
    replica_budget_mode: str
    fixed_max_total_replicas: Optional[int]
    comparisons: Tuple[FixedWindowComparison, ...]
    points: Tuple[NodeSweepMetricPoint, ...]


@dataclass(frozen=True)
class ServingSweepPoint:
    case_id: str
    node_count: int
    ingress_mode: str
    strategy: str
    capacity_rate_per_s: float
    capacity_per_node_per_s: float
    target_attainment: float
    right_censored: bool
    monotonic_attainment: bool
    warning: str
    capacity_result: SLOCapacityResult


@dataclass(frozen=True)
class ServingNodeSweepResult:
    offered_rates_per_s: Tuple[float, ...]
    seeds: Tuple[int, ...]
    points: Tuple[ServingSweepPoint, ...]


@dataclass(frozen=True)
class UncertainNodeSweepPoint:
    case_id: str
    node_count: int
    ingress_mode: str
    strategy: str
    runs: int
    completion_s: ConfidenceInterval
    tokens_per_s: ConfidenceInterval
    tokens_per_s_per_node: ConfidenceInterval
    speedup_vs_unique: ConfidenceInterval
    local_assignment_fraction: ConfidenceInterval
    network_energy_j: ConfidenceInterval
    conclusion: str


@dataclass(frozen=True)
class UncertainNodeSweepResult:
    placement_seeds: Tuple[int, ...]
    evaluation_seeds: Tuple[int, ...]
    points: Tuple[UncertainNodeSweepPoint, ...]


@dataclass(frozen=True)
class TieredNodeSweepPoint:
    case_id: str
    node_count: int
    ingress_mode: str
    policy: str
    brain_replicas: int
    brain_unique_experts: int
    gpu_fallback_experts: int
    completion_s: float
    tokens_per_s: float
    tokens_per_s_per_node: float
    speedup_vs_unique_first: float
    local_assignment_fraction: float
    network_energy_j: float


@dataclass(frozen=True)
class TieredNodeSweepResult:
    max_brain_replicas: Optional[int]
    points: Tuple[TieredNodeSweepPoint, ...]
    comparisons: Tuple[FixedWindowComparison, ...]


def _make_plan(
    strategy: str,
    experts,
    chips,
    demand,
    max_total_replicas: Optional[int],
) -> ReplicaPlan:
    if strategy == "unique-balanced":
        expected = {
            expert.key: demand.global_for(expert.key)
            for expert in experts
        }
        return unique_balanced(experts, chips, expected)
    if strategy == "layer-packed":
        return layer_packed(experts, chips)
    if strategy == "popularity-replicated":
        return popularity_replicated(
            experts, chips, demand, max_total_replicas
        )
    if strategy == "ingress-local":
        return ingress_local(
            experts, chips, demand, max_total_replicas
        )
    if strategy == "node-load-replicated":
        return node_load_replicated(
            experts, chips, demand, max_total_replicas
        )
    if strategy == "coactivation-clustered":
        return coactivation_clustered(experts, chips, demand)
    raise ValidationError(
        "unsupported fixed-window strategy: %s" % strategy
    )


def run_fixed_window_comparison(
    case: NodeSweepCase,
    traces: SweepTracePair,
    experts: Tuple[ExpertSpec, ...],
    kernel_models: Mapping[ExpertKey, object],
    hidden_bytes: int,
    strategies=FIXED_WINDOW_STRATEGIES,
    max_total_replicas: Optional[int] = None,
    dispatcher_policy: str = "earliest_finish",
    capabilities: RuntimeCapabilities = RuntimeCapabilities(),
    ingress_semantics: str = "home-node-resident",
    chip_reduce_latency_s: float = 0.0,
    node_reduce_latency_s: float = 0.0,
) -> FixedWindowComparison:
    """Run held-out assignments through several fairly configured plans."""

    expert_list = tuple(experts)
    strategy_names = tuple(strategies)
    if traces.case_id != case.case_id:
        raise ValidationError("trace pair belongs to a different sweep case")
    if traces.placement.split != "placement":
        raise ValidationError("comparison requires a placement trace")
    if traces.evaluation.split != "evaluation":
        raise ValidationError("comparison requires held-out evaluation trace")
    if not expert_list:
        raise ValidationError("comparison needs experts")
    if hidden_bytes <= 0:
        raise ValidationError("comparison hidden_bytes must be positive")
    if not strategy_names or len(set(strategy_names)) != len(strategy_names):
        raise ValidationError(
            "comparison strategies must be non-empty and unique"
        )
    if ingress_semantics not in ("home-node-resident", "gpu-origin"):
        raise ValidationError(
            "ingress_semantics must be home-node-resident or gpu-origin"
        )
    expert_keys = {expert.key for expert in expert_list}
    ratio = capacity_ratio(expert_list, case.chips)
    if ratio < 1.0 - 1e-12:
        raise ValidationError(
            "fixed-window replica comparison requires R >= 1; "
            "R=%.6g needs a separate spill/tiering execution model" % ratio
        )
    if set(kernel_models) != expert_keys:
        raise ValidationError(
            "kernel model keys must exactly match comparison experts"
        )
    evaluation_assignments = traces.evaluation.assignments()
    if any(
        assignment.expert not in expert_keys
        for assignment in evaluation_assignments
    ):
        raise ValidationError(
            "evaluation trace references an unknown expert"
        )
    if any(
        assignment.home_node not in case.routing_home_nodes
        for assignment in evaluation_assignments
    ):
        raise ValidationError(
            "evaluation trace references a home outside the case"
        )
    demand = profile_demand(traces.placement)
    by_layer = {}
    for assignment in evaluation_assignments:
        by_layer.setdefault(assignment.layer_id, []).append(assignment)
    if set(by_layer) != {key.layer_id for key in expert_keys}:
        raise ValidationError(
            "evaluation layers must exactly match expert layers"
        )

    outputs = []
    for strategy in strategy_names:
        plan = _make_plan(
            strategy, expert_list, case.chips, demand,
            max_total_replicas,
        )
        dispatcher = Dispatcher(
            replica_plan=plan,
            topology=case.cluster.topology,
            home_endpoints=case.cluster.home_endpoints,
            chip_endpoints=case.cluster.chip_endpoints,
            kernel_models=kernel_models,
            hidden_bytes=hidden_bytes,
            policy=dispatcher_policy,
            ingress_source_endpoint=(
                case.cluster.gpu_endpoint
                if ingress_semantics == "gpu-origin" else None
            ),
        )
        engine = MoELayerEngine(
            dispatcher=dispatcher,
            capabilities=capabilities,
            chip_reduce_latency_s=chip_reduce_latency_s,
            node_reduce_latency_s=node_reduce_latency_s,
        )
        now_s = 0.0
        layer_results = []
        for layer_id in sorted(by_layer):
            result = engine.run(
                by_layer[layer_id], start_time_s=now_s
            )
            layer_results.append(result)
            now_s = result.completion_s
        assignment_count = sum(
            result.assignment_count for result in layer_results
        )
        weighted_local = sum(
            result.local_assignment_fraction * result.assignment_count
            for result in layer_results
        ) / assignment_count
        if not math.isfinite(now_s):
            raise RuntimeError("comparison produced non-finite completion")
        outputs.append(FixedWindowStrategyResult(
            strategy=strategy,
            replica_plan=plan,
            layer_results=tuple(layer_results),
            completion_s=now_s,
            assignment_count=assignment_count,
            network_energy_j=sum(
                result.network_energy_j for result in layer_results
            ),
            local_assignment_fraction=weighted_local,
        ))
    return FixedWindowComparison(
        case_id=case.case_id,
        ingress_semantics=ingress_semantics,
        dispatcher_policy=dispatcher_policy,
        evaluation_split=traces.evaluation.split,
        strategies=tuple(outputs),
    )


def run_node_count_comparisons(
    cases: Tuple[NodeSweepCase, ...],
    workload: SweepWorkloadSpec,
    experts: Tuple[ExpertSpec, ...],
    kernel_models: Mapping[ExpertKey, object],
    hidden_bytes: int,
    strategies=FIXED_WINDOW_STRATEGIES,
    max_total_replicas: Optional[int] = None,
    dispatcher_policy: str = "earliest_finish",
    capabilities: RuntimeCapabilities = RuntimeCapabilities(),
    ingress_semantics: str = "home-node-resident",
    chip_reduce_latency_s: float = 0.0,
    node_reduce_latency_s: float = 0.0,
) -> NodeSweepComparisonResult:
    """Execute a declared node-count grid and derive scaling metrics."""

    case_list = tuple(cases)
    expert_list = tuple(experts)
    strategy_names = tuple(strategies)
    if not case_list:
        raise ValidationError("node comparison sweep needs cases")
    if len({case.case_id for case in case_list}) != len(case_list):
        raise ValidationError("node comparison case ids must be unique")
    fingerprints = {
        case.per_node_hardware_fingerprint for case in case_list
    }
    if len(fingerprints) != 1:
        raise ValidationError(
            "node comparison cases change per-node hardware"
        )
    if "unique-balanced" not in strategy_names:
        raise ValidationError(
            "node comparison strategies require unique-balanced baseline"
        )
    if max_total_replicas is not None and max_total_replicas <= 0:
        raise ValidationError(
            "fixed max_total_replicas must be positive"
        )
    if (max_total_replicas is not None and
            max_total_replicas < len(expert_list)):
        raise ValidationError(
            "fixed replica budget is below the unique expert count"
        )
    comparisons = []
    points = []
    for case in case_list:
        traces = build_case_trace_pair(case, workload)
        comparison = run_fixed_window_comparison(
            case=case,
            traces=traces,
            experts=expert_list,
            kernel_models=kernel_models,
            hidden_bytes=hidden_bytes,
            strategies=strategy_names,
            max_total_replicas=max_total_replicas,
            dispatcher_policy=dispatcher_policy,
            capabilities=capabilities,
            ingress_semantics=ingress_semantics,
            chip_reduce_latency_s=chip_reduce_latency_s,
            node_reduce_latency_s=node_reduce_latency_s,
        )
        comparisons.append(comparison)
        baseline = comparison.result_for("unique-balanced")
        ratio = capacity_ratio(expert_list, case.chips)
        for result in comparison.strategies:
            tokens_per_s = (
                traces.evaluation_token_count / result.completion_s
            )
            points.append(NodeSweepMetricPoint(
                case_id=case.case_id,
                node_count=case.node_count,
                ingress_mode=case.ingress_mode,
                strategy=result.strategy,
                capacity_ratio=ratio,
                resident_replicas=len(result.replica_plan.replicas),
                completion_s=result.completion_s,
                tokens_per_s=tokens_per_s,
                tokens_per_s_per_node=tokens_per_s / case.node_count,
                speedup_vs_unique=(
                    baseline.completion_s / result.completion_s
                ),
                local_assignment_fraction=
                    result.local_assignment_fraction,
                network_energy_j=result.network_energy_j,
            ))
    return NodeSweepComparisonResult(
        scaling_mode=workload.scaling_mode,
        replica_budget_mode=(
            "all-available-capacity"
            if max_total_replicas is None else "fixed-total-replicas"
        ),
        fixed_max_total_replicas=max_total_replicas,
        comparisons=tuple(comparisons),
        points=tuple(points),
    )


def run_tiered_fixed_window_comparison(
    case: NodeSweepCase,
    traces: SweepTracePair,
    experts: Tuple[ExpertSpec, ...],
    brain_kernel_models: Mapping[ExpertKey, object],
    gpu_kernel_models: Mapping[ExpertKey, object],
    gpu_provenance,
    hidden_bytes: int,
    policies=("unique-first", "hot-replica-first"),
    max_brain_replicas: Optional[int] = None,
    gpu_capacity_bytes: Optional[int] = None,
    gpu_chip_id: str = "gpu-tier0",
    dispatcher_policy: str = "earliest_finish",
    capabilities: RuntimeCapabilities = RuntimeCapabilities(),
    ingress_semantics: str = "gpu-origin",
) -> FixedWindowComparison:
    """Compare how brain SRAM is spent while every expert has GPU fallback."""

    expert_list = tuple(experts)
    policy_names = tuple(policies)
    if traces.case_id != case.case_id:
        raise ValidationError("tiered traces belong to another case")
    if (traces.placement.split != "placement" or
            traces.evaluation.split != "evaluation"):
        raise ValidationError(
            "tiered comparison requires placement and held-out traces"
        )
    if set(brain_kernel_models) != {item.key for item in expert_list}:
        raise ValidationError(
            "brain kernel keys must exactly match experts"
        )
    if set(gpu_kernel_models) != {item.key for item in expert_list}:
        raise ValidationError("GPU kernel keys must exactly match experts")
    if not policy_names or len(set(policy_names)) != len(policy_names):
        raise ValidationError("tiering policies must be non-empty and unique")
    if ingress_semantics not in ("gpu-origin", "home-node-resident"):
        raise ValidationError("invalid tiered ingress semantics")
    demand = profile_demand(traces.placement)
    expected = {
        expert.key: demand.global_for(expert.key)
        for expert in expert_list
    }
    gpu_chip = make_gpu_tier_chip(
        expert_list, gpu_chip_id, gpu_provenance,
        capacity_bytes=gpu_capacity_bytes,
    )
    assignments = traces.evaluation.assignments()
    by_layer = {}
    for assignment in assignments:
        by_layer.setdefault(assignment.layer_id, []).append(assignment)
    outputs = []
    for policy in policy_names:
        plan = tiered_expert_plan(
            experts=expert_list,
            brain_chips=case.chips,
            gpu_chip=gpu_chip,
            expected_assignments=expected,
            policy=policy,
            max_brain_replicas=max_brain_replicas,
        )
        kernel_models = {}
        for expert in expert_list:
            kernel_models[(expert.key, gpu_chip_id)] = (
                gpu_kernel_models[expert.key]
            )
            for chip in case.chips:
                kernel_models[(expert.key, chip.chip_id)] = (
                    brain_kernel_models[expert.key]
                )
        chip_endpoints = dict(case.cluster.chip_endpoints)
        chip_endpoints[gpu_chip_id] = case.cluster.gpu_endpoint
        dispatcher = Dispatcher(
            replica_plan=plan,
            topology=case.cluster.topology,
            home_endpoints=case.cluster.home_endpoints,
            chip_endpoints=chip_endpoints,
            kernel_models=kernel_models,
            hidden_bytes=hidden_bytes,
            policy=dispatcher_policy,
            ingress_source_endpoint=(
                case.cluster.gpu_endpoint
                if ingress_semantics == "gpu-origin" else None
            ),
        )
        engine = MoELayerEngine(dispatcher, capabilities)
        now_s = 0.0
        layer_results = []
        for layer_id in sorted(by_layer):
            result = engine.run(
                by_layer[layer_id], start_time_s=now_s
            )
            layer_results.append(result)
            now_s = result.completion_s
        count = sum(item.assignment_count for item in layer_results)
        outputs.append(FixedWindowStrategyResult(
            strategy=plan.strategy,
            replica_plan=plan,
            layer_results=tuple(layer_results),
            completion_s=now_s,
            assignment_count=count,
            network_energy_j=sum(
                item.network_energy_j for item in layer_results
            ),
            local_assignment_fraction=sum(
                item.local_assignment_fraction * item.assignment_count
                for item in layer_results
            ) / count,
        ))
    return FixedWindowComparison(
        case_id=case.case_id,
        ingress_semantics=ingress_semantics,
        dispatcher_policy=dispatcher_policy,
        evaluation_split=traces.evaluation.split,
        strategies=tuple(outputs),
    )


def run_node_serving_sweep(
    cases: Tuple[NodeSweepCase, ...],
    workload: SweepWorkloadSpec,
    experts: Tuple[ExpertSpec, ...],
    kernel_models: Mapping[ExpertKey, object],
    hidden_bytes: int,
    offered_rates_per_s,
    seeds,
    n_requests: int,
    output_tokens,
    max_batch: int,
    slo_ttft_s: float,
    slo_tpot_s: float,
    target_attainment: float,
    strategies=FIXED_WINDOW_STRATEGIES,
    max_total_replicas: Optional[int] = None,
    dispatcher_policy: str = "earliest_finish",
    capabilities: RuntimeCapabilities = RuntimeCapabilities(),
    ingress_semantics: str = "home-node-resident",
    admission_policy: AdmissionPolicy = None,
    chip_power_profile=None,
    compute_block_windows=(),
) -> ServingNodeSweepResult:
    """Run the complete strategy/node/load/seed serving matrix."""

    case_list = tuple(cases)
    expert_list = tuple(experts)
    strategy_names = tuple(strategies)
    rates = tuple(float(value) for value in offered_rates_per_s)
    seed_values = tuple(int(value) for value in seeds)
    if not case_list:
        raise ValidationError("serving node sweep needs cases")
    if len({
        case.per_node_hardware_fingerprint for case in case_list
    }) != 1:
        raise ValidationError(
            "serving sweep cases change per-node hardware"
        )
    if set(kernel_models) != {expert.key for expert in expert_list}:
        raise ValidationError(
            "serving sweep kernel keys must exactly match experts"
        )
    points = []
    for case in case_list:
        traces = build_case_trace_pair(case, workload)
        demand = profile_demand(traces.placement)
        if workload.routing_model == "uniform":
            router = UniformServingRouter(
                n_experts=workload.n_experts,
                top_k=workload.top_k,
                home_nodes=case.routing_home_nodes,
                seed=workload.evaluation_seed,
            )
        else:
            router = WeightedServingRouter(
                expert_weights=workload.evaluation_expert_weights,
                top_k=workload.top_k,
                home_nodes=case.routing_home_nodes,
                seed=workload.evaluation_seed,
            )
        for strategy in strategy_names:
            plan = _make_plan(
                strategy, expert_list, case.chips, demand,
                max_total_replicas,
            )
            dispatcher = Dispatcher(
                replica_plan=plan,
                topology=case.cluster.topology,
                home_endpoints=case.cluster.home_endpoints,
                chip_endpoints=case.cluster.chip_endpoints,
                kernel_models=kernel_models,
                hidden_bytes=hidden_bytes,
                policy=dispatcher_policy,
                ingress_source_endpoint=(
                    case.cluster.gpu_endpoint
                    if ingress_semantics == "gpu-origin" else None
                ),
            )
            simulator = ContinuousBatchingSimulator(
                layer_engine=MoELayerEngine(dispatcher, capabilities),
                router=router,
                n_layers=workload.n_layers,
                max_batch=max_batch,
                admission_policy=admission_policy,
                chip_power_profiles=(
                    {
                        chip.chip_id: chip_power_profile
                        for chip in case.chips
                    }
                    if chip_power_profile is not None else None
                ),
                compute_block_windows=compute_block_windows,
            )
            capacity = evaluate_slo_capacity(
                simulator=simulator,
                offered_rates_per_s=rates,
                seeds=seed_values,
                n_requests=n_requests,
                output_tokens=output_tokens,
                slo_ttft_s=slo_ttft_s,
                slo_tpot_s=slo_tpot_s,
                target_attainment=target_attainment,
            )
            points.append(ServingSweepPoint(
                case_id=case.case_id,
                node_count=case.node_count,
                ingress_mode=case.ingress_mode,
                strategy=strategy,
                capacity_rate_per_s=capacity.capacity_rate_per_s,
                capacity_per_node_per_s=(
                    capacity.capacity_rate_per_s / case.node_count
                ),
                target_attainment=target_attainment,
                right_censored="right-censored" in capacity.warning,
                monotonic_attainment=capacity.monotonic_attainment,
                warning=capacity.warning,
                capacity_result=capacity,
            ))
    return ServingNodeSweepResult(
        offered_rates_per_s=tuple(sorted(set(rates))),
        seeds=tuple(sorted(set(seed_values))),
        points=tuple(points),
    )


def run_node_count_uncertainty(
    cases: Tuple[NodeSweepCase, ...],
    workload: SweepWorkloadSpec,
    placement_seeds,
    evaluation_seeds,
    experts: Tuple[ExpertSpec, ...],
    kernel_models: Mapping[ExpertKey, object],
    hidden_bytes: int,
    strategies=FIXED_WINDOW_STRATEGIES,
    max_total_replicas: Optional[int] = None,
    dispatcher_policy: str = "earliest_finish",
    capabilities: RuntimeCapabilities = RuntimeCapabilities(),
    ingress_semantics: str = "home-node-resident",
) -> UncertainNodeSweepResult:
    """Aggregate paired strategy metrics over placement/evaluation seeds."""

    placement_values = tuple(sorted(set(
        int(value) for value in placement_seeds
    )))
    evaluation_values = tuple(sorted(set(
        int(value) for value in evaluation_seeds
    )))
    if not placement_values or not evaluation_values:
        raise ValidationError("uncertainty sweep needs both seed grids")
    if min(placement_values + evaluation_values) < 0:
        raise ValidationError("uncertainty seeds must be non-negative")
    collected = {}
    run_count = 0
    for placement_seed in placement_values:
        for evaluation_seed in evaluation_values:
            if placement_seed == evaluation_seed:
                continue
            seeded_workload = replace(
                workload,
                placement_seed=placement_seed,
                evaluation_seed=evaluation_seed,
            )
            sweep = run_node_count_comparisons(
                cases=cases,
                workload=seeded_workload,
                experts=experts,
                kernel_models=kernel_models,
                hidden_bytes=hidden_bytes,
                strategies=strategies,
                max_total_replicas=max_total_replicas,
                dispatcher_policy=dispatcher_policy,
                capabilities=capabilities,
                ingress_semantics=ingress_semantics,
            )
            run_count += 1
            for point in sweep.points:
                key = (
                    point.case_id, point.node_count,
                    point.ingress_mode, point.strategy,
                )
                values = collected.setdefault(key, {
                    "completion_s": [],
                    "tokens_per_s": [],
                    "tokens_per_s_per_node": [],
                    "speedup_vs_unique": [],
                    "local_assignment_fraction": [],
                    "network_energy_j": [],
                })
                for metric in values:
                    values[metric].append(getattr(point, metric))
    if run_count == 0:
        raise ValidationError(
            "uncertainty seed grids contain no distinct seed pairs"
        )
    points = []
    for key in sorted(collected):
        values = collected[key]
        speedup = confidence_interval(
            values["speedup_vs_unique"], lower_bound=0.0
        )
        if key[3] == "unique-balanced":
            conclusion = "baseline"
        elif speedup.low95 > 1.0:
            conclusion = "supported"
        elif speedup.high95 <= 1.0:
            conclusion = "unsupported"
        else:
            conclusion = "inconclusive"
        points.append(UncertainNodeSweepPoint(
            case_id=key[0],
            node_count=key[1],
            ingress_mode=key[2],
            strategy=key[3],
            runs=len(values["completion_s"]),
            completion_s=confidence_interval(
                values["completion_s"], lower_bound=0.0
            ),
            tokens_per_s=confidence_interval(
                values["tokens_per_s"], lower_bound=0.0
            ),
            tokens_per_s_per_node=confidence_interval(
                values["tokens_per_s_per_node"], lower_bound=0.0
            ),
            speedup_vs_unique=speedup,
            local_assignment_fraction=confidence_interval(
                values["local_assignment_fraction"],
                lower_bound=0.0, upper_bound=1.0,
            ),
            network_energy_j=confidence_interval(
                values["network_energy_j"], lower_bound=0.0
            ),
            conclusion=conclusion,
        ))
    return UncertainNodeSweepResult(
        placement_seeds=placement_values,
        evaluation_seeds=evaluation_values,
        points=tuple(points),
    )


def run_tiered_node_count_comparisons(
    cases: Tuple[NodeSweepCase, ...],
    workload: SweepWorkloadSpec,
    experts: Tuple[ExpertSpec, ...],
    brain_kernel_models: Mapping[ExpertKey, object],
    gpu_kernel_models: Mapping[ExpertKey, object],
    gpu_provenance,
    hidden_bytes: int,
    max_brain_replicas: Optional[int] = None,
    gpu_capacity_bytes: Optional[int] = None,
    dispatcher_policy: str = "earliest_finish",
    ingress_semantics: str = "gpu-origin",
) -> TieredNodeSweepResult:
    """Answer whether added brain capacity should cover cold or hot copies."""

    case_list = tuple(cases)
    expert_list = tuple(experts)
    if not case_list:
        raise ValidationError("tiered node sweep needs cases")
    if len({
        case.per_node_hardware_fingerprint for case in case_list
    }) != 1:
        raise ValidationError(
            "tiered node sweep changes per-node hardware"
        )
    comparisons = []
    points = []
    for case in case_list:
        traces = build_case_trace_pair(case, workload)
        comparison = run_tiered_fixed_window_comparison(
            case=case,
            traces=traces,
            experts=expert_list,
            brain_kernel_models=brain_kernel_models,
            gpu_kernel_models=gpu_kernel_models,
            gpu_provenance=gpu_provenance,
            hidden_bytes=hidden_bytes,
            max_brain_replicas=max_brain_replicas,
            gpu_capacity_bytes=gpu_capacity_bytes,
            dispatcher_policy=dispatcher_policy,
            ingress_semantics=ingress_semantics,
        )
        comparisons.append(comparison)
        baseline = comparison.result_for("tiered-unique-first")
        for result in comparison.strategies:
            brain_replicas = [
                replica for replica in result.replica_plan.replicas
                if result.replica_plan.chip(
                    replica.chip_id
                ).node_id != "gpu-tier"
            ]
            tokens_per_s = (
                traces.evaluation_token_count / result.completion_s
            )
            points.append(TieredNodeSweepPoint(
                case_id=case.case_id,
                node_count=case.node_count,
                ingress_mode=case.ingress_mode,
                policy=result.strategy,
                brain_replicas=len(brain_replicas),
                brain_unique_experts=len({
                    replica.expert for replica in brain_replicas
                }),
                gpu_fallback_experts=len(expert_list),
                completion_s=result.completion_s,
                tokens_per_s=tokens_per_s,
                tokens_per_s_per_node=tokens_per_s / case.node_count,
                speedup_vs_unique_first=(
                    baseline.completion_s / result.completion_s
                ),
                local_assignment_fraction=
                    result.local_assignment_fraction,
                network_energy_j=result.network_energy_j,
            ))
    return TieredNodeSweepResult(
        max_brain_replicas=max_brain_replicas,
        points=tuple(points),
        comparisons=tuple(comparisons),
    )
