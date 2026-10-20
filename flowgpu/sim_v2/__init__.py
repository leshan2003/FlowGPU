"""FlowGPU Simulator v2 foundation.

This package is intentionally separate from :mod:`flowgpu.sim` (v1).  The
public surface will grow as calibrated kernel, detailed network, dispatch,
and serving models pass their validation tiers.
"""

from .event_engine import Event, EventQueue
from .energy import (ChipPowerProfile, EnergyBreakdown,
                     account_system_energy)
from .experiments import (NodeSweepCase, NodeSweepSpec, SweepTracePair,
                          SweepWorkloadSpec, build_case_trace_pair,
                          build_node_count_sweep,
                          classify_capacity_regime)
from .fast_engine import (FastDetailedComparison, FastLayerResult,
                          FastMoELayerEngine, FastReplicaExecution,
                          compare_fast_detailed_direction)
from .calibration import (BenchmarkArtifact, CalibratedLinkFit,
                          KernelCalibrationSet, KernelCurveKey,
                          NetworkCalibrationSet, NetworkFitKey,
                          build_kernel_calibration,
                          build_network_calibration,
                          load_benchmark_artifact)
from .dispatcher import (AssignmentBinding, DispatchPlan, Dispatcher,
                         ReplicaBatch)
from .detailed_network import (DetailedFlowResult, FluidNetwork,
                               simulate_flows)
from .demand import DemandProfile, profile_demand
from .kernel_model import (AnalyticExpertKernel, ExpertKernelCurve,
                           KernelPrediction, KernelSample)
from .layer_engine import (LayerCriticalPath, MoELayerEngine, MoELayerResult,
                           ReplicaExecution)
from .network_model import (Flow, FlowResult, SerializedPathScheduler,
                            estimate_unloaded)
from .multicast import (MulticastResult, MulticastTree,
                        build_multicast_tree, simulate_multicast)
from .oracle import (OracleGapReport, OracleResult, audit_oracle_gap,
                     small_replica_oracle)
from .placement import (capacity_ratio, coactivation_clustered, ingress_local,
                        joint_greedy, layer_packed, node_load_replicated,
                        popularity_replicated, primary_coactivation_cut,
                        unique_balanced)
from .replicas import CapacityUsage, ReplicaPlan
from .results import (RESULT_SCHEMA_VERSION,
                      write_comparison_result_bundle,
                      write_layer_result_bundle,
                      write_node_sweep_bundle,
                      write_serving_sweep_bundle,
                      write_tiered_sweep_bundle,
                      write_uncertainty_bundle)
from .runner import (FIXED_WINDOW_STRATEGIES, FixedWindowComparison,
                     FixedWindowStrategyResult,
                     NodeSweepComparisonResult, NodeSweepMetricPoint,
                     ServingNodeSweepResult, ServingSweepPoint,
                     TieredNodeSweepPoint, TieredNodeSweepResult,
                     UncertainNodeSweepPoint, UncertainNodeSweepResult,
                     run_fixed_window_comparison,
                     run_node_count_comparisons,
                     run_node_count_uncertainty,
                     run_node_serving_sweep,
                     run_tiered_node_count_comparisons,
                     run_tiered_fixed_window_comparison)
from .routing import (RoutingSummary, RoutingTrace, generate_uniform_trace,
                      generate_weighted_trace, load_routing_trace_jsonl,
                      zipf_expert_weights)
from .research import (PlanExecution, TaskAwareTrace,
                       adaptive_replica_share,
                       build_hbm_fallback_plan,
                       execute_hbm_fallback_plan,
                       generate_task_community_trace,
                       hybrid_replica_plan, migration_bytes,
                       plan_replica_locations)
from .schema import (Assignment, ChipSpec, Confidence, ExpertKey, ExpertSpec,
                     Provenance, ReplicaSpec, RoutingRecord,
                     RuntimeCapabilities, ValidationError)
from .serving import (AdmissionPolicy, ContinuousBatchingSimulator,
                      DecodeRequest,
                      ConfidenceInterval, RequestServingResult, RouteChoice,
                      ResourceBlockWindow,
                      SLOCapacityPoint, SLOCapacityResult, ServingAggregate,
                      ServingResult, UniformServingRouter,
                      WeightedServingRouter,
                      aggregate_serving_results, evaluate_slo_capacity,
                      confidence_interval,
                      generate_poisson_requests, load_decode_request_trace)
from .topology import (ClusterTopology, Endpoint, LinkProfile, LinkSpec, Route,
                       Topology, UnreachableError, build_hierarchical_cluster)
from .tiering import make_gpu_tier_chip, tiered_expert_plan
from .validation import validate_layer_result

__all__ = [
    "AdmissionPolicy", "AnalyticExpertKernel", "Assignment",
    "AssignmentBinding",
    "BenchmarkArtifact", "CalibratedLinkFit", "CapacityUsage", "ChipSpec",
    "ChipPowerProfile", "ClusterTopology", "Confidence",
    "ConfidenceInterval",
    "ContinuousBatchingSimulator",
    "DecodeRequest", "DemandProfile", "DetailedFlowResult", "DispatchPlan",
    "Dispatcher",
    "EnergyBreakdown",
    "Endpoint", "Event", "EventQueue", "FastDetailedComparison",
    "FastLayerResult", "FastMoELayerEngine", "FastReplicaExecution",
    "FIXED_WINDOW_STRATEGIES", "FixedWindowComparison",
    "FixedWindowStrategyResult",
    "ExpertKernelCurve", "ExpertKey", "ExpertSpec", "Flow", "FlowResult",
    "FluidNetwork", "KernelCalibrationSet", "KernelCurveKey",
    "KernelPrediction", "KernelSample", "LayerCriticalPath",
    "LinkProfile", "LinkSpec", "MoELayerEngine", "MoELayerResult",
    "MulticastResult", "MulticastTree", "NetworkCalibrationSet",
    "NetworkFitKey", "NodeSweepCase", "NodeSweepSpec", "OracleGapReport",
    "NodeSweepComparisonResult", "NodeSweepMetricPoint", "OracleResult",
    "Provenance", "RESULT_SCHEMA_VERSION", "ReplicaBatch",
    "PlanExecution", "TaskAwareTrace",
    "adaptive_replica_share",
    "ReplicaExecution", "ReplicaPlan", "ReplicaSpec",
    "RequestServingResult", "Route", "RouteChoice",
    "ResourceBlockWindow",
    "RoutingRecord", "RoutingSummary", "RoutingTrace",
    "RuntimeCapabilities", "SLOCapacityPoint", "SLOCapacityResult",
    "SerializedPathScheduler", "ServingAggregate", "ServingResult",
    "ServingNodeSweepResult", "ServingSweepPoint",
    "SweepTracePair", "SweepWorkloadSpec", "Topology",
    "TieredNodeSweepPoint", "TieredNodeSweepResult",
    "UncertainNodeSweepPoint", "UncertainNodeSweepResult",
    "UniformServingRouter",
    "WeightedServingRouter",
    "UnreachableError", "ValidationError", "estimate_unloaded",
    "account_system_energy", "audit_oracle_gap", "build_case_trace_pair",
    "build_hierarchical_cluster",
    "build_hbm_fallback_plan",
    "build_kernel_calibration",
    "build_multicast_tree", "build_network_calibration",
    "build_node_count_sweep", "capacity_ratio",
    "aggregate_serving_results", "coactivation_clustered",
    "compare_fast_detailed_direction", "evaluate_slo_capacity",
    "confidence_interval",
    "classify_capacity_regime",
    "generate_poisson_requests",
    "generate_uniform_trace", "ingress_local",
    "generate_weighted_trace",
    "generate_task_community_trace",
    "execute_hbm_fallback_plan",
    "hybrid_replica_plan",
    "joint_greedy", "layer_packed", "load_benchmark_artifact",
    "load_decode_request_trace",
    "load_routing_trace_jsonl",
    "migration_bytes",
    "node_load_replicated", "popularity_replicated",
    "make_gpu_tier_chip", "tiered_expert_plan",
    "primary_coactivation_cut", "profile_demand", "simulate_flows",
    "plan_replica_locations",
    "run_fixed_window_comparison",
    "run_node_count_comparisons",
    "run_node_count_uncertainty",
    "run_node_serving_sweep",
    "run_tiered_node_count_comparisons",
    "run_tiered_fixed_window_comparison",
    "simulate_multicast", "small_replica_oracle", "unique_balanced",
    "validate_layer_result", "write_layer_result_bundle",
    "write_comparison_result_bundle",
    "write_node_sweep_bundle",
    "write_serving_sweep_bundle", "write_tiered_sweep_bundle",
    "write_uncertainty_bundle",
    "zipf_expert_weights",
]
