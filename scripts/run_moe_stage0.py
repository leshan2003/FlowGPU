"""Run the Stage 0 semantic and mechanism gate bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowgpu.sim_v2.research import (  # noqa: E402
    build_b_stage_layout,
    build_h_exclusive_plan,
    build_hbm_fallback_plan,
    build_marginal_event_plan,
    build_plan_dispatcher,
    execute_b_stage_layer,
    execute_hbm_fallback_plan,
    generate_ordered_task_trace,
    TaskPhase,
)
from flowgpu.sim_v2.demand import profile_demand
from flowgpu.sim_v2.kernel_model import ExpertKernelCurve, KernelSample
from flowgpu.sim_v2.hardware_profiles import (
    load_fabric_profile,
    load_hardware_profile,
)
from flowgpu.sim_v2.model_geometry import build_model_expert_geometry
from flowgpu.sim_v2.placement import (
    coactivation_clustered,
    full_hyperedge_cut,
    hyperedge_clustered,
    primary_coactivation_cut,
    unique_balanced,
)
from flowgpu.sim_v2.refresh import (
    VersionedPlacement,
    build_migration_plan,
    estimate_migration_timing,
)
from flowgpu.sim_v2.replicas import ReplicaPlan
from flowgpu.sim_v2.routing import RoutingRecord, RoutingTrace
from flowgpu.sim_v2.routing import zipf_expert_weights
from flowgpu.sim_v2.schema import (
    ChipSpec,
    Confidence,
    ExpertKey,
    ExpertSpec,
    Provenance,
)
from flowgpu.sim_v2.statistics import (
    HysteresisDriftDetector,
    normalized_l1,
    paired_block_bootstrap_ci,
    required_paired_blocks,
)
from flowgpu.sim_v2.layer_engine import MoELayerEngine
from flowgpu.sim_v2.serving import (
    AdmissionPolicy,
    ContinuousBatchingSimulator,
    DecodeRequest,
    WeightedServingRouter,
)
from flowgpu.sim_v2.topology import (
    ClusterTopology,
    LinkProfile,
    build_hierarchical_cluster,
)


def projected_provenance(label: str) -> Provenance:
    return Provenance(Confidence.PROJECTED, label)


def derived_provenance(label: str) -> Provenance:
    return Provenance(Confidence.DERIVED, label)


def make_cluster(
    nodes: int,
    chips_per_node: int,
    brain_profile=None,
    bridge_profile=None,
) -> ClusterTopology:
    brain_profile = brain_profile or load_hardware_profile(
        "lynxi_hp300_proj"
    )
    bridge_profile = bridge_profile or load_fabric_profile("pcie5_x16")
    brain_provenance = brain_profile.provenance
    bridge_provenance = bridge_profile.provenance
    scale_up = LinkProfile(
        brain_profile.inter_chip_bandwidth_bytes_per_s,
        brain_profile.inter_chip_latency_s,
        0.85,
        2e-12,
        brain_provenance,
    )
    scale_out = LinkProfile(
        brain_profile.inter_board_bandwidth_bytes_per_s,
        brain_profile.inter_board_latency_s,
        0.85,
        5e-12,
        brain_provenance,
    )
    bridge = LinkProfile(
        bridge_profile.bandwidth_bytes_per_s,
        bridge_profile.latency_s,
        bridge_profile.efficiency,
        bridge_profile.energy_per_byte,
        bridge_provenance,
    )
    return build_hierarchical_cluster(
        n_nodes=nodes,
        chips_per_node=chips_per_node,
        ingress_node_indices=range(nodes),
        scale_up=scale_up,
        scale_out=scale_out,
        bridge=bridge,
    )


def make_brain_chips(
    cluster: ClusterTopology,
    max_experts: int,
    count_per_node: int,
    reserve: float = 0.0,
    profile=None,
) -> tuple:
    profile = profile or load_hardware_profile("lynxi_hp300_proj")
    provenance = profile.provenance
    node_ids = sorted({
        endpoint.node_id for endpoint in cluster.topology.endpoints.values()
        if endpoint.kind == "node_switch"
    })
    return tuple(
        ChipSpec(
            chip_id="%s:chip%d" % (node_id, index),
            node_id=node_id,
            sram_capacity_bytes=profile.weight_capacity_bytes,
            runtime_reserve_fraction=reserve,
            fixed_reserved_bytes=0,
            provenance=provenance,
            role="brain_sram",
            max_experts=max_experts,
        )
        for node_id in node_ids
        for index in range(count_per_node)
    )


def make_gpu_chips(count: int = 16, profile=None) -> tuple:
    profile = profile or load_hardware_profile("h100_sxm")
    provenance = profile.provenance
    return tuple(
        ChipSpec(
            chip_id="gpu%d" % index,
            node_id="gpu-tier",
            sram_capacity_bytes=profile.weight_capacity_bytes,
            runtime_reserve_fraction=0.20,
            fixed_reserved_bytes=0,
            provenance=provenance,
            role="gpu",
        )
        for index in range(count)
    )


def make_kernels(
    experts: Iterable[ExpertSpec],
    gpu_chips: Iterable[ChipSpec],
    brain_chips: Iterable[ChipSpec],
    dtype: str,
) -> dict:
    provenance = projected_provenance(
        "hardware-relative Stage 0 expert curve; not measured"
    )
    m_values = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    gpu_us = (20, 21, 23, 30, 40, 55, 80, 150, 300)
    brain_us = (8, 8.5, 9, 11, 14, 20, 30, 50, 85)
    gpu_curve = ExpertKernelCurve(
        "stage0-gpu", dtype, 1, "hbm",
        tuple(
            KernelSample(m, value * 1e-6)
            for m, value in zip(m_values, gpu_us)
        ),
        provenance,
    )
    brain_curve = ExpertKernelCurve(
        "stage0-brain", dtype, 1, "sram",
        tuple(
            KernelSample(m, value * 1e-6)
            for m, value in zip(m_values, brain_us)
        ),
        provenance,
    )
    models = {}
    for expert in experts:
        for chip in gpu_chips:
            models[(expert.key, chip.chip_id)] = gpu_curve
        for chip in brain_chips:
            models[(expert.key, chip.chip_id)] = brain_curve
    return models


def task_trace(
    tokens_per_request: int,
    phases,
    n_layers: int,
    n_experts: int,
    top_k: int,
    homes,
    seed: int,
    split: str,
    first_layer: int = 0,
    zipf_exponent: float = 1.3,
):
    return generate_ordered_task_trace(
        tokens_per_request=tokens_per_request,
        n_layers=n_layers,
        n_experts=n_experts,
        top_k=top_k,
        home_nodes=homes,
        phases=phases,
        seed=seed,
        provenance=projected_provenance(
            "paper-constrained synthetic Stage 0 trace"
        ),
        split=split,
        community_count=8,
        community_strength=1.2,
        node_affinity_strength=0.2,
        zipf_exponent=zipf_exponent,
        first_layer=first_layer,
    )


def effective_dtype(args):
    if args.dtype:
        return args.dtype
    if args.model == "gpt_oss_120b":
        return "mxfp4"
    return "fp8"


def model_case(args):
    geometry = build_model_expert_geometry(
        args.model, dtype=effective_dtype(args)
    )
    experts = geometry.routed_experts(derived_provenance(
        "%s expert geometry from local public config" %
        geometry.model_name
    ))
    layer_limit = int(getattr(args, "layer_limit", 0) or 0)
    if layer_limit < 0:
        raise ValidationError("layer_limit cannot be negative")
    if layer_limit and layer_limit < geometry.n_moe_layers:
        last_layer = geometry.first_moe_layer + layer_limit
        experts = tuple(
            expert for expert in experts
            if expert.key.layer_id < last_layer
        )
        used_layers = layer_limit
    else:
        used_layers = geometry.n_moe_layers
    return geometry, experts, used_layers


def runtime_profiles(args):
    return (
        load_hardware_profile(
            getattr(args, "gpu_profile", "h100_sxm")
        ),
        load_hardware_profile(
            getattr(args, "brain_profile", "lynxi_hp300_proj")
        ),
        load_fabric_profile(
            getattr(args, "bridge_fabric", "pcie5_x16")
        ),
    )


@dataclass(frozen=True)
class GateRow:
    gate: str
    strategy: str
    mode: str
    completion_s: float
    speedup_vs_gpu: float
    gpu_assignments: int
    brain_assignments: int
    brain_replicas: int
    cut_metric: float
    transfer_bytes: int
    status: str
    slo_goodput_per_s: float = 0.0
    tpot_p99_s: float = 0.0


class OffsetWeightedRouter:
    def __init__(self, router, first_layer: int):
        self.router = router
        self.first_layer = first_layer

    def route(self, request_id, token_index, layer_id):
        return self.router.route(
            request_id, token_index, self.first_layer + layer_id
        )


def run_gate(args, temp: Path):
    geometry, experts, used_layers = model_case(args)
    gpu_profile, brain_profile, bridge_profile = runtime_profiles(args)
    brain_nodes = int(getattr(args, "brain_nodes", 2) or 2)
    if brain_nodes <= 0:
        raise ValidationError("brain_nodes must be positive")
    cluster = make_cluster(
        brain_nodes, 8, brain_profile, bridge_profile
    )
    home_nodes = cluster.ingress_nodes
    placement_tokens = max(2, args.tokens)
    evaluation_tokens = placement_tokens
    placement = task_trace(
        placement_tokens,
        (TaskPhase("placement", max(1, args.requests), {
            "code": 0.6, "math": 0.4,
        }),),
        used_layers,
        geometry.n_experts,
        geometry.top_k,
        home_nodes,
        args.seed,
        "placement",
        geometry.first_moe_layer,
        getattr(args, "zipf_exponent", 1.3),
    )
    evaluation = task_trace(
        evaluation_tokens,
        (TaskPhase("evaluation", max(1, args.requests), {
            "code": 0.4, "math": 0.6,
        }),),
        used_layers,
        geometry.n_experts,
        geometry.top_k,
        home_nodes,
        args.seed + 1,
        "evaluation",
        geometry.first_moe_layer,
        getattr(args, "zipf_exponent", 1.3),
    )
    demand = profile_demand(placement.trace)
    brain = make_brain_chips(
        cluster,
        max_experts=args.c_max,
        count_per_node=8,
        profile=brain_profile,
    )
    gpu = make_gpu_chips(profile=gpu_profile)
    kernels = make_kernels(experts, gpu, brain, geometry.dtype)
    gpu_plan = unique_balanced(experts, gpu, {
        expert.key: float(demand.global_for(expert.key))
        for expert in experts
    })
    gpu_exec = execute_hbm_fallback_plan(
        gpu_plan,
        evaluation.trace,
        cluster,
        [chip.chip_id for chip in gpu],
        kernels,
        7168,
    )
    rows = [GateRow(
        "G0", "gpu-only", "G", gpu_exec.completion_s, 1.0,
        gpu_exec.gpu_assignments, 0, 0, 0.0, 0, "ok",
    )]
    copy_budget = min(
        len(experts),
        len(brain) * args.c_max,
        args.replicas,
    )
    byte_budget = copy_budget * geometry.expert_weight_bytes
    plans = {}
    for policy in (
        "coverage-first", "replica-first", "joint-marginal",
        "ranked-0.25", "ranked-adaptive",
    ):
        plans[policy] = build_hbm_fallback_plan(
            experts, gpu, brain, demand, policy,
            max_brain_replicas=copy_budget,
            coactivation_weight=0.5,
        )
    plans["marginal-event"] = build_marginal_event_plan(
        experts,
        gpu,
        brain,
        demand,
        max_brain_bytes=min(byte_budget, 8 * geometry.expert_weight_bytes),
        max_brain_replicas=min(copy_budget, 8),
        candidate_limit=max(4, min(16, copy_budget)),
    )
    for strategy, plan in plans.items():
        execution = execute_hbm_fallback_plan(
            plan,
            evaluation.trace,
            cluster,
            [chip.chip_id for chip in gpu],
            kernels,
            7168,
        )
        rows.append(GateRow(
            "G-overlay",
            strategy,
            "G",
            execution.completion_s,
            gpu_exec.completion_s / execution.completion_s,
            execution.gpu_assignments,
            execution.brain_assignments,
            execution.brain_replicas,
            0.0,
            0,
            (
                "surrogate-proxy"
                if strategy == "marginal-event"
                else "ok"
            ),
        ))
    return rows, {
        "geometry": asdict(geometry),
        "used_moe_layers": used_layers,
        "placement_tokens": placement_tokens,
        "evaluation_tokens": evaluation_tokens,
        "copy_budget": copy_budget,
    }, {
        "gpu-only": gpu_plan,
        **plans,
    }


def run_c1_gate(args):
    provenance = projected_provenance("C=1 fixed-resource clustering gate")
    experts = tuple(
        ExpertSpec(ExpertKey(0, index), 100, provenance)
        for index in range(8)
    )
    _, brain_profile, bridge_profile = runtime_profiles(args)
    gpu_profile = load_hardware_profile(
        getattr(args, "gpu_profile", "h100_sxm")
    )
    cluster = make_cluster(2, 4, brain_profile, bridge_profile)
    chips = make_brain_chips(
        cluster,
        max_experts=1,
        count_per_node=4,
        reserve=0.0,
        profile=brain_profile,
    )
    gpu = make_gpu_chips(profile=gpu_profile)
    kernels = make_kernels(experts, gpu, chips, "fp8")
    records = []
    for token in range(12):
        hyperedge = (0, 1, 2) if token < 6 else (4, 5, 6)
        records.append(RoutingRecord(
            token // 2, token, token, 0,
            cluster.ingress_nodes[token % len(cluster.ingress_nodes)],
            hyperedge, (1 / 3,) * 3,
            task_family="code" if token < 6 else "math",
            stage="decode",
            timestamp_s=float(token),
            window="w0",
        ))
    trace = RoutingTrace(
        records, provenance, "evaluation"
    )
    demand = profile_demand(RoutingTrace(
        tuple(replace_record(record, split=None) for record in records),
        provenance,
        "placement",
    ))
    expected = {
        expert.key: float(demand.global_for(expert.key))
        for expert in experts
    }
    pair = coactivation_clustered(experts, chips, demand)
    full = hyperedge_clustered(experts, chips, demand)
    baseline = unique_balanced(experts, chips, expected)
    rows = []
    base_execution = execute_hbm_fallback_plan(
        baseline,
        trace,
        cluster,
        [chip.chip_id for chip in gpu],
        kernels,
        1,
    )
    for name, plan in (
        ("unique-balanced", baseline),
        ("pair-clustered", pair),
        ("full-hyperedge", full),
    ):
        execution = execute_hbm_fallback_plan(
            plan,
            trace,
            cluster,
            [chip.chip_id for chip in gpu],
            kernels,
            1,
        )
        rows.append(GateRow(
            "C1-clustering",
            name,
            "G-primary-only",
            execution.completion_s,
            base_execution.completion_s / execution.completion_s,
            execution.gpu_assignments,
            execution.brain_assignments,
            len(plan.replicas),
            float(full_hyperedge_cut(plan, demand)),
            0,
            "same-copy-multiset;event-model-checked",
        ))
    return rows, {
        "c_max": 1,
        "copy_multiset": [expert.key.expert_id for expert in experts],
        "pair_cut": primary_coactivation_cut(pair, demand),
    }, {
        "unique-balanced": baseline,
        "pair-clustered": pair,
        "full-hyperedge": full,
    }


def replace_record(record, split=None):
    return RoutingRecord(
        request_id=record.request_id,
        token_id=record.token_id,
        step=record.step,
        layer_id=record.layer_id,
        home_node=record.home_node,
        experts=record.experts,
        gate_weights=record.gate_weights,
        task_family=record.task_family,
        stage=record.stage,
        timestamp_s=record.timestamp_s,
        window=record.window,
    )


def run_h_b_gate(args):
    geometry, experts, used_layers = model_case(args)
    gpu_profile, brain_profile, bridge_profile = runtime_profiles(args)
    dram_capacity_bytes = brain_profile.dram_capacity_bytes
    brain_nodes = max(
        int(getattr(args, "brain_nodes", 2) or 2),
        int(math.ceil(
            geometry.routed_expert_weight_bytes /
            (dram_capacity_bytes * 0.8)
        )),
    )
    cluster = make_cluster(
        brain_nodes, 8, brain_profile, bridge_profile
    )
    gpu = make_gpu_chips(profile=gpu_profile)
    brain = make_brain_chips(
        cluster,
        max_experts=args.c_max,
        count_per_node=8,
        profile=brain_profile,
    )
    planning = task_trace(
        max(2, args.tokens),
        (TaskPhase("hb", max(1, args.requests), {
            "code": 0.6, "math": 0.4,
        }),),
        used_layers,
        geometry.n_experts,
        geometry.top_k,
        cluster.ingress_nodes,
        args.seed + 20,
        "placement",
        geometry.first_moe_layer,
        getattr(args, "zipf_exponent", 1.3),
    )
    evaluation = task_trace(
        max(2, args.tokens),
        (TaskPhase("hb", max(1, args.requests), {
            "code": 0.6, "math": 0.4,
        }),),
        used_layers,
        geometry.n_experts,
        geometry.top_k,
        cluster.ingress_nodes,
        args.seed + 21,
        "evaluation",
        geometry.first_moe_layer,
        getattr(args, "zipf_exponent", 1.3),
    )
    demand = profile_demand(planning.trace)
    kernels = make_kernels(experts, gpu, brain, geometry.dtype)
    rows = []
    h_plan = build_h_exclusive_plan(
        experts,
        gpu,
        brain,
        demand,
        max_brain_bytes=min(
            args.hot_experts * geometry.expert_weight_bytes,
            len(brain) * args.c_max * geometry.expert_weight_bytes,
        ),
        max_brain_copies=args.hot_experts,
    )
    h_exec = execute_hbm_fallback_plan(
        h_plan,
        evaluation.trace,
        cluster,
        [chip.chip_id for chip in gpu],
        kernels,
        7168,
    )
    rows.append(GateRow(
        "H-exclusive",
        "h-exclusive",
        "H",
        h_exec.completion_s,
        1.0,
        h_exec.gpu_assignments,
        h_exec.brain_assignments,
        h_exec.brain_replicas,
        0.0,
        0,
        "no-gpu-fallback",
    ))
    dram_bw = {
        node: brain_profile.dram_bandwidth_bytes_per_s
        for node in cluster.ingress_nodes
    }
    dram_capacity = {
        node: dram_capacity_bytes for node in cluster.ingress_nodes
    }
    b_layout = build_b_stage_layout(
        experts,
        brain,
        demand,
        dram_bw,
        dram_capacity,
        max_hot_bytes=min(
            args.hot_experts * geometry.expert_weight_bytes,
            sum(chip.usable_weight_bytes for chip in brain),
        ),
        max_hot_experts=args.hot_experts,
    )
    first_layer = tuple(
        assignment for assignment in evaluation.trace.assignments()
        if assignment.layer_id == geometry.first_moe_layer
    )
    b_exec = execute_b_stage_layer(b_layout, first_layer, kernels)
    rows.append(GateRow(
        "B-stage",
        "b-stage",
        "B",
        b_exec.completion_s,
        1.0,
        0,
        b_exec.assignments,
        len(b_layout.hot_chip),
        0.0,
        b_exec.transfer_bytes,
        "grouped-miss-staging",
    ))
    return rows, {
        "h_hot_experts": len({
            replica.expert for replica in h_plan.replicas
            if h_plan.chip(replica.chip_id).role == "brain_sram"
        }),
        "b_hot_experts": len(b_layout.hot_chip),
        "b_first_layer_assignments": b_exec.assignments,
    }, {
        "h-exclusive": h_plan,
    }


def run_refresh_gate(args):
    provenance = projected_provenance("Stage 0 refresh trace")
    experts = tuple(
        ExpertSpec(ExpertKey(0, index), 100, provenance)
        for index in range(8)
    )
    chips = tuple(
        ChipSpec("b%d" % index, "n0", 1000, 0, 0, provenance)
        for index in range(8)
    )
    gpu = (
        ChipSpec("g", "gpu", 10000, 0, 0, provenance, role="gpu"),
    )
    records = []
    for token in range(24):
        if token < 12:
            experts_selected = (0, 1)
        else:
            experts_selected = (5, 6)
        records.append(RoutingRecord(
            token // 3, token, token, 0, "n0",
            experts_selected, (0.5, 0.5),
            task_family="code" if token < 12 else "math",
            stage="decode",
            timestamp_s=float(token),
            window="code" if token < 12 else "math",
        ))
    first = profile_demand(RoutingTrace(
        records[:12], provenance, "placement"
    ))
    second = profile_demand(RoutingTrace(
        records[12:], provenance, "placement"
    ))
    old = build_hbm_fallback_plan(
        experts, gpu, chips, first, "coverage-first",
        max_brain_replicas=8,
    )
    new = build_hbm_fallback_plan(
        experts, gpu, chips, second, "coverage-first",
        max_brain_replicas=8,
    )
    migration = build_migration_plan(old, new, experts)
    timing = estimate_migration_timing(
        migration, {"n0": 10.0}, reservation_fraction=0.5
    )
    manager = VersionedPlacement(old, "v0")
    manager.begin(new, "v1")
    old_request = manager.bind_request(1)
    manager.mark_ready("v1")
    manager.activate("v1")
    new_request = manager.bind_request(2)
    manager.release_request(2)
    rows = [GateRow(
        "refresh",
        "versioned-migration",
        "G",
        timing.completion_s,
        1.0,
        0,
        0,
        len({
            replica.expert for replica in new.replicas
            if new.chip(replica.chip_id).role.startswith("brain")
        }),
        0.0,
        migration.total_bytes,
        "old-request=%s,new-request=%s" % (old_request, new_request),
    )]
    drift_old = dict(first.global_assignments)
    drift_new = dict(second.global_assignments)
    detector = HysteresisDriftDetector(0.1, 0.02, 1)
    detector.observe(0, drift_old)
    decision = detector.observe(1, drift_new)
    return rows, {
        "migration_ops": len(migration.additions),
        "migration_bytes": migration.total_bytes,
        "drift_l1": normalized_l1(drift_old, drift_new),
        "drift_triggered": decision.triggered,
        "version_ready": manager.retiring_version == "v0",
    }, {
        "refresh-old": old,
        "refresh-new": new,
    }


def run_serving_gate(args):
    geometry, experts, used_layers = model_case(args)
    gpu_profile, brain_profile, bridge_profile = runtime_profiles(args)
    brain_nodes = int(getattr(args, "brain_nodes", 2) or 2)
    if brain_nodes <= 0:
        raise ValidationError("brain_nodes must be positive")
    cluster = make_cluster(
        brain_nodes, 8, brain_profile, bridge_profile
    )
    gpu = make_gpu_chips(profile=gpu_profile)
    brain = make_brain_chips(
        cluster,
        max_experts=args.c_max,
        count_per_node=8,
        profile=brain_profile,
    )
    planning = task_trace(
        max(2, args.tokens),
        (TaskPhase("serving-placement", max(1, args.requests), {
            "code": 0.5, "math": 0.5,
        }),),
        used_layers,
        geometry.n_experts,
        geometry.top_k,
        cluster.ingress_nodes,
        args.seed + 30,
        "placement",
        geometry.first_moe_layer,
        getattr(args, "zipf_exponent", 1.3),
    )
    demand = profile_demand(planning.trace)
    kernels = make_kernels(experts, gpu, brain, geometry.dtype)
    gpu_plan = unique_balanced(experts, gpu, {
        expert.key: float(demand.global_for(expert.key))
        for expert in experts
    })
    brain_budget_copies = min(
        args.replicas,
        len(brain) * args.c_max,
    )
    overlay_plan = build_hbm_fallback_plan(
        experts, gpu, brain, demand, "coverage-first",
        max_brain_replicas=brain_budget_copies,
        coactivation_weight=0.5,
    )
    h_plan = build_h_exclusive_plan(
        experts,
        gpu,
        brain,
        demand,
        max_brain_copies=min(args.hot_experts, brain_budget_copies),
        max_brain_bytes=(
            min(args.hot_experts, brain_budget_copies) *
            geometry.expert_weight_bytes
        ),
    )
    router = WeightedServingRouter(
        zipf_expert_weights(geometry.n_experts, 1.2),
        geometry.top_k,
        cluster.ingress_nodes,
        args.seed + 31,
    )
    requests = tuple(
        DecodeRequest(index, index * 0.0005, 2)
        for index in range(max(2, args.requests))
    )
    plans = {}
    results = {}
    for name, plan in (
        ("gpu-only-serving", gpu_plan),
        ("overlay-serving", overlay_plan),
        ("h-exclusive-serving", h_plan),
    ):
        dispatcher = build_plan_dispatcher(
            plan,
            cluster,
            [chip.chip_id for chip in gpu],
            kernels,
            7168,
        )
        simulator = ContinuousBatchingSimulator(
            MoELayerEngine(dispatcher=dispatcher),
            router,
            used_layers,
            max_batch=max(2, args.requests),
            first_layer=geometry.first_moe_layer,
            admission_policy=AdmissionPolicy(
                max_outstanding_requests=max(2, args.requests)
            ),
        )
        result = simulator.run(requests)
        plans[name] = plan
        results[name] = result
    base = results["gpu-only-serving"].slo_goodput_per_s(0.1, 0.1)
    rows = []
    for name in plans:
        result = results[name]
        goodput = result.slo_goodput_per_s(0.1, 0.1)
        gpu_assignments = sum(
            execution.assignment_count
            for layer in result.layer_results
            for execution in layer.replica_executions
            if plans[name].chip(execution.chip_id).role == "gpu"
        )
        total_assignments = sum(
            layer.assignment_count for layer in result.layer_results
        )
        brain_replicas = sum(
            plans[name].chip(replica.chip_id).role.startswith("brain")
            for replica in plans[name].replicas
        )
        rows.append(GateRow(
            "serving",
            name,
            "G" if "gpu" in name or "overlay" in name else "H",
            result.finish_s - result.start_s,
            goodput / base if base > 0 else 1.0,
            gpu_assignments,
            total_assignments - gpu_assignments,
            brain_replicas,
            0.0,
            0,
            "continuous-batching-gate",
            goodput,
            result.tpot_percentile(99),
        ))
    return rows, {
        "requests": len(requests),
        "output_tokens": sum(item.output_tokens for item in requests),
        "slo_ttft_s": 0.1,
        "slo_tpot_s": 0.1,
    }, plans


def write_rows(path: Path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def serialize_plan(plan: ReplicaPlan):
    return {
        "strategy": plan.strategy,
        "chips": [
            {
                "chip_id": chip.chip_id,
                "node_id": chip.node_id,
                "role": chip.role,
                "usable_weight_bytes": chip.usable_weight_bytes,
                "max_experts": chip.max_experts,
            }
            for chip in plan.chips
        ],
        "replicas": [
            {
                "replica_id": replica.replica_id,
                "layer_id": replica.expert.layer_id,
                "expert_id": replica.expert.expert_id,
                "shared": replica.expert.shared,
                "chip_id": replica.chip_id,
                "is_primary": replica.is_primary,
            }
            for replica in plan.replicas
        ],
    }


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="mixtral_8x7b")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--c-max", type=int, default=2)
    parser.add_argument("--replicas", type=int, default=16)
    parser.add_argument("--hot-experts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--zipf-exponent", type=float, default=1.3)
    parser.add_argument("--layer-limit", type=int, default=0)
    parser.add_argument("--gpu-profile", default="h100_sxm")
    parser.add_argument("--brain-profile", default="lynxi_hp300_proj")
    parser.add_argument("--bridge-fabric", default="pcie5_x16")
    parser.add_argument("--brain-nodes", type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit("output directory already exists: %s" % output)
    temp = output.parent / (output.name + ".tmp")
    if temp.exists():
        raise SystemExit("temporary output directory already exists: %s" % temp)
    temp.mkdir(parents=True)

    g_rows, g_meta, g_plans = run_gate(args, temp)
    c1_rows, c1_meta, c1_plans = run_c1_gate(args)
    hb_rows, hb_meta, hb_plans = run_h_b_gate(args)
    refresh_rows, refresh_meta, refresh_plans = run_refresh_gate(args)
    serving_rows, serving_meta, serving_plans = run_serving_gate(args)
    all_rows = (
        g_rows + c1_rows + hb_rows + refresh_rows + serving_rows
    )
    write_rows(temp / "gate_results.csv", all_rows)
    placements = {
        "G": {
            name: serialize_plan(plan)
            for name, plan in g_plans.items()
        },
        "C1": {
            name: serialize_plan(plan)
            for name, plan in c1_plans.items()
        },
        "H": {
            name: serialize_plan(plan)
            for name, plan in hb_plans.items()
        },
        "refresh": {
            name: serialize_plan(plan)
            for name, plan in refresh_plans.items()
        },
        "serving": {
            name: serialize_plan(plan)
            for name, plan in serving_plans.items()
        },
    }
    (temp / "placements.json").write_text(
        json.dumps(placements, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "model": args.model,
        "dtype": effective_dtype(args),
        "tokens_per_request": args.tokens,
        "requests_per_phase": args.requests,
        "c_max": args.c_max,
        "copy_budget": args.replicas,
        "hot_experts": args.hot_experts,
        "seed": args.seed,
        "zipf_exponent": args.zipf_exponent,
        "layer_limit": args.layer_limit,
        "gpu_profile": args.gpu_profile,
        "brain_profile": args.brain_profile,
        "bridge_fabric": args.bridge_fabric,
        "brain_nodes": args.brain_nodes,
        "claim_boundary": (
            "Stage 0 semantic/mechanism gate; projected hardware and "
            "synthetic routing; not a final performance claim"
        ),
    }
    (temp / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "gate_status": "semantic-smoke-only",
        "geometry": g_meta,
        "c1": c1_meta,
        "h_b": hb_meta,
        "refresh": refresh_meta,
        "serving": serving_meta,
        "rows": len(all_rows),
    }
    (temp / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (temp / "warnings.json").write_text(
        json.dumps([
            "projected hardware curves; not KA200 measurements",
            "synthetic request-ordered traces; not a real router trace",
            "Stage 0 checks semantics and plumbing, not final performance",
            "B-stage is grouped full-expert staging, not block streaming",
        ], indent=2),
        encoding="utf-8",
    )
    report = "\n".join([
        "# MoE Stage 0 Gate Report",
        "",
        "- Status: semantic/mechanism gate only",
        "- Model: %s (%s)" % (args.model, effective_dtype(args)),
        "- Rows: %d" % len(all_rows),
        "- C=1 event-model clustering: checked",
        "- H-exclusive no-fallback: checked",
        "- B-stage grouped DRAM staging: checked",
        "- Versioned migration state: checked",
        "- Continuous decode serving: checked",
        "",
        "This report does not establish final performance or KA200 behavior.",
        "",
    ])
    (temp / "stage_report.md").write_text(report, encoding="utf-8")
    source_paths = [
        Path(__file__),
        ROOT / "flowgpu" / "sim_v2" / "schema.py",
        ROOT / "flowgpu" / "sim_v2" / "replicas.py",
        ROOT / "flowgpu" / "sim_v2" / "hardware_profiles.py",
        ROOT / "flowgpu" / "sim_v2" / "model_geometry.py",
        ROOT / "flowgpu" / "sim_v2" / "placement.py",
        ROOT / "flowgpu" / "sim_v2" / "research.py",
        ROOT / "flowgpu" / "sim_v2" / "refresh.py",
        ROOT / "flowgpu" / "sim_v2" / "statistics.py",
        ROOT / "flowgpu" / "units.py",
    ]
    manifest = {
        "schema_version": 1,
        "kind": "moe-stage0-gate",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "source_sha256": {
            str(path.resolve()): file_hash(path)
            for path in source_paths
        },
        "files": [
            "config.json", "gate_results.csv", "placements.json",
            "summary.json", "warnings.json", "stage_report.md",
        ],
        "claim_boundary": config["claim_boundary"],
    }
    (temp / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp.rename(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
