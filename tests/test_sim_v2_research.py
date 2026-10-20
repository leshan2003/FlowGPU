from flowgpu.sim_v2.demand import profile_demand
from flowgpu.sim_v2.detailed_network import _next_positive_float
from flowgpu.sim_v2.research import (
    adaptive_replica_share,
    build_b_stage_layout,
    build_hbm_fallback_plan,
    build_h_exclusive_plan,
    build_marginal_event_plan,
    execute_b_stage_layer,
    generate_ordered_task_trace,
    generate_task_community_trace,
    hybrid_replica_plan,
    migration_bytes,
    plan_replica_locations,
    TaskPhase,
)
from flowgpu.sim_v2.routing import RoutingTrace
from flowgpu.sim_v2.schema import (
    Assignment,
    ChipSpec,
    Confidence,
    ExpertKey,
    ExpertSpec,
    Provenance,
    RoutingRecord,
)
from flowgpu.sim_v2.kernel_model import ExpertKernelCurve, KernelSample


def _prov():
    return Provenance(Confidence.PROJECTED, "research test")


def test_adaptive_ranked_share_tracks_demand_concentration():
    flat = {ExpertKey(0, index): 10.0 for index in range(40)}
    skewed = {
        ExpertKey(0, index): (1000.0 if index < 2 else 1.0)
        for index in range(40)
    }
    assert adaptive_replica_share(flat) == 0.0
    assert adaptive_replica_share(skewed) == 0.50


def test_network_next_float_fallback_works_on_python_38():
    assert _next_positive_float(0.0) > 0.0
    value = 1.25
    assert _next_positive_float(value) > value


def _experts():
    return tuple(
        ExpertSpec(ExpertKey(0, expert), 10, _prov())
        for expert in range(4)
    )


def _chips():
    return (
        ChipSpec("g0", "gpu", 1000, 0, 0, _prov()),
        ChipSpec("node0:chip0", "node0", 20, 0, 0, _prov()),
        ChipSpec("node0:chip1", "node0", 20, 0, 0, _prov()),
        ChipSpec("node1:chip0", "node1", 20, 0, 0, _prov()),
    )


def _skew_trace():
    records = []
    for index in range(20):
        experts = (
            (0,) if index < 18 else
            (1,) if index == 18 else (2,)
        )
        records.append(RoutingRecord(
            request_id=index,
            token_id=index,
            step=index,
            layer_id=0,
            home_node="node0",
            experts=experts,
            gate_weights=(1.0,),
        ))
    return RoutingTrace(records, _prov(), "placement")


def test_joint_marginal_can_replicate_below_full_coverage():
    experts = _experts()
    gpu = (_chips()[0],)
    brain = _chips()[1:]
    demand = profile_demand(_skew_trace())
    coverage = build_hbm_fallback_plan(
        experts, gpu, brain, demand, "coverage-first",
        max_brain_replicas=2,
    )
    joint = build_hbm_fallback_plan(
        experts, gpu, brain, demand, "joint-marginal",
        max_brain_replicas=2,
    )

    def unique(plan):
        return len({
            replica.expert for replica in plan.replicas
            if not replica.is_primary
        })

    assert len(plan_replica_locations(coverage)) == 2
    assert len(plan_replica_locations(joint)) == 2
    assert unique(coverage) == 2
    assert unique(joint) == 1


def test_ranked_replication_allocates_graduated_copy_counts():
    experts = _experts()
    gpu = (_chips()[0],)
    brain = _chips()[1:]
    demand = profile_demand(_skew_trace())
    plan = build_hbm_fallback_plan(
        experts, gpu, brain, demand, "ranked-0.50",
        max_brain_replicas=4,
    )
    copies = {}
    for replica in plan.replicas:
        if not replica.is_primary:
            copies[replica.expert] = copies.get(replica.expert, 0) + 1
    assert len(copies) == 2
    assert sorted(copies.values()) == [1, 3]
    assert plan.strategy == "ranked-0.50"


def test_task_community_trace_is_reproducible_and_conserves_assignments():
    left = generate_task_community_trace(
        20, 3, 16, 4, ("node0", "node1"),
        {"code": 0.5, "math": 0.5}, 7, _prov(), "placement",
        community_count=4, community_strength=1.0,
    )
    right = generate_task_community_trace(
        20, 3, 16, 4, ("node0", "node1"),
        {"code": 0.5, "math": 0.5}, 7, _prov(), "placement",
        community_count=4, community_strength=1.0,
    )
    assert len(left.trace.records) == 60
    assert sum(
        len(record.experts) for record in left.trace.records
    ) == 240
    assert left.task_by_token == right.task_by_token
    assert [
        (record.experts, record.home_node)
        for record in left.trace.records
    ] == [
        (record.experts, record.home_node)
        for record in right.trace.records
    ]


def test_local_hybrid_plan_is_subset_and_only_counts_new_copies():
    experts = _experts()
    gpu = (_chips()[0],)
    brain = _chips()[1:]
    demand = profile_demand(_skew_trace())
    first = build_hbm_fallback_plan(
        experts, gpu, brain, demand, "coverage-first",
        max_brain_replicas=1,
    )
    candidate = build_hbm_fallback_plan(
        experts, gpu, brain, demand, "coverage-first",
        max_brain_replicas=3,
    )
    local = hybrid_replica_plan(
        first, candidate, demand, max_added_copies=1
    )
    assert len(plan_replica_locations(local)) <= 2
    assert migration_bytes(first, local, 10) in (0, 10)


def test_marginal_event_plan_places_extra_copies_below_full_coverage():
    experts = tuple(
        ExpertSpec(ExpertKey(0, expert), 10, _prov())
        for expert in range(3)
    )
    gpu = (ChipSpec("g", "gpu", 100, 0, 0, _prov(), role="gpu"),)
    brain = (
        ChipSpec("n0:c0", "n0", 30, 0, 0, _prov(), max_experts=2),
        ChipSpec("n0:c1", "n0", 30, 0, 0, _prov(), max_experts=2),
    )
    records = [
        RoutingRecord(
            index, index, index, 0, "n0",
            (0,), (1.0,),
        )
        for index in range(20)
    ]
    demand = profile_demand(RoutingTrace(
        records, _prov(), "placement"
    ))
    plan = build_marginal_event_plan(
        experts, gpu, brain, demand,
        max_brain_bytes=20,
        max_brain_replicas=2,
    )
    counts = {}
    for replica in plan.replicas:
        if not replica.is_primary:
            counts[replica.expert] = counts.get(replica.expert, 0) + 1
    assert counts == {experts[0].key: 2}
    assert sum(
        expert.weight_bytes
        for expert in experts
        for _ in range(counts.get(expert.key, 0))
    ) == 20


def test_hbm_fallback_plan_obeys_explicit_chip_expert_limit():
    experts = tuple(
        ExpertSpec(ExpertKey(0, expert), 10, _prov())
        for expert in range(4)
    )
    gpu = (ChipSpec("g", "gpu", 100, 0, 0, _prov(), role="gpu"),)
    brain = (
        ChipSpec("n0:c0", "n0", 100, 0, 0, _prov(), max_experts=1),
        ChipSpec("n0:c1", "n0", 100, 0, 0, _prov(), max_experts=1),
    )
    records = [
        RoutingRecord(
            index, index, index, 0, "n0",
            (index % len(experts),), (1.0,),
        )
        for index in range(8)
    ]
    demand = profile_demand(RoutingTrace(
        records, _prov(), "placement"
    ))
    plan = build_hbm_fallback_plan(
        experts, gpu, brain, demand, "coverage-first",
        max_brain_replicas=4,
    )
    assert len([
        replica for replica in plan.replicas if not replica.is_primary
    ]) == 2
    assert all(
        usage.max_experts is None or
        usage.resident_experts <= usage.max_experts
        for usage in plan.capacity_usage.values()
    )


def test_marginal_event_plan_respects_heterogeneous_bytes_and_c():
    experts = (
        ExpertSpec(ExpertKey(0, 0), 7, _prov()),
        ExpertSpec(ExpertKey(0, 1), 13, _prov()),
        ExpertSpec(ExpertKey(0, 2), 19, _prov()),
    )
    gpu = (ChipSpec("g", "gpu", 100, 0, 0, _prov(), role="gpu"),)
    brain = (
        ChipSpec("n0:c0", "n0", 30, 0, 0, _prov(), max_experts=2),
        ChipSpec("n1:c0", "n1", 30, 0, 0, _prov(), max_experts=1),
    )
    records = []
    for index, expert in enumerate((2, 2, 2, 1, 0)):
        records.append(RoutingRecord(
            index, index, index, 0, "n0",
            (expert,), (1.0,),
        ))
    demand = profile_demand(RoutingTrace(
        records, _prov(), "placement"
    ))
    plan = build_marginal_event_plan(
        experts, gpu, brain, demand,
        max_brain_bytes=20,
        max_brain_replicas=3,
    )
    brain_bytes = sum(
        next(expert.weight_bytes for expert in experts
             if expert.key == replica.expert)
        for replica in plan.replicas if not replica.is_primary
    )
    assert brain_bytes <= 20
    assert all(
        usage.max_experts is None or
        usage.resident_experts <= usage.max_experts
        for usage in plan.capacity_usage.values()
    )


def test_marginal_event_uses_explicit_event_objective():
    experts = (
        ExpertSpec(ExpertKey(0, 0), 10, _prov()),
        ExpertSpec(ExpertKey(0, 1), 10, _prov()),
    )
    gpu = (ChipSpec("g", "gpu", 100, 0, 0, _prov(), role="gpu"),)
    brain = (
        ChipSpec("n0:c0", "n0", 10, 0, 0, _prov(), max_experts=1),
    )
    records = [
        RoutingRecord(
            index, index, index, 0, "n0",
            (0 if index < 8 else 1,), (1.0,),
        )
        for index in range(10)
    ]
    demand = profile_demand(RoutingTrace(
        records, _prov(), "placement"
    ))

    def prefer_sparse_expert(plan):
        return 0.0 if any(
            not replica.is_primary and
            replica.expert == experts[1].key
            for replica in plan.replicas
        ) else 1.0

    plan = build_marginal_event_plan(
        experts, gpu, brain, demand,
        max_brain_bytes=10,
        max_brain_replicas=1,
        evaluate_plan=prefer_sparse_expert,
    )
    assert any(
        not replica.is_primary and replica.expert == experts[1].key
        for replica in plan.replicas
    )


def test_h_exclusive_has_no_gpu_fallback_for_brain_primary():
    experts = tuple(
        ExpertSpec(ExpertKey(0, expert), 10, _prov())
        for expert in range(4)
    )
    gpu = (
        ChipSpec("g0", "gpu", 30, 0, 0, _prov(), role="gpu"),
        ChipSpec("g1", "gpu", 30, 0, 0, _prov(), role="gpu"),
    )
    brain = (
        ChipSpec("n0:c0", "n0", 20, 0, 0, _prov(), max_experts=2),
    )
    records = [
        RoutingRecord(
            index, index, index, 0, "n0",
            (0,), (1.0,),
        )
        for index in range(5)
    ]
    demand = profile_demand(RoutingTrace(
        records, _prov(), "placement"
    ))
    plan = build_h_exclusive_plan(
        experts, gpu, brain, demand,
        max_brain_bytes=10,
        max_brain_copies=1,
    )
    hot = experts[0].key
    brain_locations = {
        replica.expert for replica in plan.replicas
        if plan.chip(replica.chip_id).role == "brain_sram"
    }
    gpu_locations = {
        replica.expert for replica in plan.replicas
        if plan.chip(replica.chip_id).role == "gpu"
    }
    assert brain_locations == {hot}
    assert hot not in gpu_locations
    assert len(plan.replicas) == len(experts)


def test_b_stage_coalesces_miss_and_charges_dram_transfer():
    experts = (
        ExpertSpec(ExpertKey(0, 0), 10, _prov()),
        ExpertSpec(ExpertKey(0, 1), 10, _prov()),
    )
    chips = (
        ChipSpec("n0:c0", "n0", 20, 0, 0, _prov(), max_experts=2),
    )
    records = [
        RoutingRecord(0, index, index, 0, "n0", (0,), (1.0,))
        for index in range(3)
    ] + [
        RoutingRecord(0, index, index + 3, 0, "n0", (1,), (1.0,))
        for index in range(2)
    ]
    demand = profile_demand(RoutingTrace(
        records, _prov(), "placement"
    ))
    layout = build_b_stage_layout(
        experts,
        chips,
        demand,
        dram_bandwidth_bytes_per_s={"n0": 10.0},
        dram_capacity_bytes_per_node={"n0": 100},
        dram_latency_s=0.0,
        max_hot_bytes=10,
        max_hot_experts=1,
    )
    assert layout.hot_chip[experts[0].key] == "n0:c0"
    assert experts[1].key in layout.staging_chip
    kernel = ExpertKernelCurve(
        "test", "fp8", 1, "sram",
        [KernelSample(1, 0.1), KernelSample(8, 0.2)],
        _prov(),
    )
    assignments = tuple(
        record.assignments()[0] for record in records
    )
    result = execute_b_stage_layer(
        layout,
        assignments,
        {
            (expert.key, "n0:c0"): kernel
            for expert in experts
        },
    )
    assert result.assignments == 5
    assert result.transfer_bytes == 10
    assert result.transfer_time_s == 1.0
    assert result.hot_assignment_fraction == 3 / 5


def test_ordered_task_trace_preserves_request_family_and_window_order():
    phases = (
        TaskPhase("code-phase", 3, {"code": 1.0, "math": 0.0}),
        TaskPhase("math-phase", 2, {"code": 0.0, "math": 1.0}),
    )
    trace = generate_ordered_task_trace(
        tokens_per_request=4,
        n_layers=2,
        n_experts=16,
        top_k=4,
        home_nodes=("n0", "n1"),
        phases=phases,
        seed=9,
        provenance=_prov(),
        split="evaluation",
        community_count=4,
    )
    records = trace.trace.records
    assert len(records) == 5 * 4 * 2
    assert {record.request_id for record in records} == set(range(5))
    by_request = {}
    for record in records:
        by_request.setdefault(record.request_id, set()).add(
            record.task_family
        )
    assert all(len(families) == 1 for families in by_request.values())
    assert by_request[0] == {"code"}
    assert by_request[4] == {"math"}
    assert [record.timestamp_s for record in records] == sorted(
        record.timestamp_s for record in records
    )
    assert all(record.stage == "decode" for record in records)
    assert all(record.window in {
        "code-phase", "math-phase"
    } for record in records)
    offset = generate_ordered_task_trace(
        tokens_per_request=1,
        n_layers=2,
        n_experts=16,
        top_k=4,
        home_nodes=("n0",),
        phases=(TaskPhase("p", 1, {"code": 1.0}),),
        seed=1,
        provenance=_prov(),
        split="placement",
        first_layer=3,
    )
    assert {record.layer_id for record in offset.trace.records} == {3, 4}


def test_research_execution_counts_unique_tokens_across_layers():
    experts = tuple(
        ExpertSpec(ExpertKey(layer, 0), 10, _prov())
        for layer in range(3)
    )
    gpu = (ChipSpec("g", "gpu", 100, 0, 0, _prov(), role="gpu"),)
    records = [
        RoutingRecord(
            token, token, token, layer, "node0", (0,), (1.0,),
        )
        for token in range(4)
        for layer in range(3)
    ]
    trace = RoutingTrace(records, _prov(), "evaluation")
    from flowgpu.sim_v2.research import execute_hbm_fallback_plan
    from flowgpu.sim_v2.placement import unique_balanced
    from flowgpu.sim_v2.topology import (
        LinkProfile, build_hierarchical_cluster,
    )
    p = _prov()
    link = LinkProfile(10e9, 0.0, 1.0, 0.0, p)
    cluster = build_hierarchical_cluster(
        1, 1, (0,), link, link, link,
    )
    kernel = ExpertKernelCurve(
        "test", "fp8", 1, "sram",
        [KernelSample(1, 0.1)],
        p,
    )
    plan = unique_balanced(experts, gpu)
    result = execute_hbm_fallback_plan(
        plan,
        trace,
        cluster,
        ["g"],
        {
            (expert.key, "g"): kernel
            for expert in experts
        },
        1,
    )
    assert result.evaluation_tokens == 4
