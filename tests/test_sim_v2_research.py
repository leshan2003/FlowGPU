from flowgpu.sim_v2.demand import profile_demand
from flowgpu.sim_v2.detailed_network import _next_positive_float
from flowgpu.sim_v2.research import (
    adaptive_replica_share,
    build_hbm_fallback_plan,
    generate_task_community_trace,
    hybrid_replica_plan,
    migration_bytes,
    plan_replica_locations,
)
from flowgpu.sim_v2.routing import RoutingTrace
from flowgpu.sim_v2.schema import (
    ChipSpec,
    Confidence,
    ExpertKey,
    ExpertSpec,
    Provenance,
    RoutingRecord,
)


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
