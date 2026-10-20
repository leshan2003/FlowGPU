from collections import Counter

import pytest

from flowgpu.sim_v2.eplb import (
    allocate_eplb_replicas,
    build_eplb_plan,
    placement_load_balance,
    rebalance_brain_replicas,
)
from flowgpu.sim_v2.replicas import ReplicaPlan
from flowgpu.sim_v2.routing import RoutingTrace
from flowgpu.sim_v2.schema import (
    ChipSpec,
    Confidence,
    ExpertKey,
    ExpertSpec,
    Provenance,
    ReplicaSpec,
    RoutingRecord,
    ValidationError,
)


def _fixture():
    provenance = Provenance(Confidence.PROJECTED, "EPLB test")
    experts = tuple(
        ExpertSpec(ExpertKey(0, expert), 10, provenance)
        for expert in range(2)
    )
    chips = (
        ChipSpec(
            "node0:chip0", "node0", 100, 0.0, 1, provenance,
            role="brain_sram", max_experts=1,
        ),
        ChipSpec(
            "node0:chip1", "node0", 100, 0.0, 1, provenance,
            role="brain_sram", max_experts=1,
        ),
        ChipSpec(
            "node1:chip0", "node1", 100, 0.0, 1, provenance,
            role="brain_sram", max_experts=1,
        ),
    )
    plan = ReplicaPlan(
        experts,
        chips,
        [
            ReplicaSpec("e0", experts[0].key, chips[0].chip_id, True),
            ReplicaSpec("e1", experts[1].key, chips[1].chip_id, True),
        ],
        "skewed",
    )
    records = tuple(
        RoutingRecord(
            request_id=step,
            token_id=step,
            step=step,
            layer_id=0,
            home_node="node0",
            experts=(0, 1),
            gate_weights=(0.5, 0.5),
            task_family="test",
            stage="decode",
            timestamp_s=float(step),
            window="w0",
        )
        for step in range(4)
    )
    return plan, RoutingTrace(records, provenance, "placement"), chips


def test_eplb_rebalances_without_changing_copy_multiset():
    plan, trace, chips = _fixture()
    brain_ids = [chip.chip_id for chip in chips]
    before = placement_load_balance(plan, trace, brain_ids)
    balanced = rebalance_brain_replicas(plan, trace, brain_ids)
    after = placement_load_balance(balanced, trace, brain_ids)

    assert Counter(r.expert for r in balanced.replicas) == Counter(
        r.expert for r in plan.replicas
    )
    locations = {(r.expert, r.chip_id) for r in balanced.replicas}
    assert len(locations) == 2
    assert any(
        chip_id == "node1:chip0"
        for _, chip_id in locations
    )
    assert before.node_lb == 2.0
    assert after.node_lb == 1.0
    assert after.chip_lb <= before.chip_lb


def test_eplb_graduated_replication_uses_fixed_budget():
    plan, trace, chips = _fixture()
    counts = allocate_eplb_replicas(
        trace,
        (expert.key for expert in plan.experts),
        total_copies=4,
        redundancy_fraction=0.5,
        max_copies_per_expert=len(chips),
    )

    assert sum(counts.values()) == 4
    assert max(counts.values()) >= 2
    assert len(counts) == 2


def test_eplb_plan_preserves_budget_and_gpu_primaries():
    plan, trace, chips = _fixture()
    gpu = ChipSpec(
        "gpu:0", "gpu-node", 1000, 0.0, 1,
        plan.experts[0].provenance,
        role="gpu",
    )
    eplb = build_eplb_plan(
        plan.experts,
        (gpu,),
        chips,
        trace,
        total_brain_copies=3,
        redundancy_fraction=0.5,
    )

    assert sum(
        eplb.chip(replica.chip_id).role.startswith("brain")
        for replica in eplb.replicas
    ) == 3
    assert {
        replica.expert for replica in eplb.replicas
        if replica.is_primary
    } == {expert.key for expert in plan.experts}
    assert all(
        replica.chip_id == gpu.chip_id
        for replica in eplb.replicas
        if replica.is_primary
    )


def test_eplb_swaps_when_all_chips_are_full():
    provenance = Provenance(Confidence.PROJECTED, "EPLB swap test")
    experts = tuple(
        ExpertSpec(ExpertKey(0, expert), 10, provenance)
        for expert in range(6)
    )
    chips = tuple(
        ChipSpec(
            "node0:chip%d" % index,
            "node0",
            100,
            0.0,
            1,
            provenance,
            role="brain_sram",
            max_experts=2,
        )
        for index in range(3)
    )
    placements = (0, 0, 1, 1, 2, 2)
    plan = ReplicaPlan(
        experts,
        chips,
        [
            ReplicaSpec(
                "e%d" % index,
                expert.key,
                chips[placements[index]].chip_id,
                True,
            )
            for index, expert in enumerate(experts)
        ],
        "full-chips",
    )
    loads = (6, 2, 4, 2, 3, 1)
    records = []
    token_id = 0
    for expert, count in enumerate(loads):
        for _ in range(count):
            records.append(RoutingRecord(
                request_id=token_id,
                token_id=token_id,
                step=0,
                layer_id=0,
                home_node="node0",
                experts=(expert,),
                gate_weights=(1.0,),
                task_family="test",
                stage="decode",
                timestamp_s=0.0,
                window="w0",
            ))
            token_id += 1
    trace = RoutingTrace(
        tuple(records), provenance, "placement"
    )
    brain_ids = [chip.chip_id for chip in chips]
    before = placement_load_balance(plan, trace, brain_ids)
    balanced = rebalance_brain_replicas(
        plan, trace, brain_ids
    )
    after = placement_load_balance(balanced, trace, brain_ids)

    assert Counter(r.expert for r in balanced.replicas) == Counter(
        r.expert for r in plan.replicas
    )
    assert after.chip_lb < before.chip_lb


def test_eplb_rejects_evaluation_trace_for_planning():
    plan, trace, chips = _fixture()
    evaluation_trace = RoutingTrace(
        trace.records,
        trace.provenance,
        "evaluation",
    )

    with pytest.raises(ValidationError, match="evaluation trace"):
        allocate_eplb_replicas(
            evaluation_trace,
            (expert.key for expert in plan.experts),
            total_copies=2,
        )

    with pytest.raises(ValidationError, match="evaluation trace"):
        rebalance_brain_replicas(
            plan,
            evaluation_trace,
            [chip.chip_id for chip in chips],
        )
