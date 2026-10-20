import pytest

from flowgpu.sim_v2.refresh import (
    VersionedPlacement,
    build_migration_plan,
    estimate_migration_timing,
)
from flowgpu.sim_v2.schema import (
    ChipSpec,
    Confidence,
    ExpertKey,
    ExpertSpec,
    Provenance,
    ReplicaSpec,
    ValidationError,
)
from flowgpu.sim_v2.replicas import ReplicaPlan


def _prov():
    return Provenance(Confidence.PROJECTED, "refresh test")


def _plan(locations, experts, chips):
    replicas = []
    primary_seen = set()
    for expert_key, chip_id in locations:
        primary = expert_key not in primary_seen
        primary_seen.add(expert_key)
        replicas.append(ReplicaSpec(
            "%s-%s-%d" % (expert_key, chip_id, len(replicas)),
            expert_key,
            chip_id,
            primary,
        ))
    return ReplicaPlan(experts, chips, replicas, "test")


def test_migration_prefers_brain_source_and_is_source_aware():
    expert = ExpertSpec(ExpertKey(0, 0), 10, _prov())
    chips = (
        ChipSpec("g", "gpu", 100, 0, 0, _prov(), role="gpu"),
        ChipSpec("b0", "n0", 100, 0, 0, _prov()),
        ChipSpec("b1", "n1", 100, 0, 0, _prov()),
    )
    old = _plan(
        [(expert.key, "g"), (expert.key, "b0")],
        [expert], chips,
    )
    new = _plan(
        [(expert.key, "g"), (expert.key, "b1")],
        [expert], chips,
    )
    plan = build_migration_plan(old, new, [expert])
    assert len(plan.additions) == 1
    assert plan.additions[0].source_chip_id == "b0"
    assert plan.additions[0].target_node_id == "n1"
    assert plan.total_bytes == 10
    timing = estimate_migration_timing(
        plan, {"n1": 10.0}, reservation_fraction=0.5
    )
    assert timing.completion_s == pytest.approx(2.0)


def test_version_keeps_old_plan_until_ready_and_references_drain():
    expert0 = ExpertSpec(ExpertKey(0, 0), 10, _prov())
    expert1 = ExpertSpec(ExpertKey(0, 1), 10, _prov())
    chips = (
        ChipSpec("b0", "n0", 100, 0, 0, _prov()),
        ChipSpec("b1", "n1", 100, 0, 0, _prov()),
    )
    old = _plan(
        [(expert0.key, "b0"), (expert1.key, "b1")],
        [expert0, expert1], chips,
    )
    new = _plan(
        [(expert0.key, "b1"), (expert1.key, "b0")],
        [expert0, expert1], chips,
    )
    manager = VersionedPlacement(old, "v0")
    manager.begin(new, "v1")
    with pytest.raises(ValidationError, match="not fully ready"):
        manager.activate("v1")
    manager.mark_ready("v1")
    old_request = manager.bind_request(10)
    manager.activate("v1")
    assert manager.active_version == "v1"
    assert manager.retiring_version == "v0"
    assert old_request == "v0"
    new_request = manager.bind_request(11)
    assert new_request == "v1"
    manager.release_request(10)
    assert manager.retiring_version is None
    assert manager.plan("v1") is not None
    manager.release_request(11)
    assert manager.retiring_version is None
    with pytest.raises(ValidationError, match="unknown placement version"):
        manager.plan("v0")
