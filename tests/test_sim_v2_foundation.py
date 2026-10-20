"""Tier-1 invariants for the independent Simulator v2 foundation."""

from __future__ import annotations

import os
import sys
import json
from dataclasses import replace

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowgpu.sim_v2 import (AdmissionPolicy, AnalyticExpertKernel,
                            Assignment, ChipSpec,
                            ChipPowerProfile,
                            Confidence, ContinuousBatchingSimulator,
                            DecodeRequest, Dispatcher, Endpoint, EventQueue,
                            ExpertKernelCurve, ExpertKey, ExpertSpec, Flow,
                            FastMoELayerEngine, FluidNetwork,
                            KernelSample, LinkProfile, LinkSpec, Provenance,
                            MoELayerEngine,
                            NodeSweepSpec,
                            ReplicaPlan, ReplicaSpec, RoutingRecord,
                            ResourceBlockWindow,
                            RoutingSummary, RuntimeCapabilities, RoutingTrace,
                            SerializedPathScheduler, Topology,
                            SweepWorkloadSpec,
                            UnreachableError, ValidationError,
                            build_hierarchical_cluster, build_multicast_tree,
                            build_case_trace_pair,
                            build_kernel_calibration,
                            build_network_calibration,
                            build_node_count_sweep,
                            account_system_energy,
                            audit_oracle_gap,
                            capacity_ratio,
                            coactivation_clustered,
                            full_hyperedge_cut,
                            hyperedge_clustered,
                            compare_fast_detailed_direction,
                            classify_capacity_regime,
                            aggregate_serving_results,
                            evaluate_slo_capacity,
                            estimate_unloaded, generate_uniform_trace,
                            generate_weighted_trace,
                            generate_poisson_requests,
                            ingress_local, joint_greedy, layer_packed,
                            load_benchmark_artifact,
                            load_decode_request_trace,
                            load_routing_trace_jsonl,
                            make_gpu_tier_chip,
                            node_load_replicated,
                            popularity_replicated, primary_coactivation_cut,
                            profile_demand, simulate_flows,
                            run_fixed_window_comparison,
                            run_node_count_comparisons,
                            run_node_count_uncertainty,
                            run_node_serving_sweep,
                            run_tiered_fixed_window_comparison,
                            run_tiered_node_count_comparisons,
                            simulate_multicast, UniformServingRouter,
                            small_replica_oracle, unique_balanced,
                            tiered_expert_plan,
                            write_comparison_result_bundle,
                            write_layer_result_bundle,
                            write_node_sweep_bundle)
from flowgpu.sim_v2 import (
    write_serving_sweep_bundle, write_tiered_sweep_bundle,
    write_uncertainty_bundle,
)
from flowgpu.sim_v2 import zipf_expert_weights
from flowgpu.sim_v2.config import (
    run_node_serving_config, run_node_sweep_config,
    run_node_tiered_config,
    run_node_uncertainty_config,
)


def _derived(source="test fixture"):
    return Provenance(Confidence.DERIVED, source)


def _expert(layer, expert, nbytes=40):
    return ExpertSpec(
        ExpertKey(layer, expert), nbytes, _derived("test expert")
    )


def _chip(chip_id, node_id="n0", capacity=100, reserve=0.1):
    return ChipSpec(
        chip_id, node_id, capacity,
        runtime_reserve_fraction=reserve,
        provenance=_derived("test chip"),
    )


def _link(link_id, src, dst, latency, bandwidth=100.0, energy=1e-12):
    return LinkSpec(
        link_id=link_id,
        src=src,
        dst=dst,
        bandwidth_bytes_per_s=bandwidth,
        latency_s=latency,
        efficiency=1.0,
        energy_per_byte=energy,
        duplex="full",
        provenance=_derived("test link"),
    )


def _small_topology():
    endpoints = [
        Endpoint("a", "gpu"),
        Endpoint("b", "node_switch", node_id="n0"),
        Endpoint("d", "brain_chip", node_id="n0", chip_id="c0"),
        Endpoint("isolated", "brain_chip", node_id="n1", chip_id="c1"),
    ]
    links = [
        _link("direct", "a", "d", latency=0.010),
        _link("via-1", "a", "b", latency=0.001),
        _link("via-2", "b", "d", latency=0.001),
    ]
    return Topology(endpoints, links)


def test_provenance_order_and_measured_artifact_requirement():
    assert Confidence.MEASURED.rank > Confidence.VENDOR.rank
    assert Confidence.VENDOR.rank > Confidence.PAPER.rank
    assert Confidence.PAPER.rank > Confidence.DERIVED.rank
    assert Confidence.DERIVED.rank > Confidence.PROJECTED.rank
    with pytest.raises(ValidationError, match="raw artifact"):
        Provenance(Confidence.MEASURED, "bench run")
    measured = Provenance(
        "measured", "bench run", artifact="bench/raw.csv"
    )
    assert measured.confidence is Confidence.MEASURED


def test_unique_balanced_places_one_primary_and_respects_capacity():
    experts = [_expert(0, i) for i in range(4)]
    chips = [_chip("c0"), _chip("c1")]
    demand = {
        experts[0].key: 10.0,
        experts[1].key: 9.0,
        experts[2].key: 2.0,
        experts[3].key: 1.0,
    }
    plan = unique_balanced(experts, chips, demand)
    assert len(plan.replicas) == len(experts)
    assert all(replica.is_primary for replica in plan.replicas)
    assert all(
        usage.weight_bytes <= usage.usable_weight_bytes
        for usage in plan.capacity_usage.values()
    )
    assert set(plan.capacity_usage) == {"c0", "c1"}


def test_replica_plan_enforces_explicit_chip_expert_limit():
    experts = [_expert(0, 0), _expert(0, 1)]
    chip = ChipSpec(
        "c0", "n0", 1000,
        runtime_reserve_fraction=0.0,
        provenance=_derived("test chip"),
        role="brain_sram",
        max_experts=1,
    )
    with pytest.raises(ValidationError, match="expert-count exceeded"):
        ReplicaPlan(
            experts, [chip],
            [
                ReplicaSpec("r0", experts[0].key, "c0", True),
                ReplicaSpec("r1", experts[1].key, "c0", True),
            ],
            "bad",
        )

    usage = ReplicaPlan(
        experts[:1], [chip],
        [ReplicaSpec("r0", experts[0].key, "c0", True)],
        "valid",
    ).capacity_usage["c0"]
    assert usage.resident_experts == 1
    assert usage.max_experts == 1


def test_replica_plan_rejects_missing_primary_duplicate_and_overflow():
    expert = _expert(0, 0, nbytes=60)
    chip = _chip("c0", capacity=100, reserve=0.1)
    with pytest.raises(ValidationError, match="exactly one primary"):
        ReplicaPlan(
            [expert], [chip],
            [ReplicaSpec("r0", expert.key, "c0", False)],
            "bad",
        )
    with pytest.raises(ValidationError, match="duplicated on chip"):
        ReplicaPlan(
            [expert], [chip],
            [
                ReplicaSpec("r0", expert.key, "c0", True),
                ReplicaSpec("r1", expert.key, "c0", False),
            ],
            "bad",
        )

    experts = [_expert(0, 0, 50), _expert(0, 1, 50)]
    with pytest.raises(ValidationError, match="capacity exceeded"):
        ReplicaPlan(
            experts, [chip],
            [
                ReplicaSpec("r0", experts[0].key, "c0", True),
                ReplicaSpec("r1", experts[1].key, "c0", True),
            ],
            "bad",
        )


def test_assignment_must_bind_to_existing_matching_replica():
    experts = [_expert(2, 3), _expert(2, 4)]
    chips = [_chip("c0"), _chip("c1")]
    plan = unique_balanced(experts, chips)
    assignment = Assignment(
        "a0", 0, 0, 0, 2, 3, 0.5, "n0"
    )
    right = plan.replicas_for(ExpertKey(2, 3))[0]
    assert plan.validate_binding(assignment, right.replica_id) == right
    wrong = plan.replicas_for(ExpertKey(2, 4))[0]
    with pytest.raises(ValidationError, match="cannot bind"):
        plan.validate_binding(assignment, wrong.replica_id)
    with pytest.raises(ValidationError, match="unknown replica"):
        plan.validate_binding(assignment, "does-not-exist")


def test_routing_trace_conserves_top_k_and_is_seed_deterministic():
    trace_a = generate_uniform_trace(
        n_tokens=7,
        n_layers=3,
        n_experts=16,
        top_k=4,
        home_nodes=("n0", "n1"),
        seed=19,
        provenance=_derived("synthetic"),
        split="evaluation",
    )
    trace_b = generate_uniform_trace(
        n_tokens=7,
        n_layers=3,
        n_experts=16,
        top_k=4,
        home_nodes=("n0", "n1"),
        seed=19,
        provenance=_derived("synthetic"),
        split="evaluation",
    )
    assert trace_a.records == trace_b.records
    assert len(trace_a.assignments()) == 7 * 3 * 4
    assert all(
        len(set(record.experts)) == record.top_k
        for record in trace_a.records
    )
    assert trace_a.summary().assignments == 84
    assert trace_a.summary().home_nodes == 2


def test_routing_trace_rejects_duplicate_expert_and_duplicate_token_layer():
    with pytest.raises(ValidationError, match="unique"):
        RoutingRecord(0, 0, 0, 0, "n0", (1, 1), (0.5, 0.5))
    record = RoutingRecord(0, 0, 0, 0, "n0", (1, 2), (0.5, 0.5))
    with pytest.raises(ValidationError, match="duplicate token-layer"):
        RoutingTrace(
            [record, record], _derived("synthetic"), "evaluation"
        )


def test_topology_routes_by_latency_or_hops_and_detects_unreachable():
    topology = _small_topology()
    low_latency = topology.route("a", "d", metric="latency")
    few_hops = topology.route("a", "d", metric="hops")
    assert tuple(link.link_id for link in low_latency.links) == (
        "via-1", "via-2"
    )
    assert tuple(link.link_id for link in few_hops.links) == ("direct",)
    with pytest.raises(UnreachableError):
        topology.route("a", "isolated")


def test_unloaded_flow_accounts_path_bytes_time_and_energy():
    topology = _small_topology()
    flow = Flow("f0", "a", "d", 100, traffic_class="dispatch")
    result = estimate_unloaded(topology, flow, metric="latency")
    assert result.route.hop_count == 2
    assert result.serialization_s == pytest.approx(1.0)
    assert result.propagation_s == pytest.approx(0.002)
    assert result.energy_j == pytest.approx(100 * 2e-12)


def test_serialized_scheduler_queues_conflicting_flows_and_counts_each_hop():
    topology = _small_topology()
    scheduler = SerializedPathScheduler(topology)
    first = scheduler.schedule(Flow("f0", "a", "d", 100))
    second = scheduler.schedule(Flow("f1", "a", "d", 100))
    assert second.start_s == pytest.approx(first.finish_s)
    assert second.queue_s == pytest.approx(first.finish_s)
    assert scheduler.link_bytes["via-1"] == 200
    assert scheduler.link_bytes["via-2"] == 200
    assert scheduler.link_bytes["direct"] == 0
    assert not scheduler.is_detailed


def test_event_queue_has_stable_contract_order_and_monotonic_time():
    queue = EventQueue()
    queue.schedule(1.0, "late-priority", priority=2, token_id=0)
    queue.schedule(1.0, "token-1", priority=1, token_id=1)
    queue.schedule(1.0, "token-0-first", priority=1, token_id=0)
    queue.schedule(1.0, "token-0-second", priority=1, token_id=0)
    assert [queue.pop().kind for _ in range(4)] == [
        "token-0-first",
        "token-0-second",
        "token-1",
        "late-priority",
    ]
    with pytest.raises(ValidationError, match="current time"):
        queue.schedule(0.5, "past")


def test_kernel_curve_interpolates_and_flags_extrapolation():
    curve = ExpertKernelCurve(
        expert_shape="ffn-4096-14336",
        dtype="bf16",
        concurrent_experts=1,
        residency="sram",
        samples=[
            KernelSample(1, 10e-6, p95_s=12e-6, repetitions=100),
            KernelSample(4, 16e-6, p95_s=19e-6, repetitions=100),
            KernelSample(8, 22e-6, p95_s=26e-6, repetitions=100),
        ],
        provenance=Provenance(
            Confidence.MEASURED,
            "kernel microbenchmark",
            artifact="bench/kernel.csv",
        ),
    )
    assert curve.predict(4).latency_s == 16e-6
    assert curve.predict(6).latency_s == pytest.approx(19e-6)
    assert not curve.predict(6).extrapolated
    assert curve.predict(16).extrapolated
    assert "outside calibrated range" in curve.predict(16).warning
    # Grouping eight tokens is not silently treated as eight serial M=1 calls.
    assert curve.predict(8).latency_s < 8 * curve.predict(1).latency_s


def test_analytic_kernel_is_explicitly_warned_and_not_linear_in_m():
    model = AnalyticExpertKernel(
        flops_per_token=1e9,
        weight_bytes=10_000_000,
        peak_flops_per_s=1e14,
        sram_bandwidth_bytes_per_s=1e12,
        max_efficiency=0.8,
        saturation_m=8,
        launch_overhead_s=2e-6,
        provenance=_derived("analytic fallback"),
    )
    one = model.predict(1)
    eight = model.predict(8)
    assert one.warning.startswith("analytic")
    assert one.extrapolated and eight.extrapolated
    assert eight.latency_s < 8 * one.latency_s


def _profiles():
    provenance = _derived("topology fixture")
    return (
        LinkProfile(200.0, 0.001, 1.0, 1e-12, provenance),
        LinkProfile(100.0, 0.010, 1.0, 2e-12, provenance),
        LinkProfile(50.0, 0.020, 1.0, 3e-12, provenance),
    )


def test_hierarchical_builder_expands_nodes_chips_and_ingress_explicitly():
    scale_up, scale_out, bridge = _profiles()
    cluster = build_hierarchical_cluster(
        n_nodes=3,
        chips_per_node=2,
        ingress_node_indices=(0,),
        scale_up=scale_up,
        scale_out=scale_out,
        bridge=bridge,
    )
    assert len(cluster.chip_endpoints) == 6
    assert cluster.ingress_nodes == ("node0",)
    assert set(cluster.home_endpoints) == {"node0", "node1", "node2"}
    remote = cluster.topology.route("gpu0", "node2:chip1")
    assert any(
        link.link_id.startswith("bridge:node0")
        for link in remote.links
    )
    assert any(
        link.link_id.startswith("scaleout:node2")
        for link in remote.links
    )


def _replicated_dispatch_fixture(policy, ready=None):
    scale_up, scale_out, bridge = _profiles()
    cluster = build_hierarchical_cluster(
        n_nodes=2,
        chips_per_node=1,
        ingress_node_indices=(0,),
        scale_up=scale_up,
        scale_out=scale_out,
        bridge=bridge,
    )
    key = ExpertKey(0, 0)
    expert = ExpertSpec(key, 40, _derived("expert"))
    chips = [
        _chip("node0:chip0", node_id="node0"),
        _chip("node1:chip0", node_id="node1"),
    ]
    replicas = [
        ReplicaSpec("local", key, "node0:chip0", True),
        ReplicaSpec("remote", key, "node1:chip0", False),
    ]
    replica_plan = ReplicaPlan(experts=[expert], chips=chips,
                               replicas=replicas, strategy="fixture")
    kernel = ExpertKernelCurve(
        "tiny", "bf16", 1, "sram",
        [KernelSample(1, 0.1), KernelSample(2, 0.15)],
        _derived("kernel fixture"),
    )
    dispatcher = Dispatcher(
        replica_plan=replica_plan,
        topology=cluster.topology,
        home_endpoints=cluster.home_endpoints,
        chip_endpoints=cluster.chip_endpoints,
        kernel_models={key: kernel},
        hidden_bytes=10,
        policy=policy,
    )
    assignments = [
        Assignment("a0", 0, 0, 0, 0, 0, 1.0, "node0"),
        Assignment("a1", 0, 1, 0, 0, 0, 1.0, "node0"),
    ]
    return dispatcher.plan(assignments, ready or {})


def test_dispatcher_policies_distinguish_locality_load_and_finish_time():
    nearest = _replicated_dispatch_fixture("nearest")
    assert [binding.replica.replica_id for binding in nearest.bindings] == [
        "local", "local"
    ]
    least = _replicated_dispatch_fixture("least_loaded")
    assert {binding.replica.replica_id for binding in least.bindings} == {
        "local", "remote"
    }
    earliest = _replicated_dispatch_fixture(
        "earliest_finish", {"local": 10.0}
    )
    assert all(
        binding.replica.replica_id == "remote"
        for binding in earliest.bindings
    )
    assert earliest.assignment_count == 2
    assert sum(len(batch.assignments) for batch in earliest.batches) == 2


def test_dispatcher_does_not_silently_drop_gpu_ingress_bridge():
    scale_up, scale_out, bridge = _profiles()
    cluster = build_hierarchical_cluster(
        1, 1, (0,), scale_up, scale_out, bridge
    )
    key = ExpertKey(0, 0)
    expert = ExpertSpec(key, 40, _derived("expert"))
    chip = _chip("node0:chip0", "node0")
    plan = ReplicaPlan(
        [expert], [chip],
        [ReplicaSpec("r0", key, chip.chip_id, True)],
        "fixture",
    )
    kernel = ExpertKernelCurve(
        "tiny", "bf16", 1, "sram",
        [KernelSample(1, 0.1)], _derived("kernel"),
    )
    common = dict(
        replica_plan=plan,
        topology=cluster.topology,
        home_endpoints=cluster.home_endpoints,
        chip_endpoints=cluster.chip_endpoints,
        kernel_models={key: kernel},
        hidden_bytes=10,
        policy="nearest",
    )
    assignment = Assignment("a", 0, 0, 0, 0, 0, 1.0, "node0")
    resident = Dispatcher(**common).plan([assignment]).bindings[0]
    gpu_origin = Dispatcher(
        **common, ingress_source_endpoint=cluster.gpu_endpoint
    ).plan([assignment]).bindings[0]
    # 10 B / 50 B/s + 20 ms in each direction is added by the bridge.
    assert gpu_origin.dispatch_s - resident.dispatch_s == pytest.approx(0.22)
    assert gpu_origin.combine_s - resident.combine_s == pytest.approx(0.22)


def _placement_trace(split="placement"):
    records = []
    # Both experts are globally equally popular, but node0 exclusively asks
    # for both.  This lets node-conditional placement differ from global heat.
    for token_id in range(8):
        expert_id = token_id % 2
        records.append(RoutingRecord(
            0, token_id, token_id, 0, "node0",
            (expert_id,), (1.0,),
        ))
    return RoutingTrace(records, _derived("placement window"), split)


def test_demand_profile_rejects_evaluation_future_and_conserves_counts():
    placement = profile_demand(_placement_trace("placement"))
    assert placement.assignments == 8
    assert placement.global_for(ExpertKey(0, 0)) == 4
    assert placement.node_for("node0", ExpertKey(0, 1)) == 4
    with pytest.raises(ValidationError, match="evaluation trace"):
        profile_demand(_placement_trace("evaluation"))

    oracle_records = _placement_trace("placement").records
    oracle = RoutingTrace(
        oracle_records, _derived("oracle window"), "oracle"
    )
    with pytest.raises(ValidationError, match="allow_oracle"):
        profile_demand(oracle)
    assert profile_demand(oracle, allow_oracle=True).source_split == "oracle"


def test_replication_strategies_spend_only_real_capacity():
    experts = [_expert(0, 0, 40), _expert(0, 1, 40)]
    chips = [
        _chip("node0:chip0", "node0", capacity=120, reserve=0.0),
        _chip("node1:chip0", "node1", capacity=120, reserve=0.0),
    ]
    demand = profile_demand(_placement_trace())
    assert capacity_ratio(experts, chips) == 3.0

    popularity = popularity_replicated(
        experts, chips, demand, max_total_replicas=4
    )
    assert len(popularity.replicas) == 4
    assert all(
        usage.weight_bytes <= usage.usable_weight_bytes
        for usage in popularity.capacity_usage.values()
    )

    local = ingress_local(
        experts, chips, demand, max_total_replicas=3
    )
    assert len(local.replicas) == 3
    assert all(
        len({
            replica.chip_id
            for replica in local.replicas_for(expert.key)
        }) == len(local.replicas_for(expert.key))
        for expert in experts
    )
    # unique-balanced puts E0 on node0 and E1 on node1 for this fixture;
    # ingress-local uses the spare slot to make E1 local to node0 demand.
    e1_nodes = {
        local.chip(replica.chip_id).node_id
        for replica in local.replicas_for(ExpertKey(0, 1))
    }
    assert e1_nodes == {"node0", "node1"}


def _direct_topology(injection=None):
    provenance = _derived("fluid network fixture")
    return Topology(
        [
            Endpoint(
                "s", "gpu",
                injection_bandwidth_bytes_per_s=injection,
            ),
            Endpoint("d0", "brain_chip", chip_id="c0"),
            Endpoint("d1", "brain_chip", chip_id="c1"),
        ],
        [
            LinkSpec(
                "s-d0", "s", "d0", 100.0, 0.0, 1.0, 0.0,
                "full", provenance,
            ),
            LinkSpec(
                "s-d1", "s", "d1", 100.0, 0.0, 1.0, 0.0,
                "full", provenance,
            ),
        ],
    )


def test_detailed_network_max_min_shares_one_link_and_reallocates():
    topology = _direct_topology()
    results, link_bytes = simulate_flows(
        topology,
        [
            Flow("a", "s", "d0", 100),
            Flow("b", "s", "d0", 100),
        ],
    )
    assert results["a"].finish_s == pytest.approx(2.0)
    assert results["b"].finish_s == pytest.approx(2.0)
    assert results["a"].contention_delay_s == pytest.approx(1.0)
    assert link_bytes["s-d0"] == 200

    # Flow a gets 0.5 s alone, then shares until it completes at 1.5 s;
    # b receives the released bandwidth and completes at 2.0 s.
    late, _ = simulate_flows(
        topology,
        [
            Flow("a", "s", "d0", 100),
            Flow("b", "s", "d0", 100, earliest_start_s=0.5),
        ],
    )
    assert late["a"].finish_s == pytest.approx(1.5)
    assert late["b"].finish_s == pytest.approx(2.0)


def test_detailed_network_models_injection_and_disjoint_links():
    disjoint, _ = simulate_flows(
        _direct_topology(),
        [Flow("a", "s", "d0", 100), Flow("b", "s", "d1", 100)],
    )
    assert disjoint["a"].finish_s == pytest.approx(1.0)
    assert disjoint["b"].finish_s == pytest.approx(1.0)

    injected, _ = simulate_flows(
        _direct_topology(injection=100.0),
        [Flow("a", "s", "d0", 100), Flow("b", "s", "d1", 100)],
    )
    assert injected["a"].finish_s == pytest.approx(2.0)
    assert injected["b"].finish_s == pytest.approx(2.0)


def test_detailed_network_half_duplex_directions_share_one_resource():
    provenance = _derived("half duplex fixture")
    topology = Topology(
        [
            Endpoint("a", "gpu"),
            Endpoint("b", "brain_chip", chip_id="c0"),
        ],
        [
            LinkSpec(
                "ab", "a", "b", 100.0, 0.0, 1.0, 0.0,
                "half", provenance, duplex_group="wire0",
            ),
            LinkSpec(
                "ba", "b", "a", 100.0, 0.0, 1.0, 0.0,
                "half", provenance, duplex_group="wire0",
            ),
        ],
    )
    results, _ = simulate_flows(
        topology,
        [Flow("out", "a", "b", 100), Flow("back", "b", "a", 100)],
    )
    assert results["out"].finish_s == pytest.approx(2.0)
    assert results["back"].finish_s == pytest.approx(2.0)


def test_detailed_network_max_inflight_admits_in_stable_order():
    provenance = _derived("inflight fixture")
    topology = Topology(
        [Endpoint("a", "gpu"), Endpoint("b", "brain_chip", chip_id="c0")],
        [
            LinkSpec(
                "ab", "a", "b", 100.0, 0.0, 1.0, 0.0,
                "full", provenance, max_inflight=1,
            )
        ],
    )
    results, _ = simulate_flows(
        topology,
        [Flow("a", "a", "b", 100), Flow("b", "a", "b", 100)],
    )
    assert results["a"].finish_s == pytest.approx(1.0)
    assert results["b"].first_service_s == pytest.approx(1.0)
    assert results["b"].finish_s == pytest.approx(2.0)


def test_detailed_network_sub_ulp_finish_at_nonzero_time_terminates():
    """Tiny high-bandwidth transfers must not retry at one timestamp."""

    provenance = _derived("sub-ULP network regression fixture")
    topology = Topology(
        [
            Endpoint("s", "gpu"),
            Endpoint("d", "brain_chip", chip_id="c0"),
        ],
        [
            LinkSpec(
                "fast", "s", "d", 90e9, 0.0, 1.0, 0.0,
                "full", provenance,
            ),
        ],
    )
    start = 0.012355022616991163
    network = FluidNetwork(topology)
    network.advance_to(start)
    network.add_flow(Flow("a", "s", "d", 14_000, start))
    network.add_flow(Flow("b", "s", "d", 14_000, start))
    results = network.run()

    expected = start + 28_000 / 90e9
    assert network.unfinished_count == 0
    assert results["a"].finish_s > start
    assert results["b"].finish_s > start
    assert results["a"].finish_s == pytest.approx(expected, abs=2e-18)
    assert results["b"].finish_s == pytest.approx(expected, abs=2e-18)


def _one_chip_layer_engine():
    scale_up, scale_out, bridge = _profiles()
    cluster = build_hierarchical_cluster(
        1, 1, (0,), scale_up, scale_out, bridge
    )
    key = ExpertKey(0, 0)
    expert = ExpertSpec(key, 40, _derived("layer expert"))
    chip = _chip("node0:chip0", "node0")
    replica_plan = ReplicaPlan(
        [expert], [chip],
        [ReplicaSpec("r0", key, chip.chip_id, True)],
        "fixture",
    )
    kernel = ExpertKernelCurve(
        "tiny", "bf16", 1, "sram",
        [KernelSample(1, 0.1), KernelSample(2, 0.15)],
        _derived("layer kernel"),
    )
    dispatcher = Dispatcher(
        replica_plan=replica_plan,
        topology=cluster.topology,
        home_endpoints=cluster.home_endpoints,
        chip_endpoints=cluster.chip_endpoints,
        kernel_models={key: kernel},
        hidden_bytes=10,
        policy="nearest",
    )
    return MoELayerEngine(dispatcher)


def test_moe_layer_integrates_dispatch_grouped_compute_and_combine():
    engine = _one_chip_layer_engine()
    assignments = [
        Assignment("a0", 0, 0, 0, 0, 0, 0.5, "node0"),
        Assignment("a1", 0, 1, 0, 0, 0, 0.5, "node0"),
    ]
    result = engine.run(assignments)
    assert result.assignment_count == 2
    assert len(result.replica_executions) == 1
    execution = result.replica_executions[0]
    assert execution.assignment_count == 2
    # Two 10-byte dispatches share a 200 B/s scale-up link:
    # 1 ms propagation + 100 ms transmission.  Grouped M=2 compute takes
    # 150 ms, then the two combines share the reverse link identically.
    assert execution.dispatch_arrival_s == pytest.approx(0.101)
    assert execution.compute_finish_s == pytest.approx(0.251)
    assert result.completion_s == pytest.approx(0.352)
    assert result.link_bytes["scaleup:node0:chip0:fwd"] == 20
    assert result.link_bytes["scaleup:node0:chip0:rev"] == 20
    assert result.critical_path.compute_finish_s == pytest.approx(0.251)
    assert result.local_assignment_fraction == 1.0
    assert result.network_contention_s == pytest.approx(0.2)


def test_moe_layer_rejects_capabilities_it_does_not_yet_model():
    dispatcher = _one_chip_layer_engine().dispatcher
    with pytest.raises(ValidationError, match="hardware_multicast"):
        MoELayerEngine(
            dispatcher,
            RuntimeCapabilities(hardware_multicast=True),
        )


def test_moe_layer_chip_dedup_and_reduce_save_physical_flows():
    scale_up, scale_out, bridge = _profiles()
    cluster = build_hierarchical_cluster(
        1, 1, (0,), scale_up, scale_out, bridge
    )
    keys = [ExpertKey(0, 0), ExpertKey(0, 1)]
    experts = [
        ExpertSpec(key, 40, _derived("dedup expert")) for key in keys
    ]
    chip = _chip("node0:chip0", "node0")
    replica_plan = ReplicaPlan(
        experts, [chip],
        [
            ReplicaSpec("r0", keys[0], chip.chip_id, True),
            ReplicaSpec("r1", keys[1], chip.chip_id, True),
        ],
        "fixture",
    )
    kernels = {
        key: ExpertKernelCurve(
            "tiny", "bf16", 1, "sram",
            [KernelSample(1, 0.1)], _derived("dedup kernel"),
        )
        for key in keys
    }
    dispatcher = Dispatcher(
        replica_plan=replica_plan,
        topology=cluster.topology,
        home_endpoints=cluster.home_endpoints,
        chip_endpoints=cluster.chip_endpoints,
        kernel_models=kernels,
        hidden_bytes=10,
        policy="nearest",
    )
    assignments = [
        Assignment("a0", 0, 0, 0, 0, 0, 0.5, "node0"),
        Assignment("a1", 0, 0, 0, 0, 1, 0.5, "node0"),
    ]
    plain = MoELayerEngine(dispatcher).run(assignments)
    optimized = MoELayerEngine(
        dispatcher,
        RuntimeCapabilities(
            dispatch_dedup_chip=True,
            combine_reduce_chip=True,
        ),
        chip_reduce_latency_s=0.01,
    ).run(assignments)
    assert plain.dispatch_flow_count == 2
    assert plain.combine_flow_count == 2
    assert optimized.dispatch_flow_count == 1
    assert optimized.combine_flow_count == 1
    assert optimized.logical_dispatch_bytes == 20
    assert optimized.physical_dispatch_bytes == 10
    assert optimized.logical_combine_bytes == 20
    assert optimized.physical_combine_bytes == 10
    assert optimized.completion_s < plain.completion_s
    with pytest.raises(ValidationError, match="calibrated positive"):
        MoELayerEngine(
            dispatcher,
            RuntimeCapabilities(combine_reduce_chip=True),
        )


def _two_node_two_chip_layer_fixture():
    scale_up, scale_out, bridge = _profiles()
    cluster = build_hierarchical_cluster(
        2, 2, (0,), scale_up, scale_out, bridge
    )
    keys = [ExpertKey(0, 0), ExpertKey(0, 1)]
    experts = [
        ExpertSpec(key, 40, _derived("node capability expert"))
        for key in keys
    ]
    chips = [
        _chip("node0:chip0", "node0"),
        _chip("node0:chip1", "node0"),
        _chip("node1:chip0", "node1"),
        _chip("node1:chip1", "node1"),
    ]
    replica_plan = ReplicaPlan(
        experts, chips,
        [
            ReplicaSpec("r0", keys[0], "node1:chip0", True),
            ReplicaSpec("r1", keys[1], "node1:chip1", True),
        ],
        "fixture",
    )
    kernels = {
        key: ExpertKernelCurve(
            "tiny", "bf16", 1, "sram",
            [KernelSample(1, 0.1)], _derived("node capability kernel"),
        )
        for key in keys
    }
    dispatcher = Dispatcher(
        replica_plan=replica_plan,
        topology=cluster.topology,
        home_endpoints=cluster.home_endpoints,
        chip_endpoints=cluster.chip_endpoints,
        kernel_models=kernels,
        hidden_bytes=10,
        policy="nearest",
    )
    assignments = [
        Assignment("a0", 0, 0, 0, 0, 0, 0.5, "node0"),
        Assignment("a1", 0, 0, 0, 0, 1, 0.5, "node0"),
    ]
    return dispatcher, assignments


def test_node_dispatch_dedup_reduces_scaleout_not_local_fanout():
    dispatcher, assignments = _two_node_two_chip_layer_fixture()
    plain = MoELayerEngine(dispatcher).run(assignments)
    dedup = MoELayerEngine(
        dispatcher,
        RuntimeCapabilities(dispatch_dedup_node=True),
    ).run(assignments)
    assert plain.link_bytes["scaleout:node0:fwd"] == 20
    assert dedup.link_bytes["scaleout:node0:fwd"] == 10
    assert plain.link_bytes["scaleout:node1:rev"] == 20
    assert dedup.link_bytes["scaleout:node1:rev"] == 10
    # The target node must still distribute one activation to each chip.
    assert dedup.link_bytes["scaleup:node1:chip0:fwd"] == 10
    assert dedup.link_bytes["scaleup:node1:chip1:fwd"] == 10
    assert dedup.completion_s < plain.completion_s


def test_node_combine_reduces_scaleout_after_chip_to_node_inputs():
    dispatcher, assignments = _two_node_two_chip_layer_fixture()
    plain = MoELayerEngine(dispatcher).run(assignments)
    reduced = MoELayerEngine(
        dispatcher,
        RuntimeCapabilities(combine_reduce_node=True),
        node_reduce_latency_s=0.01,
    ).run(assignments)
    assert plain.link_bytes["scaleout:node1:fwd"] == 20
    assert reduced.link_bytes["scaleout:node1:fwd"] == 10
    assert reduced.link_bytes["scaleup:node1:chip0:rev"] == 10
    assert reduced.link_bytes["scaleup:node1:chip1:rev"] == 10
    assert reduced.completion_s < plain.completion_s
    # Two chip-to-node partials plus one reduced node-to-home result.
    assert reduced.combine_flow_count == 3
    assert reduced.physical_combine_bytes == 30
    with pytest.raises(ValidationError, match="node_reduce_latency"):
        MoELayerEngine(
            dispatcher,
            RuntimeCapabilities(combine_reduce_node=True),
        )


def test_result_bundle_is_complete_hashed_and_never_overwritten(tmp_path):
    engine = _one_chip_layer_engine()
    assignments = [
        Assignment("a0", 0, 0, 0, 0, 0, 0.5, "node0"),
        Assignment("a1", 0, 1, 0, 0, 0, 0.5, "node0"),
    ]
    result = engine.run(assignments)
    artifact = tmp_path / "trace.jsonl"
    artifact.write_text('{"fixture": true}\n', encoding="utf-8")
    output = tmp_path / "run-001"
    written = write_layer_result_bundle(
        output_dir=output,
        resolved_config={"mode": "test", "seed": 7},
        replica_plan=engine.dispatcher.replica_plan,
        routing_summary=RoutingSummary(
            records=2,
            assignments=2,
            top_k=1,
            home_nodes=1,
            distinct_experts=1,
            max_expert_load=2,
        ),
        result=result,
        seed=7,
        command="pytest fixture",
        input_artifacts=[artifact],
    )
    assert written == output.resolve()
    expected = {
        "config.resolved.yaml", "manifest.json", "summary.json",
        "placement.json", "routing_summary.json", "chip_metrics.csv",
        "link_metrics.csv", "critical_paths.json", "warnings.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["execution_mode"] == "detailed:max-min-fluid"
    assert len(manifest["input_artifact_sha256"][
        str(artifact.resolve())
    ]) == 64
    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["assignments"] == 2
    assert summary["dispatch"]["logical_bytes"] == 20
    with pytest.raises(ValidationError, match="already exists"):
        write_layer_result_bundle(
            output, {}, engine.dispatcher.replica_plan,
            RoutingSummary(2, 2, 1, 1, 1, 2),
            result, 7, "do not overwrite",
        )


def test_small_oracle_exhausts_replica_locations_and_is_deterministic():
    expert = _expert(0, 0, 40)
    chips = [
        _chip("c0", "n0", capacity=100, reserve=0.0),
        _chip("c1", "n1", capacity=100, reserve=0.0),
    ]

    def objective(plan):
        nodes = {
            plan.chip(replica.chip_id).node_id
            for replica in plan.replicas_for(expert.key)
        }
        return 0.0 if nodes == {"n0", "n1"} else 1.0

    first = small_replica_oracle(
        [expert], chips, objective, max_extra_replicas=1
    )
    second = small_replica_oracle(
        [expert], chips, objective, max_extra_replicas=1
    )
    assert first.objective == 0.0
    assert len(first.plan.replicas) == 2
    assert first.evaluated_candidates == 3
    assert first.feasible_candidates == 3
    assert first.plan.replicas == second.plan.replicas


def test_small_oracle_refuses_unbounded_search():
    experts = [_expert(0, i, 1) for i in range(4)]
    chips = [_chip("c%d" % i, "n%d" % i) for i in range(4)]
    with pytest.raises(ValidationError, match="exceeds max_candidates"):
        small_replica_oracle(
            experts, chips, lambda plan: len(plan.replicas),
            max_extra_replicas=2,
            max_candidates=10,
        )


def test_oracle_gap_report_distinguishes_suboptimal_and_refused_search():
    experts = [_expert(0, 0, 40), _expert(0, 1, 40)]
    chips = [
        _chip("c0", "n0", capacity=80, reserve=0.0),
        _chip("c1", "n1", capacity=80, reserve=0.0),
    ]
    heuristic = layer_packed(experts, chips)

    def spread_objective(plan):
        locations = {
            replica.expert: replica.chip_id
            for replica in plan.replicas if replica.is_primary
        }
        return 0.0 if len(set(locations.values())) == 2 else 1.0

    report = audit_oracle_gap(
        heuristic, spread_objective, max_extra_replicas=0
    )
    assert report.oracle_available
    assert report.search_complete
    assert report.heuristic_objective == 1.0
    assert report.oracle_objective == 0.0
    assert report.absolute_gap == 1.0
    assert report.relative_gap is None
    assert not report.heuristic_is_optimal
    assert "undefined" in report.warning

    refused = audit_oracle_gap(
        heuristic, spread_objective,
        max_extra_replicas=1,
        max_candidates=1,
    )
    assert not refused.oracle_available
    assert not refused.search_complete
    assert refused.oracle_objective is None
    assert refused.heuristic_is_optimal is None
    assert "not run" in refused.warning


def test_layer_packed_keeps_each_layer_on_minimum_chips():
    experts = [
        _expert(layer, expert_id, 40)
        for layer in range(2)
        for expert_id in range(2)
    ]
    chips = [
        _chip("c0", "n0", capacity=80, reserve=0.0),
        _chip("c1", "n0", capacity=80, reserve=0.0),
    ]
    plan = layer_packed(experts, chips)
    for layer in range(2):
        layer_chips = {
            replica.chip_id for replica in plan.replicas
            if replica.expert.layer_id == layer
        }
        assert len(layer_chips) == 1
    assert {
        replica.chip_id for replica in plan.replicas
        if replica.expert.layer_id == 0
    } != {
        replica.chip_id for replica in plan.replicas
        if replica.expert.layer_id == 1
    }


def _community_demand():
    records = []
    token_id = 0
    for pair in ((0, 1), (2, 3)):
        for _ in range(4):
            records.append(RoutingRecord(
                0, token_id, token_id, 0, "n0",
                pair, (0.5, 0.5),
            ))
            token_id += 1
    trace = RoutingTrace(
        records, _derived("community placement trace"), "placement"
    )
    return profile_demand(trace)


def test_coactivation_clustered_reduces_cut_without_adding_replicas():
    experts = [_expert(0, expert_id, 40) for expert_id in range(4)]
    chips = [
        _chip("c0", "n0", capacity=80, reserve=0.0),
        _chip("c1", "n0", capacity=80, reserve=0.0),
    ]
    demand = _community_demand()
    baseline = unique_balanced(
        experts, chips,
        {expert.key: demand.global_for(expert.key) for expert in experts},
    )
    clustered = coactivation_clustered(experts, chips, demand)
    assert len(clustered.replicas) == len(baseline.replicas) == 4
    assert primary_coactivation_cut(
        clustered, demand
    ) < primary_coactivation_cut(baseline, demand)


def test_full_hyperedge_clustering_uses_fanout_and_preserves_copy_multiset():
    experts = [_expert(0, expert_id, 40) for expert_id in range(6)]
    chips = [
        ChipSpec(
            "c%d" % index,
            "n%d" % (index // 2),
            sram_capacity_bytes=40,
            runtime_reserve_fraction=0.0,
            provenance=_derived("full hyperedge chip"),
            max_experts=1,
        )
        for index in range(6)
    ]
    records = []
    for token_id, hyperedge in enumerate((
        (0, 1, 2), (0, 1, 2), (0, 1, 2),
        (3, 4, 5), (3, 4, 5), (3, 4, 5),
    )):
        records.append(RoutingRecord(
            0, token_id, token_id, 0, "n0",
            hyperedge, (1 / 3,) * 3,
        ))
    demand = profile_demand(RoutingTrace(
        records, _derived("full hyperedge trace"), "placement"
    ))
    expected = {
        expert.key: demand.global_for(expert.key) for expert in experts
    }
    baseline = unique_balanced(experts, chips, expected)
    clustered = hyperedge_clustered(
        experts, chips, demand, max_passes=8
    )
    assert len(clustered.replicas) == len(baseline.replicas) == 6
    assert sorted(
        replica.expert for replica in clustered.replicas
    ) == sorted(replica.expert for replica in baseline.replicas)
    assert full_hyperedge_cut(clustered, demand) <= full_hyperedge_cut(
        baseline, demand
    )
    assert all(
        usage.resident_experts <= usage.max_experts
        for usage in clustered.capacity_usage.values()
    )


def test_joint_greedy_uses_objective_and_stops_without_gain():
    expert = _expert(0, 0, 40)
    chips = [
        _chip("c0", "n0", capacity=100, reserve=0.0),
        _chip("c1", "n1", capacity=100, reserve=0.0),
    ]
    primary = unique_balanced([expert], chips)

    def locality_objective(plan):
        nodes = {
            plan.chip(replica.chip_id).node_id
            for replica in plan.replicas_for(expert.key)
        }
        return 0.0 if "n1" in nodes else 1.0

    improved = joint_greedy(
        primary, locality_objective, max_total_replicas=2
    )
    assert len(improved.replicas) == 2
    assert locality_objective(improved) == 0.0
    stopped = joint_greedy(
        primary, lambda plan: 1.0, max_total_replicas=2
    )
    assert len(stopped.replicas) == 1


def test_explicit_multicast_tree_sends_shared_prefix_once():
    provenance = _derived("multicast fixture")
    topology = Topology(
        [
            Endpoint("s", "gpu"),
            Endpoint("x", "fabric_switch"),
            Endpoint("d0", "brain_chip", chip_id="c0"),
            Endpoint("d1", "brain_chip", chip_id="c1"),
        ],
        [
            LinkSpec(
                "root", "s", "x", 100.0, 0.0, 1.0, 1e-12,
                "full", provenance,
            ),
            LinkSpec(
                "branch0", "x", "d0", 100.0, 0.0, 1.0, 1e-12,
                "full", provenance,
            ),
            LinkSpec(
                "branch1", "x", "d1", 100.0, 0.0, 1.0, 1e-12,
                "full", provenance,
            ),
        ],
    )
    tree = build_multicast_tree(topology, "s", ["d0", "d1"])
    assert tree.root_link_ids == ("root",)
    assert tree.children["root"] == ("branch0", "branch1")
    result = simulate_multicast(
        topology, "m0", "s", ["d0", "d1"], 100
    )
    assert result.link_bytes["root"] == 100
    assert result.link_bytes["branch0"] == 100
    assert result.link_bytes["branch1"] == 100
    assert result.completion_s == pytest.approx(2.0)
    assert result.destination_finish_s["d0"] == pytest.approx(2.0)
    assert result.destination_finish_s["d1"] == pytest.approx(2.0)
    assert result.energy_j == pytest.approx(300e-12)


def test_moe_hardware_multicast_shares_cross_node_prefix():
    scale_up, scale_out, bridge = _profiles()
    cluster = build_hierarchical_cluster(
        3, 1, (0,), scale_up, scale_out, bridge
    )
    keys = [ExpertKey(0, 0), ExpertKey(0, 1)]
    experts = [
        ExpertSpec(key, 40, _derived("multicast expert"))
        for key in keys
    ]
    chips = [
        _chip("node1:chip0", "node1"),
        _chip("node2:chip0", "node2"),
    ]
    replica_plan = ReplicaPlan(
        experts, chips,
        [
            ReplicaSpec("r0", keys[0], "node1:chip0", True),
            ReplicaSpec("r1", keys[1], "node2:chip0", True),
        ],
        "fixture",
    )
    kernels = {
        key: ExpertKernelCurve(
            "tiny", "bf16", 1, "sram",
            [KernelSample(1, 0.1)], _derived("multicast kernel"),
        )
        for key in keys
    }
    dispatcher = Dispatcher(
        replica_plan=replica_plan,
        topology=cluster.topology,
        home_endpoints=cluster.home_endpoints,
        chip_endpoints=cluster.chip_endpoints,
        kernel_models=kernels,
        hidden_bytes=10,
        policy="nearest",
    )
    assignments = [
        Assignment("a0", 0, 0, 0, 0, 0, 0.5, "node0"),
        Assignment("a1", 0, 0, 0, 0, 1, 0.5, "node0"),
    ]
    dedup = MoELayerEngine(
        dispatcher,
        RuntimeCapabilities(dispatch_dedup_node=True),
    ).run(assignments)
    multicast = MoELayerEngine(
        dispatcher,
        RuntimeCapabilities(
            dispatch_dedup_node=True,
            hardware_multicast=True,
        ),
    ).run(assignments)
    assert dedup.link_bytes["scaleout:node0:fwd"] == 20
    assert multicast.link_bytes["scaleout:node0:fwd"] == 10
    assert multicast.link_bytes["scaleout:node1:rev"] == 10
    assert multicast.link_bytes["scaleout:node2:rev"] == 10
    assert multicast.link_bytes["scaleup:node1:chip0:fwd"] == 10
    assert multicast.link_bytes["scaleup:node2:chip0:fwd"] == 10
    # Store-and-forward can trade lower bytes for extra staging latency;
    # the simulator must not assume multicast is automatically faster.
    assert multicast.completion_s > 0


def test_calibration_artifact_builds_measured_kernel_and_derived_link(tmp_path):
    artifact_path = tmp_path / "ka200_fixture.yaml"
    artifact_path.write_text(
        """
schema_version: 1
machine:
  model: KA200-fixture
  revision: rev-a
  firmware: fw-1
  runtime: rt-1
measurement:
  timestamp: "2026-10-17T10:00:00+08:00"
  repetitions: 100
  environment: lab
records:
  - kind: expert_latency
    expert_shape: ffn-4096-14336
    dtype: bf16
    m: 1
    concurrent_experts: 1
    residency: sram
    latency_s: 0.010
    p95_s: 0.012
  - kind: expert_latency
    expert_shape: ffn-4096-14336
    dtype: bf16
    m: 2
    concurrent_experts: 1
    residency: sram
    latency_s: 0.016
    p95_s: 0.019
  - kind: p2p
    src_class: node_switch
    dst_class: brain_chip
    bytes: 100
    concurrency: 1
    latency_s: 0.11
  - kind: p2p
    src_class: node_switch
    dst_class: brain_chip
    bytes: 200
    concurrency: 1
    latency_s: 0.21
""".lstrip(),
        encoding="utf-8",
    )
    artifact = load_benchmark_artifact(artifact_path)
    assert artifact.provenance.confidence is Confidence.MEASURED
    assert len(artifact.sha256) == 64

    kernels = build_kernel_calibration(artifact)
    curve = next(iter(kernels.curves.values()))
    assert curve.predict(1).latency_s == pytest.approx(0.010)
    assert curve.provenance.confidence is Confidence.MEASURED
    assert kernels.warnings

    network = build_network_calibration(artifact)
    fit = next(iter(network.fits.values()))
    assert fit.startup_latency_s == pytest.approx(0.01)
    assert fit.effective_bandwidth_bytes_per_s == pytest.approx(1000.0)
    assert fit.median_relative_error == pytest.approx(0.0)
    assert fit.provenance.confidence is Confidence.DERIVED
    profile = fit.as_link_profile(energy_per_byte=2e-12)
    assert profile.bandwidth_bytes_per_s == pytest.approx(1000.0)


def test_calibration_artifact_rejects_missing_machine_identity(tmp_path):
    artifact_path = tmp_path / "bad.yaml"
    artifact_path.write_text(
        """
schema_version: 1
machine: {model: KA200}
measurement:
  timestamp: now
  repetitions: 1
  environment: lab
records: [{kind: other}]
""".lstrip(),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="machine missing"):
        load_benchmark_artifact(artifact_path)


def test_fast_layer_uses_same_batches_but_ignores_network_contention():
    detailed_engine = _one_chip_layer_engine()
    assignments = [
        Assignment("a0", 0, 0, 0, 0, 0, 0.5, "node0"),
        Assignment("a1", 0, 1, 0, 0, 0, 0.5, "node0"),
    ]
    fast = FastMoELayerEngine(
        detailed_engine.dispatcher
    ).run(assignments)
    detailed = detailed_engine.run(assignments)
    assert fast.assignment_count == detailed.assignment_count == 2
    assert fast.replica_executions[0].assignment_count == 2
    assert detailed.replica_executions[0].assignment_count == 2
    assert fast.completion_s < detailed.completion_s
    assert any("ignores network contention" in item for item in fast.warnings)
    with pytest.raises(ValidationError, match="does not approximate"):
        FastMoELayerEngine(
            detailed_engine.dispatcher,
            RuntimeCapabilities(dispatch_dedup_chip=True),
        )


def test_fast_detailed_direction_flip_is_high_visibility_warning():
    engine = _one_chip_layer_engine()
    assignments = [
        Assignment("a0", 0, 0, 0, 0, 0, 1.0, "node0"),
    ]
    baseline_fast = FastMoELayerEngine(
        engine.dispatcher
    ).run(assignments)
    baseline_detailed = engine.run(assignments)
    # Synthetic candidate durations isolate the direction-checking contract:
    # fast predicts a 10% win while detailed observes a 10% loss.
    candidate_fast = replace(
        baseline_fast,
        completion_s=baseline_fast.completion_s * 0.9,
    )
    candidate_detailed = replace(
        baseline_detailed,
        completion_s=baseline_detailed.completion_s * 1.1,
    )
    comparison = compare_fast_detailed_direction(
        baseline_fast, candidate_fast,
        baseline_detailed, candidate_detailed,
    )
    assert not comparison.direction_consistent
    assert comparison.fast_speedup > 1
    assert comparison.detailed_speedup < 1
    assert "cannot support the claim" in comparison.warning


def _two_layer_serving_engine():
    scale_up, scale_out, bridge = _profiles()
    cluster = build_hierarchical_cluster(
        1, 1, (0,), scale_up, scale_out, bridge
    )
    keys = [ExpertKey(0, 0), ExpertKey(1, 0)]
    experts = [
        ExpertSpec(key, 40, _derived("serving expert")) for key in keys
    ]
    chip = _chip("node0:chip0", "node0")
    replica_plan = ReplicaPlan(
        experts, [chip],
        [
            ReplicaSpec("r0", keys[0], chip.chip_id, True),
            ReplicaSpec("r1", keys[1], chip.chip_id, True),
        ],
        "fixture",
    )
    kernels = {
        key: ExpertKernelCurve(
            "tiny", "bf16", 1, "sram",
            [KernelSample(1, 0.1), KernelSample(2, 0.15)],
            _derived("serving kernel"),
        )
        for key in keys
    }
    dispatcher = Dispatcher(
        replica_plan=replica_plan,
        topology=cluster.topology,
        home_endpoints=cluster.home_endpoints,
        chip_endpoints=cluster.chip_endpoints,
        kernel_models=kernels,
        hidden_bytes=10,
        policy="nearest",
    )
    return MoELayerEngine(dispatcher)


def test_decode_continuous_batching_admits_at_iteration_boundaries():
    engine = _two_layer_serving_engine()
    router = UniformServingRouter(
        n_experts=1, top_k=1, home_nodes=("node0",), seed=7
    )
    simulator = ContinuousBatchingSimulator(
        engine, router, n_layers=2, max_batch=2
    )
    result = simulator.run([
        DecodeRequest(0, ready_s=0.0, output_tokens=2),
        DecodeRequest(1, ready_s=0.05, output_tokens=1),
    ])
    assert result.iteration_batch_sizes == (1, 2)
    assert result.output_tokens == 3
    assert len(result.layer_results) == 4
    request0, request1 = result.requests
    assert request0.token_finish_s[0] == pytest.approx(0.404)
    assert request0.token_finish_s[1] == pytest.approx(1.108)
    assert request1.token_finish_s[0] == pytest.approx(1.108)
    assert request0.ttft_s == pytest.approx(0.404)
    assert request0.mean_tpot_s == pytest.approx(0.704)
    assert request1.ttft_s == pytest.approx(1.058)
    assert result.output_throughput == pytest.approx(3 / 1.108)
    assert result.ttft_percentile(99) >= result.ttft_percentile(50)
    assert result.slo_attainment(2.0, 1.0) == 1.0


def test_serving_router_is_reproducible_without_future_state():
    router_a = UniformServingRouter(8, 2, ("n0", "n1"), seed=11)
    router_b = UniformServingRouter(8, 2, ("n0", "n1"), seed=11)
    choices_a = [
        router_a.route(request, token, layer)
        for request in range(3)
        for token in range(2)
        for layer in range(2)
    ]
    choices_b = [
        router_b.route(request, token, layer)
        for request in range(3)
        for token in range(2)
        for layer in range(2)
    ]
    assert choices_a == choices_b
    assert all(
        len(set(choice.experts)) == 2 for choice in choices_a
    )


def test_poisson_and_trace_driven_arrivals_are_reproducible(tmp_path):
    first = generate_poisson_requests(
        8, arrival_rate_per_s=2.0,
        output_tokens=(2, 5), seed=23,
    )
    second = generate_poisson_requests(
        8, arrival_rate_per_s=2.0,
        output_tokens=(2, 5), seed=23,
    )
    assert first == second
    assert all(
        left.ready_s < right.ready_s
        for left, right in zip(first, first[1:])
    )
    assert all(2 <= request.output_tokens <= 5 for request in first)

    trace_path = tmp_path / "arrivals.jsonl"
    trace_path.write_text(
        '{"request_id": 1, "ready_s": 0.2, "output_tokens": 3}\n'
        '{"request_id": 0, "ready_s": 0.1, "output_tokens": 2}\n',
        encoding="utf-8",
    )
    loaded = load_decode_request_trace(trace_path)
    assert [request.request_id for request in loaded] == [0, 1]


def test_serving_multi_seed_aggregate_reports_confidence_intervals():
    engine = _two_layer_serving_engine()
    router = UniformServingRouter(1, 1, ("node0",), seed=3)
    simulator = ContinuousBatchingSimulator(
        engine, router, n_layers=2, max_batch=2
    )
    run_a = simulator.run([
        DecodeRequest(0, 0.0, 2),
        DecodeRequest(1, 0.1, 2),
    ])
    run_b = simulator.run([
        DecodeRequest(0, 0.0, 2),
        DecodeRequest(1, 0.2, 2),
    ])
    aggregate = aggregate_serving_results(
        [run_a, run_b], slo_ttft_s=2.0, slo_tpot_s=1.0
    )
    assert aggregate.throughput.samples == 2
    assert aggregate.throughput.method == "student-t-95"
    assert (
        aggregate.throughput.low95 <=
        aggregate.throughput.mean <=
        aggregate.throughput.high95
    )
    assert aggregate.energy_per_token_j.mean > 0
    assert aggregate.system_energy_per_token_j is None
    assert aggregate.network_energy_per_token_j.mean > 0
    assert 0 <= aggregate.slo_attainment.mean <= 1
    assert 0 <= aggregate.slo_attainment.low95
    assert aggregate.slo_attainment.high95 <= 1


def test_slo_capacity_uses_lower_ci_and_contiguous_load_prefix():
    engine = _two_layer_serving_engine()
    simulator = ContinuousBatchingSimulator(
        engine,
        UniformServingRouter(1, 1, ("node0",), seed=3),
        n_layers=2,
        max_batch=2,
    )
    result = evaluate_slo_capacity(
        simulator=simulator,
        offered_rates_per_s=(20.0, 0.05, 1.0),
        seeds=(0, 1, 2),
        n_requests=6,
        output_tokens=1,
        slo_ttft_s=1.0,
        slo_tpot_s=0.0,
        target_attainment=0.5,
    )
    assert [point.offered_rate_per_s for point in result.points] == [
        0.05, 1.0, 20.0,
    ]
    assert result.capacity_rate_per_s == pytest.approx(1.0)
    assert [point.lower_bound_meets_target for point in result.points] == [
        True, True, False,
    ]
    assert all(
        point.lower_bound_meets_target == (
            point.aggregate.slo_attainment.low95 >=
            result.target_attainment
        )
        for point in result.points
    )
    assert result.monotonic_attainment
    assert not result.warning


def test_slo_capacity_reports_right_censored_grid():
    engine = _two_layer_serving_engine()
    simulator = ContinuousBatchingSimulator(
        engine,
        UniformServingRouter(1, 1, ("node0",), seed=3),
        n_layers=2,
        max_batch=2,
    )
    result = evaluate_slo_capacity(
        simulator=simulator,
        offered_rates_per_s=(0.05, 1.0),
        seeds=(0, 1),
        n_requests=4,
        output_tokens=1,
        slo_ttft_s=2.0,
        slo_tpot_s=0.0,
        target_attainment=0.99,
    )
    assert result.capacity_rate_per_s == pytest.approx(1.0)
    assert "right-censored" in result.warning

    with pytest.raises(ValidationError, match="target attainment"):
        evaluate_slo_capacity(
            simulator, (1.0,), (0,), 2, 1, 1.0, 0.0,
            target_attainment=0.0,
        )


def test_node_count_sweep_fixes_chips_per_node_and_separates_ingress():
    scale_up, scale_out, bridge = _profiles()
    spec = NodeSweepSpec(
        node_counts=(4, 1, 2),
        chips_per_node=2,
        chip_sram_capacity_bytes=100,
        runtime_reserve_fraction=0.1,
        fixed_reserved_bytes=10,
        chip_provenance=_derived("sweep chip"),
        concentrated_ingress_count=1,
    )
    experts = [_expert(0, 0, 80), _expert(0, 1, 80)]
    cases = build_node_count_sweep(
        spec, scale_up, scale_out, bridge, experts
    )
    assert len(cases) == 6
    assert {case.node_count for case in cases} == {1, 2, 4}
    assert all(
        len(case.chips) == case.node_count * 2 for case in cases
    )
    assert len({
        case.per_node_hardware_fingerprint for case in cases
    }) == 1
    for case in cases:
        assert case.aggregate_usable_weight_bytes == (
            case.node_count * 2 * 80
        )
        assert case.capacity_ratio == pytest.approx(case.node_count)
        assert case.capacity_regime == (
            "unique-fit-boundary" if case.node_count == 1
            else "replication-capable"
        )
        assert case.routing_home_nodes == case.ingress_nodes
        assert case.bridge_link_pairs == len(case.ingress_nodes)
        if case.ingress_mode == "distributed":
            assert len(case.ingress_nodes) == case.node_count
        else:
            assert case.ingress_nodes == ("node0",)
        metadata = case.fairness_metadata()
        assert metadata["chips_per_node"] == 2
        assert metadata["total_chips"] == case.node_count * 2


def test_node_count_sweep_rejects_ambiguous_grids():
    provenance = _derived("sweep chip")
    with pytest.raises(ValidationError, match="unique"):
        NodeSweepSpec(
            node_counts=(1, 1),
            chips_per_node=2,
            chip_sram_capacity_bytes=100,
            chip_provenance=provenance,
        )
    with pytest.raises(ValidationError, match="unknown ingress"):
        NodeSweepSpec(
            node_counts=(1,),
            chips_per_node=2,
            chip_sram_capacity_bytes=100,
            chip_provenance=provenance,
            ingress_modes=("imaginary",),
        )
    assert classify_capacity_regime(0.99) == "spill-required"
    assert classify_capacity_regime(1.0) == "unique-fit-boundary"
    assert classify_capacity_regime(1.01) == "replication-capable"


def test_node_load_replication_separates_locality_and_compute_copies():
    experts = [_expert(0, 0, 40), _expert(0, 1, 40)]
    chips = [
        _chip("node0:chip0", "node0", capacity=100, reserve=0.0),
        _chip("node0:chip1", "node0", capacity=100, reserve=0.0),
        _chip("node1:chip0", "node1", capacity=100, reserve=0.0),
        _chip("node1:chip1", "node1", capacity=100, reserve=0.0),
    ]
    records = []
    for token_id in range(10):
        records.append(RoutingRecord(
            0, token_id, token_id, 0, "node0",
            (0,), (1.0,),
        ))
    records.append(RoutingRecord(
        0, 10, 10, 0, "node1", (0,), (1.0,),
    ))
    records.append(RoutingRecord(
        0, 11, 11, 0, "node1", (1,), (1.0,),
    ))
    trace = RoutingTrace(records, _derived("placement"), "placement")
    demand = profile_demand(trace)
    plan = node_load_replicated(
        experts, chips, demand, max_total_replicas=6
    )
    locations = [
        (replica.expert.expert_id, plan.chip(replica.chip_id).node_id)
        for replica in plan.replicas
    ]
    # First-copy locality covers both demanded experts on node1.
    assert (0, "node1") in locations
    assert (1, "node1") in locations
    # Only after locality is covered, the hottest node/expert gains a second
    # same-node copy for compute balance.
    assert locations.count((0, "node0")) == 2
    assert len({
        (replica.expert, replica.chip_id)
        for replica in plan.replicas
    }) == len(plan.replicas)
    assert plan.strategy == "node-load-replicated"


def test_node_sweep_trace_pairs_separate_splits_and_scaling_semantics():
    scale_up, scale_out, bridge = _profiles()
    cases = build_node_count_sweep(
        NodeSweepSpec(
            node_counts=(1, 4),
            chips_per_node=2,
            chip_sram_capacity_bytes=100,
            chip_provenance=_derived("sweep chip"),
            ingress_modes=("distributed",),
        ),
        scale_up, scale_out, bridge,
    )
    strong = SweepWorkloadSpec(
        n_layers=2,
        n_experts=4,
        top_k=2,
        placement_tokens=8,
        evaluation_tokens=12,
        placement_seed=3,
        evaluation_seed=5,
        provenance=_derived("synthetic routing"),
        scaling_mode="strong",
    )
    weak = replace(strong, scaling_mode="weak")
    strong_pairs = [
        build_case_trace_pair(case, strong) for case in cases
    ]
    weak_pairs = [
        build_case_trace_pair(case, weak) for case in cases
    ]
    assert [pair.placement_token_count for pair in strong_pairs] == [8, 8]
    assert [pair.evaluation_token_count for pair in strong_pairs] == [12, 12]
    assert [pair.placement_token_count for pair in weak_pairs] == [8, 32]
    assert [pair.evaluation_token_count for pair in weak_pairs] == [12, 48]
    assert all(pair.placement.split == "placement"
               for pair in strong_pairs + weak_pairs)
    assert all(pair.evaluation.split == "evaluation"
               for pair in strong_pairs + weak_pairs)
    assert (
        strong_pairs[0].placement.records !=
        strong_pairs[0].evaluation.records[:len(
            strong_pairs[0].placement.records
        )]
    )

    with pytest.raises(ValidationError, match="seeds must differ"):
        replace(strong, evaluation_seed=strong.placement_seed)

    drift = replace(
        strong,
        routing_model="weighted",
        placement_expert_weights=(10.0, 1.0, 1.0, 1.0),
        evaluation_expert_weights=(1.0, 1.0, 1.0, 10.0),
    )
    drift_pair = build_case_trace_pair(cases[-1], drift)
    placement_counts = profile_demand(drift_pair.placement)
    evaluation_counts = {
        key: 0 for key in (
            ExpertKey(0, 0), ExpertKey(0, 3)
        )
    }
    for assignment in drift_pair.evaluation.assignments():
        if assignment.expert in evaluation_counts:
            evaluation_counts[assignment.expert] += 1
    assert (
        placement_counts.global_for(ExpertKey(0, 0)) >
        placement_counts.global_for(ExpertKey(0, 3))
    )
    assert (
        evaluation_counts[ExpertKey(0, 3)] >
        evaluation_counts[ExpertKey(0, 0)]
    )


def test_system_energy_accounts_all_provisioned_chips_and_busy_intervals():
    engine = _two_layer_serving_engine()
    simulator = ContinuousBatchingSimulator(
        engine,
        UniformServingRouter(1, 1, ("node0",), seed=3),
        n_layers=2,
        max_batch=2,
    )
    serving = simulator.run([
        DecodeRequest(0, 0.0, 2),
        DecodeRequest(1, 0.05, 1),
    ])
    profile = ChipPowerProfile(
        idle_power_w=2.0,
        compute_dynamic_power_w=3.0,
        provenance=_derived("power fixture"),
    )
    energy = account_system_energy(
        serving.layer_results,
        {
            "node0:chip0": profile,
            # An unused provisioned chip still consumes idle power.
            "node0:chip1": profile,
        },
        output_tokens=serving.output_tokens,
        observation_start_s=serving.start_s,
        observation_finish_s=serving.finish_s,
    )
    assert energy.provisioned_chips == 2
    assert energy.chip_busy_s == pytest.approx(0.5)
    assert energy.compute_dynamic_j == pytest.approx(1.5)
    assert energy.idle_j == pytest.approx(serving.finish_s * 4.0)
    assert energy.network_j == pytest.approx(serving.network_energy_j)
    assert energy.total_j == pytest.approx(
        energy.compute_dynamic_j + energy.idle_j + energy.network_j
    )
    assert energy.energy_per_token_j == pytest.approx(
        energy.total_j / 3
    )
    assert energy.decisive_confidence is Confidence.DERIVED

    powered_simulator = ContinuousBatchingSimulator(
        engine,
        UniformServingRouter(1, 1, ("node0",), seed=3),
        n_layers=2,
        max_batch=2,
        chip_power_profiles={"node0:chip0": profile},
    )
    powered = powered_simulator.run([
        DecodeRequest(0, 0.0, 1),
    ])
    powered_aggregate = aggregate_serving_results(
        [powered], slo_ttft_s=2.0, slo_tpot_s=1.0
    )
    assert powered.system_energy_j is not None
    assert powered_aggregate.system_energy_per_token_j is not None
    assert (
        powered_aggregate.energy_per_token_j ==
        powered_aggregate.system_energy_per_token_j
    )

    with pytest.raises(ValidationError, match="missing power profile"):
        account_system_energy(
            serving.layer_results,
            {"unused": profile},
            output_tokens=3,
            observation_finish_s=serving.finish_s,
        )


def test_admission_rejections_count_against_offered_slo_goodput():
    simulator = ContinuousBatchingSimulator(
        _two_layer_serving_engine(),
        UniformServingRouter(1, 1, ("node0",), seed=3),
        n_layers=2,
        max_batch=1,
        admission_policy=AdmissionPolicy(max_outstanding_requests=1),
    )
    result = simulator.run([
        DecodeRequest(0, 0.0, 1),
        DecodeRequest(1, 0.0, 1),
        DecodeRequest(2, 0.0, 1),
    ])
    assert result.offered_requests == 3
    assert len(result.requests) == 1
    assert result.rejected_request_ids == (1, 2)
    assert result.admission_fraction == pytest.approx(1 / 3)
    assert result.slo_attainment(2.0, 1.0) == pytest.approx(1 / 3)
    assert result.slo_goodput_per_s(2.0, 1.0) == pytest.approx(
        1 / (result.finish_s - result.start_s)
    )
    aggregate = aggregate_serving_results(
        [result], slo_ttft_s=2.0, slo_tpot_s=1.0
    )
    assert aggregate.admission_fraction.mean == pytest.approx(1 / 3)
    assert aggregate.slo_goodput_per_s.mean == pytest.approx(
        result.slo_goodput_per_s(2.0, 1.0)
    )


def test_fixed_window_runner_compares_strategies_on_held_out_trace(
        tmp_path):
    scale_up, scale_out, bridge = _profiles()
    case = build_node_count_sweep(
        NodeSweepSpec(
            node_counts=(2,),
            chips_per_node=1,
            chip_sram_capacity_bytes=100,
            runtime_reserve_fraction=0.0,
            chip_provenance=_derived("runner chip"),
            ingress_modes=("distributed",),
        ),
        scale_up, scale_out, bridge,
    )[0]
    workload = SweepWorkloadSpec(
        n_layers=1,
        n_experts=2,
        top_k=1,
        placement_tokens=8,
        evaluation_tokens=8,
        placement_seed=3,
        evaluation_seed=5,
        provenance=_derived("runner routing"),
    )
    traces = build_case_trace_pair(case, workload)
    experts = tuple(
        _expert(0, expert_id, 40) for expert_id in range(2)
    )
    kernels = {
        expert.key: ExpertKernelCurve(
            "tiny", "bf16", 1, "sram",
            [KernelSample(1, 0.1), KernelSample(8, 0.3)],
            _derived("runner kernel"),
        )
        for expert in experts
    }
    comparison = run_fixed_window_comparison(
        case=case,
        traces=traces,
        experts=experts,
        kernel_models=kernels,
        hidden_bytes=10,
        strategies=(
            "unique-balanced",
            "popularity-replicated",
            "ingress-local",
            "node-load-replicated",
            "coactivation-clustered",
        ),
        max_total_replicas=4,
    )
    assert comparison.evaluation_split == "evaluation"
    assert comparison.case_id == case.case_id
    assert len(comparison.strategies) == 5
    for result in comparison.strategies:
        assert result.assignment_count == 8
        assert result.completion_s > 0
        assert len(result.layer_results) == 1
        assert 0 <= result.local_assignment_fraction <= 1
    assert (
        comparison.result_for("unique-balanced").replica_plan.strategy ==
        "unique-balanced"
    )
    artifact = tmp_path / "input.txt"
    artifact.write_text("fixture\n", encoding="utf-8")
    bundle = write_comparison_result_bundle(
        tmp_path / "comparison",
        {"declared": True},
        case,
        traces,
        comparison,
        seed=5,
        command="pytest comparison",
        input_artifacts=(artifact,),
    )
    assert {
        path.name for path in bundle.iterdir()
    } == {
        "case.json",
        "config.resolved.yaml",
        "layers.csv",
        "manifest.json",
        "placements.json",
        "strategies.csv",
        "traces.json",
        "warnings.json",
    }
    manifest = json.loads(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["bundle_kind"] == "fixed-window-comparison"
    assert manifest["case_id"] == case.case_id
    strategy_rows = (
        bundle / "strategies.csv"
    ).read_text(encoding="utf-8").splitlines()
    assert len(strategy_rows) == len(comparison.strategies) + 1
    with pytest.raises(ValidationError, match="already exists"):
        write_comparison_result_bundle(
            bundle, {}, case, traces, comparison, 5, "no overwrite"
        )

    with pytest.raises(ValidationError, match="held-out"):
        run_fixed_window_comparison(
            case, replace(traces, evaluation=traces.placement),
            experts, kernels, hidden_bytes=10,
            strategies=("unique-balanced",),
        )


def test_node_count_runner_reports_strong_scaling_and_budget_mode(
        tmp_path):
    scale_up, scale_out, bridge = _profiles()
    cases = build_node_count_sweep(
        NodeSweepSpec(
            node_counts=(1, 2),
            chips_per_node=2,
            chip_sram_capacity_bytes=100,
            runtime_reserve_fraction=0.0,
            chip_provenance=_derived("sweep runner chip"),
            ingress_modes=("distributed",),
        ),
        scale_up, scale_out, bridge,
    )
    workload = SweepWorkloadSpec(
        n_layers=1,
        n_experts=2,
        top_k=1,
        placement_tokens=6,
        evaluation_tokens=8,
        placement_seed=2,
        evaluation_seed=4,
        provenance=_derived("sweep runner routing"),
        scaling_mode="strong",
    )
    experts = tuple(_expert(0, expert_id, 40)
                    for expert_id in range(2))
    kernels = {
        expert.key: ExpertKernelCurve(
            "tiny", "bf16", 1, "sram",
            [KernelSample(1, 0.1), KernelSample(8, 0.3)],
            _derived("sweep runner kernel"),
        )
        for expert in experts
    }
    result = run_node_count_comparisons(
        cases=cases,
        workload=workload,
        experts=experts,
        kernel_models=kernels,
        hidden_bytes=10,
        strategies=("unique-balanced", "ingress-local"),
        max_total_replicas=2,
    )
    assert result.scaling_mode == "strong"
    assert result.replica_budget_mode == "fixed-total-replicas"
    assert result.fixed_max_total_replicas == 2
    assert len(result.comparisons) == 2
    assert len(result.points) == 4
    assert all(point.resident_replicas == 2
               for point in result.points)
    assert all(point.speedup_vs_unique == pytest.approx(1.0)
               for point in result.points
               if point.strategy == "unique-balanced")
    assert {point.capacity_ratio for point in result.points} == {2.5, 5.0}
    assert all(point.tokens_per_s_per_node == pytest.approx(
        point.tokens_per_s / point.node_count
    ) for point in result.points)
    bundle = write_node_sweep_bundle(
        tmp_path / "sweep",
        {"node_counts": [1, 2]},
        result,
        seed=4,
        command="pytest sweep",
    )
    assert {
        path.name for path in bundle.iterdir()
    } == {
        "cases.json", "config.resolved.yaml",
        "manifest.json", "sweep.csv",
    }
    manifest = json.loads(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["replica_budget_mode"] == "fixed-total-replicas"
    assert len(
        (bundle / "sweep.csv").read_text(
            encoding="utf-8"
        ).splitlines()
    ) == len(result.points) + 1


def test_strict_v2_yaml_config_runs_and_hashes_input(tmp_path):
    source_config = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "sim_v2", "node_sweep_smoke.yaml",
    )
    with open(source_config, "r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle)
    config_data["sweep"]["node_counts"] = [1]
    config_data["sweep"]["ingress_modes"] = ["distributed"]
    config_data["execution"]["strategies"] = ["unique-balanced"]
    config_data["serving"]["offered_rates_per_s"] = [1.0]
    config_data["serving"]["seeds"] = [31]
    config_data["uncertainty"]["placement_seeds"] = [11]
    config_data["uncertainty"]["evaluation_seeds"] = [17]
    config_path = tmp_path / "mini-v2.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, sort_keys=False),
        encoding="utf-8",
    )
    output, result = run_node_sweep_config(
        config_path, tmp_path / "configured-sweep"
    )
    assert len(result.comparisons) == 1
    assert len(result.points) == 1
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["bundle_kind"] == "node-count-sweep"
    assert str(config_path.resolve()) in (
        manifest["input_artifact_sha256"]
    )
    serving_output, serving = run_node_serving_config(
        config_path, tmp_path / "configured-serving"
    )
    assert len(serving.points) == 1
    assert (serving_output / "load_points.csv").is_file()
    serving_manifest = json.loads(
        (serving_output / "manifest.json").read_text(encoding="utf-8")
    )
    assert str(config_path.resolve()) in (
        serving_manifest["input_artifact_sha256"]
    )
    uncertainty_output, uncertainty = run_node_uncertainty_config(
        config_path, tmp_path / "configured-uncertainty"
    )
    assert len(uncertainty.points) == 1
    assert (uncertainty_output / "uncertainty.csv").is_file()
    uncertainty_manifest = json.loads(
        (uncertainty_output / "manifest.json").read_text(encoding="utf-8")
    )
    assert str(config_path.resolve()) in (
        uncertainty_manifest["input_artifact_sha256"]
    )
    tiered_output, tiered = run_node_tiered_config(
        config_path, tmp_path / "configured-tiered"
    )
    assert len(tiered.points) == 2
    tiered_manifest = json.loads(
        (tiered_output / "manifest.json").read_text(encoding="utf-8")
    )
    assert str(config_path.resolve()) in (
        tiered_manifest["input_artifact_sha256"]
    )


def test_v2_config_loads_case_mapped_router_traces_and_hashes_them(
        tmp_path):
    source_config = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "sim_v2", "node_sweep_smoke.yaml",
    )
    with open(source_config, "r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle)
    config_data["sweep"]["node_counts"] = [1]
    config_data["sweep"]["ingress_modes"] = ["distributed"]
    config_data["execution"]["strategies"] = ["unique-balanced"]
    config_data["workload"].update({
        "placement_tokens": 1,
        "evaluation_tokens": 1,
        "routing_model": "trace",
        "placement_trace_template":
            "placement-{case_id}.jsonl",
        "evaluation_trace_template":
            "evaluation-{case_id}.jsonl",
    })
    case_id = "nodes-1__chips-2__ingress-distributed-1"
    record = {
        "request_id": 0,
        "token_id": 0,
        "step": 0,
        "layer_id": 0,
        "home_node": "node0",
        "experts": [0, 1],
        "gate_weights": [0.5, 0.5],
    }
    trace_paths = []
    for split in ("placement", "evaluation"):
        trace_path = tmp_path / ("%s-%s.jsonl" % (split, case_id))
        trace_path.write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )
        trace_paths.append(trace_path)
    config_path = tmp_path / "trace-v2.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, sort_keys=False),
        encoding="utf-8",
    )

    output, result = run_node_sweep_config(
        config_path, tmp_path / "trace-sweep"
    )
    assert len(result.points) == 1
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    hashes = manifest["input_artifact_sha256"]
    assert str(config_path.resolve()) in hashes
    assert all(str(path.resolve()) in hashes for path in trace_paths)
    with pytest.raises(ValidationError, match="online router model"):
        run_node_serving_config(
            config_path, tmp_path / "trace-serving"
        )


def test_weighted_routing_supports_hot_experts_and_node_affinity():
    weights = zipf_expert_weights(4, exponent=1.5)
    assert sum(weights) == pytest.approx(1.0)
    trace = generate_weighted_trace(
        n_tokens=200,
        n_layers=1,
        expert_weights=weights,
        top_k=1,
        home_nodes=("n0", "n1"),
        seed=13,
        provenance=_derived("weighted synthetic"),
        split="placement",
        node_expert_multipliers={
            ("n0", 0): 20.0,
            ("n1", 0): 0.0,
            ("n1", 3): 20.0,
        },
    )
    repeated = generate_weighted_trace(
        200, 1, weights, 1, ("n0", "n1"), 13,
        _derived("weighted synthetic"), "placement",
        {
            ("n0", 0): 20.0,
            ("n1", 0): 0.0,
            ("n1", 3): 20.0,
        },
    )
    assert trace.records == repeated.records
    n0_hot = sum(
        record.experts == (0,) for record in trace.records
        if record.home_node == "n0"
    )
    n1_hot = sum(
        record.experts == (3,) for record in trace.records
        if record.home_node == "n1"
    )
    assert n0_hot > 70
    assert n1_hot > 40
    assert all(
        record.experts != (0,) for record in trace.records
        if record.home_node == "n1"
    )

    with pytest.raises(ValidationError, match="positive routing"):
        generate_weighted_trace(
            2, 1, (1.0, 0.0), 2, ("n0",), 1,
            _derived("bad weights"), "placement",
        )


def test_tiering_compares_unique_coverage_with_hot_replication():
    experts = tuple(_expert(0, expert_id, 40)
                    for expert_id in range(3))
    brain = (
        _chip("node0:chip0", "node0", capacity=80, reserve=0.0),
        _chip("node0:chip1", "node0", capacity=80, reserve=0.0),
    )
    gpu = make_gpu_tier_chip(
        experts, "gpu-tier0", _derived("GPU HBM"), capacity_bytes=200
    )
    demand = {
        experts[0].key: 100.0,
        experts[1].key: 10.0,
        experts[2].key: 1.0,
    }
    unique = tiered_expert_plan(
        experts, brain, gpu, demand,
        policy="unique-first", max_brain_replicas=4,
    )
    replicated = tiered_expert_plan(
        experts, brain, gpu, demand,
        policy="hot-replica-first", max_brain_replicas=4,
    )
    unique_brain_experts = {
        replica.expert for replica in unique.replicas
        if replica.chip_id != gpu.chip_id
    }
    hot_brain_replicas = [
        replica for replica in replicated.replicas
        if replica.chip_id != gpu.chip_id and
        replica.expert == experts[0].key
    ]
    assert unique_brain_experts == {expert.key for expert in experts}
    assert len(hot_brain_replicas) == 2
    assert len({
        (replica.expert, replica.chip_id)
        for replica in replicated.replicas
    }) == len(replicated.replicas)


def test_gpu_origin_tiered_execution_does_not_loop_over_bridge():
    scale_up, scale_out, bridge = _profiles()
    cluster = build_hierarchical_cluster(
        1, 1, (0,), scale_up, scale_out, bridge
    )
    expert = _expert(0, 0, 40)
    brain = (_chip("node0:chip0", "node0"),)
    gpu = make_gpu_tier_chip(
        (expert,), "gpu-tier0", _derived("GPU HBM")
    )
    plan = tiered_expert_plan(
        (expert,), brain, gpu, {expert.key: 1.0},
        policy="unique-first", max_brain_replicas=0,
    )
    kernel = ExpertKernelCurve(
        "gpu", "bf16", 1, "hbm",
        [KernelSample(1, 0.1)], _derived("GPU kernel"),
    )
    dispatcher = Dispatcher(
        plan,
        cluster.topology,
        cluster.home_endpoints,
        dict(cluster.chip_endpoints, **{"gpu-tier0": "gpu0"}),
        {expert.key: kernel},
        hidden_bytes=10,
        policy="nearest",
        ingress_source_endpoint="gpu0",
    )
    result = MoELayerEngine(dispatcher).run([
        Assignment("gpu-a0", 0, 0, 0, 0, 0, 1.0, "node0")
    ])
    assert result.replica_executions[0].chip_id == "gpu-tier0"
    assert all(nbytes == 0 for nbytes in result.link_bytes.values())
    assert result.network_energy_j == 0


def test_tiered_runner_uses_device_specific_kernel_curves(tmp_path):
    scale_up, scale_out, bridge = _profiles()
    cases = build_node_count_sweep(
        NodeSweepSpec(
            node_counts=(1, 2),
            chips_per_node=2,
            chip_sram_capacity_bytes=80,
            runtime_reserve_fraction=0.0,
            chip_provenance=_derived("tier brain"),
            ingress_modes=("distributed",),
        ),
        scale_up, scale_out, bridge,
    )
    case = cases[0]
    workload = SweepWorkloadSpec(
        n_layers=1, n_experts=3, top_k=1,
        placement_tokens=20, evaluation_tokens=20,
        placement_seed=1, evaluation_seed=2,
        provenance=_derived("tier routing"),
        routing_model="weighted",
        placement_expert_weights=(10.0, 2.0, 1.0),
        evaluation_expert_weights=(10.0, 2.0, 1.0),
    )
    traces = build_case_trace_pair(case, workload)
    experts = tuple(_expert(0, value, 40) for value in range(3))
    brain_models = {
        expert.key: ExpertKernelCurve(
            "brain", "bf16", 1, "sram",
            [KernelSample(1, 0.01), KernelSample(20, 0.1)],
            _derived("brain kernel"),
        ) for expert in experts
    }
    gpu_models = {
        expert.key: ExpertKernelCurve(
            "gpu", "bf16", 1, "hbm",
            [KernelSample(1, 0.2), KernelSample(20, 0.4)],
            _derived("GPU kernel"),
        ) for expert in experts
    }
    comparison = run_tiered_fixed_window_comparison(
        case, traces, experts, brain_models, gpu_models,
        _derived("GPU HBM"), hidden_bytes=10,
        max_brain_replicas=4,
    )
    assert {
        item.strategy for item in comparison.strategies
    } == {"tiered-unique-first", "tiered-hot-replica-first"}
    for item in comparison.strategies:
        assert item.assignment_count == 20
        assert any(
            replica.chip_id == "gpu-tier0"
            for replica in item.replica_plan.replicas
        )
        assert item.completion_s > 0
    sweep = run_tiered_node_count_comparisons(
        cases=cases,
        workload=workload,
        experts=experts,
        brain_kernel_models=brain_models,
        gpu_kernel_models=gpu_models,
        gpu_provenance=_derived("GPU HBM"),
        hidden_bytes=10,
        max_brain_replicas=None,
    )
    assert len(sweep.points) == 4
    assert {
        point.policy for point in sweep.points
    } == {"tiered-unique-first", "tiered-hot-replica-first"}
    assert all(
        point.gpu_fallback_experts == 3 for point in sweep.points
    )
    bundle = write_tiered_sweep_bundle(
        tmp_path / "tiered-sweep", {}, sweep, "pytest tiered sweep"
    )
    assert {
        path.name for path in bundle.iterdir()
    } == {
        "config.resolved.yaml", "manifest.json", "tiered_sweep.csv",
    }


def test_strict_jsonl_router_trace_loader_preserves_real_fields(tmp_path):
    path = tmp_path / "router.jsonl"
    path.write_text(
        '{"request_id":0,"token_id":0,"step":0,"layer_id":0,'
        '"home_node":"node0","experts":[2,1],'
        '"gate_weights":[0.7,0.3]}\n'
        '{"request_id":0,"token_id":1,"step":1,"layer_id":0,'
        '"home_node":"node1","experts":[0,2],'
        '"gate_weights":[0.6,0.4]}\n',
        encoding="utf-8",
    )
    trace = load_routing_trace_jsonl(
        path, _derived("captured router"), "evaluation"
    )
    assert trace.split == "evaluation"
    assert trace.top_k == 2
    assert trace.records[0].experts == (2, 1)
    assert trace.records[0].gate_weights == (0.7, 0.3)
    assert trace.summary().home_nodes == 2

    enriched = tmp_path / "router-enriched.jsonl"
    enriched.write_text(
        '{"request_id":7,"token_id":0,"step":0,"layer_id":0,'
        '"home_node":"node0","experts":[2,1],'
        '"gate_weights":[0.7,0.3],"task_family":"code",'
        '"stage":"decode","timestamp_s":1.25,"window":"w0"}\n',
        encoding="utf-8",
    )
    enriched_trace = load_routing_trace_jsonl(
        enriched, _derived("captured router"), "evaluation"
    )
    record = enriched_trace.records[0]
    assert record.task_family == "code"
    assert record.stage == "decode"
    assert record.timestamp_s == 1.25
    assert record.window == "w0"

    bad = tmp_path / "bad-router.jsonl"
    bad.write_text(
        '{"request_id":0,"token_id":0,"step":0,"layer_id":0,'
        '"home_node":"node0","experts":[0],"gate_weights":[1],'
        '"typo":true}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="unknown field typo"):
        load_routing_trace_jsonl(
            bad, _derived("bad router"), "evaluation"
        )


def test_node_serving_sweep_reports_slo_capacity_per_strategy(tmp_path):
    scale_up, scale_out, bridge = _profiles()
    cases = build_node_count_sweep(
        NodeSweepSpec(
            node_counts=(1, 2),
            chips_per_node=1,
            chip_sram_capacity_bytes=100,
            runtime_reserve_fraction=0.0,
            chip_provenance=_derived("serving sweep chip"),
            ingress_modes=("distributed",),
        ),
        scale_up, scale_out, bridge,
    )
    workload = SweepWorkloadSpec(
        n_layers=1, n_experts=1, top_k=1,
        placement_tokens=4, evaluation_tokens=4,
        placement_seed=1, evaluation_seed=2,
        provenance=_derived("serving sweep routing"),
    )
    expert = _expert(0, 0, 40)
    kernels = {
        expert.key: ExpertKernelCurve(
            "tiny", "bf16", 1, "sram",
            [KernelSample(1, 0.01), KernelSample(2, 0.015)],
            _derived("serving sweep kernel"),
        )
    }
    result = run_node_serving_sweep(
        cases=cases,
        workload=workload,
        experts=(expert,),
        kernel_models=kernels,
        hidden_bytes=10,
        offered_rates_per_s=(0.1, 10.0),
        seeds=(1, 2),
        n_requests=4,
        output_tokens=1,
        max_batch=2,
        slo_ttft_s=1.0,
        slo_tpot_s=0.0,
        target_attainment=0.5,
        strategies=("unique-balanced", "ingress-local"),
    )
    assert result.offered_rates_per_s == (0.1, 10.0)
    assert result.seeds == (1, 2)
    assert len(result.points) == 4
    assert all(
        point.capacity_per_node_per_s == pytest.approx(
            point.capacity_rate_per_s / point.node_count
        )
        for point in result.points
    )
    assert all(
        len(point.capacity_result.points) == 2
        for point in result.points
    )
    bundle = write_serving_sweep_bundle(
        tmp_path / "serving-sweep", {}, result, "pytest serving"
    )
    assert {
        path.name for path in bundle.iterdir()
    } == {
        "capacity.csv", "config.resolved.yaml",
        "load_points.csv", "manifest.json",
    }
    assert len(
        (bundle / "load_points.csv").read_text(
            encoding="utf-8"
        ).splitlines()
    ) == 1 + len(result.points) * 2


def test_prefill_block_windows_delay_decode_only_at_boundaries():
    request = DecodeRequest(0, 0.0, 1)
    baseline = ContinuousBatchingSimulator(
        _two_layer_serving_engine(),
        UniformServingRouter(1, 1, ("node0",), seed=1),
        n_layers=2,
        max_batch=1,
    ).run([request])
    blocked = ContinuousBatchingSimulator(
        _two_layer_serving_engine(),
        UniformServingRouter(1, 1, ("node0",), seed=1),
        n_layers=2,
        max_batch=1,
        compute_block_windows=(
            ResourceBlockWindow(0.0, 0.5, "prefill-chunk"),
        ),
    ).run([request])
    assert baseline.prefill_blocked_s == 0
    assert blocked.prefill_blocked_s == pytest.approx(0.5)
    assert blocked.finish_s == pytest.approx(
        baseline.finish_s + 0.5
    )
    assert blocked.ttft_percentile(50) == pytest.approx(
        baseline.ttft_percentile(50) + 0.5
    )

    with pytest.raises(ValidationError, match="must not overlap"):
        ContinuousBatchingSimulator(
            _two_layer_serving_engine(),
            UniformServingRouter(1, 1, ("node0",), seed=1),
            n_layers=2,
            max_batch=1,
            compute_block_windows=(
                ResourceBlockWindow(0.0, 0.5),
                ResourceBlockWindow(0.4, 0.6),
            ),
        )


def test_node_sweep_uncertainty_uses_paired_speedup_intervals(
        tmp_path):
    scale_up, scale_out, bridge = _profiles()
    cases = build_node_count_sweep(
        NodeSweepSpec(
            node_counts=(1,),
            chips_per_node=2,
            chip_sram_capacity_bytes=100,
            runtime_reserve_fraction=0.0,
            chip_provenance=_derived("uncertainty chip"),
            ingress_modes=("distributed",),
        ),
        scale_up, scale_out, bridge,
    )
    workload = SweepWorkloadSpec(
        n_layers=1, n_experts=2, top_k=1,
        placement_tokens=8, evaluation_tokens=8,
        placement_seed=1, evaluation_seed=3,
        provenance=_derived("uncertainty routing"),
        routing_model="weighted",
        placement_expert_weights=(4.0, 1.0),
        evaluation_expert_weights=(4.0, 1.0),
    )
    experts = tuple(_expert(0, index, 40) for index in range(2))
    kernels = {
        expert.key: ExpertKernelCurve(
            "tiny", "bf16", 1, "sram",
            [KernelSample(1, 0.01), KernelSample(8, 0.04)],
            _derived("uncertainty kernel"),
        ) for expert in experts
    }
    result = run_node_count_uncertainty(
        cases=cases,
        workload=workload,
        placement_seeds=(1, 2),
        evaluation_seeds=(3, 4),
        experts=experts,
        kernel_models=kernels,
        hidden_bytes=10,
        strategies=("unique-balanced", "ingress-local"),
        max_total_replicas=2,
    )
    assert result.placement_seeds == (1, 2)
    assert result.evaluation_seeds == (3, 4)
    assert len(result.points) == 2
    assert all(point.runs == 4 for point in result.points)
    baseline = next(
        point for point in result.points
        if point.strategy == "unique-balanced"
    )
    assert baseline.conclusion == "baseline"
    assert baseline.speedup_vs_unique.mean == pytest.approx(1.0)
    assert all(
        point.conclusion in {
            "baseline", "supported", "unsupported", "inconclusive"
        }
        for point in result.points
    )
    bundle = write_uncertainty_bundle(
        tmp_path / "uncertainty", {}, result, "pytest uncertainty"
    )
    assert {
        path.name for path in bundle.iterdir()
    } == {
        "config.resolved.yaml", "manifest.json", "uncertainty.csv",
    }
    manifest = json.loads(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["bundle_kind"] == "node-sweep-uncertainty"
