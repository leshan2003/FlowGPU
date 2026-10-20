from flowgpu.sim_v2.dispatcher import Dispatcher
from flowgpu.sim_v2.layer_engine import MoELayerEngine
from flowgpu.sim_v2.kernel_model import ExpertKernelCurve, KernelSample
from flowgpu.sim_v2.replicas import ReplicaPlan
from flowgpu.sim_v2.schema import (
    Assignment,
    ChipSpec,
    Confidence,
    ExpertKey,
    ExpertSpec,
    Provenance,
    ReplicaSpec,
)
from flowgpu.sim_v2.topology import LinkProfile, build_hierarchical_cluster


def _profile():
    provenance = Provenance(Confidence.PROJECTED, "dispatch mode test")
    link = LinkProfile(
        100.0,
        1e-6,
        1.0,
        0.0,
        provenance,
    )
    return provenance, link


def _objects(mode, home_node="node1"):
    provenance, link = _profile()
    cluster = build_hierarchical_cluster(
        2,
        1,
        (0, 1),
        link,
        link,
        link,
    )
    key = ExpertKey(0, 0)
    expert = ExpertSpec(key, 10, provenance)
    chip = ChipSpec(
        "node0:chip0",
        "node0",
        100,
        0.0,
        0,
        provenance,
        role="brain_sram",
        max_experts=1,
    )
    plan = ReplicaPlan(
        [expert],
        [chip],
        [ReplicaSpec("brain", key, chip.chip_id, True)],
        "test",
    )
    kernel = ExpertKernelCurve(
        "brain",
        "fp8",
        1,
        "sram",
        [KernelSample(1, 1e-6)],
        provenance,
    )
    chip_endpoints = dict(cluster.chip_endpoints)
    chip_endpoints["gpu-chip0"] = cluster.gpu_endpoint
    dispatcher = Dispatcher(
        plan,
        cluster.topology,
        cluster.home_endpoints,
        chip_endpoints,
        {(key, chip.chip_id): kernel},
        hidden_bytes=10,
        policy="nearest",
        ingress_source_endpoint=cluster.gpu_endpoint,
        dispatch_mode=mode,
        gpu_ingress_endpoints=(cluster.gpu_endpoint,),
        gpu_endpoint_ids=(cluster.gpu_endpoint,),
    )
    assignment = Assignment(
        "a0",
        request_id=0,
        token_id=0,
        step=0,
        layer_id=0,
        expert_id=key.expert_id,
        gate_weight=1.0,
        home_node=home_node,
    )
    return cluster, dispatcher, assignment


def _case(mode, home_node="node1"):
    _, dispatcher, assignment = _objects(mode, home_node)
    return dispatcher.plan([assignment]).bindings[0]


def test_gpu_direct_bypasses_remote_home_node():
    direct = _case("gpu-direct")
    via_home = _case("gpu-to-home")

    assert direct.dispatch_s < via_home.dispatch_s
    assert direct.combine_s < via_home.combine_s


def test_home_resident_starts_at_home_without_gpu_bridge():
    home = _case("home-resident", home_node="node0")
    direct = _case("gpu-direct", home_node="node0")

    assert home.dispatch_s < direct.dispatch_s
    assert home.combine_s < direct.combine_s


def test_multiple_gpu_ingress_endpoints_are_explicit():
    _, link = _profile()
    cluster = build_hierarchical_cluster(
        2,
        1,
        (0, 1),
        link,
        link,
        link,
        gpu_ingress_count=4,
    )

    assert cluster.gpu_endpoints == ("gpu0", "gpu1", "gpu2", "gpu3")
    assert all("gpu%d" % index in cluster.topology.endpoints
               for index in range(4))
    assert "bridge:gpu3:node1:fwd" in cluster.topology.links


def test_gpu_direct_event_flow_does_not_use_home_bridge():
    cluster, dispatcher, assignment = _objects("gpu-direct")
    direct = MoELayerEngine(dispatcher).run([assignment])
    direct_home_bridge = sum(
        nbytes for link_id, nbytes in direct.link_bytes.items()
        if link_id.startswith("bridge:node1:")
    )

    _, via_home_dispatcher, via_home_assignment = _objects(
        "gpu-to-home"
    )
    via_home = MoELayerEngine(via_home_dispatcher).run([
        via_home_assignment
    ])
    home_bridge = sum(
        nbytes for link_id, nbytes in via_home.link_bytes.items()
        if link_id.startswith("bridge:node1:")
    )

    assert direct_home_bridge == 0
    assert home_bridge > 0
    assert direct.completion_s < via_home.completion_s
