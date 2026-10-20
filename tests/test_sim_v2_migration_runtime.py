from flowgpu.sim_v2.migration_runtime import (
    build_migration_flows,
    replay_with_shared_migration,
)
from flowgpu.sim_v2.network_model import Flow
from flowgpu.sim_v2.refresh import MigrationCopy, MigrationPlan
from flowgpu.sim_v2.schema import Confidence, ExpertKey, Provenance
from flowgpu.sim_v2.topology import LinkProfile, build_hierarchical_cluster


def _cluster():
    provenance = Provenance(Confidence.PROJECTED, "migration runtime test")
    link = LinkProfile(10.0, 0.0, 1.0, 0.0, provenance)
    return build_hierarchical_cluster(
        1, 2, (0,), link, link, link,
    ), provenance


def test_shared_migration_asset_delays_serving_flow():
    cluster, provenance = _cluster()
    serving = Flow(
        "serve", cluster.gpu_endpoint, "node0:chip0", 10,
        traffic_class="dispatch",
    )
    migration = MigrationPlan((
        MigrationCopy(
            ExpertKey(0, 0),
            cluster.gpu_endpoint,
            "node0:chip1",
            "node0",
            20,
        ),
    ), ())
    flows = build_migration_flows(migration, cluster, ())
    assert len(flows) == 1
    assert flows[0].traffic_class == "migration"
    result = replay_with_shared_migration(
        cluster, (serving,), migration, ()
    )
    assert result.migration_bytes == 20
    assert result.serving_completion_s > 0
    assert result.shared_serving_completion_s >= result.serving_completion_s
    assert (
        result.serving_delay_s
        == result.shared_serving_completion_s - result.serving_completion_s
    )
    assert result.combined_completion_s >= result.serving_completion_s
