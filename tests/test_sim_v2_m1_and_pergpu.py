import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_moe_d0_pergpu import run_seed as run_pergpu_seed  # noqa: E402
from run_moe_m1_full import run_seed as run_m1_seed  # noqa: E402


def _common():
    return dict(
        model="mixtral_8x7b",
        dtype="fp8",
        tokens=1,
        c_max=2,
        zipf_exponent=1.3,
        layer_limit=1,
        gpu_profile="h100_sxm",
        brain_profile="lynxi_hp300_proj",
        bridge_fabric="pcie5_x16",
    )


def test_per_gpu_owner_case_is_valid():
    args = Namespace(
        **_common(),
        gpus=4,
        brain_nodes=1,
        batch=1,
        copies=4,
    )
    rows = run_pergpu_seed(args, 101)

    assert {row["variant"] for row in rows} == {
        "D0-shared-minimal",
        "D0-shared-replicated",
        "D0-per-gpu",
    }
    assert all(row["completion_s"] > 0 for row in rows)
    assert all(row["gpu_assignments"] + row["brain_assignments"] > 0
               for row in rows)


def test_m1_full_runtime_accounts_layers_and_transfers():
    args = Namespace(
        **_common(),
        hot_experts=2,
        brain_nodes=1,
        batch=1,
    )
    row = run_m1_seed(args, 101, 1)[0]

    assert row["used_moe_layers"] == 1
    assert row["assignments"] == 4
    assert row["m1_completion_s"] > 0
    assert row["dram_transfer_bytes"] > 0
