import pytest

from flowgpu.sim_v2.hardware_profiles import (
    load_fabric_profile,
    load_hardware_profile,
)
from flowgpu.sim_v2.schema import ValidationError


def test_h100_profile_uses_explicit_decimal_capacity():
    profile = load_hardware_profile("h100_sxm")
    assert profile.role == "gpu"
    assert profile.weight_capacity_bytes == 80_000_000_000
    assert profile.inter_chip_bandwidth_bytes_per_s == 450_000_000_000
    assert profile.inter_board_bandwidth_bytes_per_s == 50_000_000_000


def test_brain_profiles_preserve_estimate_and_projection():
    ka = load_hardware_profile("lynxi_ka200")
    hp = load_hardware_profile("lynxi_hp300_proj")
    assert ka.role == hp.role == "brain_sram"
    assert ka.weight_capacity_bytes == int(48_000_000 * 0.85)
    assert hp.weight_capacity_bytes == int(768_000_000 * 0.85)
    assert ka.dram_bandwidth_bytes_per_s == 68_000_000_000
    assert hp.dram_bandwidth_bytes_per_s == 200_000_000_000


def test_fabric_profile_and_unknown_device():
    bridge = load_fabric_profile("pcie5_x16")
    assert bridge.kind == "bridge"
    assert bridge.bandwidth_bytes_per_s == 64_000_000_000
    assert bridge.efficiency == 0.8
    with pytest.raises(ValidationError, match="unknown fabric"):
        load_fabric_profile("not-a-fabric")
