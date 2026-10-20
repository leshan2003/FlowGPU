import pytest

from flowgpu.sim_v2.statistics import (
    HysteresisDriftDetector,
    normalized_l1,
    paired_block_bootstrap_ci,
    required_paired_blocks,
)
from flowgpu.sim_v2.schema import ValidationError


def test_normalized_l1_and_hysteresis_trigger():
    detector = HysteresisDriftDetector(
        threshold=0.2,
        release_threshold=0.05,
        min_dwell_windows=2,
    )
    assert normalized_l1({"a": 1}, {"b": 1}) == 1.0
    first = detector.observe(0, {"a": 1})
    second = detector.observe(1, {"b": 1})
    third = detector.observe(2, {"c": 1})
    assert not first.triggered
    assert not second.triggered
    assert third.triggered


def test_paired_block_bootstrap_and_power_floor():
    ci = paired_block_bootstrap_ci(
        [(1.0, 1.1), (2.0, 2.2), (3.0, 3.3)],
        resamples=2000,
        seed=4,
    )
    assert ci.ratio == pytest.approx(1.1)
    assert ci.low95 <= 1.1 <= ci.high95
    assert required_paired_blocks(0.1, 0.5) == 20
    assert required_paired_blocks(1.0, 0.1) > 20


def test_block_bootstrap_rejects_empty_or_bad_blocks():
    with pytest.raises(ValidationError, match="paired blocks"):
        paired_block_bootstrap_ci([])
    with pytest.raises(ValidationError, match="finite"):
        paired_block_bootstrap_ci([(1.0, -1.0)])
