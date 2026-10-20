"""Tests for the EDA-calibrated power path.

These only run if `eda/results/energy_db.json` exists (i.e. the Synopsys flow
has been executed).  They check that the measured path stays physically
sensible and, crucially, that the study's conclusions do not depend on which
calibration source is used.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowgpu.hardware.registry import make_device
from flowgpu.power import calibrate, load_eda_db

DB = load_eda_db()
HAVE = DB is not None


def _skip():
    print("  SKIP  (no eda/results/energy_db.json -- run `flowgpu eda`)")


def test_eda_db_is_self_consistent():
    if not HAVE:
        return _skip()
    a = DB["default"]
    m = a["measured"]
    assert m["j_per_flop_int8"] > 0
    assert m["j_per_flop_16b"] > m["j_per_flop_int8"], \
        "a 16-bit MAC must cost more than an 8-bit one"
    r = m["j_per_flop_16b"] / m["j_per_flop_int8"]
    # multiplier energy scales ~ operand-width^2, so doubling width is ~4x
    assert 2.5 < r < 6.0, f"16b/8b MAC energy ratio {r:.2f} is not ~4x"
    assert a["ratio_sram_byte_per_flop"] > 0
    assert a["flop_sram_factor"] >= 1.0


def test_offchip_dram_energy_is_not_rescaled():
    """HBM per-bit energy is interface physics, not a free parameter."""
    for name, lo, hi in (("h100_sxm", 24, 32),        # HBM3 ~3.5 pJ/bit
                         ("a100_80gb_sxm", 26, 34),   # HBM2e ~3.7 pJ/bit
                         ("b200_sxm", 22, 30)):       # HBM3e ~3.2 pJ/bit
        for eda in (None, (DB or {}).get("default")):
            d = make_device(name)
            e = calibrate(d, eda=eda)
            pj = e.j_per_byte["hbm"] * 1e12
            assert lo <= pj <= hi, (name, pj, "pJ/byte out of range")


def test_sram_beats_dram_per_byte_under_both_calibrations():
    """The mechanism the whole study rests on must not be an artefact of
    which energy source was used."""
    ratios = []
    for eda in (None, (DB or {}).get("default")):
        gpu = make_device("h100_sxm")
        calibrate(gpu, eda=eda)
        df = make_device("groq_lpu_v1")
        calibrate(df, eda=eda)
        r = gpu.energy.j_per_byte["hbm"] / df.energy.j_per_byte["sram"]
        assert r > 3, f"SRAM/DRAM energy advantage only {r:.1f}x"
        ratios.append(r)
    if HAVE:
        # both methods must agree to within a factor of 3 on the size of
        # the advantage, or the conclusion is calibration-dependent
        assert max(ratios) / min(ratios) < 3.0, ratios


def test_eda_sram_ratio_is_scaled_by_array_size():
    """A 512 B register file and a 2 MB bank must not get the same pJ/byte."""
    if not HAVE:
        return _skip()
    d = make_device("h100_sxm")
    calibrate(d, eda=DB["default"])
    j = d.energy.j_per_byte
    assert j["regfile"] < j["smem"] < j["l2"], j


def _main():
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    npass = nfail = 0
    for name, fn in fns:
        try:
            fn()
            npass += 1
            print(f"  PASS  {name}")
        except Exception as e:                       # noqa: BLE001
            nfail += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{npass} passed, {nfail} failed, {len(fns)} total")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(_main())
