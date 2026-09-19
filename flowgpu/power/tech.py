"""Technology-node energy table.

Absolute anchors are the Horowitz ISSCC-2014 45 nm numbers, which remain the
most-cited energy-per-operation dataset in the architecture literature, plus
published per-bit figures for modern DRAM interfaces.  Node scaling factors
come from ITRS/IRDS energy-per-transition trends; note that SRAM and wires
scale *much worse* than logic below 7 nm, which is modelled separately -- that
asymmetry is precisely why on-chip-memory architectures have been gaining
ground.

These give the *apportionment* between compute and each memory level.  The
*absolute* scale is then pinned to each device's published TDP by
:func:`flowgpu.power.model.calibrate`, so the numbers can never drift far from
a vendor's own power envelope.  Optionally the compute/SRAM/NoC ratios are
replaced by measured values from the Synopsys DC + PrimeTime PX flow in
:mod:`flowgpu.power.eda`.
"""

from __future__ import annotations

PJ = 1e-12

# --- 45 nm reference energies (Horowitz, ISSCC 2014 plenary) -------------
REF_NODE = 45.0

# energy per FLOP (half a MAC) at 45 nm, joules
REF_J_PER_FLOP = {
    "fp64": 10.0 * PJ,
    "fp32": 2.3 * PJ,       # 0.9 pJ add + 3.7 pJ mult, /2
    "tf32": 1.6 * PJ,
    "bf16": 0.75 * PJ,      # 0.4 pJ add + 1.1 pJ mult, /2
    "fp16": 0.75 * PJ,
    "fp8": 0.30 * PJ,
    "int8": 0.115 * PJ,     # 0.03 pJ add + 0.2 pJ mult, /2
    "fp4": 0.09 * PJ,
    "mxfp4": 0.10 * PJ,
    "int4": 0.045 * PJ,
    "spike": 0.02 * PJ,     # accumulate-only, no multiplier
}

# logic energy scaling relative to 45 nm
LOGIC_SCALE = {
    180: 12.0, 130: 7.0, 90: 3.2, 65: 1.8, 45: 1.0, 40: 0.85,
    32: 0.62, 28: 0.50, 22: 0.38, 16: 0.29, 14: 0.28, 12: 0.25,
    10: 0.20, 8: 0.17, 7: 0.15, 6: 0.125, 5: 0.11, 4: 0.098,
    3: 0.082, 2: 0.070,
}

# SRAM energy scaling -- flattens hard below 7 nm (SRAM bitcell scaling stall)
SRAM_SCALE = {
    180: 9.0, 130: 5.5, 90: 2.8, 65: 1.7, 45: 1.0, 40: 0.88,
    32: 0.70, 28: 0.60, 22: 0.48, 16: 0.38, 14: 0.37, 12: 0.34,
    10: 0.30, 8: 0.27, 7: 0.25, 6: 0.23, 5: 0.22, 4: 0.21,
    3: 0.20, 2: 0.19,
}

# --- SRAM read energy per byte at 45 nm, as a function of array size -----
# Horowitz: 8 KB cache read = 10 pJ / 32 bit -> 2.5 pJ/B
#           32 KB            = 20 pJ / 32 bit -> 5.0 pJ/B
#           1 MB             = 100 pJ / 32 bit -> 25 pJ/B
_SRAM_REF = [
    (1 << 10, 1.2 * PJ),
    (8 << 10, 2.5 * PJ),
    (32 << 10, 5.0 * PJ),
    (256 << 10, 13.0 * PJ),
    (1 << 20, 25.0 * PJ),
    (8 << 20, 45.0 * PJ),
    (64 << 20, 80.0 * PJ),
]


def sram_j_per_byte(array_bytes: float, node_nm: float) -> float:
    """Energy per byte read from an SRAM array of the given size."""
    if array_bytes <= 0:
        array_bytes = 32 << 10
    import math
    xs = [a for a, _ in _SRAM_REF]
    ys = [e for _, e in _SRAM_REF]
    if array_bytes <= xs[0]:
        e = ys[0]
    elif array_bytes >= xs[-1]:
        # weak log growth beyond the table
        e = ys[-1] * (1.0 + 0.12 * math.log2(array_bytes / xs[-1]))
    else:
        for i in range(len(xs) - 1):
            if xs[i] <= array_bytes <= xs[i + 1]:
                t = (math.log2(array_bytes) - math.log2(xs[i])) / \
                    (math.log2(xs[i + 1]) - math.log2(xs[i]))
                e = ys[i] + t * (ys[i + 1] - ys[i])
                break
    return e * _interp(SRAM_SCALE, node_nm)


def logic_scale(node_nm: float) -> float:
    return _interp(LOGIC_SCALE, node_nm)


def _interp(table: dict, node_nm: float) -> float:
    if node_nm <= 0:
        return 0.15            # assume ~7 nm if unknown
    keys = sorted(table)
    if node_nm <= keys[0]:
        return table[keys[0]]
    if node_nm >= keys[-1]:
        return table[keys[-1]]
    import math
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        if a <= node_nm <= b:
            t = (math.log(node_nm) - math.log(a)) / (math.log(b) - math.log(a))
            return table[a] + t * (table[b] - table[a])
    return 0.15


# --- off-chip memory energy, joules per byte -----------------------------
# published per-bit figures x 8
DRAM_J_PER_BYTE = {
    "hbm2": 3.9e-12 * 8,
    "hbm2e": 3.7e-12 * 8,
    "hbm3": 3.5e-12 * 8,
    "hbm3e": 3.2e-12 * 8,
    "hbm4": 2.9e-12 * 8,
    "gddr6": 7.0e-12 * 8,
    "gddr6x": 7.5e-12 * 8,
    "lpddr4": 8.0e-12 * 8,
    "lpddr5": 5.0e-12 * 8,
    "lpddr5x": 4.2e-12 * 8,
    "ddr4": 18.0e-12 * 8,
    "ddr5": 12.0e-12 * 8,
}


def dram_kind_for(device) -> str:
    """Guess the DRAM technology from bandwidth, capacity and year."""
    lvl = device.level("hbm") or device.level("dram")
    if lvl is None:
        return "hbm3"
    bw = lvl.bandwidth
    yr = device.year or 2022
    name = (device.notes or "").lower()
    for k in DRAM_J_PER_BYTE:
        if k in name:
            return k
    if "gddr" in name:
        return "gddr6"
    if "lpddr" in name or bw < 400e9 and lvl.capacity > 32 * 2**30:
        return "lpddr5" if yr >= 2022 else "lpddr4"
    if bw >= 6e12:
        return "hbm3e"
    if bw >= 3e12:
        return "hbm3"
    if bw >= 1.5e12:
        return "hbm2e"
    if bw >= 700e9:
        return "hbm2"
    return "gddr6"


# --- interconnect energy, joules per byte --------------------------------
NOC_J_PER_BYTE_HOP = {           # per hop, per byte, by node
    28: 0.9e-12 * 8 / 8,         # ~0.9 pJ/bit/mm-class hop at 28 nm
    16: 0.6e-12,
    12: 0.5e-12,
    7: 0.35e-12,
    5: 0.3e-12,
    4: 0.28e-12,
}


def noc_j_per_byte_hop(node_nm: float) -> float:
    return _interp({k: v for k, v in NOC_J_PER_BYTE_HOP.items()}, node_nm)


SERDES_J_PER_BYTE = {
    "on_package": 0.5e-12 * 8,       # UCIe / chiplet
    "short_reach": 2.0e-12 * 8,      # NVLink / board trace
    "long_reach": 5.0e-12 * 8,       # cable, retimed
    "optical": 8.0e-12 * 8,
    "ethernet": 15.0e-12 * 8,        # incl. NIC + switch share
}
