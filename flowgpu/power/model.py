"""Energy-model construction and TDP calibration.

Two-step, so that neither literature nor vendor marketing can dominate:

1. **Apportionment** comes from the technology table (:mod:`.tech`) or, when
   available, from measured Synopsys DC + PrimeTime PX runs (:mod:`.eda`).
   This fixes the *ratios*: how much a bf16 MAC costs relative to an SRAM byte
   relative to an HBM byte relative to a NoC hop.
2. **Absolute scale** is pinned to the device's published TDP.  We define a
   reference operating point -- a heavy GEMM kernel running at
   ``util_compute`` of peak FLOPS while pulling ``util_bw`` of peak main-memory
   bandwidth -- and scale the dynamic coefficients so the modelled power at
   that point equals TDP minus static leakage.

Step 2 matters more than it looks.  Raw Horowitz numbers scaled to 4 nm would
predict an H100 doing dense bf16 at ~200 W, which is wrong by 3x: real chips
burn a large fraction of their power on clock distribution, register-file
traffic, control and data movement that a per-FLOP number ignores.  Anchoring
to TDP absorbs all of that, and keeps the *comparison* between architectures
honest because both sides get the same treatment.

The output of an over-TDP check is a warning, not silent clipping: if the
simulator computes a device drawing more than its TDP you will see it.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

from ..hardware.base import Device, EnergyModel
from ..hardware.dataflow import DataflowDevice
from ..hardware.gpu import GPUDevice
from . import tech


@dataclass
class CalibrationPoint:
    """The operating point at which modelled power is forced to equal TDP."""
    util_compute: float = 0.65     # fraction of peak FLOPS
    util_bw: float = 0.45          # fraction of peak main-memory bandwidth
    util_sram: float = 0.30        # fraction of peak on-chip SRAM bandwidth
    util_noc: float = 0.20
    dtype: str = "bf16"
    static_frac: float = 0.22      # leakage + clock tree + always-on, / TDP


_over: list = []          # devices whose off-chip interface exceeds their TDP

DEFAULT_POINT = CalibrationPoint()
# Dataflow chips run statically-scheduled kernels that keep on-chip memory
# saturated; their reference point is compute-light and SRAM-heavy.
DATAFLOW_POINT = CalibrationPoint(util_compute=0.55, util_bw=0.25,
                                  util_sram=0.60, util_noc=0.35,
                                  static_frac=0.18)


def base_coefficients(dev: Device) -> tuple:
    """Uncalibrated (j_per_flop, j_per_byte, j_per_byte_noc) from technology."""
    node = dev.process_nm or 7.0
    ls = tech.logic_scale(node)
    jpf = {dt: e * ls for dt, e in tech.REF_J_PER_FLOP.items()}

    jpb = {}
    for lvl in dev.memory:
        n = lvl.name
        if n in ("hbm", "dram"):
            kind = tech.dram_kind_for(dev)
            jpb[n] = tech.DRAM_J_PER_BYTE[kind]
        elif n == "regfile":
            jpb[n] = tech.sram_j_per_byte(
                lvl.per_unit_capacity or (lvl.capacity / max(dev.n_units, 1)),
                node) * 0.35          # register files are far cheaper per byte
        else:
            per_array = lvl.per_unit_capacity or (
                lvl.capacity / max(dev.n_units, 1) if n in ("sram", "smem")
                else lvl.capacity / 32.0)   # L2 is banked
            jpb[n] = tech.sram_j_per_byte(per_array, node)
    return jpf, jpb, tech.noc_j_per_byte_hop(node)


def calibrate(dev: Device, point: CalibrationPoint | None = None,
              eda: dict | None = None, verbose: bool = False) -> EnergyModel:
    """Attach a TDP-anchored energy model to ``dev`` and return it."""
    if point is None:
        point = DATAFLOW_POINT if dev.kind == "dataflow" else DEFAULT_POINT

    jpf, jpb, jnoc = base_coefficients(dev)

    # --- optional EDA-measured ratio override ---------------------------
    source = f"tech@{dev.process_nm or 7:g}nm + TDP"
    if eda:
        jpf, jpb, jnoc, source = _apply_eda_ratios(dev, jpf, jpb, jnoc, eda)

    # --- honour any explicit per-device overrides from the registry -----
    jpf.update(dev.energy.j_per_flop or {})
    jpb.update(dev.energy.j_per_byte or {})

    tdp = dev.energy.tdp
    if tdp <= 0:
        dev.energy = EnergyModel(j_per_flop=jpf, j_per_byte=jpb,
                                 j_per_byte_noc=jnoc, static_power=0.0,
                                 tdp=0.0, source=source + " (no TDP)")
        _attach_levels(dev, jpb)
        return dev.energy

    static = point.static_frac * tdp
    budget = tdp - static

    # modelled dynamic power at the calibration point, before scaling
    peak = dev.compute.peak_for(point.dtype)
    p_compute = peak * point.util_compute * jpf.get(point.dtype, 0.0)

    p_mem = 0.0
    for lvl in dev.memory:
        u = point.util_bw if lvl.name in ("hbm", "dram") else point.util_sram
        if lvl.name in ("regfile", "smem"):
            u = point.util_sram
        p_mem += lvl.bandwidth * u * jpb.get(lvl.name, 0.0)

    p_noc = 0.0
    if isinstance(dev, DataflowDevice) and dev.n_cores > 1:
        bis = dev.noc.bisection_bandwidth or dev.noc.link_bandwidth
        p_noc = bis * point.util_noc * jnoc

    # Off-chip DRAM/HBM energy per byte is set by the interface physics and is
    # one of the better-characterised numbers in the literature (~3.5 pJ/bit
    # for HBM3).  Scaling it to close a TDP gap would be wrong: the excess a
    # real chip burns over a naive per-FLOP model goes into clock trees,
    # register files, control and on-die data movement, not into the DRAM
    # PHY.  So the off-chip term is held fixed and only on-die terms scale.
    p_fixed = 0.0
    for lvl in dev.memory:
        if lvl.name in ("hbm", "dram"):
            u = point.util_bw
            p_fixed += lvl.bandwidth * u * jpb.get(lvl.name, 0.0)
    p_model = p_compute + p_mem + p_noc - p_fixed
    budget_scalable = budget - p_fixed
    if budget_scalable <= 0:
        # the DRAM interface alone would exceed the TDP at this operating
        # point -- report it rather than producing a negative scale
        scale = 1.0
        _over.append(f"{dev.name}: DRAM interface alone needs "
                     f"{p_fixed:.0f} W of a {tdp:.0f} W budget at the "
                     f"calibration point ({point.util_bw*100:.0f}% of peak "
                     f"bandwidth); on-die coefficients left unscaled")
    else:
        scale = budget_scalable / p_model if p_model > 0 else 1.0

    if verbose:                                        # pragma: no cover
        print(f"[calib] {dev.name}: TDP {tdp:.0f} W  static {static:.0f} W  "
              f"off-chip {p_fixed:.0f} W (fixed)  on-die model {p_model:.1f} W"
              f" -> scale {scale:.3f}  (compute {p_compute:.1f}, "
              f"mem {p_mem:.1f}, noc {p_noc:.1f})")

    jpf = {k: v * scale for k, v in jpf.items()}
    jpb = {k: (v if k in ("hbm", "dram") else v * scale)
           for k, v in jpb.items()}
    jnoc *= scale

    em = EnergyModel(
        j_per_flop=jpf, j_per_byte=jpb, j_per_byte_noc=jnoc,
        j_per_byte_link=dev.energy.j_per_byte_link,
        static_power=static, tdp=tdp,
        idle_frac=point.static_frac, source=source)
    em.calibration_scale = scale
    dev.energy = em
    _attach_levels(dev, jpb)
    return em


def _attach_levels(dev: Device, jpb: dict):
    for lvl in dev.memory:
        if not lvl.energy_per_byte:
            lvl.energy_per_byte = jpb.get(lvl.name, 0.0)
    if isinstance(dev, DataflowDevice) and not dev.noc.energy_per_byte_hop:
        dev.noc.energy_per_byte_hop = dev.energy.j_per_byte_noc


def _apply_eda_ratios(dev, jpf, jpb, jnoc, eda):
    """Replace modelled ratios with PrimeTime PX measurements.

    The EDA flow synthesises the three primitives that dominate the energy
    budget -- a MAC datapath, an SRAM-fed PE tile, and a NoC router -- to a
    real standard-cell library and measures switching power with realistic
    activity.  The *node* of that library is usually not the node of the chip
    being modelled, so what we transfer is the ratio
    ``E_sram_byte / E_mac_flop`` and ``E_noc_byte / E_mac_flop``, which is far
    more stable across nodes than any absolute value.
    """
    r_sram = eda.get("ratio_sram_byte_per_flop")
    r_noc = eda.get("ratio_noc_byte_per_flop")
    dt = eda.get("dtype", "int8")
    anchor = jpf.get(dt)
    if anchor is None or not r_sram:
        return jpf, jpb, jnoc, "tech (EDA ratios unusable)"

    # scale the MAC anchor by the measured relative cost of this dtype
    if eda.get("relative_flop_energy"):
        base = eda["relative_flop_energy"]
        ref = base.get(dt, 1.0)
        for k, v in base.items():
            if k in jpf:
                jpf[k] = anchor * v / ref

    # The measured SRAM ratio is for a small register-file array; read energy
    # per byte grows strongly with array size, so transfer the ratio at the
    # measured size and re-scale to each level's actual array using the
    # technology size curve.  Applying it flat would understate a 2 MB bank
    # by an order of magnitude.
    ref_bytes = float(eda.get("array_bytes", 512))
    node = dev.process_nm or 7.0
    ref_e = tech.sram_j_per_byte(ref_bytes, node)
    for name in ("sram", "smem", "regfile", "l2"):
        lvl = dev.level(name)
        if lvl is None or name not in jpb:
            continue
        # register files are structurally cheaper per byte than an SRAM
        # macro of the same size: no sense amps, no long bitlines, operands
        # wired straight into the datapath.  Same discount the technology
        # path applies, so the two sources stay comparable.
        discount = 1.0
        if name == "regfile":
            size = lvl.per_unit_capacity or ref_bytes
            discount = 0.35
        elif name == "l2":
            size = lvl.capacity / 32.0                # banked
        else:
            size = lvl.per_unit_capacity or (
                lvl.capacity / max(dev.n_units, 1))
        size_scale = tech.sram_j_per_byte(size, node) / max(ref_e, 1e-30)
        jpb[name] = anchor * r_sram * size_scale * discount
    if r_noc:
        jnoc = anchor * r_noc
    return jpf, jpb, jnoc, f"eda:{eda.get('run_id', 'unknown')} + TDP"


# =========================================================================
def calibrate_system(system, eda_db: dict | None = None, verbose=False):
    """Calibrate every distinct device in a system."""
    done = {}
    for pool in system.pools.values():
        key = pool.device.name
        if key in done:
            continue
        eda = None
        if eda_db:
            eda = eda_db.get(pool.device.kind) or eda_db.get("default")
        calibrate(pool.device, eda=eda, verbose=verbose)
        done[key] = pool.device.energy
    return done


def load_eda_db(path: str | None = None) -> dict | None:
    """Load EDA-measured coefficients written by :mod:`flowgpu.power.eda`."""
    path = path or os.environ.get("FLOWGPU_EDA_DB")
    if not path:
        here = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        cand = os.path.join(here, "eda", "results", "energy_db.json")
        path = cand if os.path.exists(cand) else None
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# =========================================================================
def idle_energy(system, wall_time: float, busy_by_pool: dict) -> dict:
    """Energy burnt by devices that were *not* doing useful work.

    A pool that sits idle for 90% of a decode step still draws leakage and
    clock power.  In a heterogeneous system this is a real and often decisive
    cost: an offload architecture that leaves the GPU idle 70% of the time is
    not saving 70% of the GPU's power.
    """
    out = {}
    for name, pool in system.pools.items():
        busy = min(busy_by_pool.get(name, 0.0), wall_time)
        idle = max(0.0, wall_time - busy)
        p_idle = pool.device.energy.static_power or (
            pool.device.energy.idle_frac * pool.device.energy.tdp)
        out[name] = p_idle * idle * pool.count
    return out


def host_energy(system, wall_time: float) -> float:
    return system.host_power * wall_time
