"""Device abstractions shared by GPUs and many-core dataflow chips.

Cost model
----------
Each device prices an :class:`~flowgpu.workload.graph.Op` into an
:class:`OpCost` (seconds, joules, plus a breakdown).  The analytic core is a
*multi-level roofline*:

1. Decide which memory level serves each byte class (weights / activations /
   KV state).  This is the single most important modelling decision, and it is
   where GPUs and dataflow chips diverge: a GPU streams weights from HBM on
   every decode step, a dataflow chip with enough on-chip SRAM does not.
2. ``t_mem = max_L(bytes_at_L / bw_L)`` -- levels run concurrently, a single
   level's bandwidth is the serialising resource.
3. ``t_compute = flops / (peak * efficiency(op))`` with a tile/wave
   quantisation efficiency model, so small-batch decode is correctly penalised.
4. ``t = max(t_compute, t_mem, t_network) + fixed_overhead`` when the device
   can overlap (GPU async copy / dataflow pipelining), else the sum.

Energy is accounted per byte moved at each level plus per FLOP issued, with a
leakage/idle term integrated over the op duration.  Coefficients come from
:mod:`flowgpu.power` (EDA-calibrated ratios, literature-anchored absolutes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..units import dtype_bytes, parse
from ..workload import graph as G


# =========================================================================
# memory hierarchy
# =========================================================================
@dataclass
class MemoryLevel:
    name: str
    capacity: float           # bytes, aggregate over the whole device
    bandwidth: float          # bytes/s, aggregate over the whole device
    latency: float = 0.0      # seconds, unloaded access latency
    energy_per_byte: float = 0.0   # J/B for a read or write at this level
    per_unit_capacity: float = 0.0  # bytes per SM/core (0 = shared pool)
    per_unit_bandwidth: float = 0.0

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "MemoryLevel":
        return cls(
            name=name,
            capacity=parse(d.get("capacity", 0)),
            bandwidth=parse(d.get("bandwidth", 0)),
            latency=parse(d.get("latency", 0)),
            energy_per_byte=parse(d.get("energy_per_byte", 0)),
            per_unit_capacity=parse(d.get("per_unit_capacity", 0)),
            per_unit_bandwidth=parse(d.get("per_unit_bandwidth", 0)),
        )


# =========================================================================
# compute
# =========================================================================
@dataclass
class ComputeSpec:
    """Peak math throughput, per dtype, for the whole device."""
    peak: dict = field(default_factory=dict)     # dtype -> FLOP/s (dense)
    vector_peak: float = 0.0                     # FLOP/s for non-GEMM work
    clock: float = 1.0e9

    def peak_for(self, dtype: str) -> float:
        d = dtype.lower()
        if d in self.peak:
            return self.peak[d]
        # fall back down the precision ladder
        order = ["fp4", "mxfp4", "int4", "fp8", "int8", "fp16", "bf16",
                 "tf32", "fp32", "fp64"]
        if d in order:
            i = order.index(d)
            for j in range(i, len(order)):
                if order[j] in self.peak:
                    return self.peak[order[j]]
            for j in range(i - 1, -1, -1):
                if order[j] in self.peak:
                    return self.peak[order[j]]
        return max(self.peak.values()) if self.peak else 0.0


# =========================================================================
# energy
# =========================================================================
@dataclass
class EnergyModel:
    """Energy coefficients.  All values in joules."""
    j_per_flop: dict = field(default_factory=dict)   # dtype -> J/FLOP (datapath)
    j_per_byte: dict = field(default_factory=dict)   # memory level -> J/B
    j_per_byte_noc: float = 0.0                      # J/B/hop on-chip network
    j_per_byte_link: float = 0.0                     # J/B off-chip serdes
    static_power: float = 0.0                        # W, leakage + always-on
    tdp: float = 0.0                                 # W, board power cap
    # fraction of TDP consumed when idle but powered
    idle_frac: float = 0.25
    source: str = "literature"                       # or 'eda:<run-id>'

    def flop_energy(self, flops: float, dtype: str) -> float:
        d = dtype.lower()
        if d in self.j_per_flop:
            return flops * self.j_per_flop[d]
        if self.j_per_flop:
            # scale by relative element width vs the nearest known dtype
            ref = next(iter(self.j_per_flop))
            scale = dtype_bytes(d) / dtype_bytes(ref)
            return flops * self.j_per_flop[ref] * (scale ** 1.4)
        return 0.0

    def byte_energy(self, nbytes: float, level: str) -> float:
        return nbytes * self.j_per_byte.get(level, 0.0)


# =========================================================================
# results
# =========================================================================
@dataclass
class OpCost:
    time: float = 0.0
    energy: float = 0.0
    t_compute: float = 0.0
    t_mem: float = 0.0
    t_net: float = 0.0
    bound: str = "compute"
    util: float = 0.0                     # achieved fraction of peak FLOPS
    bytes_by_level: dict = field(default_factory=dict)
    energy_by_source: dict = field(default_factory=dict)
    notes: str = ""

    def __add__(self, other: "OpCost") -> "OpCost":
        out = OpCost(
            time=self.time + other.time,
            energy=self.energy + other.energy,
            t_compute=self.t_compute + other.t_compute,
            t_mem=self.t_mem + other.t_mem,
            t_net=self.t_net + other.t_net,
        )
        for d, s in ((out.bytes_by_level, self.bytes_by_level),
                     (out.bytes_by_level, other.bytes_by_level)):
            for k, v in s.items():
                d[k] = d.get(k, 0.0) + v
        for d, s in ((out.energy_by_source, self.energy_by_source),
                     (out.energy_by_source, other.energy_by_source)):
            for k, v in s.items():
                d[k] = d.get(k, 0.0) + v
        out.bound = max(("compute", out.t_compute), ("memory", out.t_mem),
                        ("network", out.t_net), key=lambda kv: kv[1])[0]
        return out


@dataclass
class Residency:
    """Where a device keeps each byte class for the currently mapped model.

    ``weight_level_frac`` maps memory-level name -> fraction of weight bytes
    served from that level.  A dataflow chip whose SRAM holds the whole shard
    gets ``{'sram': 1.0}``; a GPU gets ``{'hbm': 1.0}``; a chip that only fits
    60% of its shard gets ``{'sram': 0.6, 'dram': 0.4}``.
    """
    weight_level_frac: dict = field(default_factory=dict)
    act_level: str = "l2"
    kv_level: str = "hbm"
    resident_bytes: float = 0.0
    capacity_bytes: float = 0.0
    spill_frac: float = 0.0


# =========================================================================
# device
# =========================================================================
@dataclass
class Device:
    name: str
    kind: str                      # 'gpu' | 'dataflow'
    vendor: str = ""
    n_units: int = 1               # SMs (GPU) or cores (dataflow)
    compute: ComputeSpec = field(default_factory=ComputeSpec)
    memory: list = field(default_factory=list)      # MemoryLevel, fast -> slow
    energy: EnergyModel = field(default_factory=EnergyModel)
    process_nm: float = 0.0
    year: int = 0
    # fixed per-op overheads
    kernel_launch_overhead: float = 2.0e-6
    overlap_compute_memory: bool = True
    # scale-out links (populated by the system builder)
    links: dict = field(default_factory=dict)
    notes: str = ""
    extra: dict = field(default_factory=dict)

    # -- lookups ----------------------------------------------------------
    def level(self, name: str) -> MemoryLevel | None:
        for m in self.memory:
            if m.name == name:
                return m
        return None

    @property
    def main_memory(self) -> MemoryLevel:
        """Largest-capacity level -- where a model shard has to live."""
        return max(self.memory, key=lambda m: m.capacity)

    @property
    def capacity(self) -> float:
        return self.main_memory.capacity

    @property
    def peak_flops(self) -> float:
        return max(self.compute.peak.values()) if self.compute.peak else 0.0

    # -- to be provided by subclasses -------------------------------------
    def plan_residency(self, weight_bytes: float, kv_bytes: float,
                       act_bytes: float) -> Residency:
        raise NotImplementedError

    def efficiency(self, op: G.Op, batch_rows: int) -> float:
        raise NotImplementedError

    def cost_op(self, op: G.Op, res: Residency, shards: int = 1) -> OpCost:
        raise NotImplementedError

    # -- shared helpers ---------------------------------------------------
    def _memory_time(self, byte_map: dict) -> tuple:
        """byte_map: level-name -> bytes.  Returns (seconds, bottleneck)."""
        t, who = 0.0, "none"
        for lvl_name, nb in byte_map.items():
            if nb <= 0:
                continue
            lvl = self.level(lvl_name)
            bw = lvl.bandwidth if lvl else self.main_memory.bandwidth
            if bw <= 0:
                continue
            tt = nb / bw
            if tt > t:
                t, who = tt, lvl_name
        return t, who

    def _memory_energy(self, byte_map: dict) -> dict:
        out = {}
        for lvl_name, nb in byte_map.items():
            if nb <= 0:
                continue
            lvl = self.level(lvl_name)
            e = lvl.energy_per_byte if lvl else 0.0
            if not e:
                e = self.energy.j_per_byte.get(lvl_name, 0.0)
            out[f"mem:{lvl_name}"] = nb * e
        return out

    def summary(self) -> str:  # pragma: no cover
        from ..units import fmt_bytes, fmt_bw, fmt_flops
        lines = [f"{self.name}  [{self.kind}] {self.vendor} "
                 f"{self.process_nm:g}nm {self.year}"]
        lines.append(f"  units: {self.n_units}   clock: "
                     f"{self.compute.clock/1e9:.2f} GHz")
        for dt, p in sorted(self.compute.peak.items(),
                            key=lambda kv: -kv[1]):
            lines.append(f"  peak {dt:6s} {fmt_flops(p)}")
        for m in self.memory:
            lines.append(f"  mem  {m.name:8s} {fmt_bytes(m.capacity):>12s} "
                         f"@ {fmt_bw(m.bandwidth):>12s}  "
                         f"lat {m.latency*1e9:7.1f} ns")
        lines.append(f"  TDP {self.energy.tdp:.0f} W")
        if self.notes:
            lines.append(f"  {self.notes}")
        return "\n".join(lines)
