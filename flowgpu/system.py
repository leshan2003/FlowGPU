"""System assembly: device pools, parallelism, and the fabrics between them.

A :class:`System` is what you actually run a workload on.  It is a set of
named :class:`DevicePool` objects -- e.g. ``gpu`` (24 x Iluvatar BI-V100 across
3 servers) and ``brain`` (192 x Lynxi chips across 3 racks) -- plus the links
that join them.

Capacity checking is done here and it is not advisory: if a pool's per-device
share of the weights it has been assigned does not fit in that device's
memory, the system reports a spill fraction and the cost model pays for it.
That is how the simulator refuses to let a 230 MB Groq chip pretend to hold a
37 GB expert stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .hardware.base import Device, Residency
from .hardware.interconnect import Link, make_link
from .hardware.registry import make_device
from .units import fmt_bytes, fmt_bw, parse


@dataclass
class Parallelism:
    tp: int = 1          # tensor parallel   (splits every GEMM)
    pp: int = 1          # pipeline parallel (splits layers)
    ep: int = 1          # expert parallel   (splits MoE experts)
    dp: int = 1          # data parallel     (replicates everything)

    @property
    def shard(self) -> int:
        """Devices cooperating on a single op (TP x EP within one stage)."""
        return max(1, self.tp * self.ep)

    @property
    def world(self) -> int:
        return max(1, self.tp * self.pp * self.ep * self.dp)


@dataclass
class DevicePool:
    name: str
    device: Device
    count: int = 1
    parallel: Parallelism = field(default_factory=Parallelism)
    # fabrics
    scaleup: Link | None = None      # within a board / node
    scaleout: Link | None = None     # across boards / nodes
    devices_per_board: int = 8
    # populated by :meth:`System.plan`
    residency: Residency | None = None
    weight_bytes: float = 0.0
    kv_bytes: float = 0.0
    power_cap: float = 0.0           # W, whole pool
    # acquisition cost, for equal-investment comparisons.  The reference
    # claim is "vs a pure-GPU cluster of equivalent investment", so without
    # prices that comparison cannot be made at all.
    price_per_device: float = 0.0    # currency units
    # W per board for CPU / NIC / fans / PSU loss.  A rack of 8-chip
    # brain-chip cards does not carry a 400 W x86 host per card, so this is
    # per-pool rather than a single system-wide number.
    host_overhead: float = 0.0

    # ------------------------------------------------------------------
    @property
    def stage_devices(self) -> int:
        """Devices that cooperate on one op (one PP stage, one DP replica)."""
        p = self.parallel
        return max(1, self.count // max(1, p.dp * p.pp))

    @property
    def replicas(self) -> int:
        return max(1, self.parallel.dp)

    @property
    def n_boards(self) -> int:
        return max(1, math.ceil(self.count / max(1, self.devices_per_board)))

    @property
    def total_capacity(self) -> float:
        return self.device.capacity * self.count

    @property
    def peak_flops(self) -> float:
        return self.device.peak_flops * self.count

    @property
    def tdp(self) -> float:
        return self.device.energy.tdp * self.count

    @property
    def capex(self) -> float:
        return self.price_per_device * self.count

    @property
    def host_power(self) -> float:
        return self.n_boards * self.host_overhead

    def link_for(self, n_peers: int) -> Link:
        """Pick the fabric a collective of ``n_peers`` ranks actually uses."""
        if n_peers <= self.devices_per_board and self.scaleup is not None:
            return self.scaleup
        return self.scaleout or self.scaleup

    # ------------------------------------------------------------------
    def plan(self, weight_bytes: float, kv_bytes: float,
             act_bytes: float = 0.0) -> Residency:
        """Assign this pool's weight share and check it fits."""
        self.weight_bytes = weight_bytes
        self.kv_bytes = kv_bytes
        per_dev_w = weight_bytes / max(1, self.stage_devices)
        per_dev_kv = kv_bytes / max(1, self.count // self.replicas)
        self.residency = self.device.plan_residency(
            per_dev_w, per_dev_kv, act_bytes / max(1, self.stage_devices))
        return self.residency

    def summary(self) -> str:  # pragma: no cover
        p = self.parallel
        lines = [f"pool '{self.name}': {self.count} x {self.device.name} "
                 f"({self.device.kind})",
                 f"  parallel tp={p.tp} pp={p.pp} ep={p.ep} dp={p.dp} "
                 f"-> {self.stage_devices} devices/stage",
                 f"  capacity {fmt_bytes(self.total_capacity)}  "
                 f"peak {self.peak_flops/1e12:.1f} TFLOPS  TDP {self.tdp:.0f} W"]
        if self.scaleup:
            lines.append(f"  scale-up  {self.scaleup.name} "
                         f"{fmt_bw(self.scaleup.eff_bw)} "
                         f"@{self.scaleup.latency*1e6:.1f} us")
        if self.scaleout:
            lines.append(f"  scale-out {self.scaleout.name} "
                         f"{fmt_bw(self.scaleout.eff_bw)} "
                         f"@{self.scaleout.latency*1e6:.1f} us")
        if self.residency:
            r = self.residency
            lines.append(f"  residency {r.weight_level_frac} "
                         f"spill={r.spill_frac*100:.1f}%")
        return "\n".join(lines)


@dataclass
class System:
    name: str
    pools: dict = field(default_factory=dict)
    # bridge[(a, b)] = Link joining pool a and pool b
    bridges: dict = field(default_factory=dict)
    host_overhead: float = 0.0        # W, CPU/NIC/fans/PSU losses per node
    pue: float = 1.0                  # datacentre overhead multiplier
    notes: str = ""

    # ------------------------------------------------------------------
    def add_pool(self, pool: DevicePool) -> DevicePool:
        self.pools[pool.name] = pool
        return pool

    def bridge(self, a: str, b: str) -> Link | None:
        return self.bridges.get((a, b)) or self.bridges.get((b, a))

    def set_bridge(self, a: str, b: str, link: Link):
        self.bridges[(a, b)] = link

    @property
    def host_power(self) -> float:
        """Per-pool host overhead, falling back to the system-wide default."""
        tot = 0.0
        for p in self.pools.values():
            tot += p.n_boards * (p.host_overhead or self.host_overhead)
        return tot

    @property
    def total_tdp(self) -> float:
        dev = sum(p.tdp for p in self.pools.values())
        return (dev + self.host_power) * self.pue

    @property
    def total_capex(self) -> float:
        return sum(p.capex for p in self.pools.values())

    @property
    def total_capacity(self) -> float:
        return sum(p.total_capacity for p in self.pools.values())

    @property
    def total_devices(self) -> int:
        return sum(p.count for p in self.pools.values())

    def peak_flops(self, dtype: str = "bf16") -> float:
        return sum(p.device.compute.peak_for(dtype) * p.count
                   for p in self.pools.values())

    # ------------------------------------------------------------------
    def summary(self) -> str:  # pragma: no cover
        lines = [f"System '{self.name}'  {self.total_devices} devices  "
                 f"TDP {self.total_tdp/1000:.2f} kW  "
                 f"capacity {fmt_bytes(self.total_capacity)}"]
        for p in self.pools.values():
            lines.append(p.summary())
        for (a, b), lk in self.bridges.items():
            lines.append(f"bridge {a} <-> {b}: {lk.name} "
                         f"{fmt_bw(lk.eff_bw)} @{lk.latency*1e6:.2f} us")
        if self.notes:
            lines.append(self.notes)
        return "\n".join(lines)


# =========================================================================
# builders
# =========================================================================
def build_pool(name: str, cfg: dict) -> DevicePool:
    dev = make_device(cfg["device"], cfg.get("overrides"))
    par = Parallelism(**{k: int(v) for k, v in cfg.get("parallel", {}).items()})
    count = int(cfg.get("count", par.world))

    dpb = int(cfg.get("devices_per_board",
                      cfg.get("devices_per_node",
                              dev.extra.get("chips_per_board", 8))))
    if dev.kind == "dataflow":
        dpb = int(cfg.get("devices_per_board",
                          getattr(dev, "chips_per_board", 8)))

    def _mk(key, default_world):
        spec = cfg.get(key)
        if spec is None:
            dl = dev.links.get("scaleup" if key == "scaleup" else "scaleout")
            if dl is None:
                return None
            return Link(name=dl.get("name", key),
                        bandwidth=parse(dl["bandwidth"]),
                        latency=parse(dl.get("latency", "2 us")),
                        efficiency=float(dl.get("efficiency", 0.85)),
                        energy_per_byte=parse(dl.get("energy_per_byte",
                                                     "10 pJ")),
                        world=int(dl.get("world", default_world)),
                        kind="inter_chip" if key == "scaleup"
                             else "inter_board")
        if isinstance(spec, str):
            return make_link(spec, world=default_world)
        return make_link(spec.get("fabric", "ib_ndr"),
                         world=int(spec.get("world", default_world)),
                         overrides={k: v for k, v in spec.items()
                                    if k != "fabric" and k != "world"})

    pool = DevicePool(
        name=name, device=dev, count=count, parallel=par,
        scaleup=_mk("scaleup", min(count, dpb)),
        scaleout=_mk("scaleout", max(1, count // max(1, dpb))),
        devices_per_board=dpb,
        price_per_device=float(cfg.get("price_per_device", 0.0)),
        host_overhead=parse(cfg.get(
            "host_overhead", "400 W" if dev.kind == "gpu" else "120 W")),
    )
    return pool


def build_system(cfg: dict) -> System:
    sysm = System(
        name=cfg.get("name", "system"),
        host_overhead=parse(cfg.get("host_overhead", "250 W")),
        pue=float(cfg.get("pue", 1.0)),
        notes=cfg.get("notes", ""),
    )
    for pname, pcfg in cfg["pools"].items():
        sysm.add_pool(build_pool(pname, pcfg))
    for b in cfg.get("bridges", []):
        a, bb = b["between"]
        lk = make_link(b.get("fabric", "pcie5_x16"),
                       world=int(b.get("world", 2)),
                       overrides={k: v for k, v in b.items()
                                  if k not in ("between", "fabric", "world")})
        # aggregate: `lanes` parallel links between the two pools
        lanes = int(b.get("lanes", 1))
        if lanes > 1:
            lk.bandwidth *= lanes
            lk.name = f"{lk.name}x{lanes}"
        sysm.set_bridge(a, bb, lk)
    return sysm


def homogeneous(device: str, count: int, tp: int = 0, pp: int = 1,
                ep: int = 1, dp: int = 1, name: str | None = None,
                **kw) -> System:
    """Convenience: a single-pool system (the pure-GPU baseline)."""
    tp = tp or count // (pp * ep * dp)
    cfg = dict(
        name=name or f"{device}x{count}",
        pools={"main": dict(device=device, count=count,
                            parallel=dict(tp=tp, pp=pp, ep=ep, dp=dp), **kw)},
        host_overhead=kw.pop("host_overhead", "250 W"),
    )
    return build_system(cfg)
