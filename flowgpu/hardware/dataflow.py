"""Brain-inspired many-core dataflow chip model.

Covers the whole spectrum the study needs:

* **Deterministic dataflow accelerators** -- Groq LPU (TSP), Graphcore IPU,
  SambaNova RDU, Cerebras WSE, Tesla Dojo.  Huge distributed SRAM, static
  compiler-generated schedules, no caches, no dynamic scheduling jitter.
* **Brain-inspired / near-memory many-core chips** -- Lynxi KA200, Tianjic /
  TianjicX, Darwin3.  Hybrid ANN+SNN cores, compute-in-memory style weight
  stationarity, mesh NoC with routed packets.
* **Pure neuromorphic** -- TrueNorth, Loihi 2, SpiNNaker 2, BrainScaleS 2.
  Event-driven, spike-sparse; modelled honestly (they cannot run a dense bf16
  transformer at scale, and the simulator says so rather than pretending).

Why this class differs from :class:`~flowgpu.hardware.gpu.GPUDevice`
--------------------------------------------------------------------
The decisive mechanism for decode-phase LLM inference is *where the weights
are read from*.  A GPU re-streams its weight shard from HBM every single token
step.  A dataflow chip that holds its shard in distributed core SRAM reads it
at aggregate on-chip bandwidth -- typically 20-100x higher and ~100x cheaper
per byte.  The cost of that is capacity: SRAM is ~1000x less dense than HBM,
so the model must be spread over many more chips, and inter-chip links become
the new bottleneck.  Both sides of that trade are modelled here.

Mapping model
-------------
A GEMM ``(M,K) x (K,N)`` is tiled over an ``R x C`` logical core grid: ``R``
splits the reduction dimension ``K``, ``C`` splits the output dimension ``N``.
The mapper picks ``(R, C)`` to minimise NoC traffic subject to per-core SRAM
capacity, which is exactly the decision a real dataflow compiler makes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..units import dtype_bytes, parse
from ..workload import graph as G
from .base import ComputeSpec, Device, EnergyModel, MemoryLevel, OpCost, Residency


@dataclass
class NoC:
    """On-chip network."""
    topology: str = "mesh2d"       # mesh2d | torus2d | crossbar | tree | wafer
    link_bandwidth: float = 0.0    # bytes/s per unidirectional link
    bisection_bandwidth: float = 0.0   # bytes/s, whole chip
    hop_latency: float = 5e-9
    flit_bytes: float = 32.0
    energy_per_byte_hop: float = 0.0
    multicast: bool = True         # hardware multicast tree for broadcasts

    def avg_hops(self, n_cores: int) -> float:
        if n_cores <= 1:
            return 0.0
        if self.topology == "crossbar":
            return 1.0
        side = math.sqrt(n_cores)
        if self.topology == "torus2d":
            return side / 2.0
        if self.topology == "tree":
            return 2.0 * math.log2(max(2.0, n_cores))
        # 2-D mesh average Manhattan distance
        return 2.0 * side / 3.0


@dataclass
class DataflowDevice(Device):
    kind: str = "dataflow"
    arch: str = "dataflow"         # dataflow | brain | snn | wafer
    # --- the six headline features -----------------------------------
    n_cores: int = 1               # (1) core count
    sram_per_core: float = 0.0     # (2) bytes of SRAM in each core
    flops_per_core: dict = field(default_factory=dict)  # (3) dtype -> FLOP/s
    noc: NoC = field(default_factory=NoC)               # (4) NoC bandwidth
    has_on_package_dram: bool = False                   # (5) DRAM presence
    inter_chip_bw: float = 0.0     # (6) bytes/s aggregate off-chip to peers
    inter_chip_latency: float = 1.0e-6
    inter_board_bw: float = 0.0    # (7) bytes/s aggregate board-to-board
    inter_board_latency: float = 5.0e-6
    chips_per_board: int = 1

    # --- micro-architecture ------------------------------------------
    sram_bw_per_core: float = 0.0  # bytes/s read bandwidth of one core's SRAM
    pe_rows: int = 16              # systolic / SIMD array shape inside a core
    pe_cols: int = 16
    pipeline_depth: int = 0        # cycles to fill the core pipeline (0=auto)
    achievable_peak: float = 0.85  # static schedules get close to peak
    vector_frac: float = 0.12
    jitter_sigma: float = 0.005    # statically scheduled -> near-deterministic
    kernel_launch_overhead: float = 2.0e-7   # no CPU-side launch
    # SNN-specific
    spiking: bool = False
    spike_rate: float = 0.05       # average fraction of neurons firing
    supported_dtypes: tuple = ("int8", "bf16", "fp16")

    # ------------------------------------------------------------------
    def __post_init__(self):
        if not self.sram_bw_per_core and self.sram_per_core:
            # default: one full PE-array operand fetch per cycle
            self.sram_bw_per_core = (self.pe_rows * self.pe_cols
                                     * 1.0 * self.compute.clock)
        if not self.pipeline_depth:
            self.pipeline_depth = self.pe_rows + self.pe_cols

    # ------------------------------------------------------------------
    def die_area_check(self) -> dict:
        """Is this much on-chip SRAM physically buildable at this node?

        SRAM bitcell area by node (um^2/bit, 6T high-density, incl. ~1.35x
        array + periphery overhead).  A monolithic reticle is ~800 mm^2, so a
        part claiming 1 GB of SRAM at 12 nm is not a chip -- it is a wish.
        Reported rather than enforced, because multi-die packages are real.
        """
        bitcell = {180: 4.0, 130: 2.2, 90: 1.0, 65: 0.52, 45: 0.25,
                   32: 0.15, 28: 0.127, 22: 0.092, 16: 0.070, 14: 0.064,
                   12: 0.055, 10: 0.042, 7: 0.027, 6: 0.024, 5: 0.021,
                   4: 0.021, 3: 0.019, 2: 0.018}
        node = self.process_nm or 7
        keys = sorted(bitcell)
        a = bitcell[min(keys, key=lambda k: abs(k - node))]
        bits = self.total_sram * 8
        area_mm2 = bits * a * 1.35 / 1e6
        reticle = 800.0
        return dict(sram_area_mm2=area_mm2,
                    dies_needed=area_mm2 / reticle,
                    plausible=area_mm2 < reticle * 0.65,
                    node_nm=node, bitcell_um2=a)

    # -- aggregate capacity / bandwidth --------------------------------
    @property
    def total_sram(self) -> float:
        return self.n_cores * self.sram_per_core

    @property
    def total_sram_bw(self) -> float:
        return self.n_cores * self.sram_bw_per_core

    # ------------------------------------------------------------------
    def plan_residency(self, weight_bytes: float, kv_bytes: float,
                       act_bytes: float) -> Residency:
        """Fill SRAM with weights first; spill the remainder to DRAM.

        Weight-stationarity is the entire point of the architecture, so the
        compiler pins as much of the shard as fits into core SRAM.  KV cache
        (which grows with context and cannot be statically pinned) goes to
        DRAM if the chip has any, otherwise it also competes for SRAM.
        """
        sram = self.level("sram")
        dram = self.level("dram") or self.level("hbm")
        sram_cap = sram.capacity if sram else self.total_sram
        # reserve room for activations and double-buffering
        usable = sram_cap * float(self.extra.get("sram_weight_frac", 0.85))

        frac = {}
        if weight_bytes <= 0:
            frac = {"sram": 1.0}
            spill = 0.0
        elif weight_bytes <= usable:
            frac = {"sram": 1.0}
            spill = 0.0
        elif dram is not None:
            r = usable / weight_bytes
            frac = {"sram": r, dram.name: 1.0 - r}
            spill = 1.0 - r
        else:
            # no DRAM: the shard simply does not fit -- flag it hard
            frac = {"sram": 1.0}
            spill = 1.0 - usable / weight_bytes

        kv_level = dram.name if dram is not None else "sram"
        cap = sram_cap + (dram.capacity if dram is not None else 0.0)
        return Residency(
            weight_level_frac=frac, act_level="sram", kv_level=kv_level,
            resident_bytes=weight_bytes + kv_bytes + act_bytes,
            capacity_bytes=cap, spill_frac=spill,
        )

    # ------------------------------------------------------------------
    def _grid(self, op: G.Op, n_cores: int, shards: int = 1) -> tuple:
        """Choose the (R, C) core-grid split that minimises NoC traffic.

        ``R`` splits the reduction dimension ``K`` (and therefore forces a
        partial-sum reduction across cores); ``C`` splits the output
        dimension ``N`` (free, but each column needs the whole activation).
        With hardware multicast the broadcast is cheap, so the mapper
        prefers ``R = 1`` whenever the weight tile fits -- which is exactly
        what a real dataflow compiler does.

        ``shards`` is the number of *chips* the op is already split across;
        each chip only holds ``1/shards`` of the weights, so the per-core
        tile shrinks accordingly.  Ignoring this made every multi-chip
        deployment look as if a single chip had to hold the whole layer.
        """
        ep = max(1, int(op.meta.get("ep", 1)))
        tp = max(1, shards // ep)
        groups = max(1.0, float(op.meta.get("groups", 1))) / ep

        K = max(1, int(op.k or 1))
        N = max(1.0, (op.n or 1) / tp)
        M = max(1, int(op.m or 1))
        eb = dtype_bytes(op.weights[0].dtype) if op.weights else 2.0
        cap = self.sram_per_core * float(
            self.extra.get("sram_weight_frac", 0.85))

        candidates = []
        r = 1
        while r <= n_cores:
            candidates.append(r)
            r *= 2
        if n_cores not in candidates:
            candidates.append(n_cores)

        best, best_traffic, best_fits = (1, n_cores), math.inf, False
        for r in candidates:
            c = max(1, n_cores // r)
            tile = (K / r) * (N / c) * eb * groups
            fits = tile <= cap
            bcast = M * K * eb * (1.0 if self.noc.multicast else c)
            reduce_ = M * N * eb * (r - 1) * 2.0
            traffic = bcast + reduce_
            # prefer a mapping that fits; among those, minimise traffic
            if (fits and not best_fits) or \
               (fits == best_fits and traffic < best_traffic):
                best, best_traffic, best_fits = (r, c), traffic, fits
        return best[0], best[1], best_traffic

    # ------------------------------------------------------------------
    def efficiency(self, op: G.Op, batch_rows: int, shards: int = 1) -> float:
        if op.kind in (G.ELEMENTWISE, G.NORM, G.SOFTMAX, G.EMBED, G.ROUTER,
                       G.A2A, G.ALLREDUCE):
            return self.vector_frac

        shards = max(1, int(shards))
        ep = max(1, int(op.meta.get("ep", 1)))
        tp = max(1, shards // ep)
        groups = max(1.0, float(op.meta.get("groups", 1)))
        M = max(1, int(op.m or batch_rows))
        N = max(1, int(op.n or 1))
        K = max(1, int(op.k or 1))
        if op.kind in (G.ATTENTION, G.BMM):
            M = max(1, M // shards)
        else:
            groups = max(1.0, groups / ep)
            N = max(1, N // tp)

        # weight-stationary systolic array: weights sit in the PEs, tokens
        # stream through the M dimension.  Utilisation is pipeline fill.
        # This is the structural difference from a GPU: a dataflow chip does
        # not pay a tile-quantisation penalty in M, only a pipeline-fill one,
        # so batch-1 decode hurts it far less.
        eff_pipe = M / (M + self.pipeline_depth)

        # tile quantisation against the PE array shape
        pk = math.ceil(K / self.pe_rows) * self.pe_rows
        pn = math.ceil(N / self.pe_cols) * self.pe_cols
        eff_tile = (K * N) / (pk * pn)

        # core-count quantisation: can the op fill every core?
        n_tiles = (math.ceil(K / self.pe_rows) * math.ceil(N / self.pe_cols)
                   * groups)
        cores_usable = min(self.n_cores, max(1, n_tiles))
        eff_cores = cores_usable / self.n_cores

        eff = self.achievable_peak * eff_pipe * eff_tile * eff_cores
        if op.kind in (G.ATTENTION, G.BMM):
            # KV is not weight-stationary: the array must be reloaded
            eff *= 0.45
        if self.spiking and op.kind in (G.GEMM, G.SPIKE):
            # event-driven: work scales with spike rate, but so does the
            # effective utilisation of a synchronous array
            eff *= 0.6
        return max(eff, 1e-4)

    # ------------------------------------------------------------------
    def _byte_map(self, op: G.Op, res: Residency, shards: int) -> dict:
        bm = {}
        wb = op.weight_bytes / max(1, shards)
        for lvl, frac in res.weight_level_frac.items():
            bm[lvl] = bm.get(lvl, 0.0) + wb * frac
        st = (op.state_read_bytes + op.state_write_bytes) / max(1, shards)
        bm[res.kv_level] = bm.get(res.kv_level, 0.0) + st
        bm["sram"] = bm.get("sram", 0.0) + op.act_bytes / max(1, shards)
        return bm

    # ------------------------------------------------------------------
    def cost_op(self, op: G.Op, res: Residency, shards: int = 1) -> OpCost:
        dt = op.meta.get("compute_dtype") or (
            op.weights[0].dtype if op.weights else "bf16")
        if dt not in self.supported_dtypes:
            # emulate at the nearest supported precision, with a penalty
            dt_native = self.supported_dtypes[0]
        else:
            dt_native = dt
        peak = self.compute.peak_for(dt_native)

        flops = op.flops / max(1, shards)
        if self.spiking and op.kind in (G.GEMM, G.SPIKE):
            # event-driven chips only do work for spikes that actually fire
            flops *= max(self.spike_rate, 1e-3)

        eff = self.efficiency(op, batch_rows=op.m or 1, shards=shards)
        t_compute = flops / (peak * eff) if peak > 0 else 0.0

        bm = self._byte_map(op, res, shards)
        t_mem, bottleneck = self._memory_time(bm)

        # --- NoC ---------------------------------------------------------
        t_net = 0.0
        noc_bytes = 0.0
        if op.kind in (G.GEMM, G.BMM, G.ATTENTION, G.CONV) and self.n_cores > 1:
            r, c, noc_bytes = self._grid(op, self.n_cores, shards)
            bw = self.noc.bisection_bandwidth or (
                self.noc.link_bandwidth * math.sqrt(self.n_cores))
            if bw > 0:
                t_net = noc_bytes / bw
            t_net += self.noc.avg_hops(self.n_cores) * self.noc.hop_latency
            op.meta.setdefault("grid", (r, c))
        elif self.n_cores > 1:
            noc_bytes = op.act_bytes / max(1, shards)
            bw = self.noc.bisection_bandwidth or self.noc.link_bandwidth
            if bw > 0:
                t_net = noc_bytes / bw

        t = max(t_compute, t_mem, t_net) + self.kernel_launch_overhead

        # ---- energy ----
        esrc = self._memory_energy(bm)
        esrc["compute"] = self.energy.flop_energy(flops, dt_native)
        e_noc = self.noc.energy_per_byte_hop or self.energy.j_per_byte_noc
        esrc["noc"] = noc_bytes * e_noc * max(1.0, self.noc.avg_hops(self.n_cores))
        esrc["static"] = self.energy.static_power * t
        energy = sum(esrc.values())

        bound = max(("compute", t_compute), ("memory", t_mem),
                    ("network", t_net), key=lambda kv: kv[1])[0]

        return OpCost(
            time=t, energy=energy, t_compute=t_compute, t_mem=t_mem,
            t_net=t_net, bound=bound,
            util=(flops / (t * peak)) if (t > 0 and peak > 0) else 0.0,
            bytes_by_level=bm, energy_by_source=esrc,
            notes=f"bottleneck={bottleneck} eff={eff:.3f} "
                  f"spill={res.spill_frac:.2f}",
        )

    # ------------------------------------------------------------------
    def summary(self) -> str:  # pragma: no cover
        from ..units import fmt_bytes, fmt_bw, fmt_flops
        base = Device.summary(self)
        extra = [
            f"  cores: {self.n_cores}  SRAM/core {fmt_bytes(self.sram_per_core)}"
            f"  -> total {fmt_bytes(self.total_sram)}",
            f"  SRAM agg BW  {fmt_bw(self.total_sram_bw)}",
            f"  NoC {self.noc.topology} bisection {fmt_bw(self.noc.bisection_bandwidth)}"
            f" hop {self.noc.hop_latency*1e9:.1f} ns",
            f"  inter-chip {fmt_bw(self.inter_chip_bw)} "
            f"({self.inter_chip_latency*1e6:.1f} us)",
            f"  inter-board {fmt_bw(self.inter_board_bw)} "
            f"({self.inter_board_latency*1e6:.1f} us)",
        ]
        return base + "\n" + "\n".join(extra)


def make_dataflow(name: str, spec: dict) -> DataflowDevice:
    n_cores = int(spec.get("cores", spec.get("n_units", 1)))
    sram_per_core = parse(spec.get("sram_per_core", 0))

    # per-core FLOPS -> device peak
    peak = {}
    if "peak" in spec:
        peak = {k: parse(v) for k, v in spec["peak"].items()}
    fpc = {k: parse(v) for k, v in spec.get("flops_per_core", {}).items()}
    for k, v in fpc.items():
        peak.setdefault(k, v * n_cores)

    compute = ComputeSpec(peak=peak, clock=parse(spec.get("clock", "1 GHz")))

    memspec = dict(spec.get("memory", {}))
    if "sram" not in memspec:
        memspec["sram"] = {
            "capacity": n_cores * sram_per_core,
            "bandwidth": spec.get("sram_bandwidth",
                                  n_cores * parse(spec.get("sram_bw_per_core", 0))),
            "latency": spec.get("sram_latency", "2 ns"),
            "per_unit_capacity": sram_per_core,
        }
    mem = [MemoryLevel.from_dict(k, v) for k, v in memspec.items()]
    mem.sort(key=lambda m: -m.bandwidth)

    nocspec = dict(spec.get("noc", {}))
    noc = NoC(
        topology=nocspec.get("topology", "mesh2d"),
        link_bandwidth=parse(nocspec.get("link_bandwidth", 0)),
        bisection_bandwidth=parse(nocspec.get("bisection_bandwidth", 0)),
        hop_latency=parse(nocspec.get("hop_latency", "5 ns")),
        flit_bytes=parse(nocspec.get("flit_bytes", 32)),
        energy_per_byte_hop=parse(nocspec.get("energy_per_byte_hop", 0)),
        multicast=bool(nocspec.get("multicast", True)),
    )

    en = EnergyModel(
        j_per_flop=dict(spec.get("j_per_flop", {})),
        j_per_byte=dict(spec.get("j_per_byte", {})),
        j_per_byte_noc=parse(spec.get("j_per_byte_noc", 0)),
        j_per_byte_link=parse(spec.get("j_per_byte_link", 0)),
        static_power=parse(spec.get("static_power", 0)),
        tdp=parse(spec.get("tdp", 0)),
    )

    dev = DataflowDevice(
        name=name, kind="dataflow", vendor=spec.get("vendor", ""),
        arch=spec.get("arch", "dataflow"),
        n_units=n_cores, n_cores=n_cores,
        sram_per_core=sram_per_core,
        flops_per_core=fpc,
        compute=compute, memory=mem, energy=en, noc=noc,
        has_on_package_dram=bool(spec.get("has_on_package_dram", False)),
        inter_chip_bw=parse(spec.get("inter_chip_bw", 0)),
        inter_chip_latency=parse(spec.get("inter_chip_latency", "1 us")),
        inter_board_bw=parse(spec.get("inter_board_bw", 0)),
        inter_board_latency=parse(spec.get("inter_board_latency", "5 us")),
        chips_per_board=int(spec.get("chips_per_board", 1)),
        sram_bw_per_core=parse(spec.get("sram_bw_per_core", 0)),
        pe_rows=int(spec.get("pe_rows", 16)),
        pe_cols=int(spec.get("pe_cols", 16)),
        spiking=bool(spec.get("spiking", False)),
        spike_rate=float(spec.get("spike_rate", 0.05)),
        supported_dtypes=tuple(spec.get("supported_dtypes",
                                        ("int8", "fp16", "bf16"))),
        process_nm=float(spec.get("process_nm", 0)),
        year=int(spec.get("year", 0)),
        notes=spec.get("notes", ""),
        extra=dict(spec.get("extra", {})),
    )
    for k in ("achievable_peak", "vector_frac", "jitter_sigma",
              "pipeline_depth"):
        if k in spec:
            setattr(dev, k, spec[k])
    if "kernel_launch_overhead" in spec:
        dev.kernel_launch_overhead = parse(spec["kernel_launch_overhead"])
    dev.links = dict(spec.get("links", {}))
    return dev
