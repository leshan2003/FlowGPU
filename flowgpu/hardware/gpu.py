"""GPU device model.

Models an SM-based accelerator with a register file / shared-memory / L1 / L2 /
HBM hierarchy.  The physics that matters for LLM inference:

* **Weights live in HBM.**  L2 is 40-120 MB; a model shard is tens of GB.  So
  every decode step re-streams the whole active weight shard from HBM.  Decode
  is therefore *HBM-bandwidth bound* at an arithmetic intensity of ~the batch
  size, and the achieved fraction of peak FLOPS is tiny.
* **Prefill is compute bound.**  Arithmetic intensity scales with chunk length,
  so tensor cores actually run near peak.
* **Attention reads the KV cache from HBM** and grows linearly with context,
  which is what makes long-context decode hurt.
* **Small-M GEMMs waste tensor cores.**  A tile/wave quantisation model
  captures the cliff: at batch 8 with a 128x256 tile, an SM computes 128 rows
  of output to get 8 useful ones.

Non-idealities modelled: tile quantisation, wave quantisation across SMs,
achievable-peak derate, kernel launch overhead, and scheduling jitter (the
"调度抖动" the article calls out) as a configurable per-kernel noise term.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..units import dtype_bytes
from ..workload import graph as G
from .base import ComputeSpec, Device, EnergyModel, MemoryLevel, OpCost, Residency


@dataclass
class GPUDevice(Device):
    kind: str = "gpu"
    # tensor-core tile shape (per SM-level MMA tile)
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 64
    # smallest M the tensor-core MMA instruction supports (Hopper bf16 = 16)
    mma_m: int = 16
    # fraction of peak a huge, perfectly-shaped GEMM actually achieves
    achievable_peak: float = 0.80
    # fraction of peak achievable by memory-bound elementwise/vector work
    vector_frac: float = 0.05
    # L2 hit rate for activation traffic
    l2_act_hit: float = 0.85
    # scheduling jitter: sigma as a fraction of kernel time (SM contention,
    # CTA-launch skew, CPU-side dispatch bubbles).  Used by the DES layer.
    jitter_sigma: float = 0.06
    # extra fixed cost paid by attention kernels for the softmax pass
    attn_overhead: float = 3.0e-6
    # number of concurrent kernels the scheduler can keep in flight
    n_streams: int = 4

    # ------------------------------------------------------------------
    def plan_residency(self, weight_bytes: float, kv_bytes: float,
                       act_bytes: float) -> Residency:
        """Weights and KV go to HBM; L2 backs activations."""
        hbm = self.level("hbm") or self.main_memory
        l2 = self.level("l2")
        cap = hbm.capacity
        need = weight_bytes + kv_bytes + act_bytes
        spill = max(0.0, (need - cap) / need) if need > 0 else 0.0
        return Residency(
            weight_level_frac={hbm.name: 1.0},
            act_level=(l2.name if l2 else hbm.name),
            kv_level=hbm.name,
            resident_bytes=need,
            capacity_bytes=cap,
            spill_frac=spill,
        )

    # ------------------------------------------------------------------
    def efficiency(self, op: G.Op, batch_rows: int, shards: int = 1) -> float:
        """Achieved fraction of peak FLOPS for this op shape.

        ``shards`` is the number of devices splitting the op.  Tensor
        parallelism narrows the *output* dimension of a GEMM (and the *head*
        dimension of attention), which shrinks the per-device tile count and
        can push a large GEMM back into the wave-quantisation regime -- a real
        and frequently-ignored cost of aggressive TP.

        ``op.meta['groups']`` is the number of independent GEMMs fused into
        this op (MoE experts).  Grouped GEMMs fill the machine far better than
        their individual shapes suggest, so ignoring this makes MoE look
        absurdly inefficient.
        """
        if op.kind in (G.ELEMENTWISE, G.NORM, G.SOFTMAX, G.EMBED, G.ROUTER):
            return self.vector_frac
        if op.kind in (G.A2A, G.ALLREDUCE):
            return self.vector_frac

        shards = max(1, int(shards))
        ep = max(1, int(op.meta.get("ep", 1)))
        tp = max(1, shards // ep)
        groups = max(1.0, float(op.meta.get("groups", 1)))
        M = max(1, int(op.m or batch_rows))
        N = max(1, int(op.n or 1))
        K = max(1, int(op.k or 1))

        if op.kind in (G.ATTENTION, G.BMM):
            # TP splits heads, i.e. the batched-M dimension
            M = max(1, M // shards)
        else:
            # EP splits the expert groups; TP splits each expert's output dim
            groups = max(1.0, groups / ep)
            N = max(1, N // tp)

        # --- tile quantisation: padded work / useful work ---------------
        # A real kernel picks its tile shape for the problem: CUTLASS ships
        # M-tiles from the MMA minimum (16 rows for bf16 on Hopper) up to 256.
        # Pinning tile_m at 128 would make every MoE decode look 8x worse than
        # it is, because each expert only sees a handful of tokens.
        tm = self.tile_m
        while tm > self.mma_m and tm // 2 >= M:
            tm //= 2
        pm = math.ceil(M / tm) * tm
        pn = math.ceil(N / self.tile_n) * self.tile_n
        pk = math.ceil(K / self.tile_k) * self.tile_k
        eff_tile = (M * N * K) / (pm * pn * pk)

        # --- wave quantisation across SMs -------------------------------
        n_tiles = (math.ceil(M / tm) * math.ceil(N / self.tile_n) * groups)
        slots = max(1, self.n_units)

        # split-K / stream-K: when the output is too small to give every SM a
        # tile -- which is the normal case for decode, where M is the batch --
        # a real kernel partitions the reduction dimension instead of leaving
        # SMs idle, then reduces the partials.  cuBLAS and CUTLASS both do
        # this; without it the model badly over-penalises decode GEMMs.
        split_k = 1
        reduce_penalty = 1.0
        if n_tiles < slots:
            max_split = max(1, int(pk // self.tile_k))
            split_k = max(1, min(int(math.ceil(slots / n_tiles)), max_split))
            if split_k > 1:
                # the cross-CTA reduction costs a workspace round trip
                reduce_penalty = 1.0 / (1.0 + 0.18 * math.log2(split_k))
        n_tiles_eff = n_tiles * split_k
        waves = math.ceil(n_tiles_eff / slots)
        eff_wave = n_tiles_eff / (waves * slots) * reduce_penalty

        # --- K-dimension pipeline fill ----------------------------------
        # short reduction chains cannot hide MMA latency; splitting K makes
        # each chain shorter, so the penalty applies to the split length
        k_per_cta = pk / split_k
        eff_k = min(1.0, k_per_cta / (4.0 * self.tile_k))
        eff_k = 0.35 + 0.65 * eff_k

        eff = self.achievable_peak * eff_tile * eff_wave * eff_k
        if op.kind in (G.ATTENTION, G.BMM):
            # flash-attention style kernels lose more to softmax/online rescale
            eff *= 0.72
        return max(eff, 1e-4)

    # ------------------------------------------------------------------
    def _byte_map(self, op: G.Op, res: Residency, shards: int) -> dict:
        """Distribute this op's bytes over memory levels."""
        bm = {}
        hbm = (self.level("hbm") or self.main_memory).name
        l2 = res.act_level

        # weights: split by residency plan, sharded across TP ranks
        wb = op.weight_bytes / max(1, shards)
        for lvl, frac in res.weight_level_frac.items():
            bm[lvl] = bm.get(lvl, 0.0) + wb * frac

        # KV cache state: always HBM (too big for L2), sharded by heads
        st = (op.state_read_bytes + op.state_write_bytes) / max(1, shards)
        bm[hbm] = bm.get(hbm, 0.0) + st

        # activations: mostly L2-resident, misses go to HBM
        ab = op.act_bytes / max(1, shards)
        bm[l2] = bm.get(l2, 0.0) + ab * self.l2_act_hit
        bm[hbm] = bm.get(hbm, 0.0) + ab * (1.0 - self.l2_act_hit)
        return bm

    # ------------------------------------------------------------------
    def cost_op(self, op: G.Op, res: Residency, shards: int = 1) -> OpCost:
        dt = op.meta.get("compute_dtype") or (
            op.weights[0].dtype if op.weights else "bf16")
        peak = self.compute.peak_for(dt)
        flops = op.flops / max(1, shards)

        eff = self.efficiency(op, batch_rows=op.m or 1, shards=shards)
        t_compute = flops / (peak * eff) if peak > 0 else 0.0

        bm = self._byte_map(op, res, shards)
        t_mem, bottleneck = self._memory_time(bm)

        # latency floor: a kernel cannot beat one round trip to its data
        hbm = self.level("hbm") or self.main_memory
        lat_floor = hbm.latency if bm.get(hbm.name, 0) > 0 else 0.0

        overhead = self.kernel_launch_overhead
        if op.kind == G.ATTENTION:
            overhead += self.attn_overhead

        if self.overlap_compute_memory:
            t = max(t_compute, t_mem, lat_floor) + overhead
        else:
            t = t_compute + t_mem + overhead

        # ---- energy ----
        esrc = self._memory_energy(bm)
        esrc["compute"] = self.energy.flop_energy(flops, dt)
        esrc["static"] = self.energy.static_power * t
        energy = sum(esrc.values())

        bound = "compute" if t_compute >= max(t_mem, lat_floor) else (
            "memory" if t_mem >= lat_floor else "latency")

        return OpCost(
            time=t, energy=energy, t_compute=t_compute, t_mem=t_mem,
            bound=bound,
            util=(flops / (t * peak)) if (t > 0 and peak > 0) else 0.0,
            bytes_by_level=bm, energy_by_source=esrc,
            notes=f"bottleneck={bottleneck} eff={eff:.3f}",
        )


def make_gpu(name: str, spec: dict) -> GPUDevice:
    """Build a :class:`GPUDevice` from a plain-dict spec (see registry)."""
    from ..units import parse
    compute = ComputeSpec(
        peak={k: parse(v) for k, v in spec["peak"].items()},
        vector_peak=parse(spec.get("vector_peak", 0)),
        clock=parse(spec.get("clock", "1.5 GHz")),
    )
    mem = [MemoryLevel.from_dict(k, v) for k, v in spec["memory"].items()]
    mem.sort(key=lambda m: -m.bandwidth)
    en = EnergyModel(
        j_per_flop=dict(spec.get("j_per_flop", {})),
        j_per_byte=dict(spec.get("j_per_byte", {})),
        j_per_byte_link=parse(spec.get("j_per_byte_link", 0)),
        static_power=parse(spec.get("static_power", 0)),
        tdp=parse(spec.get("tdp", 0)),
    )
    dev = GPUDevice(
        name=name, kind="gpu", vendor=spec.get("vendor", ""),
        n_units=int(spec.get("sms", spec.get("n_units", 1))),
        compute=compute, memory=mem, energy=en,
        process_nm=float(spec.get("process_nm", 0)),
        year=int(spec.get("year", 0)),
        notes=spec.get("notes", ""),
        extra=dict(spec.get("extra", {})),
    )
    for k in ("tile_m", "tile_n", "tile_k", "achievable_peak", "vector_frac",
              "l2_act_hit", "jitter_sigma", "n_streams"):
        if k in spec:
            setattr(dev, k, spec[k])
    if "kernel_launch_overhead" in spec:
        dev.kernel_launch_overhead = parse(spec["kernel_launch_overhead"])
    dev.links = dict(spec.get("links", {}))
    return dev
