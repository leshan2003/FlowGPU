"""Probe for the hot/cold expert-tiering proposal.

Answers, using only machinery that already exists in FlowGPU, the three
questions the proposal's first gate depends on:

1. **Where does a decode step actually spend its time on the brain pool?**
   ``t_by_bound`` splits the step into compute / memory / network / latency.
   If ``network`` is not a binding term, replicating hot experts to shorten
   NoC and all-to-all distance buys nothing -- it only multiplies SRAM.

2. **How badly does the pinned MoE stack overflow the pool?**
   ``pool.residency.spill_frac`` is the fraction of weights the pool cannot
   hold.  That fraction is exactly what hot/cold tiering is trying to move
   somewhere cheaper, so it is the upper bound on the saving.

3. **How does the answer move with batch?**
   ``experts_touched`` grows with T, so a "cold" expert only exists at
   small-to-medium batch.  At large T every expert is touched and the
   tiering question dissolves.

Not part of the package API.  See ``repro/README.md``.

    python repro/hot_cold_probe.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowgpu.mapping import placement as P            # noqa: E402
from flowgpu.power import calibrate_system            # noqa: E402
from flowgpu.sim.executor import execute              # noqa: E402
from flowgpu.system import build_system               # noqa: E402
from flowgpu.units import dtype_bytes, fmt_bw, fmt_bytes   # noqa: E402
from flowgpu.workload import graph as G                    # noqa: E402
from flowgpu.workload import llm                      # noqa: E402
from flowgpu.workload.models import get_model         # noqa: E402


GPU_COUNT = 16
BRAIN_COUNT = 256
CTX = 4096
BATCHES = (1, 8, 32, 64)


def plan_memory(model, system, graph, policy) -> dict:
    """Assign each pool its weight share and check it fits.

    This mirrors ``ServingSimulator._plan_memory``.  It matters because
    ``execute`` falls back to ``plan_residency(0, 0, 0)`` when a pool has
    never been planned -- i.e. "all weights on SRAM, zero spill", which is
    the most optimistic possible answer.

    The MoE term is the important one: the decode graph only names the
    experts it *touched*, but the pool has to *hold* all of them.
    """
    policy.annotate(graph, model.n_layers)
    w_by_pool: dict = {}
    seen = set()
    for op in graph.ops:
        pool = op.meta["pool"]
        for t in op.weights:
            if t.name in seen:
                continue
            seen.add(t.name)
            nb = t.nbytes
            if op.role == "moe" and "experts_touched" in op.meta:
                et = max(op.meta["experts_touched"], 1e-9)
                if "expert_pool" in t.name:
                    nb = nb / et * op.meta["n_experts"]
            if op.role == "embed":
                nb = model.vocab * model.d_model * dtype_bytes(model.w_dtype)
            w_by_pool[pool] = w_by_pool.get(pool, 0.0) + nb
    for name, p in system.pools.items():
        p.plan(w_by_pool.get(name, 0.0), 0.0, 0.0)
    return w_by_pool


def build(batch: int, brain_chips: int, chips_per_node: int = 8,
          hot_frac: float = 0.0, hot_traffic: float = 0.0):
    m = get_model("deepseek_v3", w_dtype="fp8", a_dtype="bf16",
                  kv_dtype="bf16", comm_dtype="fp8")
    if hot_frac:
        m.extra["hot_frac"] = hot_frac
        m.extra["hot_traffic"] = hot_traffic or hot_frac
    s = build_system(dict(
        name="h100x16 + brain",
        pools={
            "gpu": dict(device="h100_sxm", count=GPU_COUNT,
                        devices_per_board=8,
                        parallel=dict(tp=8, pp=1, ep=2, dp=1),
                        scaleup="nvlink4", scaleout="ib_ndr",
                        host_overhead=500.0, price_per_device=30000.0),
            "brain": dict(device="lynxi_hp300_proj", count=brain_chips,
                          devices_per_board=chips_per_node,
                          parallel=dict(tp=1, pp=1, ep=brain_chips, dp=1),
                          scaleup="custom_serdes_800g",
                          scaleout="roce_400g",
                          host_overhead=200.0, price_per_device=15000.0),
        },
        # NOTE: ``lanes`` *multiplies* the fabric bandwidth, so naming a
        # fabric "pcie5_x16" and then also passing lanes=16 charges 16
        # parallel x16 links (819 GB/s) instead of one (51 GB/s).  The
        # shipped experiment configs do exactly that; this probe does not.
        bridges=[dict(between=["gpu", "brain"], fabric="pcie5_x16")],
    ))
    calibrate_system(s)
    # ffn_tp_world=1: the FFN/MoE lands on the brain pool, which is tp=1, so
    # the tail all-reduce that exists to undo tensor parallelism is not
    # needed and must not be charged (nor crossed).
    g = llm.build_decode(m, batch=batch, ctx=CTX, ep_world=brain_chips,
                         tp_world=8, ffn_tp_world=1)
    return m, s, g


def main() -> int:
    policy = P.pd_a("gpu", "brain", crossing_dtype="fp8")

    print("=" * 78)
    print("A. Where a decode step spends its time  (pd_a, brain pool)")
    print("=" * 78)
    print(f"  GPU: {GPU_COUNT} x h100_sxm      ctx = {CTX}")
    print("  'memory' dominating in the 256-chip row is *spill*, not SRAM:")
    print("  the pool cannot hold the expert stack, so weights stream from DRAM")
    print("  every step.  1024 chips is the row where it all fits.")
    print()

    for chips in (BRAIN_COUNT, 1024):
        print(f"  --- brain = {chips} x hp300_proj ---")
        hdr = (f"  {'batch':>5} {'T':>5} {'experts':>8} {'step ms':>9} "
               f"{'compute':>8} {'memory':>8} {'network':>8} {'bridge':>7} "
               f"{'coll.':>7}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for b in BATCHES:
            m, s, g = build(b, chips)
            plan_memory(m, s, g, policy)
            res = execute(g, s, policy, n_layers=m.n_layers,
                          collective_overlap=0.6)
            tb = res.t_by_bound
            tot = max(res.time, 1e-12)

            # distinct experts touched, summed over MoE layers
            touched = 0.0
            for op in g.ops:
                if op.role == "moe" and "experts_touched" in op.meta:
                    touched += op.meta["experts_touched"]

            print(f"  {b:>5} {b:>5} {touched:>8.0f} "
                  f"{res.time * 1e3:>9.2f} "
                  f"{tb.get('compute', 0) / tot:>7.1%} "
                  f"{tb.get('memory', 0) / tot:>7.1%} "
                  f"{tb.get('network', 0) / tot:>7.1%} "
                  f"{res.t_bridge / tot:>6.1%} "
                  f"{res.t_collective / tot:>6.1%}")
        print()

    print("  'bridge' = GPU<->brain crossings, already inside the other terms.")
    print("  'coll.'  = MoE all-to-all dispatch/combine, after 60% overlap credit.")

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("B. Does the brain pool actually hold the MoE stack?")
    print("=" * 78)
    print(f"  {'brain chips':>12} {'per-chip demand':>17} "
          f"{'capacity':>11} {'spill':>9}")
    print("  " + "-" * 54)
    for n in (256, 512, 1024, 2048, 4096):
        m, s, g = build(8, n)
        plan_memory(m, s, g, policy)
        res = execute(g, s, policy, n_layers=m.n_layers)
        pool = s.pools["brain"]
        per_dev = pool.weight_bytes / max(1, pool.stage_devices)
        print(f"  {n:>12} {fmt_bytes(per_dev):>17} "
              f"{fmt_bytes(pool.device.capacity):>11} "
              f"{pool.residency.spill_frac:>8.1%}")

    print()
    brain = s.pools["brain"].device
    print(f"  brain SRAM per chip      : {fmt_bytes(brain.total_sram)} "
          f"({brain.n_cores} cores x {fmt_bytes(brain.sram_per_core)})")
    print(f"  brain aggregate SRAM BW  : "
          f"{fmt_bw(s.pools['brain'].device.total_sram_bw)}")
    dram = brain.level("dram")
    print(f"  brain DRAM per chip      : {fmt_bytes(dram.capacity)} "
          f"@ {fmt_bw(dram.bandwidth)}   (used only for the spilled share)")
    print(f"  GPU->brain bridge        : pcie5_x16, "
          f"{fmt_bw(s.bridge('gpu', 'brain').eff_bw)} usable")

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("C. Gate 1 -- what happens if the COLD experts move to the GPU's HBM")
    print("=" * 78)

    m = get_model("deepseek_v3", w_dtype="fp8", a_dtype="bf16",
                  kv_dtype="bf16", comm_dtype="fp8")
    d, de = m.d_model, m.d_expert_ff
    E, k = m.n_experts, m.n_active_experts
    moe_layers = m.n_layers - m.first_dense_layers
    expert_bytes = 3 * d * de * dtype_bytes("fp8")          # per expert
    total_moe = expert_bytes * E * moe_layers

    print(f"  DeepSeek-V3: {E} experts x {moe_layers} MoE layers, "
          f"{fmt_bytes(expert_bytes)} per expert")
    print(f"  full MoE stack = {fmt_bytes(total_moe)}")
    print(f"  GPU pool = {GPU_COUNT} x 80.00 GB = "
          f"{fmt_bytes(80 * 2 ** 30 * GPU_COUNT)} of HBM")
    print()

    gpu_pool = s.pools["gpu"]
    gpu = gpu_pool.device
    gpu_stage = gpu_pool.stage_devices
    attn_w = 0.0                       # attention weights stay on the GPU
    for op in g.ops:
        if op.role in ("attn", "embed", "lmhead", "norm"):
            for t in op.weights:
                attn_w += t.nbytes
    attn_per_dev = attn_w / gpu_stage

    print(f"  attention/embed/head weights that also live on the GPU: "
          f"{fmt_bytes(attn_w)}  ({fmt_bytes(attn_per_dev)}/device)")
    print()
    hdr = (f"  {'hot_frac':>8} {'hot->SRAM':>10} {'cold->GPU HBM':>14} "
           f"{'GPU/dev need':>13} {'of 80 GB':>9} {'KV headroom':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for hf in (0.0, 0.1, 0.2, 0.3, 0.5):
        hot_b = total_moe * hf
        cold_b = total_moe - hot_b
        need = (cold_b + attn_w) / gpu_stage
        head = 80 * 2 ** 30 - need
        print(f"  {hf:>8.1f} {fmt_bytes(hot_b):>10} {fmt_bytes(cold_b):>14} "
              f"{fmt_bytes(need):>13} {need / (80 * 2 ** 30):>8.0%} "
              f"{fmt_bytes(head):>12}")

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("D. Past the point where the stack fits, does another chip help?")
    print("=" * 78)
    print("  Beyond ~1024 chips the spill is already 0, so the only things a")
    print("  further chip can still buy are compute and SRAM bandwidth -- and")
    print("  the only things it costs are leakage and a wider all-to-all.")
    print()
    hdr = (f"  {'batch':>5} {'chips':>6} {'spill':>7} {'step ms':>9} "
           f"{'compute':>8} {'memory':>8} {'coll.':>7} {'coll. abs':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for b in (8, 64):
        for chips in (1024, 2048, 4096):
            m, s, g = build(b, chips)
            plan_memory(m, s, g, policy)
            res = execute(g, s, policy, n_layers=m.n_layers,
                          collective_overlap=0.6)
            tb, tot = res.t_by_bound, max(res.time, 1e-12)
            print(f"  {b:>5} {chips:>6} "
                  f"{s.pools['brain'].residency.spill_frac:>6.1%} "
                  f"{res.time * 1e3:>9.2f} "
                  f"{tb.get('compute', 0) / tot:>7.1%} "
                  f"{tb.get('memory', 0) / tot:>7.1%} "
                  f"{res.t_collective / tot:>6.1%} "
                  f"{res.t_collective * 1e3:>9.2f}ms")
        print()

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("E. What does GOING MULTI-NODE cost?")
    print("=" * 78)
    print("  Same 1024 chips, same total SRAM -- only the node granularity")
    print("  changes.  A collective that fits inside one node rides `scaleup`")
    print("  (custom_serdes_800g, 100 GB/s); anything wider falls to `scaleout`")
    print("  (roce_400g, 50 GB/s, 7 us).  EP dispatch spans all 1024 chips, so")
    print("  it crosses nodes as soon as a node holds fewer than 1024.")
    print()
    hdr = (f"  {'batch':>5} {'chips/node':>11} {'nodes':>7} {'step ms':>9} "
           f"{'coll.':>7} {'coll. abs':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for b in (8, 64):
        for cpn in (8, 32, 128, 1024):
            m, s, g = build(b, 1024, chips_per_node=cpn)
            plan_memory(m, s, g, policy)
            res = execute(g, s, policy, n_layers=m.n_layers,
                          collective_overlap=0.6)
            nodes = s.pools["brain"].n_boards
            print(f"  {b:>5} {cpn:>11} {nodes:>7} "
                  f"{res.time * 1e3:>9.2f} "
                  f"{res.t_collective / max(res.time, 1e-12):>6.1%} "
                  f"{res.t_collective * 1e3:>9.2f}ms")
        print()

    print("  Per-node SRAM if the whole 609 GB stack had to sit in ONE node:")
    for cpn in (8, 32, 128, 1024):
        print(f"    {cpn:>5} chips/node -> "
              f"{fmt_bytes(609_000_000_000 / cpn)} of SRAM per chip "
              f"(hp300_proj has 768.000 MB)")

    # ------------------------------------------------------------------
    # F. The actual trade: pin h of the experts, send the rest off-chip.
    #
    # This is a *calculation* driven by the simulator's own formulas and
    # device parameters, not a full pipeline run -- the MoE op is still a
    # single indivisible op in ``llm.py``, so it cannot yet be split across
    # two pools.  What it does give is the shape of the trade-off, which is
    # what decides whether wiring the split in is worth doing.
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("F. When does hot/cold tiering pay?")
    print("=" * 78)
    usable_per_chip = 768e6 * 0.85
    brain_dram_bw = 200e9
    gpu_hbm_bw = 3350e9
    tdp_per_chip = 150.0
    baseline_chips = total_moe / usable_per_chip
    print(f"  baseline: pin everything -> {baseline_chips:.0f} chips, "
          f"{baseline_chips * tdp_per_chip / 1e3:.1f} kW of TDP")
    print(f"  k = {k} experts per token;  a hot expert is read once per hit")
    print()
    print("  'cold hits'   = distinct cold experts read per layer, averaged")
    print("  'DRAM/HBM ms' = time to stream those cold weights for the WHOLE")
    print("                  step (all 58 MoE layers), per the device models")
    print()

    base_step = {1: 3.07e-3, 8: 3.35e-3, 64: 7.11e-3}   # section E, 1 node
    for T in (1, 8, 64):
        hdr = (f"  batch {T:>2}  {'hot_frac':>8} {'p':>5} {'chips':>6} "
               f"{'kW':>6} {'cold hits':>9} {'cold GB':>8} {'DRAM ms':>8} "
               f"{'HBM ms':>7} {'HBM vs base':>12}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for hf in (0.0, 0.1, 0.2):
            E_hot = E * hf
            E_cold = E - E_hot
            for p in (0.0, 0.5, 0.7, 0.9):
                if p < hf or (hf == 0.0 and p > 0.0):
                    continue
                k_cold = k * (1.0 - p)
                t_cold = (E_cold * (1.0 - (1.0 - k_cold / E_cold) ** T)
                          if E_cold > 0 and k_cold > 0 else 0.0)
                chips = (hf * total_moe) / usable_per_chip
                cold_gb = t_cold * expert_bytes * moe_layers / 1e9
                hbm_ms = cold_gb * 1e9 / gpu_hbm_bw * 1e3
                delta = hbm_ms / (base_step[T] * 1e3) - 1.0
                print(f"  {'':>8} {hf:>8.1f} {p:>5.1f} {chips:>6.0f} "
                      f"{chips * tdp_per_chip / 1e3:>6.1f} {t_cold:>9.2f} "
                      f"{cold_gb:>8.2f} "
                      f"{cold_gb * 1e9 / brain_dram_bw * 1e3:>8.2f} "
                      f"{hbm_ms:>7.2f} {delta:>11.0%}")
            print()
        print(f"  (baseline step for batch {T} = {base_step[T] * 1e3:.2f} ms,"
              f" from section E, single node)")
        print()

    print("  p = share of token-expert assignments landing on the pinned set;")
    print("  p == hot_frac would be perfectly uniform routing (no skew).")
    print("  hot_frac 0.0 means nothing is pinned: every hit is a cold hit.")

    # ------------------------------------------------------------------
    # G. The real thing: build the graph with the split and price it.
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("G. hot_cold vs pd_a -- priced by the simulator, not by hand")
    print("=" * 78)
    print("  GPU = 16 x h100_sxm, ctx = 4096.  The baseline is pd_a sized so")
    print("  it does NOT spill (~1000 chips); the tiered arms buy fewer chips.")
    print()
    hdr = (f"  {'batch':>5} {'arm':>28} {'chips':>6} {'kW':>6} "
           f"{'step ms':>9} {'compute':>8} {'memory':>8} {'coll.':>7} "
           f"{'xings':>6} {'spill':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    arms = [
        ("pd_a  all pinned",          1024, 0.0, 0.0, "pd_a"),
        ("pd_a  all pinned",           512, 0.0, 0.0, "pd_a"),
        ("hot_cold h=.1 p=.9",         256, 0.1, 0.9, "hot_cold"),
        ("hot_cold h=.1 p=.9",         128, 0.1, 0.9, "hot_cold"),
        ("hot_cold h=.2 p=.9",         256, 0.2, 0.9, "hot_cold"),
        ("hot_cold h=.3 p=.95",        384, 0.3, 0.95, "hot_cold"),
    ]
    for b in (1, 8, 64):
        for label, chips, hf, p, pol_name in arms:
            m, s, g = build(b, chips, hot_frac=hf, hot_traffic=p)
            policy = (P.pd_a("gpu", "brain", crossing_dtype="fp8")
                      if pol_name == "pd_a" else
                      P.hot_cold("gpu", "brain", crossing_dtype="fp8"))
            plan_memory(m, s, g, policy)
            res = execute(g, s, policy, n_layers=m.n_layers,
                          collective_overlap=0.6)
            tb, tot = res.t_by_bound, max(res.time, 1e-12)
            spill = max(s.pools["brain"].residency.spill_frac,
                        s.pools["gpu"].residency.spill_frac)
            kw = (chips * 150.0 + GPU_COUNT * 700.0) / 1e3
            print(f"  {b:>5} {label:>28} {chips:>6} {kw:>6.1f} "
                  f"{res.time * 1e3:>9.2f} "
                  f"{tb.get('compute', 0) / tot:>7.1%} "
                  f"{tb.get('memory', 0) / tot:>7.1%} "
                  f"{res.t_collective / tot:>6.1%} "
                  f"{res.n_crossings:>6} {spill:>6.1%}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
