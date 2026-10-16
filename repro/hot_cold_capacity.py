"""SLO capacity: does the tiered system actually serve less traffic?

Step latency says the tiered arms are 12-30% slower.  What decides whether
that matters is *capacity*: the arrival rate at which SLO attainment falls
to 95%.  A 12% slower step that still meets the SLO costs nothing; a 12%
slower step that drops the p99 out of SLO costs everything.

Compares, on identical GPUs:

  pd_a   @1024 brain chips   all experts pinned, no spill      (baseline)
  pd_a   @ 512 brain chips   fewer chips, 47% spill            (naive shrink)
  hot_cold @256              hot pinned, cold on the GPU's HBM
  hot_cold @128

Not part of the package API.  See ``repro/README.md``.

    python repro/hot_cold_capacity.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowgpu.capacity import find_capacity                  # noqa: E402
from flowgpu.mapping import placement as P                  # noqa: E402
from flowgpu.power import calibrate_system                  # noqa: E402
from flowgpu.sim.serving import SLO, WorkloadSpec           # noqa: E402
from flowgpu.system import build_system                     # noqa: E402
from flowgpu.workload.models import get_model               # noqa: E402

GPU_COUNT = 16


def build(brain_chips: int, hot_frac: float = 0.0, hot_traffic: float = 0.0):
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
                          devices_per_board=8,
                          parallel=dict(tp=1, pp=1, ep=brain_chips, dp=1),
                          scaleup="custom_serdes_800g",
                          scaleout="roce_400g",
                          host_overhead=200.0, price_per_device=15000.0),
        },
        bridges=[dict(between=["gpu", "brain"], fabric="pcie5_x16")],
    ))
    calibrate_system(s)
    return m, s


ARMS = [
    # label                                     chips  hf     p    policy  mb
    ("A   pd_a     @1024 all pinned   mb1", 1024, 0.00, 0.00, "pd_a", 1),
    ("A'mb2 pd_a   @1024 all pinned   mb2", 1024, 0.00, 0.00, "pd_a", 2),
    ("A'  pd_a     @ 512 naive cut    mb1", 512, 0.00, 0.00, "pd_a", 1),
    ("B   pd_a     @ 256 cold->DRAM   mb1", 256, 0.10, 0.90, "pd_a", 1),
    ("C   hot_cold @ 256 cold->HBM    mb1", 256, 0.10, 0.90, "hot_cold", 1),
    ("Cmb2 hot_cold@ 256 cold->HBM    mb2", 256, 0.10, 0.90, "hot_cold", 2),
    ("C'  hot_cold @ 128 cold->HBM    mb1", 128, 0.10, 0.90, "hot_cold", 1),
    ("C'mb2 hot_cold@128 cold->HBM    mb2", 128, 0.10, 0.90, "hot_cold", 2),
    ("D   layer_split frac_gpu=.2     mb1", 1024, 0.00, 0.00, "split", 1),
    ("Dmb2 layer_split frac_gpu=.2    mb2", 1024, 0.00, 0.00, "split", 2),
]


def make_policy(pol: str):
    if pol == "pd_a":
        return P.pd_a("gpu", "brain", crossing_dtype="fp8")
    if pol == "hot_cold":
        return P.hot_cold("gpu", "brain", crossing_dtype="fp8")
    if pol == "split":
        return P.layer_split(61, frac_gpu=0.2, gpu="gpu", brain="brain",
                             crossing_dtype="fp8")
    raise KeyError(pol)


def main() -> int:
    wl = WorkloadSpec(n_requests=48, prompt_len=1024, output_len=64,
                      prompt_sigma=0.4, output_sigma=0.5,
                      max_batch=64, max_running=192, chunk_tokens=8192,
                      seed=11)
    slo = SLO(ttft=5.0, tpot=0.05)

    print("=" * 92)
    print("SLO capacity: 16 x h100_sxm + N x hp300_proj")
    print(f"  workload: {wl.n_requests} reqs, prompt {wl.prompt_len},"
          f" output {wl.output_len}, SLO ttft {slo.ttft}s / tpot {slo.tpot}s")
    print("=" * 92)
    hdr = (f"  {'arm':>30} {'chips':>6} {'TDP kW':>7} {'req/s':>8} "
           f"{'tok/s':>8} {'TTFT p99':>9} {'TPOT p99':>9} {'J/tok':>7} "
           f"{'s':>5}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    base = None
    rows = []
    for label, chips, hf, p, pol, mb in ARMS:
        t0 = time.time()
        m, s = build(chips, hot_frac=hf, hot_traffic=p)
        policy = make_policy(pol)
        cap = find_capacity(m, s, policy, wl, slo=slo, target=0.95,
                            n_requests=48, iters=6,
                            sim_kwargs=dict(collective_overlap=0.6,
                                            jitter=True, microbatches=mb),
                            name=label.strip())
        el = time.time() - t0
        b = cap.best
        ttft = b.ttft_p99 if b else 0.0
        tp99 = b.tpot_p99 if b else 0.0
        thr = b.output_throughput if b else 0.0
        jtok = b.energy_per_output_token if b else 0.0
        if base is None:
            base = cap.max_rate
        rows.append((label, chips, thr, ttft, tp99, jtok))
        print(f"  {label:>30} {chips:>6} "
              f"{(chips * 150.0 + GPU_COUNT * 700.0) / 1e3:>7.1f} "
              f"{cap.max_rate:>8.3f} "
              f"{thr:>8.1f} "
              f"{ttft:>8.2f}s {tp99 * 1e3:>8.2f}ms "
              f"{jtok:>7.2f} {el:>5.0f}")

    print()
    if rows:
        b_thr, b_j = rows[0][2], rows[0][5]
        print("  relative to A (pd_a @1024 all pinned, mb1):")
        for label, chips, thr, ttft, tp99, jtok in rows:
            print(f"    {label:>30}  chips {chips/1024:>6.0%} of A   "
                  f"throughput {thr/b_thr:>6.0%}   J/tok {jtok/b_j:>6.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
