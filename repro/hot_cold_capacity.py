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
    # label                              chips  hot_frac  traffic  policy
    ("A   pd_a     @1024 all pinned", 1024, 0.00, 0.00, "pd_a"),
    ("B   pd_a     @ 256 cold->DRAM", 256, 0.10, 0.90, "pd_a"),
    ("C   hot_cold @ 256 cold->HBM", 256, 0.10, 0.90, "hot_cold"),
    ("C'  hot_cold @ 128 cold->HBM", 128, 0.10, 0.90, "hot_cold"),
]

# `collective_overlap` is scheduler quality, not hardware: how much of the
# MoE all-to-all a production engine hides behind compute (DualPipe and
# SGLang two-batch overlap reach ~0.6-0.9; a naive engine is ~0).  The
# package default is 0.0; the shipped configs use 0.6.  Every number
# reported before this sweep was taken at a single point on this axis.
COS = (0.0, 0.6, 0.9)
MB = 1


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
    # n_requests must exceed max_running or no queue can ever form and the
    # capacity search saturates its bracket (see find_capacity's guard).
    #
    # `co` is scheduler quality, not hardware: the fraction of the MoE
    # all-to-all a production engine hides behind compute.  See COS below.
    wl = WorkloadSpec(n_requests=256, prompt_len=1024, output_len=64,
                      prompt_sigma=0.4, output_sigma=0.5,
                      max_batch=64, max_running=64, chunk_tokens=8192,
                      seed=11)
    slo = SLO(ttft=5.0, tpot=0.05)

    print("=" * 92)
    print("SLO capacity: 16 x h100_sxm + N x hp300_proj")
    print(f"  workload: {wl.n_requests} reqs, prompt {wl.prompt_len},"
          f" output {wl.output_len}, SLO ttft {slo.ttft}s / tpot {slo.tpot}s")
    print("=" * 92)
    hdr = (f"  {'arm':>28} {'co':>5} {'chips':>6} {'kW':>6} "
           f"{'req/s':>8} {'tok/s':>8} {'TTFT p99':>9} {'TPOT p99':>9} "
           f"{'J/tok':>7} {'s':>5}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    got = {}
    for label, chips, hf, p, pol in ARMS:
        for co in COS:
            t0 = time.time()
            m, s = build(chips, hot_frac=hf, hot_traffic=p)
            cap = find_capacity(m, s, make_policy(pol), wl, slo=slo,
                                target=0.95, n_requests=wl.n_requests,
                                iters=6,
                                sim_kwargs=dict(collective_overlap=co,
                                                jitter=True,
                                                microbatches=MB),
                                name=f"{label.strip()} co={co}")
            if cap.warnings:
                for warning in cap.warnings:
                    print(f"  !! {label.strip()} co={co}: {warning}")
            el = time.time() - t0
            b = cap.best
            rec = dict(
                rate=cap.max_rate,
                thr=b.output_throughput if b else 0.0,
                ttft=b.ttft_p99 if b else 0.0,
                tp99=b.tpot_p99 if b else 0.0,
                jtok=b.energy_per_output_token if b else 0.0,
            )
            got[(label, co)] = rec
            print(f"  {label:>28} {co:>5.2f} {chips:>6} "
                  f"{(chips * 150.0 + GPU_COUNT * 700.0) / 1e3:>6.1f} "
                  f"{rec['rate']:>8.2f} {rec['thr']:>8.1f} "
                  f"{rec['ttft']:>8.2f}s "
                  f"{rec['tp99'] * 1e3:>8.2f}ms {rec['jtok']:>7.2f} "
                  f"{el:>5.0f}")
        print()

    # ---- the matrix the whole exercise is for ------------------------
    head = f"  {'arm':>28} " + "".join(f"{('co=' + str(c)):>12}" for c in COS)
    print("  SLO capacity (req/s), rows = arm, cols = collective_overlap")
    print()
    print(head)
    print("  " + "-" * (len(head) - 2))
    for label, chips, hf, p, pol in ARMS:
        row = "".join(f"{got[(label, c)]['rate']:>12.2f}" for c in COS)
        print(f"  {label:>28}{row}")
    print()
    print("  throughput (tok/s), rows = arm, cols = collective_overlap")
    print()
    head = f"  {'arm':>28} " + "".join(f"{('co=' + str(c)):>12}" for c in COS)
    print(head)
    print("  " + "-" * (len(head) - 2))
    base_thr = got[(ARMS[0][0], COS[-1])]["thr"] or 1.0
    for label, chips, hf, p, pol in ARMS:
        row = "".join(f"{got[(label, c)]['thr']:>12.1f}" for c in COS)
        print(f"  {label:>28}{row}")
    print()
    print("  same, normalised to A at co=0.9")
    print()
    print(head)
    print("  " + "-" * (len(head) - 2))
    for label, chips, hf, p, pol in ARMS:
        row = "".join(f"{got[(label, c)]['thr'] / base_thr:>11.0%} "
                      for c in COS)
        print(f"  {label:>28}{row}")
    print()
    print("  energy per output token (J), lower is better")
    print()
    print(head)
    print("  " + "-" * (len(head) - 2))
    for label, chips, hf, p, pol in ARMS:
        row = "".join(f"{got[(label, c)]['jtok']:>12.2f}" for c in COS)
        print(f"  {label:>28}{row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
