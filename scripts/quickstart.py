#!/usr/bin/env python3
"""A guided tour of the FlowGPU Python API in ~60 lines of calls.

Everything the CLI does is available directly; this is the shortest path from
"I have a model and some hardware" to "here is the latency, throughput and
energy, and here is why".

    python3 scripts/quickstart.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flowgpu as fg
from flowgpu.mapping import placement as P
from flowgpu.power import calibrate_system
from flowgpu.report import report as R
from flowgpu.sim import ServingSimulator, SLO, WorkloadSpec
from flowgpu.system import build_system
from flowgpu.units import fmt_bytes, fmt_time


def main():
    # ---- 1. pick a model -------------------------------------------
    print("=" * 66)
    print("1. MODEL")
    print("=" * 66)
    model = fg.get_model("deepseek_v3", w_dtype="fp8", comm_dtype="fp8")
    print(model.summary())

    # ---- 2. describe some hardware ---------------------------------
    print("\n" + "=" * 66)
    print("2. SYSTEMS")
    print("=" * 66)
    gpu_only = build_system(dict(name="h100x32", pools={
        "gpu": dict(device="h100_sxm", count=32, devices_per_board=8,
                    parallel=dict(tp=8, ep=4), scaleup="nvlink4",
                    scaleout="ib_ndr", host_overhead="500 W")}))

    hetero = build_system(dict(
        name="h100x16 + ipu768",
        pools={
            "gpu": dict(device="h100_sxm", count=16, devices_per_board=8,
                        parallel=dict(tp=8, ep=2), scaleup="nvlink4",
                        scaleout="ib_ndr", host_overhead="500 W"),
            "brain": dict(device="ipu_gc200", count=768, devices_per_board=4,
                          parallel=dict(ep=768), scaleup="ipulink",
                          scaleout="ib_ndr", host_overhead="250 W"),
        },
        bridges=[dict(between=["gpu", "brain"], fabric="pcie5_x16",
                      lanes=16)]))

    # energy models are attached here, anchored to each device's TDP
    for s in (gpu_only, hetero):
        calibrate_system(s)
        print(R.system_report(s))
        print()

    # ---- 3. choose how to split the work ---------------------------
    print("=" * 66)
    print("3. PLACEMENT")
    print("=" * 66)
    policies = {
        "all_gpu": (gpu_only, P.all_gpu("gpu")),
        # the architecture from the announcement
        "pd_a": (hetero, P.pd_a("gpu", "brain", crossing_dtype="fp8")),
        # MoE experts in both phases -- usually much better
        "moe_offload": (hetero, P.moe_offload("gpu", "brain",
                                              crossing_dtype="fp8")),
    }
    for name, (_, pol) in policies.items():
        print(f"  {name:14s} {pol.notes}")
        for r in pol.rules:
            print(f"      -> {r.pool:6s} {r.note or ''}")

    # ---- 4. run --------------------------------------------------
    print("\n" + "=" * 66)
    print("4. RESULTS")
    print("=" * 66)
    wl = WorkloadSpec(n_requests=96, prompt_len=4096, output_len=256,
                      prompt_sigma=0.3, arrival_rate=1.0, max_batch=64,
                      chunk_tokens=8192, seed=1)
    slo = SLO(ttft=5.0, tpot=0.05)

    results = {}
    for name, (system, pol) in policies.items():
        sim = ServingSimulator(model, system, pol, wl,
                               collective_overlap=0.6,
                               microbatches=2 if name != "all_gpu" else 1)
        print(f"\n--- {name} ---")
        print(f"  weights per pool: " + ", ".join(
            f"{k}={fmt_bytes(v)}" for k, v in sim.weights_by_pool.items()))
        print(f"  KV capacity: {sim.kv_capacity_tokens:,.0f} tokens")
        for w in sim.warnings:
            print(f"  ! {w}")
        results[name] = sim.run()

    print()
    print(R.compare(results, slo, baseline="all_gpu"))

    # ---- 5. dig into one result ------------------------------------
    print("\n" + "=" * 66)
    print("5. WHY -- attribution for the pd_a run")
    print("=" * 66)
    print(R.summarize(results["pd_a"], slo, title="pd_a", verbose=True))

    # ---- 6. price a single step by hand ----------------------------
    print("\n" + "=" * 66)
    print("6. ONE DECODE STEP, PRICED OP BY OP (top 8 by time)")
    print("=" * 66)
    from flowgpu.sim.executor import execute
    from flowgpu.workload import llm
    g = llm.build_decode(model, batch=32, ctx=4096, ep_world=768, tp_world=8)
    pol = P.pd_a("gpu", "brain", crossing_dtype="fp8")
    pol.annotate(g, model.n_layers)
    res = execute(g, hetero, pol, n_layers=model.n_layers, trace=True)
    print(f"step latency {fmt_time(res.time)}   "
          f"crossings {res.n_crossings}   "
          f"bytes crossed {fmt_bytes(res.bytes_crossed)}")
    rows = sorted(res.op_trace, key=lambda t: -t[2])[:8]
    print(R.table([[n, p, fmt_time(t), f"{u*100:.2f}%", b]
                   for n, p, t, u, b in rows],
                  ["op", "pool", "time", "util", "bound"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
