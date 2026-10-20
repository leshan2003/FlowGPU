#!/usr/bin/env python3
"""Break-even analysis for the PD+A architecture.

Answers: how good would the brain-inspired chip have to be for the
"2x inference output AND 2x energy efficiency" claim to hold?

Three sweeps, each isolating one lever:

1. **SRAM density** -- MB of on-chip SRAM per core, with the chip count
   rescaled to keep total capacity fixed.  This is the lever that decides
   how many chips (and therefore how many watts) the expert stack costs.
2. **Chip power** -- W per chip at fixed capability.  Directly tests how
   much of the vendor's efficiency claim is load-bearing.
3. **Bridge bandwidth** -- GB/s between the GPU and the dataflow pool.  In
   PD+A the hidden state crosses this link twice per layer per token, so it
   is the most likely hidden bottleneck.

Usage:  python3 scripts/run_breakeven.py [--quick]
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowgpu.breakeven import sweep_device
from flowgpu.experiment import load_config
from flowgpu.report.report import save_json
from flowgpu.sim.serving import SLO, WorkloadSpec
from flowgpu.workload.models import get_model

CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "configs", "experiments", "pd_a_reference.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CFG)
    ap.add_argument("--quick", action="store_true",
                    help="fewer points and fewer requests per point")
    ap.add_argument("--out", default="results/breakeven.json")
    ap.add_argument("--only", nargs="*",
                    choices=["sram", "power", "bridge"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    mc = dict(cfg["model"])
    model = get_model(mc.pop("name"), **mc)
    wl = WorkloadSpec(**cfg["workload"])
    slo = SLO(**cfg["slo"])
    sim_kwargs = dict(cfg.get("defaults", {}))
    sim_kwargs["microbatches"] = 2

    base = cfg["systems"]["pure_gpu_equal_capex"]
    het = cfg["systems"]["hetero_pda"]
    pol = dict(strategy="pd_a", gpu="gpu", brain="brain",
               crossing_dtype="fp8")

    n_req = 64 if args.quick else 128
    results = {}
    want = set(args.only or ["sram", "power", "bridge"])

    # ---- 1. SRAM per core -------------------------------------------
    if "sram" in want:
        vals = ["1 MB", "4 MB", "16 MB"] if args.quick else \
               ["0.5 MB", "1 MB", "2 MB", "4 MB", "8 MB", "16 MB", "32 MB"]
        print("\n" + "=" * 70)
        print("SWEEP 1: SRAM per core (chip count rescaled to hold total "
              "capacity fixed)")
        print("=" * 70)
        r = sweep_device(model, base, het, pol, wl, slo, vals,
                         pool="brain", param="sram_per_core",
                         sim_kwargs=sim_kwargs, n_requests=n_req,
                         label_fmt=lambda v: f"{v/2**20:,.3g} MB/core",
                         rescale_count=True, verbose=not args.quick)
        print("\n" + r.table())
        results["sram_per_core"] = _pack(r)

    # ---- 2. chip power ----------------------------------------------
    if "power" in want:
        vals = ["40 W", "150 W"] if args.quick else \
               ["20 W", "40 W", "60 W", "90 W", "150 W", "250 W"]
        print("\n" + "=" * 70)
        print("SWEEP 2: brain-chip TDP at fixed capability")
        print("=" * 70)
        r = sweep_device(model, base, het, pol, wl, slo, vals,
                         pool="brain", param="tdp",
                         sim_kwargs=sim_kwargs, n_requests=n_req,
                         label_fmt=lambda v: f"{v:,.0f} W",
                         rescale_count=False, verbose=not args.quick)
        print("\n" + r.table())
        results["tdp"] = _pack(r)

    # ---- 3. bridge bandwidth ----------------------------------------
    if "bridge" in want:
        print("\n" + "=" * 70)
        print("SWEEP 3: GPU <-> brain-chip bridge bandwidth")
        print("=" * 70)
        from flowgpu.breakeven import BreakevenPoint, BreakevenResult
        from flowgpu.capacity import find_capacity
        from flowgpu.mapping import placement as P
        from flowgpu.power import calibrate_system
        from flowgpu.system import build_system
        bsys = build_system(copy.deepcopy(base))
        calibrate_system(bsys)
        bcap = find_capacity(model, bsys, P.all_gpu("gpu"), wl, slo=slo,
                             n_requests=n_req, sim_kwargs=sim_kwargs,
                             name="baseline", verbose=not args.quick)
        out = BreakevenResult(param="bridge lanes (x800G)",
                              baseline_throughput=bcap.best.output_throughput,
                              baseline_tokens_per_kw=bcap.tokens_per_kw,
                              baseline_name=bsys.name)
        lanes = [1, 8, 96] if args.quick else [1, 2, 4, 8, 24, 48, 96, 192]
        for ln in lanes:
            c = copy.deepcopy(het)
            c["bridges"][0]["lanes"] = ln
            s = build_system(c)
            calibrate_system(s)
            cap = find_capacity(
                model, s, P.named("pd_a", gpu="gpu", brain="brain",
                                  crossing_dtype="fp8"),
                wl, slo=slo, n_requests=n_req, sim_kwargs=sim_kwargs,
                name=f"{ln} lanes", verbose=not args.quick)
            out.points.append(BreakevenPoint(
                value=ln, label=f"{ln}", rate=cap.max_rate,
                output_throughput=cap.best.output_throughput,
                avg_power=cap.best.avg_power,
                tokens_per_kw=cap.tokens_per_kw,
                energy_per_token=cap.best.energy_per_output_token,
                tdp=cap.tdp, capex=cap.capex,
                note=f"{ln*100} GB/s aggregate"))
        print("\n" + out.table())
        results["bridge_lanes"] = _pack(out)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_json(args.out, results)
    print(f"\n[saved] {args.out}")
    return 0


def _pack(r):
    return dict(
        param=r.param, baseline_name=r.baseline_name,
        baseline_throughput=r.baseline_throughput,
        baseline_tokens_per_kw=r.baseline_tokens_per_kw,
        throughput_parity=r.crossing("output_throughput"),
        efficiency_parity=r.crossing("tokens_per_kw"),
        throughput_2x=r.crossing("output_throughput", 2.0),
        efficiency_2x=r.crossing("tokens_per_kw", 2.0),
        points=[dict(value=p.value, label=p.label, rate=p.rate,
                     output_throughput=p.output_throughput,
                     avg_power=p.avg_power, tokens_per_kw=p.tokens_per_kw,
                     energy_per_token=p.energy_per_token,
                     tdp=p.tdp, capex=p.capex, note=p.note)
                for p in r.points])


if __name__ == "__main__":
    sys.exit(main())
