#!/usr/bin/env python3
"""Regenerate every figure in docs/figures/ from the experiment configs."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowgpu.experiment import run_file
from flowgpu.hardware.registry import make_device
from flowgpu.power import calibrate
from flowgpu.report import plots
from flowgpu.workload import llm
from flowgpu.workload.models import get_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures")


def main():
    os.makedirs(FIG, exist_ok=True)
    made = []

    # --- 1-3: the regime study ---------------------------------------
    print("running regimes.yaml ...")
    exp = run_file(os.path.join(ROOT, "configs", "experiments",
                                "regimes.yaml"))
    made.append(plots.comparison_bars(
        exp.results, os.path.join(FIG, "regimes_bars.png"),
        "PD+A vs pure GPU across four workload regimes"))
    made.append(plots.time_attribution(
        exp.results, os.path.join(FIG, "regimes_time.png"),
        "Where the critical path goes"))
    made.append(plots.energy_attribution(
        exp.results, os.path.join(FIG, "regimes_energy.png"),
        "Where the energy goes"))

    # --- 4: roofline --------------------------------------------------
    print("building roofline ...")
    devs = {}
    for n in ("h100_sxm", "iluvatar_bi_v100", "groq_lpu_v1", "ipu_gc200",
              "lynxi_hp300_proj"):
        d = make_device(n)
        calibrate(d)
        devs[n] = d
    m = get_model("deepseek_v3", w_dtype="fp8")
    pts = []
    for label, g in (("prefill 4k", llm.build_prefill(m, 1, 4096, 0)),
                     ("decode B=1", llm.build_decode(m, 1, 4096)),
                     ("decode B=32", llm.build_decode(m, 32, 4096)),
                     ("decode B=256", llm.build_decode(m, 256, 4096))):
        by = sum(o.total_bytes for o in g.ops)
        ai = g.flops / by if by else 1
        pts.append((label, ai, min(devs["h100_sxm"].peak_flops,
                                   ai * 3.35e12)))
    made.append(plots.roofline(
        devs, pts, os.path.join(FIG, "roofline.png"),
        "Roofline: DeepSeek-V3 fp8 operating points (per-device)"))

    # --- 5: break-even ------------------------------------------------
    bpath = os.path.join(ROOT, "results", "breakeven.json")
    if os.path.exists(bpath):
        print("plotting break-even ...")
        from flowgpu.breakeven import BreakevenPoint, BreakevenResult
        raw = json.load(open(bpath))
        for key, d in raw.items():
            r = BreakevenResult(param=d["param"],
                                baseline_throughput=d["baseline_throughput"],
                                baseline_tokens_per_kw=d["baseline_tokens_per_kw"],
                                baseline_name=d["baseline_name"])
            r.points = [BreakevenPoint(
                value=p["value"], label=p["label"], rate=p["rate"],
                output_throughput=p["output_throughput"],
                avg_power=p["avg_power"], tokens_per_kw=p["tokens_per_kw"],
                energy_per_token=p["energy_per_token"], tdp=p["tdp"],
                capex=p["capex"], note=p.get("note", "")) for p in d["points"]]
            made.append(plots.breakeven_curve(
                r, os.path.join(FIG, f"breakeven_{key}.png")))

    # --- 6: capacity curves -------------------------------------------
    cpath = os.path.join(ROOT, "results", "grounded_capacity.json")
    if os.path.exists(cpath):
        print("plotting capacity curves ...")
        from types import SimpleNamespace
        raw = json.load(open(cpath))
        caps = {}
        for name, d in raw.items():
            caps[name] = SimpleNamespace(points=[
                SimpleNamespace(rate=p["rate"], attainment=p["attainment"],
                                output_throughput=p["output_throughput"])
                for p in d.get("curve", [])])
        caps = {k: v for k, v in caps.items() if v.points}
        if caps:
            made.append(plots.capacity_curves(
                caps, os.path.join(FIG, "capacity_curves.png"),
                "SLO-constrained capacity, DeepSeek-V3 on published hardware"))

    print("\nwrote:")
    for p in made:
        print("  " + os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
