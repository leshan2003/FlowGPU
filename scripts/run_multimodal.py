#!/usr/bin/env python3
"""Non-LLM workloads: VLM vision tower, diffusion, CNN.

The LLM case is the hard one for SRAM-resident weights, because the weight
stack is enormous and the work per weight byte is one token.  The workloads
here have the opposite shape:

* **Diffusion** reuses the *same* weights for every one of 20-50 denoising
  steps.  A dataflow chip loads them once; a GPU re-streams them per step.
* **ViT / CNN** are compute-dense with modest weights and high activation
  reuse, so a weight-stationary array runs near peak.

If the heterogeneous architecture is going to win anywhere, it is here --
and the simulator should be able to say so as clearly as it says the
opposite for a 671 B MoE.

Usage:  python3 scripts/run_multimodal.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowgpu.mapping import placement as P
from flowgpu.power import calibrate_system
from flowgpu.report.report import save_json, table
from flowgpu.sim import oneshot
from flowgpu.system import build_system
from flowgpu.units import fmt_energy, fmt_time
from flowgpu.workload.models import get_model


def gpu_sys(device="h100_sxm", n=8):
    return build_system(dict(name=f"{device}x{n}", pools={
        "gpu": dict(device=device, count=n, parallel=dict(tp=n),
                    devices_per_board=8, host_overhead="500 W")}))


def dataflow_sys(device="ipu_gc200", n=16, per_board=4, host="250 W"):
    return build_system(dict(name=f"{device}x{n}", pools={
        "brain": dict(device=device, count=n, parallel=dict(tp=n),
                      devices_per_board=per_board,
                      host_overhead=host)}))


def hetero_sys(gpu="h100_sxm", ng=4, df="ipu_gc200", nd=16):
    return build_system(dict(name=f"{gpu}x{ng}+{df}x{nd}", pools={
        "gpu": dict(device=gpu, count=ng, parallel=dict(tp=ng),
                    devices_per_board=8, host_overhead="500 W"),
        "brain": dict(device=df, count=nd, parallel=dict(tp=nd),
                      devices_per_board=4, host_overhead="250 W"),
    }, bridges=[dict(between=["gpu", "brain"], fabric="pcie5_x16", lanes=4)]))


def row(label, r):
    return [label, fmt_time(r.latency), f"{r.throughput:,.3f}",
            f"{r.avg_power:,.0f}", fmt_energy(r.energy_per_item),
            f"{r.achieved_flops/1e12:,.1f}",
            max(r.t_by_bound, key=r.t_by_bound.get) if r.t_by_bound else "-"]


HDR = ["system", "latency", "items/s", "W", "energy/item", "TFLOP/s", "bound"]


def main():
    out = {}

    # ---------------------------------------------------------- diffusion
    print("=" * 74)
    print("DIFFUSION: FLUX.1-dev, 1024x1024, 28 steps, CFG")
    print("  The same 12 B of weights is read once per step.  On a GPU that")
    print("  is 28 full HBM sweeps; on SRAM-resident hardware it is zero.")
    print("=" * 74)
    spec = get_model("flux_dev", w_dtype="bf16")
    rows = []
    for label, sysm, pol in (
            ("8x H100", gpu_sys("h100_sxm", 8), P.all_gpu("gpu")),
            ("1x H100", gpu_sys("h100_sxm", 1), P.all_gpu("gpu")),
            ("16x IPU GC200", dataflow_sys("ipu_gc200", 16),
             P.all_dataflow("brain")),
            ("32x Groq LPU", dataflow_sys("groq_lpu_v1", 32, 8),
             P.all_dataflow("brain")),
            ("256x Lynxi(proj)", dataflow_sys("lynxi_hp300_proj", 256, 16),
             P.all_dataflow("brain")),
    ):
        calibrate_system(sysm)
        r = oneshot.run_diffusion(spec, sysm, pol, batch=1)
        rows.append(row(label, r))
        out[f"diffusion/{label}"] = _pack(r)
    print(table(rows, HDR))

    # ---------------------------------------------------------------- CNN
    print()
    print("=" * 74)
    print("CNN: ResNet-50 inference, batch 64")
    print("=" * 74)
    spec = get_model("resnet50", w_dtype="int8", a_dtype="int8")
    rows = []
    for label, sysm, pol in (
            # host overhead is set per part class: a datacentre card sits in
            # a ~500 W server, an embedded module in a ~10 W carrier board.
            # Using one number for both would make the edge parts look
            # absurdly inefficient for reasons that have nothing to do with
            # the silicon.
            ("1x H100", gpu_sys("h100_sxm", 1), P.all_gpu("gpu")),
            ("1x Jetson Orin", build_system(dict(name="orin", pools={
                "gpu": dict(device="jetson_agx_orin_64", count=1,
                            parallel=dict(tp=1), devices_per_board=1,
                            host_overhead="10 W")})), P.all_gpu("gpu")),
            ("1x IPU GC200", dataflow_sys("ipu_gc200", 1, 4, "250 W"),
             P.all_dataflow("brain")),
            ("1x Lynxi KA200", dataflow_sys("lynxi_ka200", 1, 2, "10 W"),
             P.all_dataflow("brain")),
            ("1x Groq LPU", dataflow_sys("groq_lpu_v1", 1, 8, "250 W"),
             P.all_dataflow("brain")),
    ):
        calibrate_system(sysm)
        r = oneshot.run_cnn(spec, sysm, pol, batch=64)
        rows.append(row(label, r))
        out[f"cnn/{label}"] = _pack(r)
    print(table(rows, HDR))

    # ---------------------------------------------------------------- VLM
    print()
    print("=" * 74)
    print("VLM: Qwen2.5-VL-7B prefill, 1 image (1024px) + 256 text tokens")
    print("  The split under test: ViT encoder on the dataflow chip, LLM")
    print("  decoder on the GPU -- each on the hardware that suits it.")
    print("=" * 74)
    spec = get_model("qwen2_5_vl_7b", w_dtype="bf16")
    rows = []
    for label, sysm, pol in (
            ("4x H100 (all GPU)", gpu_sys("h100_sxm", 4), P.all_gpu("gpu")),
            ("16x IPU (all dataflow)", dataflow_sys("ipu_gc200", 16),
             P.all_dataflow("brain")),
            ("4x H100 + 16x IPU, ViT->IPU",
             hetero_sys("h100_sxm", 4, "ipu_gc200", 16),
             P.PlacementPolicy(
                 name="vit_offload",
                 rules=[P.Rule(pool="brain", role="vision")],
                 default_pool="gpu",
                 notes="vision tower on the dataflow chip")),
            ("4x H100 + 16x IPU, FFN->IPU",
             hetero_sys("h100_sxm", 4, "ipu_gc200", 16),
             P.pd_a("gpu", "brain", crossing_dtype="fp8")),
    ):
        calibrate_system(sysm)
        r = oneshot.run_vlm_prefill(spec, sysm, pol, batch=1,
                                    text_tokens=256, n_images=1)
        rows.append(row(label, r))
        out[f"vlm/{label}"] = _pack(r)
    print(table(rows, HDR))

    os.makedirs("results", exist_ok=True)
    save_json("results/multimodal.json", out)
    print("\n[saved] results/multimodal.json")
    return 0


def _pack(r):
    return dict(latency=r.latency, throughput=r.throughput,
                avg_power=r.avg_power, energy_per_item=r.energy_per_item,
                achieved_flops=r.achieved_flops, flops=r.flops,
                t_by_bound=r.t_by_bound, busy_by_pool=r.busy_by_pool,
                e_by_source=r.e_by_source, bytes_crossed=r.bytes_crossed,
                warnings=r.warnings, meta=r.meta)


if __name__ == "__main__":
    sys.exit(main())
