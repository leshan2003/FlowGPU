"""Decompose the micro-batch trade, and sweep the overlap that does pay.

Two questions, and they have different answers:

1. **Does micro-batch pipelining help?**  It is two effects multiplied
   together, and reporting only the net hides which one is winning:

       overlap_gain   = (m * one_microbatch) / pipelined_time
       microbatch_tax = (m * one_microbatch) / serial_time
       net            = overlap_gain / microbatch_tax

   The tax exists because a weight-bound decode step does not get m times
   cheaper when you cut it into m pieces -- the weights still stream in full.
   The gain exists because with >=2 micro-batches the two pools can work at
   once, so the steady-state period is the *busiest* pool, not their sum.

2. **How much is left in collective overlap?**  The MoE all-to-all is the
   single largest term on the expert-parallel pool, and
   ``collective_overlap`` is how much of it a production engine hides behind
   compute (DeepSeek DualPipe, SGLang two-batch overlap).  This sweeps it.

The arithmetic in section A mirrors ``execute_pipelined`` deliberately: it
needs the two halves separately, and the package only exposes the product.
``_scale_graph`` is private for the same reason it is imported here -- this
is a diagnostic, not package API.

    python repro/pipeline_overlap_probe.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowgpu.mapping import placement as P            # noqa: E402
from flowgpu.power import calibrate_system            # noqa: E402
from flowgpu.sim.executor import _scale_graph, execute  # noqa: E402
from flowgpu.system import build_system               # noqa: E402
from flowgpu.units import fmt_bw                      # noqa: E402
from flowgpu.workload import llm                      # noqa: E402
from flowgpu.workload.models import get_model         # noqa: E402

GPU = 16
CTX = 4096


def build(batch, chips=256, hot_frac=0.0, hot_traffic=0.0):
    m = get_model("deepseek_v3", w_dtype="fp8", a_dtype="bf16",
                  kv_dtype="bf16", comm_dtype="fp8")
    if hot_frac:
        m.extra["hot_frac"] = hot_frac
        m.extra["hot_traffic"] = hot_traffic or hot_frac
    s = build_system(dict(
        name="h100x16 + brain",
        pools={
            "gpu": dict(device="h100_sxm", count=GPU, devices_per_board=8,
                        parallel=dict(tp=8, pp=1, ep=2, dp=1),
                        scaleup="nvlink4", scaleout="ib_ndr"),
            "brain": dict(device="lynxi_hp300_proj", count=chips,
                          devices_per_board=8,
                          parallel=dict(tp=1, pp=1, ep=chips, dp=1),
                          scaleup="custom_serdes_800g",
                          scaleout="roce_400g"),
        },
        bridges=[dict(between=["gpu", "brain"], fabric="pcie5_x16")],
    ))
    calibrate_system(s)
    # ffn_tp_world=1: the FFN/MoE lands on the tp=1 brain pool, so there is
    # no tensor-parallel tail all-reduce to pay for.
    g = llm.build_decode(m, batch=batch, ctx=CTX, ep_world=chips,
                         tp_world=8, ffn_tp_world=1)
    return m, s, g


ARMS = [
    ("pd_a      @ 256", 256, 0.0, 0.0, "pd_a"),
    ("hot_cold  @ 256", 256, 0.1, 0.9, "hot_cold"),
]


def policy_for(name):
    return (P.pd_a("gpu", "brain", crossing_dtype="fp8") if name == "pd_a"
            else P.hot_cold("gpu", "brain", crossing_dtype="fp8"))


def run(batch, chips, hf, p, pol_name, m, co):
    m_, s, g = build(batch, chips, hot_frac=hf, hot_traffic=p)
    pol = policy_for(pol_name)
    pol.annotate(g, 61)
    for pool in s.pools.values():          # plan residency like serving does
        pool.plan(0.0, 0.0, 0.0)
    serial = execute(g, s, pol, n_layers=61, collective_overlap=co)
    if m <= 1:
        return serial.time, serial.time, 1.0, 1.0, serial
    mb = _scale_graph(g, 1.0 / m)
    sub = execute(mb, s, pol, n_layers=61, collective_overlap=co)
    period = max(list(sub.busy_by_pool.values()) + [0.0])
    period = max(period, sub.t_bridge + sub.t_collective)
    pipelined = max(sub.time, sub.time - period + m * period)
    total = m * sub.time
    return serial.time, pipelined, total / serial.time, total / pipelined, sub


def main() -> int:
    print("=" * 86)
    print("A. The micro-batch trade, split into its two halves")
    print("=" * 86)
    print("  net = overlap_gain / microbatch_tax   (all three relative)")
    print("  tax > 1 means splitting the step made the total work larger,")
    print("  because a weight-bound decode step re-streams the weights per")
    print("  piece.  gain > 1 means the two pools actually overlapped.")
    print()

    for label, chips, hf, p, pol in ARMS:
        hdr = (f"  --- {label} ---\n"
               f"  {'batch':>6} {'m':>3} {'serial ms':>10} "
               f"{'pipelined':>10} {'tax':>6} {'gain':>6} {'net':>6} "
               f"{'verdict':>9}")
        print(hdr)
        for batch in (8, 64, 256):
            for m in (2, 4):
                ser, pip, tax, gain, sub = run(batch, chips, hf, p, pol, m,
                                               0.6)
                net = gain / tax
                verdict = ("faster" if net > 1.02 else
                           "worse" if net < 0.98 else "wash")
                print(f"  {batch:>6} {m:>3} {ser * 1e3:>10.3f} "
                      f"{pip * 1e3:>10.3f} {tax:>6.2f} {gain:>6.2f} "
                      f"{net:>6.2f} {verdict:>9}")
            print()

    print("=" * 86)
    print("B. Collective overlap: how much is still on the table")
    print("=" * 86)
    print("  `collective_overlap` is the fraction of the MoE all-to-all that")
    print("  a production engine hides behind compute.  Its worth is bounded")
    print("  by how much of the brain pool's step that collective *is*.")
    print()
    hdr = (f"  {'arm':>16} {'batch':>6} {'co':>5} {'step ms':>9} "
           f"{'collective':>11} {'coll. ms':>9} {'speedup vs co=0':>16}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, chips, hf, p, pol in ARMS:
        for batch in (8, 64):
            base = None
            for co in (0.0, 0.3, 0.6, 0.75, 0.9):
                ser, _, tax, gain, res = run(batch, chips, hf, p, pol, 1, co)
                if base is None:
                    base = ser
                share = res.t_collective / max(ser, 1e-12)
                print(f"  {label:>16} {batch:>6} {co:>5.2f} "
                      f"{ser * 1e3:>9.3f} {share:>10.1%} "
                      f"{res.t_collective * 1e3:>9.3f} {base / ser:>15.3f}x")
            print()

    s = build(8)[1]
    print("  bridge for reference: "
          f"{fmt_bw(s.bridge('gpu', 'brain').eff_bw)} usable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
