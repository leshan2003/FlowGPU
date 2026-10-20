"""Turn the hand-filled routing-skew assumption into a threshold.

The hot/cold tiering result rests on one number that has no measurement
behind it: ``hot_traffic`` (``p``), the share of token->expert assignments
that land on the pinned experts.  Everything so far used ``p=0.9``, which is
a guess.  This sweeps it instead, and reports where the tiered arm stops
paying for itself.

The comparison is against ``pd_a`` with the full stack pinned, at the same
``collective_overlap``.  ``hot_frac`` is held at 0.10, i.e. only the hottest
tenth of the expert stack is pinned and the rest is served from the GPU's
HBM -- the configuration the earlier probes found to be best.

Run it and read the crossing point, not the individual rows:

    python repro/skew_threshold_probe.py [co] [chips] [p ...]

Not part of the package API.  See ``repro/README.md``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowgpu.capacity import find_capacity              # noqa: E402
from flowgpu.sim.serving import SLO, WorkloadSpec       # noqa: E402

from hot_cold_capacity import build, make_policy        # noqa: E402

WL = WorkloadSpec(n_requests=256, prompt_len=1024, output_len=64,
                  prompt_sigma=0.4, output_sigma=0.5,
                  max_batch=64, max_running=64, chunk_tokens=8192, seed=11)
SLO_ = SLO(ttft=5.0, tpot=0.05)


def capacity(label, chips, hot_frac, hot_traffic, policy_name, co):
    m, s = build(chips, hot_frac=hot_frac, hot_traffic=hot_traffic)
    cap = find_capacity(m, s, make_policy(policy_name), WL, slo=SLO_,
                        target=0.95, n_requests=WL.n_requests, iters=6,
                        sim_kwargs=dict(collective_overlap=co, jitter=True,
                                        microbatches=1),
                        name=label)
    b = cap.best
    return (cap.max_rate, b.output_throughput if b else 0.0,
            b.tpot_p99 if b else 0.0,
            b.energy_per_output_token if b else 0.0, cap.warnings)


def main() -> int:
    co = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
    chips = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    ps = [float(x) for x in sys.argv[3:]] or [0.50, 0.70, 0.90, 0.99]

    print("=" * 84)
    print("Routing-skew threshold: at what `p` does pinning 10% of the")
    print("expert stack stop paying for itself?")
    print("=" * 84)
    print(f"  collective_overlap = {co}   brain chips = {chips}   "
          f"hot_frac = 0.10")
    print(f"  baseline: pd_a with the full stack pinned ({1024} chips)")
    print()

    base_rate, base_thr, _, base_j, _ = capacity(
        "baseline", 1024, 0.0, 0.0, "pd_a", co)
    print(f"  baseline pd_a @1024 : {base_rate:8.2f} req/s  "
          f"{base_thr:8.1f} tok/s  {base_j:6.2f} J/tok")
    print()

    hdr = (f"  {'p':>6} {'req/s':>9} {'vs base':>9} {'tok/s':>9} "
           f"{'TPOT p99':>10} {'J/tok':>8} {'verdict':>11}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for p in ps:
        rate, thr, tp99, j, warn = capacity(
            f"hot_cold p={p}", chips, 0.10, p, "hot_cold", co)
        ratio = rate / max(base_rate, 1e-9)
        verdict = ("beats base" if ratio > 1.02 else
                   "loses" if ratio < 0.98 else "level")
        print(f"  {p:>6.2f} {rate:>9.2f} {ratio:>8.2f}x {thr:>9.1f} "
              f"{tp99 * 1e3:>9.2f}ms {j:>8.2f} {verdict:>11}")
        for w in warn:
            print(f"         !! {w}")

    print()
    print(f"  A 'vs base' of 1.00 means {chips} brain chips carry the same")
    print(f"  request rate as {1024}.  Energy is the column that does not")
    print("  reverse: it favours the tiered arm at every p above the point")
    print("  where the cold traffic starts dominating.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
