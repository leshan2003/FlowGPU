"""SLO-constrained capacity analysis.

"2x inference output" is only a meaningful claim if both systems are loaded
to the point where they *stop meeting their latency targets*.  Comparing two
systems at a fixed offered load measures the load, not the systems: if you
offer 2 req/s to a machine that can do 10 and to one that can do 3, both
report ~2 req/s of throughput and the interesting difference disappears.

:func:`find_capacity` binary-searches the Poisson arrival rate at which a
system's SLO attainment crosses ``target`` (default 95%), and reports the
sustained output-token throughput, power, and energy per token at that point.
That number -- *serving capacity subject to an SLO* -- is the one to compare.

It also reports the same at equal power and equal capital cost, since the
reference claim is specifically "vs a pure-GPU cluster of equivalent
investment".
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace

from .sim.serving import SLO, ServingSimulator, SimResult, WorkloadSpec


@dataclass
class CapacityPoint:
    rate: float
    attainment: float
    output_throughput: float
    request_throughput: float
    avg_power: float
    energy_per_output_token: float
    ttft_p99: float
    tpot_p99: float
    feasible: bool
    result: SimResult = None


@dataclass
class CapacityResult:
    name: str
    slo: SLO
    points: list = field(default_factory=list)
    best: CapacityPoint = None
    capex: float = 0.0
    tdp: float = 0.0
    warnings: list = field(default_factory=list)

    @property
    def max_rate(self) -> float:
        return self.best.rate if self.best else 0.0

    @property
    def tokens_per_kw(self) -> float:
        if not self.best or self.best.avg_power <= 0:
            return 0.0
        return self.best.output_throughput / (self.best.avg_power / 1000.0)

    @property
    def tokens_per_dollar_hour(self) -> float:
        """Output tokens per second per unit of capital -- a crude TCO proxy."""
        if not self.best or self.capex <= 0:
            return 0.0
        return self.best.output_throughput / self.capex


def _probe(make_sim, rate: float, slo: SLO, n_requests: int,
           warmup_frac: float = 0.2) -> CapacityPoint:
    sim = make_sim(rate, n_requests)
    res = sim.run()
    # discard the warm-up transient: the first requests see an empty machine
    done = [r for r in res.requests if r.finish >= 0]
    done.sort(key=lambda r: r.arrival)
    keep = done[int(len(done) * warmup_frac):] or done
    ok = sum(1 for r in keep
             if 0 <= r.ttft <= slo.ttft and (r.tpot < 0 or r.tpot <= slo.tpot))
    att = ok / len(keep) if keep else 0.0
    ttft = sorted(r.ttft for r in keep) or [float("nan")]
    tpot = sorted(r.tpot for r in keep if r.tpot >= 0) or [float("nan")]
    q = lambda v, p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]
    return CapacityPoint(
        rate=rate, attainment=att,
        output_throughput=res.output_throughput,
        request_throughput=res.request_throughput,
        avg_power=res.avg_power,
        energy_per_output_token=res.energy_per_output_token,
        ttft_p99=q(ttft, 0.99), tpot_p99=q(tpot, 0.99),
        feasible=att >= 0.0, result=res)


def find_capacity(model, system, policy, workload: WorkloadSpec,
                  slo: SLO | None = None, target: float = 0.95,
                  lo: float = 0.05, hi: float = 0.0, iters: int = 9,
                  n_requests: int = 192, sim_kwargs: dict | None = None,
                  name: str = "", verbose: bool = False) -> CapacityResult:
    """Binary-search the arrival rate at which SLO attainment hits ``target``."""
    slo = slo or SLO()
    sim_kwargs = sim_kwargs or {}

    def make_sim(rate, n):
        wl = replace(workload, arrival_rate=rate, n_requests=n)
        return ServingSimulator(model, system, policy, wl, **sim_kwargs)

    out = CapacityResult(name=name or policy.name, slo=slo,
                         capex=getattr(system, "total_capex", 0.0),
                         tdp=system.total_tdp)

    # --- bracket: grow hi until the SLO breaks --------------------------
    if hi <= 0:
        hi = max(lo * 2, 0.5)
        for _ in range(12):
            p = _probe(make_sim, hi, slo, n_requests)
            out.points.append(p)
            if verbose:
                print(f"  probe {hi:8.3f} req/s -> att {p.attainment*100:5.1f}%"
                      f"  {p.output_throughput:8.1f} tok/s")
            if p.attainment < target:
                break
            lo = hi
            hi *= 2.0
        else:
            out.warnings.append(
                f"never violated the SLO up to {hi:.2f} req/s; capacity is "
                f"bounded by the probe range, not the system")

    # --- bisect ---------------------------------------------------------
    best = None
    for p in out.points:
        if p.attainment >= target and (best is None or p.rate > best.rate):
            best = p
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p = _probe(make_sim, mid, slo, n_requests)
        out.points.append(p)
        if verbose:
            print(f"  bisect {mid:8.3f} req/s -> att {p.attainment*100:5.1f}%"
                  f"  {p.output_throughput:8.1f} tok/s")
        if p.attainment >= target:
            lo = mid
            if best is None or mid > best.rate:
                best = p
        else:
            hi = mid
        if (hi - lo) / max(hi, 1e-9) < 0.04:
            break

    if best is None:
        # the SLO was never met, even at the lowest probed rate.  Report the
        # lowest point and say so rather than silently presenting it as
        # "capacity".
        best = min(out.points, key=lambda p: p.rate)
        out.warnings.append(
            f"SLO never met: at the lowest probed rate ({best.rate:.3f} "
            f"req/s) attainment was only {best.attainment*100:.1f}%. The "
            f"reported capacity is an upper bound, not an achieved point.")
    out.best = best
    for w in (out.best.result.warnings if out.best.result else []):
        if w not in out.warnings:
            out.warnings.append(w)
    return out


# =========================================================================
def capacity_table(results: dict, baseline: str | None = None) -> str:
    from .report.report import table
    from .units import fmt_time
    names = list(results)
    base = baseline or names[0]
    b = results[base]

    def rel(v, bv, hb=True):
        if not bv or bv != bv:
            return ""
        return f" ({v/bv:.2f}x)" if hb else f" ({bv/v:.2f}x)" if v else ""

    rows = []
    specs = [
        ("max req/s @ SLO", lambda r: r.max_rate, lambda v: f"{v:.3f}", True),
        ("output tok/s @ SLO", lambda r: r.best.output_throughput,
         lambda v: f"{v:,.1f}", True),
        ("TTFT p99 there", lambda r: r.best.ttft_p99, fmt_time, False),
        ("TPOT p99 there", lambda r: r.best.tpot_p99, fmt_time, False),
        ("avg power (W)", lambda r: r.best.avg_power,
         lambda v: f"{v:,.0f}", False),
        ("J / output token", lambda r: r.best.energy_per_output_token,
         lambda v: f"{v:.3f}", False),
        ("tok/s per kW", lambda r: r.tokens_per_kw, lambda v: f"{v:,.1f}", True),
        ("system TDP (kW)", lambda r: r.tdp / 1000, lambda v: f"{v:,.1f}", False),
        ("capex", lambda r: r.capex, lambda v: f"{v:,.0f}", False),
        ("tok/s per Mcapex", lambda r: r.tokens_per_dollar_hour * 1e6,
         lambda v: f"{v:,.1f}", True),
    ]
    for label, fn, fmt, hb in specs:
        bv = fn(b)
        row = [label]
        for n in names:
            v = fn(results[n])
            cell = fmt(v) if v == v else "-"
            if n != base:
                cell += rel(v, bv, hb)
            row.append(cell)
        rows.append(row)
    out = [f"SLO-CONSTRAINED CAPACITY (baseline = {base}; "
           f"TTFT<{b.slo.ttft}s, TPOT<{b.slo.tpot*1000:.0f}ms, "
           f"95% attainment)", ""]
    out.append(table(rows, ["metric"] + names))
    allw = []
    for n in names:
        for w in results[n].warnings:
            m = f"[{n}] {w}"
            if m not in allw:
                allw.append(m)
    if allw:
        out += ["", "WARNINGS"] + [f"  ! {w}" for w in allw]
    return "\n".join(out)
