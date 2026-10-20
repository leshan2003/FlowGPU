"""Result formatting: console tables, markdown, JSON."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass

from ..sim.serving import SLO, SimResult
from ..units import fmt_bytes, fmt_bw, fmt_energy, fmt_flops, fmt_time


def _f(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "-"
    return f"{x:.{nd}f}"


def table(rows: list, headers: list, align: str | None = None) -> str:
    cols = len(headers)
    w = [len(str(h)) for h in headers]
    srows = [[str(c) for c in r] for r in rows]
    for r in srows:
        for i in range(cols):
            w[i] = max(w[i], len(r[i]) if i < len(r) else 0)
    align = align or ("l" + "r" * (cols - 1))
    def fmt_row(cells):
        out = []
        for i, c in enumerate(cells):
            out.append(c.ljust(w[i]) if align[i] == "l" else c.rjust(w[i]))
        return "  ".join(out)
    sep = "  ".join("-" * x for x in w)
    return "\n".join([fmt_row([str(h) for h in headers]), sep]
                     + [fmt_row(r) for r in srows])


# =========================================================================
def summarize(res: SimResult, slo: SLO | None = None, title: str = "",
              verbose: bool = True) -> str:
    slo = slo or SLO()
    t, p, e = res.ttft_stats(), res.tpot_stats(), res.e2e_stats()
    L = []
    if title:
        L.append(f"=== {title} ===")
    m = res.meta
    L.append(f"model={m.get('model')}  system={m.get('system')}  "
             f"placement={m.get('placement')}  "
             f"tp={m.get('tp')} ep={m.get('ep')} mb={m.get('microbatches')}")
    L.append("")
    L.append("LATENCY")
    L.append(table(
        [["TTFT", fmt_time(t['mean']), fmt_time(t['p50']), fmt_time(t['p90']),
          fmt_time(t['p99'])],
         ["TPOT", fmt_time(p['mean']), fmt_time(p['p50']), fmt_time(p['p90']),
          fmt_time(p['p99'])],
         ["E2E",  fmt_time(e['mean']), fmt_time(e['p50']), fmt_time(e['p90']),
          fmt_time(e['p99'])]],
        ["metric", "mean", "p50", "p90", "p99"]))
    L.append("")
    L.append("THROUGHPUT & POWER")
    L.append(table([
        ["output tokens/s", _f(res.output_throughput, 1)],
        ["total tokens/s", _f(res.total_throughput, 1)],
        ["requests/s", _f(res.request_throughput, 3)],
        ["avg power (W)", _f(res.avg_power, 1)],
        ["energy / output token (J)", _f(res.energy_per_output_token, 4)],
        ["output tokens / joule", _f(res.tokens_per_joule, 3)],
        ["tokens/s per kW", _f(res.output_throughput /
                               max(res.avg_power / 1000, 1e-9), 1)],
        [f"SLO attainment (TTFT<{slo.ttft}s, TPOT<{slo.tpot*1000:.0f}ms)",
         f"{res.slo_attainment(slo)*100:.1f}%"],
        ["goodput (req/s)", _f(res.goodput(slo), 3)],
        ["wall time", fmt_time(res.wall_time)],
    ], ["metric", "value"]))

    if verbose:
        L.append("")
        L.append("TIME ATTRIBUTION (device-seconds on the critical path)")
        tot = sum(res.t_by_bound.values()) or 1.0
        L.append(table(
            [[k, fmt_time(v), f"{100*v/tot:5.1f}%"]
             for k, v in sorted(res.t_by_bound.items(), key=lambda x: -x[1])],
            ["bound", "time", "share"]))
        tot = sum(res.t_by_role.values()) or 1.0
        L.append("")
        L.append(table(
            [[k, fmt_time(v), f"{100*v/tot:5.1f}%"]
             for k, v in sorted(res.t_by_role.items(), key=lambda x: -x[1])],
            ["role", "time", "share"]))
        L.append("")
        L.append("POOL UTILISATION")
        L.append(table(
            [[k, fmt_time(v), f"{100*v/max(res.wall_time,1e-12):5.1f}%",
              fmt_energy(res.e_by_pool.get(k, 0.0)),
              f"{100*res.e_by_pool.get(k,0.0)/max(res.energy,1e-12):5.1f}%"]
             for k, v in sorted(res.busy_by_pool.items(), key=lambda x: -x[1])],
            ["pool", "busy", "util", "energy", "share"]))
        L.append("")
        L.append("ENERGY BREAKDOWN")
        L.append(table(
            [[k, fmt_energy(v), f"{100*v/max(res.energy,1e-12):5.1f}%"]
             for k, v in sorted(res.e_by_source.items(), key=lambda x: -x[1])],
            ["source", "energy", "share"]))
        L.append("")
        L.append("MEMORY / TRAFFIC")
        L.append(table([
            ["KV capacity (tokens)", f"{res.kv_capacity_tokens:,.0f}"],
            ["peak KV in use (tokens)", f"{res.peak_kv_tokens:,.0f}"],
            ["bytes crossed between pools", fmt_bytes(res.bytes_crossed)],
            ["prefill tokens", f"{res.prefill_tokens:,}"],
            ["decode tokens", f"{res.decode_tokens:,}"],
            ["prefill iters / decode iters",
             f"{res.n_prefill_iters} / {res.n_decode_iters}"],
        ], ["metric", "value"]))
        wb = m.get("weights_by_pool", {})
        if wb:
            L.append("")
            L.append(table([[k, f"{v:.2f} GiB"] for k, v in wb.items()],
                           ["pool", "weights"]))

    if res.warnings:
        L.append("")
        L.append("WARNINGS")
        for x in res.warnings:
            L.append(f"  ! {x}")
    return "\n".join(L)


# =========================================================================
def compare(results: dict, slo: SLO | None = None,
            baseline: str | None = None) -> str:
    """Side-by-side comparison table of several named SimResults."""
    slo = slo or SLO()
    names = list(results)
    base = baseline or names[0]
    b = results[base]

    def rel(v, bv, higher_better=True):
        if bv in (0, None) or v is None or (isinstance(bv, float)
                                            and math.isnan(bv)):
            return ""
        r = v / bv if higher_better else bv / v
        return f" ({r:.2f}x)"

    rows = []
    metrics = [
        ("TTFT p50", lambda r: r.ttft_stats()["p50"], fmt_time, False),
        ("TTFT p99", lambda r: r.ttft_stats()["p99"], fmt_time, False),
        ("TPOT mean", lambda r: r.tpot_stats()["mean"], fmt_time, False),
        ("TPOT p99", lambda r: r.tpot_stats()["p99"], fmt_time, False),
        ("E2E p50", lambda r: r.e2e_stats()["p50"], fmt_time, False),
        ("output tok/s", lambda r: r.output_throughput,
         lambda v: f"{v:,.1f}", True),
        ("total tok/s", lambda r: r.total_throughput,
         lambda v: f"{v:,.1f}", True),
        ("avg power (W)", lambda r: r.avg_power, lambda v: f"{v:,.0f}", False),
        ("J / output token", lambda r: r.energy_per_output_token,
         lambda v: f"{v:.4f}", False),
        ("tok/s per kW", lambda r: r.output_throughput /
         max(r.avg_power / 1000, 1e-9), lambda v: f"{v:,.1f}", True),
        ("SLO attainment", lambda r: r.slo_attainment(slo),
         lambda v: f"{v*100:.1f}%", True),
        ("goodput req/s", lambda r: r.goodput(slo), lambda v: f"{v:.3f}", True),
    ]
    for label, fn, fmt, hb in metrics:
        bv = fn(b)
        row = [label]
        for n in names:
            v = fn(results[n])
            cell = fmt(v) if v is not None and not (
                isinstance(v, float) and math.isnan(v)) else "-"
            if n != base:
                cell += rel(v, bv, hb)
            row.append(cell)
        rows.append(row)
    out = [f"COMPARISON (baseline = {base})", ""]
    out.append(table(rows, ["metric"] + names))
    allw = []
    for n in names:
        for x in results[n].warnings:
            msg = f"[{n}] {x}"
            if msg not in allw:
                allw.append(msg)
    if allw:
        out.append("")
        out.append("WARNINGS")
        out += [f"  ! {x}" for x in allw]
    return "\n".join(out)


# =========================================================================
def system_report(system) -> str:
    L = [f"SYSTEM: {system.name}", ""]
    rows = []
    for name, p in system.pools.items():
        d = p.device
        rows.append([
            name, d.name, d.kind, p.count,
            f"tp{p.parallel.tp}/pp{p.parallel.pp}/ep{p.parallel.ep}"
            f"/dp{p.parallel.dp}",
            fmt_bytes(d.capacity), fmt_flops(d.peak_flops),
            f"{d.energy.tdp:.0f} W", f"{p.tdp/1000:.2f} kW",
            d.extra.get("confidence", "")])
    L.append(table(rows, ["pool", "device", "kind", "n", "parallel",
                          "mem/dev", "peak/dev", "TDP/dev", "pool TDP",
                          "spec conf"]))
    L.append("")
    L.append(f"total devices {system.total_devices}   "
             f"total memory {fmt_bytes(system.total_capacity)}   "
             f"system TDP {system.total_tdp/1000:.2f} kW (PUE {system.pue})")
    if system.bridges:
        L.append("")
        L.append(table(
            [[f"{a} <-> {bb}", lk.name, fmt_bw(lk.eff_bw),
              f"{lk.latency*1e6:.2f} us", f"{lk.energy_per_byte*1e12:.1f} pJ/B"]
             for (a, bb), lk in system.bridges.items()],
            ["bridge", "fabric", "usable BW", "latency", "energy"]))
    est = [p.device.name for p in system.pools.values()
           if p.device.extra.get("confidence") == "estimate"]
    if est:
        L.append("")
        L.append("NOTE: specs for " + ", ".join(sorted(set(est))) +
                 " are ESTIMATES or PROJECTIONS, not vendor datasheets.")
    return "\n".join(L)


# =========================================================================
def to_dict(res: SimResult, slo: SLO | None = None) -> dict:
    slo = slo or SLO()
    return dict(
        meta=res.meta,
        ttft=res.ttft_stats(), tpot=res.tpot_stats(), e2e=res.e2e_stats(),
        output_throughput=res.output_throughput,
        total_throughput=res.total_throughput,
        request_throughput=res.request_throughput,
        avg_power=res.avg_power,
        energy=res.energy, device_energy=res.device_energy,
        idle_energy=res.idle_energy, host_energy=res.host_energy,
        energy_per_output_token=res.energy_per_output_token,
        tokens_per_joule=res.tokens_per_joule,
        slo_attainment=res.slo_attainment(slo), goodput=res.goodput(slo),
        wall_time=res.wall_time,
        t_by_bound=res.t_by_bound, t_by_role=res.t_by_role,
        busy_by_pool=res.busy_by_pool, e_by_pool=res.e_by_pool,
        e_by_source=res.e_by_source,
        bytes_crossed=res.bytes_crossed,
        prefill_tokens=res.prefill_tokens, decode_tokens=res.decode_tokens,
        kv_capacity_tokens=res.kv_capacity_tokens,
        peak_kv_tokens=res.peak_kv_tokens,
        warnings=res.warnings,
    )


def save_json(path: str, payload) -> None:
    def default(o):
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return str(o)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=default, ensure_ascii=False)
