"""Matplotlib figures.

Deliberately plain: one idea per figure, direct labels rather than legends
where it fits, and no chart junk.  Colours come from a small colour-blind-safe
categorical set and are used consistently across figures (GPU-only is always
the same colour, PD+A always another).
"""

from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# colour-blind-safe categorical palette (Okabe-Ito ordering)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
           "#56B4E9", "#7F7F7F", "#000000"]
GRID = dict(color="#D9D9D9", linewidth=0.7)


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8, length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8A8A8A")
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# =========================================================================
def _pair(names):
    """Group arms into (regime, {variant: name}) when they share a suffix.

    Arms called ``gpu_chat`` / ``pda_chat`` are two treatments of the same
    regime, and a reader wants them adjacent and consistently coloured, not
    scattered across a rainbow.
    """
    groups, order, variants = {}, [], []
    for n in names:
        if "_" in n:
            var, _, reg = n.partition("_")
        else:
            var, reg = n, n
        if reg not in groups:
            groups[reg] = {}
            order.append(reg)
        groups[reg][var] = n
        if var not in variants:
            variants.append(var)
    if len(variants) < 2 or any(len(g) != len(variants) for g in
                                groups.values()):
        return None, None, None
    return order, variants, groups


def comparison_bars(results: dict, path: str, title: str = "") -> str:
    """Throughput / latency / energy side-by-side for several arms."""
    names = list(results)
    metrics = [
        ("output tok/s", lambda r: r.output_throughput, False),
        ("TPOT p99 (ms)", lambda r: r.tpot_stats()["p99"] * 1e3, True),
        ("TTFT p99 (s)", lambda r: r.ttft_stats()["p99"], True),
        ("J / output token", lambda r: r.energy_per_output_token, True),
    ]
    regimes, variants, groups = _pair(names)
    grouped = regimes is not None

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(3.5 * len(metrics), 3.8))
    for ax, (label, fn, lower_better) in zip(axes, metrics):
        vals_all = [fn(results[n]) for n in names]
        finite = [v for v in vals_all if v == v and v > 0]
        # a wide dynamic range makes a linear axis useless, and adjacent
        # near-equal bars make horizontal labels collide -- both are fixed
        # by switching to log and rotating the annotations
        logscale = bool(finite) and max(finite) / min(finite) > 20

        if grouped:
            w = 0.8 / len(variants)
            for vi, var in enumerate(variants):
                vals = [fn(results[groups[r][var]]) for r in regimes]
                xs = [i + (vi - (len(variants) - 1) / 2) * w
                      for i in range(len(regimes))]
                bars = ax.bar(xs, vals, width=w * 0.92,
                              color=PALETTE[vi % len(PALETTE)],
                              label=var, zorder=3)
                for b, v in zip(bars, vals):
                    if v == v:
                        ax.annotate(f"{v:,.3g}",
                                    (b.get_x() + b.get_width() / 2, v),
                                    textcoords="offset points",
                                    xytext=(0, 3), ha="center", fontsize=7,
                                    rotation=90 if logscale else 0)
            ax.set_xticks(range(len(regimes)))
            ax.set_xticklabels(regimes, rotation=22, ha="right", fontsize=8)
        else:
            bars = ax.bar(range(len(names)), vals_all,
                          color=[PALETTE[i % len(PALETTE)]
                                 for i in range(len(names))], width=0.68,
                          zorder=3)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=28, ha="right", fontsize=7.5)
            for b, v in zip(bars, vals_all):
                if v == v:
                    ax.annotate(f"{v:,.3g}",
                                (b.get_x() + b.get_width() / 2, v),
                                textcoords="offset points", xytext=(0, 3),
                                ha="center", fontsize=7)
        _style(ax, label)
        if logscale:
            ax.set_yscale("log")
            ax.set_ylim(min(finite) * 0.45, max(finite) * 4.5)
        else:
            top = max(finite) if finite else 1
            ax.set_ylim(0, top * 1.28)
        if lower_better:
            ax.text(0.99, 0.985, "lower is better", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color="#6B6B6B")
    if grouped:
        axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    if title:
        fig.suptitle(title, fontsize=12, x=0.01, ha="left")
        fig.subplots_adjust(top=0.82)
    return _save(fig, path)


def time_attribution(results: dict, path: str, title: str = "") -> str:
    """Where the critical path goes, per arm."""
    names = list(results)
    keys = []
    for r in results.values():
        for k in r.t_by_bound:
            if k not in keys:
                keys.append(k)
    order = [k for k in ("compute", "memory", "collective", "bridge",
                         "network", "latency", "pipeline_fill") if k in keys]
    order += [k for k in keys if k not in order]

    fig, ax = plt.subplots(figsize=(1.7 * len(names) + 3.2, 3.8))
    bottom = [0.0] * len(names)
    for i, k in enumerate(order):
        vals = []
        for n in names:
            r = results[n]
            tot = sum(r.t_by_bound.values()) or 1.0
            vals.append(100 * r.t_by_bound.get(k, 0.0) / tot)
        ax.bar(range(len(names)), vals, bottom=bottom, width=0.62,
               color=PALETTE[i % len(PALETTE)], label=k)
        for j, (v, b) in enumerate(zip(vals, bottom)):
            if v > 6:
                ax.text(j, b + v / 2, f"{k}\n{v:.0f}%", ha="center",
                        va="center", fontsize=7,
                        color="white" if i < 4 else "black")
        bottom = [b + v for b, v in zip(bottom, vals)]
    _style(ax, title or "Critical-path time attribution", "",
           "share of critical path (%)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7.5, frameon=False, ncol=1,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    return _save(fig, path)


def energy_attribution(results: dict, path: str, title: str = "") -> str:
    names = list(results)
    keys = []
    for r in results.values():
        for k in r.e_by_source:
            if k not in keys:
                keys.append(k)
    keys.sort(key=lambda k: -sum(r.e_by_source.get(k, 0)
                                 for r in results.values()))
    fig, ax = plt.subplots(figsize=(1.7 * len(names) + 3.2, 3.8))
    bottom = [0.0] * len(names)
    for i, k in enumerate(keys[:8]):
        vals = [100 * results[n].e_by_source.get(k, 0.0) /
                max(results[n].energy, 1e-9) for n in names]
        ax.bar(range(len(names)), vals, bottom=bottom, width=0.62,
               color=PALETTE[i % len(PALETTE)], label=k)
        bottom = [b + v for b, v in zip(bottom, vals)]
    _style(ax, title or "Energy attribution", "", "share of energy (%)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="center left",
              bbox_to_anchor=(1.01, 0.5))
    return _save(fig, path)


def capacity_curves(caps: dict, path: str, title: str = "") -> str:
    """Attainment and throughput vs offered load, per arm."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for i, (name, r) in enumerate(caps.items()):
        pts = sorted(r.points, key=lambda p: p.rate)
        c = PALETTE[i % len(PALETTE)]
        a1.plot([p.rate for p in pts], [100 * p.attainment for p in pts],
                "o-", color=c, ms=3.5, lw=1.6, label=name)
        a2.plot([p.rate for p in pts],
                [p.output_throughput for p in pts], "o-", color=c,
                ms=3.5, lw=1.6, label=name)
    a1.axhline(95, color="#8A8A8A", ls="--", lw=1)
    a1.text(a1.get_xlim()[1], 95.5, "95% SLO", ha="right", fontsize=7.5,
            color="#6B6B6B")
    _style(a1, "SLO attainment vs offered load", "offered rate (req/s)",
           "attainment (%)")
    _style(a2, "Sustained output throughput", "offered rate (req/s)",
           "output tokens/s")
    a1.set_xscale("log")
    a2.set_xscale("log")
    a1.legend(fontsize=7.5, frameon=False)
    if title:
        fig.suptitle(title, fontsize=12, x=0.01, ha="left")
        fig.subplots_adjust(top=0.84)
    return _save(fig, path)


def breakeven_curve(bres, path: str, title: str = "") -> str:
    """Relative throughput and efficiency vs the swept parameter."""
    pts = bres.points
    x = list(range(len(pts)))
    tp = [p.output_throughput / max(bres.baseline_throughput, 1e-9)
          for p in pts]
    ef = [p.tokens_per_kw / max(bres.baseline_tokens_per_kw, 1e-9)
          for p in pts]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(x, tp, "o-", color=PALETTE[0], lw=1.9, ms=4.5,
            label="throughput vs GPU baseline")
    ax.plot(x, ef, "s-", color=PALETTE[1], lw=1.9, ms=4.5,
            label="energy efficiency vs GPU baseline")
    ax.axhline(1.0, color="#8A8A8A", ls="--", lw=1)
    ax.axhline(2.0, color="#B00020", ls=":", lw=1.2)
    ax.text(len(pts) - 1, 2.08, "the claimed 2x", ha="right", va="bottom",
            fontsize=8, color="#B00020")
    ax.text(len(pts) - 1, 1.03, "parity", ha="right", va="bottom",
            fontsize=8, color="#6B6B6B")
    lo = min(tp + ef)
    ax.set_ylim(lo * 0.55, 3.2)
    _style(ax, title or f"Break-even in {bres.param}", bres.param,
           "ratio to pure-GPU baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([p.label for p in pts], rotation=20, ha="right",
                       fontsize=8)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    # keep the legend clear of the 2x reference line at the top
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    return _save(fig, path)


def roofline(devices: dict, ops: list, path: str, title: str = "") -> str:
    """Classic roofline with the workload's operating points marked."""
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ai = [10 ** (i / 8.0) for i in range(-24, 33)]
    for i, (name, dev) in enumerate(devices.items()):
        peak = dev.peak_flops
        bw = (dev.level("hbm") or dev.main_memory).bandwidth
        if dev.kind == "dataflow":
            bw = dev.total_sram_bw
        y = [min(peak, bw * a) / 1e12 for a in ai]
        ax.plot(ai, y, color=PALETTE[i % len(PALETTE)], lw=2, label=name)
    for j, (label, a, perf) in enumerate(ops):
        ax.plot([a], [perf / 1e12], "o", color="#333333", ms=5)
        ax.annotate(label, (a, perf / 1e12), textcoords="offset points",
                    xytext=(6, 4), fontsize=7.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, title or "Roofline", "arithmetic intensity (FLOP/byte)",
           "achievable TFLOP/s")
    ax.grid(which="both", **GRID)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    return _save(fig, path)
