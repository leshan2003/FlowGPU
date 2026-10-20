"""Inverse analysis: what would the dataflow chip have to be?

Running one configuration tells you whether *that* configuration wins.  It
does not tell you what the architecture needs in order to win, which is the
more useful question when the part you are evaluating has no published specs.

:func:`sweep_device` varies one device attribute across a grid and reports the
heterogeneous system's SLO-constrained capacity and energy efficiency relative
to a pure-GPU baseline, so you can read off the crossing point directly:

    SRAM per chip   tok/s vs GPU   tok/s/kW vs GPU
    ------------    ------------   ---------------
     256 MB            0.41x            0.04x
     512 MB            0.78x            0.11x
       1 GB            1.42x            0.33x      <- throughput crossover
       4 GB            2.10x            0.95x
       8 GB            2.14x            1.71x      <- efficiency crossover

The output is deliberately two-sided: a chip can cross on throughput long
before it crosses on energy, and the reference claim asserts *both*.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .capacity import find_capacity
from .mapping import placement as P
from .power import calibrate_system
from .system import build_system
from .units import parse


@dataclass
class BreakevenPoint:
    value: float
    label: str
    rate: float
    output_throughput: float
    avg_power: float
    tokens_per_kw: float
    energy_per_token: float
    tdp: float
    capex: float
    feasible: bool = True
    note: str = ""


@dataclass
class BreakevenResult:
    param: str
    baseline_throughput: float
    baseline_tokens_per_kw: float
    baseline_name: str
    points: list = field(default_factory=list)

    def crossing(self, attr: str, ratio: float = 1.0) -> float | None:
        """Linearly interpolate the value at which ``attr`` ratio hits 1."""
        base = (self.baseline_throughput if attr == "output_throughput"
                else self.baseline_tokens_per_kw)
        prev = None
        for p in self.points:
            r = getattr(p, attr) / base if base else 0.0
            if prev is not None and (prev[1] - ratio) * (r - ratio) <= 0 \
                    and prev[1] != r:
                t = (ratio - prev[1]) / (r - prev[1])
                return prev[0] + t * (p.value - prev[0])
            prev = (p.value, r)
        return None

    def table(self) -> str:
        from .report.report import table
        rows = []
        for p in self.points:
            rows.append([
                p.label,
                f"{p.rate:.3f}",
                f"{p.output_throughput:,.1f}",
                f"{p.output_throughput / max(self.baseline_throughput, 1e-9):.2f}x",
                f"{p.avg_power / 1000:,.1f}",
                f"{p.tokens_per_kw:,.2f}",
                f"{p.tokens_per_kw / max(self.baseline_tokens_per_kw, 1e-9):.2f}x",
                p.note,
            ])
        out = [f"BREAK-EVEN SWEEP: {self.param}",
               f"baseline = {self.baseline_name}: "
               f"{self.baseline_throughput:,.1f} tok/s, "
               f"{self.baseline_tokens_per_kw:,.2f} tok/s/kW", ""]
        out.append(table(rows, [self.param, "req/s", "tok/s", "vs GPU",
                                "kW", "tok/s/kW", "vs GPU", "note"]))
        xt = self.crossing("output_throughput")
        xe = self.crossing("tokens_per_kw")
        out.append("")
        out.append(f"  throughput parity at {self.param} = "
                   f"{xt:,.4g}" if xt is not None
                   else f"  throughput parity not reached in this range")
        out.append(f"  energy-efficiency parity at {self.param} = "
                   f"{xe:,.4g}" if xe is not None
                   else f"  energy-efficiency parity not reached in this range")
        xt2 = self.crossing("output_throughput", 2.0)
        xe2 = self.crossing("tokens_per_kw", 2.0)
        out.append(f"  the claimed 2x throughput needs {self.param} >= "
                   f"{xt2:,.4g}" if xt2 is not None
                   else "  the claimed 2x throughput is not reached in this range")
        out.append(f"  the claimed 2x efficiency needs {self.param} >= "
                   f"{xe2:,.4g}" if xe2 is not None
                   else "  the claimed 2x efficiency is not reached in this range")
        return "\n".join(out)


# =========================================================================
def _set_override(syscfg: dict, pool: str, path: str, value):
    ov = syscfg["pools"][pool].setdefault("overrides", {})
    cur = ov
    parts = path.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def sweep_device(model, base_system_cfg: dict, hetero_system_cfg: dict,
                 policy_cfg, workload, slo, values: list, pool: str = "brain",
                 param: str = "sram_per_core", sim_kwargs: dict | None = None,
                 label_fmt=None, baseline_policy=None, n_requests: int = 128,
                 rescale_count: bool = True, verbose: bool = True
                 ) -> BreakevenResult:
    """Sweep one attribute of the dataflow device and find the crossings.

    ``rescale_count`` keeps the pool's *aggregate* SRAM roughly constant when
    sweeping ``sram_per_core``, so the sweep answers "denser chips" rather
    than "more chips" -- two very different questions.  Set it False to sweep
    total capacity instead.
    """
    sim_kwargs = sim_kwargs or {}
    label_fmt = label_fmt or (lambda v: f"{v:,.4g}")

    # --- baseline -------------------------------------------------------
    bsys = build_system(copy.deepcopy(base_system_cfg))
    calibrate_system(bsys)
    bpol = baseline_policy or P.all_gpu(next(iter(bsys.pools)))
    if verbose:
        print(f"--- baseline: {bsys.name} ---")
    bcap = find_capacity(model, bsys, bpol, workload, slo=slo,
                         n_requests=n_requests, sim_kwargs=sim_kwargs,
                         name="baseline", verbose=verbose)

    out = BreakevenResult(
        param=param,
        baseline_throughput=bcap.best.output_throughput,
        baseline_tokens_per_kw=bcap.tokens_per_kw,
        baseline_name=bsys.name)

    base_count = hetero_system_cfg["pools"][pool].get("count", 1)
    base_val = None

    for v in values:
        cfg = copy.deepcopy(hetero_system_cfg)
        val = parse(v) if isinstance(v, str) else v
        _set_override(cfg, pool, param, v if isinstance(v, str) else val)
        if rescale_count and param == "sram_per_core":
            if base_val is None:
                from .hardware.registry import get_spec
                _, spec = get_spec(cfg["pools"][pool]["device"])
                base_val = parse(spec.get("sram_per_core", val))
            n = max(1, int(round(base_count * base_val / val)))
            cfg["pools"][pool]["count"] = n
            ep = cfg["pools"][pool].get("parallel", {})
            if ep.get("ep", 1) > 1:
                ep["ep"] = n
        hsys = build_system(cfg)
        calibrate_system(hsys)
        pools = list(hsys.pools)
        pol = P.named(policy_cfg["strategy"],
                      **{k: x for k, x in policy_cfg.items()
                         if k != "strategy"})
        if verbose:
            print(f"--- {param}={label_fmt(val)} "
                  f"({hsys.pools[pool].count} chips) ---")
        cap = find_capacity(model, hsys, pol, workload, slo=slo,
                            n_requests=n_requests, sim_kwargs=sim_kwargs,
                            name=label_fmt(val), verbose=verbose)
        note = ""
        dev = hsys.pools[pool].device
        if hasattr(dev, "die_area_check"):
            chk = dev.die_area_check()
            if not chk["plausible"]:
                note = f"SRAM = {chk['sram_area_mm2']:.0f} mm2 " \
                       f"({chk['dies_needed']:.1f} reticles) -- not one die"
        if any("SLO never met" in w for w in cap.warnings):
            note = (note + "; " if note else "") + "SLO never met"
        out.points.append(BreakevenPoint(
            value=val, label=label_fmt(val), rate=cap.max_rate,
            output_throughput=cap.best.output_throughput,
            avg_power=cap.best.avg_power,
            tokens_per_kw=cap.tokens_per_kw,
            energy_per_token=cap.best.energy_per_output_token,
            tdp=cap.tdp, capex=cap.capex, note=note))
    return out
