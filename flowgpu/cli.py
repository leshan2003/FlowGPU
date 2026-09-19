"""Command-line interface.

    python -m flowgpu devices                  list the device zoo
    python -m flowgpu device h100_sxm          show one device
    python -m flowgpu models                   list the model zoo
    python -m flowgpu model deepseek_v3        show one model
    python -m flowgpu fabrics                  list interconnect fabrics
    python -m flowgpu placements               list placement strategies
    python -m flowgpu run configs/experiments/pd_a.yaml
    python -m flowgpu sweep  ... --param bridge_bw --values ...
"""

from __future__ import annotations

import argparse
import os
import sys


def _add_common(p):
    p.add_argument("-o", "--out", help="write JSON results here")
    p.add_argument("-q", "--quiet", action="store_true")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("flowgpu", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("devices", help="list all devices")
    p = sub.add_parser("device", help="show one device")
    p.add_argument("name")

    sub.add_parser("models", help="list all models")
    p = sub.add_parser("model", help="show one model")
    p.add_argument("name")
    p.add_argument("--w-dtype", default=None)
    p.add_argument("--kv-dtype", default=None)

    sub.add_parser("fabrics", help="list interconnect fabrics")
    sub.add_parser("placements", help="list placement strategies")

    p = sub.add_parser("run", help="run an experiment config")
    p.add_argument("config")
    p.add_argument("--only", nargs="*", help="run only these arms")
    _add_common(p)

    p = sub.add_parser("sweep", help="sweep one parameter of an experiment")
    p.add_argument("config")
    p.add_argument("--param", required=True,
                   help="dotted path into the config. Global keys like "
                        "'workload.prompt_len' are SHADOWED by any per-run "
                        "override of the same key -- address those directly "
                        "as 'runs.<run-name>.workload.prompt_len'. System "
                        "fields work too: "
                        "'systems.hetero.pools.brain.overrides.tdp'.")
    p.add_argument("--values", nargs="+", required=True)
    p.add_argument("--metric", default="output_throughput",
                   help="SimResult attribute, or ttft_/tpot_/e2e_ + "
                        "mean|p50|p90|p99 (e.g. tpot_p99)")
    p.add_argument("--only", nargs="*", help="run only these arms")
    _add_common(p)

    p = sub.add_parser("capacity",
                       help="SLO-constrained serving capacity per arm")
    p.add_argument("config")
    p.add_argument("--only", nargs="*")
    p.add_argument("--target", type=float, default=0.95)
    p.add_argument("--requests", type=int, default=192)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("-v", "--verbose", action="store_true")
    _add_common(p)

    p = sub.add_parser("eda", help="run the DC/PrimeTime energy calibration")
    p.add_argument("--host", default=None,
                   help="EDA host (default: $FLOWGPU_EDA_HOST)")
    p.add_argument("--user", default=None,
                   help="username there (default: $FLOWGPU_EDA_USER or $USER)")
    p.add_argument("--key", default=None,
                   help="ssh identity file (default: $FLOWGPU_EDA_KEY)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--local-only", action="store_true",
                   help="only generate RTL and scripts, do not submit")
    p.add_argument("--skip-syn", action="store_true",
                   help="reuse the remote netlists (synthesis is the slow "
                        "stage; skip it when iterating on power setup)")
    p.add_argument("--out", default=None)

    args = ap.parse_args(argv)

    if args.cmd == "devices":
        return _devices()
    if args.cmd == "device":
        return _device(args.name)
    if args.cmd == "models":
        return _models()
    if args.cmd == "model":
        return _model(args)
    if args.cmd == "fabrics":
        return _fabrics()
    if args.cmd == "placements":
        return _placements()
    if args.cmd == "run":
        return _run(args)
    if args.cmd == "sweep":
        return _sweep(args)
    if args.cmd == "capacity":
        return _capacity(args)
    if args.cmd == "eda":
        return _eda(args)
    return 1


# =========================================================================
def _devices() -> int:
    from .hardware.registry import DATAFLOW, GPUS
    from .report.report import table
    from .units import fmt_bw, fmt_bytes, fmt_flops
    from .hardware.registry import make_device
    rows = []
    for name in sorted(GPUS):
        d = make_device(name)
        rows.append([name, d.vendor, d.kind, d.n_units,
                     fmt_flops(d.compute.peak_for("bf16")),
                     fmt_bytes(d.capacity),
                     fmt_bw(d.main_memory.bandwidth),
                     f"{d.energy.tdp:.0f}", f"{d.process_nm:g}", d.year,
                     d.extra.get("confidence", "")])
    print("GPUs")
    print(table(rows, ["name", "vendor", "kind", "SMs", "peak bf16", "mem",
                       "mem BW", "TDP W", "nm", "yr", "conf"]))
    rows = []
    for name in sorted(DATAFLOW):
        d = make_device(name)
        rows.append([name, d.vendor, d.arch, d.n_cores,
                     fmt_bytes(d.sram_per_core), fmt_bytes(d.total_sram),
                     fmt_bw(d.total_sram_bw),
                     fmt_flops(d.compute.peak_for("bf16")),
                     fmt_bw(d.noc.bisection_bandwidth),
                     fmt_bw(d.inter_chip_bw),
                     f"{d.energy.tdp:.0f}", f"{d.process_nm:g}", d.year,
                     d.extra.get("confidence", "")])
    print()
    print("Brain-inspired many-core dataflow chips")
    print(table(rows, ["name", "vendor", "arch", "cores", "SRAM/core",
                       "SRAM tot", "SRAM BW", "peak bf16", "NoC bisect",
                       "chip2chip", "TDP W", "nm", "yr", "conf"]))
    return 0


def _device(name: str) -> int:
    from .hardware.registry import make_device
    from .power import calibrate, load_eda_db
    from .units import fmt_energy
    d = make_device(name)
    db = load_eda_db()
    calibrate(d, eda=(db or {}).get(d.kind) or (db or {}).get("default"))
    print(d.summary())
    print()
    print("energy model (%s):" % d.energy.source)
    for k, v in sorted(d.energy.j_per_flop.items(), key=lambda x: -x[1]):
        print(f"  {k:8s} {v*1e12:8.4f} pJ / FLOP")
    for k, v in sorted(d.energy.j_per_byte.items(), key=lambda x: -x[1]):
        print(f"  {k:8s} {v*1e12:8.4f} pJ / byte")
    if d.energy.j_per_byte_noc:
        print(f"  {'noc':8s} {d.energy.j_per_byte_noc*1e12:8.4f} pJ / byte/hop")
    print(f"  static   {d.energy.static_power:8.1f} W")
    src = d.extra.get("source", "")
    if src:
        print(f"\nspec source: {src}  (confidence: "
              f"{d.extra.get('confidence','?')})")
    return 0


def _models() -> int:
    from .workload.models import get_model, list_models
    from .report.report import table
    from .units import fmt_bytes
    for fam, names in list_models().items():
        rows = []
        for n in names:
            m = get_model(n)
            if m.family == "llm" or m.family == "vlm":
                rows.append([n, m.n_layers, m.d_model,
                             f"{m.total_params()/1e9:.1f}B",
                             f"{m.active_params()/1e9:.1f}B",
                             m.n_experts or "-", m.n_active_experts or "-",
                             "MLA" if m.is_mla else "GQA",
                             f"{m.kv_bytes_per_token()/1024:.1f} KiB",
                             "PROJ" if m.extra.get("projected") else ""])
            else:
                rows.append([n, "-", "-", "-", "-", "-", "-",
                             m.extra.get("kind", m.extra.get("arch", "")),
                             "-", ""])
        print(f"{fam.upper()}")
        print(table(rows, ["name", "L", "d", "params", "active", "E", "topk",
                           "attn", "KV/token", "note"]))
        print()
    return 0


def _model(args) -> int:
    from .workload.models import get_model
    kw = {}
    if args.w_dtype:
        kw["w_dtype"] = args.w_dtype
    if args.kv_dtype:
        kw["kv_dtype"] = args.kv_dtype
    m = get_model(args.name, **kw)
    print(m.summary())
    print(f"\nsource: {m.extra.get('source','')}")
    if m.extra.get("projected"):
        print("*** PROJECTED CONFIG -- not a published architecture ***")
    return 0


def _fabrics() -> int:
    from .hardware.interconnect import FABRICS
    from .report.report import table
    from .units import fmt_bw
    from .units import parse
    rows = [[k, v["kind"], v["bandwidth"], v["latency"],
             f"{v.get('efficiency',0.85):.2f}", v.get("energy_per_byte", "-")]
            for k, v in sorted(FABRICS.items(),
                               key=lambda kv: (kv[1]["kind"],
                                               -parse(kv[1]["bandwidth"])))]
    print(table(rows, ["fabric", "kind", "BW (1-way)", "latency", "eff",
                       "energy/B"]))
    return 0


def _placements() -> int:
    from .mapping.placement import STRATEGIES
    for k, fn in sorted(STRATEGIES.items()):
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        print(f"  {k:16s} {doc}")
    return 0


def _run(args) -> int:
    from .experiment import run_file
    exp = run_file(args.config, only=args.only)
    print(exp.report(verbose=not args.quiet))
    if args.out:
        exp.save(args.out)
        print(f"\n[saved] {args.out}")
    return 0


def _sweep(args) -> int:
    import copy
    from .experiment import Experiment, load_config
    from .report.report import table
    from .units import parse

    cfg0 = load_config(args.config)
    base_dir = os.path.dirname(os.path.abspath(args.config))
    rows = []
    names = None
    for raw in args.values:
        cfg = copy.deepcopy(cfg0)
        try:
            val = parse(raw)
        except Exception:
            val = raw
        _set_path(cfg, args.param, val)
        exp = Experiment(cfg, base_dir=base_dir)
        exp.run(only=args.only)
        if names is None:
            names = list(exp.results)
        row = [raw]
        for n in names:
            r = exp.results[n]
            row.append(_metric(r, args.metric))
        rows.append(row)
    print(f"SWEEP {args.param} -> {args.metric}")
    print(table(rows, [args.param] + (names or [])))
    return 0


def _metric(res, name):
    if hasattr(res, name):
        v = getattr(res, name)
        return f"{v:,.4g}" if isinstance(v, (int, float)) else str(v)
    for stat in ("ttft", "tpot", "e2e"):
        if name.startswith(stat + "_"):
            return f"{getattr(res, stat + '_stats')()[name.split('_',1)[1]]:.6g}"
    return "?"


def _set_path(d, path, value):
    parts = path.split(".")
    cur = d
    for i, p in enumerate(parts[:-1]):
        nxt = parts[i + 1]
        if isinstance(cur, list):
            cur = cur[int(p)]
            continue
        if p == "runs" and isinstance(cur.get("runs"), list):
            # allow runs.<name>.<...>
            name = nxt
            for r in cur["runs"]:
                if r.get("name") == name:
                    cur = r
                    break
            else:
                raise KeyError(f"no run named {name!r}")
            parts = parts[i + 2:]
            return _set_path(cur, ".".join(parts), value)
        cur = cur.setdefault(p, {}) if isinstance(cur, dict) else cur[int(p)]
    if isinstance(cur, list):
        cur[int(parts[-1])] = value
    else:
        cur[parts[-1]] = value


def _capacity(args) -> int:
    from .capacity import capacity_table, find_capacity
    from .experiment import Experiment, load_config, _resolve_system
    from .mapping import placement as P
    from .power import calibrate_system
    from .report.report import save_json
    from .system import build_system

    cfg = load_config(args.config)
    base_dir = os.path.dirname(os.path.abspath(args.config))
    exp = Experiment(cfg, base_dir=base_dir)
    out = {}
    for run in cfg.get("runs", []):
        name = run.get("name")
        if args.only and name not in args.only:
            continue
        model = exp._model(run)
        syscfg = _resolve_system(run["system"], exp.systems, base_dir)
        syscfg.setdefault("name", name)
        system = build_system(syscfg)
        calibrate_system(system, eda_db=exp.eda_db)
        policy = exp._placement(run, model, system)
        wl = exp._workload(run)
        opt = dict(exp.defaults)
        opt.update({k: v for k, v in run.items()
                    if k in ("microbatches", "jitter", "collective_overlap",
                             "tp_world", "ep_world", "seed")})
        if not args.quiet:
            print(f"--- capacity search: {name} ---")
        out[name] = find_capacity(
            model, system, policy, wl, slo=exp.slo, target=args.target,
            n_requests=args.requests, iters=args.iters, sim_kwargs=opt,
            name=name, verbose=args.verbose or not args.quiet)
    print()
    base = cfg.get("baseline")
    if base not in out:
        base = next(iter(out))
    print(capacity_table(out, baseline=base))
    if args.out:
        save_json(args.out, {
            n: dict(max_rate=r.max_rate,
                    output_throughput=r.best.output_throughput,
                    avg_power=r.best.avg_power,
                    energy_per_output_token=r.best.energy_per_output_token,
                    tokens_per_kw=r.tokens_per_kw,
                    ttft_p99=r.best.ttft_p99, tpot_p99=r.best.tpot_p99,
                    capex=r.capex, tdp=r.tdp, warnings=r.warnings,
                    curve=[dict(rate=p.rate, attainment=p.attainment,
                                output_throughput=p.output_throughput,
                                avg_power=p.avg_power)
                           for p in sorted(r.points, key=lambda x: x.rate)])
            for n, r in out.items()})
        print(f"\n[saved] {args.out}")
    return 0


def _eda(args) -> int:
    from .power.eda import run_flow
    return run_flow(host=args.host, user=args.user, key=args.key,
                    dry_run=args.dry_run, local_only=args.local_only,
                    out=args.out, skip_syn=args.skip_syn)


if __name__ == "__main__":
    sys.exit(main())
