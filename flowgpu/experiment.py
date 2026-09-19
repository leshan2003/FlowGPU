"""Experiment driver: load a YAML/JSON config, run every arm, compare.

Config schema
-------------
::

    name: pd_a_vs_pure_gpu
    model:      {name: deepseek_v3, w_dtype: fp8, comm_dtype: fp8}
    workload:   {n_requests: 128, prompt_len: 8192, output_len: 512,
                 max_batch: 64, arrival_rate: 0}
    slo:        {ttft: 2.0, tpot: 0.05}
    defaults:   {collective_overlap: 0.6, microbatches: 1, jitter: true}
    systems:                      # reusable named system definitions
      gpu_only: {pools: {...}}
      hetero:   {pools: {...}, bridges: [...]}
    runs:
      - {name: baseline, system: gpu_only, placement: all_gpu}
      - {name: pd_a,     system: hetero,
         placement: {strategy: pd_a, gpu: gpu, brain: brain},
         microbatches: 2}

Anything under ``defaults`` can be overridden per run.  ``system`` may be a
name from ``systems``, an inline dict, or a path to a YAML file.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field

from .mapping import placement as P
from .power import calibrate_system, load_eda_db
from .report import report as R
from .sim.serving import SLO, ServingSimulator, WorkloadSpec
from .system import build_system
from .workload.models import get_model


def load_config(path: str) -> dict:
    with open(path) as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)


def _resolve_system(spec, systems: dict, base_dir: str) -> dict:
    if isinstance(spec, dict):
        return copy.deepcopy(spec)
    if spec in systems:
        return copy.deepcopy(systems[spec])
    for cand in (spec, os.path.join(base_dir, spec),
                 os.path.join(base_dir, "systems", spec),
                 os.path.join(base_dir, "systems", spec + ".yaml")):
        if os.path.exists(cand):
            return load_config(cand)
    raise KeyError(f"unknown system {spec!r}")


@dataclass
class RunResult:
    name: str
    result: object
    system: object
    sim: object
    config: dict = field(default_factory=dict)


class Experiment:
    def __init__(self, cfg: dict, base_dir: str = ".",
                 eda_db: dict | None = None):
        self.cfg = cfg
        self.base_dir = base_dir
        self.name = cfg.get("name", "experiment")
        self.systems = cfg.get("systems", {})
        self.defaults = cfg.get("defaults", {})
        self.slo = SLO(**cfg.get("slo", {}))
        self.eda_db = eda_db if eda_db is not None else load_eda_db()
        self.results: dict = {}
        self.runs: list = []

    # ------------------------------------------------------------------
    def _model(self, run: dict):
        mc = dict(self.cfg.get("model", {}))
        mc.update(run.get("model", {}))
        name = mc.pop("name")
        return get_model(name, **mc)

    def _workload(self, run: dict) -> WorkloadSpec:
        wc = dict(self.cfg.get("workload", {}))
        wc.update(run.get("workload", {}))
        return WorkloadSpec(**wc)

    def _placement(self, run: dict, model, system):
        pc = run.get("placement", "all_gpu")
        if isinstance(pc, str):
            kw = {}
            if pc == "layer_split":
                kw["n_layers"] = model.n_layers
            pools = list(system.pools)
            if len(pools) >= 2:
                kw.setdefault("gpu", pools[0])
                kw.setdefault("brain", pools[1])
            elif pools:
                kw.setdefault("gpu", pools[0])
                kw.setdefault("pool", pools[0])
            return P.named(pc, **kw)
        if "strategy" in pc:
            kw = {k: v for k, v in pc.items() if k != "strategy"}
            if pc["strategy"] == "layer_split":
                kw.setdefault("n_layers", model.n_layers)
            return P.named(pc["strategy"], **kw)
        return P.from_config(pc)

    # ------------------------------------------------------------------
    def run_one(self, run: dict) -> RunResult:
        name = run.get("name", f"run{len(self.results)}")
        model = self._model(run)
        syscfg = _resolve_system(run["system"], self.systems, self.base_dir)
        syscfg.setdefault("name", name)
        system = build_system(syscfg)
        calibrate_system(system, eda_db=self.eda_db)
        policy = self._placement(run, model, system)
        wl = self._workload(run)

        opt = dict(self.defaults)
        opt.update({k: v for k, v in run.items()
                    if k in ("microbatches", "jitter", "collective_overlap",
                             "tp_world", "ep_world", "seed", "ctx_bucket",
                             "batch_bucket")})
        sim = ServingSimulator(model, system, policy, wl, **opt)
        res = sim.run()
        rr = RunResult(name=name, result=res, system=system, sim=sim,
                       config=run)
        self.results[name] = res
        self.runs.append(rr)
        return rr

    def run(self, only: list | None = None) -> dict:
        for r in self.cfg.get("runs", []):
            if only and r.get("name") not in only:
                continue
            self.run_one(r)
        return self.results

    # ------------------------------------------------------------------
    def report(self, verbose: bool = True) -> str:
        out = [f"################ {self.name} ################", ""]
        for rr in self.runs:
            out.append(R.system_report(rr.system))
            out.append("")
            out.append(R.summarize(rr.result, self.slo, title=rr.name,
                                   verbose=verbose))
            out.append("")
        if len(self.runs) > 1:
            base = self.cfg.get("baseline", self.runs[0].name)
            if base not in self.results:
                base = self.runs[0].name
            out.append(R.compare(self.results, self.slo, baseline=base))
        return "\n".join(out)

    def to_dict(self) -> dict:
        return dict(
            name=self.name,
            config=self.cfg,
            runs={rr.name: R.to_dict(rr.result, self.slo) for rr in self.runs},
        )

    def save(self, path: str):
        R.save_json(path, self.to_dict())


def run_file(path: str, only: list | None = None,
             verbose: bool = True) -> Experiment:
    cfg = load_config(path)
    exp = Experiment(cfg, base_dir=os.path.dirname(os.path.abspath(path)))
    exp.run(only=only)
    return exp
