"""Single-graph evaluation for non-autoregressive workloads.

CNNs, diffusion denoising steps and ViT encoders have no KV cache and no
token-by-token loop, so the request-level serving simulator is the wrong tool.
What matters for them is per-batch latency, sustained throughput under a
pipeline, and energy per image / per step -- which is one graph, evaluated.

This module also handles **VLM** and **diffusion** end to end, because those
are the cases where heterogeneous placement gets genuinely interesting:

* a VLM's vision tower is a dense, compute-bound, weight-reusing encoder --
  a near-perfect fit for a dataflow chip -- while its LLM decoder is the
  memory-bound part that wants HBM;
* a diffusion model runs the *same* graph 20-50 times with the same weights,
  so weight-stationary hardware amortises the load once instead of per step.
  That is the single most favourable case for the architecture under study,
  and the simulator should be able to show it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..mapping.placement import PlacementPolicy
from ..power import model as pmodel
from ..system import System
from ..workload.graph import Graph, ModelSpec
from ..workload import vision
from .executor import StepResult, execute, execute_pipelined


@dataclass
class OneShotResult:
    name: str
    latency: float = 0.0            # one batch, end to end
    energy: float = 0.0
    device_energy: float = 0.0
    idle_energy: float = 0.0
    host_energy: float = 0.0
    batch: int = 1
    repeats: int = 1                # diffusion steps, or images per batch
    steps: list = field(default_factory=list)
    busy_by_pool: dict = field(default_factory=dict)
    t_by_bound: dict = field(default_factory=dict)
    t_by_role: dict = field(default_factory=dict)
    e_by_source: dict = field(default_factory=dict)
    bytes_crossed: float = 0.0
    flops: float = 0.0
    warnings: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def throughput(self) -> float:
        """Items (images / prompts) per second."""
        return self.batch / self.latency if self.latency else 0.0

    @property
    def avg_power(self) -> float:
        return self.energy / self.latency if self.latency else 0.0

    @property
    def energy_per_item(self) -> float:
        return self.energy / self.batch if self.batch else 0.0

    @property
    def achieved_flops(self) -> float:
        return self.flops / self.latency if self.latency else 0.0


def run_graphs(graphs: list, system: System, policy: PlacementPolicy,
               name: str = "", batch: int = 1, repeats: dict | None = None,
               microbatches: int = 1, collective_overlap: float = 0.0,
               weight_bytes_by_pool: dict | None = None) -> OneShotResult:
    """Evaluate a sequence of graphs, each optionally repeated.

    ``repeats`` maps graph name -> repeat count (diffusion steps).  Weights
    are planned once, which is the point: a 28-step diffusion sample loads
    the DiT weights once on weight-stationary hardware and 28 times on a GPU.
    """
    repeats = repeats or {}
    n_layers = max((o.layer for g in graphs for o in g.ops), default=0) + 1

    if weight_bytes_by_pool is None:
        weight_bytes_by_pool = _weights_by_pool(graphs, policy, n_layers)
    for pname, pool in system.pools.items():
        pool.plan(weight_bytes_by_pool.get(pname, 0.0), 0.0, 0.0)

    out = OneShotResult(name=name or "oneshot", batch=batch)
    for g in graphs:
        policy.annotate(g, n_layers)
        if microbatches > 1:
            r = execute_pipelined(g, system, policy, microbatches=microbatches,
                                  n_layers=n_layers,
                                  collective_overlap=collective_overlap)
        else:
            r = execute(g, system, policy, n_layers=n_layers,
                        collective_overlap=collective_overlap)
        n = int(repeats.get(g.name, 1))
        out.steps.append((g.name, n, r))
        out.latency += r.time * n
        out.energy += r.energy * n
        out.flops += r.flops * n
        out.bytes_crossed += r.bytes_crossed * n
        for attr in ("busy_by_pool", "t_by_bound", "t_by_role",
                     "e_by_source"):
            d = getattr(out, attr)
            for k, v in getattr(r, attr).items():
                d[k] = d.get(k, 0.0) + v * n
        for w in r.warnings:
            if w not in out.warnings:
                out.warnings.append(w)

    out.device_energy = out.energy
    idle = pmodel.idle_energy(system, out.latency, out.busy_by_pool)
    out.idle_energy = sum(idle.values())
    out.host_energy = pmodel.host_energy(system, out.latency)
    out.energy = (out.device_energy + out.idle_energy
                  + out.host_energy) * system.pue
    out.e_by_source["idle"] = out.idle_energy
    out.e_by_source["host"] = out.host_energy
    out.meta.update(dict(system=system.name, placement=policy.name,
                         weights_by_pool={k: v / 2**30 for k, v
                                          in weight_bytes_by_pool.items()}))
    return out


def _weights_by_pool(graphs, policy, n_layers) -> dict:
    seen, out = set(), {}
    for g in graphs:
        policy.annotate(g, n_layers)
        for op in g.ops:
            p = op.meta["pool"]
            for t in op.weights:
                if t.name in seen:
                    continue
                seen.add(t.name)
                out[p] = out.get(p, 0.0) + t.nbytes
    return out


# =========================================================================
def run_diffusion(spec: ModelSpec, system: System, policy: PlacementPolicy,
                  batch: int = 1, steps: int | None = None,
                  cfg_scale: bool = True, **kw) -> OneShotResult:
    """A full text-to-image sample: N denoising steps plus a VAE decode."""
    steps = steps or spec.extra.get("default_steps", 28)
    eff_batch = batch * (2 if cfg_scale else 1)     # classifier-free guidance
    dit = vision.build_diffusion_step(spec, eff_batch)
    vae = vision.build_vae_decode(spec, batch)
    r = run_graphs([dit, vae], system, policy,
                   name=f"{spec.name}@{steps}steps", batch=batch,
                   repeats={dit.name: steps, vae.name: 1}, **kw)
    r.repeats = steps
    r.meta["steps"] = steps
    r.meta["cfg"] = cfg_scale
    return r


def run_cnn(spec: ModelSpec, system: System, policy: PlacementPolicy,
            batch: int = 32, **kw) -> OneShotResult:
    g = vision.build_cnn(spec, batch)
    return run_graphs([g], system, policy, name=f"{spec.name}@b{batch}",
                      batch=batch, **kw)


def run_vit(spec: ModelSpec, system: System, policy: PlacementPolicy,
            n_images: int = 1, **kw) -> OneShotResult:
    g = vision.build_vit(spec, n_images)
    return run_graphs([g], system, policy, name=f"{spec.name}.vit",
                      batch=n_images, **kw)


def run_vlm_prefill(spec: ModelSpec, system: System, policy: PlacementPolicy,
                    batch: int = 1, text_tokens: int = 256,
                    n_images: int = 1, **kw) -> OneShotResult:
    vg, lg = vision.build_vlm_prefill(spec, batch, text_tokens, n_images)
    r = run_graphs([vg, lg], system, policy,
                   name=f"{spec.name}.vlm_prefill", batch=batch, **kw)
    r.meta["visual_tokens"] = lg.meta.get("visual_tokens")
    r.meta["text_tokens"] = text_tokens
    return r


# =========================================================================
def summarize(r: OneShotResult) -> str:  # pragma: no cover
    from ..report.report import table
    from ..units import fmt_bytes, fmt_energy, fmt_flops, fmt_time
    L = [f"=== {r.name} ===",
         f"system={r.meta.get('system')} placement={r.meta.get('placement')}"]
    L.append(table([
        ["latency (one batch)", fmt_time(r.latency)],
        ["throughput (items/s)", f"{r.throughput:,.3f}"],
        ["avg power (W)", f"{r.avg_power:,.1f}"],
        ["energy / item", fmt_energy(r.energy_per_item)],
        ["total FLOP", f"{r.flops/1e12:,.2f} T"],
        ["achieved", fmt_flops(r.achieved_flops)],
        ["bytes crossed pools", fmt_bytes(r.bytes_crossed)],
    ], ["metric", "value"]))
    L.append("")
    L.append(table([[n, k, fmt_time(s.time), fmt_time(s.time * k)]
                    for n, k, s in r.steps],
                   ["graph", "repeats", "per call", "total"]))
    L.append("")
    tot = sum(r.t_by_bound.values()) or 1
    L.append(table([[k, fmt_time(v), f"{100*v/tot:.1f}%"]
                    for k, v in sorted(r.t_by_bound.items(),
                                       key=lambda x: -x[1])],
                   ["bound", "time", "share"]))
    L.append("")
    L.append(table([[k, fmt_time(v),
                     f"{100*v/max(r.latency,1e-12):.1f}%"]
                    for k, v in sorted(r.busy_by_pool.items(),
                                       key=lambda x: -x[1])],
                   ["pool", "busy", "util"]))
    if r.warnings:
        L.append("")
        for w in r.warnings:
            L.append(f"  ! {w}")
    return "\n".join(L)
