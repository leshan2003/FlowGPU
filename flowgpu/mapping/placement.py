"""Configurable workload -> device-pool placement.

The central question of the study is *which part of the model runs where*, so
placement is a first-class, fully configurable object rather than something
hard-coded in the scheduler.

A :class:`PlacementPolicy` is an ordered list of rules; the first rule whose
predicate matches an op decides its pool.  Predicates can match on:

``role``    ``attn`` | ``ffn`` | ``moe`` | ``embed`` | ``lmhead`` | ``norm``
            | ``vision`` | ``conv`` | ``diffusion`` | ``other``
``stage``   ``prefill`` | ``decode``
``kind``    the op kind (``gemm``, ``attention``, ``all2all``, ...)
``layer``   an inclusive ``[lo, hi]`` range, or a list of indices
``name``    a regular expression on the op name

Named strategies
----------------
``all_gpu``        everything on the GPU pool (the baseline)
``all_dataflow``   everything on the dataflow pool
``pd_a``           **the architecture under test**: prefill and *all* attention
                   on the GPU, FFN/MoE on the dataflow chip
``ffn_offload``    only decode-phase FFN/MoE goes to the dataflow chip
``moe_offload``    only MoE experts go to the dataflow chip
``layer_split``    first ``f`` fraction of layers on the GPU, rest on dataflow
``decode_offload`` whole decode phase on dataflow, prefill on GPU

Crossing cost
-------------
Whenever consecutive ops land on different pools the activation tensor must
cross a bridge link.  The scheduler charges that crossing.  For ``pd_a`` on a
61-layer model that is 2 crossings per layer per step -- 122 round trips of the
hidden state per decoded token.  If the bridge is thin, that cost dominates
everything else, which is precisely the risk the simulator exists to quantify.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..workload import graph as G


@dataclass
class Rule:
    pool: str
    role: tuple | None = None
    stage: tuple | None = None
    kind: tuple | None = None
    layer: tuple | None = None       # (lo, hi) inclusive
    layers: tuple | None = None      # explicit indices
    name: str | None = None          # regex
    note: str = ""

    def __post_init__(self):
        for f in ("role", "stage", "kind"):
            v = getattr(self, f)
            if isinstance(v, str):
                setattr(self, f, (v,) if v != "*" else None)
            elif isinstance(v, (list, tuple)):
                setattr(self, f, tuple(v))
        if self.name:
            self._re = re.compile(self.name)
        else:
            self._re = None

    def matches(self, op: G.Op, n_layers: int) -> bool:
        if self.role is not None and op.role not in self.role:
            return False
        if self.stage is not None and op.stage not in self.stage:
            return False
        if self.kind is not None and op.kind not in self.kind:
            return False
        if self.layer is not None:
            lo, hi = self.layer
            lo = lo if lo >= 0 else n_layers + lo
            hi = hi if hi >= 0 else n_layers + hi
            if not (lo <= op.layer <= hi):
                return False
        if self.layers is not None and op.layer not in self.layers:
            return False
        if self._re is not None and not self._re.search(op.name):
            return False
        return True


@dataclass
class PlacementPolicy:
    name: str
    rules: list = field(default_factory=list)
    default_pool: str = "gpu"
    # activation dtype used when crossing a bridge (quantising the crossing
    # is a real and very effective optimisation, so it is configurable)
    crossing_dtype: str | None = None
    notes: str = ""

    def place(self, op: G.Op, n_layers: int) -> str:
        for r in self.rules:
            if r.matches(op, n_layers):
                return r.pool
        return self.default_pool

    def annotate(self, graph: G.Graph, n_layers: int | None = None) -> dict:
        """Tag every op with ``op.meta['pool']``; return per-pool stats."""
        n_layers = n_layers or max((o.layer for o in graph.ops), default=0) + 1
        stats = {}
        prev = None
        crossings = 0
        cross_bytes = 0.0
        for op in graph.ops:
            pool = self.place(op, n_layers)
            op.meta["pool"] = pool
            s = stats.setdefault(pool, dict(ops=0, flops=0.0, weight_bytes=0.0,
                                            act_bytes=0.0, state_bytes=0.0))
            s["ops"] += 1
            s["flops"] += op.flops
            s["weight_bytes"] += op.weight_bytes
            s["act_bytes"] += op.act_bytes
            s["state_bytes"] += op.state_read_bytes + op.state_write_bytes
            if prev is not None and pool != prev:
                crossings += 1
                cross_bytes += op.input_bytes
                op.meta["crossing_from"] = prev
            prev = pool
        graph.meta["crossings"] = crossings
        graph.meta["crossing_bytes"] = cross_bytes
        graph.meta["placement"] = self.name
        return stats


# =========================================================================
# named strategies
# =========================================================================
def all_gpu(gpu: str = "gpu") -> PlacementPolicy:
    """Everything on the GPU pool -- the pure-GPU baseline."""
    return PlacementPolicy(name="all_gpu", rules=[], default_pool=gpu,
                           notes="pure-GPU baseline")


def all_dataflow(pool: str = "brain") -> PlacementPolicy:
    """Everything on the many-core dataflow pool."""
    return PlacementPolicy(name="all_dataflow", rules=[], default_pool=pool,
                           notes="everything on the many-core dataflow pool")


def pd_a(gpu: str = "gpu", brain: str = "brain",
         crossing_dtype: str | None = None) -> PlacementPolicy:
    """PD+A split as described in the 中国算力大会 announcement.

    'Prefill 阶段及 Attention 计算模块交由国产GPU承载，FFN（MOE专家）延时敏感
    模块交由类脑芯片处理.'
    """
    return PlacementPolicy(
        name="pd_a",
        rules=[
            # everything in prefill stays on the GPU: it is compute-bound and
            # that is what GPUs are good at
            Rule(pool=gpu, stage="prefill", note="prefill -> GPU"),
            # decode attention (KV-cache bound) stays on the GPU
            Rule(pool=gpu, role=(G.R_ATTN,), note="attention -> GPU"),
            Rule(pool=gpu, role=(G.R_EMBED, G.R_LMHEAD),
                 note="embed/head -> GPU"),
            # decode FFN and MoE experts go to the dataflow chip
            Rule(pool=brain, role=(G.R_FFN, G.R_MOE),
                 note="FFN/MoE -> brain chip"),
            # norms follow whichever block they belong to; cheapest is to keep
            # them next to the FFN they feed to avoid an extra crossing
            Rule(pool=brain, role=(G.R_NORM,), name=r"norm_(ffn|moe)",
                 note="pre-FFN norm -> brain chip"),
        ],
        default_pool=gpu, crossing_dtype=crossing_dtype,
        notes="Prefill+Attention on GPU, FFN/MoE experts on brain chip")


def ffn_offload(gpu: str = "gpu", brain: str = "brain",
                crossing_dtype: str | None = None) -> PlacementPolicy:
    """Only *decode* FFN/MoE is offloaded; prefill FFN stays on the GPU."""
    return PlacementPolicy(
        name="ffn_offload",
        rules=[Rule(pool=brain, stage="decode", role=(G.R_FFN, G.R_MOE)),
               Rule(pool=brain, stage="decode", role=(G.R_NORM,),
                    name=r"norm_(ffn|moe)")],
        default_pool=gpu, crossing_dtype=crossing_dtype,
        notes="decode FFN/MoE offloaded, everything else on GPU")


def moe_offload(gpu: str = "gpu", brain: str = "brain",
                crossing_dtype: str | None = None) -> PlacementPolicy:
    """MoE experts to the dataflow pool in *both* phases; rest on the GPU.

    Unlike :func:`pd_a` this does not pin prefill to the GPU, so the dataflow
    pool absorbs the prefill expert GEMMs too.  In the reference experiment
    that difference is worth 2.1x of SLO-constrained capacity on identical
    hardware -- it is the single most consequential knob in the study.
    """
    return PlacementPolicy(
        name="moe_offload",
        rules=[Rule(pool=brain, role=(G.R_MOE,))],
        default_pool=gpu, crossing_dtype=crossing_dtype,
        notes="MoE experts only (dense FFN layers stay on GPU)")


def decode_offload(gpu: str = "gpu", brain: str = "brain",
                   crossing_dtype: str | None = None) -> PlacementPolicy:
    """Classic PD disaggregation: prefill on the GPU, all decode on dataflow.

    Because the two phases then touch disjoint pools, the serving simulator
    runs them concurrently -- this is the one strategy that genuinely
    pipelines prefill against decode.
    """
    return PlacementPolicy(
        name="decode_offload",
        rules=[Rule(pool=gpu, stage="prefill"),
               Rule(pool=brain, stage="decode")],
        default_pool=gpu, crossing_dtype=crossing_dtype,
        notes="classic PD disaggregation: prefill GPU, decode dataflow")


def layer_split(n_layers: int, frac_gpu: float = 0.5, gpu: str = "gpu",
                brain: str = "brain",
                crossing_dtype: str | None = None) -> PlacementPolicy:
    """First ``frac_gpu`` of layers on the GPU, the rest on the dataflow pool.

    This is the *crossing-minimal* heterogeneous split: exactly one bridge
    traversal per token step instead of two per layer.  Worth comparing
    against ``pd_a`` -- if the bridge is the bottleneck, this wins easily.
    """
    cut = int(round(n_layers * frac_gpu))
    return PlacementPolicy(
        name=f"layer_split@{frac_gpu:.2f}",
        rules=[Rule(pool=gpu, layer=(-1, cut - 1)),
               Rule(pool=brain, layer=(cut, n_layers)),
               Rule(pool=gpu, role=(G.R_EMBED, G.R_LMHEAD))],
        default_pool=gpu, crossing_dtype=crossing_dtype,
        notes=f"layers [0,{cut}) on GPU, [{cut},{n_layers}) on dataflow")


def from_config(cfg: dict) -> PlacementPolicy:
    """Build a policy from a YAML/JSON dict.

    ::

        placement:
          name: my_split
          default_pool: gpu
          crossing_dtype: fp8
          rules:
            - {pool: brain, role: [moe, ffn], stage: decode}
            - {pool: gpu,   role: attn}
    """
    if isinstance(cfg, str):
        return named(cfg)
    if "strategy" in cfg:
        return named(cfg["strategy"], **{k: v for k, v in cfg.items()
                                         if k != "strategy"})
    rules = [Rule(**r) for r in cfg.get("rules", [])]
    return PlacementPolicy(
        name=cfg.get("name", "custom"), rules=rules,
        default_pool=cfg.get("default_pool", "gpu"),
        crossing_dtype=cfg.get("crossing_dtype"),
        notes=cfg.get("notes", ""))


STRATEGIES = {
    "all_gpu": all_gpu,
    "all_dataflow": all_dataflow,
    "pd_a": pd_a,
    "ffn_offload": ffn_offload,
    "moe_offload": moe_offload,
    "decode_offload": decode_offload,
    "layer_split": layer_split,
}


def named(strategy: str, **kw) -> PlacementPolicy:
    if strategy not in STRATEGIES:
        raise KeyError(f"unknown placement strategy {strategy!r}. "
                       f"Known: {sorted(STRATEGIES)}")
    fn = STRATEGIES[strategy]
    import inspect
    sig = inspect.signature(fn)
    kw = {k: v for k, v in kw.items() if k in sig.parameters}
    return fn(**kw)
