"""Workload intermediate representation.

A workload is a DAG of :class:`Op`.  Every Op carries enough information for a
hardware model to price it:

* ``flops``            -- multiply/add work (1 MAC = 2 FLOP)
* ``weights``          -- persistent tensors (model parameters)
* ``inputs``/``outputs`` -- transient activations
* ``state``            -- tensors that persist *across* ops within a request
                          (KV cache), read/written incrementally
* ``m/n/k``            -- GEMM shape, used by the efficiency models
* ``role``             -- semantic tag driving placement (``attn``/``ffn``/...)

The IR is deliberately coarse: one Op is roughly one fused kernel.  That is the
granularity at which hardware roofline + memory-hierarchy reasoning is
meaningful, and it keeps a 61-layer MoE model at a few hundred nodes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from ..units import dtype_bytes

# --- op kinds -------------------------------------------------------------
GEMM = "gemm"              # dense matmul, weight-stationary candidate
BMM = "bmm"                # batched matmul without persistent weights (QK^T, PV)
ATTENTION = "attention"    # fused attention (flash-style); reads KV state
ELEMENTWISE = "elementwise"
NORM = "norm"
SOFTMAX = "softmax"
CONV = "conv"
ROUTER = "router"          # MoE gate
A2A = "all2all"            # expert dispatch/combine collective
ALLREDUCE = "allreduce"    # TP collective
EMBED = "embed"            # embedding lookup (gather, ~no flops)
SPIKE = "spike"            # SNN accumulate-and-fire

# --- roles (placement keys) ----------------------------------------------
R_EMBED = "embed"
R_ATTN = "attn"
R_FFN = "ffn"
R_MOE = "moe"
R_NORM = "norm"
R_LMHEAD = "lmhead"
R_VISION = "vision"
R_CONV = "conv"
R_DIFFUSION = "diffusion"
R_OTHER = "other"

ALL_ROLES = (R_EMBED, R_ATTN, R_FFN, R_MOE, R_NORM, R_LMHEAD,
             R_VISION, R_CONV, R_DIFFUSION, R_OTHER)


@dataclass
class Tensor:
    name: str
    shape: tuple
    dtype: str = "bf16"
    kind: str = "act"        # 'weight' | 'act' | 'kv' | 'const'

    @property
    def numel(self) -> float:
        n = 1.0
        for s in self.shape:
            n *= s
        return n

    @property
    def nbytes(self) -> float:
        return self.numel * dtype_bytes(self.dtype)

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"T({self.name},{self.shape},{self.dtype},{self.kind})"


@dataclass
class Op:
    name: str
    kind: str
    flops: float = 0.0
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    # bytes of persistent per-request state touched (KV cache reads/writes)
    state_read_bytes: float = 0.0
    state_write_bytes: float = 0.0
    m: int = 0
    n: int = 0
    k: int = 0
    layer: int = -1
    role: str = R_OTHER
    stage: str = "decode"        # 'prefill' | 'decode'
    deps: list = field(default_factory=list)   # names of predecessor ops
    # for MoE: how many experts are actually touched, and their total params
    meta: dict = field(default_factory=dict)

    # -- derived sizes ----------------------------------------------------
    @property
    def weight_bytes(self) -> float:
        return sum(t.nbytes for t in self.weights)

    @property
    def input_bytes(self) -> float:
        return sum(t.nbytes for t in self.inputs)

    @property
    def output_bytes(self) -> float:
        return sum(t.nbytes for t in self.outputs)

    @property
    def act_bytes(self) -> float:
        return self.input_bytes + self.output_bytes

    @property
    def total_bytes(self) -> float:
        return (self.weight_bytes + self.act_bytes
                + self.state_read_bytes + self.state_write_bytes)

    @property
    def arithmetic_intensity(self) -> float:
        b = self.total_bytes
        return self.flops / b if b > 0 else math.inf

    def __repr__(self):  # pragma: no cover
        return (f"Op({self.name},{self.kind},L{self.layer},{self.role},"
                f"{self.flops/1e9:.2f} GFLOP,{self.total_bytes/2**20:.2f} MiB)")


@dataclass
class Graph:
    """An executable workload phase (one prefill chunk, or one decode step)."""
    name: str
    ops: list = field(default_factory=list)
    stage: str = "decode"
    batch: int = 1
    seq_len: int = 1          # tokens processed *by this graph* per sequence
    ctx_len: int = 0          # context length the KV cache holds
    meta: dict = field(default_factory=dict)

    def add(self, op: Op) -> Op:
        op.stage = self.stage
        self.ops.append(op)
        return op

    # -- aggregates -------------------------------------------------------
    @property
    def flops(self) -> float:
        return sum(o.flops for o in self.ops)

    @property
    def weight_bytes(self) -> float:
        """Unique weight bytes (a weight read once per graph, not per op)."""
        seen = {}
        for o in self.ops:
            for t in o.weights:
                seen[t.name] = t.nbytes
        return sum(seen.values())

    @property
    def n_tokens(self) -> float:
        return self.batch * self.seq_len

    def by_role(self) -> dict:
        out = {}
        for o in self.ops:
            out.setdefault(o.role, []).append(o)
        return out

    def flops_by_role(self) -> dict:
        out = {}
        for o in self.ops:
            out[o.role] = out.get(o.role, 0.0) + o.flops
        return out

    def summary(self) -> str:  # pragma: no cover
        lines = [f"Graph {self.name} stage={self.stage} B={self.batch} "
                 f"S={self.seq_len} ctx={self.ctx_len} ops={len(self.ops)}",
                 f"  total {self.flops/1e12:.3f} TFLOP, "
                 f"weights {self.weight_bytes/2**30:.3f} GiB"]
        for role, f in sorted(self.flops_by_role().items(),
                              key=lambda kv: -kv[1]):
            lines.append(f"    {role:10s} {f/1e12:9.4f} TFLOP "
                         f"({100*f/max(self.flops,1):5.1f}%)")
        return "\n".join(lines)


@dataclass
class ModelSpec:
    """Architecture description that graph builders consume."""
    name: str
    family: str                       # 'llm' | 'vlm' | 'diffusion' | 'cnn'
    n_layers: int = 0
    d_model: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0
    head_dim: int = 0
    d_ff: int = 0
    vocab: int = 0
    # MoE
    n_experts: int = 0
    n_active_experts: int = 0
    d_expert_ff: int = 0
    n_shared_experts: int = 0
    first_dense_layers: int = 0       # dense FFN layers before MoE starts
    # MLA (DeepSeek) -- if set, KV cache is the compressed latent
    mla_kv_lora_rank: int = 0
    mla_qk_rope_dim: int = 0
    mla_qk_nope_dim: int = 0
    mla_v_dim: int = 0
    mla_q_lora_rank: int = 0
    # dtypes
    w_dtype: str = "bf16"
    a_dtype: str = "bf16"
    kv_dtype: str = "bf16"
    # dtype used on the wire for collectives (fp8 dispatch is standard in
    # modern MoE engines and halves all-to-all volume)
    comm_dtype: str = ""
    # multimodal / diffusion / cnn extras
    extra: dict = field(default_factory=dict)

    # -- derived ----------------------------------------------------------
    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0

    @property
    def is_mla(self) -> bool:
        return self.mla_kv_lora_rank > 0

    def kv_bytes_per_token(self) -> float:
        """Bytes of KV cache per token per *layer-stack* (all layers)."""
        eb = dtype_bytes(self.kv_dtype)
        if self.is_mla:
            per_layer = (self.mla_kv_lora_rank + self.mla_qk_rope_dim) * eb
        else:
            per_layer = 2 * self.n_kv_heads * self.head_dim * eb
        return per_layer * self.n_layers

    def params(self) -> dict:
        """Parameter counts, in elements, split by role."""
        p = {R_EMBED: 0.0, R_ATTN: 0.0, R_FFN: 0.0, R_MOE: 0.0,
             R_LMHEAD: 0.0, R_NORM: 0.0}
        d = self.d_model
        p[R_EMBED] += self.vocab * d
        p[R_LMHEAD] += self.vocab * d

        for l in range(self.n_layers):
            # attention projections
            if self.is_mla:
                p[R_ATTN] += d * self.mla_q_lora_rank if self.mla_q_lora_rank else 0
                q_out = self.n_heads * (self.mla_qk_nope_dim + self.mla_qk_rope_dim)
                if self.mla_q_lora_rank:
                    p[R_ATTN] += self.mla_q_lora_rank * q_out
                else:
                    p[R_ATTN] += d * q_out
                p[R_ATTN] += d * (self.mla_kv_lora_rank + self.mla_qk_rope_dim)
                p[R_ATTN] += self.mla_kv_lora_rank * self.n_heads * (
                    self.mla_qk_nope_dim + self.mla_v_dim)
                p[R_ATTN] += self.n_heads * self.mla_v_dim * d
            else:
                p[R_ATTN] += d * self.n_heads * self.head_dim              # Q
                p[R_ATTN] += 2 * d * self.n_kv_heads * self.head_dim       # K,V
                p[R_ATTN] += self.n_heads * self.head_dim * d              # O
            p[R_NORM] += 2 * d

            # feed-forward
            if self.is_moe and l >= self.first_dense_layers:
                de = self.d_expert_ff
                p[R_MOE] += self.n_experts * 3 * d * de
                p[R_MOE] += self.n_shared_experts * 3 * d * de
                p[R_MOE] += d * self.n_experts                             # router
            else:
                p[R_FFN] += 3 * d * self.d_ff
        return p

    def total_params(self) -> float:
        return sum(self.params().values())

    def active_params(self) -> float:
        """Parameters touched by a single token (MoE-aware)."""
        p = self.params()
        act = p[R_ATTN] + p[R_NORM] + p[R_FFN] + p[R_LMHEAD]
        if self.is_moe:
            d, de = self.d_model, self.d_expert_ff
            n_moe_layers = self.n_layers - self.first_dense_layers
            act += n_moe_layers * (
                (self.n_active_experts + self.n_shared_experts) * 3 * d * de
                + d * self.n_experts)
        return act

    def weight_bytes(self) -> float:
        return self.total_params() * dtype_bytes(self.w_dtype)

    def summary(self) -> str:  # pragma: no cover
        p = self.params()
        lines = [f"{self.name} [{self.family}] L={self.n_layers} d={self.d_model}"]
        lines.append(f"  total params  {self.total_params()/1e9:8.2f} B")
        lines.append(f"  active params {self.active_params()/1e9:8.2f} B")
        lines.append(f"  weights       {self.weight_bytes()/2**30:8.2f} GiB "
                     f"({self.w_dtype})")
        lines.append(f"  KV / token    {self.kv_bytes_per_token()/1024:8.2f} KiB "
                     f"({self.kv_dtype})")
        for k, v in p.items():
            if v:
                lines.append(f"    {k:8s} {v/1e9:8.3f} B params")
        return "\n".join(lines)
