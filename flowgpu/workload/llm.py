"""Transformer LLM graph builders (dense, MoE, MLA).

Builds one :class:`~flowgpu.workload.graph.Graph` per execution phase:

* ``build_prefill(spec, batch, chunk, ctx_before)`` -- a prefill chunk
* ``build_decode(spec, batch, ctx)``               -- one decode step

The modelling choices that actually move the numbers:

**MoE weight traffic scales with *distinct experts touched*, not FLOPs.**
With ``E`` experts and ``k`` active, a step of ``T`` tokens touches
``E * (1 - (1 - k/E)^T)`` experts in expectation under balanced routing.  At
``T=1`` that is ``k`` experts; at ``T=512`` it is essentially all ``E``.  This
is why MoE decode is brutal on a GPU at low batch (tiny FLOPs, full weight
stream) and why parking experts in SRAM is such a large win.

**MLA decode uses the absorbed form.**  DeepSeek-style MLA caches only the
compressed latent (``kv_lora + qk_rope`` per token, e.g. 576 elements) shared
across all heads, instead of ``2 * n_kv_heads * head_dim``.  Attention runs
directly on the latent, so KV traffic drops ~10x relative to GQA.  Modelling
the naive form instead would flatter GQA models by an order of magnitude.

**Causal masking halves prefill attention FLOPs**, and attention FLOPs use the
*full* context (``ctx_before + chunk``), not just the chunk.
"""

from __future__ import annotations

import math

from ..units import dtype_bytes
from .graph import (A2A, ALLREDUCE, ATTENTION, BMM, ELEMENTWISE, EMBED, GEMM,
                    NORM, R_ATTN, R_EMBED, R_FFN, R_LMHEAD, R_MOE, R_NORM,
                    ROUTER, SOFTMAX, Graph, ModelSpec, Op, Tensor)


def _w(name: str, *shape, dtype: str) -> Tensor:
    return Tensor(name, tuple(shape), dtype, "weight")


def _a(name: str, *shape, dtype: str) -> Tensor:
    return Tensor(name, tuple(shape), dtype, "act")


def experts_touched(n_experts: int, top_k: int, n_tokens: float) -> float:
    """Expected distinct experts hit by ``n_tokens`` under balanced routing."""
    if n_experts <= 0:
        return 0.0
    if n_tokens <= 0:
        return 0.0
    p_miss = max(0.0, 1.0 - top_k / n_experts)
    return n_experts * (1.0 - p_miss ** n_tokens)


# =========================================================================
def _attention_block(g: Graph, s: ModelSpec, l: int, T: int, B: int,
                     q_len: int, kv_len: int, mla_absorbed: bool = True):
    """Emit the attention sub-graph for layer ``l``.

    ``T`` = total tokens this step (B*q_len).  ``kv_len`` = context length the
    KV cache holds *after* this step.
    """
    d = s.d_model
    wb, ab, kvb = s.w_dtype, s.a_dtype, s.kv_dtype
    pre = f"L{l}"

    g.add(Op(f"{pre}.norm_in", NORM, flops=6.0 * T * d,
             inputs=[_a(f"{pre}.x", T, d, dtype=ab)],
             outputs=[_a(f"{pre}.xn", T, d, dtype=ab)],
             weights=[_w(f"{pre}.wn1", d, dtype=wb)],
             m=T, n=d, k=1, layer=l, role=R_NORM))

    if s.is_mla:
        # ---- compressed latent-attention (DeepSeek MLA) ----------------
        qk_h = s.mla_qk_nope_dim + s.mla_qk_rope_dim
        kv_c = s.mla_kv_lora_rank
        rope = s.mla_qk_rope_dim

        if s.mla_q_lora_rank:
            g.add(Op(f"{pre}.q_a", GEMM, flops=2.0 * T * d * s.mla_q_lora_rank,
                     inputs=[_a(f"{pre}.xn", T, d, dtype=ab)],
                     outputs=[_a(f"{pre}.qa", T, s.mla_q_lora_rank, dtype=ab)],
                     weights=[_w(f"{pre}.Wqa", d, s.mla_q_lora_rank, dtype=wb)],
                     m=T, n=s.mla_q_lora_rank, k=d, layer=l, role=R_ATTN))
            q_in, q_k = f"{pre}.qa", s.mla_q_lora_rank
        else:
            q_in, q_k = f"{pre}.xn", d

        q_out = s.n_heads * qk_h
        g.add(Op(f"{pre}.q_b", GEMM, flops=2.0 * T * q_k * q_out,
                 inputs=[_a(q_in, T, q_k, dtype=ab)],
                 outputs=[_a(f"{pre}.q", T, q_out, dtype=ab)],
                 weights=[_w(f"{pre}.Wqb", q_k, q_out, dtype=wb)],
                 m=T, n=q_out, k=q_k, layer=l, role=R_ATTN))

        kv_out = kv_c + rope
        g.add(Op(f"{pre}.kv_a", GEMM, flops=2.0 * T * d * kv_out,
                 inputs=[_a(f"{pre}.xn", T, d, dtype=ab)],
                 outputs=[_a(f"{pre}.kvc", T, kv_out, dtype=ab)],
                 weights=[_w(f"{pre}.Wkva", d, kv_out, dtype=wb)],
                 state_write_bytes=T * kv_out * dtype_bytes(kvb),
                 m=T, n=kv_out, k=d, layer=l, role=R_ATTN))

        if mla_absorbed and q_len == 1:
            # decode: W_kv_b is folded into Q and O, attention runs on the
            # latent directly.  KV traffic = B * kv_len * (kv_c + rope).
            scores = 2.0 * B * s.n_heads * q_len * kv_len * kv_out
            outp = 2.0 * B * s.n_heads * q_len * kv_len * kv_c
            g.add(Op(f"{pre}.attn", ATTENTION, flops=scores + outp,
                     inputs=[_a(f"{pre}.q", T, q_out, dtype=ab)],
                     outputs=[_a(f"{pre}.ao", T, s.n_heads * kv_c, dtype=ab)],
                     state_read_bytes=B * kv_len * kv_out * dtype_bytes(kvb),
                     m=B * s.n_heads * q_len, n=kv_len, k=kv_out,
                     layer=l, role=R_ATTN,
                     meta=dict(kv_len=kv_len, absorbed=True)))
            o_k = s.n_heads * kv_c
        else:
            # prefill: materialise K/V from the latent
            kvb_out = s.n_heads * (s.mla_qk_nope_dim + s.mla_v_dim)
            g.add(Op(f"{pre}.kv_b", GEMM, flops=2.0 * T * kv_c * kvb_out,
                     inputs=[_a(f"{pre}.kvc", T, kv_out, dtype=ab)],
                     outputs=[_a(f"{pre}.kv", T, kvb_out, dtype=ab)],
                     weights=[_w(f"{pre}.Wkvb", kv_c, kvb_out, dtype=wb)],
                     m=T, n=kvb_out, k=kv_c, layer=l, role=R_ATTN))
            causal = 0.5 if q_len > 1 else 1.0
            scores = 2.0 * B * s.n_heads * q_len * kv_len * qk_h * causal
            outp = 2.0 * B * s.n_heads * q_len * kv_len * s.mla_v_dim * causal
            g.add(Op(f"{pre}.attn", ATTENTION, flops=scores + outp,
                     inputs=[_a(f"{pre}.q", T, q_out, dtype=ab),
                             _a(f"{pre}.kv", T, kvb_out, dtype=ab)],
                     outputs=[_a(f"{pre}.ao", T, s.n_heads * s.mla_v_dim,
                                 dtype=ab)],
                     state_read_bytes=(B * max(0, kv_len - q_len)
                                       * kv_out * dtype_bytes(kvb)),
                     m=B * s.n_heads * q_len, n=kv_len, k=qk_h,
                     layer=l, role=R_ATTN, meta=dict(kv_len=kv_len)))
            o_k = s.n_heads * s.mla_v_dim
    else:
        # ---- standard MHA / GQA ----------------------------------------
        hd = s.head_dim
        q_out = s.n_heads * hd
        kv_o = s.n_kv_heads * hd
        qkv = q_out + 2 * kv_o
        g.add(Op(f"{pre}.qkv", GEMM, flops=2.0 * T * d * qkv,
                 inputs=[_a(f"{pre}.xn", T, d, dtype=ab)],
                 outputs=[_a(f"{pre}.qkv", T, qkv, dtype=ab)],
                 weights=[_w(f"{pre}.Wqkv", d, qkv, dtype=wb)],
                 state_write_bytes=2 * T * kv_o * dtype_bytes(kvb),
                 m=T, n=qkv, k=d, layer=l, role=R_ATTN))
        g.add(Op(f"{pre}.rope", ELEMENTWISE, flops=6.0 * T * (q_out + kv_o),
                 inputs=[_a(f"{pre}.qkv", T, qkv, dtype=ab)],
                 outputs=[_a(f"{pre}.qkv_r", T, qkv, dtype=ab)],
                 m=T, n=qkv, k=1, layer=l, role=R_ATTN))

        causal = 0.5 if q_len > 1 else 1.0
        scores = 2.0 * B * s.n_heads * q_len * kv_len * hd * causal
        outp = 2.0 * B * s.n_heads * q_len * kv_len * hd * causal
        kv_read = B * max(0, kv_len - q_len) * 2 * kv_o * dtype_bytes(kvb)
        g.add(Op(f"{pre}.attn", ATTENTION, flops=scores + outp,
                 inputs=[_a(f"{pre}.qkv_r", T, qkv, dtype=ab)],
                 outputs=[_a(f"{pre}.ao", T, q_out, dtype=ab)],
                 state_read_bytes=kv_read,
                 m=B * s.n_heads * q_len, n=kv_len, k=hd,
                 layer=l, role=R_ATTN, meta=dict(kv_len=kv_len)))
        o_k = q_out

    g.add(Op(f"{pre}.o_proj", GEMM, flops=2.0 * T * o_k * d,
             inputs=[_a(f"{pre}.ao", T, o_k, dtype=ab)],
             outputs=[_a(f"{pre}.attn_out", T, d, dtype=ab)],
             weights=[_w(f"{pre}.Wo", o_k, d, dtype=wb)],
             m=T, n=d, k=o_k, layer=l, role=R_ATTN))


# =========================================================================
def _ffn_block(g: Graph, s: ModelSpec, l: int, T: int):
    d, dff = s.d_model, s.d_ff
    wb, ab = s.w_dtype, s.a_dtype
    pre = f"L{l}"
    g.add(Op(f"{pre}.norm_ffn", NORM, flops=6.0 * T * d,
             inputs=[_a(f"{pre}.attn_out", T, d, dtype=ab)],
             outputs=[_a(f"{pre}.h", T, d, dtype=ab)],
             weights=[_w(f"{pre}.wn2", d, dtype=wb)],
             m=T, n=d, k=1, layer=l, role=R_NORM))
    g.add(Op(f"{pre}.ffn_gate_up", GEMM, flops=2.0 * T * d * 2 * dff,
             inputs=[_a(f"{pre}.h", T, d, dtype=ab)],
             outputs=[_a(f"{pre}.gu", T, 2 * dff, dtype=ab)],
             weights=[_w(f"{pre}.Wgu", d, 2 * dff, dtype=wb)],
             m=T, n=2 * dff, k=d, layer=l, role=R_FFN))
    g.add(Op(f"{pre}.silu", ELEMENTWISE, flops=5.0 * T * dff,
             inputs=[_a(f"{pre}.gu", T, 2 * dff, dtype=ab)],
             outputs=[_a(f"{pre}.act", T, dff, dtype=ab)],
             m=T, n=dff, k=1, layer=l, role=R_FFN))
    g.add(Op(f"{pre}.ffn_down", GEMM, flops=2.0 * T * dff * d,
             inputs=[_a(f"{pre}.act", T, dff, dtype=ab)],
             outputs=[_a(f"{pre}.y", T, d, dtype=ab)],
             weights=[_w(f"{pre}.Wdown", dff, d, dtype=wb)],
             m=T, n=d, k=dff, layer=l, role=R_FFN))


def _moe_block(g: Graph, s: ModelSpec, l: int, T: int, ep_world: int = 1):
    d, de = s.d_model, s.d_expert_ff
    E, k = s.n_experts, s.n_active_experts
    wb, ab = s.w_dtype, s.a_dtype
    pre = f"L{l}"

    g.add(Op(f"{pre}.norm_moe", NORM, flops=6.0 * T * d,
             inputs=[_a(f"{pre}.attn_out", T, d, dtype=ab)],
             outputs=[_a(f"{pre}.h", T, d, dtype=ab)],
             weights=[_w(f"{pre}.wn2", d, dtype=wb)],
             m=T, n=d, k=1, layer=l, role=R_NORM))

    g.add(Op(f"{pre}.router", ROUTER, flops=2.0 * T * d * E,
             inputs=[_a(f"{pre}.h", T, d, dtype=ab)],
             outputs=[_a(f"{pre}.gate", T, E, dtype="fp32")],
             weights=[_w(f"{pre}.Wg", d, E, dtype=wb)],
             m=T, n=E, k=d, layer=l, role=R_MOE))

    cb = dtype_bytes(s.comm_dtype or ab)
    if ep_world > 1:
        disp = T * k * d * cb
        g.add(Op(f"{pre}.a2a_dispatch", A2A, flops=0.0,
                 inputs=[_a(f"{pre}.h", T, d, dtype=ab)],
                 outputs=[_a(f"{pre}.hd", T * k, d, dtype=ab)],
                 m=T * k, n=d, k=1, layer=l, role=R_MOE,
                 meta=dict(collective="all2all", bytes=disp,
                           world=ep_world, overlappable=True)))

    # --- the decisive term: how many distinct experts get read -----------
    n_touch = experts_touched(E, k, T)
    shared = s.n_shared_experts

    expert_w = [_w(f"{pre}.expert_pool", n_touch * 3 * d * de, dtype=wb)]
    if shared:
        expert_w.append(_w(f"{pre}.shared_expert", shared * 3 * d * de,
                           dtype=wb))

    # per-expert GEMM shape drives the efficiency model: M is tokens/expert
    m_per_expert = max(1, int(round(T * k / max(E, 1))))
    flops = T * k * 6.0 * d * de + T * shared * 6.0 * d * de
    g.add(Op(f"{pre}.moe_experts", GEMM, flops=flops,
             inputs=[_a(f"{pre}.h", T, d, dtype=ab)],
             outputs=[_a(f"{pre}.y", T, d, dtype=ab)],
             weights=expert_w,
             m=m_per_expert, n=de, k=d, layer=l, role=R_MOE,
             meta=dict(experts_touched=n_touch, top_k=k, n_experts=E,
                       tokens_per_expert=T * k / max(E, 1),
                       shared_experts=shared, tokens=T,
                       groups=n_touch + shared, ep=ep_world)))

    if ep_world > 1:
        comb = T * k * d * cb
        g.add(Op(f"{pre}.a2a_combine", A2A, flops=0.0,
                 inputs=[_a(f"{pre}.y", T, d, dtype=ab)],
                 outputs=[_a(f"{pre}.yc", T, d, dtype=ab)],
                 m=T, n=d, k=1, layer=l, role=R_MOE,
                 meta=dict(collective="all2all", bytes=comb,
                           world=ep_world, overlappable=True)))


# =========================================================================
def _build(s: ModelSpec, batch: int, q_len: int, kv_len: int, stage: str,
           ep_world: int = 1, tp_world: int = 1,
           mla_absorbed: bool = True, include_embed: bool = True,
           include_lmhead: bool = True) -> Graph:
    T = batch * q_len
    d = s.d_model
    ab = s.a_dtype
    g = Graph(name=f"{s.name}.{stage}", stage=stage, batch=batch,
              seq_len=q_len, ctx_len=kv_len,
              meta=dict(ep_world=ep_world, tp_world=tp_world))

    if include_embed:
        g.add(Op("embed", EMBED, flops=0.0,
                 outputs=[_a("h0", T, d, dtype=ab)],
                 weights=[_w("Wemb", T, d, dtype=s.w_dtype)],
                 m=T, n=d, k=1, layer=-1, role=R_EMBED,
                 meta=dict(gather_rows=T, table_rows=s.vocab)))

    for l in range(s.n_layers):
        _attention_block(g, s, l, T, batch, q_len, kv_len,
                         mla_absorbed=mla_absorbed)
        if tp_world > 1:
            g.add(Op(f"L{l}.ar_attn", ALLREDUCE, flops=0.0,
                     inputs=[_a(f"L{l}.attn_out", T, d, dtype=ab)],
                     outputs=[_a(f"L{l}.attn_out_r", T, d, dtype=ab)],
                     m=T, n=d, k=1, layer=l, role=R_ATTN,
                     meta=dict(collective="allreduce",
                               bytes=T * d * dtype_bytes(s.comm_dtype or ab),
                               world=tp_world, overlappable=True,
                               overlap_scale=0.45)))
        if s.is_moe and l >= s.first_dense_layers:
            _moe_block(g, s, l, T, ep_world=ep_world)
            role_tail = R_MOE
        else:
            _ffn_block(g, s, l, T)
            role_tail = R_FFN
        if tp_world > 1:
            g.add(Op(f"L{l}.ar_ffn", ALLREDUCE, flops=0.0,
                     inputs=[_a(f"L{l}.y", T, d, dtype=ab)],
                     outputs=[_a(f"L{l}.y_r", T, d, dtype=ab)],
                     m=T, n=d, k=1, layer=l, role=role_tail,
                     meta=dict(collective="allreduce",
                               bytes=T * d * dtype_bytes(s.comm_dtype or ab),
                               world=tp_world, overlappable=True,
                               overlap_scale=0.45)))

    if include_lmhead:
        g.add(Op("norm_final", NORM, flops=6.0 * T * d,
                 inputs=[_a("hL", T, d, dtype=ab)],
                 outputs=[_a("hLn", T, d, dtype=ab)],
                 weights=[_w("wnf", d, dtype=s.w_dtype)],
                 m=T, n=d, k=1, layer=s.n_layers, role=R_NORM))
        # only the last position needs logits during prefill
        t_head = batch if stage == "prefill" else T
        g.add(Op("lm_head", GEMM, flops=2.0 * t_head * d * s.vocab,
                 inputs=[_a("hLn", t_head, d, dtype=ab)],
                 outputs=[_a("logits", t_head, s.vocab, dtype="fp32")],
                 weights=[_w("Wlm", d, s.vocab, dtype=s.w_dtype)],
                 m=t_head, n=s.vocab, k=d, layer=s.n_layers, role=R_LMHEAD))
    return g


def build_prefill(s: ModelSpec, batch: int, chunk: int, ctx_before: int = 0,
                  **kw) -> Graph:
    """One prefill chunk of ``chunk`` tokens per sequence."""
    return _build(s, batch, chunk, ctx_before + chunk, "prefill", **kw)


def build_decode(s: ModelSpec, batch: int, ctx: int, **kw) -> Graph:
    """One decode step (1 token per sequence in the batch)."""
    return _build(s, batch, 1, ctx + 1, "decode", **kw)
