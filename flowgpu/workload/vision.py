"""Non-LLM workloads: ViT encoders (for VLM), diffusion (DiT / U-Net), CNN.

These share the same :class:`~flowgpu.workload.graph.Graph` IR as the LLM
builders, so the placement engine, hardware models and power model all apply
unchanged.  That is the point of having one IR: you can ask "what if the ViT
encoder goes on the dataflow chip and the LLM decoder stays on the GPU?" and
the answer comes out of the same machinery.
"""

from __future__ import annotations

import math

from ..units import dtype_bytes
from .graph import (ATTENTION, CONV, ELEMENTWISE, GEMM, NORM, R_ATTN, R_CONV,
                    R_DIFFUSION, R_FFN, R_NORM, R_VISION, Graph, ModelSpec, Op,
                    Tensor)
from .llm import build_decode, build_prefill


def _w(n, *s, dtype):
    return Tensor(n, tuple(s), dtype, "weight")


def _a(n, *s, dtype):
    return Tensor(n, tuple(s), dtype, "act")


# =========================================================================
# ViT encoder
# =========================================================================
def build_vit(spec: ModelSpec, n_images: int = 1, role: str = R_VISION,
              w_dtype: str | None = None, a_dtype: str | None = None) -> Graph:
    """Vision-transformer encoder pass over ``n_images`` images."""
    v = spec.extra.get("vision")
    if not v:
        raise ValueError(f"{spec.name} has no vision tower")
    wb = w_dtype or spec.w_dtype
    ab = a_dtype or spec.a_dtype

    img = v["image_size"]
    patch = v["patch"]
    d = v["d_vision"]
    L = v["n_vision_layers"]
    H = v["n_vision_heads"]
    dff = v["d_vision_ff"]
    hd = d // H

    n_patch = (img // patch) ** 2
    T = n_images * n_patch

    g = Graph(name=f"{spec.name}.vit", stage="prefill", batch=n_images,
              seq_len=n_patch, ctx_len=n_patch,
              meta=dict(n_patches=n_patch, n_images=n_images))

    # patchify = conv with kernel == stride
    in_ch = 3
    g.add(Op("vit.patchify", CONV, flops=2.0 * T * in_ch * patch * patch * d,
             inputs=[_a("img", n_images, 3, img, img, dtype=ab)],
             outputs=[_a("vit.h0", T, d, dtype=ab)],
             weights=[_w("vit.Wp", in_ch * patch * patch, d, dtype=wb)],
             m=T, n=d, k=in_ch * patch * patch, layer=-1, role=role))

    for l in range(L):
        p = f"vit.L{l}"
        g.add(Op(f"{p}.ln1", NORM, flops=6.0 * T * d,
                 inputs=[_a(f"{p}.x", T, d, dtype=ab)],
                 outputs=[_a(f"{p}.xn", T, d, dtype=ab)],
                 weights=[_w(f"{p}.wn1", d, dtype=wb)],
                 m=T, n=d, k=1, layer=l, role=role))
        g.add(Op(f"{p}.qkv", GEMM, flops=2.0 * T * d * 3 * d,
                 inputs=[_a(f"{p}.xn", T, d, dtype=ab)],
                 outputs=[_a(f"{p}.qkv", T, 3 * d, dtype=ab)],
                 weights=[_w(f"{p}.Wqkv", d, 3 * d, dtype=wb)],
                 m=T, n=3 * d, k=d, layer=l, role=role))
        # bidirectional attention: no causal halving
        g.add(Op(f"{p}.attn", ATTENTION,
                 flops=4.0 * n_images * H * n_patch * n_patch * hd,
                 inputs=[_a(f"{p}.qkv", T, 3 * d, dtype=ab)],
                 outputs=[_a(f"{p}.ao", T, d, dtype=ab)],
                 m=n_images * H * n_patch, n=n_patch, k=hd,
                 layer=l, role=role))
        g.add(Op(f"{p}.proj", GEMM, flops=2.0 * T * d * d,
                 inputs=[_a(f"{p}.ao", T, d, dtype=ab)],
                 outputs=[_a(f"{p}.y", T, d, dtype=ab)],
                 weights=[_w(f"{p}.Wo", d, d, dtype=wb)],
                 m=T, n=d, k=d, layer=l, role=role))
        g.add(Op(f"{p}.ln2", NORM, flops=6.0 * T * d,
                 inputs=[_a(f"{p}.y", T, d, dtype=ab)],
                 outputs=[_a(f"{p}.yn", T, d, dtype=ab)],
                 weights=[_w(f"{p}.wn2", d, dtype=wb)],
                 m=T, n=d, k=1, layer=l, role=role))
        g.add(Op(f"{p}.fc1", GEMM, flops=2.0 * T * d * dff,
                 inputs=[_a(f"{p}.yn", T, d, dtype=ab)],
                 outputs=[_a(f"{p}.h", T, dff, dtype=ab)],
                 weights=[_w(f"{p}.W1", d, dff, dtype=wb)],
                 m=T, n=dff, k=d, layer=l, role=role))
        g.add(Op(f"{p}.gelu", ELEMENTWISE, flops=8.0 * T * dff,
                 inputs=[_a(f"{p}.h", T, dff, dtype=ab)],
                 outputs=[_a(f"{p}.hg", T, dff, dtype=ab)],
                 m=T, n=dff, k=1, layer=l, role=role))
        g.add(Op(f"{p}.fc2", GEMM, flops=2.0 * T * dff * d,
                 inputs=[_a(f"{p}.hg", T, dff, dtype=ab)],
                 outputs=[_a(f"{p}.out", T, d, dtype=ab)],
                 weights=[_w(f"{p}.W2", dff, d, dtype=wb)],
                 m=T, n=d, k=dff, layer=l, role=role))

    # merge + projector into the LLM hidden size
    merge = v.get("merge", 1)
    n_out = T // (merge * merge) if merge > 1 else T
    g.add(Op("vit.projector", GEMM,
             flops=2.0 * n_out * d * merge * merge * spec.d_model,
             inputs=[_a("vit.pooled", n_out, d * merge * merge, dtype=ab)],
             outputs=[_a("vit.embeds", n_out, spec.d_model, dtype=ab)],
             weights=[_w("vit.Wproj", d * merge * merge, spec.d_model,
                         dtype=wb)],
             m=n_out, n=spec.d_model, k=d * merge * merge,
             layer=L, role=role))
    g.meta["n_visual_tokens"] = n_out
    return g


def vlm_visual_tokens(spec: ModelSpec, n_images: int = 1) -> int:
    v = spec.extra.get("vision", {})
    if not v:
        return 0
    n_patch = (v["image_size"] // v["patch"]) ** 2
    merge = v.get("merge", 1)
    return n_images * n_patch // (merge * merge)


def build_vlm_prefill(spec: ModelSpec, batch: int, text_tokens: int,
                      n_images: int = 1, **kw) -> tuple:
    """Returns (vision_graph, llm_prefill_graph).

    The two are separate graphs so they can be placed on *different* devices,
    which is exactly the kind of split the study is about.
    """
    vg = build_vit(spec, n_images=n_images * batch)
    vtok = vlm_visual_tokens(spec, n_images)
    total = text_tokens + vtok
    lg = build_prefill(spec, batch, total, ctx_before=0, **kw)
    lg.meta["visual_tokens"] = vtok
    lg.meta["text_tokens"] = text_tokens
    return vg, lg


# =========================================================================
# Diffusion
# =========================================================================
def build_diffusion_step(spec: ModelSpec, batch: int = 1) -> Graph:
    """One denoising step of a DiT or U-Net diffusion model."""
    e = spec.extra
    wb, ab = spec.w_dtype, spec.a_dtype
    if e.get("kind") == "dit":
        return _build_dit_step(spec, batch, e, wb, ab)
    return _build_unet_step(spec, batch, e, wb, ab)


def _build_dit_step(spec, batch, e, wb, ab) -> Graph:
    d = e["d_model"]
    L = e["n_layers"]
    H = e["n_heads"]
    hd = d // H
    dff = e["d_ff"]
    lat = e["latent"]
    patch = e["patch"]
    n_img_tok = (lat // patch) ** 2
    T_txt = e.get("text_tokens", 0)
    S = n_img_tok + T_txt
    T = batch * S

    g = Graph(name=f"{spec.name}.dit_step", stage="prefill", batch=batch,
              seq_len=S, ctx_len=S,
              meta=dict(image_tokens=n_img_tok, text_tokens=T_txt))
    g.add(Op("dit.embed", GEMM, flops=2.0 * batch * n_img_tok * (patch * patch * 16) * d,
             inputs=[_a("latent", batch, 16, lat, lat, dtype=ab)],
             outputs=[_a("dit.h0", T, d, dtype=ab)],
             weights=[_w("dit.Wemb", patch * patch * 16, d, dtype=wb)],
             m=batch * n_img_tok, n=d, k=patch * patch * 16,
             layer=-1, role=R_DIFFUSION))
    for l in range(L):
        p = f"dit.L{l}"
        g.add(Op(f"{p}.mod", ELEMENTWISE, flops=12.0 * T * d,
                 inputs=[_a(f"{p}.x", T, d, dtype=ab)],
                 outputs=[_a(f"{p}.xm", T, d, dtype=ab)],
                 weights=[_w(f"{p}.Wmod", d, 6 * d, dtype=wb)],
                 m=T, n=d, k=1, layer=l, role=R_DIFFUSION))
        g.add(Op(f"{p}.qkv", GEMM, flops=2.0 * T * d * 3 * d,
                 inputs=[_a(f"{p}.xm", T, d, dtype=ab)],
                 outputs=[_a(f"{p}.qkv", T, 3 * d, dtype=ab)],
                 weights=[_w(f"{p}.Wqkv", d, 3 * d, dtype=wb)],
                 m=T, n=3 * d, k=d, layer=l, role=R_DIFFUSION))
        g.add(Op(f"{p}.attn", ATTENTION, flops=4.0 * batch * H * S * S * hd,
                 inputs=[_a(f"{p}.qkv", T, 3 * d, dtype=ab)],
                 outputs=[_a(f"{p}.ao", T, d, dtype=ab)],
                 m=batch * H * S, n=S, k=hd, layer=l, role=R_DIFFUSION))
        g.add(Op(f"{p}.proj", GEMM, flops=2.0 * T * d * d,
                 inputs=[_a(f"{p}.ao", T, d, dtype=ab)],
                 outputs=[_a(f"{p}.y", T, d, dtype=ab)],
                 weights=[_w(f"{p}.Wo", d, d, dtype=wb)],
                 m=T, n=d, k=d, layer=l, role=R_DIFFUSION))
        g.add(Op(f"{p}.fc1", GEMM, flops=2.0 * T * d * dff,
                 inputs=[_a(f"{p}.y", T, d, dtype=ab)],
                 outputs=[_a(f"{p}.h", T, dff, dtype=ab)],
                 weights=[_w(f"{p}.W1", d, dff, dtype=wb)],
                 m=T, n=dff, k=d, layer=l, role=R_DIFFUSION))
        g.add(Op(f"{p}.gelu", ELEMENTWISE, flops=8.0 * T * dff,
                 inputs=[_a(f"{p}.h", T, dff, dtype=ab)],
                 outputs=[_a(f"{p}.hg", T, dff, dtype=ab)],
                 m=T, n=dff, k=1, layer=l, role=R_DIFFUSION))
        g.add(Op(f"{p}.fc2", GEMM, flops=2.0 * T * dff * d,
                 inputs=[_a(f"{p}.hg", T, dff, dtype=ab)],
                 outputs=[_a(f"{p}.out", T, d, dtype=ab)],
                 weights=[_w(f"{p}.W2", dff, d, dtype=wb)],
                 m=T, n=d, k=dff, layer=l, role=R_DIFFUSION))
    g.add(Op("dit.unpatch", GEMM,
             flops=2.0 * batch * n_img_tok * d * patch * patch * 16,
             inputs=[_a("dit.hL", batch * n_img_tok, d, dtype=ab)],
             outputs=[_a("noise", batch, 16, lat, lat, dtype=ab)],
             weights=[_w("dit.Wout", d, patch * patch * 16, dtype=wb)],
             m=batch * n_img_tok, n=patch * patch * 16, k=d,
             layer=L, role=R_DIFFUSION))
    return g


def _build_unet_step(spec, batch, e, wb, ab) -> Graph:
    """U-Net diffusion step, modelled at block granularity."""
    base = e["base_ch"]
    lat = e["latent"]
    g = Graph(name=f"{spec.name}.unet_step", stage="prefill", batch=batch,
              seq_len=1, ctx_len=1)
    # down/mid/up: channels double and resolution halves per stage
    stages = [(base, lat), (base * 2, lat // 2), (base * 4, lat // 4),
              (base * 4, lat // 8)]
    idx = 0
    for direction, seq in (("down", stages), ("up", list(reversed(stages)))):
        for ch, res in seq:
            for r in range(e.get("n_res", 2)):
                idx += 1
                n_px = batch * res * res
                fl = 2.0 * n_px * ch * ch * 9
                g.add(Op(f"unet.{direction}{idx}.res{r}", CONV, flops=fl,
                         inputs=[_a(f"unet.x{idx}_{r}", batch, ch, res, res,
                                    dtype=ab)],
                         outputs=[_a(f"unet.y{idx}_{r}", batch, ch, res, res,
                                     dtype=ab)],
                         weights=[_w(f"unet.W{idx}_{r}", ch, ch, 3, 3,
                                     dtype=wb)],
                         m=n_px, n=ch, k=ch * 9, layer=idx, role=R_CONV))
            if ch >= e.get("attn_dim", 1280) // 2:
                S = res * res
                ctx = e.get("ctx_tokens", 77)
                g.add(Op(f"unet.{direction}{idx}.attn", ATTENTION,
                         flops=4.0 * batch * S * (S + ctx) * ch,
                         inputs=[_a(f"unet.a{idx}", batch * S, ch, dtype=ab)],
                         outputs=[_a(f"unet.ao{idx}", batch * S, ch,
                                     dtype=ab)],
                         weights=[_w(f"unet.Wattn{idx}", ch, 4 * ch,
                                     dtype=wb)],
                         m=batch * S, n=S + ctx, k=ch, layer=idx,
                         role=R_DIFFUSION))
    return g


def build_vae_decode(spec: ModelSpec, batch: int = 1) -> Graph:
    e = spec.extra
    lat = e["latent"]
    img = e.get("image", lat * e.get("vae_scale", 8))
    wb, ab = spec.w_dtype, spec.a_dtype
    g = Graph(name=f"{spec.name}.vae", stage="prefill", batch=batch,
              seq_len=1, ctx_len=1)
    ch = 512
    res = lat
    i = 0
    while res <= img:
        i += 1
        n_px = batch * res * res
        g.add(Op(f"vae.up{i}", CONV, flops=2.0 * n_px * ch * ch * 9,
                 inputs=[_a(f"vae.x{i}", batch, ch, res, res, dtype=ab)],
                 outputs=[_a(f"vae.y{i}", batch, ch, res, res, dtype=ab)],
                 weights=[_w(f"vae.W{i}", ch, ch, 3, 3, dtype=wb)],
                 m=n_px, n=ch, k=ch * 9, layer=i, role=R_CONV))
        res *= 2
        ch = max(128, ch // 2)
    return g


# =========================================================================
# CNN
# =========================================================================
def build_cnn(spec: ModelSpec, batch: int = 1) -> Graph:
    """Classic CNN modelled from published FLOPs/params/activation volume.

    CNNs are layer-shape heavy and the per-layer detail changes the answer
    very little at this level of abstraction; what matters is the FLOP:byte
    ratio, which published aggregates capture accurately.
    """
    e = spec.extra
    wb, ab = spec.w_dtype, spec.a_dtype
    g = Graph(name=f"{spec.name}.infer", stage="prefill", batch=batch,
              seq_len=1, ctx_len=1)
    n_blocks = 16
    fl = e["flops_per_image"] * batch / n_blocks
    pb = e["params"] / n_blocks
    ab_mb = e["act_mb"] * batch / n_blocks * 2**20
    for i in range(n_blocks):
        g.add(Op(f"cnn.block{i}", CONV, flops=fl,
                 inputs=[Tensor(f"cnn.x{i}", (int(ab_mb / 2),), ab, "act")],
                 outputs=[Tensor(f"cnn.y{i}", (int(ab_mb / 2),), ab, "act")],
                 weights=[Tensor(f"cnn.W{i}", (int(pb),), wb, "weight")],
                 m=batch * 196, n=int(math.sqrt(max(pb, 1))),
                 k=int(math.sqrt(max(pb, 1))), layer=i, role=R_CONV))
    return g
