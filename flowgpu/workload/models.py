"""Model zoo.

Architecture configs for the workloads under study.  Every entry is a
:class:`~flowgpu.workload.graph.ModelSpec`; ``source`` notes where the numbers
come from.  Anything whose name ends in ``_proj`` is a *projection*, not a
published config, and is flagged as such in reports.
"""

from __future__ import annotations

import copy

from .graph import ModelSpec

_LLM = {
    # ------------------------------------------------------------- dense
    "llama3_8b": dict(
        n_layers=32, d_model=4096, n_heads=32, n_kv_heads=8, head_dim=128,
        d_ff=14336, vocab=128256,
        source="Meta Llama-3.1-8B config.json"),
    "llama3_70b": dict(
        n_layers=80, d_model=8192, n_heads=64, n_kv_heads=8, head_dim=128,
        d_ff=28672, vocab=128256,
        source="Meta Llama-3.1-70B config.json"),
    "llama3_405b": dict(
        n_layers=126, d_model=16384, n_heads=128, n_kv_heads=8, head_dim=128,
        d_ff=53248, vocab=128256,
        source="Meta Llama-3.1-405B config.json"),
    "llama3_1b": dict(
        n_layers=16, d_model=2048, n_heads=32, n_kv_heads=8, head_dim=64,
        d_ff=8192, vocab=128256,
        source="Meta Llama-3.2-1B config.json"),
    "qwen3_32b": dict(
        n_layers=64, d_model=5120, n_heads=64, n_kv_heads=8, head_dim=128,
        d_ff=25600, vocab=151936,
        source="Qwen3-32B config.json"),
    "qwen3_8b": dict(
        n_layers=36, d_model=4096, n_heads=32, n_kv_heads=8, head_dim=128,
        d_ff=12288, vocab=151936,
        source="Qwen3-8B config.json"),
    # --------------------------------------------------------------- MoE
    "mixtral_8x7b": dict(
        n_layers=32, d_model=4096, n_heads=32, n_kv_heads=8, head_dim=128,
        d_ff=14336, vocab=32000,
        n_experts=8, n_active_experts=2, d_expert_ff=14336,
        source="Mistral Mixtral-8x7B config.json"),
    "qwen3_235b_a22b": dict(
        n_layers=94, d_model=4096, n_heads=64, n_kv_heads=4, head_dim=128,
        d_ff=12288, vocab=151936,
        n_experts=128, n_active_experts=8, d_expert_ff=1536,
        source="Qwen3-235B-A22B config.json"),
    "gpt_oss_120b": dict(
        n_layers=36, d_model=2880, n_heads=64, n_kv_heads=8, head_dim=64,
        d_ff=2880, vocab=201088,
        n_experts=128, n_active_experts=4, d_expert_ff=2880,
        w_dtype="mxfp4",
        source="openai/gpt-oss-120b config.json"),
    # ---------------------------------------------------- DeepSeek (MLA)
    "deepseek_v3": dict(
        n_layers=61, d_model=7168, n_heads=128, n_kv_heads=128, head_dim=128,
        d_ff=18432, vocab=129280,
        n_experts=256, n_active_experts=8, d_expert_ff=2048,
        n_shared_experts=1, first_dense_layers=3,
        mla_kv_lora_rank=512, mla_qk_rope_dim=64, mla_qk_nope_dim=128,
        mla_v_dim=128, mla_q_lora_rank=1536,
        source="deepseek-ai/DeepSeek-V3 config.json"),
    "deepseek_r1": dict(
        n_layers=61, d_model=7168, n_heads=128, n_kv_heads=128, head_dim=128,
        d_ff=18432, vocab=129280,
        n_experts=256, n_active_experts=8, d_expert_ff=2048,
        n_shared_experts=1, first_dense_layers=3,
        mla_kv_lora_rank=512, mla_qk_rope_dim=64, mla_qk_nope_dim=128,
        mla_v_dim=128, mla_q_lora_rank=1536,
        source="deepseek-ai/DeepSeek-R1 config.json (same arch as V3)"),
    "deepseek_v4_flash_proj": dict(
        # PROJECTION.  The 2026 中国算力大会 announcement names "DeepSeek V4
        # Flash"; no config has been published.  "Flash" implies a small-active
        # sibling of the V3/V3.2 line, so this is modelled as ~160 B total /
        # ~7 B active: same MLA attention, many narrow experts, high sparsity.
        # Every number here is the simulator author's construction.  Run
        # `deepseek_v3` alongside it for a fully-grounded comparison.
        n_layers=48, d_model=4096, n_heads=64, n_kv_heads=64, head_dim=128,
        d_ff=11264, vocab=129280,
        n_experts=256, n_active_experts=8, d_expert_ff=1024,
        n_shared_experts=1, first_dense_layers=2,
        mla_kv_lora_rank=448, mla_qk_rope_dim=64, mla_qk_nope_dim=128,
        mla_v_dim=128, mla_q_lora_rank=1024,
        source="PROJECTION -- not a published config", projected=True),
    "deepseek_v2_lite": dict(
        n_layers=27, d_model=2048, n_heads=16, n_kv_heads=16, head_dim=128,
        d_ff=10944, vocab=102400,
        n_experts=64, n_active_experts=6, d_expert_ff=1408,
        n_shared_experts=2, first_dense_layers=1,
        mla_kv_lora_rank=512, mla_qk_rope_dim=64, mla_qk_nope_dim=128,
        mla_v_dim=128, mla_q_lora_rank=0,
        source="deepseek-ai/DeepSeek-V2-Lite config.json"),
}

# --------------------------------------------------------------- VLM
_VLM = {
    "qwen2_5_vl_7b": dict(
        base="qwen3_8b",
        n_layers=28, d_model=3584, n_heads=28, n_kv_heads=4, head_dim=128,
        d_ff=18944, vocab=152064,
        extra=dict(vision=dict(
            patch=14, merge=2, d_vision=1280, n_vision_layers=32,
            n_vision_heads=16, d_vision_ff=3420, image_size=1024)),
        source="Qwen2.5-VL-7B config.json"),
    "internvl3_38b": dict(
        n_layers=64, d_model=5120, n_heads=40, n_kv_heads=8, head_dim=128,
        d_ff=27648, vocab=151674,
        extra=dict(vision=dict(
            patch=14, merge=2, d_vision=3200, n_vision_layers=45,
            n_vision_heads=25, d_vision_ff=12800, image_size=448)),
        source="InternVL3-38B config.json"),
}

# ------------------------------------------------------- diffusion / CNN
_DIFFUSION = {
    "sd3_medium": dict(
        family="diffusion",
        extra=dict(kind="dit", d_model=1536, n_layers=24, n_heads=24,
                   patch=2, latent=128, d_ff=6144, text_tokens=333,
                   default_steps=28, vae_scale=8, image=1024)),
    "flux_dev": dict(
        family="diffusion",
        extra=dict(kind="dit", d_model=3072, n_layers=57, n_heads=24,
                   patch=2, latent=128, d_ff=12288, text_tokens=512,
                   default_steps=28, vae_scale=8, image=1024)),
    "sdxl": dict(
        family="diffusion",
        extra=dict(kind="unet", base_ch=320, n_res=2, latent=128,
                   default_steps=30, vae_scale=8, image=1024,
                   attn_dim=1280, ctx_tokens=77)),
}

_CNN = {
    "resnet50": dict(family="cnn", extra=dict(arch="resnet50", image=224,
                                              flops_per_image=8.2e9,
                                              params=25.6e6, act_mb=110)),
    "resnet18": dict(family="cnn", extra=dict(arch="resnet18", image=224,
                                              flops_per_image=3.6e9,
                                              params=11.7e6, act_mb=30)),
    "yolov8l": dict(family="cnn", extra=dict(arch="yolov8l", image=640,
                                             flops_per_image=165e9,
                                             params=43.7e6, act_mb=260)),
    "vit_l_16": dict(family="cnn", extra=dict(arch="vit_l_16", image=224,
                                              flops_per_image=61.6e9,
                                              params=304e6, act_mb=180)),
}


def list_models() -> dict:
    return {"llm": sorted(_LLM), "vlm": sorted(_VLM),
            "diffusion": sorted(_DIFFUSION), "cnn": sorted(_CNN)}


def get_model(name: str, **overrides) -> ModelSpec:
    for table, family in ((_LLM, "llm"), (_VLM, "vlm"),
                          (_DIFFUSION, "diffusion"), (_CNN, "cnn")):
        if name in table:
            cfg = copy.deepcopy(table[name])
            cfg.pop("base", None)
            src = cfg.pop("source", "")
            proj = cfg.pop("projected", False)
            fam = cfg.pop("family", family)
            cfg.update(overrides)
            extra = cfg.pop("extra", {})
            spec = ModelSpec(name=name, family=fam, extra=extra, **cfg)
            spec.extra["source"] = src
            spec.extra["projected"] = proj
            if spec.n_kv_heads == 0:
                spec.n_kv_heads = spec.n_heads
            return spec
    raise KeyError(f"unknown model {name!r}. Known: {list_models()}")
