import math

import pytest

from flowgpu.sim_v2.model_geometry import build_model_expert_geometry
from flowgpu.sim_v2.schema import Confidence, Provenance, ValidationError


def test_deepseek_geometry_matches_experiment_expert_size():
    geometry = build_model_expert_geometry("deepseek_v3", dtype="fp8")
    assert geometry.n_layers == 61
    assert geometry.n_moe_layers == 58
    assert geometry.n_experts == 256
    assert geometry.top_k == 8
    assert geometry.n_shared_experts == 1
    assert geometry.expert_weight_bytes == 3 * 7168 * 2048
    assert geometry.routed_expert_weight_bytes == 609 * 2**30
    experts = geometry.routed_experts(
        Provenance(Confidence.DERIVED, "test")
    )
    assert len(experts) == 58 * 256
    assert experts[0].key.layer_id == 3
    assert experts[-1].key.layer_id == 60


def test_mixtral_and_qwen_geometry_are_model_specific():
    mixtral = build_model_expert_geometry("mixtral_8x7b", dtype="fp8")
    qwen = build_model_expert_geometry("qwen3_235b_a22b", dtype="fp8")
    assert mixtral.n_experts == 8
    assert mixtral.top_k == 2
    assert mixtral.expert_weight_bytes == 3 * 4096 * 14336
    assert qwen.n_experts == 128
    assert qwen.top_k == 8
    assert qwen.expert_weight_bytes == 3 * 4096 * 1536
    assert mixtral.expert_weight_bytes != qwen.expert_weight_bytes


def test_non_moe_model_is_rejected():
    with pytest.raises(ValidationError, match="not a mixture-of-experts"):
        build_model_expert_geometry("llama3_8b")


def test_total_weight_bytes_is_consistent():
    geometry = build_model_expert_geometry("deepseek_v3", dtype="fp8")
    assert geometry.total_weight_bytes == (
        geometry.non_moe_weight_bytes
        + geometry.routed_expert_weight_bytes
        + geometry.shared_expert_weight_bytes
        + geometry.router_weight_bytes
    )
    assert math.isclose(
        geometry.total_weight_bytes / 1e9,
        671.0,
        rel_tol=0.02,
    )
