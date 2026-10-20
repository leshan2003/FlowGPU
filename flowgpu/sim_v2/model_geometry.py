"""Model-config-driven expert geometry for MoE research experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from flowgpu.units import dtype_bytes
from flowgpu.workload.graph import R_ATTN, R_EMBED, R_FFN, R_LMHEAD, R_MOE, R_NORM
from flowgpu.workload.models import get_model

from .schema import ExpertKey, ExpertSpec, Provenance, ValidationError


@dataclass(frozen=True)
class ModelExpertGeometry:
    model_name: str
    dtype: str
    n_layers: int
    n_moe_layers: int
    n_experts: int
    top_k: int
    n_shared_experts: int
    first_moe_layer: int
    expert_weight_bytes: int
    routed_expert_weight_bytes: int
    shared_expert_weight_bytes: int
    router_weight_bytes: int
    non_moe_weight_bytes: int
    total_weight_bytes: int

    def routed_experts(
        self,
        provenance: Provenance,
    ) -> Tuple[ExpertSpec, ...]:
        return tuple(
            ExpertSpec(
                ExpertKey(layer_id, expert_id),
                self.expert_weight_bytes,
                provenance,
            )
            for layer_id in range(self.first_moe_layer, self.n_layers)
            for expert_id in range(self.n_experts)
        )


def build_model_expert_geometry(
    model_name: str,
    dtype: str | None = None,
) -> ModelExpertGeometry:
    """Derive routed-expert size from a local public model configuration."""

    model = get_model(model_name)
    if not model.is_moe:
        raise ValidationError(
            "model %s is not a mixture-of-experts model" % model_name
        )
    weight_dtype = dtype or model.w_dtype
    element_bytes = dtype_bytes(weight_dtype)
    n_moe_layers = model.n_layers - model.first_dense_layers
    if n_moe_layers <= 0:
        raise ValidationError("model has no MoE layers")

    expert_elements = 3 * model.d_model * model.d_expert_ff
    expert_bytes = int(math.ceil(expert_elements * element_bytes))
    routed_bytes = n_moe_layers * model.n_experts * expert_bytes
    shared_bytes = (
        n_moe_layers * model.n_shared_experts * expert_bytes
    )
    router_bytes = int(math.ceil(
        n_moe_layers * model.d_model * model.n_experts * element_bytes
    ))

    params = model.params()
    non_moe_elements = sum(
        params[role]
        for role in (R_EMBED, R_ATTN, R_FFN, R_NORM, R_LMHEAD)
    )
    non_moe_bytes = int(math.ceil(non_moe_elements * element_bytes))
    moe_elements = params[R_MOE]
    expected_moe_elements = (
        n_moe_layers * (
            model.d_model * model.n_experts
            + (model.n_experts + model.n_shared_experts)
            * expert_elements
        )
    )
    if not math.isclose(
        moe_elements,
        expected_moe_elements,
        rel_tol=1e-12,
        abs_tol=1.0,
    ):
        raise ValidationError(
            "model MoE parameter accounting does not match geometry"
        )

    return ModelExpertGeometry(
        model_name=model_name,
        dtype=weight_dtype,
        n_layers=model.n_layers,
        n_moe_layers=n_moe_layers,
        n_experts=model.n_experts,
        top_k=model.n_active_experts,
        n_shared_experts=model.n_shared_experts,
        first_moe_layer=model.first_dense_layers,
        expert_weight_bytes=expert_bytes,
        routed_expert_weight_bytes=routed_bytes,
        shared_expert_weight_bytes=shared_bytes,
        router_weight_bytes=router_bytes,
        non_moe_weight_bytes=non_moe_bytes,
        total_weight_bytes=(
            non_moe_bytes + routed_bytes + shared_bytes + router_bytes
        ),
    )
