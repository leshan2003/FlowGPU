"""Strict YAML entry point for reproducible v2 node-count sweeps."""

from __future__ import annotations

from pathlib import Path

import yaml

from .experiments import (NodeSweepSpec, SweepWorkloadSpec,
                          build_node_count_sweep)
from .kernel_model import ExpertKernelCurve, KernelSample
from .energy import ChipPowerProfile
from .results import (write_node_sweep_bundle,
                      write_serving_sweep_bundle,
                      write_tiered_sweep_bundle,
                      write_uncertainty_bundle)
from .runner import (run_node_count_comparisons,
                     run_node_count_uncertainty,
                     run_node_serving_sweep,
                     run_tiered_node_count_comparisons)
from .schema import (ExpertKey, ExpertSpec, Provenance,
                     RuntimeCapabilities, ValidationError)
from .serving import AdmissionPolicy, ResourceBlockWindow
from .topology import LinkProfile


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValidationError("%s must be a mapping" % name)
    return value


def _keys(value, required, optional=()):
    unknown = set(value) - set(required) - set(optional)
    missing = set(required) - set(value)
    if missing:
        raise ValidationError(
            "config missing field: %s" % sorted(missing)[0]
        )
    if unknown:
        raise ValidationError(
            "config unknown field: %s" % sorted(unknown)[0]
        )


def _provenance(value, name):
    item = _mapping(value, name)
    _keys(
        item, ("confidence", "source"),
        ("artifact", "measured_at", "machine_revision"),
    )
    return Provenance(
        confidence=item["confidence"],
        source=str(item["source"]),
        artifact=item.get("artifact"),
        measured_at=item.get("measured_at"),
        machine_revision=item.get("machine_revision"),
    )


def _link_profile(value, name):
    item = _mapping(value, name)
    _keys(item, (
        "bandwidth_bytes_per_s", "latency_s", "efficiency",
        "energy_per_byte", "duplex", "provenance",
    ))
    return LinkProfile(
        bandwidth_bytes_per_s=float(item["bandwidth_bytes_per_s"]),
        latency_s=float(item["latency_s"]),
        efficiency=float(item["efficiency"]),
        energy_per_byte=float(item["energy_per_byte"]),
        duplex=str(item["duplex"]),
        provenance=_provenance(
            item["provenance"], name + ".provenance"
        ),
    )


def _kernel_sample(value, index):
    item = _mapping(value, "kernel.samples[%d]" % index)
    _keys(item, ("m", "latency_s"))
    return KernelSample(int(item["m"]), float(item["latency_s"]))


def _load_node_sweep_config(config_path):
    """Load and resolve the shared strict node-sweep configuration."""

    path = Path(config_path).resolve()
    if not path.is_file():
        raise ValidationError("v2 config is not a file: %s" % path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError("cannot parse v2 config: %s" % path) from exc
    config = _mapping(raw, "config")
    _keys(config, (
        "schema_version", "sweep", "chip", "links", "model",
        "kernel", "workload", "execution",
    ), ("serving", "uncertainty", "power", "tiered"))
    if config["schema_version"] != 1:
        raise ValidationError("unsupported v2 config schema_version")

    sweep = _mapping(config["sweep"], "sweep")
    _keys(sweep, (
        "node_counts", "chips_per_node", "ingress_modes",
        "concentrated_ingress_count",
    ))
    chip = _mapping(config["chip"], "chip")
    _keys(chip, (
        "sram_capacity_bytes", "runtime_reserve_fraction",
        "fixed_reserved_bytes", "provenance",
    ))
    links = _mapping(config["links"], "links")
    _keys(links, ("scale_up", "scale_out", "bridge"))
    model = _mapping(config["model"], "model")
    _keys(model, (
        "n_layers", "n_experts", "top_k", "expert_weight_bytes",
        "hidden_bytes", "expert_provenance",
    ))
    kernel = _mapping(config["kernel"], "kernel")
    _keys(kernel, (
        "expert_shape", "dtype", "concurrent_experts", "residency",
        "samples", "provenance",
    ))
    workload = _mapping(config["workload"], "workload")
    _keys(workload, (
        "placement_tokens", "evaluation_tokens", "placement_seed",
        "evaluation_seed", "scaling_mode", "routing_model",
        "routing_provenance",
    ), (
        "placement_expert_weights", "evaluation_expert_weights",
        "placement_trace_template", "evaluation_trace_template",
    ))
    execution = _mapping(config["execution"], "execution")
    _keys(execution, (
        "strategies", "max_total_replicas", "dispatcher_policy",
        "ingress_semantics", "capabilities",
        "chip_reduce_latency_s", "node_reduce_latency_s",
    ))
    capabilities = _mapping(
        execution["capabilities"], "execution.capabilities"
    )
    _keys(capabilities, (
        "dispatch_dedup_chip", "dispatch_dedup_node",
        "combine_reduce_chip", "combine_reduce_node",
        "hardware_multicast",
    ))

    node_spec = NodeSweepSpec(
        node_counts=tuple(int(value) for value in sweep["node_counts"]),
        chips_per_node=int(sweep["chips_per_node"]),
        chip_sram_capacity_bytes=int(chip["sram_capacity_bytes"]),
        runtime_reserve_fraction=float(
            chip["runtime_reserve_fraction"]
        ),
        fixed_reserved_bytes=int(chip["fixed_reserved_bytes"]),
        chip_provenance=_provenance(
            chip["provenance"], "chip.provenance"
        ),
        ingress_modes=tuple(str(value)
                            for value in sweep["ingress_modes"]),
        concentrated_ingress_count=int(
            sweep["concentrated_ingress_count"]
        ),
    )
    scale_up = _link_profile(links["scale_up"], "links.scale_up")
    scale_out = _link_profile(links["scale_out"], "links.scale_out")
    bridge = _link_profile(links["bridge"], "links.bridge")
    expert_provenance = _provenance(
        model["expert_provenance"], "model.expert_provenance"
    )
    n_layers = int(model["n_layers"])
    n_experts = int(model["n_experts"])
    experts = tuple(
        ExpertSpec(
            ExpertKey(layer_id, expert_id),
            int(model["expert_weight_bytes"]),
            expert_provenance,
        )
        for layer_id in range(n_layers)
        for expert_id in range(n_experts)
    )
    cases = build_node_count_sweep(
        node_spec, scale_up, scale_out, bridge, experts
    )
    samples = tuple(
        _kernel_sample(item, index)
        for index, item in enumerate(kernel["samples"])
    )
    curve = ExpertKernelCurve(
        expert_shape=str(kernel["expert_shape"]),
        dtype=str(kernel["dtype"]),
        concurrent_experts=int(kernel["concurrent_experts"]),
        residency=str(kernel["residency"]),
        samples=samples,
        provenance=_provenance(
            kernel["provenance"], "kernel.provenance"
        ),
    )
    kernels = {expert.key: curve for expert in experts}
    workload_spec = SweepWorkloadSpec(
        n_layers=n_layers,
        n_experts=n_experts,
        top_k=int(model["top_k"]),
        placement_tokens=int(workload["placement_tokens"]),
        evaluation_tokens=int(workload["evaluation_tokens"]),
        placement_seed=int(workload["placement_seed"]),
        evaluation_seed=int(workload["evaluation_seed"]),
        provenance=_provenance(
            workload["routing_provenance"],
            "workload.routing_provenance",
        ),
        scaling_mode=str(workload["scaling_mode"]),
        routing_model=str(workload["routing_model"]),
        placement_expert_weights=tuple(
            float(value) for value in
            workload.get("placement_expert_weights", ())
        ),
        evaluation_expert_weights=tuple(
            float(value) for value in
            workload.get("evaluation_expert_weights", ())
        ),
        placement_trace_template=(
            str((path.parent / str(
                workload["placement_trace_template"]
            )).resolve())
            if workload.get("placement_trace_template") and
            not Path(str(workload["placement_trace_template"])).is_absolute()
            else workload.get("placement_trace_template")
        ),
        evaluation_trace_template=(
            str((path.parent / str(
                workload["evaluation_trace_template"]
            )).resolve())
            if workload.get("evaluation_trace_template") and
            not Path(str(workload["evaluation_trace_template"])).is_absolute()
            else workload.get("evaluation_trace_template")
        ),
    )
    max_replicas = execution["max_total_replicas"]
    if max_replicas is not None:
        max_replicas = int(max_replicas)
    if any(not isinstance(value, bool)
           for value in capabilities.values()):
        raise ValidationError("capability values must be booleans")
    runtime_capabilities = RuntimeCapabilities(**capabilities)
    return {
        "path": path,
        "config": config,
        "cases": cases,
        "workload": workload_spec,
        "experts": experts,
        "kernels": kernels,
        "max_replicas": max_replicas,
        "capabilities": runtime_capabilities,
        "execution": execution,
        "model": model,
    }


def _input_artifacts(resolved):
    artifacts = [resolved["path"]]
    workload = resolved["workload"]
    if workload.routing_model == "trace":
        for case in resolved["cases"]:
            artifacts.extend((
                workload.placement_trace_template.format(
                    case_id=case.case_id
                ),
                workload.evaluation_trace_template.format(
                    case_id=case.case_id
                ),
            ))
    return tuple(dict.fromkeys(artifacts))


def run_node_sweep_config(config_path, output_dir):
    """Load, validate, execute, and atomically save one fixed-window sweep."""

    resolved = _load_node_sweep_config(config_path)
    execution = resolved["execution"]
    result = run_node_count_comparisons(
        cases=resolved["cases"],
        workload=resolved["workload"],
        experts=resolved["experts"],
        kernel_models=resolved["kernels"],
        hidden_bytes=int(resolved["model"]["hidden_bytes"]),
        strategies=tuple(str(value)
                         for value in execution["strategies"]),
        max_total_replicas=resolved["max_replicas"],
        dispatcher_policy=str(execution["dispatcher_policy"]),
        capabilities=resolved["capabilities"],
        ingress_semantics=str(execution["ingress_semantics"]),
        chip_reduce_latency_s=float(
            execution["chip_reduce_latency_s"]
        ),
        node_reduce_latency_s=float(
            execution["node_reduce_latency_s"]
        ),
    )
    output = write_node_sweep_bundle(
        output_dir=output_dir,
        resolved_config=resolved["config"],
        sweep=result,
        seed=resolved["workload"].evaluation_seed,
        command="python -m flowgpu v2-node-sweep %s" % resolved["path"],
        input_artifacts=_input_artifacts(resolved),
    )
    return output, result


def run_node_serving_config(config_path, output_dir):
    """Execute the strict serving load grid declared in ``serving``."""

    resolved = _load_node_sweep_config(config_path)
    config = resolved["config"]
    if resolved["workload"].routing_model == "trace":
        raise ValidationError(
            "fixed router traces are supported by fixed-window/tiered "
            "sweeps; serving needs an online router model"
        )
    if "serving" not in config:
        raise ValidationError("config missing field: serving")
    serving = _mapping(config["serving"], "serving")
    _keys(serving, (
        "offered_rates_per_s", "seeds", "n_requests",
        "output_tokens", "max_batch", "slo_ttft_s", "slo_tpot_s",
        "target_attainment", "max_outstanding_requests",
        "compute_block_windows",
    ))
    output_tokens = serving["output_tokens"]
    if isinstance(output_tokens, bool):
        raise ValidationError(
            "serving.output_tokens must be a positive integer or [low, high]"
        )
    if isinstance(output_tokens, int):
        if output_tokens <= 0:
            raise ValidationError(
                "serving.output_tokens must be positive"
            )
    elif isinstance(output_tokens, list):
        if (len(output_tokens) != 2 or
                any(not isinstance(value, int) or isinstance(value, bool)
                    for value in output_tokens) or
                output_tokens[0] <= 0 or
                output_tokens[1] < output_tokens[0]):
            raise ValidationError(
                "serving.output_tokens range must be [positive low, high]"
            )
        output_tokens = tuple(output_tokens)
    else:
        raise ValidationError(
            "serving.output_tokens must be a positive integer or [low, high]"
        )
    windows = []
    for index, raw_window in enumerate(serving["compute_block_windows"]):
        window = _mapping(
            raw_window, "serving.compute_block_windows[%d]" % index
        )
        _keys(window, ("start_s", "finish_s"), ("source",))
        windows.append(ResourceBlockWindow(
            float(window["start_s"]),
            float(window["finish_s"]),
            str(window.get("source", "prefill")),
        ))
    admission_value = serving["max_outstanding_requests"]
    admission = (
        AdmissionPolicy(int(admission_value))
        if admission_value is not None else None
    )
    power_profile = None
    if "power" in config:
        power = _mapping(config["power"], "power")
        _keys(power, (
            "idle_power_w", "compute_dynamic_power_w", "provenance",
        ))
        power_profile = ChipPowerProfile(
            idle_power_w=float(power["idle_power_w"]),
            compute_dynamic_power_w=float(
                power["compute_dynamic_power_w"]
            ),
            provenance=_provenance(
                power["provenance"], "power.provenance"
            ),
        )
    execution = resolved["execution"]
    result = run_node_serving_sweep(
        cases=resolved["cases"],
        workload=resolved["workload"],
        experts=resolved["experts"],
        kernel_models=resolved["kernels"],
        hidden_bytes=int(resolved["model"]["hidden_bytes"]),
        offered_rates_per_s=serving["offered_rates_per_s"],
        seeds=serving["seeds"],
        n_requests=int(serving["n_requests"]),
        output_tokens=output_tokens,
        max_batch=int(serving["max_batch"]),
        slo_ttft_s=float(serving["slo_ttft_s"]),
        slo_tpot_s=float(serving["slo_tpot_s"]),
        target_attainment=float(serving["target_attainment"]),
        strategies=tuple(execution["strategies"]),
        max_total_replicas=resolved["max_replicas"],
        dispatcher_policy=str(execution["dispatcher_policy"]),
        capabilities=resolved["capabilities"],
        ingress_semantics=str(execution["ingress_semantics"]),
        admission_policy=admission,
        chip_power_profile=power_profile,
        compute_block_windows=tuple(windows),
    )
    output = write_serving_sweep_bundle(
        output_dir, config, result,
        "python -m flowgpu v2-node-serving %s" % resolved["path"],
        input_artifacts=_input_artifacts(resolved),
    )
    return output, result


def run_node_uncertainty_config(config_path, output_dir):
    """Execute the declared placement/evaluation seed grid."""

    resolved = _load_node_sweep_config(config_path)
    config = resolved["config"]
    if "uncertainty" not in config:
        raise ValidationError("config missing field: uncertainty")
    uncertainty = _mapping(config["uncertainty"], "uncertainty")
    _keys(uncertainty, ("placement_seeds", "evaluation_seeds"))
    if resolved["workload"].routing_model == "trace":
        raise ValidationError(
            "seed uncertainty is not meaningful for fixed router traces"
        )
    execution = resolved["execution"]
    result = run_node_count_uncertainty(
        cases=resolved["cases"],
        workload=resolved["workload"],
        placement_seeds=uncertainty["placement_seeds"],
        evaluation_seeds=uncertainty["evaluation_seeds"],
        experts=resolved["experts"],
        kernel_models=resolved["kernels"],
        hidden_bytes=int(resolved["model"]["hidden_bytes"]),
        strategies=tuple(execution["strategies"]),
        max_total_replicas=resolved["max_replicas"],
        dispatcher_policy=str(execution["dispatcher_policy"]),
        capabilities=resolved["capabilities"],
        ingress_semantics=str(execution["ingress_semantics"]),
    )
    output = write_uncertainty_bundle(
        output_dir, config, result,
        "python -m flowgpu v2-node-uncertainty %s" %
        resolved["path"],
        input_artifacts=_input_artifacts(resolved),
    )
    return output, result


def run_node_tiered_config(config_path, output_dir):
    """Execute cold-coverage versus hot-replication GPU/brain tiering."""

    resolved = _load_node_sweep_config(config_path)
    config = resolved["config"]
    if "tiered" not in config:
        raise ValidationError("config missing field: tiered")
    tiered = _mapping(config["tiered"], "tiered")
    _keys(tiered, (
        "max_brain_replicas", "gpu_capacity_bytes",
        "gpu_provenance", "gpu_kernel",
    ))
    gpu_kernel = _mapping(tiered["gpu_kernel"], "tiered.gpu_kernel")
    _keys(gpu_kernel, (
        "expert_shape", "dtype", "concurrent_experts", "residency",
        "samples", "provenance",
    ))
    gpu_curve = ExpertKernelCurve(
        expert_shape=str(gpu_kernel["expert_shape"]),
        dtype=str(gpu_kernel["dtype"]),
        concurrent_experts=int(gpu_kernel["concurrent_experts"]),
        residency=str(gpu_kernel["residency"]),
        samples=tuple(
            _kernel_sample(item, index)
            for index, item in enumerate(gpu_kernel["samples"])
        ),
        provenance=_provenance(
            gpu_kernel["provenance"], "tiered.gpu_kernel.provenance"
        ),
    )
    max_brain_replicas = tiered["max_brain_replicas"]
    if max_brain_replicas is not None:
        max_brain_replicas = int(max_brain_replicas)
    gpu_capacity_bytes = tiered["gpu_capacity_bytes"]
    if gpu_capacity_bytes is not None:
        gpu_capacity_bytes = int(gpu_capacity_bytes)
    execution = resolved["execution"]
    result = run_tiered_node_count_comparisons(
        cases=resolved["cases"],
        workload=resolved["workload"],
        experts=resolved["experts"],
        brain_kernel_models=resolved["kernels"],
        gpu_kernel_models={
            expert.key: gpu_curve for expert in resolved["experts"]
        },
        gpu_provenance=_provenance(
            tiered["gpu_provenance"], "tiered.gpu_provenance"
        ),
        hidden_bytes=int(resolved["model"]["hidden_bytes"]),
        max_brain_replicas=max_brain_replicas,
        gpu_capacity_bytes=gpu_capacity_bytes,
        dispatcher_policy=str(execution["dispatcher_policy"]),
        ingress_semantics=str(execution["ingress_semantics"]),
    )
    output = write_tiered_sweep_bundle(
        output_dir, config, result,
        "python -m flowgpu v2-node-tiered %s" % resolved["path"],
        input_artifacts=_input_artifacts(resolved),
    )
    return output, result
