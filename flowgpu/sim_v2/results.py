"""Atomic, traceable result bundles for v2 fixed-window layer runs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from .layer_engine import MoELayerResult
from .replicas import ReplicaPlan
from .routing import RoutingSummary
from .runner import (FixedWindowComparison, NodeSweepComparisonResult,
                     ServingNodeSweepResult, TieredNodeSweepResult,
                     UncertainNodeSweepResult)
from .experiments import NodeSweepCase, SweepTracePair
from .schema import ValidationError
from .validation import validate_layer_result


RESULT_SCHEMA_VERSION = 1


def _json_default(value):
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dict__"):
        return value.__dict__
    raise TypeError("cannot serialize %r" % (value,))


def _write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value, handle, ensure_ascii=False, indent=2,
            sort_keys=True, default=_json_default,
        )
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_hashes(input_artifacts: Iterable) -> dict:
    hashes = {}
    for artifact in input_artifacts:
        path = Path(artifact).resolve()
        if not path.is_file():
            raise ValidationError(
                "input artifact is not a file: %s" % path
            )
        hashes[str(path)] = _sha256(path)
    return hashes


def _expert_key(key) -> dict:
    return {
        "layer_id": key.layer_id,
        "expert_id": key.expert_id,
        "shared": key.shared,
    }


def write_layer_result_bundle(
    output_dir,
    resolved_config: Mapping,
    replica_plan: ReplicaPlan,
    routing_summary: RoutingSummary,
    result: MoELayerResult,
    seed: int,
    command: str,
    input_artifacts: Iterable = (),
    simulator_version: str = "0.2.0-v2-foundation",
) -> Path:
    """Write the v2 result contract to a new directory atomically.

    ``config.resolved.yaml`` contains JSON syntax, which is valid YAML 1.2,
    avoiding a hidden serializer dependency in the simulator core.
    Existing result directories are never overwritten.
    """

    validate_layer_result(result)
    if seed < 0:
        raise ValidationError("result seed must be non-negative")
    if not command:
        raise ValidationError("result command must be non-empty")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise ValidationError(
            "result directory already exists: %s" % destination
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    artifact_hashes = {}
    for artifact in input_artifacts:
        path = Path(artifact).resolve()
        if not path.is_file():
            raise ValidationError(
                "input artifact is not a file: %s" % path
            )
        artifact_hashes[str(path)] = _sha256(path)

    temp_dir = Path(tempfile.mkdtemp(
        prefix=destination.name + ".tmp-",
        dir=str(destination.parent),
    ))
    try:
        # JSON is a strict subset of YAML and remains easy to diff.
        _write_json(temp_dir / "config.resolved.yaml", dict(resolved_config))

        confidences = Counter()
        for expert in replica_plan.experts:
            confidences[expert.provenance.confidence.value] += 1
        for chip in replica_plan.chips:
            if chip.provenance is not None:
                confidences[chip.provenance.confidence.value] += 1
        manifest = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "simulator_version": simulator_version,
            "command": command,
            "seed": seed,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "host": {
                "platform": platform.platform(),
                "python": sys.version,
            },
            "input_artifact_sha256": artifact_hashes,
            "decisive_provenance_counts": dict(sorted(confidences.items())),
            "placement_strategy": replica_plan.strategy,
            "execution_mode": "detailed:max-min-fluid",
        }
        _write_json(temp_dir / "manifest.json", manifest)

        summary = {
            "layer_id": result.layer_id,
            "completion_s": result.completion_s,
            "assignments": result.assignment_count,
            "resident_replicas": len(replica_plan.replicas),
            "active_replicas": len(result.replica_executions),
            "active_chips": len({
                execution.chip_id
                for execution in result.replica_executions
            }),
            "network_energy_j": result.network_energy_j,
            "network_contention_s": result.network_contention_s,
            "local_assignment_fraction": result.local_assignment_fraction,
            "dispatch": {
                "flows": result.dispatch_flow_count,
                "logical_bytes": result.logical_dispatch_bytes,
                "physical_flow_payload_bytes":
                    result.physical_dispatch_bytes,
            },
            "combine": {
                "flows": result.combine_flow_count,
                "logical_bytes": result.logical_combine_bytes,
                "physical_flow_payload_bytes":
                    result.physical_combine_bytes,
            },
        }
        _write_json(temp_dir / "summary.json", summary)

        placement = {
            "strategy": replica_plan.strategy,
            "replicas": [
                {
                    "replica_id": replica.replica_id,
                    "expert": _expert_key(replica.expert),
                    "chip_id": replica.chip_id,
                    "node_id": replica_plan.chip(
                        replica.chip_id
                    ).node_id,
                    "is_primary": replica.is_primary,
                    "weight_bytes": next(
                        expert.weight_bytes
                        for expert in replica_plan.experts
                        if expert.key == replica.expert
                    ),
                }
                for replica in replica_plan.replicas
            ],
            "capacity": [
                {
                    "chip_id": usage.chip_id,
                    "weight_bytes": usage.weight_bytes,
                    "usable_weight_bytes": usage.usable_weight_bytes,
                    "utilization": usage.utilization,
                }
                for usage in replica_plan.capacity_usage.values()
            ],
        }
        _write_json(temp_dir / "placement.json", placement)
        _write_json(
            temp_dir / "routing_summary.json",
            routing_summary.__dict__,
        )

        execution_by_chip = defaultdict(list)
        for execution in result.replica_executions:
            execution_by_chip[execution.chip_id].append(execution)
        with (temp_dir / "chip_metrics.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            fields = [
                "chip_id", "node_id", "resident_replicas",
                "active_replicas", "assignments", "weight_bytes",
                "usable_weight_bytes", "compute_queue_s", "compute_span_s",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for chip in sorted(
                    replica_plan.chips, key=lambda item: item.chip_id):
                executions = execution_by_chip[chip.chip_id]
                resident = [
                    replica for replica in replica_plan.replicas
                    if replica.chip_id == chip.chip_id
                ]
                usage = replica_plan.capacity_usage[chip.chip_id]
                writer.writerow({
                    "chip_id": chip.chip_id,
                    "node_id": chip.node_id,
                    "resident_replicas": len(resident),
                    "active_replicas": len(executions),
                    "assignments": sum(
                        item.assignment_count for item in executions
                    ),
                    "weight_bytes": usage.weight_bytes,
                    "usable_weight_bytes": usage.usable_weight_bytes,
                    "compute_queue_s": sum(
                        item.compute_queue_s for item in executions
                    ),
                    "compute_span_s": sum(
                        item.compute_finish_s - item.compute_start_s
                        for item in executions
                    ),
                })

        link_flow_count = Counter()
        link_energy = Counter()
        link_contention = Counter()
        for flow in result.flow_results.values():
            for link in flow.route.links:
                link_flow_count[link.link_id] += 1
                link_energy[link.link_id] += (
                    flow.flow.nbytes * link.energy_per_byte
                )
                # Contention is a path-level quantity.  Split evenly across
                # the path only for attribution; raw flow values remain in
                # summary/critical-path artifacts.
                if flow.route.links:
                    link_contention[link.link_id] += (
                        flow.contention_delay_s / len(flow.route.links)
                    )
        with (temp_dir / "link_metrics.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            fields = [
                "link_id", "bytes", "flows", "energy_j",
                "attributed_contention_s",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for link_id in sorted(result.link_bytes):
                writer.writerow({
                    "link_id": link_id,
                    "bytes": result.link_bytes[link_id],
                    "flows": link_flow_count[link_id],
                    "energy_j": link_energy[link_id],
                    "attributed_contention_s":
                        link_contention[link_id],
                })

        _write_json(
            temp_dir / "critical_paths.json",
            {"layer": result.critical_path.__dict__},
        )
        _write_json(
            temp_dir / "warnings.json",
            {"warnings": list(result.warnings)},
        )
        os.replace(str(temp_dir), str(destination))
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(str(temp_dir))
        raise
    return destination


def write_comparison_result_bundle(
    output_dir,
    resolved_config: Mapping,
    case: NodeSweepCase,
    traces: SweepTracePair,
    comparison: FixedWindowComparison,
    seed: int,
    command: str,
    input_artifacts: Iterable = (),
    simulator_version: str = "0.2.0-v2-foundation",
) -> Path:
    """Atomically persist one held-out fixed-window strategy comparison."""

    if seed < 0:
        raise ValidationError("result seed must be non-negative")
    if not command:
        raise ValidationError("result command must be non-empty")
    if not (
        case.case_id == traces.case_id == comparison.case_id
    ):
        raise ValidationError(
            "comparison bundle case identifiers do not match"
        )
    if traces.placement.split != "placement":
        raise ValidationError("bundle placement trace has wrong split")
    if traces.evaluation.split != "evaluation":
        raise ValidationError("bundle evaluation trace has wrong split")
    if comparison.evaluation_split != "evaluation":
        raise ValidationError("comparison did not use held-out evaluation")
    if not comparison.strategies:
        raise ValidationError("comparison bundle has no strategies")
    for strategy in comparison.strategies:
        for result in strategy.layer_results:
            validate_layer_result(result)

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise ValidationError(
            "result directory already exists: %s" % destination
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact_hashes = {}
    for artifact in input_artifacts:
        path = Path(artifact).resolve()
        if not path.is_file():
            raise ValidationError(
                "input artifact is not a file: %s" % path
            )
        artifact_hashes[str(path)] = _sha256(path)

    confidences = Counter()
    for strategy in comparison.strategies:
        for expert in strategy.replica_plan.experts:
            confidences[expert.provenance.confidence.value] += 1
        break
    for chip in case.chips:
        if chip.provenance is not None:
            confidences[chip.provenance.confidence.value] += 1
    for link in case.cluster.topology.links.values():
        confidences[link.provenance.confidence.value] += 1
    confidences[traces.placement.provenance.confidence.value] += 1
    confidences[traces.evaluation.provenance.confidence.value] += 1

    temp_dir = Path(tempfile.mkdtemp(
        prefix=destination.name + ".tmp-",
        dir=str(destination.parent),
    ))
    try:
        _write_json(temp_dir / "config.resolved.yaml", dict(resolved_config))
        _write_json(temp_dir / "manifest.json", {
            "schema_version": RESULT_SCHEMA_VERSION,
            "bundle_kind": "fixed-window-comparison",
            "simulator_version": simulator_version,
            "command": command,
            "seed": seed,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "host": {
                "platform": platform.platform(),
                "python": sys.version,
            },
            "case_id": case.case_id,
            "input_artifact_sha256": artifact_hashes,
            "decisive_provenance_counts":
                dict(sorted(confidences.items())),
            "execution_mode": "detailed:max-min-fluid",
        })
        _write_json(temp_dir / "case.json", case.fairness_metadata())
        _write_json(temp_dir / "traces.json", {
            "scaling_mode": traces.scaling_mode,
            "placement_token_count": traces.placement_token_count,
            "evaluation_token_count": traces.evaluation_token_count,
            "placement": traces.placement.summary().__dict__,
            "evaluation": traces.evaluation.summary().__dict__,
            "placement_provenance":
                traces.placement.provenance.__dict__,
            "evaluation_provenance":
                traces.evaluation.provenance.__dict__,
        })

        baseline = comparison.result_for("unique-balanced")
        with (temp_dir / "strategies.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            fields = [
                "strategy", "completion_s", "speedup_vs_unique",
                "assignments", "resident_replicas",
                "local_assignment_fraction", "network_energy_j",
                "link_bytes", "max_sram_utilization",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for strategy in comparison.strategies:
                writer.writerow({
                    "strategy": strategy.strategy,
                    "completion_s": strategy.completion_s,
                    "speedup_vs_unique":
                        baseline.completion_s / strategy.completion_s,
                    "assignments": strategy.assignment_count,
                    "resident_replicas":
                        len(strategy.replica_plan.replicas),
                    "local_assignment_fraction":
                        strategy.local_assignment_fraction,
                    "network_energy_j": strategy.network_energy_j,
                    "link_bytes": sum(
                        sum(result.link_bytes.values())
                        for result in strategy.layer_results
                    ),
                    "max_sram_utilization": max(
                        usage.utilization
                        for usage in strategy.replica_plan.
                        capacity_usage.values()
                    ),
                })

        with (temp_dir / "layers.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            fields = [
                "strategy", "layer_id", "completion_s", "assignments",
                "active_replicas", "local_assignment_fraction",
                "network_energy_j", "network_contention_s",
                "dispatch_bytes", "combine_bytes",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for strategy in comparison.strategies:
                for result in strategy.layer_results:
                    writer.writerow({
                        "strategy": strategy.strategy,
                        "layer_id": result.layer_id,
                        "completion_s": result.completion_s,
                        "assignments": result.assignment_count,
                        "active_replicas":
                            len(result.replica_executions),
                        "local_assignment_fraction":
                            result.local_assignment_fraction,
                        "network_energy_j": result.network_energy_j,
                        "network_contention_s":
                            result.network_contention_s,
                        "dispatch_bytes": result.physical_dispatch_bytes,
                        "combine_bytes": result.physical_combine_bytes,
                    })

        placements = {}
        for strategy in comparison.strategies:
            plan = strategy.replica_plan
            placements[strategy.strategy] = {
                "replicas": [
                    {
                        "replica_id": replica.replica_id,
                        "layer_id": replica.expert.layer_id,
                        "expert_id": replica.expert.expert_id,
                        "shared": replica.expert.shared,
                        "chip_id": replica.chip_id,
                        "node_id": plan.chip(replica.chip_id).node_id,
                        "is_primary": replica.is_primary,
                    }
                    for replica in plan.replicas
                ],
                "capacity": [
                    usage.__dict__
                    for usage in plan.capacity_usage.values()
                ],
            }
        _write_json(temp_dir / "placements.json", placements)
        _write_json(temp_dir / "warnings.json", {
            strategy.strategy: sorted({
                warning
                for result in strategy.layer_results
                for warning in result.warnings
            })
            for strategy in comparison.strategies
        })
        os.replace(str(temp_dir), str(destination))
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(str(temp_dir))
        raise
    return destination


def write_node_sweep_bundle(
    output_dir,
    resolved_config: Mapping,
    sweep: NodeSweepComparisonResult,
    seed: int,
    command: str,
    input_artifacts: Iterable = (),
    simulator_version: str = "0.2.0-v2-foundation",
) -> Path:
    """Atomically save an across-node-count metric table and manifest."""

    if seed < 0:
        raise ValidationError("result seed must be non-negative")
    if not command:
        raise ValidationError("result command must be non-empty")
    if not sweep.comparisons or not sweep.points:
        raise ValidationError("node sweep result is empty")
    case_ids = {comparison.case_id for comparison in sweep.comparisons}
    if {point.case_id for point in sweep.points} != case_ids:
        raise ValidationError("node sweep points/comparisons mismatch")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise ValidationError(
            "result directory already exists: %s" % destination
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact_hashes = {}
    for artifact in input_artifacts:
        path = Path(artifact).resolve()
        if not path.is_file():
            raise ValidationError(
                "input artifact is not a file: %s" % path
            )
        artifact_hashes[str(path)] = _sha256(path)

    temp_dir = Path(tempfile.mkdtemp(
        prefix=destination.name + ".tmp-",
        dir=str(destination.parent),
    ))
    try:
        _write_json(temp_dir / "config.resolved.yaml", dict(resolved_config))
        _write_json(temp_dir / "manifest.json", {
            "schema_version": RESULT_SCHEMA_VERSION,
            "bundle_kind": "node-count-sweep",
            "simulator_version": simulator_version,
            "command": command,
            "seed": seed,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "host": {
                "platform": platform.platform(),
                "python": sys.version,
            },
            "scaling_mode": sweep.scaling_mode,
            "replica_budget_mode": sweep.replica_budget_mode,
            "fixed_max_total_replicas":
                sweep.fixed_max_total_replicas,
            "input_artifact_sha256": artifact_hashes,
            "execution_mode": "detailed:max-min-fluid",
        })
        with (temp_dir / "sweep.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            fields = [
                "case_id", "node_count", "ingress_mode", "strategy",
                "capacity_ratio", "resident_replicas", "completion_s",
                "tokens_per_s", "tokens_per_s_per_node",
                "speedup_vs_unique", "local_assignment_fraction",
                "network_energy_j",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for point in sweep.points:
                writer.writerow({
                    field: getattr(point, field) for field in fields
                })
        _write_json(temp_dir / "cases.json", [
            {
                "case_id": comparison.case_id,
                "ingress_semantics": comparison.ingress_semantics,
                "dispatcher_policy": comparison.dispatcher_policy,
                "evaluation_split": comparison.evaluation_split,
                "strategies": [
                    result.strategy for result in comparison.strategies
                ],
            }
            for comparison in sweep.comparisons
        ])
        os.replace(str(temp_dir), str(destination))
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(str(temp_dir))
        raise
    return destination


def _new_bundle_dir(output_dir):
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise ValidationError(
            "result directory already exists: %s" % destination
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(
        prefix=destination.name + ".tmp-",
        dir=str(destination.parent),
    ))
    return destination, temp_dir


def write_serving_sweep_bundle(
    output_dir,
    resolved_config: Mapping,
    sweep: ServingNodeSweepResult,
    command: str,
    input_artifacts: Iterable = (),
    simulator_version: str = "0.2.0-v2-foundation",
) -> Path:
    """Persist capacity summaries and every offered-load confidence interval."""

    if not command or not sweep.points:
        raise ValidationError("serving sweep bundle is empty or unnamed")
    destination, temp_dir = _new_bundle_dir(output_dir)
    try:
        _write_json(temp_dir / "config.resolved.yaml", dict(resolved_config))
        _write_json(temp_dir / "manifest.json", {
            "schema_version": RESULT_SCHEMA_VERSION,
            "bundle_kind": "serving-node-sweep",
            "simulator_version": simulator_version,
            "command": command,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "offered_rates_per_s": list(sweep.offered_rates_per_s),
            "seeds": list(sweep.seeds),
            "input_artifact_sha256":
                _artifact_hashes(input_artifacts),
            "execution_mode": "detailed:max-min-fluid",
        })
        with (temp_dir / "capacity.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            fields = [
                "case_id", "node_count", "ingress_mode", "strategy",
                "capacity_rate_per_s", "capacity_per_node_per_s",
                "target_attainment", "right_censored",
                "monotonic_attainment", "warning",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for point in sweep.points:
                writer.writerow({
                    field: getattr(point, field) for field in fields
                })
        with (temp_dir / "load_points.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            fields = [
                "case_id", "strategy", "offered_rate_per_s",
                "lower_bound_meets_target",
                "throughput_mean", "throughput_low95",
                "throughput_high95", "ttft_mean_s", "ttft_low95_s",
                "ttft_high95_s", "tpot_mean_s", "tpot_low95_s",
                "tpot_high95_s", "slo_attainment_mean",
                "slo_attainment_low95", "slo_attainment_high95",
                "slo_goodput_mean", "slo_goodput_low95",
                "slo_goodput_high95", "admission_mean",
                "admission_low95", "admission_high95",
                "network_energy_per_token_mean_j",
                "system_energy_per_token_mean_j",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for sweep_point in sweep.points:
                for point in sweep_point.capacity_result.points:
                    aggregate = point.aggregate
                    writer.writerow({
                        "case_id": sweep_point.case_id,
                        "strategy": sweep_point.strategy,
                        "offered_rate_per_s": point.offered_rate_per_s,
                        "lower_bound_meets_target":
                            point.lower_bound_meets_target,
                        "throughput_mean": aggregate.throughput.mean,
                        "throughput_low95": aggregate.throughput.low95,
                        "throughput_high95": aggregate.throughput.high95,
                        "ttft_mean_s": aggregate.mean_ttft_s.mean,
                        "ttft_low95_s": aggregate.mean_ttft_s.low95,
                        "ttft_high95_s": aggregate.mean_ttft_s.high95,
                        "tpot_mean_s": aggregate.mean_tpot_s.mean,
                        "tpot_low95_s": aggregate.mean_tpot_s.low95,
                        "tpot_high95_s": aggregate.mean_tpot_s.high95,
                        "slo_attainment_mean":
                            aggregate.slo_attainment.mean,
                        "slo_attainment_low95":
                            aggregate.slo_attainment.low95,
                        "slo_attainment_high95":
                            aggregate.slo_attainment.high95,
                        "slo_goodput_mean":
                            aggregate.slo_goodput_per_s.mean,
                        "slo_goodput_low95":
                            aggregate.slo_goodput_per_s.low95,
                        "slo_goodput_high95":
                            aggregate.slo_goodput_per_s.high95,
                        "admission_mean":
                            aggregate.admission_fraction.mean,
                        "admission_low95":
                            aggregate.admission_fraction.low95,
                        "admission_high95":
                            aggregate.admission_fraction.high95,
                        "network_energy_per_token_mean_j":
                            aggregate.network_energy_per_token_j.mean,
                        "system_energy_per_token_mean_j": (
                            aggregate.system_energy_per_token_j.mean
                            if aggregate.system_energy_per_token_j
                            is not None else ""
                        ),
                    })
        os.replace(str(temp_dir), str(destination))
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(str(temp_dir))
        raise
    return destination


def write_uncertainty_bundle(
    output_dir,
    resolved_config: Mapping,
    sweep: UncertainNodeSweepResult,
    command: str,
    input_artifacts: Iterable = (),
    simulator_version: str = "0.2.0-v2-foundation",
) -> Path:
    """Persist paired-seed confidence intervals and conclusion labels."""

    if not command or not sweep.points:
        raise ValidationError("uncertainty bundle is empty or unnamed")
    destination, temp_dir = _new_bundle_dir(output_dir)
    try:
        _write_json(temp_dir / "config.resolved.yaml", dict(resolved_config))
        _write_json(temp_dir / "manifest.json", {
            "schema_version": RESULT_SCHEMA_VERSION,
            "bundle_kind": "node-sweep-uncertainty",
            "simulator_version": simulator_version,
            "command": command,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "placement_seeds": list(sweep.placement_seeds),
            "evaluation_seeds": list(sweep.evaluation_seeds),
            "input_artifact_sha256":
                _artifact_hashes(input_artifacts),
            "execution_mode": "detailed:max-min-fluid",
        })
        with (temp_dir / "uncertainty.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            fields = [
                "case_id", "node_count", "ingress_mode", "strategy",
                "runs", "conclusion", "completion_mean_s",
                "completion_low95_s", "completion_high95_s",
                "speedup_mean", "speedup_low95", "speedup_high95",
                "tokens_per_s_mean", "tokens_per_s_low95",
                "tokens_per_s_high95", "locality_mean",
                "locality_low95", "locality_high95",
                "network_energy_mean_j", "network_energy_low95_j",
                "network_energy_high95_j",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for point in sweep.points:
                writer.writerow({
                    "case_id": point.case_id,
                    "node_count": point.node_count,
                    "ingress_mode": point.ingress_mode,
                    "strategy": point.strategy,
                    "runs": point.runs,
                    "conclusion": point.conclusion,
                    "completion_mean_s": point.completion_s.mean,
                    "completion_low95_s": point.completion_s.low95,
                    "completion_high95_s": point.completion_s.high95,
                    "speedup_mean": point.speedup_vs_unique.mean,
                    "speedup_low95": point.speedup_vs_unique.low95,
                    "speedup_high95": point.speedup_vs_unique.high95,
                    "tokens_per_s_mean": point.tokens_per_s.mean,
                    "tokens_per_s_low95": point.tokens_per_s.low95,
                    "tokens_per_s_high95": point.tokens_per_s.high95,
                    "locality_mean":
                        point.local_assignment_fraction.mean,
                    "locality_low95":
                        point.local_assignment_fraction.low95,
                    "locality_high95":
                        point.local_assignment_fraction.high95,
                    "network_energy_mean_j":
                        point.network_energy_j.mean,
                    "network_energy_low95_j":
                        point.network_energy_j.low95,
                    "network_energy_high95_j":
                        point.network_energy_j.high95,
                })
        os.replace(str(temp_dir), str(destination))
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(str(temp_dir))
        raise
    return destination


def write_tiered_sweep_bundle(
    output_dir,
    resolved_config: Mapping,
    sweep: TieredNodeSweepResult,
    command: str,
    input_artifacts: Iterable = (),
    simulator_version: str = "0.2.0-v2-foundation",
) -> Path:
    """Persist the cold-coverage versus hot-replication node sweep."""

    if not command or not sweep.points:
        raise ValidationError("tiered sweep bundle is empty or unnamed")
    destination, temp_dir = _new_bundle_dir(output_dir)
    try:
        _write_json(temp_dir / "config.resolved.yaml", dict(resolved_config))
        _write_json(temp_dir / "manifest.json", {
            "schema_version": RESULT_SCHEMA_VERSION,
            "bundle_kind": "tiered-node-sweep",
            "simulator_version": simulator_version,
            "command": command,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "max_brain_replicas": sweep.max_brain_replicas,
            "input_artifact_sha256":
                _artifact_hashes(input_artifacts),
            "execution_mode": "detailed:max-min-fluid",
        })
        with (temp_dir / "tiered_sweep.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            fields = [
                "case_id", "node_count", "ingress_mode", "policy",
                "brain_replicas", "brain_unique_experts",
                "gpu_fallback_experts", "completion_s", "tokens_per_s",
                "tokens_per_s_per_node", "speedup_vs_unique_first",
                "local_assignment_fraction", "network_energy_j",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for point in sweep.points:
                writer.writerow({
                    field: getattr(point, field) for field in fields
                })
        os.replace(str(temp_dir), str(destination))
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(str(temp_dir))
        raise
    return destination
