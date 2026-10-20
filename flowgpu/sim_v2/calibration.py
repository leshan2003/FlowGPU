"""Validated benchmark artifacts and calibration builders for v2."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Tuple

import yaml

from .kernel_model import ExpertKernelCurve, KernelSample
from .schema import Confidence, Provenance, ValidationError
from .topology import LinkProfile


CALIBRATION_SCHEMA_VERSION = 1
RECOMMENDED_KERNEL_M = (1, 2, 4, 8, 16, 32, 64)


@dataclass(frozen=True)
class BenchmarkArtifact:
    path: str
    sha256: str
    machine: Mapping
    measurement: Mapping
    records: Tuple[Mapping, ...]
    provenance: Provenance


@dataclass(frozen=True, order=True)
class KernelCurveKey:
    expert_shape: str
    dtype: str
    concurrent_experts: int
    residency: str


@dataclass(frozen=True)
class KernelCalibrationSet:
    curves: Mapping[KernelCurveKey, ExpertKernelCurve]
    warnings: Tuple[str, ...]
    artifact: BenchmarkArtifact


@dataclass(frozen=True, order=True)
class NetworkFitKey:
    kind: str
    src_class: str
    dst_class: str
    concurrency: int


@dataclass(frozen=True)
class CalibratedLinkFit:
    key: NetworkFitKey
    startup_latency_s: float
    effective_bandwidth_bytes_per_s: float
    min_message_bytes: int
    max_message_bytes: int
    sample_count: int
    median_relative_error: float
    provenance: Provenance

    def as_link_profile(
        self,
        energy_per_byte: float,
        duplex: str = "full",
    ) -> LinkProfile:
        return LinkProfile(
            bandwidth_bytes_per_s=self.effective_bandwidth_bytes_per_s,
            latency_s=self.startup_latency_s,
            efficiency=1.0,
            energy_per_byte=energy_per_byte,
            provenance=self.provenance,
            duplex=duplex,
        )


@dataclass(frozen=True)
class NetworkCalibrationSet:
    fits: Mapping[NetworkFitKey, CalibratedLinkFit]
    warnings: Tuple[str, ...]
    artifact: BenchmarkArtifact


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required(mapping, fields, context):
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValidationError(
            "%s missing required fields: %s" %
            (context, ", ".join(missing))
        )


def load_benchmark_artifact(path) -> BenchmarkArtifact:
    artifact_path = Path(path).resolve()
    if not artifact_path.is_file():
        raise ValidationError(
            "benchmark artifact is not a file: %s" % artifact_path
        )
    try:
        with artifact_path.open("r", encoding="utf-8") as handle:
            if artifact_path.suffix.lower() == ".json":
                data = json.load(handle)
            else:
                data = yaml.safe_load(handle)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ValidationError(
            "cannot parse benchmark artifact: %s" % artifact_path
        ) from exc
    if not isinstance(data, dict):
        raise ValidationError("benchmark artifact root must be a mapping")
    _required(
        data, ("schema_version", "machine", "measurement", "records"),
        "benchmark artifact",
    )
    if data["schema_version"] != CALIBRATION_SCHEMA_VERSION:
        raise ValidationError(
            "unsupported calibration schema_version: %r" %
            data["schema_version"]
        )
    machine = data["machine"]
    measurement = data["measurement"]
    records = data["records"]
    if not isinstance(machine, dict) or not isinstance(measurement, dict):
        raise ValidationError(
            "machine and measurement must be mappings"
        )
    _required(
        machine, ("model", "revision", "firmware", "runtime"), "machine"
    )
    _required(
        measurement,
        ("timestamp", "repetitions", "environment"),
        "measurement",
    )
    for field in ("model", "revision", "firmware", "runtime"):
        if machine[field] is None or not str(machine[field]).strip():
            raise ValidationError(
                "machine %s must be non-empty" % field
            )
    for field in ("timestamp", "environment"):
        if (measurement[field] is None or
                not str(measurement[field]).strip()):
            raise ValidationError(
                "measurement %s must be non-empty" % field
            )
    if (not isinstance(measurement["repetitions"], int) or
            measurement["repetitions"] <= 0):
        raise ValidationError(
            "measurement repetitions must be a positive integer"
        )
    if not isinstance(records, list) or not records:
        raise ValidationError("benchmark records must be a non-empty list")
    if any(not isinstance(record, dict) for record in records):
        raise ValidationError("every benchmark record must be a mapping")
    provenance = Provenance(
        confidence=Confidence.MEASURED,
        source="benchmark:%s:%s" %
        (machine["model"], measurement["timestamp"]),
        artifact=str(artifact_path),
        measured_at=str(measurement["timestamp"]),
        machine_revision=str(machine["revision"]),
    )
    return BenchmarkArtifact(
        path=str(artifact_path),
        sha256=_file_sha256(artifact_path),
        machine=MappingProxyType(dict(machine)),
        measurement=MappingProxyType(dict(measurement)),
        records=tuple(MappingProxyType(dict(record)) for record in records),
        provenance=provenance,
    )


def build_kernel_calibration(
    artifact: BenchmarkArtifact,
) -> KernelCalibrationSet:
    grouped = {}
    for index, record in enumerate(artifact.records):
        if record.get("kind") != "expert_latency":
            continue
        _required(
            record,
            (
                "expert_shape", "dtype", "m", "concurrent_experts",
                "residency", "latency_s",
            ),
            "expert_latency record %d" % index,
        )
        key = KernelCurveKey(
            str(record["expert_shape"]),
            str(record["dtype"]),
            int(record["concurrent_experts"]),
            str(record["residency"]),
        )
        sample = KernelSample(
            m=int(record["m"]),
            latency_s=float(record["latency_s"]),
            p95_s=(
                float(record["p95_s"])
                if record.get("p95_s") is not None else None
            ),
            repetitions=(
                int(record["repetitions"])
                if record.get("repetitions") is not None
                else artifact.measurement["repetitions"]
            ),
        )
        grouped.setdefault(key, []).append(sample)
    if not grouped:
        raise ValidationError(
            "artifact contains no expert_latency records"
        )
    curves = {}
    warnings = []
    for key in sorted(grouped):
        samples = grouped[key]
        curves[key] = ExpertKernelCurve(
            expert_shape=key.expert_shape,
            dtype=key.dtype,
            concurrent_experts=key.concurrent_experts,
            residency=key.residency,
            samples=samples,
            provenance=artifact.provenance,
        )
        measured_m = {sample.m for sample in samples}
        missing = [
            value for value in RECOMMENDED_KERNEL_M
            if value not in measured_m
        ]
        if missing:
            warnings.append(
                "%r missing recommended M points: %s" %
                (key, ",".join(str(value) for value in missing))
            )
    return KernelCalibrationSet(
        curves=MappingProxyType(curves),
        warnings=tuple(warnings),
        artifact=artifact,
    )


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _fit_affine(samples):
    count = len(samples)
    mean_x = sum(x for x, _ in samples) / count
    mean_y = sum(y for _, y in samples) / count
    denominator = sum((x - mean_x) ** 2 for x, _ in samples)
    if denominator <= 0:
        raise ValidationError(
            "network fit needs at least two distinct message sizes"
        )
    slope = sum(
        (x - mean_x) * (y - mean_y) for x, y in samples
    ) / denominator
    intercept = mean_y - slope * mean_x
    if slope <= 0 or not math.isfinite(slope):
        raise ValidationError("network fit produced non-positive slope")
    intercept = max(0.0, intercept)
    bandwidth = 1.0 / slope
    errors = []
    for nbytes, latency in samples:
        predicted = intercept + nbytes / bandwidth
        errors.append(abs(predicted - latency) / latency)
    return intercept, bandwidth, _median(errors)


def build_network_calibration(
    artifact: BenchmarkArtifact,
) -> NetworkCalibrationSet:
    supported = {"p2p", "bridge", "multicast", "reduce"}
    grouped = {}
    for index, record in enumerate(artifact.records):
        kind = record.get("kind")
        if kind not in supported:
            continue
        _required(
            record,
            (
                "src_class", "dst_class", "bytes",
                "concurrency", "latency_s",
            ),
            "%s record %d" % (kind, index),
        )
        nbytes = int(record["bytes"])
        latency = float(record["latency_s"])
        concurrency = int(record["concurrency"])
        if nbytes <= 0 or latency <= 0 or concurrency <= 0:
            raise ValidationError(
                "network bytes, latency, and concurrency must be positive"
            )
        key = NetworkFitKey(
            str(kind), str(record["src_class"]),
            str(record["dst_class"]), concurrency,
        )
        grouped.setdefault(key, []).append((nbytes, latency))
    if not grouped:
        raise ValidationError(
            "artifact contains no supported network records"
        )
    fits = {}
    warnings = []
    derived_provenance = Provenance(
        Confidence.DERIVED,
        "affine fit from %s" % artifact.provenance.source,
        artifact=artifact.path,
        measured_at=artifact.provenance.measured_at,
        machine_revision=artifact.provenance.machine_revision,
    )
    for key in sorted(grouped):
        samples = grouped[key]
        startup, bandwidth, median_error = _fit_affine(samples)
        fits[key] = CalibratedLinkFit(
            key=key,
            startup_latency_s=startup,
            effective_bandwidth_bytes_per_s=bandwidth,
            min_message_bytes=min(item[0] for item in samples),
            max_message_bytes=max(item[0] for item in samples),
            sample_count=len(samples),
            median_relative_error=median_error,
            provenance=derived_provenance,
        )
        if median_error > 0.10:
            warnings.append(
                "%r median relative fit error %.1f%% exceeds 10%%" %
                (key, median_error * 100)
            )
    return NetworkCalibrationSet(
        fits=MappingProxyType(fits),
        warnings=tuple(warnings),
        artifact=artifact,
    )
