"""Adapters from the hardware catalogue to v2 experiment profiles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

from flowgpu.hardware.interconnect import FABRICS
from flowgpu.hardware.registry import get_spec
from flowgpu.units import parse

from .schema import Confidence, Provenance, ValidationError


@dataclass(frozen=True)
class HardwareProfile:
    device: str
    kind: str
    role: str
    weight_capacity_bytes: int
    dram_capacity_bytes: int
    dram_bandwidth_bytes_per_s: float
    dram_latency_s: float
    inter_chip_bandwidth_bytes_per_s: float
    inter_chip_latency_s: float
    inter_board_bandwidth_bytes_per_s: float
    inter_board_latency_s: float
    tdp_w: float
    supported_dtypes: tuple
    provenance: Provenance


@dataclass(frozen=True)
class FabricProfile:
    name: str
    kind: str
    bandwidth_bytes_per_s: float
    latency_s: float
    efficiency: float
    energy_per_byte: float
    provenance: Provenance


def _confidence(value: str) -> Confidence:
    value = str(value).lower()
    if value == "estimate":
        return Confidence.PROJECTED
    try:
        return Confidence(value)
    except ValueError:
        return Confidence.PROJECTED


def load_hardware_profile(name: str) -> HardwareProfile:
    kind, spec = get_spec(name)
    memory = dict(spec.get("memory", {}))
    if kind == "gpu":
        if "hbm" not in memory:
            raise ValidationError("GPU profile has no HBM tier")
        weight_memory = memory["hbm"]
        role = "gpu"
        links = spec.get("links", {})
        scaleup = links.get("scaleup", {})
        scaleout = links.get("scaleout", {})
        inter_chip_bw = parse(scaleup.get("bandwidth", 0))
        inter_chip_latency = parse(scaleup.get("latency", 0))
        inter_board_bw = parse(scaleout.get("bandwidth", 0))
        inter_board_latency = parse(scaleout.get("latency", 0))
    else:
        if "sram" not in memory:
            raise ValidationError("brain profile has no SRAM tier")
        weight_memory = memory["sram"]
        role = "brain_sram"
        inter_chip_bw = parse(spec.get("inter_chip_bw", 0))
        inter_chip_latency = parse(spec.get("inter_chip_latency", 0))
        inter_board_bw = parse(spec.get("inter_board_bw", 0))
        inter_board_latency = parse(spec.get("inter_board_latency", 0))
    dram = memory.get("dram", {})
    weight_frac = float(
        spec.get("extra", {}).get(
            "sram_weight_frac", 0.85 if kind != "gpu" else 1.0
        )
    )
    if not 0 < weight_frac <= 1:
        raise ValidationError("device weight capacity fraction must be in (0, 1]")
    capacity = int(math.floor(parse(weight_memory["capacity"]) * weight_frac))
    return HardwareProfile(
        device=name,
        kind=kind,
        role=role,
        weight_capacity_bytes=capacity,
        dram_capacity_bytes=int(parse(dram.get("capacity", 0))),
        dram_bandwidth_bytes_per_s=parse(dram.get("bandwidth", 0)),
        dram_latency_s=parse(dram.get("latency", 0)),
        inter_chip_bandwidth_bytes_per_s=inter_chip_bw,
        inter_chip_latency_s=inter_chip_latency,
        inter_board_bandwidth_bytes_per_s=inter_board_bw,
        inter_board_latency_s=inter_board_latency,
        tdp_w=parse(spec.get("tdp", 0)),
        supported_dtypes=tuple(spec.get("supported_dtypes", ())),
        provenance=Provenance(
            _confidence(spec.get("confidence", "projected")),
            str(spec.get("source", "unattributed device profile")),
        ),
    )


def load_fabric_profile(name: str) -> FabricProfile:
    try:
        spec = FABRICS[name]
    except KeyError as exc:
        raise ValidationError("unknown fabric %s" % name) from exc
    return FabricProfile(
        name=name,
        kind=str(spec.get("kind", "inter_chip")),
        bandwidth_bytes_per_s=parse(spec["bandwidth"]),
        latency_s=parse(spec["latency"]),
        efficiency=float(spec.get("efficiency", 0.85)),
        energy_per_byte=parse(spec.get("energy_per_byte", 0)),
        provenance=Provenance(
            Confidence.PROJECTED,
            "fabric catalogue: %s" % name,
        ),
    )
