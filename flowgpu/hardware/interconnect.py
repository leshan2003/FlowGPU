"""Interconnect model: links, collectives, and the heterogeneous bridge.

Three tiers, matching the feature list:

``noc``          on-chip network (modelled inside the device)
``inter_chip``   chip-to-chip inside a board/module (NVLink, IPU-Link, C2C)
``inter_board``  board-to-board / node-to-node (RoCE, InfiniBand, PCIe host)

Plus the one that decides whether the article's architecture works at all:

``bridge``       the GPU <-> dataflow-chip path.  In a PD+A split the hidden
                 state crosses this link **twice per layer per decode step**.
                 For a 61-layer model at batch 64 and d=7168 that is ~112 MB
                 per token step; at 64 GB/s of usable PCIe that alone is
                 1.75 ms of TPOT.  Getting this link right is the single most
                 important modelling decision in the whole study.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..units import parse


@dataclass
class Link:
    name: str
    bandwidth: float           # bytes/s, unidirectional usable peak
    latency: float             # seconds, one-way
    efficiency: float = 0.85   # protocol + framing + congestion derate
    energy_per_byte: float = 0.0
    bidirectional: bool = True
    world: int = 2             # endpoints sharing this fabric
    kind: str = "inter_chip"   # inter_chip | inter_board | bridge

    @property
    def eff_bw(self) -> float:
        return self.bandwidth * self.efficiency

    # -- point to point ------------------------------------------------
    def p2p_time(self, nbytes: float) -> float:
        if nbytes <= 0:
            return 0.0
        return self.latency + nbytes / max(self.eff_bw, 1.0)

    def p2p_energy(self, nbytes: float) -> float:
        return nbytes * self.energy_per_byte

    # -- collectives ---------------------------------------------------
    def allreduce_time(self, nbytes: float, n: int | None = None,
                       algo: str = "ring") -> float:
        """Ring all-reduce: 2(n-1)/n bytes per rank over the wire."""
        n = n or self.world
        if n <= 1 or nbytes <= 0:
            return 0.0
        if algo == "ring":
            vol = 2.0 * (n - 1) / n * nbytes
            hops = 2 * (n - 1)
        elif algo == "tree":
            vol = 2.0 * nbytes
            hops = 2 * math.ceil(math.log2(n))
        else:                       # direct / one-shot
            vol = (n - 1) * nbytes
            hops = 2
        return hops * self.latency + vol / max(self.eff_bw, 1.0)

    def allgather_time(self, nbytes: float, n: int | None = None) -> float:
        n = n or self.world
        if n <= 1 or nbytes <= 0:
            return 0.0
        return (n - 1) * self.latency + (n - 1) / n * nbytes / max(self.eff_bw, 1.0)

    def all2all_time(self, nbytes: float, n: int | None = None) -> float:
        """Expert dispatch/combine: each rank sends (n-1)/n of its buffer."""
        n = n or self.world
        if n <= 1 or nbytes <= 0:
            return 0.0
        return self.latency + (n - 1) / n * nbytes / max(self.eff_bw, 1.0)

    def collective_bytes(self, nbytes: float, op: str,
                         n: int | None = None) -> float:
        n = n or self.world
        if n <= 1:
            return 0.0
        if op == "allreduce":
            return 2.0 * (n - 1) / n * nbytes
        if op in ("allgather", "reducescatter"):
            return (n - 1) / n * nbytes
        if op == "all2all":
            return (n - 1) / n * nbytes
        return nbytes


# --- catalogue of real fabrics -------------------------------------------
FABRICS = {
    # scale-up (inside a node / module)
    "nvlink3":   dict(bandwidth="300 GB/s", latency="1.5 us", efficiency=0.92,
                      energy_per_byte="8 pJ", kind="inter_chip"),
    "nvlink4":   dict(bandwidth="450 GB/s", latency="1.3 us", efficiency=0.92,
                      energy_per_byte="6 pJ", kind="inter_chip"),
    "nvlink5":   dict(bandwidth="900 GB/s", latency="1.1 us", efficiency=0.92,
                      energy_per_byte="4 pJ", kind="inter_chip"),
    "hccs":      dict(bandwidth="196 GB/s", latency="2.5 us", efficiency=0.88,
                      energy_per_byte="10 pJ", kind="inter_chip"),
    "mtlink":    dict(bandwidth="120 GB/s", latency="4 us", efficiency=0.85,
                      energy_per_byte="12 pJ", kind="inter_chip"),
    "metaxlink": dict(bandwidth="224 GB/s", latency="3.5 us", efficiency=0.85,
                      energy_per_byte="11 pJ", kind="inter_chip"),
    "ipulink":   dict(bandwidth="160 GB/s", latency="1 us", efficiency=0.90,
                      energy_per_byte="9 pJ", kind="inter_chip"),
    "groq_c2c":  dict(bandwidth="100 GB/s", latency="0.6 us", efficiency=0.95,
                      energy_per_byte="5 pJ", kind="inter_chip"),
    "infinity_fabric": dict(bandwidth="448 GB/s", latency="1.8 us",
                            efficiency=0.90, energy_per_byte="7 pJ",
                            kind="inter_chip"),
    # host / bridge
    "pcie4_x16": dict(bandwidth="32 GB/s", latency="1.5 us", efficiency=0.80,
                      energy_per_byte="15 pJ", kind="bridge"),
    "pcie5_x16": dict(bandwidth="64 GB/s", latency="1.2 us", efficiency=0.80,
                      energy_per_byte="12 pJ", kind="bridge"),
    "pcie6_x16": dict(bandwidth="128 GB/s", latency="1.0 us", efficiency=0.82,
                      energy_per_byte="10 pJ", kind="bridge"),
    "cxl3":      dict(bandwidth="128 GB/s", latency="0.6 us", efficiency=0.88,
                      energy_per_byte="9 pJ", kind="bridge"),
    "custom_serdes_400g": dict(bandwidth="50 GB/s", latency="0.5 us",
                               efficiency=0.90, energy_per_byte="6 pJ",
                               kind="bridge"),
    "custom_serdes_800g": dict(bandwidth="100 GB/s", latency="0.4 us",
                               efficiency=0.90, energy_per_byte="5 pJ",
                               kind="bridge"),
    "custom_serdes_1600g": dict(bandwidth="200 GB/s", latency="0.35 us",
                                efficiency=0.90, energy_per_byte="4 pJ",
                                kind="bridge"),
    # scale-out
    "ib_hdr":    dict(bandwidth="25 GB/s", latency="5 us", efficiency=0.85,
                      energy_per_byte="25 pJ", kind="inter_board"),
    "ib_ndr":    dict(bandwidth="50 GB/s", latency="4 us", efficiency=0.85,
                      energy_per_byte="20 pJ", kind="inter_board"),
    "ib_xdr":    dict(bandwidth="100 GB/s", latency="3.5 us", efficiency=0.85,
                      energy_per_byte="16 pJ", kind="inter_board"),
    "roce_200g": dict(bandwidth="25 GB/s", latency="8 us", efficiency=0.78,
                      energy_per_byte="30 pJ", kind="inter_board"),
    "roce_400g": dict(bandwidth="50 GB/s", latency="7 us", efficiency=0.78,
                      energy_per_byte="24 pJ", kind="inter_board"),
    "eth_100g":  dict(bandwidth="12.5 GB/s", latency="20 us", efficiency=0.72,
                      energy_per_byte="40 pJ", kind="inter_board"),
}


def make_link(name: str, world: int = 2, overrides: dict | None = None) -> Link:
    if name in FABRICS:
        spec = dict(FABRICS[name])
    else:
        spec = {}
    if overrides:
        spec.update(overrides)
    if not spec:
        raise KeyError(f"unknown fabric {name!r}. Known: {sorted(FABRICS)}")
    return Link(
        name=name,
        bandwidth=parse(spec["bandwidth"]),
        latency=parse(spec["latency"]),
        efficiency=float(spec.get("efficiency", 0.85)),
        energy_per_byte=parse(spec.get("energy_per_byte", 0)),
        world=world,
        kind=spec.get("kind", "inter_chip"),
    )


def list_fabrics() -> list:
    return sorted(FABRICS)
