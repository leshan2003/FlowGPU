"""Unit helpers.

Internal canonical units throughout FlowGPU:

============  ==========================
quantity      canonical unit
============  ==========================
time          seconds (float)
data          bytes (float)
compute       FLOP (float, 1 MAC = 2 FLOP)
bandwidth     bytes / second
energy        joules
power         watts
============  ==========================

Every config file may use suffixed strings ("80 GB", "3.35 TB/s", "989 TFLOPS",
"700 W", "1.2 us") which :func:`parse` converts to canonical units.
"""

from __future__ import annotations

import re

# --- scale prefixes -------------------------------------------------------
_SI = {
    "": 1.0,
    "k": 1e3,
    "K": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
    "P": 1e15,
    "E": 1e18,
    "m": 1e-3,
    "u": 1e-6,
    "µ": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}
# binary prefixes for memory capacities
_BIN = {"": 1.0, "K": 2**10, "Ki": 2**10, "M": 2**20, "Mi": 2**20,
        "G": 2**30, "Gi": 2**30, "T": 2**40, "Ti": 2**40, "P": 2**50}

KB = 2**10
MB = 2**20
GB = 2**30
TB = 2**40

TFLOPS = 1e12
GFLOPS = 1e9
POPS = 1e15

NS = 1e-9
US = 1e-6
MS = 1e-3

_NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
_PAT = re.compile(rf"^\s*{_NUM}\s*([A-Za-zµ/%]*)\s*$")


class UnitError(ValueError):
    pass


def parse(value, kind: str | None = None) -> float:
    """Parse ``value`` into canonical units.

    ``kind`` disambiguates bare numbers and picks binary vs decimal prefixes.
    Accepted kinds: ``time``, ``bytes``, ``bandwidth``, ``flops``, ``ops``,
    ``power``, ``energy``, ``freq``, ``ratio``, ``count``.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise UnitError(f"cannot parse {value!r}")

    m = _PAT.match(value)
    if not m:
        raise UnitError(f"cannot parse {value!r}")
    num = float(m.group(1))
    suf = m.group(2)

    if not suf:
        return num

    # percentages
    if suf == "%":
        return num / 100.0

    # Strip an optional "/s" (or "ps" as in "Gbps") rate tail first.
    # Rates always use DECIMAL prefixes (a 3.35 TB/s HBM3 stack moves
    # 3.35e12 B/s, not 3.35*2^40); capacities use BINARY ones.
    tail = suf
    is_rate = False
    if tail.endswith("/s") or tail.endswith("/S"):
        tail, is_rate = tail[:-2], True
    elif tail.lower().endswith("ps") and not tail.lower().endswith("flops") \
            and not tail.lower().endswith("ops") and len(tail) > 2:
        tail, is_rate = tail[:-2], True              # "Gbps" -> "Gb"

    low = tail.lower()
    binary = False
    if low.endswith("flops") or low.endswith("flop"):
        base = "FLOP"
        pfx = tail[: -5] if low.endswith("flops") else tail[: -4]
    elif low.endswith("ops") or low.endswith("op"):
        base = "OP"
        pfx = tail[: -3] if low.endswith("ops") else tail[: -2]
    elif low.endswith("ib"):                 # KiB / MiB / GiB / TiB
        binary, base, pfx = True, "B", tail[:-2]
    elif tail.endswith("B"):                 # bytes: capacities are 2^n
        binary, base, pfx = (not is_rate), "B", tail[:-1]
    elif tail.endswith("b"):                 # bits
        base, pfx = "b", tail[:-1]
    elif low.endswith("hz"):
        base, pfx = "Hz", tail[:-2]
    elif low.endswith("s"):
        base, pfx = "s", tail[:-1]
    elif tail in ("W", "w") or (low.endswith("w") and len(tail) <= 2):
        base, pfx = "W", tail[:-1]
    elif tail in ("J", "j") or (low.endswith("j") and len(tail) <= 2):
        base, pfx = "J", tail[:-1]
    else:
        raise UnitError(f"unknown unit {suf!r} in {value!r}")

    scale = (_BIN.get(pfx) if binary else None) or _SI.get(pfx)
    if scale is None:
        raise UnitError(f"unknown prefix {pfx!r} in {value!r}")

    out = num * scale
    if base == "b":                          # bits -> bytes
        out /= 8.0
    return out


def fmt_time(t: float) -> str:
    if t is None:
        return "-"
    if t >= 1.0:
        return f"{t:.3f} s"
    if t >= 1e-3:
        return f"{t * 1e3:.3f} ms"
    if t >= 1e-6:
        return f"{t * 1e6:.3f} us"
    return f"{t * 1e9:.2f} ns"


def fmt_bytes(b: float) -> str:
    if b is None:
        return "-"
    for unit, s in (("TB", TB), ("GB", GB), ("MB", MB), ("KB", KB)):
        if abs(b) >= s:
            return f"{b / s:.3f} {unit}"
    return f"{b:.0f} B"


def fmt_bw(b: float) -> str:
    if b is None:
        return "-"
    for unit, s in (("TB/s", 1e12), ("GB/s", 1e9), ("MB/s", 1e6)):
        if abs(b) >= s:
            return f"{b / s:.2f} {unit}"
    return f"{b:.0f} B/s"


def fmt_flops(f: float) -> str:
    if f is None:
        return "-"
    for unit, s in (("PFLOPS", 1e15), ("TFLOPS", 1e12), ("GFLOPS", 1e9)):
        if abs(f) >= s:
            return f"{f / s:.2f} {unit}"
    return f"{f:.0f} FLOPS"


def fmt_energy(j: float) -> str:
    if j is None:
        return "-"
    for unit, s in (("kJ", 1e3), ("J", 1.0), ("mJ", 1e-3), ("uJ", 1e-6), ("nJ", 1e-9)):
        if abs(j) >= s:
            return f"{j / s:.3f} {unit}"
    return f"{j * 1e12:.2f} pJ"


DTYPE_BYTES = {
    "fp64": 8.0, "fp32": 4.0, "tf32": 4.0, "bf16": 2.0, "fp16": 2.0,
    "fp8": 1.0, "int8": 1.0, "fp6": 0.75, "fp4": 0.5, "int4": 0.5,
    "mxfp4": 0.5 + 1.0 / 32.0,   # 4-bit element + shared e8m0 scale per 32
    "spike": 0.125,              # 1-bit binary spike event
}


def dtype_bytes(dtype: str) -> float:
    d = dtype.lower()
    if d not in DTYPE_BYTES:
        raise UnitError(f"unknown dtype {dtype!r}")
    return DTYPE_BYTES[d]
