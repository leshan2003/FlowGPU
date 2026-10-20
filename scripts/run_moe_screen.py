"""Run a small paired-seed MoE screening study."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_moe_stage0 import (  # noqa: E402
    effective_dtype,
    file_hash,
    run_c1_gate,
    run_gate,
    run_serving_gate,
)
from flowgpu.sim_v2.serving import confidence_interval  # noqa: E402


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seeds", default="101,102,103,104,105")
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--c-max", type=int, default=2)
    parser.add_argument("--replicas", type=int, default=16)
    parser.add_argument("--hot-experts", type=int, default=8)
    parser.add_argument("--zipf-exponent", type=float, default=1.3)
    parser.add_argument("--layer-limit", type=int, default=0)
    parser.add_argument("--gpu-profile", default="h100_sxm")
    parser.add_argument("--brain-profile", default="lynxi_hp300_proj")
    parser.add_argument("--bridge-fabric", default="pcie5_x16")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit("output directory already exists: %s" % output)
    temp = output.parent / (output.name + ".tmp")
    if temp.exists():
        raise SystemExit("temporary output directory already exists: %s" % temp)
    temp.mkdir(parents=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if not seeds:
        raise SystemExit("at least one seed is required")

    raw_rows = []
    for seed in seeds:
        local = SimpleNamespace(
            model=args.model,
            dtype=args.dtype,
            tokens=args.tokens,
            requests=args.requests,
            c_max=args.c_max,
            replicas=args.replicas,
            hot_experts=args.hot_experts,
            seed=seed,
            zipf_exponent=args.zipf_exponent,
            layer_limit=args.layer_limit,
            gpu_profile=args.gpu_profile,
            brain_profile=args.brain_profile,
            bridge_fabric=args.bridge_fabric,
        )
        g_rows, _, _ = run_gate(local, temp)
        serving_rows, _, _ = run_serving_gate(local)
        for row in g_rows + serving_rows:
            raw_rows.append({
                "seed": seed,
                "gate": row.gate,
                "strategy": row.strategy,
                "mode": row.mode,
                "completion_s": row.completion_s,
                "speedup_vs_gpu": row.speedup_vs_gpu,
                "slo_goodput_per_s": row.slo_goodput_per_s,
                "tpot_p99_s": row.tpot_p99_s,
                "status": row.status,
            })
    c1_rows, c1_meta, _ = run_c1_gate(SimpleNamespace(
        model=args.model,
        dtype=args.dtype,
        tokens=args.tokens,
        requests=args.requests,
        c_max=args.c_max,
        replicas=args.replicas,
        hot_experts=args.hot_experts,
        seed=seeds[0],
        zipf_exponent=args.zipf_exponent,
        layer_limit=args.layer_limit,
        gpu_profile=args.gpu_profile,
        brain_profile=args.brain_profile,
        bridge_fabric=args.bridge_fabric,
    ))
    for row in c1_rows:
        raw_rows.append({
            "seed": seeds[0],
            "gate": row.gate,
            "strategy": row.strategy,
            "mode": row.mode,
            "completion_s": row.completion_s,
            "speedup_vs_gpu": row.speedup_vs_gpu,
            "slo_goodput_per_s": row.slo_goodput_per_s,
            "tpot_p99_s": row.tpot_p99_s,
            "status": row.status,
        })

    grouped = {}
    for row in raw_rows:
        grouped.setdefault((row["gate"], row["strategy"]), []).append(
            row["speedup_vs_gpu"]
        )
    aggregates = []
    for (gate, strategy), values in sorted(grouped.items()):
        ci = confidence_interval(values, lower_bound=0.0)
        aggregates.append({
            "gate": gate,
            "strategy": strategy,
            "runs": len(values),
            "speedup_mean": ci.mean,
            "speedup_low95": ci.low95,
            "speedup_high95": ci.high95,
        })
    write_csv(temp / "raw.csv", raw_rows)
    write_csv(temp / "aggregate.csv", aggregates)
    summary = {
        "schema_version": 1,
        "model": args.model,
        "dtype": effective_dtype(args),
        "seeds": list(seeds),
        "zipf_exponent": args.zipf_exponent,
        "layer_limit": args.layer_limit,
        "gpu_profile": args.gpu_profile,
        "brain_profile": args.brain_profile,
        "bridge_fabric": args.bridge_fabric,
        "raw_rows": len(raw_rows),
        "c1": c1_meta,
        "claim_boundary": (
            "small paired-seed Stage 1 screen; projected hardware and "
            "synthetic routing; not a final performance claim"
        ),
    }
    (temp / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    source_paths = [
        Path(__file__),
        SCRIPTS / "run_moe_stage0.py",
        ROOT / "flowgpu" / "sim_v2" / "serving.py",
        ROOT / "flowgpu" / "sim_v2" / "research.py",
        ROOT / "flowgpu" / "sim_v2" / "placement.py",
        ROOT / "flowgpu" / "sim_v2" / "hardware_profiles.py",
    ]
    manifest = {
        "schema_version": 1,
        "kind": "moe-stage1-screen",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {
            str(path.resolve()): file_hash(path)
            for path in source_paths
        },
        "claim_boundary": summary["claim_boundary"],
    }
    (temp / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp.rename(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
