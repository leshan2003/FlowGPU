"""Run the canonical D0 runner as independent seed shards and merge them."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_moe_stage0 import file_hash  # noqa: E402
from flowgpu.sim_v2.serving import confidence_interval  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="mixtral_8x7b")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--c-max", type=int, default=2)
    parser.add_argument("--copies", type=int, default=8)
    parser.add_argument("--seeds", default="101,102,103,104,105")
    parser.add_argument("--batches", default="1,2,4")
    parser.add_argument("--nodes", default="1,2,4")
    parser.add_argument(
        "--policies",
        default="earliest_finish,least_loaded,nearest,brain-first,split",
    )
    parser.add_argument("--zipf-exponent", type=float, default=1.3)
    parser.add_argument("--layer-limit", type=int, default=0)
    parser.add_argument(
        "--jobs",
        type=int,
        default=6,
        help="Independent seed-shard subprocesses (default: 6).",
    )
    parser.add_argument("--gpu-profile", default="h100_sxm")
    parser.add_argument("--brain-profile", default="lynxi_hp300_proj")
    parser.add_argument("--bridge-fabric", default="pcie5_x16")
    return parser.parse_args()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def runner_command(args, output, seeds):
    command = [
        sys.executable,
        str(SCRIPTS / "run_moe_d0.py"),
        "--output",
        str(output),
        "--model",
        args.model,
        "--tokens",
        str(args.tokens),
        "--c-max",
        str(args.c_max),
        "--copies",
        str(args.copies),
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--batches",
        ",".join(str(value) for value in args.batches),
        "--nodes",
        ",".join(str(value) for value in args.nodes),
        "--policies",
        ",".join(args.policies),
        "--zipf-exponent",
        str(args.zipf_exponent),
        "--layer-limit",
        str(args.layer_limit),
        "--gpu-profile",
        args.gpu_profile,
        "--brain-profile",
        args.brain_profile,
        "--bridge-fabric",
        args.bridge_fabric,
        "--jobs",
        "1",
    ]
    if args.dtype is not None:
        command.extend(("--dtype", args.dtype))
    return command


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_rows(raw):
    grouped = {}
    for row in raw:
        key = (
            row["variant"],
            row["nodes"],
            row["batch"],
            row["strategy"],
            row["dispatch_policy"],
        )
        grouped.setdefault(key, []).append(row)
    aggregate = []
    for key, values in sorted(grouped.items()):
        variant, nodes, batch, strategy, dispatch_policy = key
        ci = confidence_interval(
            [float(row["speedup_vs_gpu"]) for row in values],
            lower_bound=0.0,
        )
        aggregate.append({
            "variant": variant,
            "nodes": nodes,
            "batch": batch,
            "strategy": strategy,
            "dispatch_policy": dispatch_policy,
            "runs": len(values),
            "speedup_mean": ci.mean,
            "speedup_low95": ci.low95,
            "speedup_high95": ci.high95,
            "gpu_assignment_fraction": statistics.mean(
                float(row["gpu_assignment_fraction"]) for row in values
            ),
            "gpu_busy_fraction": statistics.mean(
                float(row["gpu_busy_fraction"]) for row in values
            ),
            "brain_busy_fraction": statistics.mean(
                float(row["brain_busy_fraction"]) for row in values
            ),
            "bridge_bytes_mean": statistics.mean(
                int(row["bridge_bytes"]) for row in values
            ),
            "scaleout_bytes_mean": statistics.mean(
                int(row["scaleout_bytes"]) for row in values
            ),
            "used_brain_chips_mean": statistics.mean(
                int(row["used_brain_chips"]) for row in values
            ),
            "max_experts_per_brain_chip_mean": statistics.mean(
                int(row["max_experts_per_brain_chip"])
                for row in values
            ),
            "unique_brain_experts_mean": statistics.mean(
                int(row["unique_brain_experts"]) for row in values
            ),
        })
    return aggregate


def contrast_summaries(raw, policies):
    d0_vs_d1 = {}
    for variant, dispatch_policy in sorted({
        (row["variant"], row["dispatch_policy"])
        for row in raw
        if row["strategy"] == "M3-overlay"
    }):
        if variant != "D0-shared":
            continue
        for strategy in ("M2-tier-exclusive", "M3-overlay"):
            d1 = {
                (row["seed"], row["nodes"], row["batch"]):
                    row["completion_s"]
                for row in raw
                if row["variant"] == "D1-shared"
                and row["strategy"] == strategy
                and row["dispatch_policy"] == dispatch_policy
            }
            d0 = {
                (row["seed"], row["nodes"], row["batch"]):
                    row["completion_s"]
                for row in raw
                if row["variant"] == "D0-shared"
                and row["strategy"] == strategy
                and row["dispatch_policy"] == dispatch_policy
            }
            values = [
                float(d1[key]) / float(d0[key])
                for key in d0 if key in d1
            ]
            if values:
                ci = confidence_interval(values, lower_bound=0.0)
                d0_vs_d1["%s|%s" % (
                    strategy,
                    dispatch_policy,
                )] = {
                    "ratio": "D1 completion / D0 completion",
                    "mean": ci.mean,
                    "low95": ci.low95,
                    "high95": ci.high95,
                }

    m3_vs_m2 = {}
    for variant in ("D0-shared", "D1-shared"):
        for dispatch_policy in policies:
            m2 = {
                (row["seed"], row["nodes"], row["batch"]):
                    row["completion_s"]
                for row in raw
                if row["variant"] == variant
                and row["strategy"] == "M2-tier-exclusive"
            }
            m3 = {
                (row["seed"], row["nodes"], row["batch"]):
                    row["completion_s"]
                for row in raw
                if row["variant"] == variant
                and row["strategy"] == "M3-overlay"
                and row["dispatch_policy"] == dispatch_policy
            }
            values = [
                float(m2[key]) / float(m3[key])
                for key in m3 if key in m2
            ]
            if values:
                ci = confidence_interval(values, lower_bound=0.0)
                m3_vs_m2["%s|%s" % (
                    variant,
                    dispatch_policy,
                )] = {
                    "ratio": "M2 completion / M3 completion",
                    "mean": ci.mean,
                    "low95": ci.low95,
                    "high95": ci.high95,
                }
    return d0_vs_d1, m3_vs_m2


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    args.seeds = tuple(
        int(value) for value in args.seeds.split(",") if value
    )
    args.batches = tuple(
        int(value) for value in args.batches.split(",") if value
    )
    args.nodes = tuple(
        int(value) for value in args.nodes.split(",") if value
    )
    args.policies = tuple(
        value for value in args.policies.split(",") if value
    )
    if not all((args.seeds, args.batches, args.nodes, args.policies)):
        raise SystemExit("all experiment axes must be non-empty")

    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit("output directory already exists: %s" % output)
    temp = output.parent / (output.name + ".tmp")
    if temp.exists():
        raise SystemExit("temporary output directory already exists: %s" % temp)
    temp.mkdir(parents=True)

    shard_count = min(args.jobs, len(args.seeds))
    shards = [tuple(args.seeds[index::shard_count])
              for index in range(shard_count)]
    shard_dirs = [temp / "shards" / ("seed_shard_%02d" % index)
                  for index in range(shard_count)]
    shard_dirs[0].parent.mkdir(parents=True)

    processes = []
    for shard, shard_dir in zip(shards, shard_dirs):
        command = runner_command(args, shard_dir, shard)
        processes.append((subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ), shard_dir))

    failures = []
    for process, shard_dir in processes:
        stdout, _ = process.communicate()
        if process.returncode:
            failures.append(
                "%s (exit %d)\n%s" % (
                    shard_dir,
                    process.returncode,
                    stdout,
                )
            )
    if failures:
        raise SystemExit("seed shard failure:\n" + "\n".join(failures))

    rows_by_seed = {}
    for shard_dir in shard_dirs:
        for row in read_rows(shard_dir / "raw.csv"):
            rows_by_seed.setdefault(int(row["seed"]), []).append(row)
    raw = []
    for seed in args.seeds:
        raw.extend(rows_by_seed[seed])
    write_csv(temp / "raw.csv", raw)
    write_csv(temp / "aggregate.csv", aggregate_rows(raw))

    first_summary = json.loads(
        (shard_dirs[0] / "summary.json").read_text(encoding="utf-8")
    )
    d0_vs_d1, m3_vs_m2 = contrast_summaries(raw, args.policies)
    summary = dict(first_summary)
    summary["kind"] = "moe-d0-d1-dispatch-sharded"
    summary["seeds"] = list(args.seeds)
    summary["d0_vs_d1"] = d0_vs_d1
    summary["m3_vs_m2"] = m3_vs_m2
    summary["parallel_execution"] = {
        "jobs": shard_count,
        "seed_shards": [list(shard) for shard in shards],
        "merge_is_row_identical_to_canonical_seed_order": True,
    }
    (temp / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    first_manifest = json.loads(
        (shard_dirs[0] / "manifest.json").read_text(encoding="utf-8")
    )
    source_sha256 = dict(first_manifest["source_sha256"])
    source_sha256[str(Path(__file__).resolve())] = file_hash(Path(__file__))
    manifest = {
        "schema_version": 1,
        "kind": "moe-d0-d1-dispatch-sharded",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_sha256,
        "shard_manifest_sha256": {
            str(path.relative_to(temp)): file_sha256(path)
            for path in (
                shard_dir / "manifest.json" for shard_dir in shard_dirs
            )
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
