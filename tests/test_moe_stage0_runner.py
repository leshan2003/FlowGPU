import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage0_runner_emits_atomic_gate_bundle(tmp_path):
    output = tmp_path / "gate"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_moe_stage0.py"),
            "--output", str(output),
            "--model", "mixtral_8x7b",
            "--tokens", "2",
            "--requests", "1",
            "--c-max", "1",
            "--replicas", "4",
            "--hot-experts", "2",
            "--seed", "5",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert {
        path.name for path in output.iterdir()
    } == {
        "config.json",
        "gate_results.csv",
        "manifest.json",
        "placements.json",
        "stage_report.md",
        "summary.json",
        "warnings.json",
    }
    with (output / "gate_results.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    gates = {row["gate"] for row in rows}
    assert {
        "G0", "G-overlay", "C1-clustering", "H-exclusive",
        "B-stage", "refresh", "serving",
    } <= gates
    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["geometry"]["geometry"]["model_name"] == "mixtral_8x7b"
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_sha256"]
    assert manifest["claim_boundary"].startswith("Stage 0")
