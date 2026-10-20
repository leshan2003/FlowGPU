import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_screen_runner_respects_layer_limit_and_emits_ci(tmp_path):
    output = tmp_path / "screen"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_moe_screen.py"),
            "--output", str(output),
            "--model", "mixtral_8x7b",
            "--seeds", "1,2",
            "--tokens", "2",
            "--requests", "1",
            "--c-max", "1",
            "--replicas", "4",
            "--hot-experts", "2",
            "--layer-limit", "1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["layer_limit"] == 1
    assert summary["raw_rows"] > 0
    with (output / "aggregate.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    coverage = next(
        row for row in rows
        if row["gate"] == "G-overlay" and
        row["strategy"] == "coverage-first"
    )
    assert int(coverage["runs"]) == 2
    assert float(coverage["speedup_low95"]) <= float(
        coverage["speedup_mean"]
    ) <= float(coverage["speedup_high95"])
