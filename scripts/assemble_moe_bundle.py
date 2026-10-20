"""Assemble the full-scale static and fixed-size mechanism result bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", required=True)
    parser.add_argument("--mechanisms", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    static = Path(args.static).resolve()
    mechanisms = Path(args.mechanisms).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit("output directory already exists: %s" % output)
    output.mkdir(parents=True)
    files = [
        static / "static_capacity.csv",
        mechanisms / "community.csv",
        mechanisms / "node_batch.csv",
        mechanisms / "skew.csv",
        mechanisms / "refresh.csv",
        mechanisms / "seeds.csv",
        mechanisms / "seed_ci.csv",
        mechanisms / "sensitivity.csv",
    ]
    for source in files:
        if not source.is_file():
            raise SystemExit("missing bundle file: %s" % source)
        shutil.copy2(source, output / source.name)
    ranked_full = mechanisms / "ranked_full.csv"
    if ranked_full.is_file():
        shutil.copy2(ranked_full, output / ranked_full.name)
        files.append(ranked_full)
    static_details = json.loads(
        (static / "details.json").read_text(encoding="utf-8")
    )
    mechanism_details = json.loads(
        (mechanisms / "details.json").read_text(encoding="utf-8")
    )
    (output / "details.json").write_text(
        json.dumps(
            static_details + mechanism_details,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    static_manifest = json.loads(
        (static / "manifest.json").read_text(encoding="utf-8")
        if (static / "manifest.json").is_file() else "{}"
    )
    mechanism_manifest = json.loads(
        (mechanisms / "manifest.json").read_text(encoding="utf-8")
    )
    manifest = {
        "schema_version": 1,
        "bundle_kind": "moe-research-combined",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "static_source": str(static),
        "mechanism_source": str(mechanisms),
        "static_manifest_present": bool(static_manifest),
        "static_manifest": static_manifest,
        "mechanism_manifest": mechanism_manifest,
        "source_bundle_file_sha256": {
            str(path.resolve()): sha256(path) for path in files
        },
        "combined_details_sha256": sha256(output / "details.json"),
        "claim_boundary": (
            "paper-constrained synthetic routing and projected hardware; "
            "not real router trace or KA200 measurement"
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
