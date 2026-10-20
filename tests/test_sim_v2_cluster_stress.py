import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_moe_cluster_stress import run_case  # noqa: E402


def test_aligned_c1_cluster_case_exposes_real_mechanism_gap():
    rows = run_case(-1, permutation=False)
    by_strategy = {row["strategy"]: row for row in rows}

    assert by_strategy["unique-balanced"]["plan_fingerprint"] == (
        by_strategy["pair-clustered"]["plan_fingerprint"]
    )
    assert by_strategy["full-hyperedge"]["node_cut"] == 0
    assert by_strategy["unique-balanced"]["node_cut"] > 0
    assert by_strategy["full-hyperedge"]["local_assignment_fraction"] == 1.0
    assert by_strategy["full-hyperedge"]["speedup_vs_unique"] > 1.2


def test_permuted_case_is_deterministic():
    first = run_case(7, permutation=True)
    second = run_case(7, permutation=True)
    assert [
        (row["strategy"], row["plan_fingerprint"], row["completion_s"])
        for row in first
    ] == [
        (row["strategy"], row["plan_fingerprint"], row["completion_s"])
        for row in second
    ]
