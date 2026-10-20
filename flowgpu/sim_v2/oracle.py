"""Exact small-topology replica-placement oracle for Tier-2 validation."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

from .replicas import ReplicaPlan
from .schema import ChipSpec, ExpertSpec, ReplicaSpec, ValidationError


@dataclass(frozen=True)
class OracleResult:
    plan: ReplicaPlan
    objective: float
    evaluated_candidates: int
    feasible_candidates: int
    search_space_upper_bound: int


@dataclass(frozen=True)
class OracleGapReport:
    """Tier-2 comparison of one heuristic against the exact small oracle."""

    heuristic_strategy: str
    heuristic_objective: float
    oracle_available: bool
    oracle_objective: Optional[float]
    absolute_gap: Optional[float]
    relative_gap: Optional[float]
    heuristic_is_optimal: Optional[bool]
    evaluated_candidates: int
    feasible_candidates: int
    search_space_upper_bound: Optional[int]
    search_complete: bool
    warning: str


def audit_oracle_gap(
    heuristic_plan: ReplicaPlan,
    objective: Callable[[ReplicaPlan], float],
    max_extra_replicas: int = 0,
    max_chips: int = 8,
    max_candidates: int = 1_000_000,
    tolerance: float = 1e-12,
) -> OracleGapReport:
    """Run the bounded exact oracle and report a heuristic's optimality gap.

    A refused search is returned as an unavailable report rather than silently
    treating the best heuristic as optimal.  No partial candidate search is
    performed, so ``search_complete=False`` means "not run", not an
    approximate oracle result.
    """

    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValidationError("oracle gap tolerance must be finite and >= 0")
    heuristic_value = float(objective(heuristic_plan))
    if not math.isfinite(heuristic_value):
        raise ValidationError(
            "oracle gap objective must return a finite value"
        )
    required_extra = len(heuristic_plan.replicas) - len(
        heuristic_plan.experts
    )
    if required_extra > max_extra_replicas:
        raise ValidationError(
            "heuristic uses %d extra replicas but oracle budget is %d" %
            (required_extra, max_extra_replicas)
        )
    try:
        oracle = small_replica_oracle(
            experts=heuristic_plan.experts,
            chips=heuristic_plan.chips,
            objective=objective,
            max_extra_replicas=max_extra_replicas,
            max_chips=max_chips,
            max_candidates=max_candidates,
        )
    except ValidationError as exc:
        message = str(exc)
        if ("exceeds max_candidates" not in message and
                "supports at most" not in message):
            raise
        return OracleGapReport(
            heuristic_strategy=heuristic_plan.strategy,
            heuristic_objective=heuristic_value,
            oracle_available=False,
            oracle_objective=None,
            absolute_gap=None,
            relative_gap=None,
            heuristic_is_optimal=None,
            evaluated_candidates=0,
            feasible_candidates=0,
            search_space_upper_bound=None,
            search_complete=False,
            warning=(
                "exact oracle was not run within declared bounds: " +
                message
            ),
        )

    gap = heuristic_value - oracle.objective
    if gap < -tolerance:
        raise ValidationError(
            "heuristic objective is below exact oracle; objective or "
            "candidate budgets are inconsistent"
        )
    if abs(gap) <= tolerance:
        gap = 0.0
    if abs(oracle.objective) > tolerance:
        relative_gap = gap / abs(oracle.objective)
    elif gap == 0:
        relative_gap = 0.0
    else:
        relative_gap = None
    warning = ""
    if relative_gap is None:
        warning = (
            "relative gap is undefined because oracle objective is zero; "
            "use absolute_gap"
        )
    return OracleGapReport(
        heuristic_strategy=heuristic_plan.strategy,
        heuristic_objective=heuristic_value,
        oracle_available=True,
        oracle_objective=oracle.objective,
        absolute_gap=gap,
        relative_gap=relative_gap,
        heuristic_is_optimal=(gap == 0),
        evaluated_candidates=oracle.evaluated_candidates,
        feasible_candidates=oracle.feasible_candidates,
        search_space_upper_bound=oracle.search_space_upper_bound,
        search_complete=True,
        warning=warning,
    )


def small_replica_oracle(
    experts: Iterable[ExpertSpec],
    chips: Iterable[ChipSpec],
    objective: Callable[[ReplicaPlan], float],
    max_extra_replicas: int = 0,
    max_chips: int = 8,
    max_candidates: int = 1_000_000,
) -> OracleResult:
    """Exhaustively minimize an objective over complete replica locations.

    Every expert selects a non-empty subset of chips.  The first chip in the
    deterministic subset is labelled primary; this label does not change
    residency or execution semantics.  Capacity-invalid candidates are
    rejected by :class:`ReplicaPlan`.

    The oracle is intentionally bounded and must not be used as a scalable
    production optimizer.
    """

    expert_list = tuple(sorted(experts, key=lambda item: item.key))
    chip_list = tuple(sorted(chips, key=lambda item: item.chip_id))
    if not expert_list or not chip_list:
        raise ValidationError("small oracle needs experts and chips")
    if len(chip_list) > max_chips:
        raise ValidationError(
            "small oracle supports at most %d chips" % max_chips
        )
    if max_extra_replicas < 0:
        raise ValidationError("max_extra_replicas cannot be negative")
    if max_candidates <= 0:
        raise ValidationError("max_candidates must be positive")

    max_copies_per_expert = min(
        len(chip_list), 1 + max_extra_replicas
    )
    subsets = tuple(
        combination
        for size in range(1, max_copies_per_expert + 1)
        for combination in itertools.combinations(
            range(len(chip_list)), size
        )
    )
    upper_bound = len(subsets) ** len(expert_list)
    if upper_bound > max_candidates:
        raise ValidationError(
            "oracle search upper bound %d exceeds max_candidates %d" %
            (upper_bound, max_candidates)
        )

    evaluated = 0
    feasible = 0
    best_plan = None
    best_key = None
    for selections in itertools.product(subsets, repeat=len(expert_list)):
        extra = sum(len(selection) - 1 for selection in selections)
        if extra > max_extra_replicas:
            continue
        evaluated += 1
        replicas = []
        signature = []
        for expert, selection in zip(expert_list, selections):
            for copy_index, chip_index in enumerate(selection):
                chip = chip_list[chip_index]
                signature.append((expert.key, chip.chip_id))
                replicas.append(ReplicaSpec(
                    replica_id="oracle-L%d-%sE%d-C%d" % (
                        expert.key.layer_id,
                        "S" if expert.key.shared else "R",
                        expert.key.expert_id,
                        copy_index,
                    ),
                    expert=expert.key,
                    chip_id=chip.chip_id,
                    is_primary=(copy_index == 0),
                ))
        try:
            plan = ReplicaPlan(
                expert_list, chip_list, replicas, "small-oracle"
            )
        except ValidationError:
            continue
        feasible += 1
        value = float(objective(plan))
        if not math.isfinite(value):
            raise ValidationError(
                "oracle objective must return a finite value"
            )
        key = (value, tuple(signature))
        if best_key is None or key < best_key:
            best_key = key
            best_plan = plan

    if best_plan is None:
        raise ValidationError("small oracle found no feasible placement")
    return OracleResult(
        plan=best_plan,
        objective=best_key[0],
        evaluated_candidates=evaluated,
        feasible_candidates=feasible,
        search_space_upper_bound=upper_bound,
    )
