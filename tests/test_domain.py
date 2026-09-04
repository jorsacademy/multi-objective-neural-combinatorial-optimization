from __future__ import annotations

import numpy as np
import pytest

from mopareto.dataset import generate_instance
from mopareto.domain import (
    MultiObjectiveKnapsackInstance,
    ParetoFrontier,
    ParetoSolution,
    audit_solution,
    nondominated_solutions,
    solution_from_selection,
    solve_pareto_brute_force,
    solve_pareto_dynamic_programming,
)


def test_exact_dynamic_programming_matches_exhaustive_enumeration() -> None:
    for regime in ("independent", "aligned", "strongly_conflicting", "clustered"):
        for seed in (7, 19):
            instance = generate_instance(item_count=8, regime=regime, seed=seed)
            dynamic = solve_pareto_dynamic_programming(instance)
            brute = solve_pareto_brute_force(instance)
            assert dynamic == brute


def test_dynamic_programming_trace_and_audits(
    simple_instance: MultiObjectiveKnapsackInstance,
) -> None:
    result = solve_pareto_dynamic_programming(simple_instance, return_trace=True)
    assert isinstance(result, tuple)
    frontier, trace = result
    assert trace.item_count == simple_instance.item_count
    assert trace.capacity == simple_instance.capacity
    assert trace.maximum_frontier_size >= len(frontier.solutions)
    for solution in frontier.solutions:
        audit = audit_solution(
            simple_instance,
            solution.selection,
            reported_objectives=solution.objectives,
            frontier=frontier,
        )
        assert audit.binary and audit.feasible
        assert audit.reported_objectives_consistent
        assert audit.nondominated is True


def test_nondominated_filter_canonicalizes_duplicate_objectives() -> None:
    candidates = (
        ParetoSolution((1, 0, 0), 5, 10, 4),
        ParetoSolution((0, 1, 0), 4, 10, 4),
        ParetoSolution((0, 0, 1), 3, 8, 3),
        ParetoSolution((1, 1, 0), 7, 6, 9),
    )
    filtered = nondominated_solutions(candidates)
    assert (
        filtered
        == (
            ParetoSolution((0, 1, 0), 4, 10, 4),
            ParetoSolution((1, 1, 0), 7, 6, 9),
        )[::-1]
    )
    assert filtered[0].objectives == (6, 9)
    assert filtered[1].objectives == (10, 4)


def test_solution_from_selection_accepts_numpy_integers(
    simple_instance: MultiObjectiveKnapsackInstance,
) -> None:
    solution = solution_from_selection(simple_instance, [np.int64(1), 0, True, 0])
    assert solution.selection == (1, 0, 1, 0)
    assert solution.total_weight == 6
    assert solution.objectives == (17, 7)
    with pytest.raises(ValueError, match="binary"):
        solution_from_selection(simple_instance, [1, 0, 2, 0])


def test_invalid_selection_audit_fails_closed(
    simple_instance: MultiObjectiveKnapsackInstance,
) -> None:
    audit = audit_solution(simple_instance, [1, 0])
    assert not audit.binary
    assert not audit.feasible
    assert audit.total_weight == -1


def test_instance_and_frontier_validation() -> None:
    with pytest.raises(ValueError, match="at least one item"):
        MultiObjectiveKnapsackInstance((), (), (), 1)
    with pytest.raises(ValueError, match="aligned"):
        MultiObjectiveKnapsackInstance((1, 2), (1,), (1, 2), 1)
    with pytest.raises(ValueError, match="positive"):
        MultiObjectiveKnapsackInstance((0, 2), (1, 2), (1, 2), 1)
    with pytest.raises(ValueError, match="admit at least one"):
        MultiObjectiveKnapsackInstance((3, 4), (1, 2), (1, 2), 2)
    with pytest.raises(ValueError, match="potentially excluded"):
        MultiObjectiveKnapsackInstance((1, 2), (1, 2), (1, 2), 3)
    with pytest.raises(ValueError, match="canonical"):
        ParetoFrontier(
            (
                ParetoSolution((1,), 1, 2, 2),
                ParetoSolution((0,), 0, 1, 1),
            )
        )


def test_brute_force_limit(simple_instance: MultiObjectiveKnapsackInstance) -> None:
    with pytest.raises(ValueError, match="limit"):
        solve_pareto_brute_force(simple_instance, maximum_items=3)
