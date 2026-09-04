from __future__ import annotations

import math

import pytest

from mopareto.domain import ParetoFrontier, ParetoSolution
from mopareto.pareto import (
    additive_epsilon_indicator,
    condition_grid,
    epsilon_constraint_solution,
    exact_objective_coverage,
    extreme_recovery,
    hypervolume_2d,
    hypervolume_ratio,
    inverted_generational_distance_plus,
    supported_solution_flags,
    unsupported_objective_coverage,
    weighted_sum_solution,
)


def _frontier() -> ParetoFrontier:
    return ParetoFrontier(
        (
            ParetoSolution((1, 0, 0), 1, 1, 5),
            ParetoSolution((0, 1, 0), 1, 3, 2),
            ParetoSolution((0, 0, 1), 1, 5, 1),
        )
    )


def test_supportedness_detects_discrete_unsupported_point() -> None:
    frontier = _frontier()
    assert supported_solution_flags(frontier) == (True, False, True)
    assert weighted_sum_solution(frontier, 0.0).objectives == (1, 5)
    assert weighted_sum_solution(frontier, 1.0).objectives == (5, 1)
    assert weighted_sum_solution(frontier, 0.5).objectives != (3, 2)
    assert epsilon_constraint_solution(frontier, 0.4).objectives == (3, 2)


def test_exact_front_has_identity_indicators() -> None:
    frontier = _frontier()
    assert hypervolume_ratio(frontier.solutions, frontier) == pytest.approx(1.0)
    assert additive_epsilon_indicator(frontier.solutions, frontier) == pytest.approx(0.0)
    assert inverted_generational_distance_plus(frontier.solutions, frontier) == pytest.approx(0.0)
    assert exact_objective_coverage(frontier.solutions, frontier) == pytest.approx(1.0)
    assert unsupported_objective_coverage(frontier.solutions, frontier) == pytest.approx(1.0)
    assert extreme_recovery(frontier.solutions, frontier) == pytest.approx(1.0)


def test_partial_front_indicators_are_bounded() -> None:
    frontier = _frontier()
    approximation = (frontier.solutions[0], frontier.solutions[-1])
    assert 0.0 < hypervolume_ratio(approximation, frontier) < 1.0
    assert 0.0 < additive_epsilon_indicator(approximation, frontier) <= 1.0
    assert 0.0 < inverted_generational_distance_plus(approximation, frontier) < math.sqrt(2)
    assert exact_objective_coverage(approximation, frontier) == pytest.approx(2 / 3)
    assert unsupported_objective_coverage(approximation, frontier) == pytest.approx(0.0)


def test_hypervolume_known_area() -> None:
    frontier = _frontier()
    assert hypervolume_2d(frontier.solutions, ideal_point=(5, 5)) == pytest.approx(0.44)
    assert hypervolume_2d((), ideal_point=(5, 5)) == 0.0


def test_no_unsupported_points_returns_none() -> None:
    frontier = ParetoFrontier(
        (
            ParetoSolution((1, 0), 1, 1, 3),
            ParetoSolution((0, 1), 1, 3, 1),
        )
    )
    assert supported_solution_flags(frontier) == (True, True)
    assert unsupported_objective_coverage(frontier.solutions, frontier) is None


def test_condition_validation() -> None:
    assert condition_grid(3) == (0.0, 0.5, 1.0)
    with pytest.raises(ValueError, match="at least two"):
        condition_grid(1)
    with pytest.raises(ValueError, match="unit interval"):
        weighted_sum_solution(_frontier(), float("nan"))
    with pytest.raises(ValueError, match="unit interval"):
        epsilon_constraint_solution(_frontier(), 1.1)
