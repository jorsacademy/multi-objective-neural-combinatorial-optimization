"""Exact condition oracles and two-objective Pareto quality indicators."""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

import numpy as np

from mopareto.domain import ParetoFrontier, ParetoSolution, nondominated_solutions


def _unit_fraction(value: float) -> Fraction:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("condition values must lie in the finite unit interval")
    return Fraction(value).limit_denominator(1_000_000)


def weighted_sum_solution(frontier: ParetoFrontier, weight_1: float) -> ParetoSolution:
    """Return a deterministic exact normalized weighted-sum optimum."""

    preference = _unit_fraction(weight_1)
    weight_2 = Fraction(1, 1) - preference
    ideal_1, ideal_2 = frontier.ideal_point

    scored: list[tuple[Fraction, ParetoSolution]] = []
    for solution in frontier.solutions:
        score = preference * Fraction(solution.objective_1, ideal_1) + weight_2 * Fraction(
            solution.objective_2, ideal_2
        )
        scored.append((score, solution))
    best_score = max(score for score, _ in scored)
    ties = [solution for score, solution in scored if score == best_score]
    return min(
        ties,
        key=lambda solution: (
            -min(
                Fraction(solution.objective_1, ideal_1),
                Fraction(solution.objective_2, ideal_2),
            ),
            -(Fraction(solution.objective_1, ideal_1) + Fraction(solution.objective_2, ideal_2)),
            solution.total_weight,
            solution.selection,
        ),
    )


def epsilon_constraint_solution(frontier: ParetoFrontier, target_2: float) -> ParetoSolution:
    """Maximize objective 1 subject to a normalized objective-2 floor."""

    target = _unit_fraction(target_2)
    _, ideal_2 = frontier.ideal_point
    eligible = [
        solution
        for solution in frontier.solutions
        if Fraction(solution.objective_2, ideal_2) >= target
    ]
    if not eligible:
        raise RuntimeError("the exact frontier must contain a point attaining objective-2 ideal")
    return min(
        eligible,
        key=lambda solution: (
            -solution.objective_1,
            -solution.objective_2,
            solution.total_weight,
            solution.selection,
        ),
    )


def supported_solution_flags(frontier: ParetoFrontier) -> tuple[bool, ...]:
    """Classify weakly supported points by exact normalized weight-interval feasibility."""

    ideal_1, ideal_2 = frontier.ideal_point
    normalized = [
        (
            Fraction(solution.objective_1, ideal_1),
            Fraction(solution.objective_2, ideal_2),
        )
        for solution in frontier.solutions
    ]
    flags: list[bool] = []
    for point_index, point in enumerate(normalized):
        lower = Fraction(0, 1)
        upper = Fraction(1, 1)
        feasible = True
        for competitor_index, competitor in enumerate(normalized):
            if competitor_index == point_index:
                continue
            difference_1 = point[0] - competitor[0]
            difference_2 = point[1] - competitor[1]
            slope = difference_1 - difference_2
            intercept = difference_2
            if slope == 0:
                if intercept < 0:
                    feasible = False
                    break
                continue
            boundary = Fraction(
                -intercept.numerator * slope.denominator,
                intercept.denominator * slope.numerator,
            )
            if slope > 0:
                if boundary > lower:
                    lower = boundary
            elif boundary < upper:
                upper = boundary
            if lower > upper:
                feasible = False
                break
        flags.append(feasible and upper >= Fraction(0, 1) and lower <= Fraction(1, 1))
    return tuple(flags)


def condition_grid(query_budget: int) -> tuple[float, ...]:
    """Return an endpoint-inclusive deterministic unit-interval query grid."""

    if query_budget < 2:
        raise ValueError("query budget must be at least two")
    return tuple(float(value) for value in np.linspace(0.0, 1.0, query_budget))


def normalized_objective_vectors(
    solutions: Sequence[ParetoSolution],
    *,
    ideal_point: tuple[int, int],
) -> tuple[tuple[float, float], ...]:
    ideal_1, ideal_2 = ideal_point
    if ideal_1 <= 0 or ideal_2 <= 0:
        raise ValueError("ideal objectives must be positive")
    return tuple(
        (
            solution.objective_1 / ideal_1,
            solution.objective_2 / ideal_2,
        )
        for solution in solutions
    )


def hypervolume_2d(
    solutions: Sequence[ParetoSolution],
    *,
    ideal_point: tuple[int, int],
) -> float:
    """Compute normalized dominated hypervolume against reference point (0, 0)."""

    canonical = nondominated_solutions(solutions)
    if not canonical:
        return 0.0
    points = normalized_objective_vectors(canonical, ideal_point=ideal_point)
    previous_x = 0.0
    hypervolume = 0.0
    for objective_1, objective_2 in points:
        if objective_1 < previous_x - 1e-12:
            raise RuntimeError("nondominated points are not ordered by objective 1")
        if not (-1e-12 <= objective_1 <= 1.0 + 1e-12):
            raise ValueError("candidate objective 1 exceeds the supplied ideal point")
        if not (-1e-12 <= objective_2 <= 1.0 + 1e-12):
            raise ValueError("candidate objective 2 exceeds the supplied ideal point")
        hypervolume += max(0.0, objective_1 - previous_x) * max(0.0, objective_2)
        previous_x = max(previous_x, objective_1)
    return hypervolume


def hypervolume_ratio(
    approximation: Sequence[ParetoSolution],
    exact_frontier: ParetoFrontier,
) -> float:
    ideal = exact_frontier.ideal_point
    exact_hypervolume = hypervolume_2d(exact_frontier.solutions, ideal_point=ideal)
    if exact_hypervolume <= 0.0:
        raise RuntimeError("exact frontier must dominate positive hypervolume")
    ratio = hypervolume_2d(approximation, ideal_point=ideal) / exact_hypervolume
    if ratio > 1.0 + 1e-10:
        raise RuntimeError("an approximation cannot dominate more volume than the exact frontier")
    return min(1.0, max(0.0, ratio))


def additive_epsilon_indicator(
    approximation: Sequence[ParetoSolution],
    exact_frontier: ParetoFrontier,
) -> float:
    """Return the normalized unary additive epsilon indicator for maximization."""

    if not approximation:
        return 1.0
    ideal = exact_frontier.ideal_point
    approximated = normalized_objective_vectors(
        nondominated_solutions(approximation),
        ideal_point=ideal,
    )
    exact = normalized_objective_vectors(exact_frontier.solutions, ideal_point=ideal)
    worst = 0.0
    for reference in exact:
        best = min(
            max(reference[0] - candidate[0], reference[1] - candidate[1])
            for candidate in approximated
        )
        worst = max(worst, best)
    return max(0.0, worst)


def inverted_generational_distance_plus(
    approximation: Sequence[ParetoSolution],
    exact_frontier: ParetoFrontier,
) -> float:
    """Compute normalized IGD+ from the exact front to the approximation."""

    if not approximation:
        return math.sqrt(2.0)
    ideal = exact_frontier.ideal_point
    approximated = normalized_objective_vectors(
        nondominated_solutions(approximation),
        ideal_point=ideal,
    )
    exact = normalized_objective_vectors(exact_frontier.solutions, ideal_point=ideal)
    distances: list[float] = []
    for reference in exact:
        distances.append(
            min(
                math.hypot(
                    max(0.0, reference[0] - candidate[0]),
                    max(0.0, reference[1] - candidate[1]),
                )
                for candidate in approximated
            )
        )
    return float(np.mean(np.asarray(distances, dtype=float)))


def exact_objective_coverage(
    approximation: Sequence[ParetoSolution],
    exact_frontier: ParetoFrontier,
) -> float:
    approximated_vectors = {solution.objectives for solution in approximation}
    exact_vectors = set(exact_frontier.objective_vectors)
    return len(approximated_vectors & exact_vectors) / len(exact_vectors)


def unsupported_objective_coverage(
    approximation: Sequence[ParetoSolution],
    exact_frontier: ParetoFrontier,
) -> float | None:
    flags = supported_solution_flags(exact_frontier)
    unsupported = {
        solution.objectives
        for solution, supported in zip(exact_frontier.solutions, flags, strict=True)
        if not supported
    }
    if not unsupported:
        return None
    approximated_vectors = {solution.objectives for solution in approximation}
    return len(approximated_vectors & unsupported) / len(unsupported)


def extreme_recovery(
    approximation: Sequence[ParetoSolution],
    exact_frontier: ParetoFrontier,
) -> float:
    if not approximation:
        return 0.0
    ideal_1, ideal_2 = exact_frontier.ideal_point
    recovered_1 = any(solution.objective_1 == ideal_1 for solution in approximation)
    recovered_2 = any(solution.objective_2 == ideal_2 for solution in approximation)
    return 0.5 * float(recovered_1) + 0.5 * float(recovered_2)
