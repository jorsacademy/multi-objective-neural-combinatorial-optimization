"""Deterministic feasible decoding and fixed-budget Pareto-set construction."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch

from mopareto.domain import (
    MultiObjectiveKnapsackInstance,
    ParetoFrontier,
    ParetoSolution,
    audit_solution,
    nondominated_solutions,
    solution_from_selection,
)
from mopareto.models import ConditionalParetoPolicy, policy_logits
from mopareto.pareto import condition_grid


@dataclass(frozen=True, slots=True)
class ConditionalAdvice:
    source: str
    condition: float
    scores: tuple[float, ...]
    candidate: ParetoSolution
    condition_satisfied: bool | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidate"] = self.candidate.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class ApproximationSet:
    method: str
    query_budget: int
    advices: tuple[ConditionalAdvice, ...]
    frontier: tuple[ParetoSolution, ...]
    runtime_seconds: float

    @property
    def unique_candidate_count(self) -> int:
        return len({advice.candidate.objectives for advice in self.advices})

    @property
    def condition_satisfaction_rate(self) -> float | None:
        observed = [
            advice.condition_satisfied
            for advice in self.advices
            if advice.condition_satisfied is not None
        ]
        if not observed:
            return None
        return float(np.mean(np.asarray(observed, dtype=float)))

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "query_budget": self.query_budget,
            "unique_candidate_count": self.unique_candidate_count,
            "condition_satisfaction_rate": self.condition_satisfaction_rate,
            "runtime_seconds": self.runtime_seconds,
            "advices": [advice.to_dict() for advice in self.advices],
            "frontier": [solution.to_dict() for solution in self.frontier],
        }


Criterion = Callable[[ParetoSolution], tuple[float, ...]]


def _weighted_criterion(
    ideal_point: tuple[int, int],
    condition: float,
) -> Criterion:
    ideal_1, ideal_2 = ideal_point

    def criterion(solution: ParetoSolution) -> tuple[float, ...]:
        normalized_1 = solution.objective_1 / ideal_1
        normalized_2 = solution.objective_2 / ideal_2
        utility = condition * normalized_1 + (1.0 - condition) * normalized_2
        return utility, min(normalized_1, normalized_2), normalized_1 + normalized_2

    return criterion


def _epsilon_criterion(
    ideal_point: tuple[int, int],
    target: float,
) -> Criterion:
    ideal_1, ideal_2 = ideal_point

    def criterion(solution: ParetoSolution) -> tuple[float, ...]:
        normalized_1 = solution.objective_1 / ideal_1
        normalized_2 = solution.objective_2 / ideal_2
        satisfied = normalized_2 + 1e-12 >= target
        if satisfied:
            return 1.0, normalized_1, normalized_2
        return 0.0, normalized_2, normalized_1

    return criterion


def _is_better(candidate: ParetoSolution, incumbent: ParetoSolution, criterion: Criterion) -> bool:
    candidate_key = criterion(candidate)
    incumbent_key = criterion(incumbent)
    if candidate_key != incumbent_key:
        return candidate_key > incumbent_key
    return (candidate.total_weight, candidate.selection) < (
        incumbent.total_weight,
        incumbent.selection,
    )


def _greedy_from_order(
    instance: MultiObjectiveKnapsackInstance,
    order: Sequence[int],
    *,
    allowed: set[int] | None = None,
) -> ParetoSolution:
    selection = [0] * instance.item_count
    remaining = instance.capacity
    for index in order:
        if allowed is not None and index not in allowed:
            continue
        if instance.weights[index] <= remaining:
            selection[index] = 1
            remaining -= instance.weights[index]
    return solution_from_selection(instance, selection)


def _rank_values(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=float), kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    if len(values) <= 1:
        return np.ones(len(values), dtype=float)
    return ranks / float(len(values) - 1)


def _model_orders(
    instance: MultiObjectiveKnapsackInstance,
    scores: Sequence[float],
    condition: float,
) -> tuple[tuple[int, ...], ...]:
    idealized_utility = [
        condition * (profit_1 / instance.total_profit_1)
        + (1.0 - condition) * (profit_2 / instance.total_profit_2)
        for profit_1, profit_2 in zip(instance.profits_1, instance.profits_2, strict=True)
    ]
    score_ranks = _rank_values(scores)
    utility_density = [
        utility / weight
        for utility, weight in zip(idealized_utility, instance.weights, strict=True)
    ]
    utility_ranks = _rank_values(utility_density)
    blended = 0.8 * score_ranks + 0.2 * utility_ranks
    score_order = tuple(
        sorted(
            range(instance.item_count),
            key=lambda index: (
                float(scores[index]),
                utility_density[index],
                instance.profits_1[index] + instance.profits_2[index],
                -index,
            ),
            reverse=True,
        )
    )
    probability_density_order = tuple(
        sorted(
            range(instance.item_count),
            key=lambda index: (
                1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, float(scores[index])))))
                / instance.weights[index],
                float(scores[index]),
                utility_density[index],
                -index,
            ),
            reverse=True,
        )
    )
    blended_order = tuple(
        sorted(
            range(instance.item_count),
            key=lambda index: (
                float(blended[index]),
                float(scores[index]),
                utility_density[index],
                -index,
            ),
            reverse=True,
        )
    )
    return score_order, probability_density_order, blended_order


def _density_orders(
    instance: MultiObjectiveKnapsackInstance,
    condition: float,
) -> tuple[tuple[int, ...], ...]:
    combined = [
        condition * (profit_1 / instance.total_profit_1)
        + (1.0 - condition) * (profit_2 / instance.total_profit_2)
        for profit_1, profit_2 in zip(instance.profits_1, instance.profits_2, strict=True)
    ]

    def ordered(key_values: Sequence[float]) -> tuple[int, ...]:
        return tuple(
            sorted(
                range(instance.item_count),
                key=lambda index: (
                    key_values[index] / instance.weights[index],
                    key_values[index],
                    -instance.weights[index],
                    -index,
                ),
                reverse=True,
            )
        )

    return (
        ordered(combined),
        ordered([float(value) for value in instance.profits_1]),
        ordered([float(value) for value in instance.profits_2]),
    )


def _single_swap_improvement(
    instance: MultiObjectiveKnapsackInstance,
    initial: ParetoSolution,
    criterion: Criterion,
) -> ParetoSolution:
    incumbent = initial
    for _ in range(max(1, 2 * instance.item_count)):
        best = incumbent
        selected = [index for index, chosen in enumerate(incumbent.selection) if chosen]
        unselected = [index for index, chosen in enumerate(incumbent.selection) if not chosen]
        for add_index in unselected:
            if incumbent.total_weight + instance.weights[add_index] <= instance.capacity:
                selection = list(incumbent.selection)
                selection[add_index] = 1
                candidate = solution_from_selection(instance, selection)
                if _is_better(candidate, best, criterion):
                    best = candidate
        for remove_index in selected:
            for add_index in unselected:
                next_weight = (
                    incumbent.total_weight
                    - instance.weights[remove_index]
                    + instance.weights[add_index]
                )
                if next_weight > instance.capacity:
                    continue
                selection = list(incumbent.selection)
                selection[remove_index] = 0
                selection[add_index] = 1
                candidate = solution_from_selection(instance, selection)
                if _is_better(candidate, best, criterion):
                    best = candidate
        if best == incumbent:
            break
        incumbent = best
    return incumbent


def _best_decoded_candidate(
    instance: MultiObjectiveKnapsackInstance,
    orders: Sequence[Sequence[int]],
    criterion: Criterion,
    *,
    scores: Sequence[float] | None = None,
) -> ParetoSolution:
    candidates: list[ParetoSolution] = []
    positive = (
        {index for index, score in enumerate(scores) if float(score) >= 0.0}
        if scores is not None
        else None
    )
    for order in orders:
        candidates.append(_greedy_from_order(instance, order))
        if positive is not None:
            candidates.append(_greedy_from_order(instance, order, allowed=positive))
    improved = [
        _single_swap_improvement(instance, candidate, criterion)
        for candidate in candidates
    ]
    incumbent = improved[0]
    for candidate in improved[1:]:
        if _is_better(candidate, incumbent, criterion):
            incumbent = candidate
    return incumbent


def policy_advice(
    model: ConditionalParetoPolicy,
    instance: MultiObjectiveKnapsackInstance,
    exact_frontier: ParetoFrontier,
    condition: float,
) -> ConditionalAdvice:
    with torch.no_grad():
        logits = policy_logits(model, instance, condition)
    scores = tuple(float(value) for value in logits.detach().cpu().double().tolist())
    if len(scores) != instance.item_count or not all(math.isfinite(score) for score in scores):
        raise RuntimeError("policy advice contains malformed or non-finite scores")
    if model.mode == "weighted_sum":
        criterion = _weighted_criterion(exact_frontier.ideal_point, condition)
        decoding_condition = condition
    else:
        criterion = _epsilon_criterion(exact_frontier.ideal_point, condition)
        decoding_condition = 1.0 - condition
    candidate = _best_decoded_candidate(
        instance,
        _model_orders(instance, scores, decoding_condition),
        criterion,
        scores=scores,
    )
    audit = audit_solution(instance, candidate.selection, reported_objectives=candidate.objectives)
    if not (audit.binary and audit.feasible and audit.reported_objectives_consistent):
        raise RuntimeError("decoded policy candidate failed feasibility audit")
    satisfied = None
    if model.mode == "epsilon_constraint":
        satisfied = candidate.objective_2 / exact_frontier.ideal_point[1] + 1e-12 >= condition
    return ConditionalAdvice(
        source=model.mode,
        condition=condition,
        scores=scores,
        candidate=candidate,
        condition_satisfied=satisfied,
    )


def density_advice(
    instance: MultiObjectiveKnapsackInstance,
    exact_frontier: ParetoFrontier,
    condition: float,
    *,
    epsilon_mode: bool = False,
) -> ConditionalAdvice:
    criterion = (
        _epsilon_criterion(exact_frontier.ideal_point, condition)
        if epsilon_mode
        else _weighted_criterion(exact_frontier.ideal_point, condition)
    )
    decoding_condition = 1.0 - condition if epsilon_mode else condition
    orders = _density_orders(instance, decoding_condition)
    candidate = _best_decoded_candidate(instance, orders, criterion)
    audit = audit_solution(instance, candidate.selection, reported_objectives=candidate.objectives)
    if not (audit.binary and audit.feasible and audit.reported_objectives_consistent):
        raise RuntimeError("density candidate failed feasibility audit")
    scores = tuple(
        decoding_condition * (profit_1 / instance.total_profit_1)
        + (1.0 - decoding_condition) * (profit_2 / instance.total_profit_2)
        for profit_1, profit_2 in zip(instance.profits_1, instance.profits_2, strict=True)
    )
    satisfied = None
    if epsilon_mode:
        satisfied = candidate.objective_2 / exact_frontier.ideal_point[1] + 1e-12 >= condition
    return ConditionalAdvice(
        source="epsilon_density" if epsilon_mode else "weighted_density",
        condition=condition,
        scores=scores,
        candidate=candidate,
        condition_satisfied=satisfied,
    )


def construct_approximation_set(
    *,
    instance: MultiObjectiveKnapsackInstance,
    exact_frontier: ParetoFrontier,
    query_budget: int,
    model: ConditionalParetoPolicy | None = None,
    density_mode: str | None = None,
) -> ApproximationSet:
    """Query one policy or density baseline on a fixed grid and filter nondominated outputs."""

    if (model is None) == (density_mode is None):
        raise ValueError("provide exactly one of model or density_mode")
    if density_mode not in {None, "weighted_sum", "epsilon_constraint"}:
        raise ValueError("density_mode must be weighted_sum or epsilon_constraint")
    start = time.perf_counter()
    advices: list[ConditionalAdvice] = []
    for condition in condition_grid(query_budget):
        if model is not None:
            advice = policy_advice(model, instance, exact_frontier, condition)
        else:
            advice = density_advice(
                instance,
                exact_frontier,
                condition,
                epsilon_mode=density_mode == "epsilon_constraint",
            )
        advices.append(advice)
    frontier = nondominated_solutions([advice.candidate for advice in advices])
    if not frontier:
        raise RuntimeError("fixed-budget construction produced no feasible candidate")
    method = model.mode if model is not None else f"{density_mode}_density"
    return ApproximationSet(
        method=method,
        query_budget=query_budget,
        advices=tuple(advices),
        frontier=frontier,
        runtime_seconds=time.perf_counter() - start,
    )
