"""Exact bi-objective 0-1 knapsack domain, Pareto dynamic programming, and audits."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MultiObjectiveKnapsackInstance:
    """A positive-integer bi-objective 0-1 knapsack instance."""

    weights: tuple[int, ...]
    profits_1: tuple[int, ...]
    profits_2: tuple[int, ...]
    capacity: int
    instance_id: str = "instance"
    regime: str = "unspecified"
    seed: int = 0

    def __post_init__(self) -> None:
        item_count = len(self.weights)
        if item_count == 0:
            raise ValueError("an instance must contain at least one item")
        if len(self.profits_1) != item_count or len(self.profits_2) != item_count:
            raise ValueError("weights and both profit vectors must be aligned")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.weights):
            raise ValueError("weights must contain integers")
        if any(isinstance(value, bool) or value <= 0 for value in self.weights):
            raise ValueError("weights must be positive")
        for profits in (self.profits_1, self.profits_2):
            if any(isinstance(value, bool) or not isinstance(value, int) for value in profits):
                raise ValueError("profits must contain integers")
            if any(value <= 0 for value in profits):
                raise ValueError("profits must be positive")
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int):
            raise ValueError("capacity must be an integer")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if min(self.weights) > self.capacity:
            raise ValueError("capacity must admit at least one item")
        if self.capacity >= sum(self.weights):
            raise ValueError("capacity must leave at least one item potentially excluded")
        if not self.instance_id:
            raise ValueError("instance_id must be nonempty")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")

    @property
    def item_count(self) -> int:
        return len(self.weights)

    @property
    def total_weight(self) -> int:
        return sum(self.weights)

    @property
    def total_profit_1(self) -> int:
        return sum(self.profits_1)

    @property
    def total_profit_2(self) -> int:
        return sum(self.profits_2)

    def to_dict(self) -> dict[str, object]:
        return {
            "weights": list(self.weights),
            "profits_1": list(self.profits_1),
            "profits_2": list(self.profits_2),
            "capacity": self.capacity,
            "instance_id": self.instance_id,
            "regime": self.regime,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> MultiObjectiveKnapsackInstance:
        weights = _integer_list(payload.get("weights"), "weights")
        profits_1 = _integer_list(payload.get("profits_1"), "profits_1")
        profits_2 = _integer_list(payload.get("profits_2"), "profits_2")
        capacity = payload.get("capacity")
        seed = payload.get("seed", 0)
        instance_id = payload.get("instance_id", "instance")
        regime = payload.get("regime", "unspecified")
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise ValueError("instance capacity must be an integer")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("instance seed must be an integer")
        if not isinstance(instance_id, str) or not isinstance(regime, str):
            raise ValueError("instance identifiers must be strings")
        return cls(
            tuple(weights),
            tuple(profits_1),
            tuple(profits_2),
            capacity,
            instance_id=instance_id,
            regime=regime,
            seed=seed,
        )


def _integer_list(value: object, name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"instance {name} must be a JSON array")
    if not all(isinstance(entry, int) and not isinstance(entry, bool) for entry in value):
        raise ValueError(f"instance {name} must contain integers")
    return value


@dataclass(frozen=True, slots=True)
class ParetoSolution:
    """One feasible binary selection and its two objective values."""

    selection: tuple[int, ...]
    total_weight: int
    objective_1: int
    objective_2: int

    @property
    def objectives(self) -> tuple[int, int]:
        return self.objective_1, self.objective_2

    @property
    def selected_indices(self) -> tuple[int, ...]:
        return tuple(index for index, chosen in enumerate(self.selection) if chosen)

    def to_dict(self) -> dict[str, object]:
        return {
            "selection": list(self.selection),
            "selected_indices": list(self.selected_indices),
            "total_weight": self.total_weight,
            "objective_1": self.objective_1,
            "objective_2": self.objective_2,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ParetoSolution:
        selection = _integer_list(payload.get("selection"), "selection")
        total_weight = payload.get("total_weight")
        objective_1 = payload.get("objective_1")
        objective_2 = payload.get("objective_2")
        numeric = (total_weight, objective_1, objective_2)
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in numeric):
            raise ValueError("solution totals and objectives must be integers")
        return cls(tuple(selection), int(total_weight), int(objective_1), int(objective_2))


@dataclass(frozen=True, slots=True)
class ParetoFrontier:
    """Canonical objective-sorted set of mutually nondominated solutions."""

    solutions: tuple[ParetoSolution, ...]

    def __post_init__(self) -> None:
        if not self.solutions:
            raise ValueError("a Pareto frontier must contain at least one solution")
        canonical = nondominated_solutions(self.solutions)
        if canonical != self.solutions:
            raise ValueError("frontier solutions must be canonical, unique, and nondominated")

    @property
    def objective_vectors(self) -> tuple[tuple[int, int], ...]:
        return tuple(solution.objectives for solution in self.solutions)

    @property
    def ideal_point(self) -> tuple[int, int]:
        return (
            max(solution.objective_1 for solution in self.solutions),
            max(solution.objective_2 for solution in self.solutions),
        )

    @property
    def nadir_point(self) -> tuple[int, int]:
        return (
            min(solution.objective_1 for solution in self.solutions),
            min(solution.objective_2 for solution in self.solutions),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "solutions": [solution.to_dict() for solution in self.solutions],
            "objective_vectors": [list(vector) for vector in self.objective_vectors],
            "ideal_point": list(self.ideal_point),
            "nadir_point": list(self.nadir_point),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ParetoFrontier:
        raw_solutions = payload.get("solutions")
        if not isinstance(raw_solutions, list):
            raise ValueError("frontier solutions must be a JSON array")
        solutions: list[ParetoSolution] = []
        for raw_solution in raw_solutions:
            if not isinstance(raw_solution, dict):
                raise ValueError("frontier solution entries must be JSON objects")
            solutions.append(ParetoSolution.from_dict(raw_solution))
        return cls(tuple(solutions))


@dataclass(frozen=True, slots=True)
class ParetoDynamicProgrammingTrace:
    """Compact diagnostics for the exact capacity-indexed Pareto dynamic program."""

    frontier_sizes: np.ndarray

    def __post_init__(self) -> None:
        if self.frontier_sizes.ndim != 2:
            raise ValueError("frontier-size trace must be a matrix")
        if self.frontier_sizes.dtype.kind not in {"i", "u"}:
            raise ValueError("frontier-size trace must contain integers")
        if np.any(self.frontier_sizes <= 0):
            raise ValueError("every capacity frontier must contain the zero solution")

    @property
    def item_count(self) -> int:
        return int(self.frontier_sizes.shape[0]) - 1

    @property
    def capacity(self) -> int:
        return int(self.frontier_sizes.shape[1]) - 1

    @property
    def maximum_frontier_size(self) -> int:
        return int(np.max(self.frontier_sizes))


@dataclass(frozen=True, slots=True)
class SolutionAudit:
    binary: bool
    feasible: bool
    total_weight: int
    objective_1: int
    objective_2: int
    reported_objectives_consistent: bool
    nondominated: bool | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _PartialSolution:
    selection: tuple[int, ...]
    total_weight: int
    objective_1: int
    objective_2: int


def _append_skip(solution: _PartialSolution) -> _PartialSolution:
    return _PartialSolution(
        selection=solution.selection + (0,),
        total_weight=solution.total_weight,
        objective_1=solution.objective_1,
        objective_2=solution.objective_2,
    )


def _append_take(
    solution: _PartialSolution,
    *,
    weight: int,
    profit_1: int,
    profit_2: int,
) -> _PartialSolution:
    return _PartialSolution(
        selection=solution.selection + (1,),
        total_weight=solution.total_weight + weight,
        objective_1=solution.objective_1 + profit_1,
        objective_2=solution.objective_2 + profit_2,
    )


def _canonical_partial_frontier(
    candidates: Sequence[_PartialSolution],
) -> tuple[_PartialSolution, ...]:
    by_objective: dict[tuple[int, int], _PartialSolution] = {}
    for candidate in candidates:
        key = candidate.objective_1, candidate.objective_2
        incumbent = by_objective.get(key)
        if incumbent is None or (
            candidate.total_weight,
            candidate.selection,
        ) < (
            incumbent.total_weight,
            incumbent.selection,
        ):
            by_objective[key] = candidate

    descending = sorted(
        by_objective.values(),
        key=lambda candidate: (
            -candidate.objective_1,
            -candidate.objective_2,
            candidate.total_weight,
            candidate.selection,
        ),
    )
    kept: list[_PartialSolution] = []
    best_objective_2 = -1
    for candidate in descending:
        if candidate.objective_2 > best_objective_2:
            kept.append(candidate)
            best_objective_2 = candidate.objective_2
    return tuple(
        sorted(
            kept,
            key=lambda candidate: (
                candidate.objective_1,
                -candidate.objective_2,
                candidate.total_weight,
                candidate.selection,
            ),
        )
    )


def nondominated_solutions(
    solutions: Sequence[ParetoSolution],
) -> tuple[ParetoSolution, ...]:
    """Return canonical nondominated solutions sorted by objective 1 ascending."""

    if not solutions:
        return ()
    by_objective: dict[tuple[int, int], ParetoSolution] = {}
    for solution in solutions:
        key = solution.objectives
        incumbent = by_objective.get(key)
        if incumbent is None or (
            solution.total_weight,
            solution.selection,
        ) < (
            incumbent.total_weight,
            incumbent.selection,
        ):
            by_objective[key] = solution

    descending = sorted(
        by_objective.values(),
        key=lambda solution: (
            -solution.objective_1,
            -solution.objective_2,
            solution.total_weight,
            solution.selection,
        ),
    )
    kept: list[ParetoSolution] = []
    best_objective_2 = -1
    for solution in descending:
        if solution.objective_2 > best_objective_2:
            kept.append(solution)
            best_objective_2 = solution.objective_2
    return tuple(
        sorted(
            kept,
            key=lambda solution: (
                solution.objective_1,
                -solution.objective_2,
                solution.total_weight,
                solution.selection,
            ),
        )
    )


def solution_from_selection(
    instance: MultiObjectiveKnapsackInstance,
    selection: Sequence[int],
) -> ParetoSolution:
    """Recompute all solution totals from a binary selection."""

    if len(selection) != instance.item_count:
        raise ValueError("selection length does not match item count")
    normalized: list[int] = []
    for chosen in selection:
        if (
            isinstance(chosen, bool)
            or isinstance(chosen, (int, np.integer))
            and int(chosen) in {0, 1}
        ):
            normalized.append(int(chosen))
        else:
            raise ValueError("selection entries must be binary")
    total_weight = sum(
        weight * chosen for weight, chosen in zip(instance.weights, normalized, strict=True)
    )
    objective_1 = sum(
        profit * chosen for profit, chosen in zip(instance.profits_1, normalized, strict=True)
    )
    objective_2 = sum(
        profit * chosen for profit, chosen in zip(instance.profits_2, normalized, strict=True)
    )
    return ParetoSolution(tuple(normalized), total_weight, objective_1, objective_2)


def solve_pareto_dynamic_programming(
    instance: MultiObjectiveKnapsackInstance,
    *,
    return_trace: bool = False,
) -> ParetoFrontier | tuple[ParetoFrontier, ParetoDynamicProgrammingTrace]:
    """Enumerate the exact Pareto frontier with capacity-indexed dominance pruning."""

    zero = _PartialSolution((), 0, 0, 0)
    frontiers: list[tuple[_PartialSolution, ...]] = [(zero,) for _ in range(instance.capacity + 1)]
    sizes = np.ones((instance.item_count + 1, instance.capacity + 1), dtype=np.int64)

    for item_index, (weight, profit_1, profit_2) in enumerate(
        zip(instance.weights, instance.profits_1, instance.profits_2, strict=True),
        start=1,
    ):
        next_frontiers: list[tuple[_PartialSolution, ...]] = []
        for capacity in range(instance.capacity + 1):
            candidates = [_append_skip(solution) for solution in frontiers[capacity]]
            if weight <= capacity:
                candidates.extend(
                    _append_take(
                        solution,
                        weight=weight,
                        profit_1=profit_1,
                        profit_2=profit_2,
                    )
                    for solution in frontiers[capacity - weight]
                )
            canonical = _canonical_partial_frontier(candidates)
            next_frontiers.append(canonical)
            sizes[item_index, capacity] = len(canonical)
        frontiers = next_frontiers

    final_solutions = tuple(
        ParetoSolution(
            selection=solution.selection,
            total_weight=solution.total_weight,
            objective_1=solution.objective_1,
            objective_2=solution.objective_2,
        )
        for solution in frontiers[instance.capacity]
    )
    frontier = ParetoFrontier(nondominated_solutions(final_solutions))
    for solution in frontier.solutions:
        audit = audit_solution(instance, solution.selection, frontier=frontier)
        if not (
            audit.binary
            and audit.feasible
            and audit.reported_objectives_consistent
            and audit.nondominated is True
        ):
            raise RuntimeError("dynamic-programming frontier failed its own consistency audit")
    if return_trace:
        return frontier, ParetoDynamicProgrammingTrace(frontier_sizes=sizes)
    return frontier


def solve_pareto_brute_force(
    instance: MultiObjectiveKnapsackInstance,
    *,
    maximum_items: int = 22,
) -> ParetoFrontier:
    """Enumerate all subsets for independent verification of small instances."""

    if instance.item_count > maximum_items:
        raise ValueError("instance exceeds brute-force verification limit")
    feasible: list[ParetoSolution] = []
    for selection in itertools.product((0, 1), repeat=instance.item_count):
        candidate = solution_from_selection(instance, selection)
        if candidate.total_weight <= instance.capacity:
            feasible.append(candidate)
    return ParetoFrontier(nondominated_solutions(feasible))


def audit_solution(
    instance: MultiObjectiveKnapsackInstance,
    selection: Sequence[int],
    *,
    reported_objectives: tuple[int, int] | None = None,
    frontier: ParetoFrontier | None = None,
) -> SolutionAudit:
    """Audit feasibility, objective consistency, and optional Pareto membership."""

    binary = len(selection) == instance.item_count and all(
        isinstance(chosen, (bool, int, np.integer)) and int(chosen) in {0, 1}
        for chosen in selection
    )
    if not binary:
        return SolutionAudit(
            binary=False,
            feasible=False,
            total_weight=-1,
            objective_1=-1,
            objective_2=-1,
            reported_objectives_consistent=False,
            nondominated=False if frontier is not None else None,
        )
    solution = solution_from_selection(instance, selection)
    consistent = reported_objectives is None or reported_objectives == solution.objectives
    nondominated: bool | None = None
    if frontier is not None:
        nondominated = solution.objectives in set(frontier.objective_vectors)
    return SolutionAudit(
        binary=True,
        feasible=solution.total_weight <= instance.capacity,
        total_weight=solution.total_weight,
        objective_1=solution.objective_1,
        objective_2=solution.objective_2,
        reported_objectives_consistent=consistent,
        nondominated=nondominated,
    )
