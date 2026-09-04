"""Fixed-budget Pareto quality, coverage, and diversity evaluation."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import numpy as np

from mopareto.dataset import KnapsackDataset
from mopareto.decoding import ApproximationSet, construct_approximation_set
from mopareto.domain import audit_solution, nondominated_solutions
from mopareto.models import ConditionalParetoPolicy
from mopareto.pareto import (
    additive_epsilon_indicator,
    exact_objective_coverage,
    extreme_recovery,
    hypervolume_ratio,
    inverted_generational_distance_plus,
    supported_solution_flags,
    unsupported_objective_coverage,
)
from mopareto.utils import write_json


@dataclass(frozen=True, slots=True)
class OracleFrontierMetrics:
    scenario: str
    instance_count: int
    mean_frontier_size: float
    median_frontier_size: float
    maximum_frontier_size: int
    mean_supported_fraction: float
    instances_with_unsupported_points: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FrontierMetrics:
    scenario: str
    method: str
    instance_count: int
    query_budget: int
    mean_hypervolume_ratio: float
    median_hypervolume_ratio: float
    p10_hypervolume_ratio: float
    mean_additive_epsilon: float
    mean_igd_plus: float
    mean_exact_objective_coverage: float
    mean_unsupported_objective_coverage: float | None
    mean_extreme_recovery: float
    mean_approximation_frontier_size: float
    mean_unique_candidate_count: float
    mean_nondominated_yield: float
    mean_feasibility_rate: float
    mean_condition_satisfaction_rate: float | None
    mean_runtime_seconds: float
    mean_hypervolume_gain_vs_weighted_density: float
    hypervolume_gain_ci_low: float
    hypervolume_gain_ci_high: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    oracle_metrics: OracleFrontierMetrics
    frontier_rows: tuple[FrontierMetrics, ...]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "oracle_metrics": self.oracle_metrics.to_dict(),
            "frontier_rows": [row.to_dict() for row in self.frontier_rows],
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class _InstanceMetrics:
    hypervolume_ratio: float
    additive_epsilon: float
    igd_plus: float
    exact_coverage: float
    unsupported_coverage: float | None
    extreme_recovery: float
    approximation_size: int
    unique_candidate_count: int
    nondominated_yield: float
    feasibility_rate: float
    condition_satisfaction_rate: float | None
    runtime_seconds: float


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    return float(np.percentile(np.asarray(values, dtype=float), q, method="linear"))


def _bootstrap_mean_interval(
    values: list[float],
    *,
    seed: int,
    draws: int,
) -> tuple[float, float]:
    if not values or draws <= 0:
        raise ValueError("bootstrap requires values and a positive draw count")
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for draw in range(draws):
        sample = rng.integers(0, data.size, size=data.size)
        means[draw] = float(np.mean(data[sample]))
    return (
        float(np.percentile(means, 2.5, method="linear")),
        float(np.percentile(means, 97.5, method="linear")),
    )


def _mean_optional(values: list[float | None]) -> float | None:
    observed = [value for value in values if value is not None]
    if not observed:
        return None
    return float(np.mean(np.asarray(observed, dtype=float)))


def _audit_approximation(
    dataset: KnapsackDataset,
    instance_index: int,
    approximation: ApproximationSet,
) -> float:
    instance = dataset.instances[instance_index]
    valid: list[float] = []
    for advice in approximation.advices:
        audit = audit_solution(
            instance,
            advice.candidate.selection,
            reported_objectives=advice.candidate.objectives,
        )
        valid.append(float(audit.binary and audit.feasible and audit.reported_objectives_consistent))
    if tuple(approximation.frontier) != nondominated_solutions(approximation.frontier):
        raise RuntimeError("reported approximation frontier is not canonical and nondominated")
    return float(np.mean(np.asarray(valid, dtype=float)))


def _instance_metrics(
    dataset: KnapsackDataset,
    instance_index: int,
    approximation: ApproximationSet,
) -> _InstanceMetrics:
    exact = dataset.frontiers[instance_index]
    feasibility = _audit_approximation(dataset, instance_index, approximation)
    return _InstanceMetrics(
        hypervolume_ratio=hypervolume_ratio(approximation.frontier, exact),
        additive_epsilon=additive_epsilon_indicator(approximation.frontier, exact),
        igd_plus=inverted_generational_distance_plus(approximation.frontier, exact),
        exact_coverage=exact_objective_coverage(approximation.frontier, exact),
        unsupported_coverage=unsupported_objective_coverage(approximation.frontier, exact),
        extreme_recovery=extreme_recovery(approximation.frontier, exact),
        approximation_size=len(approximation.frontier),
        unique_candidate_count=approximation.unique_candidate_count,
        nondominated_yield=len(approximation.frontier) / approximation.query_budget,
        feasibility_rate=feasibility,
        condition_satisfaction_rate=approximation.condition_satisfaction_rate,
        runtime_seconds=approximation.runtime_seconds,
    )


def _summary_row(
    *,
    scenario: str,
    method: str,
    query_budget: int,
    rows: list[_InstanceMetrics],
    baseline_hypervolumes: list[float],
    bootstrap_seed: int,
    bootstrap_draws: int,
) -> FrontierMetrics:
    hypervolumes = [row.hypervolume_ratio for row in rows]
    gains = [
        hypervolume - baseline
        for hypervolume, baseline in zip(hypervolumes, baseline_hypervolumes, strict=True)
    ]
    ci_low, ci_high = _bootstrap_mean_interval(
        gains,
        seed=bootstrap_seed,
        draws=bootstrap_draws,
    )
    return FrontierMetrics(
        scenario=scenario,
        method=method,
        instance_count=len(rows),
        query_budget=query_budget,
        mean_hypervolume_ratio=float(np.mean(np.asarray(hypervolumes, dtype=float))),
        median_hypervolume_ratio=float(median(hypervolumes)),
        p10_hypervolume_ratio=_percentile(hypervolumes, 10.0),
        mean_additive_epsilon=float(
            np.mean(np.asarray([row.additive_epsilon for row in rows], dtype=float))
        ),
        mean_igd_plus=float(np.mean(np.asarray([row.igd_plus for row in rows], dtype=float))),
        mean_exact_objective_coverage=float(
            np.mean(np.asarray([row.exact_coverage for row in rows], dtype=float))
        ),
        mean_unsupported_objective_coverage=_mean_optional(
            [row.unsupported_coverage for row in rows]
        ),
        mean_extreme_recovery=float(
            np.mean(np.asarray([row.extreme_recovery for row in rows], dtype=float))
        ),
        mean_approximation_frontier_size=float(
            np.mean(np.asarray([row.approximation_size for row in rows], dtype=float))
        ),
        mean_unique_candidate_count=float(
            np.mean(np.asarray([row.unique_candidate_count for row in rows], dtype=float))
        ),
        mean_nondominated_yield=float(
            np.mean(np.asarray([row.nondominated_yield for row in rows], dtype=float))
        ),
        mean_feasibility_rate=float(
            np.mean(np.asarray([row.feasibility_rate for row in rows], dtype=float))
        ),
        mean_condition_satisfaction_rate=_mean_optional(
            [row.condition_satisfaction_rate for row in rows]
        ),
        mean_runtime_seconds=float(
            np.mean(np.asarray([row.runtime_seconds for row in rows], dtype=float))
        ),
        mean_hypervolume_gain_vs_weighted_density=float(
            np.mean(np.asarray(gains, dtype=float))
        ),
        hypervolume_gain_ci_low=ci_low,
        hypervolume_gain_ci_high=ci_high,
    )


def evaluate_models(
    weighted_policy: ConditionalParetoPolicy,
    epsilon_policy: ConditionalParetoPolicy,
    dataset: KnapsackDataset,
    *,
    scenario: str,
    query_budget: int = 21,
    bootstrap_seed: int = 0,
    bootstrap_draws: int = 500,
) -> EvaluationReport:
    """Compare two conditional policies with matched deterministic density baselines."""

    if weighted_policy.mode != "weighted_sum":
        raise ValueError("weighted_policy checkpoint has the wrong mode")
    if epsilon_policy.mode != "epsilon_constraint":
        raise ValueError("epsilon_policy checkpoint has the wrong mode")
    if not scenario:
        raise ValueError("scenario must be nonempty")
    if query_budget < 2 or bootstrap_draws <= 0:
        raise ValueError("query_budget and bootstrap_draws are invalid")

    method_rows: dict[str, list[_InstanceMetrics]] = {
        "weighted_sum_density": [],
        "epsilon_constraint_density": [],
        "weighted_sum_policy": [],
        "epsilon_constraint_policy": [],
    }
    for index, (instance, exact) in enumerate(
        zip(dataset.instances, dataset.frontiers, strict=True)
    ):
        approximations = {
            "weighted_sum_density": construct_approximation_set(
                instance=instance,
                exact_frontier=exact,
                query_budget=query_budget,
                density_mode="weighted_sum",
            ),
            "epsilon_constraint_density": construct_approximation_set(
                instance=instance,
                exact_frontier=exact,
                query_budget=query_budget,
                density_mode="epsilon_constraint",
            ),
            "weighted_sum_policy": construct_approximation_set(
                instance=instance,
                exact_frontier=exact,
                query_budget=query_budget,
                model=weighted_policy,
            ),
            "epsilon_constraint_policy": construct_approximation_set(
                instance=instance,
                exact_frontier=exact,
                query_budget=query_budget,
                model=epsilon_policy,
            ),
        }
        for method, approximation in approximations.items():
            method_rows[method].append(_instance_metrics(dataset, index, approximation))

    baseline_hypervolumes = [
        row.hypervolume_ratio for row in method_rows["weighted_sum_density"]
    ]
    frontier_rows = tuple(
        _summary_row(
            scenario=scenario,
            method=method,
            query_budget=query_budget,
            rows=rows,
            baseline_hypervolumes=baseline_hypervolumes,
            bootstrap_seed=bootstrap_seed + offset,
            bootstrap_draws=bootstrap_draws,
        )
        for offset, (method, rows) in enumerate(method_rows.items())
    )
    frontier_sizes = [len(frontier.solutions) for frontier in dataset.frontiers]
    supported_fractions = [
        float(np.mean(np.asarray(supported_solution_flags(frontier), dtype=float)))
        for frontier in dataset.frontiers
    ]
    unsupported_instances = sum(
        int(not all(supported_solution_flags(frontier)))
        for frontier in dataset.frontiers
    )
    oracle = OracleFrontierMetrics(
        scenario=scenario,
        instance_count=len(dataset.instances),
        mean_frontier_size=float(np.mean(np.asarray(frontier_sizes, dtype=float))),
        median_frontier_size=float(median(frontier_sizes)),
        maximum_frontier_size=max(frontier_sizes),
        mean_supported_fraction=float(np.mean(np.asarray(supported_fractions, dtype=float))),
        instances_with_unsupported_points=unsupported_instances,
    )
    return EvaluationReport(
        oracle_metrics=oracle,
        frontier_rows=frontier_rows,
        metadata={
            "scenario": scenario,
            "dataset_fingerprint": dataset.fingerprint,
            "dataset_regimes": list(dataset.regimes),
            "dataset_item_counts": list(dataset.item_counts),
            "query_budget": query_budget,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_draws": bootstrap_draws,
            "all_decoded_candidates_feasible": all(
                row.mean_feasibility_rate == 1.0 for row in frontier_rows
            ),
            "indicator_reference_point": [0.0, 0.0],
            "objective_normalization": "per-instance exact ideal point",
            "claims_boundary": (
                "Synthetic bi-objective 0-1 knapsack benchmark; policies approximate "
                "Pareto sets and never certify completeness or nondominance."
            ),
        },
    )


def save_report_json(report: EvaluationReport, path: str | Path) -> None:
    write_json(report.to_dict(), path)


def save_report_csv(report: EvaluationReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    oracle = report.oracle_metrics.to_dict()
    oracle["row_type"] = "oracle"
    rows.append(oracle)
    for frontier_row in report.frontier_rows:
        payload = frontier_row.to_dict()
        payload["row_type"] = "frontier"
        rows.append(payload)
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
