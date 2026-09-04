"""Verification-first neural approximation of bi-objective knapsack Pareto fronts."""

from mopareto.dataset import KnapsackDataset, generate_dataset, generate_instance, load_dataset
from mopareto.domain import (
    MultiObjectiveKnapsackInstance,
    ParetoFrontier,
    ParetoSolution,
    solve_pareto_brute_force,
    solve_pareto_dynamic_programming,
)
from mopareto.models import ConditionalParetoPolicy, ConditionalPolicyConfig

__all__ = [
    "ConditionalParetoPolicy",
    "ConditionalPolicyConfig",
    "KnapsackDataset",
    "MultiObjectiveKnapsackInstance",
    "ParetoFrontier",
    "ParetoSolution",
    "generate_dataset",
    "generate_instance",
    "load_dataset",
    "solve_pareto_brute_force",
    "solve_pareto_dynamic_programming",
]

__version__ = "0.1.0"
