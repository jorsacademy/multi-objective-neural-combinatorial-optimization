from __future__ import annotations

import pytest

from mopareto.dataset import generate_dataset
from mopareto.decoding import construct_approximation_set, density_advice, policy_advice
from mopareto.domain import audit_solution, nondominated_solutions
from mopareto.models import ConditionalParetoPolicy, ConditionalPolicyConfig


def test_policy_and_density_decoding_are_feasible() -> None:
    dataset = generate_dataset(
        count=1,
        item_counts=(10,),
        regimes=("strongly_conflicting",),
        seed=17,
    )
    instance = dataset.instances[0]
    exact = dataset.frontiers[0]
    for mode in ("weighted_sum", "epsilon_constraint"):
        model = ConditionalParetoPolicy(
            mode,
            ConditionalPolicyConfig(hidden_dim=8, hidden_layers=1),
        )
        approximation = construct_approximation_set(
            instance=instance,
            exact_frontier=exact,
            query_budget=5,
            model=model,
        )
        assert approximation.frontier == nondominated_solutions(approximation.frontier)
        assert approximation.unique_candidate_count >= 1
        for advice in approximation.advices:
            audit = audit_solution(instance, advice.candidate.selection)
            assert audit.binary and audit.feasible
        density = construct_approximation_set(
            instance=instance,
            exact_frontier=exact,
            query_budget=5,
            density_mode=mode,
        )
        assert all(
            audit_solution(instance, advice.candidate.selection).feasible
            for advice in density.advices
        )


def test_epsilon_advice_reports_condition_satisfaction() -> None:
    dataset = generate_dataset(
        count=1,
        item_counts=(8,),
        regimes=("independent",),
        seed=22,
    )
    instance, exact = dataset.instances[0], dataset.frontiers[0]
    model = ConditionalParetoPolicy("epsilon_constraint")
    advice = policy_advice(model, instance, exact, 0.0)
    assert advice.condition_satisfied is True
    density = density_advice(instance, exact, 0.0, epsilon_mode=True)
    assert density.condition_satisfied is True


def test_approximation_argument_validation() -> None:
    dataset = generate_dataset(
        count=1,
        item_counts=(5,),
        regimes=("independent",),
        seed=2,
    )
    instance, exact = dataset.instances[0], dataset.frontiers[0]
    with pytest.raises(ValueError, match="exactly one"):
        construct_approximation_set(
            instance=instance,
            exact_frontier=exact,
            query_budget=3,
        )
    with pytest.raises(ValueError, match="exactly one"):
        construct_approximation_set(
            instance=instance,
            exact_frontier=exact,
            query_budget=3,
            model=ConditionalParetoPolicy("weighted_sum"),
            density_mode="weighted_sum",
        )
    with pytest.raises(ValueError, match="density_mode"):
        construct_approximation_set(
            instance=instance,
            exact_frontier=exact,
            query_budget=3,
            density_mode="bad",
        )
