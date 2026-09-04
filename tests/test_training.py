from __future__ import annotations

import math

import pytest

from mopareto.dataset import KnapsackDataset
from mopareto.models import ConditionalPolicyConfig
from mopareto.training import (
    TrainingConfig,
    build_supervision_examples,
    train_policy,
)


def test_supervision_examples_cover_both_condition_families(
    tiny_training_dataset: KnapsackDataset,
) -> None:
    weighted = build_supervision_examples(
        tiny_training_dataset,
        mode="weighted_sum",
        conditions_per_instance=5,
    )
    epsilon = build_supervision_examples(
        tiny_training_dataset,
        mode="epsilon_constraint",
        conditions_per_instance=5,
    )
    assert len(weighted) == len(tiny_training_dataset.instances) * 5
    assert len(epsilon) == len(tiny_training_dataset.instances) * 5
    assert {example.condition for example in weighted}.issuperset({0.0, 1.0})
    assert {example.condition for example in epsilon}.issuperset({0.0, 1.0})


@pytest.mark.parametrize("mode", ["weighted_sum", "epsilon_constraint"])
def test_training_is_finite_and_reports_metrics(
    tiny_training_dataset: KnapsackDataset,
    tiny_validation_dataset: KnapsackDataset,
    mode: str,
) -> None:
    model, report = train_policy(
        tiny_training_dataset,
        tiny_validation_dataset,
        mode=mode,  # type: ignore[arg-type]
        model_config=ConditionalPolicyConfig(hidden_dim=8, hidden_layers=1),
        training_config=TrainingConfig(
            epochs=2,
            batch_size=8,
            conditions_per_instance=3,
            seed=9,
        ),
    )
    assert model.mode == mode
    assert len(report.epochs) == 2
    assert math.isfinite(report.best_validation_loss)
    assert 0.0 <= report.epochs[-1].validation_item_accuracy <= 1.0
    assert report.training_example_count == len(tiny_training_dataset.instances) * 3


def test_training_config_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        TrainingConfig(epochs=0)
    with pytest.raises(ValueError, match="at least two"):
        TrainingConfig(conditions_per_instance=1)
    with pytest.raises(ValueError, match="nonnegative"):
        TrainingConfig(capacity_penalty=-1.0)
