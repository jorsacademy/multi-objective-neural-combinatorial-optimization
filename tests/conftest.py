from __future__ import annotations

import pytest

from mopareto.dataset import KnapsackDataset, generate_dataset
from mopareto.domain import MultiObjectiveKnapsackInstance


@pytest.fixture
def simple_instance() -> MultiObjectiveKnapsackInstance:
    return MultiObjectiveKnapsackInstance(
        weights=(2, 3, 4, 5),
        profits_1=(8, 5, 9, 6),
        profits_2=(3, 10, 4, 8),
        capacity=7,
        instance_id="simple",
        regime="test",
        seed=1,
    )


@pytest.fixture
def tiny_training_dataset() -> KnapsackDataset:
    return generate_dataset(
        count=4,
        item_counts=(4, 5),
        regimes=("independent", "weakly_conflicting"),
        seed=100,
    )


@pytest.fixture
def tiny_validation_dataset() -> KnapsackDataset:
    return generate_dataset(
        count=2,
        item_counts=(4, 5),
        regimes=("independent", "strongly_conflicting"),
        seed=200,
    )
