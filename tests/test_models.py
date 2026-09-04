from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mopareto.domain import MultiObjectiveKnapsackInstance
from mopareto.models import (
    ConditionalParetoPolicy,
    ConditionalPolicyConfig,
    condition_features,
    item_features,
    load_checkpoint,
    policy_logits,
    save_checkpoint,
)


def test_policy_shapes_and_permutation_equivariance(
    simple_instance: MultiObjectiveKnapsackInstance,
) -> None:
    torch.manual_seed(3)
    model = ConditionalParetoPolicy(
        "weighted_sum",
        ConditionalPolicyConfig(hidden_dim=12, hidden_layers=1),
    )
    logits = policy_logits(model, simple_instance, 0.3)
    assert logits.shape == (simple_instance.item_count,)
    permutation = (2, 0, 3, 1)
    permuted = MultiObjectiveKnapsackInstance(
        weights=tuple(simple_instance.weights[index] for index in permutation),
        profits_1=tuple(simple_instance.profits_1[index] for index in permutation),
        profits_2=tuple(simple_instance.profits_2[index] for index in permutation),
        capacity=simple_instance.capacity,
    )
    permuted_logits = policy_logits(model, permuted, 0.3)
    for new_index, old_index in enumerate(permutation):
        assert float(permuted_logits[new_index].detach()) == pytest.approx(
            float(logits[old_index].detach()), abs=1e-6
        )


def test_feature_validation(simple_instance: MultiObjectiveKnapsackInstance) -> None:
    features = item_features(simple_instance)
    assert features.shape == (simple_instance.item_count, 7)
    assert condition_features("weighted_sum", 0.25).shape == (4,)
    assert condition_features("epsilon_constraint", 0.25).shape == (4,)
    with pytest.raises(ValueError, match="finite unit interval"):
        condition_features("weighted_sum", -0.1)
    model = ConditionalParetoPolicy("weighted_sum")
    with pytest.raises(ValueError, match="item features"):
        model(torch.zeros(2, 6), torch.zeros(4))
    with pytest.raises(ValueError, match="condition features"):
        model(torch.zeros(2, 7), torch.zeros(3))


def test_checkpoint_round_trip(
    tmp_path: Path, simple_instance: MultiObjectiveKnapsackInstance
) -> None:
    torch.manual_seed(4)
    model = ConditionalParetoPolicy(
        "epsilon_constraint",
        ConditionalPolicyConfig(hidden_dim=10, hidden_layers=2),
    )
    expected = policy_logits(model, simple_instance, 0.6).detach().clone()
    path = tmp_path / "model.safetensors"
    save_checkpoint(model, path, metadata={"seed": 4})
    loaded, metadata = load_checkpoint(path)
    actual = policy_logits(loaded, simple_instance, 0.6)
    assert loaded.mode == "epsilon_constraint"
    assert loaded.config == model.config
    assert metadata == {"seed": 4}
    assert torch.allclose(expected, actual)


def test_model_config_and_mode_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        ConditionalPolicyConfig(hidden_dim=0)
    with pytest.raises(ValueError, match="unsupported"):
        ConditionalParetoPolicy("invalid")  # type: ignore[arg-type]
