"""Deterministic exact-oracle supervision for conditional Pareto policies."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mopareto.dataset import KnapsackDataset
from mopareto.models import (
    ConditionalParetoPolicy,
    ConditionalPolicyConfig,
    PolicyMode,
    condition_features,
    item_features,
)
from mopareto.pareto import (
    condition_grid,
    epsilon_constraint_solution,
    weighted_sum_solution,
)
from mopareto.utils import set_global_seed


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    conditions_per_instance: int = 11
    capacity_penalty: float = 0.1
    cardinality_penalty: float = 0.05
    gradient_clip_norm: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer parameters are invalid")
        if self.conditions_per_instance < 2:
            raise ValueError("conditions_per_instance must be at least two")
        if self.capacity_penalty < 0.0 or self.cardinality_penalty < 0.0:
            raise ValueError("loss penalties must be nonnegative")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


@dataclass(frozen=True, slots=True)
class SupervisionExample:
    instance_index: int
    condition: float
    target_selection: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_item_accuracy: float
    validation_exact_match_rate: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingReport:
    mode: PolicyMode
    model_config: dict[str, object]
    training_config: dict[str, object]
    parameter_count: int
    training_dataset_fingerprint: str
    validation_dataset_fingerprint: str
    training_example_count: int
    validation_example_count: int
    epochs: tuple[EpochMetrics, ...]
    best_validation_loss: float

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "parameter_count": self.parameter_count,
            "training_dataset_fingerprint": self.training_dataset_fingerprint,
            "validation_dataset_fingerprint": self.validation_dataset_fingerprint,
            "training_example_count": self.training_example_count,
            "validation_example_count": self.validation_example_count,
            "epochs": [epoch.to_dict() for epoch in self.epochs],
            "best_validation_loss": self.best_validation_loss,
        }


def _epsilon_conditions(dataset: KnapsackDataset, index: int, count: int) -> tuple[float, ...]:
    frontier = dataset.frontiers[index]
    uniform_count = max(2, count // 2)
    conditions = list(condition_grid(uniform_count))
    exact_thresholds = sorted(
        {solution.objective_2 / frontier.ideal_point[1] for solution in frontier.solutions}
    )
    if exact_thresholds:
        sample_count = min(len(exact_thresholds), count - len(conditions))
        if sample_count > 0:
            positions = np.linspace(0, len(exact_thresholds) - 1, sample_count)
            conditions.extend(exact_thresholds[int(round(position))] for position in positions)
    for condition in condition_grid(max(count, 2)):
        if len(conditions) >= count:
            break
        conditions.append(condition)
    unique = sorted({round(float(condition), 12) for condition in conditions})
    if len(unique) < count:
        for condition in condition_grid(2 * count + 1):
            rounded = round(float(condition), 12)
            if rounded not in unique:
                unique.append(rounded)
            if len(unique) == count:
                break
    return tuple(sorted(unique)[:count])


def build_supervision_examples(
    dataset: KnapsackDataset,
    *,
    mode: PolicyMode,
    conditions_per_instance: int,
) -> tuple[SupervisionExample, ...]:
    if conditions_per_instance < 2:
        raise ValueError("conditions_per_instance must be at least two")
    examples: list[SupervisionExample] = []
    for index, frontier in enumerate(dataset.frontiers):
        conditions = (
            condition_grid(conditions_per_instance)
            if mode == "weighted_sum"
            else _epsilon_conditions(dataset, index, conditions_per_instance)
        )
        for condition in conditions:
            target = (
                weighted_sum_solution(frontier, condition)
                if mode == "weighted_sum"
                else epsilon_constraint_solution(frontier, condition)
            )
            examples.append(
                SupervisionExample(
                    instance_index=index,
                    condition=condition,
                    target_selection=target.selection,
                )
            )
    return tuple(examples)


def _example_loss(
    model: ConditionalParetoPolicy,
    dataset: KnapsackDataset,
    example: SupervisionExample,
    config: TrainingConfig,
) -> Tensor:
    instance = dataset.instances[example.instance_index]
    logits = model(
        item_features(instance, device=model.device),
        condition_features(model.mode, example.condition, device=model.device),
    )
    targets = torch.tensor(
        example.target_selection,
        dtype=logits.dtype,
        device=model.device,
    )
    classification = F.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    weights = torch.tensor(instance.weights, dtype=logits.dtype, device=model.device)
    expected_weight_ratio = torch.dot(probabilities, weights) / float(instance.capacity)
    capacity_excess = torch.relu(expected_weight_ratio - 1.0)
    capacity_loss = capacity_excess.square()
    cardinality_loss = (
        torch.abs(torch.sum(probabilities) - torch.sum(targets)) / instance.item_count
    )
    loss = (
        classification
        + config.capacity_penalty * capacity_loss
        + config.cardinality_penalty * cardinality_loss
    )
    if not torch.isfinite(loss):
        raise RuntimeError("training produced a non-finite loss")
    return loss


def _evaluate_examples(
    model: ConditionalParetoPolicy,
    dataset: KnapsackDataset,
    examples: tuple[SupervisionExample, ...],
    config: TrainingConfig,
) -> tuple[float, float, float]:
    model.eval()
    losses: list[float] = []
    correct_items = 0
    total_items = 0
    exact_matches = 0
    with torch.no_grad():
        for example in examples:
            instance = dataset.instances[example.instance_index]
            loss = _example_loss(model, dataset, example, config)
            losses.append(float(loss.detach().cpu()))
            logits = model(
                item_features(instance, device=model.device),
                condition_features(model.mode, example.condition, device=model.device),
            )
            predictions = tuple(int(value >= 0.0) for value in logits.detach().cpu().tolist())
            correct_items += sum(
                int(predicted == target)
                for predicted, target in zip(
                    predictions,
                    example.target_selection,
                    strict=True,
                )
            )
            total_items += instance.item_count
            exact_matches += int(predictions == example.target_selection)
    return (
        float(np.mean(np.asarray(losses, dtype=float))),
        correct_items / total_items,
        exact_matches / len(examples),
    )


def _gradients_are_finite(model: nn.Module) -> bool:
    return all(
        parameter.grad is None or bool(torch.all(torch.isfinite(parameter.grad)))
        for parameter in model.parameters()
    )


def train_policy(
    training_dataset: KnapsackDataset,
    validation_dataset: KnapsackDataset,
    *,
    mode: PolicyMode,
    model_config: ConditionalPolicyConfig | None = None,
    training_config: TrainingConfig | None = None,
    device: torch.device | str = "cpu",
) -> tuple[ConditionalParetoPolicy, TrainingReport]:
    """Train one conditional policy from exact frontier-derived labels."""

    config = training_config or TrainingConfig()
    architecture = model_config or ConditionalPolicyConfig()
    set_global_seed(config.seed)
    model = ConditionalParetoPolicy(mode, architecture).to(device)
    training_examples = build_supervision_examples(
        training_dataset,
        mode=mode,
        conditions_per_instance=config.conditions_per_instance,
    )
    validation_examples = build_supervision_examples(
        validation_dataset,
        mode=mode,
        conditions_per_instance=config.conditions_per_instance,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = np.random.default_rng(config.seed)
    epoch_rows: list[EpochMetrics] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = generator.permutation(len(training_examples))
        batch_losses: list[float] = []
        for start in range(0, len(order), config.batch_size):
            batch_indices = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            losses = [
                _example_loss(model, training_dataset, training_examples[int(index)], config)
                for index in batch_indices
            ]
            loss = torch.stack(losses).mean()
            loss.backward()  # type: ignore[no-untyped-call]
            if not _gradients_are_finite(model):
                raise RuntimeError("training produced non-finite gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        validation_loss, item_accuracy, exact_match = _evaluate_examples(
            model,
            validation_dataset,
            validation_examples,
            config,
        )
        train_loss = float(np.mean(np.asarray(batch_losses, dtype=float)))
        if not all(math.isfinite(value) for value in (train_loss, validation_loss)):
            raise RuntimeError("training metrics became non-finite")
        epoch_rows.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                validation_item_accuracy=item_accuracy,
                validation_exact_match_rate=exact_match,
            )
        )

    model.eval()
    report = TrainingReport(
        mode=mode,
        model_config=asdict(architecture),
        training_config=asdict(config),
        parameter_count=model.parameter_count,
        training_dataset_fingerprint=training_dataset.fingerprint,
        validation_dataset_fingerprint=validation_dataset.fingerprint,
        training_example_count=len(training_examples),
        validation_example_count=len(validation_examples),
        epochs=tuple(epoch_rows),
        best_validation_loss=min(row.validation_loss for row in epoch_rows),
    )
    return model, report
