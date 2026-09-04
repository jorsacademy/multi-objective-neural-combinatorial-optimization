"""Frozen multi-scenario training and transfer protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from mopareto.dataset import KnapsackDataset, generate_dataset
from mopareto.evaluation import EvaluationReport, evaluate_models
from mopareto.models import ConditionalPolicyConfig, save_checkpoint
from mopareto.training import TrainingConfig, TrainingReport, train_policy
from mopareto.utils import sha256_json

RESEARCH_CONFIG_SCHEMA_VERSION = "mopareto-research-v1"


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    name: str
    count: int
    item_count: int
    regime: str
    capacity_ratio: float
    seed: int
    value_scale: int = 100

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name must be nonempty")
        if self.count <= 0 or self.item_count < 2:
            raise ValueError("scenario count and item_count are invalid")
        if not 0.0 < self.capacity_ratio < 1.0:
            raise ValueError("scenario capacity_ratio must be in (0, 1)")
        if self.seed < 0 or self.value_scale < 10:
            raise ValueError("scenario seed or value_scale is invalid")


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    training_count: int
    training_item_counts: tuple[int, ...]
    training_regimes: tuple[str, ...]
    training_seed: int
    validation_count: int
    validation_item_counts: tuple[int, ...]
    validation_regimes: tuple[str, ...]
    validation_seed: int
    capacity_ratio: float
    hidden_dim: int
    hidden_layers: int
    epochs: int
    batch_size: int
    learning_rate: float
    conditions_per_instance: int
    training_seed_offset: int
    query_budget: int
    bootstrap_draws: int
    bootstrap_seed: int
    scenarios: tuple[ScenarioConfig, ...]

    def __post_init__(self) -> None:
        if self.training_count <= 0 or self.validation_count <= 0:
            raise ValueError("training and validation counts must be positive")
        if not self.training_item_counts or not self.validation_item_counts:
            raise ValueError("training and validation item counts must be nonempty")
        if not self.training_regimes or not self.validation_regimes:
            raise ValueError("training and validation regimes must be nonempty")
        if not 0.0 < self.capacity_ratio < 1.0:
            raise ValueError("capacity_ratio must be in (0, 1)")
        if self.hidden_dim <= 0 or self.hidden_layers <= 0:
            raise ValueError("model dimensions must be positive")
        if self.epochs <= 0 or self.batch_size <= 0 or self.learning_rate <= 0.0:
            raise ValueError("training parameters are invalid")
        if self.conditions_per_instance < 2 or self.query_budget < 2:
            raise ValueError("condition and query budgets must be at least two")
        if self.bootstrap_draws <= 0 or not self.scenarios:
            raise ValueError("bootstrap draws and scenarios must be nonempty")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = RESEARCH_CONFIG_SCHEMA_VERSION
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ResearchConfig:
        if payload.get("schema_version") != RESEARCH_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported research configuration schema")
        raw_scenarios = payload.get("scenarios")
        if not isinstance(raw_scenarios, list):
            raise ValueError("research scenarios must be a JSON array")
        scenarios: list[ScenarioConfig] = []
        for raw in raw_scenarios:
            if not isinstance(raw, dict):
                raise ValueError("research scenario entries must be JSON objects")
            scenarios.append(
                ScenarioConfig(
                    name=_string(raw, "name"),
                    count=_integer(raw, "count"),
                    item_count=_integer(raw, "item_count"),
                    regime=_string(raw, "regime"),
                    capacity_ratio=_number(raw, "capacity_ratio"),
                    seed=_integer(raw, "seed"),
                    value_scale=_integer(raw, "value_scale", default=100),
                )
            )
        return cls(
            training_count=_integer(payload, "training_count"),
            training_item_counts=tuple(_integer_list(payload, "training_item_counts")),
            training_regimes=tuple(_string_list(payload, "training_regimes")),
            training_seed=_integer(payload, "training_seed"),
            validation_count=_integer(payload, "validation_count"),
            validation_item_counts=tuple(_integer_list(payload, "validation_item_counts")),
            validation_regimes=tuple(_string_list(payload, "validation_regimes")),
            validation_seed=_integer(payload, "validation_seed"),
            capacity_ratio=_number(payload, "capacity_ratio"),
            hidden_dim=_integer(payload, "hidden_dim"),
            hidden_layers=_integer(payload, "hidden_layers"),
            epochs=_integer(payload, "epochs"),
            batch_size=_integer(payload, "batch_size"),
            learning_rate=_number(payload, "learning_rate"),
            conditions_per_instance=_integer(payload, "conditions_per_instance"),
            training_seed_offset=_integer(payload, "training_seed_offset"),
            query_budget=_integer(payload, "query_budget"),
            bootstrap_draws=_integer(payload, "bootstrap_draws"),
            bootstrap_seed=_integer(payload, "bootstrap_seed"),
            scenarios=tuple(scenarios),
        )


@dataclass(frozen=True, slots=True)
class ResearchReport:
    config: dict[str, object]
    config_fingerprint: str
    training_dataset_metadata: dict[str, object]
    validation_dataset_metadata: dict[str, object]
    weighted_training: TrainingReport
    epsilon_training: TrainingReport
    evaluations: tuple[EvaluationReport, ...]
    checkpoints: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config,
            "config_fingerprint": self.config_fingerprint,
            "training_dataset_metadata": self.training_dataset_metadata,
            "validation_dataset_metadata": self.validation_dataset_metadata,
            "weighted_training": self.weighted_training.to_dict(),
            "epsilon_training": self.epsilon_training.to_dict(),
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
            "checkpoints": self.checkpoints,
        }


def _integer(payload: dict[str, object], name: str, *, default: int | None = None) -> int:
    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"research field {name!r} must be an integer")
    return value


def _number(payload: dict[str, object], name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"research field {name!r} must be numeric")
    return float(value)


def _string(payload: dict[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str):
        raise ValueError(f"research field {name!r} must be a string")
    return value


def _integer_list(payload: dict[str, object], name: str) -> list[int]:
    value = payload.get(name)
    if not isinstance(value, list) or not all(
        isinstance(entry, int) and not isinstance(entry, bool) for entry in value
    ):
        raise ValueError(f"research field {name!r} must be an integer array")
    return cast(list[int], value)


def _string_list(payload: dict[str, object], name: str) -> list[str]:
    value = payload.get(name)
    if not isinstance(value, list) or not all(isinstance(entry, str) for entry in value):
        raise ValueError(f"research field {name!r} must be a string array")
    return cast(list[str], value)


def _generate_training_datasets(config: ResearchConfig) -> tuple[KnapsackDataset, KnapsackDataset]:
    training = generate_dataset(
        count=config.training_count,
        item_counts=config.training_item_counts,
        regimes=config.training_regimes,
        seed=config.training_seed,
        capacity_ratio=config.capacity_ratio,
    )
    validation = generate_dataset(
        count=config.validation_count,
        item_counts=config.validation_item_counts,
        regimes=config.validation_regimes,
        seed=config.validation_seed,
        capacity_ratio=config.capacity_ratio,
    )
    return training, validation


def run_research(
    config: ResearchConfig,
    *,
    checkpoint_directory: str | Path,
) -> ResearchReport:
    """Run the frozen train-once, evaluate-many transfer protocol."""

    training, validation = _generate_training_datasets(config)
    architecture = ConditionalPolicyConfig(
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
    )
    weighted_training_config = TrainingConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        conditions_per_instance=config.conditions_per_instance,
        seed=config.training_seed_offset,
    )
    epsilon_training_config = TrainingConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        conditions_per_instance=config.conditions_per_instance,
        seed=config.training_seed_offset + 1,
    )
    weighted_policy, weighted_report = train_policy(
        training,
        validation,
        mode="weighted_sum",
        model_config=architecture,
        training_config=weighted_training_config,
    )
    epsilon_policy, epsilon_report = train_policy(
        training,
        validation,
        mode="epsilon_constraint",
        model_config=architecture,
        training_config=epsilon_training_config,
    )
    checkpoint_root = Path(checkpoint_directory)
    weighted_path = checkpoint_root / "weighted-sum-policy.safetensors"
    epsilon_path = checkpoint_root / "epsilon-constraint-policy.safetensors"
    save_checkpoint(
        weighted_policy,
        weighted_path,
        metadata={"training_report": weighted_report.to_dict()},
    )
    save_checkpoint(
        epsilon_policy,
        epsilon_path,
        metadata={"training_report": epsilon_report.to_dict()},
    )

    evaluations: list[EvaluationReport] = []
    for offset, scenario in enumerate(config.scenarios):
        dataset = generate_dataset(
            count=scenario.count,
            item_counts=(scenario.item_count,),
            regimes=(scenario.regime,),
            seed=scenario.seed,
            capacity_ratio=scenario.capacity_ratio,
            value_scale=scenario.value_scale,
        )
        evaluations.append(
            evaluate_models(
                weighted_policy,
                epsilon_policy,
                dataset,
                scenario=scenario.name,
                query_budget=config.query_budget,
                bootstrap_seed=config.bootstrap_seed + 100 * offset,
                bootstrap_draws=config.bootstrap_draws,
            )
        )
    config_payload = config.to_dict()
    return ResearchReport(
        config=config_payload,
        config_fingerprint=sha256_json(config_payload),
        training_dataset_metadata=training.to_metadata(),
        validation_dataset_metadata=validation.to_metadata(),
        weighted_training=weighted_report,
        epsilon_training=epsilon_report,
        evaluations=tuple(evaluations),
        checkpoints={
            "weighted_sum": str(weighted_path),
            "epsilon_constraint": str(epsilon_path),
        },
    )
