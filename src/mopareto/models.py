"""Permutation-equivariant condition-aware policies and safe checkpoints."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn

from mopareto.domain import MultiObjectiveKnapsackInstance

PolicyMode = Literal["weighted_sum", "epsilon_constraint"]
CHECKPOINT_SCHEMA_VERSION = "1.0"
FEATURE_SCHEMA_VERSION = "biobjective-knapsack-conditional-v1"


@dataclass(frozen=True, slots=True)
class ConditionalPolicyConfig:
    hidden_dim: int = 64
    hidden_layers: int = 2

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.hidden_layers <= 0:
            raise ValueError("hidden dimensions must be positive")


def _mlp(input_dim: int, hidden_dim: int, hidden_layers: int, output_dim: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    width = input_dim
    for _ in range(hidden_layers):
        modules.extend((nn.Linear(width, hidden_dim), nn.SiLU()))
        width = hidden_dim
    modules.append(nn.Linear(width, output_dim))
    return nn.Sequential(*modules)


class ConditionalParetoPolicy(nn.Module):
    """DeepSets-style item policy conditioned on a scalar frontier query."""

    item_encoder: nn.Sequential
    condition_encoder: nn.Sequential
    decoder: nn.Sequential

    def __init__(
        self,
        mode: PolicyMode,
        config: ConditionalPolicyConfig | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"weighted_sum", "epsilon_constraint"}:
            raise ValueError(f"unsupported policy mode: {mode}")
        self.mode: PolicyMode = mode
        self.config = config or ConditionalPolicyConfig()
        hidden_dim = self.config.hidden_dim
        self.item_encoder = _mlp(7, hidden_dim, self.config.hidden_layers, hidden_dim)
        self.condition_encoder = _mlp(4, hidden_dim, 1, hidden_dim)
        self.decoder = _mlp(4 * hidden_dim, hidden_dim, self.config.hidden_layers, 1)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, item_inputs: Tensor, condition_inputs: Tensor) -> Tensor:
        if item_inputs.ndim != 2 or item_inputs.shape[1] != 7:
            raise ValueError("item features must have shape [items, 7]")
        if item_inputs.shape[0] == 0:
            raise ValueError("the policy requires at least one item")
        if condition_inputs.ndim != 1 or condition_inputs.shape[0] != 4:
            raise ValueError("condition features must have shape [4]")
        local = cast(Tensor, self.item_encoder(item_inputs))
        condition = cast(Tensor, self.condition_encoder(condition_inputs))
        mean_pool = torch.mean(local, dim=0, keepdim=True).expand_as(local)
        max_pool = torch.max(local, dim=0, keepdim=True).values.expand_as(local)
        expanded_condition = condition.unsqueeze(0).expand(local.shape[0], -1)
        logits = cast(
            Tensor,
            self.decoder(torch.cat((local, mean_pool, max_pool, expanded_condition), dim=1)),
        ).squeeze(-1)
        if not torch.all(torch.isfinite(logits)):
            raise RuntimeError("conditional policy produced non-finite logits")
        return logits


def item_features(
    instance: MultiObjectiveKnapsackInstance,
    *,
    device: torch.device | str = "cpu",
) -> Tensor:
    weights = np.asarray(instance.weights, dtype=np.float32)
    profits_1 = np.asarray(instance.profits_1, dtype=np.float32)
    profits_2 = np.asarray(instance.profits_2, dtype=np.float32)
    density_1 = profits_1 / weights
    density_2 = profits_2 / weights
    max_density_1 = max(float(np.max(density_1)), 1.0)
    max_density_2 = max(float(np.max(density_2)), 1.0)
    features = np.column_stack(
        (
            weights / float(instance.capacity),
            weights / float(instance.total_weight),
            profits_1 / float(instance.total_profit_1),
            profits_2 / float(instance.total_profit_2),
            density_1 / max_density_1,
            density_2 / max_density_2,
            np.full(instance.item_count, 1.0 / instance.item_count, dtype=np.float32),
        )
    ).astype(np.float32)
    return torch.tensor(features, dtype=torch.float32, device=device)


def condition_features(
    mode: PolicyMode,
    condition: float,
    *,
    device: torch.device | str = "cpu",
) -> Tensor:
    if not math.isfinite(condition) or not 0.0 <= condition <= 1.0:
        raise ValueError("condition must lie in the finite unit interval")
    complement = 1.0 - condition
    if mode == "weighted_sum":
        values = (condition, complement, condition - complement, 4.0 * condition * complement)
    elif mode == "epsilon_constraint":
        values = (condition, complement, condition * condition, complement * complement)
    else:
        raise ValueError(f"unsupported policy mode: {mode}")
    return torch.tensor(values, dtype=torch.float32, device=device)


def policy_logits(
    model: ConditionalParetoPolicy,
    instance: MultiObjectiveKnapsackInstance,
    condition: float,
) -> Tensor:
    return model(
        item_features(instance, device=model.device),
        condition_features(model.mode, condition, device=model.device),
    )


def _checkpoint_header(model: ConditionalParetoPolicy) -> dict[str, str]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_type": "conditional_pareto_policy",
        "policy_mode": model.mode,
        "model_config": json.dumps(asdict(model.config), sort_keys=True),
    }


def save_checkpoint(
    model: ConditionalParetoPolicy,
    path: str | Path,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = _checkpoint_header(model)
    header["metadata"] = json.dumps(metadata or {}, sort_keys=True)
    tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(tensors, str(output), metadata=header)


def _config_integer(config: dict[str, object], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"checkpoint model field {name!r} must be an integer")
    return value


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[ConditionalParetoPolicy, dict[str, object]]:
    source = Path(path)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        header = handle.metadata()
        tensors = {
            key: handle.get_tensor(key)
            for key in handle.keys()  # noqa: SIM118 -- Safetensors is not iterable.
        }
    if header is None:
        raise ValueError("checkpoint metadata is missing")
    if header.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    if header.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("checkpoint feature schema is incompatible")
    if header.get("model_type") != "conditional_pareto_policy":
        raise ValueError("checkpoint model type is unsupported")
    mode = header.get("policy_mode")
    if mode not in {"weighted_sum", "epsilon_constraint"}:
        raise ValueError("checkpoint policy mode is unsupported")
    raw_config: object = json.loads(header["model_config"])
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint model configuration is invalid")
    config = cast(dict[str, object], raw_config)
    model = ConditionalParetoPolicy(
        cast(PolicyMode, mode),
        ConditionalPolicyConfig(
            hidden_dim=_config_integer(config, "hidden_dim"),
            hidden_layers=_config_integer(config, "hidden_layers"),
        ),
    )
    model.load_state_dict(tensors, strict=True)
    model.to(device)
    raw_metadata: object = json.loads(header.get("metadata", "{}"))
    if not isinstance(raw_metadata, dict):
        raise ValueError("checkpoint metadata payload is invalid")
    return model, cast(dict[str, object], raw_metadata)
