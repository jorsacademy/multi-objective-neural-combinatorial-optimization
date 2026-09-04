"""Controlled instance generators and tamper-evident exact Pareto corpora."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mopareto.domain import (
    MultiObjectiveKnapsackInstance,
    ParetoFrontier,
    solve_pareto_dynamic_programming,
)
from mopareto.utils import canonical_json, sha256_json

CORPUS_SCHEMA_VERSION = "mopareto-corpus-v1"
GENERATOR_SCHEMA_VERSION = "biobjective-knapsack-generator-v1"
SUPPORTED_REGIMES = (
    "independent",
    "aligned",
    "weakly_conflicting",
    "strongly_conflicting",
    "heavy_tail",
    "clustered",
    "value_scale_shift",
)


@dataclass(frozen=True, slots=True)
class KnapsackDataset:
    instances: tuple[MultiObjectiveKnapsackInstance, ...]
    frontiers: tuple[ParetoFrontier, ...]

    def __post_init__(self) -> None:
        if not self.instances or len(self.instances) != len(self.frontiers):
            raise ValueError("dataset instances and exact frontiers must be aligned and nonempty")
        for instance, frontier in zip(self.instances, self.frontiers, strict=True):
            for solution in frontier.solutions:
                if len(solution.selection) != instance.item_count:
                    raise ValueError("frontier selection length does not match its instance")
                if solution.total_weight > instance.capacity:
                    raise ValueError("dataset frontier contains an infeasible solution")

    @property
    def fingerprint(self) -> str:
        return sha256_json(
            [
                _record_payload(instance, frontier)
                for instance, frontier in zip(self.instances, self.frontiers, strict=True)
            ]
        )

    @property
    def regimes(self) -> tuple[str, ...]:
        return tuple(sorted({instance.regime for instance in self.instances}))

    @property
    def item_counts(self) -> tuple[int, ...]:
        return tuple(sorted({instance.item_count for instance in self.instances}))

    @property
    def frontier_sizes(self) -> tuple[int, ...]:
        return tuple(len(frontier.solutions) for frontier in self.frontiers)

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": CORPUS_SCHEMA_VERSION,
            "instance_count": len(self.instances),
            "item_counts": list(self.item_counts),
            "regimes": list(self.regimes),
            "frontier_sizes": list(self.frontier_sizes),
            "fingerprint": self.fingerprint,
        }


def _record_payload(
    instance: MultiObjectiveKnapsackInstance,
    frontier: ParetoFrontier,
) -> dict[str, object]:
    return {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "generator_schema_version": GENERATOR_SCHEMA_VERSION,
        "instance": instance.to_dict(),
        "frontier": frontier.to_dict(),
    }


def _clip_integer_values(values: np.ndarray, *, lower: int, upper: int) -> tuple[int, ...]:
    clipped = np.clip(np.rint(values), lower, upper).astype(np.int64)
    return tuple(int(value) for value in clipped)


def generate_instance(
    *,
    item_count: int,
    regime: str,
    seed: int,
    capacity_ratio: float = 0.45,
    value_scale: int = 100,
) -> MultiObjectiveKnapsackInstance:
    """Generate one deterministic controlled bi-objective knapsack instance."""

    if item_count < 2:
        raise ValueError("item_count must be at least two")
    if regime not in SUPPORTED_REGIMES:
        raise ValueError(f"unsupported regime: {regime}")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not 0.0 < capacity_ratio < 1.0:
        raise ValueError("capacity_ratio must lie strictly between zero and one")
    if value_scale < 10:
        raise ValueError("value_scale must be at least ten")

    rng = np.random.default_rng(seed)
    if regime == "clustered":
        clusters = rng.integers(0, 2, size=item_count)
        light = rng.integers(2, 9, size=item_count)
        heavy = rng.integers(11, 25, size=item_count)
        weights_array = np.where(clusters == 0, light, heavy)
        high_1 = rng.integers(value_scale, 2 * value_scale + 1, size=item_count)
        low_1 = rng.integers(5, value_scale // 2 + 1, size=item_count)
        high_2 = rng.integers(value_scale, 2 * value_scale + 1, size=item_count)
        low_2 = rng.integers(5, value_scale // 2 + 1, size=item_count)
        profits_1_array = np.where(clusters == 0, high_1, low_1)
        profits_2_array = np.where(clusters == 0, low_2, high_2)
    else:
        weights_array = rng.integers(1, 21, size=item_count)
        if regime == "independent":
            profits_1_array = rng.integers(10, value_scale + 1, size=item_count)
            profits_2_array = rng.integers(10, value_scale + 1, size=item_count)
        elif regime == "aligned":
            latent = rng.integers(10, value_scale + 1, size=item_count)
            noise = rng.normal(0.0, 0.08 * value_scale, size=item_count)
            profits_1_array = latent
            profits_2_array = np.asarray(latent, dtype=float) + noise
        elif regime in {"weakly_conflicting", "strongly_conflicting"}:
            profits_1_array = rng.integers(10, value_scale + 1, size=item_count)
            reflected = value_scale + 10 - profits_1_array
            random_component = rng.integers(10, value_scale + 1, size=item_count)
            if regime == "weakly_conflicting":
                profits_2_array = 0.65 * reflected + 0.35 * random_component
            else:
                profits_2_array = reflected + rng.normal(
                    0.0,
                    0.04 * value_scale,
                    size=item_count,
                )
        elif regime == "heavy_tail":
            tail_1 = 1.0 + rng.pareto(1.8, size=item_count)
            tail_2 = 1.0 + rng.pareto(1.8, size=item_count)
            profits_1_array = 8.0 + value_scale * tail_1 / np.median(tail_1)
            profits_2_array = 8.0 + value_scale * tail_2 / np.median(tail_2)
        elif regime == "value_scale_shift":
            profits_1_array = rng.integers(3 * value_scale, 7 * value_scale + 1, size=item_count)
            profits_2_array = rng.integers(3 * value_scale, 7 * value_scale + 1, size=item_count)
        else:
            raise RuntimeError("unreachable generator regime")

    weights = tuple(int(value) for value in weights_array)
    if regime == "value_scale_shift":
        upper = 8 * value_scale
    elif regime in {"clustered", "heavy_tail"}:
        upper = 4 * value_scale
    else:
        upper = 2 * value_scale
    profits_1 = _clip_integer_values(np.asarray(profits_1_array), lower=1, upper=upper)
    profits_2 = _clip_integer_values(np.asarray(profits_2_array), lower=1, upper=upper)

    total_weight = sum(weights)
    capacity = int(round(total_weight * capacity_ratio))
    capacity = max(min(weights), min(capacity, total_weight - 1))
    return MultiObjectiveKnapsackInstance(
        weights=weights,
        profits_1=profits_1,
        profits_2=profits_2,
        capacity=capacity,
        instance_id=f"{regime}-n{item_count}-s{seed}",
        regime=regime,
        seed=seed,
    )


def generate_dataset(
    *,
    count: int,
    item_counts: Sequence[int],
    regimes: Sequence[str],
    seed: int,
    capacity_ratio: float = 0.45,
    value_scale: int = 100,
) -> KnapsackDataset:
    """Generate an exact-labeled deterministic dataset by cycling supplied controls."""

    if count <= 0:
        raise ValueError("count must be positive")
    if not item_counts or not regimes:
        raise ValueError("item_counts and regimes must be nonempty")
    instances: list[MultiObjectiveKnapsackInstance] = []
    frontiers: list[ParetoFrontier] = []
    for index in range(count):
        instance_seed = seed + index * 104_729
        instance = generate_instance(
            item_count=int(item_counts[index % len(item_counts)]),
            regime=str(regimes[index % len(regimes)]),
            seed=instance_seed,
            capacity_ratio=capacity_ratio,
            value_scale=value_scale,
        )
        frontier = solve_pareto_dynamic_programming(instance)
        if not isinstance(frontier, ParetoFrontier):
            raise RuntimeError("unexpected Pareto dynamic-programming return type")
        instances.append(instance)
        frontiers.append(frontier)
    return KnapsackDataset(tuple(instances), tuple(frontiers))


def save_dataset(dataset: KnapsackDataset, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for instance, frontier in zip(dataset.instances, dataset.frontiers, strict=True):
            payload = _record_payload(instance, frontier)
            record = {
                **payload,
                "record_fingerprint": sha256_json(payload),
            }
            handle.write(canonical_json(record) + "\n")


def load_dataset(path: str | Path) -> KnapsackDataset:
    source = Path(path)
    instances: list[MultiObjectiveKnapsackInstance] = []
    frontiers: list[ParetoFrontier] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw: object = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on corpus line {line_number}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"corpus line {line_number} must contain a JSON object")
            schema = raw.get("schema_version")
            generator_schema = raw.get("generator_schema_version")
            if schema != CORPUS_SCHEMA_VERSION or generator_schema != GENERATOR_SCHEMA_VERSION:
                raise ValueError(f"unsupported corpus schema on line {line_number}")
            raw_instance = raw.get("instance")
            raw_frontier = raw.get("frontier")
            fingerprint = raw.get("record_fingerprint")
            if not isinstance(raw_instance, dict) or not isinstance(raw_frontier, dict):
                raise ValueError(f"corpus line {line_number} is missing instance/frontier objects")
            if not isinstance(fingerprint, str):
                raise ValueError(f"corpus line {line_number} is missing its fingerprint")
            payload = {
                "schema_version": schema,
                "generator_schema_version": generator_schema,
                "instance": raw_instance,
                "frontier": raw_frontier,
            }
            if sha256_json(payload) != fingerprint:
                raise ValueError(f"corpus fingerprint mismatch on line {line_number}")
            instance = MultiObjectiveKnapsackInstance.from_dict(raw_instance)
            stored_frontier = ParetoFrontier.from_dict(raw_frontier)
            recomputed = solve_pareto_dynamic_programming(instance)
            if not isinstance(recomputed, ParetoFrontier):
                raise RuntimeError("unexpected Pareto dynamic-programming return type")
            if stored_frontier != recomputed:
                raise ValueError(f"stored exact frontier is inconsistent on line {line_number}")
            instances.append(instance)
            frontiers.append(recomputed)
    if not instances:
        raise ValueError("corpus contains no records")
    return KnapsackDataset(tuple(instances), tuple(frontiers))
