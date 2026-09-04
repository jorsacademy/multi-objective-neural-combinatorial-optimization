from __future__ import annotations

import json
from pathlib import Path

import pytest

from mopareto.dataset import (
    SUPPORTED_REGIMES,
    generate_dataset,
    generate_instance,
    load_dataset,
    save_dataset,
)


def test_generation_is_deterministic_across_all_regimes() -> None:
    for regime in SUPPORTED_REGIMES:
        first = generate_instance(item_count=8, regime=regime, seed=42)
        second = generate_instance(item_count=8, regime=regime, seed=42)
        assert first == second
        assert min(first.weights) <= first.capacity < first.total_weight


def test_dataset_round_trip_and_fingerprint(tmp_path: Path) -> None:
    dataset = generate_dataset(
        count=5,
        item_counts=(6, 7),
        regimes=("independent", "strongly_conflicting"),
        seed=123,
    )
    path = tmp_path / "corpus.jsonl"
    save_dataset(dataset, path)
    loaded = load_dataset(path)
    assert loaded == dataset
    assert loaded.fingerprint == dataset.fingerprint
    assert loaded.to_metadata()["instance_count"] == 5


def test_tamper_detection_rejects_modified_record(tmp_path: Path) -> None:
    dataset = generate_dataset(
        count=1,
        item_counts=(6,),
        regimes=("independent",),
        seed=123,
    )
    path = tmp_path / "corpus.jsonl"
    save_dataset(dataset, path)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["instance"]["capacity"] -= 1
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_dataset(path)


def test_dataset_argument_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least two"):
        generate_instance(item_count=1, regime="independent", seed=0)
    with pytest.raises(ValueError, match="unsupported"):
        generate_instance(item_count=4, regime="unknown", seed=0)
    with pytest.raises(ValueError, match="strictly"):
        generate_instance(item_count=4, regime="independent", seed=0, capacity_ratio=1.0)
    with pytest.raises(ValueError, match="positive"):
        generate_dataset(count=0, item_counts=(4,), regimes=("independent",), seed=0)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no records"):
        load_dataset(empty)
