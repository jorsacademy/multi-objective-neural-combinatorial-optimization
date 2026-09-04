from __future__ import annotations

import json
from pathlib import Path

from mopareto.cli import main


def test_cli_end_to_end_workflow(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    weighted_path = tmp_path / "weighted.safetensors"
    epsilon_path = tmp_path / "epsilon.safetensors"
    oracle_path = tmp_path / "oracle.json"
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_csv = tmp_path / "benchmark.csv"

    assert main(
        [
            "collect",
            "--count",
            "3",
            "--item-counts",
            "4",
            "5",
            "--regimes",
            "independent",
            "weakly_conflicting",
            "--seed",
            "100",
            "--output",
            str(train_path),
        ]
    ) == 0
    assert main(
        [
            "collect",
            "--count",
            "2",
            "--item-counts",
            "4",
            "--regimes",
            "independent",
            "--seed",
            "200",
            "--output",
            str(validation_path),
        ]
    ) == 0
    for mode, checkpoint, seed in (
        ("weighted_sum", weighted_path, "300"),
        ("epsilon_constraint", epsilon_path, "400"),
    ):
        assert main(
            [
                "train",
                str(train_path),
                "--validation",
                str(validation_path),
                "--model",
                mode,
                "--epochs",
                "1",
                "--batch-size",
                "8",
                "--hidden-dim",
                "8",
                "--hidden-layers",
                "1",
                "--conditions-per-instance",
                "3",
                "--seed",
                seed,
                "--checkpoint",
                str(checkpoint),
            ]
        ) == 0
    assert main(
        [
            "oracle",
            str(validation_path),
            "--sample-index",
            "0",
            "--output",
            str(oracle_path),
        ]
    ) == 0
    assert main(
        [
            "benchmark",
            str(validation_path),
            "--weighted-checkpoint",
            str(weighted_path),
            "--epsilon-checkpoint",
            str(epsilon_path),
            "--scenario",
            "smoke",
            "--query-budget",
            "3",
            "--bootstrap-draws",
            "10",
            "--output-json",
            str(benchmark_path),
            "--output-csv",
            str(benchmark_csv),
        ]
    ) == 0
    assert json.loads(oracle_path.read_text(encoding="utf-8"))["brute_force_verified"] is True
    assert len(json.loads(benchmark_path.read_text(encoding="utf-8"))["frontier_rows"]) == 4
    assert benchmark_csv.exists()


def test_generate_cli(tmp_path: Path) -> None:
    output = tmp_path / "instance.json"
    assert main(
        [
            "generate",
            "--item-count",
            "6",
            "--regime",
            "strongly_conflicting",
            "--seed",
            "7",
            "--output",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["frontier"]["solutions"]
    assert len(payload["fingerprint"]) == 64
