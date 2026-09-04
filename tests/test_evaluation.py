from __future__ import annotations

import json
from pathlib import Path

import pytest

from mopareto.dataset import KnapsackDataset
from mopareto.evaluation import evaluate_models, save_report_csv, save_report_json
from mopareto.models import ConditionalParetoPolicy, ConditionalPolicyConfig


def _models() -> tuple[ConditionalParetoPolicy, ConditionalParetoPolicy]:
    config = ConditionalPolicyConfig(hidden_dim=8, hidden_layers=1)
    return (
        ConditionalParetoPolicy("weighted_sum", config),
        ConditionalParetoPolicy("epsilon_constraint", config),
    )


def test_evaluation_report_is_complete_and_serializable(
    tiny_validation_dataset: KnapsackDataset,
    tmp_path: Path,
) -> None:
    weighted, epsilon = _models()
    report = evaluate_models(
        weighted,
        epsilon,
        tiny_validation_dataset,
        scenario="test",
        query_budget=5,
        bootstrap_draws=20,
    )
    assert len(report.frontier_rows) == 4
    assert report.oracle_metrics.instance_count == len(tiny_validation_dataset.instances)
    assert report.metadata["all_decoded_candidates_feasible"] is True
    for row in report.frontier_rows:
        assert 0.0 <= row.mean_hypervolume_ratio <= 1.0
        assert row.mean_feasibility_rate == 1.0
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    save_report_json(report, json_path)
    save_report_csv(report, csv_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["frontier_rows"]
    assert "row_type" in csv_path.read_text(encoding="utf-8")


def test_evaluation_rejects_swapped_modes(tiny_validation_dataset: KnapsackDataset) -> None:
    weighted, epsilon = _models()
    with pytest.raises(ValueError, match="wrong mode"):
        evaluate_models(epsilon, weighted, tiny_validation_dataset, scenario="bad")
    with pytest.raises(ValueError, match="nonempty"):
        evaluate_models(weighted, epsilon, tiny_validation_dataset, scenario="")
    with pytest.raises(ValueError, match="invalid"):
        evaluate_models(
            weighted,
            epsilon,
            tiny_validation_dataset,
            scenario="bad",
            query_budget=1,
        )
