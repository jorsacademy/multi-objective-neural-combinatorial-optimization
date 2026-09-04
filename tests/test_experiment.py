from __future__ import annotations

from pathlib import Path

import pytest

from mopareto.experiment import ResearchConfig, run_research


def _config_payload() -> dict[str, object]:
    return {
        "schema_version": "mopareto-research-v1",
        "training_count": 3,
        "training_item_counts": [4, 5],
        "training_regimes": ["independent", "weakly_conflicting"],
        "training_seed": 10,
        "validation_count": 2,
        "validation_item_counts": [4],
        "validation_regimes": ["independent"],
        "validation_seed": 20,
        "capacity_ratio": 0.45,
        "hidden_dim": 8,
        "hidden_layers": 1,
        "epochs": 1,
        "batch_size": 8,
        "learning_rate": 0.001,
        "conditions_per_instance": 3,
        "training_seed_offset": 30,
        "query_budget": 3,
        "bootstrap_draws": 10,
        "bootstrap_seed": 40,
        "scenarios": [
            {
                "name": "smoke",
                "count": 2,
                "item_count": 5,
                "regime": "strongly_conflicting",
                "capacity_ratio": 0.45,
                "seed": 50,
                "value_scale": 100,
            }
        ],
    }


def test_frozen_protocol_runs_and_saves_checkpoints(tmp_path: Path) -> None:
    config = ResearchConfig.from_dict(_config_payload())
    report = run_research(config, checkpoint_directory=tmp_path / "checkpoints")
    assert len(report.evaluations) == 1
    assert Path(report.checkpoints["weighted_sum"]).exists()
    assert Path(report.checkpoints["epsilon_constraint"]).exists()
    assert len(report.config_fingerprint) == 64


def test_research_config_validation() -> None:
    payload = _config_payload()
    payload["schema_version"] = "bad"
    with pytest.raises(ValueError, match="unsupported"):
        ResearchConfig.from_dict(payload)
    payload = _config_payload()
    payload["scenarios"] = "bad"
    with pytest.raises(ValueError, match="JSON array"):
        ResearchConfig.from_dict(payload)
