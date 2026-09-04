"""Command-line workflows for exact data, conditional training, and Pareto evaluation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from mopareto.dataset import (
    SUPPORTED_REGIMES,
    generate_dataset,
    generate_instance,
    load_dataset,
    save_dataset,
)
from mopareto.domain import (
    ParetoFrontier,
    solve_pareto_brute_force,
    solve_pareto_dynamic_programming,
)
from mopareto.evaluation import evaluate_models, save_report_csv, save_report_json
from mopareto.experiment import ResearchConfig, run_research
from mopareto.models import (
    ConditionalPolicyConfig,
    load_checkpoint,
    save_checkpoint,
)
from mopareto.pareto import hypervolume_2d, supported_solution_flags
from mopareto.training import TrainingConfig, train_policy
from mopareto.utils import read_json, sha256_json, write_json


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _unit_open(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("value must lie strictly between zero and one")
    return parsed


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--item-count", type=_positive_integer, required=True)
    parser.add_argument("--regime", choices=SUPPORTED_REGIMES, default="independent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capacity-ratio", type=_unit_open, default=0.45)
    parser.add_argument("--value-scale", type=_positive_integer, default=100)


def _generate(args: argparse.Namespace) -> int:
    instance = generate_instance(
        item_count=args.item_count,
        regime=args.regime,
        seed=args.seed,
        capacity_ratio=args.capacity_ratio,
        value_scale=args.value_scale,
    )
    solved = solve_pareto_dynamic_programming(instance, return_trace=True)
    if not isinstance(solved, tuple):
        raise RuntimeError("expected a Pareto frontier and trace")
    frontier, trace = solved
    payload = {
        "instance": instance.to_dict(),
        "frontier": frontier.to_dict(),
        "supported_flags": list(supported_solution_flags(frontier)),
        "normalized_hypervolume": hypervolume_2d(
            frontier.solutions,
            ideal_point=frontier.ideal_point,
        ),
        "maximum_intermediate_frontier_size": trace.maximum_frontier_size,
    }
    payload["fingerprint"] = sha256_json(payload)
    write_json(payload, args.output)
    print(json.dumps({"output": str(args.output), "frontier_size": len(frontier.solutions)}))
    return 0


def _collect(args: argparse.Namespace) -> int:
    dataset = generate_dataset(
        count=args.count,
        item_counts=args.item_counts,
        regimes=args.regimes,
        seed=args.seed,
        capacity_ratio=args.capacity_ratio,
        value_scale=args.value_scale,
    )
    save_dataset(dataset, args.output)
    print(json.dumps(dataset.to_metadata(), sort_keys=True))
    return 0


def _oracle(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    if not 0 <= args.sample_index < len(dataset.instances):
        raise ValueError("sample_index is outside the dataset")
    instance = dataset.instances[args.sample_index]
    frontier = dataset.frontiers[args.sample_index]
    brute_force: ParetoFrontier | None = None
    if instance.item_count <= args.maximum_brute_force_items:
        brute_force = solve_pareto_brute_force(
            instance,
            maximum_items=args.maximum_brute_force_items,
        )
        if brute_force != frontier:
            raise RuntimeError("Pareto dynamic programming disagrees with exhaustive enumeration")
    payload = {
        "sample_index": args.sample_index,
        "instance": instance.to_dict(),
        "dynamic_programming_frontier": frontier.to_dict(),
        "brute_force_verified": brute_force == frontier if brute_force is not None else False,
        "supported_flags": list(supported_solution_flags(frontier)),
        "dataset_fingerprint": dataset.fingerprint,
    }
    write_json(payload, args.output)
    print(json.dumps({"output": str(args.output), "verified": payload["brute_force_verified"]}))
    return 0


def _train(args: argparse.Namespace) -> int:
    training = load_dataset(args.dataset)
    validation = load_dataset(args.validation)
    model, report = train_policy(
        training,
        validation,
        mode=args.model,
        model_config=ConditionalPolicyConfig(
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
        ),
        training_config=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            conditions_per_instance=args.conditions_per_instance,
            capacity_penalty=args.capacity_penalty,
            cardinality_penalty=args.cardinality_penalty,
            gradient_clip_norm=args.gradient_clip_norm,
            seed=args.seed,
        ),
        device=args.device,
    )
    save_checkpoint(model, args.checkpoint, metadata={"training_report": report.to_dict()})
    if args.output_report is not None:
        write_json(report.to_dict(), args.output_report)
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "mode": model.mode,
                "best_validation_loss": report.best_validation_loss,
            },
            sort_keys=True,
        )
    )
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    weighted, weighted_metadata = load_checkpoint(args.weighted_checkpoint, device=args.device)
    epsilon, epsilon_metadata = load_checkpoint(args.epsilon_checkpoint, device=args.device)
    report = evaluate_models(
        weighted,
        epsilon,
        dataset,
        scenario=args.scenario,
        query_budget=args.query_budget,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_draws=args.bootstrap_draws,
    )
    report.metadata["weighted_checkpoint_metadata"] = weighted_metadata
    report.metadata["epsilon_checkpoint_metadata"] = epsilon_metadata
    save_report_json(report, args.output_json)
    if args.output_csv is not None:
        save_report_csv(report, args.output_csv)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "scenario": args.scenario,
                "dataset_fingerprint": dataset.fingerprint,
            },
            sort_keys=True,
        )
    )
    return 0


def _research(args: argparse.Namespace) -> int:
    raw = read_json(args.config)
    if not isinstance(raw, dict):
        raise ValueError("research configuration must contain a JSON object")
    config = ResearchConfig.from_dict(cast(dict[str, object], raw))
    report = run_research(config, checkpoint_directory=args.checkpoint_directory)
    write_json(report.to_dict(), args.output_report)
    print(
        json.dumps(
            {
                "output_report": str(args.output_report),
                "config_fingerprint": report.config_fingerprint,
                "scenario_count": len(report.evaluations),
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mopareto",
        description="Verification-first neural approximation of exact bi-objective knapsack fronts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="generate one exact-labeled instance")
    _add_generation_arguments(generate_parser)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.set_defaults(handler=_generate)

    collect_parser = subparsers.add_parser("collect", help="build a tamper-evident JSONL corpus")
    collect_parser.add_argument("--count", type=_positive_integer, required=True)
    collect_parser.add_argument("--item-counts", type=_positive_integer, nargs="+", required=True)
    collect_parser.add_argument("--regimes", choices=SUPPORTED_REGIMES, nargs="+", required=True)
    collect_parser.add_argument("--seed", type=int, default=0)
    collect_parser.add_argument("--capacity-ratio", type=_unit_open, default=0.45)
    collect_parser.add_argument("--value-scale", type=_positive_integer, default=100)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.set_defaults(handler=_collect)

    oracle_parser = subparsers.add_parser(
        "oracle",
        help="verify a stored exact frontier by exhaustive enumeration when small enough",
    )
    oracle_parser.add_argument("dataset", type=Path)
    oracle_parser.add_argument("--sample-index", type=int, default=0)
    oracle_parser.add_argument("--maximum-brute-force-items", type=_positive_integer, default=22)
    oracle_parser.add_argument("--output", type=Path, required=True)
    oracle_parser.set_defaults(handler=_oracle)

    train_parser = subparsers.add_parser("train", help="train one exact-oracle conditional policy")
    train_parser.add_argument("dataset", type=Path)
    train_parser.add_argument("--validation", type=Path, required=True)
    train_parser.add_argument(
        "--model",
        choices=("weighted_sum", "epsilon_constraint"),
        required=True,
    )
    train_parser.add_argument("--epochs", type=_positive_integer, default=30)
    train_parser.add_argument("--batch-size", type=_positive_integer, default=16)
    train_parser.add_argument("--hidden-dim", type=_positive_integer, default=64)
    train_parser.add_argument("--hidden-layers", type=_positive_integer, default=2)
    train_parser.add_argument("--conditions-per-instance", type=_positive_integer, default=11)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-5)
    train_parser.add_argument("--capacity-penalty", type=float, default=0.1)
    train_parser.add_argument("--cardinality-penalty", type=float, default=0.05)
    train_parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--checkpoint", type=Path, required=True)
    train_parser.add_argument("--output-report", type=Path)
    train_parser.set_defaults(handler=_train)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="construct fixed-budget approximate fronts and compute exact indicators",
    )
    benchmark_parser.add_argument("dataset", type=Path)
    benchmark_parser.add_argument("--weighted-checkpoint", type=Path, required=True)
    benchmark_parser.add_argument("--epsilon-checkpoint", type=Path, required=True)
    benchmark_parser.add_argument("--scenario", required=True)
    benchmark_parser.add_argument("--query-budget", type=_positive_integer, default=21)
    benchmark_parser.add_argument("--bootstrap-seed", type=int, default=0)
    benchmark_parser.add_argument("--bootstrap-draws", type=_positive_integer, default=500)
    benchmark_parser.add_argument("--device", default="cpu")
    benchmark_parser.add_argument("--output-json", type=Path, required=True)
    benchmark_parser.add_argument("--output-csv", type=Path)
    benchmark_parser.set_defaults(handler=_benchmark)

    research_parser = subparsers.add_parser("research", help="run the frozen transfer protocol")
    research_parser.add_argument("--config", type=Path, required=True)
    research_parser.add_argument("--checkpoint-directory", type=Path, required=True)
    research_parser.add_argument("--output-report", type=Path, required=True)
    research_parser.set_defaults(handler=_research)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    if not callable(handler):
        raise RuntimeError("command handler is not callable")
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
