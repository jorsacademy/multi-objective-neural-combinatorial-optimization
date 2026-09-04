# Frozen Experiment Protocol

## Primary comparison

Train one weighted-sum-conditioned policy and one epsilon-constraint-conditioned policy with identical architecture and training budget. At evaluation, query each policy on the same endpoint-inclusive grid. Compare against weighted-density and epsilon-density baselines with the same query count.

The primary estimand is the paired per-instance hypervolume-ratio difference from `weighted_sum_density`. Secondary outcomes are exact objective coverage, unsupported-point coverage, IGD+, additive epsilon, unique candidate count, nondominated yield, condition satisfaction, and runtime.

## Data partitions

Training and validation cycle item counts 8, 10, 12, and 14. Training regimes are:

- independent objectives;
- aligned objectives;
- weakly conflicting objectives.

All generator calls use deterministic, non-overlapping seed ranges. The test scenarios use separate seeds and are never consulted during model fitting.

## Evaluation scenarios

| Scenario | Item count | Regime | Capacity ratio | Purpose |
| --- | ---: | --- | ---: | --- |
| interpolation | 12 | independent | 0.45 | nominal in-range test |
| size_18 | 18 | independent | 0.45 | size extrapolation |
| size_24 | 24 | independent | 0.45 | harder size extrapolation |
| tight_capacity | 14 | independent | 0.25 | reduced feasible volume |
| loose_capacity | 14 | independent | 0.65 | larger selected subsets |
| strong_conflict | 14 | strongly_conflicting | 0.45 | unsupported-front stress |
| aligned_objectives | 14 | aligned | 0.45 | small/easy fronts |
| heavy_tail | 14 | heavy_tail | 0.45 | profit outliers |
| clustered_tradeoffs | 14 | clustered | 0.45 | structured opposing item groups |
| value_scale_shift | 14 | value_scale_shift | 0.45 | objective scale extrapolation |

## Query budget

The default frozen budget is 21 conditions, including both endpoints. Results at one budget do not establish the full quality–cost curve. New studies should sweep budgets and retain paired instance seeds.

## Exact labels

Weighted labels maximize normalized scalar utility over the exact front. Epsilon labels maximize objective 1 subject to an exact normalized objective-2 floor. Epsilon training conditions combine a uniform grid with thresholds induced by exact frontier points.

Canonical tie-breaking is deterministic. Because multiple selections can share the same objective vector, selection-level accuracy is a diagnostic rather than the primary set-quality endpoint.

## Indicators

All objective coordinates are divided by the exact per-instance ideal point. Hypervolume uses reference point `(0, 0)`. The implementation also reports additive epsilon and IGD+ because hypervolume alone can obscure local coverage failures.

Unsupported coverage is undefined for fronts containing no unsupported points and is averaged only over applicable instances.

## Uncertainty

Hypervolume gains are paired by instance and summarized with deterministic percentile bootstrap intervals. The bootstrap seed and draw count are stored in every report. These intervals quantify finite test-sample variability, not training-seed uncertainty. Multi-seed model training requires an outer experimental loop.

## Reporting requirements

A defensible report must include:

- exact dataset fingerprints;
- train/validation/test seed ranges;
- item counts, regimes, capacities, and value scales;
- model and training configuration;
- checkpoint schema and metadata;
- query budget;
- frontier-size and supportedness statistics;
- all four method rows;
- feasibility and condition-satisfaction rates;
- runtime together with set-quality indicators;
- explicit acknowledgement of exact ideal-point normalization.
