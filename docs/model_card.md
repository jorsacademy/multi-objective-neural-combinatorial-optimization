# Model Card

## Model

`ConditionalParetoPolicy` is a small permutation-equivariant neural item scorer for synthetic bi-objective 0-1 knapsack. Two modes are supported:

- `weighted_sum`;
- `epsilon_constraint`.

The architecture is identical across modes and is trained by supervised imitation of exact oracle selections.

## Intended use

- controlled research on preference-conditioned neural combinatorial optimization;
- analysis of supported versus unsupported Pareto-point recovery;
- regression tests for multi-objective set indicators and exact labels;
- educational experiments on amortized Pareto-set approximation.

## Out-of-scope use

- commercial or safety-critical optimization;
- optimality, nondominance, or completeness certification;
- arbitrary numbers of objectives;
- unbounded item counts or capacities;
- instances with negative/zero weights or profits;
- direct use on private operational data without validation and threat modeling.

## Training data

Data are synthetic positive-integer bi-objective knapsack instances generated from documented regimes. Exact Pareto frontiers are computed locally. No external personal or proprietary dataset is bundled.

## Inputs and outputs

Input:

- a variable-length set of item features;
- one scalar condition in `[0, 1]`.

Output:

- one finite inclusion logit per item.

Logits require deterministic feasible decoding before evaluation. A logit is not a probability calibration guarantee.

## Evaluation

The frozen benchmark reports hypervolume ratio, additive epsilon, IGD+, exact objective coverage, unsupported coverage, extreme recovery, unique candidates, nondominated yield, condition satisfaction, feasibility, and runtime across controlled shifts.

## Limitations

- selection supervision chooses one canonical subset among objective-equivalent alternatives;
- finite query grids can confound model error with condition discretization;
- exact ideal-point normalization supplies oracle information;
- a DeepSets scorer has limited capacity for complex higher-order item interactions;
- deterministic greedy/swap decoding can dominate or mask raw logit quality;
- synthetic regimes do not establish real-world validity.

## Safety and reliability

Checkpoints use Safetensors. Inputs are validated for finite values and compatible schemas. Exact corpus labels are recomputed. Decoded candidates are audited. These measures reduce silent experimental corruption but do not make untrusted artifacts safe to execute outside a sandbox.
