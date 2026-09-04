# Architecture

## System boundary

```text
controlled generator
      │
      ▼
exact Pareto dynamic program ───────────────┐
      │                                     │
      ├── canonical front                   │
      ├── weighted-sum condition labels     │ exact reference
      ├── epsilon-constraint labels         │
      └── supportedness classification      │
                                            │
item set + condition                        │
      │                                     │
      ▼                                     │
conditional permutation-equivariant policy │ approximate
      │                                     │
      ▼                                     │
score-dependent feasible decoder            │
      │                                     │
      ▼                                     │
fixed-budget candidate set                  │
      │                                     │
      ▼                                     │
exact finite-set nondominance filter ◄──────┘
      │
      ▼
hypervolume, epsilon, IGD+, coverage, diversity
```

## Item representation

Each item receives seven scale-controlled features:

1. weight divided by capacity;
2. weight divided by total instance weight;
3. objective-1 profit divided by total objective-1 profit;
4. objective-2 profit divided by total objective-2 profit;
5. objective-1 density divided by the maximum objective-1 density;
6. objective-2 density divided by the maximum objective-2 density;
7. reciprocal item count.

These features are invariant to item ordering and reduce raw value-scale sensitivity. They do not remove the combinatorial coupling induced by capacity.

## Condition representation

The weighted-sum policy embeds

```text
[lambda, 1-lambda, lambda-(1-lambda), 4*lambda*(1-lambda)]
```

so the representation exposes both extreme direction and balance.

The epsilon policy embeds

```text
[epsilon, 1-epsilon, epsilon^2, (1-epsilon)^2]
```

so the model can represent nonlinear changes as the objective floor crosses discrete frontier thresholds.

## Equivariant policy

A shared MLP maps each item to a local hidden vector. Mean and maximum pooling summarize the complete set. A second MLP embeds the condition. For each item, the decoder receives:

```text
local item embedding
+ global mean embedding
+ global maximum embedding
+ expanded condition embedding
```

and emits one inclusion logit. Permuting the item order permutes logits in the same way; global pooling is permutation invariant.

The weighted and epsilon policies have identical parameterization. Only condition semantics and oracle labels differ.

## Training objective

The primary loss is binary cross-entropy against a canonical exact selection. Two weak regularizers are added:

- expected-capacity excess under sigmoid probabilities;
- difference between predicted and target expected cardinality.

These terms shape probabilities but do not guarantee feasible thresholded predictions. Feasibility is enforced only by decoding.

## Decoder

For a neural score vector, the decoder constructs orders from:

- raw score ranking;
- sigmoid-score per unit weight;
- a score-dominant blend with conditioned utility density.

It evaluates full greedy fills and positive-logit-only selections, then performs deterministic one-add and one-swap improvement. Weighted queries compare candidates by normalized scalar utility. Epsilon queries first prefer floor satisfaction, then objective 1; failed-floor candidates are ranked by objective-2 attainment.

The density baselines use matched conditioned density orders and the same local-improvement machinery, preventing differences from being attributed solely to a stronger repair routine.

## Checkpoints

Safetensors checkpoints contain only tensors plus UTF-8 metadata:

- checkpoint schema version;
- feature schema version;
- model type;
- policy mode;
- architecture configuration;
- caller-supplied training metadata.

Loading validates all fields and applies the state dictionary in strict mode.
