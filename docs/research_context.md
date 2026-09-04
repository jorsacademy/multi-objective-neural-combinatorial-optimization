# Research Context

## Preference-conditioned Pareto-set learning

Lin, Yang, and Zhang's *Pareto Set Learning for Neural Multi-objective Combinatorial Optimization* proposes a single preference-conditioned model that amortizes solutions across trade-off weights and evaluates multiobjective TSP, VRP, and knapsack. This repository adopts the single-model conditioning question but uses exact supervised labels and a complete discrete-front oracle rather than reinforcement learning.

Reference: <https://arxiv.org/abs/2203.15386>

## Diversity limitations

Chen et al.'s NeurIPS 2023 work on neural MOCO with diversity enhancement identifies repeated solutions across decomposition subproblems as a core limitation. Lu et al. further connect condition-aware sequence modeling and hypervolume-oriented objectives to diversity. The present benchmark therefore reports unique candidate count and nondominated yield in addition to hypervolume.

References:

- <https://proceedings.neurips.cc/paper_files/paper/2023/hash/7b5ae891000049b91b3b62de596b1560-Abstract-Conference.html>
- <https://arxiv.org/abs/2405.08604>

## Conditional specialization

Fan et al.'s NeurIPS 2025 POCCO framework studies conditional computation across preference subproblems. This implementation does not specialize model structure; it deliberately holds architecture constant to isolate condition semantics and labels.

Reference: <https://proceedings.neurips.cc/paper_files/paper/2025/hash/10fc83943b4540a9524af6fc67a23fef-Abstract-Conference.html>

## Scale

Singh et al.'s 2026 *Divide and Learn* reformulates large-scale multi-objective combinatorial optimization as online learning over decomposed decisions. The present repository instead chooses a small exact setting so unsupportedness and full-front coverage can be audited without surrogate ground truth.

Reference: <https://arxiv.org/abs/2602.11346>

## Narrow methodological contribution

The repository focuses on one controlled distinction:

```text
linear preference conditioning
        versus
constraint-level conditioning
```

For a discrete non-convex frontier, the two oracle families have different representational reach. Exact rational supportedness classification makes that difference observable. The learned policies, matched decoders, density baselines, and fixed query budget then expose how much of the oracle-level distinction survives amortization and extrapolation.
