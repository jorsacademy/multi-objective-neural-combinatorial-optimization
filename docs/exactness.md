# Exactness and Reliability Contract

## Exact components

For every declared bi-objective 0-1 knapsack instance:

- weights and both profit coordinates are positive integers;
- capacity admits at least one item and excludes at least one possible item;
- the capacity-indexed dynamic program retains complete nondominated objective sets;
- duplicate objective vectors use deterministic lower-weight, then lexicographic, canonicalization;
- small instances can enumerate every subset and compare the complete frontier;
- solution audits recompute weight and both objectives from the binary vector;
- corpus loading recomputes the exact frontier rather than trusting stored labels;
- supportedness is decided by exact rational feasibility of a scalarization-weight interval;
- nondominance of any reported finite approximation is recomputed exactly.

## Approximate components

- conditional policy logits are learned approximations;
- score decoding is a deterministic heuristic with add/swap local search;
- a finite query grid can miss points even for an exact conditional oracle;
- learned policies can duplicate candidates across conditions;
- epsilon-conditioned candidates can fail their requested objective floor;
- bootstrap intervals have Monte Carlo error;
- wall-clock timings depend on hardware, thread scheduling, and process noise.

## Exact Pareto dynamic program

For each item prefix and capacity limit, the dynamic program merges skip states with take states from the reduced capacity. Within that fixed capacity limit, any state whose objective vector is dominated can be removed: the dominating state is feasible under the same capacity, and lower-capacity tables preserve lighter alternatives needed by later item transitions.

The final capacity table is projected to canonical `ParetoSolution` values and audited. The implementation stores a compact matrix of intermediate frontier sizes for diagnostics rather than serializing every partial state.

## Weakly supported versus unsupported points

A point is weakly supported when some normalized nonnegative linear scalarization makes it optimal, allowing ties. For each exact point, the implementation intersects all pairwise linear inequalities in the scalarization parameter using `fractions.Fraction`. A nonempty interval proves support for the two-objective normalized weighted-sum family. This avoids misclassifying points merely because a finite preference grid did not sample the relevant weight.

Unsupported points remain exact Pareto points; they are simply below the upper convex envelope and cannot be exposed by linear scalarization. Epsilon constraints can target them.

## Oracle-derived normalization

Training labels and evaluation indicators normalize objective coordinates by the exact per-instance ideal point. This is an explicit controlled-benchmark choice. It removes scale imbalance and makes preference semantics comparable across generated regimes, but it also supplies oracle anchor information. Results must not be presented as deployment without exact or estimated normalization.

## Failure policy

The code raises rather than silently continuing when it observes:

- invalid instance domains;
- malformed binary selections;
- inconsistent reported objectives;
- corpus schema or SHA-256 fingerprint mismatch;
- a stored frontier that differs from recomputation;
- non-finite model logits, losses, or gradients;
- incompatible checkpoint metadata or tensor structure;
- infeasible decoded candidates;
- a purported approximation set that is not canonically nondominated;
- hypervolume exceeding the exact reference frontier beyond numerical tolerance;
- disagreement between Pareto dynamic programming and exhaustive enumeration.

A feasibility repair may transform a neural score vector into a valid candidate. It does not repair or conceal failures at exactness boundaries.
