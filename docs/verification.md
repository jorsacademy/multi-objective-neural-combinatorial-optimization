# Verification

Local pre-merge verification exercises the complete exact and learned pipeline:

- 34 regression tests pass;
- branch-aware coverage is 87.77%, above the configured 84% threshold;
- Python compile-all succeeds for source and tests;
- capacity-indexed Pareto dynamic programming agrees with exhaustive subset enumeration across randomized small instances;
- corpus fingerprints and exact frontiers are recomputed on load;
- all decoded candidates are audited for binary decisions, capacity feasibility, and objective consistency;
- weighted-sum and epsilon-constraint training, Safetensors round trips, fixed-budget benchmarking, the frozen protocol, and CLI workflows complete locally.

GitHub Actions is the authoritative clean-environment check. It runs Ruff linting and formatting, strict mypy, branch-aware pytest, and an end-to-end collect–train–oracle–benchmark smoke workflow on Python 3.11 and 3.12.
