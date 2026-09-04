# Contributing

Changes should preserve the separation between exact and approximate components. A learned score may influence decoding or query allocation, but it must not be represented as a proof of Pareto optimality or frontier completeness.

Before opening a pull request, run:

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

New optimization logic requires an independently checkable regression test. Changes to corpus or checkpoint schemas must increment their version constants and document migration behavior. Benchmark claims must report query budget, exact reference construction, objective normalization, random seeds, and both set quality and runtime.
