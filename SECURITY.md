# Security policy

Report suspected vulnerabilities privately through GitHub's security-advisory interface rather than opening a public issue.

The project treats datasets and checkpoints as untrusted inputs. JSONL corpora are schema-checked, fingerprinted, and recomputed against the exact Pareto dynamic program. Checkpoints use Safetensors and strict metadata/state-dictionary validation; pickle-based model loading is intentionally unsupported.

Do not run artifacts from unknown sources outside a sandbox. Exact verification protects optimization correctness, not the host environment from arbitrary external files or compromised dependencies.
