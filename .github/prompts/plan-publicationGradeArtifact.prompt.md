## Plan: Publication‑grade research artifact

Upgrade packaging, reproducibility, documentation, tests, and metadata while preserving existing algorithms. The plan adds minimal infrastructure (packaging + CLI + pinned deps), documents exact reproduction steps, and expands tests for mathematical correctness and edge cases.

### Steps 5
1. Audit core APIs in [src/entropy_regularized.py](src/entropy_regularized.py), [src/classical_shortest_path.py](src/classical_shortest_path.py), and [src/bounds.py](src/bounds.py) to define stable public entry points and docstrings.
2. Add lightweight packaging and import hygiene so experiments in [experiments/cost_margin.py](experiments/cost_margin.py), [experiments/path_multiplicity.py](experiments/path_multiplicity.py), and [experiments/temperature_analysis.py](experiments/temperature_analysis.py) run without `sys.path` hacks.
3. Make experiments deterministic and reproducible by pinning dependencies in [requirements.txt](requirements.txt) and documenting exact commands in [README.md](README.md).
4. Expand tests in [tests/test_bounds.py](tests/test_bounds.py) to cover edge cases in `soft_shortest_path()` and `soft_hard_gap_bound()` with small DAGs from [data/sample_dag.json](data/sample_dag.json).
5. Add artifact metadata (LICENSE, CITATION, and a short “reproduce figures” section) and clarify data/results policy in [README.md](README.md) and [notebooks/README.md](notebooks/README.md).

### Further Considerations 2
1. Should generated figures be versioned in results/ (Option A) or always regenerated from scripts (Option B)?
2. Prefer a minimal CLI entry point, or keep script-based execution and only document exact commands?
