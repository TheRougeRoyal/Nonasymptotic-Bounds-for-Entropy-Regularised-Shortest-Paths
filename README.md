# Entropy-Regularized Shortest Paths

Research implementation workspace for nonasymptotic bounds on entropy-regularized shortest paths.

## Overview
This project provides:
- Classical shortest-path algorithms (Dijkstra, Bellman-Ford)
- Entropy-regularized (soft) shortest-path value computation via log-sum-exp
- Nonasymptotic bound utilities (Theorem III.1)
- Reproducible experiments and plots
- IEEE paper template scaffold

## Quick Start
1. Create and activate a virtual environment.
2. Install dependencies from requirements.txt.
3. Run experiments in experiments/ to generate plots in results/.
4. Run tests with pytest.

## Project Layout
- src/: Core implementation modules
- experiments/: Numerical experiments and validation
- tests/: Unit tests for theoretical bounds verification
- notebooks/: Jupyter notebooks for visualization
- data/: Sample DAG datasets
- results/: Output figures and tables
- paper/: IEEE LaTeX template scaffold

## Experiments
- temperature_analysis.py: gap vs temperature, exponential convergence as $T \to 0^+$, and classical vs soft comparison plots
- path_multiplicity.py: impact of $N_{sub}$ and $N_{tot}$
- cost_margin.py: sensitivity to $\Delta$

## Notes
- The IEEE template uses the IEEEtran class (available in standard LaTeX distributions).
- Replace placeholder text in paper/main.tex with your final content.
