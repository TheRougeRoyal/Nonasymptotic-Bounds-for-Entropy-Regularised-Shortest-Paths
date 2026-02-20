# Entropy-Regularized Shortest Paths

This repository implements entropy-regularized shortest paths on DAGs. For a source $s$ and target $t$,
the soft value is

$$
d_T(s) = -T \log\Big(\sum_{\pi:s\to t} \exp(-C(\pi)/T)\Big).
$$

We provide classical shortest paths, soft values via log-sum-exp, and Theorem III.1 gap bounds.

## Reproduction
Install dependencies:

```
python -m pip install -r requirements.txt
```

Run all experiments (regenerates plots and CSVs in results/):

```
python run_all_experiments.py
```

Expected runtime: under 1 minute on a typical laptop.

## Project Layout
- src/: Core algorithms and public API
- experiments/: Deterministic experiments and plotting
- tests/: Mathematical verification tests
- notebooks/: Optional exploratory notebooks
- data/: Sample DAG inputs
- results/: Generated plots and CSV outputs (not versioned)
