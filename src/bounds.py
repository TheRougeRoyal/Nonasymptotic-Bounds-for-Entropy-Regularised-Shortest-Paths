from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np


def theorem_iii_1_upper_bound(temperature: float, n_sub: int, delta: float) -> float:
    """Compute the Theorem III.1 bound: T log(1 + N_sub e^{-Delta/T})."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if n_sub < 0:
        raise ValueError("n_sub must be non-negative")
    if delta < 0:
        raise ValueError("delta must be non-negative")

    return float(temperature * np.log1p(n_sub * np.exp(-delta / temperature)))


def soft_hard_gap_bound(delta: float, n_sub: int, temperature: float) -> float:
    """Return T log(1 + N_sub exp(-Delta/T)) for the soft-hard gap on a DAG.

    Requires temperature > 0 and delta, n_sub nonnegative; returns the theorem bound.
    """
    return theorem_iii_1_upper_bound(temperature, n_sub, delta)


def enumerate_path_costs(
    graph: nx.DiGraph,
    source: Any,
    target: Any,
    weight: str = "weight",
    max_paths: int | None = None,
) -> List[float]:
    costs: List[float] = []
    for i, path in enumerate(nx.all_simple_paths(graph, source, target)):
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            total += float(graph[u][v].get(weight, 1.0))
        costs.append(total)
        if max_paths is not None and i + 1 >= max_paths:
            break
    return costs


def compute_path_stats(
    graph: nx.DiGraph,
    source: Any,
    target: Any,
    weight: str = "weight",
) -> Dict[str, float | int]:
    """Compute d*, delta, N_tot, N_sub using full path enumeration (DAG)."""
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("compute_path_stats expects a DAG")

    costs = enumerate_path_costs(graph, source, target, weight=weight)
    if not costs:
        raise ValueError("No paths from source to target")

    costs_sorted = sorted(costs)
    d_star = costs_sorted[0]
    n_tot = len(costs_sorted)

    delta = float("inf")
    for c in costs_sorted[1:]:
        if c > d_star:
            delta = c - d_star
            break

    n_sub = sum(1 for c in costs_sorted if c > d_star)
    if delta == float("inf"):
        delta = 0.0
        n_sub = 0

    return {
        "d_star": float(d_star),
        "delta": float(delta),
        "n_tot": int(n_tot),
        "n_sub": int(n_sub),
    }
