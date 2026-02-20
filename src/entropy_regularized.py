from __future__ import annotations

from typing import Any, Dict, Tuple

import networkx as nx
import numpy as np
from scipy.special import logsumexp


def soft_shortest_path_dag(
    graph: nx.DiGraph,
    source: Any,
    target: Any,
    temperature: float,
    weight: str = "weight",
) -> Tuple[float, Dict[Any, float]]:
    """Compute soft shortest-path values on a DAG using log-sum-exp (equation 3)."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("soft_shortest_path_dag expects a DAG")

    topo = list(nx.topological_sort(graph))
    dT: Dict[Any, float] = {node: float("inf") for node in graph.nodes}
    dT[target] = 0.0

    for v in reversed(topo):
        if v == target:
            continue
        terms = []
        for _, u, data in graph.out_edges(v, data=True):
            if np.isfinite(dT[u]):
                w = float(data.get(weight, 1.0))
                terms.append(-(w + dT[u]) / temperature)
        if terms:
            dT[v] = -temperature * logsumexp(terms)

    return float(dT[source]), dT


def soft_shortest_path_values(
    graph: nx.DiGraph,
    target: Any,
    temperature: float,
    weight: str = "weight",
) -> Dict[Any, float]:
    """Return soft shortest-path values d_T(v) for all v to target."""
    source = target
    _, dT = soft_shortest_path_dag(graph, source, target, temperature, weight=weight)
    return dT
