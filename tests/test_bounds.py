from __future__ import annotations

import networkx as nx

from src.bounds import compute_path_stats, theorem_iii_1_upper_bound
from src.entropy_regularized import soft_shortest_path_dag


def build_test_dag() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_edge("s", "a", weight=1.0)
    graph.add_edge("s", "b", weight=1.1)
    graph.add_edge("a", "t", weight=1.0)
    graph.add_edge("b", "t", weight=0.9)
    graph.add_edge("a", "c", weight=0.8)
    graph.add_edge("c", "t", weight=0.7)
    return graph


def test_theorem_iii_1_bound() -> None:
    graph = build_test_dag()
    stats = compute_path_stats(graph, "s", "t")
    d_star = stats["d_star"]
    delta = stats["delta"]
    n_sub = stats["n_sub"]

    for T in [0.2, 0.5, 1.0]:
        dT, _ = soft_shortest_path_dag(graph, "s", "t", T)
        gap = d_star - dT
        bound = theorem_iii_1_upper_bound(T, n_sub, delta)
        assert gap <= bound + 1e-8
