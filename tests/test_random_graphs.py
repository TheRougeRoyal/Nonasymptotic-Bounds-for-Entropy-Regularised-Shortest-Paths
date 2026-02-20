from __future__ import annotations

import networkx as nx
import numpy as np

from src.bounds import compute_path_stats, theorem_iii_1_upper_bound
from src.classical_shortest_path import dijkstra_shortest_path_length
from src.entropy_regularized import soft_shortest_path_dag


def generate_random_dag(
    rng: np.random.Generator,
    min_nodes: int = 5,
    max_nodes: int = 15,
    min_prob: float = 0.15,
    max_prob: float = 0.6,
    min_cost: float = 0.1,
    max_cost: float = 2.0,
    max_attempts: int = 100,
) -> nx.DiGraph:
    for _ in range(max_attempts):
        n = int(rng.integers(min_nodes, max_nodes + 1))
        p = float(rng.uniform(min_prob, max_prob))
        graph = nx.DiGraph()
        graph.add_nodes_from(range(n))
        for i in range(n - 1):
            for j in range(i + 1, n):
                if rng.random() < p:
                    w = float(rng.uniform(min_cost, max_cost))
                    graph.add_edge(i, j, weight=w)
        if nx.has_path(graph, 0, n - 1):
            return graph
    raise RuntimeError("Failed to generate a connected DAG with a path from source to sink")


def test_random_dag_theorem_bound() -> None:
    rng = np.random.default_rng(0)
    for _ in range(100):
        graph = generate_random_dag(rng)
        source = 0
        target = max(graph.nodes)

        stats = compute_path_stats(graph, source, target)
        d_star = float(stats["d_star"])
        delta = float(stats["delta"])
        n_sub = int(stats["n_sub"])

        temperature = float(rng.uniform(0.2, 2.0))
        dT, _ = soft_shortest_path_dag(graph, source, target, temperature)

        d_star_classical = dijkstra_shortest_path_length(graph, source, target)
        assert abs(d_star_classical - d_star) <= 1e-12

        gap = d_star - dT
        bound = theorem_iii_1_upper_bound(temperature, n_sub, delta)
        assert gap <= bound + 1e-10
