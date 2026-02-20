from __future__ import annotations

import math

import networkx as nx
import numpy as np

from src.bounds import enumerate_path_costs
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


def test_partition_equivalence() -> None:
    graph = build_test_dag()
    costs = enumerate_path_costs(graph, "s", "t")
    assert costs

    for temperature in [0.3, 0.7, 1.5]:
        z = float(np.sum(np.exp(-np.array(costs) / temperature)))
        dT_bruteforce = -temperature * math.log(z)
        dT_dp, _ = soft_shortest_path_dag(graph, "s", "t", temperature)
        assert abs(dT_bruteforce - dT_dp) <= 1e-9
