from __future__ import annotations

import math

import networkx as nx
import numpy as np

from src.bounds import compute_path_stats, enumerate_path_costs, theorem_iii_1_upper_bound
from src.classical_shortest_path import shortest_path_cost
from src.entropy_regularized import soft_shortest_path, soft_shortest_path_dag
from src.graph import load_dag_from_json


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


def test_partition_equivalence_small_dag() -> None:
    graph = build_test_dag()
    costs = enumerate_path_costs(graph, "s", "t")
    temperature = 0.7
    z = float(np.sum(np.exp(-np.array(costs) / temperature)))
    dT_bruteforce = -temperature * math.log(z)
    dT_dp, _ = soft_shortest_path(graph, "s", "t", temperature)
    assert abs(dT_bruteforce - dT_dp) <= 1e-9


def test_small_temperature_convergence_sample_dag() -> None:
    dag = load_dag_from_json("data/sample_dag.json")
    graph = dag.to_networkx()
    assert dag.source is not None
    assert dag.sink is not None
    d_star = shortest_path_cost(graph, dag.source, dag.sink)
    dT, _ = soft_shortest_path(graph, dag.source, dag.sink, 1e-4)
    assert abs(dT - d_star) <= 1e-6


def test_single_path_dag_soft_equals_hard() -> None:
    graph = nx.DiGraph()
    graph.add_edge("s", "a", weight=1.2)
    graph.add_edge("a", "t", weight=0.7)
    d_star = shortest_path_cost(graph, "s", "t")
    dT, _ = soft_shortest_path(graph, "s", "t", 0.3)
    assert abs(dT - d_star) <= 1e-12
