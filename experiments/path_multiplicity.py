from __future__ import annotations

import os
import sys
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bounds import compute_path_stats, theorem_iii_1_upper_bound
from src.entropy_regularized import soft_shortest_path_dag


def _results_path(filename: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", filename))


def build_parallel_dag(n_paths: int, delta: float = 0.2) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("s")
    graph.add_node("t")
    for i in range(n_paths):
        mid = f"m{i}"
        graph.add_edge("s", mid, weight=1.0)
        graph.add_edge(mid, "t", weight=1.0 + i * delta)
    return graph


def main() -> None:
    temps = 0.5
    n_paths_list = list(range(2, 16))

    gaps = []
    bounds = []

    for n_paths in n_paths_list:
        graph = build_parallel_dag(n_paths, delta=0.3)
        stats = compute_path_stats(graph, "s", "t")
        d_star = stats["d_star"]
        delta = stats["delta"]
        n_sub = stats["n_sub"]

        dT, _ = soft_shortest_path_dag(graph, "s", "t", temps)
        gaps.append(d_star - dT)
        bounds.append(theorem_iii_1_upper_bound(temps, n_sub, delta))

    plt.figure(figsize=(6, 4))
    plt.plot(n_paths_list, gaps, marker="o", label="d*(s) - d_T(s)")
    plt.plot(n_paths_list, bounds, marker="s", linestyle="--", label="Theorem III.1 bound")
    plt.xlabel("Number of paths N_tot")
    plt.ylabel("Gap")
    plt.title("Effect of path multiplicity")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_results_path("path_multiplicity.png"))


if __name__ == "__main__":
    main()
