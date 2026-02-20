from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bounds import compute_path_stats, theorem_iii_1_upper_bound
from src.entropy_regularized import soft_shortest_path_dag


def _results_path(filename: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", filename))


def build_two_path_dag(delta: float) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_edge("s", "a", weight=1.0)
    graph.add_edge("a", "t", weight=1.0)
    graph.add_edge("s", "b", weight=1.0)
    graph.add_edge("b", "t", weight=1.0 + delta)
    return graph


def main() -> None:
    temps = 0.5
    deltas = np.linspace(0.05, 2.0, 40)

    gaps = []
    bounds = []

    for delta in deltas:
        graph = build_two_path_dag(float(delta))
        stats = compute_path_stats(graph, "s", "t")
        d_star = stats["d_star"]
        n_sub = stats["n_sub"]

        dT, _ = soft_shortest_path_dag(graph, "s", "t", temps)
        gaps.append(d_star - dT)
        bounds.append(theorem_iii_1_upper_bound(temps, n_sub, float(delta)))

    plt.figure(figsize=(6, 4))
    plt.plot(deltas, gaps, label="d*(s) - d_T(s)")
    plt.plot(deltas, bounds, linestyle="--", label="Theorem III.1 bound")
    plt.xlabel("Cost margin Δ")
    plt.ylabel("Gap")
    plt.title("Sensitivity to Δ")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_results_path("cost_margin.png"))


if __name__ == "__main__":
    main()
