from __future__ import annotations

import os
import sys
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.bounds import compute_path_stats, theorem_iii_1_upper_bound
from src.classical_shortest_path import dijkstra_shortest_path, dijkstra_shortest_path_length
from src.entropy_regularized import soft_shortest_path_dag
from src.graph import load_dag_from_json


def _results_path(filename: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", filename))


def plot_gap_vs_temperature(graph: nx.DiGraph, source: str, target: str) -> None:
    stats = compute_path_stats(graph, source, target)
    d_star = stats["d_star"]
    delta = stats["delta"]
    n_sub = stats["n_sub"]

    temps = np.logspace(-2, 1, 60)
    gaps = []
    bounds = []

    for T in temps:
        dT, _ = soft_shortest_path_dag(graph, source, target, T)
        gap = d_star - dT
        gaps.append(gap)
        bounds.append(theorem_iii_1_upper_bound(T, n_sub, delta))

    plt.figure(figsize=(6, 4))
    plt.semilogy(temps, gaps, label="d*(s) - d_T(s)")
    plt.semilogy(temps, bounds, label="Theorem III.1 bound", linestyle="--")
    plt.xlabel("Temperature T")
    plt.ylabel("Gap (log scale)")
    plt.title("Exponential convergence as T → 0+")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_results_path("temperature_gap.png"))


def plot_exponential_convergence(graph: nx.DiGraph, source: str, target: str) -> None:
    stats = compute_path_stats(graph, source, target)
    d_star = stats["d_star"]

    temps = np.logspace(-3, -0.3, 60)
    gaps = []
    for T in temps:
        dT, _ = soft_shortest_path_dag(graph, source, target, T)
        gaps.append(d_star - dT)

    plt.figure(figsize=(6, 4))
    plt.semilogy(1.0 / temps, gaps)
    plt.xlabel("1 / T")
    plt.ylabel("Gap (log scale)")
    plt.title("Exponential decay vs 1/T")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(_results_path("exponential_convergence.png"))


def plot_classical_vs_soft(graph: nx.DiGraph, source: str, target: str, temperature: float) -> None:
    d_star = dijkstra_shortest_path_length(graph, source, target)
    dT, dT_all = soft_shortest_path_dag(graph, source, target, temperature)

    pos = nx.spring_layout(graph, seed=7)
    labels = {}
    for node in graph.nodes:
        hard = dijkstra_shortest_path_length(graph, node, target)
        soft = dT_all.get(node, float("inf"))
        labels[node] = f"{node}\n d*={hard:.2f}\n dT={soft:.2f}"

    plt.figure(figsize=(7, 4.5))
    nx.draw_networkx(graph, pos=pos, node_color="#DDE7FF", node_size=900, labels=labels)
    edge_labels = {(u, v): f"{data.get('weight', 1.0):.2f}" for u, v, data in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels, font_color="#444")
    plt.title(f"Classical vs Soft Shortest Paths (T={temperature})\n d*(s)={d_star:.3f}, dT(s)={dT:.3f}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(_results_path("classical_vs_soft.png"))


def main() -> None:
    dag = load_dag_from_json(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_dag.json")))
    graph = dag.to_networkx()
    source = dag.source
    target = dag.sink

    if source is None or target is None:
        raise ValueError("sample_dag.json must define source and sink")

    plot_gap_vs_temperature(graph, source, target)
    plot_exponential_convergence(graph, source, target)
    plot_classical_vs_soft(graph, source, target, temperature=0.5)


if __name__ == "__main__":
    main()
