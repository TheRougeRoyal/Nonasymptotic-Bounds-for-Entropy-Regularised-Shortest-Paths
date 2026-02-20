from __future__ import annotations

from typing import Any, Dict, List, Tuple

import networkx as nx


def dijkstra_shortest_path_length(
    graph: nx.DiGraph,
    source: Any,
    target: Any,
    weight: str = "weight",
) -> float:
    return float(nx.dijkstra_path_length(graph, source, target, weight=weight))


def bellman_ford_shortest_path_length(
    graph: nx.DiGraph,
    source: Any,
    target: Any,
    weight: str = "weight",
) -> float:
    return float(nx.bellman_ford_path_length(graph, source, target, weight=weight))


def dijkstra_shortest_path(
    graph: nx.DiGraph,
    source: Any,
    target: Any,
    weight: str = "weight",
) -> List[Any]:
    return list(nx.dijkstra_path(graph, source, target, weight=weight))


def bellman_ford_shortest_path(
    graph: nx.DiGraph,
    source: Any,
    target: Any,
    weight: str = "weight",
) -> List[Any]:
    return list(nx.bellman_ford_path(graph, source, target, weight=weight))
