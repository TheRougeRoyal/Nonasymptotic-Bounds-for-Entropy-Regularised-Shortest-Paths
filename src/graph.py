from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx


@dataclass(frozen=True)
class Edge:
    u: Any
    v: Any
    weight: float


class DAG:
    """Finite directed acyclic graph wrapper."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.source: Optional[Any] = None
        self.sink: Optional[Any] = None

    def add_node(self, node: Any) -> None:
        self.graph.add_node(node)

    def add_edge(self, u: Any, v: Any, weight: float) -> None:
        self.graph.add_edge(u, v, weight=float(weight))

    def set_source_sink(self, source: Any, sink: Any) -> None:
        self.source = source
        self.sink = sink

    def validate_acyclic(self) -> None:
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Graph must be a DAG.")

    def topological_sort(self) -> List[Any]:
        self.validate_acyclic()
        return list(nx.topological_sort(self.graph))

    def edges(self) -> List[Edge]:
        return [Edge(u, v, data.get("weight", 1.0)) for u, v, data in self.graph.edges(data=True)]

    def nodes(self) -> List[Any]:
        return list(self.graph.nodes)

    def to_networkx(self) -> nx.DiGraph:
        return self.graph

    def path_cost(self, path: Iterable[Any]) -> float:
        total = 0.0
        nodes = list(path)
        for i in range(len(nodes) - 1):
            total += float(self.graph[nodes[i]][nodes[i + 1]].get("weight", 1.0))
        return total


def load_dag_from_json(path: str) -> DAG:
    """Load a DAG from a JSON file with fields: nodes, edges (u, v, weight), source, sink."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dag = DAG()
    for node in data.get("nodes", []):
        dag.add_node(node)
    for u, v, w in data.get("edges", []):
        dag.add_edge(u, v, w)

    source = data.get("source")
    sink = data.get("sink")
    if source is not None and sink is not None:
        dag.set_source_sink(source, sink)

    dag.validate_acyclic()
    return dag
