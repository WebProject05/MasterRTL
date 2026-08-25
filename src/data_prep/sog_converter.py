"""Build a Simple Operator Graph (SOG) from a Yosys JSON AST.

The graph is represented as a NetworkX DiGraph with node features and a
lightweight analytic delay/toggle-rate model suitable for downstream PPA graph
feature extraction.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError(
        "networkx is required for the SOG builder. Install it with: "
        "python -m pip install networkx>=3.0 or choose the correct VS Code interpreter."
    ) from exc

VALID_OPERATORS = {"AND", "OR", "XOR", "NOT", "MUX", "DFF"}
OPERATOR_ALIASES = {
    "$_AND_": "AND",
    "$_OR_": "OR",
    "$_XOR_": "XOR",
    "$_NOT_": "NOT",
    "$mux": "MUX",
    "$dff": "DFF",
    "$adff": "DFF",
    "$sdff": "DFF",
    "$_MUX_": "MUX",
    "$eq": "XOR",
    "$_DFF_": "DFF",
    "$_DFFE_": "DFF",
    "$_DFFE_PN0P_": "DFF",
    "$_DFFE_PN0N_": "DFF",
    "$_DFF_PN0_": "DFF",
    "$_DFF_PN1_": "DFF",
    "$_DFF_PN0N_": "DFF",
    "$_DFF_N_": "DFF",
    "$_DFF_NN_": "DFF",
    "$_DFF_P_": "DFF",
    "$_DFF_PP_": "DFF",
}


def _normalize_op(cell_type: str | None) -> str:
    """Normalize a Yosys cell type into a supported SOG operator."""
    if cell_type is None:
        return "DFF"
    key = str(cell_type).strip()
    op = OPERATOR_ALIASES.get(key, key.upper())
    if op not in VALID_OPERATORS:
        raise ValueError(f"Unsupported cell type in SOG: {cell_type!r}")
    return op


def _is_register_like(cell_type: str | None) -> bool:
    """Return True for Yosys register cell variants that should not form combinational loops."""
    if cell_type is None:
        return False
    key = str(cell_type).strip()
    return key.startswith("$_DFF") or key.startswith("$_DFFE") or key.startswith("$dff")


def _extract_bit_width(connection_values: Any) -> int:
    """Return a bit-width estimate from a Yosys connection list."""
    if connection_values is None:
        return 1
    if isinstance(connection_values, list):
        return max(1, len(connection_values))
    if isinstance(connection_values, int):
        return 1
    if isinstance(connection_values, str):
        return 1
    return 1


def _build_node_features_for_cell(cell_name: str, cell: dict[str, Any]) -> dict[str, Any]:
    """Return the feature dictionary for a logical cell node."""
    op = _normalize_op(cell.get("type"))
    connections = cell.get("connections", {})
    input_ports = [port for port in connections if port.upper() != "Y"]
    input_bits = []
    for port in input_ports:
        vals = connections.get(port, [])
        if isinstance(vals, list):
            input_bits.extend([int(v) for v in vals if isinstance(v, int)])
    fan_in = len(input_bits) if input_bits else 1
    width = max(1, _extract_bit_width(connections.get("Y")))
    return {
        "node_id": cell_name,
        "op": op,
        "bit_width": width,
        "fan_in": fan_in,
        "fan_out": 0,
        "operator_one_hot": {op_name: 1.0 if op_name == op else 0.0 for op_name in sorted(VALID_OPERATORS)},
    }


def _iter_net_inputs(module: dict[str, Any]) -> dict[int, list[int]]:
    """Collect all source net indices driving each net in the module."""
    net_inputs: dict[int, list[int]] = defaultdict(list)

    for cell in module.get("cells", {}).values():
        connections = cell.get("connections", {})
        for port_name, values in connections.items():
            if port_name.upper() == "Y":
                continue
            if isinstance(values, list):
                for val in values:
                    if isinstance(val, int):
                        net_inputs[int(val)].append(int(values[0]) if values and isinstance(values[0], int) else int(val))
    return net_inputs


def build_sog_from_json(json_path: str | Path) -> nx.DiGraph:
    """Build a directed SOG from a Yosys JSON AST file.

    Nodes are net indices (ints) and cell outputs. For each cell, the cell output
    net becomes a node, while input nets are connected into it with edges.
    """
    source = Path(json_path)
    with source.open("r", encoding="utf-8") as handle:
        ast = json.load(handle)

    graph = nx.DiGraph()

    modules = ast.get("modules", {})
    if not modules:
        raise ValueError(f"No modules found in Yosys JSON: {source}")

    module = next(iter(modules.values()))
    ports = module.get("ports", {})
    cells = module.get("cells", {})

    for port_name, port_def in ports.items():
        bits = port_def.get("bits", [])
        for bit in bits:
            graph.add_node(int(bit), op="IO", bit_width=1, fan_in=0, fan_out=0, operator_one_hot={op: 0.0 for op in sorted(VALID_OPERATORS)})

    for cell in cells.values():
        op = _normalize_op(cell.get("type"))
        connections = cell.get("connections", {})
        output_bits = connections.get("Y", connections.get("Q", []))
        if not isinstance(output_bits, list) or not output_bits:
            continue

        for output_bit in output_bits:
            output_bit = int(output_bit)
            features = {
                "node_id": output_bit,
                "op": op,
                "bit_width": max(1, _extract_bit_width(output_bits)),
                "fan_in": 0,
                "fan_out": 0,
                "operator_one_hot": {op_name: 1.0 if op_name == op else 0.0 for op_name in sorted(VALID_OPERATORS)},
            }
            if output_bit in graph:
                graph.nodes[output_bit].update(features)
            else:
                graph.add_node(output_bit, **features)

            for port_name, values in connections.items():
                port_key = port_name.upper()
                if port_key in {"Y", "Q"}:
                    continue
                if _is_register_like(cell.get("type")) and port_key in {"D", "C", "CLK", "E", "EN", "R", "RESET", "S", "SET"}:
                    continue
                if not isinstance(values, list):
                    continue
                for value in values:
                    if isinstance(value, int):
                        src = int(value)
                        if src not in graph:
                            graph.add_node(src, op="NET", bit_width=1, fan_in=0, fan_out=0, operator_one_hot={op_name: 0.0 for op_name in sorted(VALID_OPERATORS)})
                        graph.add_edge(src, output_bit)

    # add DFF nodes for register assignments when present
    for node in list(graph.nodes):
        if graph.nodes[node].get("op") in {"AND", "OR", "XOR", "NOT", "MUX"}:
            graph.nodes[node]["fan_in"] = len(list(graph.predecessors(node)))
            graph.nodes[node]["fan_out"] = len(list(graph.successors(node)))

    # derive fan-in/out for all nodes
    for node in list(graph.nodes):
        graph.nodes[node]["fan_in"] = len(list(graph.predecessors(node)))
        graph.nodes[node]["fan_out"] = len(list(graph.successors(node)))

    return graph


def _topological_order(graph: nx.DiGraph) -> list[Any]:
    """Return a topological order for the graph."""
    return list(nx.topological_sort(graph))


def compute_node_delays(graph: nx.DiGraph) -> dict[Any, float]:
    """Compute an analytical delay score using a linear fan-out RC approximation."""
    delays: dict[Any, float] = {}
    for node in _topological_order(graph):
        fan_out = float(graph.nodes[node].get("fan_out", 0))
        op = str(graph.nodes[node].get("op", "NET"))
        base = {"AND": 1.0, "OR": 1.1, "XOR": 1.2, "NOT": 0.8, "MUX": 1.6, "DFF": 2.4, "NET": 0.1, "IO": 0.0}.get(op, 1.0)
        delays[node] = base + 0.15 * fan_out
    return delays


def compute_path_features(graph: nx.DiGraph | dict[str, Any]) -> list[dict[str, Any]]:
    """Compute maximum-delay combinational paths between register pairs.

    Returns a list of path summaries, storing start, end, delay, and path nodes.
    """
    if isinstance(graph, dict):
        graph_obj = nx.DiGraph()
        graph_obj.add_nodes_from(graph.get("nodes", {}).items())
        graph_obj.add_edges_from(graph.get("edges", []))
    else:
        graph_obj = graph

    delays = compute_node_delays(graph_obj)
    reg_nodes = [n for n, attrs in graph_obj.nodes(data=True) if attrs.get("op") == "DFF"]
    paths: list[dict[str, Any]] = []

    for start in reg_nodes:
        for end in reg_nodes:
            if start == end:
                continue
            try:
                path = nx.shortest_path(graph_obj, source=start, target=end, weight=lambda u, v, d: 1.0)
            except nx.NetworkXNoPath:
                continue
            path_delay = sum(delays.get(node, 0.0) for node in path)
            paths.append({
                "start": start,
                "end": end,
                "delay": path_delay,
                "path": path,
            })

    return sorted(paths, key=lambda p: p["delay"], reverse=True)


def propagate_toggle_rates(graph: nx.DiGraph, start_rates: dict[Any, float]) -> dict[Any, float]:
    """Propagate toggle rates through the logic graph using a simple Boolean model."""
    rates = {node: float(start_rates.get(node, 0.0)) for node in graph.nodes}
    for node in _topological_order(graph):
        if node in start_rates:
            continue
        inputs = list(graph.predecessors(node))
        if not inputs:
            rates[node] = 0.0
            continue
        op = graph.nodes[node].get("op", "NET")
        if op == "NOT":
            rates[node] = rates[inputs[0]]
        elif op in {"AND", "OR", "XOR"}:
            if op in {"AND", "OR"}:
                rates[node] = 0.5 * sum(rates[i] for i in inputs) / max(1, len(inputs))
            else:
                rates[node] = sum(rates[i] for i in inputs) / max(1, len(inputs))
        else:
            rates[node] = sum(rates[i] for i in inputs) / max(1, len(inputs))
    return rates


def save_graph_artifacts(graph: nx.DiGraph, output_dir: str | Path, name: str) -> dict[str, Path]:
    """Persist the SOG graph and extracted features to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = output_dir / f"{name}.pkl"
    feature_path = output_dir / f"{name}_features.pkl"

    with graph_path.open("wb") as handle:
        pickle.dump(graph, handle)

    path_features = compute_path_features(graph)
    with feature_path.open("wb") as handle:
        pickle.dump(path_features, handle)

    return {"graph": graph_path, "features": feature_path}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Construct a Simple Operator Graph from a Yosys JSON AST.")
    parser.add_argument("json_path", type=Path, help="Input Yosys JSON file.")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("data/sog_graphs"), help="Directory for output artifacts.")
    parser.add_argument("--name", default=None, help="Base name for output files.")
    args = parser.parse_args()

    source = Path(args.json_path).expanduser().resolve()
    graph = build_sog_from_json(source)
    output_name = args.name or source.stem
    saved = save_graph_artifacts(graph, args.output_dir, output_name)

    print(f"SOG built for {source}")
    print(f"Graph: {saved['graph']}")
    print(f"Features: {saved['features']}")


if __name__ == "__main__":
    main()
