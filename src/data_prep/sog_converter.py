"""Build a Simple Operator Graph (SOG) from a Yosys JSON AST.

The graph is represented as a NetworkX DiGraph with single-bit operators:
- Five primitive single-bit logic operations: AND, OR, XOR, NOT, MUX
- Single-bit registers: DFF
- Primary Inputs (PI) and Primary Outputs (PO)

Includes:
- Analytical node delay modeling (linear fan-out RC approximation)
- Static Timing Analysis (STA) DAG partitioning for critical path extraction (P^R_{i->j})
- Boolean toggle-rate propagation
- Comprehensive SOG graph statistics extraction and recording
"""

from __future__ import annotations

import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError(
        "networkx is required for the SOG builder. Install it with: "
        "python -m pip install networkx>=3.0"
    ) from exc

VALID_OPERATORS = {"AND", "OR", "XOR", "NOT", "MUX", "DFF"}
OPERATOR_ALIASES = {
    "$_AND_": "AND",
    "$_OR_": "OR",
    "$_XOR_": "XOR",
    "$_NOT_": "NOT",
    "$mux": "MUX",
    "$_MUX_": "MUX",
    "$dff": "DFF",
    "$adff": "DFF",
    "$sdff": "DFF",
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

# Standard cell approximate area (NanGate 45nm standard cell library references in um^2)
AREA_WEIGHTS = {
    "DFF": 4.522,  # DFF_X1
    "AND": 1.064,  # AND2_X1
    "OR": 1.064,   # OR2_X1
    "XOR": 1.596,  # XOR2_X1
    "NOT": 0.532,  # INV_X1
    "MUX": 1.862,  # MUX2_X1
    "NET": 0.0,
    "IO": 0.0,
}

BASE_DELAYS = {
    "AND": 1.0,
    "OR": 1.1,
    "XOR": 1.2,
    "NOT": 0.8,
    "MUX": 1.6,
    "DFF": 2.4,
    "NET": 0.1,
    "IO": 0.0,
}


def _normalize_op(cell_type: str | None) -> str:
    """Normalize a Yosys cell type into a supported SOG operator."""
    if cell_type is None:
        return "DFF"
    key = str(cell_type).strip()
    if key in OPERATOR_ALIASES:
        return OPERATOR_ALIASES[key]
    if key.startswith("$_DFF") or key.startswith("$dff"):
        return "DFF"
    op = key.upper()
    if op not in VALID_OPERATORS:
        raise ValueError(f"Unsupported cell type in SOG: {cell_type!r}")
    return op


def _is_register_like(cell_type: str | None) -> bool:
    """Return True for Yosys register cell variants."""
    if cell_type is None:
        return False
    key = str(cell_type).strip()
    return key.startswith("$_DFF") or key.startswith("$_DFFE") or key.startswith("$dff") or key.startswith("$adff") or key.startswith("$sdff")


def _extract_bit_width(connection_values: Any) -> int:
    """Return a bit-width estimate from a Yosys connection list."""
    if connection_values is None:
        return 1
    if isinstance(connection_values, list):
        return max(1, len(connection_values))
    if isinstance(connection_values, (int, str)):
        return 1
    return 1


def build_sog_from_json(json_path: str | Path) -> nx.DiGraph:
    """Build a directed Simple Operator Graph (SOG) from a Yosys JSON AST file.

    Nodes represent:
    - Primary IO ports (op="IO")
    - Single-bit logic operations (op in {"AND", "OR", "XOR", "NOT", "MUX"})
    - Single-bit registers (op="DFF")

    Maintains full structural connectivity while storing pin references for STA.
    """
    source = Path(json_path)
    with source.open("r", encoding="utf-8") as handle:
        ast = json.load(handle)

    graph = nx.DiGraph()

    modules = ast.get("modules", {})
    if not modules:
        raise ValueError(f"No modules found in Yosys JSON: {source}")

    module_name = next(iter(modules.keys()))
    module = modules[module_name]
    ports = module.get("ports", {})
    cells = module.get("cells", {})

    graph.graph["module_name"] = module_name
    primary_inputs: list[int] = []
    primary_outputs: list[int] = []
    registers: list[int] = []

    # 1. Add IO port nodes
    for port_name, port_def in ports.items():
        direction = port_def.get("direction", "")
        bits = port_def.get("bits", [])
        for bit in bits:
            if isinstance(bit, int):
                node_id = bit
                if direction == "input":
                    primary_inputs.append(node_id)
                elif direction == "output":
                    primary_outputs.append(node_id)

                if node_id not in graph:
                    graph.add_node(
                        node_id,
                        node_id=node_id,
                        name=f"{port_name}[{bit}]",
                        op="IO",
                        io_type=direction,
                        bit_width=1,
                        fan_in=0,
                        fan_out=0,
                        operator_one_hot={op: 0.0 for op in sorted(VALID_OPERATORS)},
                    )

    # 2. Add combinational and sequential cells
    dff_conns: list[tuple[str, str, dict[str, Any]]] = []

    for cell_name, cell in cells.items():
        cell_type = cell.get("type")
        op = _normalize_op(cell_type)
        connections = cell.get("connections", {})

        if _is_register_like(cell_type):
            dff_conns.append((cell_name, op, connections))
            continue

        output_bits = connections.get("Y", connections.get("Q", []))
        if not isinstance(output_bits, list) or not output_bits:
            continue

        for output_bit in output_bits:
            if not isinstance(output_bit, int):
                continue
            output_bit = int(output_bit)
            features = {
                "node_id": output_bit,
                "cell_name": cell_name,
                "op": op,
                "bit_width": 1,
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
                if not isinstance(values, list):
                    continue
                for value in values:
                    if isinstance(value, int):
                        src = int(value)
                        if src not in graph:
                            graph.add_node(
                                src,
                                node_id=src,
                                op="NET",
                                bit_width=1,
                                fan_in=0,
                                fan_out=0,
                                operator_one_hot={op_name: 0.0 for op_name in sorted(VALID_OPERATORS)},
                            )
                        graph.add_edge(src, output_bit)

    # 3. Process DFF registers
    for cell_name, op, connections in dff_conns:
        q_bits = connections.get("Q", [])
        d_bits = connections.get("D", [])
        clk_bits = connections.get("C", connections.get("CLK", []))

        if not isinstance(q_bits, list) or not q_bits:
            continue

        for idx, q_bit in enumerate(q_bits):
            if not isinstance(q_bit, int):
                continue
            d_bit = d_bits[idx] if isinstance(d_bits, list) and idx < len(d_bits) and isinstance(d_bits[idx], int) else None
            clk_bit = clk_bits[0] if isinstance(clk_bits, list) and clk_bits and isinstance(clk_bits[0], int) else None

            registers.append(q_bit)
            reg_features = {
                "node_id": q_bit,
                "cell_name": cell_name,
                "op": "DFF",
                "bit_width": 1,
                "fan_in": 0,
                "fan_out": 0,
                "d_pin": d_bit,
                "q_pin": q_bit,
                "clk_pin": clk_bit,
                "operator_one_hot": {op_name: 1.0 if op_name == "DFF" else 0.0 for op_name in sorted(VALID_OPERATORS)},
            }
            if q_bit in graph:
                graph.nodes[q_bit].update(reg_features)
            else:
                graph.add_node(q_bit, **reg_features)

            # Connect data input D to register node
            if d_bit is not None:
                if d_bit not in graph:
                    graph.add_node(
                        d_bit,
                        node_id=d_bit,
                        op="NET",
                        bit_width=1,
                        fan_in=0,
                        fan_out=0,
                        operator_one_hot={op_name: 0.0 for op_name in sorted(VALID_OPERATORS)},
                    )
                graph.add_edge(d_bit, q_bit, is_sequential=True, port="D")

    # 4. Calculate fan-in and fan-out for all nodes
    for node in graph.nodes:
        graph.nodes[node]["fan_in"] = graph.in_degree(node)
        graph.nodes[node]["fan_out"] = graph.out_degree(node)

    graph.graph["primary_inputs"] = list(set(primary_inputs))
    graph.graph["primary_outputs"] = list(set(primary_outputs))
    graph.graph["registers"] = list(set(registers))

    return graph


def compute_node_delays(graph: nx.DiGraph | dict[str, Any]) -> dict[Any, float]:
    """Compute analytical node delays using a linear fan-out RC approximation.

    Formula: delay(node) = base_delay(op) + 0.15 * fan_out
    """
    delays: dict[Any, float] = {}

    if isinstance(graph, dict):
        nodes_dict = graph.get("nodes", {})
        for node_id, attrs in nodes_dict.items():
            op = attrs.get("op", "NET")
            fan_out = float(attrs.get("fan_out", 0))
            base = BASE_DELAYS.get(op, 1.0)
            delays[node_id] = base + 0.15 * fan_out
        return delays

    for node in graph.nodes:
        op = str(graph.nodes[node].get("op", "NET"))
        fan_out = float(graph.nodes[node].get("fan_out", graph.out_degree(node)))
        base = BASE_DELAYS.get(op, 1.0)
        delay = base + 0.15 * fan_out
        graph.nodes[node]["delay"] = delay
        delays[node] = delay

    return delays


def build_timing_dag(graph: nx.DiGraph | dict[str, Any]) -> tuple[nx.DiGraph, dict[str, Any]]:
    """Construct an acyclic timing graph (DAG) suitable for Static Timing Analysis (STA).

    Registers are decoupled into launch points (Q) and capture points (D),
    preventing cyclic feedback loops across clock cycles while enabling
    exact register-to-register and IO path evaluation.
    """
    dag = nx.DiGraph()

    if isinstance(graph, dict):
        nodes_dict = graph.get("nodes", {})
        edges_list = graph.get("edges", [])
        for n, attrs in nodes_dict.items():
            dag.add_node(n, **attrs)
        for u, v in edges_list:
            dag.add_edge(u, v)
        metadata = {
            "registers": [n for n, a in nodes_dict.items() if a.get("op") == "DFF"],
            "launch_nodes": [n for n, a in nodes_dict.items() if a.get("op") == "DFF"],
            "capture_nodes": [n for n, a in nodes_dict.items() if a.get("op") == "DFF"],
        }
        return dag, metadata

    delays = compute_node_delays(graph)

    # Add all combinational & IO nodes
    for node, attrs in graph.nodes(data=True):
        if attrs.get("op") != "DFF":
            node_data = dict(attrs)
            node_data["delay"] = delays.get(node, 1.0)
            dag.add_node(node, **node_data)

    launch_nodes: dict[Any, str] = {}
    capture_nodes: dict[Any, str] = {}

    # Create launch and capture nodes for each register
    for node, attrs in graph.nodes(data=True):
        if attrs.get("op") == "DFF":
            q_pin = attrs.get("q_pin", node)
            d_pin = attrs.get("d_pin")

            launch_id = f"reg_launch_{q_pin}"
            capture_id = f"reg_capture_{q_pin}"

            launch_nodes[q_pin] = launch_id
            capture_nodes[q_pin] = capture_id

            dag.add_node(launch_id, op="DFF_LAUNCH", reg_id=node, q_pin=q_pin, delay=delays.get(node, 2.4))
            dag.add_node(capture_id, op="DFF_CAPTURE", reg_id=node, d_pin=d_pin, delay=0.5)

    # Add combinational edges and register boundary edges
    for u, v, data in graph.edges(data=True):
        u_is_reg = graph.nodes[u].get("op") == "DFF"
        v_is_reg = graph.nodes[v].get("op") == "DFF"

        if u_is_reg and v_is_reg:
            # Direct register-to-register connection
            dag.add_edge(launch_nodes[u], capture_nodes[v])
        elif u_is_reg:
            # Register launch drives combinational logic
            dag.add_edge(launch_nodes[u], v)
        elif v_is_reg:
            # Combinational logic drives register capture (D input)
            dag.add_edge(u, capture_nodes[v])
        else:
            # Combinational edge
            dag.add_edge(u, v)

    metadata = {
        "registers": list(launch_nodes.keys()),
        "launch_nodes": list(launch_nodes.values()),
        "capture_nodes": list(capture_nodes.values()),
        "launch_map": launch_nodes,
        "capture_map": capture_nodes,
    }
    return dag, metadata


def compute_path_features(graph: nx.DiGraph | dict[str, Any]) -> list[dict[str, Any]]:
    """Compute maximum-delay combinational paths between register pairs.

    Returns a sorted list of path summaries (longest/most critical paths first),
    each containing start register, end register, accumulated delay, hop count,
    operator breakdown, and node sequence.
    """
    # Fallback for manual unit-test dictionaries
    if isinstance(graph, dict):
        g = nx.DiGraph()
        nodes = graph.get("nodes", {})
        for n, attrs in nodes.items():
            g.add_node(n, **attrs)
        for u, v in graph.get("edges", []):
            g.add_edge(u, v)
        delays = compute_node_delays(graph)
        reg_nodes = [n for n, attrs in nodes.items() if attrs.get("op") == "DFF"]
        paths: list[dict[str, Any]] = []
        for start in reg_nodes:
            for end in reg_nodes:
                if start == end:
                    continue
                try:
                    p = nx.shortest_path(g, source=start, target=end)
                except nx.NetworkXNoPath:
                    continue
                p_delay = sum(delays.get(n, 0.0) for n in p)
                paths.append({"start": start, "end": end, "delay": p_delay, "path": p, "length": len(p)})
        return sorted(paths, key=lambda x: x["delay"], reverse=True)

    dag, meta = build_timing_dag(graph)
    delays = {n: float(dag.nodes[n].get("delay", 1.0)) for n in dag.nodes}

    # If graph is not a DAG due to unmapped cycles, fallback to condensation/DAG view
    if not nx.is_directed_acyclic_graph(dag):
        cycles = list(nx.simple_cycles(dag))
        for cycle in cycles:
            if len(cycle) >= 2:
                dag.remove_edge(cycle[-1], cycle[0])

    topological_order = list(nx.topological_sort(dag))
    node_to_idx = {n: i for i, n in enumerate(topological_order)}

    launch_set = set(meta["launch_nodes"])
    # Include Primary Inputs as path launch points
    pi_nodes = graph.graph.get("primary_inputs", []) if hasattr(graph, "graph") else []
    for pi in pi_nodes:
        if pi in dag:
            launch_set.add(pi)

    capture_set = set(meta["capture_nodes"])
    # If circuit has no registers (pure combinational), include Primary Outputs as capture endpoints
    if not capture_set:
        po_nodes = graph.graph.get("primary_outputs", []) if hasattr(graph, "graph") else []
        for po in po_nodes:
            if po in dag:
                capture_set.add(po)

    # For each capture register, find the longest path from any launch register
    paths: list[dict[str, Any]] = []

    # Dynamic programming for longest path arrival times
    dist: dict[Any, float] = {}
    parent: dict[Any, Any] = {}
    origin: dict[Any, Any] = {}

    for node in topological_order:
        if node in launch_set:
            dist[node] = delays.get(node, 2.4)
            origin[node] = node
            parent[node] = None
        else:
            max_d = -1.0
            best_p = None
            best_orig = None
            for pred in dag.predecessors(node):
                if pred in dist:
                    cand_d = dist[pred] + delays.get(node, 1.0)
                    if cand_d > max_d:
                        max_d = cand_d
                        best_p = pred
                        best_orig = origin[pred]
            if best_p is not None:
                dist[node] = max_d
                parent[node] = best_p
                origin[node] = best_orig

    # Extract critical path for each endpoint
    for cap in capture_set:
        if cap in dist and origin[cap] is not None:
            # Backtrack path
            curr = cap
            path_nodes: list[Any] = []
            while curr is not None:
                path_nodes.append(curr)
                curr = parent.get(curr)
            path_nodes.reverse()

            start_reg = dag.nodes[origin[cap]].get("reg_id", origin[cap])
            end_reg = dag.nodes[cap].get("reg_id", cap)

            # Count operator breakdown on this path
            op_counts: dict[str, int] = defaultdict(int)
            accumulated_fanout = 0
            for pn in path_nodes:
                op = str(dag.nodes[pn].get("op", "NET"))
                if "DFF" in op:
                    op_counts["DFF"] += 1
                else:
                    op_counts[op] += 1
                accumulated_fanout += int(dag.nodes[pn].get("fan_out", 1))

            paths.append({
                "start": start_reg,
                "end": end_reg,
                "delay": dist[cap],
                "length": len(path_nodes),
                "path": path_nodes,
                "op_counts": dict(op_counts),
                "accumulated_fanout": accumulated_fanout,
            })

    return sorted(paths, key=lambda p: p["delay"], reverse=True)


def propagate_toggle_rates(
    graph: nx.DiGraph | dict[str, Any],
    start_rates: dict[Any, float] | None = None,
) -> dict[Any, float]:
    """Propagate switching activities (toggle rates) through the logic graph.

    Uses Boolean functionality formulas for each operator type:
    - NOT: alpha_out = alpha_in
    - AND / OR: probabilistic switching reduction
    - XOR: alpha_out = a*(1-b) + b*(1-a)
    - MUX: selection-weighted combination
    """
    if isinstance(graph, dict):
        g = nx.DiGraph()
        nodes = graph.get("nodes", {})
        for n, attrs in nodes.items():
            g.add_node(n, **attrs)
        for u, v in graph.get("edges", []):
            g.add_edge(u, v)
        graph_obj = g
    else:
        graph_obj = graph

    rates: dict[Any, float] = {}
    init_rates = start_rates or {}

    # Initialize rates for primary inputs and registers (default 0.1 toggle rate)
    for node, attrs in graph_obj.nodes(data=True):
        if node in init_rates:
            rates[node] = float(init_rates[node])
        elif attrs.get("op") in {"IO", "DFF"}:
            rates[node] = 0.10
        else:
            rates[node] = 0.05

    # Safely compute topological order (breaking feedback loops at registers)
    temp_graph = graph_obj.copy()
    for u, v, data in list(temp_graph.edges(data=True)):
        if temp_graph.nodes[v].get("op") == "DFF":
            temp_graph.remove_edge(u, v)

    try:
        topo_order = list(nx.topological_sort(temp_graph))
    except nx.NetworkXUnfeasible:
        # Fallback to simple degree ordering
        topo_order = sorted(temp_graph.nodes, key=lambda n: temp_graph.in_degree(n))

    for node in topo_order:
        if node in init_rates:
            continue
        op = str(graph_obj.nodes[node].get("op", "NET"))
        inputs = list(graph_obj.predecessors(node))
        if not inputs:
            continue

        in_rates = [rates.get(inp, 0.05) for inp in inputs]

        if op == "NOT":
            rates[node] = in_rates[0] if in_rates else 0.05
        elif op in {"AND", "OR"}:
            avg_r = sum(in_rates) / max(1, len(in_rates))
            rates[node] = max(0.001, min(1.0, 0.5 * avg_r * (1.0 - 0.25 * avg_r)))
        elif op == "XOR":
            if len(in_rates) >= 2:
                a, b = in_rates[0], in_rates[1]
                rates[node] = max(0.001, min(1.0, a * (1.0 - b) + b * (1.0 - a)))
            else:
                rates[node] = in_rates[0] if in_rates else 0.05
        elif op == "MUX":
            rates[node] = max(0.001, min(1.0, sum(in_rates) / max(1, len(in_rates))))
        elif op == "DFF":
            rates[node] = in_rates[0] if in_rates else 0.10
        else:
            rates[node] = sum(in_rates) / max(1, len(in_rates))

        if not isinstance(graph, dict):
            graph_obj.nodes[node]["toggle_rate"] = rates[node]

    return rates


def extract_sog_statistics(
    graph: nx.DiGraph,
    paths: list[dict[str, Any]],
    toggle_rates: dict[Any, float],
) -> dict[str, Any]:
    """Extract comprehensive topological, timing, power, and area statistics from the SOG."""
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    density = num_edges / max(1, (num_nodes * (num_nodes - 1))) if num_nodes > 1 else 0.0

    op_counts: dict[str, int] = defaultdict(int)
    fan_outs: list[int] = []
    fan_ins: list[int] = []

    for node, attrs in graph.nodes(data=True):
        op = attrs.get("op", "NET")
        op_counts[op] += 1
        fan_outs.append(attrs.get("fan_out", graph.out_degree(node)))
        fan_ins.append(attrs.get("fan_in", graph.in_degree(node)))

    num_registers = op_counts.get("DFF", 0)
    num_and = op_counts.get("AND", 0)
    num_or = op_counts.get("OR", 0)
    num_xor = op_counts.get("XOR", 0)
    num_not = op_counts.get("NOT", 0)
    num_mux = op_counts.get("MUX", 0)
    num_comb = num_and + num_or + num_xor + num_not + num_mux
    num_pi = len(graph.graph.get("primary_inputs", []))
    num_po = len(graph.graph.get("primary_outputs", []))

    # Path timing metrics
    path_delays = [p["delay"] for p in paths] if paths else [0.0]
    path_lengths = [p["length"] for p in paths] if paths else [0]

    sorted_delays = sorted(path_delays)
    crit_delay = sorted_delays[-1]
    mean_delay = sum(sorted_delays) / len(sorted_delays)
    min_delay = sorted_delays[0]

    def percentile(vals: list[float], pct: float) -> float:
        if not vals:
            return 0.0
        idx = int(math.ceil(pct / 100.0 * len(vals))) - 1
        return vals[max(0, min(idx, len(vals) - 1))]

    p10 = percentile(sorted_delays, 10.0)
    p50 = percentile(sorted_delays, 50.0)
    p90 = percentile(sorted_delays, 90.0)

    # Power switching activity metrics
    rate_values = list(toggle_rates.values()) if toggle_rates else [0.1]
    total_toggle = sum(rate_values)
    mean_toggle = total_toggle / max(1, len(rate_values))

    # Weighted switching: sum(fan_out * toggle_rate)
    weighted_toggle = sum(
        graph.nodes[n].get("fan_out", graph.out_degree(n)) * toggle_rates.get(n, 0.05)
        for n in graph.nodes
    )

    # Area approximations
    est_seq_area = num_registers * AREA_WEIGHTS["DFF"]
    est_comb_area = (
        num_and * AREA_WEIGHTS["AND"]
        + num_or * AREA_WEIGHTS["OR"]
        + num_xor * AREA_WEIGHTS["XOR"]
        + num_not * AREA_WEIGHTS["NOT"]
        + num_mux * AREA_WEIGHTS["MUX"]
    )
    est_total_area = est_seq_area + est_comb_area

    return {
        "module_name": graph.graph.get("module_name", "top"),
        "total_nodes": num_nodes,
        "total_edges": num_edges,
        "density": round(density, 6),
        "num_registers": num_registers,
        "num_comb_nodes": num_comb,
        "num_and": num_and,
        "num_or": num_or,
        "num_xor": num_xor,
        "num_not": num_not,
        "num_mux": num_mux,
        "num_primary_inputs": num_pi,
        "num_primary_outputs": num_po,
        "max_fanout": max(fan_outs) if fan_outs else 0,
        "mean_fanout": round(sum(fan_outs) / max(1, len(fan_outs)), 2),
        "max_fanin": max(fan_ins) if fan_ins else 0,
        "mean_fanin": round(sum(fan_ins) / max(1, len(fan_ins)), 2),
        "total_reg_paths": len(paths),
        "comb_depth": max(path_lengths) if path_lengths else 0,
        "critical_path_delay": round(crit_delay, 3),
        "mean_path_delay": round(mean_delay, 3),
        "min_path_delay": round(min_delay, 3),
        "path_delay_p10": round(p10, 3),
        "path_delay_p50": round(p50, 3),
        "path_delay_p90": round(p90, 3),
        "total_toggle_rate": round(total_toggle, 3),
        "mean_toggle_rate": round(mean_toggle, 4),
        "weighted_toggle_rate": round(weighted_toggle, 3),
        "est_sequential_area": round(est_seq_area, 2),
        "est_combinational_area": round(est_comb_area, 2),
        "est_total_area": round(est_total_area, 2),
    }


def save_graph_artifacts(
    graph: nx.DiGraph,
    output_dir: str | Path,
    name: str,
) -> dict[str, Path]:
    """Persist the SOG graph, path features, and statistics to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = output_dir / f"{name}.pkl"
    feature_path = output_dir / f"{name}_features.pkl"
    stats_path = output_dir / f"{name}_stats.json"

    # Compute features and toggle rates
    compute_node_delays(graph)
    paths = compute_path_features(graph)
    toggles = propagate_toggle_rates(graph)
    stats = extract_sog_statistics(graph, paths, toggles)

    with graph_path.open("wb") as handle:
        pickle.dump(graph, handle)

    with feature_path.open("wb") as handle:
        pickle.dump(paths, handle)

    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    return {"graph": graph_path, "features": feature_path, "stats": stats_path}


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
    print(f"Stats: {saved['stats']}")


if __name__ == "__main__":
    main()
