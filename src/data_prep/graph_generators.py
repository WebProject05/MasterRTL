"""Graph generation engine for all representation types referenced in the MasterRTL paper.

Supported graph representations:
1. SOG (Simple Operator Graph) - Bit-level canonical representation (AND, OR, XOR, NOT, MUX, DFF)
2. AST (Abstract Syntax Tree) - Word-level syntax tree from prior works (ICCAD'22, ISCA'22)
3. AIG (And-Inverter Graph) - Classical logic synthesis bit-level graph (AND, NOT, DFF)
4. Netlist (Gate-Level Netlist Graph) - Synthesized netlist standard cells and interconnects
5. Timing DAG (STA Acyclic Timing Graph) - Decoupled launch/capture register timing DAG
6. CDFG (Control Data Flow Graph) - Dual-edge graph with control flow and data flow paths
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from src.data_prep.sog_converter import build_sog_from_json, build_timing_dag


# =========================================================================
# 1. SOG (Simple Operator Graph) - MasterRTL Bit-Level Core Representation
# =========================================================================

def build_sog(json_ast: dict[str, Any] | str | Path, top_module: str | None = None) -> nx.DiGraph:
    """Build SOG from Yosys synthesized JSON AST."""
    return build_sog_from_json(json_ast)


# =========================================================================
# 2. Timing DAG - Static Timing Analysis Decoupled Acyclic Graph
# =========================================================================

def build_timing_dag_graph(sog_graph: nx.DiGraph) -> nx.DiGraph:
    """Build STA decoupled acyclic timing DAG from SOG graph."""
    timing_dag, _ = build_timing_dag(sog_graph)
    return timing_dag


# =========================================================================
# 3. AST (Abstract Syntax Tree) - Word-Level Operator Representation
# =========================================================================

def build_ast_graph(json_ast: dict[str, Any] | str | Path, top_module: str | None = None) -> nx.DiGraph:
    """Build Word-Level Abstract Syntax Tree (AST) graph from pre-bit-blast Yosys AST.

    Represents word-level operators (multi-bit arithmetic, logic, muxes, registers)
    preserving signal bit widths as analyzed in ICCAD'22 [21] and ISCA'22 [22].
    """
    if isinstance(json_ast, (str, Path)):
        with open(json_ast, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json_ast

    modules = data.get("modules", {})
    if not modules:
        raise ValueError("Invalid Yosys JSON: missing 'modules' object.")

    if top_module is None:
        top_candidates = [m for m, m_data in modules.items() if m_data.get("attributes", {}).get("top") == 1]
        top_module = top_candidates[0] if top_candidates else next(iter(modules.keys()))

    if top_module not in modules:
        raise KeyError(f"Top module '{top_module}' not found in modules: {list(modules.keys())}")

    mod_data = modules[top_module]
    graph = nx.DiGraph()
    graph.graph["graph_type"] = "ast"
    graph.graph["design_name"] = top_module
    graph.graph["word_level"] = True

    # 1. Primary Ports
    ports = mod_data.get("ports", {})
    primary_inputs = []
    primary_outputs = []
    bit_to_driver: dict[int | str, str] = {}

    for port_name, p_info in ports.items():
        direction = p_info.get("direction", "input").lower()
        bits = p_info.get("bits", [])
        width = len(bits)
        node_id = f"port_{port_name}"

        node_type = "PI" if direction == "input" else "PO"
        if node_type == "PI":
            primary_inputs.append(node_id)
        else:
            primary_outputs.append(node_id)

        graph.add_node(
            node_id,
            node_type=node_type,
            op=node_type,
            name=port_name,
            width=width,
            bits=bits,
            direction=direction,
        )

        if direction == "input":
            for b in bits:
                bit_to_driver[b] = node_id

    # 2. Word-Level Cells (Operations)
    cells = mod_data.get("cells", {})
    registers = []

    for cell_name, cell_data in cells.items():
        raw_type = cell_data.get("type", "UNKNOWN")
        clean_op = raw_type.lstrip("$").upper()
        conn = cell_data.get("connections", {})
        node_id = f"cell_{cell_name}"

        is_reg = "DFF" in clean_op
        if is_reg:
            registers.append(node_id)

        # Output bit width
        y_bits = conn.get("Y", conn.get("Q", []))
        out_width = len(y_bits) if isinstance(y_bits, list) else 1

        graph.add_node(
            node_id,
            node_type="REGISTER" if is_reg else "OPERATOR",
            op=clean_op,
            raw_type=raw_type,
            name=cell_name,
            width=out_width,
            is_sequential=is_reg,
        )

        # Register drivers
        for b in (y_bits if isinstance(y_bits, list) else [y_bits]):
            bit_to_driver[b] = node_id

    # 3. Word-Level Net Interconnects (Edges)
    for cell_name, cell_data in cells.items():
        dst_node = f"cell_{cell_name}"
        conn = cell_data.get("connections", {})

        for port_pin, port_bits in conn.items():
            if port_pin in {"Y", "Q", "RD_DATA"}:
                continue  # Skip outputs

            if not isinstance(port_bits, list):
                port_bits = [port_bits]

            seen_drivers = set()
            for b in port_bits:
                if b in bit_to_driver:
                    driver = bit_to_driver[b]
                    if driver != dst_node and driver not in seen_drivers:
                        seen_drivers.add(driver)
                        graph.add_edge(driver, dst_node, port=port_pin, width=len(port_bits))

    # Connect to primary outputs
    for port_name, p_info in ports.items():
        if p_info.get("direction", "").lower() == "output":
            dst_node = f"port_{port_name}"
            bits = p_info.get("bits", [])
            seen_drivers = set()
            for b in bits:
                if b in bit_to_driver:
                    driver = bit_to_driver[b]
                    if driver != dst_node and driver not in seen_drivers:
                        seen_drivers.add(driver)
                        graph.add_edge(driver, dst_node, port="OUT", width=len(bits))

    graph.graph["primary_inputs"] = primary_inputs
    graph.graph["primary_outputs"] = primary_outputs
    graph.graph["registers"] = registers

    return graph


# =========================================================================
# 4. AIG (And-Inverter Graph) - Canonical 2-Input AND + NOT Representation
# =========================================================================

def build_aig_graph(json_ast: dict[str, Any] | str | Path, top_module: str | None = None) -> nx.DiGraph:
    """Build And-Inverter Graph (AIG) from Yosys AIG-synthesized JSON.

    Combinational nodes are strictly 2-input AND ($_AND_) and NOT inverters ($_NOT_),
    along with single-bit registers (DFF) and primary I/O, as cited in [18], [19], [20].
    """
    if isinstance(json_ast, (str, Path)):
        with open(json_ast, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json_ast

    modules = data.get("modules", {})
    if not modules:
        raise ValueError("Invalid Yosys JSON: missing 'modules' object.")

    if top_module is None:
        top_candidates = [m for m, m_data in modules.items() if m_data.get("attributes", {}).get("top") == 1]
        top_module = top_candidates[0] if top_candidates else next(iter(modules.keys()))

    mod_data = modules[top_module]
    graph = nx.DiGraph()
    graph.graph["graph_type"] = "aig"
    graph.graph["design_name"] = top_module

    ports = mod_data.get("ports", {})
    primary_inputs = []
    primary_outputs = []
    bit_to_driver: dict[int | str, str] = {}

    # Ports
    for port_name, p_info in ports.items():
        direction = p_info.get("direction", "input").lower()
        bits = p_info.get("bits", [])
        for idx, bit_val in enumerate(bits):
            node_id = f"pi_{port_name}[{idx}]" if direction == "input" else f"po_{port_name}[{idx}]"
            node_type = "PI" if direction == "input" else "PO"
            if node_type == "PI":
                primary_inputs.append(node_id)
                bit_to_driver[bit_val] = node_id
            else:
                primary_outputs.append(node_id)

            graph.add_node(
                node_id,
                node_type=node_type,
                op=node_type,
                bit=bit_val,
                port=port_name,
                index=idx,
            )

    # Cells
    cells = mod_data.get("cells", {})
    registers = []

    for cell_name, cell_data in cells.items():
        c_type = cell_data.get("type", "")
        conn = cell_data.get("connections", {})

        if "AND" in c_type:
            op = "AND"
        elif "NOT" in c_type or "INV" in c_type:
            op = "NOT"
        elif "DFF" in c_type:
            op = "DFF"
        else:
            op = c_type.strip("$_")

        node_id = f"aig_{cell_name}"
        is_reg = (op == "DFF")
        if is_reg:
            registers.append(node_id)

        graph.add_node(
            node_id,
            node_type="REGISTER" if is_reg else "GATE",
            op=op,
            raw_type=c_type,
            name=cell_name,
            is_sequential=is_reg,
        )

        for out_pin in ["Y", "Q"]:
            if out_pin in conn:
                bits = conn[out_pin]
                if not isinstance(bits, list):
                    bits = [bits]
                for b in bits:
                    bit_to_driver[b] = node_id

    # Edges
    for cell_name, cell_data in cells.items():
        dst_node = f"aig_{cell_name}"
        conn = cell_data.get("connections", {})

        for in_pin in ["A", "B", "C", "D"]:
            if in_pin in conn:
                bits = conn[in_pin]
                if not isinstance(bits, list):
                    bits = [bits]
                for b in bits:
                    if b in bit_to_driver and bit_to_driver[b] != dst_node:
                        graph.add_edge(bit_to_driver[b], dst_node, pin=in_pin)

    # Output connections
    for port_name, p_info in ports.items():
        if p_info.get("direction", "").lower() == "output":
            bits = p_info.get("bits", [])
            for idx, bit_val in enumerate(bits):
                dst_node = f"po_{port_name}[{idx}]"
                if bit_val in bit_to_driver and bit_to_driver[bit_val] != dst_node:
                    graph.add_edge(bit_to_driver[bit_val], dst_node, pin="D")

    graph.graph["primary_inputs"] = primary_inputs
    graph.graph["primary_outputs"] = primary_outputs
    graph.graph["registers"] = registers

    return graph


# =========================================================================
# 5. Netlist (Gate-Level Netlist Graph) - Standard Cell Instances & Nets
# =========================================================================

def build_netlist_graph(json_ast: dict[str, Any] | str | Path, top_module: str | None = None) -> nx.DiGraph:
    """Build Gate-Level Netlist Graph (G) from synthesized netlist.

    Nodes represent standard cells (NAND, NOR, XNOR, AOI, OAI, MUX, DFF)
    and I/O pins, representing the physical logic gates as illustrated in Fig. 2(c).
    """
    if isinstance(json_ast, (str, Path)):
        with open(json_ast, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json_ast

    modules = data.get("modules", {})
    if not modules:
        raise ValueError("Invalid Yosys JSON: missing 'modules' object.")

    if top_module is None:
        top_candidates = [m for m, m_data in modules.items() if m_data.get("attributes", {}).get("top") == 1]
        top_module = top_candidates[0] if top_candidates else next(iter(modules.keys()))

    mod_data = modules[top_module]
    graph = nx.DiGraph()
    graph.graph["graph_type"] = "netlist"
    graph.graph["design_name"] = top_module

    ports = mod_data.get("ports", {})
    primary_inputs = []
    primary_outputs = []
    bit_to_driver: dict[int | str, str] = {}

    # Ports
    for port_name, p_info in ports.items():
        direction = p_info.get("direction", "input").lower()
        bits = p_info.get("bits", [])
        for idx, bit_val in enumerate(bits):
            node_id = f"net_pi_{port_name}[{idx}]" if direction == "input" else f"net_po_{port_name}[{idx}]"
            node_type = "PI" if direction == "input" else "PO"
            if node_type == "PI":
                primary_inputs.append(node_id)
                bit_to_driver[bit_val] = node_id
            else:
                primary_outputs.append(node_id)

            graph.add_node(
                node_id,
                node_type=node_type,
                cell_type=node_type,
                port=port_name,
                bit=bit_val,
            )

    # Standard cells
    cells = mod_data.get("cells", {})
    registers = []

    for cell_name, cell_data in cells.items():
        c_type = cell_data.get("type", "UNKNOWN")
        conn = cell_data.get("connections", {})
        node_id = f"cell_{cell_name}"
        is_reg = "DFF" in c_type

        if is_reg:
            registers.append(node_id)

        graph.add_node(
            node_id,
            node_type="SEQUENTIAL_CELL" if is_reg else "COMBINATIONAL_CELL",
            cell_type=c_type,
            name=cell_name,
            is_sequential=is_reg,
        )

        for out_pin in ["Y", "Q", "ZN"]:
            if out_pin in conn:
                bits = conn[out_pin]
                if not isinstance(bits, list):
                    bits = [bits]
                for b in bits:
                    bit_to_driver[b] = node_id

    # Nets
    for cell_name, cell_data in cells.items():
        dst_node = f"cell_{cell_name}"
        conn = cell_data.get("connections", {})

        for in_pin, bits in conn.items():
            if in_pin in {"Y", "Q", "ZN", "RD_DATA"}:
                continue
            if not isinstance(bits, list):
                bits = [bits]
            for b in bits:
                if b in bit_to_driver and bit_to_driver[b] != dst_node:
                    graph.add_edge(bit_to_driver[b], dst_node, net_id=b, pin=in_pin)

    # Primary outputs
    for port_name, p_info in ports.items():
        if p_info.get("direction", "").lower() == "output":
            bits = p_info.get("bits", [])
            for idx, bit_val in enumerate(bits):
                dst_node = f"net_po_{port_name}[{idx}]"
                if bit_val in bit_to_driver and bit_to_driver[bit_val] != dst_node:
                    graph.add_edge(bit_to_driver[bit_val], dst_node, net_id=bit_val, pin="OUT")

    graph.graph["primary_inputs"] = primary_inputs
    graph.graph["primary_outputs"] = primary_outputs
    graph.graph["registers"] = registers

    return graph


# =========================================================================
# 6. CDFG (Control Data Flow Graph) - Dual-Edge Control & Data Flow Graph
# =========================================================================

def build_cdfg_graph(json_ast: dict[str, Any] | str | Path, top_module: str | None = None) -> nx.DiGraph:
    """Build Control Data Flow Graph (CDFG) from RTL AST.

    Differentiates Data Flow edges (variable dependency) from Control Flow edges
    (multiplexing conditions, FSM state branches, clock/reset enables).
    """
    if isinstance(json_ast, (str, Path)):
        with open(json_ast, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json_ast

    modules = data.get("modules", {})
    if not modules:
        raise ValueError("Invalid Yosys JSON: missing 'modules' object.")

    if top_module is None:
        top_candidates = [m for m, m_data in modules.items() if m_data.get("attributes", {}).get("top") == 1]
        top_module = top_candidates[0] if top_candidates else next(iter(modules.keys()))

    mod_data = modules[top_module]
    graph = nx.DiGraph()
    graph.graph["graph_type"] = "cdfg"
    graph.graph["design_name"] = top_module

    ports = mod_data.get("ports", {})
    cells = mod_data.get("cells", {})
    bit_to_driver: dict[int | str, str] = {}

    # Ports
    for port_name, p_info in ports.items():
        direction = p_info.get("direction", "input").lower()
        bits = p_info.get("bits", [])
        node_id = f"cdfg_port_{port_name}"
        node_type = "PI" if direction == "input" else "PO"

        graph.add_node(
            node_id,
            node_type=node_type,
            name=port_name,
            width=len(bits),
            direction=direction,
        )

        if direction == "input":
            for b in bits:
                bit_to_driver[b] = node_id

    # Operations & Control Nodes
    for cell_name, cell_data in cells.items():
        c_type = cell_data.get("type", "")
        conn = cell_data.get("connections", {})
        node_id = f"cdfg_op_{cell_name}"

        clean = c_type.lstrip("$_").upper()
        if any(x in clean for x in ["MUX", "PMUX", "CASE"]):
            category = "CONTROL_BRANCH"
        elif "DFF" in clean or "FSM" in clean:
            category = "CONTROL_STATE"
        elif any(x in clean for x in ["EQ", "NE", "LT", "LE", "GT", "GE", "CMP"]):
            category = "CONTROL_CONDITION"
        elif any(x in clean for x in ["ADD", "SUB", "MUL", "DIV", "ALU", "SHL", "SHR"]):
            category = "DATA_ARITH"
        else:
            category = "DATA_LOGIC"

        graph.add_node(
            node_id,
            node_type=category,
            op=clean,
            raw_type=c_type,
            name=cell_name,
        )

        for out_pin in ["Y", "Q"]:
            if out_pin in conn:
                bits = conn[out_pin]
                if not isinstance(bits, list):
                    bits = [bits]
                for b in bits:
                    bit_to_driver[b] = node_id

    # Dual-Edge Routing
    for cell_name, cell_data in cells.items():
        dst_node = f"cdfg_op_{cell_name}"
        conn = cell_data.get("connections", {})
        category = graph.nodes[dst_node]["node_type"]

        for in_pin, bits in conn.items():
            if in_pin in {"Y", "Q"}:
                continue
            if not isinstance(bits, list):
                bits = [bits]

            is_control_edge = in_pin in {"S", "SEL", "C", "CLK", "ARST", "EN", "SET", "CLR"} or category.startswith("CONTROL_")
            edge_type = "control" if is_control_edge else "data"

            seen_drivers = set()
            for b in bits:
                if b in bit_to_driver:
                    driver = bit_to_driver[b]
                    if driver != dst_node and driver not in seen_drivers:
                        seen_drivers.add(driver)
                        graph.add_edge(driver, dst_node, edge_type=edge_type, pin=in_pin)

    # Outputs
    for port_name, p_info in ports.items():
        if p_info.get("direction", "").lower() == "output":
            dst_node = f"cdfg_port_{port_name}"
            bits = p_info.get("bits", [])
            seen_drivers = set()
            for b in bits:
                if b in bit_to_driver:
                    driver = bit_to_driver[b]
                    if driver != dst_node and driver not in seen_drivers:
                        seen_drivers.add(driver)
                        graph.add_edge(driver, dst_node, edge_type="data", pin="OUT")

    return graph


# =========================================================================
# Standardized Graph Statistics Extractor
# =========================================================================

def extract_graph_stats(graph: nx.DiGraph, graph_type: str, design_name: str) -> dict[str, Any]:
    """Extract standard topological and structural metrics for any graph type."""
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    is_dag = nx.is_directed_acyclic_graph(graph)
    density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0

    in_degrees = [d for _, d in graph.in_degree()]
    avg_degree = sum(in_degrees) / num_nodes if num_nodes > 0 else 0.0

    max_depth = 0
    if is_dag and num_nodes > 0:
        try:
            max_depth = nx.dag_longest_path_length(graph)
        except Exception:
            max_depth = 0

    op_counts: dict[str, int] = defaultdict(int)
    for _, data in graph.nodes(data=True):
        op_label = data.get("op", data.get("cell_type", data.get("node_type", "OTHER")))
        op_counts[str(op_label)] += 1

    pi_count = len(graph.graph.get("primary_inputs", [n for n, d in graph.nodes(data=True) if d.get("node_type") == "PI"]))
    po_count = len(graph.graph.get("primary_outputs", [n for n, d in graph.nodes(data=True) if d.get("node_type") == "PO"]))
    reg_count = len(graph.graph.get("registers", [n for n, d in graph.nodes(data=True) if d.get("is_sequential") or "DFF" in str(d.get("op", ""))]))

    return {
        "design_name": design_name,
        "graph_type": graph_type,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": round(density, 6),
        "avg_degree": round(avg_degree, 3),
        "is_dag": is_dag,
        "max_depth": max_depth,
        "num_registers": reg_count,
        "num_primary_inputs": pi_count,
        "num_primary_outputs": po_count,
        "top_operators": dict(sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:8]),
    }
