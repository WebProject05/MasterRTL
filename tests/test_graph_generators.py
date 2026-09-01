"""Unit tests for all 6 graph representation generators mentioned in the MasterRTL paper:
SOG, AST, AIG, Netlist, Timing DAG, and CDFG.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from src.data_prep.graph_generators import (
    build_aig_graph,
    build_ast_graph,
    build_cdfg_graph,
    build_netlist_graph,
    build_sog,
    build_timing_dag_graph,
    extract_graph_stats,
)
from src.data_prep.sog_converter import build_sog_from_json


@pytest.fixture
def mock_ast_json() -> dict:
    """Mock Yosys word-level AST JSON with arithmetic operators, mux, and DFF."""
    return {
        "modules": {
            "test_alu": {
                "attributes": {"top": 1},
                "ports": {
                    "clk": {"direction": "input", "bits": [2]},
                    "rst": {"direction": "input", "bits": [3]},
                    "a": {"direction": "input", "bits": [4, 5, 6, 7]},
                    "b": {"direction": "input", "bits": [8, 9, 10, 11]},
                    "sel": {"direction": "input", "bits": [12]},
                    "out": {"direction": "output", "bits": [20, 21, 22, 23]},
                },
                "cells": {
                    "$add$1": {
                        "type": "$add",
                        "connections": {
                            "A": [4, 5, 6, 7],
                            "B": [8, 9, 10, 11],
                            "Y": [13, 14, 15, 16],
                        },
                    },
                    "$mux$2": {
                        "type": "$mux",
                        "connections": {
                            "A": [4, 5, 6, 7],
                            "B": [13, 14, 15, 16],
                            "S": [12],
                            "Y": [17, 18, 19, 20],
                        },
                    },
                    "$dff$3": {
                        "type": "$dff",
                        "connections": {
                            "CLK": [2],
                            "D": [17, 18, 19, 20],
                            "Q": [20, 21, 22, 23],
                        },
                    },
                },
            }
        }
    }


@pytest.fixture
def mock_aig_json() -> dict:
    """Mock Yosys AIG JSON strictly with 2-input ANDs, NOTs, and DFFs."""
    return {
        "modules": {
            "test_aig": {
                "attributes": {"top": 1},
                "ports": {
                    "in_a": {"direction": "input", "bits": [1]},
                    "in_b": {"direction": "input", "bits": [2]},
                    "out_y": {"direction": "output", "bits": [5]},
                },
                "cells": {
                    "g_and1": {
                        "type": "$_AND_",
                        "connections": {"A": [1], "B": [2], "Y": [3]},
                    },
                    "g_not1": {
                        "type": "$_NOT_",
                        "connections": {"A": [3], "Y": [4]},
                    },
                    "g_dff1": {
                        "type": "$_DFF_P_",
                        "connections": {"C": [10], "D": [4], "Q": [5]},
                    },
                },
            }
        }
    }


def test_build_ast_graph(mock_ast_json: dict) -> None:
    graph = build_ast_graph(mock_ast_json, "test_alu")
    assert isinstance(graph, nx.DiGraph)
    assert graph.graph["graph_type"] == "ast"
    assert graph.graph["design_name"] == "test_alu"
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0

    # Verify ports and cells exist
    assert "port_a" in graph
    assert "port_out" in graph
    assert "cell_$add$1" in graph
    assert graph.nodes["cell_$add$1"]["op"] == "ADD"


def test_build_aig_graph(mock_aig_json: dict) -> None:
    graph = build_aig_graph(mock_aig_json, "test_aig")
    assert isinstance(graph, nx.DiGraph)
    assert graph.graph["graph_type"] == "aig"
    assert "aig_g_and1" in graph
    assert "aig_g_not1" in graph
    assert "aig_g_dff1" in graph

    assert graph.nodes["aig_g_and1"]["op"] == "AND"
    assert graph.nodes["aig_g_not1"]["op"] == "NOT"
    assert graph.nodes["aig_g_dff1"]["op"] == "DFF"


def test_build_netlist_graph(mock_aig_json: dict) -> None:
    graph = build_netlist_graph(mock_aig_json, "test_aig")
    assert isinstance(graph, nx.DiGraph)
    assert graph.graph["graph_type"] == "netlist"
    assert "cell_g_and1" in graph
    assert graph.nodes["cell_g_and1"]["cell_type"] == "$_AND_"


def test_build_cdfg_graph(mock_ast_json: dict) -> None:
    graph = build_cdfg_graph(mock_ast_json, "test_alu")
    assert isinstance(graph, nx.DiGraph)
    assert graph.graph["graph_type"] == "cdfg"
    assert "cdfg_op_$mux$2" in graph
    # Mux is categorized as control branch
    assert graph.nodes["cdfg_op_$mux$2"]["node_type"] == "CONTROL_BRANCH"


def test_build_timing_dag_graph() -> None:
    # Build a simple SOG-like graph with a DFF
    g = nx.DiGraph()
    g.graph["primary_inputs"] = ["in1"]
    g.graph["primary_outputs"] = ["out1"]
    g.graph["registers"] = ["reg1"]

    g.add_node("in1", op="IO", io_type="input", bit_width=1, fan_in=0, fan_out=1, delay=1.0)
    g.add_node("and1", op="AND", bit_width=1, fan_in=2, fan_out=1, delay=1.2)
    g.add_node("reg1", op="DFF", bit_width=1, fan_in=1, fan_out=1, delay=1.5, q_pin="reg1", d_pin="and1")
    g.add_node("out1", op="IO", io_type="output", bit_width=1, fan_in=1, fan_out=0, delay=1.0)

    g.add_edge("in1", "and1")
    g.add_edge("reg1", "and1")
    g.add_edge("and1", "reg1", is_sequential=True)
    g.add_edge("reg1", "out1")

    # In SOG, reg1 -> and1 -> reg1 is a cycle across clock cycles
    assert not nx.is_directed_acyclic_graph(g)

    # In Timing DAG, registers are decoupled into launch and capture nodes
    tdag = build_timing_dag_graph(g)
    assert nx.is_directed_acyclic_graph(tdag)
    assert "reg_launch_reg1" in tdag
    assert "reg_capture_reg1" in tdag


def test_extract_graph_stats(mock_ast_json: dict) -> None:
    graph = build_ast_graph(mock_ast_json, "test_alu")
    stats = extract_graph_stats(graph, "ast", "test_alu")
    assert stats["design_name"] == "test_alu"
    assert stats["graph_type"] == "ast"
    assert stats["num_nodes"] > 0
    assert stats["num_edges"] > 0
    assert stats["avg_degree"] >= 0.0
    assert "num_registers" in stats

