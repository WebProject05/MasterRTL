"""Unit and integration tests for the SOG graph construction and feature extraction pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from src.data_prep.batch_parse import export_aggregated_reports
from src.data_prep.sog_converter import (
    build_sog_from_json,
    build_timing_dag,
    compute_node_delays,
    compute_path_features,
    extract_sog_statistics,
    propagate_toggle_rates,
    save_graph_artifacts,
)


@pytest.fixture
def synthetic_yosys_json(tmp_path: Path) -> Path:
    """Create a minimal synthetic Yosys JSON AST file with logic gates and a DFF."""
    json_path = tmp_path / "mock_design.json"
    ast = {
        "modules": {
            "mock_top": {
                "ports": {
                    "clk": {"direction": "input", "bits": [1]},
                    "rst_n": {"direction": "input", "bits": [2]},
                    "d_in": {"direction": "input", "bits": [3]},
                    "q_out": {"direction": "output", "bits": [10]},
                },
                "cells": {
                    "gate_not": {
                        "type": "$_NOT_",
                        "connections": {"A": [3], "Y": [4]},
                    },
                    "gate_and": {
                        "type": "$_AND_",
                        "connections": {"A": [4], "B": [10], "Y": [5]},
                    },
                    "reg_dff": {
                        "type": "$_DFF_PN0_",
                        "connections": {"C": [1], "R": [2], "D": [5], "Q": [10]},
                    },
                },
            }
        }
    }
    json_path.write_text(json.dumps(ast), encoding="utf-8")
    return json_path


def test_build_sog_extracts_correct_operators_and_dff(synthetic_yosys_json: Path) -> None:
    graph = build_sog_from_json(synthetic_yosys_json)

    assert graph.number_of_nodes() >= 4
    assert 4 in graph.nodes
    assert 5 in graph.nodes
    assert 10 in graph.nodes

    assert graph.nodes[4]["op"] == "NOT"
    assert graph.nodes[5]["op"] == "AND"
    assert graph.nodes[10]["op"] == "DFF"

    # Register attributes
    assert graph.nodes[10]["q_pin"] == 10
    assert graph.nodes[10]["d_pin"] == 5
    assert graph.nodes[10]["clk_pin"] == 1


def test_timing_dag_is_strictly_acyclic(synthetic_yosys_json: Path) -> None:
    graph = build_sog_from_json(synthetic_yosys_json)
    dag, meta = build_timing_dag(graph)

    # In mock design: reg feeds gate_and, which feeds back to reg.
    # In STA DAG, this cycle must be broken by separating launch and capture points.
    assert nx.is_directed_acyclic_graph(dag)
    assert len(meta["launch_nodes"]) == 1
    assert len(meta["capture_nodes"]) == 1


def test_compute_path_features_extracts_paths(synthetic_yosys_json: Path) -> None:
    graph = build_sog_from_json(synthetic_yosys_json)
    paths = compute_path_features(graph)

    assert len(paths) >= 1
    crit_path = paths[0]
    assert crit_path["delay"] > 0.0
    assert "start" in crit_path
    assert "end" in crit_path
    assert "op_counts" in crit_path


def test_propagate_toggle_rates(synthetic_yosys_json: Path) -> None:
    graph = build_sog_from_json(synthetic_yosys_json)
    rates = propagate_toggle_rates(graph, start_rates={1: 1.0, 3: 0.2})

    assert rates[1] == 1.0
    assert rates[3] == 0.2
    assert 0.0 <= rates[4] <= 1.0
    assert 0.0 <= rates[5] <= 1.0


def test_extract_sog_statistics(synthetic_yosys_json: Path, tmp_path: Path) -> None:
    graph = build_sog_from_json(synthetic_yosys_json)
    delays = compute_node_delays(graph)
    paths = compute_path_features(graph)
    toggles = propagate_toggle_rates(graph)

    stats = extract_sog_statistics(graph, paths, toggles)
    assert stats["module_name"] == "mock_top"
    assert stats["num_registers"] == 1
    assert stats["num_comb_nodes"] == 2
    assert stats["num_and"] == 1
    assert stats["num_not"] == 1
    assert stats["critical_path_delay"] > 0.0
    assert stats["est_total_area"] > 0.0


def test_save_graph_artifacts(synthetic_yosys_json: Path, tmp_path: Path) -> None:
    graph = build_sog_from_json(synthetic_yosys_json)
    saved = save_graph_artifacts(graph, tmp_path, "mock_top")

    assert saved["graph"].is_file()
    assert saved["features"].is_file()
    assert saved["stats"].is_file()

    with saved["stats"].open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["module_name"] == "mock_top"


def test_export_aggregated_reports(tmp_path: Path) -> None:
    sample_stats = [
        {
            "module_name": "mod_a",
            "total_nodes": 100,
            "total_edges": 120,
            "density": 0.012,
            "num_registers": 16,
            "num_comb_nodes": 60,
            "num_and": 20,
            "num_or": 15,
            "num_xor": 10,
            "num_not": 10,
            "num_mux": 5,
            "critical_path_delay": 18.5,
            "mean_path_delay": 12.3,
            "weighted_toggle_rate": 8.4,
            "est_total_area": 145.2,
        }
    ]
    reports = export_aggregated_reports(sample_stats, tmp_path / "reports")
    assert reports["csv"].is_file()
    assert reports["json"].is_file()
    assert reports["md"].is_file()

