import json
from pathlib import Path

from src.data_prep.sog_converter import build_sog_from_json, compute_path_features


def test_build_sog_from_json_extracts_gate_graph(tmp_path: Path) -> None:
    json_path = tmp_path / "tiny.json"
    json_path.write_text(
        json.dumps(
            {
                "modules": {
                    "top": {
                        "ports": {
                            "a": {"bits": [1]},
                            "b": {"bits": [2]},
                            "q": {"bits": [9]},
                        },
                        "cells": {
                            "cell_and": {
                                "type": "$_AND_",
                                "connections": {"A": [1], "B": [2], "Y": [10]},
                            },
                            "cell_not": {
                                "type": "$_NOT_",
                                "connections": {"A": [10], "Y": [9]},
                            },
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    graph = build_sog_from_json(json_path)

    assert 1 in graph.nodes
    assert 10 in graph.nodes
    assert 9 in graph.nodes
    assert graph.has_edge(1, 10)
    assert graph.has_edge(2, 10)
    assert graph.nodes[10]["op"] == "AND"
    assert graph.nodes[10]["fan_in"] == 2
    assert graph.nodes[10]["fan_out"] >= 1


def test_build_sog_accepts_dff_like_yosys_register_cells(tmp_path: Path) -> None:
    json_path = tmp_path / "reg.json"
    json_path.write_text(
        json.dumps(
            {
                "modules": {
                    "top": {
                        "ports": {
                            "clk": {"bits": [1]},
                            "d": {"bits": [2]},
                            "q": {"bits": [9]},
                        },
                        "cells": {
                            "reg_cell": {
                                "type": "$_DFFE_PN0P_",
                                "connections": {"D": [2], "CLK": [1], "EN": [3], "Q": [9]},
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    graph = build_sog_from_json(json_path)
    assert graph.nodes[9]["op"] == "DFF"


def test_compute_path_features_returns_reg_to_reg_paths() -> None:
    graph = {
        "nodes": {
            "d0": {"op": "DFF", "fan_out": 1},
            "a": {"op": "NOT", "fan_out": 1},
            "b": {"op": "AND", "fan_out": 1},
            "d1": {"op": "DFF", "fan_out": 0},
        },
        "edges": [("d0", "a"), ("a", "b"), ("b", "d1")],
    }

    paths = compute_path_features(graph)
    assert len(paths) >= 1
    assert any(path["start"] == "d0" and path["end"] == "d1" for path in paths)
