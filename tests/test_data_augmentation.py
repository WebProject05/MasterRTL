"""Unit and integration tests for Data Augmentation Engine (Section II-F & III-E)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.data_prep.data_augmentation import (
    GraphBasedAugmentor,
    HierarchicalAutoDecompositionAugmentor,
    evaluate_augmentation_quality,
    generate_augmented_dataset,
)


def test_graph_based_augmentor_generates_synthesizable_verilog():
    augmentor = GraphBasedAugmentor(random_seed=42)
    des = augmentor.generate_design("test_graph_aug", target_scale="small")

    assert des.name == "test_graph_aug"
    assert des.method == "graph_based"
    assert des.is_synthesizable is True
    assert des.num_registers > 0
    assert des.num_combinational > 0
    assert "module test_graph_aug" in des.verilog_code
    assert "always @(posedge clk" in des.verilog_code
    assert "endmodule" in des.verilog_code


def test_hierarchical_augmentor_generates_multi_modules():
    augmentor = HierarchicalAutoDecompositionAugmentor(random_seed=100)
    des = augmentor.generate_design("test_hier_aug", category_id="crypto")

    assert des.name == "test_hier_aug"
    assert des.method == "hierarchical"
    assert des.num_submodules == 3
    assert "module test_hier_aug_datapath" in des.verilog_code
    assert "module test_hier_aug_control" in des.verilog_code
    assert "module test_hier_aug" in des.verilog_code


def test_evaluate_augmentation_quality_table5():
    g_aug = GraphBasedAugmentor(random_seed=1)
    h_aug = HierarchicalAutoDecompositionAugmentor(random_seed=2)

    designs = [
        g_aug.generate_design("g1"),
        g_aug.generate_design("g2"),
        h_aug.generate_design("h1"),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        report_file = Path(tmp_dir) / "table5_test.md"
        report = evaluate_augmentation_quality(designs, report_file)

        assert report["Graph-Based"]["Synthesizability"] == "100%"
        assert report["Hierarchical Auto-Decomposition"]["Synthesizability"] == "100%"
        assert report["Combined Total"]["Total Designs"] == 3
        assert report_file.is_file()
        content = report_file.read_text(encoding="utf-8")
        assert "Table V: Quality Evaluation" in content
