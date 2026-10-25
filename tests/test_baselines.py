"""Unit and integration tests for baseline models ICCAD'22 [21] and ISCA'22 [22]."""

from __future__ import annotations

import numpy as np
import pytest

from src.models import (
    ICCAD21Baseline,
    ISCA22Baseline,
    extract_iccad21_tree_features,
    extract_isca22_design_features,
    extract_isca22_path_features,
    load_ast_dataset,
    load_ppa_dataset,
)


@pytest.fixture
def synthetic_dataset() -> list[dict]:
    """Create synthetic dataset fixture with both AST and SOG properties."""
    samples = []
    for i in range(12):
        nodes = 50 + i * 20
        regs = 10 + i * 4
        comb = nodes - regs
        stats = {
            "total_nodes": nodes,
            "num_nodes": nodes,
            "total_edges": nodes * 2,
            "num_edges": nodes * 2,
            "num_registers": regs,
            "num_comb_nodes": comb,
            "num_and": int(comb * 0.4),
            "num_or": int(comb * 0.3),
            "num_xor": int(comb * 0.2),
            "num_not": int(comb * 0.05),
            "num_mux": int(comb * 0.05),
            "comb_depth": 10 + i * 2,
            "max_depth": 10 + i * 2,
            "critical_path_delay": 12.0 + i * 2.5,
            "density": 0.015,
            "weighted_toggle_rate": 5.0 + i * 2.0,
        }
        paths = [
            {
                "length": 8 + i,
                "op_counts": {"AND": 3, "OR": 2, "XOR": 1, "DFF": 1},
                "delay": 10.0 + i * 2.0,
                "accumulated_fanout": 12,
            }
        ]
        wns = min(0.0, 1.0 - 0.035 * stats["critical_path_delay"])
        tns = wns * regs * 0.45
        pwr = 0.045 * stats["weighted_toggle_rate"] + 0.003 * nodes + 0.02
        area = regs * 4.522 + comb * 1.15 + 2.5

        samples.append({
            "name": f"design_synth_{i}",
            "stats": stats,
            "paths": paths,
            "labels": {
                "wns": round(wns, 4),
                "tns": round(tns, 4),
                "power": round(pwr, 4),
                "area": round(area, 2),
                "netlist_delays": [float(p["delay"]) * 0.035 for p in paths],
            },
        })
    return samples


def test_iccad21_baseline_feature_extraction() -> None:
    stats = {
        "num_registers": 16,
        "total_nodes": 120,
        "comb_depth": 14,
        "num_and": 40,
        "num_or": 30,
        "num_xor": 15,
        "num_not": 5,
        "num_mux": 10,
        "density": 0.02,
        "num_primary_inputs": 16,
        "num_primary_outputs": 8,
    }
    feats_ast = extract_iccad21_tree_features(stats, representation="ast")
    feats_sog = extract_iccad21_tree_features(stats, representation="sog")

    assert len(feats_ast) == 17
    assert len(feats_sog) == 17
    # Register count
    assert feats_ast[0] == 16.0
    # Duplicated tree nodes should exceed raw nodes
    assert feats_ast[2] > stats["total_nodes"]


def test_iccad21_baseline_train_predict(synthetic_dataset: list[dict]) -> None:
    model = ICCAD21Baseline(representation="ast", n_estimators=10, max_depth=4)
    model.train(synthetic_dataset)
    assert model.is_fitted

    pred = model.predict(synthetic_dataset[0])
    assert "predicted_wns" in pred
    assert "predicted_tns" in pred
    assert "predicted_power_mW" in pred
    assert "predicted_area_um2" in pred

    eval_res = model.evaluate_dataset(synthetic_dataset)
    assert "TNS" in eval_res
    assert "Power" in eval_res
    assert eval_res["Power"]["R"] > 0.0


def test_isca22_baseline_path_features_omits_fanout() -> None:
    path_sample = {
        "length": 6,
        "op_counts": {"AND": 2, "OR": 2, "XOR": 1, "DFF": 1},
        "delay": 15.0,
        "accumulated_fanout": 25,  # Should NOT be in ISCA'22 features
    }
    feats = extract_isca22_path_features(path_sample)
    # Length (1) + 6 operator counts = 7 features
    assert len(feats) == 7
    # Neither fanout (25) nor delay (15.0) are in the sequence features
    assert 25.0 not in feats
    assert 15.0 not in feats


def test_isca22_baseline_train_predict(synthetic_dataset: list[dict]) -> None:
    model = ISCA22Baseline(representation="ast", n_estimators=10, max_depth=4)
    model.train(synthetic_dataset)
    assert model.is_fitted

    pred = model.predict(synthetic_dataset[0])
    assert "predicted_wns" in pred
    assert "predicted_tns" in pred
    assert "predicted_power_mW" in pred
    assert "predicted_area_um2" in pred

    eval_res = model.evaluate_dataset(synthetic_dataset)
    assert "WNS" in eval_res
    assert "Area" in eval_res
    assert eval_res["WNS"]["R"] > 0.0


def test_load_ast_dataset_structure() -> None:
    ast_dataset = load_ast_dataset("data/graphs/ast", "data/sog_graphs")
    assert len(ast_dataset) > 0
    sample = ast_dataset[0]
    assert "name" in sample
    assert "stats" in sample
    assert "labels" in sample
    assert "paths" in sample
