"""Unit tests for MasterRTL ML modeling framework."""

from __future__ import annotations

import numpy as np
import pytest

from src.models import (
    AreaPPAEstimator,
    CombinationalAreaModel,
    DesignLevelPowerCalibrationModel,
    DesignLevelTimingCalibrationModel,
    LayoutAndTechTransferModel,
    MasterRTL,
    ModuleLevelPowerModel,
    PathLevelTimingModel,
    PowerPPAEstimator,
    TimingPPAEstimator,
    TransferPPAEstimator,
    compute_mae,
    compute_mape,
    compute_pearson_r,
    compute_rrse,
    compute_sequential_area,
    evaluate_metrics,
    infer_design_timing_paths,
)
from src.models.area import extract_combinational_area_features
from src.models.power import (
    extract_design_power_calibration_features,
    extract_module_power_features,
)
from src.models.timing import (
    extract_design_timing_calibration_features,
    extract_path_features,
)
from src.models.transfer import extract_transfer_features


# =========================================================================
# Metrics Tests
# =========================================================================

def test_evaluation_metrics_exact() -> None:
    y_true = [10.0, 20.0, 30.0, 40.0]
    y_pred = [10.0, 20.0, 30.0, 40.0]

    assert compute_pearson_r(y_true, y_pred) == 1.0
    assert compute_mape(y_true, y_pred) == 0.0
    assert compute_mae(y_true, y_pred) == 0.0
    assert compute_rrse(y_true, y_pred) == 0.0

    # With small errors
    y_noisy = [11.0, 19.0, 31.0, 39.0]
    metrics = evaluate_metrics(y_true, y_noisy)
    assert metrics["R"] > 0.95
    assert metrics["MAPE"] > 0.0
    assert metrics["MAE"] == 1.0
    assert metrics["RRSE"] > 0.0


# =========================================================================
# Timing Model Tests
# =========================================================================

def test_path_level_timing_model() -> None:
    path_sample = {
        "length": 5,
        "op_counts": {"AND": 2, "OR": 1, "XOR": 1, "NOT": 0, "MUX": 0, "DFF": 1},
        "delay": 12.5,
        "accumulated_fanout": 6,
    }
    feats = extract_path_features(path_sample)
    assert len(feats) == 9
    assert feats[0] == 5.0
    assert feats[1] == 2.0  # AND count
    assert feats[7] == 12.5  # delay

    # Train path model
    X = [feats for _ in range(10)]
    y = [0.45 for _ in range(10)]

    model = PathLevelTimingModel(n_estimators=10, max_depth=5)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 10
    assert abs(preds[0] - 0.45) < 0.05


def test_design_timing_paths_inference() -> None:
    path_model = PathLevelTimingModel(n_estimators=10, max_depth=5)
    paths = [
        {"length": 4, "op_counts": {"AND": 2}, "delay": 8.0, "accumulated_fanout": 4},
        {"length": 6, "op_counts": {"AND": 4}, "delay": 14.0, "accumulated_fanout": 8},
    ]
    path_model.fit([extract_path_features(p) for p in paths], [0.30, 0.55])

    res = infer_design_timing_paths(path_model, paths, clk_period=1.0)
    assert "wns_r" in res
    assert "tns_r" in res
    assert res["wns_r"] < 1.0
    assert "slack_worst" in res


def test_design_level_timing_calibration() -> None:
    calib = DesignLevelTimingCalibrationModel(n_estimators=10, max_depth=4)
    dummy_X = [[10.0] * 18 for _ in range(8)]
    dummy_wns = [-0.1, -0.2, -0.15, -0.3, -0.05, -0.25, -0.12, -0.18]
    dummy_tns = [-1.0, -2.0, -1.5, -3.0, -0.5, -2.5, -1.2, -1.8]

    calib.fit(dummy_X, dummy_wns, dummy_tns)
    p_wns, p_tns = calib.predict(dummy_X)
    assert len(p_wns) == 8
    assert len(p_tns) == 8


# =========================================================================
# Power Model Tests
# =========================================================================

def test_power_model() -> None:
    stats = {
        "total_toggle_rate": 15.0,
        "mean_toggle_rate": 0.1,
        "weighted_toggle_rate": 22.0,
        "total_nodes": 120,
        "total_edges": 160,
        "num_registers": 16,
        "num_comb_nodes": 80,
        "num_and": 30,
        "num_or": 20,
        "num_xor": 10,
        "num_not": 10,
        "num_mux": 10,
    }
    feats = extract_module_power_features(stats)
    assert len(feats) == 10

    mod_model = ModuleLevelPowerModel(n_estimators=10, max_depth=4)
    mod_model.fit([feats] * 5, [1.25] * 5)
    pred_p = mod_model.predict([feats])
    assert abs(pred_p[0] - 1.25) < 0.1

    pwr_est = PowerPPAEstimator()
    res = pwr_est.predict_design(stats)
    assert "predicted_power" in res
    assert res["predicted_power"] > 0.0


# =========================================================================
# Area Model Tests
# =========================================================================

def test_area_model() -> None:
    assert compute_sequential_area(10, 4.522) == pytest.approx(45.22)

    stats = {
        "num_registers": 10,
        "num_comb_nodes": 50,
        "est_combinational_area": 60.5,
        "num_and": 20,
        "num_or": 15,
        "num_xor": 5,
        "num_not": 5,
        "num_mux": 5,
        "total_nodes": 70,
        "total_edges": 90,
        "density": 0.015,
        "mean_fanout": 1.2,
        "max_fanout": 4,
        "mean_fanin": 1.2,
        "max_fanin": 2,
    }
    feats = extract_combinational_area_features(stats)
    assert len(feats) == 14

    comb_model = CombinationalAreaModel(n_estimators=10, max_depth=4)
    comb_model.fit([feats] * 5, [75.0] * 5)
    pred = comb_model.predict([feats])
    assert abs(pred[0] - 75.0) < 1.0

    area_est = AreaPPAEstimator()
    res = area_est.predict_design(stats)
    assert res["sequential_area"] == pytest.approx(45.22)
    assert res["predicted_total_area"] >= res["sequential_area"]


# =========================================================================
# Transfer Model Tests
# =========================================================================

def test_transfer_model() -> None:
    src_ppa = {"wns": -0.15, "tns": -1.5, "power": 2.5, "area": 120.0}
    stats = {"total_nodes": 100, "total_edges": 140, "num_registers": 16, "num_comb_nodes": 60}
    scale = {"delay": 0.65, "power": 1.35, "area": 1.00}

    feats = extract_transfer_features(src_ppa, stats, target_scale=scale)
    assert len(feats) == 15

    transfer_est = TransferPPAEstimator()
    res = transfer_est.transfer_design(src_ppa, stats, target_technology="NanGate_45nm_MIN", target_scale=scale)
    assert "transferred_power" in res
    assert "transferred_area" in res
    assert res["transferred_power"] > 0.0


# =========================================================================
# Unified MasterRTL Tests
# =========================================================================

def test_master_rtl_end_to_end_synthetic() -> None:
    dataset = []
    for i in range(5):
        stats = {
            "module_name": f"mod_{i}",
            "total_nodes": 50 + i * 20,
            "total_edges": 70 + i * 25,
            "num_registers": 8 + i * 4,
            "num_comb_nodes": 30 + i * 15,
            "num_and": 10 + i * 5,
            "num_or": 8 + i * 3,
            "num_xor": 6 + i * 2,
            "num_not": 4 + i,
            "num_mux": 2 + i,
            "mean_fanout": 1.3,
            "max_fanout": 3,
            "density": 0.02,
            "critical_path_delay": 15.0 + i * 3,
            "total_toggle_rate": 8.0 + i * 2,
            "mean_toggle_rate": 0.08,
            "weighted_toggle_rate": 12.0 + i * 3,
            "est_combinational_area": 40.0 + i * 15,
        }
        paths = [
            {"length": 4, "op_counts": {"AND": 2, "DFF": 1}, "delay": 10.0 + i, "accumulated_fanout": 4},
            {"length": 6, "op_counts": {"OR": 2, "DFF": 1}, "delay": 14.0 + i, "accumulated_fanout": 6},
        ]
        labels = {
            "wns": -0.1 * (i + 1),
            "tns": -1.0 * (i + 1),
            "power": 0.5 * (i + 1),
            "area": 50.0 * (i + 1),
            "comb_area": 30.0 * (i + 1),
            "netlist_delays": [0.35 + i * 0.05, 0.49 + i * 0.05],
        }
        dataset.append({"name": f"mod_{i}", "stats": stats, "paths": paths, "labels": labels})

    model = MasterRTL()
    res_train = model.train(dataset)
    assert res_train["status"] == "trained"

    # Single prediction
    pred = model.predict(dataset[0]["stats"], dataset[0]["paths"])
    assert "summary" in pred
    assert "predicted_wns" in pred["summary"]
    assert "predicted_power_mW" in pred["summary"]
    assert "predicted_area_um2" in pred["summary"]

    # Evaluate dataset
    eval_res = model.evaluate_dataset(dataset)
    assert "WNS" in eval_res
    assert "Power" in eval_res
    assert "Area" in eval_res
    assert eval_res["Area"]["R"] > 0.8
