"""Unit and integration test for scripts/predict.py inference CLI."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from scripts.predict import format_ppa_table, generate_prediction_plot, predict_verilog_file
from src.models.master_rtl import MasterRTL


def test_format_ppa_table():
    stats = {
        "total_nodes": 100,
        "num_registers": 10,
        "num_comb_nodes": 90,
        "critical_path_delay": 15.0,
        "mean_toggle_rate": 0.12,
    }
    pred = {
        "wns": -0.05,
        "tns": -0.35,
        "power_mW": 1.25,
        "area_um2": 250.0,
        "summary": {
            "predicted_wns": -0.05,
            "predicted_tns": -0.35,
            "predicted_power_mW": 1.25,
            "predicted_area_um2": 250.0,
        },
        "area": {
            "predicted_seq_area": 45.22,
            "predicted_comb_area": 204.78,
        },
    }

    report = format_ppa_table("test_alu.v", "test_alu", stats, pred, elapsed_time=0.45)
    assert "test_alu.v" in report
    assert "250.00 um^2" in report
    assert "-0.050 ns" in report
    assert "1.250 mW" in report


def test_predict_verilog_file_integration():
    v_file = Path("data/raw_rtl/adder_ripple_8bit.v")
    model_checkpoint = Path("models/master_rtl_checkpoint.pkl")

    if not v_file.is_file() or not model_checkpoint.is_file():
        pytest.skip("Required test artifacts not found.")

    with model_checkpoint.open("rb") as f:
        model: MasterRTL = pickle.load(f)

    res = predict_verilog_file(v_file, model, target_tech="TSMC_28nm_TYP")
    assert res["top_module"] == "adder_ripple_8bit"
    assert "prediction" in res
    assert res["prediction"]["wns"] != 0.0
    assert res["prediction"]["area_um2"] > 0.0
    assert res["prediction"]["power_mW"] > 0.0
    assert res["transfer"] is not None
    assert res["transfer"]["transferred_area"] > 0.0


def test_generate_prediction_plot(tmp_path: Path):
    v_file = Path("data/raw_rtl/adder_ripple_8bit.v")
    model_checkpoint = Path("models/master_rtl_checkpoint.pkl")

    if not v_file.is_file() or not model_checkpoint.is_file():
        pytest.skip("Required test artifacts not found.")

    with model_checkpoint.open("rb") as f:
        model: MasterRTL = pickle.load(f)

    res = predict_verilog_file(v_file, model)
    png_path, pdf_path = generate_prediction_plot(res, tmp_path)

    assert png_path.is_file()
    assert pdf_path.is_file()
    assert png_path.stat().st_size > 10000
    assert pdf_path.stat().st_size > 5000


def test_build_dashboard_html(tmp_path: Path):
    from scripts.generate_html_dashboard import build_dashboard_html

    html_file = build_dashboard_html(tmp_path)
    assert html_file.is_file()
    content = html_file.read_text(encoding="utf-8")
    assert "MasterRTL: Presynthesis PPA Estimation" in content
    assert "Table III: Baseline Comparison" in content
