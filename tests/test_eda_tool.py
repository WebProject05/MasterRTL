"""Unit and integration tests for MasterRTL EDA Shell and Web Studio."""

from __future__ import annotations

import io
import json
import pickle
import threading
from http.server import HTTPServer
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pytest

from scripts.eda_gui import MasterRTLEDAHandler
from scripts.master_rtl_shell import MasterRTLEDAShell
from src.models.master_rtl import MasterRTL


@pytest.fixture(scope="module")
def sample_verilog() -> Path:
    p = Path("data/raw_rtl/adder_ripple_8bit.v")
    if not p.is_file():
        pytest.skip(f"Design {p} not found.")
    return p


@pytest.fixture(scope="module")
def checkpoint_path() -> Path:
    p = Path("models/master_rtl_checkpoint.pkl")
    if not p.is_file():
        pytest.skip(f"Model checkpoint {p} not found.")
    return p


def test_eda_shell_commands(sample_verilog: Path, checkpoint_path: Path, capsys: Any):
    shell = MasterRTLEDAShell(model_checkpoint=checkpoint_path)

    # 1. Test list_designs
    shell.do_list_designs("")
    captured = capsys.readouterr()
    assert "adder_ripple_8bit.v" in captured.out

    # 2. Test read_verilog
    shell.do_read_verilog(str(sample_verilog))
    captured = capsys.readouterr()
    assert "elaborated into SOG" in captured.out
    assert shell.current_result is not None

    # 3. Test report_ppa
    shell.do_report_ppa("")
    captured = capsys.readouterr()
    assert "Pre-Synthesis PPA Estimation Report" in captured.out
    assert "Top Module" in captured.out

    # 4. Test report_timing
    shell.do_report_timing("")
    captured = capsys.readouterr()
    assert "STATIC TIMING ANALYSIS (STA) REPORT" in captured.out
    assert "Worst Negative Slack" in captured.out

    # 5. Test report_area
    shell.do_report_area("")
    captured = capsys.readouterr()
    assert "SILICON AREA BREAKDOWN REPORT" in captured.out
    assert "Sequential Cell Area" in captured.out

    # 6. Test report_power
    shell.do_report_power("")
    captured = capsys.readouterr()
    assert "POWER DISSIPATION REPORT" in captured.out
    assert "Dynamic Toggle Power" in captured.out or "Dynamic Power" in captured.out or "Total Power" in captured.out

    # 7. Test set_target_technology & set_clock_period
    shell.do_set_target_technology("TSMC_28nm_TYP")
    shell.do_set_clock_period("0.8")
    assert shell.target_tech == "TSMC_28nm_TYP"
    assert shell.clk_period == 0.8

    # 8. Re-evaluate with new settings
    shell.do_report_ppa("")
    captured = capsys.readouterr()
    assert "TSMC 28nm TYP" in captured.out or "Transferred Technology" in captured.out


def test_eda_gui_endpoints(checkpoint_path: Path):
    server = HTTPServer(("127.0.0.1", 0), MasterRTLEDAHandler)
    host, port = server.server_address
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        base_url = f"http://{host}:{port}"

        # 1. GET /
        with urlopen(f"{base_url}/") as res:
            assert res.status == 200
            html = res.read().decode("utf-8")
            assert "MasterRTL EDA Studio" in html

        # 2. GET /api/designs
        with urlopen(f"{base_url}/api/designs") as res:
            assert res.status == 200
            designs = json.loads(res.read().decode("utf-8"))
            assert isinstance(designs, list)
            assert "adder_ripple_8bit.v" in designs

        # 3. GET /api/design?name=adder_ripple_8bit.v
        with urlopen(f"{base_url}/api/design?name=adder_ripple_8bit.v") as res:
            assert res.status == 200
            data = json.loads(res.read().decode("utf-8"))
            assert data["name"] == "adder_ripple_8bit.v"
            assert "module" in data["code"]

        # 4. POST /api/analyze
        payload = json.dumps({
            "design_name": "adder_ripple_8bit.v",
            "code": Path("data/raw_rtl/adder_ripple_8bit.v").read_text(encoding="utf-8"),
            "clk_period": 1.0,
            "target_tech": "NanGate_45nm",
            "generate_plot": False,
        }).encode("utf-8")

        req = Request(f"{base_url}/api/analyze", data=payload, headers={"Content-Type": "application/json"})
        with urlopen(req) as res:
            assert res.status == 200
            result = json.loads(res.read().decode("utf-8"))
            assert result["top_module"] == "adder_ripple_8bit"
            assert "prediction" in result
            assert result["prediction"]["wns"] != 0.0
            assert result["prediction"]["area_um2"] > 0.0

    finally:
        server.shutdown()
        server.server_close()
