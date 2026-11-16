"""MasterRTL Interactive EDA Shell (REPL).

Provides an authentic industry-grade EDA command-line shell (akin to dc_shell, pt_shell, genus).
Supports interactive RTL elaboration, SOG canonicalization, STA timing reports,
silicon area breakdown, toggle power analysis, foundry technology transfer, and GUI launch.
"""

from __future__ import annotations

import argparse
import cmd
import os
import pickle
import shlex
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict import (
    format_ppa_table,
    generate_prediction_plot,
    predict_verilog_file,
)
from src.models.master_rtl import MasterRTL

BANNER = """================================================================================
MasterRTL Interactive EDA Environment (v2.4.0)
Target Technology: NanGate 45nm @ 1.0 GHz (1.000 ns)
================================================================================
"""


class MasterRTLEDAShell(cmd.Cmd):
    """Interactive EDA Command Shell for MasterRTL."""

    intro = BANNER + "Type 'help' to view available EDA commands, or 'exit' / 'quit' to quit.\n"
    prompt = "master_rtl> "

    def __init__(self, model_checkpoint: Path | None = None) -> None:
        super().__init__()
        self.model_path = model_checkpoint or (PROJECT_ROOT / "models" / "master_rtl_checkpoint.pkl")
        self.model: MasterRTL | None = None
        self.current_design_path: Path | None = None
        self.current_result: dict[str, Any] | None = None
        self.clk_period: float = 1.0
        self.target_tech: str | None = None
        self._load_model()

    def _load_model(self) -> None:
        if self.model_path.is_file():
            try:
                with self.model_path.open("rb") as f:
                    self.model = pickle.load(f)
                print(f"[INIT] Loaded MasterRTL core model checkpoint: {self.model_path.name}")
            except Exception as e:
                print(f"[WARN] Failed to load checkpoint {self.model_path}: {e}")
        else:
            print(f"[WARN] Checkpoint not found at {self.model_path}. Run 'python scripts/train_all.py' first.")

    def default(self, line: str) -> None:
        cmd_name = line.split()[0] if line.split() else ""
        print(f"Error: Unknown EDA command '{cmd_name}'. Type 'help' for command list.")

    def emptyline(self) -> None:
        pass

    # -------------------------------------------------------------------------
    # Command: read_verilog
    # -------------------------------------------------------------------------
    def do_read_verilog(self, arg: str) -> None:
        """read_verilog <path/to/design.v>
        Elaborates and runs presynthesis SOG extraction on a Verilog design."""
        # Use posix=False on Windows so backslashes in paths are not treated as escape characters
        args = shlex.split(arg, posix=(os.name != "nt"))
        if not args:
            print("Syntax: read_verilog <path_to_verilog_file>")
            return

        target_str = args[0].strip('"').strip("'")
        target = Path(target_str).expanduser()
        if not target.is_absolute():
            # Check relative to cwd or data/raw_rtl
            cand1 = Path.cwd() / target
            cand2 = PROJECT_ROOT / "data" / "raw_rtl" / target.name
            target = cand1 if cand1.is_file() else (cand2 if cand2.is_file() else target)

        if not target.is_file():
            print(f"Error: File '{target}' not found.")
            return

        if self.model is None:
            print("Error: MasterRTL model checkpoint is not loaded. Train model first.")
            return

        print(f"\n[SYNTH] Reading & elaborating RTL: {target.name} ...")
        t0 = time.perf_counter()
        try:
            res = predict_verilog_file(
                verilog_file=target,
                model=self.model,
                clk_period=self.clk_period,
                target_tech=self.target_tech,
            )
            self.current_design_path = target
            self.current_result = res
            dt = time.perf_counter() - t0
            stats = res["stats"]
            print(f"[SUCCESS] Design '{res['top_module']}' elaborated into SOG in {dt:.2f}s:")
            print(f"          - 1-bit Total Nodes: {stats.get('total_nodes', 0)}")
            print(f"          - Sequential DFFs  : {stats.get('num_registers', 0)}")
            print(f"          - Comb Logic Gates : {stats.get('num_comb_nodes', 0)}")
            print(f"          - Critical Depth   : {stats.get('critical_path_delay', 0.0):.2f}")
            print(f"          - Mean Toggle Rate : {stats.get('mean_toggle_rate', 0.0):.3f}")
            print("Type 'report_ppa', 'report_timing', 'report_area', or 'report_power' for details.\n")
        except Exception as e:
            print(f"[ERROR] Synthesis failed: {e}\n")

    # -------------------------------------------------------------------------
    # Command: report_ppa
    # -------------------------------------------------------------------------
    def do_report_ppa(self, arg: str) -> None:
        """report_ppa [-plot]
        Displays the complete unified PPA estimation scorecard."""
        if not self.current_result:
            print("Error: No design currently loaded. Use 'read_verilog <file.v>' first.")
            return

        args = shlex.split(arg)
        do_plot = "-plot" in args or "--plot" in args

        res = self.current_result
        output_str = format_ppa_table(
            design_name=Path(res["file"]).name,
            top_module=res["top_module"],
            stats=res["stats"],
            pred=res["prediction"],
            transfer_res=res.get("transfer"),
            elapsed_time=res.get("elapsed_seconds", 0.0),
        )
        print(output_str)

        if do_plot:
            self.do_plot_ppa("")

    # -------------------------------------------------------------------------
    # Command: report_timing
    # -------------------------------------------------------------------------
    def do_report_timing(self, arg: str) -> None:
        """report_timing
        Generates detailed Static Timing Analysis (STA) metrics (WNS, TNS, critical paths)."""
        if not self.current_result:
            print("Error: No design currently loaded. Use 'read_verilog <file.v>' first.")
            return

        res = self.current_result
        stats = res["stats"]
        pred = res["prediction"]
        wns = pred["wns"]
        tns = pred["tns"]
        crit_delay = stats.get("critical_path_delay", 0.0)
        mean_delay = stats.get("mean_path_delay", 0.0)

        timing_status = "PASSED (Timing Met)" if wns >= 0.0 else f"VIOLATED (Negative Slack by {abs(wns):.3f} ns)"

        print("\n" + "=" * 74)
        print(f"  STATIC TIMING ANALYSIS (STA) REPORT: {res['top_module']}")
        print("=" * 74)
        print(f"  Target Clock Frequency  : {1.0 / self.clk_period:.2f} GHz (Clock Period = {self.clk_period:.3f} ns)")
        print(f"  Timing Closure Status   : {timing_status}")
        print("-" * 74)
        print(f"  Worst Negative Slack (WNS) : {wns:>9.3f} ns")
        print(f"  Total Negative Slack (TNS) : {tns:>9.3f} ns")
        print(f"  Critical Path Delay (STA)  : {crit_delay:>9.2f} topological units")
        print(f"  Mean Path Delay (STA)      : {mean_delay:>9.2f} topological units")
        print(f"  Endpoints Evaluated        : {stats.get('total_reg_paths', stats.get('num_registers', 0)):>9}")
        print("=" * 74 + "\n")

    # -------------------------------------------------------------------------
    # Command: report_area
    # -------------------------------------------------------------------------
    def do_report_area(self, arg: str) -> None:
        """report_area
        Reports standard-cell silicon area breakdown (Sequential DFFs vs Combinational Gates)."""
        if not self.current_result:
            print("Error: No design currently loaded. Use 'read_verilog <file.v>' first.")
            return

        res = self.current_result
        stats = res["stats"]
        pred = res["prediction"]
        area_dict = pred.get("area", {})
        tot_area = pred.get("area_um2", 0.0)
        seq_area = area_dict.get("predicted_seq_area", stats.get("num_registers", 0) * 4.522)
        comb_area = area_dict.get("predicted_comb_area", max(0.0, tot_area - seq_area))

        num_dff = stats.get("num_registers", 0)
        num_comb = stats.get("num_comb_nodes", 0)
        num_and = stats.get("num_and", 0)
        num_or = stats.get("num_or", 0)
        num_xor = stats.get("num_xor", 0)
        num_not = stats.get("num_not", 0)
        num_mux = stats.get("num_mux", 0)

        seq_pct = (seq_area / tot_area * 100) if tot_area > 0 else 0
        comb_pct = (comb_area / tot_area * 100) if tot_area > 0 else 0

        print("\n" + "=" * 74)
        print(f"  SILICON AREA BREAKDOWN REPORT: {res['top_module']}")
        print("=" * 74)
        print(f"  Standard Cell Library   : NanGate 45nm Open Cell Library (DFF: 4.522 um^2)")
        print(f"  Total Estimated Area    : {tot_area:>10.2f} um^2")
        print("-" * 74)
        print(f"  Sequential Cell Area    : {seq_area:>10.2f} um^2 ({seq_pct:>5.1f}%) | {num_dff} DFF Flip-Flops")
        print(f"  Combinational Cell Area : {comb_area:>10.2f} um^2 ({comb_pct:>5.1f}%) | {num_comb} 1-bit Gates")
        print("  Combinational Gate Breakdown:")
        print(f"    * AND Gates : {num_and:>6}")
        print(f"    * OR Gates  : {num_or:>6}")
        print(f"    * XOR Gates : {num_xor:>6}")
        print(f"    * NOT Gates : {num_not:>6}")
        print(f"    * MUX Gates : {num_mux:>6}")
        print("=" * 74 + "\n")

    # -------------------------------------------------------------------------
    # Command: report_power
    # -------------------------------------------------------------------------
    def do_report_power(self, arg: str) -> None:
        """report_power
        Reports toggle-rate switching dynamic power and static leakage power."""
        if not self.current_result:
            print("Error: No design currently loaded. Use 'read_verilog <file.v>' first.")
            return

        res = self.current_result
        stats = res["stats"]
        pred = res["prediction"]
        power = pred.get("power_mW", 0.0)
        toggle = stats.get("mean_toggle_rate", 0.0)
        wt_toggle = stats.get("weighted_toggle_rate", 0.0)

        # Approximate breakdown based on toggle activity ratio
        dyn_ratio = min(0.85, max(0.40, toggle * 4.0))
        dyn_pwr = power * dyn_ratio
        stat_pwr = power * (1.0 - dyn_ratio)

        print("\n" + "=" * 74)
        print(f"  POWER DISSIPATION REPORT: {res['top_module']}")
        print("=" * 74)
        print(f"  Target Clock Frequency  : {1.0 / self.clk_period:.2f} GHz (Vdd = 1.10 V)")
        print(f"  Mean Switching Activity : {toggle:.4f} (transitions / clock cycle)")
        print(f"  Weighted Capacitive Load: {wt_toggle:.2f} load units")
        print("-" * 74)
        print(f"  Dynamic Switching Power : {dyn_pwr:>10.3f} mW ({dyn_ratio*100:>5.1f}%)")
        print(f"  Static Leakage Power    : {stat_pwr:>10.3f} mW ({(1.0-dyn_ratio)*100:>5.1f}%)")
        print(f"  Total Power Consumption : {power:>10.3f} mW")
        print("=" * 74 + "\n")

    # -------------------------------------------------------------------------
    # Command: plot_ppa
    # -------------------------------------------------------------------------
    def do_plot_ppa(self, arg: str) -> None:
        """plot_ppa [-output <dir>]
        Generates and saves a high-res 4-panel visual PPA dashboard (.png and .pdf)."""
        if not self.current_result:
            print("Error: No design currently loaded. Use 'read_verilog <file.v>' first.")
            return

        out_dir = PROJECT_ROOT / "data" / "reports" / "predictions"
        args = shlex.split(arg)
        if "-output" in args:
            idx = args.index("-output")
            if idx + 1 < len(args):
                out_dir = Path(args[idx + 1]).expanduser()

        print(f"[PLOT] Rendering 4-panel visual PPA dashboard...")
        png_path, pdf_path = generate_prediction_plot(
            res=self.current_result,
            output_dir=out_dir,
            clk_period=self.clk_period,
        )
        print(f"[SUCCESS] Visual report saved to:")
        print(f"          * PNG (300 DPI): {png_path}")
        print(f"          * PDF (Vector) : {pdf_path}\n")

    # -------------------------------------------------------------------------
    # Command: set_target_technology
    # -------------------------------------------------------------------------
    def do_set_target_technology(self, arg: str) -> None:
        """set_target_technology <TSMC_28nm_TYP | TSMC_28nm_MIN | TSMC_28nm_MAX | NanGate_45nm_MIN | reset>
        Sets the target foundry process node for technology transfer scaling."""
        valid_nodes = ["TSMC_28nm_TYP", "TSMC_28nm_MIN", "TSMC_28nm_MAX", "NanGate_45nm_MIN", "NanGate_45nm_MAX"]
        node = arg.strip()
        if not node:
            print(f"Current target technology: {self.target_tech or 'NanGate 45nm (Baseline)'}")
            print(f"Available nodes: {', '.join(valid_nodes)} or 'reset'")
            return

        if node.lower() in {"reset", "none", "clear"}:
            self.target_tech = None
            print("[CONFIG] Technology transfer reset to baseline (NanGate 45nm).")
        elif node in valid_nodes:
            self.target_tech = node
            print(f"[CONFIG] Target technology set to: {node}")
        else:
            print(f"Error: Unknown technology '{node}'. Choose from: {', '.join(valid_nodes)}")

        # Re-evaluate current design if loaded
        if self.current_design_path and self.model:
            print("[UPDATE] Re-evaluating current design under new technology target...")
            self.do_read_verilog(str(self.current_design_path))

    # -------------------------------------------------------------------------
    # Command: set_clock_period
    # -------------------------------------------------------------------------
    def do_set_clock_period(self, arg: str) -> None:
        """set_clock_period <ns>
        Sets the design clock period in nanoseconds (e.g., 1.0 for 1.0 GHz)."""
        val_str = arg.strip()
        if not val_str:
            print(f"Current clock period: {self.clk_period:.3f} ns ({1.0/self.clk_period:.2f} GHz)")
            return
        try:
            val = float(val_str)
            if val <= 0.0:
                print("Error: Clock period must be positive.")
                return
            self.clk_period = val
            print(f"[CONFIG] Clock period set to {self.clk_period:.3f} ns ({1.0/self.clk_period:.2f} GHz)")
            if self.current_design_path and self.model:
                self.do_read_verilog(str(self.current_design_path))
        except ValueError:
            print(f"Error: Invalid number '{val_str}'.")

    # -------------------------------------------------------------------------
    # Command: list_designs
    # -------------------------------------------------------------------------
    def do_list_designs(self, arg: str) -> None:
        """list_designs [filter]
        Lists available synthesizable Verilog benchmark designs in data/raw_rtl/."""
        raw_dir = PROJECT_ROOT / "data" / "raw_rtl"
        if not raw_dir.is_dir():
            print("Error: data/raw_rtl directory not found.")
            return

        filter_str = arg.strip().lower()
        files = sorted(raw_dir.glob("*.v"))
        if filter_str:
            files = [f for f in files if filter_str in f.name.lower()]

        print(f"\nAvailable Verilog Benchmark Designs ({len(files)} designs in data/raw_rtl):")
        print("-" * 74)
        for i, f in enumerate(files, 1):
            print(f"  [{i:2d}] {f.name:<32}", end="\n" if i % 2 == 0 else "")
        if len(files) % 2 != 0:
            print()
        print("-" * 74)
        print("To analyze a design: read_verilog <filename>\n")

    # -------------------------------------------------------------------------
    # Command: gui
    # -------------------------------------------------------------------------
    def do_gui(self, arg: str) -> None:
        """gui
        Launches the local interactive Web EDA Studio in the default web browser."""
        import webbrowser
        from scripts.eda_gui import start_gui_server_thread

        url = "http://localhost:8080"
        print(f"\n[GUI] Starting MasterRTL Interactive EDA Web Studio at {url} ...")
        start_gui_server_thread(port=8080)
        webbrowser.open(url)
        print("[GUI] Studio opened in web browser. Press Ctrl+C in terminal if needed.\n")

    # -------------------------------------------------------------------------
    # Utility Commands
    # -------------------------------------------------------------------------
    def do_clear(self, arg: str) -> None:
        """clear
        Clears the terminal screen."""
        os.system("cls" if os.name == "nt" else "clear")

    def do_exit(self, arg: str) -> bool:
        """exit
        Exits the MasterRTL EDA shell."""
        print("Exiting MasterRTL EDA Shell. Goodbye.")
        return True

    def do_quit(self, arg: str) -> bool:
        """quit
        Exits the MasterRTL EDA shell."""
        return self.do_exit(arg)


def main() -> int:
    parser = argparse.ArgumentParser(description="MasterRTL Interactive EDA Shell")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "master_rtl_checkpoint.pkl",
        help="Path to MasterRTL model checkpoint",
    )
    args = parser.parse_args()

    shell = MasterRTLEDAShell(model_checkpoint=args.model)
    shell.cmdloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
