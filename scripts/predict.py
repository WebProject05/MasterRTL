"""MasterRTL Single-File & Batch Inference CLI with Optional Visual PPA Dashboard.

Given any Verilog (.v / .sv) file, this script:
1. Synthesizes it into single-bit primitives and DFFs via Yosys (~0.5s).
2. Converts it into the canonical Simple Operator Graph (SOG).
3. Extracts topological timing paths, toggle activity, and cell distributions.
4. Evaluates the trained MasterRTL models to predict PPA:
   - Area (um^2) [Sequential DFFs + Combinational Logic]
   - Timing (WNS and TNS in ns)
   - Power (mW) [Dynamic switching + Static leakage]
   - Optional: Technology node transfer (e.g. TSMC 28nm, NanGate 45nm MIN)
5. Optional: Generates high-res visual 4-panel graphs (--plot).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep.sog_converter import (
    build_sog_from_json,
    save_graph_artifacts,
)
from src.data_prep.yosys_synthesis import run_yosys_synthesis
from src.models.master_rtl import MasterRTL


def _get_box_chars() -> dict[str, str]:
    """Return box drawing characters with fallback to ASCII."""
    try:
        if sys.stdout.encoding and "utf" in sys.stdout.encoding.lower():
            return {
                "tl": "┌", "tr": "┐", "bl": "└", "br": "┘",
                "h": "─", "v": "│", "ml": "├", "mr": "┤",
                "tm": "┬", "bm": "┴", "mm": "┼",
            }
    except Exception:
        pass
    return {
        "tl": "+", "tr": "+", "bl": "+", "br": "+",
        "h": "-", "v": "|", "ml": "+", "mr": "+",
        "tm": "+", "bm": "+", "mm": "+",
    }


def format_ppa_table(
    design_name: str,
    top_module: str,
    stats: dict[str, Any],
    pred: dict[str, Any],
    transfer_res: dict[str, Any] | None = None,
    elapsed_time: float = 0.0,
) -> str:
    """Format PPA predictions into a clean, simple, and perfectly aligned report."""
    b = _get_box_chars()

    summary = pred.get("summary", {})
    wns = summary.get("predicted_wns", pred.get("wns", 0.0))
    tns = summary.get("predicted_tns", pred.get("tns", 0.0))
    power = summary.get("predicted_power_mW", pred.get("power_mW", 0.0))
    area = summary.get("predicted_area_um2", pred.get("area_um2", 0.0))

    area_details = pred.get("area", {})
    seq_area = area_details.get("predicted_seq_area", stats.get("num_registers", 0) * 4.522)
    comb_area = area_details.get("predicted_comb_area", max(0.0, area - seq_area))

    timing_info = "MET (PASS)" if wns >= 0.0 else f"VIOLATED ({abs(wns):.3f} ns)"

    nodes = stats.get("total_nodes", 0)
    dffs = stats.get("num_registers", 0)
    comb = stats.get("num_comb_nodes", 0)
    depth = stats.get("critical_path_delay", 0.0)
    toggle = stats.get("mean_toggle_rate", 0.0)

    # Box dimensions: total width = 76 characters
    w_inner = 74
    c1_w = 14
    c2_w = 15
    c3_w = 37

    line_top = f"{b['tl']}{b['h'] * w_inner}{b['tr']}"
    line_mid = f"{b['ml']}{b['h'] * (c1_w + 2)}{b['tm']}{b['h'] * (c2_w + 2)}{b['tm']}{b['h'] * (c3_w + 2)}{b['mr']}"
    line_sub = f"{b['ml']}{b['h'] * (c1_w + 2)}{b['mm']}{b['h'] * (c2_w + 2)}{b['mm']}{b['h'] * (c3_w + 2)}{b['mr']}"
    line_div = f"{b['ml']}{b['h'] * w_inner}{b['mr']}"
    line_bot = f"{b['bl']}{b['h'] * (c1_w + 2)}{b['bm']}{b['h'] * (c2_w + 2)}{b['bm']}{b['h'] * (c3_w + 2)}{b['br']}"

    def row_full(text: str) -> str:
        padded = text[:w_inner - 2].ljust(w_inner - 2)
        return f"{b['v']} {padded} {b['v']}"

    def row_cols(c1: str, c2: str, c3: str) -> str:
        f1 = c1[:c1_w].ljust(c1_w)
        f2 = c2[:c2_w].ljust(c2_w)
        f3 = c3[:c3_w].ljust(c3_w)
        return f"{b['v']} {f1} {b['v']} {f2} {b['v']} {f3} {b['v']}"

    circuit_str = f"{nodes} nodes | {dffs} DFFs | {comb} gates | depth {depth:.1f} | toggle {toggle:.3f}"
    timing_str = f"Slack: {timing_info}"

    lines = [
        "",
        line_top,
        row_full("MasterRTL: Pre-Synthesis PPA Estimation Report"),
        line_div,
        row_full(f"Design File : {design_name}"),
        row_full(f"Top Module  : {top_module}"),
        row_full(f"SOG Circuit : {circuit_str}"),
        row_full(f"Inference   : {elapsed_time:.2f}s (NanGate 45nm @ 1.0 GHz)"),
        line_mid,
        row_cols("TARGET METRIC", "PREDICTED VALUE", "PHYSICAL BREAKDOWN / STATUS"),
        line_sub,
        row_cols("Total Area", f"{area:>8.2f} um^2", f"Seq: {seq_area:.1f} um^2 | Comb: {comb_area:.1f} um^2"),
        row_cols("Timing (WNS)", f"{wns:>8.3f} ns", timing_str),
        row_cols("Timing (TNS)", f"{tns:>8.3f} ns", "Cumulative endpoint slack"),
        row_cols("Total Power", f"{power:>8.3f} mW", "Dynamic switching + static leakage"),
    ]

    if transfer_res:
        t_tech = transfer_res.get("target_technology", "Target Node")
        t_power = transfer_res.get("transferred_power", 0.0)
        t_area = transfer_res.get("transferred_area", 0.0)
        t_wns = transfer_res.get("transferred_wns", 0.0)
        lines.extend([
            line_sub,
            row_cols(f"Node Transfer", f"{t_tech}", "Transferred Technology Scaling"),
            row_cols("  -> Area", f"{t_area:>8.2f} um^2", f"Target silicon area on {t_tech}"),
            row_cols("  -> WNS", f"{t_wns:>8.3f} ns", f"Target worst slack on {t_tech}"),
            row_cols("  -> Power", f"{t_power:>8.3f} mW", f"Target total power on {t_tech}"),
        ])

    lines.extend([
        line_bot,
        "",
    ])
    return "\n".join(lines)


def generate_prediction_plot(
    res: dict[str, Any],
    output_dir: Path,
    clk_period: float = 1.0,
) -> tuple[Path, Path]:
    """Generate a clean, publication-grade 4-panel visual dashboard for the tested design."""
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    design_name = Path(res["file"]).stem
    top_mod = res.get("top_module", design_name)
    stats = res.get("stats", {})
    pred = res.get("prediction", {})
    transfer = res.get("transfer")
    paths = res.get("paths", [])

    # Matplotlib academic styling conforming to IEEE TCAD
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.titlesize": 12.5,
        "grid.linestyle": ":",
        "grid.alpha": 0.55,
    })

    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(f"MasterRTL Presynthesis PPA Dashboard: {design_name} (Top: {top_mod})", fontsize=13, fontweight="bold", y=0.98)

    # -------------------------------------------------------------------------
    # Panel 1 (Top Left): Gate Primitives Breakdown
    # -------------------------------------------------------------------------
    ax1 = axs[0, 0]
    gate_labels = ["DFF (Seq)", "AND", "OR", "XOR", "NOT", "MUX"]
    gate_counts = [
        stats.get("num_registers", 0),
        stats.get("num_and", 0),
        stats.get("num_or", 0),
        stats.get("num_xor", 0),
        stats.get("num_not", 0),
        stats.get("num_mux", 0),
    ]
    tot_gates = max(1, sum(gate_counts))
    gate_colors = ["#2b5c8f", "#2ca02c", "#17becf", "#ff7f0e", "#d62728", "#9467bd"]

    y_pos = np.arange(len(gate_labels))
    bars = ax1.barh(y_pos, gate_counts, color=gate_colors, edgecolor="black", height=0.6, alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(gate_labels, fontweight="normal")
    ax1.invert_yaxis()
    ax1.set_xlabel("Count (1-bit Nodes)")
    ax1.set_title("1. SOG Logic Operator Breakdown", fontweight="bold")
    ax1.grid(axis="x", linestyle=":", alpha=0.6)

    max_val = max(gate_counts) if gate_counts else 10
    ax1.set_xlim(0, max_val * 1.35)
    for bar, val in zip(bars, gate_counts):
        pct = (val / tot_gates) * 100
        ax1.text(val + max_val * 0.02, bar.get_y() + bar.get_height() / 2, f"{val} ({pct:.1f}%)", va="center", fontsize=8.5)

    # -------------------------------------------------------------------------
    # Panel 2 (Top Right): Silicon Area Allocation
    # -------------------------------------------------------------------------
    ax2 = axs[0, 1]
    tot_area = pred.get("summary", {}).get("predicted_area_um2", pred.get("area_um2", 100.0))
    area_details = pred.get("area", {})
    seq_area = area_details.get("predicted_seq_area", stats.get("num_registers", 0) * 4.522)
    comb_area = area_details.get("predicted_comb_area", max(0.0, tot_area - seq_area))

    wedges, texts, autotexts = ax2.pie(
        [seq_area, comb_area],
        labels=["Sequential (DFF)", "Combinational"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2b5c8f", "#2ca02c"],
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontsize(9)

    ax2.text(0, 0, f"{tot_area:.1f} um^2\nTotal Area", ha="center", va="center", fontsize=10.5, fontweight="bold")
    ax2.set_title("2. Silicon Area Breakdown (NanGate 45nm)", fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel 3 (Bottom Left): Path Slack Spectrum & Timing Margin
    # -------------------------------------------------------------------------
    ax3 = axs[1, 0]
    wns = pred.get("summary", {}).get("predicted_wns", pred.get("wns", 0.0))
    tns = pred.get("summary", {}).get("predicted_tns", pred.get("tns", 0.0))

    if paths:
        raw_delays = [p.get("delay", 1.0) * 0.035 for p in paths]
        path_slacks = [clk_period - d for d in raw_delays]
    else:
        s_dist = pred.get("timing", {}).get("slack_distribution", {})
        worst = s_dist.get("worst", wns)
        p50 = s_dist.get("p50", wns + 0.3)
        path_slacks = list(np.linspace(worst, max(p50 + 0.4, 0.5), 30))

    n, bins, patches = ax3.hist(path_slacks, bins=15, edgecolor="black", alpha=0.75, color="#17becf")
    for p, b_left, b_right in zip(patches, bins[:-1], bins[1:]):
        if (b_left + b_right) / 2 < 0:
            p.set_facecolor("#d62728")
        else:
            p.set_facecolor("#2ca02c")

    ax3.axvline(0.0, color="black", linestyle="--", linewidth=1.5, label="Deadline (Slack = 0 ns)")
    ax3.axvline(wns, color="#d62728", linestyle=":", linewidth=2, label=f"WNS = {wns:.3f} ns")
    ax3.set_xlabel("Slack (ns)  [Negative = Timing Violation]")
    ax3.set_ylabel("Path Count")
    ax3.set_title(f"3. Timing Slack Spectrum (TNS = {tns:.3f} ns)", fontweight="bold")
    ax3.legend(loc="upper left", framealpha=0.9)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel 4 (Bottom Right): PPA Metric Scorecard / Foundry Transfer
    # -------------------------------------------------------------------------
    ax4 = axs[1, 1]
    power = pred.get("summary", {}).get("predicted_power_mW", pred.get("power_mW", 0.0))

    if transfer:
        t_tech = transfer.get("target_technology", "TSMC 28nm")
        t_area = transfer.get("transferred_area", tot_area)
        t_power = transfer.get("transferred_power", power)
        t_wns = transfer.get("transferred_wns", wns)

        cats = ["Area (um^2 / 10)", "Power (mW)", "WNS Slack (ns)"]
        vals_45nm = [tot_area / 10.0, power, wns]
        vals_trans = [t_area / 10.0, t_power, t_wns]

        x = np.arange(len(cats))
        w_bar = 0.35
        ax4.bar(x - w_bar / 2, vals_45nm, w_bar, label="NanGate 45nm", color="#2b5c8f", edgecolor="black", alpha=0.85)
        ax4.bar(x + w_bar / 2, vals_trans, w_bar, label=t_tech, color="#ff7f0e", edgecolor="black", alpha=0.85)
        ax4.set_xticks(x)
        ax4.set_xticklabels(cats, fontweight="normal")
        ax4.set_title("4. Technology Node Transfer Scaling", fontweight="bold")
        ax4.legend(loc="upper left", framealpha=0.9)
        ax4.grid(axis="y", linestyle=":", alpha=0.6)
        ax4.axhline(0, color="gray", linewidth=0.8)
    else:
        ax4.axis("off")
        card_data = [
            ("Predicted Silicon Area", f"{tot_area:.2f} um^2", "#2b5c8f", f"Seq: {seq_area:.1f} | Comb: {comb_area:.1f}"),
            ("Worst Slack (WNS)", f"{wns:.3f} ns", "#d62728" if wns < 0 else "#2ca02c", "MET (PASS)" if wns >= 0 else f"VIOLATED ({abs(wns):.3f} ns)"),
            ("Total Slack (TNS)", f"{tns:.3f} ns", "#d62728" if tns < 0 else "#2ca02c", "Cumulative endpoint slack"),
            ("Total Power", f"{power:.3f} mW", "#9467bd", "Dynamic + static leakage"),
        ]

        y_starts = [0.80, 0.55, 0.30, 0.05]
        for (m_title, m_val, col, desc), y_s in zip(card_data, y_starts):
            bbox_props = dict(boxstyle="round,pad=0.5", facecolor=col, alpha=0.15, edgecolor=col, linewidth=1.5)
            ax4.text(0.05, y_s + 0.08, m_title, fontsize=9.5, fontweight="bold", color="#333333", transform=ax4.transAxes)
            ax4.text(0.05, y_s - 0.02, m_val, fontsize=12, fontweight="bold", color=col, bbox=bbox_props, transform=ax4.transAxes)
            ax4.text(0.55, y_s - 0.02, desc, fontsize=8.5, color="#555555", transform=ax4.transAxes, va="center")

        ax4.set_title("4. Summary PPA Estimation Scorecard", fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = output_dir / f"{design_name}_ppa.png"
    pdf_path = output_dir / f"{design_name}_ppa.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    return png_path, pdf_path


def predict_verilog_file(
    verilog_file: Path,
    model: MasterRTL,
    clk_period: float = 1.0,
    target_tech: str | None = None,
    output_dir: Path | None = None,
    yosys_bin: str = "yosys",
) -> dict[str, Any]:
    """Execute end-to-end prediction for a single Verilog design file."""
    start_time = time.perf_counter()

    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "inference_temp"
    synth_dir = output_dir / "yosys_synth"
    sog_dir = output_dir / "sog_graphs"
    synth_dir.mkdir(parents=True, exist_ok=True)
    sog_dir.mkdir(parents=True, exist_ok=True)

    # 1. Synthesize Verilog to bit-level single-bit gates via Yosys
    synth_res = run_yosys_synthesis(
        input_path=verilog_file,
        output_dir=synth_dir,
        yosys_bin=yosys_bin,
    )
    json_path = Path(synth_res["json_ast"])
    top_module = synth_res["top_module"]

    # 2. Build SOG graph and extract paths/features
    graph = build_sog_from_json(json_path)
    artifacts = save_graph_artifacts(graph, sog_dir, verilog_file.stem)

    with open(artifacts["stats"], "r", encoding="utf-8") as f:
        stats = json.load(f)
    with open(artifacts["features"], "rb") as f:
        paths = pickle.load(f)

    # 3. Model Prediction
    pred = model.predict(stats, paths, clk_period=clk_period)

    # 4. Optional Technology Transfer
    transfer_res = None
    if target_tech:
        transfer_res = model.transfer(pred, stats, target_technology=target_tech)

    elapsed = time.perf_counter() - start_time

    return {
        "file": str(verilog_file),
        "top_module": top_module,
        "stats": stats,
        "paths": paths,
        "prediction": pred,
        "transfer": transfer_res,
        "elapsed_seconds": round(elapsed, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MasterRTL: Instant Presynthesis PPA Estimation from Raw Verilog"
    )
    parser.add_argument(
        "verilog_path",
        type=Path,
        help="Path to a Verilog (.v / .sv) file or directory of Verilog files",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "master_rtl_checkpoint.pkl",
        help="Path to trained MasterRTL model checkpoint (.pkl)",
    )
    parser.add_argument(
        "--clk-period",
        type=float,
        default=1.0,
        help="Target clock period in nanoseconds (default: 1.0 ns = 1.0 GHz)",
    )
    parser.add_argument(
        "--transfer",
        type=str,
        default=None,
        choices=["TSMC_28nm_TYP", "TSMC_28nm_MIN", "TSMC_28nm_MAX", "NanGate_45nm_MIN", "NanGate_45nm_MAX"],
        help="Optional technology corner to transfer predictions to (e.g. TSMC_28nm_TYP)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a publication-grade 4-panel visual PPA dashboard (.png and .pdf)",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "reports" / "predictions",
        help="Directory to save generated visual graphs (default: data/reports/predictions)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON results instead of formatted table",
    )
    parser.add_argument(
        "--yosys-bin",
        type=str,
        default="yosys",
        help="Yosys executable name or path (auto-detects OSS CAD Suite)",
    )
    args = parser.parse_args()

    v_path = args.verilog_path.expanduser().resolve()
    if not v_path.exists():
        print(f"Error: Verilog path does not exist: {v_path}", file=sys.stderr)
        return 1

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        print(f"Error: MasterRTL checkpoint not found at: {model_path}", file=sys.stderr)
        print("Please train the model first by running: python scripts/train_all.py", file=sys.stderr)
        return 1

    # Load trained model
    with model_path.open("rb") as f:
        model: MasterRTL = pickle.load(f)

    # Collect files
    if v_path.is_file():
        files = [v_path]
    else:
        files = sorted(list(v_path.glob("*.v")) + list(v_path.glob("*.sv")))
        files = [f for f in files if not f.name.endswith(".synth.v")]

    if not files:
        print(f"Error: No Verilog (.v / .sv) files found in: {v_path}", file=sys.stderr)
        return 1

    all_results = []
    for f in files:
        try:
            res = predict_verilog_file(
                verilog_file=f,
                model=model,
                clk_period=args.clk_period,
                target_tech=args.transfer,
                yosys_bin=args.yosys_bin,
            )
            all_results.append(res)

            if not args.json:
                report_str = format_ppa_table(
                    design_name=f.name,
                    top_module=res["top_module"],
                    stats=res["stats"],
                    pred=res["prediction"],
                    transfer_res=res["transfer"],
                    elapsed_time=res["elapsed_seconds"],
                )
                print(report_str)

                # Generate graph if requested
                if args.plot:
                    png_p, pdf_p = generate_prediction_plot(
                        res=res,
                        output_dir=args.plot_dir,
                        clk_period=args.clk_period,
                    )
                    print(f"  [+] Visual PPA Dashboard saved to:\n      * PNG: {png_p}\n      * PDF: {pdf_p}\n")
        except Exception as e:
            print(f"[ERROR] Failed to process {f.name}: {e}", file=sys.stderr)

    if args.json:
        print(json.dumps(all_results if len(all_results) > 1 else all_results[0], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
