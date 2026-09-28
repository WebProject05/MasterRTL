"""Generate Figure 9 from IEEE TCAD 2025 MasterRTL paper.

Fig. 9: Comparison between prediction and ground truth for each PPA metric across all designs.
Contains 3 subplots:
  (a) WNS (Worst Negative Slack, ns)
  (b) Total Power (mW)
  (c) Total Area (um^2)
As requested, displaying the MasterRTL points against the ground truth reference line y = x.
Also exports a 4-panel version including TNS.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.models import (
    MasterRTL,
    evaluate_metrics,
    load_ppa_dataset,
)


def generate_figure_9(
    sog_dir: str | Path = "data/sog_graphs",
    output_dir: str | Path = "data/reports",
) -> tuple[Path, Path]:
    """Generate Figure 9 (3-panel publication-quality PPA comparison)."""
    out_path = Path(output_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[Fig 9] Loading dataset from {sog_dir}...")
    dataset = load_ppa_dataset(sog_dir)
    print(f"[Fig 9] Loaded {len(dataset)} designs.")

    print("[Fig 9] Training MasterRTL model on dataset...")
    model = MasterRTL()
    model.train(dataset)

    y_true_wns, y_pred_wns = [], []
    y_true_tns, y_pred_tns = [], []
    y_true_pwr, y_pred_pwr = [], []
    y_true_area, y_pred_area = [], []
    design_names = []

    for d in dataset:
        pred = model.predict(d["stats"], d["paths"])
        lbl = d["labels"]
        design_names.append(d["name"])
        y_true_wns.append(lbl["wns"])
        y_pred_wns.append(pred["summary"]["predicted_wns"])
        y_true_tns.append(lbl["tns"])
        y_pred_tns.append(pred["summary"]["predicted_tns"])
        y_true_pwr.append(lbl["power"])
        y_pred_pwr.append(pred["summary"]["predicted_power_mW"])
        y_true_area.append(lbl["area"])
        y_pred_area.append(pred["summary"]["predicted_area_um2"])

    # Compute metrics
    m_wns = evaluate_metrics(y_true_wns, y_pred_wns)
    m_tns = evaluate_metrics(y_true_tns, y_pred_tns)
    m_pwr = evaluate_metrics(y_true_pwr, y_pred_pwr)
    m_area = evaluate_metrics(y_true_area, y_pred_area)

    print("\n[Fig 9] Final Model Metrics:")
    print(f"  WNS:   R={m_wns['R']:.4f}, MAPE={m_wns['MAPE']:.1f}%, MAE={m_wns['MAE']:.4f}, RRSE={m_wns['RRSE']:.4f}")
    print(f"  Power: R={m_pwr['R']:.4f}, MAPE={m_pwr['MAPE']:.1f}%, MAE={m_pwr['MAE']:.4f}, RRSE={m_pwr['RRSE']:.4f}")
    print(f"  Area:  R={m_area['R']:.4f}, MAPE={m_area['MAPE']:.1f}%, MAE={m_area['MAE']:.2f}, RRSE={m_area['RRSE']:.4f}")
    print(f"  TNS:   R={m_tns['R']:.4f}, MAPE={m_tns['MAPE']:.1f}%, MAE={m_tns['MAE']:.4f}, RRSE={m_tns['RRSE']:.4f}")

    # Set publication styling
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    })

    # =========================================================================
    # 1. Figure 9: Exactly 3 subplots: (a) WNS, (b) Power, (c) Area
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    plots_config = [
        {
            "ax": axes[0],
            "title": "(a) Worst Negative Slack (WNS)",
            "x_label": "Ground Truth WNS (ns)",
            "y_label": "MasterRTL Predicted WNS (ns)",
            "x_data": np.array(y_true_wns),
            "y_data": np.array(y_pred_wns),
            "metrics": m_wns,
            "color": "#1f77b4",
            "edgecolor": "#0d3d63",
            "unit": "ns",
        },
        {
            "ax": axes[1],
            "title": "(b) Total Power",
            "x_label": "Ground Truth Power (mW)",
            "y_label": "MasterRTL Predicted Power (mW)",
            "x_data": np.array(y_true_pwr),
            "y_data": np.array(y_pred_pwr),
            "metrics": m_pwr,
            "color": "#2ca02c",
            "edgecolor": "#134f13",
            "unit": "mW",
        },
        {
            "ax": axes[2],
            "title": "(c) Total Area",
            "x_label": r"Ground Truth Area ($\mu m^2$)",
            "y_label": r"MasterRTL Predicted Area ($\mu m^2$)",
            "x_data": np.array(y_true_area),
            "y_data": np.array(y_pred_area),
            "metrics": m_area,
            "color": "#d62728",
            "edgecolor": "#670f10",
            "unit": r"$\mu m^2$",
        },
    ]

    for cfg in plots_config:
        ax = cfg["ax"]
        xd = cfg["x_data"]
        yd = cfg["y_data"]
        met = cfg["metrics"]

        min_val = min(float(np.min(xd)), float(np.min(yd)))
        max_val = max(float(np.max(xd)), float(np.max(yd)))
        span = max_val - min_val
        margin = max(span * 0.05, 0.05)
        lim_min = min_val - margin
        lim_max = max_val + margin

        # Ideal reference line y = x
        ax.plot([lim_min, lim_max], [lim_min, lim_max], color="#d9534f", linestyle="--", linewidth=1.8, label="Ideal (y = x)", zorder=2)

        # Scatter points for MasterRTL
        ax.scatter(
            xd,
            yd,
            c=cfg["color"],
            edgecolors=cfg["edgecolor"],
            alpha=0.82,
            s=48,
            linewidth=0.8,
            label="MasterRTL",
            zorder=3,
        )

        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel(cfg["x_label"], fontweight="bold")
        ax.set_ylabel(cfg["y_label"], fontweight="bold")
        ax.set_title(cfg["title"], fontweight="bold", pad=10)
        ax.grid(True, linestyle=":", alpha=0.6, zorder=1)
        ax.set_aspect("equal", adjustable="box")

        # Metric stats box
        textstr = (
            f"$R = {met['R']:.4f}$\n"
            f"MAPE $= {met['MAPE']:.1f}\\%$\n"
            f"MAE $= {met['MAE']:.3f}$\n"
            f"RRSE $= {met['RRSE']:.4f}$"
        )
        props = dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.92)
        ax.text(0.05, 0.70, textstr, transform=ax.transAxes, verticalalignment="top", bbox=props, fontsize=10)
        ax.legend(loc="lower right", framealpha=0.9)

    fig.suptitle("Fig. 9: Comparison Between Prediction and Ground Truth for Each PPA Metric (MasterRTL)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    fig9_png = out_path / "fig9_ppa_prediction.png"
    fig9_pdf = out_path / "fig9_ppa_prediction.pdf"
    fig.savefig(fig9_png, dpi=300, bbox_inches="tight")
    fig.savefig(fig9_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig 9] Successfully saved 3-panel figure to:\n  - {fig9_png}\n  - {fig9_pdf}")

    # =========================================================================
    # 2. Supplementary Figure: 4 subplots including TNS
    # =========================================================================
    fig4, axes4 = plt.subplots(2, 2, figsize=(13, 11), dpi=300)
    axes4_flat = axes4.flatten()

    all_4_configs = [
        {
            "ax": axes4_flat[0],
            "title": "(a) Worst Negative Slack (WNS)",
            "x_label": plots_config[0]["x_label"],
            "y_label": plots_config[0]["y_label"],
            "x_data": plots_config[0]["x_data"],
            "y_data": plots_config[0]["y_data"],
            "metrics": plots_config[0]["metrics"],
            "color": plots_config[0]["color"],
            "edgecolor": plots_config[0]["edgecolor"],
            "unit": plots_config[0]["unit"],
        },
        {
            "ax": axes4_flat[1],
            "title": "(b) Total Negative Slack (TNS)",
            "x_label": "Ground Truth TNS (ns)",
            "y_label": "MasterRTL Predicted TNS (ns)",
            "x_data": np.array(y_true_tns),
            "y_data": np.array(y_pred_tns),
            "metrics": m_tns,
            "color": "#9467bd",
            "edgecolor": "#4b286d",
            "unit": "ns",
        },
        {
            "ax": axes4_flat[2],
            "title": "(c) Total Power",
            "x_label": plots_config[1]["x_label"],
            "y_label": plots_config[1]["y_label"],
            "x_data": plots_config[1]["x_data"],
            "y_data": plots_config[1]["y_data"],
            "metrics": plots_config[1]["metrics"],
            "color": plots_config[1]["color"],
            "edgecolor": plots_config[1]["edgecolor"],
            "unit": plots_config[1]["unit"],
        },
        {
            "ax": axes4_flat[3],
            "title": "(d) Total Area",
            "x_label": plots_config[2]["x_label"],
            "y_label": plots_config[2]["y_label"],
            "x_data": plots_config[2]["x_data"],
            "y_data": plots_config[2]["y_data"],
            "metrics": plots_config[2]["metrics"],
            "color": plots_config[2]["color"],
            "edgecolor": plots_config[2]["edgecolor"],
            "unit": plots_config[2]["unit"],
        },
    ]

    for cfg in all_4_configs:
        ax = cfg["ax"]
        xd = cfg["x_data"]
        yd = cfg["y_data"]
        met = cfg["metrics"]

        min_val = min(float(np.min(xd)), float(np.min(yd)))
        max_val = max(float(np.max(xd)), float(np.max(yd)))
        span = max_val - min_val
        margin = max(span * 0.05, 0.05)
        lim_min = min_val - margin
        lim_max = max_val + margin

        ax.plot([lim_min, lim_max], [lim_min, lim_max], color="#d9534f", linestyle="--", linewidth=1.8, label="Ideal (y = x)", zorder=2)
        ax.scatter(xd, yd, c=cfg["color"], edgecolors=cfg["edgecolor"], alpha=0.82, s=48, linewidth=0.8, label="MasterRTL", zorder=3)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel(cfg["x_label"], fontweight="bold")
        ax.set_ylabel(cfg["y_label"], fontweight="bold")
        ax.set_title(cfg["title"], fontweight="bold", pad=8)
        ax.grid(True, linestyle=":", alpha=0.6, zorder=1)
        ax.set_aspect("equal", adjustable="box")

        textstr = (
            f"$R = {met['R']:.4f}$\n"
            f"MAPE $= {met['MAPE']:.1f}\\%$\n"
            f"MAE $= {met['MAE']:.3f}$\n"
            f"RRSE $= {met['RRSE']:.4f}$"
        )
        props = dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.92)
        ax.text(0.05, 0.70, textstr, transform=ax.transAxes, verticalalignment="top", bbox=props, fontsize=10)
        ax.legend(loc="lower right", framealpha=0.9)

    fig4.suptitle("MasterRTL Complete PPA Presynthesis vs Ground Truth (WNS, TNS, Power, Area)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    fig4_png = out_path / "fig9_ppa_with_tns.png"
    fig4.savefig(fig4_png, dpi=300, bbox_inches="tight")
    plt.close(fig4)
    print(f"[Fig 9] Successfully saved 4-panel figure to:\n  - {fig4_png}")

    return fig9_png, fig4_png


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 9 PPA comparison plots")
    parser.add_argument("--sog-dir", default="data/sog_graphs", help="Directory with SOG graphs and stats")
    parser.add_argument("--output-dir", default="data/reports", help="Output directory for plots")
    args = parser.parse_args()

    generate_figure_9(
        sog_dir=args.sog_dir,
        output_dir=args.output_dir,
    )
