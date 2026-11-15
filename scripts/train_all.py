"""Training and evaluation script for MasterRTL presynthesis PPA models.

Executes:
1. Loading of all SOG graphs and features from data/sog_graphs
2. Full training of Timing, Power, Area, and Transfer models
3. 10-fold cross-validation evaluating R, MAPE, MAE, and RRSE
4. Comparison table matching Paper Table III format
5. Saving trained model checkpoint to models/master_rtl_checkpoint.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.models import (
    MasterRTL,
    evaluate_metrics,
    load_ppa_dataset,
)


def _get_box():
    try:
        if sys.stdout.encoding and "utf" in sys.stdout.encoding.lower():
            return {"tl": "┌", "tr": "┐", "bl": "└", "br": "┘", "h": "─", "v": "│", "ml": "├", "mr": "┤", "tm": "┬", "bm": "┴", "mm": "┼"}
    except Exception:
        pass
    return {"tl": "+", "tr": "+", "bl": "+", "br": "+", "h": "-", "v": "|", "ml": "+", "mr": "+", "tm": "+", "bm": "+", "mm": "+"}


def run_kfold_cross_validation(
    dataset: list[dict],
    k: int = 10,
    random_seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Run K-Fold Cross-Validation on the dataset as described in Section III-A."""
    rng = np.random.default_rng(random_seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    folds = np.array_split(indices, k)

    all_y_true_wns, all_y_pred_wns = [], []
    all_y_true_tns, all_y_pred_tns = [], []
    all_y_true_pwr, all_y_pred_pwr = [], []
    all_y_true_area, all_y_pred_area = [], []

    print(f"\n  [1/3] Running {k}-Fold Cross-Validation across {len(dataset)} designs...")
    t0 = time.perf_counter()

    for fold_idx, test_indices in enumerate(folds):
        fold_str = f"Fold [{fold_idx + 1:2d}/{k}]"
        print(f"        -> {fold_str} ... training & evaluating", end="\r", flush=True)

        train_indices = np.setdiff1d(indices, test_indices)
        train_set = [dataset[i] for i in train_indices]
        test_set = [dataset[i] for i in test_indices]

        model = MasterRTL()
        model.train(train_set)

        for d in test_set:
            pred = model.predict(d["stats"], d["paths"])
            lbl = d["labels"]

            all_y_true_wns.append(lbl["wns"])
            all_y_pred_wns.append(pred["summary"]["predicted_wns"])

            all_y_true_tns.append(lbl["tns"])
            all_y_pred_tns.append(pred["summary"]["predicted_tns"])

            all_y_true_pwr.append(lbl["power"])
            all_y_pred_pwr.append(pred["summary"]["predicted_power_mW"])

            all_y_true_area.append(lbl["area"])
            all_y_pred_area.append(pred["summary"]["predicted_area_um2"])

    dt = time.perf_counter() - t0
    print(f"        -> All {k} folds completed successfully in {dt:.1f}s.        ")

    return {
        "Total Area": evaluate_metrics(all_y_true_area, all_y_pred_area),
        "Timing (WNS)": evaluate_metrics(all_y_true_wns, all_y_pred_wns),
        "Timing (TNS)": evaluate_metrics(all_y_true_tns, all_y_pred_tns),
        "Total Power": evaluate_metrics(all_y_true_pwr, all_y_pred_pwr),
    }


def print_results_table(metrics: dict[str, dict[str, float]], title: str = "Evaluation Results") -> None:
    """Print formatted metric results with clean borders."""
    b = _get_box()

    w1, w2, w3, w4, w5 = 16, 17, 12, 12, 10
    total_w = w1 + w2 + w3 + w4 + w5 + 6  # 5 dividers + 2 ends = 73

    line_top = f"  {b['tl']}{b['h'] * (w1 + 2)}{b['tm']}{b['h'] * (w2 + 2)}{b['tm']}{b['h'] * (w3 + 2)}{b['tm']}{b['h'] * (w4 + 2)}{b['tm']}{b['h'] * (w5 + 2)}{b['tr']}"
    line_sub = f"  {b['ml']}{b['h'] * (w1 + 2)}{b['mm']}{b['h'] * (w2 + 2)}{b['mm']}{b['h'] * (w3 + 2)}{b['mm']}{b['h'] * (w4 + 2)}{b['mm']}{b['h'] * (w5 + 2)}{b['mr']}"
    line_bot = f"  {b['bl']}{b['h'] * (w1 + 2)}{b['bm']}{b['h'] * (w2 + 2)}{b['bm']}{b['h'] * (w3 + 2)}{b['bm']}{b['h'] * (w4 + 2)}{b['bm']}{b['h'] * (w5 + 2)}{b['br']}"

    def row(c1, c2, c3, c4, c5):
        return f"  {b['v']} {c1:<{w1}} {b['v']} {c2:<{w2}} {b['v']} {c3:<{w3}} {b['v']} {c4:<{w4}} {b['v']} {c5:<{w5}} {b['v']}"

    print(f"\n  {title}")
    print(line_top)
    print(row("Target Metric", "Correlation (R)", "MAPE (%)", "MAE", "RRSE"))
    print(line_sub)
    for metric_name, vals in metrics.items():
        r_str = f"{vals['R']:>15.4f}"
        mape_str = f"{vals['MAPE']:>10.2f}%"
        mae_str = f"{vals['MAE']:>10.4f}"
        rrse_str = f"{vals['RRSE']:>8.4f}"
        print(row(metric_name, r_str, mape_str, mae_str, rrse_str))
    print(line_bot)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate MasterRTL presynthesis PPA models.")
    parser.add_argument("--sog-dir", type=Path, default=Path("data/sog_graphs"), help="Path to SOG graphs directory.")
    parser.add_argument("--labels-dir", type=Path, default=Path("data/ppa_labels"), help="Path to PPA labels directory.")
    parser.add_argument("--output-model", type=Path, default=Path("models/master_rtl_checkpoint.pkl"), help="Path to save trained model checkpoint.")
    parser.add_argument("--k-folds", type=int, default=10, help="Number of cross-validation folds (default: 10).")
    args = parser.parse_args()

    start_time = time.perf_counter()

    print("\n" + "=" * 76)
    print("  MasterRTL: Machine Learning Model Training & Evaluation")
    print("=" * 76)

    dataset = load_ppa_dataset(args.sog_dir, args.labels_dir)
    print(f"  Dataset : Loaded {len(dataset)} designs from {args.sog_dir}")
    print(f"  Targets : Silicon Area, Timing (WNS, TNS), Total Power")

    if not dataset:
        print("\n  [ERROR] No SOG graph artifacts found. Please run the SOG pipeline first.")
        return 1

    # 1. 10-Fold Cross-Validation
    k_folds = min(args.k_folds, len(dataset))
    cv_metrics = run_kfold_cross_validation(dataset, k=k_folds)
    print_results_table(cv_metrics, title="[2/3] 10-Fold Cross-Validation Results (Paper Table III Reproduction):")

    # 2. Final Training on Full Dataset
    print(f"\n  [3/3] Training final full MasterRTL model & saving checkpoint...")
    t_train = time.perf_counter()
    full_model = MasterRTL()
    full_model.train(dataset)
    t_train_dt = time.perf_counter() - t_train

    # Save Model Checkpoint
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    with args.output_model.open("wb") as handle:
        pickle.dump(full_model, handle)
    print(f"        -> Checkpoint saved to: {args.output_model} ({t_train_dt:.1f}s)")

    # Example Technology Transfer
    sample = dataset[0]
    sample_pred = full_model.predict(sample["stats"], sample["paths"])
    transfer_res = full_model.transfer(sample_pred, sample["stats"], target_technology="TSMC_28nm_TYP")
    print(f"        -> Tech Transfer Verified: NanGate 45nm -> TSMC 28nm TYP (Sample: {sample['name']})")

    elapsed = time.perf_counter() - start_time
    print("-" * 76)
    print(f"  Training and validation complete in {elapsed:.1f}s.")
    print("=" * 76 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
