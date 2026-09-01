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

    print(f"\nRunning {k}-Fold Cross-Validation on {len(dataset)} designs...")

    for fold_idx, test_indices in enumerate(folds):
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

    return {
        "WNS": evaluate_metrics(all_y_true_wns, all_y_pred_wns),
        "TNS": evaluate_metrics(all_y_true_tns, all_y_pred_tns),
        "Power": evaluate_metrics(all_y_true_pwr, all_y_pred_pwr),
        "Area": evaluate_metrics(all_y_true_area, all_y_pred_area),
    }


def print_results_table(metrics: dict[str, dict[str, float]], title: str = "Evaluation Results") -> None:
    """Print formatted metric results matching Paper Table III style."""
    print(f"\n==========================================================================")
    print(f" {title}")
    print(f"==========================================================================")
    print(f"{'Target Metric':<12} | {'Correlation (R)':<16} | {'MAPE (%)':<12} | {'MAE':<12} | {'RRSE':<10}")
    print(f"--------------------------------------------------------------------------")
    for metric_name, vals in metrics.items():
        r = vals["R"]
        mape = f"{vals['MAPE']:.1f}%"
        mae = f"{vals['MAE']:.4f}"
        rrse = f"{vals['RRSE']:.4f}"
        print(f"{metric_name:<12} | {r:<16.4f} | {mape:<12} | {mae:<12} | {rrse:<10}")
    print(f"==========================================================================\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate MasterRTL presynthesis PPA models.")
    parser.add_argument("--sog-dir", type=Path, default=Path("data/sog_graphs"), help="Path to SOG graphs directory.")
    parser.add_argument("--labels-dir", type=Path, default=Path("data/ppa_labels"), help="Path to PPA labels directory.")
    parser.add_argument("--output-model", type=Path, default=Path("models/master_rtl_checkpoint.pkl"), help="Path to save trained model checkpoint.")
    parser.add_argument("--k-folds", type=int, default=10, help="Number of cross-validation folds (default: 10).")
    args = parser.parse_args()

    print("=================================================================")
    print(" MasterRTL: Machine Learning Model Training & Evaluation")
    print("=================================================================")

    start_time = time.perf_counter()
    dataset = load_ppa_dataset(args.sog_dir, args.labels_dir)
    print(f"Loaded {len(dataset)} designs from {args.sog_dir}")

    if not dataset:
        print("ERROR: No SOG graph artifacts found. Please run the SOG pipeline first.")
        return 1

    # 1. 10-Fold Cross-Validation
    k_folds = min(args.k_folds, len(dataset))
    cv_metrics = run_kfold_cross_validation(dataset, k=k_folds)
    print_results_table(cv_metrics, title=f"10-Fold Cross-Validation Results (Paper Table III Reproduction)")

    # 2. Final Training on Full Dataset
    print(f"Training final MasterRTL model on all {len(dataset)} designs...")
    full_model = MasterRTL()
    full_model.train(dataset)

    train_metrics = full_model.evaluate_dataset(dataset)
    print_results_table(train_metrics, title="Full Dataset Final Model Fit Metrics")

    # 3. Test Transfer Model on an Example
    sample = dataset[0]
    sample_pred = full_model.predict(sample["stats"], sample["paths"])
    transfer_res = full_model.transfer(sample_pred, sample["stats"], target_technology="TSMC_28nm_TYP")
    print("Example Technology Transfer (NanGate 45nm -> TSMC 28nm TYP):")
    print(f"  Design:               {sample['name']}")
    print(f"  Source Power / Area:  {sample_pred['summary']['predicted_power_mW']:.4f} mW | {sample_pred['summary']['predicted_area_um2']:.1f} um^2")
    print(f"  Transferred P/A:      {transfer_res['transferred_power']:.4f} mW | {transfer_res['transferred_area']:.1f} um^2")

    # 4. Save Model Checkpoint
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    with args.output_model.open("wb") as handle:
        pickle.dump(full_model, handle)
    print(f"\nSaved trained MasterRTL model checkpoint to: {args.output_model}")

    elapsed = time.perf_counter() - start_time
    print(f"Completed in {elapsed:.2f}s.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
