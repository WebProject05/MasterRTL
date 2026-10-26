"""Master simulation script for MasterRTL and all Presynthesis PPA Baselines.

Runs the entire pipeline end-to-end with a single command:
1. Benchmark verification & registration (156 diverse hardware designs)
2. Realistic EDA post-synthesis label generation (NanGate 45nm calibration)
3. MasterRTL core model training & checkpoint saving (models/master_rtl_checkpoint.pkl)
4. Comprehensive 10-fold cross-validation baseline comparison (Table III)
   - [21] ICCAD'22 Register-Tree model on AST-alike
   - [21] ICCAD'22 Register-Tree model on SOG
   - [22] ISCA'22 SNS Pseudo-Path model on AST-alike
   - [22] ISCA'22 SNS Pseudo-Path model on SOG
   - MasterRTL on AST-alike
   - MasterRTL on SOG
5. Publication paper figure generation (Figs. 7, 8, 9, 10 in PNG & PDF)
6. Full test suite validation via pytest

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --help
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_banner(title: str) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print(f" {title}")
    print(f"{sep}\n")


def run_step(description: str, cmd: list[str], cwd: Path = PROJECT_ROOT) -> float:
    print(f"--> [RUNNING] {description} ...")
    start_t = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    
    result = subprocess.run(cmd, cwd=cwd, env=env)
    elapsed = time.perf_counter() - start_t
    
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{description}' failed with exit code {result.returncode}!")
        sys.exit(result.returncode)
        
    print(f"--> [COMPLETED] {description} in {elapsed:.2f}s\n")
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MasterRTL One-Click End-to-End Simulation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use for child processes",
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Skip registering new benchmark Verilog files",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip standalone MasterRTL model checkpoint training",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip 10-fold cross-validation of all 6 baseline configurations",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip publication paper figure generation (Figs. 7, 8, 9, 10)",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest test suite execution",
    )
    args = parser.parse_args()

    total_start = time.perf_counter()
    py = args.python

    print_banner("MasterRTL: One-Click End-to-End Presynthesis PPA Simulation Pipeline")
    print(f"Working Directory : {PROJECT_ROOT}")
    print(f"Python Executable : {py}")

    step_times: dict[str, float] = {}

    # Step 1: Ensure benchmark designs and representations exist
    if not args.skip_benchmarks:
        print_banner("Step 1/7: Verifying & Registering Diverse Hardware RTL Benchmarks")
        t = run_step(
            "Benchmark Registration (Verilog -> SOG, AST, Labels)",
            [py, str(PROJECT_ROOT / "scripts" / "add_diverse_benchmarks.py")],
        )
        step_times["Benchmarks"] = t

    # Step 2: Realistic EDA Ground Truth Verification
    print_banner("Step 2/7: Verifying Realistic NanGate 45nm Synthesis Labels")
    t = run_step(
        "EDA Label Synthesis Engine",
        [py, "-m", "src.data_prep.generate_eda_labels"],
    )
    step_times["EDA Labels"] = t

    # Step 3: Train MasterRTL Core Models
    if not args.skip_training:
        print_banner("Step 3/7: Training MasterRTL Core Models & Saving Checkpoint")
        t = run_step(
            "MasterRTL Model Checkpoint Training",
            [py, str(PROJECT_ROOT / "scripts" / "train_all.py")],
        )
        step_times["MasterRTL Training"] = t

    # Step 4: 10-Fold Cross-Validation Comparison on all 6 Configurations (Table III)
    if not args.skip_baselines:
        print_banner("Step 4/7: Evaluating All Presynthesis Baselines (Table III: 10-Fold CV)")
        t = run_step(
            "10-Fold CV Evaluation ([21] AST/SOG, [22] AST/SOG, MasterRTL AST/SOG)",
            [py, str(PROJECT_ROOT / "scripts" / "evaluate_all_baselines.py")],
        )
        step_times["Baselines (Table III)"] = t

    # Step 5: Data Augmentation & Evaluation Suite (Table V, Table VI, Fig. 13, Fig. 14)
    print_banner("Step 5/7: Executing Data Augmentation Engine & Evaluation (Sec. II-F, III-E)")
    t = run_step(
        "Data Augmentation (Table V, Table VI, Fig 13 t-SNE, Fig 14 Impact)",
        [py, str(PROJECT_ROOT / "scripts" / "evaluate_augmentation.py")],
    )
    step_times["Data Augmentation"] = t

    # Step 6: Publication Figures Generation (Figs. 7, 8, 9, 10)
    if not args.skip_figures:
        print_banner("Step 6/7: Generating Publication Paper Figures (Figs. 7, 8, 9, 10)")
        t = run_step(
            "Paper Figures Generation (PDF & PNG 300 DPI)",
            [py, str(PROJECT_ROOT / "scripts" / "generate_paper_figures.py")],
        )
        step_times["Paper Figures"] = t

    # Step 7: Pytest Suite Execution
    if not args.skip_tests:
        print_banner("Step 7/7: Running Full Pytest Test Suite")
        t = run_step(
            "Pytest Validation (Unit & Integration Tests)",
            [py, "-m", "pytest", "-v"],
        )
        step_times["Pytest Suite"] = t

    total_elapsed = time.perf_counter() - total_start

    print_banner("SIMULATION PIPELINE SUMMARY & DELIVERABLES")
    print(f"Total Execution Time: {total_elapsed:.2f} seconds\n")
    print("Execution Breakdown:")
    for step_name, dur in step_times.items():
        print(f"  - {step_name:<25}: {dur:.2f}s ({dur/total_elapsed*100:.1f}%)")

    print("\nGenerated Reports & Publications:")
    print("  [Table III Comparison] : data/reports/table3_comparison.md")
    print("  [Table III CSV]        : data/reports/table3_comparison.csv")
    print("  [Table III JSON]       : data/reports/table3_comparison.json")
    print("  [Table V Augmentation] : data/reports/table5_augmentation_quality.md")
    print("  [Table VI Proportions] : data/reports/table6_augmentation_proportions.md")
    print("  [Figure 9 (PPA Scatter): data/reports/fig9_ppa_prediction.png (.pdf)")
    print("  [Figure 7 (Intermed.)] : data/reports/fig7_intermediate_models.png (.pdf)")
    print("  [Figure 8 (Breakdown)] : data/reports/fig8_component_breakdown.png (.pdf)")
    print("  [Figure 10 (Ablation)] : data/reports/fig10_ablation_study.png (.pdf)")
    print("  [Figure 13 (t-SNE)]    : data/reports/fig13_tsne_diversity.png (.pdf)")
    print("  [Figure 14 (Aug. Imp.)]: data/reports/fig14_augmentation_impact.png (.pdf)")
    print("  [Model Checkpoint]     : models/master_rtl_checkpoint.pkl")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
