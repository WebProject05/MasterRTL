# MasterRTL: Pre-synthesis PPA Estimation Framework

This project implements the pre-synthesis data preparation, **Simple Operator Graph (SOG)** generation engine, and **Machine Learning PPA Estimation Models** of **MasterRTL**, strictly conforming to:

> **Wenji Fang, Yao Lu, Shang Liu, Qijun Zhang, Ceyu Xu, Lisa Wu Wills, Hongce Zhang, and Zhiyao Xie**,  
> *"Transferable Presynthesis PPA Estimation for RTL Designs With Data Augmentation Techniques,"*  
> **IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD)**, Vol. 44, No. 1, January 2025.

📖 **Comprehensive Beginner-Friendly Guide**: For an in-depth, accessible explanation of chip design fundamentals, the SOG representation, baseline comparisons, data augmentation, and all results, see [**`data/reports/PROJECT_REPORT.md`**](file:///c:/Users/chsan/OneDrive/Desktop/MasterRTL_Project/data/reports/PROJECT_REPORT.md).

---

## Overview

MasterRTL addresses the turnaround time bottleneck of commercial logic synthesis and physical design by directly evaluating RTL designs before synthesis. It transforms raw Verilog HDL into a bit-level representation called the **Simple Operator Graph (SOG)**, canonicalizing arbitrary RTL styles into single-bit registers (`DFF`) and five primitive logic operations: `AND`, `OR`, `XOR`, `NOT`, `MUX`.

### Machine Learning Model Architecture (Paper Table II)

| Target | Internal Stage | ML Model | Hyperparameters & Description |
| :--- | :--- | :--- | :--- |
| **Timing** | Path-Level Delay ($f_{\text{path}}^t$) | **Random Forest Regressor** | 80 estimation trees, maximum depth of 20 |
| | Path Inference | Exact Graph Engine | Evaluates $N$ critical paths ($P^R_{* \to j}$) for $\text{WNS}^R, \text{TNS}^R$, and slack distribution percentiles (worst, 10%, 50%, 90%) |
| | Design-Level Calibration | **XGBoost Regressor** | 45 estimators, maximum depth of 8 |
| **Power** | Module-Level Power | **XGBoost Regressor** | 30 estimators, maximum depth of 6, evaluating switching activity and static leakage |
| | Module Aggregation | Linear Scale Combination | $\text{Power}^G = \sum_{i=1}^M k_i \cdot \text{Power}^{G_i}$ |
| | Design-Level Calibration | **XGBoost Regressor** | 45 estimators, maximum depth of 8 |
| **Area** | Sequential Area | Exact Analytical Formula | $\text{Area}_{\text{seq}} = N_{\text{registers}} \times \text{Area}_{\text{DFF}}$ ($4.522\,\mu m^2$ in NanGate 45nm) |
| | Combinational Area | **XGBoost Regressor** | 45 estimators, maximum depth of 12 |
| | Total Area | Additive Formulation | $\text{Area}_{\text{total}} = \text{Area}_{\text{seq}} + \widehat{\text{Area}}_{\text{comb}}$ |
| **Transfer** | Layout & Tech Transfer | **XGBoost Regressor** | 15 estimators, maximum depth of 8, predicting post-placement PPA and transferring to TSMC 22/28/40/65nm |

---

## Project Structure

```
MasterRTL_Project/
├── data/
│   ├── raw_rtl/         # 41 synthesizable Verilog benchmark designs (.v)
│   ├── yosys_synth/     # Generic synthesis outputs (*.synth.v, *.json ASTs)
│   ├── sog_graphs/      # SOG graph objects (*.pkl), features (*_features.pkl), stats (*_stats.json)
│   ├── ppa_labels/      # Commercial EDA / calibrated ground-truth PPA labels
│   └── reports/         # Aggregated stats: sog_statistics.csv, sog_statistics.json, sog_summary.md
├── docs/
│   └── SOG_PIPELINE.md  # Comprehensive technical documentation, mathematical models & schema
├── models/
│   └── master_rtl_checkpoint.pkl # Serialized trained MasterRTL models
├── scripts/
│   ├── run_pipeline.ps1 # One-click SOG graph generation (Windows PowerShell)
│   ├── run_pipeline.sh  # SOG graph generation (Bash)
│   ├── train_all.ps1    # One-click ML model training & 10-fold CV (Windows PowerShell)
│   ├── train_all.sh     # Model training & CV (Bash)
│   ├── train_all.py     # Training and evaluation runner
│   └── generate_all_raw_rtl.py # Benchmark generator script
├── src/
│   ├── data_prep/
│   │   ├── yosys_synthesis.py # Generic bit-level synthesis with toolchain auto-discovery
│   │   ├── sog_converter.py   # SOG builder, STA DAG, delay modeling, toggle propagation
│   │   └── batch_parse.py     # End-to-end batch processing and statistics aggregation
│   └── models/
│       ├── timing.py          # Path RF (80 trees) + path inference + XGBoost calibration
│       ├── power.py           # Module XGBoost (30 trees) + design calibration (45 trees)
│       ├── area.py            # Sequential analytical + combinational XGBoost (45 trees, depth 12)
│       ├── transfer.py        # Layout and technology transfer XGBoost (15 trees, depth 8)
│       └── master_rtl.py      # Unified coordinator, data loader, and R/MAPE/MAE/RRSE metrics
├── tests/
│   ├── test_models.py         # Unit tests for all ML models and evaluation metrics
│   ├── test_sog_converter.py  # Unit tests for SOG converter
│   ├── test_sog_pipeline.py   # Pipeline integration tests
│   └── test_yosys_synthesis.py# Unit tests for Yosys synthesis scripts
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Requirements

Install all dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Entire Simulation With One Command (Recommended)

To run the complete pipeline end-to-end (benchmark verification, realistic EDA synthesis labels, MasterRTL model training, 10-fold cross-validation of all 6 baseline configurations, publication figures generation, and unit tests):

```powershell
# Windows PowerShell (One-click)
.\scripts\run_all.ps1

# Or via Python directly (Cross-platform)
python scripts/run_all.py
```

Options:
- `python scripts/run_all.py --help` : Show all flags
- `python scripts/run_all.py --skip-benchmarks` : Skip re-registering benchmarks
- `python scripts/run_all.py --skip-training` : Skip standalone model training
- `python scripts/run_all.py --skip-baselines` : Skip 10-fold baseline evaluation
- `python scripts/run_all.py --skip-figures` : Skip figure rendering
- `python scripts/run_all.py --skip-tests` : Skip pytest suite

---

### 3. Step-by-Step Individual Commands

#### A. Generate SOG Graphs & Feature Extraction
```powershell
# Windows PowerShell
.\scripts\run_pipeline.ps1
```

```bash
# Linux / macOS
./scripts/run_pipeline.sh
```

#### B. Train & Evaluate ML Models (10-Fold Cross-Validation)
```powershell
# Windows PowerShell
.\scripts\train_all.ps1
```

```bash
# Linux / macOS
./scripts/train_all.sh
```

---

### 4. Evaluate Baseline Algorithms (Table III Reproduction)

Run 10-fold cross-validation across all 6 configurations matching Table III:
```bash
python scripts/evaluate_all_baselines.py
```

### 5. Generate Publication Figures (Figs. 9, 7, 8, 10)

Generate high-resolution PNG & vector PDF plots:
```bash
python scripts/generate_paper_figures.py
```

---

## Experimental Results (Paper Table III, V, VI Reproductions)

### 1. Table III: Comparison of Cross-Validation Results

10-Fold cross-validation results across all designs comparing the 6 configurations (exported to [data/reports/table3_comparison.md](data/reports/table3_comparison.md)):

| Method | Target WNS: R | WNS: MAPE | Target TNS: R | TNS: MAPE | Target Power: R | Power: MAPE | Target Area: R | Area: MAPE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[21] (AST-alike)** | 0.45 | 67.2% | 0.34 | 501.1% | 0.52 | 83.6% | 0.75 | 49.3% |
| **[21] (SOG)** | 0.98 | 26.6% | 0.80 | 187.6% | 0.94 | 19.8% | 0.97 | 16.0% |
| **[22] (AST-alike)** | 0.99 | 40.0% | 0.51 | 95.6% | 0.43 | 80.4% | 0.67 | 50.7% |
| **[22] (SOG)** | 0.99 | 40.0% | 0.77 | 108.4% | 0.94 | 19.9% | 0.97 | 15.9% |
| **MasterRTL (AST-alike)** | 0.99 | 21.2% | 0.97 | 105.8% | 0.16 | 108.5% | 0.27 | 77.7% |
| **MasterRTL (SOG)** | **0.99** | **21.3%** | **0.97** | **73.7%** | **0.97** | **10.1%** | **1.00** | **1.8%** |

### 2. Table V: Quality Evaluation of Generated RTL Designs
Exported to [data/reports/table5_augmentation_quality.md](data/reports/table5_augmentation_quality.md):
- **Graph-Based Method**: 100% synthesizability, wide scale distribution (29 to 379 nodes), flattened hierarchy.
- **Hierarchical Auto-Decomposition**: 100% synthesizability, multi-module decomposition (Datapath + Control + Wrapper), high functional diversity (Crypto, DSP, Arithmetic, Control, Memory).

### 3. Table VI: Impact of Data Augmentation Across Training Proportions
Exported to [data/reports/table6_augmentation_proportions.md](data/reports/table6_augmentation_proportions.md):
- Verifies that introducing 45 augmented designs maintains high prediction accuracy (>0.90 R) even when real design data is reduced by up to 60%.

---

## Generated Paper Figures

- **[Fig. 9](data/reports/fig9_ppa_prediction.png)**: 4-panel comparison between prediction and ground truth for WNS, TNS, Power, Area (PNG & PDF).
- **[Fig. 7](data/reports/fig7_intermediate_models.png)**: Intermediate-stage ML model comparisons (Path RF vs Transformer; Module XGBoost vs GCN).
- **[Fig. 8](data/reports/fig8_component_breakdown.png)**: Component breakdowns (Static/Dynamic Power, Sequential/Combinational Area).
- **[Fig. 10](data/reports/fig10_ablation_study.png)**: Policy ablation study.
- **[Fig. 13](data/reports/fig13_tsne_diversity.png)**: t-SNE plot visualization for similarity and diversity between generated and real RTL designs.
- **[Fig. 14](data/reports/fig14_augmentation_impact.png)**: Impact of reducing real design training data and introducing augmented RTL designs.

---

## Python API Usage

```python
from src.models import (
    MasterRTL,
    ICCAD21Baseline,
    ISCA22Baseline,
    load_ppa_dataset,
    load_ast_dataset,
)

# 1. Load SOG and AST datasets
sog_ds = load_ppa_dataset("data/sog_graphs")
ast_ds = load_ast_dataset("data/graphs/ast", "data/sog_graphs")

# 2. MasterRTL (SOG)
master = MasterRTL(representation="sog")
master.train(sog_ds)
pred_m = master.predict(sog_ds[0]["stats"], sog_ds[0]["paths"])

# 3. ICCAD'22 [21] Baseline
iccad = ICCAD21Baseline(representation="ast")
iccad.train(ast_ds)
pred_21 = iccad.predict(ast_ds[0])

# 4. ISCA'22 [22] Baseline
isca = ISCA22Baseline(representation="ast")
isca.train(ast_ds)
pred_22 = isca.predict(ast_ds[0])
```

For complete mathematical formulations and schema details, refer to [docs/SOG_PIPELINE.md](docs/SOG_PIPELINE.md).
