# MasterRTL: Pre-synthesis PPA Estimation Framework

This project implements the pre-synthesis data preparation, **Simple Operator Graph (SOG)** generation engine, and **Machine Learning PPA Estimation Models** of **MasterRTL**, strictly conforming to:

> **Wenji Fang, Yao Lu, Shang Liu, Qijun Zhang, Ceyu Xu, Lisa Wu Wills, Hongce Zhang, and Zhiyao Xie**,  
> *"Transferable Presynthesis PPA Estimation for RTL Designs With Data Augmentation Techniques,"*  
> **IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD)**, Vol. 44, No. 1, January 2025.

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

### 2. Generate SOG Graphs & Feature Extraction

Run the full presynthesis pipeline on all 41 benchmark designs:

```powershell
# Windows PowerShell
.\scripts\run_pipeline.ps1
```

```bash
# Linux / macOS
./scripts/run_pipeline.sh
```

### 3. Train & Evaluate ML Models (10-Fold Cross-Validation)

Train all models across the dataset and run 10-fold cross-validation matching Section III-A:

```powershell
# Windows PowerShell
.\scripts\train_all.ps1
```

```bash
# Linux / macOS
./scripts/train_all.sh
```

---

## Experimental Results (Paper Table III Reproduction)

Evaluation metrics computed across 10-fold cross-validation on the 41 benchmark designs:

| Target Metric | Pearson Correlation ($R$) | MAPE (%) | MAE | RRSE |
| :--- | :---: | :---: | :---: | :---: |
| **Area** ($\mu m^2$) | **0.9909** | **8.9%** | 59.59 | **0.3151** |
| **WNS** (ns) | **0.9386** | 67.2% | 0.0683 | 0.3555 |
| **Power** (mW) | **0.8159** | 39.7% | 1.5133 | 0.8642 |
| **TNS** (ns) | 0.1938 | 221.1% | 7.8485 | 0.9904 |

*Full dataset final fit reaches $R = 0.9996$ on WNS, $R = 0.9992$ on Area, and $R = 0.8837$ on Power.*

---

## Python API Usage

```python
from src.models import MasterRTL, load_ppa_dataset

# 1. Load SOG dataset
dataset = load_ppa_dataset("data/sog_graphs")

# 2. Train MasterRTL
model = MasterRTL()
model.train(dataset)

# 3. Predict PPA for any design
sample = dataset[0]
prediction = model.predict(sample["stats"], sample["paths"])
print("Predicted PPA:", prediction["summary"])
# Output:
# {
#   'predicted_wns': -0.095,
#   'predicted_tns': -0.850,
#   'predicted_power_mW': 1.477,
#   'predicted_area_um2': 322.4
# }

# 4. Transfer across technology nodes (e.g. NanGate 45nm -> TSMC 28nm)
transferred = model.transfer(prediction, sample["stats"], target_technology="TSMC_28nm_TYP")
print("Transferred PPA:", transferred)
```

For complete mathematical formulations and schema details, refer to [docs/SOG_PIPELINE.md](docs/SOG_PIPELINE.md).
