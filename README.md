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

## 🚀 Instant PPA Prediction on Any Verilog File (CLI)

You can pass any synthesizable Verilog (`.v` or `.sv`) file to `scripts/predict.py` to get instant presynthesis PPA estimates in **under 1 second**:

```powershell
# 1. Predict PPA for any single Verilog file
.\.venv\Scripts\python.exe scripts/predict.py path/to/my_design.v

# 2. Predict PPA and generate a visual 4-panel dashboard (.png and .pdf)
.\.venv\Scripts\python.exe scripts/predict.py path/to/my_design.v --plot

# 3. Predict with Technology Transfer (e.g., TSMC 28nm)
.\.venv\Scripts\python.exe scripts/predict.py path/to/my_design.v --transfer TSMC_28nm_TYP --plot

# 4. Predict for an entire folder of designs
.\.venv\Scripts\python.exe scripts/predict.py data/raw_rtl/

# 5. Output machine-readable JSON (for CI/CD pipelines)
.\.venv\Scripts\python.exe scripts/predict.py path/to/my_design.v --json
```

**Example Output:**
```text
================================================================================
 MasterRTL: Pre-Synthesis PPA Estimation Report
================================================================================
 Design File        : adder_ripple_8bit.v
 Top Module         : adder_ripple_8bit
 Inference Latency  : 1.180 seconds (Skipped full commercial EDA run!)
--------------------------------------------------------------------------------
 1. CIRCUIT STRUCTURAL COMPLEXITY (SOG Graph)
--------------------------------------------------------------------------------
  * Total Graph Nodes  :       95
  * Register Bits (DFF):       26  (Sequential State)
  * Comb Gates (1-bit) :       50  (AND/OR/XOR/NOT/MUX)
  * Critical Path Depth:    34.40  (Topological Delay Units)
  * Mean Toggle Rate   :    0.076  (Switching Probability)
--------------------------------------------------------------------------------
 2. PREDICTED PPA METRICS (NanGate 45nm Typical Corner, 1.0 GHz / 1.0 ns Target)
--------------------------------------------------------------------------------
  [AREA]
    - Total Silicon Area   :     177.87 um^2
    - Sequential Cell Area :     117.57 um^2 (26 DFFs @ 4.522 um^2)
    - Combinational Area   :      60.30 um^2

  [TIMING / PERFORMANCE]
    - WNS (Worst Slack)    :     -0.187 ns   --> Timing VIOLATED by 0.187 ns
    - TNS (Total Slack)    :     -1.135 ns   --> Cumulative Endpoint Slack

  [POWER CONSUMPTION]
    - Total Power          :      0.754 mW
================================================================================
```

---

## 🛠️ Interactive EDA Tool Suite

To provide a seamless experience matching commercial IC design environments (`dc_shell`, `pt_shell`, `genus`), MasterRTL includes an interactive EDA Command Shell and a zero-dependency Web Studio GUI:

### 1. Interactive EDA Command Shell (`master_rtl>`)
Launch the interactive chip design REPL with one click:
```powershell
# Windows PowerShell
.\eda_shell.ps1

# Or Windows Command Prompt / Linux
.\eda_shell.bat
python scripts/master_rtl_shell.py
```

**Supported EDA Shell Commands:**
| Command | Description |
| :--- | :--- |
| `read_verilog <file.v>` | Synthesizes and elaborates design into SOG bit-level graph in ~1 sec. |
| `report_ppa [-plot]` | Prints complete unified PPA scorecard (with optional 4-panel visual plot). |
| `report_timing` | Static Timing Analysis (STA): WNS, TNS, critical path delay, endpoint count. |
| `report_area` | Standard-cell area breakdown: Sequential DFF area vs Combinational gate area. |
| `report_power` | Toggle-activity dynamic switching power vs static leakage dissipation. |
| `set_target_technology <node>` | Switch foundry process node (`TSMC_28nm_TYP`, `TSMC_28nm_MIN`, etc.). |
| `set_clock_period <ns>` | Dynamically update target clock period (e.g. `0.8` for 1.25 GHz). |
| `plot_ppa [-output <dir>]` | Exports publication-quality 300-DPI PNG and vector PDF PPA dashboard. |
| `list_designs [filter]` | Lists benchmark Verilog files in `data/raw_rtl/`. |
| `gui` | Spawns the Web EDA Studio and opens it in your default browser. |

### 2. Interactive Web EDA Studio (`http://localhost:8080`)
A local, zero-dependency browser workbench with real-time gauges, Verilog code editor, and live PPA telemetry:
```powershell
# Windows PowerShell
.\eda_gui.ps1

# Or Windows Command Prompt / Linux
.\eda_gui.bat
python scripts/eda_gui.py
```
Open **[http://localhost:8080](http://localhost:8080)** in any modern browser to:
- Pick from 156+ pre-synthesized benchmark RTL designs or paste custom Verilog.
- Trigger instant 1-second presynthesis PPA estimation.
- View live color-coded KPI gauge cards for Total Area, WNS Slack, TNS Slack, and Total Power.
- View live embedded 4-panel prediction dashboard plots and circuit complexity metrics.
- Switch foundry process nodes (NanGate 45nm, TSMC 28nm) in real time.

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
